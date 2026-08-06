use oxionnx_core::Tensor;

#[cfg(feature = "simd")]
use super::broadcast::is_full_reduction;
use super::broadcast::sub;
use crate::shape::basic::normalize_axis;

/// The `simd_reduce_*` full-reduction shortcuts below sum/max/min/mean the raw slice they are
/// given, with no shape awareness of their own. `Tensor::new` only checks `data.len() ==
/// shape.product()` via `debug_assert!` (release builds aren't validated -- see its doc
/// comment), so bound the slice to what `x.shape` actually describes before handing it to the
/// SIMD kernel, the same `.take(shape_numel)` cap `reduce_with`/`reduce_with_into` apply for the
/// identical reason (see the comment there). `.min(x.data.len())` additionally guards the
/// opposite mismatch (`data.len() < shape.product()`), which would otherwise panic on the slice
/// bound here; a well-formed tensor (the overwhelming common case) always has
/// `shape.product() == data.len()`, so this is a no-op slice, not a copy.
#[cfg(feature = "simd")]
#[inline]
fn shape_bounded_data(x: &Tensor) -> &[f32] {
    let n = x.shape.iter().product::<usize>().min(x.data.len());
    &x.data[..n]
}

// ── Output-shape helper ───────────────────────────────────────────────────────

/// Compute the output shape and element count for a reduction.
///
/// Used by the registry `execute_into_slots` path to pre-size an output buffer before calling
/// the real `reduce_*_into` kernel. An out-of-range axis is a typed `Err` here — via the same
/// checked `normalize_axis` the rest of the crate uses, rather than an unchecked cast that could
/// wrap a wildly negative axis into an arbitrary `usize` — returned *before* the caller touches
/// (or mis-sizes) any output buffer, the same "validate before the buffer is sized" discipline
/// `ArgMaxOp`/`ArgMinOp::execute_into_slots` already apply via `arg_output_shape`.
///
/// This used to be an infallible *hint* that silently dropped an out-of-range axis (via
/// `normalize_axis(a, ndim).ok()` in a `filter_map`), reasoning that the subsequent
/// `reduce_with`/`reduce_with_into` call downstream would re-validate and surface the error
/// itself — which it does for every *current* caller, but only by accident of call order: it
/// left this function itself able to silently compute a wrong (full-passthrough) shape for a
/// malformed `axes` list, a landmine for any future caller that trusts this shape without also
/// re-validating. Returning `Result` here makes the axis check load-bearing at its actual
/// source instead of relying on a second, separate validation elsewhere to catch what this
/// function got wrong.
pub(crate) fn reduce_output_shape(
    x: &Tensor,
    axes_raw: &[i64],
    keepdims: bool,
) -> Result<(Vec<usize>, usize), String> {
    let ndim = x.ndim();
    let axes: Vec<usize> = if axes_raw.is_empty() {
        (0..ndim).collect()
    } else {
        axes_raw
            .iter()
            .map(|&a| normalize_axis(a, ndim))
            .collect::<Result<_, _>>()?
    };
    let out_shape: Vec<usize> = if keepdims {
        x.shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
            .collect()
    } else {
        // Reducing away every axis leaves the **empty** shape — a genuine rank-0 tensor, as
        // ONNX and NumPy both specify (`np.sum(x, axis=(0,1,2), keepdims=False).shape == ()`)
        // — not the rank-1 `[1]` this used to promote it to. `len` below is unaffected: the
        // empty shape's product is the empty product 1, which is exactly the one element a
        // full reduction produces.
        x.shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
            .collect()
    };
    let len: usize = out_shape.iter().product::<usize>().max(1);
    Ok((out_shape, len))
}

// ── Zero-copy reduce primitive ────────────────────────────────────────────────

/// Like `reduce_with` but writes the final result into `out` (pre-sized).
/// Returns the output shape.
pub(crate) fn reduce_with_into<F, G>(
    x: &Tensor,
    axes_raw: &[i64],
    keepdims: bool,
    init: f32,
    accumulate: F,
    finalize: G,
    out: &mut [f32],
) -> Result<Vec<usize>, String>
where
    F: Fn(f32, f32) -> f32,
    G: Fn(f32, u32) -> f32,
{
    let ndim = x.ndim();
    // Checked normalization (shared with the rest of the crate's axis-taking ops) instead of an
    // unchecked `(a + ndim as i64) as usize` cast: an out-of-range axis (or one so negative that
    // `a + ndim` is still negative) must be reported as a typed error, not silently reinterpreted
    // as a huge `usize` that `.contains()` then just happens to never match.
    let axes: Vec<usize> = if axes_raw.is_empty() {
        (0..ndim).collect()
    } else {
        axes_raw
            .iter()
            .map(|&a| normalize_axis(a, ndim))
            .collect::<Result<_, _>>()?
    };
    // O(ndim) once, replacing the O(|axes|) `axes.contains(&d)` scan the hot loop below used to
    // repeat once per (element, dimension) pair.
    let mut axis_mask = vec![false; ndim];
    for &d in &axes {
        axis_mask[d] = true;
    }

    // keepdims-true shape used for stride computation
    let kd_shape: Vec<usize> = x
        .shape
        .iter()
        .enumerate()
        .map(|(i, &d)| if axis_mask[i] { 1 } else { d })
        .collect();
    let out_n: usize = kd_shape.iter().product::<usize>().max(1);
    debug_assert!(out.len() >= out_n, "reduce_with_into: out buffer too small");

    for v in out[..out_n].iter_mut() {
        *v = init;
    }
    let mut counts = vec![0u32; out_n];

    let mut kd_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        kd_strides[i] = s;
        s *= kd_shape[i];
    }
    // Per-input-axis contribution to `out_idx`: 0 for a reduced axis (every value that axis takes
    // lands in the same output bucket), `kd_strides[d]` otherwise. Precomputing this turns the
    // hot loop's `if !axis_mask[d] { out_idx += coord * kd_strides[d] }` into an unconditional add
    // of a per-axis constant.
    let eff_out_stride: Vec<usize> = (0..ndim)
        .map(|d| if axis_mask[d] { 0 } else { kd_strides[d] })
        .collect();

    // Walk `x.data` in its natural (already row-major) order while maintaining an odometer over
    // `x.shape`: `coord` is the standard row-major decode of the current flat index, updated by a
    // single increment plus an occasional carry instead of `ndim` fresh divisions per element.
    // `out_idx` is maintained the same way, in lockstep, via `eff_out_stride` -- this computes
    // exactly `sum_d coord[d] * eff_out_stride[d]` at every step, which is definitionally the same
    // value the original per-element `rem / in_strides[d]` decode + `axes.contains` check produced,
    // just accumulated incrementally instead of recomputed from scratch.
    //
    // `.take(shape_numel)` bounds the walk at exactly `x.shape.iter().product()` -- the odometer's
    // natural cycle length, after which `coord`/`out_idx` wrap back to their initial all-zero
    // state. `Tensor::new` only validates `data.len() == shape.product()` via `debug_assert!` (see
    // its doc comment), so a release build can hand this a tensor with `data.len() >
    // shape.product()`; without the cap, the odometer would silently begin a second lap and
    // accumulate that trailing "extra" data into buckets 0.. again instead of leaving it alone.
    // Note this is *not* `x.data.len()`: this crate's `Tensor::numel()` returns `data.len()`
    // itself, not the shape product, so `.take(x.numel())` here would be a no-op.
    let shape_numel: usize = x.shape.iter().product();
    let mut coord = vec![0usize; ndim];
    let mut out_idx = 0usize;
    for &val in x.data.iter().take(shape_numel) {
        out[out_idx] = accumulate(out[out_idx], val);
        counts[out_idx] += 1;
        for d in (0..ndim).rev() {
            coord[d] += 1;
            out_idx += eff_out_stride[d];
            if coord[d] < x.shape[d] {
                break;
            }
            out_idx -= coord[d] * eff_out_stride[d];
            coord[d] = 0;
        }
    }

    for (v, &c) in out[..out_n].iter_mut().zip(counts.iter()) {
        *v = finalize(*v, c);
    }

    if keepdims {
        Ok(kd_shape)
    } else {
        // Reducing away every axis leaves the empty shape (rank 0), not `[1]` — see
        // `reduce_output_shape` above for the ONNX/NumPy reference.
        Ok(kd_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if axis_mask[i] { None } else { Some(d) })
            .collect())
    }
}

// ── Per-op _into wrappers ─────────────────────────────────────────────────────

pub(crate) fn reduce_mean_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_mean(shape_bounded_data(x));
            if keepdims {
                let shape = vec![1; x.ndim()];
                let n: usize = shape.iter().product();
                for v in out.iter_mut().take(n) {
                    *v = val;
                }
                return Ok(shape);
            } else {
                // A full reduction with `keepdims=0` produces a rank-0 scalar (the
                // empty shape), not `[1]` — this must agree with the non-SIMD
                // `reduce_with_into` path it shortcuts.
                out[0] = val;
                return Ok(Vec::new());
            }
        }
    }
    reduce_with_into(
        x,
        axes,
        keepdims,
        0.0,
        |a, v| a + v,
        |s, c| s / c as f32,
        out,
    )
}

pub(crate) fn reduce_sum_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_sum(shape_bounded_data(x));
            if keepdims {
                let shape = vec![1; x.ndim()];
                let n: usize = shape.iter().product();
                for v in out.iter_mut().take(n) {
                    *v = val;
                }
                return Ok(shape);
            } else {
                // A full reduction with `keepdims=0` produces a rank-0 scalar (the
                // empty shape), not `[1]` — this must agree with the non-SIMD
                // `reduce_with_into` path it shortcuts.
                out[0] = val;
                return Ok(Vec::new());
            }
        }
    }
    reduce_with_into(x, axes, keepdims, 0.0, |a, v| a + v, |s, _| s, out)
}

pub(crate) fn reduce_max_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_max(shape_bounded_data(x));
            if keepdims {
                let shape = vec![1; x.ndim()];
                let n: usize = shape.iter().product();
                for v in out.iter_mut().take(n) {
                    *v = val;
                }
                return Ok(shape);
            } else {
                // A full reduction with `keepdims=0` produces a rank-0 scalar (the
                // empty shape), not `[1]` — this must agree with the non-SIMD
                // `reduce_with_into` path it shortcuts.
                out[0] = val;
                return Ok(Vec::new());
            }
        }
    }
    reduce_with_into(
        x,
        axes,
        keepdims,
        f32::NEG_INFINITY,
        |a, v| a.max(v),
        |s, _| s,
        out,
    )
}

pub(crate) fn reduce_min_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_min(shape_bounded_data(x));
            if keepdims {
                let shape = vec![1; x.ndim()];
                let n: usize = shape.iter().product();
                for v in out.iter_mut().take(n) {
                    *v = val;
                }
                return Ok(shape);
            } else {
                // A full reduction with `keepdims=0` produces a rank-0 scalar (the
                // empty shape), not `[1]` — this must agree with the non-SIMD
                // `reduce_with_into` path it shortcuts.
                out[0] = val;
                return Ok(Vec::new());
            }
        }
    }
    reduce_with_into(
        x,
        axes,
        keepdims,
        f32::INFINITY,
        |a, v| a.min(v),
        |s, _| s,
        out,
    )
}

pub(crate) fn reduce_prod_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    reduce_with_into(x, axes, keepdims, 1.0, |a, v| a * v, |s, _| s, out)
}

pub(crate) fn reduce_l1_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    reduce_with_into(x, axes, keepdims, 0.0, |a, v| a + v.abs(), |s, _| s, out)
}

pub(crate) fn reduce_l2_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    reduce_with_into(
        x,
        axes,
        keepdims,
        0.0,
        |a, v| a + v * v,
        |s, _| s.sqrt(),
        out,
    )
}

pub(crate) fn reduce_log_sum_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    reduce_with_into(x, axes, keepdims, 0.0, |a, v| a + v, |s, _| s.ln(), out)
}

pub(crate) fn reduce_sum_square_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    reduce_with_into(x, axes, keepdims, 0.0, |a, v| a + v * v, |s, _| s, out)
}

pub(crate) fn reduce_log_sum_exp_into(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    // Numerically stable: log(sum(exp(x))) = max + log(sum(exp(x - max)))
    let max_keep = reduce_max(x, axes, true)?;
    let shifted = sub(x, &max_keep)?;
    let exp_data: Vec<f32> = shifted.data.iter().map(|v| v.exp()).collect();
    let exp_tensor = Tensor::new(exp_data, shifted.shape.clone());
    let sum_exp = reduce_sum(&exp_tensor, axes, keepdims)?;
    let max_final = if keepdims {
        max_keep
    } else {
        reduce_max(x, axes, false)?
    };
    let n = sum_exp.data.len();
    if out.len() < n {
        return Err(format!(
            "reduce_log_sum_exp_into: out buffer too small ({} < {})",
            out.len(),
            n
        ));
    }
    for (i, (&s, &m)) in sum_exp.data.iter().zip(max_final.data.iter()).enumerate() {
        out[i] = s.ln() + m;
    }
    Ok(sum_exp.shape)
}

// ── Allocating helpers (hot path returns Tensor) ──────────────────────────────

pub(super) fn reduce_with<F, G>(
    x: &Tensor,
    axes: &[i64],
    keepdims: bool,
    init: f32,
    accumulate: F,
    finalize: G,
) -> Result<Tensor, String>
where
    F: Fn(f32, f32) -> f32,
    G: Fn(f32, u32) -> f32,
{
    let ndim = x.ndim();
    // Checked normalization instead of an unchecked `(a + ndim as i64) as usize` cast: an
    // out-of-range axis (or one so negative that `a + ndim` is still negative) used to become
    // an arbitrary/huge `usize` that then indexed `out_shape[ax]` directly below — an
    // out-of-bounds panic on a malformed `axes` attribute/input instead of a typed error.
    let axes: Vec<usize> = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.iter()
            .map(|&a| normalize_axis(a, ndim))
            .collect::<Result<_, _>>()?
    };
    // O(ndim) once, replacing the O(|axes|) `axes.contains(&d)` scan the hot loop below used to
    // repeat once per (element, dimension) pair.
    let mut axis_mask = vec![false; ndim];
    for &d in &axes {
        axis_mask[d] = true;
    }

    let mut out_shape: Vec<usize> = x.shape.clone();
    for &ax in &axes {
        out_shape[ax] = 1;
    }

    let out_n: usize = out_shape.iter().product();
    let mut acc = vec![init; out_n];
    let mut counts = vec![0u32; out_n];

    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    // Per-input-axis contribution to `out_idx`: 0 for a reduced axis, `out_strides[d]` otherwise
    // (see `reduce_with_into` above for the full derivation -- same technique, same guarantee of
    // bit-identical output, applied to this allocating twin).
    let eff_out_stride: Vec<usize> = (0..ndim)
        .map(|d| if axis_mask[d] { 0 } else { out_strides[d] })
        .collect();

    // See `reduce_with_into` above for why this is `.take(shape_numel)` and not
    // `.take(x.numel())` (this crate's `Tensor::numel()` is `data.len()`, not the shape product).
    let shape_numel: usize = x.shape.iter().product();
    let mut coord = vec![0usize; ndim];
    let mut out_idx = 0usize;
    for &val in x.data.iter().take(shape_numel) {
        acc[out_idx] = accumulate(acc[out_idx], val);
        counts[out_idx] += 1;
        for d in (0..ndim).rev() {
            coord[d] += 1;
            out_idx += eff_out_stride[d];
            if coord[d] < x.shape[d] {
                break;
            }
            out_idx -= coord[d] * eff_out_stride[d];
            coord[d] = 0;
        }
    }

    let data: Vec<f32> = acc
        .iter()
        .zip(counts.iter())
        .map(|(&a, &c)| finalize(a, c))
        .collect();

    if keepdims {
        Ok(Tensor::new(data, out_shape))
    } else {
        // Reducing away every axis leaves the empty shape (rank 0), not `[1]` — see
        // `reduce_output_shape` above for the ONNX/NumPy reference.
        let final_shape: Vec<usize> = out_shape
            .into_iter()
            .enumerate()
            .filter_map(|(i, d)| if axis_mask[i] { None } else { Some(d) })
            .collect();
        Ok(Tensor::new(data, final_shape))
    }
}

pub fn reduce_mean(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_mean(shape_bounded_data(x));
            // `keepdims=0` over every axis yields the empty shape (rank 0), matching
            // the non-SIMD `reduce_with` path this shortcuts.
            let shape = if keepdims {
                vec![1; x.ndim()]
            } else {
                Vec::new()
            };
            return Ok(Tensor::new(vec![val], shape));
        }
    }
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v, |s, c| s / c as f32)
}

pub fn reduce_sum(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_sum(shape_bounded_data(x));
            // `keepdims=0` over every axis yields the empty shape (rank 0), matching
            // the non-SIMD `reduce_with` path this shortcuts.
            let shape = if keepdims {
                vec![1; x.ndim()]
            } else {
                Vec::new()
            };
            return Ok(Tensor::new(vec![val], shape));
        }
    }
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v, |s, _| s)
}

pub fn reduce_max(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_max(shape_bounded_data(x));
            // `keepdims=0` over every axis yields the empty shape (rank 0), matching
            // the non-SIMD `reduce_with` path this shortcuts.
            let shape = if keepdims {
                vec![1; x.ndim()]
            } else {
                Vec::new()
            };
            return Ok(Tensor::new(vec![val], shape));
        }
    }
    reduce_with(
        x,
        axes,
        keepdims,
        f32::NEG_INFINITY,
        |a, v| a.max(v),
        |s, _| s,
    )
}

pub fn reduce_min(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    {
        if is_full_reduction(x, axes) {
            let val = crate::simd_ops::simd_reduce_min(shape_bounded_data(x));
            // `keepdims=0` over every axis yields the empty shape (rank 0), matching
            // the non-SIMD `reduce_with` path this shortcuts.
            let shape = if keepdims {
                vec![1; x.ndim()]
            } else {
                Vec::new()
            };
            return Ok(Tensor::new(vec![val], shape));
        }
    }
    reduce_with(x, axes, keepdims, f32::INFINITY, |a, v| a.min(v), |s, _| s)
}

pub fn reduce_prod(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 1.0, |a, v| a * v, |s, _| s)
}

/// ReduceL1: sum(|x|)
pub fn reduce_l1(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v.abs(), |s, _| s)
}

/// ReduceL2: sqrt(sum(x^2))
pub fn reduce_l2(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v * v, |s, _| s.sqrt())
}

/// ReduceLogSum: log(sum(x))
pub fn reduce_log_sum(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v, |s, _| s.ln())
}

/// ReduceSumSquare: sum(x^2)
pub fn reduce_sum_square(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    reduce_with(x, axes, keepdims, 0.0, |a, v| a + v * v, |s, _| s)
}

/// ReduceLogSumExp: log(sum(exp(x))) — numerically stable via max-subtract trick.
pub fn reduce_log_sum_exp(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, String> {
    // Step 1: max with keepdims=true for broadcasting
    let max_keep = reduce_max(x, axes, true)?;
    // Step 2: x - max (sub handles broadcasting internally)
    let shifted = sub(x, &max_keep)?;
    // Step 3: sum(exp(shifted)) with requested keepdims
    let exp_data: Vec<f32> = shifted.data.iter().map(|v| v.exp()).collect();
    let exp_tensor = Tensor::new(exp_data, shifted.shape.clone());
    let sum_exp = reduce_sum(&exp_tensor, axes, keepdims)?;
    // Step 4: log(sum_exp) + max_final
    let max_final = if keepdims {
        max_keep
    } else {
        reduce_max(x, axes, false)?
    };
    let out_data: Vec<f32> = sum_exp
        .data
        .iter()
        .zip(max_final.data.iter())
        .map(|(&s, &m)| s.ln() + m)
        .collect();
    Ok(Tensor::new(out_data, sum_exp.shape.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Axes lists below are deliberately shorter than `ndim` so `is_full_reduction` (in
    // `broadcast.rs`) short-circuits to `false` on the length check alone, regardless of
    // whether the `simd` feature is enabled for this test run — guaranteeing the assertions
    // below actually exercise `reduce_with`/`reduce_with_into`'s axis validation rather than a
    // SIMD fast path that never looks at axis values.

    /// [Checked axis-normalization regression] An out-of-range *positive* axis used to become
    /// `out_shape[ax] = 1` with `ax` unchecked — an out-of-bounds panic (`out_shape` only has
    /// `ndim` entries) instead of a typed error, on a value a malformed/adversarial model could
    /// supply directly as the `axes` attribute or input.
    #[test]
    fn reduce_with_out_of_range_positive_axis_errors_not_panics() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let err = reduce_sum(&x, &[5], false).unwrap_err();
        assert!(
            err.contains("5"),
            "error should mention the bad axis: {err}"
        );
    }

    /// [Checked axis-normalization regression] An axis so negative that `axis + ndim` is still
    /// negative used to wrap into an arbitrary huge `usize` via `as usize` on a negative `i64`
    /// (undefined-looking but well-defined two's-complement reinterpretation) and then index
    /// `out_shape` with it — an out-of-bounds panic, not a typed error.
    #[test]
    fn reduce_with_very_negative_axis_errors_not_panics() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let err = reduce_sum(&x, &[-10], false).unwrap_err();
        assert!(!err.is_empty());
    }

    /// Same two cases through the zero-copy `_into` path (`reduce_with_into`, via
    /// `reduce_sum_into`), which must error *before* writing anything into `out` — an
    /// undersized buffer (deliberately 0-length here) must not be touched, let alone panic.
    #[test]
    fn reduce_with_into_bad_axis_errors_not_panics() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let mut out: Vec<f32> = Vec::new();
        assert!(reduce_sum_into(&x, &[5], false, &mut out).is_err());
        assert!(reduce_sum_into(&x, &[-10], false, &mut out).is_err());
    }

    /// [G1-ops-close] `reduce_output_shape` itself used to silently drop an out-of-range axis
    /// (`normalize_axis(a, ndim).ok()` in a `filter_map`) rather than reporting it, computing a
    /// full-passthrough shape (every axis kept, none reduced) instead of erroring. Every
    /// *current* caller happens to be safe regardless -- `reduce_sum_into` above re-validates
    /// via `reduce_with_into`'s own strict `normalize_axis`, and the two tests above pin exactly
    /// that -- but that safety was incidental to call order in the registry macro
    /// (`registry/math_ops/reduce.rs`'s `execute_into_slots`), not a guarantee this function
    /// made itself. This test is the direct regression check on the function whose contract
    /// changed: unlike `reduce_with_out_of_range_positive_axis_errors_not_panics` /
    /// `reduce_with_into_bad_axis_errors_not_panics` above (which were already passing before
    /// this fix, since `reduce_with`/`reduce_with_into` always re-validated), this assertion is
    /// only true after `reduce_output_shape` gained its own `Result` return.
    #[test]
    fn reduce_output_shape_out_of_range_axis_errors_not_silently_dropped() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // ReduceSum axes=[5] on a rank-2 tensor: axis 5 is out of range for ndim=2.
        let err = reduce_output_shape(&x, &[5], false)
            .expect_err("out-of-range positive axis must be a typed error, not a dropped axis");
        assert!(
            err.contains('5'),
            "error should mention the bad axis: {err}"
        );

        // Also out of range: an axis so negative that `axis + ndim` is still negative.
        assert!(reduce_output_shape(&x, &[-10], false).is_err());

        // Sanity: a valid axis must still succeed and reduce exactly that axis, not silently
        // keep the full shape the way the out-of-range cases above used to.
        let (shape, len) = reduce_output_shape(&x, &[1], false)
            .expect("in-range axis must still succeed after the Result migration");
        assert_eq!(shape, vec![2]);
        assert_eq!(len, 2);
    }

    /// Regression: an in-range negative axis must keep behaving exactly like its positive
    /// equivalent (this is the common, valid case the checked helper must not break).
    #[test]
    fn reduce_with_negative_axis_matches_positive_equivalent() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let neg = reduce_sum(&x, &[-1], false).expect("negative axis should be valid");
        let pos = reduce_sum(&x, &[1], false).expect("positive axis should be valid");
        assert_eq!(neg.shape, pos.shape);
        assert_eq!(neg.data, pos.data);
        assert_eq!(neg.data, vec![6.0, 15.0]); // row sums: 1+2+3, 4+5+6
    }

    // ── [W2-perf-misc / a6-9] odometer + axis-mask rewrite of reduce_with /
    // reduce_with_into ─────────────────────────────────────────────────────
    //
    // The rewrite replaced (a) the per-(element, dimension) `axes.contains(&d)`
    // linear scan with an O(ndim) precomputed `bool` mask, and (b) the
    // per-element `rem / in_strides[d]` / `rem %= in_strides[d]` div/mod
    // decode with an increment-and-carry odometer. Both are pure index-space
    // changes -- for a fixed `accumulate`/`finalize` pair the *set* and
    // *order* of (bucket, value) pairs fed to `accumulate` is unchanged, so
    // sum/min/max/prod reductions below are checked for exact equality
    // against an independently written reference, not a tolerance.

    /// Independent reference: the exact per-element div/mod decode +
    /// `axes.contains` scan `reduce_with` used before this change, for SUM
    /// specifically. Deliberately not shared with the code under test.
    fn reference_reduce_sum(
        data: &[f32],
        shape: &[usize],
        axes: &[usize],
        keepdims: bool,
    ) -> (Vec<f32>, Vec<usize>) {
        let ndim = shape.len();
        let mut out_shape: Vec<usize> = shape.to_vec();
        for &ax in axes {
            out_shape[ax] = 1;
        }
        let out_n: usize = out_shape.iter().product();
        let mut acc = vec![0.0f32; out_n];
        let mut in_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            in_strides[i] = s;
            s *= shape[i];
        }
        let mut out_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = s;
            s *= out_shape[i];
        }
        let numel: usize = shape.iter().product();
        for (in_idx, &val) in data.iter().enumerate().take(numel) {
            let mut rem = in_idx;
            let mut out_idx = 0usize;
            for d in 0..ndim {
                let coord = rem / in_strides[d];
                rem %= in_strides[d];
                if !axes.contains(&d) {
                    out_idx += coord * out_strides[d];
                }
            }
            acc[out_idx] += val;
        }
        if keepdims {
            (acc, out_shape)
        } else {
            // Reducing away every axis leaves the empty shape (rank 0), not `[1]`:
            // ONNX `ReduceSum` removes the reduced axes outright when `keepdims=0`
            // (`np.sum(x, axis=tuple(range(x.ndim)), keepdims=False).shape == ()`).
            // This reference used to promote it — dead in practice, since every
            // case below keeps `axes.len() < ndim` (see the module note above), but
            // it would have silently endorsed the very promotion the migration
            // removed the moment a full-reduction case was added here.
            let final_shape: Vec<usize> = out_shape
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
                .collect();
            (acc, final_shape)
        }
    }

    #[test]
    fn matches_numpy_reference_mean_middle_axis() {
        // ```python
        // x = (np.arange(24, dtype=np.float32) * 0.5 - 6.0).reshape(2,3,4)
        // x.astype(np.float64).mean(axis=1).flatten()
        // ```
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 6.0).collect();
        let x = Tensor::new(data, vec![2, 3, 4]);
        let y = reduce_mean(&x, &[1], false).expect("reduce_mean failed");
        assert_eq!(y.shape, vec![2, 4]);
        assert_eq!(y.data, vec![-4.0, -3.5, -3.0, -2.5, 2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    fn matches_numpy_reference_sum_non_adjacent_axes() {
        // ```python
        // x = (np.arange(120, dtype=np.float32) * 0.25 - 15.0).reshape(2,3,4,5)
        // x.astype(np.float64).sum(axis=(0,2)).flatten()
        // ```
        let data: Vec<f32> = (0..120).map(|i| i as f32 * 0.25 - 15.0).collect();
        let x = Tensor::new(data, vec![2, 3, 4, 5]);
        let y = reduce_sum(&x, &[0, 2], false).expect("reduce_sum failed");
        assert_eq!(y.shape, vec![3, 5]);
        let expected: Vec<f32> = vec![
            -45.0, -43.0, -41.0, -39.0, -37.0, -5.0, -3.0, -1.0, 1.0, 3.0, 35.0, 37.0, 39.0, 41.0,
            43.0,
        ];
        assert_eq!(y.data, expected);

        // keepdims=true must give the same data, reshaped to [1,3,1,5].
        let y_kd = reduce_sum(&x, &[0, 2], true).expect("reduce_sum keepdims failed");
        assert_eq!(y_kd.shape, vec![1, 3, 1, 5]);
        assert_eq!(y_kd.data, expected);
    }

    #[test]
    fn odometer_matches_reference_across_shapes_axes_and_keepdims() {
        // Axes lists are always shorter than `ndim` (see the module-level note
        // above) so this exercises `reduce_with`'s odometer path specifically,
        // never the full-reduction SIMD shortcut.
        let cases: &[(&[usize], &[usize])] = &[
            (&[4, 3], &[0]),
            (&[4, 3], &[1]),
            (&[2, 3, 4], &[0]),
            (&[2, 3, 4], &[1]),
            (&[2, 3, 4], &[2]),
            (&[2, 3, 4], &[0, 2]), // non-adjacent
            (&[1, 5, 1, 3], &[1]),
            (&[1, 5, 1, 3], &[0, 2]),       // reducing size-1 axes
            (&[2, 2, 2, 2, 2], &[0, 2, 4]), // 5D, alternating axes
            (&[3, 1, 4], &[0, 2]),
        ];
        for &(shape, axes) in cases {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|i| (i as f32) * 1.25 - (n as f32) / 2.0)
                .collect();
            let x = Tensor::new(data.clone(), shape.to_vec());
            let axes_i64: Vec<i64> = axes.iter().map(|&a| a as i64).collect();

            for &keepdims in &[false, true] {
                let got = reduce_sum(&x, &axes_i64, keepdims).unwrap_or_else(|e| {
                    panic!("reduce_sum shape={shape:?} axes={axes:?} kd={keepdims}: {e}")
                });
                let (want_data, want_shape) = reference_reduce_sum(&data, shape, axes, keepdims);
                assert_eq!(
                    got.shape, want_shape,
                    "shape mismatch: shape={shape:?} axes={axes:?} keepdims={keepdims}"
                );
                assert_eq!(
                    got.data, want_data,
                    "data mismatch (must be bit-identical): shape={shape:?} axes={axes:?} keepdims={keepdims}"
                );

                // Same comparison through the zero-copy `_into` path.
                let (_, out_n) = reduce_output_shape(&x, &axes_i64, keepdims)
                    .expect("reduce_output_shape should succeed for valid in-range axes");
                let mut out = vec![-999.0f32; out_n];
                let got_shape_into = reduce_sum_into(&x, &axes_i64, keepdims, &mut out)
                    .unwrap_or_else(|e| {
                        panic!("reduce_sum_into shape={shape:?} axes={axes:?} kd={keepdims}: {e}")
                    });
                assert_eq!(got_shape_into, want_shape);
                assert_eq!(&out[..out_n], want_data.as_slice());
            }
        }
    }

    /// [rank-0 migration] A `keepdims=0` reduction over **every** axis produces a
    /// genuine rank-0 tensor — shape `[]`, one element — through all three
    /// surfaces that have to agree: the allocating kernel (`reduce_*`), the
    /// zero-copy `_into` kernel (whose returned shape *and* the buffer length
    /// `reduce_output_shape` pre-plans for it must both be right), and
    /// `reduce_output_shape` itself.
    ///
    /// ONNX `ReduceSum`/`ReduceMean`/`ReduceMax`/`ReduceMin`: `keepdims=0`
    /// *removes* the reduced axes, so removing all of them leaves rank 0 —
    /// `np.sum(np.arange(24).reshape(2,3,4), axis=(0,1,2), keepdims=False).shape`
    /// is `()`. Both spellings of "all axes" are exercised because they reach
    /// different code: an explicit list, and the empty list ONNX defines as
    /// "reduce every dimension" (which under the `simd` feature additionally
    /// takes the `is_full_reduction` shortcut, a separate hard-coded shape).
    #[test]
    fn full_reduction_keepdims0_is_rank0_across_every_surface() {
        let rank0: Vec<usize> = Vec::new();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();

        for shape in [vec![24usize], vec![4, 6], vec![2, 3, 4]] {
            let x = Tensor::new(data.clone(), shape.clone());
            let all_axes: Vec<i64> = (0..shape.len() as i64).collect();

            for axes in [all_axes.as_slice(), &[]] {
                // reduce_output_shape: empty shape, but still one element to size.
                let (out_shape, out_n) = reduce_output_shape(&x, axes, false)
                    .expect("reduce_output_shape should succeed for valid in-range axes");
                assert_eq!(out_shape, rank0, "shape={shape:?} axes={axes:?}");
                assert_eq!(out_n, 1, "a rank-0 output still holds exactly one element");

                // Allocating kernel.
                let y = reduce_sum(&x, axes, false).expect("reduce_sum runs");
                assert_eq!(y.shape, rank0, "shape={shape:?} axes={axes:?}");
                assert_eq!(y.data, vec![276.0], "sum(0..24)");

                // Zero-copy `_into` kernel, sized by `reduce_output_shape` exactly
                // as the registry's `execute_into_slots` sizes it.
                let mut out = vec![-999.0f32; out_n];
                let into_shape =
                    reduce_sum_into(&x, axes, false, &mut out).expect("reduce_sum_into runs");
                assert_eq!(into_shape, rank0, "shape={shape:?} axes={axes:?}");
                assert_eq!(out, vec![276.0]);

                // The other kinds carrying their own full-reduction shortcut.
                assert_eq!(reduce_mean(&x, axes, false).expect("mean").shape, rank0);
                assert_eq!(reduce_max(&x, axes, false).expect("max").shape, rank0);
                assert_eq!(reduce_min(&x, axes, false).expect("min").shape, rank0);
                // ...and one with no shortcut at all, which must not diverge.
                assert_eq!(reduce_l1(&x, axes, false).expect("l1").shape, rank0);

                // `keepdims=1` is deliberately unchanged: rank is preserved.
                let kd = reduce_sum(&x, axes, true).expect("keepdims reduce_sum runs");
                assert_eq!(kd.shape, vec![1; shape.len()]);
                assert_eq!(kd.data, vec![276.0]);
            }
        }
    }

    #[test]
    fn zero_size_axis_does_not_panic() {
        // A non-reduced axis of size 0 makes `reduce_with`'s `out_n == 0`
        // (no `.max(1)` there, unlike the `_into` twin) -- the rewrite must
        // not introduce a division keyed on `out_n` that would divide by
        // zero in that case. (It doesn't: the odometer just walks
        // `x.data.iter()`, which is empty here, so the loop body never runs.)
        let x = Tensor::new(Vec::new(), vec![0, 3]);
        let y = reduce_sum(&x, &[1], false).expect("reduce over a zero-size tensor must not panic");
        assert_eq!(y.shape, vec![0]);
        assert_eq!(y.data.len(), 0);
    }

    /// [W2-perf-misc / a6-9 hardening] `Tensor::new` only checks `data.len() ==
    /// shape.product()` via `debug_assert!` (see its doc comment) -- release builds can
    /// construct an inconsistent `Tensor` (e.g. from an internal bug elsewhere in the
    /// pipeline; this does *not* model malformed model bytes, which go through the
    /// unconditionally-validating `Tensor::try_new` instead). Before the odometer rewrite,
    /// the per-element loop bound was `0..x.numel()` -- and `Tensor::numel()` in this crate
    /// is `data.len()`, not `shape.product()` -- so old and new code both originally walked
    /// every element of an oversized `data` buffer; the difference is *how* each decodes the
    /// "extra" trailing elements once `in_idx`/`coord` runs past what `shape` describes. The
    /// old div/mod decode could compute an out-of-range coordinate for a non-reduced axis and
    /// panic indexing `out`/`acc`; the new odometer instead wraps that coordinate cleanly back
    /// to 0 and keeps going, silently re-accumulating the extra elements into buckets 0.. --
    /// no panic, but wrong output. `.take(shape_numel)` in `reduce_with`/`reduce_with_into`
    /// closes both: the walk stops at exactly the shape-implied element count, so a
    /// too-long `data` buffer's trailing elements are simply never visited, and the result
    /// matches what a correctly-shaped tensor built from just that prefix would produce.
    ///
    /// Deliberately 3D with `axes` always shorter than `ndim`: a *full* reduction (`axes.len()
    /// == ndim`) instead takes the `#[cfg(feature = "simd")] is_full_reduction` shortcut in
    /// `reduce_sum`/`reduce_mean`/etc. above, which calls `crate::simd_ops::simd_reduce_*` on
    /// `shape_bounded_data(x)` rather than `&x.data` directly for the identical reason (see
    /// `shape_bounded_data`'s doc comment at the top of this file) -- covered separately by
    /// [`full_reduction_simd_shortcut_also_respects_shape_bound`] below, since it is a distinct
    /// code path (a different function in a different, not-owned module, `simd_ops`) even though
    /// it is the same underlying `Tensor` invariant.
    #[test]
    fn oversized_data_buffer_is_capped_at_shape_product_not_corrupted() {
        let shape = vec![2usize, 3, 1];
        let shape_numel = 6usize;
        // `data` has 9 elements but `shape` only describes 6 -- constructed via the struct
        // literal (both fields are `pub`) specifically to bypass `Tensor::new`'s
        // debug_assert, since that is exactly the release-build-only state under test.
        let mut data: Vec<f32> = (0..shape_numel).map(|i| i as f32 + 1.0).collect();
        data.extend([1000.0, 2000.0, 3000.0]); // extra, out-of-contract trailing elements
        let x = Tensor {
            data: data.clone(),
            shape: shape.clone(),
        };
        // The "correctly shaped" baseline: same leading 6 elements, no oversize.
        let x_well_formed = Tensor::new(data[..shape_numel].to_vec(), shape.clone());

        for axes in [vec![0i64], vec![1], vec![0, 2]] {
            let got = reduce_sum(&x, &axes, false).expect("must not panic on oversized data");
            let want = reduce_sum(&x_well_formed, &axes, false).expect("well-formed baseline");
            assert_eq!(
                got.data, want.data,
                "axes={axes:?}: oversized-data result must match the shape-implied prefix, \
                 not silently fold the extra trailing elements into the output"
            );
            assert_eq!(got.shape, want.shape);

            let (_, out_n) = reduce_output_shape(&x, &axes, false)
                .expect("reduce_output_shape should succeed for valid in-range axes");
            let mut out = vec![-999.0f32; out_n];
            let got_into = reduce_sum_into(&x, &axes, false, &mut out)
                .expect("must not panic on oversized data (_into path)");
            assert_eq!(&out[..out_n], want.data.as_slice());
            assert_eq!(got_into, want.shape);
        }
    }

    /// [W2-perf-misc / a6-9 hardening, SIMD shortcut] Companion to
    /// [`oversized_data_buffer_is_capped_at_shape_product_not_corrupted`] above: a *full*
    /// reduction (here, `axes = []`, the ONNX "reduce over every axis" default, which
    /// `is_full_reduction` accepts immediately) takes the `simd_reduce_*(shape_bounded_data(x))`
    /// shortcut in `reduce_sum`/`reduce_mean`/`reduce_max`/`reduce_min` instead of
    /// `reduce_with`'s odometer, so it needs its own regression test: `shape_bounded_data` is
    /// what stops that shortcut from summing an oversized tensor's out-of-contract trailing
    /// elements too. Only meaningful under the `simd` feature -- without it, "full reduction"
    /// just falls through to the already-covered `reduce_with` path.
    #[test]
    #[cfg(feature = "simd")]
    fn full_reduction_simd_shortcut_also_respects_shape_bound() {
        let shape = vec![2usize, 3];
        let shape_numel = 6usize;
        let mut data: Vec<f32> = (0..shape_numel).map(|i| i as f32 + 1.0).collect();
        data.extend([1000.0, 2000.0, 3000.0]); // extra, out-of-contract trailing elements
        let x = Tensor {
            data: data.clone(),
            shape: shape.clone(),
        };
        let x_well_formed = Tensor::new(data[..shape_numel].to_vec(), shape.clone());

        for keepdims in [false, true] {
            let got_sum = reduce_sum(&x, &[], keepdims).expect("must not panic");
            let want_sum = reduce_sum(&x_well_formed, &[], keepdims).expect("baseline");
            assert_eq!(
                got_sum.data, want_sum.data,
                "keepdims={keepdims}: full-reduction SIMD shortcut must not sum the oversized \
                 tensor's out-of-contract trailing elements"
            );
            assert_eq!(got_sum.shape, want_sum.shape);

            let got_mean = reduce_mean(&x, &[], keepdims).expect("must not panic");
            let want_mean = reduce_mean(&x_well_formed, &[], keepdims).expect("baseline");
            assert_eq!(got_mean.data, want_mean.data, "keepdims={keepdims}");

            let got_max = reduce_max(&x, &[], keepdims).expect("must not panic");
            let want_max = reduce_max(&x_well_formed, &[], keepdims).expect("baseline");
            assert_eq!(got_max.data, want_max.data, "keepdims={keepdims}");

            let got_min = reduce_min(&x, &[], keepdims).expect("must not panic");
            let want_min = reduce_min(&x_well_formed, &[], keepdims).expect("baseline");
            assert_eq!(got_min.data, want_min.data, "keepdims={keepdims}");
        }
    }
}
