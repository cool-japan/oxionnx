use oxionnx_core::Tensor;

// ── ArgMax / ArgMin ─────────────────────────────────────────────────────────

/// Index of the maximum value along `axis`. `select_last_index` (opset 12)
/// picks the last, rather than the first, occurrence of a tied maximum.
pub fn arg_max(
    x: &Tensor,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
) -> Result<Tensor, String> {
    arg_reduce(x, axis, keepdims, true, select_last_index)
}

/// Index of the minimum value along `axis`. `select_last_index` (opset 12)
/// picks the last, rather than the first, occurrence of a tied minimum.
pub fn arg_min(
    x: &Tensor,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
) -> Result<Tensor, String> {
    arg_reduce(x, axis, keepdims, false, select_last_index)
}

/// True when the candidate `v` should replace `best_val` at a later index:
/// strict `>`/`<` keeps the first occurrence of the extremum, `>=`/`<=` keeps
/// the last (opset-12 `select_last_index`). NaN compares false either way (a
/// pre-existing property of `>`/`<`/`>=`/`<=` against NaN), so NaNs are still
/// skipped rather than propagated -- unchanged by this attribute.
fn is_better(v: f32, best_val: f32, find_max: bool, select_last_index: bool) -> bool {
    match (find_max, select_last_index) {
        (true, false) => v > best_val,
        (true, true) => v >= best_val,
        (false, false) => v < best_val,
        (false, true) => v <= best_val,
    }
}

fn arg_reduce(
    x: &Tensor,
    axis: i64,
    keepdims: bool,
    find_max: bool,
    select_last_index: bool,
) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "arg_reduce: axis {ax} out of range for {ndim}D tensor"
        ));
    }

    // No `.max(1)` on either product: a legitimate size-0 dimension in the leading or trailing
    // slice must stay 0 so the loops below simply do not run. Clamping it to 1 made the walk
    // index an empty `x.data` — `arg_max(Tensor::new(vec![], vec![0,3]), axis=1)` panicked with
    // index-out-of-bounds rather than returning the empty result NumPy gives (shape `(0,)`).
    // The empty product is already 1, so a rank-1 input (`x.shape[..0]`) is unaffected.
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];
    // A zero-length *reduction* axis is the separate degenerate case, and it is not rescued by
    // dropping the clamps above: `outer`/`inner` can both be legitimately non-zero while
    // `axis_len == 0` (e.g. `[3,0]` along axis 1, or `[0,3]` along axis 0), and the seed read
    // `x.data[o * axis_len * inner + i]` then indexes an empty buffer. There is no answer to
    // report — the extremum of an empty sequence is undefined — so this is a typed error, the
    // same way NumPy raises "attempt to get argmax of an empty sequence" for every one of these
    // shapes, rather than a fabricated index 0.
    if axis_len == 0 {
        return Err(format!(
            "arg_reduce: axis {ax} has length 0; the index of an extremum is undefined over an empty axis"
        ));
    }

    let mut result = vec![0.0f32; outer * inner];

    for o in 0..outer {
        for i in 0..inner {
            let base = x.data[o * axis_len * inner + i];
            let (mut best_val, mut best_idx) = (base, 0usize);
            for j in 1..axis_len {
                let v = x.data[o * axis_len * inner + j * inner + i];
                if is_better(v, best_val, find_max, select_last_index) {
                    best_val = v;
                    best_idx = j;
                }
            }
            result[o * inner + i] = best_idx as f32;
        }
    }

    if keepdims {
        let mut out_shape = x.shape.clone();
        out_shape[ax] = 1;
        Ok(Tensor::new(result, out_shape))
    } else {
        // Dropping the only axis of a 1-D input leaves the **empty** shape — a genuine rank-0
        // tensor, not the rank-1 `[1]` this used to promote it to. NumPy:
        // `np.argmax(np.array([3.,9.,4.]), axis=0).shape == ()`.
        let mut final_shape = x.shape.clone();
        final_shape.remove(ax);
        Ok(Tensor::new(result, final_shape))
    }
}

// ── Zero-copy _into variants ─────────────────────────────────────────────────

/// Compute output shape and element count for arg_max / arg_min.
///
/// Validates `axis` is in range *before* using it to index or remove from
/// the shape below. Both `s[ax] = 1` (keepdims) and `s.remove(ax)` (not
/// keepdims) panic when `ax >= s.len()` -- an out-of-range `axis` used to
/// reach them unchecked, since this function trusted callers to have
/// validated `axis` themselves. `arg_reduce_into` (below, in this file) does
/// validate before calling here, so this repeats that check; the callers
/// that mattered were the `execute_into_slots` paths in
/// `registry/math_ops/reduce.rs`, which called this function to size the
/// output slot *before* calling `arg_reduce_into` at all -- so the unchecked
/// version could panic on a bad axis before the kernel's own check ever ran.
pub(crate) fn arg_output_shape(
    x: &Tensor,
    axis: i64,
    keepdims: bool,
) -> Result<(Vec<usize>, usize), String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "arg_reduce: axis {ax} out of range for {ndim}D tensor"
        ));
    }
    // No `.max(1)` on `len` in either branch. At rank 0 it was merely redundant (the empty
    // shape's product is the empty product 1 already), but for a genuinely zero-size output
    // it was wrong: it floored a 0 to 1, so the registry's slot path resized `slots[0].data`
    // to one element while `shape` said zero. That slot is malformed, and observably so —
    // this crate's `Tensor::numel()` returns `data.len()`, so a following `Size` node would
    // report 1 element for a tensor whose shape describes none. `len` is exactly the output
    // shape product, which is also exactly the `outer * inner` write count `arg_reduce_into`
    // performs, so a 0 here means "write nothing", not "buffer too small".
    if keepdims {
        let mut s = x.shape.clone();
        s[ax] = 1;
        let len = s.iter().product::<usize>();
        Ok((s, len))
    } else {
        // Rank 0 (the empty shape) when the only axis is dropped — see `arg_reduce`.
        let mut s = x.shape.clone();
        s.remove(ax);
        let len = s.iter().product::<usize>();
        Ok((s, len))
    }
}

/// Like arg_reduce but writes directly into `out`.
pub(crate) fn arg_reduce_into(
    x: &Tensor,
    axis: i64,
    keepdims: bool,
    find_max: bool,
    select_last_index: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "arg_reduce: axis {ax} out of range for {ndim}D tensor"
        ));
    }
    // Unclamped products plus a zero-length-axis guard, for the same reasons as `arg_reduce`
    // (which see): the clamp turned a legitimate 0 into 1 and walked an empty buffer, and a
    // zero-length reduction axis has no defined extremum index.
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];
    if axis_len == 0 {
        return Err(format!(
            "arg_reduce: axis {ax} has length 0; the index of an extremum is undefined over an empty axis"
        ));
    }
    for o in 0..outer {
        for i in 0..inner {
            let base = x.data[o * axis_len * inner + i];
            let (mut best_val, mut best_idx) = (base, 0usize);
            for j in 1..axis_len {
                let v = x.data[o * axis_len * inner + j * inner + i];
                if is_better(v, best_val, find_max, select_last_index) {
                    best_val = v;
                    best_idx = j;
                }
            }
            out[o * inner + i] = best_idx as f32;
        }
    }
    let (shape, _) = arg_output_shape(x, axis, keepdims)?;
    Ok(shape)
}

/// Like cumsum but writes directly into `out` (same length and shape as x).
pub(crate) fn cumsum_into(
    x: &Tensor,
    axis: i64,
    exclusive: bool,
    reverse: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("cumsum: axis {ax} out of range for {ndim}D tensor"));
    }
    // Unclamped, as in `arg_reduce`: `cumsum(Tensor::new(vec![], vec![0,3]), axis=1)` panicked
    // with index-out-of-bounds because the clamp turned the leading `0` into a `1` and ran the
    // loop over an empty buffer. NumPy returns the input shape unchanged (`(0,3)`), which is
    // what the (now un-entered) loops plus the seed copy below produce. Unlike `arg_reduce` a
    // zero-length *axis* needs no guard here: the inner `for j in 0..axis_len` simply does not
    // run, and an empty prefix sum is correctly the empty result.
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];
    // seed out with x values so any unvisited position retains input
    out.copy_from_slice(&x.data);
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f32;
            if !reverse {
                for j in 0..axis_len {
                    let idx = o * axis_len * inner + j * inner + i;
                    let val = x.data[idx];
                    if exclusive {
                        out[idx] = acc;
                        acc += val;
                    } else {
                        acc += val;
                        out[idx] = acc;
                    }
                }
            } else {
                for j in (0..axis_len).rev() {
                    let idx = o * axis_len * inner + j * inner + i;
                    let val = x.data[idx];
                    if exclusive {
                        out[idx] = acc;
                        acc += val;
                    } else {
                        acc += val;
                        out[idx] = acc;
                    }
                }
            }
        }
    }
    Ok(x.shape.clone())
}

// ── CumSum ──────────────────────────────────────────────────────────────────

/// Prefix sum (inclusive or exclusive) along `axis`.
pub fn cumsum(x: &Tensor, axis: i64, exclusive: bool, reverse: bool) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("cumsum: axis {ax} out of range for {ndim}D tensor"));
    }

    // Unclamped — see `cumsum_into` above for the panic this removes and the NumPy reference.
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];

    let mut data = x.data.clone();

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f32;
            if !reverse {
                for j in 0..axis_len {
                    let idx = o * axis_len * inner + j * inner + i;
                    let val = x.data[idx];
                    if exclusive {
                        data[idx] = acc;
                        acc += val;
                    } else {
                        acc += val;
                        data[idx] = acc;
                    }
                }
            } else {
                for j in (0..axis_len).rev() {
                    let idx = o * axis_len * inner + j * inner + i;
                    let val = x.data[idx];
                    if exclusive {
                        data[idx] = acc;
                        acc += val;
                    } else {
                        acc += val;
                        data[idx] = acc;
                    }
                }
            }
        }
    }

    Ok(Tensor::new(data, x.shape.clone()))
}

// ── Range ───────────────────────────────────────────────────────────────────

/// Generate `[start, start+delta, ...]` up to (not including) `limit`.
pub fn range(start: f32, limit: f32, delta: f32) -> Result<Tensor, String> {
    if delta == 0.0 {
        return Err("range: delta cannot be zero".into());
    }
    let count = if delta > 0.0 {
        ((limit - start) / delta).ceil().max(0.0) as usize
    } else {
        ((start - limit) / (-delta)).ceil().max(0.0) as usize
    };
    let data: Vec<f32> = (0..count).map(|i| start + i as f32 * delta).collect();
    Ok(Tensor::new(data, vec![count]))
}
