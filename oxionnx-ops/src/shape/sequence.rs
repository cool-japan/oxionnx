//! Sequence shape operations: concat, slice, pad, split, tile.

use oxionnx_core::Tensor;

use super::basic::normalize_axis;

/// Concatenate tensors along the given axis.
pub fn concat(tensors: &[&Tensor], axis: i64) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("concat: no tensors".into());
    }
    let ndim = tensors[0].ndim();
    let ax = normalize_axis(axis, ndim)?;
    for t in tensors.iter().skip(1) {
        if t.ndim() != ndim {
            return Err("concat: tensors must have same ndim".into());
        }
        for d in 0..ndim {
            if d != ax && t.shape[d] != tensors[0].shape[d] {
                return Err(format!("concat: shape mismatch at dim {d}"));
            }
        }
    }
    let mut out_shape = tensors[0].shape.clone();
    out_shape[ax] = tensors.iter().map(|t| t.shape[ax]).sum();
    let mut out = Vec::with_capacity(out_shape.iter().product());
    // Empty-slice products are already 1 (the multiplicative identity) with no clamping needed;
    // clamping a *genuinely* zero-size leading/trailing dim up to 1 would corrupt the
    // shape/data-length invariant for legitimately empty tensors (e.g. a [0, 3] input).
    let outer: usize = tensors[0].shape[..ax].iter().product();
    let inner: usize = tensors[0].shape[ax + 1..].iter().product();
    for o in 0..outer {
        for t in tensors {
            let seg = t.shape[ax];
            for s in 0..seg {
                let src_start = (o * t.shape[ax] + s) * inner;
                out.extend_from_slice(&t.data[src_start..src_start + inner]);
            }
        }
    }
    Ok(Tensor::new(out, out_shape))
}

/// Slice tensor along given axes with start/end/step, per ONNX `Slice` (opset 10+) semantics.
///
/// For each axis: a negative `start`/`end` counts from the end of that axis (`value + dim`);
/// the result is then clamped to `[0, dim]` for a positive step or `[-1, dim-1]` for a negative
/// step (per spec — this is *not* symmetric, and is what makes `INT64_MAX`/`INT64_MIN`
/// sentinels work as "to the natural end" markers in either direction). `steps` may be negative
/// to walk an axis in reverse; a step of `0` is rejected. Axes not named in `axes` pass through
/// unsliced (full range, step 1).
pub fn slice(
    x: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let default_axes: Vec<i64> = (0..starts.len() as i64).collect();
    let axes = axes.unwrap_or(&default_axes);
    let default_steps: Vec<i64> = vec![1; starts.len()];
    let steps = steps.unwrap_or(&default_steps);

    if starts.len() != axes.len() || ends.len() != axes.len() || steps.len() != axes.len() {
        return Err(format!(
            "slice: starts/ends/axes/steps length mismatch (starts={}, ends={}, axes={}, steps={})",
            starts.len(),
            ends.len(),
            axes.len(),
            steps.len()
        ));
    }

    // Per-axis (start, end, step) as signed values; axes not mentioned default to a full
    // forward pass (0..dim, step 1).
    let mut dim_start: Vec<i64> = vec![0; ndim];
    let mut dim_end: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
    let mut dim_step: Vec<i64> = vec![1; ndim];

    for (i, &raw_ax) in axes.iter().enumerate() {
        let ax = normalize_axis(raw_ax, ndim)?;
        let dim = x.shape[ax] as i64;
        let step = steps[i];
        if step == 0 {
            return Err(format!("slice: steps[{i}] must not be 0"));
        }
        if step == i64::MIN {
            return Err(format!("slice: steps[{i}] out of range"));
        }
        let mut start = starts[i];
        let mut end = ends[i];
        // Negative start/end count from the end of the axis. `dim` is bounded by real tensor
        // sizes, so `start + dim` cannot overflow even for `start == i64::MIN`; `saturating_add`
        // is defense-in-depth against adversarial input rather than a normal-path necessity.
        if start < 0 {
            start = start.saturating_add(dim);
        }
        if end < 0 {
            end = end.saturating_add(dim);
        }
        if step > 0 {
            start = start.clamp(0, dim);
            end = end.clamp(0, dim);
        } else {
            start = start.clamp(-1, dim - 1);
            end = end.clamp(-1, dim - 1);
        }
        dim_start[ax] = start;
        dim_end[ax] = end;
        dim_step[ax] = step;
    }

    let out_shape: Vec<usize> = (0..ndim)
        .map(|d| {
            let (start, end, step) = (dim_start[d], dim_end[d], dim_step[d]);
            if step > 0 {
                if end <= start {
                    0
                } else {
                    ((end - start + step - 1) / step) as usize
                }
            } else {
                let neg_step = -step;
                if start <= end {
                    0
                } else {
                    ((start - end + neg_step - 1) / neg_step) as usize
                }
            }
        })
        .collect();

    let out_n: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(out_n);
    if out_n == 0 {
        return Ok(Tensor::new(out, out_shape));
    }

    let mut strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        strides[i] = s;
        s *= x.shape[i];
    }
    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    for out_idx in 0..out_n {
        let mut rem = out_idx;
        // Signed accumulator: a negative step walks `dim_start` downward, and by construction
        // (the clamp + out_shape length above) every coordinate visited stays in `[0, dim)`.
        let mut in_idx: i64 = 0;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            let in_coord = dim_start[d] + coord as i64 * dim_step[d];
            in_idx += in_coord * strides[d] as i64;
        }
        out.push(x.data[in_idx as usize]);
    }
    Ok(Tensor::new(out, out_shape))
}

/// Pad tensor with constant, reflect, edge, or wrap values, honoring the opset-18 `axes` input.
///
/// `pads` holds `[begin_0, begin_1, ..., end_0, end_1, ...]` in `axes` order: when `axes` is
/// `None` that's every axis of `input` in input order (`pads.len() == 2 * ndim`); when `axes`
/// is `Some`, `pads.len() == 2 * axes.len()` and any axis not named gets zero padding.
///
/// A negative pad value *crops* that many elements off the corresponding edge instead of
/// adding to it (ONNX allows this since opset 11).
pub fn pad_axes(
    input: &Tensor,
    pads: &[i64],
    mode: &str,
    constant_value: f32,
    axes: Option<&[i64]>,
) -> Result<Tensor, String> {
    let ndim = input.ndim();
    if !matches!(mode, "constant" | "reflect" | "edge" | "wrap") {
        return Err(format!(
            "pad: unsupported mode '{mode}' (expected constant, reflect, edge, or wrap)"
        ));
    }
    let axes_norm: Vec<usize> = match axes {
        Some(a) => a
            .iter()
            .map(|&ax| normalize_axis(ax, ndim))
            .collect::<Result<_, _>>()?,
        None => (0..ndim).collect(),
    };
    if pads.len() != 2 * axes_norm.len() {
        return Err(format!(
            "pad: pads length {} must be 2 * len(axes) ({})",
            pads.len(),
            2 * axes_norm.len()
        ));
    }

    // Expand the (possibly axes-compacted) pads into a full per-dimension (begin, end) pair;
    // axes not named in `axes` get zero padding on both sides.
    let mut begin: Vec<i64> = vec![0; ndim];
    let mut end: Vec<i64> = vec![0; ndim];
    for (i, &ax) in axes_norm.iter().enumerate() {
        begin[ax] = pads[i];
        end[ax] = pads[axes_norm.len() + i];
    }

    // Output extent per dim, allowing negative pads to crop the input.
    let mut out_shape = vec![0usize; ndim];
    for d in 0..ndim {
        let in_dim = input.shape[d] as i64;
        let out_dim = in_dim + begin[d] + end[d];
        if out_dim < 0 {
            return Err(format!(
                "pad: pads crop dim {d} below zero (input {in_dim}, begin {}, end {})",
                begin[d], end[d]
            ));
        }
        out_shape[d] = out_dim as usize;
    }
    let out_n: usize = out_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| "pad: output element count overflows".to_string())?;

    if out_n == 0 {
        return Ok(Tensor::new(Vec::new(), out_shape));
    }
    // reflect/edge/wrap all need at least one source element per padded axis; an empty input
    // axis being asked to grow has no data to reflect/repeat/extend from (constant mode is
    // fine either way, since every output position is simply "outside").
    if mode != "constant" && input.numel() == 0 {
        return Err(format!(
            "pad: cannot {mode}-pad an axis of a 0-element input"
        ));
    }

    let mut in_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        in_strides[i] = s;
        s *= input.shape[i];
    }
    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    let mut out = vec![constant_value; out_n];
    match mode {
        "reflect" => {
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                let mut valid = true;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let in_coord_signed = out_coord as isize - begin[d] as isize;
                    let dim = input.shape[d] as isize;
                    let mut c = in_coord_signed;
                    if dim <= 1 {
                        c = 0;
                    } else {
                        let period = 2 * (dim - 1);
                        c = c.rem_euclid(period);
                        if c >= dim {
                            c = period - c;
                        }
                    }
                    if c < 0 || c >= dim {
                        valid = false;
                        break;
                    }
                    in_idx += c as usize * in_strides[d];
                }
                if valid {
                    *out_val = input.data[in_idx];
                }
            }
        }
        "edge" => {
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let in_coord = (out_coord as isize - begin[d] as isize)
                        .max(0)
                        .min(input.shape[d] as isize - 1)
                        as usize;
                    in_idx += in_coord * in_strides[d];
                }
                *out_val = input.data[in_idx];
            }
        }
        "wrap" => {
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let in_coord_signed = out_coord as isize - begin[d] as isize;
                    let dim = input.shape[d] as isize;
                    let c = in_coord_signed.rem_euclid(dim);
                    in_idx += c as usize * in_strides[d];
                }
                *out_val = input.data[in_idx];
            }
        }
        _ => {
            // "constant" mode: fill already done above; copy input into the interior. A
            // negative `begin`/`end` (crop) is handled by the exact same in-bounds test: the
            // shifted coordinate simply starts further into `input` than `out_coord` alone
            // would suggest.
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                let mut inside = true;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let in_coord = out_coord as isize - begin[d] as isize;
                    if in_coord < 0 || in_coord >= input.shape[d] as isize {
                        inside = false;
                        break;
                    }
                    in_idx += in_coord as usize * in_strides[d];
                }
                if inside {
                    *out_val = input.data[in_idx];
                }
            }
        }
    }
    Ok(Tensor::new(out, out_shape))
}

/// Legacy 4-argument Pad entry point, equivalent to [`pad_axes`] with `axes = None` (the
/// `pads` slice covers every input axis, in order).
///
/// Kept `Tensor`-returning (rather than `Result`) for source compatibility with existing
/// callers outside this crate's `shape` module. On input `pad_axes` would reject — a `pads`
/// length that doesn't match `2 * ndim`, or an unrecognized `mode` — this returns the input
/// tensor unchanged (a documented no-op) instead of panicking. New callers, and anyone able to
/// thread the opset-18 `axes` input through, should call [`pad_axes`] directly: it reports
/// those cases as a proper `Err` instead of silently no-op'ing.
pub fn pad(input: &Tensor, pads: &[i64], mode: &str, constant_value: f32) -> Tensor {
    pad_axes(input, pads, mode, constant_value, None).unwrap_or_else(|_| input.clone())
}

/// Split tensor along axis into chunks of given sizes.
pub fn split(x: &Tensor, axis: i64, split_sizes: &[usize]) -> Result<Vec<Tensor>, String> {
    if split_sizes.is_empty() {
        return Err("split: split_sizes must not be empty".into());
    }
    let ndim = x.ndim();
    let ax = normalize_axis(axis, ndim)?;
    let axis_len = x.shape[ax];
    let total: usize = split_sizes.iter().sum();
    if total != axis_len {
        return Err(format!("split: sizes sum {total} != axis len {axis_len}"));
    }
    // See the comment in `concat`: empty-slice products are already 1; clamping would corrupt
    // a genuinely zero-size leading/trailing dim.
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let mut results = Vec::with_capacity(split_sizes.len());
    let mut start = 0usize;
    for &chunk in split_sizes {
        let n_out = outer * chunk * inner;
        let mut out = Vec::with_capacity(n_out);
        for o in 0..outer {
            for j in start..start + chunk {
                let src = o * axis_len * inner + j * inner;
                out.extend_from_slice(&x.data[src..src + inner]);
            }
        }
        let mut out_shape = x.shape.clone();
        out_shape[ax] = chunk;
        results.push(Tensor::new(out, out_shape));
        start += chunk;
    }
    Ok(results)
}

/// Repeat tensor along each axis N times.
pub fn tile(x: &Tensor, repeats: &[usize]) -> Result<Tensor, String> {
    let ndim = x.ndim();
    if repeats.len() != ndim {
        return Err(format!(
            "tile: repeats len {} != tensor ndim {}",
            repeats.len(),
            ndim
        ));
    }
    let out_shape: Vec<usize> = x.shape.iter().zip(repeats).map(|(&d, &r)| d * r).collect();
    let out_n: usize = out_shape.iter().product();
    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    let mut in_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        in_strides[i] = s;
        s *= x.shape[i];
    }
    let mut out = vec![0.0f32; out_n];
    for (out_idx, out_val) in out.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += (coord % x.shape[d]) * in_strides[d];
        }
        *out_val = x.data[in_idx];
    }
    Ok(Tensor::new(out, out_shape))
}
