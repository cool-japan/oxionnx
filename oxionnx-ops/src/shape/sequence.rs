//! Sequence shape operations: concat, slice, pad, split, tile.

use oxionnx_core::Tensor;

/// Concatenate tensors along the given axis.
pub fn concat(tensors: &[&Tensor], axis: i64) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("concat: no tensors".into());
    }
    let ndim = tensors[0].ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
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
    let outer: usize = tensors[0].shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = tensors[0].shape[ax + 1..].iter().product::<usize>().max(1);
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

/// Slice tensor along given axes with start/end/step.
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
    let mut dim_slices: Vec<(usize, usize, usize)> = x.shape.iter().map(|&d| (0, d, 1)).collect();
    for (i, &ax) in axes.iter().enumerate() {
        let ax = if ax < 0 {
            (ax + ndim as i64) as usize
        } else {
            ax as usize
        };
        let dim = x.shape[ax];
        let step = steps[i].max(1) as usize;
        let start = starts[i].clamp(0, dim as i64) as usize;
        let end = ends[i].clamp(0, dim as i64) as usize;
        dim_slices[ax] = (start, end, step);
    }
    let out_shape: Vec<usize> = dim_slices
        .iter()
        .map(|&(s, e, step)| e.saturating_sub(s).div_ceil(step))
        .collect();
    let out_n: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(out_n);
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
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            let (start, _, step) = dim_slices[d];
            in_idx += (start + coord * step) * strides[d];
        }
        out.push(x.data[in_idx]);
    }
    Ok(Tensor::new(out, out_shape))
}

/// Pad tensor with constant, reflect, or edge values.
/// pads format: [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
pub fn pad(input: &Tensor, pads: &[i64], mode: &str, constant_value: f32) -> Tensor {
    let ndim = input.ndim();
    assert!(pads.len() == 2 * ndim, "pad: pads length must be 2 * ndim");
    let begin: Vec<usize> = pads[..ndim].iter().map(|&p| p.max(0) as usize).collect();
    let end: Vec<usize> = pads[ndim..].iter().map(|&p| p.max(0) as usize).collect();
    let out_shape: Vec<usize> = (0..ndim)
        .map(|d| input.shape[d] + begin[d] + end[d])
        .collect();
    let out_n: usize = out_shape.iter().product();
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
        _ => {
            // "constant" mode (default): fill with constant_value, copy input into center
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
    Tensor::new(out, out_shape)
}

/// Split tensor along axis into chunks of given sizes.
pub fn split(x: &Tensor, axis: i64, split_sizes: &[usize]) -> Result<Vec<Tensor>, String> {
    if split_sizes.is_empty() {
        return Err("split: split_sizes must not be empty".into());
    }
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("split: axis {ax} out of range for {ndim}D tensor"));
    }
    let axis_len = x.shape[ax];
    let total: usize = split_sizes.iter().sum();
    if total != axis_len {
        return Err(format!("split: sizes sum {total} != axis len {axis_len}"));
    }
    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
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
