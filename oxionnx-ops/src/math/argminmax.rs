use oxionnx_core::Tensor;

// ── ArgMax / ArgMin ─────────────────────────────────────────────────────────

/// Index of the maximum value along `axis`.
pub fn arg_max(x: &Tensor, axis: i64, keepdims: bool) -> Result<Tensor, String> {
    arg_reduce(x, axis, keepdims, true)
}

/// Index of the minimum value along `axis`.
pub fn arg_min(x: &Tensor, axis: i64, keepdims: bool) -> Result<Tensor, String> {
    arg_reduce(x, axis, keepdims, false)
}

fn arg_reduce(x: &Tensor, axis: i64, keepdims: bool, find_max: bool) -> Result<Tensor, String> {
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

    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
    let axis_len = x.shape[ax];

    let mut result = vec![0.0f32; outer * inner];

    for o in 0..outer {
        for i in 0..inner {
            let base = x.data[o * axis_len * inner + i];
            let (mut best_val, mut best_idx) = (base, 0usize);
            for j in 1..axis_len {
                let v = x.data[o * axis_len * inner + j * inner + i];
                let better = if find_max { v > best_val } else { v < best_val };
                if better {
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
        let mut final_shape = x.shape.clone();
        final_shape.remove(ax);
        let final_shape = if final_shape.is_empty() {
            vec![1]
        } else {
            final_shape
        };
        Ok(Tensor::new(result, final_shape))
    }
}

// ── Zero-copy _into variants ─────────────────────────────────────────────────

/// Compute output shape and element count for arg_max / arg_min.
pub(crate) fn arg_output_shape(x: &Tensor, axis: i64, keepdims: bool) -> (Vec<usize>, usize) {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if keepdims {
        let mut s = x.shape.clone();
        s[ax] = 1;
        let len = s.iter().product::<usize>().max(1);
        (s, len)
    } else {
        let mut s = x.shape.clone();
        s.remove(ax);
        let s = if s.is_empty() { vec![1] } else { s };
        let len = s.iter().product::<usize>().max(1);
        (s, len)
    }
}

/// Like arg_reduce but writes directly into `out`.
pub(crate) fn arg_reduce_into(
    x: &Tensor,
    axis: i64,
    keepdims: bool,
    find_max: bool,
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
    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
    let axis_len = x.shape[ax];
    for o in 0..outer {
        for i in 0..inner {
            let base = x.data[o * axis_len * inner + i];
            let (mut best_val, mut best_idx) = (base, 0usize);
            for j in 1..axis_len {
                let v = x.data[o * axis_len * inner + j * inner + i];
                let better = if find_max { v > best_val } else { v < best_val };
                if better {
                    best_val = v;
                    best_idx = j;
                }
            }
            out[o * inner + i] = best_idx as f32;
        }
    }
    let (shape, _) = arg_output_shape(x, axis, keepdims);
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
    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
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

    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
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
