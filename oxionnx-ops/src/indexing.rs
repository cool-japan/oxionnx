use oxionnx_core::Tensor;

/// Gather elements from `data` along `axis` using `indices`.
///
/// For axis=0 (embedding lookup):
///   output[i, j, ...] = data[indices[i, j, ...], j, ...]
///
/// For axis=1:
///   output[i, j, k, ...] = data[i, indices[i, j, k, ...], k, ...]
pub fn gather(data: &Tensor, indices: &Tensor, axis: i64) -> Result<Tensor, String> {
    let ndim = data.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("gather: axis {ax} out of range for {ndim}D tensor"));
    }

    // Output shape = data.shape[:ax] + indices.shape + data.shape[ax+1:]
    let out_shape: Vec<usize> = data.shape[..ax]
        .iter()
        .copied()
        .chain(indices.shape.iter().copied())
        .chain(data.shape[ax + 1..].iter().copied())
        .collect();

    let outer: usize = data.shape[..ax].iter().product::<usize>().max(1);
    let axis_size = data.shape[ax];
    let inner: usize = data.shape[ax + 1..].iter().product::<usize>().max(1);
    let idx_n = indices.numel();

    let mut out = vec![0.0f32; out_shape.iter().product()];

    for o in 0..outer {
        for (ii, &idx_val) in indices.data.iter().enumerate() {
            let idx = idx_val as i64;
            let idx = if idx < 0 {
                (idx + axis_size as i64) as usize
            } else {
                idx as usize
            };
            if idx >= axis_size {
                return Err(format!(
                    "gather: index {idx} out of bounds for axis size {axis_size}"
                ));
            }
            let src_base = o * axis_size * inner + idx * inner;
            let dst_base = o * idx_n * inner + ii * inner;
            out[dst_base..dst_base + inner].copy_from_slice(&data.data[src_base..src_base + inner]);
        }
    }

    Ok(Tensor::new(out, out_shape))
}

/// GatherElements: gather individual elements (advanced indexing).
/// `output[i][j][k] = input[index[i][j][k]]` for axis=0
pub fn gather_elements(data: &Tensor, indices: &Tensor, axis: i64) -> Result<Tensor, String> {
    let ndim = data.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };

    if data.shape.len() != indices.shape.len() {
        return Err("gather_elements: data and indices must have same rank".into());
    }

    let out_n = indices.numel();
    let mut out = vec![0.0f32; out_n];

    // Compute strides for data
    let mut data_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        data_strides[i] = s;
        s *= data.shape[i];
    }
    let mut idx_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        idx_strides[i] = s;
        s *= indices.shape[i];
    }

    for (flat, out_val) in out.iter_mut().enumerate() {
        // Decode flat index in indices tensor
        let mut rem = flat;
        let mut data_flat = 0usize;
        for d in 0..ndim {
            let coord = rem / idx_strides[d];
            rem %= idx_strides[d];
            if d == ax {
                let idx_val = indices.data[flat] as i64;
                let idx_val = if idx_val < 0 {
                    (idx_val + data.shape[ax] as i64) as usize
                } else {
                    idx_val as usize
                };
                data_flat += idx_val * data_strides[d];
            } else {
                data_flat += coord * data_strides[d];
            }
        }
        *out_val = data.data[data_flat];
    }

    Ok(Tensor::new(out, indices.shape.clone()))
}

/// Where: select elements from x or y based on condition (bool tensor).
pub fn where_op(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(
        &Tensor::broadcast_shape(&condition.shape, &x.shape)?,
        &y.shape,
    )?;
    let n: usize = out_shape.iter().product();

    // Simple elementwise if all have same shape or scalar
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let c = condition.data[i % condition.numel()];
            if c != 0.0 {
                x.data[i % x.numel()]
            } else {
                y.data[i % y.numel()]
            }
        })
        .collect();

    Ok(Tensor::new(data, out_shape))
}

/// Expand: broadcast x to shape.
pub fn expand(x: &Tensor, shape: &[usize]) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(&x.shape, shape)?;
    let n: usize = out_shape.iter().product();

    // Pad x.shape on left
    let ndim = out_shape.len();
    let pad = ndim - x.shape.len();
    let padded: Vec<usize> = (0..pad).map(|_| 1).chain(x.shape.iter().copied()).collect();

    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    let mut in_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        in_strides[i] = if padded[i] == 1 { 0 } else { s };
        s *= padded[i];
    }

    let mut out = vec![0.0f32; n];
    for (out_idx, out_val) in out.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += coord * in_strides[d];
        }
        *out_val = x.data[in_idx];
    }
    Ok(Tensor::new(out, out_shape))
}

/// ScatterElements: for each element in `updates`, write it into `data` at the position
/// given by `indices` along `axis`.
pub fn scatter_elements(
    data: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
    axis: i64,
) -> Result<Tensor, String> {
    let ndim = data.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "scatter_elements: axis {ax} out of range for {ndim}D tensor"
        ));
    }

    let mut out = data.data.clone();

    let mut data_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        data_strides[i] = s;
        s *= data.shape[i];
    }
    let mut idx_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        idx_strides[i] = s;
        s *= indices.shape[i];
    }

    for (flat, (&idx_val, &upd_val)) in indices.data.iter().zip(updates.data.iter()).enumerate() {
        let mut rem = flat;
        let mut data_flat = 0usize;
        for d in 0..ndim {
            let coord = rem / idx_strides[d];
            rem %= idx_strides[d];
            if d == ax {
                let idx = idx_val as i64;
                let idx = if idx < 0 {
                    (idx + data.shape[ax] as i64) as usize
                } else {
                    idx as usize
                };
                data_flat += idx * data_strides[d];
            } else {
                data_flat += coord * data_strides[d];
            }
        }
        out[data_flat] = upd_val;
    }

    Ok(Tensor::new(out, data.shape.clone()))
}

/// ScatterND: updates `data` at multi-dim indices.
/// `indices` shape: `[..., k]`; `updates` shape: `indices.shape[:-1] + data.shape[k:]`
pub fn scatter_nd(data: &Tensor, indices: &Tensor, updates: &Tensor) -> Result<Tensor, String> {
    let ndim = data.ndim();
    let k = *indices
        .shape
        .last()
        .ok_or("scatter_nd: indices must be at least 1D")?;
    if k > ndim {
        return Err(format!(
            "scatter_nd: index depth {k} exceeds data ndim {ndim}"
        ));
    }

    let mut data_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        data_strides[i] = s;
        s *= data.shape[i];
    }

    let mut out = data.data.clone();
    let n_idx = indices.numel() / k;
    let inner: usize = data.shape[k..].iter().product::<usize>().max(1);

    for i in 0..n_idx {
        let idx_base = i * k;
        let mut data_flat = 0usize;
        for (j, &ds) in data_strides.iter().enumerate().take(k) {
            let coord = indices.data[idx_base + j] as usize;
            data_flat += coord * ds;
        }
        let upd_base = i * inner;
        out[data_flat..data_flat + inner]
            .copy_from_slice(&updates.data[upd_base..upd_base + inner]);
    }

    Ok(Tensor::new(out, data.shape.clone()))
}

/// QuantizeLinear: y = saturate(round(x / scale) + zero_point, int8 range)
/// Result stored as f32 with integer values in [-128, 127].
pub fn quantize_linear(
    x: &Tensor,
    y_scale: &Tensor,
    y_zero_point: Option<&Tensor>,
) -> Result<Tensor, String> {
    let zp = y_zero_point.map(|t| t.data[0]).unwrap_or(0.0);
    let scale_len = y_scale.numel();
    let data: Vec<f32> = x
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let scale = y_scale.data[i % scale_len];
            ((v / scale).round() + zp).clamp(-128.0, 127.0)
        })
        .collect();
    Ok(Tensor::new(data, x.shape.clone()))
}

/// DequantizeLinear: y = (x - zero_point) * scale
pub fn dequantize_linear(
    x: &Tensor,
    x_scale: &Tensor,
    x_zero_point: Option<&Tensor>,
) -> Result<Tensor, String> {
    let zp = x_zero_point.map(|t| t.data[0]).unwrap_or(0.0);
    let scale_len = x_scale.numel();
    let data: Vec<f32> = x
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - zp) * x_scale.data[i % scale_len])
        .collect();
    Ok(Tensor::new(data, x.shape.clone()))
}

/// GatherND: gather slices using multi-dimensional indices
/// data: any shape, indices: [..., K] where K <= data.ndim - batch_dims
/// output shape: indices.shape[:-1] + data.shape[batch_dims + K:]
pub fn gather_nd(data: &Tensor, indices: &Tensor, batch_dims: i64) -> Result<Tensor, String> {
    let bd = batch_dims as usize;
    let ind_ndim = indices.ndim();
    if ind_ndim == 0 {
        return Err("gather_nd: indices must be at least 1D".into());
    }
    let last_dim = indices.shape[ind_ndim - 1]; // K

    if bd + last_dim > data.ndim() {
        return Err(format!(
            "gather_nd: batch_dims ({bd}) + last_index_dim ({last_dim}) exceeds data ndim ({})",
            data.ndim()
        ));
    }

    // Compute output shape
    let mut out_shape: Vec<usize> = Vec::new();
    // batch dims from indices
    for d in 0..bd {
        out_shape.push(indices.shape[d]);
    }
    // index dims (all but last) from indices, excluding batch
    for d in bd..ind_ndim - 1 {
        out_shape.push(indices.shape[d]);
    }
    // slice dims from data
    for d in bd + last_dim..data.ndim() {
        out_shape.push(data.shape[d]);
    }

    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let slice_size: usize = if bd + last_dim < data.ndim() {
        data.shape[bd + last_dim..].iter().product()
    } else {
        1
    };
    let batch_size: usize = if bd > 0 {
        data.shape[..bd].iter().product()
    } else {
        1
    };
    let data_batch_stride: usize = data.shape[bd..].iter().product();
    let idx_batch_stride: usize = indices.shape[bd..].iter().product();
    let num_indices: usize = if bd < ind_ndim - 1 {
        indices.shape[bd..ind_ndim - 1].iter().product()
    } else {
        1
    };

    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![0.0f32; out_numel];

    for b in 0..batch_size {
        for i in 0..num_indices {
            // Read the K-dimensional index
            let idx_offset = b * idx_batch_stride + i * last_dim;
            let mut flat_idx = 0usize;
            let mut stride = data_batch_stride;
            for k in 0..last_dim {
                let dim_size = data.shape[bd + k];
                stride /= dim_size;
                let mut idx_val = indices.data[idx_offset + k] as i64;
                if idx_val < 0 {
                    idx_val += dim_size as i64;
                }
                if idx_val < 0 || idx_val as usize >= dim_size {
                    return Err(format!(
                        "gather_nd: index {idx_val} out of bounds for dim size {dim_size}"
                    ));
                }
                flat_idx += idx_val as usize * stride;
            }
            let src_start = b * data_batch_stride + flat_idx;
            let dst_start = b * num_indices * slice_size + i * slice_size;
            out_data[dst_start..dst_start + slice_size]
                .copy_from_slice(&data.data[src_start..src_start + slice_size]);
        }
    }

    Ok(Tensor::new(out_data, out_shape))
}

/// OneHot: create one-hot encoded tensor
/// indices: any shape, depth: number of classes, values: (off_value, on_value), axis: insertion axis
pub fn one_hot(
    indices: &Tensor,
    depth: usize,
    values: (f32, f32),
    axis: i64,
) -> Result<Tensor, String> {
    if depth == 0 {
        return Err("one_hot: depth must be > 0".into());
    }
    let ndim = indices.ndim() + 1;
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "one_hot: axis {ax} out of range for output ndim {ndim}"
        ));
    }

    let (off_val, on_val) = values;

    // Output shape: insert `depth` at position `ax` in indices.shape
    let mut out_shape = indices.shape.clone();
    out_shape.insert(ax, depth);

    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![off_val; out_numel];

    // For each element in indices, set the corresponding position to on_val
    let outer: usize = if ax < indices.ndim() {
        indices.shape[..ax].iter().product::<usize>().max(1)
    } else {
        indices.numel().max(1)
    };
    let inner: usize = if ax < indices.ndim() {
        indices.shape[ax..].iter().product::<usize>().max(1)
    } else {
        1
    };

    for o in 0..outer {
        for i in 0..inner {
            let idx_flat = o * inner + i;
            if idx_flat >= indices.data.len() {
                continue;
            }
            let mut idx_val = indices.data[idx_flat] as i64;
            if idx_val < 0 {
                idx_val += depth as i64;
            }
            if idx_val >= 0 && (idx_val as usize) < depth {
                let out_flat = (o * depth + idx_val as usize) * inner + i;
                data[out_flat] = on_val;
            }
        }
    }

    Ok(Tensor::new(data, out_shape))
}

/// Compress: select elements based on boolean condition along an axis.
/// If axis is None, flatten input first.
pub fn compress(input: &Tensor, condition: &Tensor, axis: Option<i64>) -> Result<Tensor, String> {
    if let Some(ax_val) = axis {
        let ndim = input.ndim();
        let ax = if ax_val < 0 {
            (ndim as i64 + ax_val) as usize
        } else {
            ax_val as usize
        };
        if ax >= ndim {
            return Err(format!(
                "compress: axis {ax} out of range for {ndim}D tensor"
            ));
        }

        // Count true values in condition
        let true_count = condition.data.iter().filter(|v| **v != 0.0).count();

        let mut out_shape = input.shape.clone();
        out_shape[ax] = true_count;

        let outer: usize = input.shape[..ax].iter().product::<usize>().max(1);
        let axis_size = input.shape[ax];
        let inner: usize = input.shape[ax + 1..].iter().product::<usize>().max(1);

        let mut data = Vec::with_capacity(out_shape.iter().product());

        for o in 0..outer {
            for a in 0..axis_size {
                if a < condition.data.len() && condition.data[a] != 0.0 {
                    for i in 0..inner {
                        data.push(input.data[(o * axis_size + a) * inner + i]);
                    }
                }
            }
        }

        Ok(Tensor::new(data, out_shape))
    } else {
        // Flatten and select
        let mut data = Vec::new();
        for (i, v) in input.data.iter().enumerate() {
            if i < condition.data.len() && condition.data[i] != 0.0 {
                data.push(*v);
            }
        }
        let len = data.len();
        Ok(Tensor::new(data, vec![len]))
    }
}

/// Unique: find unique elements.
/// Returns (unique_values, indices, inverse_indices, counts)
pub fn unique(
    x: &Tensor,
    axis: Option<i64>,
    sorted: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor), String> {
    if let Some(raw_axis) = axis {
        return unique_axis(x, raw_axis, sorted);
    }

    // Flatten approach (axis=None)
    let mut seen: Vec<(f32, usize)> = Vec::new(); // (value, first_index)
    let mut inverse = vec![0.0f32; x.data.len()];

    for (i, &val) in x.data.iter().enumerate() {
        if let Some(pos) = seen
            .iter()
            .position(|(v, _)| (*v - val).abs() < f32::EPSILON)
        {
            inverse[i] = pos as f32;
        } else {
            inverse[i] = seen.len() as f32;
            seen.push((val, i));
        }
    }

    if sorted {
        // Sort by value and remap
        let mut sorted_indices: Vec<usize> = (0..seen.len()).collect();
        sorted_indices.sort_by(|a, b| {
            seen[*a]
                .0
                .partial_cmp(&seen[*b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Remap inverse
        let mut remap = vec![0usize; seen.len()];
        for (new_pos, &old_pos) in sorted_indices.iter().enumerate() {
            remap[old_pos] = new_pos;
        }
        for inv in inverse.iter_mut() {
            *inv = remap[*inv as usize] as f32;
        }
        let sorted_seen: Vec<(f32, usize)> = sorted_indices.iter().map(|&i| seen[i]).collect();
        seen = sorted_seen;
    }

    let unique_vals: Vec<f32> = seen.iter().map(|(v, _)| *v).collect();
    let indices_data: Vec<f32> = seen.iter().map(|(_, i)| *i as f32).collect();
    let mut counts = vec![0.0f32; seen.len()];
    for &inv in &inverse {
        counts[inv as usize] += 1.0;
    }

    let n = unique_vals.len();
    Ok((
        Tensor::new(unique_vals, vec![n]),
        Tensor::new(indices_data, vec![n]),
        Tensor::new(inverse, vec![x.data.len()]),
        Tensor::new(counts, vec![n]),
    ))
}

/// Extract the data of a single slice along `ax` at position `idx`.
/// For shape [d0,..,d_{ax-1}, d_ax, d_{ax+1},..,d_{n-1}], the slice has
/// `outer * inner` elements where outer = product(shape[..ax]), inner = product(shape[ax+1..]).
fn extract_axis_slice(data: &[f32], shape: &[usize], ax: usize, idx: usize) -> Vec<f32> {
    let outer: usize = shape[..ax].iter().product::<usize>().max(1);
    let axis_size = shape[ax];
    let inner: usize = shape[ax + 1..].iter().product::<usize>().max(1);
    let mut slice_data = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        let base = (o * axis_size + idx) * inner;
        for j in 0..inner {
            slice_data.push(data[base + j]);
        }
    }
    slice_data
}

/// Compare two slices for exact equality (bitwise f32 comparison).
fn slices_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Unique along a specified axis.
/// Returns (unique_tensor, indices, inverse_indices, counts).
fn unique_axis(
    x: &Tensor,
    raw_axis: i64,
    sorted: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor), String> {
    let ndim = x.ndim();
    if ndim == 0 {
        return Err("unique: axis mode requires at least 1D tensor".into());
    }
    let ax = if raw_axis < 0 {
        let a = raw_axis + ndim as i64;
        if a < 0 {
            return Err(format!(
                "unique: axis {raw_axis} out of range for {ndim}D tensor"
            ));
        }
        a as usize
    } else {
        raw_axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "unique: axis {raw_axis} out of range for {ndim}D tensor"
        ));
    }

    let axis_size = x.shape[ax];

    // Extract all slices along the axis
    let all_slices: Vec<Vec<f32>> = (0..axis_size)
        .map(|i| extract_axis_slice(&x.data, &x.shape, ax, i))
        .collect();

    // Find unique slices (first occurrence order)
    // unique_map[i] = index into `unique_indices` that slice i maps to
    let mut unique_indices: Vec<usize> = Vec::new(); // original axis indices of unique slices
    let mut inverse_map: Vec<usize> = vec![0; axis_size];

    for i in 0..axis_size {
        let mut found = None;
        for (uid, &orig_idx) in unique_indices.iter().enumerate() {
            if slices_equal(&all_slices[i], &all_slices[orig_idx]) {
                found = Some(uid);
                break;
            }
        }
        match found {
            Some(uid) => {
                inverse_map[i] = uid;
            }
            None => {
                inverse_map[i] = unique_indices.len();
                unique_indices.push(i);
            }
        }
    }

    // If sorted, sort unique slices lexicographically and remap
    if sorted {
        let mut order: Vec<usize> = (0..unique_indices.len()).collect();
        order.sort_by(|&a, &b| {
            let sa = &all_slices[unique_indices[a]];
            let sb = &all_slices[unique_indices[b]];
            for (va, vb) in sa.iter().zip(sb.iter()) {
                match va.partial_cmp(vb) {
                    Some(std::cmp::Ordering::Equal) | None => continue,
                    Some(ord) => return ord,
                }
            }
            std::cmp::Ordering::Equal
        });

        // Build old-unique-pos -> new-unique-pos mapping
        let mut remap = vec![0usize; unique_indices.len()];
        for (new_pos, &old_pos) in order.iter().enumerate() {
            remap[old_pos] = new_pos;
        }
        // Remap inverse
        for inv in inverse_map.iter_mut() {
            *inv = remap[*inv];
        }
        // Reorder unique_indices
        let sorted_unique: Vec<usize> = order.iter().map(|&i| unique_indices[i]).collect();
        unique_indices = sorted_unique;
    }

    // Build output tensor by stacking unique slices along axis
    let num_unique = unique_indices.len();
    let mut out_shape = x.shape.clone();
    out_shape[ax] = num_unique;

    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);

    let total_elems: usize = out_shape.iter().product();
    let mut out_data = vec![0.0f32; total_elems];

    for (new_idx, &orig_idx) in unique_indices.iter().enumerate() {
        for o in 0..outer {
            let src_base = (o * axis_size + orig_idx) * inner;
            let dst_base = (o * num_unique + new_idx) * inner;
            out_data[dst_base..dst_base + inner]
                .copy_from_slice(&x.data[src_base..src_base + inner]);
        }
    }

    // Build counts
    let mut counts = vec![0.0f32; num_unique];
    for &inv in &inverse_map {
        counts[inv] += 1.0;
    }

    let indices_data: Vec<f32> = unique_indices.iter().map(|&i| i as f32).collect();
    let inverse_data: Vec<f32> = inverse_map.iter().map(|&i| i as f32).collect();

    Ok((
        Tensor::new(out_data, out_shape),
        Tensor::new(indices_data, vec![num_unique]),
        Tensor::new(inverse_data, vec![axis_size]),
        Tensor::new(counts, vec![num_unique]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::OnnxError;

    #[test]
    fn test_gather_embedding() -> Result<(), OnnxError> {
        // Embedding table: 5 tokens × 4 dims
        let table = Tensor::new((0..20).map(|i| i as f32).collect(), vec![5, 4]);
        // Indices: [2, 0, 4]
        let indices = Tensor::new(vec![2.0, 0.0, 4.0], vec![3]);
        let out = gather(&table, &indices, 0)?;
        assert_eq!(out.shape, vec![3, 4]);
        // token 2 → row 2 → [8, 9, 10, 11]
        assert_eq!(out.data[0], 8.0);
        assert_eq!(out.data[1], 9.0);
        // token 0 → [0, 1, 2, 3]
        assert_eq!(out.data[4], 0.0);
        Ok(())
    }

    #[test]
    fn test_scatter_elements() -> Result<(), OnnxError> {
        let data = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        let indices = Tensor::new(vec![1.0, 0.0], vec![1, 2]);
        let updates = Tensor::new(vec![7.0, 8.0], vec![1, 2]);
        let out = scatter_elements(&data, &indices, &updates, 0)?;
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data[2], 7.0); // row 1, col 0
        assert_eq!(out.data[1], 8.0); // row 0, col 1
        Ok(())
    }

    #[test]
    fn test_scatter_nd() -> Result<(), OnnxError> {
        let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let indices = Tensor::new(vec![0.0, 1.0], vec![1, 2]); // index [0][1]
        let updates = Tensor::new(vec![99.0, 100.0], vec![1, 2]); // update 2 values
        let out = scatter_nd(&data, &indices, &updates)?;
        assert_eq!(out.data[2], 99.0);
        assert_eq!(out.data[3], 100.0);
        Ok(())
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() -> Result<(), OnnxError> {
        let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]);
        let scale = Tensor::new(vec![0.01], vec![1]);
        let q = quantize_linear(&x, &scale, None)?;
        // 0/0.01=0, 1/0.01=100, -1/0.01=-100, 2/0.01=127 (clamped)
        assert_eq!(q.data[0], 0.0);
        assert_eq!(q.data[1], 100.0);
        assert_eq!(q.data[2], -100.0);
        assert_eq!(q.data[3], 127.0);
        let dq = dequantize_linear(&q, &scale, None)?;
        assert!((dq.data[1] - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_where() {
        let cond = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
        let x = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let y = Tensor::new(vec![100.0, 200.0, 300.0], vec![3]);
        let out = where_op(&cond, &x, &y).expect("where_op failed");
        assert_eq!(out.data, vec![10.0, 200.0, 30.0]);
    }

    #[test]
    fn test_gather_nd() {
        // data: [[0,1],[2,3],[4,5]], indices: [[0,0],[1,1]] -> [0, 3]
        let data = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![3, 2]);
        let indices = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]);
        let out = gather_nd(&data, &indices, 0).expect("gather_nd failed");
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.data, vec![0.0, 3.0]);
    }

    #[test]
    fn test_gather_nd_slice() {
        // data: [[0,1],[2,3],[4,5]], indices: [[0],[2]] -> [[0,1],[4,5]]
        let data = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![3, 2]);
        let indices = Tensor::new(vec![0.0, 2.0], vec![2, 1]);
        let out = gather_nd(&data, &indices, 0).expect("gather_nd slice failed");
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn test_gather_nd_negative_index() {
        let data = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![3, 2]);
        // -1 means last row -> [4, 5]
        let indices = Tensor::new(vec![-1.0], vec![1, 1]);
        let out = gather_nd(&data, &indices, 0).expect("gather_nd neg failed");
        assert_eq!(out.data, vec![4.0, 5.0]);
    }

    #[test]
    fn test_one_hot() {
        let indices = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
        let out = one_hot(&indices, 3, (0.0, 1.0), -1).expect("one_hot failed");
        assert_eq!(out.shape, vec![3, 3]);
        // Should be identity-like matrix
        assert_eq!(out.data[0], 1.0); // [0,0]
        assert_eq!(out.data[1], 0.0); // [0,1]
        assert_eq!(out.data[4], 1.0); // [1,1]
        assert_eq!(out.data[8], 1.0); // [2,2]
    }

    #[test]
    fn test_one_hot_custom_values() {
        let indices = Tensor::new(vec![1.0], vec![1]);
        let out = one_hot(&indices, 3, (5.0, 10.0), 0).expect("one_hot custom failed");
        assert_eq!(out.shape, vec![3, 1]);
        assert_eq!(out.data, vec![5.0, 10.0, 5.0]);
    }

    #[test]
    fn test_one_hot_negative_index() {
        let indices = Tensor::new(vec![-1.0], vec![1]);
        let out = one_hot(&indices, 3, (0.0, 1.0), -1).expect("one_hot neg idx failed");
        assert_eq!(out.shape, vec![1, 3]);
        // -1 + 3 = 2 -> last position
        assert_eq!(out.data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_compress() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let cond = Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0], vec![5]);
        let out = compress(&input, &cond, None).expect("compress failed");
        assert_eq!(out.data, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_compress_with_axis() {
        // 2D input, compress along axis 0
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let cond = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
        let out = compress(&input, &cond, Some(0)).expect("compress axis failed");
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_compress_all_false() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let cond = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]);
        let out = compress(&input, &cond, None).expect("compress all false failed");
        assert_eq!(out.data.len(), 0);
        assert_eq!(out.shape, vec![0]);
    }

    #[test]
    fn test_unique() {
        let x = Tensor::new(vec![2.0, 1.0, 1.0, 3.0, 2.0], vec![5]);
        let (unique_vals, _indices, inverse, counts) =
            unique(&x, None, true).expect("unique failed");
        assert_eq!(unique_vals.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(counts.data, vec![2.0, 2.0, 1.0]);
        // Verify inverse mapping
        for (i, &orig) in x.data.iter().enumerate() {
            let inv_idx = inverse.data[i] as usize;
            assert!(
                (unique_vals.data[inv_idx] - orig).abs() < f32::EPSILON,
                "inverse mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_unique_unsorted() {
        let x = Tensor::new(vec![3.0, 1.0, 3.0], vec![3]);
        let (unique_vals, indices, inverse, counts) =
            unique(&x, None, false).expect("unique unsorted failed");
        // First-seen order: 3.0, 1.0
        assert_eq!(unique_vals.data, vec![3.0, 1.0]);
        assert_eq!(indices.data, vec![0.0, 1.0]); // first occurrences
        assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
        assert_eq!(counts.data, vec![2.0, 1.0]);
    }

    #[test]
    fn test_unique_all_same() {
        let x = Tensor::new(vec![5.0, 5.0, 5.0], vec![3]);
        let (unique_vals, _, _, counts) = unique(&x, None, true).expect("unique same failed");
        assert_eq!(unique_vals.data, vec![5.0]);
        assert_eq!(counts.data, vec![3.0]);
    }

    // ---- Axis-mode unique tests ----

    #[test]
    fn test_unique_axis0_1d() {
        // 1D with axis=0 should behave like value-level unique
        let x = Tensor::new(vec![3.0, 1.0, 3.0, 2.0, 1.0], vec![5]);
        let (unique_vals, indices, inverse, counts) =
            unique(&x, Some(0), false).expect("unique axis0 1d failed");
        // First occurrence order: 3.0, 1.0, 2.0
        assert_eq!(unique_vals.data, vec![3.0, 1.0, 2.0]);
        assert_eq!(unique_vals.shape, vec![3]);
        assert_eq!(indices.data, vec![0.0, 1.0, 3.0]);
        assert_eq!(inverse.data, vec![0.0, 1.0, 0.0, 2.0, 1.0]);
        assert_eq!(counts.data, vec![2.0, 2.0, 1.0]);
    }

    #[test]
    fn test_unique_axis0_2d_rows() {
        // 3x2 matrix with duplicate rows
        // Row 0: [1, 2], Row 1: [3, 4], Row 2: [1, 2]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0], vec![3, 2]);
        let (out, indices, inverse, counts) =
            unique(&x, Some(0), false).expect("unique axis0 2d failed");
        // Unique rows (first occurrence): row 0 [1,2], row 1 [3,4]
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(indices.data, vec![0.0, 1.0]);
        assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
        assert_eq!(counts.data, vec![2.0, 1.0]);
    }

    #[test]
    fn test_unique_axis1_2d_columns() {
        // 2x4 matrix, columns: [1,5], [2,6], [1,5], [3,7]
        // Col 0 == Col 2
        let x = Tensor::new(vec![1.0, 2.0, 1.0, 3.0, 5.0, 6.0, 5.0, 7.0], vec![2, 4]);
        let (out, indices, inverse, counts) =
            unique(&x, Some(1), false).expect("unique axis1 2d failed");
        // Unique columns (first occurrence): col 0 [1,5], col 1 [2,6], col 3 [3,7]
        assert_eq!(out.shape, vec![2, 3]);
        // Row 0: [1, 2, 3], Row 1: [5, 6, 7]
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0]);
        assert_eq!(indices.data, vec![0.0, 1.0, 3.0]);
        assert_eq!(inverse.data, vec![0.0, 1.0, 0.0, 2.0]);
        assert_eq!(counts.data, vec![2.0, 1.0, 1.0]);
    }

    #[test]
    fn test_unique_axis0_sorted() {
        // Rows: [3,4], [1,2], [3,4]
        // Unsorted unique: [3,4] (idx 0), [1,2] (idx 1)
        // Sorted unique (lexicographic): [1,2], [3,4]
        let x = Tensor::new(vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0], vec![3, 2]);
        let (out, indices, inverse, counts) =
            unique(&x, Some(0), true).expect("unique axis0 sorted failed");
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(indices.data, vec![1.0, 0.0]); // row 1 first after sort, then row 0
        assert_eq!(inverse.data, vec![1.0, 0.0, 1.0]); // row 0->[3,4]->idx 1, row 1->[1,2]->idx 0
        assert_eq!(counts.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_unique_axis0_unsorted() {
        // Same data, but unsorted preserves first-occurrence order
        let x = Tensor::new(vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0], vec![3, 2]);
        let (out, _indices, inverse, counts) =
            unique(&x, Some(0), false).expect("unique axis0 unsorted failed");
        assert_eq!(out.shape, vec![2, 2]);
        // First occurrence order: [3,4] then [1,2]
        assert_eq!(out.data, vec![3.0, 4.0, 1.0, 2.0]);
        assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
        assert_eq!(counts.data, vec![2.0, 1.0]);
    }

    #[test]
    fn test_unique_axis_negative() {
        // 2x3 matrix, axis=-1 means axis=1 (columns)
        // Cols: [1,4], [2,5], [1,4] -> col 0 == col 2
        let x = Tensor::new(vec![1.0, 2.0, 1.0, 4.0, 5.0, 4.0], vec![2, 3]);
        let (out, indices, inverse, counts) =
            unique(&x, Some(-1), false).expect("unique neg axis failed");
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 4.0, 5.0]);
        assert_eq!(indices.data, vec![0.0, 1.0]);
        assert_eq!(inverse.data, vec![0.0, 1.0, 0.0]);
        assert_eq!(counts.data, vec![2.0, 1.0]);
    }

    #[test]
    fn test_unique_axis_single_element() {
        // 1x1 tensor with axis=0
        let x = Tensor::new(vec![42.0], vec![1, 1]);
        let (out, indices, inverse, counts) =
            unique(&x, Some(0), false).expect("unique single elem failed");
        assert_eq!(out.shape, vec![1, 1]);
        assert_eq!(out.data, vec![42.0]);
        assert_eq!(indices.data, vec![0.0]);
        assert_eq!(inverse.data, vec![0.0]);
        assert_eq!(counts.data, vec![1.0]);
    }

    #[test]
    fn test_unique_axis_all_same_rows() {
        // 3x2, all rows identical
        let x = Tensor::new(vec![7.0, 8.0, 7.0, 8.0, 7.0, 8.0], vec![3, 2]);
        let (out, indices, inverse, counts) =
            unique(&x, Some(0), false).expect("unique all same rows failed");
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.data, vec![7.0, 8.0]);
        assert_eq!(indices.data, vec![0.0]);
        assert_eq!(inverse.data, vec![0.0, 0.0, 0.0]);
        assert_eq!(counts.data, vec![3.0]);
    }

    #[test]
    fn test_unique_axis_all_distinct() {
        // 3x2, all rows distinct
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let (out, indices, inverse, counts) =
            unique(&x, Some(0), false).expect("unique all distinct failed");
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(indices.data, vec![0.0, 1.0, 2.0]);
        assert_eq!(inverse.data, vec![0.0, 1.0, 2.0]);
        assert_eq!(counts.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_unique_axis_3d() {
        // 2x2x2 tensor, unique along axis=0
        // Slice 0: [[1,2],[3,4]], Slice 1: [[1,2],[3,4]] -> identical
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], vec![2, 2, 2]);
        let (out, indices, inverse, counts) =
            unique(&x, Some(0), false).expect("unique 3d axis0 failed");
        assert_eq!(out.shape, vec![1, 2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(indices.data, vec![0.0]);
        assert_eq!(inverse.data, vec![0.0, 0.0]);
        assert_eq!(counts.data, vec![2.0]);
    }

    #[test]
    fn test_unique_axis_out_of_range() {
        let x = Tensor::new(vec![1.0, 2.0], vec![2]);
        let err = unique(&x, Some(1), false);
        assert!(err.is_err());
        let err_neg = unique(&x, Some(-2), false);
        assert!(err_neg.is_err());
    }
}
