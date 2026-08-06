use oxionnx_core::Tensor;

use super::index_util::{normalize_axis, normalize_index};

/// Gather elements from `data` along `axis` using `indices`.
///
/// For axis=0 (embedding lookup):
///   output[i, j, ...] = data[indices[i, j, ...], j, ...]
///
/// For axis=1:
///   output[i, j, k, ...] = data[i, indices[i, j, k, ...], k, ...]
pub fn gather(data: &Tensor, indices: &Tensor, axis: i64) -> Result<Tensor, String> {
    let ndim = data.ndim();
    let ax = normalize_axis(axis, ndim, "gather")?;

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
            let idx = normalize_index(idx_val, axis_size, "gather")?;
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
    let ax = normalize_axis(axis, ndim, "gather_elements")?;

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

    let axis_size = data.shape[ax];
    for (flat, out_val) in out.iter_mut().enumerate() {
        // Decode flat index in indices tensor
        let mut rem = flat;
        let mut data_flat = 0usize;
        for d in 0..ndim {
            let coord = rem / idx_strides[d];
            rem %= idx_strides[d];
            if d == ax {
                let idx_val = normalize_index(indices.data[flat], axis_size, "gather_elements")?;
                data_flat += idx_val * data_strides[d];
            } else {
                // `indices` need only match `data`'s rank, not its exact
                // per-dimension sizes; a malformed model could still supply
                // a non-axis dimension larger than `data`'s, which would
                // otherwise read past the end of `data.data`.
                if coord >= data.shape[d] {
                    return Err(format!(
                        "gather_elements: indices shape {:?} exceeds data shape {:?} on non-axis dim {d}",
                        indices.shape, data.shape
                    ));
                }
                data_flat += coord * data_strides[d];
            }
        }
        *out_val = data.data[data_flat];
    }

    Ok(Tensor::new(out, indices.shape.clone()))
}

/// GatherND: gather slices using multi-dimensional indices
/// data: any shape, indices: [..., K] where K <= data.ndim - batch_dims
/// output shape: indices.shape[:-1] + data.shape[batch_dims + K:]
pub fn gather_nd(data: &Tensor, indices: &Tensor, batch_dims: i64) -> Result<Tensor, String> {
    if batch_dims < 0 {
        return Err(format!(
            "gather_nd: batch_dims must be non-negative, got {batch_dims}"
        ));
    }
    let bd = batch_dims as usize;
    let ind_ndim = indices.ndim();
    if ind_ndim == 0 {
        return Err("gather_nd: indices must be at least 1D".into());
    }
    let last_dim = indices.shape[ind_ndim - 1]; // K

    let combined_dims = bd.checked_add(last_dim).ok_or_else(|| {
        format!("gather_nd: batch_dims ({bd}) + last_index_dim ({last_dim}) overflows")
    })?;
    if combined_dims > data.ndim() {
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

    // No `[1]` promotion when all three contributions above are empty: `indices.shape[:-1] +
    // data.shape[batch_dims + K:]` working out empty means every index addresses one whole
    // element, which ONNX specifies as a rank-0 output. `out_numel` below is unaffected — the
    // empty shape's product is the empty product 1, the single element such a gather yields.

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

/// Core Gather loop: writes gathered values into a pre-allocated `out` buffer.
///
/// Caller must ensure `out.len() == out_shape.iter().product()`. Returns an
/// error for any out-of-range axis or index (instead of clamping to the
/// last row), so this output-slot dispatch path agrees with `gather`'s
/// error behavior rather than silently substituting the wrong element.
pub(crate) fn gather_into(
    data: &Tensor,
    indices_data: &[f32],
    axis: usize,
    out: &mut [f32],
) -> Result<(), String> {
    if axis >= data.shape.len() {
        return Err(format!(
            "gather: axis {axis} out of range for {}D tensor",
            data.shape.len()
        ));
    }
    let outer: usize = data.shape[..axis].iter().product::<usize>().max(1);
    let axis_size = data.shape[axis];
    let inner: usize = data.shape[axis + 1..].iter().product::<usize>().max(1);
    let idx_n = indices_data.len();

    for o in 0..outer {
        for (ii, &idx_val) in indices_data.iter().enumerate() {
            let idx = normalize_index(idx_val, axis_size, "gather")?;
            let src_base = o * axis_size * inner + idx * inner;
            let dst_base = o * idx_n * inner + ii * inner;
            out[dst_base..dst_base + inner].copy_from_slice(&data.data[src_base..src_base + inner]);
        }
    }
    Ok(())
}
