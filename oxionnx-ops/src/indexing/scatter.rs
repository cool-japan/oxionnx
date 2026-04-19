use oxionnx_core::Tensor;

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

/// Core ScatterND loop: mutates `out` (already initialised to `data`) by
/// applying scatter updates.  Semantics match `scatter_nd` (no reduction).
///
/// `k` is the index depth — the last dimension of the indices tensor.
pub(crate) fn scatter_nd_into(
    data_shape: &[usize],
    k: usize,
    indices_data: &[f32],
    updates_data: &[f32],
    out: &mut [f32],
) {
    let ndim = data_shape.len();
    if k == 0 || k > ndim {
        return;
    }

    let mut data_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        data_strides[i] = s;
        s *= data_shape[i];
    }

    let n_idx = indices_data.len() / k;
    let inner: usize = data_shape[k..].iter().product::<usize>().max(1);

    for i in 0..n_idx {
        let idx_base = i * k;
        let mut data_flat = 0usize;
        for (j, &ds) in data_strides.iter().enumerate().take(k) {
            let coord = indices_data[idx_base + j] as usize;
            data_flat += coord * ds;
        }
        let upd_base = i * inner;
        out[data_flat..data_flat + inner]
            .copy_from_slice(&updates_data[upd_base..upd_base + inner]);
    }
}

/// Core ScatterElements loop: mutates `out` (already initialised to `data`)
/// by applying element-wise scatter updates.  Semantics match
/// `scatter_elements` (no reduction, overwrite).
///
/// `indices_shape` must be the actual shape of the `indices_data` slice.
pub(crate) fn scatter_elements_into(
    data_shape: &[usize],
    indices_shape: &[usize],
    indices_data: &[f32],
    updates_data: &[f32],
    axis: usize,
    out: &mut [f32],
) {
    let ndim = data_shape.len();
    if ndim == 0 {
        return;
    }

    let mut data_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        data_strides[i] = s;
        s *= data_shape[i];
    }

    let mut idx_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        idx_strides[i] = s;
        s *= indices_shape[i];
    }

    for (flat, (&idx_val, &upd_val)) in indices_data.iter().zip(updates_data.iter()).enumerate() {
        let mut rem = flat;
        let mut data_flat = 0usize;
        for d in 0..ndim {
            let coord = rem / idx_strides[d];
            rem %= idx_strides[d];
            if d == axis {
                let idx = idx_val as i64;
                let idx = if idx < 0 {
                    (idx + data_shape[axis] as i64) as usize
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
}
