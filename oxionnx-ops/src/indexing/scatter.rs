use oxionnx_core::Tensor;

use super::index_util::{normalize_axis, normalize_index};

/// How to combine a scattered update with the value already present at that
/// position. Mirrors the ONNX `reduction` attribute on `ScatterElements` /
/// `ScatterND` (opset 16 adds `none`/`add`/`mul`, opset 18 adds `max`/`min`).
/// When two updates target the same position, `None` keeps whichever update
/// is seen last (plain overwrite, the pre-opset-16 behavior); the others
/// accumulate every update that lands on that position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduction {
    None,
    Add,
    Mul,
    Max,
    Min,
}

impl ScatterReduction {
    /// Parse the ONNX `reduction` attribute string. An empty string (the
    /// attribute is absent from the node) is the spec default, `"none"`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "" | "none" => Ok(Self::None),
            "add" => Ok(Self::Add),
            "mul" => Ok(Self::Mul),
            "max" => Ok(Self::Max),
            "min" => Ok(Self::Min),
            other => Err(format!("unsupported scatter reduction '{other}'")),
        }
    }

    #[inline]
    fn combine(self, current: f32, update: f32) -> f32 {
        match self {
            Self::None => update,
            Self::Add => current + update,
            Self::Mul => current * update,
            Self::Max => current.max(update),
            Self::Min => current.min(update),
        }
    }
}

/// ScatterElements: for each element in `updates`, write it into `data` at the position
/// given by `indices` along `axis` (no reduction — last write wins on duplicate indices).
pub fn scatter_elements(
    data: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
    axis: i64,
) -> Result<Tensor, String> {
    scatter_elements_reduce(data, indices, updates, axis, ScatterReduction::None)
}

/// ScatterElements with an explicit `reduction` mode (ONNX opset 16+): when
/// `indices` repeats a position, the updates targeting it are combined via
/// `reduction` (add/mul/max/min) instead of the last one silently winning.
pub fn scatter_elements_reduce(
    data: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
    axis: i64,
    reduction: ScatterReduction,
) -> Result<Tensor, String> {
    let ndim = data.ndim();
    let ax = normalize_axis(axis, ndim, "scatter_elements")?;

    let mut out = data.data.clone();
    let axis_size = data.shape[ax];

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
                // Bounds-checked (and negative-normalized) BEFORE it ever
                // contributes to the write offset — an out-of-range index
                // must error, never land on the wrong element.
                let idx = normalize_index(idx_val, axis_size, "scatter_elements")?;
                data_flat += idx * data_strides[d];
            } else {
                // `indices`/`updates` need only match `data`'s rank, not its
                // exact per-dimension sizes; guard the non-axis dims too so a
                // malformed model can't walk `data_flat` past the buffer.
                if coord >= data.shape[d] {
                    return Err(format!(
                        "scatter_elements: indices shape {:?} exceeds data shape {:?} on non-axis dim {d}",
                        indices.shape, data.shape
                    ));
                }
                data_flat += coord * data_strides[d];
            }
        }
        out[data_flat] = reduction.combine(out[data_flat], upd_val);
    }

    Ok(Tensor::new(out, data.shape.clone()))
}

/// ScatterND: updates `data` at multi-dim indices (no reduction — last write
/// wins on duplicate indices).
/// `indices` shape: `[..., k]`; `updates` shape: `indices.shape[:-1] + data.shape[k:]`
pub fn scatter_nd(data: &Tensor, indices: &Tensor, updates: &Tensor) -> Result<Tensor, String> {
    scatter_nd_reduce(data, indices, updates, ScatterReduction::None)
}

/// ScatterND with an explicit `reduction` mode (ONNX opset 16+): when
/// `indices` repeats a position, the updates targeting it are combined via
/// `reduction` (add/mul/max/min) instead of the last one silently winning.
pub fn scatter_nd_reduce(
    data: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
    reduction: ScatterReduction,
) -> Result<Tensor, String> {
    let ndim = data.ndim();
    let k = *indices
        .shape
        .last()
        .ok_or("scatter_nd: indices must be at least 1D")?;
    if k == 0 {
        return Err("scatter_nd: indices' last dimension (index depth) must be > 0".into());
    }
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
            // Each of the k index components is bounds-checked (and
            // negative-normalized) against ITS OWN data dimension before
            // contributing to the write offset — matches gather_nd's
            // per-component validation, applied before the write instead
            // of only on read.
            let coord = normalize_index(indices.data[idx_base + j], data.shape[j], "scatter_nd")?;
            data_flat += coord * ds;
        }
        let upd_base = i * inner;
        if reduction == ScatterReduction::None {
            out[data_flat..data_flat + inner]
                .copy_from_slice(&updates.data[upd_base..upd_base + inner]);
        } else {
            for off in 0..inner {
                out[data_flat + off] =
                    reduction.combine(out[data_flat + off], updates.data[upd_base + off]);
            }
        }
    }

    Ok(Tensor::new(out, data.shape.clone()))
}

/// Core ScatterND loop: mutates `out` (already initialised to `data`) by
/// applying scatter updates with the given reduction mode.
///
/// `k` is the index depth — the last dimension of the indices tensor.
pub(crate) fn scatter_nd_into(
    data_shape: &[usize],
    k: usize,
    indices_data: &[f32],
    updates_data: &[f32],
    reduction: ScatterReduction,
    out: &mut [f32],
) -> Result<(), String> {
    let ndim = data_shape.len();
    if k == 0 {
        return Err("scatter_nd: indices' last dimension (index depth) must be > 0".into());
    }
    if k > ndim {
        return Err(format!(
            "scatter_nd: index depth {k} exceeds data ndim {ndim}"
        ));
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
            let coord = normalize_index(indices_data[idx_base + j], data_shape[j], "scatter_nd")?;
            data_flat += coord * ds;
        }
        let upd_base = i * inner;
        if reduction == ScatterReduction::None {
            out[data_flat..data_flat + inner]
                .copy_from_slice(&updates_data[upd_base..upd_base + inner]);
        } else {
            for off in 0..inner {
                out[data_flat + off] =
                    reduction.combine(out[data_flat + off], updates_data[upd_base + off]);
            }
        }
    }
    Ok(())
}

/// Core ScatterElements loop: mutates `out` (already initialised to `data`)
/// by applying element-wise scatter updates with the given reduction mode.
///
/// `indices_shape` must be the actual shape of the `indices_data` slice.
pub(crate) fn scatter_elements_into(
    data_shape: &[usize],
    indices_shape: &[usize],
    indices_data: &[f32],
    updates_data: &[f32],
    axis: usize,
    reduction: ScatterReduction,
    out: &mut [f32],
) -> Result<(), String> {
    let ndim = data_shape.len();
    if ndim == 0 {
        return Ok(());
    }
    if axis >= ndim {
        return Err(format!(
            "scatter_elements: axis {axis} out of range for {ndim}D tensor"
        ));
    }
    let axis_size = data_shape[axis];

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
                let idx = normalize_index(idx_val, axis_size, "scatter_elements")?;
                data_flat += idx * data_strides[d];
            } else {
                if coord >= data_shape[d] {
                    return Err(format!(
                        "scatter_elements: indices shape {indices_shape:?} exceeds data shape {data_shape:?} on non-axis dim {d}"
                    ));
                }
                data_flat += coord * data_strides[d];
            }
        }
        out[data_flat] = reduction.combine(out[data_flat], upd_val);
    }
    Ok(())
}
