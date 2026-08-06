use oxionnx_core::Tensor;

/// Compute strides for broadcasting a (possibly lower-rank) `shape` up to
/// `ndim` dimensions: left-pad with implicit size-1 axes, and use stride 0
/// on any axis where the operand's size is 1 (broadcast), so walking the
/// *output's* coordinates and mapping them through these strides naturally
/// repeats that axis's single slice across the output — real NumPy/ONNX
/// broadcasting, not flat-index modulo (which is only correct when an
/// operand is a scalar or already full-size).
fn padded_broadcast_strides(shape: &[usize], ndim: usize) -> Vec<usize> {
    let pad = ndim - shape.len();
    let mut strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..shape.len()).rev() {
        strides[pad + i] = if shape[i] == 1 { 0 } else { s };
        s *= shape[i];
    }
    strides
}

/// Where: select elements from `x` or `y` based on `condition` (bool tensor),
/// with full N-D broadcasting (NumPy/ONNX rules) across all three operands.
pub fn where_op(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(
        &Tensor::broadcast_shape(&condition.shape, &x.shape)?,
        &y.shape,
    )?;
    let ndim = out_shape.len();
    let n: usize = out_shape.iter().product();

    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }

    let cond_strides = padded_broadcast_strides(&condition.shape, ndim);
    let x_strides = padded_broadcast_strides(&x.shape, ndim);
    let y_strides = padded_broadcast_strides(&y.shape, ndim);

    let mut data = vec![0.0f32; n];
    for (out_idx, out_val) in data.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut cond_idx = 0usize;
        let mut x_idx = 0usize;
        let mut y_idx = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            cond_idx += coord * cond_strides[d];
            x_idx += coord * x_strides[d];
            y_idx += coord * y_strides[d];
        }
        *out_val = if condition.data[cond_idx] != 0.0 {
            x.data[x_idx]
        } else {
            y.data[y_idx]
        };
    }

    Ok(Tensor::new(data, out_shape))
}

/// Expand: broadcast x to shape.
pub fn expand(x: &Tensor, shape: &[usize]) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(&x.shape, shape)?;
    let ndim = out_shape.len();
    let n: usize = out_shape.iter().product();

    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    let in_strides = padded_broadcast_strides(&x.shape, ndim);

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
