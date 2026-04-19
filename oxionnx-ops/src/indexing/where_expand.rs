use oxionnx_core::Tensor;

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
