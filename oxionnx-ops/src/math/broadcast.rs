use oxionnx_core::Tensor;

/// Check if the reduction covers all dimensions (full-tensor reduction).
/// This is true when axes is empty (ONNX convention: all axes)
/// or when axes lists every dimension index.
#[cfg(feature = "simd")]
pub(super) fn is_full_reduction(x: &Tensor, axes: &[i64]) -> bool {
    if axes.is_empty() {
        return true;
    }
    let ndim = x.ndim();
    if axes.len() < ndim {
        return false;
    }
    let mut seen = vec![false; ndim];
    for &a in axes {
        let idx = if a < 0 {
            (a + ndim as i64) as usize
        } else {
            a as usize
        };
        if idx < ndim {
            seen[idx] = true;
        }
    }
    seen.iter().all(|&s| s)
}

/// Broadcast a tensor to the given shape.
pub fn broadcast_to(t: &Tensor, out_shape: &[usize]) -> Tensor {
    if t.shape == out_shape {
        return t.clone();
    }
    let n_out: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; n_out];

    // Pad t.shape on the left with 1s to match out_shape.ndim
    let n = out_shape.len();
    let pad = n - t.shape.len();
    let padded: Vec<usize> = (0..pad).map(|_| 1).chain(t.shape.iter().copied()).collect();

    // Compute strides for padded shape (0 for broadcast dims)
    let mut strides = vec![0usize; n];
    let mut stride = 1usize;
    for i in (0..n).rev() {
        if padded[i] == 1 && out_shape[i] != 1 {
            strides[i] = 0; // broadcast
        } else {
            strides[i] = stride;
        }
        stride *= padded[i];
    }

    // Fill output
    let mut out_strides = vec![0usize; n];
    let mut s = 1usize;
    for i in (0..n).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }

    for (out_idx, out_val) in data.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut src_idx = 0usize;
        for i in 0..n {
            let coord = rem / out_strides[i];
            rem %= out_strides[i];
            src_idx += coord * strides[i];
        }
        *out_val = t.data[src_idx];
    }

    Tensor::new(data, out_shape.to_vec())
}

pub(super) fn elementwise_binary(
    a: &Tensor,
    b: &Tensor,
    op: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)?;
    let a = broadcast_to(a, &out_shape);
    let b = broadcast_to(b, &out_shape);
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();
    Ok(Tensor::new(data, out_shape))
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    if a.shape == b.shape {
        let mut out = vec![0.0f32; a.data.len()];
        crate::simd_ops::simd_add(&a.data, &b.data, &mut out);
        return Ok(Tensor::new(out, a.shape.clone()));
    }
    elementwise_binary(a, b, |x, y| x + y)
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    if a.shape == b.shape {
        let mut out = vec![0.0f32; a.data.len()];
        crate::simd_ops::simd_sub(&a.data, &b.data, &mut out);
        return Ok(Tensor::new(out, a.shape.clone()));
    }
    elementwise_binary(a, b, |x, y| x - y)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    if a.shape == b.shape {
        let mut out = vec![0.0f32; a.data.len()];
        crate::simd_ops::simd_mul(&a.data, &b.data, &mut out);
        return Ok(Tensor::new(out, a.shape.clone()));
    }
    elementwise_binary(a, b, |x, y| x * y)
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    #[cfg(feature = "simd")]
    if a.shape == b.shape {
        let mut out = vec![0.0f32; a.data.len()];
        crate::simd_ops::simd_div(&a.data, &b.data, &mut out);
        return Ok(Tensor::new(out, a.shape.clone()));
    }
    elementwise_binary(a, b, |x, y| x / y)
}

pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    elementwise_binary(a, b, |x, y| x.powf(y))
}

pub fn sqrt(a: &Tensor) -> Tensor {
    #[cfg(feature = "simd")]
    {
        let mut data = a.data.clone();
        crate::simd_ops::simd_sqrt(&mut data);
        Tensor::new(data, a.shape.clone())
    }
    #[cfg(not(feature = "simd"))]
    Tensor::new(a.data.iter().map(|x| x.sqrt()).collect(), a.shape.clone())
}

pub fn reciprocal(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.recip()).collect(), a.shape.clone())
}

pub fn neg(a: &Tensor) -> Tensor {
    #[cfg(feature = "simd")]
    {
        let mut data = a.data.clone();
        crate::simd_ops::simd_neg(&mut data);
        Tensor::new(data, a.shape.clone())
    }
    #[cfg(not(feature = "simd"))]
    Tensor::new(a.data.iter().map(|x| -x).collect(), a.shape.clone())
}
