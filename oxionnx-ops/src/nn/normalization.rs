//! Normalization operations: softmax, layer_norm, batch_norm and related ops.

use oxionnx_core::Tensor;

pub(crate) fn layer_norm_into(
    x: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
    axis: i64,
    out: &mut [f32],
) -> Result<(), String> {
    out.copy_from_slice(&x.data);
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    let norm_size: usize = x.shape[ax..].iter().product();

    #[cfg(feature = "simd")]
    {
        let bias_data = bias.map(|b| b.data.as_slice());
        crate::simd_ops::simd_layer_norm_strided(out, norm_size, &scale.data, bias_data, eps);
        Ok(())
    }

    #[cfg(not(feature = "simd"))]
    {
        let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
        for o in 0..outer {
            let slice = &mut out[o * norm_size..(o + 1) * norm_size];
            let mean = slice.iter().sum::<f32>() / norm_size as f32;
            let var = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / norm_size as f32;
            let inv_std = (var + eps).sqrt().recip();
            for v in slice.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }
        let scale_len = scale.numel();
        for (i, v) in out.iter_mut().enumerate() {
            *v *= scale.data[i % scale_len];
        }
        if let Some(b) = bias {
            let bias_len = b.numel();
            for (i, v) in out.iter_mut().enumerate() {
                *v += b.data[i % bias_len];
            }
        }
        Ok(())
    }
}

pub(crate) fn rms_norm_into(
    x: &Tensor,
    scale: &Tensor,
    eps: f32,
    out: &mut [f32],
) -> Result<(), String> {
    if x.ndim() < 1 {
        return Err("rms_norm: input must be at least 1D".into());
    }
    let norm_size = *x
        .shape
        .last()
        .ok_or_else(|| "rms_norm: empty shape".to_string())?;
    let outer = x.numel() / norm_size;
    let scale_len = scale.numel();

    for o in 0..outer {
        let slice = &x.data[o * norm_size..(o + 1) * norm_size];
        let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / norm_size as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for j in 0..norm_size {
            out[o * norm_size + j] =
                x.data[o * norm_size + j] * inv_rms * scale.data[j % scale_len];
        }
    }
    Ok(())
}

pub(crate) fn batch_norm_into(
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    eps: f32,
    out: &mut [f32],
) -> Result<(), String> {
    if x.ndim() < 2 {
        return Err("batch_norm: need at least 2D".into());
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = if x.ndim() > 2 {
        x.shape[2..].iter().product()
    } else {
        1
    };

    out.copy_from_slice(&x.data);
    for ni in 0..n {
        for ci in 0..c {
            let s = scale.data[ci];
            let b = bias.data[ci];
            let m = mean.data[ci];
            let v = var.data[ci];
            let inv_std = (v + eps).sqrt().recip();
            for si in 0..spatial {
                let idx = ni * c * spatial + ci * spatial + si;
                out[idx] = (out[idx] - m) * inv_std * s + b;
            }
        }
    }
    Ok(())
}

pub(crate) fn group_norm_into(
    x: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    num_groups: usize,
    eps: f32,
    out: &mut [f32],
) -> Result<(), String> {
    if x.ndim() < 2 {
        return Err("group_norm: need at least 2D input".into());
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product::<usize>().max(1);
    if c % num_groups != 0 {
        return Err(format!(
            "group_norm: C={c} not divisible by num_groups={num_groups}"
        ));
    }
    let group_size = c / num_groups * spatial;

    out.copy_from_slice(&x.data);
    for ni in 0..n {
        for g in 0..num_groups {
            let start = ni * c * spatial + g * group_size;
            let slice = &mut out[start..start + group_size];
            let mean = slice.iter().sum::<f32>() / group_size as f32;
            let var = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / group_size as f32;
            let inv_std = (var + eps).sqrt().recip();
            for v in slice.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }
    }
    for ni in 0..n {
        for ci in 0..c {
            let s = scale.data[ci % scale.numel()];
            let b = bias.map(|b| b.data[ci % b.numel()]).unwrap_or(0.0);
            for si in 0..spatial {
                let idx = ni * c * spatial + ci * spatial + si;
                out[idx] = out[idx] * s + b;
            }
        }
    }
    Ok(())
}

pub(crate) fn instance_norm_into(
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    eps: f32,
    out: &mut [f32],
) -> Result<(), String> {
    if x.ndim() < 3 {
        return Err("instance_norm: input must have at least 3 dimensions".into());
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();

    for batch in 0..n {
        for ch in 0..c {
            let offset = (batch * c + ch) * spatial;
            let slice = &x.data[offset..offset + spatial];
            let mean = slice.iter().sum::<f32>() / spatial as f32;
            let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / spatial as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let s = scale.data[ch];
            let b = bias.data[ch];
            for i in 0..spatial {
                out[offset + i] = (slice[i] - mean) * inv_std * s + b;
            }
        }
    }
    Ok(())
}

pub(crate) fn softmax_into(x: &Tensor, axis: i64, out: &mut [f32]) -> Result<(), String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "softmax: axis {axis} out of range for {ndim}D tensor"
        ));
    }

    out.copy_from_slice(&x.data);
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];

    #[cfg(feature = "simd")]
    {
        if inner == 1 {
            crate::simd_ops::simd_softmax_strided(out, axis_len);
            return Ok(());
        }
    }

    let outer: usize = x.shape[..ax].iter().product();
    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                if out[idx] > max_val {
                    max_val = out[idx];
                }
            }
            let mut sum = 0.0f32;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                out[idx] = (out[idx] - max_val).exp();
                sum += out[idx];
            }
            let inv = sum.recip();
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                out[idx] *= inv;
            }
        }
    }
    Ok(())
}

pub(crate) fn log_softmax_into(x: &Tensor, axis: i64, out: &mut [f32]) -> Result<(), String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "log_softmax: axis {axis} out of range for {ndim}D tensor"
        ));
    }

    out.copy_from_slice(&x.data);
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                if out[idx] > max_val {
                    max_val = out[idx];
                }
            }
            let mut sum_exp = 0.0f32;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                sum_exp += (out[idx] - max_val).exp();
            }
            let log_sum_exp = sum_exp.ln();
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                out[idx] = out[idx] - max_val - log_sum_exp;
            }
        }
    }
    Ok(())
}

/// Numerically stable softmax along the specified axis.
pub fn softmax(x: &Tensor, axis: i64) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "softmax: axis {axis} out of range for {ndim}D tensor"
        ));
    }

    let mut data = x.data.clone();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];

    // SIMD fast path: softmax along the last axis (contiguous in memory)
    #[cfg(feature = "simd")]
    {
        if inner == 1 {
            crate::simd_ops::simd_softmax_strided(&mut data, axis_len);
            return Ok(Tensor::new(data, x.shape.clone()));
        }
    }

    let outer: usize = x.shape[..ax].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            // find max
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }
            // subtract max and exp
            let mut sum = 0.0f32;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                data[idx] = (data[idx] - max_val).exp();
                sum += data[idx];
            }
            // normalize
            let inv = sum.recip();
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                data[idx] *= inv;
            }
        }
    }

    Ok(Tensor::new(data, x.shape.clone()))
}

/// Layer normalization: normalize over the last dimension.
/// y = (x - mean) / sqrt(var + eps) * scale + bias
pub fn layer_norm(
    x: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
    axis: i64,
) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };

    let norm_size: usize = x.shape[ax..].iter().product();

    let mut data = x.data.clone();

    // SIMD fast path: LayerNorm over last `norm_size` elements per chunk
    #[cfg(feature = "simd")]
    {
        let bias_data = bias.map(|b| b.data.as_slice());
        crate::simd_ops::simd_layer_norm_strided(&mut data, norm_size, &scale.data, bias_data, eps);
        Ok(Tensor::new(data, x.shape.clone()))
    }

    #[cfg(not(feature = "simd"))]
    {
        let outer: usize = x.shape[..ax].iter().product();
        for o in 0..outer {
            let slice = &mut data[o * norm_size..(o + 1) * norm_size];
            // mean
            let mean = slice.iter().sum::<f32>() / norm_size as f32;
            // variance
            let var = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / norm_size as f32;
            let inv_std = (var + eps).sqrt().recip();
            for v in slice.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }

        // scale and bias (broadcast over outer dims)
        let scale_len = scale.numel();
        for (i, v) in data.iter_mut().enumerate() {
            let s = scale.data[i % scale_len];
            *v *= s;
        }
        if let Some(bias) = bias {
            let bias_len = bias.numel();
            for (i, v) in data.iter_mut().enumerate() {
                *v += bias.data[i % bias_len];
            }
        }

        Ok(Tensor::new(data, x.shape.clone()))
    }
}

/// Group normalization (special case: group_norm with groups=1 = layer_norm over C*H*W).
pub fn group_norm(
    x: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    num_groups: usize,
    eps: f32,
) -> Result<Tensor, String> {
    // x: [N, C, *] — normalize within each group of channels
    if x.ndim() < 2 {
        return Err("group_norm: need at least 2D input".into());
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product::<usize>().max(1);
    if c % num_groups != 0 {
        return Err(format!(
            "group_norm: C={c} not divisible by num_groups={num_groups}"
        ));
    }
    let group_size = c / num_groups * spatial;

    let mut data = x.data.clone();
    for ni in 0..n {
        for g in 0..num_groups {
            let start = ni * c * spatial + g * group_size;
            let slice = &mut data[start..start + group_size];
            let mean = slice.iter().sum::<f32>() / group_size as f32;
            let var = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / group_size as f32;
            let inv_std = (var + eps).sqrt().recip();
            for v in slice.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }
    }
    // Apply per-channel scale and bias
    for ni in 0..n {
        for ci in 0..c {
            let s = scale.data[ci % scale.numel()];
            let b = bias.map(|b| b.data[ci % b.numel()]).unwrap_or(0.0);
            for si in 0..spatial {
                let idx = ni * c * spatial + ci * spatial + si;
                data[idx] = data[idx] * s + b;
            }
        }
    }
    Ok(Tensor::new(data, x.shape.clone()))
}

/// LogSoftmax: log(softmax(x)) computed in a numerically stable way.
/// log_softmax(x) = x - max - log(sum(exp(x - max))) along axis.
pub fn log_softmax(x: &Tensor, axis: i64) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "log_softmax: axis {axis} out of range for {ndim}D tensor"
        ));
    }

    let mut data = x.data.clone();
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];

    for o in 0..outer {
        for i in 0..inner {
            // find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }
            // compute log(sum(exp(x - max)))
            let mut sum_exp = 0.0f32;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                sum_exp += (data[idx] - max_val).exp();
            }
            let log_sum_exp = sum_exp.ln();
            // log_softmax = x - max - log_sum_exp
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                data[idx] = data[idx] - max_val - log_sum_exp;
            }
        }
    }

    Ok(Tensor::new(data, x.shape.clone()))
}

/// InstanceNorm: per-instance, per-channel normalization across spatial dims.
/// x: [N, C, d1, d2, ...], normalize across d1,d2,... for each (n,c) pair.
pub fn instance_norm(
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor, String> {
    if x.ndim() < 3 {
        return Err("instance_norm: input must have at least 3 dimensions".into());
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();
    let mut data = vec![0.0f32; x.data.len()];

    for batch in 0..n {
        for ch in 0..c {
            let offset = (batch * c + ch) * spatial;
            let slice = &x.data[offset..offset + spatial];

            let mean = slice.iter().sum::<f32>() / spatial as f32;
            let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / spatial as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let s = scale.data[ch];
            let b = bias.data[ch];

            for i in 0..spatial {
                data[offset + i] = (slice[i] - mean) * inv_std * s + b;
            }
        }
    }
    Ok(Tensor::new(data, x.shape.clone()))
}

/// LpNormalization: normalize along axis using Lp norm.
/// p=1: L1 norm, p=2: L2 norm (default).
pub fn lp_norm(x: &Tensor, axis: i64, p: i64) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("lp_norm: axis {ax} out of range for ndim {ndim}"));
    }

    let mut data = x.data.clone();
    let axis_size = x.shape[ax];
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            let norm: f32 = if p == 1 {
                (0..axis_size)
                    .map(|a| x.data[(o * axis_size + a) * inner + i].abs())
                    .sum()
            } else {
                (0..axis_size)
                    .map(|a| {
                        let v = x.data[(o * axis_size + a) * inner + i];
                        v * v
                    })
                    .sum::<f32>()
                    .sqrt()
            };
            let norm = if norm == 0.0 { 1.0 } else { norm };
            for a in 0..axis_size {
                data[(o * axis_size + a) * inner + i] /= norm;
            }
        }
    }
    Ok(Tensor::new(data, x.shape.clone()))
}

/// MeanVarianceNormalization: (x - mean(x)) / sqrt(var(x) + epsilon) across specified axes.
/// Default axes = [0, 2, 3] (across batch and spatial, keeping channel).
pub fn mean_variance_normalization(x: &Tensor, axes: &[i64]) -> Result<Tensor, String> {
    let ndim = x.ndim();
    if ndim == 0 {
        return Err("mean_variance_normalization: input must have at least 1 dimension".into());
    }

    // Normalize axes to positive values
    let norm_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (a + ndim as i64) as usize
            } else {
                a as usize
            }
        })
        .collect();

    for &ax in &norm_axes {
        if ax >= ndim {
            return Err(format!(
                "mean_variance_normalization: axis {ax} out of range for {ndim}D tensor"
            ));
        }
    }

    let total = x.numel();
    let mut data = x.data.clone();

    // Build strides
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * x.shape[d + 1];
    }

    // Determine which dims are reduced
    let mut is_reduced = vec![false; ndim];
    for &ax in &norm_axes {
        is_reduced[ax] = true;
    }

    // Compute size of reduced portion and non-reduced portion
    let reduced_size: usize = (0..ndim)
        .filter(|&d| is_reduced[d])
        .map(|d| x.shape[d])
        .product::<usize>()
        .max(1);
    let non_reduced_size = total / reduced_size;

    // Collect non-reduced dims
    let non_reduced_dims: Vec<usize> = (0..ndim).filter(|&d| !is_reduced[d]).collect();
    let reduced_dims: Vec<usize> = (0..ndim).filter(|&d| is_reduced[d]).collect();

    // For each non-reduced index combination, gather indices, compute mean/var, normalize
    for nr_idx in 0..non_reduced_size {
        // Decode nr_idx into multi-index for non-reduced dims
        let mut nr_coords = vec![0usize; non_reduced_dims.len()];
        let mut rem = nr_idx;
        for j in (0..non_reduced_dims.len()).rev() {
            let dim = non_reduced_dims[j];
            nr_coords[j] = rem % x.shape[dim];
            rem /= x.shape[dim];
        }

        // Collect all flat indices for this non-reduced combo
        let mut flat_indices = Vec::with_capacity(reduced_size);
        collect_reduced_indices(
            x,
            &strides,
            &reduced_dims,
            &non_reduced_dims,
            &nr_coords,
            0,
            0,
            &mut flat_indices,
        );

        // Compute mean
        let mean: f32 =
            flat_indices.iter().map(|&fi| x.data[fi]).sum::<f32>() / reduced_size as f32;
        // Compute variance
        let var: f32 = flat_indices
            .iter()
            .map(|&fi| (x.data[fi] - mean) * (x.data[fi] - mean))
            .sum::<f32>()
            / reduced_size as f32;
        let inv_std = 1.0 / (var + 1e-9_f32).sqrt();

        for &fi in &flat_indices {
            data[fi] = (data[fi] - mean) * inv_std;
        }
    }

    Ok(Tensor::new(data, x.shape.clone()))
}

/// Helper: recursively collect flat indices for reduced dimensions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_reduced_indices(
    x: &Tensor,
    strides: &[usize],
    reduced_dims: &[usize],
    non_reduced_dims: &[usize],
    nr_coords: &[usize],
    rd_pos: usize,
    base_offset: usize,
    out: &mut Vec<usize>,
) {
    if rd_pos == reduced_dims.len() {
        // All reduced dims assigned; compute final flat index
        let mut idx = base_offset;
        for (j, &dim) in non_reduced_dims.iter().enumerate() {
            idx += nr_coords[j] * strides[dim];
        }
        out.push(idx);
        return;
    }
    let dim = reduced_dims[rd_pos];
    for coord in 0..x.shape[dim] {
        collect_reduced_indices(
            x,
            strides,
            reduced_dims,
            non_reduced_dims,
            nr_coords,
            rd_pos + 1,
            base_offset + coord * strides[dim],
            out,
        );
    }
}

/// RMS normalization over the last dimension: y = x / sqrt(mean(x²) + eps) * scale
pub fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor, String> {
    if x.ndim() < 1 {
        return Err("rms_norm: input must be at least 1D".into());
    }
    let norm_size = *x
        .shape
        .last()
        .ok_or_else(|| "rms_norm: empty shape".to_string())?;
    let outer = x.numel() / norm_size;
    let scale_len = scale.numel();
    let mut data = x.data.clone();

    for o in 0..outer {
        let slice = &x.data[o * norm_size..(o + 1) * norm_size];
        let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / norm_size as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for j in 0..norm_size {
            data[o * norm_size + j] =
                x.data[o * norm_size + j] * inv_rms * scale.data[j % scale_len];
        }
    }

    Ok(Tensor::new(data, x.shape.clone()))
}

/// Batch normalization (inference mode): y = (x - mean) / sqrt(var + eps) * scale + bias
pub fn batch_norm(
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    eps: f32,
) -> Result<Tensor, String> {
    if x.ndim() < 2 {
        return Err("batch_norm: need at least 2D".into());
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = if x.ndim() > 2 {
        x.shape[2..].iter().product()
    } else {
        1
    };

    let mut data = x.data.clone();
    for ni in 0..n {
        for ci in 0..c {
            let s = scale.data[ci];
            let b = bias.data[ci];
            let m = mean.data[ci];
            let v = var.data[ci];
            let inv_std = (v + eps).sqrt().recip();
            for si in 0..spatial {
                let idx = ni * c * spatial + ci * spatial + si;
                data[idx] = (data[idx] - m) * inv_std * s + b;
            }
        }
    }
    Ok(Tensor::new(data, x.shape.clone()))
}

/// Hardmax: along `axis`, produces a one-hot tensor with 1.0 at the argmax, 0.0 elsewhere.
///
/// ONNX spec: same shape as input, 1.0 at position of maximum value along `axis`.
pub fn hardmax(x: &Tensor, axis: i64) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!(
            "hardmax: axis {axis} out of range for {ndim}D tensor"
        ));
    }
    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
    let axis_len = x.shape[ax];
    let mut out = vec![0.0f32; x.numel()];
    for o in 0..outer {
        for i in 0..inner {
            let mut best_k = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for k in 0..axis_len {
                let idx = o * axis_len * inner + k * inner + i;
                if x.data[idx] > best_v {
                    best_v = x.data[idx];
                    best_k = k;
                }
            }
            out[o * axis_len * inner + best_k * inner + i] = 1.0;
        }
    }
    Ok(Tensor::new(out, x.shape.clone()))
}
