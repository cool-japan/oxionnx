use oxionnx_core::Tensor;

pub fn relu(x: &Tensor) -> Tensor {
    relu_impl(x)
}

#[cfg(feature = "simd")]
fn relu_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_relu(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn relu_impl(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| v.max(0.0)).collect(),
        x.shape.clone(),
    )
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    sigmoid_impl(x)
}

#[cfg(feature = "simd")]
fn sigmoid_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_sigmoid(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn sigmoid_impl(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
        x.shape.clone(),
    )
}

pub fn tanh_op(x: &Tensor) -> Tensor {
    tanh_op_impl(x)
}

#[cfg(feature = "simd")]
fn tanh_op_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_tanh(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn tanh_op_impl(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|&v| v.tanh()).collect(), x.shape.clone())
}

/// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &Tensor) -> Tensor {
    gelu_impl(x)
}

#[cfg(feature = "simd")]
fn gelu_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_gelu(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn gelu_impl(x: &Tensor) -> Tensor {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEF: f32 = 0.044_715;
    Tensor::new(
        x.data
            .iter()
            .map(|&v| {
                let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
                0.5 * v * (1.0 + inner.tanh())
            })
            .collect(),
        x.shape.clone(),
    )
}

/// LeakyRelu: f(x) = x if x >= 0, alpha * x if x < 0
pub fn leaky_relu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * v })
            .collect(),
        x.shape.clone(),
    )
}

/// PRelu: `f(x) = x if x >= 0, slope[c] * x if x < 0`
/// slope shape is typically \[C\] or \[1, C, 1, 1\] -- broadcast per-channel
pub fn prelu(x: &Tensor, slope: &Tensor) -> Tensor {
    let slope_numel = slope.numel();
    if slope_numel == 1 {
        // scalar slope
        let alpha = slope.data[0];
        return Tensor::new(
            x.data
                .iter()
                .map(|&v| if v >= 0.0 { v } else { alpha * v })
                .collect(),
            x.shape.clone(),
        );
    }

    // Per-channel: x is [N, C, ...], slope is [C] or [1, C, 1, 1]
    // Determine channel count from slope
    let c = slope_numel;

    if x.ndim() >= 2 {
        let spatial: usize = if x.ndim() > 2 {
            x.shape[2..].iter().product()
        } else {
            1
        };
        let n = x.shape[0];
        let x_c = x.shape[1];

        let mut data = x.data.clone();
        if x_c == c {
            for ni in 0..n {
                for ci in 0..c {
                    let alpha = slope.data[ci];
                    for si in 0..spatial {
                        let idx = ni * c * spatial + ci * spatial + si;
                        if data[idx] < 0.0 {
                            data[idx] *= alpha;
                        }
                    }
                }
            }
        } else {
            // Fallback: broadcast element-wise
            for (i, v) in data.iter_mut().enumerate() {
                if *v < 0.0 {
                    *v *= slope.data[i % slope_numel];
                }
            }
        }
        Tensor::new(data, x.shape.clone())
    } else {
        // 1D case: broadcast
        Tensor::new(
            x.data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if v >= 0.0 {
                        v
                    } else {
                        slope.data[i % slope_numel] * v
                    }
                })
                .collect(),
            x.shape.clone(),
        )
    }
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

/// Softplus: ln(1 + exp(x)), with numerical stability for large x.
pub fn softplus(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| {
                if v > 20.0 {
                    v
                } else if v < -20.0 {
                    0.0
                } else {
                    (1.0 + v.exp()).ln()
                }
            })
            .collect(),
        x.shape.clone(),
    )
}

/// Softsign: x / (1 + |x|)
pub fn softsign(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| v / (1.0 + v.abs())).collect(),
        x.shape.clone(),
    )
}

/// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
pub fn mish(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| {
                let sp = if v > 20.0 {
                    v
                } else if v < -20.0 {
                    0.0
                } else {
                    (1.0 + v.exp()).ln()
                };
                v * sp.tanh()
            })
            .collect(),
        x.shape.clone(),
    )
}

/// CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1))
pub fn celu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| {
                if v >= 0.0 {
                    v
                } else {
                    alpha * ((v / alpha).exp() - 1.0)
                }
            })
            .collect(),
        x.shape.clone(),
    )
}

/// ELU: x if x >= 0, alpha*(exp(x)-1) if x < 0
pub fn elu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) })
            .collect(),
        x.shape.clone(),
    )
}

/// SELU: gamma * (x if x > 0, alpha*exp(x) - alpha if x <= 0)
/// Default alpha=1.6732632423543772, gamma=1.0507009873554805
pub fn selu(x: &Tensor, alpha: f32, gamma: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| gamma * if v > 0.0 { v } else { alpha * v.exp() - alpha })
            .collect(),
        x.shape.clone(),
    )
}

/// ThresholdedRelu: x if x > alpha, 0 otherwise
pub fn thresholded_relu(x: &Tensor, alpha: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| if v > alpha { v } else { 0.0 })
            .collect(),
        x.shape.clone(),
    )
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

    // Determine the size of the reduction (axes we average over)
    // and the size of the remaining dims (which we iterate independently).
    // For each unique combination of non-reduced indices, compute mean/var over reduced indices.

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
fn collect_reduced_indices(
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

/// Dropout: identity in inference mode (passthrough).
pub fn dropout(x: &Tensor) -> Tensor {
    x.clone()
}

/// SiLU / Swish: y = x * sigmoid(x)
pub fn silu(x: &Tensor) -> Tensor {
    silu_impl(x)
}

#[cfg(feature = "simd")]
fn silu_impl(x: &Tensor) -> Tensor {
    let mut data = x.data.clone();
    crate::simd_ops::simd_silu(&mut data);
    Tensor::new(data, x.shape.clone())
}

#[cfg(not(feature = "simd"))]
fn silu_impl(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| v / (1.0 + (-v).exp())).collect(),
        x.shape.clone(),
    )
}

/// HardSigmoid: y = clamp(alpha * x + beta, 0, 1)
pub fn hard_sigmoid(x: &Tensor, alpha: f32, beta: f32) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| (alpha * v + beta).clamp(0.0, 1.0))
            .collect(),
        x.shape.clone(),
    )
}

/// HardSwish: y = x * HardSigmoid(x, 1/6, 1/2) = x * clamp(x/6 + 0.5, 0, 1)
pub fn hard_swish(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|&v| v * (v / 6.0 + 0.5).clamp(0.0, 1.0))
            .collect(),
        x.shape.clone(),
    )
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

/// Shrink activation: y = x + bias if x < -lambd; x - bias if x > lambd; else 0.
///
/// ONNX spec defaults: bias=0.0, lambd=0.5.
pub fn shrink(x: &Tensor, bias: f32, lambd: f32) -> Tensor {
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|&v| {
            if v < -lambd {
                v + bias
            } else if v > lambd {
                v - bias
            } else {
                0.0
            }
        })
        .collect();
    Tensor::new(data, x.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::OnnxError;

    #[test]
    fn test_softmax_last_dim() -> Result<(), OnnxError> {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let y = softmax(&x, -1)?;
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(y.data[2] > y.data[1] && y.data[1] > y.data[0]);
        Ok(())
    }

    #[test]
    fn test_layer_norm() -> Result<(), OnnxError> {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let scale = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
        let bias = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]);
        let y = layer_norm(&x, &scale, Some(&bias), 1e-5, -1)?;
        // Each row should have mean≈0
        let mean0: f32 = y.data[..3].iter().sum::<f32>() / 3.0;
        assert!(mean0.abs() < 1e-5, "mean={mean0}");
        Ok(())
    }

    #[test]
    fn test_gelu() {
        let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
        let y = gelu(&x);
        assert!((y.data[0]).abs() < 1e-6); // gelu(0) = 0
        assert!(y.data[1] > 0.0); // gelu(1) > 0
        assert!(y.data[2] < 0.0); // gelu(-1) < 0
    }

    #[test]
    fn test_leaky_relu() {
        let x = Tensor::new(vec![2.0, -3.0, 0.0, -1.0], vec![4]);
        let y = leaky_relu(&x, 0.01);
        assert_eq!(y.data[0], 2.0);
        assert!((y.data[1] - (-0.03)).abs() < 1e-6);
        assert_eq!(y.data[2], 0.0);
        assert!((y.data[3] - (-0.01)).abs() < 1e-6);
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
        let y = silu(&x);
        assert!((y.data[0]).abs() < 1e-6);
        assert!(y.data[1] > 0.0 && y.data[1] < 1.0);
        assert!(y.data[2] > -0.5 && y.data[2] < 0.0);
    }

    #[test]
    fn test_hard_sigmoid() {
        // clamp(alpha*x + beta, 0, 1) with alpha=0.2, beta=0.5
        let x = Tensor::new(vec![-10.0, 0.0, 10.0, 1.0], vec![4]);
        let y = hard_sigmoid(&x, 0.2, 0.5);
        assert_eq!(y.data[0], 0.0);
        assert!((y.data[1] - 0.5).abs() < 1e-6);
        assert_eq!(y.data[2], 1.0);
    }

    #[test]
    fn test_hard_swish() {
        // hard_swish(0) = 0 * 0.5 = 0
        let x = Tensor::new(vec![0.0, 3.0, -3.0, 6.0], vec![4]);
        let y = hard_swish(&x);
        assert!((y.data[0]).abs() < 1e-6);
        assert!((y.data[1] - 3.0 * (3.0 / 6.0 + 0.5)).abs() < 1e-5);
        assert_eq!(y.data[2], 0.0); // -3: clamp(-3/6+0.5, 0, 1) = 0
        assert_eq!(y.data[3], 6.0); // 6: 6 * clamp(1.5, 0, 1) = 6
    }

    #[test]
    fn test_rms_norm() -> Result<(), OnnxError> {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let scale = Tensor::new(vec![1.0; 4], vec![4]);
        let y = rms_norm(&x, &scale, 1e-6)?;
        // Each row should have RMS ≈ 1
        let sq_mean: f32 = y.data.iter().map(|&v| v * v).sum::<f32>() / 4.0;
        assert!((sq_mean - 1.0).abs() < 1e-4, "sq_mean={sq_mean}");
        Ok(())
    }

    #[test]
    fn test_prelu_per_channel() {
        // [1, 2, 2, 2] input, 2 channels
        #[rustfmt::skip]
        let x = Tensor::new(vec![
            1.0, -2.0, 3.0, -4.0,  // channel 0
            -1.0, 2.0, -3.0, 4.0,  // channel 1
        ], vec![1, 2, 2, 2]);
        let slope = Tensor::new(vec![0.1, 0.2], vec![2]);
        let y = prelu(&x, &slope);
        assert_eq!(y.data[0], 1.0);
        assert!((y.data[1] - (-0.2)).abs() < 1e-6); // -2 * 0.1
        assert_eq!(y.data[2], 3.0);
        assert!((y.data[3] - (-0.4)).abs() < 1e-6); // -4 * 0.1
        assert!((y.data[4] - (-0.2)).abs() < 1e-6); // -1 * 0.2
        assert_eq!(y.data[5], 2.0);
        assert!((y.data[6] - (-0.6)).abs() < 1e-6); // -3 * 0.2
        assert_eq!(y.data[7], 4.0);
    }

    #[test]
    fn test_log_softmax() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out = log_softmax(&t, -1).expect("log_softmax failed");
        // exp(log_softmax) should sum to ~1.0
        let sum: f32 = out.data.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // All values should be negative (log of probability)
        assert!(out.data.iter().all(|v| *v <= 0.0));
    }

    #[test]
    fn test_log_softmax_2d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
        let out = log_softmax(&t, 1).expect("log_softmax failed");
        // Each row should sum to ~1.0 after exp
        let sum0: f32 = out.data[0..3].iter().map(|v| v.exp()).sum();
        let sum1: f32 = out.data[3..6].iter().map(|v| v.exp()).sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_invalid_axis() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(log_softmax(&t, 5).is_err());
    }

    #[allow(clippy::approx_constant)]
    #[test]
    fn test_softplus() {
        let t = Tensor::new(vec![0.0, 1.0, -1.0, 20.0, -20.0], vec![5]);
        let out = softplus(&t);
        assert!((out.data[0] - 0.6931).abs() < 1e-3); // ln(2)
        assert!(out.data[3] > 19.9); // large x ~ x
        assert!(out.data[4] < 0.01); // large negative ~ 0
    }

    #[test]
    fn test_softsign() {
        let t = Tensor::new(vec![0.0, 1.0, -1.0, 100.0], vec![4]);
        let out = softsign(&t);
        assert!((out.data[0]).abs() < 1e-6);
        assert!((out.data[1] - 0.5).abs() < 1e-6);
        assert!((out.data[2] + 0.5).abs() < 1e-6);
        assert!((out.data[3] - 100.0 / 101.0).abs() < 1e-4);
    }

    #[test]
    fn test_mish() {
        let t = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
        let out = mish(&t);
        assert!((out.data[0]).abs() < 1e-6); // mish(0) = 0
                                             // mish(1) = 1 * tanh(ln(1+e)) ~ 0.8651
        assert!((out.data[1] - 0.8651).abs() < 1e-3);
    }

    #[test]
    fn test_elu() {
        let t = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
        let out = elu(&t, 1.0);
        assert!((out.data[0] - 1.0).abs() < 1e-6);
        assert!((out.data[1]).abs() < 1e-6);
        // alpha*(exp(-1)-1) ~ -0.6321
        assert!((out.data[2] - ((-1.0_f32).exp() - 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_elu_custom_alpha() {
        let t = Tensor::new(vec![-1.0], vec![1]);
        let out = elu(&t, 2.0);
        assert!((out.data[0] - 2.0 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_celu() {
        let t = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
        let out = celu(&t, 1.0);
        assert!((out.data[0] - 1.0).abs() < 1e-6);
        assert!((out.data[1]).abs() < 1e-6);
        // celu(-1, alpha=1) = 1*(exp(-1/1)-1) = exp(-1)-1 ~ -0.6321
        assert!((out.data[2] - ((-1.0_f32).exp() - 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_celu_custom_alpha() {
        let t = Tensor::new(vec![-2.0], vec![1]);
        let out = celu(&t, 0.5);
        let expected = 0.5 * ((-2.0_f32 / 0.5).exp() - 1.0);
        assert!((out.data[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_selu() {
        let t = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
        let alpha = 1.673_263_2_f32;
        let gamma = 1.050_701_f32;
        let out = selu(&t, alpha, gamma);
        assert!((out.data[0] - gamma).abs() < 1e-4);
        // selu(0) = gamma * (alpha*exp(0) - alpha) = gamma * 0 = 0
        assert!((out.data[1]).abs() < 1e-5);
    }

    #[test]
    fn test_thresholded_relu() {
        let t = Tensor::new(vec![-1.0, 0.0, 0.5, 1.0, 2.0], vec![5]);
        let out = thresholded_relu(&t, 1.0);
        assert_eq!(out.data, vec![0.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_instance_norm() {
        // [1, 2, 2, 2] - 1 batch, 2 channels, 2x2 spatial
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = Tensor::new(data, vec![1, 2, 2, 2]);
        let scale = Tensor::new(vec![1.0, 1.0], vec![2]);
        let bias = Tensor::new(vec![0.0, 0.0], vec![2]);
        let out = instance_norm(&x, &scale, &bias, 1e-5).expect("instance_norm failed");
        // Each channel should have approximately zero mean
        let ch0_mean: f32 = out.data[0..4].iter().sum::<f32>() / 4.0;
        assert!(ch0_mean.abs() < 1e-4);
        let ch1_mean: f32 = out.data[4..8].iter().sum::<f32>() / 4.0;
        assert!(ch1_mean.abs() < 1e-4);
    }

    #[test]
    fn test_instance_norm_with_scale_bias() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = Tensor::new(data, vec![1, 2, 2, 2]);
        let scale = Tensor::new(vec![2.0, 3.0], vec![2]);
        let bias = Tensor::new(vec![1.0, -1.0], vec![2]);
        let out = instance_norm(&x, &scale, &bias, 1e-5).expect("instance_norm failed");
        // Channel 0 mean should be bias[0] = 1.0
        let ch0_mean: f32 = out.data[0..4].iter().sum::<f32>() / 4.0;
        assert!((ch0_mean - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_instance_norm_too_few_dims() {
        let x = Tensor::new(vec![1.0, 2.0], vec![2]);
        let scale = Tensor::new(vec![1.0], vec![1]);
        let bias = Tensor::new(vec![0.0], vec![1]);
        assert!(instance_norm(&x, &scale, &bias, 1e-5).is_err());
    }

    #[test]
    fn test_lp_norm_l2() {
        let t = Tensor::new(vec![3.0, 4.0], vec![2]);
        let out = lp_norm(&t, 0, 2).expect("lp_norm failed");
        // L2 norm of [3,4] = 5, so normalized = [0.6, 0.8]
        assert!((out.data[0] - 0.6).abs() < 1e-5);
        assert!((out.data[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_lp_norm_l1() {
        let t = Tensor::new(vec![3.0, -4.0], vec![2]);
        let out = lp_norm(&t, 0, 1).expect("lp_norm failed");
        // L1 norm = 7, so [3/7, -4/7]
        assert!((out.data[0] - 3.0 / 7.0).abs() < 1e-5);
        assert!((out.data[1] - (-4.0 / 7.0)).abs() < 1e-5);
    }

    #[test]
    fn test_lp_norm_invalid_axis() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]);
        assert!(lp_norm(&t, 5, 2).is_err());
    }

    #[test]
    fn test_lp_norm_2d() {
        // [2, 3] tensor, normalize along axis=1
        let t = Tensor::new(vec![3.0, 4.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]);
        let out = lp_norm(&t, 1, 2).expect("lp_norm failed");
        // Row 0: norm = 5, [0.6, 0.8, 0.0]
        assert!((out.data[0] - 0.6).abs() < 1e-5);
        assert!((out.data[1] - 0.8).abs() < 1e-5);
        assert!((out.data[2]).abs() < 1e-5);
        // Row 1: norm = 1, [1.0, 0.0, 0.0]
        assert!((out.data[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_variance_normalization() {
        // Simple 4D case: [1, 2, 1, 2], axes=[0, 2, 3]
        let data = vec![1.0, 3.0, 5.0, 7.0];
        let x = Tensor::new(data, vec![1, 2, 1, 2]);
        let out = mean_variance_normalization(&x, &[0, 2, 3]).expect("mean_var_norm failed");
        // Channel 0 slice: [1, 3], mean=2, var=1, normalized=[-1, 1]
        assert!((out.data[0] - (-1.0)).abs() < 0.1);
        assert!((out.data[1] - 1.0).abs() < 0.1);
        // Channel 1 slice: [5, 7], mean=6, var=1, normalized=[-1, 1]
        assert!((out.data[2] - (-1.0)).abs() < 0.1);
        assert!((out.data[3] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_mean_variance_normalization_default_axes() {
        // 4D [2, 1, 1, 1], axes=[0,2,3]
        let data = vec![2.0, 4.0];
        let x = Tensor::new(data, vec![2, 1, 1, 1]);
        let out = mean_variance_normalization(&x, &[0, 2, 3]).expect("mean_var_norm failed");
        // mean=3, var=1, normalized: [-1, 1]
        assert!((out.data[0] - (-1.0)).abs() < 0.1);
        assert!((out.data[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_mean_variance_normalization_invalid_axis() {
        let x = Tensor::new(vec![1.0], vec![1]);
        assert!(mean_variance_normalization(&x, &[5]).is_err());
    }

    #[test]
    fn test_dropout_identity() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let out = dropout(&t);
        assert_eq!(out.data, t.data);
    }

    // ── J-phase nn ops tests ────────────────────────────────────────────────

    #[test]
    fn test_hardmax_basic() {
        let x = Tensor::new(vec![1.0, 3.0, 2.0], vec![3]);
        let out = hardmax(&x, 0).unwrap();
        assert_eq!(out.data, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_hardmax_negative_axis() {
        let x = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
        let out = hardmax(&x, -1).unwrap();
        // row0: [1,3] → max at idx 1 → [0,1]
        // row1: [2,4] → max at idx 1 → [0,1]
        assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_shrink_basic() {
        let x = Tensor::new(vec![-2.0, -0.3, 0.0, 0.3, 2.0], vec![5]);
        let out = shrink(&x, 0.0, 0.5);
        // -2 < -0.5 → -2+0=-2; -0.3 in [-0.5, 0.5] → 0; 0 → 0; 0.3 → 0; 2 > 0.5 → 2-0=2
        assert!((out.data[0] - (-2.0)).abs() < 1e-5);
        assert!((out.data[1] - 0.0).abs() < 1e-5);
        assert!((out.data[4] - 2.0).abs() < 1e-5);
    }
}
