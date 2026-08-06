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
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    let norm_size: usize = x.shape[ax..].iter().product();

    #[cfg(feature = "simd")]
    {
        // `simd_layer_norm_strided` normalizes in place, so (only on this
        // path) `out` needs to start as a copy of `x`; the non-SIMD path
        // below reads `x.data` directly and writes `out` in a single pass,
        // so it never needs this copy at all.
        out.copy_from_slice(&x.data);
        let bias_data = bias.map(|b| b.data.as_slice());
        crate::simd_ops::simd_layer_norm_strided(out, norm_size, &scale.data, bias_data, eps);
        Ok(())
    }

    #[cfg(not(feature = "simd"))]
    {
        let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
        let scale_len = scale.numel();
        let bias_len = bias.map(|b| b.numel());
        // ONNX-conformant case (`scale`/`bias` cover exactly one normalized
        // group): index directly, no modulo -- `j % norm_size == j` for
        // every `j < norm_size` anyway, so this is bit-identical to the
        // modulo form for every input that hits it, just without the
        // per-element hardware division. A model supplying a shorter/longer
        // scale or bias (legal but unusual broadcast) falls back to the
        // original modulo-indexed loop just below.
        let fast_path = scale_len == norm_size && bias_len.map_or(true, |bl| bl == norm_size);

        for o in 0..outer {
            let in_slice = &x.data[o * norm_size..(o + 1) * norm_size];
            let out_slice = &mut out[o * norm_size..(o + 1) * norm_size];
            let mean = in_slice.iter().sum::<f32>() / norm_size as f32;
            let var = in_slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / norm_size as f32;
            let inv_std = (var + eps).sqrt().recip();

            if fast_path {
                for ((o_v, &i_v), &s) in out_slice
                    .iter_mut()
                    .zip(in_slice.iter())
                    .zip(scale.data[..norm_size].iter())
                {
                    *o_v = (i_v - mean) * inv_std * s;
                }
                if let Some(b) = bias {
                    for (o_v, &bv) in out_slice.iter_mut().zip(b.data[..norm_size].iter()) {
                        *o_v += bv;
                    }
                }
            } else {
                for (j, (o_v, &i_v)) in out_slice.iter_mut().zip(in_slice.iter()).enumerate() {
                    *o_v = (i_v - mean) * inv_std * scale.data[j % scale_len];
                }
                if let Some(b) = bias {
                    let bl = b.numel();
                    for (j, o_v) in out_slice.iter_mut().enumerate() {
                        *o_v += b.data[j % bl];
                    }
                }
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
    if norm_size == 0 {
        // A zero-size trailing dim makes the whole tensor empty (nothing to
        // normalize); `x.numel() / norm_size` below would otherwise be an
        // integer division by zero for a shape a dynamic-axis model can
        // legitimately produce.
        return Ok(());
    }
    let outer = x.numel() / norm_size;
    let scale_len = scale.numel();
    // See `layer_norm_into` above: bit-identical to the modulo form for
    // every input that hits it, since `j % norm_size == j` when `j <
    // norm_size`.
    let fast_path = scale_len == norm_size;

    for o in 0..outer {
        let in_slice = &x.data[o * norm_size..(o + 1) * norm_size];
        let out_slice = &mut out[o * norm_size..(o + 1) * norm_size];
        let mean_sq = in_slice.iter().map(|&v| v * v).sum::<f32>() / norm_size as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        if fast_path {
            for ((o_v, &i_v), &s) in out_slice
                .iter_mut()
                .zip(in_slice.iter())
                .zip(scale.data[..norm_size].iter())
            {
                *o_v = i_v * inv_rms * s;
            }
        } else {
            for (j, (o_v, &i_v)) in out_slice.iter_mut().zip(in_slice.iter()).enumerate() {
                *o_v = i_v * inv_rms * scale.data[j % scale_len];
            }
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

/// [a1-14] Resolve GroupNormalization's `scale`/`bias` to one value per
/// channel. ONNX has specified two different affine-parameter shapes across
/// opsets: per-group (length `num_groups`, pre-opset-21) and per-channel
/// (length `C`, opset 21+). The two must NOT be conflated via a blind
/// `data[ci % numel]` modulo -- that only happens to look right when
/// `numel == C` and silently scrambles the group->channel mapping when
/// `numel == num_groups` (e.g. C=4, num_groups=2, scale=[a,b] must apply
/// a,a,b,b to channels 0..3, but `ci % 2` gives a,b,a,b instead). Any other
/// length is an unambiguous shape error rather than a silently-wrapped guess.
fn group_norm_channel_affine(
    param: &Tensor,
    c: usize,
    num_groups: usize,
    param_name: &str,
) -> Result<Vec<f32>, String> {
    let numel = param.numel();
    if numel == c {
        Ok(param.data.clone())
    } else if numel == num_groups {
        let channels_per_group = c / num_groups;
        Ok((0..c)
            .map(|ci| param.data[ci / channels_per_group])
            .collect())
    } else {
        Err(format!(
            "group_norm: {param_name} has {numel} elements, expected {c} (per-channel) or {num_groups} (per-group)"
        ))
    }
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
    let scale_per_channel = group_norm_channel_affine(scale, c, num_groups, "scale")?;
    let bias_per_channel: Vec<f32> = match bias {
        Some(b) => group_norm_channel_affine(b, c, num_groups, "bias")?,
        None => vec![0.0; c],
    };
    for ni in 0..n {
        for ci in 0..c {
            let s = scale_per_channel[ci];
            let b = bias_per_channel[ci];
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

    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];

    #[cfg(feature = "simd")]
    {
        if inner == 1 {
            // Only this path needs `out` pre-loaded with `x`'s data (the
            // kernel normalizes each axis-length chunk in place); the
            // general path below reads `x.data` through a small gather
            // buffer and never needs a whole-tensor copy.
            out.copy_from_slice(&x.data);
            crate::simd_ops::simd_softmax_strided(out, axis_len);
            return Ok(());
        }
    }

    // Strided gather into a small contiguous scratch buffer, compute
    // max/exp/sum/normalize entirely within that buffer (sequential access,
    // no repeated `o*axis_len*inner + k*inner + i` multiply per element --
    // the strided offset is hoisted once per (o, i) and then just advances by
    // `inner`), then a strided scatter back. That is two strided touches per
    // element (one gather read, one scatter write) instead of the previous
    // four full-tensor traversals (the whole-tensor copy, plus a strided
    // read for the max scan, a strided read+write for exp/sum, and a strided
    // read+write for normalize) -- with the exact same max -> exp/sum ->
    // normalize arithmetic in the exact same order, so the result is
    // bit-identical, not just within tolerance.
    let outer: usize = x.shape[..ax].iter().product();
    let mut scratch = vec![0.0f32; axis_len];
    for o in 0..outer {
        let o_base = o * axis_len * inner;
        for i in 0..inner {
            let mut idx = o_base + i;
            for slot in scratch.iter_mut() {
                *slot = x.data[idx];
                idx += inner;
            }

            let mut max_val = f32::NEG_INFINITY;
            for &v in scratch.iter() {
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum = 0.0f32;
            for v in scratch.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            let inv = sum.recip();
            for v in scratch.iter_mut() {
                *v *= inv;
            }

            let mut idx = o_base + i;
            for &v in scratch.iter() {
                out[idx] = v;
                idx += inner;
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

    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();
    let axis_len = x.shape[ax];

    // Same strided-gather / contiguous-compute / strided-scatter
    // restructuring as `softmax_into` above -- bit-identical to the previous
    // four-traversal version. Deliberately *no* SIMD escape here (unlike
    // `softmax_into`'s `inner == 1` fast path): reusing the SIMD softmax
    // kernel via `ln(softmax(x))` is not a safe substitute for the direct
    // `x - max - ln(sum)` formula below. For an `x_i` far enough below the
    // axis max, `exp(x_i - max)` underflows to exactly `0.0f32`, and
    // `ln(0.0)` is `-inf` -- an unbounded error versus the finite value the
    // direct formula produces for the same input.
    let mut scratch = vec![0.0f32; axis_len];
    for o in 0..outer {
        let o_base = o * axis_len * inner;
        for i in 0..inner {
            let mut idx = o_base + i;
            for slot in scratch.iter_mut() {
                *slot = x.data[idx];
                idx += inner;
            }

            let mut max_val = f32::NEG_INFINITY;
            for &v in scratch.iter() {
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum_exp = 0.0f32;
            for &v in scratch.iter() {
                sum_exp += (v - max_val).exp();
            }
            let log_sum_exp = sum_exp.ln();

            let mut idx = o_base + i;
            for &v in scratch.iter() {
                out[idx] = v - max_val - log_sum_exp;
                idx += inner;
            }
        }
    }
    Ok(())
}

/// Numerically stable softmax along the specified axis.
/// Numerically stable softmax along the specified axis.
///
/// Delegates to `softmax_into` (allocating the output buffer here instead
/// of mutating a clone of `x.data` in place): the two used to be independent,
/// fully-duplicated implementations of the same max/exp/sum/normalize
/// algorithm, which meant a perf fix landing in one could silently not land
/// in the other. Sharing the implementation is free here -- both still do
/// exactly one output-sized allocation -- and guarantees they can't diverge.
pub fn softmax(x: &Tensor, axis: i64) -> Result<Tensor, String> {
    let mut data = vec![0.0f32; x.data.len()];
    softmax_into(x, axis, &mut data)?;
    Ok(Tensor::new(data, x.shape.clone()))
}

/// Layer normalization: normalize over the last dimension.
/// y = (x - mean) / sqrt(var + eps) * scale + bias
/// Delegates to `layer_norm_into` -- see [`softmax`]'s doc comment for why.
pub fn layer_norm(
    x: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
    axis: i64,
) -> Result<Tensor, String> {
    let mut data = vec![0.0f32; x.data.len()];
    layer_norm_into(x, scale, bias, eps, axis, &mut data)?;
    Ok(Tensor::new(data, x.shape.clone()))
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
    // Apply per-channel scale and bias (per-group affine mapped to its
    // channels, or per-channel affine used directly -- see
    // `group_norm_channel_affine` for why this can't be a blind modulo).
    let scale_per_channel = group_norm_channel_affine(scale, c, num_groups, "scale")?;
    let bias_per_channel: Vec<f32> = match bias {
        Some(b) => group_norm_channel_affine(b, c, num_groups, "bias")?,
        None => vec![0.0; c],
    };
    for ni in 0..n {
        for ci in 0..c {
            let s = scale_per_channel[ci];
            let b = bias_per_channel[ci];
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
/// Delegates to `log_softmax_into` -- see [`softmax`]'s doc comment for why.
pub fn log_softmax(x: &Tensor, axis: i64) -> Result<Tensor, String> {
    let mut data = vec![0.0f32; x.data.len()];
    log_softmax_into(x, axis, &mut data)?;
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
/// Delegates to `rms_norm_into` -- see [`softmax`]'s doc comment for why.
pub fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor, String> {
    let mut data = vec![0.0f32; x.data.len()];
    rms_norm_into(x, scale, eps, &mut data)?;
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

/// [W2-perf-misc] Correctness of the perf rewrites in this file:
/// - `layer_norm`/`rms_norm` (a6-20): hoisting the per-element `% norm_size`
///   into a one-time `scale_len == norm_size` branch.
/// - `softmax`/`log_softmax` (a6-21): replacing the whole-tensor
///   `copy_from_slice` prologue + three strided passes with a strided
///   gather into a small contiguous scratch buffer, contiguous compute, and
///   a strided scatter back.
///
/// Every case is checked against an independent reference implementation of
/// the *original* algorithm (not sharing any code with the rewrite) for
/// exact equality -- both rewrites are pure index-space/data-movement
/// changes with the max/exp/sum/normalize (or mean/var/scale/bias)
/// arithmetic performed in the same order, so there is no floating-point
/// reassociation to tolerate. A handful of cases are additionally
/// cross-checked against NumPy with a small tolerance (float32 Rust vs
/// NumPy's higher-precision internals do not agree to the last bit).
#[cfg(test)]
mod perf_rewrite_tests {
    use super::*;

    fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{label}[{i}]: got {g}, want {w} (delta {}, tol {tol})",
                (g - w).abs()
            );
        }
    }

    // ── layer_norm / rms_norm ───────────────────────────────────────────────

    /// Independent reference: the exact modulo-indexed loop `layer_norm_into`
    /// used before this change. Only exercised under `#[cfg(not(feature =
    /// "simd"))]` below -- see the comment there.
    #[cfg(not(feature = "simd"))]
    fn reference_layer_norm(
        x: &[f32],
        shape: &[usize],
        scale: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
        ax: usize,
    ) -> Vec<f32> {
        let norm_size: usize = shape[ax..].iter().product();
        let outer: usize = shape[..ax].iter().product::<usize>().max(1);
        let mut out = x.to_vec();
        for o in 0..outer {
            let slice = &mut out[o * norm_size..(o + 1) * norm_size];
            let mean = slice.iter().sum::<f32>() / norm_size as f32;
            let var = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / norm_size as f32;
            let inv_std = (var + eps).sqrt().recip();
            for v in slice.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }
        let scale_len = scale.len();
        for (i, v) in out.iter_mut().enumerate() {
            *v *= scale[i % scale_len];
        }
        if let Some(b) = bias {
            let bias_len = b.len();
            for (i, v) in out.iter_mut().enumerate() {
                *v += b[i % bias_len];
            }
        }
        out
    }

    /// Independent reference: the exact modulo-indexed loop `rms_norm_into`
    /// used before this change.
    fn reference_rms_norm(x: &[f32], shape: &[usize], scale: &[f32], eps: f32) -> Vec<f32> {
        let norm_size = *shape.last().expect("non-empty shape");
        let outer = x.len() / norm_size;
        let scale_len = scale.len();
        let mut out = vec![0.0f32; x.len()];
        for o in 0..outer {
            let slice = &x[o * norm_size..(o + 1) * norm_size];
            let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / norm_size as f32;
            let inv_rms = (mean_sq + eps).sqrt().recip();
            for j in 0..norm_size {
                out[o * norm_size + j] = x[o * norm_size + j] * inv_rms * scale[j % scale_len];
            }
        }
        out
    }

    #[test]
    fn matches_numpy_reference_layer_norm_with_scale_and_bias() {
        // ```python
        // x = (np.arange(24, dtype=np.float32) * 0.3 - 3.0).reshape(2,3,4)
        // scale = np.array([1.5, 0.5, -1.0, 2.0], dtype=np.float32)
        // bias = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        // mean = x.mean(-1, keepdims=True); var = ((x-mean)**2).mean(-1, keepdims=True)
        // (x - mean) / np.sqrt(var + 1e-5) * scale + bias
        // ```
        let x_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.3 - 3.0).collect();
        let x = Tensor::new(x_data, vec![2, 3, 4]);
        let scale = Tensor::new(vec![1.5, 0.5, -1.0, 2.0], vec![4]);
        let bias = Tensor::new(vec![0.1, -0.2, 0.3, 0.0], vec![4]);
        let y = layer_norm(&x, &scale, Some(&bias), 1e-5, -1).expect("layer_norm failed");
        let expected = [
            -1.912371, -0.4235967, -0.147194, 2.683164, -1.912372, -0.423597, -0.1471936, 2.683162,
            -1.912372, -0.4235969, -0.1471935, 2.683163, -1.912372, -0.423597, -0.147194, 2.683161,
            -1.912372, -0.4235967, -0.1471933, 2.683162, -1.912373, -0.423597, -0.1471939,
            2.683161,
        ];
        assert_close(&y.data, &expected, 1e-4, "layer_norm vs numpy");
    }

    #[test]
    fn matches_numpy_reference_rms_norm() {
        // ```python
        // x = np.array([1.0, 2.0, -1.0, 0.5, -0.5, 2.0], dtype=np.float32).reshape(2,3)
        // scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        // ms = (x**2).mean(-1, keepdims=True)
        // x / np.sqrt(ms + 1e-6) * scale
        // ```
        let x = Tensor::new(vec![1.0, 2.0, -1.0, 0.5, -0.5, 2.0], vec![2, 3]);
        let scale = Tensor::new(vec![1.0, 2.0, 0.5], vec![3]);
        let y = rms_norm(&x, &scale, 1e-6).expect("rms_norm failed");
        let expected = [
            0.7071066, 2.828426, -0.3535533, 0.4082482, -0.8164963, 0.8164963,
        ];
        assert_close(&y.data, &expected, 1e-4, "rms_norm vs numpy");
    }

    #[test]
    fn layer_norm_and_rms_norm_broadcast_fallback_matches_reference() {
        // `scale`/`bias` shorter than `norm_size` (broadcast/repeat) is legal
        // but takes the modulo fallback branch, not the fast path -- exercise
        // it explicitly against the independent reference above.
        let shapes_and_norms: &[(&[usize], usize, usize)] = &[
            (&[2, 6], 6, 2),    // scale len 2, norm_size 6 (3x repeat)
            (&[3, 1, 8], 8, 4), // scale len 4, norm_size 8 (2x repeat)
            (&[5, 3], 3, 1),    // scale len 1 (pure broadcast)
        ];
        for &(shape, _norm_size, scale_len) in shapes_and_norms {
            let n: usize = shape.iter().product();
            let x_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - 2.0).collect();
            let x = Tensor::new(x_data.clone(), shape.to_vec());
            let scale_data: Vec<f32> = (0..scale_len).map(|i| 0.5 + i as f32 * 0.25).collect();
            let scale = Tensor::new(scale_data.clone(), vec![scale_len]);

            // [Found during this change, NOT introduced by it, and out of
            // this file's owned scope to fix] `layer_norm`'s SIMD path
            // (`simd_ops::{neon,avx2}::layer_norm_inplace`, unconditionally
            // reached whenever the `simd` feature is enabled -- this
            // dispatch already existed before this rewrite, unchanged here)
            // loads `scale`/`bias` in SIMD-lane-width chunks via
            // `scale.as_ptr().add(offset % scale.len())` with no bound on
            // how far that load reads past `scale.len()`. For a `scale`
            // shorter than the lane width (4 on NEON, 8 on AVX2) -- exactly
            // this broadcast/repeat case -- that is an out-of-bounds SIMD
            // load: real undefined behaviour, not a rounding difference, so
            // it must not even be invoked here (a "close enough" tolerance
            // check on the output of UB would be meaningless). Skip the
            // layer_norm side of this check under `simd`; `rms_norm` has no
            // SIMD path at all and is unaffected.
            #[cfg(not(feature = "simd"))]
            {
                // Every shape/norm_size pair above normalizes over exactly
                // the last axis (`norm_size` is that axis's own size), so
                // `axis = -1` always resolves to `shape.len() - 1`.
                debug_assert_eq!(_norm_size, shape[shape.len() - 1]);
                let bias_data: Vec<f32> = (0..scale_len).map(|i| i as f32 * 0.1 - 0.05).collect();
                let bias = Tensor::new(bias_data.clone(), vec![scale_len]);
                let y_ln =
                    layer_norm(&x, &scale, Some(&bias), 1e-5, -1i64).expect("layer_norm failed");
                let want_ln = reference_layer_norm(
                    &x_data,
                    shape,
                    &scale_data,
                    Some(&bias_data),
                    1e-5,
                    shape.len() - 1,
                );
                assert_eq!(
                    y_ln.data, want_ln,
                    "layer_norm broadcast fallback: shape={shape:?}"
                );
            }

            let y_rms = rms_norm(&x, &scale, 1e-6).expect("rms_norm failed");
            let want_rms = reference_rms_norm(&x_data, shape, &scale_data, 1e-6);
            assert_eq!(
                y_rms.data, want_rms,
                "rms_norm broadcast fallback: shape={shape:?}"
            );
        }
    }

    #[test]
    fn rms_norm_zero_norm_size_does_not_panic() {
        let x = Tensor::new(Vec::new(), vec![3, 0]);
        let scale = Tensor::new(Vec::new(), vec![0]);
        let y = rms_norm(&x, &scale, 1e-6).expect("rms_norm over a zero-size axis must not panic");
        assert_eq!(y.data.len(), 0);
    }

    // ── softmax / log_softmax ───────────────────────────────────────────────

    /// Independent reference: the exact `copy_from_slice` + three-strided-pass
    /// softmax `softmax_into` used before this change.
    fn reference_softmax(x: &[f32], shape: &[usize], ax: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        let inner: usize = shape[ax + 1..].iter().product();
        let axis_len = shape[ax];
        let outer: usize = shape[..ax].iter().product();
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
        out
    }

    /// Independent reference: the exact `copy_from_slice` + three-strided-pass
    /// log_softmax `log_softmax_into` used before this change.
    fn reference_log_softmax(x: &[f32], shape: &[usize], ax: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        let outer: usize = shape[..ax].iter().product();
        let inner: usize = shape[ax + 1..].iter().product();
        let axis_len = shape[ax];
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
        out
    }

    #[test]
    fn matches_numpy_reference_softmax_non_contiguous_axis() {
        // ```python
        // x = (np.arange(24, dtype=np.float32) * 0.4 - 4.0).reshape(2,3,4)
        // m = x.max(axis=1, keepdims=True)
        // e = np.exp((x - m).astype(np.float64))
        // (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        // ```
        let x_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.4 - 4.0).collect();
        let x = Tensor::new(x_data, vec![2, 3, 4]);
        let y = softmax(&x, 1).expect("softmax failed");
        let expected = [
            0.03280241, 0.03280241, 0.03280241, 0.03280241, 0.1624714, 0.1624714, 0.1624714,
            0.1624714, 0.8047262, 0.8047262, 0.8047262, 0.8047262, 0.03280242, 0.03280241,
            0.0328024, 0.03280242, 0.1624714, 0.1624714, 0.1624714, 0.1624714, 0.8047262,
            0.8047262, 0.8047262, 0.8047262,
        ];
        assert_close(&y.data, &expected, 1e-5, "softmax vs numpy");
    }

    #[test]
    fn matches_numpy_reference_log_softmax_non_contiguous_axis() {
        // Same input as above; NumPy reference computed the same way with `log`.
        let x_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.4 - 4.0).collect();
        let x = Tensor::new(x_data, vec![2, 3, 4]);
        let y = log_softmax(&x, 1).expect("log_softmax failed");
        let expected = [
            -3.417253, -3.417253, -3.417253, -3.417253, -1.817253, -1.817253, -1.817253, -1.817253,
            -0.2172532, -0.2172532, -0.2172532, -0.2172532, -3.417253, -3.417253, -3.417253,
            -3.417253, -1.817253, -1.817253, -1.817253, -1.817253, -0.2172532, -0.2172531,
            -0.2172532, -0.2172532,
        ];
        assert_close(&y.data, &expected, 1e-5, "log_softmax vs numpy");
    }

    #[test]
    fn softmax_and_log_softmax_gather_scatter_matches_reference_four_pass() {
        // (shape, axis) combinations spanning: axis on a size-1 dim, axis at
        // the front/middle/back, `inner == 1` (contiguous, exercises the SIMD
        // escape when the `simd` feature is on) and `inner > 1` (strided,
        // the case a6-21 is about), plus a case crossing the scratch-buffer
        // reuse across a larger `outer`.
        let cases: &[(&[usize], usize)] = &[
            (&[5], 0),
            (&[4, 5], 0),
            (&[4, 5], 1),
            (&[2, 3, 4], 0),
            (&[2, 3, 4], 1),
            (&[2, 3, 4], 2),
            (&[1, 6, 1], 1),
            (&[6, 5, 4, 3], 1), // larger outer, exercises scratch reuse across many (o,i)
            (&[6, 5, 4, 3], 2),
        ];
        for &(shape, ax) in cases {
            let n: usize = shape.iter().product();
            // Deterministic, non-monotonic values so ties/argmax positions vary.
            let data: Vec<f32> = (0..n)
                .map(|i| (((i * 2654435761u64 as usize) % 4000) as f32) * 0.01 - 20.0)
                .collect();
            let x = Tensor::new(data.clone(), shape.to_vec());

            // `softmax` (unlike `log_softmax`) keeps its pre-existing `inner
            // == 1` SIMD escape (`simd_softmax_strided`, untouched by this
            // change): under the `simd` feature that path performs an
            // actually-vectorized max/exp/sum reduction, which reassociates
            // relative to this test's strictly-sequential scalar reference
            // and so is only guaranteed to match within float tolerance, not
            // bit-for-bit. Every other combination below (the `simd` feature
            // off, or `inner != 1`) runs this file's rewritten scalar
            // gather/contiguous-compute/scatter path, which performs the
            // exact same operations in the exact same order as the
            // reference and so must match it exactly.
            let inner: usize = shape[ax + 1..].iter().product();
            let softmax_may_reassociate = cfg!(feature = "simd") && inner == 1;

            let got_sm = softmax(&x, ax as i64).expect("softmax failed");
            let want_sm = reference_softmax(&data, shape, ax);
            if softmax_may_reassociate {
                assert_close(&got_sm.data, &want_sm, 1e-4, "softmax (simd reduction)");
            } else {
                assert_eq!(got_sm.data, want_sm, "softmax shape={shape:?} axis={ax}");
            }

            let got_lsm = log_softmax(&x, ax as i64).expect("log_softmax failed");
            let want_lsm = reference_log_softmax(&data, shape, ax);
            assert_eq!(
                got_lsm.data, want_lsm,
                "log_softmax shape={shape:?} axis={ax}"
            );

            // Same comparison through the zero-copy `_into` path.
            let mut out_sm = vec![-999.0f32; n];
            softmax_into(&x, ax as i64, &mut out_sm).expect("softmax_into failed");
            if softmax_may_reassociate {
                assert_close(&out_sm, &want_sm, 1e-4, "softmax_into (simd reduction)");
            } else {
                assert_eq!(out_sm, want_sm, "softmax_into shape={shape:?} axis={ax}");
            }

            let mut out_lsm = vec![-999.0f32; n];
            log_softmax_into(&x, ax as i64, &mut out_lsm).expect("log_softmax_into failed");
            assert_eq!(
                out_lsm, want_lsm,
                "log_softmax_into shape={shape:?} axis={ax}"
            );
        }
    }

    #[test]
    fn log_softmax_stays_finite_for_extreme_dynamic_range() {
        // The reason `log_softmax` does NOT reuse the SIMD softmax kernel via
        // `ln(softmax(x))`: for `x_i` far enough below the max, `exp(x_i -
        // max)` underflows to exactly `0.0f32` and `ln(0.0) == -inf`. The
        // direct `x - max - ln(sum)` formula used here stays finite. `-200.0`
        // relative to a `0.0` max is far past the ~-104 underflow threshold
        // for `f32::exp`.
        let x = Tensor::new(vec![0.0, -200.0, -5.0], vec![3]);
        let y = log_softmax(&x, 0).expect("log_softmax failed");
        assert!(y.data.iter().all(|v| v.is_finite()), "{:?}", y.data);
        assert!(
            y.data[1] < -190.0,
            "should stay a large finite negative, got {}",
            y.data[1]
        );
    }
}
