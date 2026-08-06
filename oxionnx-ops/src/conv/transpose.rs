//! Rank-generic transposed convolution (`ConvTranspose1D/2D/3D`).
//!
//! A transposed convolution is a scatter-accumulate: every input element
//! contributes `in_val * weight[k]` to the output position
//! `i * stride + k * dilation - pad_begin`, dropped when that lands outside
//! the (cropped) output extent.
//!
//! The rank-2 loop nest is kept as a specialisation because it is the shape
//! decoders actually use; rank 1 and rank ≥ 3 run the generic nest. The two
//! visit output elements in the same order — batch, group, input channel,
//! input position, output channel, kernel offset — so their floating-point
//! accumulation is bit-identical, which
//! `conv_transpose_generic_matches_rank2_bitwise` asserts directly.

use oxionnx_core::{OnnxError, Tensor};

use super::spatial::{self, odometer_next};

/// Validated spatial parameters of an N-D transposed convolution.
pub(crate) struct ConvTransposeParams<'a> {
    /// One stride per spatial axis.
    pub strides: &'a [usize],
    /// `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]`.
    pub pads: &'a [usize],
    /// One dilation per spatial axis.
    pub dilations: &'a [usize],
    /// Channel-group count.
    pub group: usize,
}

/// Compute output shape for transposed 2D convolution.
///
/// `input_shape`:  `[N, C_in, H, W]`
/// `weight_shape`: `[C_in, C_out/group, kH, kW]`
/// `pads`:         `[top, left, bottom, right]`
///
/// Returns `[N, C_out, oH, oW]` where `C_out = weight_shape[1] * group`.
///
/// Every size computation is checked: a malformed model (rank mismatch,
/// zero-size input, zero stride/dilation, `group == 0`, or padding that
/// exceeds the natural output extent) yields a typed
/// [`OnnxError::ShapeMismatch`] instead of an unsigned underflow.
pub(crate) fn compute_conv_transpose2d_out_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    output_padding: &[usize],
    dilations: &[usize],
    group: usize,
) -> Result<Vec<usize>, OnnxError> {
    if input_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "ConvTranspose: input must be 4D [N,C,H,W], got rank {}",
            input_shape.len()
        )));
    }
    if weight_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "ConvTranspose: weight must be 4D [C_in,C_out/group,kH,kW], got rank {}",
            weight_shape.len()
        )));
    }
    spatial::compute_conv_transpose_out_shape(
        "ConvTranspose",
        input_shape,
        weight_shape,
        strides,
        pads,
        output_padding,
        dilations,
        group,
    )
}

/// Rank-generic transposed convolution into a pre-allocated buffer.
///
/// `out` is zeroed before accumulation begins. `out_shape` must be the shape
/// [`spatial::compute_conv_transpose_out_shape`] derives for the same
/// parameters — the padding is *not* re-applied here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose_into(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    params: &ConvTransposeParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<(), OnnxError> {
    let op = "ConvTranspose";
    let rank = spatial::spatial_rank(input_shape, op, "input")?;
    if weight_shape.len() != input_shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: weight rank {} must equal input rank {} ([C_in, C_out/group, k_0, ...])",
            weight_shape.len(),
            input_shape.len()
        )));
    }
    if out_shape.len() != input_shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: output rank {} must equal input rank {}",
            out_shape.len(),
            input_shape.len()
        )));
    }
    if params.group == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: group must be >= 1, got 0"
        )));
    }
    if params.strides.len() != rank || params.dilations.len() != rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: strides ({}) and dilations ({}) need {rank} entries",
            params.strides.len(),
            params.dilations.len()
        )));
    }
    if params.pads.len() != 2 * rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: pads needs {} entries, got {}",
            2 * rank,
            params.pads.len()
        )));
    }
    if params.strides.contains(&0) || params.dilations.contains(&0) {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: strides and dilations must be >= 1"
        )));
    }
    let c_in = input_shape[1];
    if c_in != weight_shape[0] {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: input channels {c_in} != weight input channels {}",
            weight_shape[0]
        )));
    }
    if c_in % params.group != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: c_in ({c_in}) not divisible by group ({})",
            params.group
        )));
    }
    let c_out_per_group = weight_shape[1];
    let c_out = c_out_per_group
        .checked_mul(params.group)
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: output channel count overflows")))?;
    if out_shape[1] != c_out {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: output channels {} != weight output channels {c_out_per_group} * group {}",
            out_shape[1], params.group
        )));
    }
    let in_len = volume(input_shape, op, "input")?;
    let w_len = volume(weight_shape, op, "weight")?;
    let out_len = volume(out_shape, op, "output")?;
    if input.len() < in_len || weight.len() < w_len || out.len() < out_len {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: buffer too small for its shape (input {} < {in_len}, weight {} < {w_len}, \
             output {} < {out_len})",
            input.len(),
            weight.len(),
            out.len()
        )));
    }
    if let Some(b) = bias {
        if b.len() < c_out {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: bias has {} entries, expected {c_out}",
                b.len()
            )));
        }
    }

    out.fill(0.0_f32);
    if out_len == 0 || in_len == 0 {
        return Ok(());
    }

    if rank == 2 {
        scatter_rank2(
            input,
            input_shape,
            weight,
            weight_shape,
            params,
            out,
            out_shape,
        );
    } else {
        scatter_generic(
            input,
            input_shape,
            weight,
            weight_shape,
            params,
            out,
            out_shape,
        );
    }

    if let Some(b) = bias {
        let n = out_shape[0];
        let out_plane: usize = out_shape[2..].iter().product();
        for ni in 0..n {
            for (co, &bias_val) in b.iter().take(c_out).enumerate() {
                let base = (ni * c_out + co) * out_plane;
                for v in &mut out[base..base + out_plane] {
                    *v += bias_val;
                }
            }
        }
    }
    Ok(())
}

/// Element count of a shape, rejecting an overflowing product.
fn volume(shape: &[usize], op: &str, what: &str) -> Result<usize, OnnxError> {
    shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("{op}: {what} shape {shape:?} overflows usize"))
        })
}

/// Rank-2 scatter-accumulate (the decoder / GAN shape).
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_rank2(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    params: &ConvTransposeParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) {
    let (n, c_in, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let c_out_per_group = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];
    let group = params.group;
    let c_out = c_out_per_group * group;
    let c_in_per_group = c_in / group;
    let oh = out_shape[2];
    let ow = out_shape[3];
    let [s_h, s_w] = [params.strides[0], params.strides[1]];
    let [d_h, d_w] = [params.dilations[0], params.dilations[1]];
    let [p_top, p_left] = [params.pads[0], params.pads[1]];

    for ni in 0..n {
        for g in 0..group {
            for ic in 0..c_in_per_group {
                let ci = g * c_in_per_group + ic;
                for iy in 0..h {
                    for ix in 0..w {
                        let in_val = input[((ni * c_in + ci) * h + iy) * w + ix];
                        for oc in 0..c_out_per_group {
                            let co = g * c_out_per_group + oc;
                            let w_base = (ci * c_out_per_group + oc) * kh * kw;
                            let o_base = (ni * c_out + co) * oh * ow;
                            for ky in 0..kh {
                                let oy_raw = iy * s_h + ky * d_h;
                                if oy_raw < p_top {
                                    continue;
                                }
                                let oy = oy_raw - p_top;
                                if oy >= oh {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ox_raw = ix * s_w + kx * d_w;
                                    if ox_raw < p_left {
                                        continue;
                                    }
                                    let ox = ox_raw - p_left;
                                    if ox >= ow {
                                        continue;
                                    }
                                    out[o_base + oy * ow + ox] +=
                                        in_val * weight[w_base + ky * kw + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Rank-generic scatter-accumulate.
///
/// Visits `(batch, group, in-channel, in-position, out-channel, kernel-offset)`
/// in exactly the order [`scatter_rank2`] does, so both produce bit-identical
/// sums. The leading spatial axes are resolved once per kernel row and only the
/// last axis is walked element-by-element.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_generic(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    params: &ConvTransposeParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) {
    let rank = input_shape.len() - 2;
    if rank == 0 {
        return;
    }
    let last = rank - 1;
    let n = input_shape[0];
    let c_in = input_shape[1];
    let in_spatial = &input_shape[2..];
    let kernel = &weight_shape[2..];
    let out_spatial = &out_shape[2..];
    let c_out_per_group = weight_shape[1];
    let group = params.group;
    let c_out = c_out_per_group * group;
    let c_in_per_group = c_in / group;

    let in_plane: usize = in_spatial.iter().product();
    let out_plane: usize = out_spatial.iter().product();
    let ksize: usize = kernel.iter().product();
    if in_plane == 0 || out_plane == 0 || ksize == 0 {
        return;
    }

    // Row-major strides of the output spatial axes.
    let mut out_stride = vec![1_usize; rank];
    for d in (0..last).rev() {
        out_stride[d] = out_stride[d + 1] * out_spatial[d + 1];
    }

    let k_last = kernel[last];
    let k_outer: usize = kernel[..last].iter().product();
    let stride_last = params.strides[last];
    let dilation_last = params.dilations[last];
    let pad_last = params.pads[last];
    let out_last = out_spatial[last];

    let mut iidx = vec![0_usize; rank];
    let mut kidx = vec![0_usize; last];

    for ni in 0..n {
        for g in 0..group {
            for ic in 0..c_in_per_group {
                let ci = g * c_in_per_group + ic;
                let in_base = (ni * c_in + ci) * in_plane;
                iidx.iter_mut().for_each(|v| *v = 0);
                for iflat in 0..in_plane {
                    let in_val = input[in_base + iflat];
                    for oc in 0..c_out_per_group {
                        let co = g * c_out_per_group + oc;
                        let w_base = (ci * c_out_per_group + oc) * ksize;
                        let o_base = (ni * c_out + co) * out_plane;
                        kidx.iter_mut().for_each(|v| *v = 0);
                        for ko in 0..k_outer {
                            let mut ok = true;
                            let mut off = 0_usize;
                            for d in 0..last {
                                let raw =
                                    iidx[d] * params.strides[d] + kidx[d] * params.dilations[d];
                                if raw < params.pads[d] {
                                    ok = false;
                                    break;
                                }
                                let o = raw - params.pads[d];
                                if o >= out_spatial[d] {
                                    ok = false;
                                    break;
                                }
                                off += o * out_stride[d];
                            }
                            if ok {
                                let w_row = w_base + ko * k_last;
                                let o_row = o_base + off;
                                for kl in 0..k_last {
                                    let raw = iidx[last] * stride_last + kl * dilation_last;
                                    if raw < pad_last {
                                        continue;
                                    }
                                    let o = raw - pad_last;
                                    if o >= out_last {
                                        continue;
                                    }
                                    out[o_row + o] += in_val * weight[w_row + kl];
                                }
                            }
                            if odometer_next(&mut kidx, &kernel[..last]) {
                                break;
                            }
                        }
                    }
                    if odometer_next(&mut iidx, in_spatial) {
                        break;
                    }
                }
            }
        }
    }
}

/// Write transposed conv2d result directly into a pre-allocated output buffer.
///
/// `out` must have length == product of `out_shape` elements.
/// `out` is zeroed before accumulation begins.
///
/// `pads`: `[top, left, bottom, right]`
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose2d_into(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: &[usize],
    pads: &[usize],
    // output_padding is not used here because the caller pre-computes out_shape
    // and passes it directly; the parameter is kept for API symmetry with the
    // public conv_transpose2d function.
    _output_padding: &[usize],
    dilations: &[usize],
    group: usize,
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<(), String> {
    if input.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: input must be 4D, got {}D",
            input.ndim()
        ));
    }
    if weight.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: weight must be 4D, got {}D",
            weight.ndim()
        ));
    }
    if out_shape.len() != 4 {
        return Err(format!(
            "conv_transpose2d: out_shape must be 4D, got rank {}",
            out_shape.len()
        ));
    }
    if strides.len() < 2 || pads.len() < 4 || dilations.len() < 2 {
        return Err(
            "conv_transpose2d: strides/dilations need 2 entries and pads needs 4".to_string(),
        );
    }
    conv_transpose_into(
        &input.data,
        &input.shape,
        &weight.data,
        &weight.shape,
        bias.map(|b| b.data.as_slice()),
        &ConvTransposeParams {
            strides: &strides[..2],
            pads: &pads[..4],
            dilations: &dilations[..2],
            group,
        },
        out,
        out_shape,
    )
    .map_err(|e| e.to_string())
}

/// ConvTranspose2D: fractionally-strided convolution (deconvolution)
/// input: [N, C_in, H, W]
/// weight: [C_in, C_out/group, kH, kW]
/// output: [N, C_out, oH, oW]
/// oH = stride*(H-1) + output_padding + ((kH-1)*dilation + 1) - pad_top - pad_bottom
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4], // [top, left, bottom, right]
    output_padding: [usize; 2],
    dilations: [usize; 2],
    group: usize,
) -> Result<Tensor, String> {
    // Validate dimensions early — before compute_conv_transpose2d_out_shape
    // indexes into the shape slices, which would panic on invalid inputs.
    if input.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: input must be 4D, got {}D",
            input.ndim()
        ));
    }
    if weight.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: weight must be 4D, got {}D",
            weight.ndim()
        ));
    }
    let out_shape = compute_conv_transpose2d_out_shape(
        &input.shape,
        &weight.shape,
        &strides,
        &pads,
        &output_padding,
        &dilations,
        group,
    )
    .map_err(|e| e.to_string())?;
    let out_len: usize = out_shape.iter().product();
    let mut data = vec![0.0_f32; out_len];
    conv_transpose2d_into(
        input,
        weight,
        bias,
        &strides,
        &pads,
        &output_padding,
        &dilations,
        group,
        &mut data,
        &out_shape,
    )?;
    Ok(Tensor::new(data, out_shape))
}

/// Rank-generic transposed convolution returning a fresh tensor.
///
/// `pads` uses the ONNX layout `[begin_0, …, end_{r-1}]`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: &[usize],
    pads: &[usize],
    output_padding: &[usize],
    dilations: &[usize],
    group: usize,
) -> Result<Tensor, OnnxError> {
    let out_shape = spatial::compute_conv_transpose_out_shape(
        "ConvTranspose",
        &input.shape,
        &weight.shape,
        strides,
        pads,
        output_padding,
        dilations,
        group,
    )?;
    let out_len: usize = out_shape.iter().product();
    let mut data = vec![0.0_f32; out_len];
    conv_transpose_into(
        &input.data,
        &input.shape,
        &weight.data,
        &weight.shape,
        bias.map(|b| b.data.as_slice()),
        &ConvTransposeParams {
            strides,
            pads,
            dilations,
            group,
        },
        &mut data,
        &out_shape,
    )?;
    Ok(Tensor::new(data, out_shape))
}
