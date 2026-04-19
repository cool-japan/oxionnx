use oxionnx_core::Tensor;

/// 2D max pooling.
/// input: [N, C, H, W]
pub fn max_pool2d(
    input: &Tensor,
    kernel_shape: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
) -> Tensor {
    let n = input.shape[0];
    let c = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let [kh, kw] = kernel_shape;
    let oh = (h + pads[0] + pads[2] - kh) / strides[0] + 1;
    let ow = (w + pads[1] + pads[3] - kw) / strides[1] + 1;

    let mut out = vec![f32::NEG_INFINITY; n * c * oh * ow];

    for batch in 0..n {
        for ch in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut max_val = f32::NEG_INFINITY;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * strides[0] + ky) as isize - pads[0] as isize;
                            let ix = (ox * strides[1] + kx) as isize - pads[1] as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let iy = iy as usize;
                                let ix = ix as usize;
                                let idx = ((batch * c + ch) * h + iy) * w + ix;
                                if input.data[idx] > max_val {
                                    max_val = input.data[idx];
                                }
                            }
                        }
                    }
                    out[((batch * c + ch) * oh + oy) * ow + ox] = max_val;
                }
            }
        }
    }

    Tensor::new(out, vec![n, c, oh, ow])
}

/// 2D average pooling.
/// input: [N, C, H, W]
pub fn avg_pool2d(
    input: &Tensor,
    kernel_shape: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
    count_include_pad: bool,
) -> Tensor {
    let n = input.shape[0];
    let c = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let [kh, kw] = kernel_shape;
    let oh = (h + pads[0] + pads[2] - kh) / strides[0] + 1;
    let ow = (w + pads[1] + pads[3] - kw) / strides[1] + 1;

    let mut out = vec![0.0f32; n * c * oh * ow];

    for batch in 0..n {
        for ch in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * strides[0] + ky) as isize - pads[0] as isize;
                            let ix = (ox * strides[1] + kx) as isize - pads[1] as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let iy = iy as usize;
                                let ix = ix as usize;
                                let idx = ((batch * c + ch) * h + iy) * w + ix;
                                sum += input.data[idx];
                                count += 1;
                            } else if count_include_pad {
                                count += 1;
                            }
                        }
                    }
                    let divisor = if count_include_pad { kh * kw } else { count };
                    out[((batch * c + ch) * oh + oy) * ow + ox] = if divisor > 0 {
                        sum / divisor as f32
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    Tensor::new(out, vec![n, c, oh, ow])
}

/// Global average pooling: reduce all spatial dimensions to 1.
/// Input: [N, C, d0, d1, ...] → Output: [N, C, 1, 1, ...]
pub fn global_avg_pool(x: &Tensor) -> Tensor {
    if x.ndim() < 3 {
        return x.clone();
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();
    let mut out = vec![0.0f32; n * c];
    for ni in 0..n {
        for ci in 0..c {
            let base = ni * c * spatial + ci * spatial;
            let sum: f32 = x.data[base..base + spatial].iter().sum();
            out[ni * c + ci] = sum / spatial as f32;
        }
    }
    let mut out_shape = vec![n, c];
    out_shape.extend(vec![1usize; x.ndim() - 2]);
    Tensor::new(out, out_shape)
}

/// Global max pooling: reduce all spatial dimensions to 1.
/// Input: [N, C, d0, d1, ...] → Output: [N, C, 1, 1, ...]
pub fn global_max_pool(x: &Tensor) -> Tensor {
    if x.ndim() < 3 {
        return x.clone();
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();
    let mut out = vec![f32::NEG_INFINITY; n * c];
    for ni in 0..n {
        for ci in 0..c {
            let base = ni * c * spatial + ci * spatial;
            let max_val = x.data[base..base + spatial]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            out[ni * c + ci] = max_val;
        }
    }
    let mut out_shape = vec![n, c];
    out_shape.extend(vec![1usize; x.ndim() - 2]);
    Tensor::new(out, out_shape)
}

/// Compute output shape for transposed 2D convolution.
///
/// `input_shape`:  `[N, C_in, H, W]`
/// `weight_shape`: `[C_in, C_out/group, kH, kW]`
/// `pads`:         `[top, left, bottom, right]`
///
/// Returns `[N, C_out, oH, oW]` where `C_out = weight_shape[1] * group`.
pub(crate) fn compute_conv_transpose2d_out_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    output_padding: &[usize],
    dilations: &[usize],
    group: usize,
) -> Vec<usize> {
    let n = input_shape[0];
    let h = input_shape[2];
    let w = input_shape[3];
    let c_out_per_group = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];
    let c_out = c_out_per_group * group;
    let eff_kh = (kh - 1) * dilations[0] + 1;
    let eff_kw = (kw - 1) * dilations[1] + 1;
    let oh = strides[0] * (h - 1) + output_padding[0] + eff_kh - pads[0] - pads[2];
    let ow = strides[1] * (w - 1) + output_padding[1] + eff_kw - pads[1] - pads[3];
    vec![n, c_out, oh, ow]
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

    let (n, c_in, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let c_out_per_group = weight.shape[1];
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    let c_out = c_out_per_group * group;
    let c_in_per_group = c_in / group;

    if c_in % group != 0 {
        return Err(format!(
            "conv_transpose2d: c_in ({}) not divisible by group ({})",
            c_in, group
        ));
    }

    let oh = out_shape[2];
    let ow = out_shape[3];

    // Zero the output buffer before scatter-accumulation.
    out.fill(0.0_f32);

    // For each input element, scatter its contribution to the output.
    for ni in 0..n {
        for g in 0..group {
            for ic in 0..c_in_per_group {
                let ci = g * c_in_per_group + ic;
                for iy in 0..h {
                    for ix in 0..w {
                        let in_val = input.data[((ni * c_in + ci) * h + iy) * w + ix];
                        for oc in 0..c_out_per_group {
                            let co = g * c_out_per_group + oc;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oy_raw = iy * strides[0] + ky * dilations[0];
                                    let ox_raw = ix * strides[1] + kx * dilations[1];
                                    if oy_raw < pads[0] || ox_raw < pads[1] {
                                        continue;
                                    }
                                    let oy = oy_raw - pads[0];
                                    let ox = ox_raw - pads[1];
                                    if oy >= oh || ox >= ow {
                                        continue;
                                    }
                                    let w_val = weight.data
                                        [((ci * c_out_per_group + oc) * kh + ky) * kw + kx];
                                    out[((ni * c_out + co) * oh + oy) * ow + ox] += in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias.
    if let Some(b) = bias {
        for ni in 0..n {
            for co in 0..c_out {
                let bias_val = b.data[co];
                for oy in 0..oh {
                    for ox in 0..ow {
                        out[((ni * c_out + co) * oh + oy) * ow + ox] += bias_val;
                    }
                }
            }
        }
    }

    Ok(())
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
    );
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
