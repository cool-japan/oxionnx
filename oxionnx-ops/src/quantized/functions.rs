//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxionnx_core::{OnnxError, Tensor};

use super::types::QuantizedTensor;

/// Quantized matrix multiplication: A (f32) x B (i8) -> C (f32).
///
/// This is the most common pattern in quantized inference:
/// activations remain in f32, weights are quantized to i8.
/// Accumulation is done in f32 after dequantizing i8 values.
///
/// A: \[M, K\] f32 activations
/// B: QuantizedTensor \[K, N\] i8 weights
/// Returns: \[M, N\] f32 output
pub fn quantized_matmul(a: &Tensor, b: &QuantizedTensor) -> Result<Tensor, OnnxError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(OnnxError::ShapeMismatch(
            "quantized_matmul: expected 2D tensors".into(),
        ));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    if k != b.shape[0] {
        return Err(OnnxError::ShapeMismatch(format!(
            "quantized_matmul: inner dims mismatch: {} vs {}",
            k, b.shape[0]
        )));
    }
    let mut out = vec![0.0f32; m * n];
    if !b.params.per_channel {
        let scale = b.params.scale[0];
        let zp = b.params.zero_point[0] as i32;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let a_val = a.data[i * k + p];
                    let b_val = (b.data[p * n + j] as i32 - zp) as f32 * scale;
                    acc += a_val * b_val;
                }
                out[i * n + j] = acc;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                let ch_scale = if j < b.params.scale.len() {
                    b.params.scale[j]
                } else {
                    1.0
                };
                let ch_zp = if j < b.params.zero_point.len() {
                    b.params.zero_point[j] as i32
                } else {
                    0
                };
                let mut acc = 0.0f32;
                for p in 0..k {
                    let a_val = a.data[i * k + p];
                    let b_val = (b.data[p * n + j] as i32 - ch_zp) as f32 * ch_scale;
                    acc += a_val * b_val;
                }
                out[i * n + j] = acc;
            }
        }
    }
    Ok(Tensor::new(out, vec![m, n]))
}
/// Fully quantized matmul: A (i8) x B (i8) -> C (f32).
///
/// Both inputs are quantized. Uses optimized integer arithmetic with
/// precomputed row/column sums to handle non-zero zero points efficiently.
///
/// Mathematical decomposition:
///   `C[i][j] = scale_a * scale_b * Σ_k (A_q[i][k] - zp_a) * (B_q[k][j] - zp_b)`
///   `= scale_a * scale_b * (A_q@B_q - zp_a*colsum(B_q) - zp_b*rowsum(A_q) + K*zp_a*zp_b)[i][j]`
pub fn fully_quantized_matmul(
    a: &QuantizedTensor,
    b: &QuantizedTensor,
) -> Result<Tensor, OnnxError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(OnnxError::ShapeMismatch(
            "fully_quantized_matmul: expected 2D".into(),
        ));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    if k != b.shape[0] {
        return Err(OnnxError::ShapeMismatch(format!(
            "K mismatch: {} vs {}",
            k, b.shape[0]
        )));
    }
    let a_scale = a.params.scale[0];
    let a_zp = a.params.zero_point[0] as i32;
    let b_scale = b.params.scale[0];
    let b_zp = b.params.zero_point[0] as i32;
    let output_scale = a_scale * b_scale;
    if a_zp == 0 && b_zp == 0 {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for p in 0..k {
                    acc += a.data[i * k + p] as i32 * b.data[p * n + j] as i32;
                }
                out[i * n + j] = acc as f32 * output_scale;
            }
        }
        return Ok(Tensor::new(out, vec![m, n]));
    }
    let row_sum_a: Vec<i32> = (0..m)
        .map(|i| {
            let mut s = 0i32;
            for p in 0..k {
                s += a.data[i * k + p] as i32;
            }
            s
        })
        .collect();
    let mut col_sum_b = vec![0i32; n];
    for p in 0..k {
        for (j, cs) in col_sum_b.iter_mut().enumerate() {
            *cs += b.data[p * n + j] as i32;
        }
    }
    let k_zp_product = k as i32 * a_zp * b_zp;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut raw = 0i32;
            for p in 0..k {
                raw += a.data[i * k + p] as i32 * b.data[p * n + j] as i32;
            }
            let corrected = raw - a_zp * col_sum_b[j] - b_zp * row_sum_a[i] + k_zp_product;
            out[i * n + j] = corrected as f32 * output_scale;
        }
    }
    Ok(Tensor::new(out, vec![m, n]))
}
/// QLinearConv: Fully quantized 2D convolution.
///
/// Performs convolution in integer arithmetic with per-channel weight scales,
/// then requantizes the output. Implements the ONNX QLinearConv operator.
///
/// # Arguments
/// * `x_q` - Quantized input \[N,C,H,W\] as i8 values stored in f32
/// * `x_scale` - Input quantization scale
/// * `x_zero_point` - Input zero point
/// * `w_q` - Quantized weights \[OC,IC/g,kH,kW\] as i8 values stored in f32
/// * `w_scale` - Per-channel or per-tensor weight scales
/// * `w_zero_point` - Per-channel or per-tensor weight zero points
/// * `y_scale` - Output quantization scale
/// * `y_zero_point` - Output zero point
/// * `bias` - Optional bias \[OC\] in float
/// * `strides` - Convolution strides \[sH, sW\]
/// * `pads` - Padding \[pad_top, pad_left, pad_bottom, pad_right\]
/// * `group` - Number of groups
#[allow(clippy::too_many_arguments)]
pub fn qlinear_conv2d(
    x_q: &Tensor,
    x_scale: f32,
    x_zero_point: i8,
    w_q: &Tensor,
    w_scale: &[f32],
    w_zero_point: &[i8],
    y_scale: f32,
    y_zero_point: i8,
    bias: Option<&Tensor>,
    strides: &[usize],
    pads: &[usize],
    group: usize,
) -> Result<Tensor, OnnxError> {
    if x_q.shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: input must be 4D [N,C,H,W], got {:?}",
            x_q.shape
        )));
    }
    if w_q.shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: weight must be 4D [OC,IC/g,kH,kW], got {:?}",
            w_q.shape
        )));
    }
    if strides.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "qlinear_conv2d: strides must have at least 2 elements".into(),
        ));
    }
    if pads.len() < 4 {
        return Err(OnnxError::ShapeMismatch(
            "qlinear_conv2d: pads must have at least 4 elements".into(),
        ));
    }
    if y_scale.abs() < 1e-15 {
        return Err(OnnxError::ShapeMismatch(
            "qlinear_conv2d: y_scale is effectively zero".into(),
        ));
    }
    let batch_size = x_q.shape[0];
    let c_in = x_q.shape[1];
    let h_in = x_q.shape[2];
    let w_in = x_q.shape[3];
    let c_out = w_q.shape[0];
    let c_per_group = w_q.shape[1];
    let k_h = w_q.shape[2];
    let k_w = w_q.shape[3];
    if c_in != c_per_group * group {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: input channels {} != weight IC/g {} * group {}",
            c_in, c_per_group, group
        )));
    }
    if c_out % group != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: c_out {} not divisible by group {}",
            c_out, group
        )));
    }
    let per_channel_w = w_scale.len() > 1;
    if per_channel_w && w_scale.len() != c_out {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: w_scale len {} != c_out {}",
            w_scale.len(),
            c_out
        )));
    }
    if per_channel_w && w_zero_point.len() != c_out {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: w_zero_point len {} != c_out {}",
            w_zero_point.len(),
            c_out
        )));
    }
    let h_out = (h_in + pads[0] + pads[2] - k_h) / strides[0] + 1;
    let w_out = (w_in + pads[1] + pads[3] - k_w) / strides[1] + 1;
    let c_out_per_group = c_out / group;
    let col_rows = c_per_group * k_h * k_w;
    let col_cols = h_out * w_out;
    let x_zp_i32 = x_zero_point as i32;
    let mut output = vec![0.0f32; batch_size * c_out * h_out * w_out];
    for batch in 0..batch_size {
        for g in 0..group {
            let in_c_start = g * c_per_group;
            let mut col = vec![0i32; col_rows * col_cols];
            let mut row = 0usize;
            for ic in 0..c_per_group {
                let in_c = in_c_start + ic;
                let plane_off = (batch * c_in + in_c) * h_in * w_in;
                for ky in 0..k_h {
                    for kx in 0..k_w {
                        for oy in 0..h_out {
                            let iy = (oy * strides[0] + ky) as isize - pads[0] as isize;
                            let base = row * col_cols + oy * w_out;
                            if iy < 0 || iy >= h_in as isize {
                                for ox in 0..w_out {
                                    col[base + ox] = x_zp_i32;
                                }
                            } else {
                                let iy_u = iy as usize;
                                for ox in 0..w_out {
                                    let ix = (ox * strides[1] + kx) as isize - pads[1] as isize;
                                    col[base + ox] = if ix >= 0 && ix < w_in as isize {
                                        x_q.data[plane_off + iy_u * w_in + ix as usize] as i32
                                    } else {
                                        x_zp_i32
                                    };
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
            let mut col_sums = vec![0i32; col_cols];
            for r in 0..col_rows {
                for c_idx in 0..col_cols {
                    col_sums[c_idx] += col[r * col_cols + c_idx];
                }
            }
            for oc in 0..c_out_per_group {
                let global_oc = g * c_out_per_group + oc;
                let w_sc = if per_channel_w {
                    w_scale[global_oc]
                } else {
                    w_scale[0]
                };
                let w_zp_i32 = if per_channel_w {
                    w_zero_point[global_oc] as i32
                } else {
                    w_zero_point[0] as i32
                };
                let w_base = global_oc * col_rows;
                let mut w_row_sum = 0i32;
                for r in 0..col_rows {
                    w_row_sum += w_q.data[w_base + r] as i32;
                }
                let bias_i32 = if let Some(b) = bias {
                    let combined_scale = x_scale * w_sc;
                    if combined_scale.abs() < 1e-15 {
                        0i32
                    } else {
                        (b.data[global_oc] / combined_scale).round() as i32
                    }
                } else {
                    0i32
                };
                let requant_scale = x_scale * w_sc / y_scale;
                let y_zp_f = y_zero_point as f32;
                let zp_correction = col_rows as i32 * x_zp_i32 * w_zp_i32;
                let o_base = (batch * c_out + global_oc) * col_cols;
                for sp in 0..col_cols {
                    let mut raw_sum = 0i32;
                    for r in 0..col_rows {
                        raw_sum += w_q.data[w_base + r] as i32 * col[r * col_cols + sp];
                    }
                    let corrected = raw_sum - x_zp_i32 * w_row_sum - w_zp_i32 * col_sums[sp]
                        + zp_correction
                        + bias_i32;
                    let y_q = (corrected as f32 * requant_scale + y_zp_f)
                        .round()
                        .clamp(-128.0, 127.0);
                    output[o_base + sp] = y_q;
                }
            }
        }
    }
    Ok(Tensor::new(output, vec![batch_size, c_out, h_out, w_out]))
}
/// Dynamic quantization: compute optimal uint8 quantization parameters from data.
///
/// Returns `(quantized_tensor, scale, zero_point)` where values are in \[0, 255\]
/// stored as f32 (uint8 semantics). The zero\_point is returned as i8 per ONNX
/// convention (reinterpret as u8).
///
/// The range always includes 0 to avoid bias in ReLU-like activations.
pub fn dynamic_quantize(x: &Tensor) -> Result<(Tensor, f32, i8), String> {
    if x.data.is_empty() {
        return Err("dynamic_quantize: empty tensor".into());
    }
    let min_val = x
        .data
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .min(0.0);
    let max_val = x
        .data
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.0);
    let range = max_val - min_val;
    let scale = if range < 1e-10 { 1e-10 } else { range / 255.0 };
    let zp_f = (-min_val / scale).round().clamp(0.0, 255.0);
    let zero_point = zp_f as u8 as i8;
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|&v| (v / scale + zp_f).round().clamp(0.0, 255.0))
        .collect();
    Ok((Tensor::new(data, x.shape.clone()), scale, zero_point))
}
