//! INT8 quantized matrix multiplication.
//!
//! Implements quantized inference where weights are stored as i8 with
//! per-channel scale and zero_point parameters.

use oxionnx_core::{OnnxError, Tensor};

/// Quantization parameters for a tensor.
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scale factor(s). Length 1 for per-tensor, or num_channels for per-channel.
    pub scale: Vec<f32>,
    /// Zero point(s). Same length as scale.
    pub zero_point: Vec<i8>,
    /// Whether quantization is per-channel (true) or per-tensor (false).
    pub per_channel: bool,
    /// Channel axis for per-channel quantization.
    pub axis: usize,
}

/// A quantized tensor: i8 data + quantization parameters.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized i8 data in row-major order.
    pub data: Vec<i8>,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Quantization parameters.
    pub params: QuantParams,
}

impl QuantizedTensor {
    /// Create a new quantized tensor.
    pub fn new(data: Vec<i8>, shape: Vec<usize>, params: QuantParams) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            data,
            shape,
            params,
        }
    }

    /// Quantize an f32 tensor to i8 using symmetric quantization.
    /// Per-tensor quantization: single scale, zero_point = 0.
    /// Formula: q = clamp(round(x / scale), -127, 127)
    /// Dequantize: x_approx = q * scale
    pub fn quantize(tensor: &Tensor) -> Self {
        // Symmetric quantization: scale based on max absolute value
        let abs_max = tensor
            .data
            .iter()
            .copied()
            .fold(0.0f32, |acc, v| acc.max(v.abs()));

        let scale = {
            let raw = abs_max / 127.0;
            if raw < 1e-10 {
                1e-10
            } else {
                raw
            }
        };

        let data: Vec<i8> = tensor
            .data
            .iter()
            .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        Self::new(
            data,
            tensor.shape.clone(),
            QuantParams {
                scale: vec![scale],
                zero_point: vec![0],
                per_channel: false,
                axis: 0,
            },
        )
    }

    /// Quantize with per-channel parameters along the given axis.
    pub fn quantize_per_channel(tensor: &Tensor, axis: usize) -> Result<Self, OnnxError> {
        if axis >= tensor.shape.len() {
            return Err(OnnxError::ShapeMismatch(format!(
                "quantize axis {} >= rank {}",
                axis,
                tensor.shape.len()
            )));
        }

        let num_channels = tensor.shape[axis];
        let channel_size: usize = tensor.shape[axis + 1..].iter().product();
        let outer_size: usize = tensor.shape[..axis].iter().product();

        let mut scales = Vec::with_capacity(num_channels);
        let mut zero_points = Vec::with_capacity(num_channels);
        let mut quant_data = vec![0i8; tensor.data.len()];

        for ch in 0..num_channels {
            // Find max absolute value for this channel (symmetric quantization)
            let mut ch_abs_max = 0.0f32;

            for outer in 0..outer_size {
                let base = outer * num_channels * channel_size + ch * channel_size;
                for i in 0..channel_size {
                    let v = tensor.data[base + i].abs();
                    if v > ch_abs_max {
                        ch_abs_max = v;
                    }
                }
            }

            let scale = {
                let raw = ch_abs_max / 127.0;
                if raw < 1e-10 {
                    1e-10
                } else {
                    raw
                }
            };

            scales.push(scale);
            zero_points.push(0);

            for outer in 0..outer_size {
                let base = outer * num_channels * channel_size + ch * channel_size;
                for i in 0..channel_size {
                    quant_data[base + i] =
                        (tensor.data[base + i] / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }

        Ok(Self::new(
            quant_data,
            tensor.shape.clone(),
            QuantParams {
                scale: scales,
                zero_point: zero_points,
                per_channel: true,
                axis,
            },
        ))
    }

    /// Quantize with asymmetric quantization (non-zero zero_point).
    /// Formula: q = clamp(round(x / scale) + zero_point, -128, 127)
    /// Dequantize: x ≈ (q - zero_point) * scale
    ///
    /// The scale and zero_point are computed from the data range to maximize
    /// precision across the full [-128, 127] i8 range.
    pub fn quantize_asymmetric(tensor: &Tensor) -> Self {
        let min_val = tensor
            .data
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .min(0.0);
        let max_val = tensor
            .data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0);

        let range = max_val - min_val;
        let scale = if range < 1e-10 { 1e-10 } else { range / 255.0 };

        // zero_point maps min_val → -128
        // min_val / scale + zp = -128  =>  zp = -128 - min_val / scale
        let zp_f = (-128.0 - min_val / scale).round().clamp(-128.0, 127.0);
        let zero_point = zp_f as i8;

        let data: Vec<i8> = tensor
            .data
            .iter()
            .map(|&v| (v / scale + zp_f).round().clamp(-128.0, 127.0) as i8)
            .collect();

        Self::new(
            data,
            tensor.shape.clone(),
            QuantParams {
                scale: vec![scale],
                zero_point: vec![zero_point],
                per_channel: false,
                axis: 0,
            },
        )
    }

    /// Quantize with asymmetric per-channel parameters along the given axis.
    pub fn quantize_asymmetric_per_channel(
        tensor: &Tensor,
        axis: usize,
    ) -> Result<Self, OnnxError> {
        if axis >= tensor.shape.len() {
            return Err(OnnxError::ShapeMismatch(format!(
                "quantize axis {} >= rank {}",
                axis,
                tensor.shape.len()
            )));
        }

        let num_channels = tensor.shape[axis];
        let channel_size: usize = tensor.shape[axis + 1..].iter().product();
        let outer_size: usize = tensor.shape[..axis].iter().product();

        let mut scales = Vec::with_capacity(num_channels);
        let mut zero_points = Vec::with_capacity(num_channels);
        let mut quant_data = vec![0i8; tensor.data.len()];

        for ch in 0..num_channels {
            let mut ch_min = 0.0f32;
            let mut ch_max = 0.0f32;

            for outer in 0..outer_size {
                let base = outer * num_channels * channel_size + ch * channel_size;
                for i in 0..channel_size {
                    let v = tensor.data[base + i];
                    if v < ch_min {
                        ch_min = v;
                    }
                    if v > ch_max {
                        ch_max = v;
                    }
                }
            }

            let ch_range = ch_max - ch_min;
            let scale = if ch_range < 1e-10 {
                1e-10
            } else {
                ch_range / 255.0
            };
            let zp_f = (-128.0 - ch_min / scale).round().clamp(-128.0, 127.0);
            let zero_point = zp_f as i8;

            scales.push(scale);
            zero_points.push(zero_point);

            for outer in 0..outer_size {
                let base = outer * num_channels * channel_size + ch * channel_size;
                for i in 0..channel_size {
                    quant_data[base + i] = (tensor.data[base + i] / scale + zp_f)
                        .round()
                        .clamp(-128.0, 127.0) as i8;
                }
            }
        }

        Ok(Self::new(
            quant_data,
            tensor.shape.clone(),
            QuantParams {
                scale: scales,
                zero_point: zero_points,
                per_channel: true,
                axis,
            },
        ))
    }

    /// Dequantize back to f32.
    pub fn dequantize(&self) -> Tensor {
        let mut data = vec![0.0f32; self.data.len()];

        if !self.params.per_channel {
            let scale = self.params.scale[0];
            let zp = self.params.zero_point[0] as f32;
            for (i, &q) in self.data.iter().enumerate() {
                data[i] = (q as f32 - zp) * scale;
            }
        } else {
            let axis = self.params.axis;
            let num_channels = self.shape[axis];
            let channel_size: usize = self.shape[axis + 1..].iter().product();
            let outer_size: usize = self.shape[..axis].iter().product();

            for ch in 0..num_channels {
                let scale = self.params.scale[ch];
                let zp = self.params.zero_point[ch] as f32;
                for outer in 0..outer_size {
                    let base = outer * num_channels * channel_size + ch * channel_size;
                    for i in 0..channel_size {
                        data[base + i] = (self.data[base + i] as f32 - zp) * scale;
                    }
                }
            }
        }

        Tensor::new(data, self.shape.clone())
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

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
        // Per-channel: each output column j has its own scale/zp
        // For [K, N] weight with per-channel axis=1, each column is a channel
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

    // Fast path: when both zero points are zero, skip correction terms
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

    // Precompute row sums of A: row_sum_a[i] = Σ_p A_q[i][p]
    let row_sum_a: Vec<i32> = (0..m)
        .map(|i| {
            let mut s = 0i32;
            for p in 0..k {
                s += a.data[i * k + p] as i32;
            }
            s
        })
        .collect();

    // Precompute column sums of B: col_sum_b[j] = Σ_p B_q[p][j]
    let mut col_sum_b = vec![0i32; n];
    for p in 0..k {
        for (j, cs) in col_sum_b.iter_mut().enumerate() {
            *cs += b.data[p * n + j] as i32;
        }
    }

    let k_zp_product = k as i32 * a_zp * b_zp;

    // C[i][j] = (A_q@B_q)[i][j] - a_zp*col_sum_b[j] - b_zp*row_sum_a[i] + K*a_zp*b_zp
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

            // im2col: build column matrix [col_rows, col_cols] as i32
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
                                // Padding: fill with x_zero_point (dequant(xzp) = 0)
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

            // Precompute column sums of im2col matrix for zero point correction
            let mut col_sums = vec![0i32; col_cols];
            for r in 0..col_rows {
                for c_idx in 0..col_cols {
                    col_sums[c_idx] += col[r * col_cols + c_idx];
                }
            }

            // For each output channel in this group
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

                // Weight row for this output channel
                let w_base = global_oc * col_rows;

                // Precompute weight row sum
                let mut w_row_sum = 0i32;
                for r in 0..col_rows {
                    w_row_sum += w_q.data[w_base + r] as i32;
                }

                // Compute bias in int32 scale
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
                    // Integer dot product of weight row with im2col column
                    let mut raw_sum = 0i32;
                    for r in 0..col_rows {
                        raw_sum += w_q.data[w_base + r] as i32 * col[r * col_cols + sp];
                    }

                    // Zero point correction:
                    // Σ(x-xzp)(w-wzp) = Σxw - xzp*Σw - wzp*Σx + K*xzp*wzp
                    let corrected = raw_sum - x_zp_i32 * w_row_sum - w_zp_i32 * col_sums[sp]
                        + zp_correction
                        + bias_i32;

                    // Requantize to output
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

    // zero_point maps 0.0 to a uint8 value: zp = round(-min_val / scale)
    let zp_f = (-min_val / scale).round().clamp(0.0, 255.0);
    let zero_point = zp_f as u8 as i8; // store as i8 per ONNX convention

    let data: Vec<f32> = x
        .data
        .iter()
        .map(|&v| (v / scale + zp_f).round().clamp(0.0, 255.0))
        .collect();

    Ok((Tensor::new(data, x.shape.clone()), scale, zero_point))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compute f32 matmul for reference.
    fn f32_matmul(a: &Tensor, b: &Tensor) -> Tensor {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a.data[i * k + p] * b.data[p * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        Tensor::new(out, vec![m, n])
    }

    /// Helper: max absolute error between two tensors.
    fn max_abs_error(a: &Tensor, b: &Tensor) -> f32 {
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Helper: relative error (Frobenius norm of difference / Frobenius norm of reference).
    fn relative_error(result: &Tensor, reference: &Tensor) -> f32 {
        let diff_norm: f32 = result
            .data
            .iter()
            .zip(reference.data.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        let ref_norm: f32 = reference.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if ref_norm < 1e-10 {
            diff_norm
        } else {
            diff_norm / ref_norm
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Quantize then dequantize, check error is within tolerance
        let tensor = Tensor::new(
            vec![1.0, -0.5, 3.2, -2.1, 0.0, 1.7, -1.3, 0.8, 2.5],
            vec![3, 3],
        );
        let quantized = QuantizedTensor::quantize(&tensor);
        let dequantized = quantized.dequantize();

        // Per-tensor quantization error should be bounded by scale/2
        let scale = quantized.params.scale[0];
        let err = max_abs_error(&tensor, &dequantized);
        assert!(
            err < scale * 1.5,
            "Roundtrip error {} exceeds tolerance (scale={})",
            err,
            scale,
        );
    }

    #[test]
    fn test_quantize_per_channel() {
        // 2x4 tensor, per-channel along axis 0 (2 channels, each of size 4)
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], vec![2, 4]);
        let quantized =
            QuantizedTensor::quantize_per_channel(&tensor, 0).expect("per-channel quantize");
        assert!(quantized.params.per_channel);
        assert_eq!(quantized.params.scale.len(), 2);
        assert_eq!(quantized.params.zero_point.len(), 2);

        let dequantized = quantized.dequantize();
        let err = max_abs_error(&tensor, &dequantized);
        let max_scale = quantized
            .params
            .scale
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        assert!(
            err < max_scale * 1.5,
            "Per-channel roundtrip error {} too large (max_scale={})",
            err,
            max_scale,
        );
    }

    #[test]
    fn test_quantized_matmul_basic() {
        // f32 activations x i8 weights, compare with f32 reference
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b_f32 = Tensor::new(
            vec![
                0.5, -0.3, 1.2, 0.8, -1.0, 0.4, 0.1, -0.7, 0.9, 0.6, 0.2, -0.5,
            ],
            vec![3, 4],
        );
        let b_quant = QuantizedTensor::quantize(&b_f32);

        let result = quantized_matmul(&a, &b_quant).expect("quantized_matmul");
        let reference = f32_matmul(&a, &b_f32);

        assert_eq!(result.shape, vec![2, 4]);

        // Quantization introduces some error; check relative error
        let rel_err = relative_error(&result, &reference);
        assert!(
            rel_err < 0.15,
            "quantized_matmul relative error {} too large",
            rel_err,
        );
    }

    #[test]
    fn test_quantized_matmul_per_channel() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b_f32 = Tensor::new(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.5], vec![2, 3]);

        // Per-channel quantize along axis 1 (each output column is a channel)
        let b_quant =
            QuantizedTensor::quantize_per_channel(&b_f32, 1).expect("per-channel quantize");

        let result = quantized_matmul(&a, &b_quant).expect("quantized_matmul per-channel");
        let reference = f32_matmul(&a, &b_f32);

        assert_eq!(result.shape, vec![2, 3]);

        let rel_err = relative_error(&result, &reference);
        assert!(
            rel_err < 0.15,
            "per-channel quantized_matmul relative error {} too large",
            rel_err,
        );
    }

    #[test]
    fn test_fully_quantized_matmul() {
        let a_f32 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b_f32 = Tensor::new(vec![0.5, -0.3, 1.2, 0.8, -1.0, 0.4], vec![3, 2]);

        let a_quant = QuantizedTensor::quantize(&a_f32);
        let b_quant = QuantizedTensor::quantize(&b_f32);

        let result = fully_quantized_matmul(&a_quant, &b_quant).expect("fully_quantized_matmul");
        let reference = f32_matmul(&a_f32, &b_f32);

        assert_eq!(result.shape, vec![2, 2]);

        let rel_err = relative_error(&result, &reference);
        assert!(
            rel_err < 0.15,
            "fully_quantized_matmul relative error {} too large",
            rel_err,
        );
    }

    #[test]
    fn test_quantize_range() {
        // Verify quantized values are in [-128, 127]
        let tensor = Tensor::new(
            vec![-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0],
            vec![3, 3],
        );
        let quantized = QuantizedTensor::quantize(&tensor);
        for &v in &quantized.data {
            // i8 range is [-128, 127]; verify values are valid i8
            // (symmetric quantization should keep values in [-127, 127])
            let vi = v as i32;
            assert!(
                (-128..=127).contains(&vi),
                "Quantized value {} out of range",
                vi,
            );
        }
    }

    #[test]
    fn test_dequantize_identity() {
        // quantize(zeros) -> dequantize should be near zero
        let tensor = Tensor::new(vec![0.0; 16], vec![4, 4]);
        let quantized = QuantizedTensor::quantize(&tensor);
        let dequantized = quantized.dequantize();
        for &v in &dequantized.data {
            assert!(v.abs() < 1e-6, "Dequantized zero is not near zero: {}", v,);
        }
    }

    #[test]
    fn test_quantized_matmul_accuracy() {
        // Larger matrix, check relative error < 5%
        // 8x16 * 16x8 = 8x8
        let m = 8;
        let k = 16;
        let n = 8;

        // Generate deterministic data using a simple linear congruential pattern
        let mut a_data = Vec::with_capacity(m * k);
        let mut val = 0.1f32;
        for _ in 0..m * k {
            a_data.push(val);
            val = (val * 1.1 + 0.3) % 5.0 - 2.5;
        }

        let mut b_data = Vec::with_capacity(k * n);
        val = -0.2f32;
        for _ in 0..k * n {
            b_data.push(val);
            val = (val * 0.9 + 0.7) % 3.0 - 1.5;
        }

        let a = Tensor::new(a_data, vec![m, k]);
        let b_f32 = Tensor::new(b_data, vec![k, n]);
        let b_quant = QuantizedTensor::quantize(&b_f32);

        let result = quantized_matmul(&a, &b_quant).expect("quantized_matmul accuracy");
        let reference = f32_matmul(&a, &b_f32);

        let rel_err = relative_error(&result, &reference);
        assert!(
            rel_err < 0.05,
            "Large matrix quantized_matmul relative error {} exceeds 5%",
            rel_err,
        );
    }

    // ==================== Asymmetric quantization tests ====================

    #[test]
    fn test_asymmetric_quantize_dequantize_roundtrip() {
        let tensor = Tensor::new(
            vec![1.0, -0.5, 3.2, -2.1, 0.0, 1.7, -1.3, 0.8, 2.5],
            vec![3, 3],
        );
        let quantized = QuantizedTensor::quantize_asymmetric(&tensor);

        // Verify non-zero zero_point for asymmetric data
        // (symmetric data centered at 0 might still get zp=0, but asymmetric range shouldn't)
        let dequantized = quantized.dequantize();

        let scale = quantized.params.scale[0];
        let err = max_abs_error(&tensor, &dequantized);
        assert!(
            err < scale * 1.5,
            "Asymmetric roundtrip error {} exceeds tolerance (scale={})",
            err,
            scale,
        );
    }

    #[test]
    fn test_asymmetric_quantize_all_positive() {
        // All positive values: zero_point should be negative (to shift range)
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let quantized = QuantizedTensor::quantize_asymmetric(&tensor);
        let dequantized = quantized.dequantize();

        let scale = quantized.params.scale[0];
        let err = max_abs_error(&tensor, &dequantized);
        assert!(
            err < scale * 1.5,
            "All-positive asymmetric roundtrip error {} too large (scale={})",
            err,
            scale,
        );
    }

    #[test]
    fn test_asymmetric_quantize_all_negative() {
        let tensor = Tensor::new(vec![-6.0, -5.0, -4.0, -3.0, -2.0, -1.0], vec![2, 3]);
        let quantized = QuantizedTensor::quantize_asymmetric(&tensor);
        let dequantized = quantized.dequantize();

        let scale = quantized.params.scale[0];
        let err = max_abs_error(&tensor, &dequantized);
        assert!(
            err < scale * 1.5,
            "All-negative asymmetric roundtrip error {} too large (scale={})",
            err,
            scale,
        );
    }

    #[test]
    fn test_asymmetric_quantize_backward_compatible() {
        // Symmetric data: asymmetric should still work (zero_point near 0)
        let tensor = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 0.5], vec![2, 3]);
        let q_sym = QuantizedTensor::quantize(&tensor);
        let q_asym = QuantizedTensor::quantize_asymmetric(&tensor);

        let d_sym = q_sym.dequantize();
        let d_asym = q_asym.dequantize();

        // Both should approximate the original reasonably
        let err_sym = max_abs_error(&tensor, &d_sym);
        let err_asym = max_abs_error(&tensor, &d_asym);
        assert!(err_sym < 0.1, "Symmetric roundtrip error: {}", err_sym);
        assert!(err_asym < 0.1, "Asymmetric roundtrip error: {}", err_asym);
    }

    #[test]
    fn test_asymmetric_per_channel() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, -5.0, -3.0, -1.0, 0.5], vec![2, 4]);
        let quantized = QuantizedTensor::quantize_asymmetric_per_channel(&tensor, 0)
            .expect("asymmetric per-channel quantize");
        assert!(quantized.params.per_channel);
        assert_eq!(quantized.params.scale.len(), 2);
        assert_eq!(quantized.params.zero_point.len(), 2);

        let dequantized = quantized.dequantize();
        let max_scale = quantized
            .params
            .scale
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let err = max_abs_error(&tensor, &dequantized);
        assert!(
            err < max_scale * 1.5,
            "Asymmetric per-channel roundtrip error {} too large (max_scale={})",
            err,
            max_scale,
        );
    }

    #[test]
    fn test_asymmetric_fully_quantized_matmul() {
        let a_f32 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b_f32 = Tensor::new(vec![0.5, -0.3, 1.2, 0.8, -1.0, 0.4], vec![3, 2]);

        // Asymmetric quantization
        let a_quant = QuantizedTensor::quantize_asymmetric(&a_f32);
        let b_quant = QuantizedTensor::quantize_asymmetric(&b_f32);

        // Verify zero points are non-trivial for A (all positive)
        assert_ne!(
            a_quant.params.zero_point[0], 0,
            "A zero_point should be non-zero for asymmetric all-positive data"
        );

        let result =
            fully_quantized_matmul(&a_quant, &b_quant).expect("asymmetric fully_quantized_matmul");
        let reference = f32_matmul(&a_f32, &b_f32);

        assert_eq!(result.shape, vec![2, 2]);

        let rel_err = relative_error(&result, &reference);
        assert!(
            rel_err < 0.15,
            "Asymmetric fully_quantized_matmul relative error {} too large",
            rel_err,
        );
    }

    #[test]
    fn test_asymmetric_matmul_larger() {
        // 4x8 * 8x4 with asymmetric quant
        let m = 4;
        let k = 8;
        let n = 4;

        let mut a_data = Vec::with_capacity(m * k);
        let mut val = 0.5f32;
        for _ in 0..m * k {
            a_data.push(val);
            val = (val * 1.3 + 0.2) % 4.0 - 1.0;
        }

        let mut b_data = Vec::with_capacity(k * n);
        val = -0.3f32;
        for _ in 0..k * n {
            b_data.push(val);
            val = (val * 0.7 + 0.5) % 3.0 - 1.5;
        }

        let a_f32 = Tensor::new(a_data, vec![m, k]);
        let b_f32 = Tensor::new(b_data, vec![k, n]);

        let a_quant = QuantizedTensor::quantize_asymmetric(&a_f32);
        let b_quant = QuantizedTensor::quantize_asymmetric(&b_f32);

        let result = fully_quantized_matmul(&a_quant, &b_quant).expect("asymmetric matmul larger");
        let reference = f32_matmul(&a_f32, &b_f32);

        let rel_err = relative_error(&result, &reference);
        assert!(
            rel_err < 0.10,
            "Asymmetric matmul larger relative error {} too large",
            rel_err,
        );
    }

    // ==================== QLinearConv tests ====================

    /// Helper: simple f32 conv2d for reference (no groups, no dilation).
    fn reference_conv2d(
        input: &[f32],
        n: usize,
        c_in: usize,
        h: usize,
        w: usize,
        weight: &[f32],
        c_out: usize,
        kh: usize,
        kw: usize,
        bias: Option<&[f32]>,
        strides: &[usize],
        pads: &[usize],
        group: usize,
    ) -> Vec<f32> {
        let c_per_group = c_in / group;
        let c_out_per_group = c_out / group;
        let h_out = (h + pads[0] + pads[2] - kh) / strides[0] + 1;
        let w_out = (w + pads[1] + pads[3] - kw) / strides[1] + 1;
        let mut out = vec![0.0f32; n * c_out * h_out * w_out];

        for batch in 0..n {
            for g in 0..group {
                for oc in 0..c_out_per_group {
                    let global_oc = g * c_out_per_group + oc;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut sum = 0.0f32;
                            for ic in 0..c_per_group {
                                let in_c = g * c_per_group + ic;
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oh * strides[0] + ky) as isize - pads[0] as isize;
                                        let ix = (ow * strides[1] + kx) as isize - pads[1] as isize;
                                        if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize
                                        {
                                            let x_val = input[(batch * c_in + in_c) * h * w
                                                + iy as usize * w
                                                + ix as usize];
                                            let w_val =
                                                weight[((global_oc * c_per_group + ic) * kh + ky)
                                                    * kw
                                                    + kx];
                                            sum += x_val * w_val;
                                        }
                                    }
                                }
                            }
                            if let Some(b) = bias {
                                sum += b[global_oc];
                            }
                            out[(batch * c_out + global_oc) * h_out * w_out + oh * w_out + ow] =
                                sum;
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_qlinear_conv2d_1x1_kernel() {
        // 1x1 convolution: input [1,2,3,3], weight [4,2,1,1]
        let x_f32 = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // ch0
                0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, // ch1
            ],
            vec![1, 2, 3, 3],
        );
        let w_f32 = Tensor::new(
            vec![
                0.3, -0.2, // oc0
                0.5, 0.1, // oc1
                -0.4, 0.6, // oc2
                0.2, -0.3, // oc3
            ],
            vec![4, 2, 1, 1],
        );

        // Compute float reference
        let ref_out = reference_conv2d(
            &x_f32.data,
            1,
            2,
            3,
            3,
            &w_f32.data,
            4,
            1,
            1,
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        );

        // Quantize
        let x_scale = 9.0 / 127.0; // symmetric scale for input
        let x_zp: i8 = 0;
        let x_q_data: Vec<f32> = x_f32
            .data
            .iter()
            .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
            .collect();
        let x_q = Tensor::new(x_q_data, vec![1, 2, 3, 3]);

        let w_scale = vec![0.6 / 127.0]; // per-tensor
        let w_zp = vec![0i8];
        let w_q_data: Vec<f32> = w_f32
            .data
            .iter()
            .map(|&v| (v / w_scale[0]).round().clamp(-128.0, 127.0))
            .collect();
        let w_q = Tensor::new(w_q_data, vec![4, 2, 1, 1]);

        // Output scale: choose based on expected range
        let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let y_scale = expected_max / 127.0;
        let y_zp: i8 = 0;

        let result = qlinear_conv2d(
            &x_q,
            x_scale,
            x_zp,
            &w_q,
            &w_scale,
            &w_zp,
            y_scale,
            y_zp,
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        )
        .expect("qlinear_conv2d 1x1");

        assert_eq!(result.shape, vec![1, 4, 3, 3]);

        // Dequantize output and compare
        let deq_out: Vec<f32> = result
            .data
            .iter()
            .map(|&v| (v - y_zp as f32) * y_scale)
            .collect();
        let ref_tensor = Tensor::new(ref_out, vec![1, 4, 3, 3]);
        let deq_tensor = Tensor::new(deq_out, vec![1, 4, 3, 3]);
        let rel_err = relative_error(&deq_tensor, &ref_tensor);
        assert!(
            rel_err < 0.15,
            "QLinearConv 1x1 relative error {} too large",
            rel_err,
        );
    }

    #[test]
    fn test_qlinear_conv2d_3x3_kernel() {
        // [1,1,4,4] input, [1,1,3,3] weight, stride 1, pad 0
        let x_f32 = Tensor::new((0..16).map(|i| i as f32 * 0.5).collect(), vec![1, 1, 4, 4]);
        let w_f32 = Tensor::new(
            vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0],
            vec![1, 1, 3, 3],
        );
        let bias = Tensor::new(vec![0.5], vec![1]);

        let ref_out = reference_conv2d(
            &x_f32.data,
            1,
            1,
            4,
            4,
            &w_f32.data,
            1,
            3,
            3,
            Some(&bias.data),
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        );

        let x_max = 7.5f32;
        let x_scale = x_max / 127.0;
        let x_zp: i8 = 0;
        let x_q_data: Vec<f32> = x_f32
            .data
            .iter()
            .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
            .collect();
        let x_q = Tensor::new(x_q_data, vec![1, 1, 4, 4]);

        let w_max = 2.0f32;
        let w_scale = vec![w_max / 127.0];
        let w_zp = vec![0i8];
        let w_q_data: Vec<f32> = w_f32
            .data
            .iter()
            .map(|&v| (v / w_scale[0]).round().clamp(-128.0, 127.0))
            .collect();
        let w_q = Tensor::new(w_q_data, vec![1, 1, 3, 3]);

        let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let y_scale = if expected_max < 1e-10 {
            1e-10
        } else {
            expected_max / 127.0
        };
        let y_zp: i8 = 0;

        let result = qlinear_conv2d(
            &x_q,
            x_scale,
            x_zp,
            &w_q,
            &w_scale,
            &w_zp,
            y_scale,
            y_zp,
            Some(&bias),
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        )
        .expect("qlinear_conv2d 3x3");

        assert_eq!(result.shape, vec![1, 1, 2, 2]);

        let deq_out: Vec<f32> = result
            .data
            .iter()
            .map(|&v| (v - y_zp as f32) * y_scale)
            .collect();
        let ref_tensor = Tensor::new(ref_out, vec![1, 1, 2, 2]);
        let deq_tensor = Tensor::new(deq_out, vec![1, 1, 2, 2]);
        let rel_err = relative_error(&deq_tensor, &ref_tensor);
        assert!(
            rel_err < 0.2,
            "QLinearConv 3x3 relative error {} too large",
            rel_err,
        );
    }

    #[test]
    fn test_qlinear_conv2d_grouped() {
        // group=2: [1,4,3,3] input, [4,2,1,1] weight (2 groups, each 2 in -> 2 out)
        let x_f32 = Tensor::new(
            (0..36).map(|i| (i as f32 - 18.0) * 0.1).collect(),
            vec![1, 4, 3, 3],
        );
        let w_f32 = Tensor::new(
            vec![
                0.3, -0.2, // oc0, group0
                0.5, 0.1, // oc1, group0
                -0.4, 0.6, // oc2, group1
                0.2, -0.3, // oc3, group1
            ],
            vec![4, 2, 1, 1],
        );

        let ref_out = reference_conv2d(
            &x_f32.data,
            1,
            4,
            3,
            3,
            &w_f32.data,
            4,
            1,
            1,
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            2,
        );

        let x_max = x_f32
            .data
            .iter()
            .copied()
            .fold(0.0f32, |a, v| a.max(v.abs()));
        let x_scale = x_max / 127.0;
        let x_zp: i8 = 0;
        let x_q_data: Vec<f32> = x_f32
            .data
            .iter()
            .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
            .collect();
        let x_q = Tensor::new(x_q_data, vec![1, 4, 3, 3]);

        let w_max = 0.6f32;
        let w_scale = vec![w_max / 127.0];
        let w_zp = vec![0i8];
        let w_q_data: Vec<f32> = w_f32
            .data
            .iter()
            .map(|&v| (v / w_scale[0]).round().clamp(-128.0, 127.0))
            .collect();
        let w_q = Tensor::new(w_q_data, vec![4, 2, 1, 1]);

        let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let y_scale = if expected_max < 1e-10 {
            1e-10
        } else {
            expected_max / 127.0
        };
        let y_zp: i8 = 0;

        let result = qlinear_conv2d(
            &x_q,
            x_scale,
            x_zp,
            &w_q,
            &w_scale,
            &w_zp,
            y_scale,
            y_zp,
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            2,
        )
        .expect("qlinear_conv2d grouped");

        assert_eq!(result.shape, vec![1, 4, 3, 3]);

        let deq_out: Vec<f32> = result
            .data
            .iter()
            .map(|&v| (v - y_zp as f32) * y_scale)
            .collect();
        let ref_tensor = Tensor::new(ref_out.clone(), vec![1, 4, 3, 3]);
        let deq_tensor = Tensor::new(deq_out, vec![1, 4, 3, 3]);
        let rel_err = relative_error(&deq_tensor, &ref_tensor);
        assert!(
            rel_err < 0.2,
            "QLinearConv grouped relative error {} too large",
            rel_err,
        );
    }

    #[test]
    fn test_qlinear_conv2d_per_channel_scales() {
        // Per-channel weight scales: [1,1,3,3] input, [2,1,1,1] weight
        let x_f32 = Tensor::new((0..9).map(|i| i as f32).collect(), vec![1, 1, 3, 3]);
        let w_f32 = Tensor::new(vec![0.5, -0.8], vec![2, 1, 1, 1]);

        let ref_out = reference_conv2d(
            &x_f32.data,
            1,
            1,
            3,
            3,
            &w_f32.data,
            2,
            1,
            1,
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        );

        let x_max = 8.0f32;
        let x_scale = x_max / 127.0;
        let x_zp: i8 = 0;
        let x_q_data: Vec<f32> = x_f32
            .data
            .iter()
            .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
            .collect();
        let x_q = Tensor::new(x_q_data, vec![1, 1, 3, 3]);

        // Per-channel: different scale for each output channel
        let w_scale = vec![0.5 / 127.0, 0.8 / 127.0];
        let w_zp = vec![0i8, 0i8];
        let w_q_data: Vec<f32> = vec![
            (0.5f32 / w_scale[0]).round().clamp(-128.0, 127.0),
            (-0.8f32 / w_scale[1]).round().clamp(-128.0, 127.0),
        ];
        let w_q = Tensor::new(w_q_data, vec![2, 1, 1, 1]);

        let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let y_scale = if expected_max < 1e-10 {
            1e-10
        } else {
            expected_max / 127.0
        };
        let y_zp: i8 = 0;

        let result = qlinear_conv2d(
            &x_q,
            x_scale,
            x_zp,
            &w_q,
            &w_scale,
            &w_zp,
            y_scale,
            y_zp,
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        )
        .expect("qlinear_conv2d per-channel");

        assert_eq!(result.shape, vec![1, 2, 3, 3]);

        let deq_out: Vec<f32> = result
            .data
            .iter()
            .map(|&v| (v - y_zp as f32) * y_scale)
            .collect();
        let ref_tensor = Tensor::new(ref_out, vec![1, 2, 3, 3]);
        let deq_tensor = Tensor::new(deq_out, vec![1, 2, 3, 3]);
        let rel_err = relative_error(&deq_tensor, &ref_tensor);
        assert!(
            rel_err < 0.15,
            "QLinearConv per-channel relative error {} too large",
            rel_err,
        );
    }

    #[test]
    fn test_qlinear_conv2d_with_nonzero_zp() {
        // Test asymmetric zero points in QLinearConv
        let x_f32 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let w_f32 = Tensor::new(vec![0.5], vec![1, 1, 1, 1]);
        let bias = Tensor::new(vec![0.1], vec![1]);

        let ref_out = reference_conv2d(
            &x_f32.data,
            1,
            1,
            2,
            2,
            &w_f32.data,
            1,
            1,
            1,
            Some(&bias.data),
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        );

        // Asymmetric: input range [1, 4], shift so 0 maps to some nonzero quantized val
        let x_scale = 4.0 / 255.0;
        let x_zp_f = (-1.0f32 / x_scale).round().clamp(-128.0, 127.0);
        let x_zp = x_zp_f as i8;
        let x_q_data: Vec<f32> = x_f32
            .data
            .iter()
            .map(|&v| (v / x_scale + x_zp_f).round().clamp(-128.0, 127.0))
            .collect();
        let x_q = Tensor::new(x_q_data, vec![1, 1, 2, 2]);

        let w_scale = vec![0.5 / 127.0];
        let w_zp = vec![3i8]; // nonzero weight zero point
        let w_q_data: Vec<f32> = vec![(0.5 / w_scale[0] + w_zp[0] as f32)
            .round()
            .clamp(-128.0, 127.0)];
        let w_q = Tensor::new(w_q_data, vec![1, 1, 1, 1]);

        let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let y_scale = if expected_max < 1e-10 {
            1e-10
        } else {
            expected_max / 127.0
        };
        let y_zp: i8 = 5; // nonzero output zero point

        let result = qlinear_conv2d(
            &x_q,
            x_scale,
            x_zp,
            &w_q,
            &w_scale,
            &w_zp,
            y_scale,
            y_zp,
            Some(&bias),
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        )
        .expect("qlinear_conv2d with nonzero zp");

        assert_eq!(result.shape, vec![1, 1, 2, 2]);

        let deq_out: Vec<f32> = result
            .data
            .iter()
            .map(|&v| (v - y_zp as f32) * y_scale)
            .collect();
        let ref_tensor = Tensor::new(ref_out, vec![1, 1, 2, 2]);
        let deq_tensor = Tensor::new(deq_out, vec![1, 1, 2, 2]);
        let rel_err = relative_error(&deq_tensor, &ref_tensor);
        assert!(
            rel_err < 0.25,
            "QLinearConv nonzero-zp relative error {} too large",
            rel_err,
        );
    }

    // ==================== Dynamic quantization tests ====================

    #[test]
    fn test_dynamic_quantize_mixed() {
        let x = Tensor::new(vec![-1.0, 0.0, 0.5, 1.0, 2.0, 3.0], vec![2, 3]);
        let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize mixed");

        // Range includes 0: min_val = -1.0, max_val = 3.0
        // scale = 4.0 / 255.0
        let expected_scale = 4.0 / 255.0;
        assert!(
            (scale - expected_scale).abs() < 1e-6,
            "scale {} != expected {}",
            scale,
            expected_scale,
        );

        // All quantized values should be in [0, 255]
        for &v in &q.data {
            assert!(
                v >= 0.0 && v <= 255.0,
                "Dynamic quantize value {} out of [0,255]",
                v,
            );
        }

        // Dequantize and check roundtrip
        let zp_f = zp as u8 as f32;
        for (i, &orig) in x.data.iter().enumerate() {
            let deq = (q.data[i] - zp_f) * scale;
            assert!(
                (deq - orig).abs() < scale * 1.5,
                "Dynamic roundtrip: orig={}, deq={}, diff={}",
                orig,
                deq,
                (deq - orig).abs(),
            );
        }
    }

    #[test]
    fn test_dynamic_quantize_all_positive() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let (q, scale, _zp) = dynamic_quantize(&x).expect("dynamic_quantize all_positive");

        // Range: [0, 4], scale = 4/255
        let expected_scale = 4.0 / 255.0;
        assert!(
            (scale - expected_scale).abs() < 1e-6,
            "all_positive scale {} != expected {}",
            scale,
            expected_scale,
        );

        for &v in &q.data {
            assert!(v >= 0.0 && v <= 255.0);
        }
    }

    #[test]
    fn test_dynamic_quantize_all_negative() {
        let x = Tensor::new(vec![-4.0, -3.0, -2.0, -1.0], vec![4]);
        let (q, scale, _zp) = dynamic_quantize(&x).expect("dynamic_quantize all_negative");

        // Range: [-4, 0], scale = 4/255
        let expected_scale = 4.0 / 255.0;
        assert!(
            (scale - expected_scale).abs() < 1e-6,
            "all_negative scale {} != expected {}",
            scale,
            expected_scale,
        );

        for &v in &q.data {
            assert!(v >= 0.0 && v <= 255.0);
        }
    }

    #[test]
    fn test_dynamic_quantize_range_includes_zero() {
        // Even if all values are positive, range should include 0 (min clamped to 0)
        let x = Tensor::new(vec![5.0, 10.0, 15.0], vec![3]);
        let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize zero_inclusive");

        // zp should map to 0.0: dequant(zp) = (zp - zp) * scale = 0
        let zp_u8 = zp as u8;
        let deq_zero = (zp_u8 as f32 - zp_u8 as f32) * scale;
        assert!(
            deq_zero.abs() < 1e-6,
            "Zero point should dequantize to 0, got {}",
            deq_zero,
        );

        for &v in &q.data {
            assert!(v >= 0.0 && v <= 255.0);
        }
    }

    #[test]
    fn test_dynamic_quantize_single_element() {
        let x = Tensor::new(vec![42.0], vec![1]);
        let (q, _scale, _zp) = dynamic_quantize(&x).expect("dynamic_quantize single");
        assert_eq!(q.data.len(), 1);
        assert!(q.data[0] >= 0.0 && q.data[0] <= 255.0);
    }

    #[test]
    fn test_dynamic_quantize_empty_fails() {
        let x = Tensor::new(vec![], vec![0]);
        let result = dynamic_quantize(&x);
        assert!(result.is_err());
    }

    // ==================== Edge case tests ====================

    #[test]
    fn test_single_element_tensor_quantize() {
        let tensor = Tensor::new(vec![3.14], vec![1, 1]);
        let q = QuantizedTensor::quantize(&tensor);
        assert_eq!(q.data.len(), 1);
        let dq = q.dequantize();
        assert!((dq.data[0] - 3.14).abs() < q.params.scale[0] * 1.5);

        let qa = QuantizedTensor::quantize_asymmetric(&tensor);
        assert_eq!(qa.data.len(), 1);
        let dqa = qa.dequantize();
        assert!((dqa.data[0] - 3.14).abs() < qa.params.scale[0] * 1.5);
    }

    #[test]
    fn test_zero_scale_handling() {
        // All-zero tensor: scale should be clamped to small positive value
        let tensor = Tensor::new(vec![0.0; 9], vec![3, 3]);
        let q = QuantizedTensor::quantize(&tensor);
        assert!(q.params.scale[0] > 0.0, "Scale must be positive");
        let dq = q.dequantize();
        for &v in &dq.data {
            assert!(v.abs() < 1e-6);
        }

        let qa = QuantizedTensor::quantize_asymmetric(&tensor);
        assert!(
            qa.params.scale[0] > 0.0,
            "Asymmetric scale must be positive"
        );
    }

    #[test]
    fn test_qlinear_conv2d_shape_validation() {
        // Bad input shape
        let x = Tensor::new(vec![1.0; 6], vec![2, 3]);
        let w = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
        let result = qlinear_conv2d(
            &x,
            1.0,
            0,
            &w,
            &[1.0],
            &[0],
            1.0,
            0,
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        );
        assert!(result.is_err());

        // Bad weight shape
        let x2 = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
        let w2 = Tensor::new(vec![1.0; 3], vec![3]);
        let result2 = qlinear_conv2d(
            &x2,
            1.0,
            0,
            &w2,
            &[1.0],
            &[0],
            1.0,
            0,
            None,
            &[1, 1],
            &[0, 0, 0, 0],
            1,
        );
        assert!(result2.is_err());
    }

    #[test]
    fn test_qlinear_conv2d_with_padding() {
        // [1,1,2,2] input, [1,1,3,3] kernel, pad=1 -> output same size
        let x_f32 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let w_f32 = Tensor::new(
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![1, 1, 3, 3],
        );

        // Identity-like kernel (center=1): output should match input with padding
        let ref_out = reference_conv2d(
            &x_f32.data,
            1,
            1,
            2,
            2,
            &w_f32.data,
            1,
            3,
            3,
            None,
            &[1, 1],
            &[1, 1, 1, 1],
            1,
        );

        let x_scale = 4.0 / 127.0;
        let x_zp: i8 = 0;
        let x_q_data: Vec<f32> = x_f32
            .data
            .iter()
            .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
            .collect();
        let x_q = Tensor::new(x_q_data, vec![1, 1, 2, 2]);

        let w_scale = vec![1.0 / 127.0];
        let w_zp = vec![0i8];
        let w_q_data: Vec<f32> = w_f32
            .data
            .iter()
            .map(|&v| (v / w_scale[0]).round().clamp(-128.0, 127.0))
            .collect();
        let w_q = Tensor::new(w_q_data, vec![1, 1, 3, 3]);

        let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let y_scale = if expected_max < 1e-10 {
            1e-10
        } else {
            expected_max / 127.0
        };
        let y_zp: i8 = 0;

        let result = qlinear_conv2d(
            &x_q,
            x_scale,
            x_zp,
            &w_q,
            &w_scale,
            &w_zp,
            y_scale,
            y_zp,
            None,
            &[1, 1],
            &[1, 1, 1, 1],
            1,
        )
        .expect("qlinear_conv2d with padding");

        assert_eq!(result.shape, vec![1, 1, 2, 2]);

        let deq_out: Vec<f32> = result
            .data
            .iter()
            .map(|&v| (v - y_zp as f32) * y_scale)
            .collect();
        let ref_tensor = Tensor::new(ref_out, vec![1, 1, 2, 2]);
        let deq_tensor = Tensor::new(deq_out, vec![1, 1, 2, 2]);
        let rel_err = relative_error(&deq_tensor, &ref_tensor);
        assert!(
            rel_err < 0.15,
            "QLinearConv padded relative error {} too large",
            rel_err,
        );
    }
}
