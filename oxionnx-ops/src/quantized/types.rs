//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxionnx_core::{OnnxError, Tensor};

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
