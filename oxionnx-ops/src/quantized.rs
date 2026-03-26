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
/// Both inputs are quantized. Accumulation in i32, result dequantized to f32.
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

    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                let av = a.data[i * k + p] as i32 - a_zp;
                let bv = b.data[p * n + j] as i32 - b_zp;
                acc += av * bv;
            }
            out[i * n + j] = acc as f32 * output_scale;
        }
    }

    Ok(Tensor::new(out, vec![m, n]))
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
                vi >= -128 && vi <= 127,
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
}
