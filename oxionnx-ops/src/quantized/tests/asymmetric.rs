//! Asymmetric quantization tests.

use oxionnx_core::Tensor;

use crate::quantized::{fully_quantized_matmul, QuantizedTensor};

use super::common::{f32_matmul, max_abs_error, relative_error};

#[test]
fn test_asymmetric_quantize_dequantize_roundtrip() {
    let tensor = Tensor::new(
        vec![1.0, -0.5, 3.2, -2.1, 0.0, 1.7, -1.3, 0.8, 2.5],
        vec![3, 3],
    );
    let quantized = QuantizedTensor::quantize_asymmetric(&tensor);
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
    let tensor = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 0.5], vec![2, 3]);
    let q_sym = QuantizedTensor::quantize(&tensor);
    let q_asym = QuantizedTensor::quantize_asymmetric(&tensor);
    let d_sym = q_sym.dequantize();
    let d_asym = q_asym.dequantize();
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
    let a_quant = QuantizedTensor::quantize_asymmetric(&a_f32);
    let b_quant = QuantizedTensor::quantize_asymmetric(&b_f32);
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
