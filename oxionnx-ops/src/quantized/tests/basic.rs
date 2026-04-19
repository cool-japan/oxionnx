//! Basic quantize/dequantize and matmul tests.

use oxionnx_core::Tensor;

use crate::quantized::{fully_quantized_matmul, quantized_matmul, QuantizedTensor};

use super::common::{f32_matmul, max_abs_error, relative_error};

#[test]
fn test_quantize_dequantize_roundtrip() {
    let tensor = Tensor::new(
        vec![1.0, -0.5, 3.2, -2.1, 0.0, 1.7, -1.3, 0.8, 2.5],
        vec![3, 3],
    );
    let quantized = QuantizedTensor::quantize(&tensor);
    let dequantized = quantized.dequantize();
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
    let b_quant = QuantizedTensor::quantize_per_channel(&b_f32, 1).expect("per-channel quantize");
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
    let tensor = Tensor::new(
        vec![-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0],
        vec![3, 3],
    );
    let quantized = QuantizedTensor::quantize(&tensor);
    for &v in &quantized.data {
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
    let tensor = Tensor::new(vec![0.0; 16], vec![4, 4]);
    let quantized = QuantizedTensor::quantize(&tensor);
    let dequantized = quantized.dequantize();
    for &v in &dequantized.data {
        assert!(v.abs() < 1e-6, "Dequantized zero is not near zero: {}", v,);
    }
}

#[test]
fn test_quantized_matmul_accuracy() {
    let m = 8;
    let k = 16;
    let n = 8;
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

#[test]
fn test_single_element_tensor_quantize() {
    let val = std::f32::consts::PI;
    let tensor = Tensor::new(vec![val], vec![1, 1]);
    let q = QuantizedTensor::quantize(&tensor);
    assert_eq!(q.data.len(), 1);
    let dq = q.dequantize();
    assert!((dq.data[0] - val).abs() < q.params.scale[0] * 1.5);
    let qa = QuantizedTensor::quantize_asymmetric(&tensor);
    assert_eq!(qa.data.len(), 1);
    let dqa = qa.dequantize();
    assert!((dqa.data[0] - val).abs() < qa.params.scale[0] * 1.5);
}

#[test]
fn test_zero_scale_handling() {
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
