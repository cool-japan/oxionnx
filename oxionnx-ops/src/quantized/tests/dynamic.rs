//! Dynamic quantization tests.

use oxionnx_core::Tensor;

use crate::quantized::dynamic_quantize;

#[test]
fn test_dynamic_quantize_mixed() {
    let x = Tensor::new(vec![-1.0, 0.0, 0.5, 1.0, 2.0, 3.0], vec![2, 3]);
    let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize mixed");
    let expected_scale = 4.0 / 255.0;
    assert!(
        (scale - expected_scale).abs() < 1e-6,
        "scale {} != expected {}",
        scale,
        expected_scale,
    );
    for &v in &q.data {
        assert!(
            (0.0_f32..=255.0_f32).contains(&v),
            "Dynamic quantize value {} out of [0,255]",
            v,
        );
    }
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
    let expected_scale = 4.0 / 255.0;
    assert!(
        (scale - expected_scale).abs() < 1e-6,
        "all_positive scale {} != expected {}",
        scale,
        expected_scale,
    );
    for &v in &q.data {
        assert!((0.0_f32..=255.0_f32).contains(&v));
    }
}

#[test]
fn test_dynamic_quantize_all_negative() {
    let x = Tensor::new(vec![-4.0, -3.0, -2.0, -1.0], vec![4]);
    let (q, scale, _zp) = dynamic_quantize(&x).expect("dynamic_quantize all_negative");
    let expected_scale = 4.0 / 255.0;
    assert!(
        (scale - expected_scale).abs() < 1e-6,
        "all_negative scale {} != expected {}",
        scale,
        expected_scale,
    );
    for &v in &q.data {
        assert!((0.0_f32..=255.0_f32).contains(&v));
    }
}

#[test]
fn test_dynamic_quantize_range_includes_zero() {
    let x = Tensor::new(vec![5.0, 10.0, 15.0], vec![3]);
    let (q, scale, zp) = dynamic_quantize(&x).expect("dynamic_quantize zero_inclusive");
    let zp_u8 = zp as u8;
    let deq_zero = (zp_u8 as f32 - zp_u8 as f32) * scale;
    assert!(
        deq_zero.abs() < 1e-6,
        "Zero point should dequantize to 0, got {}",
        deq_zero,
    );
    for &v in &q.data {
        assert!((0.0_f32..=255.0_f32).contains(&v));
    }
}

#[test]
fn test_dynamic_quantize_single_element() {
    let x = Tensor::new(vec![42.0], vec![1]);
    let (q, _scale, _zp) = dynamic_quantize(&x).expect("dynamic_quantize single");
    assert_eq!(q.data.len(), 1);
    assert!((0.0_f32..=255.0_f32).contains(&q.data[0]));
}

#[test]
fn test_dynamic_quantize_empty_fails() {
    let x = Tensor::new(vec![], vec![0]);
    let result = dynamic_quantize(&x);
    assert!(result.is_err());
}
