#![cfg(feature = "simd")]
#![allow(clippy::needless_range_loop, clippy::useless_vec)]

use oxionnx_ops::simd_ops::*;

const TOL: f32 = 1e-2;

fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{msg}: expected {b}, got {a}, diff={}",
        (a - b).abs()
    );
}

// ── Existing tests (moved from simd_ops.rs) ─────────────────────────────────

#[test]
fn test_simd_add() {
    let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..33).map(|i| i as f32 * 0.3 + 1.0).collect();
    let mut out = vec![0.0f32; 33];
    simd_add(&a, &b, &mut out);
    for i in 0..33 {
        assert_close(out[i], a[i] + b[i], 1e-6, "simd_add");
    }
}

#[test]
fn test_simd_mul() {
    let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..33).map(|i| i as f32 * 0.3 + 1.0).collect();
    let mut out = vec![0.0f32; 33];
    simd_mul(&a, &b, &mut out);
    for i in 0..33 {
        assert_close(out[i], a[i] * b[i], 1e-6, "simd_mul");
    }
}

#[test]
fn test_simd_relu() {
    let mut data: Vec<f32> = vec![-3.0, -1.5, -0.1, 0.0, 0.1, 1.5, 3.0, -100.0, 100.0];
    simd_relu(&mut data);
    assert_close(data[0], 0.0, 1e-6, "relu neg");
    assert_close(data[1], 0.0, 1e-6, "relu neg");
    assert_close(data[2], 0.0, 1e-6, "relu neg");
    assert_close(data[3], 0.0, 1e-6, "relu zero");
    assert_close(data[4], 0.1, 1e-6, "relu pos");
    assert_close(data[5], 1.5, 1e-6, "relu pos");
    assert_close(data[6], 3.0, 1e-6, "relu pos");
    assert_close(data[7], 0.0, 1e-6, "relu large neg");
    assert_close(data[8], 100.0, 1e-6, "relu large pos");
}

#[test]
fn test_simd_sigmoid() {
    let mut data: Vec<f32> = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    simd_sigmoid(&mut data);
    for &v in &data {
        assert!((0.0..=1.0).contains(&v), "sigmoid out of range: {v}");
    }
    assert_close(data[3], 0.5, TOL, "sigmoid(0)");
    assert_close(data[0] + data[6], 1.0, TOL, "sigmoid symmetry");
    assert_close(data[1] + data[5], 1.0, TOL, "sigmoid symmetry");
    assert_close(data[2] + data[4], 1.0, TOL, "sigmoid symmetry");
}

#[test]
fn test_simd_tanh() {
    let mut data: Vec<f32> = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    simd_tanh(&mut data);
    for &v in &data {
        assert!(
            (-1.0 - TOL..=1.0 + TOL).contains(&v),
            "tanh out of range: {v}"
        );
    }
    assert_close(data[3], 0.0, TOL, "tanh(0)");
    assert_close(data[0] + data[6], 0.0, TOL, "tanh odd");
    assert_close(data[1] + data[5], 0.0, TOL, "tanh odd");
}

#[test]
fn test_simd_gelu() {
    let mut data: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    simd_gelu(&mut data);
    assert_close(data[2], 0.0, TOL, "gelu(0)");
    assert_close(data[3], 0.8412, 0.05, "gelu(1)");
    assert_close(data[1], -0.1588, 0.05, "gelu(-1)");
}

#[test]
fn test_simd_silu() {
    let mut data: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    simd_silu(&mut data);
    assert_close(data[2], 0.0, TOL, "silu(0)");
    assert_close(data[3], 0.7311, 0.05, "silu(1)");
}

#[test]
fn test_simd_exp() {
    let mut data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0, -2.0];
    simd_exp(&mut data);
    assert_close(data[0], 1.0, TOL, "exp(0)");
    assert_close(data[1], std::f32::consts::E, 0.1, "exp(1)");
    assert_close(data[2], 1.0 / std::f32::consts::E, 0.05, "exp(-1)");
}

#[test]
fn test_simd_small_arrays() {
    let a = vec![1.0f32, 2.0];
    let b = vec![3.0f32, 4.0];
    let mut out = vec![0.0f32; 2];
    simd_add(&a, &b, &mut out);
    assert_close(out[0], 4.0, 1e-6, "small add 0");
    assert_close(out[1], 6.0, 1e-6, "small add 1");
    let mut small = vec![0.5f32];
    simd_relu(&mut small);
    assert_close(small[0], 0.5, 1e-6, "small relu");
    let mut small = vec![-0.5f32];
    simd_relu(&mut small);
    assert_close(small[0], 0.0, 1e-6, "small relu neg");
}

#[test]
fn test_simd_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let mut out: Vec<f32> = vec![];
    simd_add(&a, &b, &mut out);
    simd_mul(&a, &b, &mut out);
    assert!(out.is_empty());
    let mut empty: Vec<f32> = vec![];
    simd_relu(&mut empty);
    simd_sigmoid(&mut empty);
    simd_tanh(&mut empty);
    simd_gelu(&mut empty);
    simd_silu(&mut empty);
    simd_exp(&mut empty);
    assert!(empty.is_empty());
}

// ── Reduction / dot product tests ───────────────────────────────────────────

#[test]
fn test_reduce_sum_known() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_close(simd_reduce_sum(&data), 15.0, 1e-6, "reduce_sum known");
}

#[test]
fn test_reduce_sum_large() {
    let n = 1_000_000;
    let data = vec![1.0f32; n];
    let result = simd_reduce_sum(&data);
    assert_close(result, n as f32, 1.0, "reduce_sum 1M");
}

#[test]
fn test_reduce_sum_matches_scalar() {
    let data: Vec<f32> = (0..1_000_000).map(|i| (i as f32) * 0.001).collect();
    let simd_result = simd_reduce_sum(&data);
    let scalar_result: f64 = data.iter().map(|&v| v as f64).sum();
    let rel_err = ((simd_result as f64 - scalar_result) / scalar_result).abs();
    assert!(rel_err < 1e-4, "reduce_sum large rel_err={rel_err}");
}

#[test]
fn test_reduce_sum_empty() {
    assert_close(simd_reduce_sum(&[]), 0.0, 1e-6, "reduce_sum empty");
}

#[test]
fn test_reduce_sum_single() {
    assert_close(simd_reduce_sum(&[42.0]), 42.0, 1e-6, "reduce_sum single");
}

#[test]
fn test_reduce_sum_sub_lane() {
    let data = vec![1.5, 2.5, 3.5];
    assert_close(simd_reduce_sum(&data), 7.5, 1e-6, "reduce_sum sub-lane");
}

#[test]
fn test_reduce_max_positive() {
    let data = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0, 8.0];
    assert_close(simd_reduce_max(&data), 9.0, 1e-6, "reduce_max pos");
}

#[test]
fn test_reduce_max_negative() {
    let data = vec![-10.0, -5.0, -3.0, -9.0, -2.0, -7.0];
    assert_close(simd_reduce_max(&data), -2.0, 1e-6, "reduce_max neg");
}

#[test]
fn test_reduce_max_mixed() {
    let data = vec![-100.0, 0.0, 50.0, -50.0, 100.0, 25.0, -25.0];
    assert_close(simd_reduce_max(&data), 100.0, 1e-6, "reduce_max mixed");
}

#[test]
fn test_reduce_max_with_neg_inf() {
    let data = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 2.0];
    assert_close(simd_reduce_max(&data), 2.0, 1e-6, "reduce_max neg_inf");
}

#[test]
fn test_reduce_max_empty() {
    assert_eq!(simd_reduce_max(&[]), f32::NEG_INFINITY);
}

#[test]
fn test_reduce_max_single() {
    assert_close(simd_reduce_max(&[-42.0]), -42.0, 1e-6, "reduce_max single");
}

#[test]
fn test_reduce_min_positive() {
    let data = vec![10.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0, 8.0];
    assert_close(simd_reduce_min(&data), 2.0, 1e-6, "reduce_min pos");
}

#[test]
fn test_reduce_min_negative() {
    let data = vec![-1.0, -5.0, -3.0, -9.0, -2.0, -7.0];
    assert_close(simd_reduce_min(&data), -9.0, 1e-6, "reduce_min neg");
}

#[test]
fn test_reduce_min_mixed() {
    let data = vec![-100.0, 0.0, 50.0, -50.0, 100.0, 25.0, -200.0];
    assert_close(simd_reduce_min(&data), -200.0, 1e-6, "reduce_min mixed");
}

#[test]
fn test_reduce_min_with_inf() {
    let data = vec![f32::INFINITY, 1.0, f32::INFINITY, -1.0];
    assert_close(simd_reduce_min(&data), -1.0, 1e-6, "reduce_min inf");
}

#[test]
fn test_reduce_min_empty() {
    assert_eq!(simd_reduce_min(&[]), f32::INFINITY);
}

#[test]
fn test_reduce_min_single() {
    assert_close(simd_reduce_min(&[42.0]), 42.0, 1e-6, "reduce_min single");
}

#[test]
fn test_dot_product_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert_close(simd_dot_product(&a, &b), 0.0, 1e-6, "dot orthogonal");
}

#[test]
fn test_dot_product_parallel() {
    let a = vec![2.0, 3.0, 4.0];
    assert_close(simd_dot_product(&a, &a), 29.0, 1e-6, "dot parallel");
}

#[test]
fn test_dot_product_known() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    assert_close(simd_dot_product(&a, &b), 165.0, 1e-4, "dot known");
}

#[test]
fn test_dot_product_large_matches_naive() {
    let n = 100_000;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.01).collect();
    let simd_result = simd_dot_product(&a, &b);
    let naive: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let rel_err = ((simd_result as f64 - naive) / naive).abs();
    assert!(rel_err < 1e-3, "dot large rel_err={rel_err}");
}

#[test]
fn test_dot_product_empty() {
    assert_close(simd_dot_product(&[], &[]), 0.0, 1e-6, "dot empty");
}

#[test]
fn test_dot_product_single() {
    assert_close(simd_dot_product(&[3.0], &[4.0]), 12.0, 1e-6, "dot single");
}

#[test]
fn test_dot_product_different_lengths() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![10.0, 20.0, 30.0];
    assert_close(simd_dot_product(&a, &b), 140.0, 1e-4, "dot diff len");
}

#[test]
fn test_reduce_mean_known() {
    let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    assert_close(simd_reduce_mean(&data), 6.0, 1e-6, "reduce_mean known");
}

#[test]
fn test_reduce_mean_single() {
    assert_close(simd_reduce_mean(&[7.0]), 7.0, 1e-6, "reduce_mean single");
}

#[test]
fn test_reduce_mean_empty() {
    assert_close(simd_reduce_mean(&[]), 0.0, 1e-6, "reduce_mean empty");
}

#[test]
fn test_reduce_mean_negative() {
    let data = vec![-4.0, -2.0, 0.0, 2.0, 4.0];
    assert_close(simd_reduce_mean(&data), 0.0, 1e-6, "reduce_mean neg");
}

#[test]
fn test_reduce_sum_tail_elements() {
    let data: Vec<f32> = (1..=11).map(|i| i as f32).collect();
    assert_close(simd_reduce_sum(&data), 66.0, 1e-4, "reduce_sum tail");
}

#[test]
fn test_reduce_max_sub_lane() {
    let data = vec![3.0, 1.0, 2.0];
    assert_close(simd_reduce_max(&data), 3.0, 1e-6, "reduce_max sub-lane");
}

#[test]
fn test_reduce_min_sub_lane() {
    let data = vec![3.0, 1.0, 2.0];
    assert_close(simd_reduce_min(&data), 1.0, 1e-6, "reduce_min sub-lane");
}

// ── New tests: simd_sub ─────────────────────────────────────────────────────

#[test]
fn test_simd_sub_basic() {
    let a = vec![5.0, 10.0, 3.0, 7.0, 1.0];
    let b = vec![2.0, 3.0, 1.0, 4.0, 0.5];
    let mut out = vec![0.0f32; 5];
    simd_sub(&a, &b, &mut out);
    for i in 0..5 {
        assert_close(out[i], a[i] - b[i], 1e-6, "simd_sub basic");
    }
}

#[test]
fn test_simd_sub_large() {
    let n = 10_003; // odd size to test tail handling
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.7).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
    let mut out = vec![0.0f32; n];
    simd_sub(&a, &b, &mut out);
    for i in 0..n {
        assert_close(out[i], a[i] - b[i], 1e-5, "simd_sub large");
    }
}

#[test]
fn test_simd_sub_self_is_zero() {
    let a: Vec<f32> = (0..33).map(|i| i as f32 * 1.5 - 20.0).collect();
    let mut out = vec![0.0f32; 33];
    simd_sub(&a, &a, &mut out);
    for (i, &v) in out.iter().enumerate() {
        assert_close(v, 0.0, 1e-6, &format!("sub self [{i}]"));
    }
}

// ── New tests: simd_div ─────────────────────────────────────────────────────

#[test]
fn test_simd_div_basic() {
    let a = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let b = vec![2.0, 4.0, 5.0, 8.0, 10.0];
    let mut out = vec![0.0f32; 5];
    simd_div(&a, &b, &mut out);
    for i in 0..5 {
        assert_close(out[i], a[i] / b[i], 1e-6, "simd_div basic");
    }
}

#[test]
fn test_simd_div_by_one() {
    let a: Vec<f32> = (1..=20).map(|i| i as f32).collect();
    let b = vec![1.0f32; 20];
    let mut out = vec![0.0f32; 20];
    simd_div(&a, &b, &mut out);
    for i in 0..20 {
        assert_close(out[i], a[i], 1e-6, "simd_div by one");
    }
}

#[test]
fn test_simd_div_large() {
    let n = 10_003;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 3.0).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    let mut out = vec![0.0f32; n];
    simd_div(&a, &b, &mut out);
    for i in 0..n {
        assert_close(out[i], 3.0, 1e-4, "simd_div large");
    }
}

// ── New tests: simd_neg ─────────────────────────────────────────────────────

#[test]
fn test_simd_neg_basic() {
    let mut data = vec![1.0, -2.0, 3.0, -4.0, 0.0, 5.5, -7.5];
    let expected = vec![-1.0, 2.0, -3.0, 4.0, 0.0, -5.5, 7.5];
    simd_neg(&mut data);
    for i in 0..data.len() {
        assert_close(data[i], expected[i], 1e-6, "simd_neg basic");
    }
}

#[test]
fn test_simd_neg_large() {
    let n = 10_003;
    let original: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 2500.0).collect();
    let mut data = original.clone();
    simd_neg(&mut data);
    for i in 0..n {
        assert_close(data[i], -original[i], 1e-6, "simd_neg large");
    }
}

// ── New tests: simd_abs ─────────────────────────────────────────────────────

#[test]
fn test_simd_abs_basic() {
    let mut data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 0.0];
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
    simd_abs(&mut data);
    for i in 0..data.len() {
        assert_close(data[i], expected[i], 1e-6, "simd_abs basic");
    }
}

#[test]
fn test_simd_abs_mixed() {
    let n = 10_003;
    let mut data: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 })
        .collect();
    simd_abs(&mut data);
    for (i, &v) in data.iter().enumerate() {
        assert!(
            v >= 0.0,
            "abs result should be non-negative at [{i}]: got {v}"
        );
        assert_close(v, i as f32, 1e-6, "simd_abs mixed");
    }
}

// ── New tests: simd_sqrt ────────────────────────────────────────────────────

#[test]
fn test_simd_sqrt_basic() {
    let mut data = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0];
    let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
    simd_sqrt(&mut data);
    for i in 0..data.len() {
        assert_close(data[i], expected[i], 1e-5, "simd_sqrt basic");
    }
}

#[test]
fn test_simd_sqrt_large() {
    let n = 10_003;
    let squares: Vec<f32> = (0..n).map(|i| (i as f32) * (i as f32)).collect();
    let mut data = squares;
    simd_sqrt(&mut data);
    for i in 0..n {
        assert_close(data[i], i as f32, 1e-3, "simd_sqrt large");
    }
}

// ── New tests: simd_log ─────────────────────────────────────────────────────

#[test]
fn test_simd_log_basic() {
    let mut data = vec![
        1.0,
        std::f32::consts::E,
        std::f32::consts::E * std::f32::consts::E,
    ];
    simd_log(&mut data);
    assert_close(data[0], 0.0, 0.01, "ln(1)");
    assert_close(data[1], 1.0, 0.02, "ln(e)");
    assert_close(data[2], 2.0, 0.05, "ln(e^2)");
}

#[test]
fn test_simd_log_known_values() {
    let mut data = vec![2.0, 10.0, 0.5, 100.0];
    simd_log(&mut data);
    assert_close(data[0], 2.0f32.ln(), 0.02, "ln(2)");
    assert_close(data[1], 10.0f32.ln(), 0.05, "ln(10)");
    assert_close(data[2], 0.5f32.ln(), 0.02, "ln(0.5)");
    assert_close(data[3], 100.0f32.ln(), 0.05, "ln(100)");
}

#[test]
fn test_simd_log_large() {
    let n = 10_003;
    let original: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let mut data = original.clone();
    simd_log(&mut data);
    for i in 0..n {
        let expected = original[i].ln();
        let tol = 0.02 + expected.abs() * 0.01; // relative tolerance
        assert!(
            (data[i] - expected).abs() < tol,
            "simd_log large [{i}]: got {}, expected {}, diff={}",
            data[i],
            expected,
            (data[i] - expected).abs()
        );
    }
}
