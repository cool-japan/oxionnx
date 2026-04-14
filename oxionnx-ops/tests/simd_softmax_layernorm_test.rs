//! Integration tests for SIMD-accelerated softmax and LayerNorm.

#![cfg(feature = "simd")]

use oxionnx_ops::simd_ops::{
    simd_layer_norm, simd_layer_norm_strided, simd_softmax_inplace, simd_softmax_strided,
};

const TOL: f32 = 1e-4;

fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{msg}: expected {b}, got {a}, diff={}",
        (a - b).abs()
    );
}

// ── Reference implementations ───────────────────────────────────────────────

fn reference_softmax(data: &[f32]) -> Vec<f32> {
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = data.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

fn reference_layer_norm(data: &[f32], scale: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let n = data.len() as f32;
    let mean: f32 = data.iter().sum::<f32>() / n;
    let var: f32 = data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = (var + eps).sqrt().recip();
    data.iter()
        .enumerate()
        .map(|(i, &v)| {
            let normalized = (v - mean) * inv_std;
            let scaled = normalized * scale[i % scale.len()];
            if let Some(b) = bias {
                scaled + b[i % b.len()]
            } else {
                scaled
            }
        })
        .collect()
}

// ── Softmax tests ───────────────────────────────────────────────────────────

#[test]
fn test_softmax_small_matches_scalar() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 1.0];
    let expected = reference_softmax(&input);
    let mut data = input;
    simd_softmax_inplace(&mut data);
    for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert_close(got, exp, 1e-5, &format!("softmax small [{i}]"));
    }
}

#[test]
fn test_softmax_large_matches_scalar() {
    let n = 10_000;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin() * 5.0).collect();
    let expected = reference_softmax(&input);
    let mut data = input;
    simd_softmax_inplace(&mut data);
    for i in 0..n {
        assert_close(data[i], expected[i], 1e-5, &format!("softmax large [{i}]"));
    }
}

#[test]
fn test_softmax_strided_3x5() {
    // [3, 5] tensor, softmax on dim=-1
    let rows: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![-1.0, 0.0, 1.0, 0.0, -1.0],
        vec![10.0, 10.0, 10.0, 10.0, 10.0],
    ];
    let mut flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    simd_softmax_strided(&mut flat, 5);

    for (r, row) in rows.iter().enumerate() {
        let expected = reference_softmax(row);
        for c in 0..5 {
            assert_close(
                flat[r * 5 + c],
                expected[c],
                1e-5,
                &format!("softmax strided [{r},{c}]"),
            );
        }
    }
}

#[test]
fn test_softmax_numerical_stability() {
    // Very large values should not produce NaN
    let mut data = vec![1000.0, 1001.0, 999.0, 1000.5];
    simd_softmax_inplace(&mut data);
    for (i, &v) in data.iter().enumerate() {
        assert!(!v.is_nan(), "softmax NaN at [{i}]");
        assert!(!v.is_infinite(), "softmax Inf at [{i}]");
        assert!(v >= 0.0, "softmax negative at [{i}]");
    }
    let sum: f32 = data.iter().sum();
    assert_close(sum, 1.0, TOL, "softmax large values sum");
}

#[test]
fn test_softmax_uniform_input() {
    // Uniform input → uniform output (1/N)
    let n = 16;
    let mut data = vec![5.0f32; n];
    simd_softmax_inplace(&mut data);
    let expected = 1.0 / n as f32;
    for (i, &v) in data.iter().enumerate() {
        assert_close(v, expected, 1e-5, &format!("softmax uniform [{i}]"));
    }
}

// ── LayerNorm tests ─────────────────────────────────────────────────────────

#[test]
fn test_layer_norm_known() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let scale = vec![1.0, 1.0, 1.0, 1.0];
    let bias = vec![0.0, 0.0, 0.0, 0.0];
    let expected = reference_layer_norm(&input, &scale, Some(&bias), 1e-5);
    let mut data = input;
    simd_layer_norm(&mut data, &scale, Some(&bias), 1e-5);
    for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert_close(got, exp, TOL, &format!("layer_norm known [{i}]"));
    }
}

#[test]
fn test_layer_norm_zero_mean_unit_var() {
    // scale=1, bias=0 → output should have ~zero mean and ~unit variance
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 3.0 + 2.0).collect();
    let scale = vec![1.0f32; n];
    let mut data = input;
    simd_layer_norm(&mut data, &scale, None, 1e-5);
    let mean: f32 = data.iter().sum::<f32>() / n as f32;
    let var: f32 = data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    assert_close(mean, 0.0, TOL, "layer_norm zero mean");
    assert_close(var, 1.0, 1e-3, "layer_norm unit var");
}

#[test]
fn test_layer_norm_strided_2x4() {
    // [2, 4] tensor, layer norm over last dim
    let row0 = vec![1.0f32, 2.0, 3.0, 4.0];
    let row1 = vec![10.0f32, 20.0, 30.0, 40.0];
    let scale = vec![1.0, 1.0, 1.0, 1.0];
    let bias = vec![0.0, 0.0, 0.0, 0.0];
    let exp0 = reference_layer_norm(&row0, &scale, Some(&bias), 1e-5);
    let exp1 = reference_layer_norm(&row1, &scale, Some(&bias), 1e-5);

    let mut flat: Vec<f32> = row0.iter().chain(row1.iter()).copied().collect();
    simd_layer_norm_strided(&mut flat, 4, &scale, Some(&bias), 1e-5);

    for i in 0..4 {
        assert_close(flat[i], exp0[i], TOL, &format!("ln strided [0,{i}]"));
        assert_close(flat[4 + i], exp1[i], TOL, &format!("ln strided [1,{i}]"));
    }
}

#[test]
fn test_layer_norm_with_scale_bias() {
    let input = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let scale = vec![2.0, 0.5, 1.0, 3.0, 2.0, 0.5, 1.0, 3.0];
    let bias = vec![1.0, -1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.5];
    let expected = reference_layer_norm(&input, &scale, Some(&bias), 1e-5);
    let mut data = input;
    simd_layer_norm(&mut data, &scale, Some(&bias), 1e-5);
    for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert_close(got, exp, TOL, &format!("ln scale+bias [{i}]"));
    }
}
