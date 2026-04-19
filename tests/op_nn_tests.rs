//! Neural network operator integration tests: Softmax, LayerNorm, BatchNorm,
//! activation functions (Relu, Sigmoid, Tanh, Gelu, LeakyRelu, Hardmax, Shrink).

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_tensor_approx, run_single_op};

// ═══════════════════════════════════════════════════════════════════════════════
// Softmax
// ═══════════════════════════════════════════════════════════════════════════════

// 6. test_softmax_axis1 - Softmax([1,4]) along axis 1, verify sum=1
#[test]
fn test_softmax_axis1() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let outputs = run_single_op(
        OpKind::Softmax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4]);

    // Sum should be 1
    let sum: f32 = out.data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum should be 1.0, got {}",
        sum
    );

    // Values should be monotonically increasing
    for i in 0..3 {
        assert!(
            out.data[i] < out.data[i + 1],
            "softmax should be monotonic: {} >= {}",
            out.data[i],
            out.data[i + 1]
        );
    }

    // Check specific values: softmax([1,2,3,4])
    let expected_denom = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp() + 4.0_f32.exp();
    let expected = [
        1.0_f32.exp() / expected_denom,
        2.0_f32.exp() / expected_denom,
        3.0_f32.exp() / expected_denom,
        4.0_f32.exp() / expected_denom,
    ];
    assert_tensor_approx(out, &expected, 1e-5);
}

// 21. test_softmax_single_element - Softmax on [1,1]
#[test]
fn test_softmax_single_element() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    let x = Tensor::new(vec![42.0], vec![1, 1]);
    let outputs = run_single_op(
        OpKind::Softmax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1]);
    // Softmax of single element = 1.0
    assert_tensor_approx(out, &[1.0], 1e-5);
}

// 25. test_softmax_large_values - Softmax with large values (numerical stability)
#[test]
fn test_softmax_large_values() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    // Large values that would overflow naive exp() computation
    let x = Tensor::new(vec![1000.0, 1001.0, 1002.0, 1003.0], vec![1, 4]);
    let outputs = run_single_op(
        OpKind::Softmax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();

    // Should still sum to 1.0 (numerically stable implementation subtracts max)
    let sum: f32 = out.data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax of large values should sum to 1.0, got {}",
        sum
    );

    // No NaN or Inf
    for (i, &v) in out.data.iter().enumerate() {
        assert!(v.is_finite(), "softmax output[{}] = {} is not finite", i, v);
        assert!(v > 0.0, "softmax output[{}] = {} should be positive", i, v);
    }

    // Values should be monotonically increasing
    for i in 0..3 {
        assert!(out.data[i] < out.data[i + 1]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LayerNorm / BatchNorm
// ═══════════════════════════════════════════════════════════════════════════════

// 7. test_layer_norm - LayerNorm with scale+bias
#[test]
fn test_layer_norm() {
    // x = [[1, 2, 3, 4]] shape [1,4]
    // mean = 2.5, var = 1.25
    // normalized = [-1.3416, -0.4472, 0.4472, 1.3416] (approx)
    // scale = [2, 2, 2, 2], bias = [1, 1, 1, 1]
    // output = normalized * scale + bias
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let scale = Tensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
    let bias = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);

    let outputs = run_single_op(
        OpKind::LayerNorm,
        vec![("x", x)],
        vec![("scale", scale), ("bias", bias)],
        vec!["x"],
        vec!["x", "scale", "bias"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4]);

    // Compute expected: mean=2.5, var=1.25, inv_std = 1/sqrt(1.25+1e-5) ~ 0.89442
    let mean = 2.5_f32;
    let var = 1.25_f32;
    let inv_std = (var + 1e-5_f32).sqrt().recip();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| (v - mean) * inv_std * 2.0 + 1.0)
        .collect();
    assert_tensor_approx(out, &expected, 1e-4);
}

// 26. test_layer_norm_epsilon - LayerNorm with near-zero variance
#[test]
fn test_layer_norm_epsilon() {
    // All same values => variance = 0, relying on epsilon for stability
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    let x = Tensor::new(vec![5.0, 5.0, 5.0, 5.0], vec![1, 4]);
    let scale = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let bias = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]);

    let outputs = run_single_op(
        OpKind::LayerNorm,
        vec![("x", x)],
        vec![("scale", scale), ("bias", bias)],
        vec!["x"],
        vec!["x", "scale", "bias"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4]);

    // (5 - 5) / sqrt(0 + 1e-5) * 1 + 0 = 0 for all elements
    for (i, &v) in out.data.iter().enumerate() {
        assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
        assert!(v.abs() < 1e-2, "output[{}] = {} should be near zero", i, v);
    }
}

// 8. test_batch_norm_inference
#[test]
fn test_batch_norm_inference() {
    // x = [[[[1, 2], [3, 4]]]] shape [1,1,2,2]
    // scale=[2], bias=[1], mean=[2.5], var=[1.25], eps=1e-5
    // BN: (x - mean) / sqrt(var + eps) * scale + bias
    let mut attrs = Attributes::default();
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let scale = Tensor::new(vec![2.0], vec![1]);
    let bias = Tensor::new(vec![1.0], vec![1]);
    let bn_mean = Tensor::new(vec![2.5], vec![1]);
    let bn_var = Tensor::new(vec![1.25], vec![1]);

    let outputs = run_single_op(
        OpKind::BatchNorm,
        vec![("x", x)],
        vec![
            ("scale", scale),
            ("bias", bias),
            ("mean", bn_mean),
            ("var", bn_var),
        ],
        vec!["x"],
        vec!["x", "scale", "bias", "mean", "var"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);

    let mean_val = 2.5_f32;
    let var_val = 1.25_f32;
    let inv_std = (var_val + 1e-5_f32).sqrt().recip();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| (v - mean_val) * inv_std * 2.0 + 1.0)
        .collect();
    assert_tensor_approx(out, &expected, 1e-4);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Activation functions
// ═══════════════════════════════════════════════════════════════════════════════

// 20. test_relu_empty - ReLU on empty tensor (0 elements)
#[test]
fn test_relu_empty() {
    let x = Tensor::new(vec![], vec![0]);
    let outputs = run_single_op(
        OpKind::Relu,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert!(out.data.is_empty());
}

// test_relu_mixed
#[test]
fn test_relu_mixed() {
    let x = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Relu,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[0.0, 0.0, 0.0, 1.0, 3.0], 1e-5);
}

// test_sigmoid
#[test]
fn test_sigmoid() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 100.0, -100.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Sigmoid,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    // sigmoid(0) = 0.5, sigmoid(1) ~ 0.7310, sigmoid(-1) ~ 0.2689
    // sigmoid(100) ~ 1.0, sigmoid(-100) ~ 0.0
    assert!((out.data[0] - 0.5).abs() < 1e-5);
    assert!((out.data[1] - 0.7310586).abs() < 1e-4);
    assert!((out.data[2] - 0.2689414).abs() < 1e-4);
    assert!((out.data[3] - 1.0).abs() < 1e-5);
    assert!(out.data[4].abs() < 1e-5);
}

// test_tanh
#[test]
fn test_tanh() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Tanh,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert!((out.data[0] - 0.0).abs() < 1e-5);
    assert!((out.data[1] - 1.0_f32.tanh()).abs() < 1e-5);
    assert!((out.data[2] - (-1.0_f32).tanh()).abs() < 1e-5);
}

// test_gelu
#[test]
fn test_gelu() {
    // GELU(0) = 0, GELU(x) ~ x for large x, GELU(x) ~ 0 for large negative x
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 3.0, -3.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Gelu,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    // GELU(0) = 0
    assert!(out.data[0].abs() < 1e-5, "gelu(0) = {}", out.data[0]);
    // GELU(1) ~ 0.8412
    assert!(
        (out.data[1] - 0.8412).abs() < 0.01,
        "gelu(1) = {}",
        out.data[1]
    );
    // GELU(-1) ~ -0.1588
    assert!(
        (out.data[2] - (-0.1588)).abs() < 0.01,
        "gelu(-1) = {}",
        out.data[2]
    );
    // GELU(3) ~ 2.9960
    assert!(
        (out.data[3] - 3.0).abs() < 0.01,
        "gelu(3) = {}",
        out.data[3]
    );
    // GELU(-3) ~ -0.0040
    assert!(out.data[4].abs() < 0.01, "gelu(-3) = {}", out.data[4]);
}

// test_hardmax
#[test]
fn test_hardmax() {
    // input [1, 3, 2] shape [3], max at index 1 → [0, 1, 0]
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 0);

    let x = Tensor::new(vec![1.0, 3.0, 2.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Hardmax,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![3]);
    assert_tensor_approx(out, &[0.0, 1.0, 0.0], 1e-5);
}

// test_shrink
#[test]
fn test_shrink() {
    // input [-2, 0, 2] shape [3], lambd=0.5, bias=0.0
    // -2 < -0.5 → -2 + 0.0 = -2; 0 in [-0.5,0.5] → 0; 2 > 0.5 → 2 - 0.0 = 2
    let mut attrs = Attributes::default();
    attrs.floats.insert("lambd".to_string(), 0.5);
    attrs.floats.insert("bias".to_string(), 0.0);

    let x = Tensor::new(vec![-2.0, 0.0, 2.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Shrink,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[-2.0, 0.0, 2.0], 1e-5);
}

// test_leaky_relu
#[test]
fn test_leaky_relu() {
    let mut attrs = Attributes::default();
    attrs.floats.insert("alpha".to_string(), 0.1);

    let x = Tensor::new(vec![-10.0, -1.0, 0.0, 1.0, 10.0], vec![5]);
    let outputs = run_single_op(
        OpKind::LeakyRelu,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[-1.0, -0.1, 0.0, 1.0, 10.0], 1e-5);
}
