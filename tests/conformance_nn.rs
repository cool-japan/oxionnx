//! Conformance tests 16–21, 36–37, 40: Neural network activation and normalization operators.

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_close, assert_shape, run_op};

// ═══════════════════════════════════════════════════════════════════════════════
// 16–21: NN conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 16. conformance_relu — clamp negatives
#[test]
fn conformance_relu() {
    let x = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0, -0.5], vec![2, 3]);
    let out = run_op(
        OpKind::Relu,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "relu");
    assert_close(&t.data, &[0.0, 0.0, 0.0, 1.0, 3.0, 0.0], 1e-5, "relu");
}

/// 17. conformance_sigmoid — sigmoid(0)=0.5
#[test]
fn conformance_sigmoid() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 100.0, -100.0], vec![5]);
    let out = run_op(
        OpKind::Sigmoid,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    // sigmoid(0)=0.5, sigmoid(1)=1/(1+e^-1)≈0.7311, sigmoid(-1)≈0.2689
    // sigmoid(100)≈1.0, sigmoid(-100)≈0.0
    let expected = [
        0.5,
        1.0 / (1.0 + (-1.0_f32).exp()),
        1.0 / (1.0 + 1.0_f32.exp()),
        1.0,
        0.0,
    ];
    assert_close(&t.data, &expected, 1e-4, "sigmoid");
}

/// 18. conformance_tanh — tanh(0)=0
#[test]
fn conformance_tanh() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]);
    let out = run_op(
        OpKind::Tanh,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    let expected = [
        0.0_f32.tanh(),
        1.0_f32.tanh(),
        (-1.0_f32).tanh(),
        2.0_f32.tanh(),
    ];
    assert_close(&t.data, &expected, 1e-5, "tanh");
}

/// 19. conformance_softmax — sum=1, non-negative
#[test]
fn conformance_softmax() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    // Two rows to test batched behavior
    // row0: [1,2,3], row1: [0,0,0]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], vec![2, 3]);
    let out = run_op(
        OpKind::Softmax,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "softmax");

    // Row 0: softmax([1,2,3])
    let denom0 = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp();
    let expected_row0 = [
        1.0_f32.exp() / denom0,
        2.0_f32.exp() / denom0,
        3.0_f32.exp() / denom0,
    ];
    assert_close(&t.data[0..3], &expected_row0, 1e-5, "softmax_row0");

    // Row 1: softmax([0,0,0]) = [1/3, 1/3, 1/3]
    let third = 1.0 / 3.0;
    assert_close(&t.data[3..6], &[third, third, third], 1e-5, "softmax_row1");

    // All values non-negative and each row sums to 1
    let sum0: f32 = t.data[0..3].iter().sum();
    let sum1: f32 = t.data[3..6].iter().sum();
    assert!((sum0 - 1.0).abs() < 1e-5, "softmax row0 sum = {}", sum0);
    assert!((sum1 - 1.0).abs() < 1e-5, "softmax row1 sum = {}", sum1);
}

/// 20. conformance_layer_norm — normalize + scale + bias
#[test]
fn conformance_layer_norm() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    // x = [[1,2,3,4]] shape [1,4]
    // mean=2.5, var=1.25, inv_std = 1/sqrt(1.25+1e-5)
    // scale=[2,2,2,2], bias=[1,1,1,1]
    // output = (x - mean) * inv_std * scale + bias
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let scale = Tensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
    let bias = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);

    let out = run_op(
        OpKind::LayerNorm,
        vec!["x", "scale", "bias"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![("scale", scale), ("bias", bias)],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 4], "layer_norm");

    let mean = 2.5_f32;
    let var = 1.25_f32;
    let inv_std = (var + 1e-5_f32).sqrt().recip();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| (v - mean) * inv_std * 2.0 + 1.0)
        .collect();
    assert_close(&t.data, &expected, 1e-4, "layer_norm");
}

/// 21. conformance_batch_norm — (x-mean)/sqrt(var+eps)*gamma+beta
#[test]
fn conformance_batch_norm() {
    let mut attrs = Attributes::default();
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    // x = [[[[1,2],[3,4]]]] shape [1,1,2,2]
    // scale=[2], bias=[1], mean=[2.5], var=[1.25]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let scale = Tensor::new(vec![2.0], vec![1]);
    let bias = Tensor::new(vec![1.0], vec![1]);
    let bn_mean = Tensor::new(vec![2.5], vec![1]);
    let bn_var = Tensor::new(vec![1.25], vec![1]);

    let out = run_op(
        OpKind::BatchNorm,
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![
            ("scale", scale),
            ("bias", bias),
            ("mean", bn_mean),
            ("var", bn_var),
        ],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 1, 2, 2], "batch_norm");

    let m = 2.5_f32;
    let v = 1.25_f32;
    let inv_std = (v + 1e-5_f32).sqrt().recip();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&x| (x - m) * inv_std * 2.0 + 1.0)
        .collect();
    assert_close(&t.data, &expected, 1e-4, "batch_norm");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 36–37: Numerical stability conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 36. conformance_softmax_large_input — softmax with values > 100
#[test]
fn conformance_softmax_large_input() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);

    // Large values that would overflow naive exp()
    let x = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]);
    let out = run_op(
        OpKind::Softmax,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 3], "softmax_large");

    // All values should be finite
    for (i, &v) in t.data.iter().enumerate() {
        assert!(v.is_finite(), "softmax_large[{}] = {} not finite", i, v);
        assert!(v >= 0.0, "softmax_large[{}] = {} negative", i, v);
    }

    // Sum should be 1
    let sum: f32 = t.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax_large sum = {}", sum);

    // Largest input should dominate: output[2] should be close to 1.0
    assert!(
        t.data[2] > 0.99,
        "softmax_large: max input should dominate, got {}",
        t.data[2]
    );
}

/// 37. conformance_layernorm_zero_var — near-zero variance (epsilon test)
#[test]
fn conformance_layernorm_zero_var() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);
    attrs.floats.insert("epsilon".to_string(), 1e-5);

    // All same values => variance = 0
    let x = Tensor::new(vec![7.0, 7.0, 7.0, 7.0], vec![1, 4]);
    let scale = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let bias = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]);

    let out = run_op(
        OpKind::LayerNorm,
        vec!["x", "scale", "bias"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![("scale", scale), ("bias", bias)],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 4], "layernorm_zero_var");

    // (7 - 7) / sqrt(0 + eps) * 1 + 0 = 0 for all
    for (i, &v) in t.data.iter().enumerate() {
        assert!(v.is_finite(), "layernorm_zero_var[{}] not finite: {}", i, v);
        assert!(
            v.abs() < 1e-2,
            "layernorm_zero_var[{}] should be near zero: {}",
            i,
            v
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 40: GELU conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 40. conformance_gelu — GELU activation
#[test]
fn conformance_gelu() {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    // GELU(0) = 0, GELU(large) ≈ x
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 3.0], vec![4]);
    let out = run_op(
        OpKind::Gelu,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();

    // GELU(0) = 0
    assert!(
        (t.data[0]).abs() < 1e-5,
        "gelu(0) should be 0, got {}",
        t.data[0]
    );
    // GELU(1) ≈ 0.8413
    assert!(
        (t.data[1] - 0.8413).abs() < 0.01,
        "gelu(1) ≈ 0.8413, got {}",
        t.data[1]
    );
    // GELU(-1) ≈ -0.1587
    assert!(
        (t.data[2] - (-0.1587)).abs() < 0.01,
        "gelu(-1) ≈ -0.1587, got {}",
        t.data[2]
    );
    // GELU(3) ≈ 2.9960
    assert!(
        (t.data[3] - 2.9960).abs() < 0.01,
        "gelu(3) ≈ 2.9960, got {}",
        t.data[3]
    );
}
