//! Tests for all conv fusion passes.

use crate::graph::OpKind;
use crate::optimizer::test_utils::make_node;
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::{fold_batch_norm_inference, fuse_conv_add_relu, fuse_conv_batchnorm};
use super::{fuse_conv_clip_to_conv_relu6, fuse_conv_relu};

#[test]
fn test_fuse_conv_batchnorm() {
    let conv = make_node(
        OpKind::Conv,
        "conv",
        vec!["x", "conv_w", "conv_b"],
        vec!["conv_out"],
    );
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

    let nodes = vec![conv, bn];
    let mut weights = HashMap::new();
    weights.insert(
        "conv_w".to_string(),
        Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
    );
    weights.insert("conv_b".to_string(), Tensor::new(vec![0.0], vec![1]));
    weights.insert("bn_scale".to_string(), Tensor::new(vec![1.0], vec![1]));
    weights.insert("bn_bias".to_string(), Tensor::new(vec![0.0], vec![1]));
    weights.insert("bn_mean".to_string(), Tensor::new(vec![0.0], vec![1]));
    weights.insert("bn_var".to_string(), Tensor::new(vec![1.0], vec![1]));

    let result = fuse_conv_batchnorm(nodes, &mut weights);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Conv));
    assert_eq!(result[0].outputs[0], "bn_out");
    assert!(weights.contains_key("conv_fused_weight"));
    assert!(weights.contains_key("conv_fused_bias"));
}

#[test]
fn test_fuse_conv_batchnorm_no_conv_bias() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "conv_w"], vec!["conv_out"]);
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

    let nodes = vec![conv, bn];
    let mut weights = HashMap::new();
    weights.insert(
        "conv_w".to_string(),
        Tensor::new(vec![2.0], vec![1, 1, 1, 1]),
    );
    weights.insert("bn_scale".to_string(), Tensor::new(vec![3.0], vec![1]));
    weights.insert("bn_bias".to_string(), Tensor::new(vec![0.5], vec![1]));
    weights.insert("bn_mean".to_string(), Tensor::new(vec![1.0], vec![1]));
    weights.insert("bn_var".to_string(), Tensor::new(vec![4.0], vec![1]));

    let result = fuse_conv_batchnorm(nodes, &mut weights);
    assert_eq!(result.len(), 1);

    let fused_w = weights.get("conv_fused_weight").expect("fused weight");
    let inv_std = 1.0 / (4.0f32 + 1e-5).sqrt();
    let expected_w = 2.0 * 3.0 * inv_std;
    assert!((fused_w.data[0] - expected_w).abs() < 1e-5);

    let fused_b = weights.get("conv_fused_bias").expect("fused bias");
    let expected_b = (0.0 - 1.0) * 3.0 * inv_std + 0.5;
    assert!((fused_b.data[0] - expected_b).abs() < 1e-5);
}

#[test]
fn test_fuse_conv_batchnorm_multiple_consumers() {
    let conv = make_node(
        OpKind::Conv,
        "conv",
        vec!["x", "conv_w", "conv_b"],
        vec!["conv_out"],
    );
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);
    let relu = make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["relu_out"]);

    let nodes = vec![conv, bn, relu];
    let mut weights = HashMap::new();
    weights.insert(
        "conv_w".to_string(),
        Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
    );
    weights.insert("conv_b".to_string(), Tensor::new(vec![0.0], vec![1]));
    weights.insert("bn_scale".to_string(), Tensor::new(vec![1.0], vec![1]));
    weights.insert("bn_bias".to_string(), Tensor::new(vec![0.0], vec![1]));
    weights.insert("bn_mean".to_string(), Tensor::new(vec![0.0], vec![1]));
    weights.insert("bn_var".to_string(), Tensor::new(vec![1.0], vec![1]));

    let result = fuse_conv_batchnorm(nodes, &mut weights);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_fuse_conv_relu() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
    let relu = make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["relu_out"]);

    let nodes = vec![conv, relu];
    let result = fuse_conv_relu(nodes);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Conv));
    assert_eq!(result[0].outputs[0], "relu_out");
    assert_eq!(result[0].attrs.s("activation"), "relu");
}

#[test]
fn test_fuse_conv_clip_as_relu() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
    clip.attrs.floats.insert("min".to_string(), 0.0);
    clip.attrs.floats.insert("max".to_string(), f32::INFINITY);

    let nodes = vec![conv, clip];
    let result = fuse_conv_relu(nodes);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Conv));
    assert_eq!(result[0].outputs[0], "clip_out");
    assert_eq!(result[0].attrs.s("activation"), "relu");
}

#[test]
fn test_fuse_conv_clip_general() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
    clip.attrs.floats.insert("min".to_string(), 0.0);
    clip.attrs.floats.insert("max".to_string(), 6.0);

    let nodes = vec![conv, clip];
    let result = fuse_conv_relu(nodes);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].attrs.s("activation"), "clip");
    assert_eq!(result[0].attrs.f("activation_min", -1.0), 0.0);
    assert_eq!(result[0].attrs.f("activation_max", -1.0), 6.0);
}

#[test]
fn test_fuse_conv_relu_no_fusion_multiple_consumers() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let relu = make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["relu_out"]);
    let add = make_node(
        OpKind::Add,
        "add",
        vec!["conv_out", "other"],
        vec!["add_out"],
    );

    let nodes = vec![conv, relu, add];
    let result = fuse_conv_relu(nodes);

    assert_eq!(result.len(), 3);
}

// --- fuse_conv_clip_to_conv_relu6 tests ---

#[test]
fn test_fuse_conv_clip_to_conv_relu6_basic() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
    clip.attrs.floats.insert("min".to_string(), 0.0);
    clip.attrs.floats.insert("max".to_string(), 6.0);

    let nodes = vec![conv, clip];
    let result = fuse_conv_clip_to_conv_relu6(nodes);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Conv));
    assert_eq!(result[0].attrs.s("activation"), "relu6");
    assert_eq!(result[0].attrs.f("activation_min", -1.0), 0.0);
    assert_eq!(result[0].attrs.f("activation_max", -1.0), 6.0);
    assert_eq!(result[0].outputs, vec!["clip_out"]);
}

#[test]
fn test_fuse_conv_clip_to_conv_relu6_wrong_range() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
    clip.attrs.floats.insert("min".to_string(), 0.0);
    clip.attrs.floats.insert("max".to_string(), 1.0); // Not 6.0

    let nodes = vec![conv, clip];
    let result = fuse_conv_clip_to_conv_relu6(nodes);

    // Not ReLU6 range, no fusion
    assert_eq!(result.len(), 2);
}

#[test]
fn test_fuse_conv_clip_to_conv_relu6_not_conv() {
    // Relu followed by Clip(0,6) — not a Conv, so no fusion
    let relu = make_node(OpKind::Relu, "relu", vec!["x"], vec!["relu_out"]);
    let mut clip = make_node(OpKind::Clip, "clip", vec!["relu_out"], vec!["clip_out"]);
    clip.attrs.floats.insert("min".to_string(), 0.0);
    clip.attrs.floats.insert("max".to_string(), 6.0);

    let nodes = vec![relu, clip];
    let result = fuse_conv_clip_to_conv_relu6(nodes);

    assert_eq!(result.len(), 2);
}

#[test]
fn test_fuse_conv_clip_to_conv_relu6_multiple_consumers() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
    clip.attrs.floats.insert("min".to_string(), 0.0);
    clip.attrs.floats.insert("max".to_string(), 6.0);
    let add = make_node(
        OpKind::Add,
        "add",
        vec!["conv_out", "other"],
        vec!["add_out"],
    );

    let nodes = vec![conv, clip, add];
    let result = fuse_conv_clip_to_conv_relu6(nodes);

    // conv_out has 2 consumers, no fusion
    assert_eq!(result.len(), 3);
}

// --- fold_batch_norm_inference tests ---

#[test]
fn test_fold_batch_norm_inference_basic() {
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

    let nodes = vec![bn];
    let mut weights = HashMap::new();
    weights.insert("scale".to_string(), Tensor::new(vec![2.0], vec![1]));
    weights.insert("bias".to_string(), Tensor::new(vec![0.5], vec![1]));
    weights.insert("mean".to_string(), Tensor::new(vec![1.0], vec![1]));
    weights.insert("var".to_string(), Tensor::new(vec![4.0], vec![1]));

    // BN input is `[N=1, C=1, H=4, W=4]` — gives synthesized constants
    // shape `[1, 1, 1, 1]`.
    let mut known_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    known_shapes.insert("x".to_string(), vec![1, 1, 4, 4]);

    let result = fold_batch_norm_inference(nodes, &mut weights, &known_shapes);

    // BN replaced with Mul + Add
    assert_eq!(result.len(), 2);
    assert!(matches!(result[0].op, OpKind::Mul));
    assert!(matches!(result[1].op, OpKind::Add));

    // Check outputs — final output should match BN's output
    assert_eq!(result[1].outputs, vec!["bn_out"]);
    // Mul takes X as input
    assert_eq!(result[0].inputs[0], "x");

    // Verify precomputed factor and shift
    let inv_std = 1.0 / (4.0f32 + 1e-5).sqrt();
    let expected_factor = 2.0 * inv_std;
    let expected_shift = 0.5 - 1.0 * expected_factor;

    let factor = weights.get("bn_bn_factor").expect("factor weight");
    assert_eq!(factor.shape, vec![1, 1, 1, 1]);
    assert!((factor.data[0] - expected_factor).abs() < 1e-5);

    let shift = weights.get("bn_bn_shift").expect("shift weight");
    assert_eq!(shift.shape, vec![1, 1, 1, 1]);
    assert!((shift.data[0] - expected_shift).abs() < 1e-5);
}

#[test]
fn test_fold_batch_norm_inference_skips_conv_preceded() {
    // When BN is preceded by Conv, fuse_conv_batchnorm handles it
    let conv = make_node(OpKind::Conv, "conv", vec!["inp", "w"], vec!["conv_out"]);
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["conv_out", "scale", "bias", "mean", "var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

    let nodes = vec![conv, bn];
    let mut weights = HashMap::new();
    weights.insert("scale".to_string(), Tensor::new(vec![1.0], vec![1]));
    weights.insert("bias".to_string(), Tensor::new(vec![0.0], vec![1]));
    weights.insert("mean".to_string(), Tensor::new(vec![0.0], vec![1]));
    weights.insert("var".to_string(), Tensor::new(vec![1.0], vec![1]));

    let mut known_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    known_shapes.insert("conv_out".to_string(), vec![1, 1, 4, 4]);

    let result = fold_batch_norm_inference(nodes, &mut weights, &known_shapes);

    // Should NOT fold — Conv precedes BN
    assert_eq!(result.len(), 2);
    assert!(matches!(result[0].op, OpKind::Conv));
    assert!(matches!(result[1].op, OpKind::BatchNorm));
}

#[test]
fn test_fold_batch_norm_inference_missing_weights() {
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

    let nodes = vec![bn];
    let mut weights = HashMap::new();
    weights.insert("scale".to_string(), Tensor::new(vec![1.0], vec![1]));
    // Missing bias, mean, var weights

    let mut known_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    known_shapes.insert("x".to_string(), vec![1, 1, 4, 4]);

    let result = fold_batch_norm_inference(nodes, &mut weights, &known_shapes);

    // Should not fold — missing weights
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::BatchNorm));
}

#[test]
fn test_fold_batch_norm_inference_multi_channel() {
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 0.001);

    let nodes = vec![bn];
    let mut weights = HashMap::new();
    weights.insert(
        "scale".to_string(),
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
    );
    weights.insert(
        "bias".to_string(),
        Tensor::new(vec![0.1, 0.2, 0.3], vec![3]),
    );
    weights.insert(
        "mean".to_string(),
        Tensor::new(vec![0.5, 1.0, 1.5], vec![3]),
    );
    weights.insert("var".to_string(), Tensor::new(vec![1.0, 2.0, 4.0], vec![3]));

    // BN input is `[1, 3, 8, 8]` → emitted constants shape `[1, 3, 1, 1]`.
    let mut known_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    known_shapes.insert("x".to_string(), vec![1, 3, 8, 8]);

    let result = fold_batch_norm_inference(nodes, &mut weights, &known_shapes);

    assert_eq!(result.len(), 2);
    assert!(matches!(result[0].op, OpKind::Mul));
    assert!(matches!(result[1].op, OpKind::Add));

    let factor = weights.get("bn_bn_factor").expect("factor");
    assert_eq!(factor.shape, vec![1, 3, 1, 1]);
    let shift = weights.get("bn_bn_shift").expect("shift");
    assert_eq!(shift.shape, vec![1, 3, 1, 1]);

    // Verify channel 0: scale=1.0, var=1.0, eps=0.001
    let inv_std_0 = 1.0 / (1.0f32 + 0.001).sqrt();
    let expected_f0 = 1.0 * inv_std_0;
    assert!((factor.data[0] - expected_f0).abs() < 1e-5);
}

#[test]
fn test_fold_batch_norm_inference_shape_mismatch() {
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

    let nodes = vec![bn];
    let mut weights = HashMap::new();
    weights.insert("scale".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
    weights.insert("bias".to_string(), Tensor::new(vec![0.0], vec![1])); // Mismatch!
    weights.insert("mean".to_string(), Tensor::new(vec![0.0, 0.0], vec![2]));
    weights.insert("var".to_string(), Tensor::new(vec![1.0, 1.0], vec![2]));

    let mut known_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    known_shapes.insert("x".to_string(), vec![1, 2, 4, 4]);

    let result = fold_batch_norm_inference(nodes, &mut weights, &known_shapes);

    // Shape mismatch — should not fold
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::BatchNorm));
}

/// Regression: ArcFace (ResNet50) trips on the standalone BN fold when the
/// input is `[1, 64, 112, 112]` — earlier versions emitted `factor`/`shift`
/// as `[64]`, which fails strict NumPy alignment against `[N, C, H, W]`
/// (the trailing dim becomes `64 vs 112`).  Verify both shapes and runtime
/// elementwise correctness against the reference `BatchNormalization` op.
#[test]
fn test_fold_batch_norm_resnet_arcface_first_layer() {
    use oxionnx_ops::math;
    use oxionnx_ops::nn;

    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["x", "scale", "bias", "mean", "var"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

    let nodes = vec![bn];
    let mut weights = HashMap::new();

    // Deterministic per-channel parameters (C=64).
    let c = 64usize;
    let scale: Vec<f32> = (0..c).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let bias: Vec<f32> = (0..c).map(|i| 0.1 + (i as f32) * 0.005).collect();
    let mean: Vec<f32> = (0..c).map(|i| (i as f32) * 0.02 - 0.5).collect();
    let var: Vec<f32> = (0..c).map(|i| 0.25 + (i as f32) * 0.01).collect();
    weights.insert("scale".to_string(), Tensor::new(scale.clone(), vec![c]));
    weights.insert("bias".to_string(), Tensor::new(bias.clone(), vec![c]));
    weights.insert("mean".to_string(), Tensor::new(mean.clone(), vec![c]));
    weights.insert("var".to_string(), Tensor::new(var.clone(), vec![c]));

    // ResNet first conv output: `[1, 64, 112, 112]`.
    let n = 1usize;
    let h = 112usize;
    let w = 112usize;
    let mut known_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    known_shapes.insert("x".to_string(), vec![n, c, h, w]);

    let folded = fold_batch_norm_inference(nodes, &mut weights, &known_shapes);

    // BN folded into Mul + Add.
    assert_eq!(folded.len(), 2, "expected Mul + Add after fold");
    assert!(matches!(folded[0].op, OpKind::Mul));
    assert!(matches!(folded[1].op, OpKind::Add));

    // Synthesized constants must broadcast against `[N, C, H, W]` under strict
    // NumPy alignment from the trailing dim — i.e. `[1, C, 1, 1]`.
    let factor = weights.get("bn_bn_factor").expect("factor weight");
    assert_eq!(factor.shape, vec![1, c, 1, 1]);
    let shift = weights.get("bn_bn_shift").expect("shift weight");
    assert_eq!(shift.shape, vec![1, c, 1, 1]);

    // Build a deterministic input tensor and run both paths.
    let total = n * c * h * w;
    let x_data: Vec<f32> = (0..total)
        .map(|i| ((i % 257) as f32 - 128.0) / 64.0)
        .collect();
    let x_tensor = Tensor::new(x_data, vec![n, c, h, w]);

    // Reference: run the bona fide BatchNormalization op.
    let scale_t = Tensor::new(scale, vec![c]);
    let bias_t = Tensor::new(bias, vec![c]);
    let mean_t = Tensor::new(mean, vec![c]);
    let var_t = Tensor::new(var, vec![c]);
    let reference =
        nn::batch_norm(&x_tensor, &scale_t, &bias_t, &mean_t, &var_t, 1e-5).expect("reference BN");

    // Folded path: Mul(x, factor) → Add(_, shift).
    let mul_out = math::mul(&x_tensor, factor).expect("mul broadcast");
    let folded_out = math::add(&mul_out, shift).expect("add broadcast");

    assert_eq!(folded_out.shape, reference.shape);
    let mut max_abs_err = 0.0f32;
    for (a, b) in folded_out.data.iter().zip(reference.data.iter()) {
        let e = (a - b).abs();
        if e > max_abs_err {
            max_abs_err = e;
        }
    }
    assert!(
        max_abs_err < 1e-4,
        "folded path diverges from BatchNorm: max abs err {max_abs_err}"
    );

    // Sanity: also verify the broadcaster alone accepts both `[1,C,1,1]` and
    // `[C]` (trailing-dim broadcast still works for 1-D when matched).
    let small = Tensor::new(vec![1.0_f32; n * c * h * w], vec![n, c, h, w]);
    let bias_4d = Tensor::new(vec![0.5_f32; c], vec![1, c, 1, 1]);
    let _ = math::add(&small, &bias_4d).expect("[N,C,H,W] + [1,C,1,1] should broadcast");
}

// --- fuse_conv_add_relu tests ---

#[test]
fn test_fuse_conv_add_relu_basic() {
    // Conv(x, w, b) → Add(conv_out, residual) → Relu → fused ConvAddRelu
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
    let add = make_node(
        OpKind::Add,
        "add",
        vec!["conv_out", "residual"],
        vec!["add_out"],
    );
    let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);

    let nodes = vec![conv, add, relu];
    let result = fuse_conv_add_relu(nodes);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::ConvAddRelu));
    assert_eq!(result[0].inputs, vec!["x", "w", "b", "residual"]);
    assert_eq!(result[0].outputs, vec!["relu_out"]);
}

#[test]
fn test_fuse_conv_add_relu_reversed_add_inputs() {
    // Add(residual, conv_out) — reversed order
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
    let add = make_node(
        OpKind::Add,
        "add",
        vec!["residual", "conv_out"],
        vec!["add_out"],
    );
    let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);

    let nodes = vec![conv, add, relu];
    let result = fuse_conv_add_relu(nodes);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::ConvAddRelu));
    assert_eq!(result[0].inputs, vec!["x", "w", "b", "residual"]);
}

#[test]
fn test_fuse_conv_add_relu_no_bias() {
    // Conv without bias input
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let add = make_node(
        OpKind::Add,
        "add",
        vec!["conv_out", "residual"],
        vec!["add_out"],
    );
    let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);

    let nodes = vec![conv, add, relu];
    let result = fuse_conv_add_relu(nodes);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::ConvAddRelu));
    // Bias slot should be empty string
    assert_eq!(result[0].inputs[2], "");
    assert_eq!(result[0].inputs[3], "residual");
}

#[test]
fn test_fuse_conv_add_relu_no_fusion_conv_multiple_consumers() {
    // conv_out consumed by both Add and another node — don't fuse
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
    let add = make_node(
        OpKind::Add,
        "add",
        vec!["conv_out", "residual"],
        vec!["add_out"],
    );
    let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);
    let extra = make_node(OpKind::Relu, "extra", vec!["conv_out"], vec!["extra_out"]);

    let nodes = vec![conv, add, relu, extra];
    let result = fuse_conv_add_relu(nodes);

    assert_eq!(result.len(), 4);
    assert!(matches!(result[0].op, OpKind::Conv));
}

#[test]
fn test_fuse_conv_add_relu_no_fusion_add_multiple_consumers() {
    // add_out consumed by both Relu and another node — don't fuse
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
    let add = make_node(
        OpKind::Add,
        "add",
        vec!["conv_out", "residual"],
        vec!["add_out"],
    );
    let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);
    let extra = make_node(OpKind::Sigmoid, "extra", vec!["add_out"], vec!["extra_out"]);

    let nodes = vec![conv, add, relu, extra];
    let result = fuse_conv_add_relu(nodes);

    assert_eq!(result.len(), 4);
    assert!(matches!(result[0].op, OpKind::Conv));
}

#[test]
fn test_fuse_conv_add_relu_no_fusion_not_relu() {
    // Pattern ends with Sigmoid instead of Relu — don't fuse
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
    let add = make_node(
        OpKind::Add,
        "add",
        vec!["conv_out", "residual"],
        vec!["add_out"],
    );
    let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["add_out"], vec!["sig_out"]);

    let nodes = vec![conv, add, sigmoid];
    let result = fuse_conv_add_relu(nodes);

    // No fusion: pattern requires Relu at the end
    assert_eq!(result.len(), 3);
}
