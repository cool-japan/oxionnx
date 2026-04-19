//! Tests for MatMul-related fusion passes.

use crate::graph::OpKind;
use crate::optimizer::fusion::matmul::{
    fuse_add_matmul_to_gemm, fuse_layer_norm, fuse_matmul_add, fuse_matmul_transpose,
};
use crate::optimizer::test_utils::{make_layer_norm_pattern, make_node};
use crate::tensor::Tensor;
use std::collections::HashMap;

#[test]
fn test_fuse_matmul_add() {
    let nodes = vec![
        make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
        make_node(OpKind::Add, "add", vec!["mm_out", "bias"], vec!["add_out"]),
    ];
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));
    weights.insert("bias".to_string(), Tensor::new(vec![0.5, 0.5], vec![2]));

    let result = fuse_matmul_add(nodes, &weights);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Gemm));
    assert_eq!(result[0].outputs[0], "add_out");
    assert_eq!(result[0].inputs.len(), 3);
    assert_eq!(result[0].inputs[0], "x");
    assert_eq!(result[0].inputs[1], "w");
    assert_eq!(result[0].inputs[2], "bias");
}

#[test]
fn test_fuse_matmul_add_single_node() {
    let nodes = vec![make_node(
        OpKind::MatMul,
        "mm",
        vec!["x", "w"],
        vec!["mm_out"],
    )];
    let weights = HashMap::new();
    let result = fuse_matmul_add(nodes, &weights);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_fuse_matmul_add_bias_not_1d() {
    let nodes = vec![
        make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
        make_node(OpKind::Add, "add", vec!["mm_out", "bias"], vec!["add_out"]),
    ];
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));
    weights.insert("bias".to_string(), Tensor::new(vec![0.5; 4], vec![2, 2]));
    let result = fuse_matmul_add(nodes, &weights);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_no_fusion_when_multiple_consumers() {
    let nodes = vec![
        make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
        make_node(OpKind::Add, "add", vec!["mm_out", "bias"], vec!["add_out"]),
        make_node(OpKind::Relu, "relu", vec!["mm_out"], vec!["relu_out"]),
    ];
    let weights = {
        let mut w = HashMap::new();
        w.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));
        w.insert("bias".to_string(), Tensor::new(vec![0.5, 0.5], vec![2]));
        w
    };

    let result = fuse_matmul_add(nodes, &weights);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_fuse_layer_norm_basic() {
    let (nodes, weights) = make_layer_norm_pattern(false);
    let result = fuse_layer_norm(nodes, &weights);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::LayerNorm));
    assert_eq!(result[0].inputs[0], "X");
    assert_eq!(result[0].outputs[0], "normalized");
    let eps = result[0].attrs.f("epsilon", 0.0);
    assert!((eps - 1e-5).abs() < 1e-8);
}

#[test]
fn test_fuse_layer_norm_with_scale_bias() {
    let (nodes, weights) = make_layer_norm_pattern(true);
    let result = fuse_layer_norm(nodes, &weights);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::LayerNorm));
    assert_eq!(result[0].inputs.len(), 3);
    assert_eq!(result[0].inputs[0], "X");
    assert_eq!(result[0].inputs[1], "scale");
    assert_eq!(result[0].inputs[2], "bias");
    assert_eq!(result[0].outputs[0], "output");
}

#[test]
fn test_fuse_layer_norm_no_match_wrong_pow() {
    let (nodes, mut weights) = make_layer_norm_pattern(false);
    weights.insert("pow_exp".to_string(), Tensor::new(vec![3.0], vec![1]));

    let original_len = nodes.len();
    let result = fuse_layer_norm(nodes, &weights);

    assert_eq!(result.len(), original_len);
}

// --- fuse_matmul_transpose tests ---

#[test]
fn test_fuse_matmul_transpose_2d() {
    let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
    let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["t_out"]);
    transpose
        .attrs
        .int_lists
        .insert("perm".to_string(), vec![1, 0]);

    let nodes = vec![matmul, transpose];
    let result = fuse_matmul_transpose(nodes);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Gemm));
    // Should be Gemm(B, A) with transA=1, transB=1
    assert_eq!(result[0].inputs[0], "b");
    assert_eq!(result[0].inputs[1], "a");
    assert_eq!(result[0].attrs.i("transA", 0), 1);
    assert_eq!(result[0].attrs.i("transB", 0), 1);
    assert_eq!(result[0].outputs[0], "t_out");
}

#[test]
fn test_fuse_matmul_transpose_3d_last_two() {
    let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
    let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["t_out"]);
    transpose
        .attrs
        .int_lists
        .insert("perm".to_string(), vec![0, 2, 1]);

    let nodes = vec![matmul, transpose];
    let result = fuse_matmul_transpose(nodes);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Gemm));
}

#[test]
fn test_fuse_matmul_transpose_no_fusion_wrong_perm() {
    let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
    let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["t_out"]);
    // Perm [2, 0, 1] doesn't just swap last two dims
    transpose
        .attrs
        .int_lists
        .insert("perm".to_string(), vec![2, 0, 1]);

    let nodes = vec![matmul, transpose];
    let result = fuse_matmul_transpose(nodes);

    assert_eq!(result.len(), 2);
}

#[test]
fn test_fuse_matmul_transpose_no_fusion_multiple_consumers() {
    let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
    let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["t_out"]);
    transpose
        .attrs
        .int_lists
        .insert("perm".to_string(), vec![1, 0]);
    let relu = make_node(OpKind::Relu, "relu", vec!["mm_out"], vec!["relu_out"]);

    let nodes = vec![matmul, transpose, relu];
    let result = fuse_matmul_transpose(nodes);

    assert_eq!(result.len(), 3);
}

// --- fuse_add_matmul_to_gemm tests ---

#[test]
fn test_fuse_add_matmul_to_gemm() {
    let add = make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]);
    let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

    let nodes = vec![add, matmul];
    let mut weights = HashMap::new();
    // bias: [2], W: [2, 3]
    weights.insert("bias".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
    weights.insert(
        "w".to_string(),
        Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]),
    );

    let result = fuse_add_matmul_to_gemm(nodes, &mut weights);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Gemm));
    assert_eq!(result[0].inputs[0], "x");
    assert_eq!(result[0].inputs[1], "w");
    assert_eq!(result[0].inputs.len(), 3);
    assert_eq!(result[0].outputs[0], "mm_out");
    assert_eq!(result[0].attrs.f("alpha", 0.0), 1.0);
    assert_eq!(result[0].attrs.f("beta", 0.0), 1.0);

    // fused_bias = [1, 2] @ [[1,0,0],[0,1,0]] = [1, 2, 0]
    let fused_bias_name = &result[0].inputs[2];
    let fused_bias = weights
        .get(fused_bias_name)
        .expect("fused bias should exist");
    assert_eq!(fused_bias.shape, vec![3]);
    assert!((fused_bias.data[0] - 1.0).abs() < 1e-6);
    assert!((fused_bias.data[1] - 2.0).abs() < 1e-6);
    assert!((fused_bias.data[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_fuse_add_matmul_to_gemm_bias_first_input() {
    // Add with bias as first input
    let add = make_node(OpKind::Add, "add", vec!["bias", "x"], vec!["add_out"]);
    let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

    let nodes = vec![add, matmul];
    let mut weights = HashMap::new();
    weights.insert("bias".to_string(), Tensor::new(vec![3.0, 4.0], vec![2]));
    weights.insert(
        "w".to_string(),
        Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
    );

    let result = fuse_add_matmul_to_gemm(nodes, &mut weights);

    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Gemm));
    assert_eq!(result[0].inputs[0], "x");
}

#[test]
fn test_fuse_add_matmul_no_fusion_bias_not_1d() {
    let add = make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]);
    let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

    let nodes = vec![add, matmul];
    let mut weights = HashMap::new();
    weights.insert("bias".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));
    weights.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));

    let result = fuse_add_matmul_to_gemm(nodes, &mut weights);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_fuse_add_matmul_no_fusion_w_not_in_weights() {
    let add = make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]);
    let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

    let nodes = vec![add, matmul];
    let mut weights = HashMap::new();
    weights.insert("bias".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
    // w not in weights

    let result = fuse_add_matmul_to_gemm(nodes, &mut weights);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_fuse_add_matmul_no_fusion_shape_mismatch() {
    let add = make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]);
    let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

    let nodes = vec![add, matmul];
    let mut weights = HashMap::new();
    // bias [3] vs W [2, 2] — K mismatch
    weights.insert(
        "bias".to_string(),
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
    );
    weights.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));

    let result = fuse_add_matmul_to_gemm(nodes, &mut weights);
    assert_eq!(result.len(), 2);
}
