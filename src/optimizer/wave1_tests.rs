//! Wave-1 correctness regression tests for the optimizer passes.
//!
//! Each test pins one soundness property that a pass previously violated:
//! opset-11 Clip bounds, CSE operand order / subgraph attributes /
//! non-determinism, graph outputs surviving every rewrite, LayerNorm axis
//! resolution, collision-free generated weight names, the Transpose/Reshape
//! rewrites' layout preconditions, `OptLevel` pass gating, constant folding of
//! graph outputs, and the 2-D `Gemm` contract.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::missing_graph_outputs;
use crate::optimizer::test_utils::make_node;
use crate::optimizer::{fusion, optimize, optimize_with_level, PassLevel};
use crate::tensor::Tensor;
use oxionnx_core::{OnnxError, OpContext, Operator, OperatorRegistry};
use std::collections::HashMap;

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::new(data, shape)
}

// ───────────────────────── a4-0: Conv + Clip bounds ─────────────────────────

/// A modern (opset ≥ 11) `Conv → Clip(0, 6)` keeps its clamp: the bounds live
/// in input slots 1 and 2, not in attributes.
#[test]
fn test_conv_clip_opset11_input_bounds_are_fused() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let clip = make_node(
        OpKind::Clip,
        "clip",
        vec!["conv_out", "clip_min", "clip_max"],
        vec!["y"],
    );
    let mut weights = HashMap::new();
    weights.insert("clip_min".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("clip_max".to_string(), tensor(vec![6.0], vec![1]));

    let result = fusion::fuse_conv_relu(vec![conv, clip], &weights, &names(&["y"]));

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].attrs.s("activation"), "clip");
    assert_eq!(result[0].attrs.f("activation_min", -1.0), 0.0);
    assert_eq!(result[0].attrs.f("activation_max", -1.0), 6.0);
}

/// End-to-end: the fused Conv actually clamps.  A 1×1 identity convolution over
/// `[1, 2, 3, 10]` must come out as `[1, 2, 3, 6]`, not `[1, 2, 3, 10]`.
#[test]
fn test_conv_clip_opset11_fused_node_still_clamps() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let clip = make_node(
        OpKind::Clip,
        "clip",
        vec!["conv_out", "clip_min", "clip_max"],
        vec!["y"],
    );
    let mut weights = HashMap::new();
    weights.insert("clip_min".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("clip_max".to_string(), tensor(vec![6.0], vec![1]));

    let fused = fusion::fuse_conv_relu(vec![conv, clip], &weights, &names(&["y"]));
    assert_eq!(fused.len(), 1);

    let x = tensor(vec![1.0, 2.0, 3.0, 10.0], vec![1, 1, 1, 4]);
    let w = tensor(vec![1.0], vec![1, 1, 1, 1]);
    let registry = oxionnx_ops::default_registry();
    let conv_op = registry.get("Conv").expect("Conv operator");
    let ctx = OpContext {
        node: &fused[0],
        inputs: vec![Some(&x), Some(&w)],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let out = conv_op.execute(&ctx).expect("fused conv executes");
    assert_eq!(out[0].data, vec![1.0, 2.0, 3.0, 6.0]);
}

/// Bounds that are only known at run time must not be folded away.
#[test]
fn test_conv_clip_dynamic_bounds_not_fused() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let clip = make_node(
        OpKind::Clip,
        "clip",
        vec!["conv_out", "dyn_min", "dyn_max"],
        vec!["y"],
    );
    let weights = HashMap::new();
    let result = fusion::fuse_conv_relu(vec![conv, clip], &weights, &names(&["y"]));
    assert_eq!(result.len(), 2);
}

/// `min > max` and NaN bounds would make `f32::clamp` panic inside the kernel.
#[test]
fn test_conv_clip_inverted_bounds_not_fused() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["y"]);
    clip.attrs.floats.insert("min".to_string(), 6.0);
    clip.attrs.floats.insert("max".to_string(), 0.0);
    let weights = HashMap::new();
    let result = fusion::fuse_conv_relu(vec![conv, clip], &weights, &names(&["y"]));
    assert_eq!(result.len(), 2);

    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["y"]);
    clip.attrs.floats.insert("min".to_string(), f32::NAN);
    let result = fusion::fuse_conv_relu(vec![conv, clip], &weights, &names(&["y"]));
    assert_eq!(result.len(), 2);
}

/// The ReLU6 pass must emit a label the Conv kernels implement.
#[test]
fn test_relu6_pass_emits_kernel_understood_clip_label() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let clip = make_node(
        OpKind::Clip,
        "clip",
        vec!["conv_out", "clip_min", "clip_max"],
        vec!["y"],
    );
    let mut weights = HashMap::new();
    weights.insert("clip_min".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("clip_max".to_string(), tensor(vec![6.0], vec![1]));

    let result = fusion::fuse_conv_clip_to_conv_relu6(vec![conv, clip], &weights, &names(&["y"]));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].attrs.s("activation"), "clip");
    assert_eq!(result[0].attrs.f("activation_max", -1.0), 6.0);
}

// ─────────────────────────── a4-1 / a4-2: CSE ───────────────────────────────

/// `Sub(a, b)` and `Sub(b, a)` are different expressions.
#[test]
fn test_cse_keeps_swapped_non_commutative_operands() {
    let nodes = vec![
        make_node(OpKind::Sub, "d1", vec!["a", "b"], vec!["out1"]),
        make_node(OpKind::Sub, "d2", vec!["b", "a"], vec!["out2"]),
        make_node(OpKind::Concat, "cat", vec!["out1", "out2"], vec!["y"]),
    ];
    let result = crate::optimizer::cse::eliminate_common_subexpressions(nodes, &names(&["y"]));
    assert_eq!(result.len(), 3);
    assert_eq!(result[2].inputs, names(&["out1", "out2"]));
}

/// Same for `MatMul(Q, K)` vs `MatMul(K, Q)` and `Gather(data, idx)` vs
/// `Gather(idx, data)`.
#[test]
fn test_cse_keeps_swapped_matmul_and_gather_operands() {
    let nodes = vec![
        make_node(OpKind::MatMul, "m1", vec!["q", "k"], vec!["o1"]),
        make_node(OpKind::MatMul, "m2", vec!["k", "q"], vec!["o2"]),
        make_node(OpKind::Gather, "g1", vec!["data", "idx"], vec!["o3"]),
        make_node(OpKind::Gather, "g2", vec!["idx", "data"], vec!["o4"]),
    ];
    let result = crate::optimizer::cse::eliminate_common_subexpressions(nodes, &[]);
    assert_eq!(result.len(), 4);
}

/// Two `If` nodes sharing a condition but carrying different branches are not
/// the same expression.
#[test]
fn test_cse_keeps_if_nodes_with_different_subgraphs() {
    let mut branch_a = crate::graph::Graph::default();
    branch_a.nodes.push(make_node(
        OpKind::Relu,
        "inner_a",
        vec!["v"],
        vec!["branch_out"],
    ));
    branch_a.output_names = names(&["branch_out"]);

    let mut branch_b = crate::graph::Graph::default();
    branch_b.nodes.push(make_node(
        OpKind::Sigmoid,
        "inner_b",
        vec!["v"],
        vec!["branch_out"],
    ));
    branch_b.output_names = names(&["branch_out"]);

    let mut if_a = make_node(OpKind::If, "if_a", vec!["cond"], vec!["out1"]);
    if_a.attrs
        .graphs
        .insert("then_branch".to_string(), branch_a);
    let mut if_b = make_node(OpKind::If, "if_b", vec!["cond"], vec!["out2"]);
    if_b.attrs
        .graphs
        .insert("then_branch".to_string(), branch_b);

    let result = crate::optimizer::cse::eliminate_common_subexpressions(vec![if_a, if_b], &[]);
    assert_eq!(result.len(), 2);
}

/// String-list attributes participate in the fingerprint.
#[test]
fn test_cse_keeps_nodes_with_different_string_lists() {
    let mut n1 = make_node(OpKind::LSTM, "l1", vec!["x", "w"], vec!["o1"]);
    n1.attrs
        .string_lists
        .insert("activations".to_string(), names(&["Sigmoid", "Tanh"]));
    let mut n2 = make_node(OpKind::LSTM, "l2", vec!["x", "w"], vec!["o2"]);
    n2.attrs
        .string_lists
        .insert("activations".to_string(), names(&["Relu", "Tanh"]));

    let result = crate::optimizer::cse::eliminate_common_subexpressions(vec![n1, n2], &[]);
    assert_eq!(result.len(), 2);
}

/// The constant-tensor hash must be order sensitive: `[1, 2]` ≠ `[2, 1]`.
#[test]
fn test_cse_keeps_constants_with_permuted_data() {
    let mut c1 = make_node(OpKind::Constant, "c1", vec!["seed"], vec!["o1"]);
    c1.attrs
        .tensors
        .insert("value".to_string(), tensor(vec![1.0, 2.0], vec![2]));
    let mut c2 = make_node(OpKind::Constant, "c2", vec!["seed"], vec!["o2"]);
    c2.attrs
        .tensors
        .insert("value".to_string(), tensor(vec![2.0, 1.0], vec![2]));

    let result = crate::optimizer::cse::eliminate_common_subexpressions(vec![c1, c2], &[]);
    assert_eq!(result.len(), 2);
}

/// Non-deterministic ops are never merged, however identical they look.
#[test]
fn test_cse_keeps_non_deterministic_nodes() {
    let nodes = vec![
        make_node(OpKind::Dropout, "d1", vec!["x"], vec!["o1"]),
        make_node(OpKind::Dropout, "d2", vec!["x"], vec!["o2"]),
        make_node(OpKind::Bernoulli, "b1", vec!["x"], vec!["o3"]),
        make_node(OpKind::Bernoulli, "b2", vec!["x"], vec!["o4"]),
    ];
    let result = crate::optimizer::cse::eliminate_common_subexpressions(nodes, &[]);
    assert_eq!(result.len(), 4);
}

/// A duplicate whose output the model exports keeps its node.
#[test]
fn test_cse_keeps_duplicate_producing_graph_output() {
    let nodes = vec![
        make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]),
        make_node(OpKind::Add, "add2", vec!["x", "y"], vec!["out2"]),
    ];
    let result =
        crate::optimizer::cse::eliminate_common_subexpressions(nodes, &names(&["out1", "out2"]));
    assert_eq!(result.len(), 2);
    assert!(missing_graph_outputs(&result, &HashMap::new(), &names(&["out1", "out2"])).is_empty());
}

// ─────────────────── a4-3: graph outputs survive every pass ─────────────────

#[test]
fn test_conv_batchnorm_not_fused_when_conv_output_is_exported() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let mut bn = make_node(
        OpKind::BatchNorm,
        "bn",
        vec!["conv_out", "s", "b", "m", "v"],
        vec!["bn_out"],
    );
    bn.attrs.floats.insert("epsilon".to_string(), 1e-5);
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), tensor(vec![1.0], vec![1, 1, 1, 1]));
    weights.insert("s".to_string(), tensor(vec![1.0], vec![1]));
    weights.insert("b".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("m".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("v".to_string(), tensor(vec![1.0], vec![1]));

    let outputs = names(&["conv_out", "bn_out"]);
    let result = fusion::fuse_conv_batchnorm(vec![conv, bn], &mut weights, &outputs);
    assert_eq!(result.len(), 2);
    assert!(missing_graph_outputs(&result, &weights, &outputs).is_empty());
}

#[test]
fn test_matmul_add_not_fused_when_matmul_output_is_exported() {
    let nodes = vec![
        make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
        make_node(OpKind::Add, "add", vec!["mm_out", "bias"], vec!["y"]),
    ];
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), tensor(vec![1.0; 4], vec![2, 2]));
    weights.insert("bias".to_string(), tensor(vec![0.5, 0.5], vec![2]));
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![2, 2]);

    let outputs = names(&["mm_out", "y"]);
    let result = fusion::fuse_matmul_add(nodes, &weights, &shapes, &outputs);
    assert_eq!(result.len(), 2);
    assert!(missing_graph_outputs(&result, &weights, &outputs).is_empty());
}

#[test]
fn test_silu_not_fused_when_sigmoid_output_is_exported() {
    let nodes = vec![
        make_node(OpKind::Sigmoid, "sig", vec!["x"], vec!["sig_out"]),
        make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["y"]),
    ];
    let outputs = names(&["sig_out", "y"]);
    let result = fusion::fuse_mul_sigmoid_to_silu(nodes, &outputs);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_transpose_cancel_keeps_exported_output_name() {
    let mut t1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
    t1.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
    let mut t2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
    t2.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);

    let outputs = names(&["t2_out"]);
    let result = fusion::cancel_consecutive_transpose(vec![t1, t2], &outputs);
    assert!(missing_graph_outputs(&result, &HashMap::new(), &outputs).is_empty());
}

/// A three-Transpose chain must not leave a node pointing at a tensor the
/// collapse already deleted.
#[test]
fn test_transpose_chain_collapse_leaves_no_dangling_tensor() {
    let mut t1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
    t1.attrs.int_lists.insert("perm".to_string(), vec![1, 2, 0]);
    let mut t2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
    t2.attrs.int_lists.insert("perm".to_string(), vec![1, 2, 0]);
    let mut t3 = make_node(OpKind::Transpose, "t3", vec!["t2_out"], vec!["y"]);
    t3.attrs.int_lists.insert("perm".to_string(), vec![1, 2, 0]);

    let outputs = names(&["y"]);
    let result = fusion::cancel_consecutive_transpose(vec![t1, t2, t3], &outputs);

    let produced: Vec<&str> = result
        .iter()
        .flat_map(|n| n.outputs.iter())
        .map(String::as_str)
        .collect();
    for node in &result {
        for input in &node.inputs {
            assert!(
                input == "x" || produced.contains(&input.as_str()),
                "node {} consumes '{input}', which nothing produces",
                node.name
            );
        }
    }
    assert!(missing_graph_outputs(&result, &HashMap::new(), &outputs).is_empty());
}

/// Full pipeline: an exported intermediate feature map must survive.
#[test]
fn test_optimize_preserves_every_declared_output() {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["features"]);
    let relu = make_node(OpKind::Relu, "relu", vec!["features"], vec!["logits"]);
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), tensor(vec![1.0], vec![1, 1, 1, 1]));

    let outputs = names(&["features", "logits"]);
    let registry = oxionnx_ops::default_registry();
    let result = optimize(vec![conv, relu], &mut weights, &outputs, &registry);
    assert!(
        missing_graph_outputs(&result, &weights, &outputs).is_empty(),
        "optimizer dropped a declared graph output: {:?}",
        missing_graph_outputs(&result, &weights, &outputs)
    );
}

// ─────────────────────────── a4-7: LayerNorm axis ───────────────────────────

/// Build a LayerNorm pattern over `x` with the given ReduceMean configuration.
fn layer_norm_pattern(configure: impl Fn(&mut Node)) -> (Vec<Node>, HashMap<String, Tensor>) {
    let mut weights = HashMap::new();
    let mut mean = make_node(OpKind::ReduceMean, "mean", vec!["X"], vec!["mean_out"]);
    let mut var_mean = make_node(OpKind::ReduceMean, "var_mean", vec!["sq"], vec!["var"]);
    configure(&mut mean);
    configure(&mut var_mean);

    let sub = make_node(OpKind::Sub, "sub", vec!["X", "mean_out"], vec!["diff"]);
    let pow = make_node(OpKind::Pow, "pow", vec!["diff", "two"], vec!["sq"]);
    weights.insert("two".to_string(), tensor(vec![2.0], vec![1]));
    let add_eps = make_node(OpKind::Add, "add_eps", vec!["var", "eps"], vec!["var_eps"]);
    weights.insert("eps".to_string(), tensor(vec![1e-5], vec![1]));
    let sqrt = make_node(OpKind::Sqrt, "sqrt", vec!["var_eps"], vec!["std"]);
    let div = make_node(OpKind::Div, "div", vec!["diff", "std"], vec!["normalized"]);
    let mul = make_node(OpKind::Mul, "mul", vec!["normalized", "scale"], vec!["y"]);
    weights.insert("scale".to_string(), tensor(vec![1.0; 4], vec![4]));

    (
        vec![mean, sub, pow, var_mean, add_eps, sqrt, div, mul],
        weights,
    )
}

/// opset 18 moves `axes` from an attribute to input 1 — the fusion must read it
/// there instead of silently defaulting to `axis = -1`.
#[test]
fn test_layer_norm_reads_axes_from_input_tensor() {
    let (nodes, mut weights) = layer_norm_pattern(|node| {
        node.inputs.push("axes_const".to_string());
    });
    weights.insert("axes_const".to_string(), tensor(vec![-1.0], vec![1]));

    let shapes = HashMap::new();
    let result = fusion::fuse_layer_norm(nodes, &weights, &shapes, &names(&["y"]));
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::LayerNorm));
    assert_eq!(result[0].attrs.i("axis", 0), -1);
}

/// An opset-18 reduction over a *non-trailing* axis is not a LayerNorm.
#[test]
fn test_layer_norm_rejects_non_trailing_axis() {
    let (nodes, mut weights) = layer_norm_pattern(|node| {
        node.inputs.push("axes_const".to_string());
    });
    weights.insert("axes_const".to_string(), tensor(vec![1.0], vec![1]));
    let mut shapes = HashMap::new();
    shapes.insert("X".to_string(), vec![2, 3, 4, 5]);
    shapes.insert("sq".to_string(), vec![2, 3, 4, 5]);

    let before = nodes.len();
    let result = fusion::fuse_layer_norm(nodes, &weights, &shapes, &names(&["y"]));
    assert_eq!(result.len(), before);
}

/// A contiguous trailing run of two axes yields `axis = -2`.
#[test]
fn test_layer_norm_trailing_two_axes_gives_axis_minus_two() {
    let (nodes, weights) = layer_norm_pattern(|node| {
        node.attrs
            .int_lists
            .insert("axes".to_string(), vec![-2, -1]);
    });
    let shapes = HashMap::new();
    let result = fusion::fuse_layer_norm(nodes, &weights, &shapes, &names(&["y"]));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].attrs.i("axis", 0), -2);
}

/// `keepdims = 0` breaks the broadcast the pattern relies on.
#[test]
fn test_layer_norm_requires_keepdims() {
    let (nodes, weights) = layer_norm_pattern(|node| {
        node.attrs.int_lists.insert("axes".to_string(), vec![-1]);
        node.attrs.ints.insert("keepdims".to_string(), 0);
    });
    let before = nodes.len();
    let result = fusion::fuse_layer_norm(nodes, &weights, &HashMap::new(), &names(&["y"]));
    assert_eq!(result.len(), before);
}

/// Without a scale operand the emitted node would fail at run time
/// (`LayerNormOp` requires input 1), so the pattern is left alone.
#[test]
fn test_layer_norm_without_scale_is_not_fused() {
    let (mut nodes, weights) = layer_norm_pattern(|node| {
        node.attrs.int_lists.insert("axes".to_string(), vec![-1]);
    });
    nodes.pop(); // drop the Mul(scale)
    let before = nodes.len();
    let result = fusion::fuse_layer_norm(nodes, &weights, &HashMap::new(), &names(&["normalized"]));
    assert_eq!(result.len(), before);
}

// ─────────────────── a4-9: collision-free generated names ───────────────────

/// Two *unnamed* Conv+BatchNorm pairs must not overwrite each other's folded
/// weights.
#[test]
fn test_conv_batchnorm_unnamed_nodes_get_distinct_weight_names() {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();
    for (idx, scale) in [2.0f32, 5.0f32].iter().enumerate() {
        let conv_out = format!("conv_out{idx}");
        let bn_out = format!("bn_out{idx}");
        nodes.push(make_node(
            OpKind::Conv,
            "",
            vec!["x", "w"],
            vec![conv_out.as_str()],
        ));
        let mut bn = make_node(
            OpKind::BatchNorm,
            "",
            vec![conv_out.as_str(), "s", "b", "m", "v"],
            vec![bn_out.as_str()],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 0.0);
        nodes.push(bn);
        let _ = scale;
    }
    weights.insert("w".to_string(), tensor(vec![3.0], vec![1, 1, 1, 1]));
    weights.insert("s".to_string(), tensor(vec![2.0], vec![1]));
    weights.insert("b".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("m".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("v".to_string(), tensor(vec![1.0], vec![1]));

    let outputs = names(&["bn_out0", "bn_out1"]);
    let result = fusion::fuse_conv_batchnorm(nodes, &mut weights, &outputs);
    assert_eq!(result.len(), 2);

    let w0 = &result[0].inputs[1];
    let w1 = &result[1].inputs[1];
    assert_ne!(w0, w1, "two unnamed folds collided on one weight name");
    assert!(weights.contains_key(w0));
    assert!(weights.contains_key(w1));
    // factor = scale / sqrt(var + eps) = 2 / 1 = 2 ⇒ folded kernel = 3 * 2 = 6.
    for name in [w0, w1] {
        let folded = weights.get(name).expect("folded weight");
        assert!((folded.data[0] - 6.0).abs() < 1e-6);
    }
}

/// Same for the standalone BatchNorm fold's `factor` / `shift` constants.
#[test]
fn test_standalone_batchnorm_unnamed_nodes_get_distinct_constant_names() {
    let mut nodes = Vec::new();
    for idx in 0..2 {
        let out = format!("bn_out{idx}");
        let mut bn = make_node(
            OpKind::BatchNorm,
            "",
            vec!["x", "s", "b", "m", "v"],
            vec![out.as_str()],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 0.0);
        nodes.push(bn);
    }
    let mut weights = HashMap::new();
    weights.insert("s".to_string(), tensor(vec![2.0], vec![1]));
    weights.insert("b".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("m".to_string(), tensor(vec![0.0], vec![1]));
    weights.insert("v".to_string(), tensor(vec![1.0], vec![1]));
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 1, 2, 2]);

    let result = fusion::fold_batch_norm_inference(nodes, &mut weights, &shapes);
    // Two BatchNorms → two Mul + two Add.
    assert_eq!(result.len(), 4);
    let factor0 = &result[0].inputs[1];
    let factor1 = &result[2].inputs[1];
    assert_ne!(factor0, factor1);
    assert!(weights.contains_key(factor0));
    assert!(weights.contains_key(factor1));
}

// ───────────── a4-14: Transpose/Reshape layout preconditions ────────────────

/// `Transpose(NCHW → NHWC) → Reshape([N, -1])` genuinely moves data: the
/// Transpose must stay.
#[test]
fn test_simplify_transpose_reshape_keeps_nchw_to_nhwc() {
    let mut transpose = make_node(OpKind::Transpose, "t", vec!["x"], vec!["t_out"]);
    transpose
        .attrs
        .int_lists
        .insert("perm".to_string(), vec![0, 2, 3, 1]);
    let reshape = make_node(OpKind::Reshape, "r", vec!["t_out", "shape"], vec!["y"]);
    let mut weights = HashMap::new();
    weights.insert("shape".to_string(), tensor(vec![2.0, -1.0], vec![2]));
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![2, 3, 4, 5]);

    let result = fusion::simplify_transpose_reshape(
        vec![transpose, reshape],
        &weights,
        &shapes,
        &names(&["y"]),
    );
    assert_eq!(result.len(), 2);
    assert!(matches!(result[0].op, OpKind::Transpose));
}

/// A perm that only moves extent-1 axes leaves the buffer untouched, so the
/// Transpose can go.
#[test]
fn test_simplify_transpose_reshape_drops_extent_one_permutation() {
    let mut transpose = make_node(OpKind::Transpose, "t", vec!["x"], vec!["t_out"]);
    transpose
        .attrs
        .int_lists
        .insert("perm".to_string(), vec![0, 2, 1]);
    let reshape = make_node(OpKind::Reshape, "r", vec!["t_out", "shape"], vec!["y"]);
    let mut weights = HashMap::new();
    weights.insert("shape".to_string(), tensor(vec![6.0], vec![1]));
    let mut shapes = HashMap::new();
    // [2, 1, 3] with perm [0, 2, 1] → [2, 3, 1]: only the extent-1 axis moves.
    shapes.insert("x".to_string(), vec![2, 1, 3]);

    let result = fusion::simplify_transpose_reshape(
        vec![transpose, reshape],
        &weights,
        &shapes,
        &names(&["y"]),
    );
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Reshape));
    assert_eq!(result[0].inputs[0], "x");
}

/// A `0` in the target shape refers to the tensor being reshaped, so the
/// rewrite cannot re-parent the Reshape.
#[test]
fn test_simplify_transpose_reshape_respects_zero_dim_copy() {
    let mut transpose = make_node(OpKind::Transpose, "t", vec!["x"], vec!["t_out"]);
    transpose
        .attrs
        .int_lists
        .insert("perm".to_string(), vec![0, 2, 1]);
    let reshape = make_node(OpKind::Reshape, "r", vec!["t_out", "shape"], vec!["y"]);
    let mut weights = HashMap::new();
    weights.insert("shape".to_string(), tensor(vec![0.0, -1.0], vec![2]));
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![2, 1, 3]);

    let result = fusion::simplify_transpose_reshape(
        vec![transpose, reshape],
        &weights,
        &shapes,
        &names(&["y"]),
    );
    assert_eq!(result.len(), 2);
}

/// `Reshape(x, [6]) → Reshape(_, [6])` does **not** restore `x`'s `[2, 3]`
/// shape: the pair must collapse into one Reshape, not disappear.
#[test]
fn test_cancel_consecutive_reshape_same_target_is_not_identity() {
    let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "s"], vec!["r1_out"]);
    let r2 = make_node(OpKind::Reshape, "r2", vec!["r1_out", "s"], vec!["y"]);
    let relu = make_node(OpKind::Relu, "relu", vec!["y"], vec!["z"]);
    let mut weights = HashMap::new();
    weights.insert("s".to_string(), tensor(vec![6.0], vec![1]));
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![2, 3]);

    let result =
        fusion::cancel_consecutive_reshape(vec![r1, r2, relu], &weights, &shapes, &names(&["z"]));
    assert_eq!(result.len(), 2);
    assert!(matches!(result[0].op, OpKind::Reshape));
    assert_eq!(result[0].inputs[0], "x");
    assert_eq!(result[0].outputs[0], "y");
    assert_eq!(result[1].inputs[0], "y");
}

/// When the second Reshape provably restores the original shape, both go.
#[test]
fn test_cancel_consecutive_reshape_removes_true_round_trip() {
    let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "s1"], vec!["r1_out"]);
    let r2 = make_node(OpKind::Reshape, "r2", vec!["r1_out", "s2"], vec!["r2_out"]);
    let relu = make_node(OpKind::Relu, "relu", vec!["r2_out"], vec!["z"]);
    let mut weights = HashMap::new();
    weights.insert("s1".to_string(), tensor(vec![6.0], vec![1]));
    weights.insert("s2".to_string(), tensor(vec![2.0, 3.0], vec![2]));
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![2, 3]);

    let result =
        fusion::cancel_consecutive_reshape(vec![r1, r2, relu], &weights, &shapes, &names(&["z"]));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "relu");
    assert_eq!(result[0].inputs[0], "x");
}

// ───────────────────────── a4-21: OptLevel gating ───────────────────────────

fn gating_graph() -> (Vec<Node>, HashMap<String, Tensor>, Vec<String>) {
    let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
    let relu = make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["y"]);
    let fold_add = make_node(OpKind::Add, "fold", vec!["c1", "c2"], vec!["folded"]);
    let use_folded = make_node(OpKind::Mul, "use", vec!["y", "folded"], vec!["out"]);
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), tensor(vec![1.0], vec![1, 1, 1, 1]));
    weights.insert("c1".to_string(), tensor(vec![1.0], vec![1]));
    weights.insert("c2".to_string(), tensor(vec![2.0], vec![1]));
    (
        vec![conv, relu, fold_add, use_folded],
        weights,
        names(&["out"]),
    )
}

#[test]
fn test_opt_level_basic_runs_dead_code_only() {
    let (mut nodes, mut weights, outputs) = gating_graph();
    nodes.push(make_node(OpKind::Relu, "dead", vec!["x"], vec!["dead_out"]));
    let registry = oxionnx_ops::default_registry();
    let result = optimize_with_level(nodes, &mut weights, &outputs, &registry, PassLevel::Basic);
    // Dead node dropped, but no fusion and no constant folding.
    assert_eq!(result.len(), 4);
    assert!(result.iter().any(|n| n.name == "conv"));
    assert!(result.iter().any(|n| n.name == "relu"));
    assert!(result.iter().any(|n| n.name == "fold"));
    assert!(!weights.contains_key("folded"));
}

#[test]
fn test_opt_level_extended_fuses_but_does_not_constant_fold() {
    let (nodes, mut weights, outputs) = gating_graph();
    let registry = oxionnx_ops::default_registry();
    let result = optimize_with_level(
        nodes,
        &mut weights,
        &outputs,
        &registry,
        PassLevel::Extended,
    );
    // Conv+Relu fused → 3 nodes; the foldable Add is still there.
    assert_eq!(result.len(), 3);
    assert!(result
        .iter()
        .any(|n| matches!(n.op, OpKind::Conv) && n.attrs.s("activation") == "relu"));
    assert!(result.iter().any(|n| matches!(n.op, OpKind::Add)));
    assert!(!weights.contains_key("folded"));
}

#[test]
fn test_opt_level_all_folds_and_fuses() {
    let (nodes, mut weights, outputs) = gating_graph();
    let registry = oxionnx_ops::default_registry();
    let result = optimize_with_level(nodes, &mut weights, &outputs, &registry, PassLevel::All);
    assert_eq!(result.len(), 2);
    assert!(weights.contains_key("folded"));
}

// ────────────────── a4-22: the formerly dead passes are wired ───────────────

#[test]
fn test_optimize_eliminates_inference_dropout() {
    let softmax = make_node(OpKind::Softmax, "softmax", vec!["x"], vec!["sm"]);
    let dropout = make_node(OpKind::Dropout, "drop", vec!["sm"], vec!["dropped"]);
    let matmul = make_node(OpKind::MatMul, "mm", vec!["dropped", "v"], vec!["out"]);
    let mut weights = HashMap::new();
    let outputs = names(&["out"]);
    let registry = oxionnx_ops::default_registry();
    let result = optimize(
        vec![softmax, dropout, matmul],
        &mut weights,
        &outputs,
        &registry,
    );
    assert_eq!(result.len(), 2);
    assert!(!result.iter().any(|n| matches!(n.op, OpKind::Dropout)));
    assert!(missing_graph_outputs(&result, &weights, &outputs).is_empty());
}

#[test]
fn test_dropout_with_runtime_training_mode_is_kept() {
    let dropout = make_node(
        OpKind::Dropout,
        "drop",
        vec!["x", "ratio", "mode"],
        vec!["dropped"],
    );
    let relu = make_node(OpKind::Relu, "relu", vec!["dropped"], vec!["out"]);
    let weights = HashMap::new();
    let result =
        fusion::eliminate_dropout_inference(vec![dropout, relu], &weights, &names(&["out"]));
    assert_eq!(result.len(), 2);
}

#[test]
fn test_dropout_output_that_is_exported_is_kept() {
    let dropout = make_node(OpKind::Dropout, "drop", vec!["x"], vec!["dropped"]);
    let relu = make_node(OpKind::Relu, "relu", vec!["dropped"], vec!["out"]);
    let weights = HashMap::new();
    let result = fusion::eliminate_dropout_inference(
        vec![dropout, relu],
        &weights,
        &names(&["dropped", "out"]),
    );
    assert_eq!(result.len(), 2);
}

#[test]
fn test_gather_composition_normalizes_negative_indices() {
    let mut g1 = make_node(OpKind::Gather, "g1", vec!["data", "idx1"], vec!["g1_out"]);
    g1.attrs.ints.insert("axis".to_string(), 0);
    let mut g2 = make_node(OpKind::Gather, "g2", vec!["g1_out", "idx2"], vec!["y"]);
    g2.attrs.ints.insert("axis".to_string(), 0);
    let mut weights = HashMap::new();
    weights.insert("idx1".to_string(), tensor(vec![2.0, 0.0, 1.0], vec![3]));
    // -1 refers to the last element of idx1 (value 1), 0 to the first (value 2).
    weights.insert("idx2".to_string(), tensor(vec![-1.0, 0.0], vec![2]));

    let result = fusion::fuse_gather_composition(vec![g1, g2], &mut weights, &names(&["y"]));
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Gather));
    assert_eq!(result[0].inputs[0], "data");
    let composed = weights
        .get(&result[0].inputs[1])
        .expect("composed indices are an initializer");
    assert_eq!(composed.data, vec![1.0, 2.0]);
}

#[test]
fn test_gather_composition_requires_1d_inner_indices() {
    let mut g1 = make_node(OpKind::Gather, "g1", vec!["data", "idx1"], vec!["g1_out"]);
    g1.attrs.ints.insert("axis".to_string(), 0);
    let mut g2 = make_node(OpKind::Gather, "g2", vec!["g1_out", "idx2"], vec!["y"]);
    g2.attrs.ints.insert("axis".to_string(), 0);
    let mut weights = HashMap::new();
    weights.insert(
        "idx1".to_string(),
        tensor(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]),
    );
    weights.insert("idx2".to_string(), tensor(vec![0.0], vec![1]));

    let result = fusion::fuse_gather_composition(vec![g1, g2], &mut weights, &names(&["y"]));
    assert_eq!(result.len(), 2);
}

struct StubConvAddRelu;
impl Operator for StubConvAddRelu {
    fn op_type(&self) -> &str {
        "ConvAddRelu"
    }
    fn execute(&self, _ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Err(OnnxError::Internal("stub".into()))
    }
}

/// `ConvAddRelu` has no kernel in the default registry, so the ResNet fusion
/// must stay off there — and switch on for a registry that provides one.
#[test]
fn test_conv_add_relu_fusion_follows_registry_support() {
    let build = || {
        vec![
            make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]),
            make_node(OpKind::Add, "add", vec!["conv_out", "res"], vec!["add_out"]),
            make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["out"]),
        ]
    };
    let outputs = names(&["out"]);

    let mut weights = HashMap::new();
    let default_registry = oxionnx_ops::default_registry();
    let unfused = optimize(build(), &mut weights, &outputs, &default_registry);
    assert!(!unfused.iter().any(|n| matches!(n.op, OpKind::ConvAddRelu)));

    let mut registry = OperatorRegistry::new();
    registry.register(Box::new(StubConvAddRelu));
    let mut weights = HashMap::new();
    let fused = optimize(build(), &mut weights, &outputs, &registry);
    assert!(fused.iter().any(|n| matches!(n.op, OpKind::ConvAddRelu)));
}

// ─────────────────────── a4-23: constant folding limits ─────────────────────

#[test]
fn test_constant_fold_keeps_nodes_producing_graph_outputs() {
    let add = make_node(OpKind::Add, "add", vec!["a", "b"], vec!["sum"]);
    let mut weights = HashMap::new();
    weights.insert("a".to_string(), tensor(vec![1.0], vec![1]));
    weights.insert("b".to_string(), tensor(vec![2.0], vec![1]));
    let registry = oxionnx_ops::default_registry();
    let outputs = names(&["sum"]);

    let result = crate::optimizer::constant_fold::constant_fold(
        vec![add],
        &mut weights,
        &registry,
        &outputs,
    );
    assert_eq!(result.len(), 1);
    assert!(missing_graph_outputs(&result, &weights, &outputs).is_empty());
}

#[test]
fn test_constant_fold_skips_non_deterministic_ops() {
    let bernoulli = make_node(OpKind::Bernoulli, "b", vec!["p"], vec!["sample"]);
    let relu = make_node(OpKind::Relu, "relu", vec!["sample"], vec!["out"]);
    let mut weights = HashMap::new();
    weights.insert("p".to_string(), tensor(vec![0.5, 0.5], vec![2]));
    let registry = oxionnx_ops::default_registry();

    let result = crate::optimizer::constant_fold::constant_fold(
        vec![bernoulli, relu],
        &mut weights,
        &registry,
        &names(&["out"]),
    );
    assert!(result.iter().any(|n| matches!(n.op, OpKind::Bernoulli)));
    assert!(!weights.contains_key("sample"));
}

// ─────────────────────── a4-24: the 2-D Gemm contract ───────────────────────

#[test]
fn test_matmul_add_fuses_only_for_rank_2_activations() {
    let build = || {
        vec![
            make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
            make_node(OpKind::Add, "add", vec!["mm_out", "bias"], vec!["y"]),
        ]
    };
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), tensor(vec![1.0; 4], vec![2, 2]));
    weights.insert("bias".to_string(), tensor(vec![0.5, 0.5], vec![2]));
    let outputs = names(&["y"]);

    // rank 2 → fused
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![3, 2]);
    let fused = fusion::fuse_matmul_add(build(), &weights, &shapes, &outputs);
    assert_eq!(fused.len(), 1);
    assert!(matches!(fused[0].op, OpKind::Gemm));

    // rank 3 (transformer projection) → not fused
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![1, 128, 2]);
    let unfused = fusion::fuse_matmul_add(build(), &weights, &shapes, &outputs);
    assert_eq!(unfused.len(), 2);

    // unknown rank → not fused
    let unfused = fusion::fuse_matmul_add(build(), &weights, &HashMap::new(), &outputs);
    assert_eq!(unfused.len(), 2);
}

#[test]
fn test_matmul_transpose_fuses_only_the_2d_case() {
    let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
    let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["y"]);
    transpose
        .attrs
        .int_lists
        .insert("perm".to_string(), vec![0, 2, 1]);
    let result = fusion::fuse_matmul_transpose(vec![matmul, transpose], &names(&["y"]));
    assert_eq!(result.len(), 2);
}

#[test]
fn test_add_matmul_to_gemm_requires_rank_2_input() {
    let build = || {
        vec![
            make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]),
            make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["y"]),
        ]
    };
    let mut weights = HashMap::new();
    weights.insert("bias".to_string(), tensor(vec![1.0, 2.0], vec![2]));
    weights.insert(
        "w".to_string(),
        tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
    );
    let outputs = names(&["y"]);

    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), vec![4, 2]);
    let fused = fusion::fuse_add_matmul_to_gemm(build(), &mut weights, &shapes, &outputs);
    assert_eq!(fused.len(), 1);
    assert!(matches!(fused[0].op, OpKind::Gemm));

    let unfused = fusion::fuse_add_matmul_to_gemm(build(), &mut weights, &HashMap::new(), &outputs);
    assert_eq!(unfused.len(), 2);
}

// ───────────────────── malformed models never panic ─────────────────────────

/// Nodes with missing operands, empty names and out-of-range attributes must be
/// passed through, never indexed into blindly.
#[test]
fn test_passes_tolerate_malformed_nodes() {
    let mut weird = vec![
        make_node(OpKind::MatMul, "", vec!["only_one"], vec!["a"]),
        make_node(OpKind::Add, "", vec!["a"], vec!["b"]),
        make_node(OpKind::Conv, "", vec![], vec!["c"]),
        make_node(OpKind::BatchNorm, "", vec!["c"], vec!["d"]),
        make_node(OpKind::Clip, "", vec![], vec![]),
        make_node(OpKind::Relu, "", vec!["d"], vec![]),
    ];
    let mut t = make_node(OpKind::Transpose, "", vec!["b"], vec!["e"]);
    t.attrs
        .int_lists
        .insert("perm".to_string(), vec![9, -3, 100]);
    weird.push(t);
    weird.push(make_node(OpKind::Reshape, "", vec!["e"], vec!["f"]));
    weird.push(make_node(OpKind::Gather, "", vec!["f"], vec!["g"]));
    let mut dropout = make_node(OpKind::Dropout, "", vec!["g"], vec!["h"]);
    dropout.attrs = Attributes::default();
    weird.push(dropout);

    let mut weights = HashMap::new();
    weights.insert("only_one".to_string(), tensor(vec![1.0], vec![1]));
    let outputs = names(&["h"]);
    let registry = oxionnx_ops::default_registry();
    let result = optimize(weird, &mut weights, &outputs, &registry);
    assert!(!result.is_empty());
}
