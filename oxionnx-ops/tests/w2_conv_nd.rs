//! Wave-2 rank-generic spatial-operator tests: `Conv1D`/`Conv3D`,
//! `ConvTranspose1D`/`ConvTranspose3D` and `MaxPool`/`AveragePool` at
//! spatial rank 1 and 3, each with `auto_pad`, `ceil_mode`, `dilations`,
//! `group`, `output_padding` and `output_shape` exercised.
//!
//! Reference values come from `onnx.reference.ReferenceEvaluator` (opset 21,
//! the ONNX specification's own implementation) for Conv/MaxPool/AveragePool,
//! and from a direct NumPy scatter-accumulate for ConvTranspose that was
//! asserted equal to `ReferenceEvaluator` on every configuration that
//! implementation can evaluate (it rejects `group != 1` and an explicit
//! `output_shape`). Every input is exactly representable in binary32, so the
//! Rust side reproduces the identical bits from the same closed form.
//!
//! Generator: scratchpad/gen_rust.py

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    OnnxError, Tensor,
};
use oxionnx_ops::registry::conv_ops::{AveragePoolOp, ConvOp, ConvTransposeOp, MaxPoolOp};

// ── Test infrastructure ──────────────────────────────────────────────────────

fn node_with(op: OpKind, outputs: &[&str]) -> Node {
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

fn ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn ints(node: &mut Node, name: &str, values: &[i64]) {
    node.attrs.int_lists.insert(name.into(), values.to_vec());
}

fn int(node: &mut Node, name: &str, value: i64) {
    node.attrs.ints.insert(name.into(), value);
}

fn string(node: &mut Node, name: &str, value: &str) {
    node.attrs.strings.insert(name.into(), value.into());
}

fn assert_close(got: &Tensor, want_shape: &[usize], want: &[f32], label: &str) {
    assert_eq!(got.shape, want_shape, "{label}: shape");
    assert_eq!(got.data.len(), want.len(), "{label}: element count");
    for (i, (&g, &e)) in got.data.iter().zip(want.iter()).enumerate() {
        let tol = 1e-4_f32 * e.abs().max(1.0);
        assert!((g - e).abs() <= tol, "{label}[{i}]: got {g}, expected {e}");
    }
}

/// Run an operator through both `execute` and `execute_into_slots` and assert
/// the two agree, returning the `execute` result.
fn run_both(op: &dyn Operator, node: &Node, inputs: Vec<Option<&Tensor>>) -> Vec<Tensor> {
    let context = ctx(node, inputs.clone());
    let direct = op.execute(&context).expect("execute must succeed");

    if op.supports_output_slots() {
        let mut slots: Vec<Tensor> = node
            .outputs
            .iter()
            .map(|_| Tensor::new(Vec::new(), vec![0]))
            .collect();
        let slot_ctx = ctx(node, inputs);
        op.execute_into_slots(&slot_ctx, &mut slots)
            .expect("execute_into_slots must succeed");
        for (i, expected) in direct.iter().enumerate() {
            assert_eq!(slots[i].shape, expected.shape, "slot[{i}] shape parity");
            assert_eq!(slots[i].data, expected.data, "slot[{i}] data parity");
        }
    }
    direct
}

// ── Conv1D: [N, C, W] ────────────────────────────────────────────────────────

#[test]
fn conv1d_basic_with_bias() {
    // Conv1D [1,2,9] * [3,2,3], no padding, stride 1.
    let x = Tensor::new(
        (0..18).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 9],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![3, 2, 3],
    );
    let bias = Tensor::new(
        (0..3).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect(),
        vec![3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight), Some(&bias)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        0.125, -0.125, -0.375, 1.125, -0.875, -2.0,
        -1.375, -0.125, -0.25, -0.375, 0.375, 2.0,
        -0.75, -0.875, 0.25, 0.25, 0.25, 0.25,
        1.125, 1.125, 0.25,
    ];
    assert_close(&got[0], &[1, 3, 7], &want0, "conv1d_basic_with_bias[0]");
}

#[test]
fn conv1d_stride2_pads() {
    // Conv1D with stride 2 and asymmetric explicit pads [2, 1].
    let x = Tensor::new(
        (0..18).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 9],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![3, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "pads", &[2, 1]);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        0.25, 0.625, 0.125, -0.375, -0.875, -0.25,
        -0.125, -0.375, 2.0, -0.875, -0.75, -0.25,
        -0.25, 0.625, -0.25,
    ];
    assert_close(&got[0], &[1, 3, 5], &want0, "conv1d_stride2_pads[0]");
}

#[test]
fn conv1d_dilation3() {
    // Conv1D with dilation 3 (dilated/TCN convolution).
    let x = Tensor::new(
        (0..18).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 9],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![3, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "dilations", &[3]);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        1.625, 1.375, -0.625, 0.25, -0.75, 0.875,
        0.75, -1.0, -0.125,
    ];
    assert_close(&got[0], &[1, 3, 3], &want0, "conv1d_dilation3[0]");
}

#[test]
fn conv1d_auto_pad_same_upper() {
    // Conv1D auto_pad=SAME_UPPER, stride 2 -> out = ceil(9/2) = 5.
    let x = Tensor::new(
        (0..18).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 9],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![3, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        0.125, 0.375, 1.625, -1.5, 0.875, 0.0,
        -0.25, 0.375, -0.75, 0.125, -0.75, -0.25,
        -0.25, 0.625, 0.0,
    ];
    assert_close(&got[0], &[1, 3, 5], &want0, "conv1d_auto_pad_same_upper[0]");
}

#[test]
fn conv1d_auto_pad_same_lower() {
    // Conv1D auto_pad=SAME_LOWER: the odd pad pixel goes to the front.
    let x = Tensor::new(
        (0..18).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 9],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![3, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "dilations", &[2]);
    string(&mut node, "auto_pad", "SAME_LOWER");
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.125, 0.25, 1.5, -0.75, -0.75, 0.0,
        -0.375, 0.25, 0.875, -0.875, -0.5, 0.25,
        0.25, -0.625, 0.25,
    ];
    assert_close(&got[0], &[1, 3, 5], &want0, "conv1d_auto_pad_same_lower[0]");
}

#[test]
fn conv1d_auto_pad_valid() {
    // Conv1D auto_pad=VALID: no padding at all, out = 9 - 3 + 1 = 7.
    // (The spec forbids combining auto_pad with pads, so that combination
    // is covered by the rank-2 test in w1_conv_pool.rs, not here.)
    let x = Tensor::new(
        (0..18).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 9],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![3, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    string(&mut node, "auto_pad", "VALID");
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        0.625, 0.125, -0.375, -0.875, -0.125, -0.375,
        2.0, -0.875, -0.25, -0.25, 0.625, -0.25,
    ];
    assert_close(&got[0], &[1, 3, 4], &want0, "conv1d_auto_pad_valid[0]");
}

#[test]
fn conv1d_group2() {
    // Grouped Conv1D: [1,4,7] with group=2 -> weight [4,2,3].
    let x = Tensor::new(
        (0..28).map(|i| ((i % 6) as f32 - 2.0) * 0.5).collect(),
        vec![1, 4, 7],
    );
    let weight = Tensor::new(
        (0..24).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![4, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "pads", &[1, 1]);
    int(&mut node, "group", 2);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        0.5, 1.25, 0.875, 0.5, -1.375, -1.75,
        -0.75, 1.0, -0.125, -0.375, -0.625, -1.625,
        -0.375, 2.125, -0.5, -0.875, -1.0, 1.875,
        1.0, -0.625, -0.375, -0.125, -0.25, 0.5,
        -0.25, 0.5, -0.25, -0.375,
    ];
    assert_close(&got[0], &[1, 4, 7], &want0, "conv1d_group2[0]");
}

#[test]
fn conv1d_batch2_stride_dilation() {
    // Batched Conv1D (N=2) with stride 2, dilation 2 and padding.
    let x = Tensor::new(
        (0..66).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect(),
        vec![2, 3, 11],
    );
    let weight = Tensor::new(
        (0..36).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect(),
        vec![4, 3, 3],
    );
    let bias = Tensor::new(
        (0..4).map(|i| ((i % 4) as f32 - 2.0) * 0.5).collect(),
        vec![4],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "pads", &[2, 2]);
    ints(&mut node, "dilations", &[2]);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight), Some(&bias)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        2.75, 5.0, -0.625, -5.125, -4.0, -1.0,
        -0.75, -2.75, -5.125, -4.125, 4.75, 1.625,
        -1.5, -5.0, -4.125, 2.375, 6.625, -1.25,
        -2.25, -1.75, 2.375, 2.0, 1.625, -1.375,
        -5.125, -4.0, 0.5, 0.5, -4.0, -4.375,
        -2.625, 4.75, 2.375, -4.5, -3.5, 2.375,
        2.625, 6.625, -2.625, -4.0, 2.5, 7.75,
        3.75, 1.625, -2.125, 2.0, 1.625, 3.5,
    ];
    assert_close(
        &got[0],
        &[2, 4, 6],
        &want0,
        "conv1d_batch2_stride_dilation[0]",
    );
}

// ── Conv3D: [N, C, D, H, W] ──────────────────────────────────────────────────

#[test]
fn conv3d_basic_with_bias() {
    // Conv3D [1,2,4,4,5] * [3,2,2,2,3], no padding.
    let x = Tensor::new(
        (0..160).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 4, 5],
    );
    let weight = Tensor::new(
        (0..72).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![3, 2, 2, 2, 3],
    );
    let bias = Tensor::new(
        (0..3).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect(),
        vec![3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight), Some(&bias)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        5.875, 6.75, 1.125, 3.75, -1.875, -4.25,
        -4.875, -4.0, 1.75, -4.25, -5.0, -0.875,
        1.75, 5.875, 6.75, -2.0, 3.75, -1.875,
        6.75, 1.125, -2.875, -1.875, -4.25, -5.0,
        -4.0, 1.75, 5.875, -1.75, -4.625, -2.625,
        -3.125, 0.5, 5.75, 0.375, -0.875, 2.75,
        5.75, 1.25, 1.625, 2.75, -1.75, -4.625,
        -0.25, -3.125, 0.5, -4.625, -2.625, 1.0,
        0.5, 5.75, 1.25, -0.875, 2.75, -1.75,
        -0.625, 2.375, -1.125, 1.375, 1.125, 0.875,
        -3.125, 3.125, 2.875, 0.875, 2.25, -2.875,
        2.875, -0.625, 2.375, 3.25, 1.375, 1.125,
        2.375, -1.125, -3.0, 1.125, 0.875, 2.25,
        3.125, 2.875, -0.625,
    ];
    assert_close(
        &got[0],
        &[1, 3, 3, 3, 3],
        &want0,
        "conv3d_basic_with_bias[0]",
    );
}

#[test]
fn conv3d_stride_and_asymmetric_pads() {
    // Conv3D with per-axis strides and asymmetric pads
    // (ONNX layout [b_d, b_h, b_w, e_d, e_h, e_w]).
    let x = Tensor::new(
        (0..160).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 4, 5],
    );
    let weight = Tensor::new(
        (0..72).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![3, 2, 2, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    ints(&mut node, "strides", &[2, 1, 2]);
    ints(&mut node, "pads", &[1, 0, 2, 0, 1, 1]);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.25, -2.375, -2.875, 0.375, 1.25, 4.0,
        1.0, -3.25, 2.75, 0.25, -0.375, 0.625,
        0.125, -3.75, -0.375, 2.75, 2.25, 7.25,
        -2.75, -1.5, -1.375, -1.875, 3.125, -3.375,
        0.375, 3.125, 2.875, -0.25, -2.375, -2.625,
        0.75, -1.375, 1.625, -0.25, 1.25, -0.25,
        -2.875, 5.75, 1.625, -1.625, 2.75, -4.625,
        2.875, -0.25, 0.5, 0.5, 0.25, 0.375,
        1.0, -3.625, -3.625, -0.875, 4.5, 1.25,
        0.5, -0.375, -0.375, 1.0, -1.5, -0.25,
        2.875, 0.375, -3.375, -1.625, 2.375, 1.875,
        -2.875, 2.75, 0.625, 2.0, -2.625, -2.0,
    ];
    assert_close(
        &got[0],
        &[1, 3, 2, 4, 3],
        &want0,
        "conv3d_stride_and_asymmetric_pads[0]",
    );
}

#[test]
fn conv3d_dilations() {
    // Conv3D with a different dilation on each spatial axis.
    let x = Tensor::new(
        (0..160).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 4, 5],
    );
    let weight = Tensor::new(
        (0..72).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![3, 2, 2, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    ints(&mut node, "dilations", &[2, 2, 1]);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        5.75, 5.0, 4.25, 2.0, 1.25, 0.5,
        0.5, -5.125, -4.25, -3.25, 5.75, 5.0,
        1.375, -1.5, 0.5, -3.25, -2.875, -2.5,
        -2.5, -0.5, 3.125, -0.625, 1.375, -1.5,
        -4.75, -0.125, 2.875, 3.75, 3.5, 3.25,
        3.25, -0.25, -7.0, 2.0, -4.75, -0.125,
    ];
    assert_close(&got[0], &[1, 3, 2, 2, 3], &want0, "conv3d_dilations[0]");
}

#[test]
fn conv3d_auto_pad_same_upper() {
    // Conv3D auto_pad=SAME_UPPER preserves [D,H,W] at stride 1.
    let x = Tensor::new(
        (0..160).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 4, 5],
    );
    let weight = Tensor::new(
        (0..72).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![3, 2, 2, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        3.875, 6.375, 7.25, 1.625, 0.5, -2.875,
        4.25, -1.375, -3.75, -1.0, 1.75, -4.375,
        -3.5, 2.25, 4.0, 2.375, -0.125, -3.375,
        -1.75, 4.25, -0.375, -3.75, -4.5, -0.375,
        -7.125, 2.625, 2.25, 6.375, 7.25, 4.375,
        -5.75, -1.5, 4.25, -1.375, -0.375, -0.25,
        3.125, -1.75, -3.375, -2.875, 3.5, 7.25,
        1.625, -2.375, -1.75, 3.25, -1.375, -3.75,
        -4.5, -1.625, -1.875, -3.5, 2.25, 6.375,
        5.0, -2.875, -3.375, -1.75, 3.125, -0.25,
        -2.75, -1.625, 2.75, -2.625, -6.75, 1.5,
        2.375, 1.875, 1.375, 0.125, 0.875, 1.5,
        1.0, -1.125, 2.125, 2.875, -2.375, -4.5,
        -1.75, 1.375, 1.75, -1.75, -4.625, -2.625,
        -2.125, -2.875, -3.125, 0.5, 5.75, 0.125,
        3.875, 0.375, -0.875, 2.75, -2.5, -2.5,
        -2.625, -0.125, 2.375, 1.625, -2.125, 5.75,
        1.25, 1.625, 2.0, -1.875, 2.75, -1.75,
        -4.625, -5.5, 3.25, -0.25, -3.125, 0.5,
        4.875, 0.75, 0.25, 1.125, 0.375, -0.5,
        -2.75, -4.625, -2.625, 1.0, 2.875, -5.75,
        0.5, 5.75, 1.25, 0.25, 2.625, -0.875,
        2.75, -1.75, -5.625, -0.875, -0.125, 2.375,
        1.625, 2.25, 1.875, -1.625, -1.125, 1.0,
        2.375, -0.125, 0.875, 1.375, 1.875, 1.375,
        -3.75, -4.75, -2.625, 2.75, -2.875, 1.0,
        1.375, 1.375, -0.25, 0.75, 4.0, -1.125,
        1.875, -1.625, -0.375, 4.125, 0.875, 0.625,
        0.375, 1.25, -7.125, -3.625, 2.625, 2.375,
        -2.0, 3.125, -0.75, -1.25, 4.75, -1.0,
        -0.375, 0.375, 1.75, -3.375, -2.0, -0.25,
        2.375, -1.125, 1.875, 1.25, -1.75, 2.75,
        0.875, 0.625, -0.375, 0.875, -2.625, -4.75,
        -2.0, 3.625, 5.0, 1.875, -1.625, -3.5,
        -0.375, 1.875, 0.625, 0.375, 1.75, 1.25,
        -7.75, 2.625, 2.375, -1.125, 1.25, 0.25,
        -1.25, 4.75, 1.0, -3.125, 2.125, 5.375,
        0.25, -3.25, -1.625, 0.875, -2.375, -2.625,
        -2.875, -3.5, 1.25, 1.25, 4.25, 4.0,
        4.375, -3.5, -4.5, -2.375, 3.0, -1.625,
    ];
    assert_close(
        &got[0],
        &[1, 3, 4, 4, 5],
        &want0,
        "conv3d_auto_pad_same_upper[0]",
    );
}

#[test]
fn conv3d_auto_pad_same_lower_strided() {
    // Conv3D auto_pad=SAME_LOWER with mixed strides.
    let x = Tensor::new(
        (0..160).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 4, 5],
    );
    let weight = Tensor::new(
        (0..72).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![3, 2, 2, 2, 3],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    ints(&mut node, "strides", &[2, 2, 1]);
    string(&mut node, "auto_pad", "SAME_LOWER");
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        3.875, 6.375, 7.25, 1.625, 0.5, 1.75,
        -4.375, -3.5, 2.25, 4.0, 3.5, 7.25,
        1.625, -2.375, -1.75, -1.875, -3.5, 2.25,
        6.375, 5.0, 1.75, -1.75, -4.625, -2.625,
        -2.125, 3.875, 0.375, -0.875, 2.75, -2.5,
        -2.75, -4.625, -2.625, 1.0, 2.875, 2.625,
        -0.875, 2.75, -1.75, -5.625, 4.0, -1.125,
        1.875, -1.625, -0.375, -7.125, -3.625, 2.625,
        2.375, -2.0, 5.0, 1.875, -1.625, -3.5,
        -0.375, -7.75, 2.625, 2.375, -1.125, 1.25,
    ];
    assert_close(
        &got[0],
        &[1, 3, 2, 2, 5],
        &want0,
        "conv3d_auto_pad_same_lower_strided[0]",
    );
}

#[test]
fn conv3d_group2() {
    // Grouped Conv3D: 4 input channels, group=2 -> weight [6,2,2,2,2].
    let x = Tensor::new(
        (0..144).map(|i| ((i % 11) as f32 - 5.0) * 0.5).collect(),
        vec![1, 4, 3, 3, 4],
    );
    let weight = Tensor::new(
        (0..96).map(|i| ((i % 9) as f32 - 4.0) * 0.25).collect(),
        vec![6, 2, 2, 2, 2],
    );
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 2]);
    ints(&mut node, "pads", &[1, 0, 1, 0, 1, 0]);
    int(&mut node, "group", 2);
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.5, 0.5, 1.5, 2.5, -0.25, 0.375,
        1.375, -1.75, 0.375, 1.125, 1.125, -0.25,
        2.75, 3.75, 2.875, -0.75, 0.75, 3.0,
        -2.0, -4.25, 1.875, -2.375, -4.625, -1.375,
        2.25, 2.875, -0.75, -0.25, 0.25, -2.0,
        -4.25, 4.5, 1.125, -4.625, -1.375, 3.25,
        1.0, 2.5, 1.5, 0.5, 0.0, -0.125,
        -1.125, -3.5, 0.875, 1.625, 0.625, 1.0,
        -3.125, -3.25, -3.625, -4.0, 1.375, -3.375,
        -5.125, 1.375, 0.125, 1.0, 2.875, 0.625,
        -3.375, -3.625, -4.0, 2.5, 1.125, -5.125,
        1.375, 6.5, 0.625, 2.875, 0.625, -3.0,
        2.5, 2.25, 0.375, -1.5, 0.25, 1.625,
        -0.25, -0.75, 1.375, -3.5, -4.375, -1.125,
        0.0, -0.125, 0.0, 2.875, -1.375, 0.375,
        1.875, 4.75, 0.625, -3.5, 0.25, 2.625,
        0.0, 0.0, 2.875, 3.0, -1.375, 1.875,
        4.75, -6.125, 0.125, 0.25, 2.625, 0.875,
        1.625, 3.125, 5.125, 3.0, 1.625, -1.25,
        -4.75, -4.125, -0.625, -0.25, 1.25, 2.75,
        6.125, 2.5, 0.375, 3.75, -3.875, -4.625,
        -2.625, -2.0, 0.125, 1.0, 2.0, 3.0,
        -0.5, 0.375, 3.75, -1.125, -3.625, -2.625,
        -2.0, 4.125, 0.875, 2.0, 3.0, -0.125,
        2.125, -0.875, -4.375, -2.375, 0.25, -1.125,
        -0.5, 1.5, -0.375, -0.25, 0.25, 0.75,
        -3.5, -8.75, -3.25, 2.25, -1.5, 0.875,
        5.0, 5.0, 0.125, 0.0, -1.0, -2.0,
        -4.375, -3.25, 2.25, -0.5, 1.75, 5.0,
        5.0, 0.875, -0.125, -1.0, -2.0, -4.375,
        0.375, -2.625, -2.625, 0.125, 0.0, 0.125,
        1.5, 1.5, -0.125, -0.25, -0.75, -1.25,
        -0.75, 2.5, 3.25, -1.5, -3.625, 4.125,
        -2.0, -2.625, 1.25, 0.125, 0.5, 0.875,
        3.0, 3.25, -1.5, -2.125, 1.5, -2.0,
        -2.625, -4.625, 1.125, 0.5, 0.875, 2.625,
    ];
    assert_close(&got[0], &[1, 6, 3, 3, 4], &want0, "conv3d_group2[0]");
}

// ── ConvTranspose1D ──────────────────────────────────────────────────────────

#[test]
fn conv_transpose1d_basic_with_bias() {
    // ConvTranspose1D [1,2,5] with weight [C_in=2, C_out=3, kW=3].
    let x = Tensor::new(
        (0..10).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 5],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![2, 3, 3],
    );
    let bias = Tensor::new(
        (0..3).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect(),
        vec![3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    let got = run_both(
        &ConvTransposeOp,
        &node,
        vec![Some(&x), Some(&weight), Some(&bias)],
    );
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        0.75, 0.625, -1.75, -0.5, -0.125, -0.125,
        -0.375, -0.375, -0.75, 1.0, 0.625, -0.625,
        -0.375, -0.5, 0.375, -0.25, 0.625, 1.125,
        0.75, 0.625, 0.625,
    ];
    assert_close(
        &got[0],
        &[1, 3, 7],
        &want0,
        "conv_transpose1d_basic_with_bias[0]",
    );
}

#[test]
fn conv_transpose1d_stride2_pads() {
    // ConvTranspose1D stride 2 with explicit pads cropping both ends.
    let x = Tensor::new(
        (0..10).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 5],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![2, 3, 3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "pads", &[1, 1]);
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.125, 1.0, -0.5, -0.875, 0.875, -0.125,
        0.5, -0.25, 0.125, -0.5, 1.0, -0.125,
        1.125, -0.625, -0.5, -0.25, -0.375, 0.125,
        -0.25, -0.875, -0.375, 0.625, 0.375, 0.375,
        0.25, 0.125, 0.125,
    ];
    assert_close(
        &got[0],
        &[1, 3, 9],
        &want0,
        "conv_transpose1d_stride2_pads[0]",
    );
}

#[test]
fn conv_transpose1d_output_padding() {
    // ConvTranspose1D output_padding=1 extends the tail by one sample.
    let x = Tensor::new(
        (0..10).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 5],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![2, 3, 3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "output_padding", &[1]);
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        1.25, -0.125, 1.0, -0.5, -0.875, 0.875,
        -0.125, 0.5, -0.25, 0.125, 0.125, 0.0,
        -0.375, -0.5, 1.0, -0.125, 1.125, -0.625,
        -0.5, -0.25, -0.375, 0.125, -0.5, 0.0,
        -0.125, -0.25, -0.875, -0.375, 0.625, 0.375,
        0.375, 0.25, 0.125, 0.125, 0.125, 0.0,
    ];
    assert_close(
        &got[0],
        &[1, 3, 12],
        &want0,
        "conv_transpose1d_output_padding[0]",
    );
}

#[test]
fn conv_transpose1d_auto_pad_same_upper() {
    // ConvTranspose1D auto_pad=SAME_UPPER targets out = in * stride.
    let x = Tensor::new(
        (0..10).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 5],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![2, 3, 3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        1.25, -0.125, 1.0, -0.5, -0.875, 0.875,
        -0.125, 0.5, -0.25, 0.125, -0.375, -0.5,
        1.0, -0.125, 1.125, -0.625, -0.5, -0.25,
        -0.375, 0.125, -0.125, -0.25, -0.875, -0.375,
        0.625, 0.375, 0.375, 0.25, 0.125, 0.125,
    ];
    assert_close(
        &got[0],
        &[1, 3, 10],
        &want0,
        "conv_transpose1d_auto_pad_same_upper[0]",
    );
}

#[test]
fn conv_transpose1d_output_shape() {
    // ConvTranspose1D with an explicit output_shape: the pads are derived
    // from it (natural extent 11, requested 9 -> total crop 2).
    let x = Tensor::new(
        (0..10).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 5],
    );
    let weight = Tensor::new(
        (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect(),
        vec![2, 3, 3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "output_shape", &[9]);
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.125, 1.0, -0.5, -0.875, 0.875, -0.125,
        0.5, -0.25, 0.125, -0.5, 1.0, -0.125,
        1.125, -0.625, -0.5, -0.25, -0.375, 0.125,
        -0.25, -0.875, -0.375, 0.625, 0.375, 0.375,
        0.25, 0.125, 0.125,
    ];
    assert_close(
        &got[0],
        &[1, 3, 9],
        &want0,
        "conv_transpose1d_output_shape[0]",
    );
}

#[test]
fn conv_transpose1d_group2() {
    // Grouped ConvTranspose1D: [1,4,4], group=2, weight [4,1,3].
    let x = Tensor::new(
        (0..16).map(|i| ((i % 6) as f32 - 2.0) * 0.5).collect(),
        vec![1, 4, 4],
    );
    let weight = Tensor::new(
        (0..12).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![4, 1, 3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    int(&mut node, "group", 2);
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        0.75, 0.75, 1.125, 0.625, 0.875, -0.25,
        -0.875, -0.375, -0.375, 0.25, 0.0, 0.25,
        -0.375, 0.375, -0.75, 0.5, -1.125, -0.625,
    ];
    assert_close(&got[0], &[1, 2, 9], &want0, "conv_transpose1d_group2[0]");
}

// ── ConvTranspose3D ──────────────────────────────────────────────────────────

#[test]
fn conv_transpose3d_basic_with_bias() {
    // ConvTranspose3D [1,2,2,3,3] with weight [2,2,2,2,3].
    let x = Tensor::new(
        (0..36).map(|i| ((i % 11) as f32 - 5.0) * 0.5).collect(),
        vec![1, 2, 2, 3, 3],
    );
    let weight = Tensor::new(
        (0..48).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![2, 2, 2, 2, 3],
    );
    let bias = Tensor::new(
        (0..2).map(|i| ((i % 2) as f32 - 1.0) * 0.5).collect(),
        vec![2],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    let got = run_both(
        &ConvTransposeOp,
        &node,
        vec![Some(&x), Some(&weight), Some(&bias)],
    );
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        1.375, 2.5, 3.125, 2.0, 0.875, 1.0,
        0.75, -1.25, -5.75, -3.25, 1.0, -5.875,
        -4.75, 0.375, -0.125, -1.625, 0.0, 1.125,
        1.25, 0.5, -4.125, -3.375, 1.125, 3.125,
        1.875, 0.75, 5.875, 7.875, 1.125, -3.125,
        3.25, 0.375, -9.375, -7.625, -2.5, -3.25,
        -2.125, 0.75, 2.375, 1.25, 1.0, -0.25,
        -5.5, 0.25, 1.0, -2.875, -1.0, 2.625,
        2.5, -0.75, 1.0, 2.875, 1.25, -1.375,
        -3.25, -1.625, -3.375, -1.125, -0.125, 0.75,
        -1.75, -3.875, -1.75, -0.125, 1.125, -0.25,
        2.125, 5.25, 3.375, 1.5, 2.125, 2.875,
        1.875, -2.375, -2.625, -0.625, -1.625, -2.75,
        -1.375, -0.375, -0.375, -0.875, -7.25, -7.625,
        0.25, -3.0, -2.25, 4.0, 5.0, 3.375,
        4.375, 7.25, 9.125, 4.125, 1.75, -0.75,
        -3.0, -6.25, -4.0, -1.75, 0.5, 1.25,
        1.125, 0.0, -2.125, -3.125, -6.5, -5.5,
        -2.25, -0.25, 3.25, 5.0, 5.75, 4.0,
        2.25, 0.375, -0.375, -2.0, -1.625, -0.875,
    ];
    assert_close(
        &got[0],
        &[1, 2, 3, 4, 5],
        &want0,
        "conv_transpose3d_basic_with_bias[0]",
    );
}

#[test]
fn conv_transpose3d_stride_and_pads() {
    // ConvTranspose3D with per-axis strides and asymmetric cropping.
    let x = Tensor::new(
        (0..36).map(|i| ((i % 11) as f32 - 5.0) * 0.5).collect(),
        vec![1, 2, 2, 3, 3],
    );
    let weight = Tensor::new(
        (0..48).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![2, 2, 2, 2, 3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    ints(&mut node, "strides", &[2, 2, 1]);
    ints(&mut node, "pads", &[0, 1, 1, 1, 1, 0]);
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.25, -1.875, -3.625, -1.75, 1.5, 1.125,
        -1.625, -1.0, -4.0, -1.5, 2.5, 1.0,
        -1.375, -2.75, -1.625, -0.625, 2.0, 1.125,
        -0.125, -1.875, 1.0, 2.0, -0.375, -0.5,
        0.75, -5.0, 0.25, 1.5, 0.625, -0.125,
        -1.875, -0.875, 0.875, 2.0, -0.375, -1.75,
        2.5, 3.25, 2.5, 1.5, -0.125, -1.75,
        -3.875, -1.75, -0.375, -2.0, -1.625, -0.875,
        2.5, 3.25, 2.5, 1.5, -0.375, 2.0,
        0.875, 0.0, 1.125, 0.25, -2.875, -1.5,
        1.75, 1.625, 0.5, -1.125, 3.0, 3.625,
        2.5, 1.375, 0.0, 1.125, 1.25, 0.5,
        1.5, 1.125, -1.625, -1.0, 2.0, 2.5,
        2.0, 1.25, -1.625, 1.125, 1.5, 0.75,
        -3.625, -1.875, -0.25, 0.75, 2.5, 3.625,
        3.0, 1.875, 1.25, 1.875, 0.75, -0.375,
    ];
    assert_close(
        &got[0],
        &[1, 2, 3, 4, 4],
        &want0,
        "conv_transpose3d_stride_and_pads[0]",
    );
}

#[test]
fn conv_transpose3d_dilations() {
    // ConvTranspose3D with a different dilation per spatial axis.
    let x = Tensor::new(
        (0..36).map(|i| ((i % 11) as f32 - 5.0) * 0.5).collect(),
        vec![1, 2, 2, 3, 3],
    );
    let weight = Tensor::new(
        (0..48).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![2, 2, 2, 2, 3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    ints(&mut node, "dilations", &[2, 1, 2]);
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        1.875, 1.5, 2.625, 1.375, 2.375, 1.25,
        1.375, 1.5, 1.5, 1.25, -2.0, -2.625,
        -2.875, -2.75, 1.5, -2.625, -5.375, 1.0,
        -2.0, 0.25, 0.375, -1.125, -0.75, 0.875,
        1.0, 1.75, 1.0, 1.0, -1.5, -1.875,
        0.875, -1.125, 1.0, -0.375, 1.125, 1.5,
        1.5, 3.375, 1.5, 2.0, 2.375, -0.25,
        1.5, 1.5, -0.5, -2.375, -5.625, -2.75,
        -2.625, -1.875, -1.5, 0.625, 1.5, 2.25,
        1.0, 1.0, -2.125, -1.875, 0.25, 1.5,
        2.625, 1.375, 1.25, -0.25, 1.5, 3.375,
        1.5, 1.25, -2.0, -2.375, 2.25, -0.125,
        1.75, -2.625, -5.375, 1.0, 0.625, -0.875,
        -0.75, -1.75, -0.75, 0.875, 1.0, 0.75,
        1.5, 1.75, -3.625, -1.875, 0.875, -1.125,
        1.5, -2.375, -2.0, 1.25, 1.5, 3.375,
        1.5, -0.25, 1.5, 1.875, 3.75, 1.5,
        -0.5, -2.375, -2.75, -1.125, -1.0, -2.75,
        -1.5, 0.625, 1.5, 1.25, -1.75, -1.75,
        -3.875, -1.875, 0.25, 1.5, 1.125, -0.25,
        2.375, 2.0, 1.5, 3.375, 1.5, 1.5,
        2.125, 0.625, 2.75, -0.125, 1.75, -2.625,
        -2.625, -0.625, -0.75, -1.75, -0.75, -1.75,
        -0.75, -0.375, 1.0, 1.0, -0.25, 1.75,
        -3.625, -1.875, 1.875, -2.75, -2.875, -2.625,
        -2.0, 1.25, 1.5, 1.5, 2.375, 2.25,
        3.625, 1.875, 3.75, 1.5, 1.5, -0.375,
        -0.5, -1.75, -1.0, -2.75, -1.5, -1.125,
        -1.375, -1.625, -3.625, -1.75, -3.875, -1.875,
        -1.625, -0.25, 3.25, 2.375, 2.375, 2.0,
        1.5, 1.875, 2.0, 1.375, 2.875, 0.625,
        2.75, -0.125, 0.25, -0.375, -0.75, -1.75,
        -0.75, -1.75, -0.75, -0.625, 0.5, 0.25,
        -0.375, 1.0, -0.25, 1.75, -2.125, -3.125,
        -3.75, -3.0, -2.875, -2.625, -2.0, -0.25,
        3.25, 2.625, 4.375, 2.25, 3.625, 1.875,
        2.25, 0.375, 0.0, -0.75, -0.5, -1.75,
        -1.0, -0.875,
    ];
    assert_close(
        &got[0],
        &[1, 2, 4, 4, 7],
        &want0,
        "conv_transpose3d_dilations[0]",
    );
}

#[test]
fn conv_transpose3d_auto_pad_same_upper() {
    // ConvTranspose3D auto_pad=SAME_UPPER targets out = in * stride.
    let x = Tensor::new(
        (0..36).map(|i| ((i % 11) as f32 - 5.0) * 0.5).collect(),
        vec![1, 2, 2, 3, 3],
    );
    let weight = Tensor::new(
        (0..48).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect(),
        vec![2, 2, 2, 2, 3],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 3]);
    ints(&mut node, "strides", &[2, 2, 2]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        1.875, 1.5, 2.625, 1.375, 2.375, 1.25,
        0.75, -1.375, -0.625, -1.625, -0.25, -1.875,
        0.75, 1.125, 1.875, -0.375, -1.125, -0.5,
        1.875, -2.125, -3.625, 1.75, -0.5, 1.5,
        -0.375, -0.625, -1.625, -0.75, -1.875, -0.875,
        -1.125, 1.25, 0.25, 1.0, 0.625, 0.75,
        -2.125, 1.875, -0.375, 1.5, -0.25, 1.125,
        1.125, 0.75, -0.125, 1.125, -0.25, 1.5,
        -1.375, 0.75, 1.375, 0.375, 0.125, 0.0,
        1.5, 1.875, -3.25, -1.875, 0.75, -1.5,
        0.75, -0.375, 0.375, -0.75, 0.5, -1.125,
        -0.875, -1.125, 0.5, -0.75, 0.375, -0.375,
        -1.5, -1.0, -2.375, -1.125, 1.5, 1.5,
        0.0, 0.5, 1.375, 0.25, 1.75, -1.375,
        1.5, 1.375, 2.375, 1.25, 2.125, 1.125,
        1.125, -1.625, -0.25, -1.875, 0.125, -2.125,
        0.375, -0.375, -1.125, -0.5, -1.375, -0.625,
        -1.875, 1.75, -0.5, 1.5, -0.125, 1.25,
        1.5, -1.5, 0.75, -1.875, -3.25, 1.875,
        -0.5, 0.0, 0.125, 0.375, 1.375, 0.75,
        -1.875, 1.5, -0.25, 1.125, -0.125, 0.75,
        1.25, 1.125, -0.25, 1.5, -0.375, 1.875,
        0.25, 0.375, 0.125, 0.0, 0.25, -0.375,
        -1.125, -1.875, 0.75, -1.5, 0.625, -1.125,
        -1.75, -2.125, 0.125, -1.875, -0.25, -1.625,
        1.5, 1.125, 2.125, 1.25, 2.375, 1.375,
        -1.75, -1.375, 1.75, 0.25, 1.375, 0.5,
        1.125, 1.5, 1.5, -1.125, -2.375, -1.0,
        1.0, 0.75, 0.625, 1.0, 0.25, 1.25,
        -0.625, -0.875, -1.875, -0.75, -1.625, -0.625,
        -1.375, -1.75, -3.75, -1.75, -3.75, -1.75,
        1.875, 1.5, 2.625, 1.375, 2.375, 1.25,
        -2.125, -1.75, 0.375, 1.0, 1.75, 1.0,
        0.75, 1.125, 1.875, -0.375, -1.125, -0.5,
        1.25, 1.0, 1.75, 1.0, 1.75, 1.0,
        -0.375, -0.625, -1.625, -0.75, -1.875, -0.875,
        1.0, 1.5, -0.5, 1.75, -3.625, -2.125,
        -1.0, -0.5, -1.125, -0.375, 1.875, 1.125,
        -1.75, -1.875, -0.25, -1.625, -0.625, -1.375,
        1.375, 1.25, 2.375, 1.375, 2.625, 1.5,
        1.0, 0.25, 1.375, 0.5, 1.0, 0.75,
        -0.375, -1.125, -2.375, -1.0, -2.125, -0.875,
        0.5, 1.0, 1.75, 1.0, 0.375, -1.75,
        -1.5, -1.0, -2.375, -1.125, 1.5, 1.5,
        -1.625, -1.75, -3.75, -1.75, -3.75, -1.75,
        1.5, 1.375, 2.375, 1.25, 2.125, 1.125,
        1.75, 1.0, 1.75, 1.0, 1.75, 1.0,
        0.375, -0.375, -1.125, -0.5, -1.375, -0.625,
    ];
    assert_close(
        &got[0],
        &[1, 2, 4, 6, 6],
        &want0,
        "conv_transpose3d_auto_pad_same_upper[0]",
    );
}

#[test]
fn conv_transpose3d_group2() {
    // Grouped ConvTranspose3D: 4 input channels, group=2.
    let x = Tensor::new(
        (0..48).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect(),
        vec![1, 4, 2, 2, 3],
    );
    let weight = Tensor::new(
        (0..32).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect(),
        vec![4, 1, 2, 2, 2],
    );
    let mut node = node_with(OpKind::ConvTranspose, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 2]);
    ints(&mut node, "strides", &[1, 2, 1]);
    int(&mut node, "group", 2);
    let got = run_both(&ConvTransposeOp, &node, vec![Some(&x), Some(&weight)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        2.125, 3.375, 3.125, 1.5, 0.875, 2.75,
        2.125, -0.125, 1.375, 2.625, 2.375, 1.5,
        1.625, 0.875, 0.25, -2.75, -1.75, -5.25,
        -6.5, -3.375, -3.5, -2.25, -2.875, -0.375,
        1.25, 1.125, -0.125, 0.0, 0.25, 1.5,
        0.875, -0.375, 1.75, 2.625, 1.625, 0.75,
        1.25, 2.125, 2.125, 1.25, 1.0, 0.75,
        -0.25, -0.375, -0.25, -1.25, -1.25, -0.625,
        1.5, 2.375, 1.875, 1.0, 1.0, 1.875,
        2.375, 1.5, 0.375, -0.25, -0.75, -0.5,
        -0.875, -2.25, -1.75, -0.75, -0.25, -0.375,
        0.625, 1.125, -3.25, -6.75, -6.5, -2.625,
        -0.625, -1.875, -0.875, 0.0, 3.125, 5.25,
        5.5, 3.0, -0.25, 0.25, 1.75, 1.625,
        1.375, 2.625, 2.375, 1.5, 0.5, 1.375,
        2.875, 2.0, -2.75, -6.0, -6.25, -3.0,
    ];
    assert_close(
        &got[0],
        &[1, 2, 3, 4, 4],
        &want0,
        "conv_transpose3d_group2[0]",
    );
}

// ── MaxPool at spatial rank 1 and 3 ──────────────────────────────────────────

#[test]
fn max_pool1d_basic() {
    // MaxPool1D kernel 3, stride 2.
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.5, 0.5, 1.5, 1.5, 0.0, 1.5,
        1.5, 0.0, 1.0, 1.5,
    ];
    assert_close(&got[0], &[1, 2, 5], &want0, "max_pool1d_basic[0]");
}

#[test]
fn max_pool1d_ceil_mode() {
    // MaxPool1D ceil_mode keeps the trailing partial window.
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[3]);
    int(&mut node, "ceil_mode", 1);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.5, 1.0, 1.5, 0.0, 1.5, -0.5,
        1.0, 1.5,
    ];
    assert_close(&got[0], &[1, 2, 4], &want0, "max_pool1d_ceil_mode[0]");
}

#[test]
fn max_pool1d_dilation2() {
    // MaxPool1D with dilation 2.
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "dilations", &[2]);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        0.5, 1.5, 1.5, 1.5, 1.5, 1.5,
        1.0, 1.0,
    ];
    assert_close(&got[0], &[1, 2, 4], &want0, "max_pool1d_dilation2[0]");
}

#[test]
fn max_pool1d_auto_pad_same_upper() {
    // MaxPool1D auto_pad=SAME_UPPER -> out = ceil(11/2) = 6.
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -1.0, 0.0, 1.0, 1.5, -0.5, 0.0,
        1.0, 1.5, -0.5, 0.5, 1.5, 1.5,
    ];
    assert_close(
        &got[0],
        &[1, 2, 6],
        &want0,
        "max_pool1d_auto_pad_same_upper[0]",
    );
}

#[test]
fn max_pool1d_indices() {
    // MaxPool1D also emitting the flat Indices output.
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::MaxPool, &["y", "indices"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "pads", &[1, 1]);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -1.0, 0.0, 1.0, 1.5, -0.5, 0.0,
        1.0, 1.5, -0.5, 0.5, 1.5, 1.5,
    ];
    assert_close(&got[0], &[1, 2, 6], &want0, "max_pool1d_indices[0]");
    #[rustfmt::skip]
    let want1: Vec<f32> = vec![
        1.0, 3.0, 5.0, 6.0, 9.0, 10.0,
        12.0, 13.0, 16.0, 18.0, 20.0, 20.0,
    ];
    assert_close(&got[1], &[1, 2, 6], &want1, "max_pool1d_indices[1]");
}

#[test]
fn max_pool1d_ceil_mode_drops_window_starting_in_right_padding() {
    // in=5, k=3, s=3, pads=[0,2], ceil_mode=1: the naive ceil formula gives
    // 3, but the third window would *start* at index 6 — inside the right
    // padding — so ONNX Runtime (and onnx.reference) drop it and return 2.
    // Pins the per-axis pad indexing of that correction at spatial rank 1;
    // the rank-2 case is max_pool_ceil_mode_drops_window_starting_in_right_padding
    // in w1_conv_pool.rs. (onnx.shape_inference disagrees and predicts 3 —
    // it does not implement the correction; the kernel follows the
    // evaluator, which matches ORT.)
    let x = Tensor::new(
        (0..5).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 1, 5],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[3]);
    ints(&mut node, "pads", &[0, 2]);
    int(&mut node, "ceil_mode", 1);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.5, 0.5,
    ];
    assert_close(
        &got[0],
        &[1, 1, 2],
        &want0,
        "max_pool1d_ceil_mode_drops_window_starting_in_right_padding[0]",
    );
}

#[test]
fn max_pool3d_basic() {
    // MaxPool3D kernel 2x2x2, stride 2.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 2]);
    ints(&mut node, "strides", &[2, 2, 2]);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        3.0, 1.0, 2.5, 3.0, 3.0, 3.0,
        3.0, 2.5, 2.0, 3.0, 3.0, 3.0,
        3.0, 2.0, 2.0, 3.0,
    ];
    assert_close(&got[0], &[1, 2, 2, 2, 2], &want0, "max_pool3d_basic[0]");
}

#[test]
fn max_pool3d_ceil_mode_pads() {
    // MaxPool3D with ceil_mode and asymmetric padding.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3, 3, 3]);
    ints(&mut node, "strides", &[2, 2, 2]);
    ints(&mut node, "pads", &[1, 1, 1, 0, 0, 0]);
    int(&mut node, "ceil_mode", 1);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        3.0, 1.0, 1.5, 2.5, 3.0, 3.0,
        1.0, 2.0, 2.5, 3.0, 3.0, 1.0,
        3.0, 3.0, 3.0, 3.0, 1.5, 2.0,
        2.0, 3.0, 3.0, 3.0, 3.0, 2.0,
        3.0, 3.0, 0.5, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 2.5, 3.0, 3.0,
    ];
    assert_close(
        &got[0],
        &[1, 2, 2, 3, 3],
        &want0,
        "max_pool3d_ceil_mode_pads[0]",
    );
}

#[test]
fn max_pool3d_dilations() {
    // MaxPool3D with a different dilation per axis.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 2]);
    ints(&mut node, "strides", &[1, 2, 1]);
    ints(&mut node, "dilations", &[2, 2, 2]);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        3.0, 3.0, 3.0, 3.0, 2.5, 3.0,
        3.0, 3.0, 3.0, 2.5, 3.0, 2.5,
        2.5, 3.0, 2.5, 1.0, 1.5, 2.0,
        2.0, 2.5, 3.0, 0.5, 1.0, 1.5,
    ];
    assert_close(&got[0], &[1, 2, 2, 2, 3], &want0, "max_pool3d_dilations[0]");
}

#[test]
fn max_pool3d_auto_pad_same_upper() {
    // MaxPool3D auto_pad=SAME_UPPER preserves ceil(in/stride) per axis.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[3, 3, 3]);
    ints(&mut node, "strides", &[2, 2, 2]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        3.0, 3.0, 1.5, 2.5, 3.0, 3.0,
        1.0, 2.0, 2.5, 3.0, 3.0, 0.5,
        3.0, 2.5, 3.0, 3.0, 1.0, 1.5,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 2.5,
        3.0, 3.0, 3.0, 2.0, 3.0, 3.0,
    ];
    assert_close(
        &got[0],
        &[1, 2, 2, 3, 3],
        &want0,
        "max_pool3d_auto_pad_same_upper[0]",
    );
}

#[test]
fn max_pool3d_indices() {
    // MaxPool3D flat Indices over a rank-5 input.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::MaxPool, &["y", "indices"]);
    ints(&mut node, "kernel_shape", &[2, 2, 2]);
    ints(&mut node, "strides", &[2, 2, 2]);
    let got = run_both(&MaxPoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        3.0, 1.0, 2.5, 3.0, 3.0, 3.0,
        3.0, 2.5, 2.0, 3.0, 3.0, 3.0,
        3.0, 2.0, 2.0, 3.0,
    ];
    assert_close(&got[0], &[1, 2, 2, 2, 2], &want0, "max_pool3d_indices[0]");
    #[rustfmt::skip]
    let want1: Vec<f32> = vec![
        25.0, 8.0, 11.0, 12.0, 51.0, 77.0,
        90.0, 63.0, 101.0, 103.0, 116.0, 142.0,
        155.0, 153.0, 166.0, 168.0,
    ];
    assert_close(&got[1], &[1, 2, 2, 2, 2], &want1, "max_pool3d_indices[1]");
}

// ── AveragePool at spatial rank 1 and 3 ──────────────────────────────────────

#[test]
fn avg_pool1d_basic() {
    // AveragePool1D kernel 3, stride 2.
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -1.0, 0.0, 1.0, -0.33333334, -0.5, 1.0,
        -0.33333334, -0.5, 0.5, 0.33333334,
    ];
    assert_close(&got[0], &[1, 2, 5], &want0, "avg_pool1d_basic[0]");
}

#[test]
fn avg_pool1d_pads_exclude() {
    // AveragePool1D with padding and count_include_pad=0 (divide by the
    // number of in-bounds samples).
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "pads", &[1, 1]);
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -1.25, -0.5, 0.5, 0.33333334, -1.0, -0.25,
        0.75, 0.33333334, -1.0, 0.0, 1.0, 0.0,
    ];
    assert_close(&got[0], &[1, 2, 6], &want0, "avg_pool1d_pads_exclude[0]");
}

#[test]
fn avg_pool1d_pads_include() {
    // AveragePool1D with count_include_pad=1 (divide by the full window).
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    ints(&mut node, "pads", &[1, 1]);
    int(&mut node, "count_include_pad", 1);
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.8333333, -0.5, 0.5, 0.33333334, -1.0, -0.16666667,
        0.5, 0.33333334, -1.0, 0.0, 1.0, 0.0,
    ];
    assert_close(&got[0], &[1, 2, 6], &want0, "avg_pool1d_pads_include[0]");
}

#[test]
fn avg_pool1d_auto_pad_same_upper() {
    // AveragePool1D auto_pad=SAME_UPPER.
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[2]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -1.25, -0.5, 0.5, 0.33333334, -1.0, -0.25,
        0.75, 0.33333334, -1.0, 0.0, 1.0, 0.0,
    ];
    assert_close(
        &got[0],
        &[1, 2, 6],
        &want0,
        "avg_pool1d_auto_pad_same_upper[0]",
    );
}

#[test]
fn avg_pool1d_ceil_mode() {
    // AveragePool1D ceil_mode with no padding.
    let x = Tensor::new(
        (0..22).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        vec![1, 2, 11],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[3]);
    ints(&mut node, "strides", &[3]);
    int(&mut node, "ceil_mode", 1);
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -1.0, 0.5, -0.33333334, -0.25, 1.0, -1.0,
        0.5, 0.0,
    ];
    assert_close(&got[0], &[1, 2, 4], &want0, "avg_pool1d_ceil_mode[0]");
}

#[test]
fn avg_pool3d_basic() {
    // AveragePool3D kernel 2x2x2, stride 2.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 2]);
    ints(&mut node, "strides", &[2, 2, 2]);
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.9375, -0.75, 0.0, 0.1875, 0.5, -0.9375,
        -0.1875, 0.0, -0.5, 0.5, 1.25, -0.1875,
        0.9375, -0.5, 0.25, 1.25,
    ];
    assert_close(&got[0], &[1, 2, 2, 2, 2], &want0, "avg_pool3d_basic[0]");
}

#[test]
fn avg_pool3d_pads_exclude() {
    // AveragePool3D with symmetric padding, count_include_pad=0.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[3, 3, 3]);
    ints(&mut node, "strides", &[2, 2, 2]);
    ints(&mut node, "pads", &[1, 1, 1, 1, 1, 1]);
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.9375, -1.0, -0.25, -0.16666667, 0.22222222, -0.29166666,
        -0.75, 0.0, 0.75, 0.20833333, -0.6666667, -1.0,
        -0.5555556, -0.16666667, 0.22222222, -0.9583333, -0.75, 0.0,
        -0.5, 0.25, 0.1875, 0.0, -0.33333334, -0.6666667,
        0.5, -0.375, -1.25, 0.375, -0.1388889, 0.25,
        0.33333334, 0.0, -0.33333334, -0.25, 0.1388889, -0.375,
    ];
    assert_close(
        &got[0],
        &[1, 2, 2, 3, 3],
        &want0,
        "avg_pool3d_pads_exclude[0]",
    );
}

#[test]
fn avg_pool3d_pads_include() {
    // AveragePool3D with count_include_pad=1.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[3, 3, 3]);
    ints(&mut node, "strides", &[2, 2, 2]);
    ints(&mut node, "pads", &[1, 1, 1, 1, 1, 1]);
    int(&mut node, "count_include_pad", 1);
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.2777778, -0.44444445, -0.074074075, -0.074074075, 0.14814815, -0.12962963,
        -0.22222222, 0.0, 0.22222222, 0.09259259, -0.44444445, -0.44444445,
        -0.37037036, -0.16666667, 0.14814815, -0.42592594, -0.5, 0.0,
        -0.14814815, 0.11111111, 0.055555556, 0.0, -0.22222222, -0.2962963,
        0.14814815, -0.16666667, -0.37037036, 0.16666667, -0.09259259, 0.11111111,
        0.22222222, 0.0, -0.22222222, -0.11111111, 0.09259259, -0.16666667,
    ];
    assert_close(
        &got[0],
        &[1, 2, 2, 3, 3],
        &want0,
        "avg_pool3d_pads_include[0]",
    );
}

#[test]
fn avg_pool3d_auto_pad_same_upper() {
    // AveragePool3D auto_pad=SAME_UPPER.
    let x = Tensor::new(
        (0..200).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
        vec![1, 2, 4, 5, 5],
    );
    let mut node = node_with(OpKind::AveragePool, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2, 2]);
    ints(&mut node, "strides", &[2, 2, 2]);
    string(&mut node, "auto_pad", "SAME_UPPER");
    let got = run_both(&AveragePoolOp, &node, vec![Some(&x)]);
    #[rustfmt::skip]
    let want0: Vec<f32> = vec![
        -0.9375, -0.75, 0.0, 0.0, 0.1875, -1.5,
        0.5, 1.5, 2.25, 0.5, -0.9375, -1.0,
        -0.1875, 0.0, 0.75, -0.5, 0.5, 1.25,
        -0.5, 0.5, -0.375, 1.25, -0.1875, -0.25,
        -1.5, -0.5, 0.25, 0.9375, -0.5, 0.25,
        0.25, 1.25, 0.375, -2.5, -1.5, -0.75,
    ];
    assert_close(
        &got[0],
        &[1, 2, 2, 3, 3],
        &want0,
        "avg_pool3d_auto_pad_same_upper[0]",
    );
}

// ── Malformed rank-N nodes must be typed errors, never panics ────────────────

#[test]
fn conv_rank_mismatch_between_input_and_weight_is_typed_error() {
    // Conv1D input with a 2D weight: the spatial ranks disagree.
    let x = Tensor::new(vec![0.0_f32; 10], vec![1, 2, 5]);
    let weight = Tensor::new(vec![1.0_f32; 18], vec![1, 2, 3, 3]);
    let node = node_with(OpKind::Conv, &["y"]);
    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&x), Some(&weight)]))
        .expect_err("rank mismatch must be rejected");
    assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
}

#[test]
fn conv_rank2_input_has_no_spatial_axis_and_is_a_typed_error() {
    // [N, C] has zero spatial axes — Conv requires at least one.
    let x = Tensor::new(vec![0.0_f32; 6], vec![2, 3]);
    let weight = Tensor::new(vec![1.0_f32; 3], vec![1, 3]);
    let node = node_with(OpKind::Conv, &["y"]);
    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&x), Some(&weight)]))
        .expect_err("rank-2 input must be rejected");
    assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
}

#[test]
fn conv3d_short_strides_defaults_the_missing_axes_like_rank_2_does() {
    // A `strides` shorter than the spatial rank is malformed per the spec, but
    // this engine has always defaulted the missing axes to 1 rather than
    // rejecting the model, and the planner agrees (see
    // `a_short_strides_attribute_defaults_the_missing_axis_instead_of_panicking`
    // in tests/s4_engine_stitch.rs). Rank 3 must follow the same convention.
    let x = Tensor::new(vec![1.0_f32; 2 * 4 * 4 * 4], vec![1, 2, 4, 4, 4]);
    let weight = Tensor::new(vec![1.0_f32; 2 * 8], vec![1, 2, 2, 2, 2]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "strides", &[2]); // only the D axis is supplied
    let got = run_both(&ConvOp, &node, vec![Some(&x), Some(&weight)]);
    // D: (4-2)/2+1 = 2;  H and W default to stride 1 -> (4-2)/1+1 = 3.
    assert_eq!(got[0].shape, vec![1, 1, 2, 3, 3]);
    assert!(got[0].data.iter().all(|v| (*v - 16.0).abs() < 1e-6));
}

#[test]
fn max_pool_short_strides_defaults_the_missing_axis_at_every_rank() {
    // The rank-2 form of this is pinned end-to-end through the session by
    // `a_short_strides_attribute_defaults_the_missing_axis_instead_of_panicking`
    // in tests/s4_engine_stitch.rs; the operator must keep that convention
    // (and apply it identically at rank 1 and 3) or the planner and the
    // kernel disagree about the output extent.
    let x2 = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]);
    let mut node2 = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node2, "kernel_shape", &[2, 2]);
    ints(&mut node2, "strides", &[2]); // W defaults to stride 1
    let got2 = run_both(&MaxPoolOp, &node2, vec![Some(&x2)]);
    assert_eq!(got2[0].shape, vec![1, 1, 2, 3]);

    let x1 = Tensor::new((0..8).map(|i| i as f32).collect(), vec![1, 1, 8]);
    let mut node1 = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node1, "kernel_shape", &[2]);
    let got1 = run_both(&MaxPoolOp, &node1, vec![Some(&x1)]); // strides absent
    assert_eq!(got1[0].shape, vec![1, 1, 7]);

    let x3 = Tensor::new((0..64).map(|i| i as f32).collect(), vec![1, 1, 4, 4, 4]);
    let mut node3 = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node3, "kernel_shape", &[2, 2, 2]);
    ints(&mut node3, "strides", &[2, 2]); // W defaults to stride 1
    let got3 = run_both(&MaxPoolOp, &node3, vec![Some(&x3)]);
    assert_eq!(got3[0].shape, vec![1, 1, 2, 2, 3]);
}

#[test]
fn pool3d_kernel_shape_arity_must_match_the_input_rank() {
    let x = Tensor::new(vec![0.0_f32; 27], vec![1, 1, 3, 3, 3]);
    let mut node = node_with(OpKind::MaxPool, &["y"]);
    ints(&mut node, "kernel_shape", &[2, 2]); // 2 entries for a rank-3 pool
    let err = MaxPoolOp
        .execute(&ctx(&node, vec![Some(&x)]))
        .expect_err("wrong kernel_shape arity must be rejected");
    assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
}

#[test]
fn max_pool3d_column_major_storage_order_is_unsupported_not_wrong() {
    // ONNX defines the column-major Indices encoding for the 2D case only;
    // guessing a rank-3 generalisation would silently emit wrong indices.
    let x = Tensor::new(vec![0.0_f32; 27], vec![1, 1, 3, 3, 3]);
    let mut node = node_with(OpKind::MaxPool, &["y", "indices"]);
    ints(&mut node, "kernel_shape", &[2, 2, 2]);
    int(&mut node, "storage_order", 1);
    let err = MaxPoolOp
        .execute(&ctx(&node, vec![Some(&x)]))
        .expect_err("storage_order=1 at rank 3 must be rejected");
    assert!(matches!(err, OnnxError::Unsupported(_)), "got {err:?}");
}

#[test]
fn conv1d_padded_input_smaller_than_dilated_kernel_is_a_typed_error() {
    let x = Tensor::new(vec![0.0_f32; 4], vec![1, 1, 4]);
    let weight = Tensor::new(vec![1.0_f32; 3], vec![1, 1, 3]);
    let mut node = node_with(OpKind::Conv, &["y"]);
    ints(&mut node, "dilations", &[4]); // dilated extent 9 > 4
    let err = ConvOp
        .execute(&ctx(&node, vec![Some(&x), Some(&weight)]))
        .expect_err("over-dilated kernel must be rejected");
    assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
}
