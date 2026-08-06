//! Stitch-wave (S2-shape-stitch) cross-file regression tests.
//!
//! Covers the registry-level wiring fixes that Wave-1 could not reach because the files were
//! unowned at the time (`registry/conv_ops/pad.rs`) plus end-to-end (registry-operator, not just
//! bare-function) coverage for the `.max(1)` zero-dim idiom and unchecked-axis-cast fixes in
//! this domain's owned files (`indexing/compress.rs`, `indexing/unique.rs`, `math/reduce.rs`,
//! `math/topk.rs`, and `comparison::trilu`).
//!
//! Function-level coverage for these fixes lives as `#[cfg(test)]` modules inside the owned
//! source files themselves; this file specifically exercises the `Operator` trait entry points
//! (`execute` / `execute_into_slots`) so a routing regression (e.g. `execute_into_slots` drifting
//! back to a hand-rolled duplicate that skips a fix `execute` has) would be caught.

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::{OnnxError, Tensor};
use oxionnx_ops::registry::conv_ops::PadOp;
use oxionnx_ops::registry::indexing_ops::{CompressOp, UniqueOp};
use oxionnx_ops::registry::math_ops::{ReduceSumOp, TopKOp};
use oxionnx_ops::registry::misc_ops::TriluOp;

// ── Test infrastructure ──────────────────────────────────────────────────────

fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn dummy_node(op: OpKind) -> Node {
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn node_with_mode(op: OpKind, mode: &str) -> Node {
    let mut n = dummy_node(op);
    n.attrs.strings.insert("mode".into(), mode.into());
    n
}

fn empty_slot(shape: Vec<usize>) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new(vec![0.0; n], shape)
}

fn assert_tensor_eq(a: &Tensor, b: &Tensor, label: &str) {
    assert_eq!(a.shape, b.shape, "{label}: shape mismatch");
    assert_eq!(a.data, b.data, "{label}: data mismatch");
}

// ── PadOp: opset-18 `axes` input (ctx.optional_input(3)) ─────────────────────

#[test]
fn pad_op_execute_routes_opset18_axes_input() -> Result<(), OnnxError> {
    // Same scenario as `w1_shape_ops.rs::pad_axes_opset18_partial_axes`, this time driven
    // through the registry `PadOp::execute` entry point (input 3 = axes) rather than calling
    // `pad_axes` directly, to prove the routing itself works: x shape [1,2,3,3], pads=[1,1] on
    // axis 2 alone (axes=[2]) -> shape [1,2,5,3]; axes 0,1,3 untouched.
    let node = dummy_node(OpKind::Pad);
    let x = Tensor::new((0..18).map(|i| i as f32).collect(), vec![1, 2, 3, 3]);
    let pads = Tensor::new(vec![1.0, 1.0], vec![2]);
    let axes = Tensor::new(vec![2.0], vec![1]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&pads), None, Some(&axes)]);
    let out = PadOp.execute(&ctx)?;
    assert_eq!(out[0].shape, vec![1, 2, 5, 3]);
    Ok(())
}

#[test]
fn pad_op_execute_into_slots_matches_execute_for_opset18_axes() -> Result<(), OnnxError> {
    // The old `execute_into_slots` was a hand-inlined duplicate that never read input(3) at
    // all (it derived `ndim` from the data tensor and required `pads.len() == 2*ndim`
    // unconditionally) -- an axes-shortened `pads` would have hit its length-mismatch guard.
    // Both entry points must now agree.
    let node = dummy_node(OpKind::Pad);
    let x = Tensor::new((0..18).map(|i| i as f32).collect(), vec![1, 2, 3, 3]);
    let pads = Tensor::new(vec![1.0, 1.0], vec![2]);
    let axes = Tensor::new(vec![2.0], vec![1]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&pads), None, Some(&axes)]);

    let expected = PadOp.execute(&ctx)?;
    let mut slots = vec![empty_slot(vec![1, 1, 1, 1])]; // deliberately wrong initial shape
    PadOp.execute_into_slots(&ctx, &mut slots)?;
    assert_tensor_eq(&slots[0], &expected[0], "pad opset18 axes slots[0]");
    Ok(())
}

#[test]
fn pad_op_execute_into_slots_applies_negative_pads_crop() -> Result<(), OnnxError> {
    // The old `execute_into_slots` did `p.max(0) as usize`, silently dropping negative
    // (crop) pads. x shape [1,1,4,4] values 0..16; pads=[0,0,-1,-1,0,0,-1,-1] crops a
    // 1-element border off both spatial dims -> [1,1,2,2] == interior [[5,6],[9,10]].
    let node = dummy_node(OpKind::Pad);
    let x = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]);
    let pads = Tensor::new(vec![0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0], vec![8]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&pads)]);

    let expected = PadOp.execute(&ctx)?;
    assert_eq!(expected[0].shape, vec![1, 1, 2, 2]);
    assert_eq!(expected[0].data, vec![5.0, 6.0, 9.0, 10.0]);

    let mut slots = vec![empty_slot(vec![1, 1, 4, 4])];
    PadOp.execute_into_slots(&ctx, &mut slots)?;
    assert_tensor_eq(&slots[0], &expected[0], "pad negative-crop slots[0]");
    Ok(())
}

#[test]
fn pad_op_execute_into_slots_applies_wrap_mode() -> Result<(), OnnxError> {
    // The old `execute_into_slots` had no "wrap" arm at all (its `match` fell through to the
    // "constant" default). x = [1,2,3], pads=[2,2], mode="wrap" -> [2,3,1,2,3,1,2].
    let node = node_with_mode(OpKind::Pad, "wrap");
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let pads = Tensor::new(vec![2.0, 2.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&pads)]);

    let expected = PadOp.execute(&ctx)?;
    assert_eq!(expected[0].data, vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);

    let mut slots = vec![empty_slot(vec![3])];
    PadOp.execute_into_slots(&ctx, &mut slots)?;
    assert_tensor_eq(&slots[0], &expected[0], "pad wrap-mode slots[0]");
    Ok(())
}

#[test]
fn pad_op_execute_and_execute_into_slots_agree_on_reflect_mode() -> Result<(), OnnxError> {
    let node = node_with_mode(OpKind::Pad, "reflect");
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let pads = Tensor::new(vec![2.0, 2.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&pads)]);

    let expected = PadOp.execute(&ctx)?;
    let mut slots = vec![empty_slot(expected[0].shape.clone())];
    PadOp.execute_into_slots(&ctx, &mut slots)?;
    assert_tensor_eq(&slots[0], &expected[0], "pad reflect-mode slots[0]");
    Ok(())
}

#[test]
fn pad_op_constant_value_present_but_empty_tensor_does_not_panic() -> Result<(), OnnxError> {
    // A model can supply a *present* but 0-element constant_value tensor; `.data[0]` on that
    // panics, `.data.first()` falls back to the ONNX default (0.0) instead.
    let node = dummy_node(OpKind::Pad);
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let pads = Tensor::new(vec![1.0, 1.0], vec![2]);
    let empty_constant = Tensor::new(Vec::new(), vec![0]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&pads), Some(&empty_constant)]);

    let out = PadOp.execute(&ctx)?;
    assert_eq!(out[0].data, vec![0.0, 1.0, 2.0, 0.0]);

    let mut slots = vec![empty_slot(vec![4])];
    PadOp.execute_into_slots(&ctx, &mut slots)?;
    assert_tensor_eq(&slots[0], &out[0], "pad empty-constant slots[0]");
    Ok(())
}

#[test]
fn pad_op_missing_required_pads_input_errors_not_panics() {
    let node = dummy_node(OpKind::Pad);
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&x)]); // no `pads` at index 1
    assert!(PadOp.execute(&ctx).is_err());
    let mut slots = vec![empty_slot(vec![2])];
    assert!(PadOp.execute_into_slots(&ctx, &mut slots).is_err());
}

// ── ReduceSum / TopK: end-to-end (registry) coverage for the fixes above ─────

#[test]
fn reduce_sum_op_execute_into_slots_bad_axis_errors_not_panics() {
    let mut node = dummy_node(OpKind::ReduceSum);
    node.attrs.int_lists.insert("axes".into(), vec![5]); // out of range for a 2D tensor
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let ctx = make_ctx(&node, vec![Some(&x)]);

    assert!(ReduceSumOp.execute(&ctx).is_err());
    let mut slots = vec![empty_slot(vec![2, 3])];
    assert!(ReduceSumOp.execute_into_slots(&ctx, &mut slots).is_err());
}

#[test]
fn top_k_op_execute_into_slots_zero_size_dim_does_not_panic() -> Result<(), OnnxError> {
    let mut node = dummy_node(OpKind::TopK);
    node.attrs.ints.insert("axis".into(), 0); // reduce axis 0 (size 2); axis 1 is the genuine 0
    let x = Tensor::new(Vec::new(), vec![2, 0, 3]); // 0 elements
    let k = Tensor::new(vec![1.0], vec![1]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&k)]);

    let out = TopKOp.execute(&ctx)?;
    assert_eq!(out[0].shape, vec![1, 0, 3]);
    assert!(out[0].data.is_empty());

    let mut slots = vec![empty_slot(vec![1, 1, 1]), empty_slot(vec![1, 1, 1])];
    TopKOp.execute_into_slots(&ctx, &mut slots)?;
    assert_eq!(slots[0].shape, vec![1, 0, 3]);
    assert_eq!(slots[1].shape, vec![1, 0, 3]);
    Ok(())
}

// ── Compress / Unique / Trilu: registry-level smoke coverage ────────────────

#[test]
fn compress_op_execute_zero_size_outer_dim_does_not_panic() -> Result<(), OnnxError> {
    let mut node = dummy_node(OpKind::Compress);
    node.attrs.ints.insert("axis".into(), 1);
    let x = Tensor::new(Vec::new(), vec![0, 3, 4]);
    let cond = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&cond)]);
    let out = CompressOp.execute(&ctx)?;
    assert_eq!(out[0].shape, vec![0, 2, 4]);
    Ok(())
}

#[test]
fn unique_op_execute_uses_exact_bit_pattern_equality() -> Result<(), OnnxError> {
    let node = dummy_node(OpKind::Unique);
    // See `indexing::unique::tests` for the derivation: these two values are distinct f32 bit
    // patterns whose difference is smaller than `f32::EPSILON`.
    let x = Tensor::new(vec![1e-7_f32, 2e-7_f32, 1e-7_f32], vec![3]);
    let ctx = make_ctx(&node, vec![Some(&x)]);
    let out = UniqueOp.execute(&ctx)?;
    assert_eq!(out[0].data, vec![1e-7_f32, 2e-7_f32]);
    Ok(())
}

#[test]
fn trilu_op_execute_into_slots_zero_size_batch_does_not_panic() -> Result<(), OnnxError> {
    let mut node = dummy_node(OpKind::Trilu);
    node.attrs.ints.insert("upper".into(), 1);
    let x = Tensor::new(Vec::new(), vec![0, 3, 3]);
    let ctx = make_ctx(&node, vec![Some(&x)]);

    let out = TriluOp.execute(&ctx)?;
    assert_eq!(out[0].shape, vec![0, 3, 3]);
    assert!(out[0].data.is_empty());

    let mut slots = vec![empty_slot(vec![1, 1, 1])];
    TriluOp.execute_into_slots(&ctx, &mut slots)?;
    assert_eq!(slots[0].shape, vec![0, 3, 3]);
    Ok(())
}
