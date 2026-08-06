//! Wave-1 correctness regression tests for the `C-shape-ops` domain
//! (oxionnx-ops/src/shape/{basic,sequence}.rs, oxionnx-ops/src/registry/shape_ops/**).
//!
//! Each test below is tied to a specific audit finding and carries a hand-derived reference
//! value (traced by hand against the ONNX operator semantics, not copied from the
//! implementation under test).

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::{OnnxError, Tensor};
use oxionnx_ops::registry::shape_ops::{ConcatOp, ExpandOp, SplitOp, SqueezeOp, TransposeOp};
use oxionnx_ops::shape;
// `pad_axes` is not re-exported at the `shape` module root (that re-export list lives in
// `shape/mod.rs`, which is outside this domain's owned files), so it is reached via its
// defining submodule instead.
use oxionnx_ops::shape::sequence::pad_axes;

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

fn node_with_int_attrs(op: OpKind, pairs: &[(&str, i64)]) -> Node {
    let mut n = dummy_node(op);
    for &(k, v) in pairs {
        n.attrs.ints.insert(k.to_string(), v);
    }
    n
}

// ── [a0-1] / [a10-3]: Slice negative starts/ends, INT64_MAX/MIN sentinels, negative steps ──

#[test]
fn slice_negative_start_with_int_max_end() -> Result<(), OnnxError> {
    // x = [10,20,30,40,50], Slice(starts=[-2], ends=[INT64_MAX], axes=[0]) -> [40, 50].
    let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5]);
    let y = shape::slice(&x, &[-2], &[i64::MAX], Some(&[0]), None)?;
    assert_eq!(y.shape, vec![2]);
    assert_eq!(y.data, vec![40.0, 50.0]);
    Ok(())
}

#[test]
fn slice_negative_end() -> Result<(), OnnxError> {
    // x = [10,20,30,40,50], Slice(starts=[0], ends=[-1]) -> [10,20,30,40].
    let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5]);
    let y = shape::slice(&x, &[0], &[-1], None, None)?;
    assert_eq!(y.data, vec![10.0, 20.0, 30.0, 40.0]);
    Ok(())
}

#[test]
fn slice_negative_step_reverses_with_int_min_end() -> Result<(), OnnxError> {
    // x = [10,20,30,40,50], Slice(starts=[-1], ends=[INT64_MIN], steps=[-1]) -> full reverse.
    let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5]);
    let y = shape::slice(&x, &[-1], &[i64::MIN], None, Some(&[-1]))?;
    assert_eq!(y.data, vec![50.0, 40.0, 30.0, 20.0, 10.0]);
    Ok(())
}

#[test]
fn slice_negative_step_partial_reverse() -> Result<(), OnnxError> {
    // x = [0,1,2,3,4,5] (shape [6]); starts=[4], ends=[0], steps=[-2] -> indices 4, 2 -> [4, 2].
    let x = Tensor::new((0..6).map(|i| i as f32).collect(), vec![6]);
    let y = shape::slice(&x, &[4], &[0], None, Some(&[-2]))?;
    assert_eq!(y.data, vec![4.0, 2.0]);
    Ok(())
}

#[test]
fn slice_2d_negative_axis_and_step() -> Result<(), OnnxError> {
    // x = [[0,1,2],[3,4,5]] (shape [2,3]); slice axis=-1 (=1) reversed -> columns reversed.
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    let y = shape::slice(&x, &[-1], &[i64::MIN], Some(&[-1]), Some(&[-1]))?;
    assert_eq!(y.shape, vec![2, 3]);
    assert_eq!(y.data, vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0]);
    Ok(())
}

#[test]
fn slice_rejects_axis_out_of_range() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let err = shape::slice(&x, &[0], &[1], Some(&[5]), None);
    assert!(err.is_err(), "axis 5 on a 1D tensor must error, not panic");
}

#[test]
fn slice_rejects_length_mismatch() {
    // axes has 2 entries but starts only has 1 -> must error, not index out of bounds.
    let x = Tensor::new(vec![0.0; 6], vec![2, 3]);
    let err = shape::slice(&x, &[0], &[1, 2], Some(&[0, 1]), None);
    assert!(err.is_err());
}

#[test]
fn slice_rejects_zero_step() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let err = shape::slice(&x, &[0], &[3], None, Some(&[0]));
    assert!(err.is_err());
}

#[test]
fn slice_op_execute_rejects_axis_out_of_range_without_panicking() {
    // Reproduces [a10-3]: a 3-element `axes` with only a 1-element `starts`, purely via
    // runtime input tensors on an otherwise-valid node, must not panic.
    let node = dummy_node(OpKind::Slice);
    let data = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let starts = Tensor::new(vec![0.0], vec![1]);
    let ends = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let axes = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let ctx = make_ctx(
        &node,
        vec![Some(&data), Some(&starts), Some(&ends), Some(&axes), None],
    );
    use oxionnx_ops::registry::shape_ops::SliceOp;
    let result = SliceOp.execute(&ctx);
    assert!(result.is_err());
}

// ── [a0-3] / [a0-4] / [a0-5] / [a5-7] / [a10-9]: Pad ────────────────────────────────────────

#[test]
fn pad_axes_opset18_partial_axes() -> Result<(), OnnxError> {
    // Only axis 2 gets padding (opset-18 `axes` input): x shape [1,2,3,3], pads=[1,1] on
    // axis 2 alone -> shape [1,2,5,3]; axes 0,1,3 are untouched.
    let x = Tensor::new((0..18).map(|i| i as f32).collect(), vec![1, 2, 3, 3]);
    let y = pad_axes(&x, &[1, 1], "constant", 0.0, Some(&[2]))?;
    assert_eq!(y.shape, vec![1, 2, 5, 3]);
    Ok(())
}

#[test]
fn pad_axes_negative_axis_is_normalized() -> Result<(), OnnxError> {
    // axes=[-1] on a rank-2 tensor means axis 1.
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let y = pad_axes(&x, &[1, 1], "constant", 0.0, Some(&[-1]))?;
    assert_eq!(y.shape, vec![1, 5]);
    Ok(())
}

#[test]
fn pad_negative_pads_crop_the_interior() -> Result<(), OnnxError> {
    // x shape [1,1,4,4] with values 0..16; pads=[0,0,-1,-1, 0,0,-1,-1] crops a 1-element
    // border off both spatial dims -> shape [1,1,2,2] containing rows/cols {1,2} of x, i.e.
    // the interior 2x2 block [[5,6],[9,10]].
    let x = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]);
    let y = pad_axes(&x, &[0, 0, -1, -1, 0, 0, -1, -1], "constant", 0.0, None)?;
    assert_eq!(y.shape, vec![1, 1, 2, 2]);
    assert_eq!(y.data, vec![5.0, 6.0, 9.0, 10.0]);
    Ok(())
}

#[test]
fn pad_wrap_mode_is_circular() -> Result<(), OnnxError> {
    // x = [1,2,3] (shape [3]); pads=[2,2], mode="wrap" -> [2,3,1,2,3,1,2] (hand-traced against
    // out_coord - begin, rem_euclid(3)).
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let y = pad_axes(&x, &[2, 2], "wrap", 0.0, None)?;
    assert_eq!(y.shape, vec![7]);
    assert_eq!(y.data, vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
    Ok(())
}

#[test]
fn pad_axes_rejects_unknown_mode() {
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let err = pad_axes(&x, &[1, 1], "bogus", 0.0, None);
    assert!(
        err.is_err(),
        "an unrecognized mode must error, not silently act as constant"
    );
}

#[test]
fn pad_axes_rejects_pads_length_mismatch() {
    // pads has 3 entries; for a rank-2 tensor with axes=None it must be exactly 4.
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let err = pad_axes(&x, &[1, 1, 1], "constant", 0.0, None);
    assert!(err.is_err());
}

#[test]
fn pad_legacy_wrapper_never_panics_on_malformed_input() {
    // The 4-arg legacy `pad()` cannot report an Err (it returns `Tensor`, not `Result`); on
    // input `pad_axes` would reject it must fall back to a documented no-op instead of the
    // previous `assert!`-triggered panic.
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = shape::pad(&x, &[1, 1, 1], "constant", 0.0); // wrong pads length
    assert_eq!(y.shape, x.shape);
    assert_eq!(y.data, x.data);

    let y2 = shape::pad(&x, &[1, 1, 1, 1], "bogus-mode", 0.0); // unknown mode
    assert_eq!(y2.shape, x.shape);
    assert_eq!(y2.data, x.data);
}

#[test]
fn pad_legacy_wrapper_still_applies_valid_negative_pads_and_wrap() {
    // The bug fixes for crop and wrap must be visible through the *unchanged* 4-arg
    // signature too, since that is the only entry point `registry/conv_ops/pad.rs` currently
    // calls.
    let x = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]);
    let cropped = shape::pad(&x, &[0, 0, -1, -1, 0, 0, -1, -1], "constant", 0.0);
    assert_eq!(cropped.shape, vec![1, 1, 2, 2]);
    assert_eq!(cropped.data, vec![5.0, 6.0, 9.0, 10.0]);

    let x1d = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let wrapped = shape::pad(&x1d, &[2, 2], "wrap", 0.0);
    assert_eq!(wrapped.data, vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
}

// ── [a0-12] / [a10-6]: Unsqueeze normalizes against OUTPUT rank, rejects OOB/duplicates ─────

#[test]
fn unsqueeze_mixed_sign_axes_normalize_against_output_rank() -> Result<(), OnnxError> {
    // x shape [3,4], axes=[-1,-3] -> out_rank=4, normalized {1,3} -> [3,1,4,1].
    let x = Tensor::new((0..12).map(|i| i as f32).collect(), vec![3, 4]);
    let y = shape::unsqueeze(&x, &[-1, -3])?;
    assert_eq!(y.shape, vec![3, 1, 4, 1]);
    Ok(())
}

#[test]
fn unsqueeze_rejects_out_of_range_axis_without_panicking() {
    // Previously `Vec::insert` panicked ("insertion index ... should be <= len").
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let err = shape::unsqueeze(&x, &[100]);
    assert!(err.is_err());
}

#[test]
fn unsqueeze_rejects_duplicate_axes() {
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let err = shape::unsqueeze(&x, &[0, 0]);
    assert!(err.is_err());
}

// ── [a0-22]: empty tensors must not have their shape/data invariant corrupted by `.max(1)` ──

#[test]
fn flatten_zero_size_leading_dim_stays_zero() -> Result<(), OnnxError> {
    // x shape [0, 3] (0 elements); Flatten(axis=1) must give shape [0, 3], not [1, 3].
    let x = Tensor::new(Vec::new(), vec![0, 3]);
    let y = shape::flatten(&x, 1)?;
    assert_eq!(y.shape, vec![0, 3]);
    assert_eq!(y.data.len(), 0);
    Ok(())
}

#[test]
fn flatten_axis_equal_to_rank_is_legal() -> Result<(), OnnxError> {
    // ONNX Flatten accepts axis in the *inclusive* range [-r, r]; axis == r puts everything
    // into the outer dim (inner = 1).
    let x = Tensor::new((0..6).map(|i| i as f32).collect(), vec![2, 3]);
    let y = shape::flatten(&x, 2)?;
    assert_eq!(y.shape, vec![6, 1]);
    let err = shape::flatten(&x, 3);
    assert!(
        err.is_err(),
        "axis 3 exceeds the inclusive range for a rank-2 tensor"
    );
    Ok(())
}

#[test]
fn concat_zero_size_dim_preserves_shape() -> Result<(), OnnxError> {
    // Two tensors with a zero-size dim 0, concatenated along dim 1: outer product over an
    // empty prefix slice must stay the identity (1), not corrupt a real zero elsewhere.
    let a = Tensor::new(Vec::new(), vec![0, 2]);
    let b = Tensor::new(Vec::new(), vec![0, 3]);
    let c = shape::concat(&[&a, &b], 1)?;
    assert_eq!(c.shape, vec![0, 5]);
    assert_eq!(c.data.len(), 0);
    Ok(())
}

// ── [a0-23]: Split must emit exactly num_outputs outputs, including zero-size chunks ────────

#[test]
fn split_equal_emits_exactly_num_outputs_with_trailing_zero_chunk() -> Result<(), OnnxError> {
    // axis length 5, num_outputs=4 -> sizes [2,2,1,0]: 4 tensors, last one legitimately empty.
    let node = node_with_int_attrs(OpKind::Split, &[("axis", 0), ("num_outputs", 4)]);
    let x = Tensor::new((0..5).map(|i| i as f32).collect(), vec![5]);
    let ctx = make_ctx(&node, vec![Some(&x)]);
    let outputs = SplitOp.execute(&ctx)?;
    assert_eq!(outputs.len(), 4, "must produce exactly num_outputs tensors");
    let sizes: Vec<usize> = outputs.iter().map(|t| t.shape[0]).collect();
    assert_eq!(sizes, vec![2, 2, 1, 0]);
    assert_eq!(outputs[0].data, vec![0.0, 1.0]);
    assert_eq!(outputs[2].data, vec![4.0]);
    assert!(outputs[3].data.is_empty());

    // The output-slot path must agree, given 4 pre-allocated slots to match.
    let mut slots = vec![
        Tensor::new(Vec::new(), vec![0]),
        Tensor::new(Vec::new(), vec![0]),
        Tensor::new(Vec::new(), vec![0]),
        Tensor::new(Vec::new(), vec![0]),
    ];
    SplitOp.execute_into_slots(&ctx, &mut slots)?;
    let slot_sizes: Vec<usize> = slots.iter().map(|t| t.shape[0]).collect();
    assert_eq!(slot_sizes, vec![2, 2, 1, 0]);
    Ok(())
}

// ── [a10-7] / [a10-8] / [a0-24]: axis bounds checks across Concat/Split ─────────────────────

#[test]
fn concat_rejects_out_of_range_axis_on_identically_shaped_tensors() {
    // Same-shape tensors pass the per-dim mismatch guard; only the direct `out_shape[ax]`
    // index used to catch an out-of-range axis, and it did so by panicking.
    let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    let b = Tensor::new(vec![3.0, 4.0], vec![1, 2]);
    let err = shape::concat(&[&a, &b], 999);
    assert!(err.is_err());
}

#[test]
fn concat_op_execute_rejects_out_of_range_axis() {
    let node = node_with_int_attrs(OpKind::Concat, &[("axis", 999)]);
    let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    let b = Tensor::new(vec![3.0, 4.0], vec![1, 2]);
    let ctx = make_ctx(&node, vec![Some(&a), Some(&b)]);
    assert!(ConcatOp.execute(&ctx).is_err());
}

#[test]
fn split_op_execute_rejects_out_of_range_axis() {
    // [a10-8]: execute() indexed `x.shape[ax_u]` before any bounds check, unlike
    // execute_into_slots(); an out-of-range `axis` attribute must not panic via either path.
    let node = node_with_int_attrs(OpKind::Split, &[("axis", 7)]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let ctx = make_ctx(&node, vec![Some(&x)]);
    assert!(SplitOp.execute(&ctx).is_err());

    let mut slots = vec![Tensor::new(vec![0.0; 4], vec![2, 2])];
    assert!(SplitOp.execute_into_slots(&ctx, &mut slots).is_err());
}

#[test]
fn squeeze_rejects_out_of_range_axis() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let err = shape::squeeze(&x, &[10]);
    assert!(err.is_err());
}

#[test]
fn squeeze_op_execute_into_slots_rejects_out_of_range_axis() {
    let node = node_with_int_attrs(OpKind::Squeeze, &[]);
    let mut n = node;
    n.attrs.int_lists.insert("axes".into(), vec![10]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let ctx = make_ctx(&n, vec![Some(&x)]);
    assert!(SqueezeOp.execute(&ctx).is_err());
    let mut slots = vec![Tensor::new(vec![0.0; 3], vec![1, 3])];
    assert!(SqueezeOp.execute_into_slots(&ctx, &mut slots).is_err());
}

// ── [a10-5]: Transpose perm validation (bounds + genuine permutation) ───────────────────────

#[test]
fn transpose_op_rejects_out_of_range_perm_entry() {
    // perm=[0,0,99] on a 3D tensor: previously the raw i64 -> usize cast plus an unvalidated
    // `x.shape[p]` indexed out of bounds and panicked.
    let node = node_with_int_attrs(OpKind::Transpose, &[]);
    let mut n = node;
    n.attrs.int_lists.insert("perm".into(), vec![0, 0, 99]);
    let x = Tensor::new((0..8).map(|i| i as f32).collect(), vec![2, 2, 2]);
    let ctx = make_ctx(&n, vec![Some(&x)]);
    assert!(TransposeOp.execute(&ctx).is_err());
}

#[test]
fn transpose_op_rejects_negative_perm_entry() {
    let mut n = node_with_int_attrs(OpKind::Transpose, &[]);
    n.attrs.int_lists.insert("perm".into(), vec![-1, 1, 2]);
    let x = Tensor::new((0..8).map(|i| i as f32).collect(), vec![2, 2, 2]);
    let ctx = make_ctx(&n, vec![Some(&x)]);
    assert!(TransposeOp.execute(&ctx).is_err());
}

#[test]
fn transpose_rejects_repeated_perm_entry() {
    // perm=[0,0,1] is in-range per-entry but not a genuine permutation; letting it through
    // would build an out_shape whose product no longer matches x.numel().
    let x = Tensor::new((0..8).map(|i| i as f32).collect(), vec![2, 2, 2]);
    let err = shape::transpose(&x, &[0, 0, 1]);
    assert!(err.is_err());
}

#[test]
fn transpose_valid_perm_still_works() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let y = shape::transpose(&x, &[1, 0])?;
    assert_eq!(y.shape, vec![3, 2]);
    assert_eq!(y.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    Ok(())
}

// ── [a10-11]: Expand rejects negative shape entries instead of a capacity-overflow panic ────

#[test]
fn expand_op_execute_rejects_negative_shape_entry() {
    let node = dummy_node(OpKind::Expand);
    let x = Tensor::new(vec![1.0], vec![1, 1]);
    // -1 cast naively to usize would wrap to near-usize::MAX and blow up the allocation.
    let shape_t = Tensor::new(vec![-1.0, 4.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&shape_t)]);
    assert!(ExpandOp.execute(&ctx).is_err());
}

#[test]
fn expand_op_execute_into_slots_rejects_negative_shape_entry() {
    let node = dummy_node(OpKind::Expand);
    let x = Tensor::new(vec![1.0], vec![1, 1]);
    let shape_t = Tensor::new(vec![2.0, -8.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&shape_t)]);
    let mut slots = vec![Tensor::new(Vec::new(), vec![0])];
    assert!(ExpandOp.execute_into_slots(&ctx, &mut slots).is_err());
}

#[test]
fn expand_op_execute_still_broadcasts_normally() -> Result<(), OnnxError> {
    let node = dummy_node(OpKind::Expand);
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let shape_t = Tensor::new(vec![2.0, 3.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&shape_t)]);
    let out = ExpandOp.execute(&ctx)?;
    assert_eq!(out[0].shape, vec![2, 3]);
    assert_eq!(out[0].data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    Ok(())
}

// ── [a10-19]: Reshape shape entries below -1 are rejected, not wrapped ──────────────────────

#[test]
fn resolve_reshape_rejects_entries_below_negative_one() {
    // Previously `d as usize` on a raw `d < -1` wrapped to a huge value that then blew up
    // the `.product()` multiplication (overflow-panic in debug, silent wrap in release).
    let err = shape::resolve_reshape(&[2, 3], 6, &[-2, 3], false);
    assert!(err.is_err());

    let err2 = shape::resolve_reshape(&[2, 3], 6, &[i64::MIN, 3], false);
    assert!(err2.is_err());
}

#[test]
fn resolve_reshape_neg_one_inference_still_works() -> Result<(), OnnxError> {
    let resolved = shape::resolve_reshape(&[2, 3, 4], 24, &[-1, 4], false)?;
    assert_eq!(resolved, vec![6, 4]);
    Ok(())
}
