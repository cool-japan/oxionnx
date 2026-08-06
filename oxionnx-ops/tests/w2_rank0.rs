//! Wave-2 `W2-rank0`: rank-0 (scalar, shape `[]`) tensors at the operator layer.
//!
//! ONNX distinguishes a rank-0 tensor from the rank-1 single-element tensor
//! `[1]`. The core tensor layer (`oxionnx-core/src/tensor/**`) now treats shape
//! `[]` as a first-class value — see the `oxionnx_core::tensor` module docs for
//! the contract — and this file pins down what the operator layer does with it.
//!
//! It has two halves:
//!
//! 1. **Rank-0 flows correctly** through the consuming half of the operator
//!    layer — `Shape`, `Reshape`, `Unsqueeze`, `Gather`, `Where`, the
//!    elementwise ops. Those tests exist so the behaviour cannot silently
//!    regress.
//! 2. The rank-**producing** ops — `Squeeze`, `ReduceX`, `ArgMax`/`ArgMin`,
//!    `GatherND`, `Size`, `Constant` — which finding `[a0-21]` recorded as
//!    still promoting an emptied output shape to `[1]`. Wave-3 migrated all of
//!    them; these tests now assert the opset-21 result rather than the
//!    pre-migration behaviour they were originally written to characterize.
//!
//! Reference values are NumPy's, whose rank-0 arrays implement the semantics
//! ONNX specifies. Computed with `python3`:
//!
//! ```text
//! np.squeeze(np.array([5.0]), axis=0).shape                          -> ()
//! len(())                                                             -> 0    # ONNX Shape of a scalar
//! np.sum(np.arange(24).reshape(2,3,4), axis=(0,1,2), keepdims=False)  -> shape (), 276.0
//! np.expand_dims(<that>, 0).shape                                     -> (1,)
//! np.argmax(np.array([3.,9.,4.]), axis=0)                             -> shape (), 1
//! np.array(7.0) + np.arange(6).reshape(2,3)                           -> shape (2,3), [7..12]
//! ```

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::{OnnxError, Tensor};
use oxionnx_ops::registry::misc_ops::{ShapeOp, SizeOp};
use oxionnx_ops::{indexing, math, shape};

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

fn dummy_node(op_type: &str) -> Node {
    Node {
        name: "test".into(),
        op: OpKind::parse(op_type),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

/// The empty shape, spelled once so the assertions below stay readable.
fn rank0_shape() -> Vec<usize> {
    Vec::new()
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 1 — rank-0 already flows correctly; lock it in
// ═══════════════════════════════════════════════════════════════════════════

/// `Shape` of a rank-0 tensor is the **empty** 1-D vector: a length-0 tensor of
/// shape `[0]`, not `[1]`. This is the observable that makes the whole rank-0
/// distinction matter — everything downstream (`Reshape`, `Concat`) is driven by
/// this vector. `ShapeOp` derives its length from `x.shape.len()`, so it is
/// already correct; what it needs is an input that is genuinely rank 0.
#[test]
fn shape_of_rank0_is_the_empty_vector() -> Result<(), OnnxError> {
    let x = Tensor::rank0(7.0);
    let node = dummy_node("Shape");
    let out = ShapeOp.execute(&make_ctx(&node, vec![Some(&x)]))?;

    assert_eq!(out[0].shape, vec![0], "a length-0 vector has shape [0]");
    assert_eq!(out[0].data, Vec::<f32>::new());
    assert_eq!(out[0].numel(), 0);

    // Contrast: the legacy `[1]` scalar reports a length-1 vector holding 1.
    let legacy = Tensor::scalar(7.0);
    let out = ShapeOp.execute(&make_ctx(&node, vec![Some(&legacy)]))?;
    assert_eq!(out[0].shape, vec![1]);
    assert_eq!(out[0].data, vec![1.0]);
    Ok(())
}

/// `Reshape` to an empty target shape produces a genuine rank-0 tensor, so a
/// model that round-trips through `Shape`/`Reshape` can express rank 0 today.
#[test]
fn reshape_to_empty_shape_produces_rank0() -> Result<(), String> {
    let one = Tensor::new(vec![5.0], vec![1]);
    let out = shape::reshape(&one, &[], false)?;
    assert_eq!(out.shape, rank0_shape());
    assert_eq!(out.data, vec![5.0]);

    // And back the other way.
    let back = shape::reshape(&out, &[1], false)?;
    assert_eq!(back.shape, vec![1]);
    Ok(())
}

/// `Unsqueeze(axes=[0])` lifts a rank-0 tensor to `[1]` — the output rank is
/// `input_rank + len(axes)`, which is already computed correctly.
/// NumPy: `np.expand_dims(np.array(7.0), 0).shape == (1,)`.
#[test]
fn unsqueeze_lifts_rank0_to_rank1() -> Result<(), String> {
    let out = shape::unsqueeze(&Tensor::rank0(7.0), &[0])?;
    assert_eq!(out.shape, vec![1]);
    assert_eq!(out.data, vec![7.0]);
    Ok(())
}

/// `Transpose` of a rank-0 tensor with an empty perm is the identity, and
/// `Flatten(axis=0)` of a scalar is `[1, 1]` (both the leading and trailing
/// slices are empty, and each multiplies to the identity 1).
#[test]
fn transpose_and_flatten_handle_rank0() -> Result<(), String> {
    let x = Tensor::rank0(7.0);

    let t = shape::transpose(&x, &[])?;
    assert_eq!(t.shape, rank0_shape());
    assert_eq!(t.data, vec![7.0]);

    let f = shape::flatten(&x, 0)?;
    assert_eq!(f.shape, vec![1, 1]);
    assert_eq!(f.data, vec![7.0]);
    Ok(())
}

/// Elementwise binary ops broadcast a rank-0 operand against any shape without
/// raising the output rank, in either argument position, and rank-0 with rank-0
/// stays rank 0. NumPy: `np.array(7.0) + np.arange(6).reshape(2,3)` has shape
/// `(2,3)` and values `[7,8,9,10,11,12]`.
#[test]
fn elementwise_ops_broadcast_rank0_without_raising_rank() -> Result<(), String> {
    let scalar = Tensor::rank0(7.0);
    let mat = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);

    let left = math::add(&scalar, &mat)?;
    assert_eq!(left.shape, vec![2, 3]);
    assert_eq!(left.data, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let right = math::add(&mat, &scalar)?;
    assert_eq!(right.shape, vec![2, 3]);
    assert_eq!(right.data, left.data);

    // rank 0 + rank 0 stays rank 0 (a `[1]` operand would raise it to `[1]`).
    let both = math::add(&scalar, &Tensor::rank0(3.0))?;
    assert_eq!(both.shape, rank0_shape());
    assert_eq!(both.data, vec![10.0]);

    let quotient = math::div(&scalar, &Tensor::rank0(2.0))?;
    assert_eq!(quotient.shape, rank0_shape());
    assert_eq!(quotient.data, vec![3.5]);
    Ok(())
}

/// Unary ops preserve rank 0 rather than promoting it.
#[test]
fn unary_ops_preserve_rank0() {
    let s = math::sqrt(&Tensor::rank0(9.0));
    assert_eq!(s.shape, rank0_shape());
    assert_eq!(s.data, vec![3.0]);

    let n = math::neg(&Tensor::rank0(7.0));
    assert_eq!(n.shape, rank0_shape());
    assert_eq!(n.data, vec![-7.0]);
}

/// `Gather` with rank-0 indices drops the gathered axis entirely: output rank is
/// `data_rank + indices_rank - 1`, so `[3]` gathered by a scalar index is rank 0.
#[test]
fn gather_with_rank0_indices_produces_rank0() -> Result<(), String> {
    let data = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let out = indexing::gather(&data, &Tensor::rank0(1.0), 0)?;
    assert_eq!(out.shape, rank0_shape());
    assert_eq!(out.data, vec![20.0]);
    Ok(())
}

/// `Expand` and `Where` accept rank-0 operands, and `Where` on three rank-0
/// operands stays rank 0.
#[test]
fn expand_and_where_handle_rank0() -> Result<(), String> {
    let expanded = indexing::expand(&Tensor::rank0(7.0), &[2, 3])?;
    assert_eq!(expanded.shape, vec![2, 3]);
    assert_eq!(expanded.data, vec![7.0; 6]);

    let picked = indexing::where_op(
        &Tensor::rank0(1.0),
        &Tensor::rank0(2.0),
        &Tensor::rank0(3.0),
    )?;
    assert_eq!(picked.shape, rank0_shape());
    assert_eq!(picked.data, vec![2.0]);
    Ok(())
}

/// `Identity` and `Cast` are pure pass-throughs and must not re-rank a scalar,
/// and `Clip` accepts rank-0 `min`/`max` bounds (they are declared scalars in
/// the spec).
#[test]
fn registry_passthroughs_and_clip_accept_rank0() -> Result<(), OnnxError> {
    let registry = oxionnx_ops::default_registry();
    let x = Tensor::rank0(7.0);

    for op_type in ["Identity", "Cast"] {
        let node = dummy_node(op_type);
        let op = registry
            .get(op_type)
            .ok_or_else(|| OnnxError::Internal(format!("{op_type} not registered")))?;
        let out = op.execute(&make_ctx(&node, vec![Some(&x)]))?;
        assert_eq!(
            out[0].shape,
            rank0_shape(),
            "{op_type} must not re-rank a scalar"
        );
        assert_eq!(out[0].data, vec![7.0]);
    }

    let node = dummy_node("Clip");
    let op = registry
        .get("Clip")
        .ok_or_else(|| OnnxError::Internal("Clip not registered".into()))?;
    let values = Tensor::new(vec![-1.0, 3.0, 9.0], vec![3]);
    let lo = Tensor::rank0(0.0);
    let hi = Tensor::rank0(5.0);
    let out = op.execute(&make_ctx(&node, vec![Some(&values), Some(&lo), Some(&hi)]))?;
    assert_eq!(out[0].shape, vec![3]);
    assert_eq!(out[0].data, vec![0.0, 3.0, 5.0]);
    Ok(())
}

/// Operators whose axis attribute has no valid value at rank 0 must report a
/// typed error, not panic and not silently invent an axis. `Concat` requires
/// rank >= 1 and `CumSum`'s axis range `[-r, r-1]` is empty at `r == 0`.
#[test]
fn rank0_is_rejected_where_no_axis_exists() {
    let x = Tensor::rank0(7.0);
    assert!(
        shape::concat(&[&x, &x], 0).is_err(),
        "Concat has no axis to concatenate along at rank 0"
    );
    assert!(
        math::cumsum(&x, 0, false, false).is_err(),
        "CumSum has no axis to accumulate along at rank 0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 2 — [a0-21] the rank-producing ops, migrated in Wave-3
//
// Each of these used to promote an emptied output shape to `[1]`. They now
// return it unchanged, so a model that reduces, squeezes or sizes its way down
// to a single value gets the rank-0 tensor ONNX specifies. Reference values are
// NumPy's, quoted per test.
// ═══════════════════════════════════════════════════════════════════════════

/// [a0-21] `Squeeze` that removes every axis yields rank 0.
///
/// ONNX opset-21 / NumPy: `np.squeeze(np.array([5.0]), axis=0).shape == ()`.
/// `resolve_squeeze_shape` used to end with
/// `Ok(if new_shape.is_empty() { vec![1] } else { new_shape })`, promoting the
/// emptied shape to `[1]`; it now returns it unchanged.
#[test]
fn squeeze_to_scalar_is_rank0() -> Result<(), String> {
    let x = Tensor::new(vec![5.0], vec![1]);
    let out = shape::squeeze(&x, &[0])?;

    assert_eq!(out.data, vec![5.0]);
    assert_eq!(out.shape, rank0_shape());

    // The axes-less form ("drop every size-1 axis") reaches the same tail.
    let all = shape::squeeze(&Tensor::new(vec![5.0], vec![1, 1, 1]), &[])?;
    assert_eq!(all.shape, rank0_shape());
    assert_eq!(all.data, vec![5.0]);
    Ok(())
}

/// [a0-21] The consequence the finding named: `Squeeze` -> `Shape`.
///
/// This is the observable that makes the migration matter. A correct rank-0
/// squeeze makes the following `Shape` node emit the **empty** (length-0)
/// vector; under the old `[1]` promotion it emitted the length-1 vector `[1]`,
/// and any `Reshape`/`Concat` driven by it got one dimension too many.
/// `ShapeOp` itself needed no change (see `shape_of_rank0_is_the_empty_vector`).
#[test]
fn squeeze_then_shape_is_the_empty_vector() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![5.0], vec![1]);
    let squeezed = shape::squeeze(&x, &[0]).map_err(OnnxError::Internal)?;

    let node = dummy_node("Shape");
    let out = ShapeOp.execute(&make_ctx(&node, vec![Some(&squeezed)]))?;

    assert_eq!(out[0].data, Vec::<f32>::new(), "no axes to report");
    assert_eq!(out[0].shape, vec![0], "a length-0 vector has shape [0]");
    Ok(())
}

/// [a0-21] `ReduceSum(keepdims=0)` over every axis yields rank 0.
///
/// NumPy: `np.sum(np.arange(24).reshape(2,3,4), axis=(0,1,2), keepdims=False)`
/// has shape `()` and value `276.0`.
///
/// Several sites in oxionnx-ops/src/math/reduce.rs used to promote the emptied
/// shape to `vec![1]`: `reduce_output_shape`, `reduce_with_into`, `reduce_with`,
/// the `simd`-gated full-reduction fast paths in `reduce_{mean,sum,max,min}` and
/// their `_into` twins. **Which of those runs depends on the `simd` feature**,
/// so the empty-axes spelling below (`&[]`, "reduce every axis") is asserted
/// alongside the explicit one: both are full reductions, and under `simd` both
/// take the fast path that the explicit `[0,1,2]` list also takes.
#[test]
fn reduce_all_axes_keepdims0_is_rank0() -> Result<(), String> {
    let y = Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);

    let out = math::reduce_sum(&y, &[0, 1, 2], false)?;
    assert_eq!(out.data, vec![276.0]);
    assert_eq!(out.shape, rank0_shape());

    // Same reduction, spelled as "no axes given".
    let implicit = math::reduce_sum(&y, &[], false)?;
    assert_eq!(implicit.data, vec![276.0]);
    assert_eq!(implicit.shape, rank0_shape());

    // The other three SIMD-shortcut kinds take the same tail.
    assert_eq!(math::reduce_max(&y, &[], false)?.shape, rank0_shape());
    assert_eq!(math::reduce_min(&y, &[], false)?.shape, rank0_shape());
    let mean = math::reduce_mean(&y, &[], false)?;
    assert_eq!(mean.shape, rank0_shape());
    assert_eq!(mean.data, vec![11.5]); // np.arange(24).mean() == 11.5

    // `keepdims=1` is untouched by the migration: all axes collapse to 1.
    assert_eq!(math::reduce_sum(&y, &[], true)?.shape, vec![1, 1, 1]);
    Ok(())
}

/// [a0-21] The second consequence the finding named: `ReduceSum(keepdims=0)`
/// over all axes followed by `Unsqueeze(axes=[0])`.
///
/// NumPy: reducing gives shape `()`, and `np.expand_dims(<that>, 0).shape` is
/// `(1,)`. When the reduction stopped at `[1]` the unsqueeze lifted it to
/// `[1, 1]` — one rank too many, exactly as the finding described. `Unsqueeze`
/// itself needed no change (see `unsqueeze_lifts_rank0_to_rank1`).
#[test]
fn reduce_then_unsqueeze_is_rank1() -> Result<(), String> {
    let y = Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
    let reduced = math::reduce_sum(&y, &[0, 1, 2], false)?;
    let out = shape::unsqueeze(&reduced, &[0])?;

    assert_eq!(out.data, vec![276.0]);
    assert_eq!(out.shape, vec![1]);
    Ok(())
}

/// [a0-21] `ArgMax`/`ArgMin` with `keepdims=0` on a 1-D input must yield rank 0.
///
/// NumPy: `np.argmax(np.array([3.,9.,4.]), axis=0)` has shape `()` and value 1;
/// `np.argmin` gives shape `()` and value 0.
///
/// `arg_reduce` and `arg_output_shape` (oxionnx-ops/src/math/argminmax.rs) used
/// to promote the emptied shape to `vec![1]`; both now return it unchanged.
#[test]
fn arg_reduce_keepdims0_on_1d_is_rank0() -> Result<(), String> {
    let z = Tensor::new(vec![3.0, 9.0, 4.0], vec![3]);

    let out = math::arg_max(&z, 0, false, false)?;
    assert_eq!(out.data, vec![1.0]);
    assert_eq!(out.shape, rank0_shape());

    let out = math::arg_min(&z, 0, false, false)?;
    assert_eq!(out.data, vec![0.0]);
    assert_eq!(out.shape, rank0_shape());

    // `keepdims=1` is untouched: the axis collapses to 1 rather than vanishing.
    assert_eq!(math::arg_max(&z, 0, true, false)?.shape, vec![1]);
    Ok(())
}

/// [a0-21] `GatherND` whose output shape works out empty must yield rank 0.
///
/// With `data` of shape `[3]` and `indices` of shape `[1]` (its last dimension
/// equal to the data rank, so the index addresses a full element), the output
/// shape is `indices.shape[:-1] + data.shape[1:]`, i.e. `[]`.
///
/// `gather_nd` (oxionnx-ops/src/indexing/gather.rs) used to end with
/// `if out_shape.is_empty() { out_shape.push(1); }`; that promotion is gone.
#[test]
fn gather_nd_to_scalar_is_rank0() -> Result<(), String> {
    let data = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let indices = Tensor::new(vec![1.0], vec![1]);
    let out = indexing::gather_nd(&data, &indices, 0)?;

    assert_eq!(out.data, vec![20.0]);
    assert_eq!(out.shape, rank0_shape());

    // A partial index still leaves the trailing data axes behind, unchanged by
    // the migration: `[2,2]` indexed by `[1]` (K=1) keeps `data.shape[1:]`.
    let mat = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let row = indexing::gather_nd(&mat, &Tensor::new(vec![1.0], vec![1]), 0)?;
    assert_eq!(row.shape, vec![2]);
    assert_eq!(row.data, vec![3.0, 4.0]);
    Ok(())
}

/// [a0-21] `Size` is specified to return a **scalar** ("outputs an int64 scalar
/// that equals the total number of elements of the input tensor"), so its output
/// is rank 0 for every input.
///
/// `SizeOp` (oxionnx-ops/src/registry/misc_ops.rs) used to build
/// `Tensor::new(vec![n as f32], vec![1])` in `execute` and set
/// `slots[0].shape = vec![1]` in `execute_into_slots`; both now use the empty
/// shape. The two paths are held equal by
/// `oxionnx-ops/tests/output_slots_f14_test.rs::test_size_slot`.
#[test]
fn size_output_is_rank0() -> Result<(), OnnxError> {
    let node = dummy_node("Size");

    // Rank-0 input: 1 element. The count itself was already right, because
    // `Tensor::numel` reads the data buffer rather than clamping a shape product.
    let x = Tensor::rank0(7.0);
    let out = SizeOp.execute(&make_ctx(&node, vec![Some(&x)]))?;
    assert_eq!(out[0].data, vec![1.0], "a scalar holds exactly one element");
    assert_eq!(out[0].shape, rank0_shape());

    // Higher-rank input: the output rank is 0 regardless of the input rank.
    let y = Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
    let out = SizeOp.execute(&make_ctx(&node, vec![Some(&y)]))?;
    assert_eq!(out[0].data, vec![24.0]);
    assert_eq!(out[0].shape, rank0_shape());
    Ok(())
}
