//! G1-ops-close regression tests.
//!
//! Covers, in order:
//! - `DepthToSpace`/`SpaceToDepth`: the `blocksize` i64-boundary fix
//!   (`registry/shape_ops/spatial_ops.rs`), specifically the slot-write
//!   (`execute_into_slots`) path, which previously had no `blocksize > 0`
//!   guard at all (unlike `execute()`, which at least rejected `blocksize ==
//!   0` via `shape::depth_to_space`/`shape::space_to_depth` -- see
//!   `w3_malformed_attrs.rs` for that pre-existing pin). Negative `blocksize`
//!   was unguarded on *both* paths, since the `as usize` cast happened before
//!   either guard ran.
//! - `ReduceSum`: an operator-level pin proving `execute()`/
//!   `execute_into_slots()` already reject an out-of-range `axes` entry (the
//!   primary regression test for the `reduce_output_shape` fix itself lives
//!   inline in `oxionnx-ops/src/math/reduce.rs`, since that function is
//!   `pub(crate)` and unreachable from here).
//! - `Round`: `execute()`, `execute_inplace()`, and `execute_into_slots()`
//!   must all agree on round-half-to-even.
//! - `Abs`/`Log`/`Exp`: the typed (`execute_typed`/`native_dtypes`) dtype
//!   boundary -- `Abs` is exact on `I32`/`I64`, `Log`/`Exp` are not and no
//!   longer claim to be.
//!
//! The RNN `direction` validation fix (also part of this wave) is instead
//! appended to `w3_malformed_attrs.rs`, alongside the pre-existing `LSTM`/
//! `GRU` siblings of the same regression.

use oxionnx_core::{
    Attributes, DType, Node, OpContext, OpKind, Operator, Tensor, TensorStorage, TypedOpContext,
    TypedTensor,
};
use oxionnx_ops::registry::math_ops::{ReduceSumOp, RoundOp};
use oxionnx_ops::registry::nn_ops::{AbsOp, ExpOp, LogOp};
use oxionnx_ops::registry::shape_ops::{DepthToSpaceOp, SpaceToDepthOp};

// ── Test infrastructure ──────────────────────────────────────────────────────

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

fn node_with_int_list_attrs(op: OpKind, lists: &[(&str, Vec<i64>)]) -> Node {
    let mut n = dummy_node(op);
    for (k, v) in lists {
        n.attrs.int_lists.insert(k.to_string(), v.clone());
    }
    n
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

fn typed_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a TypedTensor>>) -> TypedOpContext<'a> {
    TypedOpContext {
        node,
        inputs,
        outer_scope: None,
        registry: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DepthToSpace / SpaceToDepth: blocksize i64-boundary validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn depth_to_space_execute_rejects_negative_blocksize() {
    // `attrs.i("blocksize", 1) as usize` on `-1` used to wrap to a huge
    // `usize` instead of tripping `shape::depth_to_space`'s `blocksize == 0`
    // guard (which never saw a `0` -- it saw the wrapped huge value), so the
    // `channels % blocksize^2` check ran with an overflowing `r * r`.
    let node = node_with_int_attrs(OpKind::DepthToSpace, &[("blocksize", -1)]);
    let x = Tensor::new(vec![0.0; 16], vec![1, 4, 2, 2]);
    assert!(DepthToSpaceOp.execute(&ctx(&node, vec![Some(&x)])).is_err());
}

#[test]
fn depth_to_space_execute_into_slots_rejects_zero_blocksize() {
    // `execute_into_slots` duplicated the rearrangement math inline with no
    // `blocksize > 0` guard at all (unlike `execute()`, which delegated to
    // `shape::depth_to_space`'s guard) -- `r = 0` reached `c_total % (r * r)`
    // directly, a modulo-by-zero panic.
    let node = node_with_int_attrs(OpKind::DepthToSpace, &[("blocksize", 0)]);
    let x = Tensor::new(vec![0.0; 16], vec![1, 4, 2, 2]);
    let mut slots = vec![Tensor::new(vec![0.0; 16], vec![1, 4, 2, 2])];
    assert!(DepthToSpaceOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .is_err());
}

#[test]
fn depth_to_space_execute_into_slots_rejects_negative_blocksize() {
    let node = node_with_int_attrs(OpKind::DepthToSpace, &[("blocksize", -3)]);
    let x = Tensor::new(vec![0.0; 16], vec![1, 4, 2, 2]);
    let mut slots = vec![Tensor::new(vec![0.0; 16], vec![1, 4, 2, 2])];
    assert!(DepthToSpaceOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .is_err());
}

#[test]
fn space_to_depth_execute_rejects_negative_blocksize() {
    let node = node_with_int_attrs(OpKind::SpaceToDepth, &[("blocksize", -1)]);
    let x = Tensor::new(vec![0.0; 16], vec![1, 1, 4, 4]);
    assert!(SpaceToDepthOp.execute(&ctx(&node, vec![Some(&x)])).is_err());
}

#[test]
fn space_to_depth_execute_into_slots_rejects_zero_blocksize() {
    let node = node_with_int_attrs(OpKind::SpaceToDepth, &[("blocksize", 0)]);
    let x = Tensor::new(vec![0.0; 16], vec![1, 1, 4, 4]);
    let mut slots = vec![Tensor::new(vec![0.0; 16], vec![1, 1, 4, 4])];
    assert!(SpaceToDepthOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .is_err());
}

#[test]
fn space_to_depth_execute_into_slots_rejects_negative_blocksize() {
    let node = node_with_int_attrs(OpKind::SpaceToDepth, &[("blocksize", -3)]);
    let x = Tensor::new(vec![0.0; 16], vec![1, 1, 4, 4]);
    let mut slots = vec![Tensor::new(vec![0.0; 16], vec![1, 1, 4, 4])];
    assert!(SpaceToDepthOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .is_err());
}

#[test]
fn depth_to_space_execute_into_slots_still_accepts_valid_blocksize() {
    // Regression guard alongside the rejections above: a valid blocksize must
    // still work through the now-guarded slot path. [1,4,2,2], blocksize=2,
    // DCR -> [1,1,4,4], matching `output_slots_shape_test.rs`'s coverage of
    // the happy path (kept minimal here since that file already owns it).
    let node = {
        let mut n = node_with_int_attrs(OpKind::DepthToSpace, &[("blocksize", 2)]);
        n.attrs.strings.insert("mode".into(), "DCR".into());
        n
    };
    let x = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 4, 2, 2]);
    let mut slots = vec![Tensor::new(vec![0.0; 16], vec![1, 1, 4, 4])];
    DepthToSpaceOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .expect("valid blocksize must still succeed");
    assert_eq!(slots[0].shape, vec![1, 1, 4, 4]);
}

// ═══════════════════════════════════════════════════════════════════════════
// ReduceSum: operator-level pin for an out-of-range axis
//
// The primary regression test for the `reduce_output_shape` fix is inline in
// `oxionnx-ops/src/math/reduce.rs` (that function is `pub(crate)`). Both
// `execute()` and `execute_into_slots()` were *already* correct before that
// fix -- `reduce_sum`/`reduce_sum_into` re-validate `axes` themselves via
// `reduce_with`/`reduce_with_into`'s own strict `normalize_axis`, independent
// of `reduce_output_shape`'s (formerly silently-permissive) shape hint -- so
// these two assertions pass identically before and after the fix. They are
// kept here as a pin on observable operator behavior, not as proof of the
// fix itself.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reduce_sum_execute_rejects_out_of_range_axis() {
    let node = node_with_int_list_attrs(OpKind::ReduceSum, &[("axes", vec![5i64])]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(ReduceSumOp.execute(&ctx(&node, vec![Some(&x)])).is_err());
}

#[test]
fn reduce_sum_execute_into_slots_rejects_out_of_range_axis() {
    let node = node_with_int_list_attrs(OpKind::ReduceSum, &[("axes", vec![5i64])]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let mut slots = vec![Tensor::new(vec![-999.0; 6], vec![2, 3])];
    assert!(ReduceSumOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Round: execute() / execute_inplace() / execute_into_slots() must agree on
// round-half-to-even (the ONNX spec's tie-breaking rule).
//
// Before this fix, `execute()` (via `math::round_op`) used round-half-to-even
// while `execute_inplace()`/`execute_into_slots()` (both driven by the same
// `$inplace_fn` macro argument) used bare `f32::round`, which is
// round-half-*away*-from-zero -- e.g. `2.5 -> 3.0` instead of the spec's
// `2.0`. All three paths now share `math::round_half_to_even`.
// ═══════════════════════════════════════════════════════════════════════════

/// (input, expected round-half-to-even output) pairs. `f32::round` (the old
/// `execute_inplace`/`execute_into_slots` behavior) would instead give
/// (3.0, 4.0, -3.0, -4.0, 1.0, 2.0, -1.0, -2.0) for the first eight entries
/// -- every one of them wrong -- since it rounds every exact half away from
/// zero regardless of even/odd.
const ROUND_HALF_CASES: &[(f32, f32)] = &[
    (2.5, 2.0),
    (3.5, 4.0),
    (-2.5, -2.0),
    (-3.5, -4.0),
    (0.5, 0.0),
    (1.5, 2.0),
    (-0.5, 0.0),
    (-1.5, -2.0),
    // Non-half values: both rounding rules must agree here.
    (2.4, 2.0),
    (2.6, 3.0),
    (-2.4, -2.0),
    (-2.6, -3.0),
    (0.0, 0.0),
];

fn round_inputs_and_expected() -> (Vec<f32>, Vec<f32>) {
    (
        ROUND_HALF_CASES.iter().map(|&(v, _)| v).collect(),
        ROUND_HALF_CASES.iter().map(|&(_, e)| e).collect(),
    )
}

#[test]
fn round_execute_matches_spec_half_to_even() {
    let (inputs, expected) = round_inputs_and_expected();
    let node = dummy_node(OpKind::Round);
    let x = Tensor::new(inputs, vec![ROUND_HALF_CASES.len()]);
    let out = RoundOp
        .execute(&ctx(&node, vec![Some(&x)]))
        .expect("Round execute");
    assert_eq!(out[0].data, expected);
}

#[test]
fn round_execute_inplace_matches_spec_half_to_even() {
    let (inputs, expected) = round_inputs_and_expected();
    let node = dummy_node(OpKind::Round);
    let x = Tensor::new(inputs, vec![ROUND_HALF_CASES.len()]);
    // `execute_inplace`'s `ctx` is unused by `RoundOp` (input 0 is consumed
    // as the owned `Tensor` argument instead), so slot 0 is `None` per the
    // trait's documented contract.
    let out = RoundOp
        .execute_inplace(x, &ctx(&node, vec![None]))
        .expect("Round execute_inplace");
    assert_eq!(out[0].data, expected);
}

#[test]
fn round_execute_into_slots_matches_spec_half_to_even() {
    let (inputs, expected) = round_inputs_and_expected();
    let node = dummy_node(OpKind::Round);
    let n = ROUND_HALF_CASES.len();
    let x = Tensor::new(inputs, vec![n]);
    let mut slots = vec![Tensor::new(vec![0.0; n], vec![n])];
    RoundOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .expect("Round execute_into_slots");
    assert_eq!(slots[0].data, expected);
}

#[test]
fn round_execute_and_execute_inplace_agree_bit_for_bit() {
    // Broader cross-check beyond the pinned half-values above: every path
    // must produce identical output for the same input, not just on the
    // specific values the spec's tie-breaking rule is about.
    let values: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect();
    let node = dummy_node(OpKind::Round);
    let x = Tensor::new(values.clone(), vec![values.len()]);
    let via_execute = RoundOp
        .execute(&ctx(&node, vec![Some(&x)]))
        .expect("Round execute");
    let via_inplace = RoundOp
        .execute_inplace(x, &ctx(&node, vec![None]))
        .expect("Round execute_inplace");
    assert_eq!(via_execute[0].data, via_inplace[0].data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Abs / Log / Exp: the typed (execute_typed / native_dtypes) dtype boundary
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn abs_execute_typed_i64_is_exact_no_f32_roundtrip() {
    // -123_456_789_012_345 exceeds f32's 24-bit exact-integer range: an f32
    // round trip corrupts it to -123_456_788_103_168 (`(v as f32) as i64 !=
    // v`), so an exact result here proves the I64 arm never touched f32.
    let v: i64 = -123_456_789_012_345;
    let tt = TypedTensor::new(TensorStorage::I64(vec![v]), vec![1]);
    let node = dummy_node(OpKind::Abs);
    let results = AbsOp
        .execute_typed(&typed_ctx(&node, vec![Some(&tt)]))
        .expect("Abs execute_typed I64");
    match &results[0].storage {
        TensorStorage::I64(data) => assert_eq!(data, &[123_456_789_012_345i64]),
        other => panic!("expected I64 storage, got {other:?}"),
    }
}

#[test]
fn abs_execute_typed_i32_is_exact_no_f32_roundtrip() {
    let v: i32 = -123_456_789;
    let tt = TypedTensor::new(TensorStorage::I32(vec![v]), vec![1]);
    let node = dummy_node(OpKind::Abs);
    let results = AbsOp
        .execute_typed(&typed_ctx(&node, vec![Some(&tt)]))
        .expect("Abs execute_typed I32");
    match &results[0].storage {
        TensorStorage::I32(data) => assert_eq!(data, &[123_456_789i32]),
        other => panic!("expected I32 storage, got {other:?}"),
    }
}

#[test]
fn abs_execute_typed_i64_min_does_not_panic_and_wraps_like_neg() {
    // i64::MIN has no positive representation in i64 (two's complement);
    // `wrapping_abs` returns it unchanged instead of panicking, matching
    // `NegOp`'s established `wrapping_neg` precedent for the identical edge
    // case (`registry/math_ops/elementwise.rs`).
    let tt = TypedTensor::new(TensorStorage::I64(vec![i64::MIN]), vec![1]);
    let node = dummy_node(OpKind::Abs);
    let results = AbsOp
        .execute_typed(&typed_ctx(&node, vec![Some(&tt)]))
        .expect("Abs execute_typed I64::MIN must not panic");
    match &results[0].storage {
        TensorStorage::I64(data) => assert_eq!(data, &[i64::MIN]),
        other => panic!("expected I64 storage, got {other:?}"),
    }
}

#[test]
fn abs_execute_typed_f32_still_uses_f32_fallback() {
    // Non-integer storage must still go through `default_typed_via_f32`
    // (unaffected by the new I32/I64 arms).
    let tt = TypedTensor::new(TensorStorage::F32(vec![-3.5]), vec![1]);
    let node = dummy_node(OpKind::Abs);
    let results = AbsOp
        .execute_typed(&typed_ctx(&node, vec![Some(&tt)]))
        .expect("Abs execute_typed F32");
    match &results[0].storage {
        TensorStorage::F32(data) => assert_eq!(data, &[3.5f32]),
        other => panic!("expected F32 storage, got {other:?}"),
    }
}

#[test]
fn abs_native_dtypes_still_includes_i32_i64() {
    // Regression guard: unlike Log/Exp below, Abs genuinely can execute
    // I32/I64 exactly now, so it must keep advertising them.
    let dtypes = AbsOp.native_dtypes();
    assert!(dtypes.contains(&DType::F32));
    assert!(dtypes.contains(&DType::I32));
    assert!(dtypes.contains(&DType::I64));
}

#[test]
fn log_native_dtypes_excludes_i32_i64_but_keeps_float_family() {
    let dtypes = LogOp.native_dtypes();
    assert!(dtypes.contains(&DType::F32));
    assert!(dtypes.contains(&DType::F16));
    assert!(dtypes.contains(&DType::BF16));
    assert!(
        !dtypes.contains(&DType::I32),
        "Log is real-valued: ln(i) has no exact I32 result, so native_dtypes must not claim it"
    );
    assert!(
        !dtypes.contains(&DType::I64),
        "Log is real-valued: ln(i) has no exact I64 result, so native_dtypes must not claim it"
    );
}

#[test]
fn exp_native_dtypes_excludes_i32_i64_but_keeps_float_family() {
    let dtypes = ExpOp.native_dtypes();
    assert!(dtypes.contains(&DType::F32));
    assert!(dtypes.contains(&DType::F16));
    assert!(dtypes.contains(&DType::BF16));
    assert!(
        !dtypes.contains(&DType::I32),
        "Exp is real-valued: exp(i) has no exact I32 result, so native_dtypes must not claim it"
    );
    assert!(
        !dtypes.contains(&DType::I64),
        "Exp is real-valued: exp(i) has no exact I64 result, so native_dtypes must not claim it"
    );
}
