//! Wave-1 correctness regression tests for the `G-activations-misc` domain
//! (oxionnx-ops/src/registry/nn_ops/{activations,parameterized}.rs,
//! oxionnx-ops/src/registry/misc_ops.rs, oxionnx-ops/src/math/{variadic,argminmax}.rs,
//! oxionnx-ops/src/comparison.rs, oxionnx-ops/src/bitwise.rs,
//! oxionnx-ops/src/registry/math_ops/reduce.rs, and the GroupNorm scale/bias mapping
//! in oxionnx-ops/src/nn/normalization.rs).
//!
//! Each test is tied to a specific audit finding and carries a hand- (or
//! python3/`math`-) derived reference value, not a value copied from the
//! implementation under test. See the per-section comments for the finding ID.

use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::{OpContext, Operator},
    Tensor,
};
use oxionnx_ops::registry::math_ops::{
    ArgMaxOp, ArgMinOp, BitShiftOp, ModOp, ReduceMeanOp, ReduceSumOp,
};
use oxionnx_ops::registry::misc_ops::{BitwiseAndOp, BitwiseNotOp, CastOp, EqualOp, ShapeOp};
use oxionnx_ops::registry::nn_ops::{ClipOp, DropoutOp, GeluOp, GroupNormOp};

// ── Test infrastructure (mirrors the pattern used by sibling w1_*.rs files) ──

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

fn int(node: &mut Node, name: &str, value: i64) {
    node.attrs.ints.insert(name.into(), value);
}

fn float(node: &mut Node, name: &str, value: f32) {
    node.attrs.floats.insert(name.into(), value);
}

fn string(node: &mut Node, name: &str, value: &str) {
    node.attrs.strings.insert(name.into(), value.into());
}

/// Run an operator through both `execute` and `execute_into_slots` (when
/// supported) and assert the two dispatch paths agree, returning the
/// `execute` result. `node.outputs` must be sized to the operator's actual
/// output count for the slot path to line up.
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
        assert_eq!(
            slots.len(),
            direct.len(),
            "slot count vs execute() output count"
        );
        for (i, expected) in direct.iter().enumerate() {
            assert_eq!(slots[i].shape, expected.shape, "slot[{i}] shape parity");
            assert_eq!(slots[i].data, expected.data, "slot[{i}] data parity");
        }
    }
    direct
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: element count");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < tol, "{label}[{i}]: got {g}, expected {w}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// [a0-2] Clip: opset-6 min/max ATTRIBUTES when inputs 1/2 are absent
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn clip_opset6_uses_min_max_attributes_when_inputs_absent() {
    // MobileNetV2/V3-style Relu6: Clip(x, min=0.0, max=6.0) in ATTRIBUTE form
    // (opset < 11), i.e. no input[1]/input[2] tensors at all.
    let mut node = node_with(OpKind::Clip, &["y"]);
    float(&mut node, "min", 0.0);
    float(&mut node, "max", 6.0);
    let x = Tensor::new(vec![-3.0, 0.0, 3.0, 6.0, 9.0], vec![5]);
    let out = run_both(&ClipOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![0.0, 0.0, 3.0, 6.0, 6.0]);
}

#[test]
fn clip_no_bounds_at_all_is_identity() {
    let node = node_with(OpKind::Clip, &["y"]);
    let x = Tensor::new(vec![-3.0, 0.0, 9.0], vec![3]);
    let out = run_both(&ClipOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![-3.0, 0.0, 9.0]);
}

#[test]
fn clip_opset11_input_tensors_still_work() {
    let node = node_with(OpKind::Clip, &["y"]);
    let x = Tensor::new(vec![-3.0, 0.0, 3.0, 6.0, 9.0], vec![5]);
    let min_t = Tensor::new(vec![0.0], vec![1]);
    let max_t = Tensor::new(vec![6.0], vec![1]);
    let out = run_both(&ClipOp, &node, vec![Some(&x), Some(&min_t), Some(&max_t)]);
    assert_eq!(out[0].data, vec![0.0, 0.0, 3.0, 6.0, 6.0]);
}

/// A malformed model whose min > max must not panic: Rust's `f32::clamp`
/// asserts `min <= max`. The fix degrades to numpy's/onnxruntime's own
/// max-then-min clip order, which yields a well-defined constant (`max`)
/// output instead of crashing.
#[test]
fn clip_min_greater_than_max_degrades_gracefully_instead_of_panicking() {
    let mut node = node_with(OpKind::Clip, &["y"]);
    float(&mut node, "min", 10.0);
    float(&mut node, "max", 5.0);
    let x = Tensor::new(vec![-3.0, 0.0, 100.0], vec![3]);
    let out = run_both(&ClipOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![5.0, 5.0, 5.0]);
}

#[test]
fn clip_empty_bound_tensor_falls_back_to_attribute_without_panicking() {
    let mut node = node_with(OpKind::Clip, &["y"]);
    float(&mut node, "max", 6.0);
    let x = Tensor::new(vec![-3.0, 9.0], vec![2]);
    let empty_min = Tensor::new(vec![], vec![0]);
    // input[1] is present but empty (malformed) -> falls back to the (absent)
    // "min" attribute's own default of -inf; input[2] is absent -> falls back
    // to the "max" attribute (6.0).
    let out = run_both(&ClipOp, &node, vec![Some(&x), Some(&empty_min), None]);
    assert_eq!(out[0].data, vec![-3.0, 6.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// [a0-9] Cast float->int: truncate toward zero (not round), saturate to range
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cast_float_to_int_truncates_toward_zero_not_rounds() {
    let mut node = node_with(OpKind::Cast, &["y"]);
    int(&mut node, "to", 7); // INT64
    let x = Tensor::new(vec![1.7, -1.5, 2.9999997, -2.9999997], vec![4]);
    let out = run_both(&CastOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![1.0, -1.0, 2.0, -2.0]);
}

#[test]
fn cast_saturates_to_destination_range_uint8() {
    let mut node = node_with(OpKind::Cast, &["y"]);
    int(&mut node, "to", 2); // UINT8
    let x = Tensor::new(vec![300.0, -5.0, 128.0], vec![3]);
    let out = run_both(&CastOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![255.0, 0.0, 128.0]);
}

#[test]
fn cast_saturates_to_destination_range_int8() {
    let mut node = node_with(OpKind::Cast, &["y"]);
    int(&mut node, "to", 3); // INT8
    let x = Tensor::new(vec![200.0, -200.0, 100.0], vec![3]);
    let out = run_both(&CastOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![127.0, -128.0, 100.0]);
}

#[test]
fn cast_narrow_int16_types_are_handled_not_left_as_no_op() {
    let mut node = node_with(OpKind::Cast, &["y"]);
    int(&mut node, "to", 5); // INT16
    let x = Tensor::new(vec![40000.0, -40000.0], vec![2]);
    let out = run_both(&CastOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![i16::MAX as f32, i16::MIN as f32]);
}

#[test]
fn cast_bool_target_is_unaffected_by_the_int_truncation_fix() {
    let mut node = node_with(OpKind::Cast, &["y"]);
    int(&mut node, "to", 9); // BOOL
    let x = Tensor::new(vec![0.0, 1.0, -2.5, 0.0001], vec![4]);
    let out = run_both(&CastOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![0.0, 1.0, 1.0, 1.0]);
}

/// The f32 `execute()` path and the typed `execute_typed` path (already
/// correct via `TypedTensor::cast`) must now agree on the same model.
#[test]
fn cast_f32_path_and_typed_path_agree_on_int_truncation() {
    use oxionnx_core::{TensorStorage, TypedOpContext, TypedTensor};

    let mut node = node_with(OpKind::Cast, &["y"]);
    int(&mut node, "to", 7); // INT64
    let x = Tensor::new(vec![1.7, -1.5, 300.0], vec![3]);
    let f32_out = run_both(&CastOp, &node, vec![Some(&x)]);

    let tt = TypedTensor::new(TensorStorage::F32(x.data.clone()), x.shape.clone());
    let typed_ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&tt)],
        outer_scope: None,
        registry: None,
    };
    let typed_out = CastOp.execute_typed(&typed_ctx).expect("execute_typed");
    match &typed_out[0].storage {
        TensorStorage::I64(v) => {
            let f32_as_i64: Vec<i64> = f32_out[0].data.iter().map(|&v| v as i64).collect();
            assert_eq!(v, &f32_as_i64);
        }
        other => panic!("expected I64 storage, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// [a0-13] Mod: fmod=0 is numpy-floored (sign of divisor), fmod=1 is C fmod
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mod_fmod0_is_numpy_floored_not_c_truncated() {
    let mut node = node_with(OpKind::Mod, &["y"]);
    int(&mut node, "fmod", 0);
    let a = Tensor::new(vec![-7.0, 7.0], vec![2]);
    let b = Tensor::new(vec![3.0, -3.0], vec![2]);
    let out = run_both(&ModOp, &node, vec![Some(&a), Some(&b)]);
    // Mod(-7,3)=2, Mod(7,-3)=-2 (numpy semantics: sign follows the divisor).
    assert_eq!(out[0].data, vec![2.0, -2.0]);
}

#[test]
fn mod_fmod1_is_c_style_sign_of_dividend() {
    let mut node = node_with(OpKind::Mod, &["y"]);
    int(&mut node, "fmod", 1);
    let a = Tensor::new(vec![-7.0, 7.0], vec![2]);
    let b = Tensor::new(vec![3.0, -3.0], vec![2]);
    let out = run_both(&ModOp, &node, vec![Some(&a), Some(&b)]);
    assert_eq!(out[0].data, vec![-1.0, 1.0]);
}

#[test]
fn mod_fmod0_default_matches_python_percent_on_floats() {
    // Default fmod (attribute absent) is 0. 5.5 % 2.0 (Python) == 1.5;
    // -5.5 % 2.0 (Python) == 0.5.
    let node = node_with(OpKind::Mod, &["y"]);
    let a = Tensor::new(vec![5.5, -5.5], vec![2]);
    let b = Tensor::new(vec![2.0], vec![1]);
    let out = run_both(&ModOp, &node, vec![Some(&a), Some(&b)]);
    assert_close(&out[0].data, &[1.5, 0.5], 1e-5, "mod_default");
}

// ═══════════════════════════════════════════════════════════════════════════
// [a0-14 / a11-24] Bitwise{And,Not} + BitShift: value-preserving, no u32
// round-trip, no shift-overflow panic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bitwise_and_registry_op_preserves_negative_operand_value() {
    let node = node_with(OpKind::BitwiseAnd, &["y"]);
    let a = Tensor::new(vec![-1.0], vec![1]);
    let b = Tensor::new(vec![5.0], vec![1]);
    let out = run_both(&BitwiseAndOp, &node, vec![Some(&a), Some(&b)]);
    assert_eq!(out[0].data, vec![5.0]);
}

/// Exercises `BitwiseNotOp::execute_into_slots` specifically (misc_ops.rs),
/// which had its own separate `as u32` bug duplicate of `bitwise::bitwise_not`.
#[test]
fn bitwise_not_registry_execute_and_execute_into_slots_agree_and_are_signed() {
    let node = node_with(OpKind::BitwiseNot, &["y"]);
    let x = Tensor::new(vec![0.0, -1.0], vec![2]);
    let out = run_both(&BitwiseNotOp, &node, vec![Some(&x)]);
    // NOT(0) = -1, NOT(-1) = 0.
    assert_eq!(out[0].data, vec![-1.0, 0.0]);
}

#[test]
fn bit_shift_left_and_right_basic() {
    let mut node = node_with(OpKind::BitShift, &["y"]);
    string(&mut node, "direction", "LEFT");
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let y = Tensor::new(vec![3.0], vec![1]);
    let out = run_both(&BitShiftOp, &node, vec![Some(&x), Some(&y)]);
    assert_eq!(out[0].data, vec![8.0, 16.0]);

    let mut node_r = node_with(OpKind::BitShift, &["y"]);
    string(&mut node_r, "direction", "RIGHT");
    let xr = Tensor::new(vec![16.0, 8.0], vec![2]);
    let out_r = run_both(&BitShiftOp, &node_r, vec![Some(&xr), Some(&y)]);
    assert_eq!(out_r[0].data, vec![2.0, 1.0]);
}

/// A shift amount >= the operand's bit width must saturate to 0, not panic
/// ("attempt to shift left/right with overflow" in a debug build) -- a
/// malformed model must produce a value, never crash the process.
#[test]
fn bit_shift_overflow_amount_saturates_to_zero_without_panicking() {
    let mut node = node_with(OpKind::BitShift, &["y"]);
    string(&mut node, "direction", "LEFT");
    let x = Tensor::new(vec![1.0], vec![1]);
    let y = Tensor::new(vec![100.0], vec![1]); // shift amount >= 64
    let out = run_both(&BitShiftOp, &node, vec![Some(&x), Some(&y)]);
    assert_eq!(out[0].data, vec![0.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// [a0-15 / a11-7] Reduce*: noop_with_empty_axes (opset 18)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reduce_sum_noop_with_empty_axes_is_identity() {
    let mut node = node_with(OpKind::ReduceSum, &["y"]);
    int(&mut node, "noop_with_empty_axes", 1);
    int(&mut node, "keepdims", 1);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = run_both(&ReduceSumOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].shape, vec![2, 3]);
    assert_eq!(out[0].data, x.data);
}

#[test]
fn reduce_sum_empty_axes_tensor_input_with_noop_flag_is_also_identity() {
    // Axes provided as an explicitly-empty *tensor* input, not merely absent --
    // must be treated identically to an absent axes input.
    let mut node = node_with(OpKind::ReduceSum, &["y"]);
    int(&mut node, "noop_with_empty_axes", 1);
    int(&mut node, "keepdims", 1);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let empty_axes = Tensor::new(vec![], vec![0]);
    let out = run_both(&ReduceSumOp, &node, vec![Some(&x), Some(&empty_axes)]);
    assert_eq!(out[0].shape, vec![2, 2]);
    assert_eq!(out[0].data, x.data);
}

#[test]
fn reduce_sum_empty_axes_without_flag_still_reduces_all_dimensions() {
    // noop_with_empty_axes defaults to 0 -> unchanged legacy behavior.
    let node = node_with(OpKind::ReduceSum, &["y"]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = run_both(&ReduceSumOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].shape, vec![1, 1]);
    assert_eq!(out[0].data, vec![21.0]);
}

#[test]
fn reduce_mean_noop_with_empty_axes_is_identity() {
    let mut node = node_with(OpKind::ReduceMean, &["y"]);
    int(&mut node, "noop_with_empty_axes", 1);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = run_both(&ReduceMeanOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, x.data);
}

// ═══════════════════════════════════════════════════════════════════════════
// [a0-16] ArgMax/ArgMin: select_last_index (opset 12)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn argmax_select_last_index_breaks_ties_toward_the_end() {
    let mut node = node_with(OpKind::ArgMax, &["y"]);
    int(&mut node, "axis", 0);
    int(&mut node, "select_last_index", 1);
    int(&mut node, "keepdims", 0);
    let x = Tensor::new(vec![1.0, 3.0, 3.0, 2.0], vec![4]);
    let out = run_both(&ArgMaxOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![2.0]);
}

#[test]
fn argmax_default_selects_first_index_on_tie() {
    let mut node = node_with(OpKind::ArgMax, &["y"]);
    int(&mut node, "axis", 0);
    int(&mut node, "keepdims", 0);
    let x = Tensor::new(vec![1.0, 3.0, 3.0, 2.0], vec![4]);
    let out = run_both(&ArgMaxOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![1.0]);
}

#[test]
fn argmin_select_last_index_breaks_ties_toward_the_end() {
    let mut node = node_with(OpKind::ArgMin, &["y"]);
    int(&mut node, "axis", 0);
    int(&mut node, "select_last_index", 1);
    int(&mut node, "keepdims", 0);
    let x = Tensor::new(vec![3.0, 1.0, 1.0, 2.0], vec![4]);
    let out = run_both(&ArgMinOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![2.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// [a0-17] Shape: start/end attributes (opset 15)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn shape_start_attribute_slices_from_the_given_dimension() {
    let mut node = node_with(OpKind::Shape, &["y"]);
    int(&mut node, "start", 1);
    let x = Tensor::new(vec![0.0; 2 * 3 * 4 * 5], vec![2, 3, 4, 5]);
    let out = run_both(&ShapeOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![3.0, 4.0, 5.0]);
}

#[test]
fn shape_negative_end_excludes_from_the_back() {
    let mut node = node_with(OpKind::Shape, &["y"]);
    int(&mut node, "end", -1);
    let x = Tensor::new(vec![0.0; 2 * 3 * 4 * 5], vec![2, 3, 4, 5]);
    let out = run_both(&ShapeOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![2.0, 3.0, 4.0]);
}

#[test]
fn shape_start_beyond_end_yields_empty_not_a_panic() {
    let mut node = node_with(OpKind::Shape, &["y"]);
    int(&mut node, "start", 3);
    int(&mut node, "end", 1);
    let x = Tensor::new(vec![0.0; 6], vec![2, 3]);
    let out = run_both(&ShapeOp, &node, vec![Some(&x)]);
    assert!(out[0].data.is_empty());
    assert_eq!(out[0].shape, vec![0]);
}

#[test]
fn shape_no_attrs_returns_the_full_shape() {
    let node = node_with(OpKind::Shape, &["y"]);
    let x = Tensor::new(vec![0.0; 6], vec![2, 3]);
    let out = run_both(&ShapeOp, &node, vec![Some(&x)]);
    assert_eq!(out[0].data, vec![2.0, 3.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// [a0-20] Equal: exact equality, no epsilon tolerance
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equal_registry_op_is_exact_not_epsilon() {
    let node = node_with(OpKind::Equal, &["y"]);
    let a = Tensor::new(vec![0.0], vec![1]);
    let b = Tensor::new(vec![1e-8], vec![1]);
    let out = run_both(&EqualOp, &node, vec![Some(&a), Some(&b)]);
    assert_eq!(out[0].data, vec![0.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// [a1-14] GroupNormalization: per-group vs per-channel scale/bias mapping
// ═══════════════════════════════════════════════════════════════════════════
//
// Reference values computed with python3 (see PR notes): x is [1,4,2]
// (N=1,C=4,spatial=2), num_groups=2 (channels_per_group=2). Channels 0,1 are
// group 0 (mean=2.5, var=1.25); channels 2,3 are group 1 (mean=6.5,
// var=1.25); eps=1e-5. A per-group scale/bias of length `num_groups` must
// apply scale[0],bias[0] to channels 0,1 and scale[1],bias[1] to channels
// 2,3 (i.e. [a,a,b,b]) -- the old `ci % num_groups` mapping produced
// [a,b,a,b] instead, which is what this test would catch.

#[test]
fn group_norm_per_group_scale_bias_maps_to_all_channels_in_the_group() {
    let mut node = node_with(OpKind::GroupNorm, &["y"]);
    int(&mut node, "num_groups", 2);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 4, 2]);
    let scale = Tensor::new(vec![10.0, 100.0], vec![2]); // per-group (len == num_groups)
    let bias = Tensor::new(vec![0.0, 0.0], vec![2]);
    let out = run_both(
        &GroupNormOp,
        &node,
        vec![Some(&x), Some(&scale), Some(&bias)],
    );
    let expected = [
        -13.416354, -4.472_118, 4.472_118, 13.416354, -134.16354, -44.721_18, 44.721_18, 134.16354,
    ];
    assert_close(&out[0].data, &expected, 1e-2, "group_norm_per_group");
}

#[test]
fn group_norm_per_group_and_per_channel_scale_agree() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 4, 2]);
    let scale_group = Tensor::new(vec![10.0, 100.0], vec![2]);
    let scale_channel = Tensor::new(vec![10.0, 10.0, 100.0, 100.0], vec![4]);
    let out_group =
        oxionnx_ops::nn::group_norm(&x, &scale_group, None, 2, 1e-5).expect("group_norm");
    let out_channel =
        oxionnx_ops::nn::group_norm(&x, &scale_channel, None, 2, 1e-5).expect("group_norm");
    assert_close(
        &out_group.data,
        &out_channel.data,
        1e-5,
        "group_norm_group_vs_channel",
    );
}

#[test]
fn group_norm_scale_with_invalid_length_is_a_typed_error_not_a_silent_wrap() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1]);
    let bad_scale = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]); // neither C=4 nor num_groups=2
    let result = oxionnx_ops::nn::group_norm(&x, &bad_scale, None, 2, 1e-5);
    assert!(
        result.is_err(),
        "a scale tensor that is neither per-channel nor per-group must error, not silently wrap"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// [a1-16] Dropout: optional mask output, training_mode guard
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dropout_single_output_is_identity_in_inference_mode() {
    let node = node_with(OpKind::Dropout, &["y"]);
    let x = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]);
    let out = run_both(&DropoutOp, &node, vec![Some(&x)]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].data, x.data);
}

#[test]
fn dropout_two_declared_outputs_emits_an_all_true_mask() {
    let node = node_with(OpKind::Dropout, &["y", "mask"]);
    let x = Tensor::new(vec![1.0, -2.0, 3.0, 4.0], vec![2, 2]);
    let out = run_both(&DropoutOp, &node, vec![Some(&x)]);
    assert_eq!(
        out.len(),
        2,
        "a 2-output Dropout node must resolve both outputs"
    );
    assert_eq!(out[0].data, x.data, "output 0 is still identity");
    assert_eq!(
        out[1].data,
        vec![1.0, 1.0, 1.0, 1.0],
        "output 1 is an all-true mask"
    );
    assert_eq!(out[1].shape, vec![2, 2]);
}

#[test]
fn dropout_training_mode_true_is_a_typed_error_not_silently_wrong_output() {
    let node = node_with(OpKind::Dropout, &["y"]);
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let ratio = Tensor::new(vec![0.5], vec![1]);
    let training_mode = Tensor::new(vec![1.0], vec![1]); // true
    let context = ctx(&node, vec![Some(&x), Some(&ratio), Some(&training_mode)]);
    let result = DropoutOp.execute(&context);
    assert!(
        result.is_err(),
        "training_mode=true must error, not silently run inference-mode dropout"
    );
}

#[test]
fn dropout_training_mode_explicit_false_is_still_identity() {
    let node = node_with(OpKind::Dropout, &["y"]);
    let x = Tensor::new(vec![5.0, 6.0], vec![2]);
    let ratio = Tensor::new(vec![0.5], vec![1]);
    let training_mode = Tensor::new(vec![0.0], vec![1]); // explicit false
    let out = run_both(
        &DropoutOp,
        &node,
        vec![Some(&x), Some(&ratio), Some(&training_mode)],
    );
    assert_eq!(out[0].data, x.data);
}

// ═══════════════════════════════════════════════════════════════════════════
// [a3-16 / a5-3] Gelu: honors `approximate` (default "none" = exact erf)
// ═══════════════════════════════════════════════════════════════════════════
//
// Reference values from python3 `math.erf` / `math.tanh` (see PR notes).
// exact(1.0)=0.8413447, tanh(1.0)=0.8411920 -- a ~1.5e-4 gap, so a 1e-4
// tolerance against one formula's reference also proves the other formula is
// NOT what produced the result.

const GELU_X: [f32; 6] = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
const GELU_EXACT: [f32; 6] = [-0.0455003, -0.1586553, 0.0, 0.8413447, 1.9544997, 2.9959503];
const GELU_TANH: [f32; 6] = [-0.0454023, -0.158_808, 0.0, 0.841_192, 1.9545977, 2.9963626];

#[test]
fn gelu_default_uses_exact_erf_formula_not_tanh_approx() {
    // No `approximate` attribute set => spec default "none" => exact erf formula.
    let node = node_with(OpKind::Gelu, &["y"]);
    let x = Tensor::new(GELU_X.to_vec(), vec![6]);
    let out = run_both(&GeluOp, &node, vec![Some(&x)]);
    assert_close(&out[0].data, &GELU_EXACT, 1e-4, "gelu_default");
}

#[test]
fn gelu_approximate_none_explicit_matches_default() {
    let mut node = node_with(OpKind::Gelu, &["y"]);
    string(&mut node, "approximate", "none");
    let x = Tensor::new(GELU_X.to_vec(), vec![6]);
    let out = run_both(&GeluOp, &node, vec![Some(&x)]);
    assert_close(&out[0].data, &GELU_EXACT, 1e-4, "gelu_none_explicit");
}

#[test]
fn gelu_approximate_tanh_uses_the_tanh_formula() {
    let mut node = node_with(OpKind::Gelu, &["y"]);
    string(&mut node, "approximate", "tanh");
    let x = Tensor::new(GELU_X.to_vec(), vec![6]);
    let out = run_both(&GeluOp, &node, vec![Some(&x)]);
    assert_close(&out[0].data, &GELU_TANH, 1e-4, "gelu_tanh");
}

#[test]
fn gelu_execute_inplace_honors_approximate_attribute() {
    let node_exact = node_with(OpKind::Gelu, &["y"]);
    let ctx_exact = ctx(&node_exact, vec![None]);
    let out_exact = GeluOp
        .execute_inplace(Tensor::new(vec![1.0], vec![1]), &ctx_exact)
        .expect("execute_inplace exact");
    assert_close(&out_exact[0].data, &[0.8413447], 1e-4, "gelu_inplace_exact");

    let mut node_tanh = node_with(OpKind::Gelu, &["y"]);
    string(&mut node_tanh, "approximate", "tanh");
    let ctx_tanh = ctx(&node_tanh, vec![None]);
    let out_tanh = GeluOp
        .execute_inplace(Tensor::new(vec![1.0], vec![1]), &ctx_tanh)
        .expect("execute_inplace tanh");
    assert_close(&out_tanh[0].data, &[0.841_192], 1e-4, "gelu_inplace_tanh");
}

#[test]
fn gelu_execute_typed_honors_approximate_attribute() {
    use oxionnx_core::{TensorStorage, TypedOpContext, TypedTensor};

    let node_default = node_with(OpKind::Gelu, &["y"]); // no attr => exact
    let tt = TypedTensor::new(TensorStorage::F32(vec![1.0]), vec![1]);
    let typed_ctx = TypedOpContext {
        node: &node_default,
        inputs: vec![Some(&tt)],
        outer_scope: None,
        registry: None,
    };
    let out = GeluOp
        .execute_typed(&typed_ctx)
        .expect("execute_typed default");
    match &out[0].storage {
        TensorStorage::F32(v) => assert_close(v, &[0.8413447], 1e-4, "gelu_typed_default"),
        other => panic!("expected F32 storage, got {other:?}"),
    }

    let mut node_tanh = node_with(OpKind::Gelu, &["y"]);
    string(&mut node_tanh, "approximate", "tanh");
    let typed_ctx_tanh = TypedOpContext {
        node: &node_tanh,
        inputs: vec![Some(&tt)],
        outer_scope: None,
        registry: None,
    };
    let out_tanh = GeluOp
        .execute_typed(&typed_ctx_tanh)
        .expect("execute_typed tanh");
    match &out_tanh[0].storage {
        TensorStorage::F32(v) => assert_close(v, &[0.841_192], 1e-4, "gelu_typed_tanh"),
        other => panic!("expected F32 storage, got {other:?}"),
    }
}

/// `native_dtypes()` must be unaffected by the `execute_typed` rewrite (still
/// declares F32/F16/BF16/I32/I64, matching what it declared before this fix).
#[test]
fn gelu_native_dtypes_unchanged() {
    use oxionnx_core::DType;
    let dtypes = GeluOp.native_dtypes();
    for want in [DType::F32, DType::F16, DType::BF16, DType::I32, DType::I64] {
        assert!(
            dtypes.contains(&want),
            "GeluOp.native_dtypes() must still contain {want:?}"
        );
    }
}
