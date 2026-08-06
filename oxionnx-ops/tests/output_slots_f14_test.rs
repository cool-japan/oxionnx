//! Output-slot correctness tests for Phase F.14 operators.
//!
//! Verifies that `execute_into_slots` produces byte-identical results to `execute`
//! for: LayerNorm, GroupNorm, BatchNorm, RmsNorm, InstanceNorm, Softmax, LogSoftmax,
//! Not, IsInf, IsNaN, BitwiseNot, ConstantOfShape, Shape, Size, Constant,
//! ReduceSum, ReduceMean, ReduceMax, ReduceMin, ArgMax, ArgMin, CumSum, TopK.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::registry::{
    math_ops::{
        ArgMaxOp, ArgMinOp, CumSumOp, ReduceMaxOp, ReduceMeanOp, ReduceMinOp, ReduceSumOp, TopKOp,
    },
    misc_ops::{
        BitwiseNotOp, ConstantOfShapeOp, ConstantOp, IsInfOp, IsNaNOp, NotOp, ShapeOp, SizeOp,
    },
    nn_ops::{
        BatchNormOp, GroupNormOp, InstanceNormOp, LayerNormOp, LogSoftmaxOp, RmsNormOp, SoftmaxOp,
    },
};

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

fn assert_tensor_eq(a: &Tensor, b: &Tensor, label: &str) {
    assert_eq!(a.shape, b.shape, "{label}: shape mismatch");
    assert_eq!(a.data.len(), b.data.len(), "{label}: data len mismatch");
    for (i, (&av, &bv)) in a.data.iter().zip(b.data.iter()).enumerate() {
        assert!(
            (av - bv).abs() < 1e-5 || (av.is_nan() && bv.is_nan()),
            "{label}[{i}]: got {av}, expected {bv}",
        );
    }
}

fn linspace(shape: &[usize], start: f32, step: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
    Tensor::new(data, shape.to_vec())
}

fn const_tensor(shape: &[usize], val: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new(vec![val; n], shape.to_vec())
}

fn run_and_compare(op: &dyn Operator, node: &Node, inputs: Vec<Option<&Tensor>>, label: &str) {
    let ctx = make_ctx(node, inputs);
    let expected = op.execute(&ctx).expect("execute failed");
    let n_outs = expected.len();
    let mut slots: Vec<Tensor> = expected
        .iter()
        .map(|t| Tensor::new(vec![0.0f32; t.numel()], t.shape.clone()))
        .collect();
    op.execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    for (i, (slot, exp)) in slots.iter().zip(expected.iter()).enumerate() {
        assert_tensor_eq(slot, exp, &format!("{label}[out{i}]"));
    }
    assert_eq!(slots.len(), n_outs);
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

#[test]
fn test_layer_norm_slot_basic() {
    let node = dummy_node(OpKind::LayerNorm);
    let x = linspace(&[2, 4, 8], 0.1, 0.05);
    let scale = const_tensor(&[8], 1.0);
    let bias = const_tensor(&[8], 0.0);
    run_and_compare(
        &LayerNormOp,
        &node,
        vec![Some(&x), Some(&scale), Some(&bias)],
        "layernorm_basic",
    );
}

#[test]
fn test_layer_norm_slot_no_bias() {
    let node = dummy_node(OpKind::LayerNorm);
    let x = linspace(&[3, 6], -1.0, 0.1);
    let scale = const_tensor(&[6], 2.0);
    run_and_compare(
        &LayerNormOp,
        &node,
        vec![Some(&x), Some(&scale), None],
        "layernorm_no_bias",
    );
}

#[test]
fn test_layer_norm_slot_second_call_resize() {
    // First call with shape [2, 4], second call with shape [3, 6]
    let node = dummy_node(OpKind::LayerNorm);
    let x1 = linspace(&[2, 4], 0.1, 0.1);
    let s1 = const_tensor(&[4], 1.0);
    let ctx1 = make_ctx(&node, vec![Some(&x1), Some(&s1), None]);
    let mut slots = vec![Tensor::new(vec![0.0f32; 8], vec![2, 4])];
    LayerNormOp.execute_into_slots(&ctx1, &mut slots).unwrap();
    // Second call with larger shape
    let x2 = linspace(&[3, 6], 0.2, 0.05);
    let s2 = const_tensor(&[6], 1.5);
    let ctx2 = make_ctx(&node, vec![Some(&x2), Some(&s2), None]);
    LayerNormOp.execute_into_slots(&ctx2, &mut slots).unwrap();
    let expected = LayerNormOp.execute(&ctx2).unwrap();
    assert_tensor_eq(&slots[0], &expected[0], "layernorm_resize");
}

// ── RmsNorm ──────────────────────────────────────────────────────────────────

#[test]
fn test_rms_norm_slot() {
    let node = dummy_node(OpKind::RMSNorm);
    let x = linspace(&[2, 8], 0.5, 0.1);
    let scale = const_tensor(&[8], 1.0);
    run_and_compare(&RmsNormOp, &node, vec![Some(&x), Some(&scale)], "rmsnorm");
}

#[test]
fn test_rms_norm_slot_scale_variation() {
    let node = dummy_node(OpKind::RMSNorm);
    let x = linspace(&[4, 4], -2.0, 0.3);
    let scale = linspace(&[4], 0.5, 0.5);
    run_and_compare(
        &RmsNormOp,
        &node,
        vec![Some(&x), Some(&scale)],
        "rmsnorm_scale_var",
    );
}

// ── BatchNorm ─────────────────────────────────────────────────────────────────

#[test]
fn test_batch_norm_slot() {
    let node = dummy_node(OpKind::BatchNorm);
    let x = linspace(&[2, 4, 3, 3], 0.1, 0.05);
    let scale = const_tensor(&[4], 1.0);
    let bias = const_tensor(&[4], 0.0);
    let mean = const_tensor(&[4], 0.5);
    let var = const_tensor(&[4], 1.0);
    run_and_compare(
        &BatchNormOp,
        &node,
        vec![Some(&x), Some(&scale), Some(&bias), Some(&mean), Some(&var)],
        "batchnorm",
    );
}

// ── GroupNorm ─────────────────────────────────────────────────────────────────

#[test]
fn test_group_norm_slot() {
    let node = node_with_int_attrs(OpKind::GroupNorm, &[("num_groups", 2)]);
    let x = linspace(&[2, 4, 3], 0.1, 0.07);
    let scale = linspace(&[4], 0.8, 0.1);
    let bias = linspace(&[4], 0.0, 0.05);
    run_and_compare(
        &GroupNormOp,
        &node,
        vec![Some(&x), Some(&scale), Some(&bias)],
        "groupnorm",
    );
}

// ── InstanceNorm ──────────────────────────────────────────────────────────────

#[test]
fn test_instance_norm_slot() {
    let node = dummy_node(OpKind::InstanceNorm);
    let x = linspace(&[2, 3, 8], 0.2, 0.03);
    let scale = const_tensor(&[3], 1.0);
    let bias = const_tensor(&[3], 0.0);
    run_and_compare(
        &InstanceNormOp,
        &node,
        vec![Some(&x), Some(&scale), Some(&bias)],
        "instancenorm",
    );
}

// ── Softmax ───────────────────────────────────────────────────────────────────

#[test]
fn test_softmax_slot_last_axis() {
    let node = node_with_int_attrs(OpKind::Softmax, &[("axis", -1)]);
    let x = linspace(&[4, 8], -1.0, 0.3);
    run_and_compare(&SoftmaxOp, &node, vec![Some(&x)], "softmax_last_axis");
}

#[test]
fn test_softmax_slot_mid_axis() {
    let node = node_with_int_attrs(OpKind::Softmax, &[("axis", 1)]);
    let x = linspace(&[2, 4, 4], -0.5, 0.15);
    run_and_compare(&SoftmaxOp, &node, vec![Some(&x)], "softmax_mid_axis");
}

#[test]
fn test_softmax_slot_second_call_same_shape() {
    let node = node_with_int_attrs(OpKind::Softmax, &[("axis", -1)]);
    let x = linspace(&[3, 5], 0.0, 0.2);
    let ctx = make_ctx(&node, vec![Some(&x)]);
    let mut slots = vec![Tensor::new(vec![0.0f32; 15], vec![3, 5])];
    SoftmaxOp.execute_into_slots(&ctx, &mut slots).unwrap();
    let buf_ptr = slots[0].data.as_ptr();
    SoftmaxOp.execute_into_slots(&ctx, &mut slots).unwrap();
    // Pool reuse: same pointer (no reallocation on second call with same shape)
    assert_eq!(
        slots[0].data.as_ptr(),
        buf_ptr,
        "softmax: ptr must not change on same-shape reuse"
    );
}

// ── LogSoftmax ────────────────────────────────────────────────────────────────

#[test]
fn test_log_softmax_slot() {
    let node = node_with_int_attrs(OpKind::LogSoftmax, &[("axis", -1)]);
    let x = linspace(&[3, 6], -0.5, 0.2);
    run_and_compare(&LogSoftmaxOp, &node, vec![Some(&x)], "log_softmax");
}

// ── Not ───────────────────────────────────────────────────────────────────────

#[test]
fn test_not_slot() {
    let node = dummy_node(OpKind::Not);
    let x = Tensor::new(vec![0.0f32, 1.0, 0.0, -1.0, 0.0], vec![5]);
    run_and_compare(&NotOp, &node, vec![Some(&x)], "not");
}

// ── IsInf ─────────────────────────────────────────────────────────────────────

#[test]
fn test_is_inf_slot_both() {
    let node = dummy_node(OpKind::IsInf);
    let x = Tensor::new(
        vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 0.0, f32::NAN],
        vec![5],
    );
    run_and_compare(&IsInfOp, &node, vec![Some(&x)], "is_inf_both");
}

#[test]
fn test_is_inf_slot_pos_only() {
    let node = node_with_int_attrs(
        OpKind::IsInf,
        &[("detect_positive", 1), ("detect_negative", 0)],
    );
    let x = Tensor::new(vec![f32::INFINITY, -f32::INFINITY, 2.0], vec![3]);
    run_and_compare(&IsInfOp, &node, vec![Some(&x)], "is_inf_pos_only");
}

// ── IsNaN ─────────────────────────────────────────────────────────────────────

#[test]
fn test_is_nan_slot() {
    let node = dummy_node(OpKind::IsNaN);
    let x = Tensor::new(vec![f32::NAN, 0.0, f32::INFINITY, f32::NAN], vec![4]);
    run_and_compare(&IsNaNOp, &node, vec![Some(&x)], "is_nan");
}

// ── BitwiseNot ────────────────────────────────────────────────────────────────

#[test]
fn test_bitwise_not_slot() {
    let node = dummy_node(OpKind::BitwiseNot);
    let x = Tensor::new(vec![0.0f32, 5.0, 255.0, 1.0], vec![4]);
    run_and_compare(&BitwiseNotOp, &node, vec![Some(&x)], "bitwise_not");
}

// ── ConstantOfShape ───────────────────────────────────────────────────────────

#[test]
fn test_constant_of_shape_slot_zeros() {
    let node = dummy_node(OpKind::ConstantOfShape);
    let shape = Tensor::new(vec![3.0, 4.0], vec![2]);
    run_and_compare(
        &ConstantOfShapeOp,
        &node,
        vec![Some(&shape)],
        "constant_of_shape_zeros",
    );
}

#[test]
fn test_constant_of_shape_slot_resize() {
    // First call shape [2,3], then [4,5]
    let node = dummy_node(OpKind::ConstantOfShape);
    let s1 = Tensor::new(vec![2.0, 3.0], vec![2]);
    let ctx1 = make_ctx(&node, vec![Some(&s1)]);
    let mut slots = vec![Tensor::new(vec![0.0f32; 6], vec![2, 3])];
    ConstantOfShapeOp
        .execute_into_slots(&ctx1, &mut slots)
        .unwrap();
    assert_eq!(slots[0].shape, vec![2, 3]);

    let s2 = Tensor::new(vec![4.0, 5.0], vec![2]);
    let ctx2 = make_ctx(&node, vec![Some(&s2)]);
    ConstantOfShapeOp
        .execute_into_slots(&ctx2, &mut slots)
        .unwrap();
    assert_eq!(slots[0].shape, vec![4, 5]);
    assert_eq!(slots[0].data.len(), 20);
}

// ── Shape ─────────────────────────────────────────────────────────────────────

#[test]
fn test_shape_slot() {
    let node = dummy_node(OpKind::Shape);
    let x = linspace(&[2, 3, 4], 0.0, 1.0);
    run_and_compare(&ShapeOp, &node, vec![Some(&x)], "shape_op");
}

#[test]
fn test_shape_slot_second_call_different_ndim() {
    let node = dummy_node(OpKind::Shape);
    let x1 = linspace(&[4, 4], 0.0, 1.0);
    let ctx1 = make_ctx(&node, vec![Some(&x1)]);
    let mut slots = vec![Tensor::new(vec![0.0f32; 2], vec![2])];
    ShapeOp.execute_into_slots(&ctx1, &mut slots).unwrap();
    assert_eq!(slots[0].data, vec![4.0, 4.0]);

    let x2 = linspace(&[2, 3, 5], 0.0, 1.0);
    let ctx2 = make_ctx(&node, vec![Some(&x2)]);
    ShapeOp.execute_into_slots(&ctx2, &mut slots).unwrap();
    assert_eq!(slots[0].shape, vec![3]);
    assert_eq!(slots[0].data, vec![2.0, 3.0, 5.0]);
}

// ── Size ─────────────────────────────────────────────────────────────────────

#[test]
fn test_size_slot() {
    let node = dummy_node(OpKind::Size);
    let x = linspace(&[3, 4, 5], 0.0, 1.0);
    run_and_compare(&SizeOp, &node, vec![Some(&x)], "size_op");
}

// ── Constant ─────────────────────────────────────────────────────────────────

#[test]
fn test_constant_slot_float() {
    let mut node = dummy_node(OpKind::Constant);
    node.attrs.floats.insert("value_float".into(), 3.15_f32);
    run_and_compare(&ConstantOp, &node, vec![], "constant_float");
}

#[test]
fn test_constant_slot_int() {
    let mut node = dummy_node(OpKind::Constant);
    node.attrs.ints.insert("value_int".into(), 42);
    run_and_compare(&ConstantOp, &node, vec![], "constant_int");
}

// ── ReduceSum ─────────────────────────────────────────────────────────────────

#[test]
fn test_reduce_sum_slot_keepdims() {
    let node = node_with_int_attrs(OpKind::ReduceSum, &[("keepdims", 1)]);
    let x = linspace(&[3, 4, 5], 1.0, 0.1);
    let axes = Tensor::new(vec![1.0], vec![1]);
    run_and_compare(
        &ReduceSumOp,
        &node,
        vec![Some(&x), Some(&axes)],
        "reduce_sum_keepdims",
    );
}

#[test]
fn test_reduce_sum_slot_no_keepdims() {
    let node = node_with_int_attrs(OpKind::ReduceSum, &[("keepdims", 0)]);
    let x = linspace(&[2, 6], 0.0, 0.5);
    let axes = Tensor::new(vec![1.0], vec![1]);
    run_and_compare(
        &ReduceSumOp,
        &node,
        vec![Some(&x), Some(&axes)],
        "reduce_sum_no_keepdims",
    );
}

#[test]
fn test_reduce_sum_slot_all_axes() {
    let node = node_with_int_attrs(OpKind::ReduceSum, &[("keepdims", 0)]);
    let x = linspace(&[4, 4], 1.0, 1.0);
    run_and_compare(&ReduceSumOp, &node, vec![Some(&x), None], "reduce_sum_all");
}

// ── ReduceMean ────────────────────────────────────────────────────────────────

#[test]
fn test_reduce_mean_slot() {
    let node = node_with_int_attrs(OpKind::ReduceMean, &[("keepdims", 1)]);
    let x = linspace(&[2, 3, 4], 0.0, 0.25);
    let axes = Tensor::new(vec![2.0], vec![1]);
    run_and_compare(
        &ReduceMeanOp,
        &node,
        vec![Some(&x), Some(&axes)],
        "reduce_mean",
    );
}

// ── ReduceMax / ReduceMin ─────────────────────────────────────────────────────

#[test]
fn test_reduce_max_slot() {
    let node = node_with_int_attrs(OpKind::ReduceMax, &[("keepdims", 0)]);
    let x = linspace(&[5, 5], -3.0, 0.5);
    let axes = Tensor::new(vec![0.0], vec![1]);
    run_and_compare(
        &ReduceMaxOp,
        &node,
        vec![Some(&x), Some(&axes)],
        "reduce_max",
    );
}

#[test]
fn test_reduce_min_slot() {
    let node = node_with_int_attrs(OpKind::ReduceMin, &[("keepdims", 0)]);
    let x = linspace(&[4, 6], 10.0, -0.2);
    let axes = Tensor::new(vec![1.0], vec![1]);
    run_and_compare(
        &ReduceMinOp,
        &node,
        vec![Some(&x), Some(&axes)],
        "reduce_min",
    );
}

// ── ArgMax / ArgMin ───────────────────────────────────────────────────────────

#[test]
fn test_arg_max_slot() {
    let node = node_with_int_attrs(OpKind::ArgMax, &[("axis", 1), ("keepdims", 0)]);
    let x = Tensor::new(vec![3.0, 1.0, 2.0, 0.0, 4.0, 1.0], vec![2, 3]);
    run_and_compare(&ArgMaxOp, &node, vec![Some(&x)], "arg_max");
}

#[test]
fn test_arg_max_slot_keepdims() {
    let node = node_with_int_attrs(OpKind::ArgMax, &[("axis", 0), ("keepdims", 1)]);
    let x = linspace(&[3, 4], -1.0, 0.5);
    run_and_compare(&ArgMaxOp, &node, vec![Some(&x)], "arg_max_keepdims");
}

#[test]
fn test_arg_min_slot() {
    let node = node_with_int_attrs(OpKind::ArgMin, &[("axis", 1), ("keepdims", 0)]);
    let x = Tensor::new(vec![3.0, 1.0, 2.0, 0.0, 4.0, 1.0], vec![2, 3]);
    run_and_compare(&ArgMinOp, &node, vec![Some(&x)], "arg_min");
}

// ── CumSum ────────────────────────────────────────────────────────────────────

#[test]
fn test_cumsum_slot_forward_inclusive() {
    let node = node_with_int_attrs(OpKind::CumSum, &[("exclusive", 0), ("reverse", 0)]);
    let x = linspace(&[4, 5], 1.0, 1.0);
    let axis_t = Tensor::new(vec![1.0], vec![1]);
    run_and_compare(
        &CumSumOp,
        &node,
        vec![Some(&x), Some(&axis_t)],
        "cumsum_fwd",
    );
}

#[test]
fn test_cumsum_slot_exclusive_reverse() {
    let node = node_with_int_attrs(OpKind::CumSum, &[("exclusive", 1), ("reverse", 1)]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let axis_t = Tensor::new(vec![0.0], vec![1]);
    run_and_compare(
        &CumSumOp,
        &node,
        vec![Some(&x), Some(&axis_t)],
        "cumsum_excl_rev",
    );
}

#[test]
fn test_cumsum_slot_pool_reuse() {
    let node = node_with_int_attrs(OpKind::CumSum, &[("exclusive", 0), ("reverse", 0)]);
    let x = linspace(&[3, 4], 0.0, 1.0);
    let axis_t = Tensor::new(vec![0.0], vec![1]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&axis_t)]);
    let mut slots = vec![Tensor::new(vec![0.0f32; 12], vec![3, 4])];
    CumSumOp.execute_into_slots(&ctx, &mut slots).unwrap();
    let ptr = slots[0].data.as_ptr();
    CumSumOp.execute_into_slots(&ctx, &mut slots).unwrap();
    assert_eq!(
        slots[0].data.as_ptr(),
        ptr,
        "cumsum: no realloc on same-shape second call"
    );
}

// ── TopK ─────────────────────────────────────────────────────────────────────

#[test]
fn test_topk_slot_largest() {
    let node = node_with_int_attrs(OpKind::TopK, &[("axis", -1), ("largest", 1), ("sorted", 1)]);
    let x = Tensor::new(vec![5.0, 1.0, 3.0, 2.0, 4.0], vec![1, 5]);
    let k_t = Tensor::new(vec![3.0], vec![1]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&k_t)]);
    let expected = TopKOp.execute(&ctx).unwrap();
    let mut slots = vec![
        Tensor::new(vec![0.0f32; 3], vec![1, 3]),
        Tensor::new(vec![0.0f32; 3], vec![1, 3]),
    ];
    TopKOp.execute_into_slots(&ctx, &mut slots).unwrap();
    assert_tensor_eq(&slots[0], &expected[0], "topk_values");
    assert_tensor_eq(&slots[1], &expected[1], "topk_indices");
}

#[test]
fn test_topk_slot_smallest() {
    let node = node_with_int_attrs(OpKind::TopK, &[("axis", -1), ("largest", 0), ("sorted", 1)]);
    let x = Tensor::new(vec![3.0, 5.0, 1.0, 4.0, 2.0], vec![1, 5]);
    let k_t = Tensor::new(vec![2.0], vec![1]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&k_t)]);
    let expected = TopKOp.execute(&ctx).unwrap();
    let mut slots = vec![
        Tensor::new(vec![0.0f32; 2], vec![1, 2]),
        Tensor::new(vec![0.0f32; 2], vec![1, 2]),
    ];
    TopKOp.execute_into_slots(&ctx, &mut slots).unwrap();
    assert_tensor_eq(&slots[0], &expected[0], "topk_smallest_values");
    assert_tensor_eq(&slots[1], &expected[1], "topk_smallest_indices");
}

#[test]
fn test_topk_slot_2d_axis0() {
    let node = node_with_int_attrs(OpKind::TopK, &[("axis", 0), ("largest", 1), ("sorted", 1)]);
    let x = linspace(&[4, 3], 0.0, 1.0);
    let k_t = Tensor::new(vec![2.0], vec![1]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&k_t)]);
    let expected = TopKOp.execute(&ctx).unwrap();
    let out_shape = expected[0].shape.clone();
    let out_len = expected[0].numel();
    let mut slots = vec![
        Tensor::new(vec![0.0f32; out_len], out_shape.clone()),
        Tensor::new(vec![0.0f32; out_len], out_shape),
    ];
    TopKOp.execute_into_slots(&ctx, &mut slots).unwrap();
    assert_tensor_eq(&slots[0], &expected[0], "topk_2d_values");
    assert_tensor_eq(&slots[1], &expected[1], "topk_2d_indices");
}

// ── Cross-op: reuse slot across different shapes ──────────────────────────────

#[test]
fn test_reduce_sum_slot_shape_change_resize() {
    // Verify slot is correctly resized when output shape changes between calls
    let node = node_with_int_attrs(OpKind::ReduceSum, &[("keepdims", 0)]);
    let x1 = linspace(&[3, 4], 1.0, 1.0);
    let axes1 = Tensor::new(vec![1.0], vec![1]);
    let ctx1 = make_ctx(&node, vec![Some(&x1), Some(&axes1)]);
    let mut slots = vec![Tensor::new(vec![0.0f32; 3], vec![3])];
    ReduceSumOp.execute_into_slots(&ctx1, &mut slots).unwrap();
    assert_eq!(slots[0].shape, vec![3]);

    let x2 = linspace(&[2, 5], 0.0, 0.5);
    let axes2 = Tensor::new(vec![0.0], vec![1]);
    let ctx2 = make_ctx(&node, vec![Some(&x2), Some(&axes2)]);
    ReduceSumOp.execute_into_slots(&ctx2, &mut slots).unwrap();
    assert_eq!(slots[0].shape, vec![5]);
    let expected2 = ReduceSumOp.execute(&ctx2).unwrap();
    assert_tensor_eq(&slots[0], &expected2[0], "reduce_sum_resize");
}

// ── Supports flag verification ────────────────────────────────────────────────

#[test]
fn test_f14_supports_output_slots_flags() {
    assert!(LayerNormOp.supports_output_slots());
    assert!(GroupNormOp.supports_output_slots());
    assert!(BatchNormOp.supports_output_slots());
    assert!(RmsNormOp.supports_output_slots());
    assert!(InstanceNormOp.supports_output_slots());
    assert!(SoftmaxOp.supports_output_slots());
    assert!(LogSoftmaxOp.supports_output_slots());
    assert!(NotOp.supports_output_slots());
    assert!(IsInfOp.supports_output_slots());
    assert!(IsNaNOp.supports_output_slots());
    assert!(BitwiseNotOp.supports_output_slots());
    assert!(ConstantOfShapeOp.supports_output_slots());
    assert!(ShapeOp.supports_output_slots());
    assert!(SizeOp.supports_output_slots());
    assert!(ConstantOp.supports_output_slots());
    assert!(ReduceSumOp.supports_output_slots());
    assert!(ReduceMeanOp.supports_output_slots());
    assert!(ReduceMaxOp.supports_output_slots());
    assert!(ReduceMinOp.supports_output_slots());
    assert!(ArgMaxOp.supports_output_slots());
    assert!(ArgMinOp.supports_output_slots());
    assert!(CumSumOp.supports_output_slots());
    assert!(TopKOp.supports_output_slots());
}
