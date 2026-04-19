//! Output-slot correctness tests for basic ops:
//! SoftmaxOp, ReduceSumOp, TransposeOp, LogSoftmaxOp, LayerNormOp, ReduceMeanOp,
//! MatMulOp, GemmOp, ConcatOp, BatchNormOp, RmsNormOp, InstanceNormOp, SliceOp.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::registry::{
    math_ops::{GemmOp, MatMulOp, ReduceMeanOp, ReduceSumOp},
    nn_ops::{BatchNormOp, InstanceNormOp, LayerNormOp, LogSoftmaxOp, RmsNormOp, SoftmaxOp},
    shape_ops::{ConcatOp, SliceOp, TransposeOp},
};

// ── Test infrastructure ──────────────────────────────────────────────────────

fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
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

fn node_with_float_attrs(op: OpKind, pairs: &[(&str, f32)]) -> Node {
    let mut n = dummy_node(op);
    for &(k, v) in pairs {
        n.attrs.floats.insert(k.to_string(), v);
    }
    n
}

fn assert_tensor_eq(a: &Tensor, b: &Tensor, label: &str) {
    assert_eq!(a.shape, b.shape, "{label}: shape mismatch");
    assert_eq!(a.data.len(), b.data.len(), "{label}: data len mismatch");
    for (i, (&av, &bv)) in a.data.iter().zip(b.data.iter()).enumerate() {
        assert!(
            (av - bv).abs() < 1e-6 || (av.is_nan() && bv.is_nan()),
            "{label}[{i}]: got {av}, expected {bv}",
        );
    }
}

// ── Test: SoftmaxOp ─────────────────────────────────────────────────────────

#[test]
fn test_softmax_execute_into_slots_correctness() {
    assert!(
        SoftmaxOp.supports_output_slots(),
        "SoftmaxOp must support output slots"
    );

    let node = node_with_int_attrs(OpKind::Softmax, &[("axis", -1)]);
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0, 1.0, 0.5], vec![2, 3]);

    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = SoftmaxOp.execute(&ctx).expect("Softmax execute failed");
    assert_eq!(expected.len(), 1);

    // First call into a fresh slot.
    let mut slots = vec![Tensor::new(vec![0.0_f32; 6], vec![2, 3])];
    SoftmaxOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Softmax execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "softmax slots[0] vs execute");

    // Second call: same slot, different-shaped input (exercises shape mismatch / replace).
    let input2 = Tensor::new(vec![0.1_f32, 0.2, 0.7], vec![1, 3]);
    let ctx2 = make_ctx(&node, vec![Some(&input2)]);
    let expected2 = SoftmaxOp.execute(&ctx2).expect("Softmax execute2 failed");
    SoftmaxOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("Softmax execute_into_slots2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "softmax slots[0] second call");
}

// ── Test: ReduceSumOp ────────────────────────────────────────────────────────

#[test]
fn test_reduce_sum_execute_into_slots_correctness() {
    assert!(
        ReduceSumOp.supports_output_slots(),
        "ReduceSumOp must support output slots"
    );

    // keepdims=1, reduce along axis 1
    let node = {
        let mut n = dummy_node(OpKind::ReduceSum);
        n.attrs.ints.insert("keepdims".into(), 1);
        n.attrs.int_lists.insert("axes".into(), vec![1]);
        n
    };
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = ReduceSumOp.execute(&ctx).expect("ReduceSum execute failed");
    assert_eq!(expected.len(), 1);

    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    ReduceSumOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ReduceSum execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "reduce_sum slots[0] vs execute");

    // Second call with a different shape (keepdims=0 reduces to scalar).
    let node2 = {
        let mut n = dummy_node(OpKind::ReduceSum);
        n.attrs.ints.insert("keepdims".into(), 0);
        n
    };
    let input2 = Tensor::new(vec![1.0_f32, 2.0, 3.0], vec![3]);
    let ctx2 = make_ctx(&node2, vec![Some(&input2)]);
    let expected2 = ReduceSumOp
        .execute(&ctx2)
        .expect("ReduceSum execute2 failed");
    ReduceSumOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("ReduceSum execute_into_slots2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "reduce_sum slots[0] second call");
}

// ── Test: TransposeOp ────────────────────────────────────────────────────────

#[test]
fn test_transpose_execute_into_slots_correctness() {
    assert!(
        TransposeOp.supports_output_slots(),
        "TransposeOp must support output slots"
    );

    let node = {
        let mut n = dummy_node(OpKind::Transpose);
        n.attrs.int_lists.insert("perm".into(), vec![1, 0]);
        n
    };
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = TransposeOp.execute(&ctx).expect("Transpose execute failed");
    assert_eq!(expected.len(), 1);

    // Pre-allocate slot with matching shape.
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    TransposeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Transpose execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "transpose slots[0] vs execute");

    // Second call: different permutation produces different output shape.
    let node3d = {
        let mut n = dummy_node(OpKind::Transpose);
        n.attrs.int_lists.insert("perm".into(), vec![2, 0, 1]);
        n
    };
    let input2 = Tensor::new((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);
    let ctx2 = make_ctx(&node3d, vec![Some(&input2)]);
    let expected2 = TransposeOp
        .execute(&ctx2)
        .expect("Transpose execute2 failed");
    TransposeOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("Transpose execute_into_slots2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "transpose slots[0] second call");
}

// ── Tests: additional basic ops ──────────────────────────────────────────────

#[test]
fn test_log_softmax_supports_output_slots() {
    assert!(LogSoftmaxOp.supports_output_slots());
    let node = node_with_int_attrs(OpKind::LogSoftmax, &[("axis", -1)]);
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0], vec![1, 3]);
    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = LogSoftmaxOp
        .execute(&ctx)
        .expect("LogSoftmax execute failed");
    let mut slots = vec![Tensor::new(vec![0.0_f32; 3], vec![1, 3])];
    LogSoftmaxOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("LogSoftmax execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "log_softmax slot correctness");
}

#[test]
fn test_layer_norm_supports_output_slots() {
    assert!(LayerNormOp.supports_output_slots());
    let node = node_with_float_attrs(OpKind::LayerNorm, &[("epsilon", 1e-5)]);
    let x = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let scale = Tensor::new(vec![1.0_f32; 4], vec![4]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&scale)]);
    let expected = LayerNormOp.execute(&ctx).expect("LayerNorm execute failed");
    let mut slots = vec![Tensor::new(vec![0.0_f32; 4], vec![1, 4])];
    LayerNormOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("LayerNorm execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "layer_norm slot correctness");
}

#[test]
fn test_reduce_mean_supports_output_slots() {
    assert!(ReduceMeanOp.supports_output_slots());
    let node = {
        let mut n = dummy_node(OpKind::ReduceMean);
        n.attrs.ints.insert("keepdims".into(), 0);
        n
    };
    let input = Tensor::new(vec![2.0_f32, 4.0, 6.0, 8.0], vec![4]);
    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = ReduceMeanOp
        .execute(&ctx)
        .expect("ReduceMean execute failed");
    let mut slots = vec![Tensor::new(vec![0.0_f32; 1], vec![1])];
    ReduceMeanOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ReduceMean execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "reduce_mean slot correctness");
}

#[test]
fn test_matmul_supports_output_slots() {
    assert!(MatMulOp.supports_output_slots());
    let node = dummy_node(OpKind::MatMul);
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]);
    let ctx = make_ctx(&node, vec![Some(&a), Some(&b)]);
    let expected = MatMulOp.execute(&ctx).expect("MatMul execute failed");
    let mut slots = vec![Tensor::new(vec![0.0_f32; 4], vec![2, 2])];
    MatMulOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("MatMul execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "matmul slot correctness");
}

#[test]
fn test_gemm_supports_output_slots() {
    assert!(GemmOp.supports_output_slots());
    let node = dummy_node(OpKind::Gemm);
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]);
    let ctx = make_ctx(&node, vec![Some(&a), Some(&b), None]);
    let expected = GemmOp.execute(&ctx).expect("Gemm execute failed");
    let mut slots = vec![Tensor::new(vec![0.0_f32; 4], vec![2, 2])];
    GemmOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Gemm execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "gemm slot correctness");
}

#[test]
fn test_concat_supports_output_slots() {
    assert!(ConcatOp.supports_output_slots());
    let node = node_with_int_attrs(OpKind::Concat, &[("axis", 0)]);
    let a = Tensor::new(vec![1.0_f32, 2.0], vec![2]);
    let b = Tensor::new(vec![3.0_f32, 4.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&a), Some(&b)]);
    let expected = ConcatOp.execute(&ctx).expect("Concat execute failed");
    let mut slots = vec![Tensor::new(vec![0.0_f32; 4], vec![4])];
    ConcatOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Concat execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "concat slot correctness");
}

#[test]
fn test_batch_norm_supports_output_slots() {
    assert!(BatchNormOp.supports_output_slots());
    let node = node_with_float_attrs(OpKind::BatchNorm, &[("epsilon", 1e-5)]);
    let n = 4usize;
    let x = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![1, n, 1, 1]);
    let scale = Tensor::new(vec![1.0_f32; n], vec![n]);
    let bias = Tensor::new(vec![0.0_f32; n], vec![n]);
    let mean = Tensor::new(vec![0.0_f32; n], vec![n]);
    let var = Tensor::new(vec![1.0_f32; n], vec![n]);
    let ctx = make_ctx(
        &node,
        vec![Some(&x), Some(&scale), Some(&bias), Some(&mean), Some(&var)],
    );
    let expected = BatchNormOp.execute(&ctx).expect("BatchNorm execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    BatchNormOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("BatchNorm execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "batch_norm slot correctness");
}

#[test]
fn test_rms_norm_supports_output_slots() {
    assert!(RmsNormOp.supports_output_slots());
    let node = node_with_float_attrs(OpKind::RMSNorm, &[("epsilon", 1e-6)]);
    let x = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let scale = Tensor::new(vec![1.0_f32; 4], vec![4]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&scale)]);
    let expected = RmsNormOp.execute(&ctx).expect("RmsNorm execute failed");
    let mut slots = vec![Tensor::new(vec![0.0_f32; 4], vec![1, 4])];
    RmsNormOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("RmsNorm execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "rms_norm slot correctness");
}

#[test]
fn test_instance_norm_supports_output_slots() {
    assert!(InstanceNormOp.supports_output_slots());
    let node = node_with_float_attrs(OpKind::InstanceNorm, &[("epsilon", 1e-5)]);
    // x: [N=1, C=2, H=2, W=2]
    let x = Tensor::new(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 2, 2, 2],
    );
    let scale = Tensor::new(vec![1.0_f32, 1.0], vec![2]);
    let bias = Tensor::new(vec![0.0_f32, 0.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&scale), Some(&bias)]);
    let expected = InstanceNormOp
        .execute(&ctx)
        .expect("InstanceNorm execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    InstanceNormOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("InstanceNorm execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "instance_norm slot correctness");
}

#[test]
fn test_slice_supports_output_slots() {
    assert!(SliceOp.supports_output_slots());
    let node = dummy_node(OpKind::Slice);
    let data = Tensor::new(vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    let starts = Tensor::new(vec![0.0_f32], vec![1]);
    let ends = Tensor::new(vec![2.0_f32], vec![1]);
    let ctx = make_ctx(
        &node,
        vec![Some(&data), Some(&starts), Some(&ends), None, None],
    );
    let expected = SliceOp.execute(&ctx).expect("Slice execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    SliceOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Slice execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "slice slot correctness");
}
