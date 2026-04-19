//! Output-slot correctness tests for indexing ops (F.13):
//! GatherOp, ScatterNDOp, ScatterElementsOp.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::registry::indexing_ops::{GatherOp, ScatterElementsOp, ScatterNDOp};

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

// ── F.13 tests: indexing ops slot reuse ──────────────────────────────────────

#[test]
fn test_gather_slot_reuse() {
    // data: shape [4, 3], indices: [3] (axis 0)
    // output shape: [3, 3] — rows 0, 2, 1 of data
    assert!(
        GatherOp.supports_output_slots(),
        "GatherOp must support output slots"
    );

    let data = Tensor::new(
        vec![
            1.0_f32, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ],
        vec![4, 3],
    );
    let indices = Tensor::new(vec![0.0_f32, 2.0, 1.0], vec![3]);
    let node = node_with_int_attrs(OpKind::Gather, &[("axis", 0)]);
    let ctx = make_ctx(&node, vec![Some(&data), Some(&indices)]);

    let expected = GatherOp.execute(&ctx).expect("GatherOp execute failed");
    assert_eq!(expected[0].shape, vec![3, 3]);

    // First call — pre-allocate a slot with matching shape.
    let mut slots = vec![Tensor::new(vec![0.0_f32; 9], vec![3, 3])];
    GatherOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("GatherOp execute_into_slots failed");
    assert_tensor_eq(
        &slots[0],
        &expected[0],
        "gather slot correctness first call",
    );

    // Second call — same shape, assert pointer stability (no reallocation).
    let indices2 = Tensor::new(vec![3.0_f32, 0.0, 2.0], vec![3]);
    let ctx2 = make_ctx(&node, vec![Some(&data), Some(&indices2)]);
    let expected2 = GatherOp.execute(&ctx2).expect("GatherOp execute2 failed");

    let ptr_before = slots[0].data.as_ptr() as usize;
    GatherOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("GatherOp execute_into_slots2 failed");
    let ptr_after = slots[0].data.as_ptr() as usize;

    assert_eq!(
        ptr_before, ptr_after,
        "gather slot pointer must be stable on same-shape second call"
    );
    assert_tensor_eq(
        &slots[0],
        &expected2[0],
        "gather slot correctness second call",
    );
}

#[test]
fn test_scatter_nd_slot_reuse() {
    // data: [4, 3], indices: [2, 1] (k=1), updates: [2, 3]
    // output shape == data.shape == [4, 3]
    assert!(
        ScatterNDOp.supports_output_slots(),
        "ScatterNDOp must support output slots"
    );

    let data = Tensor::new(
        vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![4, 3],
    );
    // Scatter rows 1 and 3
    let indices = Tensor::new(vec![1.0_f32, 3.0], vec![2, 1]);
    let updates = Tensor::new(
        vec![100.0_f32, 101.0, 102.0, 200.0, 201.0, 202.0],
        vec![2, 3],
    );

    let node = dummy_node(OpKind::ScatterND);
    let ctx = make_ctx(&node, vec![Some(&data), Some(&indices), Some(&updates)]);

    let expected = ScatterNDOp
        .execute(&ctx)
        .expect("ScatterNDOp execute failed");
    assert_eq!(expected[0].shape, vec![4, 3]);

    // First call.
    let mut slots = vec![Tensor::new(vec![0.0_f32; 12], vec![4, 3])];
    ScatterNDOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ScatterNDOp execute_into_slots failed");
    assert_tensor_eq(
        &slots[0],
        &expected[0],
        "scatter_nd slot correctness first call",
    );

    // Second call — same shape (scatter rows 0 and 2), assert pointer stability.
    let indices2 = Tensor::new(vec![0.0_f32, 2.0], vec![2, 1]);
    let updates2 = Tensor::new(vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
    let ctx2 = make_ctx(&node, vec![Some(&data), Some(&indices2), Some(&updates2)]);
    let expected2 = ScatterNDOp
        .execute(&ctx2)
        .expect("ScatterNDOp execute2 failed");

    let ptr_before = slots[0].data.as_ptr() as usize;
    ScatterNDOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("ScatterNDOp execute_into_slots2 failed");
    let ptr_after = slots[0].data.as_ptr() as usize;

    assert_eq!(
        ptr_before, ptr_after,
        "scatter_nd slot pointer must be stable on same-shape second call"
    );
    assert_tensor_eq(
        &slots[0],
        &expected2[0],
        "scatter_nd slot correctness second call",
    );
}

#[test]
fn test_scatter_elements_slot_reuse() {
    // data: [3, 3], indices: [2, 2], updates: [2, 2], axis 1 (overwrite, no reduction)
    // output shape == data.shape == [3, 3]
    assert!(
        ScatterElementsOp.supports_output_slots(),
        "ScatterElementsOp must support output slots"
    );

    let data = Tensor::new(
        vec![0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![3, 3],
    );
    // For rows 0 and 1, scatter two elements along axis 1
    let indices = Tensor::new(vec![2.0_f32, 0.0, 1.0, 2.0], vec![2, 2]);
    let updates = Tensor::new(vec![9.0_f32, 7.0, 5.0, 3.0], vec![2, 2]);

    let node = node_with_int_attrs(OpKind::ScatterElements, &[("axis", 1)]);
    let ctx = make_ctx(&node, vec![Some(&data), Some(&indices), Some(&updates)]);

    let expected = ScatterElementsOp
        .execute(&ctx)
        .expect("ScatterElementsOp execute failed");
    assert_eq!(expected[0].shape, vec![3, 3]);

    // First call.
    let mut slots = vec![Tensor::new(vec![0.0_f32; 9], vec![3, 3])];
    ScatterElementsOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ScatterElementsOp execute_into_slots failed");
    assert_tensor_eq(
        &slots[0],
        &expected[0],
        "scatter_elements slot correctness first call",
    );

    // Second call — same-shape scatter to different positions, assert pointer stability.
    let indices2 = Tensor::new(vec![0.0_f32, 1.0, 0.0, 2.0], vec![2, 2]);
    let updates2 = Tensor::new(vec![11.0_f32, 22.0, 33.0, 44.0], vec![2, 2]);
    let ctx2 = make_ctx(&node, vec![Some(&data), Some(&indices2), Some(&updates2)]);
    let expected2 = ScatterElementsOp
        .execute(&ctx2)
        .expect("ScatterElementsOp execute2 failed");

    let ptr_before = slots[0].data.as_ptr() as usize;
    ScatterElementsOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("ScatterElementsOp execute_into_slots2 failed");
    let ptr_after = slots[0].data.as_ptr() as usize;

    assert_eq!(
        ptr_before, ptr_after,
        "scatter_elements slot pointer must be stable on same-shape second call"
    );
    assert_tensor_eq(
        &slots[0],
        &expected2[0],
        "scatter_elements slot correctness second call",
    );
}
