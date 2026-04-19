//! Output-slot correctness tests for convolution ops (F.13):
//! ConvOp (with and without fused ReLU), ConvTransposeOp.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::registry::conv_ops::{ConvOp, ConvTransposeOp};

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

// ── F.13 Block 2: ConvOp slot reuse ──────────────────────────────────────────

#[test]
fn test_conv_slot_reuse() {
    // ConvOp with input [1,3,8,8], weights [4,3,3,3]
    // pads [1,1,1,1], strides [1,1], dilations [1,1], group 1, bias [4]
    // Output shape: [1,4,8,8] = 256 elements
    assert!(
        ConvOp.supports_output_slots(),
        "ConvOp must support output slots"
    );

    let input = Tensor::new(
        (0..192).map(|i| (i as f32) * 0.01).collect(),
        vec![1, 3, 8, 8],
    );
    // weights [4, 3, 3, 3]
    let weights = Tensor::new(
        (0..108).map(|i| (i as f32) * 0.005 - 0.27).collect(),
        vec![4, 3, 3, 3],
    );
    let bias = Tensor::new(vec![0.1_f32, -0.1, 0.2, -0.2], vec![4]);

    let node = {
        let mut n = dummy_node(OpKind::Conv);
        n.attrs.int_lists.insert("strides".into(), vec![1i64, 1]);
        n.attrs.int_lists.insert("pads".into(), vec![1i64, 1, 1, 1]);
        n.attrs.int_lists.insert("dilations".into(), vec![1i64, 1]);
        n.attrs.ints.insert("group".into(), 1);
        n
    };
    let ctx = make_ctx(&node, vec![Some(&input), Some(&weights), Some(&bias)]);

    let expected = ConvOp.execute(&ctx).expect("ConvOp execute failed");
    assert_eq!(expected[0].shape, vec![1, 4, 8, 8]);
    assert_eq!(expected[0].data.len(), 256);

    // First call — pre-allocate a slot with matching shape.
    let mut slots = vec![Tensor::new(vec![0.0_f32; 256], vec![1, 4, 8, 8])];
    ConvOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ConvOp execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "conv slot correctness first call");

    // Second call — same shape, verify pointer stability (no reallocation).
    let input2 = Tensor::new(
        (0..192).map(|i| (i as f32) * 0.02).collect(),
        vec![1, 3, 8, 8],
    );
    let ctx2 = make_ctx(&node, vec![Some(&input2), Some(&weights), Some(&bias)]);
    let expected2 = ConvOp.execute(&ctx2).expect("ConvOp execute2 failed");

    let ptr_before = slots[0].data.as_ptr() as usize;
    ConvOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("ConvOp execute_into_slots2 failed");
    let ptr_after = slots[0].data.as_ptr() as usize;

    assert_eq!(
        ptr_before, ptr_after,
        "conv slot pointer must be stable on same-shape second call"
    );
    assert_tensor_eq(
        &slots[0],
        &expected2[0],
        "conv slot correctness second call",
    );
}

#[test]
fn test_conv_slot_reuse_with_fused_relu() {
    // Same conv configuration as test_conv_slot_reuse but with activation="relu".
    // Verifies that execute_into_slots applies fused activation identically to execute().
    assert!(
        ConvOp.supports_output_slots(),
        "ConvOp must support output slots"
    );

    let input = Tensor::new(
        (0..192).map(|i| (i as f32) * 0.01 - 0.96).collect(),
        vec![1, 3, 8, 8],
    );
    let weights = Tensor::new(
        (0..108).map(|i| (i as f32) * 0.005 - 0.27).collect(),
        vec![4, 3, 3, 3],
    );
    let bias = Tensor::new(vec![0.1_f32, -0.1, 0.2, -0.2], vec![4]);

    let node = {
        let mut n = dummy_node(OpKind::Conv);
        n.attrs.int_lists.insert("strides".into(), vec![1i64, 1]);
        n.attrs.int_lists.insert("pads".into(), vec![1i64, 1, 1, 1]);
        n.attrs.int_lists.insert("dilations".into(), vec![1i64, 1]);
        n.attrs.ints.insert("group".into(), 1);
        n.attrs.strings.insert("activation".into(), "relu".into());
        n
    };
    let ctx = make_ctx(&node, vec![Some(&input), Some(&weights), Some(&bias)]);

    let expected = ConvOp.execute(&ctx).expect("ConvOp execute (relu) failed");
    assert_eq!(expected[0].shape, vec![1, 4, 8, 8]);

    // Verify relu was applied: all values must be >= 0.
    for &v in &expected[0].data {
        assert!(
            v >= 0.0,
            "execute output contains negative value {v} despite relu"
        );
    }

    // First call into a fresh slot.
    let mut slots = vec![Tensor::new(vec![0.0_f32; 256], vec![1, 4, 8, 8])];
    ConvOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ConvOp execute_into_slots (relu) failed");
    assert_tensor_eq(
        &slots[0],
        &expected[0],
        "conv relu slot correctness first call",
    );

    // Second call with different input — pointer must remain stable.
    let input2 = Tensor::new(
        (0..192).map(|i| (i as f32) * 0.03 - 1.5).collect(),
        vec![1, 3, 8, 8],
    );
    let ctx2 = make_ctx(&node, vec![Some(&input2), Some(&weights), Some(&bias)]);
    let expected2 = ConvOp
        .execute(&ctx2)
        .expect("ConvOp execute2 (relu) failed");

    let ptr_before = slots[0].data.as_ptr() as usize;
    ConvOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("ConvOp execute_into_slots2 (relu) failed");
    let ptr_after = slots[0].data.as_ptr() as usize;

    assert_eq!(
        ptr_before, ptr_after,
        "conv relu slot pointer must be stable on same-shape second call"
    );
    assert_tensor_eq(
        &slots[0],
        &expected2[0],
        "conv relu slot correctness second call",
    );
}

// ── F.13 Block 3: ConvTransposeOp slot reuse ─────────────────────────────────

#[test]
fn test_conv_transpose_slot_reuse() {
    // ConvTransposeOp: input [1,4,4,4], weights [4,2,3,3], strides [2,2],
    // pads [0,0,0,0], output_padding [0,0], dilations [1,1], group 1
    // out_h = (4-1)*2 + 0 + (3-1)*1 + 1 - 0 - 0 = 6 + 2 + 1 = 9
    // out_w = same => output shape [1, 2, 9, 9] = 162 elements
    assert!(
        ConvTransposeOp.supports_output_slots(),
        "ConvTransposeOp must support output slots"
    );

    let node = {
        let mut n = dummy_node(OpKind::ConvTranspose);
        n.attrs.int_lists.insert("strides".into(), vec![2_i64, 2]);
        n.attrs
            .int_lists
            .insert("pads".into(), vec![0_i64, 0, 0, 0]);
        n.attrs
            .int_lists
            .insert("output_padding".into(), vec![0_i64, 0]);
        n.attrs.int_lists.insert("dilations".into(), vec![1_i64, 1]);
        n.attrs.ints.insert("group".into(), 1);
        n
    };

    let input = Tensor::new((0..64).map(|v| v as f32 * 0.1).collect(), vec![1, 4, 4, 4]);
    // weights: [C_in=4, C_out/group=2, kH=3, kW=3]
    let weight = Tensor::new(
        (0..72).map(|v| (v as f32 - 36.0) * 0.05).collect(),
        vec![4, 2, 3, 3],
    );
    // bias: [C_out=2]
    let bias = Tensor::new(vec![0.1_f32, -0.1], vec![2]);

    let ctx = make_ctx(&node, vec![Some(&input), Some(&weight), Some(&bias)]);
    let expected = ConvTransposeOp
        .execute(&ctx)
        .expect("ConvTransposeOp execute failed");
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0].shape, vec![1, 2, 9, 9]);
    assert_eq!(expected[0].data.len(), 162);

    // First call: fresh pre-allocated slot.
    let mut slots = vec![Tensor::new(vec![0.0_f32; 162], vec![1, 2, 9, 9])];
    ConvTransposeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ConvTransposeOp execute_into_slots failed");
    assert_tensor_eq(
        &slots[0],
        &expected[0],
        "conv_transpose slot correctness first call",
    );

    // Second call: same input/weight/bias — pointer must be stable (no realloc).
    let ptr_before = slots[0].data.as_ptr() as usize;
    ConvTransposeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ConvTransposeOp execute_into_slots second call failed");
    let ptr_after = slots[0].data.as_ptr() as usize;

    assert_eq!(
        ptr_before, ptr_after,
        "conv_transpose slot pointer must be stable on same-shape second call"
    );
    assert_tensor_eq(
        &slots[0],
        &expected[0],
        "conv_transpose slot correctness second call",
    );
}
