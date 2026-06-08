//! Output-slot correctness tests for shape ops (F.12):
//! SqueezeOp, UnsqueezeOp, FlattenOp, ExpandOp, SplitOp, TileOp,
//! DepthToSpaceOp, SpaceToDepthOp, ReverseSequenceOp.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::registry::shape_ops::{
    DepthToSpaceOp, ExpandOp, FlattenOp, ReverseSequenceOp, SpaceToDepthOp, SplitOp, SqueezeOp,
    TileOp, UnsqueezeOp,
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

fn node_with_int_list_attrs(op: OpKind, lists: &[(&str, Vec<i64>)]) -> Node {
    let mut n = dummy_node(op);
    for (k, v) in lists {
        n.attrs.int_lists.insert(k.to_string(), v.clone());
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

// ── F.12 tests: shape ops ─────────────────────────────────────────────────────

#[test]
fn test_squeeze_execute_into_slots_correctness() {
    assert!(SqueezeOp.supports_output_slots());
    // Input [1, 3, 1] → squeeze all size-1 axes → [3]
    let node = dummy_node(OpKind::Squeeze);
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0], vec![1, 3, 1]);
    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = SqueezeOp.execute(&ctx).expect("Squeeze execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    SqueezeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Squeeze execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "squeeze slots[0]");

    // Second call: squeeze specific axis on different input [1, 2, 1, 4] → axis 0,2 → [2, 4]
    let node2 = {
        let mut n = dummy_node(OpKind::Squeeze);
        n.attrs.int_lists.insert("axes".into(), vec![0i64, 2]);
        n
    };
    let input2 = Tensor::new((0..8).map(|i| i as f32).collect(), vec![1, 2, 1, 4]);
    let ctx2 = make_ctx(&node2, vec![Some(&input2)]);
    let expected2 = SqueezeOp.execute(&ctx2).expect("Squeeze execute2 failed");
    SqueezeOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("Squeeze execute_into_slots2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "squeeze slots[0] second call");
}

#[test]
fn test_unsqueeze_execute_into_slots_correctness() {
    assert!(UnsqueezeOp.supports_output_slots());
    // Input [3] → unsqueeze axis 0 → [1, 3]
    let node = node_with_int_list_attrs(OpKind::Unsqueeze, &[("axes", vec![0i64])]);
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0], vec![3]);
    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = UnsqueezeOp.execute(&ctx).expect("Unsqueeze execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    UnsqueezeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Unsqueeze execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "unsqueeze slots[0]");

    // Second call: unsqueeze axes [0, 2] on [2, 3] → [1, 2, 1, 3]
    let node2 = node_with_int_list_attrs(OpKind::Unsqueeze, &[("axes", vec![0i64, 2])]);
    let input2 = Tensor::new((0..6).map(|i| i as f32).collect(), vec![2, 3]);
    let ctx2 = make_ctx(&node2, vec![Some(&input2)]);
    let expected2 = UnsqueezeOp
        .execute(&ctx2)
        .expect("Unsqueeze execute2 failed");
    UnsqueezeOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("Unsqueeze execute_into_slots2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "unsqueeze slots[0] second call");
}

#[test]
fn test_flatten_execute_into_slots_correctness() {
    assert!(FlattenOp.supports_output_slots());
    // Input [2, 3, 4] axis=1 → [2, 12]
    let node = node_with_int_attrs(OpKind::Flatten, &[("axis", 1)]);
    let input = Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = FlattenOp.execute(&ctx).expect("Flatten execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    FlattenOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Flatten execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "flatten slots[0]");

    // Second call: axis=2 on [3, 4, 2] → [12, 2]
    let node2 = node_with_int_attrs(OpKind::Flatten, &[("axis", 2)]);
    let input2 = Tensor::new((0..24).map(|i| i as f32).collect(), vec![3, 4, 2]);
    let ctx2 = make_ctx(&node2, vec![Some(&input2)]);
    let expected2 = FlattenOp.execute(&ctx2).expect("Flatten execute2 failed");
    FlattenOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("Flatten execute_into_slots2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "flatten slots[0] second call");
}

#[test]
fn test_expand_execute_into_slots_correctness() {
    assert!(ExpandOp.supports_output_slots());
    // Broadcast [1, 3] to [2, 3]
    let node = dummy_node(OpKind::Expand);
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0], vec![1, 3]);
    let shape_t = Tensor::new(vec![2.0_f32, 3.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&input), Some(&shape_t)]);
    let expected = ExpandOp.execute(&ctx).expect("Expand execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    ExpandOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Expand execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "expand slots[0]");

    // Second call: scalar broadcast [1] to [4]
    let input2 = Tensor::new(vec![5.0_f32], vec![1]);
    let shape_t2 = Tensor::new(vec![4.0_f32], vec![1]);
    let ctx2 = make_ctx(&node, vec![Some(&input2), Some(&shape_t2)]);
    let expected2 = ExpandOp.execute(&ctx2).expect("Expand execute2 failed");
    ExpandOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("Expand execute_into_slots2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "expand slots[0] second call");
}

#[test]
fn test_split_execute_into_slots_correctness() {
    assert!(SplitOp.supports_output_slots());
    // Split [2, 6] along axis 1 into [2,2] and [2,4]
    let node = {
        let mut n = node_with_int_attrs(OpKind::Split, &[("axis", 1)]);
        n.outputs = vec!["a".into(), "b".into()];
        n
    };
    let input = Tensor::new((0..12).map(|i| i as f32).collect(), vec![2, 6]);
    let split_t = Tensor::new(vec![2.0_f32, 4.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&input), Some(&split_t)]);
    let expected = SplitOp.execute(&ctx).expect("Split execute failed");
    assert_eq!(expected.len(), 2);
    let mut slots = vec![
        Tensor::new(
            vec![0.0_f32; expected[0].data.len()],
            expected[0].shape.clone(),
        ),
        Tensor::new(
            vec![0.0_f32; expected[1].data.len()],
            expected[1].shape.clone(),
        ),
    ];
    SplitOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Split execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "split slots[0]");
    assert_tensor_eq(&slots[1], &expected[1], "split slots[1]");
}

/// Regression for issue #1 (YOLO11 fails with
/// `reshape: element count mismatch (33600 vs [1, 128, 20, 20])`).
///
/// YOLO11's C2PSA attention block is exported at opset 11, where `Split`
/// carries its chunk sizes in the `split` *attribute* (an int list), not in a
/// second input tensor. The qkv tensor `[1, 2, 128, 400]` must be split along
/// axis 2 into `[32, 32, 64]` (query/key/value channels). The previous
/// implementation ignored the `split` attribute and fell back to an even
/// 3-way split of 128 → `[43, 43, 42]`, so the value branch became
/// `[1, 2, 42, 400]` (33600 elements) and the downstream Reshape to
/// `[1, 128, 20, 20]` (51200 elements) failed.
#[test]
fn test_issue_1_split_uses_split_attribute() {
    // Reproduce the failing C2PSA qkv split: [1, 2, 128, 400] split on axis 2
    // into [32, 32, 64] via the opset-11 `split` attribute.
    let node = {
        let mut n = node_with_int_attrs(OpKind::Split, &[("axis", 2)]);
        n.attrs
            .int_lists
            .insert("split".into(), vec![32i64, 32, 64]);
        n.outputs = vec!["q".into(), "k".into(), "v".into()];
        n
    };
    let input = Tensor::new(vec![0.0_f32; 2 * 128 * 400], vec![1, 2, 128, 400]);
    // No second input tensor — sizes must come from the attribute.
    let ctx = make_ctx(&node, vec![Some(&input)]);
    let outputs = SplitOp.execute(&ctx).expect("Split execute failed");
    assert_eq!(outputs.len(), 3, "expected 3 split outputs");
    assert_eq!(outputs[0].shape, vec![1, 2, 32, 400], "query chunk shape");
    assert_eq!(outputs[1].shape, vec![1, 2, 32, 400], "key chunk shape");
    assert_eq!(
        outputs[2].shape,
        vec![1, 2, 64, 400],
        "value chunk shape (33600-vs-51200 regression)",
    );

    // The output-slot path must honour the attribute identically.
    let mut slots = vec![
        Tensor::new(
            vec![0.0_f32; outputs[0].data.len()],
            outputs[0].shape.clone(),
        ),
        Tensor::new(
            vec![0.0_f32; outputs[1].data.len()],
            outputs[1].shape.clone(),
        ),
        Tensor::new(
            vec![0.0_f32; outputs[2].data.len()],
            outputs[2].shape.clone(),
        ),
    ];
    SplitOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Split execute_into_slots failed");
    assert_eq!(slots[2].shape, vec![1, 2, 64, 400], "value slot shape");
}

#[test]
fn test_tile_execute_into_slots_correctness() {
    assert!(TileOp.supports_output_slots());
    // Tile [1, 3] by [2, 1] → [2, 3]
    let node = dummy_node(OpKind::Tile);
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0], vec![1, 3]);
    let repeats = Tensor::new(vec![2.0_f32, 1.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&input), Some(&repeats)]);
    let expected = TileOp.execute(&ctx).expect("Tile execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    TileOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("Tile execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "tile slots[0]");

    // Second call: different repeats → different output size
    let input2 = Tensor::new(vec![1.0_f32, 2.0], vec![1, 2]);
    let repeats2 = Tensor::new(vec![3.0_f32, 2.0], vec![2]);
    let ctx2 = make_ctx(&node, vec![Some(&input2), Some(&repeats2)]);
    let expected2 = TileOp.execute(&ctx2).expect("Tile execute2 failed");
    TileOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("Tile execute_into_slots2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "tile slots[0] second call");
}

#[test]
fn test_depth_to_space_execute_into_slots_correctness() {
    assert!(DepthToSpaceOp.supports_output_slots());
    // [1, 4, 2, 2] with blocksize=2, mode=DCR → [1, 1, 4, 4]
    let node = {
        let mut n = node_with_int_attrs(OpKind::DepthToSpace, &[("blocksize", 2)]);
        n.attrs.strings.insert("mode".into(), "DCR".into());
        n
    };
    let input = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 4, 2, 2]);
    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = DepthToSpaceOp
        .execute(&ctx)
        .expect("DepthToSpace execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    DepthToSpaceOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("DepthToSpace execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "depth_to_space slots[0] DCR");

    // Second call: CRD mode
    let node_crd = {
        let mut n = node_with_int_attrs(OpKind::DepthToSpace, &[("blocksize", 2)]);
        n.attrs.strings.insert("mode".into(), "CRD".into());
        n
    };
    let ctx_crd = make_ctx(&node_crd, vec![Some(&input)]);
    let expected_crd = DepthToSpaceOp
        .execute(&ctx_crd)
        .expect("DepthToSpace CRD execute failed");
    DepthToSpaceOp
        .execute_into_slots(&ctx_crd, &mut slots)
        .expect("DepthToSpace CRD execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected_crd[0], "depth_to_space slots[0] CRD");
}

#[test]
fn test_space_to_depth_execute_into_slots_correctness() {
    assert!(SpaceToDepthOp.supports_output_slots());
    // [1, 1, 4, 4] with blocksize=2 → [1, 4, 2, 2]
    let node = node_with_int_attrs(OpKind::SpaceToDepth, &[("blocksize", 2)]);
    let input = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]);
    let ctx = make_ctx(&node, vec![Some(&input)]);
    let expected = SpaceToDepthOp
        .execute(&ctx)
        .expect("SpaceToDepth execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    SpaceToDepthOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("SpaceToDepth execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "space_to_depth slots[0]");

    // Second call: different input [1, 2, 2, 2] → [1, 8, 1, 1]
    let input2 = Tensor::new(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 2, 2, 2],
    );
    let ctx2 = make_ctx(&node, vec![Some(&input2)]);
    let expected2 = SpaceToDepthOp
        .execute(&ctx2)
        .expect("SpaceToDepth execute2 failed");
    SpaceToDepthOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("SpaceToDepth execute_into_slots2 failed");
    assert_tensor_eq(
        &slots[0],
        &expected2[0],
        "space_to_depth slots[0] second call",
    );
}

#[test]
fn test_reverse_sequence_execute_into_slots_correctness() {
    assert!(ReverseSequenceOp.supports_output_slots());
    // [2, 4] batch_axis=0, time_axis=1, seq_lens=[3, 2]
    let node = node_with_int_attrs(
        OpKind::ReverseSequence,
        &[("batch_axis", 0), ("time_axis", 1)],
    );
    let input = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let seq_lens = Tensor::new(vec![3.0_f32, 2.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&input), Some(&seq_lens)]);
    let expected = ReverseSequenceOp
        .execute(&ctx)
        .expect("ReverseSequence execute failed");
    let mut slots = vec![Tensor::new(
        vec![0.0_f32; expected[0].data.len()],
        expected[0].shape.clone(),
    )];
    ReverseSequenceOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ReverseSequence execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "reverse_sequence slots[0]");

    // Second call: full reversal
    let input2 = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let seq_lens2 = Tensor::new(vec![4.0_f32], vec![1]);
    let node2 = node_with_int_attrs(
        OpKind::ReverseSequence,
        &[("batch_axis", 0), ("time_axis", 1)],
    );
    let ctx2 = make_ctx(&node2, vec![Some(&input2), Some(&seq_lens2)]);
    let expected2 = ReverseSequenceOp
        .execute(&ctx2)
        .expect("ReverseSequence execute2 failed");
    ReverseSequenceOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("ReverseSequence execute_into_slots2 failed");
    assert_tensor_eq(
        &slots[0],
        &expected2[0],
        "reverse_sequence slots[0] second call",
    );
}
