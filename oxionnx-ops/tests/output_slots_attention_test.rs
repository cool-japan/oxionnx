//! Output-slot correctness tests for AttentionOp and MultiHeadAttentionOp (Phase F.13).
//!
//! Verifies that `execute_into_slots` produces byte-identical results to `execute`,
//! covers pool-reuse reshape, broadcast masks, custom scales, qkv-projection paths,
//! and out-projection present/absent branches.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::registry::rnn_ops::{AttentionOp, MultiHeadAttentionOp};

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
            (av - bv).abs() < 1e-5 || (av.is_nan() && bv.is_nan()),
            "{label}[{i}]: got {av}, expected {bv}",
        );
    }
}

/// Create a linearly-spaced f32 tensor.
fn linspace(shape: &[usize], start: f32, step: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
    Tensor::new(data, shape.to_vec())
}

// ── AttentionOp tests ────────────────────────────────────────────────────────

#[test]
fn test_attention_supports_output_slots() {
    assert!(
        AttentionOp.supports_output_slots(),
        "AttentionOp must support output slots"
    );
}

#[test]
fn test_attention_into_slots_no_mask_default_scale() {
    let node = dummy_node(OpKind::Attention);
    // Q: [2, 3, 4], K: [2, 3, 4], V: [2, 3, 4]
    let q = linspace(&[2, 3, 4], 0.1, 0.05);
    let k = linspace(&[2, 3, 4], 0.2, 0.04);
    let v = linspace(&[2, 3, 4], 0.3, 0.03);

    let ctx = make_ctx(&node, vec![Some(&q), Some(&k), Some(&v), None]);
    let expected = AttentionOp.execute(&ctx).expect("execute failed");

    let mut slots = vec![Tensor::new(vec![0.0f32; 24], vec![2, 3, 4])];
    AttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "no-mask default-scale");
}

#[test]
fn test_attention_into_slots_with_full_mask() {
    let node = dummy_node(OpKind::Attention);
    let q = linspace(&[2, 4, 8], 0.1, 0.02);
    let k = linspace(&[2, 4, 8], 0.05, 0.03);
    let v = linspace(&[2, 4, 8], 0.15, 0.02);
    // Additive mask [2, 4, 4] — same batch as Q/K
    let mask = Tensor::new(vec![0.0f32; 2 * 4 * 4], vec![2, 4, 4]);

    let ctx = make_ctx(&node, vec![Some(&q), Some(&k), Some(&v), Some(&mask)]);
    let expected = AttentionOp.execute(&ctx).expect("execute failed");

    let mut slots = vec![Tensor::new(vec![0.0f32; 2 * 4 * 8], vec![2, 4, 8])];
    AttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "full mask");
}

#[test]
fn test_attention_into_slots_broadcast_mask_batch1() {
    let node = dummy_node(OpKind::Attention);
    // batch=2 Q/K, broadcast mask with batch=1
    let q = linspace(&[2, 3, 4], 0.1, 0.05);
    let k = linspace(&[2, 3, 4], 0.2, 0.04);
    let v = linspace(&[2, 3, 4], 0.0, 0.06);
    let mask = Tensor::new(vec![-0.5f32; 3 * 3], vec![1, 3, 3]);

    let ctx = make_ctx(&node, vec![Some(&q), Some(&k), Some(&v), Some(&mask)]);
    let expected = AttentionOp.execute(&ctx).expect("execute failed");

    let mut slots = vec![Tensor::new(vec![0.0f32; 2 * 3 * 4], vec![2, 3, 4])];
    AttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "broadcast mask batch=1");
}

#[test]
fn test_attention_into_slots_custom_scale() {
    let node = node_with_float_attrs(OpKind::Attention, &[("scale", 0.5)]);
    let q = linspace(&[1, 4, 4], 0.1, 0.1);
    let k = linspace(&[1, 4, 4], 0.0, 0.1);
    let v = linspace(&[1, 4, 4], 0.2, 0.1);

    let ctx = make_ctx(&node, vec![Some(&q), Some(&k), Some(&v), None]);
    let expected = AttentionOp.execute(&ctx).expect("execute failed");

    let mut slots = vec![Tensor::new(vec![0.0f32; 4 * 4], vec![1, 4, 4])];
    AttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "custom scale=0.5");
}

#[test]
fn test_attention_into_slots_scale_zero_is_default() {
    // scale=0.0 must behave identically to not providing scale (uses 1/sqrt(d_k))
    let node_default = dummy_node(OpKind::Attention);
    let node_zero = node_with_float_attrs(OpKind::Attention, &[("scale", 0.0)]);
    let q = linspace(&[1, 3, 8], 0.1, 0.05);
    let k = linspace(&[1, 3, 8], 0.0, 0.07);
    let v = linspace(&[1, 3, 8], 0.3, 0.04);

    let ctx_default = make_ctx(&node_default, vec![Some(&q), Some(&k), Some(&v), None]);
    let ctx_zero = make_ctx(&node_zero, vec![Some(&q), Some(&k), Some(&v), None]);

    let expected = AttentionOp.execute(&ctx_default).expect("execute failed");

    let mut slots = vec![Tensor::new(vec![0.0f32; 3 * 8], vec![1, 3, 8])];
    AttentionOp
        .execute_into_slots(&ctx_zero, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "scale=0.0 == default");
}

#[test]
fn test_attention_into_slots_batch_gt1_3d() {
    let node = dummy_node(OpKind::Attention);
    // batch=4, seq_q=seq_k=5, d_k=d_v=6
    let q = linspace(&[4, 5, 6], 0.0, 0.02);
    let k = linspace(&[4, 5, 6], 0.1, 0.02);
    let v = linspace(&[4, 5, 6], 0.5, 0.01);

    let ctx = make_ctx(&node, vec![Some(&q), Some(&k), Some(&v), None]);
    let expected = AttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    AttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "batch=4 3D");
}

#[test]
fn test_attention_into_slots_4d_q_mha_style() {
    let node = dummy_node(OpKind::Attention);
    // [batch=2, num_heads=3, seq=4, head_dim=8]
    let q = linspace(&[2, 3, 4, 8], 0.0, 0.01);
    let k = linspace(&[2, 3, 4, 8], 0.5, 0.01);
    let v = linspace(&[2, 3, 4, 8], 1.0, 0.01);

    let ctx = make_ctx(&node, vec![Some(&q), Some(&k), Some(&v), None]);
    let expected = AttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    AttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "4D Q MHA-style");
}

#[test]
fn test_attention_into_slots_cross_attention_seq_k_ne_seq_q() {
    let node = dummy_node(OpKind::Attention);
    // Cross-attention: seq_q=3, seq_k=5
    let q = linspace(&[2, 3, 8], 0.0, 0.05);
    let k = linspace(&[2, 5, 8], 0.1, 0.04);
    let v = linspace(&[2, 5, 8], 0.2, 0.03);

    let ctx = make_ctx(&node, vec![Some(&q), Some(&k), Some(&v), None]);
    let expected = AttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    AttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "cross-attention seq_k≠seq_q");
}

#[test]
fn test_attention_into_slots_second_call_different_shape_resizes() {
    let node = dummy_node(OpKind::Attention);

    // First call: [1, 2, 4]
    let q1 = linspace(&[1, 2, 4], 0.1, 0.1);
    let k1 = linspace(&[1, 2, 4], 0.0, 0.1);
    let v1 = linspace(&[1, 2, 4], 0.5, 0.1);
    let ctx1 = make_ctx(&node, vec![Some(&q1), Some(&k1), Some(&v1), None]);
    let expected1 = AttentionOp.execute(&ctx1).expect("execute 1 failed");

    let mut slots = vec![Tensor::new(vec![0.0f32; 8], vec![1, 2, 4])];
    AttentionOp
        .execute_into_slots(&ctx1, &mut slots)
        .expect("slot write 1 failed");
    assert_tensor_eq(&slots[0], &expected1[0], "first call shape [1,2,4]");

    // Second call: [2, 3, 4] — different shape → slot must resize
    let q2 = linspace(&[2, 3, 4], 0.2, 0.05);
    let k2 = linspace(&[2, 3, 4], 0.1, 0.05);
    let v2 = linspace(&[2, 3, 4], 0.0, 0.05);
    let ctx2 = make_ctx(&node, vec![Some(&q2), Some(&k2), Some(&v2), None]);
    let expected2 = AttentionOp.execute(&ctx2).expect("execute 2 failed");

    AttentionOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("slot write 2 failed");
    assert_tensor_eq(
        &slots[0],
        &expected2[0],
        "second call shape [2,3,4] (resized)",
    );
}

#[test]
fn test_attention_into_slots_oversized_incoming_slot_shrinks() {
    let node = dummy_node(OpKind::Attention);
    let q = linspace(&[1, 2, 4], 0.1, 0.1);
    let k = linspace(&[1, 2, 4], 0.0, 0.1);
    let v = linspace(&[1, 2, 4], 0.5, 0.1);

    let ctx = make_ctx(&node, vec![Some(&q), Some(&k), Some(&v), None]);
    let expected = AttentionOp.execute(&ctx).expect("execute failed");

    // Slot pre-allocated far too large (64 elements vs 8 needed)
    let mut slots = vec![Tensor::new(vec![99.0f32; 64], vec![8, 8])];
    AttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "oversized slot shrinks");
}

// ── MultiHeadAttentionOp tests ───────────────────────────────────────────────

#[test]
fn test_mha_supports_output_slots() {
    assert!(
        MultiHeadAttentionOp.supports_output_slots(),
        "MultiHeadAttentionOp must support output slots"
    );
}

#[test]
fn test_mha_into_slots_no_qkv_no_outproj() {
    // No qkv_weight, no out_proj_weight → uses reshape_from_heads_into path.
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 2)]);
    let query = linspace(&[1, 4, 8], 0.0, 0.1);
    let key = linspace(&[1, 4, 8], 0.5, 0.1);
    let value = linspace(&[1, 4, 8], 1.0, 0.05);

    let ctx = make_ctx(
        &node,
        vec![
            Some(&query),
            Some(&key),
            Some(&value),
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let expected = MultiHeadAttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    MultiHeadAttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "no-qkv no-outproj");
}

#[test]
fn test_mha_into_slots_with_outproj_no_bias() {
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 2)]);
    let embed = 8usize;
    let query = linspace(&[1, 3, embed], 0.0, 0.1);
    let key = linspace(&[1, 3, embed], 0.3, 0.1);
    let value = linspace(&[1, 3, embed], 0.6, 0.05);
    // out_proj_weight: [embed, embed]
    let out_proj_w = linspace(&[embed, embed], 0.01, 0.01);

    let ctx = make_ctx(
        &node,
        vec![
            Some(&query),
            Some(&key),
            Some(&value),
            None,
            None,
            Some(&out_proj_w),
            None,
            None,
        ],
    );
    let expected = MultiHeadAttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    MultiHeadAttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "outproj no bias");
}

#[test]
fn test_mha_into_slots_with_outproj_and_bias() {
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 2)]);
    let embed = 8usize;
    let seq = 4usize;
    let query = linspace(&[2, seq, embed], 0.0, 0.05);
    let key = linspace(&[2, seq, embed], 0.2, 0.05);
    let value = linspace(&[2, seq, embed], 0.4, 0.05);
    let out_proj_w = linspace(&[embed, embed], 0.0, 0.02);
    let out_proj_b = linspace(&[embed], 0.1, 0.01);

    let ctx = make_ctx(
        &node,
        vec![
            Some(&query),
            Some(&key),
            Some(&value),
            None,
            None,
            Some(&out_proj_w),
            Some(&out_proj_b),
            None,
        ],
    );
    let expected = MultiHeadAttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    MultiHeadAttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "outproj + bias");
}

#[test]
fn test_mha_into_slots_with_qkv_weight_and_bias() {
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 2)]);
    let embed = 8usize;
    let seq = 3usize;
    let query = linspace(&[1, seq, embed], 0.0, 0.1);
    let key = linspace(&[1, seq, embed], 0.5, 0.1);
    let value = linspace(&[1, seq, embed], 1.0, 0.05);
    // qkv_weight: [3*embed, embed], qkv_bias: [3*embed]
    let qkv_w = linspace(&[3 * embed, embed], 0.01, 0.01);
    let qkv_b = linspace(&[3 * embed], 0.0, 0.05);

    let ctx = make_ctx(
        &node,
        vec![
            Some(&query),
            Some(&key),
            Some(&value),
            Some(&qkv_w),
            Some(&qkv_b),
            None,
            None,
            None,
        ],
    );
    let expected = MultiHeadAttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    MultiHeadAttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "qkv_weight+bias");
}

#[test]
fn test_mha_into_slots_num_heads_gt1() {
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 4)]);
    let embed = 16usize;
    let query = linspace(&[1, 5, embed], 0.0, 0.03);
    let key = linspace(&[1, 5, embed], 0.5, 0.03);
    let value = linspace(&[1, 5, embed], 1.0, 0.02);

    let ctx = make_ctx(
        &node,
        vec![
            Some(&query),
            Some(&key),
            Some(&value),
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let expected = MultiHeadAttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    MultiHeadAttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "num_heads=4");
}

#[test]
fn test_mha_into_slots_embed_not_divisible_by_heads_error_parity() {
    // embed_dim=9 with num_heads=4 → ShapeMismatch (same error as execute)
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 4)]);
    let query = linspace(&[1, 3, 9], 0.0, 0.1);
    let key = linspace(&[1, 3, 9], 0.1, 0.1);
    let value = linspace(&[1, 3, 9], 0.2, 0.1);

    let ctx = make_ctx(
        &node,
        vec![
            Some(&query),
            Some(&key),
            Some(&value),
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let execute_err = MultiHeadAttentionOp.execute(&ctx);
    assert!(
        execute_err.is_err(),
        "execute must error on bad divisibility"
    );

    let mut slots = vec![Tensor::new(vec![0.0f32; 27], vec![1, 3, 9])];
    let slot_err = MultiHeadAttentionOp.execute_into_slots(&ctx, &mut slots);
    assert!(
        slot_err.is_err(),
        "execute_into_slots must error on bad divisibility"
    );
}

#[test]
fn test_mha_into_slots_with_mask() {
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 2)]);
    let embed = 8usize;
    let seq = 4usize;
    let query = linspace(&[2, seq, embed], 0.0, 0.05);
    let key = linspace(&[2, seq, embed], 0.2, 0.05);
    let value = linspace(&[2, seq, embed], 0.4, 0.05);
    let mask = Tensor::new(vec![0.0f32; 2 * seq * seq], vec![2, seq, seq]);

    let ctx = make_ctx(
        &node,
        vec![
            Some(&query),
            Some(&key),
            Some(&value),
            None,
            None,
            None,
            None,
            Some(&mask),
        ],
    );
    let expected = MultiHeadAttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    MultiHeadAttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "MHA with mask");
}

#[test]
fn test_mha_into_slots_batch_gt1() {
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 2)]);
    let embed = 8usize;
    let query = linspace(&[3, 4, embed], 0.0, 0.04);
    let key = linspace(&[3, 4, embed], 0.3, 0.04);
    let value = linspace(&[3, 4, embed], 0.6, 0.03);

    let ctx = make_ctx(
        &node,
        vec![
            Some(&query),
            Some(&key),
            Some(&value),
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let expected = MultiHeadAttentionOp.execute(&ctx).expect("execute failed");

    let out_len: usize = expected[0].data.len();
    let mut slots = vec![Tensor::new(
        vec![0.0f32; out_len],
        expected[0].shape.clone(),
    )];
    MultiHeadAttentionOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_tensor_eq(&slots[0], &expected[0], "batch=3");
}

#[test]
fn test_mha_into_slots_second_call_different_shape_resizes() {
    let node = node_with_int_attrs(OpKind::MultiHeadAttention, &[("num_heads", 2)]);
    let embed = 8usize;

    // First call: [1, 2, 8]
    let q1 = linspace(&[1, 2, embed], 0.0, 0.1);
    let k1 = linspace(&[1, 2, embed], 0.5, 0.1);
    let v1 = linspace(&[1, 2, embed], 1.0, 0.05);
    let ctx1 = make_ctx(
        &node,
        vec![
            Some(&q1),
            Some(&k1),
            Some(&v1),
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let expected1 = MultiHeadAttentionOp
        .execute(&ctx1)
        .expect("execute 1 failed");

    let mut slots = vec![Tensor::new(vec![0.0f32; 2 * embed], vec![1, 2, embed])];
    MultiHeadAttentionOp
        .execute_into_slots(&ctx1, &mut slots)
        .expect("slot write 1 failed");
    assert_tensor_eq(&slots[0], &expected1[0], "first call [1,2,8]");

    // Second call: [2, 5, 8] — slot must resize
    let q2 = linspace(&[2, 5, embed], 0.1, 0.05);
    let k2 = linspace(&[2, 5, embed], 0.6, 0.05);
    let v2 = linspace(&[2, 5, embed], 1.1, 0.03);
    let ctx2 = make_ctx(
        &node,
        vec![
            Some(&q2),
            Some(&k2),
            Some(&v2),
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let expected2 = MultiHeadAttentionOp
        .execute(&ctx2)
        .expect("execute 2 failed");

    MultiHeadAttentionOp
        .execute_into_slots(&ctx2, &mut slots)
        .expect("slot write 2 failed");
    assert_tensor_eq(&slots[0], &expected2[0], "second call [2,5,8] (resized)");
}
