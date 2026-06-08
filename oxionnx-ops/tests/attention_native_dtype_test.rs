//! Tests for AttentionOp and MultiHeadAttentionOp native typed dispatch (Block W2.1).
//!
//! Covers native_dtypes() reporting, F32 parity, F16 parity (tol=1e-2), and BF16 parity (tol=5e-2).

use oxionnx_core::{
    dtype::{DType, TensorStorage, TypedTensor},
    graph::{Attributes, Node, OpKind},
    operator::{Operator, TypedOpContext},
};
use oxionnx_ops::registry::rnn_ops::{AttentionOp, MultiHeadAttentionOp};

// ── Conversion helpers ────────────────────────────────────────────────────────

fn f32_to_f16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

fn f32_to_bf16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

fn f16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::bf16::from_bits(b).to_f32())
        .collect()
}

// ── Node builders ─────────────────────────────────────────────────────────────

fn attention_node(scale: Option<f32>) -> Node {
    let mut attrs = Attributes::default();
    if let Some(s) = scale {
        attrs.floats.insert("scale".into(), s);
    }
    Node {
        name: "test_attention".into(),
        op: OpKind::Attention,
        inputs: vec![],
        outputs: vec![],
        attrs,
    }
}

fn mha_node(num_heads: usize) -> Node {
    let mut attrs = Attributes::default();
    attrs.ints.insert("num_heads".into(), num_heads as i64);
    Node {
        name: "test_mha".into(),
        op: OpKind::MultiHeadAttention,
        inputs: vec![],
        outputs: vec![],
        attrs,
    }
}

// ── TypedOpContext builders ───────────────────────────────────────────────────

fn make_ctx_3<'a>(
    node: &'a Node,
    q: &'a TypedTensor,
    k: &'a TypedTensor,
    v: &'a TypedTensor,
) -> TypedOpContext<'a> {
    TypedOpContext {
        node,
        inputs: vec![Some(q), Some(k), Some(v)],
        outer_scope: None,
        registry: None,
    }
}

fn make_ctx_6<'a>(
    node: &'a Node,
    q: &'a TypedTensor,
    k: &'a TypedTensor,
    v: &'a TypedTensor,
    out_w: &'a TypedTensor,
) -> TypedOpContext<'a> {
    // Slots: 0=query 1=key 2=value 3=qkv_weight(None) 4=qkv_bias(None) 5=out_proj_weight
    TypedOpContext {
        node,
        inputs: vec![Some(q), Some(k), Some(v), None, None, Some(out_w)],
        outer_scope: None,
        registry: None,
    }
}

// ── Assert helpers ────────────────────────────────────────────────────────────

fn assert_slices_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() <= tol,
            "{label} at [{i}]: got {av}, ref {bv}, diff={:.6} > tol={tol}",
            (av - bv).abs()
        );
    }
}

// ── Deterministic test data ───────────────────────────────────────────────────

/// Sequential floats starting from `start`, step `step`, count `n`.
fn seq_f32(start: f32, step: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| start + i as f32 * step).collect()
}

// ── Test: native_dtypes includes F32/F16/BF16 ────────────────────────────────

#[test]
fn test_attention_native_dtypes_includes_all_three() {
    let op = AttentionOp;
    let dtypes = op.native_dtypes();
    assert!(dtypes.contains(&DType::F32), "AttentionOp must support F32");
    assert!(dtypes.contains(&DType::F16), "AttentionOp must support F16");
    assert!(
        dtypes.contains(&DType::BF16),
        "AttentionOp must support BF16"
    );
}

#[test]
fn test_mha_native_dtypes_includes_all_three() {
    let op = MultiHeadAttentionOp;
    let dtypes = op.native_dtypes();
    assert!(
        dtypes.contains(&DType::F32),
        "MultiHeadAttentionOp must support F32"
    );
    assert!(
        dtypes.contains(&DType::F16),
        "MultiHeadAttentionOp must support F16"
    );
    assert!(
        dtypes.contains(&DType::BF16),
        "MultiHeadAttentionOp must support BF16"
    );
}

// ── Test: F32 execute_typed parity vs execute ────────────────────────────────

#[test]
fn test_sdpa_f32_baseline() {
    // [batch=1, heads=1 flat, seq_q=4, head_dim=8] → Q shape [4, 8], K [4, 8], V [4, 8]
    let n = 4 * 8;
    let q_data = seq_f32(0.01, 0.01, n);
    let k_data = seq_f32(0.02, 0.01, n);
    let v_data = seq_f32(0.05, 0.02, n);

    let node = attention_node(None);
    let q_tt = TypedTensor::new(TensorStorage::F32(q_data.clone()), vec![4, 8]);
    let k_tt = TypedTensor::new(TensorStorage::F32(k_data.clone()), vec![4, 8]);
    let v_tt = TypedTensor::new(TensorStorage::F32(v_data.clone()), vec![4, 8]);

    let ctx = make_ctx_3(&node, &q_tt, &k_tt, &v_tt);
    let out = AttentionOp.execute_typed(&ctx).expect("F32 SDPA typed");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape, vec![4, 8]);

    // Run via f32 execute for reference.
    use oxionnx_core::OpContext;
    let q_ref = oxionnx_core::Tensor::new(q_data, vec![4, 8]);
    let k_ref = oxionnx_core::Tensor::new(k_data, vec![4, 8]);
    let v_ref = oxionnx_core::Tensor::new(v_data, vec![4, 8]);
    let ref_ctx = OpContext {
        node: &node,
        inputs: vec![Some(&q_ref), Some(&k_ref), Some(&v_ref), None],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let ref_out = AttentionOp.execute(&ref_ctx).expect("F32 SDPA execute");
    let typed_f32 = match &out[0].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("expected F32 storage"),
    };
    assert_slices_close(&typed_f32, &ref_out[0].data, 1e-5, "F32 parity");
}

// ── Test: F16 SDPA parity ────────────────────────────────────────────────────

#[test]
fn test_sdpa_f16_parity() {
    // [batch=1, heads=2 flat → batch_total=2, seq_q=4, head_dim=8]
    // We pass Q shape [2, 4, 8] (batch_total=2, seq_q=4, head_dim=8).
    let batch_total = 2;
    let seq = 4;
    let hd = 8;
    let n = batch_total * seq * hd;

    let q_f32 = seq_f32(0.01, 0.01, n);
    let k_f32 = seq_f32(0.03, 0.01, n);
    let v_f32 = seq_f32(0.05, 0.02, n);

    // F16 typed path
    let node = attention_node(None);
    let q_tt = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&q_f32)),
        vec![batch_total, seq, hd],
    );
    let k_tt = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&k_f32)),
        vec![batch_total, seq, hd],
    );
    let v_tt = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&v_f32)),
        vec![batch_total, seq, hd],
    );

    let ctx = make_ctx_3(&node, &q_tt, &k_tt, &v_tt);
    let out = AttentionOp.execute_typed(&ctx).expect("F16 SDPA typed");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape, vec![batch_total, seq, hd]);

    let out_f32 = match &out[0].storage {
        TensorStorage::F16(b) => f16_bits_to_f32(b),
        _ => panic!("expected F16 storage"),
    };

    // F32 reference
    let q_ref = oxionnx_core::Tensor::new(q_f32, vec![batch_total, seq, hd]);
    let k_ref = oxionnx_core::Tensor::new(k_f32, vec![batch_total, seq, hd]);
    let v_ref = oxionnx_core::Tensor::new(v_f32, vec![batch_total, seq, hd]);
    use oxionnx_core::OpContext;
    let ref_ctx = OpContext {
        node: &node,
        inputs: vec![Some(&q_ref), Some(&k_ref), Some(&v_ref), None],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let ref_out = AttentionOp.execute(&ref_ctx).expect("F32 ref");
    assert_slices_close(&out_f32, &ref_out[0].data, 1e-2, "F16 SDPA parity");
}

// ── Test: BF16 SDPA parity ───────────────────────────────────────────────────

#[test]
fn test_sdpa_bf16_parity() {
    let batch_total = 2;
    let seq = 4;
    let hd = 8;
    let n = batch_total * seq * hd;

    let q_f32 = seq_f32(0.01, 0.01, n);
    let k_f32 = seq_f32(0.03, 0.01, n);
    let v_f32 = seq_f32(0.05, 0.02, n);

    let node = attention_node(None);
    let q_tt = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&q_f32)),
        vec![batch_total, seq, hd],
    );
    let k_tt = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&k_f32)),
        vec![batch_total, seq, hd],
    );
    let v_tt = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&v_f32)),
        vec![batch_total, seq, hd],
    );

    let ctx = make_ctx_3(&node, &q_tt, &k_tt, &v_tt);
    let out = AttentionOp.execute_typed(&ctx).expect("BF16 SDPA typed");
    assert_eq!(out[0].shape, vec![batch_total, seq, hd]);

    let out_f32 = match &out[0].storage {
        TensorStorage::BF16(b) => bf16_bits_to_f32(b),
        _ => panic!("expected BF16 storage"),
    };

    let q_ref = oxionnx_core::Tensor::new(q_f32, vec![batch_total, seq, hd]);
    let k_ref = oxionnx_core::Tensor::new(k_f32, vec![batch_total, seq, hd]);
    let v_ref = oxionnx_core::Tensor::new(v_f32, vec![batch_total, seq, hd]);
    use oxionnx_core::OpContext;
    let ref_ctx = OpContext {
        node: &node,
        inputs: vec![Some(&q_ref), Some(&k_ref), Some(&v_ref), None],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let ref_out = AttentionOp.execute(&ref_ctx).expect("F32 ref");
    assert_slices_close(&out_f32, &ref_out[0].data, 5e-2, "BF16 SDPA parity");
}

// ── Test: F16 MHA parity ─────────────────────────────────────────────────────

#[test]
fn test_mha_f16_parity() {
    // 2-head MHA, [1, 4, 16] embed_dim=16, head_dim=8
    let batch = 1;
    let seq = 4;
    let embed_dim = 16;
    let num_heads = 2;
    let n = batch * seq * embed_dim;
    let w_n = embed_dim * embed_dim;

    let q_f32 = seq_f32(0.01, 0.01, n);
    let k_f32 = seq_f32(0.02, 0.01, n);
    let v_f32 = seq_f32(0.03, 0.01, n);
    // Simple identity-like out_proj weight (permuted to avoid near-zero outputs)
    let mut w_f32 = vec![0.0f32; w_n];
    for i in 0..embed_dim {
        w_f32[i * embed_dim + i] = 1.0;
    }

    let node = mha_node(num_heads);

    let q_tt = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&q_f32)),
        vec![batch, seq, embed_dim],
    );
    let k_tt = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&k_f32)),
        vec![batch, seq, embed_dim],
    );
    let v_tt = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&v_f32)),
        vec![batch, seq, embed_dim],
    );
    let w_tt = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&w_f32)),
        vec![embed_dim, embed_dim],
    );

    let ctx = make_ctx_6(&node, &q_tt, &k_tt, &v_tt, &w_tt);
    let out = MultiHeadAttentionOp
        .execute_typed(&ctx)
        .expect("F16 MHA typed");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape, vec![batch, seq, embed_dim]);

    let out_f32 = match &out[0].storage {
        TensorStorage::F16(b) => f16_bits_to_f32(b),
        _ => panic!("expected F16 storage"),
    };

    // F32 reference: no qkv_weight but identity out_proj.
    let q_ref = oxionnx_core::Tensor::new(q_f32, vec![batch, seq, embed_dim]);
    let k_ref = oxionnx_core::Tensor::new(k_f32, vec![batch, seq, embed_dim]);
    let v_ref = oxionnx_core::Tensor::new(v_f32, vec![batch, seq, embed_dim]);
    let w_ref = oxionnx_core::Tensor::new(w_f32, vec![embed_dim, embed_dim]);
    use oxionnx_core::OpContext;
    let ref_ctx = OpContext {
        node: &node,
        inputs: vec![
            Some(&q_ref),
            Some(&k_ref),
            Some(&v_ref),
            None,
            None,
            Some(&w_ref),
            None,
            None,
        ],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let ref_out = MultiHeadAttentionOp.execute(&ref_ctx).expect("F32 MHA ref");
    assert_slices_close(&out_f32, &ref_out[0].data, 1e-2, "F16 MHA parity");
}

// ── Test: BF16 MHA parity ────────────────────────────────────────────────────

#[test]
fn test_mha_bf16_parity() {
    let batch = 1;
    let seq = 4;
    let embed_dim = 16;
    let num_heads = 2;
    let n = batch * seq * embed_dim;
    let w_n = embed_dim * embed_dim;

    let q_f32 = seq_f32(0.01, 0.01, n);
    let k_f32 = seq_f32(0.02, 0.01, n);
    let v_f32 = seq_f32(0.03, 0.01, n);
    let mut w_f32 = vec![0.0f32; w_n];
    for i in 0..embed_dim {
        w_f32[i * embed_dim + i] = 1.0;
    }

    let node = mha_node(num_heads);

    let q_tt = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&q_f32)),
        vec![batch, seq, embed_dim],
    );
    let k_tt = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&k_f32)),
        vec![batch, seq, embed_dim],
    );
    let v_tt = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&v_f32)),
        vec![batch, seq, embed_dim],
    );
    let w_tt = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&w_f32)),
        vec![embed_dim, embed_dim],
    );

    let ctx = make_ctx_6(&node, &q_tt, &k_tt, &v_tt, &w_tt);
    let out = MultiHeadAttentionOp
        .execute_typed(&ctx)
        .expect("BF16 MHA typed");
    assert_eq!(out[0].shape, vec![batch, seq, embed_dim]);

    let out_f32 = match &out[0].storage {
        TensorStorage::BF16(b) => bf16_bits_to_f32(b),
        _ => panic!("expected BF16 storage"),
    };

    let q_ref = oxionnx_core::Tensor::new(q_f32, vec![batch, seq, embed_dim]);
    let k_ref = oxionnx_core::Tensor::new(k_f32, vec![batch, seq, embed_dim]);
    let v_ref = oxionnx_core::Tensor::new(v_f32, vec![batch, seq, embed_dim]);
    let w_ref = oxionnx_core::Tensor::new(w_f32, vec![embed_dim, embed_dim]);
    use oxionnx_core::OpContext;
    let ref_ctx = OpContext {
        node: &node,
        inputs: vec![
            Some(&q_ref),
            Some(&k_ref),
            Some(&v_ref),
            None,
            None,
            Some(&w_ref),
            None,
            None,
        ],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let ref_out = MultiHeadAttentionOp.execute(&ref_ctx).expect("F32 MHA ref");
    assert_slices_close(&out_f32, &ref_out[0].data, 5e-2, "BF16 MHA parity");
}
