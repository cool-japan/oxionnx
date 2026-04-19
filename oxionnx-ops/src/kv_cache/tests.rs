//! Tests for the KvCache module.

#![cfg(test)]
#![allow(clippy::identity_op)]

use crate::attention::{cached_attention, grouped_query_attention, scaled_dot_product_attention};
use crate::flash::{cached_flash_attention, flash_attention};
use crate::kv_cache::KvCache;
use oxionnx_core::Tensor;
/// Helper: compare two tensors element-wise within tolerance.
fn assert_tensors_close(a: &Tensor, b: &Tensor, tol: f32, label: &str) {
    assert_eq!(
        a.shape, b.shape,
        "{label}: shape mismatch {:?} vs {:?}",
        a.shape, b.shape
    );
    for (i, (av, bv)) in a.data.iter().zip(b.data.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "{label}: mismatch at idx {i}: {av} vs {bv} (diff={})",
            (av - bv).abs()
        );
    }
}
/// Create a deterministic test tensor from a seed.
fn make_tensor(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32 * seed + 0.37).sin() * 0.5)
        .collect();
    Tensor::new(data, shape.to_vec())
}
#[test]
fn test_empty_cache_creation() {
    let cache = KvCache::new(4);
    assert_eq!(cache.num_layers(), 4);
    for layer in 0..4 {
        assert_eq!(cache.seq_len(layer), 0);
    }
}
#[test]
fn test_single_update() {
    let mut cache = KvCache::new(2);
    let k = make_tensor(&[1, 2, 3, 4], 0.1);
    let v = make_tensor(&[1, 2, 3, 4], 0.2);
    let (full_k, full_v) = cache.update(0, &k, &v).unwrap();
    assert_eq!(full_k.shape, vec![1, 2, 3, 4]);
    assert_eq!(full_v.shape, vec![1, 2, 3, 4]);
    assert_eq!(cache.seq_len(0), 3);
    assert_eq!(cache.seq_len(1), 0);
    assert_tensors_close(&full_k, &k, 1e-7, "first_update_k");
    assert_tensors_close(&full_v, &v, 1e-7, "first_update_v");
}
#[test]
fn test_multiple_updates_grow_sequence() {
    let mut cache = KvCache::new(1);
    let k1 = make_tensor(&[1, 2, 1, 4], 0.3);
    let v1 = make_tensor(&[1, 2, 1, 4], 0.4);
    let (fk1, _fv1) = cache.update(0, &k1, &v1).unwrap();
    assert_eq!(fk1.shape, vec![1, 2, 1, 4]);
    assert_eq!(cache.seq_len(0), 1);
    let k2 = make_tensor(&[1, 2, 1, 4], 0.5);
    let v2 = make_tensor(&[1, 2, 1, 4], 0.6);
    let (fk2, _fv2) = cache.update(0, &k2, &v2).unwrap();
    assert_eq!(fk2.shape, vec![1, 2, 2, 4]);
    assert_eq!(cache.seq_len(0), 2);
    let k3 = make_tensor(&[1, 2, 1, 4], 0.7);
    let v3 = make_tensor(&[1, 2, 1, 4], 0.8);
    let (fk3, _fv3) = cache.update(0, &k3, &v3).unwrap();
    assert_eq!(fk3.shape, vec![1, 2, 3, 4]);
    assert_eq!(cache.seq_len(0), 3);
    for h in 0..2 {
        for d in 0..4 {
            let expected = k1.data[h * 1 * 4 + d];
            let got = fk3.data[h * 3 * 4 + d];
            assert!(
                (expected - got).abs() < 1e-7,
                "data preserved: h={h} d={d} expected={expected} got={got}"
            );
        }
    }
}
#[test]
fn test_ring_buffer_truncation() {
    let mut cache = KvCache::with_max_seq_len(1, 3);
    for step in 0..5 {
        let k = make_tensor(&[1, 1, 1, 2], 0.1 * step as f32 + 0.01);
        let v = make_tensor(&[1, 1, 1, 2], 0.1 * step as f32 + 0.02);
        let (full_k, _full_v) = cache.update(0, &k, &v).unwrap();
        let expected_full_seq = (step + 1).min(3 + 1);
        assert_eq!(
            full_k.shape[2], expected_full_seq,
            "step {step}: expected full_seq={expected_full_seq}, got {}",
            full_k.shape[2]
        );
        let cached = cache.seq_len(0);
        assert!(
            cached <= 3,
            "step {step}: cached seq_len={cached} should be <=3"
        );
    }
    assert_eq!(cache.seq_len(0), 3);
}
#[test]
fn test_clear_resets() {
    let mut cache = KvCache::new(2);
    let k = make_tensor(&[1, 2, 5, 4], 0.1);
    let v = make_tensor(&[1, 2, 5, 4], 0.2);
    cache.update(0, &k, &v).unwrap();
    cache.update(1, &k, &v).unwrap();
    assert_eq!(cache.seq_len(0), 5);
    assert_eq!(cache.seq_len(1), 5);
    cache.clear();
    assert_eq!(cache.seq_len(0), 0);
    assert_eq!(cache.seq_len(1), 0);
}
#[test]
fn test_layer_out_of_range() {
    let mut cache = KvCache::new(2);
    let k = make_tensor(&[1, 1, 1, 4], 0.1);
    let v = make_tensor(&[1, 1, 1, 4], 0.2);
    let result = cache.update(5, &k, &v);
    assert!(result.is_err());
}
#[test]
fn test_cached_attention_matches_full_sdpa() {
    let batch = 1;
    let num_heads = 4;
    let head_dim = 8;
    let seq_len = 4;
    let full_q = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.11);
    let full_k = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.22);
    let full_v = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.33);
    let mut causal_mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                causal_mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let causal_mask = Tensor::new(causal_mask_data, vec![seq_len, seq_len]);
    let full_out =
        scaled_dot_product_attention(&full_q, &full_k, &full_v, Some(&causal_mask), None).unwrap();
    let mut cache = KvCache::new(1);
    for t in 0..seq_len {
        let q_t = extract_token(&full_q, t);
        let k_t = extract_token(&full_k, t);
        let v_t = extract_token(&full_v, t);
        let (cached_k, cached_v) = cache.update(0, &k_t, &v_t).unwrap();
        let past_len = cached_k.shape[2];
        let step_mask_data = vec![0.0f32; 1 * past_len];
        let step_mask = Tensor::new(step_mask_data, vec![1, past_len]);
        let out_t =
            scaled_dot_product_attention(&q_t, &cached_k, &cached_v, Some(&step_mask), None)
                .unwrap();
        let expected_t = extract_token(&full_out, t);
        assert_tensors_close(&out_t, &expected_t, 1e-5, &format!("cached_sdpa_token_{t}"));
    }
}
#[test]
fn test_cached_attention_2_layers() {
    let batch = 1;
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let num_layers = 2;
    let mut cache = KvCache::new(num_layers);
    for layer in 0..num_layers {
        let _full_q = make_tensor(
            &[batch, num_heads, seq_len, head_dim],
            0.1 * (layer + 1) as f32,
        );
        let full_k = make_tensor(
            &[batch, num_heads, seq_len, head_dim],
            0.2 * (layer + 1) as f32,
        );
        let full_v = make_tensor(
            &[batch, num_heads, seq_len, head_dim],
            0.3 * (layer + 1) as f32,
        );
        for t in 0..seq_len {
            let k_t = extract_token(&full_k, t);
            let v_t = extract_token(&full_v, t);
            cache.update(layer, &k_t, &v_t).unwrap();
        }
        assert_eq!(cache.seq_len(layer), seq_len);
    }
    assert_eq!(cache.seq_len(0), seq_len);
    assert_eq!(cache.seq_len(1), seq_len);
}
#[test]
fn test_cached_flash_attention_matches_full() {
    let batch = 1;
    let num_heads = 2;
    let head_dim = 8;
    let seq_len = 5;
    let full_q = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.17);
    let full_k = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.23);
    let full_v = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.31);
    let full_out = flash_attention(&full_q, &full_k, &full_v, None, true).unwrap();
    let mut cache = KvCache::new(1);
    for t in 0..seq_len {
        let q_t = extract_token(&full_q, t);
        let k_t = extract_token(&full_k, t);
        let v_t = extract_token(&full_v, t);
        let out_t =
            crate::flash::cached_flash_attention(&q_t, &k_t, &v_t, &mut cache, 0, true).unwrap();
        let expected_t = extract_token(&full_out, t);
        assert_tensors_close(
            &out_t,
            &expected_t,
            1e-4,
            &format!("cached_flash_token_{t}"),
        );
    }
}
#[test]
fn test_gqa_with_cache() {
    let batch = 1;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let full_q = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.13);
    let full_k = make_tensor(&[batch, num_kv_heads, seq_len, head_dim], 0.19);
    let full_v = make_tensor(&[batch, num_kv_heads, seq_len, head_dim], 0.29);
    let mut causal_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                causal_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let causal_mask = Tensor::new(causal_data, vec![seq_len, seq_len]);
    let full_out = grouped_query_attention(
        &full_q,
        &full_k,
        &full_v,
        num_kv_heads,
        Some(&causal_mask),
        None,
    )
    .unwrap();
    let mut cache = KvCache::new(1);
    for t in 0..seq_len {
        let q_t = extract_token(&full_q, t);
        let k_t = extract_token_heads(&full_k, t, num_kv_heads);
        let v_t = extract_token_heads(&full_v, t, num_kv_heads);
        let (cached_k, cached_v) = cache.update(0, &k_t, &v_t).unwrap();
        let past_len = cached_k.shape[2];
        let step_mask = Tensor::new(vec![0.0f32; 1 * past_len], vec![1, past_len]);
        let out_t = grouped_query_attention(
            &q_t,
            &cached_k,
            &cached_v,
            num_kv_heads,
            Some(&step_mask),
            None,
        )
        .unwrap();
        let expected_t = extract_token(&full_out, t);
        assert_tensors_close(&out_t, &expected_t, 1e-5, &format!("gqa_cached_token_{t}"));
    }
}
#[test]
fn test_mqa_with_cache() {
    let batch = 1;
    let num_heads = 4;
    let head_dim = 4;
    let seq_len = 3;
    let full_q = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.07);
    let full_k = make_tensor(&[batch, 1, seq_len, head_dim], 0.11);
    let full_v = make_tensor(&[batch, 1, seq_len, head_dim], 0.17);
    let mut causal_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                causal_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let causal_mask = Tensor::new(causal_data, vec![seq_len, seq_len]);
    let full_out =
        grouped_query_attention(&full_q, &full_k, &full_v, 1, Some(&causal_mask), None).unwrap();
    let mut cache = KvCache::new(1);
    for t in 0..seq_len {
        let q_t = extract_token(&full_q, t);
        let k_t = extract_token_heads(&full_k, t, 1);
        let v_t = extract_token_heads(&full_v, t, 1);
        let (cached_k, cached_v) = cache.update(0, &k_t, &v_t).unwrap();
        let past_len = cached_k.shape[2];
        let step_mask = Tensor::new(vec![0.0f32; past_len], vec![1, past_len]);
        let out_t =
            grouped_query_attention(&q_t, &cached_k, &cached_v, 1, Some(&step_mask), None).unwrap();
        let expected_t = extract_token(&full_out, t);
        assert_tensors_close(&out_t, &expected_t, 1e-5, &format!("mqa_cached_token_{t}"));
    }
}
#[test]
fn test_cache_batch_gt_1() {
    let batch = 2;
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let full_q = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.09);
    let full_k = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.14);
    let full_v = make_tensor(&[batch, num_heads, seq_len, head_dim], 0.21);
    let mut causal_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                causal_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let causal_mask = Tensor::new(causal_data, vec![seq_len, seq_len]);
    let full_out =
        scaled_dot_product_attention(&full_q, &full_k, &full_v, Some(&causal_mask), None).unwrap();
    let mut cache = KvCache::new(1);
    for t in 0..seq_len {
        let q_t = extract_token_batched(&full_q, t, batch, num_heads, head_dim);
        let k_t = extract_token_batched(&full_k, t, batch, num_heads, head_dim);
        let v_t = extract_token_batched(&full_v, t, batch, num_heads, head_dim);
        let (cached_k, cached_v) = cache.update(0, &k_t, &v_t).unwrap();
        let past_len = cached_k.shape[2];
        let step_mask = Tensor::new(vec![0.0f32; past_len], vec![1, past_len]);
        let out_t =
            scaled_dot_product_attention(&q_t, &cached_k, &cached_v, Some(&step_mask), None)
                .unwrap();
        let expected_t = extract_token_batched(&full_out, t, batch, num_heads, head_dim);
        assert_tensors_close(&out_t, &expected_t, 1e-5, &format!("batch2_token_{t}"));
    }
}
#[test]
fn test_edge_single_head_dim1() {
    let mut cache = KvCache::new(1);
    let k1 = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
    let v1 = Tensor::new(vec![2.0], vec![1, 1, 1, 1]);
    let (fk1, fv1) = cache.update(0, &k1, &v1).unwrap();
    assert_eq!(fk1.shape, vec![1, 1, 1, 1]);
    assert!((fk1.data[0] - 1.0).abs() < 1e-7);
    assert!((fv1.data[0] - 2.0).abs() < 1e-7);
    let k2 = Tensor::new(vec![3.0], vec![1, 1, 1, 1]);
    let v2 = Tensor::new(vec![4.0], vec![1, 1, 1, 1]);
    let (fk2, fv2) = cache.update(0, &k2, &v2).unwrap();
    assert_eq!(fk2.shape, vec![1, 1, 2, 1]);
    assert!((fk2.data[0] - 1.0).abs() < 1e-7);
    assert!((fk2.data[1] - 3.0).abs() < 1e-7);
    assert!((fv2.data[0] - 2.0).abs() < 1e-7);
    assert!((fv2.data[1] - 4.0).abs() < 1e-7);
}
#[test]
fn test_ring_buffer_data_correctness() {
    let mut cache = KvCache::with_max_seq_len(1, 2);
    let k0 = Tensor::new(vec![10.0], vec![1, 1, 1, 1]);
    let v0 = Tensor::new(vec![20.0], vec![1, 1, 1, 1]);
    cache.update(0, &k0, &v0).unwrap();
    let k1 = Tensor::new(vec![30.0], vec![1, 1, 1, 1]);
    let v1 = Tensor::new(vec![40.0], vec![1, 1, 1, 1]);
    cache.update(0, &k1, &v1).unwrap();
    assert_eq!(cache.seq_len(0), 2);
    let k2 = Tensor::new(vec![50.0], vec![1, 1, 1, 1]);
    let v2 = Tensor::new(vec![60.0], vec![1, 1, 1, 1]);
    let (fk, _fv) = cache.update(0, &k2, &v2).unwrap();
    assert_eq!(fk.shape[2], 3);
    assert_eq!(cache.seq_len(0), 2);
}
#[test]
fn test_with_max_seq_len_creation() {
    let cache = KvCache::with_max_seq_len(3, 128);
    assert_eq!(cache.num_layers(), 3);
    for layer in 0..3 {
        assert_eq!(cache.seq_len(layer), 0);
    }
}
#[test]
fn test_kv_cache_shape_mismatch_error() {
    let mut cache = KvCache::new(1);
    let k1 = make_tensor(&[1, 2, 1, 4], 0.1);
    let v1 = make_tensor(&[1, 2, 1, 4], 0.2);
    cache.update(0, &k1, &v1).unwrap();
    let k2 = make_tensor(&[1, 2, 1, 8], 0.3);
    let v2 = make_tensor(&[1, 2, 1, 8], 0.4);
    let result = cache.update(0, &k2, &v2);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("head_dim"),
        "error should mention head_dim: {err_msg}"
    );
}
#[test]
fn test_kv_cache_head_count_mismatch() {
    let mut cache = KvCache::new(1);
    let k1 = make_tensor(&[1, 2, 1, 4], 0.1);
    let v1 = make_tensor(&[1, 2, 1, 4], 0.2);
    cache.update(0, &k1, &v1).unwrap();
    let k2 = make_tensor(&[1, 3, 1, 4], 0.3);
    let v2 = make_tensor(&[1, 3, 1, 4], 0.4);
    let result = cache.update(0, &k2, &v2);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("num_heads"),
        "error should mention num_heads: {err_msg}"
    );
}
#[test]
fn test_kv_cache_empty_input_error() {
    let mut cache = KvCache::new(1);
    let k = Tensor::new(vec![], vec![1, 2, 0, 4]);
    let v = Tensor::new(vec![], vec![1, 2, 0, 4]);
    let result = cache.update(0, &k, &v);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("empty"),
        "error should mention empty sequence: {err_msg}"
    );
}
#[test]
fn test_kv_cache_rank_mismatch_error() {
    let mut cache = KvCache::new(1);
    let k = Tensor::new(vec![1.0; 16], vec![4, 4]);
    let v = Tensor::new(vec![1.0; 16], vec![4, 4]);
    let result = cache.update(0, &k, &v);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("4D"),
        "error should mention 4D requirement: {err_msg}"
    );
}
#[test]
fn test_cached_attention_shape_error() {
    let mut cache = KvCache::new(1);
    let q = Tensor::new(vec![1.0; 8], vec![1, 2, 4]);
    let k = make_tensor(&[1, 2, 1, 4], 0.2);
    let v = make_tensor(&[1, 2, 1, 4], 0.3);
    let result = cached_attention(&q, &k, &v, &mut cache, 0, None, None);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("4D"),
        "error should mention 4D requirement: {err_msg}"
    );
}
#[test]
fn test_cached_flash_attention_error() {
    let mut cache = KvCache::new(1);
    let q = Tensor::new(vec![1.0; 8], vec![2, 4]);
    let k = make_tensor(&[1, 2, 1, 4], 0.1);
    let v = make_tensor(&[1, 2, 1, 4], 0.2);
    let result = cached_flash_attention(&q, &k, &v, &mut cache, 0, false);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("4D"),
        "error should mention 4D requirement: {err_msg}"
    );
}
/// Extract token at position `t` from a 4D tensor [batch=1, heads, seq, dim].
fn extract_token(tensor: &Tensor, t: usize) -> Tensor {
    let batch = tensor.shape[0];
    let heads = tensor.shape[1];
    let head_dim = tensor.shape[3];
    extract_token_batched(tensor, t, batch, heads, head_dim)
}
/// Extract token at position `t` from a 4D tensor with given num_heads.
fn extract_token_heads(tensor: &Tensor, t: usize, num_heads: usize) -> Tensor {
    let batch = tensor.shape[0];
    let head_dim = tensor.shape[3];
    extract_token_batched(tensor, t, batch, num_heads, head_dim)
}
/// Extract token at position `t` from a 4D tensor [batch, heads, seq, dim].
fn extract_token_batched(
    tensor: &Tensor,
    t: usize,
    batch: usize,
    heads: usize,
    head_dim: usize,
) -> Tensor {
    let seq = tensor.shape[2];
    let mut data = vec![0.0f32; batch * heads * 1 * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            let src_off = (b * heads + h) * seq * head_dim + t * head_dim;
            let dst_off = (b * heads + h) * head_dim;
            data[dst_off..dst_off + head_dim]
                .copy_from_slice(&tensor.data[src_off..src_off + head_dim]);
        }
    }
    Tensor::new(data, vec![batch, heads, 1, head_dim])
}
