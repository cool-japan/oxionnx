//! Cached attention: KV-cache backed attention for autoregressive inference.

use super::core::{reshape_from_heads, reshape_to_heads, scaled_dot_product_attention};
use crate::kv_cache::KvCache;
use oxionnx_core::{OnnxError, Tensor};

// ── Cached SDPA ──────────────────────────────────────────────────────────────

/// Scaled dot-product attention with KV cache support for incremental inference.
///
/// Appends the new token's K/V to the cache, then computes attention of the
/// new query against the entire cached sequence.
///
/// # Arguments
/// * `q` — Query for the new token(s) `[batch, num_heads, new_seq, head_dim]`
/// * `k` — Key for the new token(s) `[batch, num_heads, new_seq, head_dim]`
/// * `v` — Value for the new token(s) `[batch, num_heads, new_seq, head_dim]`
/// * `cache` — KV cache containing past keys/values
/// * `layer_idx` — which layer's cache slot to use
/// * `mask` — optional additive mask broadcastable to `[..., new_seq, full_seq]`
/// * `scale` — custom scale factor (default: 1/sqrt(head_dim))
///
/// # Returns
/// Output tensor `[batch, num_heads, new_seq, head_dim]`.
pub fn cached_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &mut KvCache,
    layer_idx: usize,
    mask: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor, OnnxError> {
    if q.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "cached_attention: Q must be 4D [batch, num_heads, seq, head_dim]".into(),
        ));
    }
    if k.ndim() != 4 || v.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "cached_attention: K and V must be 4D".into(),
        ));
    }
    let (full_k, full_v) = cache.update(layer_idx, k, v).map_err(OnnxError::Internal)?;
    scaled_dot_product_attention(q, &full_k, &full_v, mask, scale)
}

// ── Cached Multi-Head Attention ──────────────────────────────────────────────

/// Multi-head attention with KV cache support for incremental inference.
///
/// Handles head projection (split Q/K/V into heads), then delegates to
/// [`cached_attention`] for the actual SDPA computation.
///
/// # Arguments
/// * `query` — `[batch, new_seq, embed_dim]`
/// * `key` — `[batch, new_seq, embed_dim]`
/// * `value` — `[batch, new_seq, embed_dim]`
/// * `cache` — KV cache
/// * `layer_idx` — which layer's cache slot to use
/// * `num_heads` — number of attention heads
/// * `mask` — optional additive mask
/// * `scale` — custom scale factor
///
/// # Returns
/// Output tensor `[batch, new_seq, embed_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn cached_multi_head_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    cache: &mut KvCache,
    layer_idx: usize,
    num_heads: usize,
    mask: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor, OnnxError> {
    if query.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "cached_multi_head_attention: query must be 3D [batch, seq, embed], got {}D",
            query.ndim()
        )));
    }
    if key.ndim() != 3 || value.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(
            "cached_multi_head_attention: key and value must be 3D".into(),
        ));
    }
    let batch = query.shape[0];
    let new_seq = query.shape[1];
    let embed_dim = query.shape[2];
    if embed_dim % num_heads != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "cached_multi_head_attention: embed_dim {} not divisible by num_heads {}",
            embed_dim, num_heads
        )));
    }
    let head_dim = embed_dim / num_heads;
    let new_seq_k = key.shape[1];
    let q_heads = reshape_to_heads(query, batch, new_seq, num_heads, head_dim);
    let k_heads = reshape_to_heads(key, batch, new_seq_k, num_heads, head_dim);
    let v_heads = reshape_to_heads(value, batch, new_seq_k, num_heads, head_dim);
    let attn_out = cached_attention(&q_heads, &k_heads, &v_heads, cache, layer_idx, mask, scale)?;
    Ok(reshape_from_heads(
        &attn_out, batch, new_seq, num_heads, head_dim,
    ))
}
