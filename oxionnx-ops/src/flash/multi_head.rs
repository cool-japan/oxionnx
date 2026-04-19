//! Multi-head Flash Attention wrapper — reshapes [batch, seq, embed] tensors
//! into per-head form, calls the tiled kernel, and reshapes back.

use oxionnx_core::{OnnxError, Tensor};

use crate::attention::{reshape_from_heads, reshape_to_heads};

use super::kernel::flash_attention;

// ── Public API ────────────────────────────────────────────────────────────────

/// Multi-head Flash Attention with automatic head splitting / merging.
///
/// Takes Q, K, V in `[batch, seq, embed_dim]` format, splits into
/// `num_heads` heads, runs Flash Attention v2, and concatenates heads back.
///
/// # Arguments
/// * `q` - Query `[batch, seq_q, embed_dim]`
/// * `k` - Key   `[batch, seq_k, embed_dim]`
/// * `v` - Value `[batch, seq_k, embed_dim]`
/// * `mask` - Optional additive mask broadcastable to `[batch, heads, seq_q, seq_k]`
/// * `causal` - Whether to apply causal masking
/// * `num_heads` - Number of attention heads (`embed_dim` must be divisible)
///
/// # Returns
/// Output tensor `[batch, seq_q, embed_dim]`.
pub fn multi_head_flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    num_heads: usize,
) -> Result<Tensor, OnnxError> {
    if q.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "multi_head_flash_attention: Q must be 3D [batch, seq, embed], got {}D",
            q.ndim()
        )));
    }
    if k.ndim() != 3 || v.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(
            "multi_head_flash_attention: K and V must be 3D".to_string(),
        ));
    }

    let batch = q.shape[0];
    let seq_q = q.shape[1];
    let embed_dim = q.shape[2];
    let seq_k = k.shape[1];

    if embed_dim % num_heads != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "multi_head_flash_attention: embed_dim {} not divisible by num_heads {}",
            embed_dim, num_heads
        )));
    }
    let head_dim = embed_dim / num_heads;

    if k.shape[0] != batch || k.shape[2] != embed_dim {
        return Err(OnnxError::ShapeMismatch(format!(
            "multi_head_flash_attention: K shape {:?} incompatible with Q shape {:?}",
            k.shape, q.shape
        )));
    }
    if v.shape[0] != batch || v.shape[1] != seq_k || v.shape[2] != embed_dim {
        return Err(OnnxError::ShapeMismatch(format!(
            "multi_head_flash_attention: V shape {:?} incompatible",
            v.shape
        )));
    }

    // Reshape [batch, seq, embed] → [batch, num_heads, seq, head_dim]
    let q_heads = reshape_to_heads(q, batch, seq_q, num_heads, head_dim);
    let k_heads = reshape_to_heads(k, batch, seq_k, num_heads, head_dim);
    let v_heads = reshape_to_heads(v, batch, seq_k, num_heads, head_dim);

    let attn_out = flash_attention(&q_heads, &k_heads, &v_heads, mask, causal)?;

    // Reshape [batch, num_heads, seq_q, head_dim] → [batch, seq_q, embed_dim]
    Ok(reshape_from_heads(
        &attn_out, batch, seq_q, num_heads, head_dim,
    ))
}
