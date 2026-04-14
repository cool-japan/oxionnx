//! Flash Attention v2 (Pure Rust) — tiled attention with online softmax.
//!
//! Processes Q, K, V in blocks to achieve O(Br × Bc) extra memory
//! instead of O(N²) for the full attention matrix.

use oxionnx_core::{OnnxError, Tensor};

use crate::attention::{reshape_from_heads, reshape_to_heads, scaled_dot_product_attention};

// ── Constants ───────────────────────────────────────────────────────────────

/// Default block size for Flash Attention query rows.
const FLASH_DEFAULT_BLOCK_R: usize = 64;
/// Default block size for Flash Attention key/value columns.
const FLASH_DEFAULT_BLOCK_C: usize = 64;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Retrieve an additive mask value with broadcasting support.
///
/// Supports 2D `[seq_q, seq_k]`, 3D `[batch, seq_q, seq_k]`, and
/// 4D `[batch, heads, seq_q, seq_k]` masks (dims of size 1 are broadcast).
fn flash_mask_value(mask: &Tensor, b: usize, h: usize, i: usize, j: usize) -> f32 {
    match mask.ndim() {
        2 => mask.data[i * mask.shape[1] + j],
        3 => {
            let mb = if mask.shape[0] == 1 { 0 } else { b };
            let (s1, s2) = (mask.shape[1], mask.shape[2]);
            mask.data[mb * s1 * s2 + i * s2 + j]
        }
        4 => {
            let mb = if mask.shape[0] == 1 { 0 } else { b };
            let mh = if mask.shape[1] == 1 { 0 } else { h };
            let (s1, s2, s3) = (mask.shape[1], mask.shape[2], mask.shape[3]);
            mask.data[mb * s1 * s2 * s3 + mh * s2 * s3 + i * s3 + j]
        }
        _ => 0.0,
    }
}

// ── Flash Attention v2 ─────────────────────────────────────────────────────

/// Flash Attention v2 with configurable block sizes (Pure Rust).
///
/// Implements the tiled Flash Attention v2 algorithm with online softmax.
/// Requires only O(Br × Bc) extra memory per block instead of O(N²) for
/// the full attention matrix.
///
/// # Arguments
/// * `q` - Query `[batch, num_heads, seq_q, head_dim]`
/// * `k` - Key `[batch, num_heads, seq_k, head_dim]`
/// * `v` - Value `[batch, num_heads, seq_k, d_v]`
/// * `mask` - Optional additive mask broadcastable to `[batch, heads, seq_q, seq_k]`
/// * `causal` - Whether to apply causal (lower-triangular) masking
/// * `block_r` - Block size for query rows (Br)
/// * `block_c` - Block size for key/value columns (Bc)
///
/// # Returns
/// Output tensor `[batch, num_heads, seq_q, d_v]`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_with_block_size(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    block_r: usize,
    block_c: usize,
) -> Result<Tensor, OnnxError> {
    // ── Validate inputs ─────────────────────────────────────────────────
    if q.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "flash_attention: Q must be 4D [batch, heads, seq, dim], got {}D",
            q.ndim()
        )));
    }
    if k.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "flash_attention: K must be 4D, got {}D",
            k.ndim()
        )));
    }
    if v.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "flash_attention: V must be 4D, got {}D",
            v.ndim()
        )));
    }
    if block_r == 0 || block_c == 0 {
        return Err(OnnxError::ShapeMismatch(
            "flash_attention: block sizes must be > 0".to_string(),
        ));
    }

    let batch = q.shape[0];
    let num_heads = q.shape[1];
    let seq_q = q.shape[2];
    let head_dim = q.shape[3];
    let seq_k = k.shape[2];
    let d_v = v.shape[3];

    if k.shape[0] != batch || k.shape[1] != num_heads || k.shape[3] != head_dim {
        return Err(OnnxError::ShapeMismatch(format!(
            "flash_attention: K shape {:?} incompatible with Q shape {:?}",
            k.shape, q.shape
        )));
    }
    if v.shape[0] != batch || v.shape[1] != num_heads || v.shape[2] != seq_k {
        return Err(OnnxError::ShapeMismatch(format!(
            "flash_attention: V seq_k ({}) differs from K seq_k ({})",
            v.shape[2], seq_k
        )));
    }
    if head_dim == 0 {
        return Ok(Tensor::new(vec![], vec![batch, num_heads, seq_q, d_v]));
    }

    // ── Short-sequence fallback to standard SDPA ────────────────────────
    if seq_q <= block_r && seq_k <= block_c {
        let combined_mask = if causal || mask.is_some() {
            let mut data = vec![0.0f32; batch * num_heads * seq_q * seq_k];
            for b in 0..batch {
                for h in 0..num_heads {
                    for i in 0..seq_q {
                        for j in 0..seq_k {
                            let idx = ((b * num_heads + h) * seq_q + i) * seq_k + j;
                            if causal && j > i {
                                data[idx] = f32::NEG_INFINITY;
                            }
                            if let Some(m) = mask {
                                data[idx] += flash_mask_value(m, b, h, i, j);
                            }
                        }
                    }
                }
            }
            Some(Tensor::new(data, vec![batch, num_heads, seq_q, seq_k]))
        } else {
            None
        };
        return scaled_dot_product_attention(q, k, v, combined_mask.as_ref(), None);
    }

    // ── Flash Attention v2 blocked algorithm ─────────────────────────────
    let scale = 1.0 / (head_dim as f32).sqrt();
    let bh_stride_q = seq_q * head_dim;
    let bh_stride_k = seq_k * head_dim;
    let bh_stride_v = seq_k * d_v;
    let bh_stride_o = seq_q * d_v;

    let mut output = vec![0.0f32; batch * num_heads * seq_q * d_v];

    let num_blocks_r = seq_q.div_ceil(block_r);
    let num_blocks_c = seq_k.div_ceil(block_c);

    for b in 0..batch {
        for h in 0..num_heads {
            let bh = b * num_heads + h;
            let q_base = bh * bh_stride_q;
            let k_base = bh * bh_stride_k;
            let v_base = bh * bh_stride_v;
            let o_base = bh * bh_stride_o;

            for br_idx in 0..num_blocks_r {
                let i_start = br_idx * block_r;
                let i_end = (i_start + block_r).min(seq_q);
                let br_size = i_end - i_start;

                // Running statistics per row of the Q block
                let mut m_i = vec![f32::NEG_INFINITY; br_size];
                let mut l_i = vec![0.0f32; br_size];
                let mut o_i = vec![0.0f32; br_size * d_v];

                for bc_idx in 0..num_blocks_c {
                    let j_start = bc_idx * block_c;
                    let j_end = (j_start + block_c).min(seq_k);
                    let bc_size = j_end - j_start;

                    // Causal early-exit: skip if entire K block is after Q block
                    if causal && j_start > i_end - 1 {
                        continue;
                    }

                    // ── S_ij = Q_i @ K_j^T × scale  [br_size × bc_size] ─────
                    let mut s_ij = vec![0.0f32; br_size * bc_size];
                    for ri in 0..br_size {
                        let q_row = q_base + (i_start + ri) * head_dim;
                        for ci in 0..bc_size {
                            let k_row = k_base + (j_start + ci) * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q.data[q_row + d] * k.data[k_row + d];
                            }
                            s_ij[ri * bc_size + ci] = dot * scale;
                        }
                    }

                    // ── Apply causal mask ────────────────────────────────────
                    if causal {
                        for ri in 0..br_size {
                            let gi = i_start + ri;
                            for ci in 0..bc_size {
                                if j_start + ci > gi {
                                    s_ij[ri * bc_size + ci] = f32::NEG_INFINITY;
                                }
                            }
                        }
                    }

                    // ── Apply additive mask ──────────────────────────────────
                    if let Some(m) = mask {
                        for ri in 0..br_size {
                            for ci in 0..bc_size {
                                s_ij[ri * bc_size + ci] +=
                                    flash_mask_value(m, b, h, i_start + ri, j_start + ci);
                            }
                        }
                    }

                    // ── Online softmax: block statistics ─────────────────────
                    let mut m_ij = vec![f32::NEG_INFINITY; br_size];
                    for ri in 0..br_size {
                        for ci in 0..bc_size {
                            let val = s_ij[ri * bc_size + ci];
                            if val > m_ij[ri] {
                                m_ij[ri] = val;
                            }
                        }
                    }

                    // P_ij = exp(S_ij − m_ij),  l_ij = rowsum(P_ij)
                    let mut p_ij = vec![0.0f32; br_size * bc_size];
                    let mut l_ij = vec![0.0f32; br_size];
                    for ri in 0..br_size {
                        for ci in 0..bc_size {
                            let val = (s_ij[ri * bc_size + ci] - m_ij[ri]).exp();
                            // Guard: exp(-inf − (-inf)) = exp(NaN) → 0
                            let val = if val.is_nan() { 0.0 } else { val };
                            p_ij[ri * bc_size + ci] = val;
                            l_ij[ri] += val;
                        }
                    }

                    // ── Rescale accumulator and update ───────────────────────
                    for ri in 0..br_size {
                        let m_new = m_i[ri].max(m_ij[ri]);

                        // Both −∞ ⇒ no valid scores yet; skip
                        if m_new == f32::NEG_INFINITY {
                            continue;
                        }

                        let alpha = {
                            let v = (m_i[ri] - m_new).exp();
                            if v.is_nan() {
                                0.0
                            } else {
                                v
                            }
                        };
                        let beta = {
                            let v = (m_ij[ri] - m_new).exp();
                            if v.is_nan() {
                                0.0
                            } else {
                                v
                            }
                        };

                        let l_new = alpha * l_i[ri] + beta * l_ij[ri];

                        if l_new > 0.0 {
                            let inv_l = 1.0 / l_new;
                            let w_old = alpha * l_i[ri] * inv_l;
                            let w_new = beta * inv_l;

                            for d in 0..d_v {
                                // P_ij[ri, :] @ V_j[:, d]
                                let mut pv = 0.0f32;
                                for ci in 0..bc_size {
                                    pv += p_ij[ri * bc_size + ci]
                                        * v.data[v_base + (j_start + ci) * d_v + d];
                                }
                                o_i[ri * d_v + d] = w_old * o_i[ri * d_v + d] + w_new * pv;
                            }
                        }

                        m_i[ri] = m_new;
                        l_i[ri] = l_new;
                    }
                }

                // ── Copy block result to output ─────────────────────────────
                for ri in 0..br_size {
                    let dst = o_base + (i_start + ri) * d_v;
                    let src_off = ri * d_v;
                    output[dst..dst + d_v].copy_from_slice(&o_i[src_off..src_off + d_v]);
                }
            }
        }
    }

    Ok(Tensor::new(output, vec![batch, num_heads, seq_q, d_v]))
}

/// Flash Attention v2 with default block sizes (64 × 64).
///
/// See [`flash_attention_with_block_size`] for full documentation.
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
) -> Result<Tensor, OnnxError> {
    flash_attention_with_block_size(
        q,
        k,
        v,
        mask,
        causal,
        FLASH_DEFAULT_BLOCK_R,
        FLASH_DEFAULT_BLOCK_C,
    )
}

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

use crate::kv_cache::KvCache;

/// Cached flash attention for incremental (autoregressive) inference.
///
/// Updates the KV cache with the new token's key/value, then computes
/// attention of the query against the full cached sequence.
///
/// When `q` has `seq_len=1` (the common autoregressive case), the tiled
/// flash algorithm is unnecessary — we use a direct single-query attention
/// path: `scores = q @ full_k^T`, softmax, `@ full_v`.
///
/// # Arguments
/// * `q` — Query `[batch, num_heads, new_seq, head_dim]` (typically new_seq=1)
/// * `k_new` — New key `[batch, num_heads, new_seq, head_dim]`
/// * `v_new` — New value `[batch, num_heads, new_seq, head_dim]`
/// * `cache` — KV cache
/// * `layer_idx` — which layer's cache slot to use
/// * `causal` — whether to apply causal masking
///
/// # Returns
/// Output tensor `[batch, num_heads, new_seq, head_dim]`.
pub fn cached_flash_attention(
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    cache: &mut KvCache,
    layer_idx: usize,
    causal: bool,
) -> Result<Tensor, OnnxError> {
    if q.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "cached_flash_attention: Q must be 4D, got {}D",
            q.ndim()
        )));
    }
    if k_new.ndim() != 4 || v_new.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "cached_flash_attention: K_new and V_new must be 4D".into(),
        ));
    }

    let (full_k, full_v) = cache
        .update(layer_idx, k_new, v_new)
        .map_err(OnnxError::Internal)?;

    let batch = q.shape[0];
    let num_heads = q.shape[1];
    let new_seq = q.shape[2];
    let head_dim = q.shape[3];
    let full_seq = full_k.shape[2];
    let d_v = full_v.shape[3];

    // Optimized path for single-token query (common autoregressive case).
    // No block tiling needed: just a direct vector-matrix product.
    if new_seq == 1 {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; batch * num_heads * d_v];

        for b in 0..batch {
            for h in 0..num_heads {
                let bh = b * num_heads + h;
                let q_off = bh * head_dim; // new_seq=1
                let k_off = bh * full_seq * head_dim;
                let v_off = bh * full_seq * d_v;

                // scores = q[1, head_dim] @ full_k[full_seq, head_dim]^T => [full_seq]
                let mut scores = vec![0.0f32; full_seq];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q.data[q_off + d] * full_k.data[k_off + j * head_dim + d];
                    }
                    *score = dot * scale;
                }

                // Causal mask: for the last token (position = full_seq - 1),
                // all positions 0..full_seq are valid, so no masking needed
                // in the typical autoregressive case. But if causal is set
                // and new_seq > 1 (handled by the else branch), we would mask.

                // Softmax
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    for s in scores.iter_mut() {
                        *s *= inv_sum;
                    }
                }

                // output = scores @ full_v => [d_v]
                let o_off = bh * d_v;
                for d in 0..d_v {
                    let mut val = 0.0f32;
                    for (j, &score) in scores.iter().enumerate() {
                        val += score * full_v.data[v_off + j * d_v + d];
                    }
                    output[o_off + d] = val;
                }
            }
        }

        return Ok(Tensor::new(output, vec![batch, num_heads, 1, d_v]));
    }

    // Multi-token query: fall back to full flash attention
    flash_attention(q, &full_k, &full_v, None, causal)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{multi_head_attention, scaled_dot_product_attention};

    /// Helper: compare two tensors element-wise within tolerance.
    fn assert_tensors_close(a: &Tensor, b: &Tensor, tol: f32, label: &str) {
        assert_eq!(a.shape, b.shape, "{label}: shape mismatch");
        for (i, (av, bv)) in a.data.iter().zip(b.data.iter()).enumerate() {
            assert!(
                (av - bv).abs() < tol,
                "{label}: mismatch at index {i}: {av} vs {bv} (diff={})",
                (av - bv).abs()
            );
        }
    }

    #[test]
    fn test_flash_attention_matches_sdpa_small() {
        // 4×4 attention with block_size=2 to exercise the tiled algorithm
        let (batch, heads, seq, dim) = (1, 1, 4, 8);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.2).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.15 + 1.0).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 2, 2)
            .expect("flash should succeed");
        let sdpa =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

        assert_tensors_close(&flash, &sdpa, 1e-5, "flash_vs_sdpa_4x4");
    }

    #[test]
    fn test_flash_attention_matches_sdpa_block3() {
        // block_size=3 with seq=4 (not divisible) to exercise boundary handling
        let (batch, heads, seq, dim) = (1, 1, 4, 8);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05 + 2.0).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 3, 3)
            .expect("flash should succeed");
        let sdpa =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

        assert_tensors_close(&flash, &sdpa, 1e-5, "flash_vs_sdpa_block3");
    }

    #[test]
    fn test_flash_attention_causal() {
        // Verify causal masking matches SDPA with an explicit causal mask
        let (batch, heads, seq, dim) = (1, 2, 6, 4);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11 + 0.5).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        // Flash attention with causal=true, block_size=2
        let flash = flash_attention_with_block_size(&q, &k, &v, None, true, 2, 2)
            .expect("flash causal should succeed");

        // Reference: SDPA with explicit causal mask
        let mut mask_data = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        let causal_mask = Tensor::new(mask_data, vec![seq, seq]);
        let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&causal_mask), None)
            .expect("SDPA with causal mask should succeed");

        assert_tensors_close(&flash, &sdpa, 1e-5, "flash_causal");
    }

    #[test]
    fn test_flash_attention_causal_with_additive_mask() {
        // Causal + additive mask combined
        let (batch, heads, seq, dim) = (1, 1, 5, 4);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.09).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.12).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.08 + 1.5).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        // Additive mask: penalize position 0 key by -5
        let mut add_mask = vec![0.0f32; seq * seq];
        for i in 0..seq {
            add_mask[i * seq] = -5.0;
        }
        let add_tensor = Tensor::new(add_mask.clone(), vec![seq, seq]);

        let flash = flash_attention_with_block_size(&q, &k, &v, Some(&add_tensor), true, 2, 2)
            .expect("flash causal+mask should succeed");

        // Reference: SDPA with combined causal + additive mask
        let mut combined = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    combined[i * seq + j] = f32::NEG_INFINITY;
                }
                combined[i * seq + j] += add_mask[i * seq + j];
            }
        }
        let combined_mask = Tensor::new(combined, vec![seq, seq]);
        let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&combined_mask), None)
            .expect("SDPA with combined mask should succeed");

        assert_tensors_close(&flash, &sdpa, 1e-5, "flash_causal_additive");
    }

    #[test]
    fn test_flash_attention_batch_multihead() {
        // batch=2, heads=4
        let (batch, heads, seq, dim) = (2, 4, 8, 8);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.04 + 0.7).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        // Use block_size=3 (not dividing seq=8 evenly)
        let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 3, 3)
            .expect("flash batch+heads should succeed");
        let sdpa =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

        assert_eq!(flash.shape, vec![batch, heads, seq, dim]);
        assert_tensors_close(&flash, &sdpa, 1e-5, "flash_batch_multihead");
    }

    #[test]
    fn test_flash_attention_large_seq_stability() {
        // 256 tokens, verify no NaN/Inf
        let (batch, heads, seq, dim) = (1, 2, 256, 16);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.025).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.018 + 0.3).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        // block_size=32, so we get 256/32 = 8 blocks
        let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 32, 32)
            .expect("flash large seq should succeed");

        assert_eq!(flash.shape, vec![batch, heads, seq, dim]);
        for (i, &val) in flash.data.iter().enumerate() {
            assert!(val.is_finite(), "NaN/Inf at index {i} (val={val})",);
        }

        // Also verify against SDPA reference
        let sdpa =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");
        assert_tensors_close(&flash, &sdpa, 1e-4, "flash_large_seq");
    }

    #[test]
    fn test_flash_attention_large_seq_causal_stability() {
        // 256 tokens, causal, verify numerical stability
        let (batch, heads, seq, dim) = (1, 1, 256, 16);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.04).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.06).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.035 + 1.2).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        let flash = flash_attention_with_block_size(&q, &k, &v, None, true, 32, 32)
            .expect("flash large causal should succeed");

        for (i, &val) in flash.data.iter().enumerate() {
            assert!(val.is_finite(), "NaN/Inf at index {i}");
        }

        // Reference with explicit causal mask
        let mut mask_data = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        let causal_mask = Tensor::new(mask_data, vec![seq, seq]);
        let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&causal_mask), None)
            .expect("SDPA should succeed");
        assert_tensors_close(&flash, &sdpa, 1e-4, "flash_large_causal");
    }

    #[test]
    fn test_flash_attention_block_boundary_edge() {
        // seq=7, block_size=3 → blocks of [3,3,1]
        let (batch, heads, seq, dim) = (1, 1, 7, 4);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.23).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.19).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.27 + 0.4).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 3, 3)
            .expect("flash boundary should succeed");
        let sdpa =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

        assert_tensors_close(&flash, &sdpa, 1e-5, "flash_block_boundary");
    }

    #[test]
    fn test_flash_attention_asymmetric_blocks() {
        // Different block_r and block_c
        let (batch, heads, seq, dim) = (1, 1, 10, 4);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.14).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.21).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17 + 0.9).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        // block_r=2, block_c=4
        let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 2, 4)
            .expect("flash asymmetric blocks should succeed");
        let sdpa =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

        assert_tensors_close(&flash, &sdpa, 1e-5, "flash_asymmetric_blocks");
    }

    #[test]
    fn test_flash_attention_default_block_fallback() {
        // seq < default block (64), should fall through to SDPA
        let (batch, heads, seq, dim) = (1, 1, 4, 8);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.2).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.15 + 1.0).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        let flash = flash_attention(&q, &k, &v, None, false).expect("flash default should succeed");
        let sdpa =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

        assert_tensors_close(&flash, &sdpa, 1e-6, "flash_default_fallback");
    }

    #[test]
    fn test_multi_head_flash_attention_basic() {
        // batch=1, seq=4, embed=8, heads=2
        let (batch, seq, embed, heads) = (1, 4, 8, 2);
        let n = batch * seq * embed;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.08).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.12).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1 + 0.5).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, seq, embed]);
        let k = Tensor::new(k_data, vec![batch, seq, embed]);
        let v = Tensor::new(v_data, vec![batch, seq, embed]);

        let mhfa = multi_head_flash_attention(&q, &k, &v, None, false, heads)
            .expect("MHFA should succeed");

        assert_eq!(mhfa.shape, vec![batch, seq, embed]);

        // Compare against standard MHA (which uses SDPA internally)
        let mha = multi_head_attention(&q, &k, &v, None, None, None, None, None, heads)
            .expect("MHA should succeed");

        assert_tensors_close(&mhfa, &mha, 1e-5, "mhfa_vs_mha");
    }

    #[test]
    fn test_multi_head_flash_attention_causal() {
        let (batch, seq, embed, heads) = (2, 6, 16, 4);
        let n = batch * seq * embed;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.09 + 0.8).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, seq, embed]);
        let k = Tensor::new(k_data, vec![batch, seq, embed]);
        let v = Tensor::new(v_data, vec![batch, seq, embed]);

        let mhfa = multi_head_flash_attention(&q, &k, &v, None, true, heads)
            .expect("MHFA causal should succeed");

        assert_eq!(mhfa.shape, vec![batch, seq, embed]);
        for (i, &val) in mhfa.data.iter().enumerate() {
            assert!(val.is_finite(), "NaN/Inf at index {i}");
        }
    }

    #[test]
    fn test_flash_attention_4d_mask() {
        // 4D mask [batch, heads, seq_q, seq_k]
        let (batch, heads, seq, dim) = (2, 2, 4, 4);
        let n = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.06).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.09).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07 + 1.0).sin()).collect();

        let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
        let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
        let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

        // 4D mask that zeros out last key position
        let mask_n = batch * heads * seq * seq;
        let mut mask_data = vec![0.0f32; mask_n];
        for b in 0..batch {
            for h in 0..heads {
                for i in 0..seq {
                    let idx = ((b * heads + h) * seq + i) * seq + (seq - 1);
                    mask_data[idx] = -1e9;
                }
            }
        }
        let mask = Tensor::new(mask_data, vec![batch, heads, seq, seq]);

        // block_size=2 to exercise the flash algorithm
        let flash = flash_attention_with_block_size(&q, &k, &v, Some(&mask), false, 2, 2)
            .expect("flash with 4D mask should succeed");
        let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None)
            .expect("SDPA should succeed");

        assert_tensors_close(&flash, &sdpa, 1e-5, "flash_4d_mask");
    }

    #[test]
    fn test_flash_attention_error_on_wrong_dims() {
        let q = Tensor::new(vec![1.0; 12], vec![3, 4]);
        let k = Tensor::new(vec![1.0; 12], vec![3, 4]);
        let v = Tensor::new(vec![1.0; 12], vec![3, 4]);

        let result = flash_attention(&q, &k, &v, None, false);
        assert!(result.is_err(), "should fail on 2D inputs");
    }

    #[test]
    fn test_flash_attention_uniform_values() {
        // All-ones Q,K,V: output should also be all-ones V
        let (batch, heads, seq, dim) = (1, 1, 8, 4);
        let n = batch * heads * seq * dim;
        let q = Tensor::new(vec![1.0f32; n], vec![batch, heads, seq, dim]);
        let k = Tensor::new(vec![1.0f32; n], vec![batch, heads, seq, dim]);
        let v = Tensor::new(vec![1.0f32; n], vec![batch, heads, seq, dim]);

        let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 3, 3)
            .expect("flash uniform should succeed");

        for (i, &val) in flash.data.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "Expected ~1.0 at index {i}, got {val}",
            );
        }
    }
}
