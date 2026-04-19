//! Flash Attention v2 tiled kernel — `flash_attention_with_block_size` and
//! `flash_attention`.
//!
//! The outer block-tiling loop (br_idx × bc_idx) and online-softmax
//! accumulator (m_i, l_i, o_i) are preserved exactly as in Flash Attention v2.
//!
//! Inner Q·K^T scoring delegates to `compute_qk_scores` from `simd_sdpa`
//! (when the `simd` feature is active), which dispatches to NEON / AVX2+FMA
//! or scalar based on the runtime CPU.
//!
//! **Note:** `softmax_inplace` is intentionally NOT used here.  The online
//! softmax in the tiled loop maintains unnormalized `p_ij = exp(s - m_ij)`
//! values whose partial sums are blended across blocks.  Normalizing them
//! inside the loop (as `softmax_inplace` does) would break the Flash Attention
//! v2 algebra.

use oxionnx_core::{OnnxError, Tensor};

use crate::attention::scaled_dot_product_attention;

use super::{flash_mask_value, FLASH_DEFAULT_BLOCK_C, FLASH_DEFAULT_BLOCK_R};

// ── SIMD Q·K^T helper ────────────────────────────────────────────────────────

/// Compute Q·K^T scores for one row of the Q block against the full K block.
///
/// When the `simd` feature is enabled delegates to `simd_sdpa::compute_qk_scores`
/// (NEON / AVX2+FMA / scalar dispatch).  Otherwise falls back to an inline
/// scalar loop so the crate compiles without the feature.
#[cfg(feature = "simd")]
#[inline]
fn qk_scores_row(
    q_row: &[f32],
    k_block: &[f32],
    scale: f32,
    head_dim: usize,
    bc_size: usize,
    out: &mut [f32],
) {
    use crate::attention::simd_sdpa::compute_qk_scores;
    compute_qk_scores(q_row, k_block, scale, head_dim, bc_size, out);
}

#[cfg(not(feature = "simd"))]
#[inline]
fn qk_scores_row(
    q_row: &[f32],
    k_block: &[f32],
    scale: f32,
    head_dim: usize,
    bc_size: usize,
    out: &mut [f32],
) {
    for ci in 0..bc_size {
        let k_row = &k_block[ci * head_dim..(ci + 1) * head_dim];
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q_row[d] * k_row[d];
        }
        out[ci] = dot * scale;
    }
}

// ── Flash Attention v2 blocked kernel ────────────────────────────────────────

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
                    // `compute_qk_scores` dispatches to NEON/AVX2+FMA/scalar.
                    // Called once per Q row; the K block slice starts at
                    // k_base + j_start*head_dim and has bc_size rows.
                    let mut s_ij = vec![0.0f32; br_size * bc_size];
                    let k_block = &k.data[k_base + j_start * head_dim
                        ..k_base + j_start * head_dim + bc_size * head_dim];
                    for ri in 0..br_size {
                        let q_row = &q.data[q_base + (i_start + ri) * head_dim
                            ..q_base + (i_start + ri) * head_dim + head_dim];
                        let score_row = &mut s_ij[ri * bc_size..(ri + 1) * bc_size];
                        qk_scores_row(q_row, k_block, scale, head_dim, bc_size, score_row);
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
                    // NOTE: We intentionally do NOT call `softmax_inplace` here.
                    // Flash Attention v2 keeps p_ij = exp(s - m_ij) unnormalized
                    // so that partial sums can be blended across blocks.
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
