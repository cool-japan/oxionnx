//! Cached flash attention for incremental (autoregressive) inference.
//!
//! The single-token (new_seq == 1) path is a standard SDPA, so all three
//! `simd_sdpa` primitives apply directly:
//!
//! - `compute_qk_scores`  — replaces the scalar Q·K^T dot-product loop
//! - `softmax_inplace`    — replaces the manual max/exp/sum/normalize steps
//! - `weighted_sum_v`     — replaces the weighted V accumulation
//!
//! The multi-token fallback delegates to `flash_attention` which has its own
//! inner SIMD wiring in `kernel.rs`.

use oxionnx_core::{OnnxError, Tensor};

use crate::kv_cache::KvCache;

use super::kernel::flash_attention;

// ── SIMD helpers (feature-gated) ─────────────────────────────────────────────

/// Compute Q·K^T scores for a single query row.
#[cfg(feature = "simd")]
#[inline]
fn compute_scores(
    q_row: &[f32],
    k_mat: &[f32],
    scale: f32,
    head_dim: usize,
    seq: usize,
    out: &mut [f32],
) {
    use crate::attention::simd_sdpa::compute_qk_scores;
    compute_qk_scores(q_row, k_mat, scale, head_dim, seq, out);
}

#[cfg(not(feature = "simd"))]
#[inline]
fn compute_scores(
    q_row: &[f32],
    k_mat: &[f32],
    scale: f32,
    head_dim: usize,
    _seq: usize,
    out: &mut [f32],
) {
    for (j, score) in out.iter_mut().enumerate() {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q_row[d] * k_mat[j * head_dim + d];
        }
        *score = dot * scale;
    }
}

/// In-place numerically-stable softmax.
#[cfg(feature = "simd")]
#[inline]
fn apply_softmax(scores: &mut [f32]) {
    use crate::attention::simd_sdpa::softmax_inplace;
    softmax_inplace(scores);
}

#[cfg(not(feature = "simd"))]
#[inline]
fn apply_softmax(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    if sum > 0.0 {
        let inv = sum.recip();
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
}

/// Softmax-weighted sum of V rows into `output`.
#[cfg(feature = "simd")]
#[inline]
fn accumulate_v(weights: &[f32], v_mat: &[f32], head_dim: usize, seq: usize, output: &mut [f32]) {
    use crate::attention::simd_sdpa::weighted_sum_v;
    // weighted_sum_v zeros output internally before accumulating
    weighted_sum_v(weights, v_mat, head_dim, seq, output);
}

#[cfg(not(feature = "simd"))]
#[inline]
fn accumulate_v(weights: &[f32], v_mat: &[f32], head_dim: usize, seq: usize, output: &mut [f32]) {
    output.fill(0.0);
    for j in 0..seq {
        let w_j = weights[j];
        let v_row = &v_mat[j * head_dim..(j + 1) * head_dim];
        for d in 0..head_dim {
            output[d] += w_j * v_row[d];
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Cached flash attention for incremental (autoregressive) inference.
///
/// Updates the KV cache with the new token's key/value, then computes
/// attention of the query against the full cached sequence.
///
/// When `q` has `seq_len=1` (the common autoregressive case), the tiled
/// flash algorithm is unnecessary — we use a direct single-query attention
/// path: `scores = q @ full_k^T`, softmax, `@ full_v`.
///
/// Inner loops use SIMD-accelerated primitives from `simd_sdpa` when the
/// `simd` feature is active.
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
    // A standard SDPA: scores, softmax, weighted V — all SIMD-accelerated.
    if new_seq == 1 {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; batch * num_heads * d_v];

        let mut scores = vec![0.0f32; full_seq];
        let mut out_row = vec![0.0f32; d_v];

        for b in 0..batch {
            for h in 0..num_heads {
                let bh = b * num_heads + h;
                let q_off = bh * head_dim; // new_seq=1
                let k_off = bh * full_seq * head_dim;
                let v_off = bh * full_seq * d_v;

                // scores[j] = dot(q, k_j) * scale  — SIMD via compute_qk_scores
                let q_row = &q.data[q_off..q_off + head_dim];
                let k_mat = &full_k.data[k_off..k_off + full_seq * head_dim];
                compute_scores(q_row, k_mat, scale, head_dim, full_seq, &mut scores);

                // Causal mask: for new_seq=1 the query sits at position
                // full_seq-1 and all cached keys 0..full_seq are valid, so no
                // masking needed in the standard autoregressive case.

                // softmax in-place — SIMD via softmax_inplace
                apply_softmax(&mut scores);

                // output = scores @ full_v  — SIMD via weighted_sum_v
                let v_mat = &full_v.data[v_off..v_off + full_seq * d_v];
                accumulate_v(&scores, v_mat, d_v, full_seq, &mut out_row);

                let o_off = bh * d_v;
                output[o_off..o_off + d_v].copy_from_slice(&out_row);
            }
        }

        return Ok(Tensor::new(output, vec![batch, num_heads, 1, d_v]));
    }

    // Multi-token query: fall back to full flash attention
    flash_attention(q, &full_k, &full_v, None, causal)
}
