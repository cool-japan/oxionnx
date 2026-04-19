//! Core attention kernels: SDPA, multi-head attention, reshape helpers, and rotary embedding.

use oxionnx_core::{OnnxError, Tensor};

// ── Private helpers ──────────────────────────────────────────────────────────

/// Softmax along last dimension for a flat buffer with given inner dimension.
pub(super) fn softmax_last_dim(data: &mut [f32], inner: usize) {
    for chunk in data.chunks_exact_mut(inner) {
        let max_val = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in chunk.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in chunk.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// 2D matmul: [m, k] @ [k, n] -> [m, n]
pub(super) fn mm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_val = a[i * k + kk];
            for j in 0..n {
                out[i * n + j] += a_val * b[kk * n + j];
            }
        }
    }
    out
}

/// [m, k] @ [n, k]^T -> [m, n]
pub(super) fn mm_a_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[j * k + kk];
            }
            out[i * n + j] = s;
        }
    }
    out
}

// ── Scaled Dot-Product Attention ─────────────────────────────────────────────

/// Scaled dot-product attention.
///
/// # Arguments
/// * `q` - Query `[..., seq_q, d_k]`
/// * `k` - Key `[..., seq_k, d_k]`
/// * `v` - Value `[..., seq_k, d_v]`
/// * `mask` - Additive mask (optional), broadcastable to `[..., seq_q, seq_k]`
/// * `scale` - Custom scale factor (default: 1/sqrt(d_k))
///
/// # Returns
/// `[..., seq_q, d_v]`
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor, OnnxError> {
    let q_ndim = q.ndim();
    if q_ndim < 2 {
        return Err(OnnxError::ShapeMismatch(
            "scaled_dot_product_attention: Q must be at least 2D".to_string(),
        ));
    }
    let d_k = q.shape[q_ndim - 1];
    let seq_q = q.shape[q_ndim - 2];
    let seq_k = k.shape[k.ndim() - 2];
    let d_v = v.shape[v.ndim() - 1];
    let scale_val = scale.unwrap_or(1.0 / (d_k as f32).sqrt());
    let q_batch: usize = q.shape[..q_ndim - 2].iter().product::<usize>().max(1);
    let k_batch: usize = k.shape[..k.ndim() - 2].iter().product::<usize>().max(1);
    let v_batch: usize = v.shape[..v.ndim() - 2].iter().product::<usize>().max(1);
    let batch = q_batch.max(k_batch).max(v_batch);
    let q_stride = seq_q * d_k;
    let k_stride = seq_k * d_k;
    let v_stride = seq_k * d_v;
    let mask_data = mask.map(|m| &m.data[..]);
    let mask_stride = if let Some(m) = mask {
        let m_batch: usize = m.shape[..m.ndim().saturating_sub(2)]
            .iter()
            .product::<usize>()
            .max(1);
        if m_batch == 1 {
            0
        } else {
            seq_q * seq_k
        }
    } else {
        0
    };
    let mut output = vec![0.0f32; batch * seq_q * d_v];

    #[cfg(feature = "simd")]
    {
        let mut row_scores = vec![0.0f32; seq_k];
        let mut row_out = vec![0.0f32; d_v];
        for b in 0..batch {
            let q_off = (b % q_batch) * q_stride;
            let k_off = (b % k_batch) * k_stride;
            let v_off = (b % v_batch) * v_stride;
            let o_off = b * seq_q * d_v;
            let q_slice = &q.data[q_off..q_off + q_stride];
            let k_slice = &k.data[k_off..k_off + k_stride];
            let v_slice = &v.data[v_off..v_off + v_stride];
            for q_i in 0..seq_q {
                let q_row = &q_slice[q_i * d_k..(q_i + 1) * d_k];
                // (1) Q·K^T dot products
                crate::attention::simd_sdpa::compute_qk_scores(
                    q_row,
                    k_slice,
                    scale_val,
                    d_k,
                    seq_k,
                    &mut row_scores,
                );
                // Apply additive mask if present
                if let Some(md) = mask_data {
                    let m_off = if mask_stride == 0 { 0 } else { b * mask_stride };
                    let row_m_off = m_off + q_i * seq_k;
                    for (kv_j, s) in row_scores.iter_mut().enumerate() {
                        let m_idx = row_m_off + kv_j;
                        if m_idx < md.len() {
                            *s += md[m_idx];
                        }
                    }
                }
                // (2) Row-wise softmax
                crate::attention::simd_sdpa::softmax_inplace(&mut row_scores);
                // (3) Softmax · V weighted sum
                crate::attention::simd_sdpa::weighted_sum_v(
                    &row_scores,
                    v_slice,
                    d_v,
                    seq_k,
                    &mut row_out,
                );
                output[o_off + q_i * d_v..o_off + (q_i + 1) * d_v].copy_from_slice(&row_out);
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for b in 0..batch {
            let q_off = (b % q_batch) * q_stride;
            let k_off = (b % k_batch) * k_stride;
            let v_off = (b % v_batch) * v_stride;
            let o_off = b * seq_q * d_v;
            let q_slice = &q.data[q_off..q_off + q_stride];
            let k_slice = &k.data[k_off..k_off + k_stride];
            let mut scores = mm_a_bt(q_slice, k_slice, seq_q, d_k, seq_k);
            for s in scores.iter_mut() {
                *s *= scale_val;
            }
            if let Some(md) = mask_data {
                let m_off = if mask_stride == 0 { 0 } else { b * mask_stride };
                for (i, s) in scores.iter_mut().enumerate() {
                    let m_idx = m_off + i;
                    if m_idx < md.len() {
                        *s += md[m_idx];
                    }
                }
            }
            softmax_last_dim(&mut scores, seq_k);
            let v_slice = &v.data[v_off..v_off + v_stride];
            let attn_out = mm(&scores, v_slice, seq_q, seq_k, d_v);
            output[o_off..o_off + seq_q * d_v].copy_from_slice(&attn_out);
        }
    }
    let mut out_shape = q.shape[..q_ndim - 2].to_vec();
    out_shape.push(seq_q);
    out_shape.push(d_v);
    Ok(Tensor::new(output, out_shape))
}

// ── Reshape helpers ──────────────────────────────────────────────────────────

/// Reshape `[batch, seq, num_heads*head_dim]` to `[batch, num_heads, seq, head_dim]`.
pub(crate) fn reshape_to_heads(
    t: &Tensor,
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    let mut out = vec![0.0f32; batch * num_heads * seq * head_dim];
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src = b * seq * num_heads * head_dim
                        + s * num_heads * head_dim
                        + h * head_dim
                        + d;
                    let dst =
                        b * num_heads * seq * head_dim + h * seq * head_dim + s * head_dim + d;
                    out[dst] = t.data[src];
                }
            }
        }
    }
    Tensor::new(out, vec![batch, num_heads, seq, head_dim])
}

/// Reshape `[batch, num_heads, seq, head_dim]` back to `[batch, seq, num_heads*head_dim]`.
pub(crate) fn reshape_from_heads(
    t: &Tensor,
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    let embed_dim = num_heads * head_dim;
    let mut out = vec![0.0f32; batch * seq * embed_dim];
    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq {
                for d in 0..head_dim {
                    let src =
                        b * num_heads * seq * head_dim + h * seq * head_dim + s * head_dim + d;
                    let dst = b * seq * embed_dim + s * embed_dim + h * head_dim + d;
                    out[dst] = t.data[src];
                }
            }
        }
    }
    Tensor::new(out, vec![batch, seq, embed_dim])
}

// ── Multi-Head Attention ─────────────────────────────────────────────────────

/// Multi-head attention.
///
/// # Arguments
/// * `query` - `[batch, seq_q, embed_dim]`
/// * `key` - `[batch, seq_k, embed_dim]`
/// * `value` - `[batch, seq_k, embed_dim]`
/// * `qkv_weight` - Packed QKV projection `[3*embed_dim, embed_dim]` (optional)
/// * `qkv_bias` - `[3*embed_dim]` (optional)
/// * `out_proj_weight` - `[embed_dim, embed_dim]` (optional)
/// * `out_proj_bias` - `[embed_dim]` (optional)
/// * `mask` - Additive mask (optional)
/// * `num_heads` - Number of attention heads
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    qkv_weight: Option<&Tensor>,
    qkv_bias: Option<&Tensor>,
    out_proj_weight: Option<&Tensor>,
    out_proj_bias: Option<&Tensor>,
    mask: Option<&Tensor>,
    num_heads: usize,
) -> Result<Tensor, OnnxError> {
    let batch = query.shape[0];
    let seq_q = query.shape[1];
    let embed_dim = query.shape[2];
    let seq_k = key.shape[1];
    if embed_dim % num_heads != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "embed_dim {} not divisible by num_heads {}",
            embed_dim, num_heads
        )));
    }
    let head_dim = embed_dim / num_heads;
    let (q_proj, k_proj, v_proj) = if let Some(w) = qkv_weight {
        let dim3 = 3 * embed_dim;
        let bias_data = qkv_bias.map(|b| &b.data[..]);
        let mut q_data = vec![0.0f32; batch * seq_q * embed_dim];
        let mut k_data = vec![0.0f32; batch * seq_k * embed_dim];
        let mut v_data = vec![0.0f32; batch * seq_k * embed_dim];
        let w_q = &w.data[..embed_dim * embed_dim];
        let w_k = &w.data[embed_dim * embed_dim..2 * embed_dim * embed_dim];
        let w_v = &w.data[2 * embed_dim * embed_dim..dim3 * embed_dim];
        for b_idx in 0..batch {
            let q_off = b_idx * seq_q * embed_dim;
            let q_src = &query.data[q_off..q_off + seq_q * embed_dim];
            let projected = mm_a_bt(q_src, w_q, seq_q, embed_dim, embed_dim);
            q_data[q_off..q_off + seq_q * embed_dim].copy_from_slice(&projected);
        }
        for b_idx in 0..batch {
            let k_off = b_idx * seq_k * embed_dim;
            let k_src = &key.data[k_off..k_off + seq_k * embed_dim];
            let projected = mm_a_bt(k_src, w_k, seq_k, embed_dim, embed_dim);
            k_data[k_off..k_off + seq_k * embed_dim].copy_from_slice(&projected);
        }
        for b_idx in 0..batch {
            let v_off = b_idx * seq_k * embed_dim;
            let v_src = &value.data[v_off..v_off + seq_k * embed_dim];
            let projected = mm_a_bt(v_src, w_v, seq_k, embed_dim, embed_dim);
            v_data[v_off..v_off + seq_k * embed_dim].copy_from_slice(&projected);
        }
        if let Some(bd) = bias_data {
            let bq = &bd[..embed_dim];
            let bk = &bd[embed_dim..2 * embed_dim];
            let bv = &bd[2 * embed_dim..3 * embed_dim];
            for b_idx in 0..batch {
                for s in 0..seq_q {
                    for d in 0..embed_dim {
                        q_data[b_idx * seq_q * embed_dim + s * embed_dim + d] += bq[d];
                    }
                }
                for s in 0..seq_k {
                    for d in 0..embed_dim {
                        k_data[b_idx * seq_k * embed_dim + s * embed_dim + d] += bk[d];
                        v_data[b_idx * seq_k * embed_dim + s * embed_dim + d] += bv[d];
                    }
                }
            }
        }
        (
            Tensor::new(q_data, vec![batch, seq_q, embed_dim]),
            Tensor::new(k_data, vec![batch, seq_k, embed_dim]),
            Tensor::new(v_data, vec![batch, seq_k, embed_dim]),
        )
    } else {
        (query.clone(), key.clone(), value.clone())
    };
    let q_heads = reshape_to_heads(&q_proj, batch, seq_q, num_heads, head_dim);
    let k_heads = reshape_to_heads(&k_proj, batch, seq_k, num_heads, head_dim);
    let v_heads = reshape_to_heads(&v_proj, batch, seq_k, num_heads, head_dim);
    let attn_out = scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask, None)?;
    let concat = reshape_from_heads(&attn_out, batch, seq_q, num_heads, head_dim);
    let result = if let Some(w_out) = out_proj_weight {
        let mut out_data = vec![0.0f32; batch * seq_q * embed_dim];
        for b_idx in 0..batch {
            let off = b_idx * seq_q * embed_dim;
            let src = &concat.data[off..off + seq_q * embed_dim];
            let projected = mm_a_bt(src, &w_out.data, seq_q, embed_dim, embed_dim);
            out_data[off..off + seq_q * embed_dim].copy_from_slice(&projected);
        }
        if let Some(bias) = out_proj_bias {
            for b_idx in 0..batch {
                for s in 0..seq_q {
                    for d in 0..embed_dim {
                        out_data[b_idx * seq_q * embed_dim + s * embed_dim + d] += bias.data[d];
                    }
                }
            }
        }
        Tensor::new(out_data, vec![batch, seq_q, embed_dim])
    } else {
        concat
    };
    Ok(result)
}

// ── Rotary Embedding (RoPE) ──────────────────────────────────────────────────

/// Apply rotary positional embedding (RoPE).
///
/// # Arguments
/// * `input` - `[..., seq_len, head_dim]`
/// * `position_ids` - `[..., seq_len]`
/// * `cos_cache` - Precomputed cos values (optional)
/// * `sin_cache` - Precomputed sin values (optional)
/// * `base` - Base frequency (default 10000.0)
///
/// # Returns
/// Tensor with same shape as input, with rotary embedding applied.
pub fn rotary_embedding(
    input: &Tensor,
    position_ids: &Tensor,
    cos_cache: Option<&Tensor>,
    sin_cache: Option<&Tensor>,
    base: f32,
) -> Result<Tensor, OnnxError> {
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(OnnxError::ShapeMismatch(
            "rotary_embedding: input must be at least 2D".to_string(),
        ));
    }
    let head_dim = input.shape[ndim - 1];
    let seq_len = input.shape[ndim - 2];
    let half_dim = head_dim / 2;
    if head_dim % 2 != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "rotary_embedding: head_dim {} must be even",
            head_dim
        )));
    }
    let batch_dims: usize = input.shape[..ndim - 2].iter().product::<usize>().max(1);
    let pos_stride = seq_len;
    let mut output = input.data.clone();
    let (cos_vals, sin_vals) = if let (Some(cc), Some(sc)) = (cos_cache, sin_cache) {
        (cc.data.clone(), sc.data.clone())
    } else {
        let freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let max_pos = position_ids.data.iter().copied().fold(0.0f32, f32::max) as usize + 1;
        let mut cos_v = vec![0.0f32; max_pos * half_dim];
        let mut sin_v = vec![0.0f32; max_pos * half_dim];
        for p in 0..max_pos {
            for (i, &f) in freq.iter().enumerate() {
                let angle = p as f32 * f;
                cos_v[p * half_dim + i] = angle.cos();
                sin_v[p * half_dim + i] = angle.sin();
            }
        }
        (cos_v, sin_v)
    };
    let stride = seq_len * head_dim;
    for b in 0..batch_dims {
        for s in 0..seq_len {
            let pos_idx = b % (position_ids.data.len() / pos_stride.max(1));
            let pos = position_ids.data[pos_idx * pos_stride + s] as usize;
            let base_idx = b * stride + s * head_dim;
            for i in 0..half_dim {
                let cos_val = cos_vals[pos * half_dim + i];
                let sin_val = sin_vals[pos * half_dim + i];
                let x0 = input.data[base_idx + i];
                let x1 = input.data[base_idx + half_dim + i];
                output[base_idx + i] = x0 * cos_val - x1 * sin_val;
                output[base_idx + half_dim + i] = x1 * cos_val + x0 * sin_val;
            }
        }
    }
    Ok(Tensor::new(output, input.shape.clone()))
}
