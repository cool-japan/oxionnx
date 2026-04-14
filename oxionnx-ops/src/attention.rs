//! Attention mechanism kernels: scaled dot-product, multi-head, and rotary embedding.

use oxionnx_core::{OnnxError, Tensor};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Softmax along last dimension for a flat buffer with given inner dimension.
fn softmax_last_dim(data: &mut [f32], inner: usize) {
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
fn mm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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
fn mm_a_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

// ── Scaled Dot-Product Attention ────────────────────────────────────────────

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

    // Compute batch dimensions
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

    for b in 0..batch {
        let q_off = (b % q_batch) * q_stride;
        let k_off = (b % k_batch) * k_stride;
        let v_off = (b % v_batch) * v_stride;
        let o_off = b * seq_q * d_v;

        // scores = Q @ K^T -> [seq_q, seq_k]
        let q_slice = &q.data[q_off..q_off + q_stride];
        let k_slice = &k.data[k_off..k_off + k_stride];
        let mut scores = mm_a_bt(q_slice, k_slice, seq_q, d_k, seq_k);

        // Scale
        for s in scores.iter_mut() {
            *s *= scale_val;
        }

        // Add mask
        if let Some(md) = mask_data {
            let m_off = if mask_stride == 0 { 0 } else { b * mask_stride };
            for (i, s) in scores.iter_mut().enumerate() {
                let m_idx = m_off + i;
                if m_idx < md.len() {
                    *s += md[m_idx];
                }
            }
        }

        // Softmax along last dim (seq_k)
        softmax_last_dim(&mut scores, seq_k);

        // output = scores @ V -> [seq_q, d_v]
        let v_slice = &v.data[v_off..v_off + v_stride];
        let attn_out = mm(&scores, v_slice, seq_q, seq_k, d_v);
        output[o_off..o_off + seq_q * d_v].copy_from_slice(&attn_out);
    }

    let mut out_shape = q.shape[..q_ndim - 2].to_vec();
    out_shape.push(seq_q);
    out_shape.push(d_v);

    Ok(Tensor::new(output, out_shape))
}

// ── Multi-Head Attention ────────────────────────────────────────────────────

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

    // Project Q, K, V
    let (q_proj, k_proj, v_proj) = if let Some(w) = qkv_weight {
        // w: [3*embed_dim, embed_dim]
        let dim3 = 3 * embed_dim;
        let bias_data = qkv_bias.map(|b| &b.data[..]);

        let mut q_data = vec![0.0f32; batch * seq_q * embed_dim];
        let mut k_data = vec![0.0f32; batch * seq_k * embed_dim];
        let mut v_data = vec![0.0f32; batch * seq_k * embed_dim];

        // Project query -> Q: [batch, seq_q, embed_dim] @ [embed_dim, 3*embed_dim] but we only need first embed_dim cols
        // Actually w is [3*embed, embed], so we do query @ w^T to get [batch, seq_q, 3*embed]
        // Then split into Q, K, V each of [batch, seq, embed]
        // But key and value may differ from query, so we project them separately through their parts of w.

        let w_q = &w.data[..embed_dim * embed_dim]; // rows 0..embed_dim
        let w_k = &w.data[embed_dim * embed_dim..2 * embed_dim * embed_dim];
        let w_v = &w.data[2 * embed_dim * embed_dim..dim3 * embed_dim];

        // Q = query @ W_q^T + bias_q
        for b_idx in 0..batch {
            let q_off = b_idx * seq_q * embed_dim;
            let q_src = &query.data[q_off..q_off + seq_q * embed_dim];
            let projected = mm_a_bt(q_src, w_q, seq_q, embed_dim, embed_dim);
            q_data[q_off..q_off + seq_q * embed_dim].copy_from_slice(&projected);
        }
        // K = key @ W_k^T + bias_k
        for b_idx in 0..batch {
            let k_off = b_idx * seq_k * embed_dim;
            let k_src = &key.data[k_off..k_off + seq_k * embed_dim];
            let projected = mm_a_bt(k_src, w_k, seq_k, embed_dim, embed_dim);
            k_data[k_off..k_off + seq_k * embed_dim].copy_from_slice(&projected);
        }
        // V = value @ W_v^T + bias_v
        for b_idx in 0..batch {
            let v_off = b_idx * seq_k * embed_dim;
            let v_src = &value.data[v_off..v_off + seq_k * embed_dim];
            let projected = mm_a_bt(v_src, w_v, seq_k, embed_dim, embed_dim);
            v_data[v_off..v_off + seq_k * embed_dim].copy_from_slice(&projected);
        }

        // Add bias
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

    // Reshape to [batch, num_heads, seq, head_dim]
    // From [batch, seq, embed_dim] -> [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
    let q_heads = reshape_to_heads(&q_proj, batch, seq_q, num_heads, head_dim);
    let k_heads = reshape_to_heads(&k_proj, batch, seq_k, num_heads, head_dim);
    let v_heads = reshape_to_heads(&v_proj, batch, seq_k, num_heads, head_dim);

    // Apply scaled dot-product attention
    // q_heads, k_heads, v_heads are [batch, num_heads, seq, head_dim]
    let attn_out = scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask, None)?;

    // attn_out: [batch, num_heads, seq_q, head_dim] -> [batch, seq_q, embed_dim]
    let concat = reshape_from_heads(&attn_out, batch, seq_q, num_heads, head_dim);

    // Apply output projection
    let result = if let Some(w_out) = out_proj_weight {
        // concat @ w_out^T + bias
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

/// Reshape [batch, seq, num_heads*head_dim] to [batch, num_heads, seq, head_dim]
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

/// Reshape [batch, num_heads, seq, head_dim] back to [batch, seq, num_heads*head_dim]
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

// ── Rotary Embedding ────────────────────────────────────────────────────────

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
    let pos_stride = seq_len; // last dim of position_ids

    let mut output = input.data.clone();

    // Compute or use cached cos/sin
    let (cos_vals, sin_vals) = if let (Some(cc), Some(sc)) = (cos_cache, sin_cache) {
        (cc.data.clone(), sc.data.clone())
    } else {
        // Compute frequencies: freq[i] = 1 / base^(2i/d) for i in 0..d/2
        let freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Find max position
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

// ── Multi-Query Attention (MQA) ─────────────────────────────────────────────

/// Multi-query attention: single K/V head shared across all Q heads.
///
/// # Arguments
/// * `q` - Query `[batch, num_heads, seq_len, head_dim]`
/// * `k` - Key   `[batch, 1, seq_len, head_dim]`   (single head)
/// * `v` - Value `[batch, 1, seq_len, head_dim]`   (single head)
/// * `mask` - Additive mask (optional), broadcastable to `[batch, num_heads, seq_q, seq_k]`
/// * `scale` - Custom scale factor (default: 1/sqrt(head_dim))
///
/// K/V are broadcast across all Q heads without allocating expanded copies.
pub fn multi_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor, OnnxError> {
    if q.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "multi_query_attention: Q must be 4D [batch, num_heads, seq_len, head_dim]".into(),
        ));
    }
    if k.ndim() != 4 || v.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "multi_query_attention: K and V must be 4D [batch, 1, seq_len, head_dim]".into(),
        ));
    }
    if k.shape[1] != 1 || v.shape[1] != 1 {
        return Err(OnnxError::ShapeMismatch(format!(
            "multi_query_attention: K/V must have 1 head, got K={} V={}",
            k.shape[1], v.shape[1]
        )));
    }

    // Delegate to GQA with num_kv_heads = 1
    grouped_query_attention(q, k, v, 1, mask, scale)
}

// ── Grouped-Query Attention (GQA) ───────────────────────────────────────────

/// Grouped-query attention: K/V have `num_kv_heads` groups, each shared by
/// `num_heads / num_kv_heads` Q heads.
///
/// # Arguments
/// * `q` - Query `[batch, num_heads, seq_len, head_dim]`
/// * `k` - Key   `[batch, num_kv_heads, seq_len, head_dim]`
/// * `v` - Value `[batch, num_kv_heads, seq_len, head_dim]`
/// * `num_kv_heads` - Number of key/value head groups
/// * `mask` - Additive mask (optional), broadcastable to `[batch, num_heads, seq_q, seq_k]`
/// * `scale` - Custom scale factor (default: 1/sqrt(head_dim))
pub fn grouped_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_kv_heads: usize,
    mask: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor, OnnxError> {
    if q.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "grouped_query_attention: Q must be 4D [batch, num_heads, seq_len, head_dim]".into(),
        ));
    }
    if k.ndim() != 4 || v.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "grouped_query_attention: K and V must be 4D".into(),
        ));
    }

    let batch = q.shape[0];
    let num_heads = q.shape[1];
    let seq_q = q.shape[2];
    let head_dim = q.shape[3];
    let seq_k = k.shape[2];

    if k.shape[1] != num_kv_heads || v.shape[1] != num_kv_heads {
        return Err(OnnxError::ShapeMismatch(format!(
            "grouped_query_attention: K/V head dim mismatch, expected {} got K={} V={}",
            num_kv_heads, k.shape[1], v.shape[1]
        )));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "grouped_query_attention: num_heads ({}) must be divisible by num_kv_heads ({})",
            num_heads, num_kv_heads
        )));
    }

    let heads_per_group = num_heads / num_kv_heads;
    let d_v = v.shape[3];
    let scale_val = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Precompute mask metadata
    let mask_data = mask.map(|m| &m.data[..]);
    let mask_batch_stride = if let Some(m) = mask {
        let m_ndim = m.ndim();
        if m_ndim >= 3 {
            let m_batch: usize = m.shape[..m_ndim.saturating_sub(2)]
                .iter()
                .product::<usize>()
                .max(1);
            if m_batch == 1 {
                0
            } else {
                m.shape[m_ndim - 2] * m.shape[m_ndim - 1]
            }
        } else {
            0
        }
    } else {
        0
    };
    let _mask_sq_sk = mask.map_or(0, |m| {
        let nd = m.ndim();
        if nd >= 2 {
            m.shape[nd - 2] * m.shape[nd - 1]
        } else {
            m.data.len()
        }
    });

    let q_head_stride = seq_q * head_dim;
    let k_head_stride = seq_k * head_dim;
    let v_head_stride = seq_k * d_v;

    let mut output = vec![0.0f32; batch * num_heads * seq_q * d_v];

    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / heads_per_group; // which KV head this Q head maps to

            let q_off = b * num_heads * q_head_stride + h * q_head_stride;
            let k_off = b * num_kv_heads * k_head_stride + kv_h * k_head_stride;
            let v_off = b * num_kv_heads * v_head_stride + kv_h * v_head_stride;
            let o_off = b * num_heads * seq_q * d_v + h * seq_q * d_v;

            let q_slice = &q.data[q_off..q_off + q_head_stride];
            let k_slice = &k.data[k_off..k_off + k_head_stride];

            // scores = Q_h @ K_{kv_h}^T -> [seq_q, seq_k]
            let mut scores = mm_a_bt(q_slice, k_slice, seq_q, head_dim, seq_k);

            // Scale
            for s in scores.iter_mut() {
                *s *= scale_val;
            }

            // Add mask (broadcast over heads dimension via modular indexing)
            if let Some(md) = mask_data {
                // Determine mask offset: support [seq_q, seq_k] or [batch, *, seq_q, seq_k]
                let m_batch_off = if mask_batch_stride == 0 {
                    0
                } else {
                    b * mask_batch_stride
                };
                let m_base = m_batch_off;
                for i in 0..seq_q {
                    for j in 0..seq_k {
                        let m_idx = m_base + i * seq_k + j;
                        if m_idx < md.len() {
                            scores[i * seq_k + j] += md[m_idx];
                        }
                    }
                }
            }

            // Softmax along last dim
            softmax_last_dim(&mut scores, seq_k);

            // output_h = scores @ V_{kv_h} -> [seq_q, d_v]
            let v_slice = &v.data[v_off..v_off + v_head_stride];
            let attn_out = mm(&scores, v_slice, seq_q, seq_k, d_v);
            output[o_off..o_off + seq_q * d_v].copy_from_slice(&attn_out);
        }
    }

    Ok(Tensor::new(output, vec![batch, num_heads, seq_q, d_v]))
}

// ── ALiBi (Attention with Linear Biases) ────────────────────────────────────

/// Attention with Linear Biases (ALiBi).
///
/// Adds position-dependent linear bias to attention scores instead of positional
/// embeddings. The bias for head h at positions (i, j) is:
/// `bias[i][j] = -slope_h * |i - j|` where `slope_h = 2^(-8*h/H)`.
///
/// # Arguments
/// * `q` - Query `[batch, num_heads, seq_len, head_dim]`
/// * `k` - Key   `[batch, num_heads, seq_len, head_dim]`
/// * `v` - Value `[batch, num_heads, seq_len, head_dim]`
/// * `num_heads` - Total number of attention heads (H)
/// * `mask` - Additional additive mask (optional)
pub fn alibi_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor, OnnxError> {
    if q.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "alibi_attention: Q must be 4D [batch, num_heads, seq_len, head_dim]".into(),
        ));
    }
    if k.ndim() != 4 || v.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(
            "alibi_attention: K and V must be 4D".into(),
        ));
    }

    let batch = q.shape[0];
    let q_heads = q.shape[1];
    let seq_q = q.shape[2];
    let head_dim = q.shape[3];
    let seq_k = k.shape[2];
    let d_v = v.shape[3];

    if q_heads != num_heads {
        return Err(OnnxError::ShapeMismatch(format!(
            "alibi_attention: Q has {} heads but num_heads={}",
            q_heads, num_heads
        )));
    }

    let scale_val = 1.0 / (head_dim as f32).sqrt();

    // Compute ALiBi slopes: slope_h = 2^(-8*h/H)
    let slopes: Vec<f32> = (0..num_heads)
        .map(|h| 2.0f32.powf(-8.0 * (h as f32 + 1.0) / num_heads as f32))
        .collect();

    // Precompute mask metadata
    let mask_data = mask.map(|m| &m.data[..]);
    let mask_batch_stride = if let Some(m) = mask {
        let m_ndim = m.ndim();
        if m_ndim >= 3 {
            let m_batch: usize = m.shape[..m_ndim.saturating_sub(2)]
                .iter()
                .product::<usize>()
                .max(1);
            if m_batch == 1 {
                0
            } else {
                m.shape[m_ndim - 2] * m.shape[m_ndim - 1]
            }
        } else {
            0
        }
    } else {
        0
    };

    let q_head_stride = seq_q * head_dim;
    let k_head_stride = seq_k * head_dim;
    let v_head_stride = seq_k * d_v;

    let mut output = vec![0.0f32; batch * num_heads * seq_q * d_v];

    for b in 0..batch {
        #[allow(clippy::needless_range_loop)]
        for h in 0..num_heads {
            let q_off = b * num_heads * q_head_stride + h * q_head_stride;
            let k_off = b * num_heads * k_head_stride + h * k_head_stride;
            let v_off = b * num_heads * v_head_stride + h * v_head_stride;
            let o_off = b * num_heads * seq_q * d_v + h * seq_q * d_v;

            let q_slice = &q.data[q_off..q_off + q_head_stride];
            let k_slice = &k.data[k_off..k_off + k_head_stride];

            // scores = Q_h @ K_h^T -> [seq_q, seq_k]
            let mut scores = mm_a_bt(q_slice, k_slice, seq_q, head_dim, seq_k);

            // Scale
            for s in scores.iter_mut() {
                *s *= scale_val;
            }

            // Add ALiBi bias: -slope_h * |i - j|
            let slope = slopes[h];
            for i in 0..seq_q {
                for j in 0..seq_k {
                    let dist = i.abs_diff(j);
                    scores[i * seq_k + j] -= slope * dist as f32;
                }
            }

            // Add optional mask
            if let Some(md) = mask_data {
                let m_batch_off = if mask_batch_stride == 0 {
                    0
                } else {
                    b * mask_batch_stride
                };
                for i in 0..seq_q {
                    for j in 0..seq_k {
                        let m_idx = m_batch_off + i * seq_k + j;
                        if m_idx < md.len() {
                            scores[i * seq_k + j] += md[m_idx];
                        }
                    }
                }
            }

            // Softmax
            softmax_last_dim(&mut scores, seq_k);

            // output_h = scores @ V_h
            let v_slice = &v.data[v_off..v_off + v_head_stride];
            let attn_out = mm(&scores, v_slice, seq_q, seq_k, d_v);
            output[o_off..o_off + seq_q * d_v].copy_from_slice(&attn_out);
        }
    }

    Ok(Tensor::new(output, vec![batch, num_heads, seq_q, d_v]))
}

// ── Cached Attention ────────────────────────────────────────────────────────

use crate::kv_cache::KvCache;

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

    // Update cache and get full K, V (past + current)
    let (full_k, full_v) = cache.update(layer_idx, k, v).map_err(OnnxError::Internal)?;

    // Compute standard SDPA: Q (new tokens) vs full K/V
    scaled_dot_product_attention(q, &full_k, &full_v, mask, scale)
}

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

    // Reshape [batch, seq, embed] -> [batch, num_heads, seq, head_dim]
    let q_heads = reshape_to_heads(query, batch, new_seq, num_heads, head_dim);
    let k_heads = reshape_to_heads(key, batch, new_seq_k, num_heads, head_dim);
    let v_heads = reshape_to_heads(value, batch, new_seq_k, num_heads, head_dim);

    // Cached SDPA
    let attn_out = cached_attention(&q_heads, &k_heads, &v_heads, cache, layer_idx, mask, scale)?;

    // Reshape back: [batch, num_heads, new_seq, head_dim] -> [batch, new_seq, embed_dim]
    Ok(reshape_from_heads(
        &attn_out, batch, new_seq, num_heads, head_dim,
    ))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: got {a}, expected {b} (diff={})",
            (a - b).abs()
        );
    }

    fn assert_tensor_close(a: &Tensor, b: &Tensor, tol: f32, msg: &str) {
        assert_eq!(a.shape, b.shape, "{msg}: shape mismatch");
        for (i, (&av, &bv)) in a.data.iter().zip(b.data.iter()).enumerate() {
            assert!(
                (av - bv).abs() < tol,
                "{msg} at idx {i}: got {av}, expected {bv} (diff={})",
                (av - bv).abs()
            );
        }
    }

    // ── Standard SDPA tests ────────────────────────────────────────────

    #[test]
    fn test_sdpa_basic() {
        // batch=1, seq_q=2, seq_k=3, d_k=4, d_v=4
        let q = Tensor::new(vec![1.0; 2 * 4], vec![2, 4]);
        let k = Tensor::new(vec![1.0; 3 * 4], vec![3, 4]);
        let v = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![3, 4],
        );

        let out =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should not fail");

        assert_eq!(out.shape, vec![2, 4]);

        // All scores should be equal (uniform Q,K) -> uniform attention -> avg of V
        let expected_avg = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0];
        for s in 0..2 {
            for (d, &expected) in expected_avg.iter().enumerate() {
                let val = out.data[s * 4 + d];
                assert!(
                    (val - expected).abs() < 1e-4,
                    "s={s}, d={d}: got {val}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn test_sdpa_hand_computed_2x2() {
        // 2x2 Q/K/V hand-computed verification
        // Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1]], V = [[1, 2], [3, 4]]
        // scores = Q @ K^T = [[1, 0], [0, 1]], scale = 1/sqrt(2)
        // scaled = [[0.707, 0], [0, 0.707]]
        // softmax row 0: [exp(0.707)/(exp(0.707)+exp(0)), exp(0)/(exp(0.707)+exp(0))]
        let q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let k = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("hand-computed SDPA");

        assert_eq!(out.shape, vec![2, 2]);

        let s = 1.0 / (2.0f32).sqrt(); // ~0.7071
        let e_s = s.exp();
        let e_0 = 1.0f32; // exp(0)
        let p0 = e_s / (e_s + e_0); // weight on V[0] for row 0
        let p1 = e_0 / (e_s + e_0); // weight on V[1] for row 0

        // Row 0 output: p0 * [1,2] + p1 * [3,4]
        assert_close(out.data[0], p0 * 1.0 + p1 * 3.0, 1e-4, "row0 col0");
        assert_close(out.data[1], p0 * 2.0 + p1 * 4.0, 1e-4, "row0 col1");

        // Row 1 output: p1 * [1,2] + p0 * [3,4] (swapped because Q[1]@K^T = [0, s])
        assert_close(out.data[2], p1 * 1.0 + p0 * 3.0, 1e-4, "row1 col0");
        assert_close(out.data[3], p1 * 2.0 + p0 * 4.0, 1e-4, "row1 col1");
    }

    #[test]
    fn test_sdpa_with_mask() {
        let q = Tensor::new(vec![1.0; 2 * 4], vec![2, 4]);
        let k = Tensor::new(vec![1.0; 3 * 4], vec![3, 4]);
        let v = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![3, 4],
        );

        let mask = Tensor::new(vec![0.0, 0.0, -1e9, 0.0, 0.0, -1e9], vec![2, 3]);

        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None)
            .expect("SDPA with mask should not fail");

        assert_eq!(out.shape, vec![2, 4]);
        for s in 0..2 {
            assert!((out.data[s * 4] - 0.5).abs() < 1e-4);
            assert!((out.data[s * 4 + 1] - 0.5).abs() < 1e-4);
            assert!(out.data[s * 4 + 2].abs() < 1e-4);
        }
    }

    #[test]
    fn test_sdpa_batched() {
        let q = Tensor::new(vec![0.5; 2 * 3 * 4], vec![2, 3, 4]);
        let k = Tensor::new(vec![0.5; 2 * 3 * 4], vec![2, 3, 4]);
        let v = Tensor::new(vec![1.0; 2 * 3 * 4], vec![2, 3, 4]);

        let out = scaled_dot_product_attention(&q, &k, &v, None, None)
            .expect("batched SDPA should not fail");

        assert_eq!(out.shape, vec![2, 3, 4]);
        for &val in &out.data {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sdpa_causal_mask() {
        // Verify upper-triangle causal masking
        let seq = 4;
        let d = 2;
        let q = Tensor::new(vec![1.0; seq * d], vec![seq, d]);
        let k = Tensor::new(vec![1.0; seq * d], vec![seq, d]);
        let v_data: Vec<f32> = (0..seq * d).map(|i| (i / d) as f32).collect();
        let v = Tensor::new(v_data, vec![seq, d]);

        // Causal mask: future positions get -inf
        let mut mask_data = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_data[i * seq + j] = -1e9;
            }
        }
        let mask = Tensor::new(mask_data, vec![seq, seq]);

        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("causal SDPA");

        // Row 0 can only attend to position 0 -> output = V[0] = [0, 0]
        assert_close(out.data[0], 0.0, 1e-4, "causal row0 d0");
        assert_close(out.data[1], 0.0, 1e-4, "causal row0 d1");

        // Row 3 can attend to positions 0,1,2,3 -> mixture but with uniform Q/K
        // all allowed positions have equal score, so output = avg(V[0..4])
        let avg = (0.0 + 1.0 + 2.0 + 3.0) / 4.0;
        assert_close(out.data[6], avg, 1e-3, "causal row3 d0");
    }

    // ── MHA tests ──────────────────────────────────────────────────────

    #[allow(clippy::identity_op)]
    #[test]
    fn test_multi_head_attention_no_projection() {
        let q = Tensor::new(vec![1.0; 1 * 2 * 4], vec![1, 2, 4]);
        let k = Tensor::new(vec![1.0; 1 * 2 * 4], vec![1, 2, 4]);
        let v = Tensor::new(vec![1.0; 1 * 2 * 4], vec![1, 2, 4]);

        let out = multi_head_attention(&q, &k, &v, None, None, None, None, None, 2)
            .expect("MHA should not fail");

        assert_eq!(out.shape, vec![1, 2, 4]);
        for &val in &out.data {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mha_4heads_batch2() {
        let batch = 2;
        let seq = 3;
        let embed_dim = 8;
        let num_heads = 4;

        let q = Tensor::new(
            vec![0.3; batch * seq * embed_dim],
            vec![batch, seq, embed_dim],
        );
        let k = q.clone();
        let v = Tensor::new(
            vec![1.0; batch * seq * embed_dim],
            vec![batch, seq, embed_dim],
        );

        let out = multi_head_attention(&q, &k, &v, None, None, None, None, None, num_heads)
            .expect("MHA 4 heads batch 2");

        assert_eq!(out.shape, vec![batch, seq, embed_dim]);
        // Uniform V -> output all 1.0
        for &val in &out.data {
            assert!((val - 1.0).abs() < 1e-4, "MHA batch2: got {val}");
        }
    }

    #[test]
    fn test_multi_head_attention_with_projection() {
        let batch = 1;
        let seq = 2;
        let embed_dim = 4;
        let num_heads = 2;

        let q = Tensor::new(
            vec![0.5; batch * seq * embed_dim],
            vec![batch, seq, embed_dim],
        );
        let k = q.clone();
        let v = q.clone();

        let mut qkv_w = vec![0.0f32; 3 * embed_dim * embed_dim];
        for i in 0..3 * embed_dim {
            qkv_w[i * embed_dim + (i % embed_dim)] = 1.0;
        }
        let qkv_weight = Tensor::new(qkv_w, vec![3 * embed_dim, embed_dim]);

        let mut out_w = vec![0.0f32; embed_dim * embed_dim];
        for i in 0..embed_dim {
            out_w[i * embed_dim + i] = 1.0;
        }
        let out_weight = Tensor::new(out_w, vec![embed_dim, embed_dim]);

        let out = multi_head_attention(
            &q,
            &k,
            &v,
            Some(&qkv_weight),
            None,
            Some(&out_weight),
            None,
            None,
            num_heads,
        )
        .expect("MHA with proj should not fail");

        assert_eq!(out.shape, vec![1, 2, 4]);
        for &val in &out.data {
            assert!((val - 0.5).abs() < 1e-4, "val={val}");
        }
    }

    // ── MQA tests ──────────────────────────────────────────────────────

    #[test]
    fn test_mqa_basic() {
        // 4 Q heads, 1 KV head, batch=1, seq=3, head_dim=4
        let batch = 1;
        let num_heads = 4;
        let seq = 3;
        let hd = 4;

        let q = Tensor::new(
            vec![0.5; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );
        let k = Tensor::new(vec![0.5; batch * 1 * seq * hd], vec![batch, 1, seq, hd]);
        let v = Tensor::new(vec![1.0; batch * 1 * seq * hd], vec![batch, 1, seq, hd]);

        let out = multi_query_attention(&q, &k, &v, None, None).expect("MQA basic");

        assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
        // Uniform V -> all 1.0
        for &val in &out.data {
            assert!((val - 1.0).abs() < 1e-4, "MQA basic: got {val}");
        }
    }

    #[test]
    fn test_mqa_equivalent_to_broadcast() {
        // MQA should produce the same result as manually broadcasting K/V
        // then running standard SDPA for each head
        let batch = 1;
        let num_heads = 4;
        let seq = 2;
        let hd = 3;

        // Non-uniform Q, K, V
        let q_data: Vec<f32> = (0..batch * num_heads * seq * hd)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..batch * 1 * seq * hd)
            .map(|i| (i as f32) * 0.2 + 0.1)
            .collect();
        let v_data: Vec<f32> = (0..batch * 1 * seq * hd)
            .map(|i| (i as f32) * 0.15 + 0.05)
            .collect();

        let q = Tensor::new(q_data.clone(), vec![batch, num_heads, seq, hd]);
        let k = Tensor::new(k_data.clone(), vec![batch, 1, seq, hd]);
        let v = Tensor::new(v_data.clone(), vec![batch, 1, seq, hd]);

        let mqa_out = multi_query_attention(&q, &k, &v, None, None).expect("MQA broadcast test");

        // Manually broadcast K/V and run per-head SDPA
        for b in 0..batch {
            for h in 0..num_heads {
                let q_off = b * num_heads * seq * hd + h * seq * hd;
                let q_head = Tensor::new(q_data[q_off..q_off + seq * hd].to_vec(), vec![seq, hd]);
                // K/V are the same single head for all
                let k_head = Tensor::new(k_data.clone(), vec![seq, hd]);
                let v_head = Tensor::new(v_data.clone(), vec![seq, hd]);

                let ref_out = scaled_dot_product_attention(&q_head, &k_head, &v_head, None, None)
                    .expect("ref SDPA for MQA");

                let o_off = b * num_heads * seq * hd + h * seq * hd;
                for i in 0..seq * hd {
                    assert_close(
                        mqa_out.data[o_off + i],
                        ref_out.data[i],
                        1e-4,
                        &format!("MQA broadcast b={b} h={h} i={i}"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_mqa_error_wrong_kv_heads() {
        let q = Tensor::new(vec![0.0; 1 * 4 * 2 * 3], vec![1, 4, 2, 3]);
        let k = Tensor::new(vec![0.0; 1 * 2 * 2 * 3], vec![1, 2, 2, 3]); // 2 heads, not 1
        let v = Tensor::new(vec![0.0; 1 * 1 * 2 * 3], vec![1, 1, 2, 3]);

        assert!(multi_query_attention(&q, &k, &v, None, None).is_err());
    }

    // ── GQA tests ──────────────────────────────────────────────────────

    #[test]
    fn test_gqa_basic() {
        // 8 Q heads, 2 KV groups
        let batch = 1;
        let num_heads = 8;
        let num_kv = 2;
        let seq = 3;
        let hd = 4;

        let q = Tensor::new(
            vec![0.5; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );
        let k = Tensor::new(
            vec![0.5; batch * num_kv * seq * hd],
            vec![batch, num_kv, seq, hd],
        );
        let v = Tensor::new(
            vec![1.0; batch * num_kv * seq * hd],
            vec![batch, num_kv, seq, hd],
        );

        let out = grouped_query_attention(&q, &k, &v, num_kv, None, None).expect("GQA basic");
        assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
        for &val in &out.data {
            assert!((val - 1.0).abs() < 1e-4, "GQA basic: got {val}");
        }
    }

    #[test]
    fn test_gqa_group_mapping() {
        // Verify that Q heads within the same group use the same KV head
        // 8 Q heads, 2 KV groups => heads 0-3 use KV[0], heads 4-7 use KV[1]
        let batch = 1;
        let num_heads = 8;
        let num_kv = 2;
        let seq = 2;
        let hd = 2;

        let q = Tensor::new(
            vec![1.0; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );

        // Make KV groups distinguishable: group 0 has V=1, group 1 has V=2
        let k_data = vec![1.0; batch * num_kv * seq * hd];
        let mut v_data = vec![0.0f32; batch * num_kv * seq * hd];
        // KV group 0: V = 1.0
        for i in 0..seq * hd {
            v_data[i] = 1.0;
        }
        // KV group 1: V = 2.0
        for i in seq * hd..2 * seq * hd {
            v_data[i] = 2.0;
        }

        let k = Tensor::new(k_data, vec![batch, num_kv, seq, hd]);
        let v = Tensor::new(v_data, vec![batch, num_kv, seq, hd]);

        let out =
            grouped_query_attention(&q, &k, &v, num_kv, None, None).expect("GQA group mapping");

        // Heads 0-3 should output ~1.0 (use KV group 0)
        for h in 0..4 {
            let off = h * seq * hd;
            for i in 0..seq * hd {
                assert_close(
                    out.data[off + i],
                    1.0,
                    1e-4,
                    &format!("GQA head {h} should use KV group 0"),
                );
            }
        }
        // Heads 4-7 should output ~2.0 (use KV group 1)
        for h in 4..8 {
            let off = h * seq * hd;
            for i in 0..seq * hd {
                assert_close(
                    out.data[off + i],
                    2.0,
                    1e-4,
                    &format!("GQA head {h} should use KV group 1"),
                );
            }
        }
    }

    #[test]
    fn test_gqa_degenerates_to_mha() {
        // GQA with num_kv_heads == num_heads should be equivalent to standard attention
        let batch = 1;
        let num_heads = 4;
        let seq = 3;
        let hd = 4;

        let q_data: Vec<f32> = (0..batch * num_heads * seq * hd)
            .map(|i| (i as f32) * 0.05)
            .collect();
        let k_data: Vec<f32> = (0..batch * num_heads * seq * hd)
            .map(|i| (i as f32) * 0.03 + 0.1)
            .collect();
        let v_data: Vec<f32> = (0..batch * num_heads * seq * hd)
            .map(|i| (i as f32) * 0.02 + 0.5)
            .collect();

        let q = Tensor::new(q_data.clone(), vec![batch, num_heads, seq, hd]);
        let k = Tensor::new(k_data.clone(), vec![batch, num_heads, seq, hd]);
        let v = Tensor::new(v_data.clone(), vec![batch, num_heads, seq, hd]);

        // GQA with all heads
        let gqa_out =
            grouped_query_attention(&q, &k, &v, num_heads, None, None).expect("GQA degenerate MHA");

        // Standard SDPA (already in 4D head format)
        let sdpa_out =
            scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA for comparison");

        assert_tensor_close(&gqa_out, &sdpa_out, 1e-4, "GQA == MHA");
    }

    #[test]
    fn test_gqa_degenerates_to_mqa() {
        // GQA with num_kv_heads == 1 should be equivalent to MQA
        let batch = 1;
        let num_heads = 4;
        let seq = 3;
        let hd = 4;

        let q_data: Vec<f32> = (0..batch * num_heads * seq * hd)
            .map(|i| (i as f32) * 0.05)
            .collect();
        let k_data: Vec<f32> = (0..batch * 1 * seq * hd)
            .map(|i| (i as f32) * 0.03 + 0.1)
            .collect();
        let v_data: Vec<f32> = (0..batch * 1 * seq * hd)
            .map(|i| (i as f32) * 0.02 + 0.5)
            .collect();

        let q = Tensor::new(q_data.clone(), vec![batch, num_heads, seq, hd]);
        let k = Tensor::new(k_data.clone(), vec![batch, 1, seq, hd]);
        let v = Tensor::new(v_data.clone(), vec![batch, 1, seq, hd]);

        let gqa_out =
            grouped_query_attention(&q, &k, &v, 1, None, None).expect("GQA degenerate MQA");
        let mqa_out = multi_query_attention(&q, &k, &v, None, None).expect("MQA for comparison");

        assert_tensor_close(&gqa_out, &mqa_out, 1e-6, "GQA(1) == MQA");
    }

    #[test]
    fn test_gqa_error_indivisible() {
        let q = Tensor::new(vec![0.0; 1 * 5 * 2 * 3], vec![1, 5, 2, 3]);
        let k = Tensor::new(vec![0.0; 1 * 3 * 2 * 3], vec![1, 3, 2, 3]);
        let v = Tensor::new(vec![0.0; 1 * 3 * 2 * 3], vec![1, 3, 2, 3]);

        // 5 heads not divisible by 3 KV heads
        assert!(grouped_query_attention(&q, &k, &v, 3, None, None).is_err());
    }

    // ── ALiBi tests ────────────────────────────────────────────────────

    #[test]
    fn test_alibi_slopes() {
        // Verify slopes: slope_h = 2^(-8*(h+1)/H)
        let num_heads = 4;
        let slopes: Vec<f32> = (0..num_heads)
            .map(|h| 2.0f32.powf(-8.0 * (h as f32 + 1.0) / num_heads as f32))
            .collect();

        // H=4: slopes = 2^(-2), 2^(-4), 2^(-6), 2^(-8)
        assert_close(slopes[0], 0.25, 1e-6, "slope 0");
        assert_close(slopes[1], 0.0625, 1e-6, "slope 1");
        assert_close(slopes[2], 0.015625, 1e-6, "slope 2");
        assert_close(slopes[3], 0.00390625, 1e-6, "slope 3");
    }

    #[test]
    fn test_alibi_attention_basic() {
        let batch = 1;
        let num_heads = 2;
        let seq = 4;
        let hd = 4;

        let q = Tensor::new(
            vec![0.5; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );
        let k = Tensor::new(
            vec![0.5; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );
        let v = Tensor::new(
            vec![1.0; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );

        let out = alibi_attention(&q, &k, &v, num_heads, None).expect("ALiBi basic");
        assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);

        // Uniform V -> output is 1.0 regardless of bias (softmax normalises)
        for &val in &out.data {
            assert!((val - 1.0).abs() < 1e-3, "ALiBi basic: got {val}");
        }
    }

    #[test]
    fn test_alibi_bias_pattern() {
        // With distinguishable V, ALiBi should bias toward nearby tokens
        let batch = 1;
        let num_heads = 1;
        let seq = 4;
        let hd = 2;

        // All Q/K the same so base scores are equal; only ALiBi bias differentiates
        let q = Tensor::new(
            vec![1.0; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );
        let k = q.clone();

        // V[i] = i for each seq position
        let mut v_data = vec![0.0f32; batch * num_heads * seq * hd];
        for s in 0..seq {
            for d in 0..hd {
                v_data[s * hd + d] = s as f32;
            }
        }
        let v = Tensor::new(v_data, vec![batch, num_heads, seq, hd]);

        let out = alibi_attention(&q, &k, &v, num_heads, None).expect("ALiBi bias pattern");

        // With ALiBi, position 0 should attend most strongly to position 0
        // Position 3 should attend most strongly to position 3
        // So output[0] should be closer to 0 than output[3] is to 0
        let out_pos0 = out.data[0]; // output for seq pos 0, should be low (biased toward V[0]=0)
        let out_pos3 = out.data[3 * hd]; // output for seq pos 3, should be higher (biased toward V[3]=3)
        assert!(
            out_pos3 > out_pos0,
            "ALiBi: pos3 output ({out_pos3}) should be > pos0 output ({out_pos0})"
        );
    }

    #[test]
    fn test_alibi_with_mask() {
        let batch = 1;
        let num_heads = 2;
        let seq = 3;
        let hd = 2;

        let q = Tensor::new(
            vec![1.0; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );
        let k = q.clone();
        let v = Tensor::new(
            vec![1.0; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );

        // Causal mask
        let mut mask_data = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_data[i * seq + j] = -1e9;
            }
        }
        let mask = Tensor::new(mask_data, vec![seq, seq]);

        let out = alibi_attention(&q, &k, &v, num_heads, Some(&mask)).expect("ALiBi + mask");
        assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
        // Shouldn't produce NaN
        for &val in &out.data {
            assert!(!val.is_nan(), "ALiBi+mask produced NaN");
        }
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_sdpa_seq_len_1() {
        let q = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let k = Tensor::new(vec![3.0, 4.0], vec![1, 2]);
        let v = Tensor::new(vec![5.0, 6.0], vec![1, 2]);

        let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("seq_len=1");
        // Single key -> softmax is just 1.0 -> output = V
        assert_eq!(out.shape, vec![1, 2]);
        assert_close(out.data[0], 5.0, 1e-5, "seq1 d0");
        assert_close(out.data[1], 6.0, 1e-5, "seq1 d1");
    }

    #[test]
    fn test_gqa_head_dim_1() {
        let batch = 1;
        let num_heads = 4;
        let num_kv = 2;
        let seq = 2;
        let hd = 1;

        let q = Tensor::new(
            vec![1.0; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );
        let k = Tensor::new(
            vec![1.0; batch * num_kv * seq * hd],
            vec![batch, num_kv, seq, hd],
        );
        let v = Tensor::new(
            vec![2.0; batch * num_kv * seq * hd],
            vec![batch, num_kv, seq, hd],
        );

        let out = grouped_query_attention(&q, &k, &v, num_kv, None, None).expect("head_dim=1");
        assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
        for &val in &out.data {
            assert_close(val, 2.0, 1e-4, "head_dim=1");
        }
    }

    #[test]
    fn test_mqa_batch_1_seq_1() {
        let q = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 1, 2]);
        let k = Tensor::new(vec![0.5, 0.5], vec![1, 1, 1, 2]);
        let v = Tensor::new(vec![10.0, 20.0], vec![1, 1, 1, 2]);

        let out = multi_query_attention(&q, &k, &v, None, None).expect("batch1 seq1 MQA");
        assert_eq!(out.shape, vec![1, 2, 1, 2]);
        // Single key position -> output = V
        for h in 0..2 {
            let off = h * 1 * 2;
            assert_close(out.data[off], 10.0, 1e-4, "b1s1 V0");
            assert_close(out.data[off + 1], 20.0, 1e-4, "b1s1 V1");
        }
    }

    // ── Numerical stability ────────────────────────────────────────────

    #[test]
    fn test_sdpa_large_values_no_nan() {
        // Test with large values up to 100.0
        let seq = 4;
        let d = 4;
        let data: Vec<f32> = (0..seq * d).map(|i| (i as f32) * 25.0).collect();

        let q = Tensor::new(data.clone(), vec![seq, d]);
        let k = Tensor::new(data.clone(), vec![seq, d]);
        let v = Tensor::new(vec![1.0; seq * d], vec![seq, d]);

        let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("large values SDPA");

        for &val in &out.data {
            assert!(!val.is_nan(), "Large values produced NaN");
            assert!(!val.is_infinite(), "Large values produced Inf");
        }
    }

    #[test]
    fn test_gqa_large_values_no_nan() {
        let batch = 1;
        let num_heads = 4;
        let num_kv = 2;
        let seq = 3;
        let hd = 4;

        let q_data: Vec<f32> = (0..batch * num_heads * seq * hd)
            .map(|i| ((i % 7) as f32) * 14.0 + 2.0)
            .collect();
        let k_data: Vec<f32> = (0..batch * num_kv * seq * hd)
            .map(|i| ((i % 5) as f32) * 20.0)
            .collect();
        let v_data: Vec<f32> = (0..batch * num_kv * seq * hd)
            .map(|i| ((i % 3) as f32) * 33.0 + 1.0)
            .collect();

        let q = Tensor::new(q_data, vec![batch, num_heads, seq, hd]);
        let k = Tensor::new(k_data, vec![batch, num_kv, seq, hd]);
        let v = Tensor::new(v_data, vec![batch, num_kv, seq, hd]);

        let out =
            grouped_query_attention(&q, &k, &v, num_kv, None, None).expect("GQA large values");

        for &val in &out.data {
            assert!(!val.is_nan(), "GQA large values produced NaN");
            assert!(!val.is_infinite(), "GQA large values produced Inf");
        }
    }

    #[test]
    fn test_alibi_large_values_no_nan() {
        let batch = 1;
        let num_heads = 4;
        let seq = 4;
        let hd = 4;

        let data: Vec<f32> = (0..batch * num_heads * seq * hd)
            .map(|i| ((i % 10) as f32) * 10.0)
            .collect();
        let q = Tensor::new(data.clone(), vec![batch, num_heads, seq, hd]);
        let k = Tensor::new(data.clone(), vec![batch, num_heads, seq, hd]);
        let v = Tensor::new(
            vec![1.0; batch * num_heads * seq * hd],
            vec![batch, num_heads, seq, hd],
        );

        let out = alibi_attention(&q, &k, &v, num_heads, None).expect("ALiBi large values");
        for &val in &out.data {
            assert!(!val.is_nan(), "ALiBi large values produced NaN");
            assert!(!val.is_infinite(), "ALiBi large values produced Inf");
        }
    }

    // ── RoPE tests (existing) ──────────────────────────────────────────

    #[test]
    fn test_rotary_embedding_basic() {
        let input = Tensor::new(
            vec![
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            ],
            vec![1, 4, 4],
        );
        let pos_ids = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]);

        let out =
            rotary_embedding(&input, &pos_ids, None, None, 10000.0).expect("RoPE should not fail");

        assert_eq!(out.shape, vec![1, 4, 4]);

        assert!((out.data[0] - 1.0).abs() < 1e-5);
        assert!((out.data[1] - 0.0).abs() < 1e-5);
        assert!((out.data[2] - 0.0).abs() < 1e-5);
        assert!((out.data[3] - 1.0).abs() < 1e-5);

        let p1 = &out.data[4..8];
        assert!(
            (p1[0] - 1.0).abs() > 1e-3 || (p1[2] - 0.0).abs() > 1e-3,
            "RoPE should modify values at position 1"
        );
    }

    #[test]
    fn test_rotary_embedding_preserves_norm() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let pos_ids = Tensor::new(vec![5.0], vec![1, 1]);

        let out = rotary_embedding(&input, &pos_ids, None, None, 10000.0).expect("RoPE norm test");

        let in_norm: f32 = input.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let out_norm: f32 = out.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (in_norm - out_norm).abs() < 1e-4,
            "RoPE should preserve norm: in={in_norm}, out={out_norm}"
        );
    }
}
