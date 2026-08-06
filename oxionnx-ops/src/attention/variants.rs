//! Attention variants: multi-query, grouped-query, and ALiBi attention.

use super::core::softmax_last_dim;
#[cfg(not(target_arch = "wasm32"))]
use super::gemm::should_parallelize;
use super::gemm::{matmul_nn_into, matmul_nt_into};
use oxionnx_core::{OnnxError, Tensor};

// ── Shared per-head driver ───────────────────────────────────────────────────

/// Shapes describing one `(batch × head)` attention loop.
struct HeadLoop {
    batch: usize,
    num_heads: usize,
    seq_q: usize,
    seq_k: usize,
    d_v: usize,
    /// Multiply-accumulates per head — the parallelism threshold input.
    macs_per_head: usize,
}

/// Run `head_fn` once per `(batch, head)` pair, writing that pair's
/// `seq_q × d_v` output chunk.
///
/// Every pair is independent, so this hands the loop to rayon when the problem
/// is large enough to amortise task dispatch. `head_fn` receives a reusable
/// per-worker `seq_q × seq_k` score scratch buffer, so the score matrix is
/// allocated once per worker instead of once per head.
///
/// `head_fn` must be side-effect free apart from the two buffers it is handed —
/// it is called from multiple threads with different `(b, h)` values.
fn for_each_head<F>(loop_dims: &HeadLoop, output: &mut [f32], head_fn: F)
where
    F: Fn(usize, usize, &mut [f32], &mut [f32]) + Sync + Send,
{
    let HeadLoop {
        batch,
        num_heads,
        seq_q,
        seq_k,
        d_v,
        macs_per_head,
    } = *loop_dims;
    let unit = seq_q * d_v;
    if unit == 0 || batch * num_heads == 0 {
        return;
    }
    let scratch_len = seq_q * seq_k;

    #[cfg(not(target_arch = "wasm32"))]
    if should_parallelize(batch * num_heads, macs_per_head) {
        use rayon::prelude::*;
        output[..batch * num_heads * unit]
            .par_chunks_mut(unit)
            .enumerate()
            .for_each_init(
                || vec![0.0f32; scratch_len],
                |scores, (bh, out_slice)| {
                    head_fn(bh / num_heads, bh % num_heads, scores, out_slice);
                },
            );
        return;
    }
    #[cfg(target_arch = "wasm32")]
    let _ = macs_per_head;

    let mut scores = vec![0.0f32; scratch_len];
    for bh in 0..batch * num_heads {
        let out_slice = &mut output[bh * unit..(bh + 1) * unit];
        head_fn(bh / num_heads, bh % num_heads, &mut scores, out_slice);
    }
}

// ── Multi-Query Attention ────────────────────────────────────────────────────

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
    grouped_query_attention(q, k, v, 1, mask, scale)
}

// ── Grouped-Query Attention ──────────────────────────────────────────────────

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
    let loop_dims = HeadLoop {
        batch,
        num_heads,
        seq_q,
        seq_k,
        d_v,
        macs_per_head: seq_q * seq_k * (head_dim + d_v),
    };
    for_each_head(&loop_dims, &mut output, |b, h, scores, out_slice| {
        let kv_h = h / heads_per_group;
        let q_off = b * num_heads * q_head_stride + h * q_head_stride;
        let k_off = b * num_kv_heads * k_head_stride + kv_h * k_head_stride;
        let v_off = b * num_kv_heads * v_head_stride + kv_h * v_head_stride;
        let q_slice = &q.data[q_off..q_off + q_head_stride];
        let k_slice = &k.data[k_off..k_off + k_head_stride];
        matmul_nt_into(q_slice, k_slice, seq_q, head_dim, seq_k, scores);
        for s in scores.iter_mut() {
            *s *= scale_val;
        }
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
        softmax_last_dim(scores, seq_k);
        let v_slice = &v.data[v_off..v_off + v_head_stride];
        matmul_nn_into(scores, v_slice, seq_q, seq_k, d_v, out_slice);
    });
    Ok(Tensor::new(output, vec![batch, num_heads, seq_q, d_v]))
}

// ── ALiBi Attention ──────────────────────────────────────────────────────────

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
    let slopes: Vec<f32> = (0..num_heads)
        .map(|h| 2.0f32.powf(-8.0 * (h as f32 + 1.0) / num_heads as f32))
        .collect();
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
    let loop_dims = HeadLoop {
        batch,
        num_heads,
        seq_q,
        seq_k,
        d_v,
        macs_per_head: seq_q * seq_k * (head_dim + d_v),
    };
    for_each_head(&loop_dims, &mut output, |b, h, scores, out_slice| {
        let q_off = b * num_heads * q_head_stride + h * q_head_stride;
        let k_off = b * num_heads * k_head_stride + h * k_head_stride;
        let v_off = b * num_heads * v_head_stride + h * v_head_stride;
        let q_slice = &q.data[q_off..q_off + q_head_stride];
        let k_slice = &k.data[k_off..k_off + k_head_stride];
        matmul_nt_into(q_slice, k_slice, seq_q, head_dim, seq_k, scores);
        for s in scores.iter_mut() {
            *s *= scale_val;
        }
        let slope = slopes[h];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let dist = i.abs_diff(j);
                scores[i * seq_k + j] -= slope * dist as f32;
            }
        }
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
        softmax_last_dim(scores, seq_k);
        let v_slice = &v.data[v_off..v_off + v_head_stride];
        matmul_nn_into(scores, v_slice, seq_q, seq_k, d_v, out_slice);
    });
    Ok(Tensor::new(output, vec![batch, num_heads, seq_q, d_v]))
}
