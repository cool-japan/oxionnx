//! F16/BF16 typed kernels for scaled dot-product attention and multi-head attention.
//!
//! All softmax accumulators operate in f32 even when inputs are f16/bf16
//! (f16 max ≈ 65504 would overflow during exponential accumulation).
//! Q/K/V are decoded to f32 only for arithmetic; outputs are re-encoded to f16/bf16.

// ── Dim structs ──────────────────────────────────────────────────────────────

/// Dimensions for a single SDPA kernel call.
///
/// `batch_total` = outer_batch × num_heads (both dimensions are treated as a
/// flat loop count — the kernel does not distinguish batch from heads).
pub(crate) struct SdpaDims {
    /// Flat loop count: batch_size × num_heads (or just batch for pure SDPA).
    pub batch_total: usize,
    pub seq_q: usize,
    pub seq_kv: usize,
    pub head_dim: usize,
}

/// Dimensions for a single MHA kernel call.
pub(crate) struct MhaDims {
    pub batch: usize,
    pub seq_q: usize,
    pub seq_kv: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Decode f16 bit slice to f32.
#[inline]
fn f16_to_f32_slice(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

/// Decode bf16 bit slice to f32.
#[inline]
fn bf16_to_f32_slice(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::bf16::from_bits(b).to_f32())
        .collect()
}

/// Encode f32 slice to f16 bits.
#[inline]
fn f32_to_f16_slice(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect()
}

/// Encode f32 slice to bf16 bits.
#[inline]
fn f32_to_bf16_slice(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect()
}

/// Softmax in-place on f32 slice, operating on `inner`-sized chunks along last dim.
fn softmax_f32(data: &mut [f32], inner: usize) {
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

/// Naive [m, k] @ [k, n] → [m, n] in f32.
fn mm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += av * b[p * n + j];
            }
        }
    }
    out
}

/// Naive [m, k] @ [n, k]^T → [m, n] in f32.
fn mm_a_bt_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = s;
        }
    }
    out
}

// ── SDPA inner kernel (operates fully in f32) ────────────────────────────────

/// Core SDPA operating in f32.
///
/// All three input matrices are provided as f32 slices (already decoded from their
/// native dtype by the caller).  Output is written into `out_f32` (pre-allocated,
/// length = batch_total × seq_q × head_dim).
fn sdpa_f32_kernel(
    q_f32: &[f32],
    k_f32: &[f32],
    v_f32: &[f32],
    dims: &SdpaDims,
    mask: Option<(&[f32], bool)>, // (mask_data, is_broadcast) where is_broadcast means stride=0
    scale: f32,
    out_f32: &mut [f32],
) {
    let SdpaDims {
        batch_total,
        seq_q,
        seq_kv,
        head_dim,
    } = *dims;

    let q_stride = seq_q * head_dim;
    let k_stride = seq_kv * head_dim;
    let v_stride = seq_kv * head_dim;
    let out_stride = seq_q * head_dim;
    let mask_stride_per_batch = seq_q * seq_kv;

    for b in 0..batch_total {
        let q_off = b * q_stride;
        let k_off = b * k_stride;
        let v_off = b * v_stride;
        let o_off = b * out_stride;

        let q_slice = &q_f32[q_off..q_off + q_stride];
        let k_slice = &k_f32[k_off..k_off + k_stride];

        // scores: [seq_q, seq_kv]
        let mut scores = mm_a_bt_f32(q_slice, k_slice, seq_q, head_dim, seq_kv);
        for s in scores.iter_mut() {
            *s *= scale;
        }

        // Add mask (f32 additive) — handle broadcast.
        if let Some((mask_data, is_broadcast)) = mask {
            let m_off = if is_broadcast {
                0
            } else {
                b * mask_stride_per_batch
            };
            for (i, sc) in scores.iter_mut().enumerate() {
                let m_idx = m_off + i;
                if m_idx < mask_data.len() {
                    *sc += mask_data[m_idx];
                }
            }
        }

        // Softmax in f32 accumulator.
        softmax_f32(&mut scores, seq_kv);

        // output = softmax_scores @ V, shape [seq_q, head_dim].
        let v_slice = &v_f32[v_off..v_off + v_stride];
        let attn_out = mm_f32(&scores, v_slice, seq_q, seq_kv, head_dim);
        out_f32[o_off..o_off + out_stride].copy_from_slice(&attn_out);
    }
}

// ── Public F16 SDPA kernel ────────────────────────────────────────────────────

/// F16 SDPA kernel. Q/K/V decoded to f32 for arithmetic; softmax accumulates in f32.
/// Output is encoded back to f16 bits.
///
/// `mask` is always f32 (standard ONNX attention mask dtype).
/// `mask_is_broadcast`: when `true`, the mask has a batch dimension of 1 and
/// the same slice is reused for every batch element.
pub(crate) fn scaled_dot_product_attention_f16(
    q_bits: &[u16],
    k_bits: &[u16],
    v_bits: &[u16],
    dims: &SdpaDims,
    mask: Option<(&[f32], bool)>,
    scale: f32,
    out_bits: &mut [u16],
) {
    let q_f32 = f16_to_f32_slice(q_bits);
    let k_f32 = f16_to_f32_slice(k_bits);
    let v_f32 = f16_to_f32_slice(v_bits);

    let out_len = dims.batch_total * dims.seq_q * dims.head_dim;
    let mut out_f32 = vec![0.0f32; out_len];
    sdpa_f32_kernel(&q_f32, &k_f32, &v_f32, dims, mask, scale, &mut out_f32);

    let encoded = f32_to_f16_slice(&out_f32);
    out_bits.copy_from_slice(&encoded);
}

/// BF16 SDPA kernel — same structure as f16 but uses bf16 bit representation.
pub(crate) fn scaled_dot_product_attention_bf16(
    q_bits: &[u16],
    k_bits: &[u16],
    v_bits: &[u16],
    dims: &SdpaDims,
    mask: Option<(&[f32], bool)>,
    scale: f32,
    out_bits: &mut [u16],
) {
    let q_f32 = bf16_to_f32_slice(q_bits);
    let k_f32 = bf16_to_f32_slice(k_bits);
    let v_f32 = bf16_to_f32_slice(v_bits);

    let out_len = dims.batch_total * dims.seq_q * dims.head_dim;
    let mut out_f32 = vec![0.0f32; out_len];
    sdpa_f32_kernel(&q_f32, &k_f32, &v_f32, dims, mask, scale, &mut out_f32);

    let encoded = f32_to_bf16_slice(&out_f32);
    out_bits.copy_from_slice(&encoded);
}

// ── MHA inner kernel (operates in f32) ──────────────────────────────────────

/// Multi-head attention kernel operating in f32.
///
/// Inputs are already decoded from their native dtype (no qkv_weight projection).
/// Performs per-head SDPA then concatenates and applies out_proj.
#[allow(clippy::too_many_arguments)]
fn mha_f32_kernel(
    q_f32: &[f32],
    k_f32: &[f32],
    v_f32: &[f32],
    out_proj_w_f32: &[f32],
    out_proj_b_f32: Option<&[f32]>,
    dims: &MhaDims,
    mask: Option<(&[f32], bool)>,
    scale: f32,
) -> Result<Vec<f32>, String> {
    let MhaDims {
        batch,
        seq_q,
        seq_kv,
        num_heads,
        head_dim,
        embed_dim,
    } = *dims;

    // Reshape Q/K/V from [batch, seq, embed_dim] to [batch, num_heads, seq, head_dim].
    // Input layout: [batch, seq, num_heads * head_dim]
    let mut q_heads = vec![0.0f32; batch * num_heads * seq_q * head_dim];
    let mut k_heads = vec![0.0f32; batch * num_heads * seq_kv * head_dim];
    let mut v_heads = vec![0.0f32; batch * num_heads * seq_kv * head_dim];

    // Reshape Q: [batch, seq_q, embed_dim] → [batch, num_heads, seq_q, head_dim]
    for b in 0..batch {
        for s in 0..seq_q {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src = b * seq_q * embed_dim + s * embed_dim + h * head_dim + d;
                    let dst =
                        b * num_heads * seq_q * head_dim + h * seq_q * head_dim + s * head_dim + d;
                    q_heads[dst] = q_f32[src];
                }
            }
        }
    }

    // Reshape K: [batch, seq_kv, embed_dim] → [batch, num_heads, seq_kv, head_dim]
    for b in 0..batch {
        for s in 0..seq_kv {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src = b * seq_kv * embed_dim + s * embed_dim + h * head_dim + d;
                    let dst = b * num_heads * seq_kv * head_dim
                        + h * seq_kv * head_dim
                        + s * head_dim
                        + d;
                    k_heads[dst] = k_f32[src];
                }
            }
        }
    }

    // Reshape V: [batch, seq_kv, embed_dim] → [batch, num_heads, seq_kv, head_dim]
    for b in 0..batch {
        for s in 0..seq_kv {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src = b * seq_kv * embed_dim + s * embed_dim + h * head_dim + d;
                    let dst = b * num_heads * seq_kv * head_dim
                        + h * seq_kv * head_dim
                        + s * head_dim
                        + d;
                    v_heads[dst] = v_f32[src];
                }
            }
        }
    }

    // Run SDPA for each (batch, head) pair — flat batch_total = batch * num_heads.
    let sdpa_dims = SdpaDims {
        batch_total: batch * num_heads,
        seq_q,
        seq_kv,
        head_dim,
    };
    let out_len = batch * num_heads * seq_q * head_dim;
    let mut sdpa_out_f32 = vec![0.0f32; out_len];
    sdpa_f32_kernel(
        &q_heads,
        &k_heads,
        &v_heads,
        &sdpa_dims,
        mask,
        scale,
        &mut sdpa_out_f32,
    );

    // Reshape SDPA output from [batch, num_heads, seq_q, head_dim] to [batch, seq_q, embed_dim].
    let mut concat = vec![0.0f32; batch * seq_q * embed_dim];
    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq_q {
                for d in 0..head_dim {
                    let src =
                        b * num_heads * seq_q * head_dim + h * seq_q * head_dim + s * head_dim + d;
                    let dst = b * seq_q * embed_dim + s * embed_dim + h * head_dim + d;
                    concat[dst] = sdpa_out_f32[src];
                }
            }
        }
    }

    // Apply out_proj: [batch, seq_q, embed_dim] @ [embed_dim, embed_dim]^T → [batch, seq_q, embed_dim].
    let mut out_data = vec![0.0f32; batch * seq_q * embed_dim];
    for b in 0..batch {
        let off = b * seq_q * embed_dim;
        let src = &concat[off..off + seq_q * embed_dim];
        // out_proj_w shape is [embed_dim, embed_dim] stored row-major; we compute A @ W^T.
        let projected = mm_a_bt_f32(src, out_proj_w_f32, seq_q, embed_dim, embed_dim);
        out_data[off..off + seq_q * embed_dim].copy_from_slice(&projected);
    }

    // Add optional bias.
    if let Some(bias) = out_proj_b_f32 {
        for b in 0..batch {
            for s in 0..seq_q {
                for d in 0..embed_dim {
                    out_data[b * seq_q * embed_dim + s * embed_dim + d] += bias[d];
                }
            }
        }
    }

    Ok(out_data)
}

// ── Public F16 MHA kernel ────────────────────────────────────────────────────

/// F16 MHA kernel. No qkv_weight projection (call site must ensure inputs are
/// already projected or use the default f32 fallback for projected paths).
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_head_attention_f16(
    q_bits: &[u16],
    k_bits: &[u16],
    v_bits: &[u16],
    out_proj_w_bits: &[u16],
    out_proj_b_bits: Option<&[u16]>,
    dims: &MhaDims,
    mask: Option<(&[f32], bool)>,
    scale: f32,
    out_bits: &mut [u16],
) -> Result<(), String> {
    let q_f32 = f16_to_f32_slice(q_bits);
    let k_f32 = f16_to_f32_slice(k_bits);
    let v_f32 = f16_to_f32_slice(v_bits);
    let w_f32 = f16_to_f32_slice(out_proj_w_bits);
    let b_f32: Option<Vec<f32>> = out_proj_b_bits.map(f16_to_f32_slice);

    let out_f32 = mha_f32_kernel(
        &q_f32,
        &k_f32,
        &v_f32,
        &w_f32,
        b_f32.as_deref(),
        dims,
        mask,
        scale,
    )?;

    let encoded = f32_to_f16_slice(&out_f32);
    out_bits.copy_from_slice(&encoded);
    Ok(())
}

/// BF16 MHA kernel — same structure as f16 but uses bf16 bit representation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_head_attention_bf16(
    q_bits: &[u16],
    k_bits: &[u16],
    v_bits: &[u16],
    out_proj_w_bits: &[u16],
    out_proj_b_bits: Option<&[u16]>,
    dims: &MhaDims,
    mask: Option<(&[f32], bool)>,
    scale: f32,
    out_bits: &mut [u16],
) -> Result<(), String> {
    let q_f32 = bf16_to_f32_slice(q_bits);
    let k_f32 = bf16_to_f32_slice(k_bits);
    let v_f32 = bf16_to_f32_slice(v_bits);
    let w_f32 = bf16_to_f32_slice(out_proj_w_bits);
    let b_f32: Option<Vec<f32>> = out_proj_b_bits.map(bf16_to_f32_slice);

    let out_f32 = mha_f32_kernel(
        &q_f32,
        &k_f32,
        &v_f32,
        &w_f32,
        b_f32.as_deref(),
        dims,
        mask,
        scale,
    )?;

    let encoded = f32_to_bf16_slice(&out_f32);
    out_bits.copy_from_slice(&encoded);
    Ok(())
}
