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

use crate::attention::core::qk_scores_block;
use crate::attention::gemm::matmul_nn_into;
#[cfg(not(target_arch = "wasm32"))]
use crate::attention::gemm::should_parallelize;
use crate::attention::scaled_dot_product_attention;

use super::{flash_mask_value, FLASH_DEFAULT_BLOCK_C, FLASH_DEFAULT_BLOCK_R};

// ── Per-(batch, head) working set ────────────────────────────────────────────

/// Shape/stride constants shared by every `(batch, head)` task.
struct FlashDims {
    num_heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    d_v: usize,
    block_r: usize,
    block_c: usize,
    num_blocks_r: usize,
    num_blocks_c: usize,
    scale: f32,
    bh_stride_q: usize,
    bh_stride_k: usize,
    bh_stride_v: usize,
}

/// All scratch buffers a Flash-Attention task needs.
///
/// Previously `s_ij`, `m_ij`, `p_ij` and `l_ij` were `vec![]`-allocated inside
/// the innermost `bc_idx` loop — for a 32-head, 4096-token call with 64×64
/// blocks that is ~524 k heap allocations per attention call.  They are now
/// allocated once per worker and reset per block.
struct FlashScratch {
    /// Running max per Q-block row.
    m_i: Vec<f32>,
    /// Running softmax denominator per Q-block row.
    l_i: Vec<f32>,
    /// Running output accumulator `[br_cap, d_v]`.
    o_i: Vec<f32>,
    /// Block scores `[br_cap, bc_cap]`.
    s_ij: Vec<f32>,
    /// Block max per row.
    m_ij: Vec<f32>,
    /// Block softmax denominator per row.
    l_ij: Vec<f32>,
    /// Block probabilities `[br_cap, bc_cap]` (unnormalised — see module docs).
    p_ij: Vec<f32>,
    /// `P_ij · V_j` product `[br_cap, d_v]`.
    pv: Vec<f32>,
}

impl FlashScratch {
    fn new(br_cap: usize, bc_cap: usize, d_v: usize) -> Self {
        Self {
            m_i: vec![f32::NEG_INFINITY; br_cap],
            l_i: vec![0.0f32; br_cap],
            o_i: vec![0.0f32; br_cap * d_v],
            s_ij: vec![0.0f32; br_cap * bc_cap],
            m_ij: vec![f32::NEG_INFINITY; br_cap],
            l_ij: vec![0.0f32; br_cap],
            p_ij: vec![0.0f32; br_cap * bc_cap],
            pv: vec![0.0f32; br_cap * d_v],
        }
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

    let dims = FlashDims {
        num_heads,
        seq_q,
        seq_k,
        head_dim,
        d_v,
        block_r,
        block_c,
        num_blocks_r,
        num_blocks_c,
        scale,
        bh_stride_q,
        bh_stride_k,
        bh_stride_v,
    };
    let br_cap = block_r.min(seq_q);
    let bc_cap = block_c.min(seq_k);

    // `par_chunks_mut(0)` panics, and `d_v == 0` slips past the `head_dim == 0`
    // early return above — with a zero-length output chunk there is nothing to
    // compute, so skip the drivers entirely.
    if bh_stride_o != 0 {
        #[cfg(not(target_arch = "wasm32"))]
        if should_parallelize(
            batch * num_heads,
            seq_q
                .saturating_mul(seq_k)
                .saturating_mul(head_dim.saturating_add(d_v)),
        ) {
            use rayon::prelude::*;
            // Every (batch, head) pair is fully independent; each becomes one
            // rayon task writing its own `seq_q × d_v` output chunk, with its
            // own reusable scratch (allocated per worker, not per block).
            output
                .par_chunks_mut(bh_stride_o)
                .enumerate()
                .for_each_init(
                    || FlashScratch::new(br_cap, bc_cap, d_v),
                    |sc, (bh, out_slice)| {
                        flash_head(
                            bh / num_heads,
                            bh % num_heads,
                            q,
                            k,
                            v,
                            mask,
                            causal,
                            &dims,
                            sc,
                            out_slice,
                        );
                    },
                );
        } else {
            flash_serial(
                q,
                k,
                v,
                mask,
                causal,
                &dims,
                batch,
                br_cap,
                bc_cap,
                &mut output,
            );
        }
        #[cfg(target_arch = "wasm32")]
        flash_serial(
            q,
            k,
            v,
            mask,
            causal,
            &dims,
            batch,
            br_cap,
            bc_cap,
            &mut output,
        );
    }

    Ok(Tensor::new(output, vec![batch, num_heads, seq_q, d_v]))
}

/// Serial driver: one `FlashScratch`, reused across every `(batch, head)`.
#[allow(clippy::too_many_arguments)]
fn flash_serial(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    dims: &FlashDims,
    batch: usize,
    br_cap: usize,
    bc_cap: usize,
    output: &mut [f32],
) {
    let unit = dims.seq_q * dims.d_v;
    let mut sc = FlashScratch::new(br_cap, bc_cap, dims.d_v);
    for bh in 0..batch * dims.num_heads {
        let out_slice = &mut output[bh * unit..(bh + 1) * unit];
        flash_head(
            bh / dims.num_heads,
            bh % dims.num_heads,
            q,
            k,
            v,
            mask,
            causal,
            dims,
            &mut sc,
            out_slice,
        );
    }
}

/// Flash Attention v2 over one `(batch, head)` pair.
///
/// The block-tiling loop (`br_idx × bc_idx`) and the online-softmax
/// accumulator (`m_i`, `l_i`, `o_i`) are the unchanged Flash Attention v2
/// algebra; only the buffers they live in are now reused across blocks.
#[allow(clippy::too_many_arguments)]
fn flash_head(
    b: usize,
    h: usize,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    dims: &FlashDims,
    sc: &mut FlashScratch,
    out: &mut [f32],
) {
    let d_v = dims.d_v;
    let head_dim = dims.head_dim;
    let bh = b * dims.num_heads + h;
    let q_base = bh * dims.bh_stride_q;
    let k_base = bh * dims.bh_stride_k;
    let v_base = bh * dims.bh_stride_v;

    for br_idx in 0..dims.num_blocks_r {
        let i_start = br_idx * dims.block_r;
        let i_end = (i_start + dims.block_r).min(dims.seq_q);
        let br_size = i_end - i_start;

        // Running statistics per row of the Q block — reset, not reallocated.
        let m_i = &mut sc.m_i[..br_size];
        let l_i = &mut sc.l_i[..br_size];
        let o_i = &mut sc.o_i[..br_size * d_v];
        m_i.fill(f32::NEG_INFINITY);
        l_i.fill(0.0);
        o_i.fill(0.0);

        for bc_idx in 0..dims.num_blocks_c {
            let j_start = bc_idx * dims.block_c;
            let j_end = (j_start + dims.block_c).min(dims.seq_k);
            let bc_size = j_end - j_start;

            // Causal early-exit: skip if entire K block is after Q block
            if causal && j_start > i_end - 1 {
                continue;
            }

            // ── S_ij = Q_i @ K_j^T × scale  [br_size × bc_size] ─────────────
            // `qk_scores_block` runs one `sgemm` for the whole tile (and keeps
            // the SIMD per-row dot products for tiles under 4 rows).
            let s_ij = &mut sc.s_ij[..br_size * bc_size];
            let k_block = &k.data
                [k_base + j_start * head_dim..k_base + j_start * head_dim + bc_size * head_dim];
            let q_block = &q.data[q_base + i_start * head_dim..q_base + i_end * head_dim];
            qk_scores_block(
                q_block, k_block, dims.scale, head_dim, br_size, bc_size, s_ij,
            );

            // ── Apply causal mask ────────────────────────────────────────────
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

            // ── Apply additive mask ──────────────────────────────────────────
            if let Some(m) = mask {
                for ri in 0..br_size {
                    for ci in 0..bc_size {
                        s_ij[ri * bc_size + ci] +=
                            flash_mask_value(m, b, h, i_start + ri, j_start + ci);
                    }
                }
            }

            // ── Online softmax: block statistics ─────────────────────────────
            // NOTE: We intentionally do NOT call `softmax_inplace` here.
            // Flash Attention v2 keeps p_ij = exp(s - m_ij) unnormalized
            // so that partial sums can be blended across blocks.
            //
            // `m_ij` is a max accumulator and `l_ij` a sum accumulator: now
            // that they outlive one block iteration they MUST be reset here.
            let m_ij = &mut sc.m_ij[..br_size];
            let l_ij = &mut sc.l_ij[..br_size];
            m_ij.fill(f32::NEG_INFINITY);
            l_ij.fill(0.0);
            for (ri, m_row) in m_ij.iter_mut().enumerate() {
                for ci in 0..bc_size {
                    let val = s_ij[ri * bc_size + ci];
                    if val > *m_row {
                        *m_row = val;
                    }
                }
            }

            // P_ij = exp(S_ij − m_ij),  l_ij = rowsum(P_ij)
            let p_ij = &mut sc.p_ij[..br_size * bc_size];
            for ri in 0..br_size {
                for ci in 0..bc_size {
                    let val = (s_ij[ri * bc_size + ci] - m_ij[ri]).exp();
                    // Guard: exp(-inf − (-inf)) = exp(NaN) → 0
                    let val = if val.is_nan() { 0.0 } else { val };
                    p_ij[ri * bc_size + ci] = val;
                    l_ij[ri] += val;
                }
            }

            // ── P_ij @ V_j  [br_size × d_v] ─────────────────────────────────
            // The old loop nest walked V with stride `d_v` in its innermost
            // loop (cache-hostile, non-vectorisable). A GEMM over the
            // contiguous V block reads it row-major and reduces over `ci` in
            // exactly the same order.
            let pv = &mut sc.pv[..br_size * d_v];
            let v_block = &v.data[v_base + j_start * d_v..v_base + j_start * d_v + bc_size * d_v];
            matmul_nn_into(p_ij, v_block, br_size, bc_size, d_v, pv);

            // ── Rescale accumulator and update ───────────────────────────────
            for ri in 0..br_size {
                let m_new = m_i[ri].max(m_ij[ri]);

                // Both −∞ ⇒ no valid scores yet; skip
                if m_new == f32::NEG_INFINITY {
                    continue;
                }

                let alpha = {
                    let val = (m_i[ri] - m_new).exp();
                    if val.is_nan() {
                        0.0
                    } else {
                        val
                    }
                };
                let beta = {
                    let val = (m_ij[ri] - m_new).exp();
                    if val.is_nan() {
                        0.0
                    } else {
                        val
                    }
                };

                let l_new = alpha * l_i[ri] + beta * l_ij[ri];

                if l_new > 0.0 {
                    let inv_l = 1.0 / l_new;
                    let w_old = alpha * l_i[ri] * inv_l;
                    let w_new = beta * inv_l;
                    let o_row = &mut o_i[ri * d_v..(ri + 1) * d_v];
                    let pv_row = &pv[ri * d_v..(ri + 1) * d_v];
                    for (o, &p) in o_row.iter_mut().zip(pv_row.iter()) {
                        *o = w_old * *o + w_new * p;
                    }
                }

                m_i[ri] = m_new;
                l_i[ri] = l_new;
            }
        }

        // ── Copy block result to output ─────────────────────────────────────
        out[i_start * d_v..i_end * d_v].copy_from_slice(&o_i[..br_size * d_v]);
    }
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
