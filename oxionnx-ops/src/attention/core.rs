//! Core attention kernels: SDPA, multi-head attention, reshape helpers, and rotary embedding.

use oxionnx_core::{OnnxError, Tensor};

#[cfg(not(target_arch = "wasm32"))]
use super::gemm::should_parallelize;
#[cfg(feature = "simd")]
use super::gemm::SGEMM_MIN_ROWS;
use super::gemm::{matmul_nn_into, matmul_nt_into, matmul_nt_into_par};

// ── Private helpers ──────────────────────────────────────────────────────────

/// Softmax along last dimension for a flat buffer with given inner dimension.
pub(super) fn softmax_last_dim(data: &mut [f32], inner: usize) {
    if inner == 0 {
        return;
    }
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

/// Rows of Q processed per score-matrix tile.
///
/// The kernel materialises only `Q_BLOCK × seq_k` scores at a time instead of
/// the full `seq_q × seq_k` matrix, so scratch memory stays bounded (and each
/// rayon worker's tile stays cache-resident) while `m = Q_BLOCK` is still large
/// enough for `sgemm`'s register blocking to pay off.
const Q_BLOCK: usize = 64;

/// In-place softmax over one score row (`simd` feature routes to the
/// SIMD reduce/normalise kernel; otherwise the scalar reference).
#[inline]
fn softmax_row(row: &mut [f32]) {
    #[cfg(feature = "simd")]
    {
        crate::attention::simd_sdpa::softmax_inplace(row);
    }
    #[cfg(not(feature = "simd"))]
    {
        let inner = row.len();
        softmax_last_dim(row, inner);
    }
}

/// `scores[rows, seq_k] = (q_block[rows, d_k] · k[seq_k, d_k]^T) * scale`.
///
/// Short blocks (`rows < SGEMM_MIN_ROWS`, i.e. autoregressive decode) keep the
/// per-row dot-product kernels — SIMD-dispatched when the feature is on —
/// because `sgemm`'s packing overhead is not worth it for one or two rows.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_scores_block(
    q_block: &[f32],
    k_slice: &[f32],
    scale: f32,
    d_k: usize,
    rows: usize,
    seq_k: usize,
    scores: &mut [f32],
) {
    #[cfg(feature = "simd")]
    if rows < SGEMM_MIN_ROWS {
        for r in 0..rows {
            crate::attention::simd_sdpa::compute_qk_scores(
                &q_block[r * d_k..(r + 1) * d_k],
                k_slice,
                scale,
                d_k,
                seq_k,
                &mut scores[r * seq_k..(r + 1) * seq_k],
            );
        }
        return;
    }
    matmul_nt_into(q_block, k_slice, rows, d_k, seq_k, scores);
    for s in scores[..rows * seq_k].iter_mut() {
        *s *= scale;
    }
}

/// `out[rows, d_v] = probs[rows, seq_k] · v[seq_k, d_v]`.
#[inline]
fn pv_block(
    probs: &[f32],
    v_slice: &[f32],
    rows: usize,
    seq_k: usize,
    d_v: usize,
    out: &mut [f32],
) {
    #[cfg(feature = "simd")]
    if rows < SGEMM_MIN_ROWS {
        for r in 0..rows {
            // `weighted_sum_v` zeroes its output before accumulating.
            crate::attention::simd_sdpa::weighted_sum_v(
                &probs[r * seq_k..(r + 1) * seq_k],
                v_slice,
                d_v,
                seq_k,
                &mut out[r * d_v..(r + 1) * d_v],
            );
        }
        return;
    }
    matmul_nn_into(probs, v_slice, rows, seq_k, d_v, out);
}

// ── Scaled Dot-Product Attention ─────────────────────────────────────────────

/// Whether `lead`'s own flat batch index can be recovered from a broadcast
/// flat index `b` via `b % lead.iter().product()`.
///
/// `sdpa_into`'s kernel loop (`SdpaJob::run_slice`) decomposes one flat index
/// over `lead_bcast` into each operand's own offset with exactly that
/// modulo — which is only correct when, scanning `lead` right-aligned to
/// `lead_bcast` from the outermost axis inward, every axis where `lead`
/// broadcasts (extent 1 while `lead_bcast`'s extent is `> 1`) comes *before*
/// every axis where `lead` matches `lead_bcast` (extent `> 1` on both). Once
/// a "real" (matching, `> 1`) axis has been seen, a later broadcast axis
/// means the operand's true position depends on inner (faster-varying) axes
/// the modulo cannot see — e.g. `lead = [B, 1]` against `lead_bcast = [B, H]`
/// with `H > 1`: `b % B == (i*H + j) % B`, which is not `i` in general, so
/// the kernel would silently read the wrong slice instead of erroring.
///
/// A single-axis (or empty) `lead` is always tileable — there is nothing
/// "after" its one axis to violate the order — which covers every reachable
/// caller of `sdpa_into` today (GQA/MQA build their own expanded K/V in
/// `attention::variants` before ever reaching this kernel). `lead.len() >
/// lead_bcast.len()` is only reachable when [`broadcast_lead_dims`] fell
/// back to its collapsed single-axis scheme (the three shapes are not
/// literally NumPy-broadcastable per axis); that fallback preserves the
/// pre-existing, unvalidated "flatten to one axis" behaviour and is not
/// re-validated here.
fn lead_is_tileable(lead: &[usize], lead_bcast: &[usize]) -> bool {
    if lead.len() > lead_bcast.len() {
        return true;
    }
    let pad = lead_bcast.len() - lead.len();
    let mut seen_real = false;
    for (i, &c) in lead_bcast.iter().enumerate() {
        let l = if i < pad { 1 } else { lead[i - pad] };
        if l == c && c > 1 {
            seen_real = true;
        } else if l == 1 && c > 1 && seen_real {
            return false;
        }
    }
    true
}

/// NumPy-broadcast the leading (batch-like) dimensions of Q, K and V.
///
/// Shared by [`sdpa_output_shape`] and [`sdpa_into`] so the declared output
/// shape and the shape the kernel actually fills always agree by
/// construction: both derive `len`/the output shape's leading part from this
/// same call, so `out_shape.iter().product() == len` unconditionally (see
/// `sdpa_output_shape`'s doc comment). Previously each used only Q's own
/// leading dims, which undersold the data written whenever Q's batch was
/// smaller than K's or V's.
///
/// Falls back to the pre-existing "collapse each operand's own leading dims
/// to a single flat count, then take the max" scheme when the three leading
/// shapes are not literally broadcastable per axis (e.g. an incompatible
/// `[2]` against `[3]`, which no real caller of this crate's SDPA entry
/// points produces). That fallback is exactly the flat batch this kernel
/// already computed before this fix, so it is not tightened further here —
/// see [`lead_is_tileable`] for why it also skips tileability validation.
fn broadcast_lead_dims(q_lead: &[usize], k_lead: &[usize], v_lead: &[usize]) -> Vec<usize> {
    Tensor::broadcast_shape(q_lead, k_lead)
        .and_then(|qk| Tensor::broadcast_shape(&qk, v_lead))
        .unwrap_or_else(|_| {
            let q_batch = q_lead.iter().product::<usize>().max(1);
            let k_batch = k_lead.iter().product::<usize>().max(1);
            let v_batch = v_lead.iter().product::<usize>().max(1);
            vec![q_batch.max(k_batch).max(v_batch)]
        })
}

/// Compute output shape and flat buffer length for SDPA without running the kernel.
///
/// Returns `(out_shape, len)` where `out_shape` is the NumPy broadcast of
/// Q/K/V's leading (batch-like) dims — via [`broadcast_lead_dims`] — plus
/// `[seq_q, d_v]`, and `len == out_shape.iter().product()` always, since both
/// are derived from that same leading shape. This agrees with what
/// [`sdpa_into`] writes and returns by construction, not merely by
/// convention — a caller may size a buffer from this function and always
/// find it exactly matches what `sdpa_into` fills.
///
/// This is infallible and does not check that Q/K/V's leading dims are
/// individually *tileable* against the broadcast result (see
/// [`lead_is_tileable`]) — `sdpa_into` is the actual gate: it rejects a
/// non-tileable combination with a typed error before running the kernel, so
/// a caller that sizes a buffer here and then calls `sdpa_into` never
/// observes a wrong answer, only a `Result`.
pub(crate) fn sdpa_output_shape(q: &Tensor, k: &Tensor, v: &Tensor) -> (Vec<usize>, usize) {
    let q_ndim = q.ndim();
    let seq_q = if q_ndim >= 2 { q.shape[q_ndim - 2] } else { 0 };
    let d_v = if v.ndim() >= 1 {
        v.shape[v.ndim() - 1]
    } else {
        0
    };
    let q_lead = &q.shape[..q_ndim.saturating_sub(2)];
    let k_lead = &k.shape[..k.ndim().saturating_sub(2)];
    let v_lead = &v.shape[..v.ndim().saturating_sub(2)];
    let mut out_shape = broadcast_lead_dims(q_lead, k_lead, v_lead);
    let batch: usize = out_shape.iter().product::<usize>().max(1);
    out_shape.push(seq_q);
    out_shape.push(d_v);
    (out_shape, batch * seq_q * d_v)
}

// ── Additive-mask broadcasting ───────────────────────────────────────────────

/// Upper bound on the number of leading (batch-like) dimensions an SDPA mask may
/// have. Real models use at most two (`batch`, `num_heads`).
const MAX_MASK_LEAD_DIMS: usize = 8;

/// Resolved NumPy-style broadcast plan for an additive attention mask.
///
/// The kernel iterates over a *flattened* batch (`batch × num_heads` for
/// multi-head attention), so a `[B, 1, S_q, S_k]` padding mask must be
/// broadcast over the head axis rather than strided by a single flat step.
/// [`build_mask_plan`] decomposes the flat index back into per-dimension
/// coordinates and derives one stride per leading dimension (`0` where the mask
/// broadcasts), so every `(batch, head)` slice reads the right mask.
struct MaskPlan<'a> {
    data: &'a [f32],
    /// Leading dims of the *output* batch, e.g. `[batch, num_heads]`.
    lead_dims: [usize; MAX_MASK_LEAD_DIMS],
    /// Mask stride (in slices) for each entry of `lead_dims`; `0` = broadcast.
    strides: [usize; MAX_MASK_LEAD_DIMS],
    n_lead: usize,
    /// Elements in one `[rows, cols]` mask slice.
    slice: usize,
    /// Rows in one mask slice: `seq_q`, or `1` when broadcast over queries.
    rows: usize,
    /// Columns in one mask slice: `seq_k`, or `1` when broadcast over keys.
    cols: usize,
}

impl MaskPlan<'_> {
    /// Element offset of the mask slice used by flat batch index `b`.
    #[inline]
    fn batch_offset(&self, b: usize) -> usize {
        let mut rem = b;
        let mut acc = 0usize;
        for i in (0..self.n_lead).rev() {
            let d = self.lead_dims[i];
            acc += (rem % d) * self.strides[i];
            rem /= d;
        }
        acc * self.slice
    }

    /// Add the mask row for query position `q_i` into `scores` (`seq_k` entries).
    #[inline]
    fn add_row(&self, batch_offset: usize, q_i: usize, scores: &mut [f32]) {
        let off = batch_offset + if self.rows == 1 { 0 } else { q_i * self.cols };
        if self.cols == 1 {
            let m = self.data[off];
            for s in scores.iter_mut() {
                *s += m;
            }
        } else {
            for (s, &m) in scores.iter_mut().zip(&self.data[off..off + self.cols]) {
                *s += m;
            }
        }
    }
}

/// Validate an additive mask against the attention shapes and build its
/// broadcast plan.
///
/// Returns [`OnnxError::ShapeMismatch`] for any mask that is not broadcastable
/// to `[lead_dims..., seq_q, seq_k]` — previously such masks were silently
/// dropped for part of the batch.
fn build_mask_plan<'a>(
    mask: &'a Tensor,
    lead_dims: &[usize],
    seq_q: usize,
    seq_k: usize,
) -> Result<MaskPlan<'a>, OnnxError> {
    let m_ndim = mask.ndim();
    let (rows, cols) = match m_ndim {
        0 => (1usize, 1usize),
        1 => (1usize, mask.shape[0]),
        _ => (mask.shape[m_ndim - 2], mask.shape[m_ndim - 1]),
    };
    let mask_lead: &[usize] = if m_ndim >= 2 {
        &mask.shape[..m_ndim - 2]
    } else {
        &[]
    };

    if rows != seq_q && rows != 1 {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: mask shape {:?} is not broadcastable to \
             [..., seq_q={seq_q}, seq_k={seq_k}] (query axis is {rows})",
            mask.shape
        )));
    }
    if cols != seq_k && cols != 1 {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: mask shape {:?} is not broadcastable to \
             [..., seq_q={seq_q}, seq_k={seq_k}] (key axis is {cols})",
            mask.shape
        )));
    }

    let n_lead = lead_dims.len();
    if n_lead > MAX_MASK_LEAD_DIMS {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: at most {MAX_MASK_LEAD_DIMS} leading \
             attention dimensions are supported, got {n_lead}"
        )));
    }
    let lm = mask_lead.len();
    if lm > n_lead && mask_lead[..lm - n_lead].iter().any(|&d| d != 1) {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: mask shape {:?} has more leading \
             dimensions than the attention batch {lead_dims:?}",
            mask.shape
        )));
    }

    let mut dims = [1usize; MAX_MASK_LEAD_DIMS];
    let mut strides = [0usize; MAX_MASK_LEAD_DIMS];
    let mut running = 1usize;
    for i in 0..n_lead {
        let pos = n_lead - 1 - i;
        let out_dim = lead_dims[pos];
        dims[pos] = out_dim;
        let m_dim = if i < lm { mask_lead[lm - 1 - i] } else { 1 };
        if m_dim == out_dim {
            strides[pos] = running;
        } else if m_dim == 1 {
            strides[pos] = 0;
        } else {
            return Err(OnnxError::ShapeMismatch(format!(
                "scaled_dot_product_attention: mask shape {:?} is not broadcastable \
                 to the attention batch {lead_dims:?}",
                mask.shape
            )));
        }
        running = running.saturating_mul(m_dim);
    }

    let slice = rows.saturating_mul(cols);
    let needed = running.saturating_mul(slice);
    if mask.data.len() < needed {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: mask holds {} elements but shape {:?} \
             needs {needed}",
            mask.data.len(),
            mask.shape
        )));
    }

    Ok(MaskPlan {
        data: &mask.data,
        lead_dims: dims,
        strides,
        n_lead,
        slice,
        rows,
        cols,
    })
}

/// Apply the causal (lower-triangular, upper-left aligned) mask to one score row.
///
/// ONNX `Attention-23`: with `is_causal = 1` query `i` may attend to keys
/// `j <= i`; for non-square score matrices the triangle is aligned to the upper
/// left, matching the `flash` kernels and PyTorch's `is_causal`.
#[inline]
fn apply_causal_row(q_i: usize, scores: &mut [f32]) {
    if q_i + 1 < scores.len() {
        for s in scores[q_i + 1..].iter_mut() {
            *s = f32::NEG_INFINITY;
        }
    }
}

// ── Blocked SDPA worker ──────────────────────────────────────────────────────

/// Everything one `(batch, head)` SDPA slice needs, resolved once up front.
///
/// Splitting the kernel out of [`sdpa_into`] lets the serial and rayon drivers
/// share a single implementation (no duplicated loop bodies) and makes the
/// per-worker scratch requirement explicit.
struct SdpaJob<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    mask: Option<MaskPlan<'a>>,
    q_batch: usize,
    k_batch: usize,
    v_batch: usize,
    q_stride: usize,
    k_stride: usize,
    v_stride: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
    is_causal: bool,
}

impl SdpaJob<'_> {
    /// Score-tile scratch each worker needs (`min(seq_q, Q_BLOCK) × seq_k`).
    fn scratch_len(&self) -> usize {
        self.seq_q.min(Q_BLOCK).saturating_mul(self.seq_k)
    }

    /// Multiply-accumulates for one `(batch, head)` slice — the parallelism
    /// threshold input.
    fn macs_per_slice(&self) -> usize {
        self.seq_q
            .saturating_mul(self.seq_k)
            .saturating_mul(self.d_k.saturating_add(self.d_v))
    }

    /// Compute the `seq_q × d_v` output slice for flat batch index `b`.
    fn run_slice(&self, b: usize, scratch: &mut [f32], out: &mut [f32]) {
        // No keys to attend to: softmax over an empty row is undefined, so the
        // output is zero-filled rather than panicking on a 0-length chunk.
        if self.seq_k == 0 || self.d_v == 0 {
            out.fill(0.0);
            return;
        }
        let q_off = (b % self.q_batch) * self.q_stride;
        let k_off = (b % self.k_batch) * self.k_stride;
        let v_off = (b % self.v_batch) * self.v_stride;
        let q_slice = &self.q[q_off..q_off + self.q_stride];
        let k_slice = &self.k[k_off..k_off + self.k_stride];
        let v_slice = &self.v[v_off..v_off + self.v_stride];
        let mask_off = self.mask.as_ref().map(|p| p.batch_offset(b));

        let mut i0 = 0usize;
        while i0 < self.seq_q {
            let rows = (self.seq_q - i0).min(Q_BLOCK);
            let scores = &mut scratch[..rows * self.seq_k];
            qk_scores_block(
                &q_slice[i0 * self.d_k..(i0 + rows) * self.d_k],
                k_slice,
                self.scale,
                self.d_k,
                rows,
                self.seq_k,
                scores,
            );
            for (r, row) in scores.chunks_exact_mut(self.seq_k).enumerate() {
                if let (Some(plan), Some(off)) = (self.mask.as_ref(), mask_off) {
                    plan.add_row(off, i0 + r, row);
                }
                if self.is_causal {
                    apply_causal_row(i0 + r, row);
                }
                softmax_row(row);
            }
            pv_block(
                scores,
                v_slice,
                rows,
                self.seq_k,
                self.d_v,
                &mut out[i0 * self.d_v..(i0 + rows) * self.d_v],
            );
            i0 += rows;
        }
    }

    /// Serial driver over the flattened `batch × num_heads` axis.
    fn run_serial(&self, batch: usize, out: &mut [f32]) {
        let unit = self.seq_q * self.d_v;
        let mut scratch = vec![0.0f32; self.scratch_len()];
        for b in 0..batch {
            self.run_slice(b, &mut scratch, &mut out[b * unit..(b + 1) * unit]);
        }
    }

    /// Rayon driver: every `(batch, head)` pair is independent, so each becomes
    /// one task writing its own `seq_q × d_v` output chunk. Scratch is
    /// allocated per worker (via `for_each_init`), not per slice.
    #[cfg(not(target_arch = "wasm32"))]
    fn run_parallel(&self, batch: usize, out: &mut [f32]) {
        use rayon::prelude::*;
        let unit = self.seq_q * self.d_v;
        let scratch_len = self.scratch_len();
        out[..batch * unit]
            .par_chunks_mut(unit)
            .enumerate()
            .for_each_init(
                || vec![0.0f32; scratch_len],
                |scratch, (b, o_slice)| self.run_slice(b, scratch, o_slice),
            );
    }
}

/// Allocation-free SDPA kernel core: writes results into a pre-sized buffer.
///
/// `out` must be pre-sized to `len` from `sdpa_output_shape`. Returns the output shape.
/// All shape validation and scale/mask/stride logic is identical to
/// `scaled_dot_product_attention`.
///
/// `is_causal` applies the ONNX `Attention` causal mask on top of any explicit
/// additive `mask`.
///
/// # Errors
/// Returns [`OnnxError::ShapeMismatch`] when Q/K/V's leading (batch-like)
/// dims broadcast (per [`broadcast_lead_dims`]) to a shape this kernel's flat
/// batch index cannot correctly tile (see [`lead_is_tileable`]) — this is the
/// authority on whether a Q/K/V combination is actually usable, not just
/// shape-compatible; `sdpa_output_shape` sizes a buffer without checking it.
pub(crate) fn sdpa_into(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
    is_causal: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, OnnxError> {
    let q_ndim = q.ndim();
    if q_ndim < 2 {
        return Err(OnnxError::ShapeMismatch(
            "scaled_dot_product_attention: Q must be at least 2D".to_string(),
        ));
    }
    if k.ndim() < 2 || v.ndim() < 2 {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: K and V must be at least 2D, got {:?} and {:?}",
            k.shape, v.shape
        )));
    }
    let d_k = q.shape[q_ndim - 1];
    let seq_q = q.shape[q_ndim - 2];
    let seq_k = k.shape[k.ndim() - 2];
    let d_v = v.shape[v.ndim() - 1];
    let scale_val = scale.unwrap_or(1.0 / (d_k as f32).sqrt());
    let q_lead = &q.shape[..q_ndim - 2];
    let k_lead = &k.shape[..k.ndim() - 2];
    let v_lead = &v.shape[..v.ndim() - 2];
    let q_batch: usize = q_lead.iter().product::<usize>().max(1);
    let k_batch: usize = k_lead.iter().product::<usize>().max(1);
    let v_batch: usize = v_lead.iter().product::<usize>().max(1);

    // NumPy-broadcast Q/K/V's leading dims — see `broadcast_lead_dims` and
    // `sdpa_output_shape`'s doc comment for why this must be the same
    // computation the shape-only path uses. A combination whose broadcast is
    // not *tileable* (see `lead_is_tileable`) is rejected here, before any
    // stride below is derived from it or any data is read through it.
    let lead_bcast = broadcast_lead_dims(q_lead, k_lead, v_lead);
    if !lead_is_tileable(q_lead, &lead_bcast)
        || !lead_is_tileable(k_lead, &lead_bcast)
        || !lead_is_tileable(v_lead, &lead_bcast)
    {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: Q/K/V leading dims {q_lead:?} / {k_lead:?} / \
             {v_lead:?} broadcast to {lead_bcast:?}, but not in a pattern this kernel's flat \
             batch index supports (a broadcast (size-1) axis of one operand may not be inner to \
             one of that operand's own non-broadcast axes)"
        )));
    }
    let batch: usize = lead_bcast.iter().product::<usize>().max(1);

    // Checked size math: a model-supplied shape must never wrap into a small
    // stride that then lets the slicing below read out of bounds. The stride
    // multiplications themselves are checked — not just the batch × stride
    // totals just below — so an overflow is caught at its actual source
    // instead of silently wrapping (or panicking in debug) first.
    let stride_overflow = |what: &str| {
        OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: {what} stride overflows usize"
        ))
    };
    let q_stride = seq_q.checked_mul(d_k).ok_or_else(|| stride_overflow("Q"))?;
    let k_stride = seq_k.checked_mul(d_k).ok_or_else(|| stride_overflow("K"))?;
    let v_stride = seq_k.checked_mul(d_v).ok_or_else(|| stride_overflow("V"))?;

    // Checked size math: a model-supplied shape must never wrap into a small
    // length that then lets the slicing below read out of bounds.
    let need = |b: usize, stride: usize| -> Result<usize, OnnxError> {
        b.checked_mul(stride).ok_or_else(|| {
            OnnxError::ShapeMismatch(
                "scaled_dot_product_attention: Q/K/V element count overflows usize".to_string(),
            )
        })
    };
    if q.data.len() < need(q_batch, q_stride)?
        || k.data.len() < need(k_batch, k_stride)?
        || v.data.len() < need(v_batch, v_stride)?
    {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: Q/K/V buffers are smaller than their shapes \
             {:?} / {:?} / {:?}",
            q.shape, k.shape, v.shape
        )));
    }

    // Leading dims the flat batch index decomposes into. Prefer whichever of
    // Q/K/V actually spans the loop so masks broadcast per (batch, head).
    let spans_batch = |lead: &[usize], lead_batch: usize| {
        !lead.is_empty() && lead_batch == batch && lead.iter().all(|&d| d > 0)
    };
    let fallback_lead = [batch];
    let lead_dims: &[usize] = if spans_batch(q_lead, q_batch) {
        q_lead
    } else if spans_batch(k_lead, k_batch) {
        k_lead
    } else if spans_batch(v_lead, v_batch) {
        v_lead
    } else {
        &fallback_lead
    };

    let mask_plan = mask
        .map(|m| build_mask_plan(m, lead_dims, seq_q, seq_k))
        .transpose()?;

    // A caller-supplied output buffer that is too small used to panic inside
    // the slicing below; report it as a typed error instead.
    let out_len = batch
        .checked_mul(seq_q)
        .and_then(|n| n.checked_mul(d_v))
        .ok_or_else(|| {
            OnnxError::ShapeMismatch(
                "scaled_dot_product_attention: output element count overflows usize".to_string(),
            )
        })?;
    if out.len() < out_len {
        return Err(OnnxError::ShapeMismatch(format!(
            "scaled_dot_product_attention: output buffer holds {} element(s), needs {out_len}",
            out.len()
        )));
    }

    let job = SdpaJob {
        q: &q.data,
        k: &k.data,
        v: &v.data,
        mask: mask_plan,
        q_batch,
        k_batch,
        v_batch,
        q_stride,
        k_stride,
        v_stride,
        seq_q,
        seq_k,
        d_k,
        d_v,
        scale: scale_val,
        is_causal,
    };

    // `par_chunks_mut(0)` panics, so degenerate slices stay on the serial path
    // (where they are a no-op / zero fill).
    if seq_q != 0 && d_v != 0 {
        #[cfg(not(target_arch = "wasm32"))]
        if should_parallelize(batch, job.macs_per_slice()) {
            job.run_parallel(batch, out);
        } else {
            job.run_serial(batch, out);
        }
        #[cfg(target_arch = "wasm32")]
        job.run_serial(batch, out);
    }

    let mut out_shape = lead_bcast;
    out_shape.push(seq_q);
    out_shape.push(d_v);
    Ok(out_shape)
}

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
    sdpa_causal(q, k, v, mask, scale, false)
}

/// Scaled dot-product attention with the ONNX `Attention` `is_causal` flag.
///
/// With `is_causal = true` query `i` may only attend to keys `j <= i`; the
/// triangle is aligned to the upper left for non-square score matrices, and it
/// is applied *in addition to* any explicit additive `mask`.
pub(crate) fn sdpa_causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
    is_causal: bool,
) -> Result<Tensor, OnnxError> {
    let (out_shape, len) = sdpa_output_shape(q, k, v);
    // `out_shape.iter().product() == len` always — both are derived from the
    // same `broadcast_lead_dims(q_lead, k_lead, v_lead)` leading shape (see
    // `sdpa_output_shape`'s doc comment), so `Tensor::new` below can never
    // observe a length mismatch. A Q/K/V combination whose broadcast is not
    // tileable by the kernel's flat batch index is rejected inside
    // `sdpa_into` itself (see `lead_is_tileable`), before `output` is
    // touched.
    let mut output = vec![0.0f32; len];
    sdpa_into(q, k, v, mask, scale, is_causal, &mut output)?;
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

/// Write `[batch, num_heads, seq, head_dim]` → `[batch, seq, embed_dim]` scatter into `out`.
///
/// Caller must pre-size `out` to `batch * seq * num_heads * head_dim`.
pub(crate) fn reshape_from_heads_into(
    t: &Tensor,
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let embed_dim = num_heads * head_dim;
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
    reshape_from_heads_into(t, batch, seq, num_heads, head_dim, &mut out);
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
/// * `out` - Pre-allocated output buffer, must be sized `batch * seq_q * embed_dim`.
///
/// Returns `[batch, seq_q, embed_dim]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_head_attention_into(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    qkv_weight: Option<&Tensor>,
    qkv_bias: Option<&Tensor>,
    out_proj_weight: Option<&Tensor>,
    out_proj_bias: Option<&Tensor>,
    mask: Option<&Tensor>,
    num_heads: usize,
    out: &mut [f32],
) -> Result<Vec<usize>, OnnxError> {
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
    // --- QKV projection (identical to multi_head_attention) ---
    let (q_proj, k_proj, v_proj) = if let Some(w) = qkv_weight {
        let dim3 = 3 * embed_dim;
        let bias_data = qkv_bias.map(|b| &b.data[..]);
        let mut q_data = vec![0.0f32; batch * seq_q * embed_dim];
        let mut k_data = vec![0.0f32; batch * seq_k * embed_dim];
        let mut v_data = vec![0.0f32; batch * seq_k * embed_dim];
        let w_q = &w.data[..embed_dim * embed_dim];
        let w_k = &w.data[embed_dim * embed_dim..2 * embed_dim * embed_dim];
        let w_v = &w.data[2 * embed_dim * embed_dim..dim3 * embed_dim];
        // Each projection is a plain `[seq, embed] @ [embed, embed]^T` GEMM.
        // The `b_idx` loop is usually a single iteration (batch = 1), so the
        // parallelism has to come from splitting the GEMM's output rows —
        // `matmul_nt_into_par` does exactly that and writes straight into the
        // destination (no `projected` temporary + copy).
        for b_idx in 0..batch {
            let q_off = b_idx * seq_q * embed_dim;
            let q_src = &query.data[q_off..q_off + seq_q * embed_dim];
            matmul_nt_into_par(
                q_src,
                w_q,
                seq_q,
                embed_dim,
                embed_dim,
                &mut q_data[q_off..q_off + seq_q * embed_dim],
            );
        }
        for b_idx in 0..batch {
            let k_off = b_idx * seq_k * embed_dim;
            let k_src = &key.data[k_off..k_off + seq_k * embed_dim];
            matmul_nt_into_par(
                k_src,
                w_k,
                seq_k,
                embed_dim,
                embed_dim,
                &mut k_data[k_off..k_off + seq_k * embed_dim],
            );
        }
        for b_idx in 0..batch {
            let v_off = b_idx * seq_k * embed_dim;
            let v_src = &value.data[v_off..v_off + seq_k * embed_dim];
            matmul_nt_into_par(
                v_src,
                w_v,
                seq_k,
                embed_dim,
                embed_dim,
                &mut v_data[v_off..v_off + seq_k * embed_dim],
            );
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
    // --- SDPA over per-head views ---
    let q_heads = reshape_to_heads(&q_proj, batch, seq_q, num_heads, head_dim);
    let k_heads = reshape_to_heads(&k_proj, batch, seq_k, num_heads, head_dim);
    let v_heads = reshape_to_heads(&v_proj, batch, seq_k, num_heads, head_dim);
    let attn_out = scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask, None)?;
    // --- Final write into out ---
    if let Some(w_out) = out_proj_weight {
        // Build concat (one allocation); project directly into out.
        let concat = reshape_from_heads(&attn_out, batch, seq_q, num_heads, head_dim);
        for b_idx in 0..batch {
            let off = b_idx * seq_q * embed_dim;
            let src = &concat.data[off..off + seq_q * embed_dim];
            let o_slice = &mut out[off..off + seq_q * embed_dim];
            // src[seq_q, embed] @ w_out[embed, embed]^T -> o_slice[seq_q, embed]
            matmul_nt_into_par(src, &w_out.data, seq_q, embed_dim, embed_dim, o_slice);
        }
        if let Some(bias) = out_proj_bias {
            for b_idx in 0..batch {
                for s in 0..seq_q {
                    for d in 0..embed_dim {
                        out[b_idx * seq_q * embed_dim + s * embed_dim + d] += bias.data[d];
                    }
                }
            }
        }
    } else {
        // No out-projection: scatter attn_out directly into out (avoids concat alloc).
        reshape_from_heads_into(&attn_out, batch, seq_q, num_heads, head_dim, out);
    }
    Ok(vec![batch, seq_q, embed_dim])
}

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
    let out_len = batch * seq_q * embed_dim;
    let mut out = vec![0.0f32; out_len];
    let shape = multi_head_attention_into(
        query,
        key,
        value,
        qkv_weight,
        qkv_bias,
        out_proj_weight,
        out_proj_bias,
        mask,
        num_heads,
        &mut out,
    )?;
    Ok(Tensor::new(out, shape))
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
    // `position_ids` must supply at least one full `[.., seq_len]` slice per
    // broadcast batch below; a shorter buffer previously reached
    // `position_ids.data.len() / pos_stride` == 0 and then `b % 0`, an integer
    // divide-by-zero panic. Reject it as a typed error instead. (When
    // `seq_len == 0` the loop that consumes `pos_batch` never runs, so no
    // check is needed in that case.)
    let pos_batch = if seq_len == 0 {
        1
    } else if position_ids.data.len() < pos_stride {
        return Err(OnnxError::ShapeMismatch(format!(
            "rotary_embedding: position_ids has {} element(s), fewer than seq_len={seq_len}",
            position_ids.data.len()
        )));
    } else {
        position_ids.data.len() / pos_stride
    };
    let stride = seq_len * head_dim;
    // A shape whose implied element count exceeds the buffer used to index
    // out of bounds below (a panic on malformed model input).
    let needed = batch_dims.checked_mul(stride).ok_or_else(|| {
        OnnxError::ShapeMismatch(
            "rotary_embedding: input element count overflows usize".to_string(),
        )
    })?;
    if input.data.len() < needed {
        return Err(OnnxError::ShapeMismatch(format!(
            "rotary_embedding: input holds {} element(s) but shape {:?} needs {needed}",
            input.data.len(),
            input.shape
        )));
    }

    // Only the "compute the table" branch owns a `Vec`; supplied caches are
    // *borrowed*. Cloning them cost `2 × max_pos × head_dim/2` floats per RoPE
    // node, per layer, per decoded token — tens of MB of pure memcpy per token
    // for a 32-layer model — and every element was read-only.
    let mut owned_cos: Vec<f32> = Vec::new();
    let mut owned_sin: Vec<f32> = Vec::new();
    if !(cos_cache.is_some() && sin_cache.is_some()) {
        let freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let max_pos = position_ids.data.iter().copied().fold(0.0f32, f32::max) as usize + 1;
        owned_cos = vec![0.0f32; max_pos * half_dim];
        owned_sin = vec![0.0f32; max_pos * half_dim];
        for p in 0..max_pos {
            for (i, &f) in freq.iter().enumerate() {
                let angle = p as f32 * f;
                owned_cos[p * half_dim + i] = angle.cos();
                owned_sin[p * half_dim + i] = angle.sin();
            }
        }
    }
    let (cos_vals, sin_vals): (&[f32], &[f32]) = match (cos_cache, sin_cache) {
        (Some(cc), Some(sc)) => (&cc.data, &sc.data),
        _ => (&owned_cos, &owned_sin),
    };

    // Every element of the first `batch_dims * stride` is written by the loop
    // below (`head_dim` is even, so the two halves cover the whole row), so
    // there is no need to clone the input first — only a trailing region past
    // the shape's extent, if the buffer is oversized, is carried over.
    let mut output = vec![0.0f32; input.data.len()];
    if input.data.len() > needed {
        output[needed..].copy_from_slice(&input.data[needed..]);
    }
    for b in 0..batch_dims {
        for s in 0..seq_len {
            let pos_idx = b % pos_batch;
            let pos = position_ids.data[pos_idx * pos_stride + s] as usize;
            // Bonus hardening alongside the `pos_batch` fix above, same class of
            // bug: an externally supplied `cos_cache`/`sin_cache` (unlike the
            // internally computed table, which is always sized to the largest
            // position actually present) is caller-controlled and may be too
            // short for a malformed model's `position_ids` — guard the row
            // instead of indexing straight into it.
            let row_start = pos.checked_mul(half_dim).filter(|&start| {
                start
                    .checked_add(half_dim)
                    .is_some_and(|end| end <= cos_vals.len() && end <= sin_vals.len())
            });
            let Some(row_start) = row_start else {
                return Err(OnnxError::ShapeMismatch(format!(
                    "rotary_embedding: position id {pos} (batch {b}, seq {s}) is out of range \
                     for the {}-row cos/sin table",
                    cos_vals.len() / half_dim.max(1)
                )));
            };
            let base_idx = b * stride + s * head_dim;
            for i in 0..half_dim {
                let cos_val = cos_vals[row_start + i];
                let sin_val = sin_vals[row_start + i];
                let x0 = input.data[base_idx + i];
                let x1 = input.data[base_idx + half_dim + i];
                output[base_idx + i] = x0 * cos_val - x1 * sin_val;
                output[base_idx + half_dim + i] = x1 * cos_val + x0 * sin_val;
            }
        }
    }
    Ok(Tensor::new(output, input.shape.clone()))
}
