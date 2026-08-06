//! 2D convolution: the specialised, vectorised and parallelised spatial-rank-2
//! kernel.
//!
//! Ranks other than 2 are handled by [`super::conv_nd`], which lowers 1D to
//! this kernel (`[N, C, W]` → `[N, C, 1, W]`) and runs a generic im2col + GEMM
//! for rank ≥ 3.

use oxionnx_core::{OnnxError, Tensor};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use super::im2col::im2col_adaptive;
use super::spatial;
use super::winograd::conv2d_winograd_f2x3;

/// Compute the output shape for a 2D convolution.
///
/// `input_shape` must be `[N, C, H, W]`.
/// `weight_shape` must be `[F, C/group, kH, kW]`.
/// `pads` must be `[pad_top, pad_left, pad_bottom, pad_right]` (length 4).
///
/// Returns a typed [`OnnxError::ShapeMismatch`] — never panics — for any
/// model-derived combination that cannot produce a valid output extent
/// (rank mismatch, zero stride/dilation, padded input below the dilated
/// kernel extent, or a size computation that overflows).
pub(crate) fn compute_conv2d_out_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    dilations: &[usize],
) -> Result<Vec<usize>, OnnxError> {
    if input_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "Conv: input must be 4D [N,C,H,W], got rank {}",
            input_shape.len()
        )));
    }
    if weight_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "Conv: weight must be 4D [F,C/group,kH,kW], got rank {}",
            weight_shape.len()
        )));
    }
    let strides2 = [
        strides.first().copied().unwrap_or(1),
        strides.get(1).copied().unwrap_or(1),
    ];
    let dilations2 = [
        dilations.first().copied().unwrap_or(1),
        dilations.get(1).copied().unwrap_or(1),
    ];
    let pads4 = [
        pads.first().copied().unwrap_or(0),
        pads.get(1).copied().unwrap_or(0),
        pads.get(2).copied().unwrap_or(0),
        pads.get(3).copied().unwrap_or(0),
    ];
    spatial::compute_conv_out_shape(
        "Conv",
        input_shape,
        weight_shape,
        &strides2,
        &pads4,
        &dilations2,
    )
}

/// Whether the Winograd F(2,3) path is expected to beat im2col + SGEMM for a
/// layer of this size.
///
/// Measured on this implementation (`perf_probe_winograd_vs_im2col`, Apple M-series,
/// release build, single thread): Winograd is **slower** than im2col + SGEMM at
/// every realistic CNN layer size, by 1.34× (8 channels, 64×64) up to 2.81×
/// (3→64 channels, 224×224). The cause is structural rather than the filter
/// transform: the accumulation re-streams the whole `oc * c * 16` transformed
/// filter bank once per 2×2 output tile, so its traffic grows as
/// `oc * c * tiles` while GEMM's grows as `oc * c + c * tiles`.
///
/// The path is therefore retained only below `WINOGRAD_MAX_WORK`, where the
/// absolute cost of either algorithm is a few microseconds and the choice is
/// immaterial — keeping the long-standing reference values of the small
/// Winograd fixtures bit-identical — and declined above it, where the measured
/// 1.3–2.8× SGEMM advantage applies. Because the two algorithms differ in
/// rounding (≈1e-5 relative), the threshold sits an order of magnitude above
/// every shape exercised by the test-suite so no existing expectation moves.
const WINOGRAD_MAX_WORK: usize = 4096;

/// Cost proxy for the Winograd path: transformed-filter traffic per image.
fn winograd_work(oc: usize, c: usize, oh: usize, ow: usize) -> usize {
    oc.saturating_mul(c)
        .saturating_mul(oh.div_ceil(2))
        .saturating_mul(ow.div_ceil(2))
}

/// Write conv2d result directly into a pre-allocated output buffer.
///
/// `out_shape` must be the result of `compute_conv2d_out_shape` for these inputs.
/// `out` must have length equal to `out_shape.iter().product()`.
///
/// Degenerate parameters that would divide by zero or index out of range
/// (`group == 0`, a non-4D input/weight, a zero-volume output) leave `out`
/// zero-filled instead of panicking; callers on the model-execution path
/// reject them earlier with a typed error.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_into(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    out: &mut [f32],
    out_shape: &[usize],
) {
    conv2d_into_slices(
        &input.data,
        &input.shape,
        &weight.data,
        &weight.shape,
        bias.map(|b| b.data.as_slice()),
        strides,
        pads,
        dilations,
        group,
        out,
        out_shape,
    );
}

/// Slice-based core of [`conv2d_into`].
///
/// Taking raw slices plus shapes (rather than [`Tensor`]s) lets the rank-1
/// lowering in [`super::conv_nd`] re-interpret a `[N, C, W]` buffer as
/// `[N, C, 1, W]` with no copy, and lets the typed (f16/bf16) wrappers pass
/// their promoted scratch buffers directly.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn conv2d_into_slices(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    out: &mut [f32],
    out_shape: &[usize],
) {
    // Defensive guards: never panic on malformed shapes/attributes.
    if group == 0 || input_shape.len() != 4 || weight_shape.len() != 4 || out_shape.len() != 4 {
        out.fill(0.0_f32);
        return;
    }
    let n = input_shape[0];
    let c_in = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];
    let c_out = weight_shape[0];
    let c_per_group = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];
    let oh = out_shape[2];
    let ow = out_shape[3];

    // Every branch below fully overwrites the first `needed` elements, so the
    // buffer is not pre-zeroed (that memset used to cost a full pass over the
    // largest tensors in a CNN). Only a caller-supplied tail beyond `needed`
    // — which no in-tree caller passes — is cleared, preserving the previous
    // contract exactly.
    let needed: usize = out_shape.iter().product();
    if out.len() < needed || out.is_empty() || oh == 0 || ow == 0 {
        out.fill(0.0_f32);
        return;
    }
    if out.len() > needed {
        out[needed..].fill(0.0_f32);
    }
    // Reject buffers too small for the declared geometry rather than indexing
    // past their end inside the GEMM.
    if input.len() < n * c_in * h * w
        || weight.len() < c_out * c_per_group * kh * kw
        || c_out % group != 0
        || c_per_group * group != c_in
    {
        out.fill(0.0_f32);
        return;
    }

    let c_out_per_group = c_out / group;
    let no_pad = pads == [0, 0, 0, 0];

    // ── 1×1 conv fast path: skip im2col entirely ──────────────────────────
    // When kernel is 1×1, stride=1, no padding, dilation=1: input channels
    // are already contiguous per spatial position → direct matmul.
    if kh == 1 && kw == 1 && strides == [1, 1] && no_pad && dilations == [1, 1] {
        conv2d_1x1_into(
            input,
            weight,
            bias,
            n,
            c_in,
            c_out,
            c_per_group,
            c_out_per_group,
            h,
            w,
            group,
            out,
        );
        return;
    }

    // ── Winograd F(2,3) fast path: 3×3 kernel, stride 1, dilation 1 ──────
    if kh == 3
        && kw == 3
        && strides == [1, 1]
        && dilations == [1, 1]
        && group == 1
        && oh >= 4
        && ow >= 4
        && pads[0] == pads[1]
        && pads[0] == pads[2]
        && pads[0] == pads[3]
        && winograd_work(c_out, c_in, oh, ow) <= WINOGRAD_MAX_WORK
    {
        let pad = pads[0];
        if let Ok(data) = conv2d_winograd_f2x3(input, weight, bias, n, c_in, h, w, c_out, pad) {
            // Length is guaranteed by construction, but this file must stay
            // panic-free even if a future edit changes the extent formula.
            if data.len() == needed {
                out[..needed].copy_from_slice(&data);
                return;
            }
        }
    }

    let col_rows = c_per_group * kh * kw;
    let col_cols = oh * ow;
    let total_jobs = n * group;

    if total_jobs <= 1 {
        // ── Single job: im2col + GEMM straight into the output buffer ────
        //
        // This is the standard batch-1, group-1 inference shape. Both halves
        // are split across rayon when the work justifies it: im2col by input
        // channel (disjoint row bands of the column matrix) and the GEMM by
        // output channel (disjoint row bands of the result), so the arithmetic
        // per element — and therefore the result — is bit-identical to the
        // sequential path.
        let mut col = vec![0.0f32; col_rows * col_cols];
        for batch in 0..n {
            for g in 0..group {
                let in_c_start = g * c_per_group;
                im2col_maybe_parallel(
                    input,
                    c_in,
                    h,
                    w,
                    in_c_start,
                    c_per_group,
                    kh,
                    kw,
                    strides,
                    pads,
                    dilations,
                    oh,
                    ow,
                    batch,
                    &mut col,
                );
                let w_off = g * c_out_per_group * col_rows;
                let o_off = (batch * c_out + g * c_out_per_group) * col_cols;
                sgemm_maybe_parallel(
                    c_out_per_group,
                    col_rows,
                    col_cols,
                    &weight[w_off..],
                    &col,
                    &mut out[o_off..],
                );
                if let Some(b) = bias {
                    for oc in 0..c_out_per_group {
                        let bv = b.get(g * c_out_per_group + oc).copied().unwrap_or(0.0_f32);
                        let start = o_off + oc * col_cols;
                        for j in 0..col_cols {
                            out[start + j] += bv;
                        }
                    }
                }
            }
        }
    } else {
        // ── Parallel path: multiple (batch, group) jobs ─────────────────
        //
        // Each job writes straight into its slice of `out` — no per-job
        // `Vec<f32>` collected and then `copy_from_slice`d back — mirroring
        // the batched-matmul fix in `math_typed::matmul_f32_into`. Job `idx`
        // (`= batch * group + g`) owns `out[idx*job_out_size..][..job_out_size]`:
        // since `c_out == group * c_out_per_group` is already guaranteed by
        // the `c_out % group != 0` guard above, `(batch * c_out + g *
        // c_out_per_group) * col_cols` — the offset the sequential branch
        // above uses — is exactly `idx * job_out_size`, so this is the same
        // destination, just written directly instead of staged through an
        // intermediate buffer.
        //
        // The im2col scratch buffer is amortised across every job one rayon
        // worker handles via `for_each_init` (same pattern as
        // `attention::core::SdpaJob::run_parallel`), rather than allocated
        // fresh per job as before. Reusing a *dirty* scratch buffer across
        // jobs is sound because `im2col_adaptive` (all three of its dispatch
        // targets) unconditionally overwrites every element of its output —
        // including explicit zero-fills for padding — so nothing is ever
        // read from `col` before this job writes it.
        let job_out_size = c_out_per_group * col_cols;
        // `c_out_per_group == 0` (a weight with zero output channels) makes
        // `job_out_size == 0`; `par_chunks_mut(0)` panics, and there is
        // nothing to write in that case anyway (the top-of-function
        // `out[needed..].fill(0.0)` already zeroed everything, since
        // `needed == total_jobs * job_out_size == 0` too).
        if job_out_size > 0 {
            #[cfg(not(target_arch = "wasm32"))]
            out[..total_jobs * job_out_size]
                .par_chunks_mut(job_out_size)
                .enumerate()
                .for_each_init(
                    || vec![0.0f32; col_rows * col_cols],
                    |col_scratch, (idx, dst)| {
                        conv2d_single_job_into(
                            input,
                            weight,
                            bias,
                            strides,
                            pads,
                            dilations,
                            c_in,
                            h,
                            w,
                            c_per_group,
                            c_out_per_group,
                            kh,
                            kw,
                            oh,
                            ow,
                            col_rows,
                            col_cols,
                            idx / group,
                            idx % group,
                            col_scratch,
                            dst,
                        );
                    },
                );

            #[cfg(target_arch = "wasm32")]
            {
                let mut col_scratch = vec![0.0f32; col_rows * col_cols];
                for (idx, dst) in out[..total_jobs * job_out_size]
                    .chunks_mut(job_out_size)
                    .enumerate()
                {
                    conv2d_single_job_into(
                        input,
                        weight,
                        bias,
                        strides,
                        pads,
                        dilations,
                        c_in,
                        h,
                        w,
                        c_per_group,
                        c_out_per_group,
                        kh,
                        kw,
                        oh,
                        ow,
                        col_rows,
                        col_cols,
                        idx / group,
                        idx % group,
                        &mut col_scratch,
                        dst,
                    );
                }
            }
        }
    }
}

/// 2D convolution via im2col + GEMM with Rayon parallelization.
///
/// Each (batch, group) pair is independent and processed in parallel on
/// native targets.  On WASM (single-threaded), falls back to sequential.
///
/// input: [N, C_in, H, W]
/// weight: \[C_out, C_in/group, kH, kW\]
/// bias: \[C_out\] (optional)
///
/// Parameter combinations with no valid output extent (padded input below the
/// dilated kernel, zero stride/dilation, non-4D operands) yield an empty
/// `[0, 0, 0, 0]` tensor rather than a panic. The `Conv` operator on the model
/// path uses `compute_conv2d_out_shape` directly and surfaces the typed error.
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Tensor {
    let Ok(out_shape) =
        compute_conv2d_out_shape(&input.shape, &weight.shape, &strides, &pads, &dilations)
    else {
        return Tensor::new(Vec::new(), vec![0, 0, 0, 0]);
    };
    let out_len: usize = out_shape.iter().product();
    let mut data = vec![0.0_f32; out_len];
    conv2d_into(
        input, weight, bias, strides, pads, dilations, group, &mut data, &out_shape,
    );
    Tensor::new(data, out_shape)
}

/// Minimum `M * K * N` before a GEMM (or an im2col of comparable size) is worth
/// splitting across threads. Shared by the 1×1 path and the single-job path so
/// small convolutions keep their sequential, allocation-free behaviour.
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_GEMM_THRESHOLD: usize = 64 * 64 * 64;

/// Run `im2col_adaptive`, splitting the column matrix by input channel across
/// rayon when the work justifies it.
///
/// Row `r` of the column matrix belongs to input channel `r / (kH * kW)`, so a
/// contiguous band of input channels owns a contiguous band of rows: each
/// thread calls the *same* `im2col_adaptive` with a shifted `in_c_start` and a
/// disjoint sub-slice of `col`. The gather is element-wise, so the result is
/// byte-identical to the sequential fill.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col_maybe_parallel(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    in_c_start: usize,
    c_per_group: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    oh: usize,
    ow: usize,
    batch: usize,
    col: &mut [f32],
) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let col_cols = oh * ow;
        let num_threads = rayon::current_num_threads();
        let work = c_per_group * kh * kw * col_cols;
        if num_threads > 1 && c_per_group >= 2 && work >= PARALLEL_GEMM_THRESHOLD {
            let chunk_ic = c_per_group.div_ceil(num_threads).max(1);
            let chunk_len = chunk_ic * kh * kw * col_cols;
            col.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(t, sub)| {
                    let first = t * chunk_ic;
                    if first >= c_per_group {
                        return;
                    }
                    let count = (c_per_group - first).min(chunk_ic);
                    im2col_adaptive(
                        input,
                        c_in,
                        h,
                        w,
                        in_c_start + first,
                        count,
                        kh,
                        kw,
                        strides,
                        pads,
                        dilations,
                        oh,
                        ow,
                        batch,
                        sub,
                    );
                });
            return;
        }
    }

    im2col_adaptive(
        input,
        c_in,
        h,
        w,
        in_c_start,
        c_per_group,
        kh,
        kw,
        strides,
        pads,
        dilations,
        oh,
        ow,
        batch,
        col,
    );
}

/// `C = A × B` with `beta = 0`, split by rows of `A` across rayon when large.
#[inline]
#[allow(unsafe_code)]
pub(crate) fn sgemm_maybe_parallel(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let num_threads = rayon::current_num_threads();
        if m.saturating_mul(k).saturating_mul(n) >= PARALLEL_GEMM_THRESHOLD && m >= num_threads * 2
        {
            parallel_sgemm(m, k, n, a, b, c);
            return;
        }
    }
    // SAFETY: `a` is [m, k], `b` is [k, n] and `c` is at least [m, n], all
    // row-major and checked by the caller; the row/column strides below match
    // that layout exactly.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// Process one (batch, group) slice: im2col → sgemm → bias, writing directly
/// into `dst` (`[c_out_per_group, col_cols]`, row-major) instead of
/// allocating and returning a fresh `Vec<f32>`.
///
/// `col_scratch` must have length `>= col_rows * col_cols`; it is reused
/// across calls by the caller (one buffer per rayon worker via
/// `for_each_init`, or one buffer for the whole wasm32 loop) rather than
/// allocated fresh per job. Safe to reuse dirty, since `im2col_adaptive`
/// unconditionally overwrites every element it uses of `col_scratch`,
/// including explicit zero-fills for padding.
#[allow(clippy::too_many_arguments)]
fn conv2d_single_job_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    c_in: usize,
    h: usize,
    w: usize,
    c_per_group: usize,
    c_out_per_group: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    col_rows: usize,
    col_cols: usize,
    batch: usize,
    g: usize,
    col_scratch: &mut [f32],
    dst: &mut [f32],
) {
    let col = &mut col_scratch[..col_rows * col_cols];
    let in_c_start = g * c_per_group;
    im2col_adaptive(
        input,
        c_in,
        h,
        w,
        in_c_start,
        c_per_group,
        kh,
        kw,
        strides,
        pads,
        dilations,
        oh,
        ow,
        batch,
        col,
    );

    let w_off = g * c_out_per_group * col_rows;

    // Already inside a rayon job — keep the inner GEMM sequential.
    // SAFETY: same layout contract as `sgemm_maybe_parallel`; `dst` is at
    // least `c_out_per_group * col_cols` long (the caller's `par_chunks_mut`
    // / `chunks_mut` chunk size), matching `[c_out_per_group, col_cols]`.
    #[allow(unsafe_code)]
    unsafe {
        matrixmultiply::sgemm(
            c_out_per_group,
            col_rows,
            col_cols,
            1.0,
            weight[w_off..].as_ptr(),
            col_rows as isize,
            1,
            col.as_ptr(),
            col_cols as isize,
            1,
            0.0,
            dst.as_mut_ptr(),
            col_cols as isize,
            1,
        );
    }

    if let Some(b) = bias {
        for oc in 0..c_out_per_group {
            let bias_val = b.get(g * c_out_per_group + oc).copied().unwrap_or(0.0_f32);
            let row_start = oc * col_cols;
            for j in 0..col_cols {
                dst[row_start + j] += bias_val;
            }
        }
    }
}

/// 1×1 convolution fast path, writing directly into a pre-allocated buffer.
///
/// For 1×1 kernel with stride=1, no padding, dilation=1:
///   input  [N, C_in, H, W]  → treat as [N, C_in, H*W]
///   weight [C_out, C_in/g, 1, 1] → treat as [C_out/g, C_in/g]
///   output = weight × input_slice (matmul, no copy)
///
/// This saves allocating and filling the im2col column matrix.
#[allow(clippy::too_many_arguments)]
fn conv2d_1x1_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    c_in: usize,
    c_out: usize,
    c_per_group: usize,
    c_out_per_group: usize,
    h: usize,
    w: usize,
    group: usize,
    out: &mut [f32],
) {
    let spatial = h * w;

    for batch in 0..n {
        for g in 0..group {
            let in_c_start = g * c_per_group;
            let in_off = (batch * c_in + in_c_start) * spatial;
            let w_off = g * c_out_per_group * c_per_group;
            let o_off = (batch * c_out + g * c_out_per_group) * spatial;

            sgemm_maybe_parallel(
                c_out_per_group,
                c_per_group,
                spatial,
                &weight[w_off..],
                &input[in_off..],
                &mut out[o_off..],
            );

            if let Some(b) = bias {
                for oc in 0..c_out_per_group {
                    let bv = b.get(g * c_out_per_group + oc).copied().unwrap_or(0.0_f32);
                    let start = o_off + oc * spatial;
                    for j in 0..spatial {
                        out[start + j] += bv;
                    }
                }
            }
        }
    }
}

/// Split a large sgemm (C = A × B) by rows of A across rayon threads.
/// A: [m, k] row-major, B: [k, n] row-major, C: [m, n] row-major.
///
/// Each thread runs the *same* `matrixmultiply::sgemm` over a disjoint row
/// band, and the K-blocking (hence the accumulation order of every output
/// element) does not depend on `m`, so the result is bit-identical to one
/// sequential call — asserted by `parallel_sgemm_is_bitwise_identical`.
#[cfg(not(target_arch = "wasm32"))]
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
pub(crate) fn parallel_sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let num_threads = rayon::current_num_threads();
    let chunk = m.div_ceil(num_threads.max(1)).max(1);

    // `par_chunks_mut` hands each thread a disjoint, correctly-sized row band
    // of C, so the tiles are written in place — no scratch allocation and no
    // second pass over the result.
    c[..m * n]
        .par_chunks_mut(chunk * n)
        .enumerate()
        .for_each(|(t, tile)| {
            let row_start = t * chunk;
            if row_start >= m {
                return;
            }
            let tile_m = (m - row_start).min(chunk);
            // SAFETY: `a` is [m, k] row-major so row `row_start` starts at
            // `row_start * k`; `tile` is exactly `tile_m * n` elements, the
            // last chunk being the short one.
            unsafe {
                matrixmultiply::sgemm(
                    tile_m,
                    k,
                    n,
                    1.0,
                    a[row_start * k..].as_ptr(),
                    k as isize,
                    1,
                    b.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    tile.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        });
}
