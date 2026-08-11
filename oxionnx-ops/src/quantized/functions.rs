//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxionnx_core::{OnnxError, Tensor};

use super::types::QuantizedTensor;

// ── W2-perf-matmul (a6-19): cache-friendly row kernels ──────────────────────
//
// `quantized_matmul`/`fully_quantized_matmul` used an i-j-k loop order, so
// the innermost reduction strided B by `n` (a new cache line every
// iteration) and the accumulator's serial `+=` chain blocked vectorisation.
// Reordering to i-p-j (`p` — the reduction axis — in the middle, `j` —
// contiguous in B and in the output row — innermost) walks both B and the
// output row contiguously and gives LLVM a vectorisable inner loop, with
// *no change to the accumulation order for any individual output element*:
// for a fixed `(i,j)`, both loop orders still sum the `p = 0..k` terms in
// increasing `p` order — i-p-j only changes which *other* accumulators get
// touched in between, and those are independent memory locations. The f32
// per-tensor/per-channel kernels are therefore bit-identical to the
// pre-reorder code (verified in `oxionnx-ops/tests/w2_perf_matmul.rs` against
// numpy-derived values, and cross-checked bit-for-bit against a standalone
// i-j-k reimplementation); the i32 kernels are exactly identical for the
// same reason, and doubly so since integer addition is
// associative/commutative even under wraparound.
//
// Per-column scale/zero-point are precomputed once per call (not once per
// row, and not once per `(row, col)` as the original per-channel loop
// implicitly repeated across every `p`), matching the brief's "hoist the
// per-column scale/zero-point out of the p loop". The per-tensor scale/zp
// are already loop-invariant scalars and need no such hoist.
//
// Row-blocks are parallelised over `i` with rayon once there are enough rows
// to amortise the dispatch (`m >= 4`, matching the threshold used for batch
// parallelism in `math_typed`/`math::matmul`) — at `m == 1` (the decode-phase
// shape) there is nothing to parallelise across, only within a row, which
// rayon's row-level split cannot help with anyway.

/// Quantized matrix multiplication: A (f32) x B (i8) -> C (f32).
///
/// This is the most common pattern in quantized inference:
/// activations remain in f32, weights are quantized to i8.
/// Accumulation is done in f32 after dequantizing i8 values.
///
/// A: \[M, K\] f32 activations
/// B: QuantizedTensor \[K, N\] i8 weights
/// Returns: \[M, N\] f32 output
pub fn quantized_matmul(a: &Tensor, b: &QuantizedTensor) -> Result<Tensor, OnnxError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(OnnxError::ShapeMismatch(
            "quantized_matmul: expected 2D tensors".into(),
        ));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    if k != b.shape[0] {
        return Err(OnnxError::ShapeMismatch(format!(
            "quantized_matmul: inner dims mismatch: {} vs {}",
            k, b.shape[0]
        )));
    }
    let mut out = vec![0.0f32; m * n];
    let a_data = &a.data;
    let b_data = &b.data;
    if !b.params.per_channel {
        let scale = b.params.scale[0];
        let zp = b.params.zero_point[0] as i32;
        let compute = |i: usize, out_row: &mut [f32]| {
            per_tensor_row(&a_data[i * k..(i + 1) * k], b_data, out_row, k, scale, zp);
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if m >= 4 {
            use rayon::prelude::*;
            out.par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| compute(i, row));
        } else {
            for (i, row) in out.chunks_mut(n).enumerate() {
                compute(i, row);
            }
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        for (i, row) in out.chunks_mut(n).enumerate() {
            compute(i, row);
        }
    } else {
        let ch_scale: Vec<f32> = (0..n)
            .map(|j| b.params.scale.get(j).copied().unwrap_or(1.0))
            .collect();
        let ch_zp: Vec<i32> = (0..n)
            .map(|j| b.params.zero_point.get(j).map(|&z| z as i32).unwrap_or(0))
            .collect();
        let compute = |i: usize, out_row: &mut [f32]| {
            per_channel_row(
                &a_data[i * k..(i + 1) * k],
                b_data,
                out_row,
                k,
                &ch_scale,
                &ch_zp,
            );
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if m >= 4 {
            use rayon::prelude::*;
            out.par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| compute(i, row));
        } else {
            for (i, row) in out.chunks_mut(n).enumerate() {
                compute(i, row);
            }
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        for (i, row) in out.chunks_mut(n).enumerate() {
            compute(i, row);
        }
    }
    Ok(Tensor::new(out, vec![m, n]))
}

/// i-p-j per-tensor-scale row kernel: walks `b_data` and `out_row`
/// contiguously (stride 1) instead of striding `b_data` by `n`.
#[inline]
fn per_tensor_row(
    a_row: &[f32],
    b_data: &[i8],
    out_row: &mut [f32],
    k: usize,
    scale: f32,
    zp: i32,
) {
    let n = out_row.len();
    for (p, &av) in a_row.iter().enumerate().take(k) {
        let b_row = &b_data[p * n..(p + 1) * n];
        for (o, &bq) in out_row.iter_mut().zip(b_row.iter()) {
            *o += av * ((bq as i32 - zp) as f32 * scale);
        }
    }
}

/// i-p-j per-channel row kernel: same access pattern as [`per_tensor_row`],
/// with `ch_scale[j]`/`ch_zp[j]` (precomputed once per call, not per row or
/// per element) indexed alongside `j` in the innermost loop.
#[inline]
fn per_channel_row(
    a_row: &[f32],
    b_data: &[i8],
    out_row: &mut [f32],
    k: usize,
    ch_scale: &[f32],
    ch_zp: &[i32],
) {
    let n = out_row.len();
    for (p, &av) in a_row.iter().enumerate().take(k) {
        let b_row = &b_data[p * n..(p + 1) * n];
        for (((o, &bq), &cs), &cz) in out_row
            .iter_mut()
            .zip(b_row.iter())
            .zip(ch_scale.iter())
            .zip(ch_zp.iter())
        {
            *o += av * ((bq as i32 - cz) as f32 * cs);
        }
    }
}
/// i-p-j row kernel for the zero-zero-point fast path.
///
/// `row_acc[j] = Σ_p a_row[p] * b[p,j]`, accumulated in exact i32 (matching
/// the original `acc: i32` per `(i,j)` — integer addition is associative
/// under wraparound, so even the overflow behaviour is identical regardless
/// of accumulation order), then converted to `f32 * output_scale` once per
/// element after the full sum is known — never once per partial product, so
/// this is not just numerically equivalent but performs the exact same
/// float conversions as the pre-reorder code, just in `(i,p,j)` order.
///
/// `row_acc` is caller-provided scratch (length `n`) so the sequential path
/// can reuse one buffer across all `m` rows instead of allocating per row.
#[inline]
fn fully_quantized_row_fast(
    a_row: &[i8],
    b_data: &[i8],
    out_row: &mut [f32],
    row_acc: &mut [i32],
    k: usize,
    output_scale: f32,
) {
    row_acc.iter_mut().for_each(|v| *v = 0);
    let n = row_acc.len();
    for (p, &av8) in a_row.iter().enumerate().take(k) {
        let av = av8 as i32;
        let b_row = &b_data[p * n..(p + 1) * n];
        for (acc, &bq) in row_acc.iter_mut().zip(b_row.iter()) {
            *acc += av * bq as i32;
        }
    }
    for (o, &acc) in out_row.iter_mut().zip(row_acc.iter()) {
        *o = acc as f32 * output_scale;
    }
}

/// Sequential pass over every row for [`fully_quantized_row_fast`], reusing
/// one `row_acc` scratch buffer across all `m` rows. Used directly when
/// there's too little row-level work to amortise rayon's dispatch, and
/// unconditionally on a serial wasm32 build (where rayon is not a
/// dependency — see the crate's `wasm-threads` feature).
fn fully_quantized_sequential_fast(
    a_data: &[i8],
    b_data: &[i8],
    out: &mut [f32],
    k: usize,
    n: usize,
    output_scale: f32,
) {
    let mut row_acc = vec![0i32; n];
    for (i, row) in out.chunks_mut(n).enumerate() {
        fully_quantized_row_fast(
            &a_data[i * k..(i + 1) * k],
            b_data,
            row,
            &mut row_acc,
            k,
            output_scale,
        );
    }
}

/// i-p-j row kernel for the general (non-zero zero-point) path — same access
/// pattern as [`fully_quantized_row_fast`], with the zero-point correction
/// (`- a_zp*col_sum_b[j] - b_zp*row_sum_a[i] + k_zp_product`) applied once
/// per element in the final conversion pass, exactly matching the original
/// per-`(i,j)` formula and evaluation order.
#[allow(clippy::too_many_arguments)]
#[inline]
fn fully_quantized_row_corrected(
    a_row: &[i8],
    b_data: &[i8],
    out_row: &mut [f32],
    row_acc: &mut [i32],
    k: usize,
    col_sum_b: &[i32],
    a_zp: i32,
    b_zp: i32,
    row_sum_a_i: i32,
    k_zp_product: i32,
    output_scale: f32,
) {
    row_acc.iter_mut().for_each(|v| *v = 0);
    let n = row_acc.len();
    for (p, &av8) in a_row.iter().enumerate().take(k) {
        let av = av8 as i32;
        let b_row = &b_data[p * n..(p + 1) * n];
        for (acc, &bq) in row_acc.iter_mut().zip(b_row.iter()) {
            *acc += av * bq as i32;
        }
    }
    for ((o, &raw), &cs) in out_row.iter_mut().zip(row_acc.iter()).zip(col_sum_b.iter()) {
        let corrected = raw - a_zp * cs - b_zp * row_sum_a_i + k_zp_product;
        *o = corrected as f32 * output_scale;
    }
}

/// Sequential pass over every row for [`fully_quantized_row_corrected`],
/// reusing one `row_acc` scratch buffer across all `m` rows. See
/// [`fully_quantized_sequential_fast`] for why this is a separate function
/// rather than inline at both call sites.
#[allow(clippy::too_many_arguments)]
fn fully_quantized_sequential_corrected(
    a_data: &[i8],
    b_data: &[i8],
    out: &mut [f32],
    k: usize,
    n: usize,
    col_sum_b: &[i32],
    row_sum_a: &[i32],
    a_zp: i32,
    b_zp: i32,
    k_zp_product: i32,
    output_scale: f32,
) {
    let mut row_acc = vec![0i32; n];
    for (i, row) in out.chunks_mut(n).enumerate() {
        fully_quantized_row_corrected(
            &a_data[i * k..(i + 1) * k],
            b_data,
            row,
            &mut row_acc,
            k,
            col_sum_b,
            a_zp,
            b_zp,
            row_sum_a[i],
            k_zp_product,
            output_scale,
        );
    }
}

/// Fully quantized matmul: A (i8) x B (i8) -> C (f32).
///
/// Both inputs are quantized. Uses optimized integer arithmetic with
/// precomputed row/column sums to handle non-zero zero points efficiently.
///
/// Mathematical decomposition:
///   `C[i][j] = scale_a * scale_b * Σ_k (A_q[i][k] - zp_a) * (B_q[k][j] - zp_b)`
///   `= scale_a * scale_b * (A_q@B_q - zp_a*colsum(B_q) - zp_b*rowsum(A_q) + K*zp_a*zp_b)[i][j]`
pub fn fully_quantized_matmul(
    a: &QuantizedTensor,
    b: &QuantizedTensor,
) -> Result<Tensor, OnnxError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(OnnxError::ShapeMismatch(
            "fully_quantized_matmul: expected 2D".into(),
        ));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    if k != b.shape[0] {
        return Err(OnnxError::ShapeMismatch(format!(
            "K mismatch: {} vs {}",
            k, b.shape[0]
        )));
    }
    let a_scale = a.params.scale[0];
    let a_zp = a.params.zero_point[0] as i32;
    let b_scale = b.params.scale[0];
    let b_zp = b.params.zero_point[0] as i32;
    let output_scale = a_scale * b_scale;
    let a_data = &a.data;
    let b_data = &b.data;

    if a_zp == 0 && b_zp == 0 {
        let mut out = vec![0.0f32; m * n];
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if m >= 4 {
            use rayon::prelude::*;
            out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                let mut row_acc = vec![0i32; n];
                fully_quantized_row_fast(
                    &a_data[i * k..(i + 1) * k],
                    b_data,
                    row,
                    &mut row_acc,
                    k,
                    output_scale,
                );
            });
        } else {
            fully_quantized_sequential_fast(a_data, b_data, &mut out, k, n, output_scale);
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        fully_quantized_sequential_fast(a_data, b_data, &mut out, k, n, output_scale);
        return Ok(Tensor::new(out, vec![m, n]));
    }

    let row_sum_a: Vec<i32> = (0..m)
        .map(|i| a_data[i * k..(i + 1) * k].iter().map(|&v| v as i32).sum())
        .collect();
    let mut col_sum_b = vec![0i32; n];
    for p in 0..k {
        for (j, cs) in col_sum_b.iter_mut().enumerate() {
            *cs += b_data[p * n + j] as i32;
        }
    }
    let k_zp_product = k as i32 * a_zp * b_zp;
    let mut out = vec![0.0f32; m * n];

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if m >= 4 {
        use rayon::prelude::*;
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let mut row_acc = vec![0i32; n];
            fully_quantized_row_corrected(
                &a_data[i * k..(i + 1) * k],
                b_data,
                row,
                &mut row_acc,
                k,
                &col_sum_b,
                a_zp,
                b_zp,
                row_sum_a[i],
                k_zp_product,
                output_scale,
            );
        });
    } else {
        fully_quantized_sequential_corrected(
            a_data,
            b_data,
            &mut out,
            k,
            n,
            &col_sum_b,
            &row_sum_a,
            a_zp,
            b_zp,
            k_zp_product,
            output_scale,
        );
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    fully_quantized_sequential_corrected(
        a_data,
        b_data,
        &mut out,
        k,
        n,
        &col_sum_b,
        &row_sum_a,
        a_zp,
        b_zp,
        k_zp_product,
        output_scale,
    );

    Ok(Tensor::new(out, vec![m, n]))
}
/// QLinearConv: Fully quantized 2D convolution.
///
/// Performs convolution in integer arithmetic with per-channel weight scales,
/// then requantizes the output. Implements the ONNX QLinearConv operator.
///
/// # Arguments
/// * `x_q` - Quantized input \[N,C,H,W\] as i8 values stored in f32
/// * `x_scale` - Input quantization scale
/// * `x_zero_point` - Input zero point
/// * `w_q` - Quantized weights \[OC,IC/g,kH,kW\] as i8 values stored in f32
/// * `w_scale` - Per-channel or per-tensor weight scales
/// * `w_zero_point` - Per-channel or per-tensor weight zero points
/// * `y_scale` - Output quantization scale
/// * `y_zero_point` - Output zero point
/// * `bias` - Optional bias \[OC\] in float
/// * `strides` - Convolution strides \[sH, sW\]
/// * `pads` - Padding \[pad_top, pad_left, pad_bottom, pad_right\]
/// * `group` - Number of groups
#[allow(clippy::too_many_arguments)]
pub fn qlinear_conv2d(
    x_q: &Tensor,
    x_scale: f32,
    x_zero_point: i8,
    w_q: &Tensor,
    w_scale: &[f32],
    w_zero_point: &[i8],
    y_scale: f32,
    y_zero_point: i8,
    bias: Option<&Tensor>,
    strides: &[usize],
    pads: &[usize],
    group: usize,
) -> Result<Tensor, OnnxError> {
    if x_q.shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: input must be 4D [N,C,H,W], got {:?}",
            x_q.shape
        )));
    }
    if w_q.shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: weight must be 4D [OC,IC/g,kH,kW], got {:?}",
            w_q.shape
        )));
    }
    if strides.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "qlinear_conv2d: strides must have at least 2 elements".into(),
        ));
    }
    if pads.len() < 4 {
        return Err(OnnxError::ShapeMismatch(
            "qlinear_conv2d: pads must have at least 4 elements".into(),
        ));
    }
    if y_scale.abs() < 1e-15 {
        return Err(OnnxError::ShapeMismatch(
            "qlinear_conv2d: y_scale is effectively zero".into(),
        ));
    }
    let batch_size = x_q.shape[0];
    let c_in = x_q.shape[1];
    let h_in = x_q.shape[2];
    let w_in = x_q.shape[3];
    let c_out = w_q.shape[0];
    let c_per_group = w_q.shape[1];
    let k_h = w_q.shape[2];
    let k_w = w_q.shape[3];
    if c_in != c_per_group * group {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: input channels {} != weight IC/g {} * group {}",
            c_in, c_per_group, group
        )));
    }
    if c_out % group != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: c_out {} not divisible by group {}",
            c_out, group
        )));
    }
    let per_channel_w = w_scale.len() > 1;
    if per_channel_w && w_scale.len() != c_out {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: w_scale len {} != c_out {}",
            w_scale.len(),
            c_out
        )));
    }
    if per_channel_w && w_zero_point.len() != c_out {
        return Err(OnnxError::ShapeMismatch(format!(
            "qlinear_conv2d: w_zero_point len {} != c_out {}",
            w_zero_point.len(),
            c_out
        )));
    }
    let h_out = (h_in + pads[0] + pads[2] - k_h) / strides[0] + 1;
    let w_out = (w_in + pads[1] + pads[3] - k_w) / strides[1] + 1;
    let c_out_per_group = c_out / group;
    let col_rows = c_per_group * k_h * k_w;
    let col_cols = h_out * w_out;
    let x_zp_i32 = x_zero_point as i32;
    let mut output = vec![0.0f32; batch_size * c_out * h_out * w_out];
    for batch in 0..batch_size {
        for g in 0..group {
            let in_c_start = g * c_per_group;
            let mut col = vec![0i32; col_rows * col_cols];
            let mut row = 0usize;
            for ic in 0..c_per_group {
                let in_c = in_c_start + ic;
                let plane_off = (batch * c_in + in_c) * h_in * w_in;
                for ky in 0..k_h {
                    for kx in 0..k_w {
                        for oy in 0..h_out {
                            let iy = (oy * strides[0] + ky) as isize - pads[0] as isize;
                            let base = row * col_cols + oy * w_out;
                            if iy < 0 || iy >= h_in as isize {
                                for ox in 0..w_out {
                                    col[base + ox] = x_zp_i32;
                                }
                            } else {
                                let iy_u = iy as usize;
                                for ox in 0..w_out {
                                    let ix = (ox * strides[1] + kx) as isize - pads[1] as isize;
                                    col[base + ox] = if ix >= 0 && ix < w_in as isize {
                                        x_q.data[plane_off + iy_u * w_in + ix as usize] as i32
                                    } else {
                                        x_zp_i32
                                    };
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
            let mut col_sums = vec![0i32; col_cols];
            for r in 0..col_rows {
                for c_idx in 0..col_cols {
                    col_sums[c_idx] += col[r * col_cols + c_idx];
                }
            }
            for oc in 0..c_out_per_group {
                let global_oc = g * c_out_per_group + oc;
                let w_sc = if per_channel_w {
                    w_scale[global_oc]
                } else {
                    w_scale[0]
                };
                let w_zp_i32 = if per_channel_w {
                    w_zero_point[global_oc] as i32
                } else {
                    w_zero_point[0] as i32
                };
                let w_base = global_oc * col_rows;
                let mut w_row_sum = 0i32;
                for r in 0..col_rows {
                    w_row_sum += w_q.data[w_base + r] as i32;
                }
                let bias_i32 = if let Some(b) = bias {
                    let combined_scale = x_scale * w_sc;
                    if combined_scale.abs() < 1e-15 {
                        0i32
                    } else {
                        (b.data[global_oc] / combined_scale).round() as i32
                    }
                } else {
                    0i32
                };
                let requant_scale = x_scale * w_sc / y_scale;
                let y_zp_f = y_zero_point as f32;
                let zp_correction = col_rows as i32 * x_zp_i32 * w_zp_i32;
                let o_base = (batch * c_out + global_oc) * col_cols;
                for sp in 0..col_cols {
                    let mut raw_sum = 0i32;
                    for r in 0..col_rows {
                        raw_sum += w_q.data[w_base + r] as i32 * col[r * col_cols + sp];
                    }
                    let corrected = raw_sum - x_zp_i32 * w_row_sum - w_zp_i32 * col_sums[sp]
                        + zp_correction
                        + bias_i32;
                    let y_q = (corrected as f32 * requant_scale + y_zp_f)
                        .round()
                        .clamp(-128.0, 127.0);
                    output[o_base + sp] = y_q;
                }
            }
        }
    }
    Ok(Tensor::new(output, vec![batch_size, c_out, h_out, w_out]))
}
/// Round to the nearest integer, ties to even (banker's rounding) — the
/// rounding mode the ONNX `DynamicQuantizeLinear` spec mandates for both
/// `round(qmin - min_x/y_scale)` and `round(x/y_scale)`. Rust's `f32::round`
/// is ties-away-from-zero, which diverges exactly at the `.5` boundary
/// (`2.5 -> 3` instead of the spec-correct `2`), so it cannot be used
/// directly here. Hand-rolled rather than the standard-library
/// `f32::round_ties_even` (stabilized in Rust 1.77) to stay compatible with
/// this workspace's `rust-version` (1.75) MSRV — the same shape as
/// `registry::quant_ops::round_ties_even_f32` and
/// `indexing::quantize::round_ties_even`, kept file-local here rather than
/// shared across modules, matching those two.
#[inline]
fn round_ties_even(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let floor = v.floor();
    let diff = v - floor;
    if diff > 0.5 {
        floor + 1.0
    } else if diff < 0.5 {
        floor
    } else {
        // Exact tie: round to the neighbouring even integer.
        if floor.rem_euclid(2.0) == 0.0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

/// Dynamic quantization: compute optimal uint8 quantization parameters from data.
///
/// Returns `(quantized_tensor, scale, zero_point)` where values are in \[0, 255\]
/// stored as f32 (uint8 semantics). The zero\_point is returned as i8 per ONNX
/// convention (reinterpret as u8).
///
/// The range always includes 0 to avoid bias in ReLU-like activations.
///
/// Matches the ONNX `DynamicQuantizeLinear` spec (and this crate's registered
/// `DynamicQuantizeLinearOp`, which already implements it correctly) in two
/// details that are easy to get backwards:
///
/// * the zero point is added **after** rounding (`round(x/s) + zp`), not
///   inside it — for a negative half-way value the two orders differ by a
///   full quantization step, and
/// * rounding is **ties-to-even**, not half-away-from-zero.
pub fn dynamic_quantize(x: &Tensor) -> Result<(Tensor, f32, i8), String> {
    if x.data.is_empty() {
        return Err("dynamic_quantize: empty tensor".into());
    }
    let min_val = x
        .data
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .min(0.0);
    let max_val = x
        .data
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.0);
    let range = max_val - min_val;
    let scale = if range < 1e-10 { 1e-10 } else { range / 255.0 };
    let zp_f = round_ties_even((-min_val / scale).clamp(0.0, 255.0));
    let zero_point = zp_f as u8 as i8;
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|&v| (round_ties_even(v / scale) + zp_f).clamp(0.0, 255.0))
        .collect();
    Ok((Tensor::new(data, x.shape.clone()), scale, zero_point))
}
