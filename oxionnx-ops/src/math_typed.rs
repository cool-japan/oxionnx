//! Typed matmul and GEMM kernels for native dispatch (Phase D.3+).
//!
//! Relocated from `math.rs` — v0.1.8 batched MatMul kernels (I8, I32, F16, BF16).
//! Added in v0.1.9 — GEMM kernels with alpha/beta/transA/transB/optional-C support.
//! Added in v0.1.6 (W2-perf-matmul) — [`sgemm_strided`], the single stride-aware
//! `matrixmultiply::sgemm` entry point, plus F32 kernels ([`matmul_f32`],
//! [`matmul_f32_into`], [`gemm_f32`]) that operate on borrowed slices instead of
//! an owned `Tensor`. This module is declared `pub(crate)` at the crate root, so
//! it is visible from both `crate::math::matmul` (a descendant of the root) and
//! `crate::registry::math_ops::matmul_gemm` (a cousin module) without either one
//! needing to reach into the other's private submodules — it is the shared home
//! for the sgemm call so the F32 hot path (`crate::math::matmul`/`gemm`) and the
//! typed dispatch F32 arms (`MatMulOp`/`GemmOp::execute_typed`) share one
//! implementation instead of duplicating the `unsafe` FFI call.
//!
//! F16/BF16 tensors are stored as raw `u16` bit patterns (matching `TensorStorage::F16/BF16`).

use oxionnx_core::Tensor;

// ── W2-perf-matmul: shared stride-aware sgemm entry point ───────────────────

/// Call `matrixmultiply::sgemm` with explicit row/col strides for A, B, and C.
///
/// Computes `C := alpha * A @ B + beta * C`, where `A[i,p] = *(a.as_ptr() +
/// i*rsa + p*csa)` (and likewise for B/C — `rs`/`cs` are element strides, not
/// byte strides). Passing swapped strides is how a logical transpose is
/// expressed without copying: e.g. `rsa=1, csa=m` reads a `[k,m]`-stored
/// buffer as if it were the transposed `[m,k]` matrix A.
///
/// `matrixmultiply`'s packing step gathers `A`/`B` into cache-blocked panels
/// using exactly these strides before the microkernel ever runs, so the
/// accumulation order over `k` — and therefore the rounding of the result —
/// depends only on `(m, k, n)`, never on `(rsa, csa, rsb, csb)`. A strided
/// (transposed) call is bit-identical to physically transposing the operand
/// first and calling with unit strides.
///
/// Safe on degenerate shapes without any extra guard: `matrixmultiply::sgemm`
/// checks `m == 0 || k == 0 || n == 0` internally and returns `beta*C` without
/// ever dereferencing `a`/`b` in that case (see `gemm_loop` in the
/// `matrixmultiply` source) — an empty `a_data[a_off..a_off]` slice's pointer
/// is valid-but-not-dereferenced per `<[T]>::as_ptr`'s guarantee, and is never
/// read through.
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe; safety covered above.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm_strided(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    rsa: isize,
    csa: isize,
    b: &[f32],
    rsb: isize,
    csb: isize,
    beta: f32,
    c: &mut [f32],
    rsc: isize,
    csc: isize,
) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            rsa,
            csa,
            b.as_ptr(),
            rsb,
            csb,
            beta,
            c.as_mut_ptr(),
            rsc,
            csc,
        );
    }
}

// ── v0.1.8: Batch-info helper ────────────────────────────────────────────────

/// Compute batch info `(batch_size, a_batches, b_batches, out_shape)` shared
/// across all typed matmul kernels.  Returns `Err(String)` on shape mismatch.
pub(crate) fn typed_matmul_batch_info(
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<(usize, usize, usize, Vec<usize>), String> {
    let an = a_shape.len();
    let bn = b_shape.len();
    if an < 2 || bn < 2 {
        return Err(format!(
            "typed matmul requires at least 2D tensors, got {an}D and {bn}D",
        ));
    }
    let k = a_shape[an - 1];
    let k2 = b_shape[bn - 2];
    if k != k2 {
        return Err(format!(
            "typed matmul: inner dimensions mismatch {k} != {k2}"
        ));
    }
    let a_batch: Vec<usize> = a_shape[..an - 2].to_vec();
    let b_batch: Vec<usize> = b_shape[..bn - 2].to_vec();
    let out_batch = Tensor::broadcast_shape(&a_batch, &b_batch)?;
    let batch_size: usize = out_batch.iter().product::<usize>().max(1);
    let m = a_shape[an - 2];
    let n = b_shape[bn - 1];
    let a_batches = (a_shape[..an - 2].iter().product::<usize>()).max(1);
    let b_batches = (b_shape[..bn - 2].iter().product::<usize>()).max(1);
    let mut out_shape = out_batch;
    out_shape.push(m);
    out_shape.push(n);
    Ok((batch_size, a_batches, b_batches, out_shape))
}

// ── v0.1.8: Slice-level kernels (no bounds — caller ensures correctness) ─────

/// Naive triple-loop kernel for i8×i8→i32 (accumulate in i32).
fn matmul_slice_i8_i32(a: &[i8], b: &[i8], out: &mut [i32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += (a[i * k + p] as i32) * (b[p * n + j] as i32);
            }
            out[i * n + j] = acc;
        }
    }
}

/// Naive triple-loop kernel for i32×i32→i32.
fn matmul_slice_i32(a: &[i32], b: &[i32], out: &mut [i32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc = acc.wrapping_add(a[i * k + p].wrapping_mul(b[p * n + j]));
            }
            out[i * n + j] = acc;
        }
    }
}

/// Naive triple-loop kernel for f16 (stored as u16 bits) — accumulates in f32,
/// converts output back to f16 bits.
fn matmul_slice_f16(
    a_bits: &[u16],
    b_bits: &[u16],
    out_bits: &mut [u16],
    m: usize,
    n: usize,
    k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let av = half::f16::from_bits(a_bits[i * k + p]).to_f32();
                let bv = half::f16::from_bits(b_bits[p * n + j]).to_f32();
                acc += av * bv;
            }
            out_bits[i * n + j] = half::f16::from_f32(acc).to_bits();
        }
    }
}

/// Naive triple-loop kernel for bf16 (stored as u16 bits) — accumulates in f32,
/// converts output back to bf16 bits.
fn matmul_slice_bf16(
    a_bits: &[u16],
    b_bits: &[u16],
    out_bits: &mut [u16],
    m: usize,
    n: usize,
    k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let av = half::bf16::from_bits(a_bits[i * k + p]).to_f32();
                let bv = half::bf16::from_bits(b_bits[p * n + j]).to_f32();
                acc += av * bv;
            }
            out_bits[i * n + j] = half::bf16::from_f32(acc).to_bits();
        }
    }
}

// ── v0.1.8: Batched matmul public API ────────────────────────────────────────

/// Batched matmul for typed i8 inputs, returning i32 output.
///
/// Follows the same batch-broadcast semantics as [`oxionnx_ops::math::matmul`].
pub(crate) fn matmul_i8_i32(
    a_data: &[i8],
    a_shape: &[usize],
    b_data: &[i8],
    b_shape: &[usize],
) -> Result<(Vec<i32>, Vec<usize>), String> {
    let (batch_size, a_batches, b_batches, out_shape) = typed_matmul_batch_info(a_shape, b_shape)?;
    let an = a_shape.len();
    let bn = b_shape.len();
    let m = a_shape[an - 2];
    let k = a_shape[an - 1];
    let n = b_shape[bn - 1];
    let mn = m * n;
    let mut out = vec![0i32; batch_size * mn];
    for b_idx in 0..batch_size {
        let a_off = (b_idx % a_batches) * (m * k);
        let b_off = (b_idx % b_batches) * (k * n);
        let c_off = b_idx * mn;
        matmul_slice_i8_i32(
            &a_data[a_off..a_off + m * k],
            &b_data[b_off..b_off + k * n],
            &mut out[c_off..c_off + mn],
            m,
            n,
            k,
        );
    }
    Ok((out, out_shape))
}

/// Batched matmul for typed i32 inputs, returning i32 output.
pub(crate) fn matmul_i32(
    a_data: &[i32],
    a_shape: &[usize],
    b_data: &[i32],
    b_shape: &[usize],
) -> Result<(Vec<i32>, Vec<usize>), String> {
    let (batch_size, a_batches, b_batches, out_shape) = typed_matmul_batch_info(a_shape, b_shape)?;
    let an = a_shape.len();
    let bn = b_shape.len();
    let m = a_shape[an - 2];
    let k = a_shape[an - 1];
    let n = b_shape[bn - 1];
    let mn = m * n;
    let mut out = vec![0i32; batch_size * mn];
    for b_idx in 0..batch_size {
        let a_off = (b_idx % a_batches) * (m * k);
        let b_off = (b_idx % b_batches) * (k * n);
        let c_off = b_idx * mn;
        matmul_slice_i32(
            &a_data[a_off..a_off + m * k],
            &b_data[b_off..b_off + k * n],
            &mut out[c_off..c_off + mn],
            m,
            n,
            k,
        );
    }
    Ok((out, out_shape))
}

/// Batched matmul for typed f16 (u16 bits) inputs, returning f16 (u16 bits) output.
pub(crate) fn matmul_f16(
    a_data: &[u16],
    a_shape: &[usize],
    b_data: &[u16],
    b_shape: &[usize],
) -> Result<(Vec<u16>, Vec<usize>), String> {
    let (batch_size, a_batches, b_batches, out_shape) = typed_matmul_batch_info(a_shape, b_shape)?;
    let an = a_shape.len();
    let bn = b_shape.len();
    let m = a_shape[an - 2];
    let k = a_shape[an - 1];
    let n = b_shape[bn - 1];
    let mn = m * n;
    let mut out = vec![0u16; batch_size * mn];
    for b_idx in 0..batch_size {
        let a_off = (b_idx % a_batches) * (m * k);
        let b_off = (b_idx % b_batches) * (k * n);
        let c_off = b_idx * mn;
        matmul_slice_f16(
            &a_data[a_off..a_off + m * k],
            &b_data[b_off..b_off + k * n],
            &mut out[c_off..c_off + mn],
            m,
            n,
            k,
        );
    }
    Ok((out, out_shape))
}

/// Batched matmul for typed bf16 (u16 bits) inputs, returning bf16 (u16 bits) output.
pub(crate) fn matmul_bf16(
    a_data: &[u16],
    a_shape: &[usize],
    b_data: &[u16],
    b_shape: &[usize],
) -> Result<(Vec<u16>, Vec<usize>), String> {
    let (batch_size, a_batches, b_batches, out_shape) = typed_matmul_batch_info(a_shape, b_shape)?;
    let an = a_shape.len();
    let bn = b_shape.len();
    let m = a_shape[an - 2];
    let k = a_shape[an - 1];
    let n = b_shape[bn - 1];
    let mn = m * n;
    let mut out = vec![0u16; batch_size * mn];
    for b_idx in 0..batch_size {
        let a_off = (b_idx % a_batches) * (m * k);
        let b_off = (b_idx % b_batches) * (k * n);
        let c_off = b_idx * mn;
        matmul_slice_bf16(
            &a_data[a_off..a_off + m * k],
            &b_data[b_off..b_off + k * n],
            &mut out[c_off..c_off + mn],
            m,
            n,
            k,
        );
    }
    Ok((out, out_shape))
}

// ── W2-perf-matmul: batched F32 matmul on borrowed slices ───────────────────
//
// Mirrors `matmul_i8_i32`/`matmul_i32`/`matmul_f16`/`matmul_bf16` above, but
// for F32 via `sgemm_strided` instead of a naive triple loop, and it is what
// both `crate::math::matmul_into` and `MatMulOp::execute_typed`'s F32 arm
// delegate to — the former after building its own `&Tensor`-shaped
// validation errors, the latter directly on `TensorStorage::F32`'s borrowed
// `Vec<f32>` (no per-call clone of the operands, which matters most for B
// when it is a multi-megabyte weight matrix).

/// Batched F32 matmul, writing directly into `out` (resized to fit).
///
/// Follows the same batch-broadcast semantics as the other `matmul_*`
/// kernels in this module (and as `crate::math::matmul_into`, which this
/// backs after its own shape validation).
pub(crate) fn matmul_f32_into(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    out: &mut Vec<f32>,
) -> Result<Vec<usize>, String> {
    let (batch_size, a_batches, b_batches, out_shape) = typed_matmul_batch_info(a_shape, b_shape)?;
    let an = a_shape.len();
    let bn = b_shape.len();
    let m = a_shape[an - 2];
    let k = a_shape[an - 1];
    let n = b_shape[bn - 1];
    let mn = m * n;
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    out.resize(batch_size * mn, 0.0f32);

    #[cfg(not(target_arch = "wasm32"))]
    if batch_size >= 4 {
        use rayon::prelude::*;
        out.par_chunks_mut(mn).enumerate().for_each(|(b_idx, dst)| {
            let a_off = (b_idx % a_batches) * a_batch_stride;
            let b_off = (b_idx % b_batches) * b_batch_stride;
            sgemm_strided(
                m,
                k,
                n,
                1.0,
                &a_data[a_off..a_off + m * k],
                k as isize,
                1,
                &b_data[b_off..b_off + k * n],
                n as isize,
                1,
                0.0,
                dst,
                n as isize,
                1,
            );
        });
    } else {
        for b_idx in 0..batch_size {
            let a_off = (b_idx % a_batches) * a_batch_stride;
            let b_off = (b_idx % b_batches) * b_batch_stride;
            let c_off = b_idx * mn;
            sgemm_strided(
                m,
                k,
                n,
                1.0,
                &a_data[a_off..a_off + m * k],
                k as isize,
                1,
                &b_data[b_off..b_off + k * n],
                n as isize,
                1,
                0.0,
                &mut out[c_off..c_off + mn],
                n as isize,
                1,
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    for b_idx in 0..batch_size {
        let a_off = (b_idx % a_batches) * a_batch_stride;
        let b_off = (b_idx % b_batches) * b_batch_stride;
        let c_off = b_idx * mn;
        sgemm_strided(
            m,
            k,
            n,
            1.0,
            &a_data[a_off..a_off + m * k],
            k as isize,
            1,
            &b_data[b_off..b_off + k * n],
            n as isize,
            1,
            0.0,
            &mut out[c_off..c_off + mn],
            n as isize,
            1,
        );
    }

    Ok(out_shape)
}

/// Batched F32 matmul, returning a freshly-allocated `(data, shape)` pair.
///
/// Thin wrapper over [`matmul_f32_into`] — used by `MatMulOp::execute_typed`'s
/// F32 arm, which (like its I8/I32/F16/BF16 siblings) returns a new
/// `TypedTensor` rather than writing into a caller-supplied slot.
pub(crate) fn matmul_f32(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let mut out = Vec::new();
    let out_shape = matmul_f32_into(a_data, a_shape, b_data, b_shape, &mut out)?;
    Ok((out, out_shape))
}

// ── v0.1.9: GEMM parameter bundle ────────────────────────────────────────────

/// Output dimensions for a GEMM: `m` (rows of A/out), `n` (cols of B/out), `k` (inner dim).
///
/// Bundling these reduces the public-API argument count below the clippy limit.
pub(crate) struct GemmDims {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Scalar GEMM parameters: `alpha`, `beta`, `transA`, `transB`.
///
/// Bundling these reduces the public-API argument count below the clippy limit.
pub(crate) struct GemmParams {
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

// ── v0.1.9: GEMM helper — compute c_bias_f32 for a given (row, col) ──────────

/// Resolve the optional C bias term to a scalar f32 value for element `(row, col)`.
///
/// Supported C shapes:
///   - `[]` (scalar): broadcast to every output element.
///   - `[n]` (1-D):  broadcast row-wise, index by `col`.
///   - `[m, n]` (2-D): full bias, index by `row * n + col`.
///
/// # Returns
///
/// `0.0` when `c_opt` is `None`.
#[inline]
fn resolve_bias_f32(c_opt: Option<(&[f32], &[usize])>, row: usize, col: usize, n: usize) -> f32 {
    match c_opt {
        None => 0.0,
        Some((c_data, c_shape)) => match c_shape.len() {
            0 => *c_data.first().unwrap_or(&0.0),
            1 => c_data[col % n],
            _ => c_data[row * n + col],
        },
    }
}

#[inline]
fn resolve_bias_i32(c_opt: Option<(&[i32], &[usize])>, row: usize, col: usize, n: usize) -> i32 {
    match c_opt {
        None => 0,
        Some((c_data, c_shape)) => match c_shape.len() {
            0 => *c_data.first().unwrap_or(&0),
            1 => c_data[col % n],
            _ => c_data[row * n + col],
        },
    }
}

// ── v0.1.9: GEMM kernels ──────────────────────────────────────────────────────

/// Compute GEMM for I8 inputs with I32 accumulator.
///
/// `out[row * n + col] = round(params.alpha * A[row, :] · B[:, col] + params.beta * C[row, col])`
///
/// where:
/// - A/B are indexed with optional transposition in `params`.
/// - C is an optional I32 bias with broadcast (shapes `[]`, `[n]`, or `[m, n]`).
pub(crate) fn gemm_i8_i32(
    a: &[i8],
    b: &[i8],
    dims: &GemmDims,
    params: &GemmParams,
    c: Option<(&[i32], &[usize])>,
    out: &mut [i32],
) {
    let GemmDims { m, n, k } = *dims;
    let GemmParams {
        alpha,
        beta,
        trans_a,
        trans_b,
    } = *params;
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                let a_val = if trans_a {
                    a[p * m + row] as i32
                } else {
                    a[row * k + p] as i32
                };
                let b_val = if trans_b {
                    b[col * k + p] as i32
                } else {
                    b[p * n + col] as i32
                };
                acc += a_val * b_val;
            }
            let c_val = resolve_bias_i32(c, row, col, n);
            out[row * n + col] = (acc as f32 * alpha + beta * c_val as f32).round() as i32;
        }
    }
}

/// Compute GEMM for I32 inputs with I32 accumulator.
///
/// `out[row * n + col] = round(params.alpha * A[row, :] · B[:, col] + params.beta * C[row, col])`
pub(crate) fn gemm_i32(
    a: &[i32],
    b: &[i32],
    dims: &GemmDims,
    params: &GemmParams,
    c: Option<(&[i32], &[usize])>,
    out: &mut [i32],
) {
    let GemmDims { m, n, k } = *dims;
    let GemmParams {
        alpha,
        beta,
        trans_a,
        trans_b,
    } = *params;
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                let a_val = if trans_a {
                    a[p * m + row]
                } else {
                    a[row * k + p]
                };
                let b_val = if trans_b {
                    b[col * k + p]
                } else {
                    b[p * n + col]
                };
                acc = acc.wrapping_add(a_val.wrapping_mul(b_val));
            }
            let c_val = resolve_bias_i32(c, row, col, n);
            out[row * n + col] = (acc as f32 * alpha + beta * c_val as f32).round() as i32;
        }
    }
}

/// Compute GEMM for F16 inputs (stored as u16 bit patterns), accumulating in f32.
///
/// `out[row * n + col] = f16(params.alpha * A[row, :] · B[:, col] + params.beta * C[row, col])`
///
/// C is optional F16 bias (also stored as u16 bits) with broadcast (shapes `[]`, `[n]`, `[m, n]`).
pub(crate) fn gemm_f16(
    a: &[u16],
    b: &[u16],
    dims: &GemmDims,
    params: &GemmParams,
    c: Option<(&[u16], &[usize])>,
    out: &mut [u16],
) {
    let GemmDims { m, n, k } = *dims;
    let GemmParams {
        alpha,
        beta,
        trans_a,
        trans_b,
    } = *params;
    // Convert optional C bias to a temporary f32 view to unify bias handling.
    let c_f32_opt: Option<(Vec<f32>, &[usize])> = c.map(|(c_bits, c_shape)| {
        let vals: Vec<f32> = c_bits
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        (vals, c_shape)
    });

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let a_val = if trans_a {
                    half::f16::from_bits(a[p * m + row]).to_f32()
                } else {
                    half::f16::from_bits(a[row * k + p]).to_f32()
                };
                let b_val = if trans_b {
                    half::f16::from_bits(b[col * k + p]).to_f32()
                } else {
                    half::f16::from_bits(b[p * n + col]).to_f32()
                };
                acc += a_val * b_val;
            }
            let c_ref = c_f32_opt.as_ref().map(|(cv, cs)| (cv.as_slice(), *cs));
            let bias = resolve_bias_f32(c_ref, row, col, n);
            out[row * n + col] = half::f16::from_f32(acc * alpha + beta * bias).to_bits();
        }
    }
}

/// Compute GEMM for BF16 inputs (stored as u16 bit patterns), accumulating in f32.
///
/// `out[row * n + col] = bf16(params.alpha * A[row, :] · B[:, col] + params.beta * C[row, col])`
///
/// C is optional BF16 bias (also stored as u16 bits) with broadcast (shapes `[]`, `[n]`, `[m, n]`).
pub(crate) fn gemm_bf16(
    a: &[u16],
    b: &[u16],
    dims: &GemmDims,
    params: &GemmParams,
    c: Option<(&[u16], &[usize])>,
    out: &mut [u16],
) {
    let GemmDims { m, n, k } = *dims;
    let GemmParams {
        alpha,
        beta,
        trans_a,
        trans_b,
    } = *params;
    // Convert optional C bias to a temporary f32 view to unify bias handling.
    let c_f32_opt: Option<(Vec<f32>, &[usize])> = c.map(|(c_bits, c_shape)| {
        let vals: Vec<f32> = c_bits
            .iter()
            .map(|&bits| half::bf16::from_bits(bits).to_f32())
            .collect();
        (vals, c_shape)
    });

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let a_val = if trans_a {
                    half::bf16::from_bits(a[p * m + row]).to_f32()
                } else {
                    half::bf16::from_bits(a[row * k + p]).to_f32()
                };
                let b_val = if trans_b {
                    half::bf16::from_bits(b[col * k + p]).to_f32()
                } else {
                    half::bf16::from_bits(b[p * n + col]).to_f32()
                };
                acc += a_val * b_val;
            }
            let c_ref = c_f32_opt.as_ref().map(|(cv, cs)| (cv.as_slice(), *cs));
            let bias = resolve_bias_f32(c_ref, row, col, n);
            out[row * n + col] = half::bf16::from_f32(acc * alpha + beta * bias).to_bits();
        }
    }
}

// ── W2-perf-matmul: F32 GEMM on borrowed slices ──────────────────────────────

/// Whether a Gemm `C` bias of `c_shape` is one of the three shapes
/// [`resolve_bias_f32`]/[`resolve_bias_i32`] implement: scalar (`[]`), row
/// vector (`[n]`), or the full bias (`[m, n]`).
///
/// ONNX `Gemm`'s C is broadcast against `[m, n]` per the general NumPy rules
/// (also allowing e.g. `[1, n]`, `[m, 1]`, `[1, 1]` — all exercised against
/// `crate::math::gemm` by `oxionnx-directml/tests/reference_vs_ops.rs`), but
/// `resolve_bias_f32`/`resolve_bias_i32` only implement the three shapes
/// above: the `1 => c_data[col % n]` arm indexes out of bounds for any
/// rank-1 C whose length isn't exactly `n` (e.g. `[1]`), and the catch-all
/// `_ => c_data[row * n + col]` arm indexes out of bounds for any rank>=2 C
/// whose element count isn't exactly `m * n` (e.g. `[m, 1]`).
///
/// Callers of `gemm_i8_i32`/`gemm_i32`/`gemm_f16`/`gemm_bf16` (the kernels
/// that use `resolve_bias_i32`/`resolve_bias_f32`) MUST check this before
/// passing a non-`None` C through, and fall back to
/// `oxionnx_core::default_typed_via_f32` (which routes through the fully
/// general `crate::math::gemm`) when it returns `false` — a hand-authored
/// model supplying a `[1, n]` I32 C is malformed-input territory, not a
/// panic. [`gemm_f32`] does not need this check: it broadcasts C via
/// [`crate::math::broadcast_to`], which handles every NumPy-broadcastable
/// shape.
///
/// Behavioural note: `default_typed_via_f32` always produces an **F32**
/// `TypedTensor`, never the operator's native dtype — so for I8/I32/F16/BF16
/// Gemm, a C shaped e.g. `[1, n]` or `[m, 1]` (previously an out-of-bounds
/// panic) now yields correct values in F32 storage rather than I32/F16/BF16
/// storage from a single `execute_typed` call. This is not a regression for
/// any shape that previously worked (`[]`/`[n]`/`[m,n]` are unaffected and
/// stay on the native-dtype fast path); it only changes what happens for
/// shapes that previously panicked. Native-dtype recovery for the fallback
/// case happens at the session level (`run_typed`), not inside this
/// operator — see `w2_perf_matmul.rs`'s
/// `gemm_c_shape_1xn_does_not_panic_for_any_dtype` test, which asserts
/// `DType::F32` (not `DType::I32`) for exactly this case.
pub(crate) fn gemm_bias_shape_supported(c_shape: &[usize], m: usize, n: usize) -> bool {
    c_shape.is_empty() || c_shape == [n] || c_shape == [m, n]
}

/// Compute GEMM for F32 inputs directly on borrowed slices via
/// [`sgemm_strided`], avoiding the clone that wrapping `a`/`b` in an owned
/// `Tensor` (to call `crate::math::gemm`) would require — `B` is normally
/// the layer weight, so for a `[4096, 4096]` F32 weight this is the 64 MB
/// `memcpy` per node that `GemmOp::execute_typed`'s F32 arm no longer pays.
///
/// `A`/`B` are always exactly 2D here (`GemmOp::execute_typed` validates rank
/// before dispatch), so there is no batching and no non-2D fallback to
/// consider — unlike `crate::math::gemm`, which must also support the (rare,
/// spec-noncompliant) case of a higher-rank input reaching the untyped
/// `Gemm::execute` path.
///
/// `alpha`/`beta` are applied as separate passes after the raw `A @ B`
/// (matching `crate::math::gemm`'s own ordering) rather than folded into
/// `sgemm_strided`'s native `alpha`/`beta` parameters — algebraically
/// equivalent, but keeps the rounding of the dominant `A @ B` term the only
/// thing this function changes relative to the pre-optimization code path.
///
/// C, when present, is broadcast via [`crate::math::broadcast_to`] (see
/// [`gemm_bias_shape_supported`]'s doc comment for why the narrower
/// `resolve_bias_f32` helper is not used here).
pub(crate) fn gemm_f32(
    a: &[f32],
    b: &[f32],
    dims: &GemmDims,
    params: &GemmParams,
    c: Option<(&[f32], &[usize])>,
    out: &mut [f32],
) {
    let GemmDims { m, n, k } = *dims;
    let GemmParams {
        alpha,
        beta,
        trans_a,
        trans_b,
    } = *params;
    let (rsa, csa) = if trans_a {
        (1isize, m as isize)
    } else {
        (k as isize, 1isize)
    };
    let (rsb, csb) = if trans_b {
        (1isize, k as isize)
    } else {
        (n as isize, 1isize)
    };
    sgemm_strided(
        m, k, n, 1.0, a, rsa, csa, b, rsb, csb, 0.0, out, n as isize, 1,
    );
    if alpha != 1.0 {
        out.iter_mut().for_each(|v| *v *= alpha);
    }
    if let Some((c_data, c_shape)) = c {
        let c_tensor = Tensor::new(c_data.to_vec(), c_shape.to_vec());
        let c_bcast = crate::math::broadcast_to(&c_tensor, &[m, n]);
        for (o, &cv) in out.iter_mut().zip(c_bcast.data.iter()) {
            *o += beta * cv;
        }
    }
}
