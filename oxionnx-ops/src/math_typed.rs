//! Typed (non-f32) matmul and GEMM kernels for native dispatch (Phase D.3+).
//!
//! Relocated from `math.rs` — v0.1.8 batched MatMul kernels (I8, I32, F16, BF16).
//! Added in v0.1.9 — GEMM kernels with alpha/beta/transA/transB/optional-C support.
//!
//! F16/BF16 tensors are stored as raw `u16` bit patterns (matching `TensorStorage::F16/BF16`).

use oxionnx_core::Tensor;

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
