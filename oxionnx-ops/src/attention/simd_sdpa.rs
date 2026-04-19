//! SIMD-accelerated SDPA inner loops.
//!
//! Three hot-loop kernels for scaled dot-product attention:
//! 1. `compute_qk_scores` — Q·K^T dot products with scale
//! 2. `softmax_inplace`    — numerically-stable row-wise softmax
//! 3. `weighted_sum_v`     — softmax-weighted V accumulation
//!
//! Architecture dispatch:
//! - aarch64: NEON via vfmaq_f32 / vaddvq_f32 / vst1q_f32
//! - x86_64:  AVX2+FMA via runtime `is_x86_feature_detected!` guards
//! - other:   scalar fallback
//!
//! Softmax exp is always scalar `f32::exp` for numerical correctness — fast-exp
//! polynomial approximation has relative error that can exceed 1e-5 absolute at
//! large magnitudes, which would break correctness tests.

use crate::simd_ops::{simd_dot_product, simd_reduce_max, simd_reduce_sum};

// ── Q·K^T scores ─────────────────────────────────────────────────────────────

/// Compute attention scores: `out_scores[kv_j] = dot(q_row, k_mat[kv_j]) * scale`
/// for all `kv_j` in `0..seq_kv`.
///
/// Uses `simd_dot_product` which dispatches to NEON / AVX2+FMA / scalar.
///
/// # Arguments
/// * `q_row`     — single query row `[head_dim]`
/// * `k_mat`     — key matrix `[seq_kv * head_dim]`, row-major
/// * `scale`     — multiplicative scale (1/sqrt(head_dim))
/// * `head_dim`  — feature dimension per head
/// * `seq_kv`    — number of key/value positions
/// * `out_scores` — output scores `[seq_kv]`
pub(crate) fn compute_qk_scores(
    q_row: &[f32],
    k_mat: &[f32],
    scale: f32,
    head_dim: usize,
    seq_kv: usize,
    out_scores: &mut [f32],
) {
    debug_assert_eq!(q_row.len(), head_dim);
    debug_assert_eq!(k_mat.len(), seq_kv * head_dim);
    debug_assert_eq!(out_scores.len(), seq_kv);

    for kv_j in 0..seq_kv {
        let k_row = &k_mat[kv_j * head_dim..(kv_j + 1) * head_dim];
        out_scores[kv_j] = simd_dot_product(q_row, k_row) * scale;
    }
}

// ── Softmax ───────────────────────────────────────────────────────────────────

/// In-place numerically-stable softmax over a single row.
///
/// Algorithm:
/// 1. SIMD reduce_max (stability shift)
/// 2. Scalar exp loop (correctness — polynomial approx has too large an abs error)
/// 3. SIMD reduce_sum
/// 4. SIMD scalar multiply by inv_sum
///
/// Works correctly on empty slices (no-op) and single-element slices.
pub(crate) fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }

    // Pass 1: numerically-stable max via SIMD
    let max_val = simd_reduce_max(scores);

    // Pass 2: exp(x - max) — scalar for correctness
    for v in scores.iter_mut() {
        *v = (*v - max_val).exp();
    }

    // Pass 3: sum via SIMD
    let sum = simd_reduce_sum(scores);

    // Pass 4: normalize
    if sum > 0.0 {
        let inv_sum = sum.recip();
        normalize_slice(scores, inv_sum);
    }
}

// ── Normalize helper (NEON / AVX2 / scalar) ──────────────────────────────────

/// Multiply every element of `data` by `scale` in-place.
/// Dispatches to NEON vmulq_f32 or AVX2 _mm256_mul_ps where available.
#[inline]
fn normalize_slice(data: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        normalize_slice_neon(data, scale);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { normalize_slice_avx2(data, scale) };
            return;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for v in data.iter_mut() {
            *v *= scale;
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    for v in data.iter_mut() {
        *v *= scale;
    }
}

#[cfg(target_arch = "aarch64")]
fn normalize_slice_neon(data: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;
    const LANES: usize = 4;
    let n = data.len();
    let chunks = n / LANES;
    unsafe {
        let v_scale = vdupq_n_f32(scale);
        for i in 0..chunks {
            let offset = i * LANES;
            let v = vld1q_f32(data.as_ptr().add(offset));
            vst1q_f32(data.as_mut_ptr().add(offset), vmulq_f32(v, v_scale));
        }
    }
    for v in data[chunks * LANES..].iter_mut() {
        *v *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_slice_avx2(data: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    let n = data.len();
    let chunks = n / LANES;
    let v_scale = _mm256_set1_ps(scale);
    for i in 0..chunks {
        let offset = i * LANES;
        let v = _mm256_loadu_ps(data.as_ptr().add(offset));
        _mm256_storeu_ps(data.as_mut_ptr().add(offset), _mm256_mul_ps(v, v_scale));
    }
    for v in data[chunks * LANES..].iter_mut() {
        *v *= scale;
    }
}

// ── V weighted sum ────────────────────────────────────────────────────────────

/// Compute attention output row: `output[d] = sum_j(weights[j] * v_mat[j*head_dim + d])`
///
/// Architecture dispatch:
/// - aarch64: NEON vfmaq_f32 (FMA: output += w_j * v_row)
/// - x86_64:  AVX2+FMA _mm256_fmadd_ps with runtime detection
/// - other:   scalar
///
/// # Arguments
/// * `weights`   — softmax weights `[seq_kv]`
/// * `v_mat`     — value matrix `[seq_kv * head_dim]`, row-major
/// * `head_dim`  — feature dimension per head
/// * `seq_kv`    — number of key/value positions
/// * `output`    — output slice `[head_dim]`, zeroed by caller or accumulated
pub(crate) fn weighted_sum_v(
    weights: &[f32],
    v_mat: &[f32],
    head_dim: usize,
    seq_kv: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(weights.len(), seq_kv);
    debug_assert_eq!(v_mat.len(), seq_kv * head_dim);
    debug_assert_eq!(output.len(), head_dim);

    // Zero output buffer before accumulation
    output.fill(0.0);

    #[cfg(target_arch = "aarch64")]
    {
        weighted_sum_v_neon(weights, v_mat, head_dim, seq_kv, output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { weighted_sum_v_avx2(weights, v_mat, head_dim, seq_kv, output) };
            return;
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        weighted_sum_v_scalar(weights, v_mat, head_dim, seq_kv, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    weighted_sum_v_scalar(weights, v_mat, head_dim, seq_kv, output);
}

// ── NEON weighted_sum_v ───────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn weighted_sum_v_neon(
    weights: &[f32],
    v_mat: &[f32],
    head_dim: usize,
    seq_kv: usize,
    output: &mut [f32],
) {
    use std::arch::aarch64::*;
    const LANES: usize = 4;
    let chunks = head_dim / LANES;
    let tail_start = chunks * LANES;

    for j in 0..seq_kv {
        let w_j = weights[j];
        let v_row = &v_mat[j * head_dim..(j + 1) * head_dim];
        unsafe {
            let v_wj = vdupq_n_f32(w_j);
            for c in 0..chunks {
                let offset = c * LANES;
                let v_v = vld1q_f32(v_row.as_ptr().add(offset));
                let v_out = vld1q_f32(output.as_ptr().add(offset));
                // output[offset..offset+4] += w_j * v_row[offset..offset+4]
                let updated = vfmaq_f32(v_out, v_wj, v_v);
                vst1q_f32(output.as_mut_ptr().add(offset), updated);
            }
        }
        for d in tail_start..head_dim {
            output[d] += w_j * v_row[d];
        }
    }
}

// ── AVX2 weighted_sum_v ───────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn weighted_sum_v_avx2(
    weights: &[f32],
    v_mat: &[f32],
    head_dim: usize,
    seq_kv: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    let chunks = head_dim / LANES;
    let tail_start = chunks * LANES;

    for j in 0..seq_kv {
        let w_j = weights[j];
        let v_row = &v_mat[j * head_dim..(j + 1) * head_dim];
        let v_wj = _mm256_set1_ps(w_j);
        for c in 0..chunks {
            let offset = c * LANES;
            let v_v = _mm256_loadu_ps(v_row.as_ptr().add(offset));
            let v_out = _mm256_loadu_ps(output.as_ptr().add(offset));
            // output[offset..offset+8] = w_j * v_row + output
            let updated = _mm256_fmadd_ps(v_wj, v_v, v_out);
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), updated);
        }
        for d in tail_start..head_dim {
            output[d] += w_j * v_row[d];
        }
    }
}

// ── Scalar fallback for weighted_sum_v ───────────────────────────────────────

#[allow(dead_code)]
fn weighted_sum_v_scalar(
    weights: &[f32],
    v_mat: &[f32],
    head_dim: usize,
    seq_kv: usize,
    output: &mut [f32],
) {
    for j in 0..seq_kv {
        let w_j = weights[j];
        let v_row = &v_mat[j * head_dim..(j + 1) * head_dim];
        for d in 0..head_dim {
            output[d] += w_j * v_row[d];
        }
    }
}
