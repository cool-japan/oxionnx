//! SIMD-accelerated element-wise operations.
//!
//! Enabled with the `simd` feature flag. Provides accelerated versions of
//! add, mul, relu, sigmoid, tanh, gelu, silu, and exp element-wise ops.
//! Falls back to scalar loops when SIMD is not available.

// ── Public dispatch functions ───────────────────────────────────────────────

/// SIMD-accelerated element-wise addition: out[i] = a[i] + b[i]
pub fn simd_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    dispatch_binary(a, b, out, Op::Add);
}

/// SIMD-accelerated element-wise multiplication: out[i] = a[i] * b[i]
pub fn simd_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    dispatch_binary(a, b, out, Op::Mul);
}

/// SIMD-accelerated in-place ReLU: data[i] = max(data[i], 0)
pub fn simd_relu(data: &mut [f32]) {
    dispatch_unary(data, Op::Relu);
}

/// SIMD-accelerated in-place sigmoid approximation
pub fn simd_sigmoid(data: &mut [f32]) {
    dispatch_unary(data, Op::Sigmoid);
}

/// SIMD-accelerated in-place tanh approximation
pub fn simd_tanh(data: &mut [f32]) {
    dispatch_unary(data, Op::Tanh);
}

/// SIMD-accelerated in-place GELU approximation
pub fn simd_gelu(data: &mut [f32]) {
    dispatch_unary(data, Op::Gelu);
}

/// SIMD-accelerated in-place SiLU (x * sigmoid(x))
pub fn simd_silu(data: &mut [f32]) {
    dispatch_unary(data, Op::Silu);
}

/// SIMD-accelerated in-place fast exp approximation
pub fn simd_exp(data: &mut [f32]) {
    dispatch_unary(data, Op::Exp);
}

/// SIMD-accelerated element-wise subtraction: out[i] = a[i] - b[i]
pub fn simd_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    dispatch_binary(a, b, out, Op::Sub);
}

/// SIMD-accelerated element-wise division: out[i] = a[i] / b[i]
pub fn simd_div(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    dispatch_binary(a, b, out, Op::Div);
}

/// SIMD-accelerated in-place negation: data[i] = -data[i]
pub fn simd_neg(data: &mut [f32]) {
    dispatch_unary(data, Op::Neg);
}

/// SIMD-accelerated in-place absolute value: data[i] = |data[i]|
pub fn simd_abs(data: &mut [f32]) {
    dispatch_unary(data, Op::Abs);
}

/// SIMD-accelerated in-place square root: data[i] = sqrt(data[i])
pub fn simd_sqrt(data: &mut [f32]) {
    dispatch_unary(data, Op::Sqrt);
}

/// SIMD-accelerated in-place natural logarithm: data[i] = ln(data[i])
pub fn simd_log(data: &mut [f32]) {
    dispatch_unary(data, Op::Log);
}

// ── Horizontal reduction dispatch functions ─────────────────────────────────

/// SIMD-accelerated sum reduction over a flat slice.
///
/// Returns 0.0 for an empty slice.
pub fn simd_reduce_sum(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        neon_impl::reduce_sum(data)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected at runtime
            return unsafe { avx2_impl::reduce_sum(data) };
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        return scalar_reduce_sum(data);
    }

    // Fallback for x86_64 without AVX2
    #[cfg(target_arch = "x86_64")]
    scalar_reduce_sum(data)
}

/// SIMD-accelerated max reduction over a flat slice.
///
/// Returns `f32::NEG_INFINITY` for an empty slice.
pub fn simd_reduce_max(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }

    #[cfg(target_arch = "aarch64")]
    {
        neon_impl::reduce_max(data)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime
            return unsafe { avx2_impl::reduce_max(data) };
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        return scalar_reduce_max(data);
    }

    #[cfg(target_arch = "x86_64")]
    scalar_reduce_max(data)
}

/// SIMD-accelerated min reduction over a flat slice.
///
/// Returns `f32::INFINITY` for an empty slice.
pub fn simd_reduce_min(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::INFINITY;
    }

    #[cfg(target_arch = "aarch64")]
    {
        neon_impl::reduce_min(data)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime
            return unsafe { avx2_impl::reduce_min(data) };
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        return scalar_reduce_min(data);
    }

    #[cfg(target_arch = "x86_64")]
    scalar_reduce_min(data)
}

/// SIMD-accelerated dot product of two f32 slices.
///
/// Uses the minimum of `a.len()` and `b.len()` as the effective length.
/// Returns 0.0 if either slice is empty.
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let a = &a[..n];
    let b = &b[..n];

    #[cfg(target_arch = "aarch64")]
    {
        neon_impl::dot_product(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected at runtime
            return unsafe { avx2_impl::dot_product(a, b) };
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        return scalar_dot_product(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    scalar_dot_product(a, b)
}

/// SIMD-accelerated mean reduction over a flat slice.
///
/// Returns 0.0 for an empty slice.
pub fn simd_reduce_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    simd_reduce_sum(data) / data.len() as f32
}

// ── Softmax & LayerNorm dispatch functions ──────────────────────────────────

/// SIMD-accelerated in-place softmax over the entire slice.
///
/// Computes softmax: `data[i] = exp(data[i] - max) / sum(exp(data - max))`
pub fn simd_softmax_inplace(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        neon_impl::softmax_inplace(data);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected at runtime
            unsafe { avx2_impl::softmax_inplace(data) };
            return;
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_softmax_inplace(data);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    scalar_softmax_inplace(data);
}

/// SIMD-accelerated strided softmax: applies softmax to consecutive chunks
/// of `inner_dim` elements within `data`.
///
/// Used when softmax is along the last axis of a multi-dimensional tensor.
pub fn simd_softmax_strided(data: &mut [f32], inner_dim: usize) {
    if inner_dim == 0 || data.is_empty() {
        return;
    }
    let n_chunks = data.len() / inner_dim;
    for c in 0..n_chunks {
        let start = c * inner_dim;
        let end = start + inner_dim;
        simd_softmax_inplace(&mut data[start..end]);
    }
}

/// SIMD-accelerated in-place LayerNorm over the entire slice.
///
/// Computes: `data[i] = (data[i] - mean) / sqrt(var + eps) * scale[i] + bias[i]`
pub fn simd_layer_norm(data: &mut [f32], scale: &[f32], bias: Option<&[f32]>, eps: f32) {
    if data.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        neon_impl::layer_norm_inplace(data, scale, bias, eps);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected at runtime
            unsafe { avx2_impl::layer_norm_inplace(data, scale, bias, eps) };
            return;
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_layer_norm_inplace(data, scale, bias, eps);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    scalar_layer_norm_inplace(data, scale, bias, eps);
}

/// SIMD-accelerated strided LayerNorm: applies LayerNorm to consecutive chunks
/// of `inner_dim` elements within `data`.
///
/// `scale` and `bias` have length `inner_dim` and are reused for each chunk.
pub fn simd_layer_norm_strided(
    data: &mut [f32],
    inner_dim: usize,
    scale: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) {
    if inner_dim == 0 || data.is_empty() {
        return;
    }
    let n_chunks = data.len() / inner_dim;
    for c in 0..n_chunks {
        let start = c * inner_dim;
        let end = start + inner_dim;
        simd_layer_norm(&mut data[start..end], scale, bias, eps);
    }
}

// ── Dispatch helpers ────────────────────────────────────────────────────────

enum Op {
    Add,
    Mul,
    Sub,
    Div,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Exp,
    Neg,
    Abs,
    Sqrt,
    Log,
}

fn dispatch_binary(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    #[cfg(target_arch = "aarch64")]
    {
        match op {
            Op::Add => neon_impl::add(a, b, out),
            Op::Mul => neon_impl::mul(a, b, out),
            Op::Sub => neon_impl::sub(a, b, out),
            Op::Div => neon_impl::div(a, b, out),
            _ => scalar_binary(a, b, out, op),
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime
            unsafe {
                match op {
                    Op::Add => avx2_impl::add(a, b, out),
                    Op::Mul => avx2_impl::mul(a, b, out),
                    Op::Sub => avx2_impl::sub(a, b, out),
                    Op::Div => avx2_impl::div(a, b, out),
                    _ => scalar_binary(a, b, out, op),
                }
            }
        } else {
            scalar_binary(a, b, out, op);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_binary(a, b, out, op);
    }
}

fn dispatch_unary(data: &mut [f32], op: Op) {
    #[cfg(target_arch = "aarch64")]
    {
        match op {
            Op::Relu => neon_impl::relu(data),
            Op::Sigmoid => neon_impl::sigmoid(data),
            Op::Tanh => neon_impl::tanh_approx(data),
            Op::Gelu => neon_impl::gelu(data),
            Op::Silu => neon_impl::silu(data),
            Op::Exp => neon_impl::exp(data),
            Op::Neg => neon_impl::neg(data),
            Op::Abs => neon_impl::abs(data),
            Op::Sqrt => neon_impl::sqrt(data),
            Op::Log => neon_impl::log(data),
            _ => scalar_unary(data, op),
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime
            unsafe {
                match op {
                    Op::Relu => avx2_impl::relu(data),
                    Op::Sigmoid => avx2_impl::sigmoid(data),
                    Op::Tanh => avx2_impl::tanh_approx(data),
                    Op::Gelu => avx2_impl::gelu(data),
                    Op::Silu => avx2_impl::silu(data),
                    Op::Exp => avx2_impl::exp(data),
                    Op::Neg => avx2_impl::neg(data),
                    Op::Abs => avx2_impl::abs(data),
                    Op::Sqrt => avx2_impl::sqrt(data),
                    Op::Log => avx2_impl::log(data),
                    _ => scalar_unary(data, op),
                }
            }
        } else {
            scalar_unary(data, op);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_unary(data, op);
    }
}

// ── Scalar fallbacks ────────────────────────────────────────────────────────

/// Fast exp approximation (scalar) using range reduction.
///
/// Uses exp(x) = 2^n * exp(r) where x = n*ln(2) + r, |r| <= ln(2)/2.
/// The reduced-range exp(r) is computed with a degree-5 minimax polynomial.
#[inline]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    // Range reduction: n = round(x / ln2), r = x - n*ln2
    const LOG2E: f32 = std::f32::consts::LOG2_E; // 1/ln(2)
    const LN2_HI: f32 = 0.693_145_75; // ln(2) high part
    const LN2_LO: f32 = 1.428_606_8e-6; // ln(2) low part

    let n = (x * LOG2E + 0.5).floor();
    let r = x - n * LN2_HI - n * LN2_LO;

    // Polynomial approximation for exp(r), |r| <= ln(2)/2
    // Coefficients from a minimax fit
    let r2 = r * r;
    let p = 1.0
        + r
        + r2 * 0.5
        + r2 * r * (1.0 / 6.0)
        + r2 * r2 * (1.0 / 24.0)
        + r2 * r2 * r * (1.0 / 120.0);

    // Multiply by 2^n using bit manipulation
    let n_i = n as i32;
    let bits = ((n_i + 127) as u32) << 23;
    let pow2n = f32::from_bits(bits);
    p * pow2n
}

#[inline]
fn fast_sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp_scalar(-x))
}

/// Fast natural log approximation (scalar) using IEEE 754 bit decomposition.
///
/// Extracts exponent and mantissa from the float's bit representation, then uses
/// a polynomial correction on the mantissa.
#[inline]
fn fast_log_scalar(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    // Reconstruct mantissa in [1, 2)
    let mantissa_bits = (bits & 0x007F_FFFF) | 0x3F80_0000;
    let m = f32::from_bits(mantissa_bits); // m in [1.0, 2.0)

    // Polynomial approximation for ln(m) where m in [1.0, 2.0)
    // ln(m) ≈ (m - 1) * (2.0 - (m - 1) * 0.333_333_3)  -- Padé-like
    // More accurate: use a degree-3 minimax polynomial
    let f = m - 1.0;
    // Degree-5 minimax polynomial for ln(1+f) where f in [0, 1)
    let ln_m = f
        * (0.999_999_7
            + f * (-0.499_999_4 + f * (0.333_319_8 + f * (-0.249_989_5 + f * 0.150_198_6))));

    (exponent as f32) * std::f32::consts::LN_2 + ln_m
}

#[allow(dead_code)]
fn scalar_binary(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    match op {
        Op::Add => {
            for i in 0..a.len() {
                out[i] = a[i] + b[i];
            }
        }
        Op::Mul => {
            for i in 0..a.len() {
                out[i] = a[i] * b[i];
            }
        }
        Op::Sub => {
            for i in 0..a.len() {
                out[i] = a[i] - b[i];
            }
        }
        Op::Div => {
            for i in 0..a.len() {
                out[i] = a[i] / b[i];
            }
        }
        _ => {}
    }
}

#[allow(dead_code)]
fn scalar_unary(data: &mut [f32], op: Op) {
    match op {
        Op::Relu => {
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
        }
        Op::Sigmoid => {
            for v in data.iter_mut() {
                *v = fast_sigmoid_scalar(*v);
            }
        }
        Op::Tanh => {
            for v in data.iter_mut() {
                *v = 2.0 * fast_sigmoid_scalar(2.0 * *v) - 1.0;
            }
        }
        Op::Gelu => {
            const SQRT_2_OVER_PI: f32 = 0.797_884_6;
            const COEF: f32 = 0.044_715;
            for v in data.iter_mut() {
                let inner = SQRT_2_OVER_PI * (*v + COEF * *v * *v * *v);
                let tanh_val = 2.0 * fast_sigmoid_scalar(2.0 * inner) - 1.0;
                *v = 0.5 * *v * (1.0 + tanh_val);
            }
        }
        Op::Silu => {
            for v in data.iter_mut() {
                *v *= fast_sigmoid_scalar(*v);
            }
        }
        Op::Exp => {
            for v in data.iter_mut() {
                *v = fast_exp_scalar(*v);
            }
        }
        Op::Neg => {
            for v in data.iter_mut() {
                *v = -*v;
            }
        }
        Op::Abs => {
            for v in data.iter_mut() {
                *v = v.abs();
            }
        }
        Op::Sqrt => {
            for v in data.iter_mut() {
                *v = v.sqrt();
            }
        }
        Op::Log => {
            for v in data.iter_mut() {
                *v = fast_log_scalar(*v);
            }
        }
        _ => {}
    }
}

// ── Scalar reduction fallbacks ───────────────────────────────────────────────

#[allow(dead_code)]
fn scalar_reduce_sum(data: &[f32]) -> f32 {
    // Kahan compensated summation for improved accuracy on large arrays
    let mut sum = 0.0f32;
    let mut comp = 0.0f32;
    for &v in data {
        let y = v - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

#[allow(dead_code)]
fn scalar_reduce_max(data: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &v in data {
        if v > m {
            m = v;
        }
    }
    m
}

#[allow(dead_code)]
fn scalar_reduce_min(data: &[f32]) -> f32 {
    let mut m = f32::INFINITY;
    for &v in data {
        if v < m {
            m = v;
        }
    }
    m
}

#[allow(dead_code)]
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut comp = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let prod = x * y;
        let t_y = prod - comp;
        let t = sum + t_y;
        comp = (t - sum) - t_y;
        sum = t;
    }
    sum
}

// ── Scalar softmax & layer_norm fallbacks ───────────────────────────────────

#[allow(dead_code)]
fn scalar_softmax_inplace(data: &mut [f32]) {
    let max_val = scalar_reduce_max(data);
    let mut sum = 0.0f32;
    for v in data.iter_mut() {
        *v = fast_exp_scalar(*v - max_val);
        sum += *v;
    }
    if sum > 0.0 {
        let inv = sum.recip();
        for v in data.iter_mut() {
            *v *= inv;
        }
    }
}

#[allow(dead_code)]
fn scalar_layer_norm_inplace(data: &mut [f32], scale: &[f32], bias: Option<&[f32]>, eps: f32) {
    let n = data.len();
    if n == 0 {
        return;
    }
    let n_f = n as f32;
    let mean = scalar_reduce_sum(data) / n_f;
    let mut var_sum = 0.0f32;
    for &v in data.iter() {
        let d = v - mean;
        var_sum += d * d;
    }
    let inv_std = (var_sum / n_f + eps).sqrt().recip();
    for i in 0..n {
        let normalized = (data[i] - mean) * inv_std;
        data[i] = normalized * scale[i % scale.len()];
        if let Some(b) = bias {
            data[i] += b[i % b.len()];
        }
    }
}

// ── NEON (aarch64) implementations ──────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    const LANE_WIDTH: usize = 4; // f32x4

    /// Fast exp approximation for NEON f32x4 using range reduction.
    ///
    /// # Safety
    /// Requires aarch64 NEON support (always available on aarch64).
    #[inline]
    unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
        let min_val = vdupq_n_f32(-88.0);
        let max_val = vdupq_n_f32(88.0);
        let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2_hi = vdupq_n_f32(0.693_145_75);
        let ln2_lo = vdupq_n_f32(1.428_606_8e-6);
        let half = vdupq_n_f32(0.5);

        // n = floor(x * log2e + 0.5)
        let n_f = vrndmq_f32(vaddq_f32(vmulq_f32(x, log2e), half));
        // r = x - n * ln2
        let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n_f, ln2_hi)), vmulq_f32(n_f, ln2_lo));

        // Polynomial: 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
        let one = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(0.5);
        let c3 = vdupq_n_f32(1.0 / 6.0);
        let c4 = vdupq_n_f32(1.0 / 24.0);
        let c5 = vdupq_n_f32(1.0 / 120.0);

        let r2 = vmulq_f32(r, r);
        let r3 = vmulq_f32(r2, r);
        let r4 = vmulq_f32(r2, r2);
        let r5 = vmulq_f32(r4, r);

        let mut p = one;
        p = vaddq_f32(p, r);
        p = vaddq_f32(p, vmulq_f32(r2, c2));
        p = vaddq_f32(p, vmulq_f32(r3, c3));
        p = vaddq_f32(p, vmulq_f32(r4, c4));
        p = vaddq_f32(p, vmulq_f32(r5, c5));

        // 2^n via integer bit manipulation
        let n_i = vcvtq_s32_f32(n_f);
        let bias = vdupq_n_s32(127);
        let pow2n_bits = vshlq_n_s32::<23>(vaddq_s32(n_i, bias));
        let pow2n: float32x4_t = vreinterpretq_f32_s32(pow2n_bits);

        vmulq_f32(p, pow2n)
    }

    /// Fast sigmoid for NEON: 1 / (1 + exp(-x))
    ///
    /// # Safety
    /// Requires aarch64 NEON support.
    #[inline]
    unsafe fn fast_sigmoid_neon(x: float32x4_t) -> float32x4_t {
        let one = vdupq_n_f32(1.0);
        let neg_x = vnegq_f32(x);
        let exp_neg = fast_exp_neon(neg_x);
        let denom = vaddq_f32(one, exp_neg);
        let recip = vrecpeq_f32(denom);
        vmulq_f32(vrecpsq_f32(denom, recip), recip)
    }

    pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            // SAFETY: NEON is always available on aarch64; bounds checked by chunk iteration
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                vst1q_f32(out.as_mut_ptr().add(offset), vaddq_f32(va, vb));
            }
        }

        let start = chunks * LANE_WIDTH;
        for i in start..n {
            out[i] = a[i] + b[i];
        }
    }

    pub fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            // SAFETY: NEON always available on aarch64; bounds checked
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                vst1q_f32(out.as_mut_ptr().add(offset), vmulq_f32(va, vb));
            }
        }

        let start = chunks * LANE_WIDTH;
        for i in start..n {
            out[i] = a[i] * b[i];
        }
    }

    pub fn relu(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            // SAFETY: NEON always available on aarch64; bounds checked
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                vst1q_f32(
                    data.as_mut_ptr().add(offset),
                    vmaxq_f32(v, vdupq_n_f32(0.0)),
                );
            }
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = v.max(0.0);
        }
    }

    pub fn sigmoid(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            // SAFETY: NEON always available on aarch64; bounds checked
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                vst1q_f32(data.as_mut_ptr().add(offset), fast_sigmoid_neon(v));
            }
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = super::fast_sigmoid_scalar(*v);
        }
    }

    pub fn tanh_approx(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            // SAFETY: NEON always available on aarch64; bounds checked
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                let two = vdupq_n_f32(2.0);
                let one = vdupq_n_f32(1.0);
                let sig = fast_sigmoid_neon(vmulq_f32(two, v));
                vst1q_f32(
                    data.as_mut_ptr().add(offset),
                    vsubq_f32(vmulq_f32(two, sig), one),
                );
            }
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = 2.0 * super::fast_sigmoid_scalar(2.0 * *v) - 1.0;
        }
    }

    pub fn gelu(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            // SAFETY: NEON always available on aarch64; bounds checked
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                let sqrt2pi = vdupq_n_f32(0.797_884_6);
                let coef = vdupq_n_f32(0.044_715);
                let half = vdupq_n_f32(0.5);
                let one = vdupq_n_f32(1.0);
                let two = vdupq_n_f32(2.0);

                let v3 = vmulq_f32(vmulq_f32(v, v), v);
                let inner = vmulq_f32(sqrt2pi, vaddq_f32(v, vmulq_f32(coef, v3)));
                let tanh_val = vsubq_f32(
                    vmulq_f32(two, fast_sigmoid_neon(vmulq_f32(two, inner))),
                    one,
                );
                vst1q_f32(
                    data.as_mut_ptr().add(offset),
                    vmulq_f32(vmulq_f32(half, v), vaddq_f32(one, tanh_val)),
                );
            }
        }

        let sqrt2pi: f32 = 0.797_884_6;
        let coef: f32 = 0.044_715;
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            let inner = sqrt2pi * (*v + coef * *v * *v * *v);
            let tanh_val = 2.0 * super::fast_sigmoid_scalar(2.0 * inner) - 1.0;
            *v = 0.5 * *v * (1.0 + tanh_val);
        }
    }

    pub fn silu(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            // SAFETY: NEON always available on aarch64; bounds checked
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                let sig = fast_sigmoid_neon(v);
                vst1q_f32(data.as_mut_ptr().add(offset), vmulq_f32(v, sig));
            }
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v *= super::fast_sigmoid_scalar(*v);
        }
    }

    pub fn exp(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            // SAFETY: NEON always available on aarch64; bounds checked
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                vst1q_f32(data.as_mut_ptr().add(offset), fast_exp_neon(v));
            }
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = super::fast_exp_scalar(*v);
        }
    }

    pub fn sub(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                vst1q_f32(out.as_mut_ptr().add(offset), vsubq_f32(va, vb));
            }
        }
        for i in (chunks * LANE_WIDTH)..n {
            out[i] = a[i] - b[i];
        }
    }

    pub fn div(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                vst1q_f32(out.as_mut_ptr().add(offset), vdivq_f32(va, vb));
            }
        }
        for i in (chunks * LANE_WIDTH)..n {
            out[i] = a[i] / b[i];
        }
    }

    pub fn neg(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                vst1q_f32(data.as_mut_ptr().add(offset), vnegq_f32(v));
            }
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = -*v;
        }
    }

    pub fn abs(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                vst1q_f32(data.as_mut_ptr().add(offset), vabsq_f32(v));
            }
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = v.abs();
        }
    }

    pub fn sqrt(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                vst1q_f32(data.as_mut_ptr().add(offset), vsqrtq_f32(v));
            }
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = v.sqrt();
        }
    }

    /// Fast natural log approximation for NEON f32x4 using IEEE 754 bit decomposition.
    #[inline]
    unsafe fn fast_log_neon(x: float32x4_t) -> float32x4_t {
        // Extract exponent: (bits >> 23) - 127
        let bits = vreinterpretq_s32_f32(x);
        let exponent = vcvtq_f32_s32(vsubq_s32(
            vshrq_n_s32::<23>(vandq_s32(bits, vdupq_n_s32(0x7F80_0000u32 as i32))),
            vdupq_n_s32(127),
        ));
        // Reconstruct mantissa in [1, 2): (bits & 0x007FFFFF) | 0x3F800000
        let mantissa_bits = vorrq_s32(
            vandq_s32(bits, vdupq_n_s32(0x007F_FFFFu32 as i32)),
            vdupq_n_s32(0x3F80_0000u32 as i32),
        );
        let m = vreinterpretq_f32_s32(mantissa_bits);
        let one = vdupq_n_f32(1.0);
        let f = vsubq_f32(m, one);
        // Polynomial: ln(1+f) ≈ f*(c0 + f*(c1 + f*(c2 + f*(c3 + f*c4))))
        let c0 = vdupq_n_f32(0.999_999_7);
        let c1 = vdupq_n_f32(-0.499_999_4);
        let c2 = vdupq_n_f32(0.333_319_8);
        let c3 = vdupq_n_f32(-0.249_989_5);
        let c4 = vdupq_n_f32(0.150_198_6);
        let ln_m = vmulq_f32(
            f,
            vaddq_f32(
                c0,
                vmulq_f32(
                    f,
                    vaddq_f32(
                        c1,
                        vmulq_f32(
                            f,
                            vaddq_f32(c2, vmulq_f32(f, vaddq_f32(c3, vmulq_f32(f, c4)))),
                        ),
                    ),
                ),
            ),
        );
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        vaddq_f32(vmulq_f32(exponent, ln2), ln_m)
    }

    pub fn log(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            unsafe {
                let v = vld1q_f32(data.as_ptr().add(offset));
                vst1q_f32(data.as_mut_ptr().add(offset), fast_log_neon(v));
            }
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = super::fast_log_scalar(*v);
        }
    }

    // ── NEON horizontal reductions ──────────────────────────────────────────

    pub fn reduce_sum(data: &[f32]) -> f32 {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        // SAFETY: NEON is always available on aarch64; bounds checked by chunk iteration
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let v = vld1q_f32(data.as_ptr().add(offset));
                acc = vaddq_f32(acc, v);
            }
            let mut sum = vaddvq_f32(acc);
            // Tail elements
            for &v in &data[chunks * LANE_WIDTH..] {
                sum += v;
            }
            sum
        }
    }

    pub fn reduce_max(data: &[f32]) -> f32 {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        // SAFETY: NEON is always available on aarch64; bounds checked
        unsafe {
            let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let v = vld1q_f32(data.as_ptr().add(offset));
                acc = vmaxq_f32(acc, v);
            }
            let mut m = vmaxvq_f32(acc);
            for &v in &data[chunks * LANE_WIDTH..] {
                if v > m {
                    m = v;
                }
            }
            m
        }
    }

    pub fn reduce_min(data: &[f32]) -> f32 {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        // SAFETY: NEON is always available on aarch64; bounds checked
        unsafe {
            let mut acc = vdupq_n_f32(f32::INFINITY);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let v = vld1q_f32(data.as_ptr().add(offset));
                acc = vminq_f32(acc, v);
            }
            let mut m = vminvq_f32(acc);
            for &v in &data[chunks * LANE_WIDTH..] {
                if v < m {
                    m = v;
                }
            }
            m
        }
    }

    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / LANE_WIDTH;

        // SAFETY: NEON is always available on aarch64; bounds checked
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                acc = vfmaq_f32(acc, va, vb);
            }
            let mut sum = vaddvq_f32(acc);
            let start = chunks * LANE_WIDTH;
            for i in start..n {
                sum += a[i] * b[i];
            }
            sum
        }
    }

    pub fn softmax_inplace(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        // Step 1: Find max
        let max_val = reduce_max(data);

        // Step 2: Subtract max and compute exp (vectorized)
        // SAFETY: NEON is always available on aarch64; bounds checked by chunk iteration
        unsafe {
            let v_max = vdupq_n_f32(max_val);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let v = vld1q_f32(data.as_ptr().add(offset));
                let shifted = vsubq_f32(v, v_max);
                vst1q_f32(data.as_mut_ptr().add(offset), fast_exp_neon(shifted));
            }
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = super::fast_exp_scalar(*v - max_val);
        }

        // Step 3: Sum all exp values
        let sum = reduce_sum(data);

        // Step 4: Divide all by sum
        if sum > 0.0 {
            let inv_sum = sum.recip();
            // SAFETY: NEON always available; bounds checked
            unsafe {
                let v_inv = vdupq_n_f32(inv_sum);
                for i in 0..chunks {
                    let offset = i * LANE_WIDTH;
                    let v = vld1q_f32(data.as_ptr().add(offset));
                    vst1q_f32(data.as_mut_ptr().add(offset), vmulq_f32(v, v_inv));
                }
            }
            for v in data[chunks * LANE_WIDTH..].iter_mut() {
                *v *= inv_sum;
            }
        }
    }

    pub fn layer_norm_inplace(data: &mut [f32], scale: &[f32], bias: Option<&[f32]>, eps: f32) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        // Step 1: mean
        let mean = reduce_sum(data) / n as f32;

        // Step 2: variance (vectorized)
        let var_sum;
        // SAFETY: NEON always available; bounds checked
        unsafe {
            let v_mean = vdupq_n_f32(mean);
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let v = vld1q_f32(data.as_ptr().add(offset));
                let diff = vsubq_f32(v, v_mean);
                acc = vfmaq_f32(acc, diff, diff);
            }
            let mut vs = vaddvq_f32(acc);
            for &v in data[chunks * LANE_WIDTH..].iter() {
                let d = v - mean;
                vs += d * d;
            }
            var_sum = vs;
        }
        let inv_std = (var_sum / n as f32 + eps).sqrt().recip();

        // Step 3: normalize + scale + bias (vectorized)
        // SAFETY: NEON always available; bounds checked
        unsafe {
            let v_mean = vdupq_n_f32(mean);
            let v_inv_std = vdupq_n_f32(inv_std);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let v = vld1q_f32(data.as_ptr().add(offset));
                let mut result = vmulq_f32(vsubq_f32(v, v_mean), v_inv_std);
                let s = vld1q_f32(scale.as_ptr().add(offset % scale.len()));
                result = vmulq_f32(result, s);
                if let Some(b) = bias {
                    let vb = vld1q_f32(b.as_ptr().add(offset % b.len()));
                    result = vaddq_f32(result, vb);
                }
                vst1q_f32(data.as_mut_ptr().add(offset), result);
            }
        }
        // Tail elements
        for i in (chunks * LANE_WIDTH)..n {
            let normalized = (data[i] - mean) * inv_std;
            data[i] = normalized * scale[i % scale.len()];
            if let Some(b) = bias {
                data[i] += b[i % b.len()];
            }
        }
    }
}

// ── AVX2 (x86_64) implementations ──────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use std::arch::x86_64::*;

    const LANE_WIDTH: usize = 8; // f32x8

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
        let min_val = _mm256_set1_ps(-88.0);
        let max_val = _mm256_set1_ps(88.0);
        let x = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);

        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let ln2_hi = _mm256_set1_ps(0.693_145_75);
        let ln2_lo = _mm256_set1_ps(1.428_606_8e-6);
        let half = _mm256_set1_ps(0.5);

        // n = floor(x * log2e + 0.5)
        let n_f = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(x, log2e), half));
        // r = x - n * ln2
        let r = _mm256_sub_ps(
            _mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)),
            _mm256_mul_ps(n_f, ln2_lo),
        );

        // Polynomial: 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
        let one = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(1.0 / 6.0);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c5 = _mm256_set1_ps(1.0 / 120.0);

        let r2 = _mm256_mul_ps(r, r);
        let r3 = _mm256_mul_ps(r2, r);
        let r4 = _mm256_mul_ps(r2, r2);
        let r5 = _mm256_mul_ps(r4, r);

        let mut p = one;
        p = _mm256_add_ps(p, r);
        p = _mm256_add_ps(p, _mm256_mul_ps(r2, c2));
        p = _mm256_add_ps(p, _mm256_mul_ps(r3, c3));
        p = _mm256_add_ps(p, _mm256_mul_ps(r4, c4));
        p = _mm256_add_ps(p, _mm256_mul_ps(r5, c5));

        // 2^n via integer bit manipulation
        let n_i = _mm256_cvtps_epi32(n_f);
        let bias = _mm256_set1_epi32(127);
        let pow2n_bits = _mm256_slli_epi32(_mm256_add_epi32(n_i, bias), 23);
        let pow2n = _mm256_castsi256_ps(pow2n_bits);

        _mm256_mul_ps(p, pow2n)
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn fast_sigmoid_avx2(x: __m256) -> __m256 {
        let one = _mm256_set1_ps(1.0);
        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let exp_neg = fast_exp_avx2(neg_x);
        let denom = _mm256_add_ps(one, exp_neg);
        let recip = _mm256_rcp_ps(denom);
        let two = _mm256_set1_ps(2.0);
        _mm256_mul_ps(recip, _mm256_sub_ps(two, _mm256_mul_ps(denom, recip)))
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_add_ps(va, vb));
        }

        for i in (chunks * LANE_WIDTH)..n {
            out[i] = a[i] + b[i];
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_mul_ps(va, vb));
        }

        for i in (chunks * LANE_WIDTH)..n {
            out[i] = a[i] * b[i];
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn relu(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        let zero = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), _mm256_max_ps(v, zero));
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = v.max(0.0);
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sigmoid(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), fast_sigmoid_avx2(v));
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = super::fast_sigmoid_scalar(*v);
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn tanh_approx(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            let two = _mm256_set1_ps(2.0);
            let one = _mm256_set1_ps(1.0);
            let sig = fast_sigmoid_avx2(_mm256_mul_ps(two, v));
            _mm256_storeu_ps(
                data.as_mut_ptr().add(offset),
                _mm256_sub_ps(_mm256_mul_ps(two, sig), one),
            );
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = 2.0 * super::fast_sigmoid_scalar(2.0 * *v) - 1.0;
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn gelu(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            let sqrt2pi = _mm256_set1_ps(0.797_884_6);
            let coef = _mm256_set1_ps(0.044_715);
            let half = _mm256_set1_ps(0.5);
            let one = _mm256_set1_ps(1.0);
            let two = _mm256_set1_ps(2.0);

            let v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
            let inner = _mm256_mul_ps(sqrt2pi, _mm256_add_ps(v, _mm256_mul_ps(coef, v3)));
            let tanh_val = _mm256_sub_ps(
                _mm256_mul_ps(two, fast_sigmoid_avx2(_mm256_mul_ps(two, inner))),
                one,
            );
            _mm256_storeu_ps(
                data.as_mut_ptr().add(offset),
                _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, tanh_val)),
            );
        }

        let sqrt2pi: f32 = 0.797_884_6;
        let coef_val: f32 = 0.044_715;
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            let inner = sqrt2pi * (*v + coef_val * *v * *v * *v);
            let tanh_val = 2.0 * super::fast_sigmoid_scalar(2.0 * inner) - 1.0;
            *v = 0.5 * *v * (1.0 + tanh_val);
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn silu(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            let sig = fast_sigmoid_avx2(v);
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), _mm256_mul_ps(v, sig));
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v *= super::fast_sigmoid_scalar(*v);
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn exp(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), fast_exp_avx2(v));
        }

        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = super::fast_exp_scalar(*v);
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sub(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_sub_ps(va, vb));
        }
        for i in (chunks * LANE_WIDTH)..n {
            out[i] = a[i] - b[i];
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn div(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_div_ps(va, vb));
        }
        for i in (chunks * LANE_WIDTH)..n {
            out[i] = a[i] / b[i];
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn neg(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        let zero = _mm256_setzero_ps();
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), _mm256_sub_ps(zero, v));
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = -*v;
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn abs(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        // Mask off the sign bit: abs(x) = x & 0x7FFFFFFF
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), _mm256_and_ps(v, sign_mask));
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = v.abs();
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sqrt(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), _mm256_sqrt_ps(v));
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = v.sqrt();
        }
    }

    /// Fast natural log approximation for AVX2 using IEEE 754 bit decomposition.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn fast_log_avx2(x: __m256) -> __m256 {
        // Extract exponent: (bits >> 23) & 0xFF - 127
        let bits = _mm256_castps_si256(x);
        let exponent = _mm256_cvtepi32_ps(_mm256_sub_epi32(
            _mm256_srli_epi32(
                _mm256_and_si256(bits, _mm256_set1_epi32(0x7F80_0000u32 as i32)),
                23,
            ),
            _mm256_set1_epi32(127),
        ));
        // Reconstruct mantissa in [1, 2)
        let mantissa_bits = _mm256_or_si256(
            _mm256_and_si256(bits, _mm256_set1_epi32(0x007F_FFFFu32 as i32)),
            _mm256_set1_epi32(0x3F80_0000u32 as i32),
        );
        let m = _mm256_castsi256_ps(mantissa_bits);
        let one = _mm256_set1_ps(1.0);
        let f = _mm256_sub_ps(m, one);
        // Polynomial: ln(1+f) ≈ f*(c0 + f*(c1 + f*(c2 + f*(c3 + f*c4))))
        let c0 = _mm256_set1_ps(0.999_999_7);
        let c1 = _mm256_set1_ps(-0.499_999_4);
        let c2 = _mm256_set1_ps(0.333_319_8);
        let c3 = _mm256_set1_ps(-0.249_989_5);
        let c4 = _mm256_set1_ps(0.150_198_6);
        let inner = _mm256_add_ps(c3, _mm256_mul_ps(f, c4));
        let inner = _mm256_add_ps(c2, _mm256_mul_ps(f, inner));
        let inner = _mm256_add_ps(c1, _mm256_mul_ps(f, inner));
        let inner = _mm256_add_ps(c0, _mm256_mul_ps(f, inner));
        let ln_m = _mm256_mul_ps(f, inner);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        _mm256_add_ps(_mm256_mul_ps(exponent, ln2), ln_m)
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn log(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), fast_log_avx2(v));
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = super::fast_log_scalar(*v);
        }
    }

    // ── AVX2 horizontal reductions ──────────────────────────────────────────

    /// Horizontal sum of a __m256 register.
    /// hadd → [a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7] pattern
    /// Two hadd's then extract low and high 128-bit lanes and add.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum_avx2(v: __m256) -> f32 {
        // hadd pairs: [a0+a1, a2+a3, a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7]
        let sum1 = _mm256_hadd_ps(v, v);
        // hadd again: [a0+a1+a2+a3, ..., a4+a5+a6+a7, ...]
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        // Extract low 128 and high 128
        let lo = _mm256_castps256_ps128(sum2);
        let hi = _mm256_extractf128_ps(sum2, 1);
        let total = _mm_add_ss(lo, hi);
        _mm_cvtss_f32(total)
    }

    /// # Safety
    /// Caller must ensure AVX2 and FMA are supported.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn reduce_sum(data: &[f32]) -> f32 {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            acc = _mm256_add_ps(acc, v);
        }
        let mut sum = hsum_avx2(acc);
        for &v in &data[chunks * LANE_WIDTH..] {
            sum += v;
        }
        sum
    }

    /// Horizontal max across a __m256 register.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hmax_avx2(v: __m256) -> f32 {
        // Compare high and low 128-bit halves
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let m128 = _mm_max_ps(lo, hi); // 4 elements
                                       // Shuffle and max within 128 bits
        let shuf = _mm_movehdup_ps(m128); // [1,1,3,3]
        let m2 = _mm_max_ps(m128, shuf); // max of pairs
        let shuf2 = _mm_movehl_ps(m2, m2); // move high 64 to low
        let m1 = _mm_max_ss(m2, shuf2);
        _mm_cvtss_f32(m1)
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn reduce_max(data: &[f32]) -> f32 {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            acc = _mm256_max_ps(acc, v);
        }
        let mut m = hmax_avx2(acc);
        for &v in &data[chunks * LANE_WIDTH..] {
            if v > m {
                m = v;
            }
        }
        m
    }

    /// Horizontal min across a __m256 register.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hmin_avx2(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let m128 = _mm_min_ps(lo, hi);
        let shuf = _mm_movehdup_ps(m128);
        let m2 = _mm_min_ps(m128, shuf);
        let shuf2 = _mm_movehl_ps(m2, m2);
        let m1 = _mm_min_ss(m2, shuf2);
        _mm_cvtss_f32(m1)
    }

    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    pub unsafe fn reduce_min(data: &[f32]) -> f32 {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        let mut acc = _mm256_set1_ps(f32::INFINITY);
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            acc = _mm256_min_ps(acc, v);
        }
        let mut m = hmin_avx2(acc);
        for &v in &data[chunks * LANE_WIDTH..] {
            if v < m {
                m = v;
            }
        }
        m
    }

    /// # Safety
    /// Caller must ensure AVX2 and FMA are supported.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / LANE_WIDTH;

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        let mut sum = hsum_avx2(acc);
        let start = chunks * LANE_WIDTH;
        for i in start..n {
            sum += a[i] * b[i];
        }
        sum
    }

    /// # Safety
    /// Caller must ensure AVX2 and FMA are supported.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn softmax_inplace(data: &mut [f32]) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        // Step 1: Find max
        let max_val = reduce_max(data);

        // Step 2: Subtract max and compute exp (vectorized)
        let v_max = _mm256_set1_ps(max_val);
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            let shifted = _mm256_sub_ps(v, v_max);
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), fast_exp_avx2(shifted));
        }
        for v in data[chunks * LANE_WIDTH..].iter_mut() {
            *v = super::fast_exp_scalar(*v - max_val);
        }

        // Step 3: Sum all exp values
        let sum = reduce_sum(data);

        // Step 4: Divide all by sum
        if sum > 0.0 {
            let inv_sum = sum.recip();
            let v_inv = _mm256_set1_ps(inv_sum);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let v = _mm256_loadu_ps(data.as_ptr().add(offset));
                _mm256_storeu_ps(data.as_mut_ptr().add(offset), _mm256_mul_ps(v, v_inv));
            }
            for v in data[chunks * LANE_WIDTH..].iter_mut() {
                *v *= inv_sum;
            }
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 and FMA are supported.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn layer_norm_inplace(
        data: &mut [f32],
        scale: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) {
        let n = data.len();
        let chunks = n / LANE_WIDTH;

        // Step 1: mean
        let mean = reduce_sum(data) / n as f32;

        // Step 2: variance (vectorized)
        let v_mean = _mm256_set1_ps(mean);
        let mut var_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            let diff = _mm256_sub_ps(v, v_mean);
            var_acc = _mm256_fmadd_ps(diff, diff, var_acc);
        }
        let mut var_sum = hsum_avx2(var_acc);
        for &v in data[chunks * LANE_WIDTH..].iter() {
            let d = v - mean;
            var_sum += d * d;
        }
        let inv_std = (var_sum / n as f32 + eps).sqrt().recip();

        // Step 3: normalize + scale + bias (vectorized)
        let v_inv_std = _mm256_set1_ps(inv_std);
        for i in 0..chunks {
            let offset = i * LANE_WIDTH;
            let v = _mm256_loadu_ps(data.as_ptr().add(offset));
            let mut result = _mm256_mul_ps(_mm256_sub_ps(v, v_mean), v_inv_std);
            let s = _mm256_loadu_ps(scale.as_ptr().add(offset % scale.len()));
            result = _mm256_mul_ps(result, s);
            if let Some(b) = bias {
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset % b.len()));
                result = _mm256_add_ps(result, vb);
            }
            _mm256_storeu_ps(data.as_mut_ptr().add(offset), result);
        }
        // Tail elements
        for i in (chunks * LANE_WIDTH)..n {
            let normalized = (data[i] - mean) * inv_std;
            data[i] = normalized * scale[i % scale.len()];
            if let Some(b) = bias {
                data[i] += b[i % b.len()];
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────
