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

// ── Dispatch helpers ────────────────────────────────────────────────────────

enum Op {
    Add,
    Mul,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Exp,
}

fn dispatch_binary(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    #[cfg(target_arch = "aarch64")]
    {
        match op {
            Op::Add => neon_impl::add(a, b, out),
            Op::Mul => neon_impl::mul(a, b, out),
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
        _ => {}
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
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-2;

    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: expected {b}, got {a}, diff={}",
            (a - b).abs()
        );
    }

    #[test]
    fn test_simd_add() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..33).map(|i| i as f32 * 0.3 + 1.0).collect();
        let mut out = vec![0.0f32; 33];
        simd_add(&a, &b, &mut out);

        for i in 0..33 {
            assert_close(out[i], a[i] + b[i], 1e-6, "simd_add");
        }
    }

    #[test]
    fn test_simd_mul() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..33).map(|i| i as f32 * 0.3 + 1.0).collect();
        let mut out = vec![0.0f32; 33];
        simd_mul(&a, &b, &mut out);

        for i in 0..33 {
            assert_close(out[i], a[i] * b[i], 1e-6, "simd_mul");
        }
    }

    #[test]
    fn test_simd_relu() {
        let mut data: Vec<f32> = vec![-3.0, -1.5, -0.1, 0.0, 0.1, 1.5, 3.0, -100.0, 100.0];
        simd_relu(&mut data);

        assert_close(data[0], 0.0, 1e-6, "relu neg");
        assert_close(data[1], 0.0, 1e-6, "relu neg");
        assert_close(data[2], 0.0, 1e-6, "relu neg");
        assert_close(data[3], 0.0, 1e-6, "relu zero");
        assert_close(data[4], 0.1, 1e-6, "relu pos");
        assert_close(data[5], 1.5, 1e-6, "relu pos");
        assert_close(data[6], 3.0, 1e-6, "relu pos");
        assert_close(data[7], 0.0, 1e-6, "relu large neg");
        assert_close(data[8], 100.0, 1e-6, "relu large pos");
    }

    #[test]
    fn test_simd_sigmoid() {
        let mut data: Vec<f32> = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        simd_sigmoid(&mut data);

        for &v in &data {
            assert!(v >= 0.0 && v <= 1.0, "sigmoid out of range: {v}");
        }

        assert_close(data[3], 0.5, TOL, "sigmoid(0)");
        assert_close(data[0] + data[6], 1.0, TOL, "sigmoid symmetry");
        assert_close(data[1] + data[5], 1.0, TOL, "sigmoid symmetry");
        assert_close(data[2] + data[4], 1.0, TOL, "sigmoid symmetry");
    }

    #[test]
    fn test_simd_tanh() {
        let mut data: Vec<f32> = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        simd_tanh(&mut data);

        for &v in &data {
            assert!(v >= -1.0 - TOL && v <= 1.0 + TOL, "tanh out of range: {v}");
        }

        assert_close(data[3], 0.0, TOL, "tanh(0)");
        assert_close(data[0] + data[6], 0.0, TOL, "tanh odd");
        assert_close(data[1] + data[5], 0.0, TOL, "tanh odd");
    }

    #[test]
    fn test_simd_gelu() {
        let mut data: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        simd_gelu(&mut data);

        assert_close(data[2], 0.0, TOL, "gelu(0)");
        assert_close(data[3], 0.8412, 0.05, "gelu(1)");
        assert_close(data[1], -0.1588, 0.05, "gelu(-1)");
    }

    #[test]
    fn test_simd_silu() {
        let mut data: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        simd_silu(&mut data);

        assert_close(data[2], 0.0, TOL, "silu(0)");
        assert_close(data[3], 0.7311, 0.05, "silu(1)");
    }

    #[test]
    fn test_simd_exp() {
        let mut data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        simd_exp(&mut data);

        assert_close(data[0], 1.0, TOL, "exp(0)");
        assert_close(data[1], std::f32::consts::E, 0.1, "exp(1)");
        assert_close(data[2], 1.0 / std::f32::consts::E, 0.05, "exp(-1)");
    }

    #[test]
    fn test_simd_small_arrays() {
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 4.0];
        let mut out = vec![0.0f32; 2];
        simd_add(&a, &b, &mut out);
        assert_close(out[0], 4.0, 1e-6, "small add 0");
        assert_close(out[1], 6.0, 1e-6, "small add 1");

        let mut small = vec![0.5f32];
        simd_relu(&mut small);
        assert_close(small[0], 0.5, 1e-6, "small relu");

        let mut small = vec![-0.5f32];
        simd_relu(&mut small);
        assert_close(small[0], 0.0, 1e-6, "small relu neg");
    }

    #[test]
    fn test_simd_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        simd_add(&a, &b, &mut out);
        simd_mul(&a, &b, &mut out);
        assert!(out.is_empty());

        let mut empty: Vec<f32> = vec![];
        simd_relu(&mut empty);
        simd_sigmoid(&mut empty);
        simd_tanh(&mut empty);
        simd_gelu(&mut empty);
        simd_silu(&mut empty);
        simd_exp(&mut empty);
        assert!(empty.is_empty());
    }
}
