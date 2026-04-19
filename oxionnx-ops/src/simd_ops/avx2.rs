//! AVX2 (x86_64) SIMD implementations.

// Re-export scalar helpers so that the avx2_impl submodule can reach them via `super::`.
// These are used inside avx2_impl (cfg-gated to x86_64), so they appear unused on other targets.
#[allow(unused_imports)]
pub(super) use super::functions::{fast_exp_scalar, fast_log_scalar, fast_sigmoid_scalar};

#[cfg(target_arch = "x86_64")]
pub(super) mod avx2_impl {
    use std::arch::x86_64::*;
    const LANE_WIDTH: usize = 8;
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
        let n_f = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(x, log2e), half));
        let r = _mm256_sub_ps(
            _mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)),
            _mm256_mul_ps(n_f, ln2_lo),
        );
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
        let bits = _mm256_castps_si256(x);
        let exponent = _mm256_cvtepi32_ps(_mm256_sub_epi32(
            _mm256_srli_epi32(
                _mm256_and_si256(bits, _mm256_set1_epi32(0x7F80_0000u32 as i32)),
                23,
            ),
            _mm256_set1_epi32(127),
        ));
        let mantissa_bits = _mm256_or_si256(
            _mm256_and_si256(bits, _mm256_set1_epi32(0x007F_FFFFu32 as i32)),
            _mm256_set1_epi32(0x3F80_0000u32 as i32),
        );
        let m = _mm256_castsi256_ps(mantissa_bits);
        let one = _mm256_set1_ps(1.0);
        let f = _mm256_sub_ps(m, one);
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
    /// Horizontal sum of a __m256 register.
    /// hadd → [a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7] pattern
    /// Two hadd's then extract low and high 128-bit lanes and add.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is supported.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum_avx2(v: __m256) -> f32 {
        let sum1 = _mm256_hadd_ps(v, v);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
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
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let m128 = _mm_max_ps(lo, hi);
        let shuf = _mm_movehdup_ps(m128);
        let m2 = _mm_max_ps(m128, shuf);
        let shuf2 = _mm_movehl_ps(m2, m2);
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
        let max_val = reduce_max(data);
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
        let sum = reduce_sum(data);
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
        let mean = reduce_sum(data) / n as f32;
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
        for i in (chunks * LANE_WIDTH)..n {
            let normalized = (data[i] - mean) * inv_std;
            data[i] = normalized * scale[i % scale.len()];
            if let Some(b) = bias {
                data[i] += b[i % b.len()];
            }
        }
    }
}
