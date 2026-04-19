//! NEON (aarch64) SIMD implementations.

// Re-export scalar helpers so that the neon_impl submodule can reach them via `super::`.
// These are used inside neon_impl (cfg-gated to aarch64), so they appear unused on other targets.
#[allow(unused_imports)]
pub(super) use super::functions::{fast_exp_scalar, fast_log_scalar, fast_sigmoid_scalar};

#[cfg(target_arch = "aarch64")]
pub(super) mod neon_impl {
    use std::arch::aarch64::*;
    const LANE_WIDTH: usize = 4;
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
        let n_f = vrndmq_f32(vaddq_f32(vmulq_f32(x, log2e), half));
        let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n_f, ln2_hi)), vmulq_f32(n_f, ln2_lo));
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
        let bits = vreinterpretq_s32_f32(x);
        let exponent = vcvtq_f32_s32(vsubq_s32(
            vshrq_n_s32::<23>(vandq_s32(bits, vdupq_n_s32(0x7F80_0000u32 as i32))),
            vdupq_n_s32(127),
        ));
        let mantissa_bits = vorrq_s32(
            vandq_s32(bits, vdupq_n_s32(0x007F_FFFFu32 as i32)),
            vdupq_n_s32(0x3F80_0000u32 as i32),
        );
        let m = vreinterpretq_f32_s32(mantissa_bits);
        let one = vdupq_n_f32(1.0);
        let f = vsubq_f32(m, one);
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
    pub fn reduce_sum(data: &[f32]) -> f32 {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let offset = i * LANE_WIDTH;
                let v = vld1q_f32(data.as_ptr().add(offset));
                acc = vaddq_f32(acc, v);
            }
            let mut sum = vaddvq_f32(acc);
            for &v in &data[chunks * LANE_WIDTH..] {
                sum += v;
            }
            sum
        }
    }
    pub fn reduce_max(data: &[f32]) -> f32 {
        let n = data.len();
        let chunks = n / LANE_WIDTH;
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
        let max_val = reduce_max(data);
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
        let sum = reduce_sum(data);
        if sum > 0.0 {
            let inv_sum = sum.recip();
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
        let mean = reduce_sum(data) / n as f32;
        let var_sum;
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
        for i in (chunks * LANE_WIDTH)..n {
            let normalized = (data[i] - mean) * inv_std;
            data[i] = normalized * scale[i % scale.len()];
            if let Some(b) = bias {
                data[i] += b[i % b.len()];
            }
        }
    }
}
