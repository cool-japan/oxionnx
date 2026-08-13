//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Op;

// Test-only instrumentation for the runtime AVX2 dispatch mechanism.
//
// Every numeric SIMD-vs-scalar test in this crate is blind to a broken
// `is_x86_feature_detected!("avx2")` guard for pure elementwise ops (add,
// mul, relu, ...): AVX2 and scalar are bit-identical for those (no
// reduction/reassociation involved), so a dispatcher that silently always
// fell through to the scalar arm would still pass every existing output
// comparison while quietly discarding the AVX2 speedup on every AVX2-capable
// machine. For the reduction-shaped ops (`simd_reduce_sum`/`_max`/`_min`,
// `simd_dot_product`, `simd_softmax_inplace`, `simd_layer_norm`) the same
// blindness risk exists from the opposite direction: their scalar fallbacks
// use compensated (Kahan) summation while the AVX2 kernels reduce
// lane-parallel, so a small numeric difference between the two is *expected*
// and already tolerance-checked elsewhere -- which makes a silent
// always-scalar dispatch bug even easier to miss by output comparison alone
// (the "difference" a broken dispatcher would produce is simply zero, which
// looks like success). This counter lets `simd_ops::tests` assert the AVX2
// arm was *actually entered*, independent of what it computed, at all eight
// of this module's `is_x86_feature_detected!("avx2")` call sites: the two
// shared chokepoints (`dispatch_binary`, `dispatch_unary`) and the six
// standalone reduction/normalization functions listed above.
//
// `thread_local!` (not a shared `AtomicUsize`) deliberately, so the counter
// is immune to cross-talk from unrelated `#[test]` functions running
// concurrently on other threads elsewhere in this crate's test binary; each
// `#[test]` fn gets its own thread under the default libtest harness, so
// reading "did *my* call bump *my* thread's counter" is race-free without
// any locking. `#[cfg(test)]`-only: does not exist in a shipped build.
#[cfg(test)]
thread_local! {
    pub(crate) static AVX2_DISPATCH_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// SIMD-accelerated element-wise addition: `out[i] = a[i] + b[i]`
pub fn simd_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    dispatch_binary(a, b, out, Op::Add);
}
/// SIMD-accelerated element-wise multiplication: `out[i] = a[i] * b[i]`
pub fn simd_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    dispatch_binary(a, b, out, Op::Mul);
}
/// SIMD-accelerated in-place ReLU: `data[i] = max(data[i], 0)`
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
/// SIMD-accelerated element-wise subtraction: `out[i] = a[i] - b[i]`
pub fn simd_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    dispatch_binary(a, b, out, Op::Sub);
}
/// SIMD-accelerated element-wise division: `out[i] = a[i] / b[i]`
pub fn simd_div(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    dispatch_binary(a, b, out, Op::Div);
}
/// SIMD-accelerated in-place negation: `data[i] = -data[i]`
pub fn simd_neg(data: &mut [f32]) {
    dispatch_unary(data, Op::Neg);
}
/// SIMD-accelerated in-place absolute value: `data[i] = |data[i]|`
pub fn simd_abs(data: &mut [f32]) {
    dispatch_unary(data, Op::Abs);
}
/// SIMD-accelerated in-place square root: `data[i] = sqrt(data[i])`
pub fn simd_sqrt(data: &mut [f32]) {
    dispatch_unary(data, Op::Sqrt);
}
/// SIMD-accelerated in-place natural logarithm: `data[i] = ln(data[i])`
pub fn simd_log(data: &mut [f32]) {
    dispatch_unary(data, Op::Log);
}
/// SIMD-accelerated sum reduction over a flat slice.
///
/// Returns 0.0 for an empty slice.
pub fn simd_reduce_sum(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::neon_impl::reduce_sum(data)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            #[cfg(test)]
            AVX2_DISPATCH_HITS.with(|c| c.set(c.get() + 1));
            return unsafe { super::avx2::avx2_impl::reduce_sum(data) };
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        return scalar_reduce_sum(data);
    }
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
        super::neon::neon_impl::reduce_max(data)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            #[cfg(test)]
            AVX2_DISPATCH_HITS.with(|c| c.set(c.get() + 1));
            return unsafe { super::avx2::avx2_impl::reduce_max(data) };
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
        super::neon::neon_impl::reduce_min(data)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            #[cfg(test)]
            AVX2_DISPATCH_HITS.with(|c| c.set(c.get() + 1));
            return unsafe { super::avx2::avx2_impl::reduce_min(data) };
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
        super::neon::neon_impl::dot_product(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            #[cfg(test)]
            AVX2_DISPATCH_HITS.with(|c| c.set(c.get() + 1));
            return unsafe { super::avx2::avx2_impl::dot_product(a, b) };
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
/// SIMD-accelerated in-place softmax over the entire slice.
///
/// Computes softmax: `data[i] = exp(data[i] - max) / sum(exp(data - max))`
pub fn simd_softmax_inplace(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::neon_impl::softmax_inplace(data);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            #[cfg(test)]
            AVX2_DISPATCH_HITS.with(|c| c.set(c.get() + 1));
            unsafe { super::avx2::avx2_impl::softmax_inplace(data) };
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
        super::neon::neon_impl::layer_norm_inplace(data, scale, bias, eps);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            #[cfg(test)]
            AVX2_DISPATCH_HITS.with(|c| c.set(c.get() + 1));
            unsafe { super::avx2::avx2_impl::layer_norm_inplace(data, scale, bias, eps) };
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
fn dispatch_binary(a: &[f32], b: &[f32], out: &mut [f32], op: Op) {
    #[cfg(target_arch = "aarch64")]
    {
        match op {
            Op::Add => super::neon::neon_impl::add(a, b, out),
            Op::Mul => super::neon::neon_impl::mul(a, b, out),
            Op::Sub => super::neon::neon_impl::sub(a, b, out),
            Op::Div => super::neon::neon_impl::div(a, b, out),
            _ => scalar_binary(a, b, out, op),
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            #[cfg(test)]
            AVX2_DISPATCH_HITS.with(|c| c.set(c.get() + 1));
            unsafe {
                match op {
                    Op::Add => super::avx2::avx2_impl::add(a, b, out),
                    Op::Mul => super::avx2::avx2_impl::mul(a, b, out),
                    Op::Sub => super::avx2::avx2_impl::sub(a, b, out),
                    Op::Div => super::avx2::avx2_impl::div(a, b, out),
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
            Op::Relu => super::neon::neon_impl::relu(data),
            Op::Sigmoid => super::neon::neon_impl::sigmoid(data),
            Op::Tanh => super::neon::neon_impl::tanh_approx(data),
            Op::Gelu => super::neon::neon_impl::gelu(data),
            Op::Silu => super::neon::neon_impl::silu(data),
            Op::Exp => super::neon::neon_impl::exp(data),
            Op::Neg => super::neon::neon_impl::neg(data),
            Op::Abs => super::neon::neon_impl::abs(data),
            Op::Sqrt => super::neon::neon_impl::sqrt(data),
            Op::Log => super::neon::neon_impl::log(data),
            _ => scalar_unary(data, op),
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            #[cfg(test)]
            AVX2_DISPATCH_HITS.with(|c| c.set(c.get() + 1));
            unsafe {
                match op {
                    Op::Relu => super::avx2::avx2_impl::relu(data),
                    Op::Sigmoid => super::avx2::avx2_impl::sigmoid(data),
                    Op::Tanh => super::avx2::avx2_impl::tanh_approx(data),
                    Op::Gelu => super::avx2::avx2_impl::gelu(data),
                    Op::Silu => super::avx2::avx2_impl::silu(data),
                    Op::Exp => super::avx2::avx2_impl::exp(data),
                    Op::Neg => super::avx2::avx2_impl::neg(data),
                    Op::Abs => super::avx2::avx2_impl::abs(data),
                    Op::Sqrt => super::avx2::avx2_impl::sqrt(data),
                    Op::Log => super::avx2::avx2_impl::log(data),
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
/// Fast exp approximation (scalar) using range reduction.
///
/// Uses exp(x) = 2^n * exp(r) where x = n*ln(2) + r, |r| <= ln(2)/2.
/// The reduced-range exp(r) is computed with a degree-5 minimax polynomial.
#[inline]
pub(crate) fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    let n = (x * LOG2E + 0.5).floor();
    let r = x - n * LN2_HI - n * LN2_LO;
    let r2 = r * r;
    let p = 1.0
        + r
        + r2 * 0.5
        + r2 * r * (1.0 / 6.0)
        + r2 * r2 * (1.0 / 24.0)
        + r2 * r2 * r * (1.0 / 120.0);
    let n_i = n as i32;
    let bits = ((n_i + 127) as u32) << 23;
    let pow2n = f32::from_bits(bits);
    p * pow2n
}
#[inline]
pub(crate) fn fast_sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp_scalar(-x))
}
/// Fast natural log approximation (scalar) using IEEE 754 bit decomposition.
///
/// Extracts exponent and mantissa from the float's bit representation, then uses
/// a polynomial correction on the mantissa.
#[inline]
pub(crate) fn fast_log_scalar(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (bits & 0x007F_FFFF) | 0x3F80_0000;
    let m = f32::from_bits(mantissa_bits);
    let f = m - 1.0;
    let ln_m = f
        * (0.999_999_7
            + f * (-0.499_999_4 + f * (0.333_319_8 + f * (-0.249_989_5 + f * 0.150_198_6))));
    (exponent as f32) * std::f32::consts::LN_2 + ln_m
}
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
#[allow(dead_code)]
fn scalar_reduce_sum(data: &[f32]) -> f32 {
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
