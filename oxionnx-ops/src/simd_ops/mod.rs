//! SIMD-accelerated element-wise operations.
//!
//! Enabled with the `simd` feature flag. Provides accelerated versions of
//! add, mul, relu, sigmoid, tanh, gelu, silu, and exp element-wise ops.
//! Falls back to scalar loops when SIMD is not available.

pub(crate) mod avx2;
pub(crate) mod functions;
pub(crate) mod neon;
pub(crate) mod types;

// Re-export the public dispatch API so callers use `simd_ops::simd_add` etc.
pub use functions::{
    simd_abs, simd_add, simd_div, simd_dot_product, simd_exp, simd_gelu, simd_layer_norm,
    simd_layer_norm_strided, simd_log, simd_mul, simd_neg, simd_reduce_max, simd_reduce_mean,
    simd_reduce_min, simd_reduce_sum, simd_relu, simd_sigmoid, simd_silu, simd_softmax_inplace,
    simd_softmax_strided, simd_sqrt, simd_sub, simd_tanh,
};

// AVX2 intrinsics only exist on x86_64 (see `avx2::avx2_impl`'s own
// `#[cfg(target_arch = "x86_64")]`), so the regression test that exercises
// them directly is gated the same way.
#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
