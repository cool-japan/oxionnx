//! Convolution, pooling, and related spatial operations.
//!
//! This module is organized into focused submodules:
//! - `conv2d` — 2D convolution dispatch (im2col+GEMM, 1×1 fast path, Winograd)
//! - `im2col` — im2col transformation variants (plain, cache-blocked, SIMD)
//! - `winograd` — Winograd F(2,3) algorithm for 3×3 kernels
//! - `pooling` — max/avg/global pooling and transposed convolution

mod conv2d;
mod im2col;
mod pooling;
mod winograd;

#[cfg(test)]
mod tests;

// ── Public API re-exports ────────────────────────────────────────────────────

pub use conv2d::conv2d;
pub(crate) use conv2d::{compute_conv2d_out_shape, conv2d_into};
#[cfg(feature = "simd")]
pub use im2col::im2col_simd_stride1;
pub use im2col::pack_weights_panel;
pub use pooling::{avg_pool2d, conv_transpose2d, global_avg_pool, global_max_pool, max_pool2d};
pub(crate) use pooling::{compute_conv_transpose2d_out_shape, conv_transpose2d_into};
pub use winograd::conv2d_winograd_f2x3;
