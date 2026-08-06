//! Convolution, pooling, and related spatial operations.
//!
//! Every operator here is **rank-generic**: `Conv`, `ConvTranspose`, `MaxPool`
//! and `AveragePool` all accept `[N, C, d_0, …, d_{r-1}]` for any spatial rank
//! `r >= 1`, so 1D (audio / TCN), 2D (vision) and 3D (video / volumetric)
//! models run through the same code path with `auto_pad`, `ceil_mode` and
//! `dilations` honoured at every rank.
//!
//! This module is organized into focused submodules:
//! - `spatial` — shared N-D geometry: `auto_pad`, attribute validation, extents
//! - `conv2d` — the specialised rank-2 kernel (im2col+GEMM, 1×1, Winograd)
//! - `conv_nd` — rank dispatch: 1D lowering to the 2D kernel, generic rank ≥ 3
//! - `im2col` — im2col transformation variants (plain, cache-blocked, SIMD)
//! - `winograd` — Winograd F(2,3) algorithm for 3×3 kernels
//! - `pooling` — the single max/avg pooling kernel plus global pooling
//! - `transpose` — transposed convolution (rank-2 specialisation + generic)

mod conv2d;
mod conv_nd;
mod im2col;
mod pooling;
pub(crate) mod spatial;
mod transpose;
mod winograd;

#[cfg(test)]
mod tests;

// ── Public API re-exports ────────────────────────────────────────────────────

pub use conv2d::conv2d;
pub use conv_nd::conv;
pub(crate) use conv_nd::{conv_into, ConvParams};
#[cfg(feature = "simd")]
pub use im2col::im2col_simd_stride1;
pub use im2col::pack_weights_panel;
pub use pooling::{avg_pool2d, global_avg_pool, global_max_pool, max_pool2d};
pub(crate) use pooling::{avg_pool_into, max_pool_into, PoolGeometry};
pub use transpose::{conv_transpose, conv_transpose2d};
pub(crate) use transpose::{conv_transpose_into, ConvTransposeParams};
pub use winograd::conv2d_winograd_f2x3;
