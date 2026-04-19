//! MatMul-related fusion passes:
//! - MatMul + Add → Gemm
//! - LayerNorm pattern fusion
//! - MatMul + Transpose → transposed Gemm
//! - Add(bias) + MatMul → Gemm with pre-computed bias

mod add_matmul;
mod layer_norm;
mod matmul_add;
mod matmul_transpose;

#[cfg(test)]
mod tests;

pub use add_matmul::fuse_add_matmul_to_gemm;
pub use layer_norm::fuse_layer_norm;
pub use matmul_add::fuse_matmul_add;
pub use matmul_transpose::fuse_matmul_transpose;
