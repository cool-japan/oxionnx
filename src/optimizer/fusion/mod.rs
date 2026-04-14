//! Fusion optimization passes: MatMul+Add, Conv+BatchNorm, Conv+Relu,
//! Conv+ReLU6, SiLU (Mul+Sigmoid), Div+Sqrt→Rsqrt, standalone BatchNorm
//! folding, LayerNorm pattern, consecutive Transpose/Reshape cancellation,
//! MatMul+Transpose fusion, Add+MatMul→Gemm fusion, Conv+Add+Relu (ResNet),
//! Gather+Gather composition, Dropout elimination, Transpose+Reshape simplification.

mod conv;
mod matmul;
mod simplify;

pub use conv::{
    fold_batch_norm_inference, fuse_conv_add_relu, fuse_conv_batchnorm,
    fuse_conv_clip_to_conv_relu6, fuse_conv_relu,
};
pub use matmul::{
    fuse_add_matmul_to_gemm, fuse_layer_norm, fuse_matmul_add, fuse_matmul_transpose,
};
pub use simplify::{
    cancel_consecutive_reshape, cancel_consecutive_transpose, eliminate_dropout_inference,
    fuse_div_sqrt_to_rsqrt, fuse_gather_composition, fuse_mul_sigmoid_to_silu,
    simplify_transpose_reshape,
};
