//! GPU compute shader dispatch functions for softmax, element-wise ops, reductions,
//! normalization, and transpose.

mod common;
mod elementwise;
mod normalization;
mod reduction;
mod softmax;
mod transpose;

#[cfg(test)]
mod tests;

pub use elementwise::{
    gpu_abs, gpu_add, gpu_exp, gpu_gelu, gpu_leaky_relu, gpu_leaky_relu_alpha, gpu_log, gpu_mul,
    gpu_neg, gpu_relu, gpu_sigmoid, gpu_silu, gpu_sqrt, gpu_tanh, DEFAULT_LEAKY_RELU_ALPHA,
};
pub use normalization::{gpu_batch_norm, gpu_layer_norm, gpu_layer_norm_axis};
pub use reduction::{gpu_reduce_max, gpu_reduce_mean, gpu_reduce_min, gpu_reduce_sum};
pub use softmax::gpu_softmax;
pub use transpose::gpu_transpose;
