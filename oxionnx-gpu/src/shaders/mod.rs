//! GPU compute shader dispatch functions for softmax, element-wise ops, reductions,
//! normalization, and transpose.

mod broadcast_binary;
mod common;
mod conv2d;
mod elementwise;
/// [w2-f16] Derives each half-precision kernel from its `f32` source.
mod f16_variant;
mod gemm;
mod instance_norm;
mod kernel_support;
mod normalization;
mod pad;
mod prelu;
mod reduction;
mod resize;
mod softmax;
mod transpose;

#[cfg(test)]
mod tests;

pub use broadcast_binary::{
    gpu_broadcast_add, gpu_broadcast_div, gpu_broadcast_mul, gpu_broadcast_sub,
};
pub use conv2d::{gpu_conv2d_implicit, ConvActivation};
pub use elementwise::{
    gpu_abs, gpu_add, gpu_exp, gpu_gelu, gpu_leaky_relu, gpu_leaky_relu_alpha, gpu_log, gpu_mul,
    gpu_neg, gpu_relu, gpu_sigmoid, gpu_silu, gpu_sqrt, gpu_tanh, DEFAULT_LEAKY_RELU_ALPHA,
};
pub use gemm::gpu_gemm_nt;
pub use instance_norm::gpu_instance_norm;
pub use normalization::{gpu_batch_norm, gpu_layer_norm, gpu_layer_norm_axis};
pub use pad::{gpu_pad, PadMode};
pub use prelu::gpu_prelu;
pub use reduction::{gpu_reduce_max, gpu_reduce_mean, gpu_reduce_min, gpu_reduce_sum};
pub use resize::{gpu_resize_bilinear_pytorch_half_pixel, gpu_resize_nearest_asymmetric};
pub use softmax::gpu_softmax;
pub use transpose::gpu_transpose;

// The `async` half of the same surface — the actual implementations, and the
// only ones that produce a value in a browser. See the crate docs.
pub use broadcast_binary::{
    gpu_broadcast_add_async, gpu_broadcast_div_async, gpu_broadcast_mul_async,
    gpu_broadcast_sub_async,
};
pub use conv2d::{gpu_conv2d_implicit_async, gpu_conv2d_implicit_resident_async};
pub use elementwise::{
    gpu_abs_async, gpu_add_async, gpu_exp_async, gpu_gelu_async, gpu_leaky_relu_alpha_async,
    gpu_leaky_relu_async, gpu_log_async, gpu_mul_async, gpu_neg_async, gpu_relu_async,
    gpu_sigmoid_async, gpu_silu_async, gpu_sqrt_async, gpu_tanh_async,
};
pub use gemm::{gpu_gemm_nt_async, gpu_gemm_nt_resident_async};
pub use instance_norm::gpu_instance_norm_async;
pub use normalization::{gpu_batch_norm_async, gpu_layer_norm_async, gpu_layer_norm_axis_async};
pub use pad::gpu_pad_async;
pub use prelu::gpu_prelu_async;
pub use reduction::{
    gpu_reduce_max_async, gpu_reduce_mean_async, gpu_reduce_min_async, gpu_reduce_sum_async,
};
pub use resize::{
    gpu_resize_bilinear_pytorch_half_pixel_async, gpu_resize_nearest_asymmetric_async,
};
pub use softmax::gpu_softmax_async;
pub use transpose::gpu_transpose_async;

// The residency-aware half of the same surface: entry points whose operands may
// already be on the device and whose result may stay there. Each one *is* the
// body its two siblings above delegate to — see `context::activation`.
pub use broadcast_binary::{gpu_broadcast_placed_async, BroadcastKind};
pub use conv2d::gpu_conv2d_implicit_placed_async;
pub use elementwise::{
    gpu_abs_placed_async, gpu_add_placed_async, gpu_exp_placed_async, gpu_gelu_placed_async,
    gpu_leaky_relu_placed_async, gpu_log_placed_async, gpu_mul_placed_async, gpu_neg_placed_async,
    gpu_relu_placed_async, gpu_sigmoid_placed_async, gpu_silu_placed_async, gpu_sqrt_placed_async,
    gpu_tanh_placed_async,
};
pub use gemm::gpu_gemm_nt_placed_async;
pub use instance_norm::gpu_instance_norm_placed_async;
pub use pad::gpu_pad_placed_async;
pub use prelu::gpu_prelu_placed_async;
pub use resize::{gpu_resize_placed_async, ResizeKind};

// [w5] The standalone kernel batch's `build_*_pipeline` helpers used to be
// re-exported here, unused, so that a later wave could hoist their compiles
// onto `GpuContext` as a call-site move. That hoist has happened: every one of
// them now memoizes into `GpuContext`'s own `pipeline_cache`
// (`kernel_support::build_pipeline`), so each stays private to the module that
// owns its shader source and there is nothing left to re-export.
