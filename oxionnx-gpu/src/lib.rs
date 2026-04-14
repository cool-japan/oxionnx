//! oxionnx-gpu — wgpu GPU compute backend for oxionnx.
//!
//! On native targets, synchronous GPU initialization is provided via `pollster`.
//! On `wasm32` targets, `pollster` is unavailable (no blocking in the browser),
//! so callers must use the async API ([`GpuContext::try_new_async`]).

pub mod compute;
pub mod context;
pub mod shaders;

pub use compute::{gpu_conv2d, gpu_matmul, gpu_matmul_tiled};
pub use context::{GpuBufferPool, GpuContext, GpuTensorTracker};
pub use shaders::{
    gpu_abs, gpu_add, gpu_batch_norm, gpu_exp, gpu_gelu, gpu_layer_norm, gpu_leaky_relu, gpu_log,
    gpu_mul, gpu_neg, gpu_reduce_max, gpu_reduce_mean, gpu_reduce_min, gpu_reduce_sum, gpu_relu,
    gpu_sigmoid, gpu_silu, gpu_softmax, gpu_sqrt, gpu_tanh, gpu_transpose,
};
