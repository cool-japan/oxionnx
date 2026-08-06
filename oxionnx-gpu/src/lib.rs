//! oxionnx-gpu — wgpu GPU compute backend for oxionnx.
//!
//! Every entry point returns `Option<_>`, where `None` means "this crate
//! declines; run the CPU operator instead". A missing adapter, a tensor larger
//! than the device can bind, a dispatch wider than the device allows, or a
//! driver error all degrade to CPU execution rather than failing the session.
//!
//! # Platforms
//!
//! **Native** (Vulkan / Metal / DX12): [`GpuContext::try_new`] blocks on
//! initialization via `pollster`. The device is requested with the *adapter's*
//! limits rather than the WebGPU baseline, so a GPU that can bind gigabytes is
//! not capped at the 128 MiB / 256 MiB defaults.
//!
//! **wasm32 / WebGPU**: not supported. Both [`GpuContext::try_new`] and
//! [`GpuContext::try_new_async`] return `None` in the browser, so no work is
//! submitted and the session runs on the CPU. The kernels here are synchronous
//! and end in a blocking read-back, which WebGPU cannot do; supporting the
//! browser needs an `async` variant of every `gpu_*` entry point, not a flag.
//! See [`GpuContext::try_new_async`] for the full rationale.

pub mod compute;
pub mod context;
pub mod device_guard;
pub mod shaders;

pub use compute::{gpu_conv2d, gpu_matmul, gpu_matmul_tiled};
pub use context::{GpuBufferPool, GpuContext, DEFAULT_POOL_BYTE_BUDGET};
pub use device_guard::GpuLimits;
pub use shaders::{
    gpu_abs, gpu_add, gpu_batch_norm, gpu_exp, gpu_gelu, gpu_layer_norm, gpu_layer_norm_axis,
    gpu_leaky_relu, gpu_leaky_relu_alpha, gpu_log, gpu_mul, gpu_neg, gpu_reduce_max,
    gpu_reduce_mean, gpu_reduce_min, gpu_reduce_sum, gpu_relu, gpu_sigmoid, gpu_silu, gpu_softmax,
    gpu_sqrt, gpu_tanh, gpu_transpose,
};
