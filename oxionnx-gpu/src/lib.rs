//! oxionnx-gpu — wgpu GPU compute backend for oxionnx.
//!
//! Every entry point returns `Option<_>`, where `None` means "this crate
//! declines; run the CPU operator instead". A missing adapter, a tensor larger
//! than the device can bind, a dispatch wider than the device allows, or a
//! driver error all degrade to CPU execution rather than failing the session.
//!
//! # Sync and async
//!
//! Every kernel exists twice, from one body: `gpu_x_async` **is** the
//! implementation, and `gpu_x` is a `pollster::block_on` wrapper around it.
//! Pick by target, not by taste:
//!
//! * **Native** — use either. The synchronous form blocks the calling thread on
//!   the GPU fence exactly as it always has, and the async form completes in a
//!   single `poll` (its read-back *is* the blocking one), so awaiting it in a
//!   real executor would block that executor's thread. Async is there for API
//!   parity and tests, not for concurrency.
//! * **wasm32** — you must use `gpu_x_async`. The synchronous forms return
//!   `None` there, deliberately: blocking a browser's only thread on a GPU
//!   fence deadlocks the page, so declining sends the node to the CPU operator
//!   instead of hanging.
//!
//! Two GPU calls from this crate must never be in flight on one device at the
//! same time — wgpu error scopes are a per-thread LIFO stack that the native
//! backend panics on if popped out of order. The engine's async run loop is
//! sequential by construction; do not `join!` two `gpu_*_async` calls yourself.
//!
//! # Platforms
//!
//! **Native** (Vulkan / Metal / DX12): [`GpuContext::try_new`] blocks on
//! initialization via `pollster`. The device is requested with the *adapter's*
//! limits rather than the WebGPU baseline, so a GPU that can bind gigabytes is
//! not capped at the 128 MiB / 256 MiB defaults.
//!
//! **wasm32 / WebGPU**: supported through [`GpuContext::try_new_async`], which
//! requests a [`wgpu::Backends::BROWSER_WEBGPU`] adapter. The browser read-back
//! awaits the `mapAsync` promise instead of polling the device (`Device::poll`
//! is a no-op on that backend and would hang), and validation errors are
//! surfaced by awaiting the error-scope pop rather than discarding it.
//! [`GpuContext::try_new`] still returns `None` there — see it for why.
//! Note that a browser device is capped at the WebGPU baseline limits
//! (typically 128 MiB per storage binding), so large tensors decline to the CPU
//! by the usual `checked_storage_bytes` route.

pub mod compute;
pub mod context;
pub mod device_guard;
pub mod shaders;

pub use compute::gpu_conv2d_fused_placed_async;
pub use compute::{
    gpu_conv2d, gpu_conv2d_async, gpu_conv2d_fused, gpu_conv2d_fused_async,
    gpu_conv2d_fused_resident_async, gpu_matmul, gpu_matmul_async, gpu_matmul_tiled,
    gpu_matmul_tiled_async,
};
pub use context::{
    skips_size_threshold, DeviceTensor, GemmWeightTraffic, GpuBufferPool, GpuContext,
    GpuInitDiagnostic, GpuInitError, GpuMemoryBudget, GpuOutput, GpuPerfClass, GpuTuning,
    OutputPlacement, ResidentCounters, TensorSource, TrackedBuffer, WeightBytes, WeightFormat,
    WeightKeys, DEFAULT_LIVE_BYTE_BUDGET, DEFAULT_POOL_BYTE_BUDGET,
};
pub use device_guard::{read_device_tensor_async, GpuLimits};
pub use shaders::{
    gpu_abs, gpu_add, gpu_batch_norm, gpu_broadcast_add, gpu_broadcast_div, gpu_broadcast_mul,
    gpu_broadcast_sub, gpu_exp, gpu_gelu, gpu_gemm_nt, gpu_instance_norm, gpu_layer_norm,
    gpu_layer_norm_axis, gpu_leaky_relu, gpu_leaky_relu_alpha, gpu_log, gpu_mul, gpu_neg, gpu_pad,
    gpu_prelu, gpu_reduce_max, gpu_reduce_mean, gpu_reduce_min, gpu_reduce_sum, gpu_relu,
    gpu_resize_bilinear_pytorch_half_pixel, gpu_resize_nearest_asymmetric, gpu_sigmoid, gpu_silu,
    gpu_softmax, gpu_sqrt, gpu_tanh, gpu_transpose, ConvActivation, PadMode,
};
pub use shaders::{
    gpu_abs_async, gpu_add_async, gpu_batch_norm_async, gpu_broadcast_add_async,
    gpu_broadcast_div_async, gpu_broadcast_mul_async, gpu_broadcast_sub_async, gpu_exp_async,
    gpu_gelu_async, gpu_gemm_nt_async, gpu_instance_norm_async, gpu_layer_norm_async,
    gpu_layer_norm_axis_async, gpu_leaky_relu_alpha_async, gpu_leaky_relu_async, gpu_log_async,
    gpu_mul_async, gpu_neg_async, gpu_pad_async, gpu_prelu_async, gpu_reduce_max_async,
    gpu_reduce_mean_async, gpu_reduce_min_async, gpu_reduce_sum_async, gpu_relu_async,
    gpu_resize_bilinear_pytorch_half_pixel_async, gpu_resize_nearest_asymmetric_async,
    gpu_sigmoid_async, gpu_silu_async, gpu_softmax_async, gpu_sqrt_async, gpu_tanh_async,
    gpu_transpose_async,
};
pub use shaders::{
    gpu_abs_placed_async, gpu_add_placed_async, gpu_broadcast_placed_async,
    gpu_conv2d_implicit_placed_async, gpu_exp_placed_async, gpu_gelu_placed_async,
    gpu_gemm_nt_placed_async, gpu_instance_norm_placed_async, gpu_leaky_relu_placed_async,
    gpu_log_placed_async, gpu_mul_placed_async, gpu_neg_placed_async, gpu_pad_placed_async,
    gpu_prelu_placed_async, gpu_relu_placed_async, gpu_resize_placed_async,
    gpu_sigmoid_placed_async, gpu_silu_placed_async, gpu_sqrt_placed_async, gpu_tanh_placed_async,
    BroadcastKind, ResizeKind,
};
pub use shaders::{gpu_conv2d_implicit_resident_async, gpu_gemm_nt_resident_async};
