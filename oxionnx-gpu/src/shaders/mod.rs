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

/// \[w4\] Drop every entry this thread holds for `device` across all five
/// device-keyed thread-local caches: `kernel_support`'s `PIPELINES`,
/// `conv2d`'s `CONV2D_PIPELINE` and `CONV2D_F16_UNAVAILABLE`, and `gemm`'s
/// `GEMM_F16_UNAVAILABLE` and `GEMM_F16_READY`.
///
/// Called from [`crate::GpuContext`]'s `Drop`. Every one of those caches stores
/// the `wgpu::Device` handle itself — handle equality is `Arc` identity, and
/// only holding the handle stops a later, different device from comparing equal
/// by landing in a freed slot — so every entry keeps its device alive. The
/// retain-on-insert rule in `kernel_support::insert_for_current_device` already
/// bounds each cache at one device, but it runs only on an insert, leaving a
/// dropped context's handle resident on a thread that has stopped dispatching
/// until that thread's next compile. This is the other half.
///
/// **Same-thread, and best-effort.** It reaches the caches of whichever thread
/// runs the `Drop`, never another's; entries a worker thread populated for this
/// device stay until that thread's next insert evicts them, which is by
/// construction (a thread-local is not addressable from another thread, and
/// these caches cannot be global — `wgpu::Device` is neither `Send` nor `Sync`
/// on wasm32). Retain-on-insert therefore remains the backstop and is not
/// weakened by this existing. See `kernel_support::purge_thread_local` for the
/// two teardown cases it must not panic in.
pub(crate) fn purge_thread_local_caches_for(device: &wgpu::Device) {
    kernel_support::purge_device(device);
    conv2d::purge_device(device);
    gemm::purge_device(device);
}

/// \[w4\] Entries this thread holds for `device` across all five caches, i.e.
/// what [`purge_thread_local_caches_for`] would remove. Test-only.
///
/// Counting *for a named device* rather than in total is what makes an
/// assertion on it order-independent: a test can only ever ask about the
/// context it constructed itself, so nothing another test left on the thread
/// can move the answer.
#[cfg(test)]
pub(crate) fn thread_local_entries_for_device(device: &wgpu::Device) -> usize {
    kernel_support::cached_entries_for_device(device)
        + conv2d::cached_entries_for_device(device)
        + gemm::cached_entries_for_device(device)
}

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

// Crate-internal only: the standalone kernel batch's per-call pipeline
// builders (broadcast_binary/gemm/pad/prelu/resize -- see
// `kernel_support`'s module docs). Re-exported here, not from their private
// submodules directly, so a future integration wave can hoist these into
// `GpuContext`'s cached pipeline fields (`context/types.rs`) as a call-site
// move (`crate::shaders::build_broadcast_pipeline(...)`) without also having
// to make `broadcast_binary`/`pad`/`prelu`/`resize`/`gemm` public modules.
// Unused for now (nothing in this crate calls them yet -- that integration
// is a follow-up wave's job, not this one's); the `#[allow]` is deliberate,
// not a stray import to clean up.
#[allow(unused_imports)]
pub(crate) use broadcast_binary::{build_broadcast_pipeline, BroadcastOp};
// `conv2d` already caches its own compiled pipeline per device (see its
// `CONV2D_PIPELINE` docs), so this export exists for the same reason as the
// others: hoisting the compile into `GpuContext` later stays a call-site move.
#[allow(unused_imports)]
pub(crate) use conv2d::build_conv2d_pipeline;
#[allow(unused_imports)]
pub(crate) use gemm::build_gemm_nt_pipeline;
#[allow(unused_imports)]
pub(crate) use instance_norm::build_instance_norm_pipeline;
#[allow(unused_imports)]
pub(crate) use pad::build_pad_pipeline;
#[allow(unused_imports)]
pub(crate) use prelu::build_prelu_pipeline;
#[allow(unused_imports)]
pub(crate) use resize::build_resize_pipeline;
