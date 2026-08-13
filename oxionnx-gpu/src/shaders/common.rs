//! Shared parameter structs and GPU read-back helper.
//!
//! # Where the thresholds went
//!
//! This module used to own seven flat constants — `EW_GPU_THRESHOLD` and
//! friends — each answering "is a GPU round trip cheaper than the CPU kernel
//! at this size?" with one compile-time number for every adapter in existence.
//! They now live on the context as [`crate::context::tuning::GpuTuning`],
//! because that question has no adapter-independent answer and, for the
//! memory-bound kernels, has no size-independent answer either: measured on an
//! RTX A4000, `gpu_relu`, `gpu_add`, `gpu_layer_norm` and `gpu_batch_norm` lose
//! to their CPU kernels at *every* size while their operands transfer, by
//! 1.8x to 45x. The historical values are preserved as
//! [`crate::context::tuning::GpuTuning::LEGACY_FLAT`], and every kernel below
//! now reads `ctx.tuning()` at its gate.

// --- Uniform param structs ---

/// Uniform block for the softmax kernel.
///
/// `wg_per_row` is the dispatch grid's X extent so the kernel can rebuild the
/// row index from a 2-D grid (`wid.y * wg_per_row + wid.x`), exactly as
/// [`LayerNormParams`] does for normalization instances.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SoftmaxParams {
    pub num_rows: u32,
    pub row_len: u32,
    pub wg_per_row: u32,
    pub _pad: u32,
}

/// Uniform block shared by the unary and binary element-wise kernels.
///
/// `alpha` carries the LeakyRelu slope (ignored by every other entry point);
/// `row_threads` is `grid_x * 256` so the kernels can rebuild a flat index from
/// a 2-D dispatch grid.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EwParams {
    pub len: u32,
    pub alpha: f32,
    pub row_threads: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ReduceParams {
    pub outer_size: u32,
    pub axis_len: u32,
    pub inner_size: u32,
    pub row_threads: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LayerNormParams {
    pub n_elements: u32,
    pub batch_count: u32,
    pub eps: f32,
    /// Workgroups along X, used to rebuild the instance index in a 2-D grid.
    pub wg_per_row: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BatchNormParams {
    pub total_elements: u32,
    pub channels: u32,
    pub spatial_size: u32,
    pub eps: f32,
    pub row_threads: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TransposeParams {
    pub total_elements: u32,
    pub ndim: u32,
    pub row_threads: u32,
    pub _pad1: u32,
}

// ========================================================================
// Shared device guards
// ========================================================================

// Read-back, dispatch planning and limit checks all live in `device_guard` so
// the matmul path in `compute.rs` and the shader paths here share one
// implementation (and one bounded, non-panicking failure mode).
pub(super) use crate::device_guard::{
    block_on_gpu, checked_storage_bytes, finish_output_async, plan_dispatch,
    read_back_and_recycle_async, DispatchGrid, ErrorScope,
};

/// Workgroup size (threads along X) used by every element-wise-style kernel.
pub(super) const WG_SIZE: u32 = 256;
