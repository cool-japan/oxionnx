//! Shared parameter structs, thresholds, and GPU read-back helper.

/// Minimum tensor elements before GPU dispatch is worthwhile for element-wise ops.
pub(super) const EW_GPU_THRESHOLD: usize = 100_000;

/// Minimum last-dimension size before GPU softmax is worthwhile.
pub(super) const SOFTMAX_DIM_THRESHOLD: usize = 1000;

/// Minimum output elements before GPU reduction is worthwhile.
pub(super) const REDUCE_GPU_THRESHOLD: usize = 50_000;

/// Minimum elements before GPU LayerNorm is worthwhile.
pub(super) const LAYER_NORM_GPU_THRESHOLD: usize = 50_000;

/// Minimum elements before GPU BatchNorm is worthwhile.
pub(super) const BATCH_NORM_GPU_THRESHOLD: usize = 50_000;

/// Minimum elements before GPU Transpose is worthwhile.
pub(super) const TRANSPOSE_GPU_THRESHOLD: usize = 50_000;

/// Minimum tensor elements before GPU dispatch is worthwhile for binary element-wise ops.
pub(super) const BINARY_EW_GPU_THRESHOLD: usize = 100_000;

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
