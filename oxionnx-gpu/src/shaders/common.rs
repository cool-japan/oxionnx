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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SoftmaxParams {
    pub num_rows: u32,
    pub row_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EwParams {
    pub len: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ReduceParams {
    pub outer_size: u32,
    pub axis_len: u32,
    pub inner_size: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LayerNormParams {
    pub n_elements: u32,
    pub batch_count: u32,
    pub eps: f32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BatchNormParams {
    pub total_elements: u32,
    pub channels: u32,
    pub spatial_size: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TransposeParams {
    pub total_elements: u32,
    pub ndim: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

// ========================================================================
// Helper: read back a staging buffer into Vec<f32>
// ========================================================================

/// Read back GPU staging buffer contents into a `Vec<f32>`.
///
/// On wasm32, blocking device poll is not supported, so this returns `None`.
pub(super) fn read_back(
    _device: &wgpu::Device,
    _staging: &wgpu::Buffer,
    _count: usize,
) -> Option<Vec<f32>> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let slice = _staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        _device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if receiver.recv().ok()?.is_err() {
            return None;
        }
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data)[.._count].to_vec();
        drop(data);
        _staging.unmap();
        Some(result)
    }
}
