//! GpuTensorTracker and GpuBufferPool — lightweight GPU memory management helpers.

use std::collections::HashMap;

/// Track whether a tensor's data is on GPU to avoid redundant host-device transfers.
///
/// When executing consecutive GPU-capable operations, data can remain on the GPU
/// between operations without being read back to the CPU.
pub struct GpuTensorTracker {
    /// Map from tensor name to its GPU buffer (if currently on GPU).
    gpu_buffers: HashMap<String, (wgpu::Buffer, u64)>,
}

impl GpuTensorTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            gpu_buffers: HashMap::new(),
        }
    }

    /// Check if a tensor is currently on GPU.
    pub fn is_on_gpu(&self, name: &str) -> bool {
        self.gpu_buffers.contains_key(name)
    }

    /// Store a GPU buffer for a tensor.
    pub fn store(&mut self, name: String, buffer: wgpu::Buffer, size: u64) {
        self.gpu_buffers.insert(name, (buffer, size));
    }

    /// Remove and return a GPU buffer.
    pub fn take(&mut self, name: &str) -> Option<(wgpu::Buffer, u64)> {
        self.gpu_buffers.remove(name)
    }

    /// Clear all tracked GPU buffers.
    pub fn clear(&mut self) {
        self.gpu_buffers.clear();
    }

    /// Number of tensors currently tracked on GPU.
    pub fn count(&self) -> usize {
        self.gpu_buffers.len()
    }
}

impl Default for GpuTensorTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Pool of reusable wgpu::Buffer allocations to reduce allocation overhead.
pub struct GpuBufferPool {
    /// Available buffers sorted by size (ascending).
    buffers: Vec<(u64, wgpu::Buffer)>,
    /// Maximum buffers to retain.
    max_buffers: usize,
}

impl GpuBufferPool {
    /// Create a new buffer pool that retains up to `max_buffers` idle buffers.
    pub fn new(max_buffers: usize) -> Self {
        Self {
            buffers: Vec::new(),
            max_buffers,
        }
    }

    /// Get a buffer of at least `min_size` bytes.
    /// Returns a reused buffer if available (within 2x of requested size), or creates a new one.
    pub fn get_buffer(
        &mut self,
        device: &wgpu::Device,
        min_size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        // Find smallest buffer that is >= min_size and <= 2*min_size (to avoid waste).
        let max_acceptable = min_size.saturating_mul(2);
        let pos = self
            .buffers
            .iter()
            .position(|(sz, _)| *sz >= min_size && *sz <= max_acceptable);
        if let Some(idx) = pos {
            let (_sz, buf) = self.buffers.remove(idx);
            return buf;
        }
        // No suitable buffer found — create a new one.
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_buf"),
            size: min_size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&mut self, buffer: wgpu::Buffer, size: u64) {
        if self.buffers.len() >= self.max_buffers {
            // Drop the smallest buffer to make room (the new one might be more useful).
            if let Some(min_idx) = self
                .buffers
                .iter()
                .enumerate()
                .min_by_key(|(_, (sz, _))| *sz)
                .map(|(i, _)| i)
            {
                if self.buffers[min_idx].0 < size {
                    self.buffers.remove(min_idx);
                } else {
                    // New buffer is smaller than all existing ones — just drop it.
                    return;
                }
            }
        }
        // Insert sorted by size.
        let insert_pos = self.buffers.partition_point(|(sz, _)| *sz < size);
        self.buffers.insert(insert_pos, (size, buffer));
    }

    /// Clear all pooled buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Number of buffers currently available for reuse.
    pub fn available_count(&self) -> usize {
        self.buffers.len()
    }
}
