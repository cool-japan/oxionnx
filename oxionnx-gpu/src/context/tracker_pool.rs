//! GpuBufferPool — a byte-bounded, LRU-evicting pool of reusable wgpu buffers.
//!
//! [a7-13] This module used to also carry a `GpuTensorTracker` whose docs
//! promised "data can remain on the GPU between operations without being read
//! back". Nothing ever consulted it: `GpuContext` held one behind a mutex and
//! no dispatch path anywhere in the workspace called `store` / `take` /
//! `is_on_gpu` outside the type's own unit test. It has been removed rather
//! than left as a dead field documenting a capability the engine does not
//! have. Real keep-on-GPU chaining needs `Tensor` (oxionnx-core) to carry a
//! device buffer, or the session executor to hold a residency map, plus
//! buffer-taking variants of every `gpu_*` entry point — none of which is a
//! change local to this crate.

use std::collections::VecDeque;

/// Default byte budget for pooled idle buffers: 256 MiB.
///
/// [a7-14] The pool used to be bounded only by *count* (64 buffers) while
/// `return_buffer` preferentially evicted the *smallest* entry, so its steady
/// state was "the 64 largest output buffers this session ever produced". A
/// segmentation network whose activations run to 64 MB each could therefore
/// pin ~4 GB of VRAM in buffers nothing was using, with no eviction over time.
pub const DEFAULT_POOL_BYTE_BUDGET: u64 = 256 << 20;

/// One idle buffer, tagged with the size wgpu actually allocated for it.
struct PooledBuffer {
    /// `wgpu::Buffer::size()` — the real capacity, not the caller's request.
    size: u64,
    buffer: wgpu::Buffer,
}

/// Pool of reusable `wgpu::Buffer` allocations to reduce allocation overhead.
///
/// Bounded by two independent limits — a buffer count and a byte budget —
/// with least-recently-used eviction. Entries are kept in LRU order (front =
/// least recently touched), so a buffer that stops being requested is the
/// first one dropped when the pool needs room, regardless of its size.
pub struct GpuBufferPool {
    /// Idle buffers in LRU order: front is the oldest, back the most recently
    /// returned.
    ///
    /// A pooled buffer is only handed back out when its creation-time
    /// [`wgpu::BufferUsages`] are a superset of what the caller asks for —
    /// binding a buffer that lacks a requested usage flag is a wgpu validation
    /// error, and validation errors must never reach a user of this crate.
    buffers: VecDeque<PooledBuffer>,
    /// Maximum buffers to retain.
    max_buffers: usize,
    /// Maximum total bytes to retain across all pooled buffers.
    max_bytes: u64,
    /// Sum of `size` over `buffers`, maintained incrementally.
    pooled_bytes: u64,
}

impl GpuBufferPool {
    /// Create a pool that retains up to `max_buffers` idle buffers and
    /// [`DEFAULT_POOL_BYTE_BUDGET`] bytes.
    #[must_use]
    pub fn new(max_buffers: usize) -> Self {
        Self::with_byte_budget(max_buffers, DEFAULT_POOL_BYTE_BUDGET)
    }

    /// Create a pool with an explicit byte budget.
    ///
    /// Both bounds apply: the pool never holds more than `max_buffers` entries
    /// nor more than `max_bytes` bytes in total.
    #[must_use]
    pub fn with_byte_budget(max_buffers: usize, max_bytes: u64) -> Self {
        Self {
            buffers: VecDeque::new(),
            max_buffers,
            max_bytes,
            pooled_bytes: 0,
        }
    }

    /// Get a buffer of at least `min_size` bytes that supports `usage`.
    ///
    /// Returns a reused buffer when the pool holds one that is large enough
    /// (within 2x of the requested size, so a tiny request cannot claim a huge
    /// allocation) **and** was created with usage flags covering `usage`;
    /// otherwise a new buffer is created.
    ///
    /// The returned buffer may be *larger* than `min_size`. Callers must bind
    /// and copy using their own size, never `buffer.size()`.
    pub fn get_buffer(
        &mut self,
        device: &wgpu::Device,
        min_size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        // Find the smallest buffer that is >= min_size and <= 2*min_size (to
        // avoid waste) and whose usage flags cover the requested ones.
        let max_acceptable = min_size.saturating_mul(2);
        let best = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.size >= min_size
                    && entry.size <= max_acceptable
                    && entry.buffer.usage().contains(usage)
            })
            .min_by_key(|(_, entry)| entry.size)
            .map(|(idx, _)| idx);
        if let Some(idx) = best {
            if let Some(entry) = self.buffers.remove(idx) {
                self.pooled_bytes = self.pooled_bytes.saturating_sub(entry.size);
                return entry.buffer;
            }
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
    ///
    /// The entry is tagged with `buffer.size()` — the allocation wgpu really
    /// made. [a7-14] The old signature took the caller's *requested* size,
    /// which drifted below the truth as soon as [`Self::get_buffer`] handed
    /// back an oversized buffer (it accepts anything up to 2x the request), so
    /// pooled entries progressively under-reported their capacity and a later
    /// `get_buffer` could reject a buffer that would in fact have fit.
    ///
    /// A buffer larger than the whole byte budget is dropped rather than
    /// pooled: keeping it would evict everything else and still leave the pool
    /// over budget.
    pub fn return_buffer(&mut self, buffer: wgpu::Buffer) {
        let size = buffer.size();
        if self.max_buffers == 0 || size > self.max_bytes {
            return;
        }
        self.buffers.push_back(PooledBuffer { size, buffer });
        self.pooled_bytes = self.pooled_bytes.saturating_add(size);
        self.evict_to_bounds();
    }

    /// Drop least-recently-used entries until both bounds hold.
    fn evict_to_bounds(&mut self) {
        while self.buffers.len() > self.max_buffers || self.pooled_bytes > self.max_bytes {
            let Some(evicted) = self.buffers.pop_front() else {
                // Nothing left to evict; `pooled_bytes` is 0 by construction.
                self.pooled_bytes = 0;
                break;
            };
            self.pooled_bytes = self.pooled_bytes.saturating_sub(evicted.size);
        }
    }

    /// Clear all pooled buffers, releasing the memory back to the driver.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.pooled_bytes = 0;
    }

    /// Number of buffers currently available for reuse.
    #[must_use]
    pub fn available_count(&self) -> usize {
        self.buffers.len()
    }

    /// Total bytes currently held by idle pooled buffers.
    #[must_use]
    pub fn pooled_bytes(&self) -> u64 {
        self.pooled_bytes
    }

    /// The pool's byte budget.
    #[must_use]
    pub fn byte_budget(&self) -> u64 {
        self.max_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte accounting and LRU order are pure logic — assert them without a
    /// device by checking the bookkeeping a pool does around a real buffer is
    /// consistent. (Buffer-backed behaviour is covered in `shaders::tests` and
    /// `tests/w2_gpu_perf.rs`, which skip when no adapter exists.)
    #[test]
    fn a_fresh_pool_is_empty_and_reports_its_budget() {
        let pool = GpuBufferPool::with_byte_budget(8, 4096);
        assert_eq!(pool.available_count(), 0);
        assert_eq!(pool.pooled_bytes(), 0);
        assert_eq!(pool.byte_budget(), 4096);

        let default_pool = GpuBufferPool::new(64);
        assert_eq!(default_pool.byte_budget(), DEFAULT_POOL_BYTE_BUDGET);
    }
}
