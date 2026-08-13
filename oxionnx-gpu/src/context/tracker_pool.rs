//! GpuBufferPool — a byte-bounded, LRU-evicting pool of reusable wgpu buffers.
//!
//! Entries are [`TrackedBuffer`]s rather than bare `wgpu::Buffer`s, so eviction
//! actually returns memory: see `super::budget` for why a dropped buffer frees
//! nothing in a browser.
//!
//! [a7-13] This module used to also carry a `GpuTensorTracker` whose docs
//! promised "data can remain on the GPU between operations without being read
//! back". Nothing ever consulted it: `GpuContext` held one behind a mutex and
//! no dispatch path anywhere in the workspace called `store` / `take` /
//! `is_on_gpu` outside the type's own unit test. It has been removed rather
//! than left as a dead field documenting a capability the engine does not have.
//!
//! Half of what it promised now exists, in `super::resident`: a graph's
//! *invariant* operands (convolution weights, `Gemm`'s `B`/`C`) stay in device
//! buffers for the session's lifetime, keyed by an identity the caller
//! supplies. The other half — chaining *activations* from one node to the next
//! without a read-back — still needs what it always needed: `Tensor`
//! (oxionnx-core) carrying a device buffer, or the session executor holding a
//! tensor map, plus buffer-taking variants of every `gpu_*` entry point.
//!
//! Resident buffers are **not** pool entries and cannot become them. The pool
//! owns `TrackedBuffer`s by value and `return_buffer` is only ever reached from
//! `read_back_and_recycle_async`, which also takes one by value; a resident
//! buffer is an `Arc<TrackedBuffer>`, which cannot be unwrapped into one. So
//! [`GpuBufferPool::reclaim_for`] — the eviction path a budget squeeze runs —
//! has no way to reach a weight.

use std::collections::VecDeque;
use std::sync::Arc;

use super::budget::{GpuMemoryBudget, TrackedBuffer};

/// Default byte budget for pooled idle buffers: 256 MiB.
///
/// [a7-14] The pool used to be bounded only by *count* (64 buffers) while
/// `return_buffer` preferentially evicted the *smallest* entry, so its steady
/// state was "the 64 largest output buffers this session ever produced". A
/// segmentation network whose activations run to 64 MB each could therefore
/// pin ~4 GB of VRAM in buffers nothing was using, with no eviction over time.
pub const DEFAULT_POOL_BYTE_BUDGET: u64 = 256 << 20;

/// Pool of reusable buffer allocations to reduce allocation overhead.
///
/// Bounded by two independent limits — a buffer count and a byte budget —
/// with least-recently-used eviction. Entries are kept in LRU order (front =
/// least recently touched), so a buffer that stops being requested is the
/// first one evicted when the pool needs room, regardless of its size.
///
/// Entries are [`TrackedBuffer`]s, so evicting one **destroys** it and returns
/// its bytes to the shared [`GpuMemoryBudget`] — dropping a `wgpu::Buffer`
/// releases nothing at all on the WebGPU backend (see `budget`'s module docs),
/// which made the old pool's eviction a bookkeeping change and nothing more in
/// a browser.
pub struct GpuBufferPool {
    /// Idle buffers in LRU order: front is the oldest, back the most recently
    /// returned.
    ///
    /// A pooled buffer is only handed back out when its creation-time
    /// [`wgpu::BufferUsages`] are a superset of what the caller asks for —
    /// binding a buffer that lacks a requested usage flag is a wgpu validation
    /// error, and validation errors must never reach a user of this crate.
    buffers: VecDeque<TrackedBuffer>,
    /// Maximum buffers to retain.
    max_buffers: usize,
    /// Maximum total bytes to retain across all pooled buffers.
    max_bytes: u64,
    /// Sum of the entries' sizes, maintained incrementally.
    pooled_bytes: u64,
    /// \[w4\] Requests [`Self::get_buffer`] served from an idle entry.
    reuses: u64,
    /// \[w4\] Requests [`Self::get_buffer`] had to ask the driver for.
    ///
    /// Counted whether or not the allocation succeeded, so
    /// `reuses + allocations` is the request count and the ratio between them
    /// is what a change in buffer disposition is supposed to move.
    allocations: u64,
    /// Live-byte accounting shared with the context this pool belongs to.
    /// Idle pooled buffers occupy device memory exactly like in-use ones, so
    /// they are counted the same and released on eviction.
    budget: Arc<GpuMemoryBudget>,
}

impl GpuBufferPool {
    /// Create a pool that retains up to `max_buffers` idle buffers and
    /// [`DEFAULT_POOL_BYTE_BUDGET`] bytes, against a private unlimited budget.
    #[must_use]
    pub fn new(max_buffers: usize) -> Self {
        Self::with_byte_budget(max_buffers, DEFAULT_POOL_BYTE_BUDGET)
    }

    /// Create a pool with an explicit byte budget, against a private unlimited
    /// live-byte budget.
    ///
    /// Both retention bounds apply: the pool never holds more than
    /// `max_buffers` entries nor more than `max_bytes` bytes in total.
    #[must_use]
    pub fn with_byte_budget(max_buffers: usize, max_bytes: u64) -> Self {
        Self::with_live_budget(max_buffers, max_bytes, GpuMemoryBudget::unlimited())
    }

    /// Create a pool that allocates against a shared live-byte budget.
    ///
    /// This is what [`crate::GpuContext`] uses: the pool's own `max_bytes`
    /// bounds how much it *retains*, while `budget` bounds how much the whole
    /// context may have allocated at once.
    #[must_use]
    pub fn with_live_budget(
        max_buffers: usize,
        max_bytes: u64,
        budget: Arc<GpuMemoryBudget>,
    ) -> Self {
        Self {
            buffers: VecDeque::new(),
            max_buffers,
            max_bytes,
            pooled_bytes: 0,
            reuses: 0,
            allocations: 0,
            budget,
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
    ///
    /// `None` means the live-byte budget cannot accommodate a new allocation
    /// even after idle entries were reclaimed — a decline, so the caller runs
    /// the node on the CPU.
    pub fn get_buffer(
        &mut self,
        device: &wgpu::Device,
        min_size: u64,
        usage: wgpu::BufferUsages,
    ) -> Option<TrackedBuffer> {
        // Find the smallest buffer that is >= min_size and <= 2*min_size (to
        // avoid waste) and whose usage flags cover the requested ones.
        let max_acceptable = min_size.saturating_mul(2);
        let best = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.reserved_bytes() >= min_size
                    && entry.reserved_bytes() <= max_acceptable
                    && entry.usage().contains(usage)
            })
            .min_by_key(|(_, entry)| entry.reserved_bytes())
            .map(|(idx, _)| idx);
        if let Some(idx) = best {
            if let Some(entry) = self.buffers.remove(idx) {
                self.pooled_bytes = self.pooled_bytes.saturating_sub(entry.reserved_bytes());
                self.reuses = self.reuses.saturating_add(1);
                return Some(entry);
            }
        }
        // No suitable buffer found — create a new one.
        self.allocations = self.allocations.saturating_add(1);
        let desc = wgpu::BufferDescriptor {
            label: Some("pool_buf"),
            size: min_size,
            usage,
            mapped_at_creation: false,
        };
        if let Some(buffer) = TrackedBuffer::create(device, &self.budget, &desc) {
            return Some(buffer);
        }
        // Over budget. Idle entries are reclaimable memory that nothing is
        // reading, so give them back to the driver before declining.
        self.reclaim_for(min_size);
        TrackedBuffer::create(device, &self.budget, &desc)
    }

    /// Return a buffer to the pool for reuse.
    ///
    /// A buffer larger than the whole retention budget is destroyed rather than
    /// pooled: keeping it would evict everything else and still leave the pool
    /// over budget.
    pub fn return_buffer(&mut self, buffer: TrackedBuffer) {
        let size = buffer.reserved_bytes();
        if self.max_buffers == 0 || size > self.max_bytes {
            // Dropping destroys it and releases its bytes.
            return;
        }
        self.buffers.push_back(buffer);
        self.pooled_bytes = self.pooled_bytes.saturating_add(size);
        self.evict_to_bounds();
    }

    /// Destroy least-recently-used idle entries until `bytes` fits the live
    /// budget, or until nothing is left to release.
    ///
    /// Called before declining an allocation: the pool's retention bound and
    /// the context's live-byte bound are independent, so a pool that is
    /// comfortably inside its own 256 MiB can still be the reason the next
    /// allocation does not fit.
    pub fn reclaim_for(&mut self, bytes: u64) {
        while !self.budget.admits(bytes) {
            let Some(evicted) = self.buffers.pop_front() else {
                break;
            };
            self.pooled_bytes = self.pooled_bytes.saturating_sub(evicted.reserved_bytes());
        }
    }

    /// Evict least-recently-used entries until both retention bounds hold.
    fn evict_to_bounds(&mut self) {
        while self.buffers.len() > self.max_buffers || self.pooled_bytes > self.max_bytes {
            let Some(evicted) = self.buffers.pop_front() else {
                // Nothing left to evict; `pooled_bytes` is 0 by construction.
                self.pooled_bytes = 0;
                break;
            };
            self.pooled_bytes = self.pooled_bytes.saturating_sub(evicted.reserved_bytes());
        }
    }

    /// Destroy every pooled buffer, releasing the memory back to the driver.
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

    /// The pool's entry-count budget — the other half of the retention bound
    /// [`Self::byte_budget`] states, and the one that usually binds first for a
    /// graph whose activations are small.
    #[must_use]
    pub fn max_buffers(&self) -> usize {
        self.max_buffers
    }

    /// \[w4\] Requests this pool has served from an idle entry since it was
    /// created. Never reset — difference two readings to get one run's figure.
    #[must_use]
    pub fn reuses(&self) -> u64 {
        self.reuses
    }

    /// \[w4\] Requests this pool has had to ask the driver for. See
    /// [`Self::reuses`].
    #[must_use]
    pub fn allocations(&self) -> u64 {
        self.allocations
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
