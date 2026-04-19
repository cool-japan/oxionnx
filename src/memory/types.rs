//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::Node;
use std::collections::HashMap;

use super::functions::{bucket_for, compute_peak_memory, MAX_BUCKETS_PER_CLASS, MAX_POOL_BUFFERS};

/// Size-class-based memory pool that reduces fragmentation.
///
/// Buckets: tiny (<128 elements), small (<1024), medium (<16384), large (≥16384).
/// Within each bucket, best-fit allocation picks the smallest buffer ≥ requested size.
pub struct SizeClassPool {
    /// Free lists per size class. Each entry is a `Vec<f32>` buffer.
    tiny: Vec<Vec<f32>>,
    small: Vec<Vec<f32>>,
    medium: Vec<Vec<f32>>,
    large: Vec<Vec<f32>>,
    /// Allocation and reuse statistics.
    stats: PoolStats,
}
impl SizeClassPool {
    /// Create a new empty size-class pool.
    pub fn new() -> Self {
        Self {
            tiny: Vec::new(),
            small: Vec::new(),
            medium: Vec::new(),
            large: Vec::new(),
            stats: PoolStats::default(),
        }
    }
    /// Acquire a buffer with at least `size` f32 elements.
    ///
    /// Searches the appropriate size-class bucket for the smallest buffer
    /// that satisfies the request (best-fit). If no suitable buffer is cached,
    /// a fresh allocation is made. The returned buffer is zeroed and has
    /// exactly `size` elements.
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        let class = bucket_for(size);
        let bucket = self.bucket_mut(class);
        let best_idx = Self::best_fit_index(bucket, size);
        if let Some(idx) = best_idx {
            let mut buf = bucket.remove(idx);
            self.stats.reuse_count += 1;
            let freed_bytes = buf.capacity() * std::mem::size_of::<f32>();
            self.stats.current_bytes = self.stats.current_bytes.saturating_sub(freed_bytes);
            buf.clear();
            buf.resize(size, 0.0);
            buf
        } else {
            if let Some((found_class, idx)) = self.find_in_larger_buckets(class, size) {
                let bucket = self.bucket_mut(found_class);
                let mut buf = bucket.remove(idx);
                self.stats.reuse_count += 1;
                let freed_bytes = buf.capacity() * std::mem::size_of::<f32>();
                self.stats.current_bytes = self.stats.current_bytes.saturating_sub(freed_bytes);
                buf.clear();
                buf.resize(size, 0.0);
                buf
            } else {
                self.stats.alloc_count += 1;
                let buf = vec![0.0_f32; size];
                let allocated_bytes = buf.capacity() * std::mem::size_of::<f32>();
                let total = self.stats.current_bytes + allocated_bytes;
                if total > self.stats.peak_bytes {
                    self.stats.peak_bytes = total;
                }
                buf
            }
        }
    }
    /// Release a buffer back into the pool.
    ///
    /// The buffer is placed into the bucket corresponding to its length (requested size).
    /// Buffers are not shrunk on return. Per-class limits prevent unbounded growth.
    pub fn release(&mut self, buf: Vec<f32>) {
        if buf.capacity() == 0 {
            return;
        }
        let class = bucket_for(buf.len());
        let added_bytes = buf.capacity() * std::mem::size_of::<f32>();
        let bucket = self.bucket_mut(class);
        if bucket.len() >= MAX_BUCKETS_PER_CLASS {
            if let Some(smallest_cap) = bucket.first().map(|b| b.capacity()) {
                if buf.capacity() > smallest_cap {
                    let evicted = bucket.remove(0);
                    let evicted_bytes = evicted.capacity() * std::mem::size_of::<f32>();
                    self.stats.current_bytes =
                        self.stats.current_bytes.saturating_sub(evicted_bytes);
                } else {
                    return;
                }
            }
        }
        let cap = buf.capacity();
        let bucket = self.bucket_mut(class);
        let pos = bucket.partition_point(|b| b.capacity() < cap);
        bucket.insert(pos, buf);
        self.stats.current_bytes += added_bytes;
        if self.stats.current_bytes > self.stats.peak_bytes {
            self.stats.peak_bytes = self.stats.current_bytes;
        }
        self.update_fragmentation();
    }
    /// Return a reference to the pool statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }
    /// Drop all cached buffers, resetting the pool.
    pub fn clear(&mut self) {
        self.tiny.clear();
        self.small.clear();
        self.medium.clear();
        self.large.clear();
        self.stats.current_bytes = 0;
    }
    /// Compact the pool by dropping oversized buffers.
    ///
    /// If fragmentation exceeds 20%, removes buffers whose capacity is more
    /// than 2× the upper bound of their size class. For the `Large` class,
    /// no compaction is applied since there is no meaningful upper bound.
    pub fn compact(&mut self) {
        if self.stats.fragmentation_ratio <= 0.20 {
            return;
        }
        self.compact_bucket(SizeClass::Tiny);
        self.compact_bucket(SizeClass::Small);
        self.compact_bucket(SizeClass::Medium);
        self.update_fragmentation();
    }
    fn bucket_mut(&mut self, class: SizeClass) -> &mut Vec<Vec<f32>> {
        match class {
            SizeClass::Tiny => &mut self.tiny,
            SizeClass::Small => &mut self.small,
            SizeClass::Medium => &mut self.medium,
            SizeClass::Large => &mut self.large,
        }
    }
    fn bucket_ref(&self, class: SizeClass) -> &Vec<Vec<f32>> {
        match class {
            SizeClass::Tiny => &self.tiny,
            SizeClass::Small => &self.small,
            SizeClass::Medium => &self.medium,
            SizeClass::Large => &self.large,
        }
    }
    /// Find the index of the smallest buffer with capacity >= `size` in a bucket.
    fn best_fit_index(bucket: &[Vec<f32>], size: usize) -> Option<usize> {
        let pos = bucket.partition_point(|b| b.capacity() < size);
        if pos < bucket.len() {
            Some(pos)
        } else {
            None
        }
    }
    /// Search buckets larger than `class` for a buffer with capacity >= `size`.
    fn find_in_larger_buckets(&self, class: SizeClass, size: usize) -> Option<(SizeClass, usize)> {
        let larger_classes: &[SizeClass] = match class {
            SizeClass::Tiny => &[SizeClass::Small, SizeClass::Medium, SizeClass::Large],
            SizeClass::Small => &[SizeClass::Medium, SizeClass::Large],
            SizeClass::Medium => &[SizeClass::Large],
            SizeClass::Large => &[],
        };
        for &lc in larger_classes {
            let bucket = self.bucket_ref(lc);
            if let Some(idx) = Self::best_fit_index(bucket, size) {
                return Some((lc, idx));
            }
        }
        None
    }
    /// Compact a single bucket by removing buffers whose capacity exceeds
    /// 2× the class maximum.
    fn compact_bucket(&mut self, class: SizeClass) {
        let threshold = class.max_elements().saturating_mul(2);
        let bucket = self.bucket_mut(class);
        let mut freed_bytes: usize = 0;
        bucket.retain(|buf| {
            if buf.capacity() > threshold {
                freed_bytes += buf.capacity() * std::mem::size_of::<f32>();
                false
            } else {
                true
            }
        });
        self.stats.current_bytes = self.stats.current_bytes.saturating_sub(freed_bytes);
    }
    /// Recompute fragmentation ratio as (wasted elements) / (total cached elements).
    /// Wasted = sum of (capacity − len) for all cached buffers.
    fn update_fragmentation(&mut self) {
        let mut total_capacity: usize = 0;
        let mut total_wasted: usize = 0;
        for bucket in [&self.tiny, &self.small, &self.medium, &self.large] {
            for buf in bucket {
                total_capacity += buf.capacity();
            }
        }
        for (class, bucket) in [
            (SizeClass::Tiny, &self.tiny),
            (SizeClass::Small, &self.small),
            (SizeClass::Medium, &self.medium),
            (SizeClass::Large, &self.large),
        ] {
            let class_min = match class {
                SizeClass::Tiny => 1,
                SizeClass::Small => 128,
                SizeClass::Medium => 1024,
                SizeClass::Large => 16384,
            };
            for buf in bucket {
                total_wasted += buf.capacity().saturating_sub(class_min);
            }
        }
        if total_capacity == 0 {
            self.stats.fragmentation_ratio = 0.0;
        } else {
            self.stats.fragmentation_ratio = total_wasted as f32 / total_capacity as f32;
        }
    }
}
/// Size class categories for the bucketing allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeClass {
    /// < 128 elements
    Tiny,
    /// < 1024 elements
    Small,
    /// < 16384 elements
    Medium,
    /// ≥ 16384 elements
    Large,
}
impl SizeClass {
    /// Return the exclusive upper bound for this size class (elements).
    /// For `Large`, returns `usize::MAX` since there is no upper bound.
    pub fn max_elements(self) -> usize {
        match self {
            SizeClass::Tiny => 128,
            SizeClass::Small => 1024,
            SizeClass::Medium => 16384,
            SizeClass::Large => usize::MAX,
        }
    }
}
/// Memory plan with buffer slot assignments for intermediate tensors.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Lifetime intervals for each intermediate tensor.
    pub lifetimes: Vec<TensorLifetime>,
    /// Mapping from tensor name to buffer slot index.
    pub buffer_assignments: HashMap<String, usize>,
    /// Size (in f32 elements) of each buffer slot.
    pub buffer_sizes: Vec<usize>,
    /// Peak concurrent memory in f32 elements across all execution steps.
    pub peak_memory_elements: usize,
}
impl MemoryPlan {
    /// Compute a memory plan from a topologically sorted node list.
    ///
    /// `sorted_nodes`: nodes in execution order.
    /// `output_names`: graph output tensor names (never freed early).
    /// `shape_map`: tensor name -> shape dimensions (from shape inference).
    pub fn compute(
        sorted_nodes: &[Node],
        output_names: &[String],
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Self {
        let mut produced: HashMap<String, usize> = HashMap::new();
        let mut last_consumed: HashMap<String, usize> = HashMap::new();
        for (i, node) in sorted_nodes.iter().enumerate() {
            for out_name in &node.outputs {
                if !out_name.is_empty() {
                    produced.entry(out_name.clone()).or_insert(i);
                }
            }
            for inp_name in &node.inputs {
                if !inp_name.is_empty() {
                    last_consumed.insert(inp_name.clone(), i);
                }
            }
        }
        let final_step = sorted_nodes.len();
        for name in output_names {
            last_consumed.insert(name.clone(), final_step);
        }
        let mut lifetimes: Vec<TensorLifetime> = Vec::new();
        for (name, &prod) in &produced {
            let consumed = last_consumed.get(name).copied().unwrap_or(prod);
            let size_elements = shape_map
                .get(name)
                .map(|dims| {
                    if dims.is_empty() {
                        1
                    } else {
                        dims.iter().product()
                    }
                })
                .unwrap_or(0);
            lifetimes.push(TensorLifetime {
                name: name.clone(),
                produced_at: prod,
                last_consumed_at: consumed,
                size_elements,
            });
        }
        lifetimes.sort_by_key(|lt| lt.produced_at);
        let mut buffer_assignments: HashMap<String, usize> = HashMap::new();
        let mut buffer_sizes: Vec<usize> = Vec::new();
        let mut slot_free_after: Vec<usize> = Vec::new();
        for lt in &lifetimes {
            if lt.size_elements == 0 {
                let slot = buffer_sizes.len();
                buffer_sizes.push(0);
                slot_free_after.push(lt.last_consumed_at);
                buffer_assignments.insert(lt.name.clone(), slot);
                continue;
            }
            let mut best_slot: Option<usize> = None;
            let mut best_size: usize = usize::MAX;
            for (slot_idx, &free_after) in slot_free_after.iter().enumerate() {
                if free_after < lt.produced_at
                    && buffer_sizes[slot_idx] >= lt.size_elements
                    && buffer_sizes[slot_idx] < best_size
                {
                    best_size = buffer_sizes[slot_idx];
                    best_slot = Some(slot_idx);
                }
            }
            if best_slot.is_none() {
                let mut smallest_available: Option<(usize, usize)> = None;
                for (slot_idx, &free_after) in slot_free_after.iter().enumerate() {
                    if free_after < lt.produced_at {
                        let sz = buffer_sizes[slot_idx];
                        if smallest_available.is_none()
                            || sz < smallest_available.map(|(_, s)| s).unwrap_or(usize::MAX)
                        {
                            smallest_available = Some((slot_idx, sz));
                        }
                    }
                }
                if let Some((slot_idx, _)) = smallest_available {
                    best_slot = Some(slot_idx);
                }
            }
            match best_slot {
                Some(slot_idx) => {
                    if buffer_sizes[slot_idx] < lt.size_elements {
                        buffer_sizes[slot_idx] = lt.size_elements;
                    }
                    slot_free_after[slot_idx] = lt.last_consumed_at;
                    buffer_assignments.insert(lt.name.clone(), slot_idx);
                }
                None => {
                    let slot = buffer_sizes.len();
                    buffer_sizes.push(lt.size_elements);
                    slot_free_after.push(lt.last_consumed_at);
                    buffer_assignments.insert(lt.name.clone(), slot);
                }
            }
        }
        let peak_memory_elements = compute_peak_memory(&lifetimes, final_step);
        Self {
            lifetimes,
            buffer_assignments,
            buffer_sizes,
            peak_memory_elements,
        }
    }
}
/// Lifetime interval for an intermediate tensor.
#[derive(Debug, Clone)]
pub struct TensorLifetime {
    /// Tensor name in the graph.
    pub name: String,
    /// Node execution index where the tensor is first produced.
    pub produced_at: usize,
    /// Node execution index where the tensor is last consumed.
    pub last_consumed_at: usize,
    /// Number of f32 elements (0 if shape is unknown).
    pub size_elements: usize,
}
/// Per-pool allocation and reuse statistics.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of allocations (new buffers created).
    pub alloc_count: u64,
    /// Number of times a buffer was reused from the pool.
    pub reuse_count: u64,
    /// Peak total bytes held by the pool (cached + in-flight).
    pub peak_bytes: usize,
    /// Current bytes held in the pool's free lists.
    pub current_bytes: usize,
    /// Fragmentation ratio: wasted bytes / total cached bytes. 0.0 = perfect.
    pub fragmentation_ratio: f32,
}
/// Buffer pool for reusing tensor allocations during inference.
///
/// Maintains a sorted list of available buffers and returns the smallest
/// buffer that satisfies the requested size.
pub struct BufferPool {
    /// Available buffers sorted by capacity (ascending).
    buffers: Vec<Vec<f32>>,
}
impl BufferPool {
    /// Create a new empty buffer pool.
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }
    /// Get a buffer with at least `min_size` f32 elements.
    ///
    /// Returns a recycled buffer if one of sufficient size is available,
    /// otherwise allocates a new one. The returned buffer is zeroed and
    /// has exactly `min_size` elements.
    pub fn get_buffer(&mut self, min_size: usize) -> Vec<f32> {
        let pos = self
            .buffers
            .partition_point(|buf| buf.capacity() < min_size);
        if pos < self.buffers.len() {
            let mut buf = self.buffers.remove(pos);
            buf.clear();
            buf.resize(min_size, 0.0);
            buf
        } else {
            vec![0.0; min_size]
        }
    }
    /// Return a buffer to the pool for future reuse.
    ///
    /// The pool maintains at most `MAX_POOL_BUFFERS` buffers to prevent
    /// unbounded memory growth.
    pub fn return_buffer(&mut self, buf: Vec<f32>) {
        if self.buffers.len() >= MAX_POOL_BUFFERS {
            if let Some(smallest_cap) = self.buffers.first().map(|b| b.capacity()) {
                if buf.capacity() > smallest_cap {
                    self.buffers.remove(0);
                } else {
                    return;
                }
            }
        }
        let cap = buf.capacity();
        let pos = self.buffers.partition_point(|b| b.capacity() < cap);
        self.buffers.insert(pos, buf);
    }
    /// Clear all pooled buffers, releasing their memory.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
    /// Number of buffers currently available in the pool.
    pub fn available_count(&self) -> usize {
        self.buffers.len()
    }
}
