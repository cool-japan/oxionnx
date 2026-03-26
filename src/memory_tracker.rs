//! Peak memory usage tracking for inference runs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Tracks peak memory allocation during inference.
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    current_bytes: Arc<AtomicUsize>,
    peak_bytes: Arc<AtomicUsize>,
    allocation_count: Arc<AtomicUsize>,
}

impl MemoryTracker {
    /// Create a new tracker with all counters at zero.
    pub fn new() -> Self {
        Self {
            current_bytes: Arc::new(AtomicUsize::new(0)),
            peak_bytes: Arc::new(AtomicUsize::new(0)),
            allocation_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Record an allocation of `bytes` bytes.
    pub fn record_alloc(&self, bytes: usize) {
        let prev = self.current_bytes.fetch_add(bytes, Ordering::Relaxed);
        let new_total = prev + bytes;
        // Update peak if new total exceeds it (lock-free CAS loop)
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a deallocation of `bytes` bytes.
    pub fn record_dealloc(&self, bytes: usize) {
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Current allocated bytes.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Peak allocated bytes during lifetime.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Total number of allocations recorded.
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }

    /// Format a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "Memory: current={:.2} MB, peak={:.2} MB, allocations={}",
            self.current_bytes() as f64 / (1024.0 * 1024.0),
            self.peak_bytes() as f64 / (1024.0 * 1024.0),
            self.allocation_count(),
        )
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker_basic() {
        let tracker = MemoryTracker::new();
        assert_eq!(tracker.current_bytes(), 0);
        assert_eq!(tracker.peak_bytes(), 0);
        assert_eq!(tracker.allocation_count(), 0);

        tracker.record_alloc(1024);
        assert_eq!(tracker.current_bytes(), 1024);
        assert_eq!(tracker.peak_bytes(), 1024);
        assert_eq!(tracker.allocation_count(), 1);

        tracker.record_dealloc(512);
        assert_eq!(tracker.current_bytes(), 512);
        // Peak should still be 1024
        assert_eq!(tracker.peak_bytes(), 1024);
        assert_eq!(tracker.allocation_count(), 1);
    }

    #[test]
    fn test_memory_tracker_peak() {
        let tracker = MemoryTracker::new();

        // Allocate 1000, then 2000 more => peak should be 3000
        tracker.record_alloc(1000);
        tracker.record_alloc(2000);
        assert_eq!(tracker.current_bytes(), 3000);
        assert_eq!(tracker.peak_bytes(), 3000);

        // Free 1500 => current=1500, peak still 3000
        tracker.record_dealloc(1500);
        assert_eq!(tracker.current_bytes(), 1500);
        assert_eq!(tracker.peak_bytes(), 3000);

        // Allocate 2000 => current=3500, new peak
        tracker.record_alloc(2000);
        assert_eq!(tracker.current_bytes(), 3500);
        assert_eq!(tracker.peak_bytes(), 3500);

        // Free everything => peak unchanged
        tracker.record_dealloc(3500);
        assert_eq!(tracker.current_bytes(), 0);
        assert_eq!(tracker.peak_bytes(), 3500);
        assert_eq!(tracker.allocation_count(), 3);
    }

    #[test]
    fn test_memory_tracker_reset() {
        let tracker = MemoryTracker::new();

        tracker.record_alloc(4096);
        tracker.record_alloc(8192);
        assert_eq!(tracker.peak_bytes(), 12288);
        assert_eq!(tracker.allocation_count(), 2);

        tracker.reset();
        assert_eq!(tracker.current_bytes(), 0);
        assert_eq!(tracker.peak_bytes(), 0);
        assert_eq!(tracker.allocation_count(), 0);

        // After reset, new allocations should work correctly
        tracker.record_alloc(100);
        assert_eq!(tracker.current_bytes(), 100);
        assert_eq!(tracker.peak_bytes(), 100);
        assert_eq!(tracker.allocation_count(), 1);
    }
}
