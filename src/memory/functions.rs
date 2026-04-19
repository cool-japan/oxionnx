//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{SizeClass, TensorLifetime};

/// Compute peak concurrent memory usage (in f32 elements) across all execution steps.
pub(super) fn compute_peak_memory(lifetimes: &[TensorLifetime], total_steps: usize) -> usize {
    let mut peak: usize = 0;
    for step in 0..=total_steps {
        let live_sum: usize = lifetimes
            .iter()
            .filter(|lt| lt.produced_at <= step && lt.last_consumed_at >= step)
            .map(|lt| lt.size_elements)
            .sum();
        if live_sum > peak {
            peak = live_sum;
        }
    }
    peak
}
/// Maximum number of buffers retained in the pool.
pub(super) const MAX_POOL_BUFFERS: usize = 64;
/// Determine the size class for a given number of elements.
pub fn bucket_for(size: usize) -> SizeClass {
    if size < 128 {
        SizeClass::Tiny
    } else if size < 1024 {
        SizeClass::Small
    } else if size < 16384 {
        SizeClass::Medium
    } else {
        SizeClass::Large
    }
}
/// Maximum buffers per size class to prevent unbounded growth.
pub(super) const MAX_BUCKETS_PER_CLASS: usize = 32;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Node, OpKind};
    use crate::memory::{BufferPool, MemoryPlan, SizeClassPool};
    use std::collections::HashMap;
    fn make_node(op: OpKind, name: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> Node {
        Node {
            op,
            name: name.to_string(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
            attrs: Attributes::default(),
        }
    }
    #[test]
    fn test_lifetime_computation() {
        let nodes = vec![
            make_node(OpKind::Relu, "relu", vec!["input"], vec!["a"]),
            make_node(OpKind::Sigmoid, "sigmoid", vec!["a"], vec!["b"]),
            make_node(OpKind::Tanh, "tanh", vec!["b"], vec!["output"]),
        ];
        let output_names = vec!["output".to_string()];
        let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();
        shape_map.insert("a".to_string(), vec![1, 10]);
        shape_map.insert("b".to_string(), vec![1, 10]);
        shape_map.insert("output".to_string(), vec![1, 10]);
        let plan = MemoryPlan::compute(&nodes, &output_names, &shape_map);
        let lt_a = plan.lifetimes.iter().find(|lt| lt.name == "a").expect("a");
        let lt_b = plan.lifetimes.iter().find(|lt| lt.name == "b").expect("b");
        let lt_out = plan
            .lifetimes
            .iter()
            .find(|lt| lt.name == "output")
            .expect("output");
        assert_eq!(lt_a.produced_at, 0);
        assert_eq!(lt_a.last_consumed_at, 1);
        assert_eq!(lt_b.produced_at, 1);
        assert_eq!(lt_b.last_consumed_at, 2);
        assert_eq!(lt_out.produced_at, 2);
        assert_eq!(lt_out.last_consumed_at, 3);
    }
    #[test]
    fn test_buffer_reuse_non_overlapping() {
        let nodes = vec![
            make_node(OpKind::Relu, "n0", vec!["input"], vec!["a"]),
            make_node(OpKind::Relu, "n1", vec!["a"], vec!["b"]),
            make_node(OpKind::Relu, "n2", vec!["b"], vec!["output"]),
        ];
        let output_names = vec!["output".to_string()];
        let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();
        shape_map.insert("a".to_string(), vec![10]);
        shape_map.insert("b".to_string(), vec![10]);
        shape_map.insert("output".to_string(), vec![10]);
        let plan = MemoryPlan::compute(&nodes, &output_names, &shape_map);
        let slot_a = plan.buffer_assignments.get("a");
        let slot_b = plan.buffer_assignments.get("b");
        assert!(slot_a.is_some());
        assert!(slot_b.is_some());
        assert!(plan.buffer_sizes.len() >= 2);
    }
    #[test]
    fn test_buffer_reuse_strictly_non_overlapping() {
        let nodes = vec![
            make_node(OpKind::Relu, "n0", vec!["input"], vec!["a"]),
            make_node(OpKind::Relu, "n1", vec!["a"], vec!["c"]),
            make_node(OpKind::Relu, "n2", vec!["c"], vec!["b"]),
            make_node(OpKind::Relu, "n3", vec!["b"], vec!["output"]),
        ];
        let output_names = vec!["output".to_string()];
        let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();
        shape_map.insert("a".to_string(), vec![10]);
        shape_map.insert("b".to_string(), vec![10]);
        shape_map.insert("c".to_string(), vec![10]);
        shape_map.insert("output".to_string(), vec![10]);
        let plan = MemoryPlan::compute(&nodes, &output_names, &shape_map);
        let slot_a = plan.buffer_assignments.get("a");
        let slot_b = plan.buffer_assignments.get("b");
        assert!(slot_a.is_some());
        assert!(slot_b.is_some());
        assert_eq!(
            slot_a, slot_b,
            "non-overlapping tensors should share a slot"
        );
    }
    #[test]
    fn test_no_reuse_overlapping() {
        let nodes = vec![
            Node {
                op: OpKind::Split,
                name: "split".to_string(),
                inputs: vec!["input".to_string()],
                outputs: vec!["a".to_string(), "b".to_string()],
                attrs: Attributes::default(),
            },
            Node {
                op: OpKind::Add,
                name: "add".to_string(),
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["output".to_string()],
                attrs: Attributes::default(),
            },
        ];
        let output_names = vec!["output".to_string()];
        let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();
        shape_map.insert("a".to_string(), vec![5]);
        shape_map.insert("b".to_string(), vec![5]);
        shape_map.insert("output".to_string(), vec![5]);
        let plan = MemoryPlan::compute(&nodes, &output_names, &shape_map);
        let slot_a = plan.buffer_assignments.get("a");
        let slot_b = plan.buffer_assignments.get("b");
        assert!(slot_a.is_some());
        assert!(slot_b.is_some());
        assert_ne!(
            slot_a, slot_b,
            "overlapping tensors must have different slots"
        );
    }
    #[test]
    fn test_peak_memory_calculation() {
        let nodes = vec![
            Node {
                op: OpKind::Split,
                name: "split".to_string(),
                inputs: vec!["input".to_string()],
                outputs: vec!["a".to_string(), "b".to_string()],
                attrs: Attributes::default(),
            },
            Node {
                op: OpKind::Add,
                name: "add".to_string(),
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["c".to_string()],
                attrs: Attributes::default(),
            },
            make_node(OpKind::Relu, "relu", vec!["c"], vec!["output"]),
        ];
        let output_names = vec!["output".to_string()];
        let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();
        shape_map.insert("a".to_string(), vec![10]);
        shape_map.insert("b".to_string(), vec![10]);
        shape_map.insert("c".to_string(), vec![20]);
        shape_map.insert("output".to_string(), vec![20]);
        let plan = MemoryPlan::compute(&nodes, &output_names, &shape_map);
        assert_eq!(plan.peak_memory_elements, 40);
    }
    #[test]
    fn test_buffer_pool_get_return() {
        let mut pool = BufferPool::new();
        assert_eq!(pool.available_count(), 0);
        let buf = pool.get_buffer(100);
        assert_eq!(buf.len(), 100);
        assert!(buf.iter().all(|&v| v == 0.0));
        pool.return_buffer(buf);
        assert_eq!(pool.available_count(), 1);
        let buf2 = pool.get_buffer(100);
        assert_eq!(buf2.len(), 100);
        assert_eq!(pool.available_count(), 0);
    }
    #[test]
    fn test_buffer_pool_size_matching() {
        let mut pool = BufferPool::new();
        let small = vec![0.0_f32; 50];
        let medium = vec![0.0_f32; 200];
        let large = vec![0.0_f32; 500];
        pool.return_buffer(small);
        pool.return_buffer(large);
        pool.return_buffer(medium);
        assert_eq!(pool.available_count(), 3);
        let buf = pool.get_buffer(150);
        assert_eq!(buf.len(), 150);
        assert_eq!(pool.available_count(), 2);
        let buf2 = pool.get_buffer(10);
        assert_eq!(buf2.len(), 10);
        assert_eq!(pool.available_count(), 1);
    }
    #[test]
    fn test_buffer_pool_capacity_limit() {
        let mut pool = BufferPool::new();
        for i in 0..(MAX_POOL_BUFFERS + 10) {
            let buf = vec![0.0_f32; i + 1];
            pool.return_buffer(buf);
        }
        assert!(
            pool.available_count() <= MAX_POOL_BUFFERS,
            "pool size {} exceeds max {}",
            pool.available_count(),
            MAX_POOL_BUFFERS
        );
    }
    #[test]
    fn test_buffer_pool_clear() {
        let mut pool = BufferPool::new();
        pool.return_buffer(vec![0.0; 100]);
        pool.return_buffer(vec![0.0; 200]);
        assert_eq!(pool.available_count(), 2);
        pool.clear();
        assert_eq!(pool.available_count(), 0);
    }
    #[test]
    fn test_estimated_memory_bytes() {
        use crate::graph::Graph;
        use crate::tensor::Tensor;
        let nodes = vec![
            make_node(OpKind::Relu, "relu", vec!["x"], vec!["a"]),
            make_node(OpKind::Sigmoid, "sigmoid", vec!["a"], vec!["output"]),
        ];
        let graph = Graph {
            nodes,
            input_names: vec!["x".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };
        let weights: HashMap<String, Tensor> = HashMap::new();
        let session = crate::session::Session::builder()
            .with_optimization_level(crate::session::OptLevel::None)
            .with_memory_pool(true)
            .build_from_graph(graph, weights)
            .expect("build should succeed");
        let mut inputs = HashMap::new();
        inputs.insert("x", Tensor::new(vec![0.0; 10], vec![1, 10]));
        let result = session.run(&inputs);
        assert!(result.is_ok());
        let est = session.estimated_memory_bytes();
        let _ = est;
    }
    #[test]
    fn test_empty_graph_memory_plan() {
        let nodes: Vec<Node> = vec![];
        let output_names: Vec<String> = vec![];
        let shape_map: HashMap<String, Vec<usize>> = HashMap::new();
        let plan = MemoryPlan::compute(&nodes, &output_names, &shape_map);
        assert!(plan.lifetimes.is_empty());
        assert!(plan.buffer_assignments.is_empty());
        assert!(plan.buffer_sizes.is_empty());
        assert_eq!(plan.peak_memory_elements, 0);
    }
    #[test]
    fn test_size_class_acquire_release_preserves_content() {
        let mut pool = SizeClassPool::new();
        let mut buf = pool.acquire(100);
        for (i, val) in buf.iter_mut().enumerate() {
            *val = i as f32;
        }
        for (i, val) in buf.iter().enumerate() {
            assert_eq!(*val, i as f32);
        }
        pool.release(buf);
        let buf2 = pool.acquire(100);
        assert_eq!(buf2.len(), 100);
        assert!(buf2.iter().all(|&v| v == 0.0));
    }
    #[test]
    fn test_size_class_reuse_increments_count() {
        let mut pool = SizeClassPool::new();
        assert_eq!(pool.stats().alloc_count, 0);
        assert_eq!(pool.stats().reuse_count, 0);
        let buf = pool.acquire(64);
        assert_eq!(pool.stats().alloc_count, 1);
        assert_eq!(pool.stats().reuse_count, 0);
        pool.release(buf);
        let _buf2 = pool.acquire(64);
        assert_eq!(pool.stats().alloc_count, 1);
        assert_eq!(pool.stats().reuse_count, 1);
    }
    #[test]
    fn test_size_class_selection_tiny() {
        assert_eq!(bucket_for(0), SizeClass::Tiny);
        assert_eq!(bucket_for(1), SizeClass::Tiny);
        assert_eq!(bucket_for(127), SizeClass::Tiny);
    }
    #[test]
    fn test_size_class_selection_small() {
        assert_eq!(bucket_for(128), SizeClass::Small);
        assert_eq!(bucket_for(500), SizeClass::Small);
        assert_eq!(bucket_for(1023), SizeClass::Small);
    }
    #[test]
    fn test_size_class_selection_medium() {
        assert_eq!(bucket_for(1024), SizeClass::Medium);
        assert_eq!(bucket_for(8000), SizeClass::Medium);
        assert_eq!(bucket_for(16383), SizeClass::Medium);
    }
    #[test]
    fn test_size_class_selection_large() {
        assert_eq!(bucket_for(16384), SizeClass::Large);
        assert_eq!(bucket_for(100_000), SizeClass::Large);
        assert_eq!(bucket_for(1_000_000), SizeClass::Large);
    }
    #[test]
    fn test_size_class_best_fit() {
        let mut pool = SizeClassPool::new();
        let buf = vec![0.0_f32; 1000];
        pool.release(buf);
        let acquired = pool.acquire(500);
        assert_eq!(acquired.len(), 500);
        assert!(acquired.capacity() >= 500);
        assert_eq!(pool.stats().reuse_count, 1);
    }
    #[test]
    fn test_size_class_compact() {
        let mut pool = SizeClassPool::new();
        let mut oversized = Vec::with_capacity(512);
        oversized.resize(50, 0.0_f32);
        pool.release(oversized);
        let normal = vec![0.0_f32; 32];
        pool.release(normal);
        let bytes_before = pool.stats().current_bytes;
        assert!(bytes_before > 0);
        pool.compact();
        let bytes_after = pool.stats().current_bytes;
        assert!(
            bytes_after < bytes_before,
            "compact should free oversized buffers: before={bytes_before} after={bytes_after}"
        );
    }
    #[test]
    fn test_size_class_stats_tracking() {
        let mut pool = SizeClassPool::new();
        let b1 = pool.acquire(50);
        let b2 = pool.acquire(500);
        let b3 = pool.acquire(5000);
        assert_eq!(pool.stats().alloc_count, 3);
        assert_eq!(pool.stats().reuse_count, 0);
        pool.release(b1);
        pool.release(b2);
        pool.release(b3);
        assert!(pool.stats().current_bytes > 0);
        let _b4 = pool.acquire(50);
        let _b5 = pool.acquire(500);
        assert_eq!(pool.stats().alloc_count, 3);
        assert_eq!(pool.stats().reuse_count, 2);
    }
    #[test]
    fn test_size_class_default_enable() {
        let builder = crate::session::SessionBuilder::new();
        assert!(
            builder.enable_memory_pool,
            "memory pool should be enabled by default"
        );
    }
    #[test]
    fn test_size_class_multiple_cycles_no_leak() {
        let mut pool = SizeClassPool::new();
        for _ in 0..100 {
            let b1 = pool.acquire(64);
            let b2 = pool.acquire(256);
            let b3 = pool.acquire(2048);
            let b4 = pool.acquire(32768);
            pool.release(b1);
            pool.release(b2);
            pool.release(b3);
            pool.release(b4);
        }
        assert_eq!(
            pool.stats().alloc_count,
            4,
            "only first cycle should allocate new buffers"
        );
        assert_eq!(
            pool.stats().reuse_count,
            396,
            "remaining cycles should reuse"
        );
        pool.clear();
        assert_eq!(pool.stats().current_bytes, 0);
    }
    #[test]
    fn test_size_class_clear() {
        let mut pool = SizeClassPool::new();
        pool.release(vec![0.0_f32; 50]);
        pool.release(vec![0.0_f32; 200]);
        pool.release(vec![0.0_f32; 5000]);
        assert!(pool.stats().current_bytes > 0);
        pool.clear();
        assert_eq!(pool.stats().current_bytes, 0);
    }
    #[test]
    fn test_size_class_pool_stats_api() {
        use crate::graph::Graph;
        use crate::tensor::Tensor;
        let nodes = vec![
            make_node(OpKind::Relu, "relu", vec!["x"], vec!["a"]),
            make_node(OpKind::Sigmoid, "sigmoid", vec!["a"], vec!["output"]),
        ];
        let graph = Graph {
            nodes,
            input_names: vec!["x".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };
        let weights: HashMap<String, Tensor> = HashMap::new();
        let session = crate::session::Session::builder()
            .with_optimization_level(crate::session::OptLevel::None)
            .with_memory_pool(true)
            .build_from_graph(graph, weights)
            .expect("build should succeed");
        let stats = session.pool_stats();
        assert!(
            stats.is_some(),
            "pool_stats should return Some when pool is enabled"
        );
    }
    #[test]
    fn test_size_class_pool_stats_none_when_disabled() {
        use crate::graph::Graph;
        use crate::tensor::Tensor;
        let nodes = vec![
            make_node(OpKind::Relu, "relu", vec!["x"], vec!["a"]),
            make_node(OpKind::Sigmoid, "sigmoid", vec!["a"], vec!["output"]),
        ];
        let graph = Graph {
            nodes,
            input_names: vec!["x".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };
        let weights: HashMap<String, Tensor> = HashMap::new();
        let session = crate::session::Session::builder()
            .with_optimization_level(crate::session::OptLevel::None)
            .with_memory_pool(false)
            .build_from_graph(graph, weights)
            .expect("build should succeed");
        let stats = session.pool_stats();
        assert!(
            stats.is_none(),
            "pool_stats should return None when pool is disabled"
        );
    }
    #[test]
    fn test_size_class_zero_size_acquire() {
        let mut pool = SizeClassPool::new();
        let buf = pool.acquire(0);
        assert_eq!(buf.len(), 0);
        pool.release(buf);
        assert_eq!(pool.stats().current_bytes, 0);
    }
    #[test]
    fn test_size_class_cross_bucket_reuse() {
        let mut pool = SizeClassPool::new();
        let buf = vec![0.0_f32; 10000];
        pool.release(buf);
        let acquired = pool.acquire(5000);
        assert_eq!(acquired.len(), 5000);
        assert_eq!(pool.stats().reuse_count, 1);
    }
    #[test]
    fn test_size_class_peak_bytes_tracking() {
        let mut pool = SizeClassPool::new();
        let b1 = pool.acquire(1000);
        let b2 = pool.acquire(2000);
        pool.release(b1);
        pool.release(b2);
        let peak = pool.stats().peak_bytes;
        assert!(peak > 0, "peak_bytes should be positive after allocations");
        pool.clear();
        assert_eq!(pool.stats().peak_bytes, peak);
    }
}
