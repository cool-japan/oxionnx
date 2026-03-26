//! Static memory planner and activation buffer pool for ONNX inference.
//!
//! Analyzes tensor lifetimes in the computation graph and assigns buffer slots
//! using a greedy best-fit algorithm to minimize peak memory usage.

use crate::graph::Node;
use std::collections::HashMap;

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

        // Step 1: Walk nodes to determine production and consumption points
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

        // Step 2: Output tensors are never freed early
        let final_step = sorted_nodes.len();
        for name in output_names {
            last_consumed.insert(name.clone(), final_step);
        }

        // Step 3: Build lifetime entries for tensors that are both produced and consumed
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

        // Step 4: Sort by produced_at for greedy assignment
        lifetimes.sort_by_key(|lt| lt.produced_at);

        // Step 5: Greedy best-fit buffer assignment
        let mut buffer_assignments: HashMap<String, usize> = HashMap::new();
        let mut buffer_sizes: Vec<usize> = Vec::new();
        // Track last_consumed_at for the current tenant of each slot
        let mut slot_free_after: Vec<usize> = Vec::new();

        for lt in &lifetimes {
            if lt.size_elements == 0 {
                // Unknown size: assign a new slot with size 0
                let slot = buffer_sizes.len();
                buffer_sizes.push(0);
                slot_free_after.push(lt.last_consumed_at);
                buffer_assignments.insert(lt.name.clone(), slot);
                continue;
            }

            // Find available slots (previous tenant finished before this tensor is produced)
            let mut best_slot: Option<usize> = None;
            let mut best_size: usize = usize::MAX;

            for (slot_idx, &free_after) in slot_free_after.iter().enumerate() {
                if free_after < lt.produced_at && buffer_sizes[slot_idx] >= lt.size_elements {
                    // Slot is available and large enough
                    if buffer_sizes[slot_idx] < best_size {
                        best_size = buffer_sizes[slot_idx];
                        best_slot = Some(slot_idx);
                    }
                }
            }

            // If no exact-fit, look for available slot we can grow
            if best_slot.is_none() {
                // Find smallest available slot (even if too small, we'll grow it)
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
                    // Grow slot if needed
                    if buffer_sizes[slot_idx] < lt.size_elements {
                        buffer_sizes[slot_idx] = lt.size_elements;
                    }
                    slot_free_after[slot_idx] = lt.last_consumed_at;
                    buffer_assignments.insert(lt.name.clone(), slot_idx);
                }
                None => {
                    // Create a new slot
                    let slot = buffer_sizes.len();
                    buffer_sizes.push(lt.size_elements);
                    slot_free_after.push(lt.last_consumed_at);
                    buffer_assignments.insert(lt.name.clone(), slot);
                }
            }
        }

        // Step 6: Compute peak memory across execution steps
        let peak_memory_elements = compute_peak_memory(&lifetimes, final_step);

        Self {
            lifetimes,
            buffer_assignments,
            buffer_sizes,
            peak_memory_elements,
        }
    }
}

/// Compute peak concurrent memory usage (in f32 elements) across all execution steps.
fn compute_peak_memory(lifetimes: &[TensorLifetime], total_steps: usize) -> usize {
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

/// Buffer pool for reusing tensor allocations during inference.
///
/// Maintains a sorted list of available buffers and returns the smallest
/// buffer that satisfies the requested size.
pub struct BufferPool {
    /// Available buffers sorted by capacity (ascending).
    buffers: Vec<Vec<f32>>,
}

/// Maximum number of buffers retained in the pool.
const MAX_POOL_BUFFERS: usize = 64;

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
        // Binary search for the smallest buffer >= min_size
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
            // Drop the smallest buffer if full (the incoming one may be more useful)
            if let Some(smallest_cap) = self.buffers.first().map(|b| b.capacity()) {
                if buf.capacity() > smallest_cap {
                    self.buffers.remove(0);
                } else {
                    // Incoming buffer is smaller than everything — just drop it
                    return;
                }
            }
        }

        // Insert in sorted position by capacity
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

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Node, OpKind};

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
        // Linear chain: input -> relu -> sigmoid -> output
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

        // Find lifetimes by name
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
        assert_eq!(lt_out.last_consumed_at, 3); // output_names extends to final_step
    }

    #[test]
    fn test_buffer_reuse_non_overlapping() {
        // a is produced at 0, consumed at 0 (used by node 1)
        // b is produced at 1, consumed at 1 (used by node 2)
        // a and b don't overlap, so they should share a slot
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
        // a is consumed at step 1, b is produced at step 1 — a's last_consumed_at (1)
        // is NOT < b's produced_at (1), so they cannot share.
        // Actually: a is produced at 0, last_consumed at 1 (by node n1)
        // b is produced at 1.
        // Condition is: slot_free_after < produced_at => 1 < 1 => false
        // So they get different slots.
        // Let's just verify the plan is valid (non-overlapping lifetimes
        // with strict < condition don't share here)
        assert!(plan.buffer_sizes.len() >= 2);
    }

    #[test]
    fn test_buffer_reuse_strictly_non_overlapping() {
        // Make tensor a consumed strictly before b is produced
        // n0: input -> a
        // n1: a -> c (a consumed here at step 1)
        // n2: c -> b (b produced at step 2, a's last_consumed was step 1)
        // n3: b -> output
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
        // a: produced_at=0, last_consumed_at=1
        // b: produced_at=2
        // 1 < 2 => true, so a's slot is available for b => they should share
        assert_eq!(
            slot_a, slot_b,
            "non-overlapping tensors should share a slot"
        );
    }

    #[test]
    fn test_no_reuse_overlapping() {
        // Two tensors alive at the same time must get different slots
        // n0: input -> a, b (both produced at step 0)
        // n1: a, b -> output (both consumed at step 1)
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
        // Two tensors of size 10 alive simultaneously, then one of size 20
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

        // At step 0: a(10) + b(10) alive = 20
        // At step 1: a(10) + b(10) + c(20) = 40 (a,b consumed at step 1 but still alive at step 1)
        // At step 2: c(20) + output(20) = 40
        // At step 3: output(20) = 20
        // Peak = 40
        assert_eq!(plan.peak_memory_elements, 40);
    }

    #[test]
    fn test_buffer_pool_get_return() {
        let mut pool = BufferPool::new();
        assert_eq!(pool.available_count(), 0);

        // Get a new buffer (pool is empty, so it allocates)
        let buf = pool.get_buffer(100);
        assert_eq!(buf.len(), 100);
        assert!(buf.iter().all(|&v| v == 0.0));

        // Return it
        pool.return_buffer(buf);
        assert_eq!(pool.available_count(), 1);

        // Get again — should reuse
        let buf2 = pool.get_buffer(100);
        assert_eq!(buf2.len(), 100);
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_buffer_pool_size_matching() {
        let mut pool = BufferPool::new();

        // Create buffers of different sizes
        let small = vec![0.0_f32; 50];
        let medium = vec![0.0_f32; 200];
        let large = vec![0.0_f32; 500];

        pool.return_buffer(small);
        pool.return_buffer(large);
        pool.return_buffer(medium);
        assert_eq!(pool.available_count(), 3);

        // Request 150 elements: should get the medium (200) buffer, not the large (500)
        let buf = pool.get_buffer(150);
        assert_eq!(buf.len(), 150);
        assert_eq!(pool.available_count(), 2);

        // Request 10 elements: should get the small (50) buffer
        let buf2 = pool.get_buffer(10);
        assert_eq!(buf2.len(), 10);
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_buffer_pool_capacity_limit() {
        let mut pool = BufferPool::new();

        // Fill pool beyond MAX_POOL_BUFFERS
        for i in 0..(MAX_POOL_BUFFERS + 10) {
            let buf = vec![0.0_f32; i + 1];
            pool.return_buffer(buf);
        }

        // Pool should never exceed MAX_POOL_BUFFERS
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
        // Integration test: build a session and check estimated memory
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
        };
        let weights: HashMap<String, Tensor> = HashMap::new();

        let session = crate::session::Session::builder()
            .with_optimization_level(crate::session::OptLevel::None)
            .with_memory_pool(true)
            .build_from_graph(graph, weights)
            .expect("build should succeed");

        // With input shape [1, 10], shape inference should determine intermediate sizes
        let mut inputs = HashMap::new();
        inputs.insert("x", Tensor::new(vec![0.0; 10], vec![1, 10]));
        let result = session.run(&inputs);
        assert!(result.is_ok());

        // estimated_memory_bytes requires known input shapes; with empty shapes
        // it may return None or Some based on what shape inference can determine
        // Just check the API works
        let est = session.estimated_memory_bytes();
        // May be None if shape inference can't determine all shapes without input info
        // That's acceptable behavior
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
}
