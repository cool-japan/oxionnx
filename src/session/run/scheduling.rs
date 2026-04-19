use crate::graph::Node;
use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Mutex;

use super::super::Session;
use super::state::SessionRunState;

impl Session {
    /// Compute the topological depth for each node in `sorted_nodes`.
    /// Depth 0 = all inputs come from model inputs / weights (no graph predecessors).
    /// For others, depth = max(depth of predecessor nodes) + 1.
    pub(crate) fn compute_node_depths(
        sorted_nodes: &[Node],
        weights: &HashMap<String, Tensor>,
    ) -> Vec<usize> {
        let mut tensor_depth: HashMap<&str, usize> = HashMap::new();
        let mut depths = Vec::with_capacity(sorted_nodes.len());

        for node in sorted_nodes {
            let mut max_pred_depth: Option<usize> = None;
            for inp in &node.inputs {
                if inp.is_empty() || weights.contains_key(inp) {
                    continue;
                }
                if let Some(&d) = tensor_depth.get(inp.as_str()) {
                    max_pred_depth = Some(match max_pred_depth {
                        Some(cur) => cur.max(d),
                        None => d,
                    });
                }
            }
            let depth = match max_pred_depth {
                Some(d) => d + 1,
                None => 0,
            };
            depths.push(depth);
            for out in &node.outputs {
                if !out.is_empty() {
                    tensor_depth.insert(out.as_str(), depth);
                }
            }
        }
        depths
    }

    /// Group node indices by their topological depth.
    pub(crate) fn group_by_depth(depths: &[usize]) -> Vec<Vec<usize>> {
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        let mut groups = vec![Vec::new(); max_depth + 1];
        for (i, &d) in depths.iter().enumerate() {
            groups[d].push(i);
        }
        groups
    }

    /// Decrement reference counts for a node's inputs via `SessionRunState`,
    /// freeing tensors that are no longer needed and returning buffers to the pool.
    pub(crate) fn decrement_refs_state(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) {
        for inp in &node.inputs {
            if inp.is_empty() || self.weights.contains_key(inp) {
                continue;
            }
            if let Some(count) = ref_counts.get_mut(inp) {
                *count = count.saturating_sub(1);
                if *count == 0 && !output_set.contains(inp.as_str()) {
                    // take from state and release buffer to pool
                    if let Some(mut tensor) = state.take(inp) {
                        if let Some(ref pool_mutex) = self.pool {
                            if let Ok(mut pool) = pool_mutex.lock() {
                                let buf = std::mem::take(&mut tensor.data);
                                if !buf.is_empty() {
                                    pool.release(buf);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Helper: pool reference from a Mutex<SizeClassPool>.
#[allow(dead_code)]
pub(super) fn pool_ref(pool: &Option<Mutex<SizeClassPool>>) -> Option<&Mutex<SizeClassPool>> {
    pool.as_ref().map(|m| m as &Mutex<SizeClassPool>)
}
