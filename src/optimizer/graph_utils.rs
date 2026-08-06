//! Shared preconditions and helpers for the optimizer passes.
//!
//! Every rewrite that *removes* an intermediate tensor has the same two
//! soundness obligations:
//!
//! 1. the tensor must be consumed by exactly one node — otherwise the other
//!    consumers lose their producer, and
//! 2. the tensor must not be a declared graph output — otherwise `Session::run`
//!    silently returns fewer outputs than the model declares.
//!
//! [`TensorUsage`] answers both questions from one pre-computed map so that a
//! pass never has to re-derive them (and never forgets the second one).
//!
//! Passes that synthesise new tensors (folded weights, composed indices, …)
//! must not key those names on `Node::name`: `NodeProto.name` is optional in
//! the ONNX spec and exporters routinely emit `""` or duplicates, which makes
//! two independent fusions collide on one `weights` entry.  [`NameAllocator`]
//! derives collision-free names from the (spec-unique) tensor names instead.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Per-tensor usage information for one node list.
///
/// Combines the number of node-level consumers of each tensor with the set of
/// declared graph outputs, which a pass must treat as an extra consumer that
/// can never be rewritten away.
pub(crate) struct TensorUsage {
    consumer_count: HashMap<String, usize>,
    graph_outputs: HashSet<String>,
}

impl TensorUsage {
    /// Build the usage map for `nodes` given the graph's declared outputs.
    pub(crate) fn new(nodes: &[Node], output_names: &[String]) -> Self {
        let mut consumer_count: HashMap<String, usize> = HashMap::new();
        for node in nodes {
            for inp in &node.inputs {
                if !inp.is_empty() {
                    *consumer_count.entry(inp.clone()).or_insert(0) += 1;
                }
            }
        }
        let graph_outputs: HashSet<String> = output_names
            .iter()
            .filter(|name| !name.is_empty())
            .cloned()
            .collect();
        Self {
            consumer_count,
            graph_outputs,
        }
    }

    /// Number of nodes that consume `name` as an input.
    pub(crate) fn consumers(&self, name: &str) -> usize {
        self.consumer_count.get(name).copied().unwrap_or(0)
    }

    /// `true` when `name` is one of the graph's declared output tensors.
    pub(crate) fn is_graph_output(&self, name: &str) -> bool {
        self.graph_outputs.contains(name)
    }

    /// A pass may fuse away (delete, rename or reparent) the tensor `name`
    /// only when it has exactly one consumer and is not a graph output.
    pub(crate) fn is_fusable_intermediate(&self, name: &str) -> bool {
        !name.is_empty() && !self.is_graph_output(name) && self.consumers(name) == 1
    }

    /// `true` when *none* of `names` is a declared graph output.  Used before
    /// deleting a node outright (CSE duplicates, constant-folded nodes): the
    /// node's outputs would stop being produced by anything.
    pub(crate) fn none_is_graph_output(&self, names: &[String]) -> bool {
        !names.iter().any(|name| self.is_graph_output(name))
    }
}

/// Allocator for tensor names synthesised by a pass.
///
/// Seeded with every name already present in the graph (node inputs, node
/// outputs) and in `weights`, so a generated name can never shadow a real
/// tensor, and two generated names can never collide with each other.
pub(crate) struct NameAllocator {
    taken: HashSet<String>,
}

impl NameAllocator {
    /// Seed the allocator from the graph's tensor names and the weight map.
    pub(crate) fn new(nodes: &[Node], weights: &HashMap<String, Tensor>) -> Self {
        let mut taken: HashSet<String> = weights.keys().cloned().collect();
        for node in nodes {
            for name in node.inputs.iter().chain(node.outputs.iter()) {
                if !name.is_empty() {
                    taken.insert(name.clone());
                }
            }
        }
        Self { taken }
    }

    /// Return an unused name of the form `{base}{suffix}`, appending `_1`,
    /// `_2`, … until it is unique.  The result is recorded as taken.
    ///
    /// `base` should be a tensor name (ONNX requires those to be unique within
    /// a graph), never `Node::name`, which may be empty or duplicated.
    pub(crate) fn allocate(&mut self, base: &str, suffix: &str) -> String {
        let mut candidate = format!("{base}{suffix}");
        let mut counter: usize = 1;
        while self.taken.contains(&candidate) {
            candidate = format!("{base}{suffix}_{counter}");
            counter += 1;
        }
        self.taken.insert(candidate.clone());
        candidate
    }
}

/// Names in `output_names` that no node produces and that are not constants.
///
/// A non-empty result means some pass removed or renamed a declared graph
/// output — `SessionRunState::take_outputs` would silently drop it from the
/// returned map.  Used by the optimizer regression tests as a blanket
/// post-condition for every pass.
#[cfg(test)]
pub(crate) fn missing_graph_outputs(
    nodes: &[Node],
    weights: &HashMap<String, Tensor>,
    output_names: &[String],
) -> Vec<String> {
    let produced: HashSet<&str> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter())
        .map(String::as_str)
        .collect();
    output_names
        .iter()
        .filter(|name| !name.is_empty())
        .filter(|name| !produced.contains(name.as_str()) && !weights.contains_key(name.as_str()))
        .cloned()
        .collect()
}

/// Resolve a constant scalar operand (a graph initializer) to `f32`.
///
/// Returns `None` when the name is empty (an omitted optional input), when the
/// tensor is not a compile-time constant, or when it is not a single value.
pub(crate) fn const_scalar(weights: &HashMap<String, Tensor>, name: &str) -> Option<f32> {
    if name.is_empty() {
        return None;
    }
    let tensor = weights.get(name)?;
    if tensor.numel() != 1 {
        return None;
    }
    tensor.data.first().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    #[test]
    fn test_fusable_intermediate_requires_single_consumer() {
        let nodes = vec![
            make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]),
            make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["y"]),
            make_node(OpKind::Sigmoid, "sig", vec!["conv_out"], vec!["z"]),
        ];
        let usage = TensorUsage::new(&nodes, &["y".to_string(), "z".to_string()]);
        assert_eq!(usage.consumers("conv_out"), 2);
        assert!(!usage.is_fusable_intermediate("conv_out"));
    }

    #[test]
    fn test_fusable_intermediate_rejects_graph_output() {
        let nodes = vec![
            make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]),
            make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["y"]),
        ];
        let usage = TensorUsage::new(&nodes, &["y".to_string(), "conv_out".to_string()]);
        assert_eq!(usage.consumers("conv_out"), 1);
        assert!(usage.is_graph_output("conv_out"));
        assert!(!usage.is_fusable_intermediate("conv_out"));
        assert!(!usage.none_is_graph_output(&["conv_out".to_string()]));
    }

    #[test]
    fn test_name_allocator_avoids_existing_names() {
        let nodes = vec![make_node(
            OpKind::Conv,
            "",
            vec!["x", "w"],
            vec!["conv_out"],
        )];
        let mut weights = HashMap::new();
        weights.insert(
            "conv_out_fused_weight".to_string(),
            Tensor::new(vec![1.0], vec![1]),
        );
        let mut alloc = NameAllocator::new(&nodes, &weights);
        let first = alloc.allocate("conv_out", "_fused_weight");
        assert_eq!(first, "conv_out_fused_weight_1");
        let second = alloc.allocate("conv_out", "_fused_weight");
        assert_eq!(second, "conv_out_fused_weight_2");
        assert_ne!(first, second);
    }

    #[test]
    fn test_missing_graph_outputs_detects_dropped_output() {
        let nodes = vec![make_node(OpKind::Relu, "relu", vec!["x"], vec!["y"])];
        let mut weights = HashMap::new();
        weights.insert("c".to_string(), Tensor::new(vec![1.0], vec![1]));
        let outputs = vec!["y".to_string(), "c".to_string(), "gone".to_string()];
        let missing = missing_graph_outputs(&nodes, &weights, &outputs);
        assert_eq!(missing, vec!["gone".to_string()]);
    }

    #[test]
    fn test_const_scalar() {
        let mut weights = HashMap::new();
        weights.insert("s".to_string(), Tensor::new(vec![6.0], vec![1]));
        weights.insert("v".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
        assert_eq!(const_scalar(&weights, "s"), Some(6.0));
        assert_eq!(const_scalar(&weights, "v"), None);
        assert_eq!(const_scalar(&weights, ""), None);
        assert_eq!(const_scalar(&weights, "missing"), None);
    }
}
