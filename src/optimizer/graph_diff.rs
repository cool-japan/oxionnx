//! Graph diff utility for debugging optimization passes.
//!
//! Compares two graphs and reports differences in nodes, edges, and attributes.

use oxionnx_core::graph::{Graph, Node};
use std::fmt;

/// A single difference between two graphs.
#[derive(Debug, Clone)]
pub enum GraphChange {
    /// A node was added (present in `after` but not `before`).
    NodeAdded { name: String, op_type: String },
    /// A node was removed (present in `before` but not `after`).
    NodeRemoved { name: String, op_type: String },
    /// A node's operator type changed.
    OpChanged {
        name: String,
        before: String,
        after: String,
    },
    /// A node's inputs changed.
    InputsChanged {
        name: String,
        before: Vec<String>,
        after: Vec<String>,
    },
    /// A node's outputs changed.
    OutputsChanged {
        name: String,
        before: Vec<String>,
        after: Vec<String>,
    },
    /// Graph input list changed.
    GraphInputsChanged {
        before: Vec<String>,
        after: Vec<String>,
    },
    /// Graph output list changed.
    GraphOutputsChanged {
        before: Vec<String>,
        after: Vec<String>,
    },
    /// Node count changed.
    NodeCountChanged { before: usize, after: usize },
}

impl fmt::Display for GraphChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeAdded { name, op_type } => {
                write!(f, "+ Node '{}' ({})", name, op_type)
            }
            Self::NodeRemoved { name, op_type } => {
                write!(f, "- Node '{}' ({})", name, op_type)
            }
            Self::OpChanged {
                name,
                before,
                after,
            } => {
                write!(f, "~ Node '{}': op {} -> {}", name, before, after)
            }
            Self::InputsChanged {
                name,
                before,
                after,
            } => {
                write!(
                    f,
                    "~ Node '{}': inputs [{}] -> [{}]",
                    name,
                    before.join(", "),
                    after.join(", ")
                )
            }
            Self::OutputsChanged {
                name,
                before,
                after,
            } => {
                write!(
                    f,
                    "~ Node '{}': outputs [{}] -> [{}]",
                    name,
                    before.join(", "),
                    after.join(", ")
                )
            }
            Self::GraphInputsChanged { before, after } => {
                write!(
                    f,
                    "~ Graph inputs: [{}] -> [{}]",
                    before.join(", "),
                    after.join(", ")
                )
            }
            Self::GraphOutputsChanged { before, after } => {
                write!(
                    f,
                    "~ Graph outputs: [{}] -> [{}]",
                    before.join(", "),
                    after.join(", ")
                )
            }
            Self::NodeCountChanged { before, after } => {
                write!(f, "~ Node count: {} -> {}", before, after)
            }
        }
    }
}

/// Result of comparing two graphs.
#[derive(Debug, Clone)]
pub struct GraphDiff {
    pub changes: Vec<GraphChange>,
}

impl GraphDiff {
    /// Compare two graphs and return all differences.
    pub fn compare(before: &Graph, after: &Graph) -> Self {
        let mut changes = Vec::new();

        // Check node count
        if before.nodes.len() != after.nodes.len() {
            changes.push(GraphChange::NodeCountChanged {
                before: before.nodes.len(),
                after: after.nodes.len(),
            });
        }

        // Check graph inputs/outputs
        if before.input_names != after.input_names {
            changes.push(GraphChange::GraphInputsChanged {
                before: before.input_names.clone(),
                after: after.input_names.clone(),
            });
        }
        if before.output_names != after.output_names {
            changes.push(GraphChange::GraphOutputsChanged {
                before: before.output_names.clone(),
                after: after.output_names.clone(),
            });
        }

        // Build maps by node name
        let before_map: std::collections::HashMap<&str, &Node> =
            before.nodes.iter().map(|n| (n.name.as_str(), n)).collect();
        let after_map: std::collections::HashMap<&str, &Node> =
            after.nodes.iter().map(|n| (n.name.as_str(), n)).collect();

        // Find removed nodes
        for (name, node) in &before_map {
            if !after_map.contains_key(name) {
                changes.push(GraphChange::NodeRemoved {
                    name: name.to_string(),
                    op_type: node.op.as_str().to_string(),
                });
            }
        }

        // Find added nodes and changed nodes
        for (name, after_node) in &after_map {
            match before_map.get(name) {
                None => {
                    changes.push(GraphChange::NodeAdded {
                        name: name.to_string(),
                        op_type: after_node.op.as_str().to_string(),
                    });
                }
                Some(before_node) => {
                    // Check for op type change
                    if before_node.op != after_node.op {
                        changes.push(GraphChange::OpChanged {
                            name: name.to_string(),
                            before: before_node.op.as_str().to_string(),
                            after: after_node.op.as_str().to_string(),
                        });
                    }
                    // Check for input changes
                    if before_node.inputs != after_node.inputs {
                        changes.push(GraphChange::InputsChanged {
                            name: name.to_string(),
                            before: before_node.inputs.clone(),
                            after: after_node.inputs.clone(),
                        });
                    }
                    // Check for output changes
                    if before_node.outputs != after_node.outputs {
                        changes.push(GraphChange::OutputsChanged {
                            name: name.to_string(),
                            before: before_node.outputs.clone(),
                            after: after_node.outputs.clone(),
                        });
                    }
                }
            }
        }

        Self { changes }
    }

    /// Whether the two graphs are identical.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Number of differences.
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Format as a human-readable report.
    pub fn report(&self) -> String {
        if self.changes.is_empty() {
            return "No differences found.".to_string();
        }
        let mut lines = Vec::with_capacity(self.changes.len() + 1);
        lines.push(format!("Graph diff: {} change(s)", self.changes.len()));
        for change in &self.changes {
            lines.push(format!("  {}", change));
        }
        lines.join("\n")
    }
}

impl fmt::Display for GraphDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Graph, Node, OpKind};

    fn make_node(op: OpKind, name: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> Node {
        Node {
            op,
            name: name.to_string(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
            attrs: Attributes::default(),
        }
    }

    fn make_graph(nodes: Vec<Node>) -> Graph {
        Graph {
            nodes,
            input_names: vec!["x".to_string()],
            output_names: vec!["out".to_string()],
        }
    }

    #[test]
    fn test_graph_diff_identical() {
        let g1 = make_graph(vec![
            make_node(OpKind::Relu, "relu1", vec!["x"], vec!["r1"]),
            make_node(OpKind::Add, "add1", vec!["r1", "r1"], vec!["out"]),
        ]);
        let g2 = make_graph(vec![
            make_node(OpKind::Relu, "relu1", vec!["x"], vec!["r1"]),
            make_node(OpKind::Add, "add1", vec!["r1", "r1"], vec!["out"]),
        ]);
        let diff = GraphDiff::compare(&g1, &g2);
        assert!(diff.is_empty());
        assert_eq!(diff.len(), 0);
    }

    #[test]
    fn test_graph_diff_node_added() {
        let g1 = make_graph(vec![make_node(
            OpKind::Relu,
            "relu1",
            vec!["x"],
            vec!["out"],
        )]);
        let g2 = make_graph(vec![
            make_node(OpKind::Relu, "relu1", vec!["x"], vec!["r1"]),
            make_node(OpKind::Sigmoid, "sig1", vec!["r1"], vec!["out"]),
        ]);
        let diff = GraphDiff::compare(&g1, &g2);
        assert!(!diff.is_empty());
        // Should have NodeCountChanged + NodeAdded + OutputsChanged for relu1
        let has_added = diff.changes.iter().any(|c| {
            matches!(c, GraphChange::NodeAdded { name, op_type }
                if name == "sig1" && op_type == "Sigmoid")
        });
        assert!(has_added, "Expected NodeAdded for sig1");
    }

    #[test]
    fn test_graph_diff_node_removed() {
        let g1 = make_graph(vec![
            make_node(OpKind::Relu, "relu1", vec!["x"], vec!["r1"]),
            make_node(OpKind::Sigmoid, "sig1", vec!["r1"], vec!["out"]),
        ]);
        let g2 = make_graph(vec![make_node(
            OpKind::Relu,
            "relu1",
            vec!["x"],
            vec!["out"],
        )]);
        let diff = GraphDiff::compare(&g1, &g2);
        assert!(!diff.is_empty());
        let has_removed = diff.changes.iter().any(|c| {
            matches!(c, GraphChange::NodeRemoved { name, op_type }
                if name == "sig1" && op_type == "Sigmoid")
        });
        assert!(has_removed, "Expected NodeRemoved for sig1");
    }

    #[test]
    fn test_graph_diff_op_changed() {
        let g1 = make_graph(vec![make_node(
            OpKind::Relu,
            "act1",
            vec!["x"],
            vec!["out"],
        )]);
        let g2 = make_graph(vec![make_node(
            OpKind::Sigmoid,
            "act1",
            vec!["x"],
            vec!["out"],
        )]);
        let diff = GraphDiff::compare(&g1, &g2);
        assert!(!diff.is_empty());
        let has_op_changed = diff.changes.iter().any(|c| {
            matches!(c, GraphChange::OpChanged { name, before, after }
                if name == "act1" && before == "Relu" && after == "Sigmoid")
        });
        assert!(has_op_changed, "Expected OpChanged for act1");
    }

    #[test]
    fn test_graph_diff_inputs_changed() {
        let g1 = make_graph(vec![make_node(
            OpKind::Add,
            "add1",
            vec!["x", "y"],
            vec!["out"],
        )]);
        let g2 = make_graph(vec![make_node(
            OpKind::Add,
            "add1",
            vec!["x", "z"],
            vec!["out"],
        )]);
        let diff = GraphDiff::compare(&g1, &g2);
        assert!(!diff.is_empty());
        let has_inputs_changed = diff
            .changes
            .iter()
            .any(|c| matches!(c, GraphChange::InputsChanged { name, .. } if name == "add1"));
        assert!(has_inputs_changed, "Expected InputsChanged for add1");
    }

    #[test]
    fn test_graph_diff_report() {
        let g1 = make_graph(vec![make_node(
            OpKind::Relu,
            "relu1",
            vec!["x"],
            vec!["out"],
        )]);
        let g2 = make_graph(vec![make_node(
            OpKind::Sigmoid,
            "relu1",
            vec!["x"],
            vec!["out"],
        )]);
        let diff = GraphDiff::compare(&g1, &g2);
        let report = diff.report();
        assert!(report.contains("Graph diff:"));
        assert!(report.contains("change(s)"));
        assert!(report.contains("relu1"));
        assert!(report.contains("Relu"));
        assert!(report.contains("Sigmoid"));
    }
}
