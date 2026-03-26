//! Dead node elimination pass.

use crate::graph::Node;
use std::collections::HashSet;

/// Dead node elimination: remove nodes whose outputs are never consumed.
/// Walk backward from output_names and keep only reachable nodes.
pub fn dead_node_elimination(nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
    let mut live_tensors: HashSet<String> = output_names.iter().cloned().collect();
    let mut live_nodes: Vec<bool> = vec![false; nodes.len()];

    // Backward pass
    for (i, node) in nodes.iter().enumerate().rev() {
        let node_is_live = node.outputs.iter().any(|o| live_tensors.contains(o));
        if node_is_live {
            live_nodes[i] = true;
            for inp in &node.inputs {
                if !inp.is_empty() {
                    live_tensors.insert(inp.clone());
                }
            }
        }
    }

    nodes
        .into_iter()
        .zip(live_nodes)
        .filter(|(_, live)| *live)
        .map(|(n, _)| n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    #[test]
    fn test_dead_node_elimination() {
        let nodes = vec![
            make_node(OpKind::MatMul, "matmul1", vec!["x", "w1"], vec!["mm1"]),
            make_node(OpKind::Relu, "relu1", vec!["mm1"], vec!["r1"]),
            make_node(OpKind::MatMul, "matmul2", vec!["x", "w2"], vec!["mm2"]), // dead branch
            make_node(OpKind::Relu, "relu2", vec!["mm2"], vec!["r2"]),          // dead
        ];
        let output_names = vec!["r1".to_string()];
        let result = dead_node_elimination(nodes, &output_names);
        assert_eq!(result.len(), 2); // Only matmul1 + relu1
        assert_eq!(result[0].name, "matmul1");
        assert_eq!(result[1].name, "relu1");
    }

    #[test]
    fn test_dead_node_elimination_empty() {
        let nodes: Vec<Node> = vec![];
        let output_names: Vec<String> = vec![];
        let result = dead_node_elimination(nodes, &output_names);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dead_node_elimination_all_live() {
        let nodes = vec![
            make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
            make_node(OpKind::Relu, "relu", vec!["mm_out"], vec!["out"]),
        ];
        let output_names = vec!["out".to_string()];
        let result = dead_node_elimination(nodes, &output_names);
        assert_eq!(result.len(), 2);
    }
}
