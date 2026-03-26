//! Constant folding optimization pass.

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use oxionnx_core::{OpContext, OperatorRegistry};
use std::collections::{HashMap, HashSet};

/// Constant folding: evaluate nodes whose inputs are all known constants
/// (present in `weights`). Store results back in `weights` and remove the node.
/// Iterates until no more nodes can be folded.
pub fn constant_fold(
    mut nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    registry: &OperatorRegistry,
) -> Vec<Node> {
    // Ops that depend on runtime state or have side effects — skip these.
    let skip_ops: HashSet<&str> = [
        "Shape",
        "Dropout",
        "RandomNormal",
        "RandomUniform",
        "RandomNormalLike",
        "RandomUniformLike",
    ]
    .iter()
    .copied()
    .collect();

    loop {
        let mut folded_any = false;
        let mut keep = Vec::with_capacity(nodes.len());

        for node in nodes {
            // Skip unknown ops and ops with runtime dependencies
            if matches!(node.op, OpKind::Unknown(_)) {
                keep.push(node);
                continue;
            }
            let op_name = node.op.as_str();
            if skip_ops.contains(op_name) {
                keep.push(node);
                continue;
            }

            // Check if ALL non-empty inputs are in weights
            let all_inputs_const = node
                .inputs
                .iter()
                .all(|inp| inp.is_empty() || weights.contains_key(inp));

            if !all_inputs_const {
                keep.push(node);
                continue;
            }

            // Must have at least one non-empty input to be worth folding
            let has_inputs = node.inputs.iter().any(|inp| !inp.is_empty());
            if !has_inputs {
                keep.push(node);
                continue;
            }

            // Look up the operator
            let operator = match registry.get(op_name) {
                Some(op) => op,
                None => {
                    keep.push(node);
                    continue;
                }
            };

            // Build OpContext with resolved inputs
            let resolved_inputs: Vec<Option<&Tensor>> = node
                .inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        weights.get(name)
                    }
                })
                .collect();

            let ctx = OpContext {
                node: &node,
                inputs: resolved_inputs,
                outer_scope: None,
                registry: None,
            };

            // Execute the operator
            match operator.execute(&ctx) {
                Ok(results) => {
                    // Store each output tensor in weights
                    for (out_name, tensor) in node.outputs.iter().zip(results) {
                        if !out_name.is_empty() {
                            weights.insert(out_name.clone(), tensor);
                        }
                    }
                    folded_any = true;
                    // Node is removed (not pushed to `keep`)
                }
                Err(_) => {
                    // If execution fails, keep the node
                    keep.push(node);
                }
            }
        }

        nodes = keep;
        if !folded_any {
            break;
        }
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    #[test]
    fn test_constant_fold_both_constants() {
        let add_node = make_node(OpKind::Add, "add1", vec!["a", "b"], vec!["sum"]);
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));
        weights.insert("b".to_string(), Tensor::new(vec![4.0, 5.0, 6.0], vec![3]));

        let registry = oxionnx_ops::default_registry();
        let result = constant_fold(vec![add_node], &mut weights, &registry);

        assert!(
            result.is_empty(),
            "Expected no nodes after folding, got {}",
            result.len()
        );
        let sum = weights.get("sum").expect("sum should be in weights");
        assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_constant_fold_mixed_inputs() {
        let add_runtime = make_node(OpKind::Add, "add_rt", vec!["x", "a"], vec!["rt_out"]);
        let add_const = make_node(OpKind::Add, "add_const", vec!["b", "c"], vec!["const_out"]);

        let mut weights = HashMap::new();
        weights.insert("a".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("b".to_string(), Tensor::new(vec![2.0, 3.0], vec![2]));
        weights.insert("c".to_string(), Tensor::new(vec![4.0, 5.0], vec![2]));

        let registry = oxionnx_ops::default_registry();
        let result = constant_fold(vec![add_runtime, add_const], &mut weights, &registry);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "add_rt");
        let const_out = weights.get("const_out").expect("const_out in weights");
        assert_eq!(const_out.data, vec![6.0, 8.0]);
    }

    #[test]
    fn test_constant_fold_no_registry_op() {
        let add_node = make_node(OpKind::Add, "add1", vec!["a", "b"], vec!["sum"]);
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("b".to_string(), Tensor::new(vec![2.0], vec![1]));

        let registry = OperatorRegistry::new(); // empty registry
        let result = constant_fold(vec![add_node], &mut weights, &registry);

        assert_eq!(result.len(), 1);
        assert!(!weights.contains_key("sum"));
    }

    #[test]
    fn test_constant_fold_chain() {
        let add1 = make_node(OpKind::Add, "add1", vec!["a", "b"], vec!["c"]);
        let add2 = make_node(OpKind::Add, "add2", vec!["c", "d"], vec!["e"]);

        let mut weights = HashMap::new();
        weights.insert("a".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("b".to_string(), Tensor::new(vec![2.0], vec![1]));
        weights.insert("d".to_string(), Tensor::new(vec![10.0], vec![1]));

        let registry = oxionnx_ops::default_registry();
        let result = constant_fold(vec![add1, add2], &mut weights, &registry);

        assert!(result.is_empty());
        let e = weights.get("e").expect("e in weights");
        assert_eq!(e.data, vec![13.0]);
    }
}
