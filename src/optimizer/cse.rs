//! Common Subexpression Elimination (CSE) optimization pass.

use crate::graph::Node;
use std::collections::HashMap;

/// Compute a deterministic fingerprint for a node based on its op type,
/// sorted inputs, and canonical attribute representation.
fn node_fingerprint(node: &Node) -> String {
    let op_str = node.op.as_str();

    // Sort inputs for canonical ordering
    let mut sorted_inputs = node.inputs.clone();
    sorted_inputs.sort();
    let inputs_str = sorted_inputs.join(",");

    // Build canonical attribute string with sorted keys
    let mut attr_parts: Vec<String> = Vec::new();

    // Floats
    let mut float_keys: Vec<&String> = node.attrs.floats.keys().collect();
    float_keys.sort();
    for k in float_keys {
        if let Some(v) = node.attrs.floats.get(k) {
            attr_parts.push(format!("f:{k}={v}"));
        }
    }

    // Ints
    let mut int_keys: Vec<&String> = node.attrs.ints.keys().collect();
    int_keys.sort();
    for k in int_keys {
        if let Some(v) = node.attrs.ints.get(k) {
            attr_parts.push(format!("i:{k}={v}"));
        }
    }

    // Strings
    let mut str_keys: Vec<&String> = node.attrs.strings.keys().collect();
    str_keys.sort();
    for k in str_keys {
        if let Some(v) = node.attrs.strings.get(k) {
            attr_parts.push(format!("s:{k}={v}"));
        }
    }

    // Int lists
    let mut il_keys: Vec<&String> = node.attrs.int_lists.keys().collect();
    il_keys.sort();
    for k in il_keys {
        if let Some(v) = node.attrs.int_lists.get(k) {
            attr_parts.push(format!("il:{k}={v:?}"));
        }
    }

    // Float lists
    let mut fl_keys: Vec<&String> = node.attrs.float_lists.keys().collect();
    fl_keys.sort();
    for k in fl_keys {
        if let Some(v) = node.attrs.float_lists.get(k) {
            attr_parts.push(format!("fl:{k}={v:?}"));
        }
    }

    // Tensors: include shape and a hash of the data
    let mut t_keys: Vec<&String> = node.attrs.tensors.keys().collect();
    t_keys.sort();
    for k in t_keys {
        if let Some(t) = node.attrs.tensors.get(k) {
            // Simple hash: sum of all elements as a fingerprint component
            let data_hash: u64 = t
                .data
                .iter()
                .fold(0u64, |acc, &val| acc.wrapping_add(val.to_bits() as u64));
            attr_parts.push(format!("t:{k}=shape{:?}hash{data_hash}", t.shape));
        }
    }

    let attrs_str = attr_parts.join(";");
    format!("{op_str}|{inputs_str}|{attrs_str}")
}

/// Eliminate common subexpressions from the node list.
///
/// Two nodes are considered duplicates when they have the same op type,
/// the same set of inputs (order-independent), and identical attributes.
/// The first occurrence is kept; later duplicates are removed and all
/// references to their outputs are redirected to the original's outputs.
pub fn eliminate_common_subexpressions(nodes: Vec<Node>) -> Vec<Node> {
    // fingerprint -> output names of the first node with that fingerprint
    let mut seen: HashMap<String, Vec<String>> = HashMap::new();
    // redirect map: duplicate output name -> original output name
    let mut redirects: HashMap<String, String> = HashMap::new();
    // Indices of duplicate nodes to remove
    let mut duplicate_indices: Vec<bool> = vec![false; nodes.len()];

    for (idx, node) in nodes.iter().enumerate() {
        let fp = node_fingerprint(node);

        if let Some(original_outputs) = seen.get(&fp) {
            // This is a duplicate — build redirect map
            for (dup_out, orig_out) in node.outputs.iter().zip(original_outputs.iter()) {
                if !dup_out.is_empty() && !orig_out.is_empty() {
                    redirects.insert(dup_out.clone(), orig_out.clone());
                }
            }
            duplicate_indices[idx] = true;
        } else {
            seen.insert(fp, node.outputs.clone());
        }
    }

    // Apply redirects and filter out duplicates
    nodes
        .into_iter()
        .zip(duplicate_indices)
        .filter(|(_, is_dup)| !is_dup)
        .map(|(mut node, _)| {
            // Redirect any inputs that point to removed duplicates
            for inp in &mut node.inputs {
                if let Some(redirect) = redirects.get(inp) {
                    *inp = redirect.clone();
                }
            }
            node
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    #[test]
    fn test_cse_removes_duplicate_nodes() {
        // Two identical Add nodes with same inputs
        let nodes = vec![
            make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]),
            make_node(OpKind::Add, "add2", vec!["x", "y"], vec!["out2"]),
        ];
        let result = eliminate_common_subexpressions(nodes);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "add1");
    }

    #[test]
    fn test_cse_preserves_different_attrs() {
        // Same op and inputs, but different attributes -> both kept
        let mut node1 = make_node(OpKind::Conv, "conv1", vec!["x", "w"], vec!["out1"]);
        node1.attrs.ints.insert("group".to_string(), 1);

        let mut node2 = make_node(OpKind::Conv, "conv2", vec!["x", "w"], vec!["out2"]);
        node2.attrs.ints.insert("group".to_string(), 4);

        let nodes = vec![node1, node2];
        let result = eliminate_common_subexpressions(nodes);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_redirects_consumers() {
        // add1 and add2 are identical: Add(x, y)
        // relu consumes add2's output -> should be redirected to add1's output
        let nodes = vec![
            make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]),
            make_node(OpKind::Add, "add2", vec!["x", "y"], vec!["out2"]),
            make_node(OpKind::Relu, "relu", vec!["out2"], vec!["relu_out"]),
        ];
        let result = eliminate_common_subexpressions(nodes);
        assert_eq!(result.len(), 2); // add1 + relu
        assert_eq!(result[0].name, "add1");
        assert_eq!(result[1].name, "relu");
        assert_eq!(result[1].inputs[0], "out1"); // redirected from out2 -> out1
    }

    #[test]
    fn test_cse_different_inputs_not_removed() {
        // Same op but different inputs -> both kept
        let nodes = vec![
            make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]),
            make_node(OpKind::Add, "add2", vec!["x", "z"], vec!["out2"]),
        ];
        let result = eliminate_common_subexpressions(nodes);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_different_ops_not_removed() {
        // Different ops with same inputs -> both kept
        let nodes = vec![
            make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]),
            make_node(OpKind::Mul, "mul1", vec!["x", "y"], vec!["out2"]),
        ];
        let result = eliminate_common_subexpressions(nodes);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_preserves_different_float_attrs() {
        let mut node1 = make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]);
        node1.attrs.floats.insert("alpha".to_string(), 1.0);

        let mut node2 = make_node(OpKind::Add, "add2", vec!["x", "y"], vec!["out2"]);
        node2.attrs.floats.insert("alpha".to_string(), 2.0);

        let nodes = vec![node1, node2];
        let result = eliminate_common_subexpressions(nodes);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_empty_graph() {
        let nodes: Vec<Node> = vec![];
        let result = eliminate_common_subexpressions(nodes);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cse_with_attrs_match() {
        // Same op, same inputs, same attributes -> duplicate removed
        let mut node1 = make_node(OpKind::Gemm, "gemm1", vec!["x", "w", "b"], vec!["out1"]);
        node1.attrs.floats.insert("alpha".to_string(), 1.0);
        node1.attrs.ints.insert("transB".to_string(), 1);

        let mut node2 = make_node(OpKind::Gemm, "gemm2", vec!["x", "w", "b"], vec!["out2"]);
        node2.attrs.floats.insert("alpha".to_string(), 1.0);
        node2.attrs.ints.insert("transB".to_string(), 1);

        let nodes = vec![node1, node2];
        let result = eliminate_common_subexpressions(nodes);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "gemm1");
    }
}
