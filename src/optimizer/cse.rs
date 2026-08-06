//! Common Subexpression Elimination (CSE) optimization pass.

use crate::graph::{Attributes, Graph, Node};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Ops whose operand order does not change the result, and whose inputs may
/// therefore be canonicalised by sorting.  Everything else keeps its positional
/// order: `Sub(a, b)` and `Sub(b, a)` are different expressions, and so are
/// `MatMul(q, k)` / `MatMul(k, q)`, `Gather(data, idx)` / `Gather(idx, data)`,
/// `Div`, `Pow`, `Reshape`, `Concat`, …
const COMMUTATIVE_OPS: &[&str] = &[
    "Add",
    "Mul",
    "And",
    "Or",
    "Xor",
    "Min",
    "Max",
    "Mean",
    "Sum",
    "Equal",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
];

/// Ops that draw on a random source (or otherwise differ between two
/// evaluations with identical inputs).  Two such nodes are never the same
/// expression, no matter how identical their inputs and attributes look.
const NON_DETERMINISTIC_OPS: &[&str] = &[
    "Dropout",
    "Bernoulli",
    "Multinomial",
    "RandomNormal",
    "RandomUniform",
    "RandomNormalLike",
    "RandomUniformLike",
];

/// Order-sensitive 64-bit hash (FNV-1a) of a constant tensor.
///
/// A commutative accumulation would make `[1.0, 2.0]` and `[2.0, 1.0]` collide
/// and merge two different constants, so shape and element order both feed the
/// hash.
fn tensor_hash(tensor: &Tensor) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    mix(tensor.shape.len() as u64);
    for &dim in &tensor.shape {
        mix(dim as u64);
    }
    mix(tensor.data.len() as u64);
    for &value in &tensor.data {
        // Normalise the two zero encodings so `-0.0` and `0.0` hash alike; NaN
        // payloads stay distinct, which is the conservative direction.
        let bits = if value == 0.0 { 0f32 } else { value }.to_bits();
        mix(u64::from(bits));
    }
    hash
}

/// Structural fingerprint of a subgraph attribute (`If` branches, `Loop` /
/// `Scan` bodies).  Two nodes carrying different bodies must never be merged.
fn graph_fingerprint(graph: &Graph) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(graph.nodes.len() + 2);
    parts.push(format!("in[{}]", graph.input_names.join(",")));
    parts.push(format!("out[{}]", graph.output_names.join(",")));
    for node in &graph.nodes {
        parts.push(format!(
            "{}({})->({}){{{}}}",
            node.op.as_str(),
            node.inputs.join(","),
            node.outputs.join(","),
            attrs_fingerprint(&node.attrs)
        ));
    }
    parts.join(";")
}

/// Canonical, fully-covering rendering of a node's attributes.
///
/// Every attribute kind carried by [`Attributes`] contributes — including
/// string lists and subgraphs, whose omission would make two `If` nodes with
/// different branches look identical.
fn attrs_fingerprint(attrs: &Attributes) -> String {
    let mut parts: Vec<String> = Vec::new();

    let mut float_keys: Vec<&String> = attrs.floats.keys().collect();
    float_keys.sort();
    for k in float_keys {
        if let Some(v) = attrs.floats.get(k) {
            parts.push(format!("f:{k}={}", v.to_bits()));
        }
    }

    let mut int_keys: Vec<&String> = attrs.ints.keys().collect();
    int_keys.sort();
    for k in int_keys {
        if let Some(v) = attrs.ints.get(k) {
            parts.push(format!("i:{k}={v}"));
        }
    }

    let mut str_keys: Vec<&String> = attrs.strings.keys().collect();
    str_keys.sort();
    for k in str_keys {
        if let Some(v) = attrs.strings.get(k) {
            parts.push(format!("s:{k}={v}"));
        }
    }

    let mut il_keys: Vec<&String> = attrs.int_lists.keys().collect();
    il_keys.sort();
    for k in il_keys {
        if let Some(v) = attrs.int_lists.get(k) {
            parts.push(format!("il:{k}={v:?}"));
        }
    }

    let mut fl_keys: Vec<&String> = attrs.float_lists.keys().collect();
    fl_keys.sort();
    for k in fl_keys {
        if let Some(v) = attrs.float_lists.get(k) {
            let bits: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
            parts.push(format!("fl:{k}={bits:?}"));
        }
    }

    let mut sl_keys: Vec<&String> = attrs.string_lists.keys().collect();
    sl_keys.sort();
    for k in sl_keys {
        if let Some(v) = attrs.string_lists.get(k) {
            parts.push(format!("sl:{k}={v:?}"));
        }
    }

    let mut t_keys: Vec<&String> = attrs.tensors.keys().collect();
    t_keys.sort();
    for k in t_keys {
        if let Some(t) = attrs.tensors.get(k) {
            parts.push(format!("t:{k}=shape{:?}hash{}", t.shape, tensor_hash(t)));
        }
    }

    let mut g_keys: Vec<&String> = attrs.graphs.keys().collect();
    g_keys.sort();
    for k in g_keys {
        if let Some(g) = attrs.graphs.get(k) {
            parts.push(format!("g:{k}=[{}]", graph_fingerprint(g)));
        }
    }

    parts.join(";")
}

/// Compute a deterministic fingerprint for a node: op type, operands (sorted
/// only for commutative ops), output arity and every attribute.
fn node_fingerprint(node: &Node) -> String {
    let op_str = node.op.as_str();

    let inputs_str = if COMMUTATIVE_OPS.contains(&op_str) {
        let mut sorted_inputs = node.inputs.clone();
        sorted_inputs.sort();
        sorted_inputs.join(",")
    } else {
        node.inputs.join(",")
    };

    let attrs_str = attrs_fingerprint(&node.attrs);
    let arity = node.outputs.len();
    format!("{op_str}|{inputs_str}|{arity}|{attrs_str}")
}

/// Eliminate common subexpressions from the node list.
///
/// Two nodes are duplicates when they have the same op type, the same operands
/// (order-independent only for commutative ops), the same output arity and
/// identical attributes — including subgraph bodies.  The first occurrence is
/// kept; later duplicates are removed and references to their outputs are
/// redirected to the original's outputs.
///
/// A node is never removed when it is non-deterministic, or when any of its
/// outputs is a declared graph output (the name would stop being produced).
pub fn eliminate_common_subexpressions(nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
    // fingerprint -> output names of the first node with that fingerprint
    let mut seen: HashMap<String, Vec<String>> = HashMap::new();
    // redirect map: duplicate output name -> original output name
    let mut redirects: HashMap<String, String> = HashMap::new();
    // Indices of duplicate nodes to remove
    let mut duplicate_indices: Vec<bool> = vec![false; nodes.len()];

    let usage = TensorUsage::new(&nodes, output_names);

    for (idx, node) in nodes.iter().enumerate() {
        if NON_DETERMINISTIC_OPS.contains(&node.op.as_str()) {
            continue;
        }
        let fp = node_fingerprint(node);

        if let Some(original_outputs) = seen.get(&fp) {
            // Removing this node would drop a declared graph output.
            if !usage.none_is_graph_output(&node.outputs) {
                continue;
            }
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

    fn no_outputs() -> Vec<String> {
        Vec::new()
    }

    #[test]
    fn test_cse_removes_duplicate_nodes() {
        // Two identical Add nodes with same inputs
        let nodes = vec![
            make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]),
            make_node(OpKind::Add, "add2", vec!["x", "y"], vec!["out2"]),
        ];
        let result = eliminate_common_subexpressions(nodes, &no_outputs());
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
        let result = eliminate_common_subexpressions(nodes, &no_outputs());
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
        let result = eliminate_common_subexpressions(nodes, &["relu_out".to_string()]);
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
        let result = eliminate_common_subexpressions(nodes, &no_outputs());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_different_ops_not_removed() {
        // Different ops with same inputs -> both kept
        let nodes = vec![
            make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]),
            make_node(OpKind::Mul, "mul1", vec!["x", "y"], vec!["out2"]),
        ];
        let result = eliminate_common_subexpressions(nodes, &no_outputs());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_preserves_different_float_attrs() {
        let mut node1 = make_node(OpKind::Add, "add1", vec!["x", "y"], vec!["out1"]);
        node1.attrs.floats.insert("alpha".to_string(), 1.0);

        let mut node2 = make_node(OpKind::Add, "add2", vec!["x", "y"], vec!["out2"]);
        node2.attrs.floats.insert("alpha".to_string(), 2.0);

        let nodes = vec![node1, node2];
        let result = eliminate_common_subexpressions(nodes, &no_outputs());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_empty_graph() {
        let nodes: Vec<Node> = vec![];
        let result = eliminate_common_subexpressions(nodes, &no_outputs());
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
        let result = eliminate_common_subexpressions(nodes, &no_outputs());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "gemm1");
    }

    #[test]
    fn test_cse_commutative_add_still_merges_swapped_operands() {
        let nodes = vec![
            make_node(OpKind::Add, "add1", vec!["a", "b"], vec!["out1"]),
            make_node(OpKind::Add, "add2", vec!["b", "a"], vec!["out2"]),
        ];
        let result = eliminate_common_subexpressions(nodes, &no_outputs());
        assert_eq!(result.len(), 1);
    }
}
