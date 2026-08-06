//! Conv + Add + ReLU fusion pass (ResNet residual block pattern).

use crate::graph::{Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use std::collections::{HashMap, HashSet};

/// Conv + Add + ReLU fusion (ResNet residual block pattern).
///
/// Pattern: `Conv(X, W, B) → Add(conv_out, residual) → Relu`
///
/// This is the core building block in ResNet architectures.  The three-op
/// sequence can be fused into a single `ConvAddRelu` node that computes
/// `relu(conv(X, W, B) + residual)` in one pass, eliminating two intermediate
/// tensors and enabling execution engines to use a single fused kernel.
///
/// The fused node has inputs `[X, W, B, residual]`.  If the original Conv has
/// no bias, an empty string is used for the B slot.
///
/// Conditions:
/// - Conv output has exactly one consumer (the Add node) and is not a graph output.
/// - Add output has exactly one consumer (the Relu node) and is not a graph output.
/// - One of Add's inputs comes from Conv; the other is the residual / skip connection.
///
/// The fused `ConvAddRelu` op has no kernel in the default registry, so
/// [`crate::optimizer::optimize`] only runs this pass when the active
/// `OperatorRegistry` provides one.
pub fn fuse_conv_add_relu(nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
    if nodes.len() < 3 {
        return nodes;
    }

    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.clone(), i);
        }
    }

    let usage = TensorUsage::new(&nodes, output_names);

    let mut skip: HashSet<usize> = HashSet::new();
    let mut replacements: HashMap<usize, Node> = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        if !matches!(node.op, OpKind::Relu) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }

        let relu_input = &node.inputs[0];

        // Relu input must have exactly one consumer (this Relu) and must not
        // be a graph output — the fused node no longer produces it.
        if !usage.is_fusable_intermediate(relu_input) {
            continue;
        }

        // Find the Add node producing the Relu's input
        let add_idx = match producer.get(relu_input) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&add_idx) {
            continue;
        }
        if !matches!(nodes[add_idx].op, OpKind::Add) {
            continue;
        }
        if nodes[add_idx].inputs.len() < 2 {
            continue;
        }

        // Identify which Add input comes from Conv and which is the residual.
        // Try both orderings: Add(conv_out, residual) and Add(residual, conv_out).
        let add_inp0 = &nodes[add_idx].inputs[0];
        let add_inp1 = &nodes[add_idx].inputs[1];

        let (conv_idx, residual_name) = {
            let try_find_conv =
                |conv_candidate: &str, residual_candidate: &str| -> Option<(usize, String)> {
                    // Conv output must have exactly one consumer (this Add)
                    // and must not be a declared graph output.
                    if !usage.is_fusable_intermediate(conv_candidate) {
                        return None;
                    }
                    let idx = producer.get(conv_candidate)?;
                    if skip.contains(idx) {
                        return None;
                    }
                    if !matches!(nodes[*idx].op, OpKind::Conv) {
                        return None;
                    }
                    // An already-fused activation would have to run between the
                    // Conv and the Add, which `ConvAddRelu` cannot express.
                    if !nodes[*idx].attrs.s("activation").is_empty() {
                        return None;
                    }
                    Some((*idx, residual_candidate.to_string()))
                };

            match try_find_conv(add_inp0, add_inp1).or_else(|| try_find_conv(add_inp1, add_inp0)) {
                Some(result) => result,
                None => continue,
            }
        };

        let conv_node = &nodes[conv_idx];

        // Build fused ConvAddRelu node
        // Inputs: [X, W, B (or ""), residual]
        let mut fused_inputs = Vec::with_capacity(4);
        // X (data input)
        if conv_node.inputs.is_empty() {
            continue;
        }
        fused_inputs.push(conv_node.inputs[0].clone());
        // W (weight)
        if conv_node.inputs.len() < 2 {
            continue;
        }
        fused_inputs.push(conv_node.inputs[1].clone());
        // B (bias; may be absent)
        fused_inputs.push(conv_node.inputs.get(2).cloned().unwrap_or_default());
        // residual (skip connection)
        fused_inputs.push(residual_name);

        let mut fused_attrs = conv_node.attrs.clone();
        fused_attrs
            .strings
            .insert("fused_ops".to_string(), "Add,Relu".to_string());

        let fused = Node {
            op: OpKind::ConvAddRelu,
            name: format!("{}_fused_conv_add_relu", conv_node.name),
            inputs: fused_inputs,
            outputs: node.outputs.clone(),
            attrs: fused_attrs,
        };

        replacements.insert(conv_idx, fused);
        skip.insert(add_idx);
        skip.insert(i);
    }

    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}
