//! Conv + Clip(0, 6) → Conv with ReLU6 activation fusion pass.

use crate::graph::{Node, OpKind};
use std::collections::{HashMap, HashSet};

/// Conv + Clip(min=0, max=6) → Conv with ReLU6 activation.
///
/// MobileNet and EfficientNet architectures use ReLU6 (Clip(0, 6)) extensively
/// after convolutions.  This pass specifically recognises the ReLU6 pattern and
/// marks the fused Conv with `activation = "relu6"`, allowing execution engines
/// to dispatch a dedicated fused kernel.
///
/// This is complementary to `fuse_conv_relu` which handles plain Relu and
/// general Clip ranges.  ReLU6 gets its own label because many hardware
/// accelerators have a dedicated ReLU6 instruction.
///
/// Conditions:
/// - Clip's min attribute == 0.0 and max attribute == 6.0.
/// - Clip's sole input comes from a Conv node with a single consumer.
pub fn fuse_conv_clip_to_conv_relu6(nodes: Vec<Node>) -> Vec<Node> {
    if nodes.len() < 2 {
        return nodes;
    }

    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.clone(), i);
        }
    }

    let mut consumer_count: HashMap<String, usize> = HashMap::new();
    for node in &nodes {
        for inp in &node.inputs {
            if !inp.is_empty() {
                *consumer_count.entry(inp.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut skip: HashSet<usize> = HashSet::new();
    let mut replacements: HashMap<usize, Node> = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        if !matches!(node.op, OpKind::Clip) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }

        // Must be exactly ReLU6: min=0, max=6
        let min_val = node.attrs.f("min", f32::NEG_INFINITY);
        let max_val = node.attrs.f("max", f32::INFINITY);
        if (min_val - 0.0).abs() > 1e-7 || (max_val - 6.0).abs() > 1e-7 {
            continue;
        }

        let conv_tensor = &node.inputs[0];

        if consumer_count.get(conv_tensor).copied().unwrap_or(0) != 1 {
            continue;
        }

        let conv_idx = match producer.get(conv_tensor) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&conv_idx) {
            continue;
        }
        if !matches!(nodes[conv_idx].op, OpKind::Conv) {
            continue;
        }

        let mut fused_attrs = nodes[conv_idx].attrs.clone();
        fused_attrs
            .strings
            .insert("activation".to_string(), "relu6".to_string());
        fused_attrs.floats.insert("activation_min".to_string(), 0.0);
        fused_attrs.floats.insert("activation_max".to_string(), 6.0);

        let fused = Node {
            op: OpKind::Conv,
            name: format!("{}_fused_relu6", nodes[conv_idx].name),
            inputs: nodes[conv_idx].inputs.clone(),
            outputs: node.outputs.clone(),
            attrs: fused_attrs,
        };

        replacements.insert(conv_idx, fused);
        skip.insert(i);
    }

    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}
