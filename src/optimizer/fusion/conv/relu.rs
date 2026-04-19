//! Conv + Relu/Clip activation fusion pass.

use crate::graph::{Node, OpKind};
use std::collections::{HashMap, HashSet};

/// Conv + Relu/Clip fusion.
/// Pattern: Conv node -> Relu node (or Clip with min=0, max=inf)
/// Merges activation into the Conv node as an attribute.
pub fn fuse_conv_relu(nodes: Vec<Node>) -> Vec<Node> {
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

        let is_relu = matches!(node.op, OpKind::Relu);
        let is_clip = matches!(node.op, OpKind::Clip);
        if !is_relu && !is_clip {
            continue;
        }

        if node.inputs.is_empty() {
            continue;
        }

        if is_clip {
            let min_val = node.attrs.f("min", f32::NEG_INFINITY);
            if min_val != 0.0 && min_val != f32::NEG_INFINITY {
                continue;
            }
        }

        let conv_tensor = &node.inputs[0];

        if consumer_count.get(conv_tensor).copied().unwrap_or(0) != 1 {
            continue;
        }

        let conv_idx = match producer.get(conv_tensor) {
            Some(&idx) => idx,
            None => continue,
        };

        if !matches!(nodes[conv_idx].op, OpKind::Conv) {
            continue;
        }

        let mut fused_attrs = nodes[conv_idx].attrs.clone();

        if is_relu {
            fused_attrs
                .strings
                .insert("activation".to_string(), "relu".to_string());
        } else {
            let min_val = node.attrs.f("min", f32::NEG_INFINITY);
            let max_val = node.attrs.f("max", f32::INFINITY);
            if min_val == 0.0 && max_val == f32::INFINITY {
                fused_attrs
                    .strings
                    .insert("activation".to_string(), "relu".to_string());
            } else {
                fused_attrs
                    .strings
                    .insert("activation".to_string(), "clip".to_string());
                fused_attrs
                    .floats
                    .insert("activation_min".to_string(), min_val);
                fused_attrs
                    .floats
                    .insert("activation_max".to_string(), max_val);
            }
        }

        let fused = Node {
            op: OpKind::Conv,
            name: format!("{}_fused_activation", nodes[conv_idx].name),
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
