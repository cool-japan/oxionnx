//! Conv + Clip(0, 6) → Conv with a fused ReLU6 clamp.

use crate::graph::{Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

use super::relu::{resolve_clip_bounds, ClipBounds};

/// Conv + Clip(min=0, max=6) → Conv with a fused ReLU6 clamp.
///
/// MobileNet and EfficientNet architectures use ReLU6 (`Clip(0, 6)`) extensively
/// after convolutions.  This pass recognises exactly that range and folds it
/// into the Conv node.
///
/// The clamp is emitted as `activation = "clip"` with `activation_min = 0` and
/// `activation_max = 6`, which is the contract the Conv kernels implement (a
/// bare `"relu6"` label would be ignored by every kernel, i.e. the clamp would
/// be silently dropped).  Backends that have a dedicated ReLU6 instruction can
/// recognise the fused Conv by its `[0, 6]` range.
///
/// Since opset 11 the bounds are Clip **inputs** rather than attributes; they
/// are resolved from the initializer map, and a Clip with runtime bounds is
/// never fused.
///
/// Conditions:
/// - Clip's bounds are compile-time constants equal to `0` and `6`.
/// - Clip's data input comes from a Conv whose output has exactly one consumer
///   and is not a declared graph output.
pub fn fuse_conv_clip_to_conv_relu6(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
    output_names: &[String],
) -> Vec<Node> {
    if nodes.len() < 2 {
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
        if !matches!(node.op, OpKind::Clip) {
            continue;
        }
        if node.inputs.is_empty() || node.outputs.is_empty() {
            continue;
        }

        // Must be exactly ReLU6: min = 0, max = 6, both compile-time constants.
        let ClipBounds { min, max } = match resolve_clip_bounds(node, weights) {
            Some(b) => b,
            None => continue,
        };
        if (min - 0.0).abs() > 1e-7 || (max - 6.0).abs() > 1e-7 {
            continue;
        }

        let conv_tensor = &node.inputs[0];

        if !usage.is_fusable_intermediate(conv_tensor) {
            continue;
        }

        let conv_idx = match producer.get(conv_tensor) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&conv_idx) || replacements.contains_key(&conv_idx) {
            continue;
        }
        if !matches!(nodes[conv_idx].op, OpKind::Conv) {
            continue;
        }
        // Never stack a second activation onto an already-fused Conv.
        if !nodes[conv_idx].attrs.s("activation").is_empty() {
            continue;
        }

        let mut fused_attrs = nodes[conv_idx].attrs.clone();
        fused_attrs
            .strings
            .insert("activation".to_string(), "clip".to_string());
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
