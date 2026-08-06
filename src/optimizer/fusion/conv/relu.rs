//! Conv + Relu/Clip activation fusion pass.

use crate::graph::{Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Compile-time bounds of a `Clip` node, resolved from either the opset-11+
/// `min`/`max` **inputs** or the opset-6..10 `min`/`max` **attributes**.
pub(super) struct ClipBounds {
    pub(super) min: f32,
    pub(super) max: f32,
}

/// Resolve one Clip bound.
///
/// Since ONNX opset 11 the bound lives in input slot `slot` (an initializer for
/// a static bound); before that it was the attribute `attr_name`.  An absent /
/// empty input slot means "unbounded on this side".
///
/// Returns `None` when the input slot is present but is *not* a compile-time
/// constant — the bound is only known at runtime and the Clip must not be
/// folded into the Conv.
fn resolve_bound(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    slot: usize,
    attr_name: &str,
    unbounded: f32,
) -> Option<f32> {
    match node.inputs.get(slot) {
        Some(name) if !name.is_empty() => {
            // opset 11+ form: the bound is an input tensor.
            crate::optimizer::graph_utils::const_scalar(weights, name)
        }
        // No input slot (or an explicitly omitted one): fall back to the
        // pre-opset-11 attribute, else the op is unbounded on this side.
        _ => Some(node.attrs.f(attr_name, unbounded)),
    }
}

/// Resolve both Clip bounds, rejecting anything that is dynamic or that would
/// make `f32::clamp` panic (`NaN` bounds, or `min > max`).
pub(super) fn resolve_clip_bounds(
    node: &Node,
    weights: &HashMap<String, Tensor>,
) -> Option<ClipBounds> {
    let min = resolve_bound(node, weights, 1, "min", f32::NEG_INFINITY)?;
    let max = resolve_bound(node, weights, 2, "max", f32::INFINITY)?;
    if min.is_nan() || max.is_nan() || min > max {
        return None;
    }
    Some(ClipBounds { min, max })
}

/// Conv + Relu/Clip fusion.
///
/// Pattern: `Conv → Relu`, or `Conv → Clip(min, max)` with compile-time bounds.
/// The activation is merged into the Conv node as the `activation` attribute
/// (`"relu"`, or `"clip"` plus `activation_min` / `activation_max`), which is
/// exactly the contract `ConvOp` implements.
///
/// Since opset 11 a Clip's bounds are **inputs**, not attributes; they are
/// resolved from the initializer map.  A Clip whose bounds are only known at
/// runtime is never fused (folding it would silently drop the clamp).
///
/// The Conv output must have exactly one consumer *and* must not be a declared
/// graph output — the fused node produces only the activation's output name.
pub fn fuse_conv_relu(
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

        let is_relu = matches!(node.op, OpKind::Relu);
        let is_clip = matches!(node.op, OpKind::Clip);
        if !is_relu && !is_clip {
            continue;
        }

        if node.inputs.is_empty() || node.outputs.is_empty() {
            continue;
        }

        // Resolve the clamp range before touching the graph: a Clip with
        // runtime bounds cannot be folded into the Conv at all.
        let bounds = if is_clip {
            match resolve_clip_bounds(node, weights) {
                Some(b) => Some(b),
                None => continue,
            }
        } else {
            None
        };

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

        match bounds {
            None => {
                fused_attrs
                    .strings
                    .insert("activation".to_string(), "relu".to_string());
            }
            Some(ClipBounds { min, max }) => {
                if min == 0.0 && max == f32::INFINITY {
                    fused_attrs
                        .strings
                        .insert("activation".to_string(), "relu".to_string());
                } else {
                    fused_attrs
                        .strings
                        .insert("activation".to_string(), "clip".to_string());
                    fused_attrs.floats.insert("activation_min".to_string(), min);
                    fused_attrs.floats.insert("activation_max".to_string(), max);
                }
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
