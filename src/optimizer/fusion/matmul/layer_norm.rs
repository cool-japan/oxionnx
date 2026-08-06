//! LayerNorm pattern fusion pass.
//!
//! Matches: ReduceMean → Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div
//! followed by Mul(scale) and optionally Add(bias).
//! Replaces with a single LayerNorm node.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Resolve a `ReduceMean` node's reduction axis for the LayerNorm pattern.
///
/// Returns the axis in ONNX `LayerNormalization` form — a **negative** index
/// `-k` meaning "normalise over the last `k` dimensions" — or `None` when the
/// node cannot be part of a LayerNorm:
///
/// * `keepdims` must be 1 (a squeezed mean does not broadcast back over `X`);
/// * the axes must be present, either as the pre-opset-18 `axes` **attribute**
///   or as the opset-18 `axes` **input** (resolved from the initializer map) —
///   an absent axes list means "reduce everything", which is not this pattern;
/// * the axes must form a contiguous trailing run (`[-1]`, `[-2, -1]`, …).
///   Non-negative axes need the input rank to be known, otherwise their
///   position relative to the end cannot be established.
fn trailing_reduce_axis(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    known_shapes: &HashMap<String, Vec<usize>>,
) -> Option<i64> {
    if node.attrs.i("keepdims", 1) != 1 {
        return None;
    }

    let attr_axes = node.attrs.ints("axes");
    let raw: Vec<i64> = if !attr_axes.is_empty() {
        attr_axes.to_vec()
    } else {
        // opset 18+: `axes` moved from an attribute to input slot 1.
        let axes_name = node.inputs.get(1)?;
        if axes_name.is_empty() {
            return None;
        }
        let axes_tensor = weights.get(axes_name)?;
        if axes_tensor.data.is_empty() {
            return None;
        }
        axes_tensor.data.iter().map(|&v| v as i64).collect()
    };

    // Rank of the reduced tensor, needed only to place non-negative axes.
    let rank: Option<i64> = node
        .inputs
        .first()
        .and_then(|x_name| {
            known_shapes
                .get(x_name)
                .map(|s| s.len())
                .or_else(|| weights.get(x_name).map(|t| t.ndim()))
        })
        .and_then(|r| i64::try_from(r).ok());

    let mut from_end: Vec<i64> = Vec::with_capacity(raw.len());
    for axis in raw {
        let normalized = if axis < 0 {
            if let Some(r) = rank {
                if axis < -r {
                    return None;
                }
            }
            axis
        } else {
            let r = rank?;
            if axis >= r {
                return None;
            }
            axis - r
        };
        from_end.push(normalized);
    }

    from_end.sort_unstable();
    from_end.dedup();
    let k = i64::try_from(from_end.len()).ok()?;
    // Must be exactly the last `k` axes: [-k, -k+1, …, -1].
    for (offset, &axis) in from_end.iter().enumerate() {
        let expected = -k + i64::try_from(offset).ok()?;
        if axis != expected {
            return None;
        }
    }
    Some(-k)
}

/// LayerNorm fusion: match the canonical pattern
///   `ReduceMean → Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div`
/// followed by `Mul(scale)` and optionally `Add(bias)`, and replace it with a
/// single `LayerNormalization` node.
///
/// A scale operand is mandatory: `LayerNormOp` requires input 1, so a pattern
/// without the trailing `Mul(scale)` is left untouched rather than fused into a
/// node that would fail at run time.
///
/// Every tensor the fusion removes must have exactly the consumers the pattern
/// accounts for and must not be a declared graph output.
pub fn fuse_layer_norm(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
    known_shapes: &HashMap<String, Vec<usize>>,
    output_names: &[String],
) -> Vec<Node> {
    if nodes.len() < 7 {
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

    let get_producer = |name: &str| -> Option<usize> { producer.get(name).copied() };

    // Sole output of a node in the chain, when it can be fused away: exactly
    // one consumer and not a graph output.
    let fusable_sole_output = |node: &Node| -> bool {
        match node.outputs.first() {
            Some(name) => usage.is_fusable_intermediate(name),
            None => false,
        }
    };

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }

        if !matches!(node.op, OpKind::Div) {
            continue;
        }
        if node.inputs.len() < 2 || node.outputs.is_empty() {
            continue;
        }

        let div_input0 = &node.inputs[0];
        let div_input1 = &node.inputs[1];

        // Step 7: div_input1 should come from Sqrt
        let sqrt_idx = match get_producer(div_input1) {
            Some(idx) if !skip.contains(&idx) => idx,
            _ => continue,
        };
        if !matches!(nodes[sqrt_idx].op, OpKind::Sqrt) {
            continue;
        }
        if !fusable_sole_output(&nodes[sqrt_idx]) {
            continue;
        }

        // Step 6: Sqrt input should come from Add(var, eps)
        if nodes[sqrt_idx].inputs.is_empty() {
            continue;
        }
        let add_eps_idx = match get_producer(&nodes[sqrt_idx].inputs[0]) {
            Some(idx) if !skip.contains(&idx) => idx,
            _ => continue,
        };
        if !matches!(nodes[add_eps_idx].op, OpKind::Add) {
            continue;
        }
        if !fusable_sole_output(&nodes[add_eps_idx]) {
            continue;
        }

        // Step 5: Add(var, eps) - one input should be a small constant (epsilon)
        if nodes[add_eps_idx].inputs.len() < 2 {
            continue;
        }
        let (var_tensor, epsilon) = {
            let inp0 = &nodes[add_eps_idx].inputs[0];
            let inp1 = &nodes[add_eps_idx].inputs[1];
            let is_eps = |t: &Tensor| -> bool {
                t.numel() == 1 && t.data.first().is_some_and(|&v| (0.0..0.01).contains(&v))
            };
            let eps1 = weights.get(inp1).filter(|t| is_eps(t));
            let eps0 = weights.get(inp0).filter(|t| is_eps(t));
            match (eps1, eps0) {
                (Some(t), _) => (inp0.clone(), t.data.first().copied().unwrap_or(1e-5)),
                (None, Some(t)) => (inp1.clone(), t.data.first().copied().unwrap_or(1e-5)),
                (None, None) => continue,
            }
        };

        // Step 4: var should come from ReduceMean(sq, axes)
        let var_reduce_idx = match get_producer(&var_tensor) {
            Some(idx) if !skip.contains(&idx) => idx,
            _ => continue,
        };
        if !matches!(nodes[var_reduce_idx].op, OpKind::ReduceMean) {
            continue;
        }
        if !fusable_sole_output(&nodes[var_reduce_idx]) {
            continue;
        }

        // Step 3: sq should come from Pow(diff, 2)
        if nodes[var_reduce_idx].inputs.is_empty() {
            continue;
        }
        let pow_idx = match get_producer(&nodes[var_reduce_idx].inputs[0]) {
            Some(idx) if !skip.contains(&idx) => idx,
            _ => continue,
        };
        if !matches!(nodes[pow_idx].op, OpKind::Pow) {
            continue;
        }
        if !fusable_sole_output(&nodes[pow_idx]) {
            continue;
        }
        if nodes[pow_idx].inputs.len() < 2 {
            continue;
        }
        let pow_exp_name = &nodes[pow_idx].inputs[1];
        let is_pow2 = match weights.get(pow_exp_name) {
            Some(exp_t) => {
                exp_t.numel() == 1 && exp_t.data.first().is_some_and(|&v| (v - 2.0).abs() < 1e-6)
            }
            None => false,
        };
        if !is_pow2 {
            continue;
        }

        // Step 2: Pow input[0] should come from Sub(X, mean) = diff
        let pow_diff_name = &nodes[pow_idx].inputs[0];
        if pow_diff_name != div_input0 {
            continue;
        }
        // `diff` feeds exactly the Pow and the Div, and nothing else: any third
        // consumer (or an export of `diff`) would lose its producer.
        if usage.consumers(pow_diff_name) != 2 || usage.is_graph_output(pow_diff_name) {
            continue;
        }
        let sub_idx = match get_producer(pow_diff_name) {
            Some(idx) if !skip.contains(&idx) => idx,
            _ => continue,
        };
        if !matches!(nodes[sub_idx].op, OpKind::Sub) {
            continue;
        }
        if nodes[sub_idx].inputs.len() < 2 {
            continue;
        }

        // Step 1: Sub input[1] should come from ReduceMean(X, axes) = mean
        let x_name = &nodes[sub_idx].inputs[0];
        let mean_name = &nodes[sub_idx].inputs[1];
        let mean_reduce_idx = match get_producer(mean_name) {
            Some(idx) if !skip.contains(&idx) => idx,
            _ => continue,
        };
        if !matches!(nodes[mean_reduce_idx].op, OpKind::ReduceMean) {
            continue;
        }
        if !fusable_sole_output(&nodes[mean_reduce_idx]) {
            continue;
        }

        if nodes[mean_reduce_idx].inputs.is_empty() {
            continue;
        }
        if &nodes[mean_reduce_idx].inputs[0] != x_name {
            continue;
        }

        // Both reductions must normalise over the same contiguous trailing axes.
        let axis = match trailing_reduce_axis(&nodes[mean_reduce_idx], weights, known_shapes) {
            Some(a) => a,
            None => continue,
        };
        let var_axis = match trailing_reduce_axis(&nodes[var_reduce_idx], weights, known_shapes) {
            Some(a) => a,
            None => continue,
        };
        if axis != var_axis {
            continue;
        }

        // Mandatory Mul(scale), optional Add(bias) after the Div.
        let div_out = match node.outputs.first() {
            Some(name) => name.clone(),
            None => continue,
        };
        let mut final_output = div_out.clone();
        let mut scale_name: Option<String> = None;
        let mut bias_name: Option<String> = None;
        let mut extra_skip = Vec::new();

        if usage.is_fusable_intermediate(&div_out) {
            for (j, next_node) in nodes.iter().enumerate() {
                if skip.contains(&j) || j == i {
                    continue;
                }
                if !matches!(next_node.op, OpKind::Mul) {
                    continue;
                }
                if next_node.inputs.len() < 2 || next_node.outputs.is_empty() {
                    continue;
                }
                let s_name = if next_node.inputs[0] == div_out
                    && weights.contains_key(&next_node.inputs[1])
                {
                    next_node.inputs[1].clone()
                } else if next_node.inputs[1] == div_out
                    && weights.contains_key(&next_node.inputs[0])
                {
                    next_node.inputs[0].clone()
                } else {
                    continue;
                };

                let mul_out = match next_node.outputs.first() {
                    Some(name) => name.clone(),
                    None => continue,
                };
                scale_name = Some(s_name);
                final_output = mul_out.clone();
                extra_skip.push(j);

                if usage.is_fusable_intermediate(&mul_out) {
                    for (k, add_node) in nodes.iter().enumerate() {
                        if skip.contains(&k) || k == j || k == i {
                            continue;
                        }
                        if !matches!(add_node.op, OpKind::Add) {
                            continue;
                        }
                        if add_node.inputs.len() < 2 || add_node.outputs.is_empty() {
                            continue;
                        }
                        let b_name = if add_node.inputs[0] == mul_out
                            && weights.contains_key(&add_node.inputs[1])
                        {
                            add_node.inputs[1].clone()
                        } else if add_node.inputs[1] == mul_out
                            && weights.contains_key(&add_node.inputs[0])
                        {
                            add_node.inputs[0].clone()
                        } else {
                            continue;
                        };
                        bias_name = Some(b_name);
                        if let Some(name) = add_node.outputs.first() {
                            final_output = name.clone();
                        }
                        extra_skip.push(k);
                        break;
                    }
                }
                break;
            }
        }

        // `LayerNormalization` requires a scale operand; without one there is
        // nothing valid to emit.
        let scale = match scale_name {
            Some(s) => s,
            None => continue,
        };

        let mut inputs = vec![x_name.clone(), scale];
        if let Some(b) = bias_name {
            inputs.push(b);
        }

        let mut attrs = Attributes::default();
        attrs.floats.insert("epsilon".to_string(), epsilon);
        attrs.ints.insert("axis".to_string(), axis);

        let fused = Node {
            op: OpKind::LayerNorm,
            name: format!("{}_fused_layernorm", nodes[mean_reduce_idx].name),
            inputs,
            outputs: vec![final_output],
            attrs,
        };

        skip.insert(sub_idx);
        skip.insert(pow_idx);
        skip.insert(var_reduce_idx);
        skip.insert(add_eps_idx);
        skip.insert(sqrt_idx);
        skip.insert(i);
        for idx in &extra_skip {
            skip.insert(*idx);
        }

        replacements.insert(mean_reduce_idx, fused);
    }

    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}
