//! LayerNorm pattern fusion pass.
//!
//! Matches: ReduceMean → Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div
//! Optionally followed by Mul(scale) → Add(bias).
//! Replaces with a single LayerNorm node.

use crate::graph::{Attributes, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// LayerNorm fusion: match the canonical pattern of 7+ nodes:
///   ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div
/// Optionally followed by Mul(scale) -> Add(bias).
/// Replace with a single LayerNorm node.
pub fn fuse_layer_norm(nodes: Vec<Node>, weights: &HashMap<String, Tensor>) -> Vec<Node> {
    if nodes.len() < 7 {
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

    let single_consumer =
        |name: &str| -> bool { consumer_count.get(name).copied().unwrap_or(0) == 1 };

    let get_producer = |name: &str| -> Option<usize> { producer.get(name).copied() };

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }

        if !matches!(node.op, OpKind::Div) {
            continue;
        }
        if node.inputs.len() < 2 {
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
        if !single_consumer(&nodes[sqrt_idx].outputs[0]) {
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
        if !single_consumer(&nodes[add_eps_idx].outputs[0]) {
            continue;
        }

        // Step 5: Add(var, eps) - one input should be a small constant (epsilon)
        if nodes[add_eps_idx].inputs.len() < 2 {
            continue;
        }
        let (var_tensor, epsilon) = {
            let inp0 = &nodes[add_eps_idx].inputs[0];
            let inp1 = &nodes[add_eps_idx].inputs[1];
            if let Some(eps_t) = weights.get(inp1) {
                if eps_t.numel() == 1 && eps_t.data[0] < 0.01 {
                    (inp0.clone(), eps_t.data[0])
                } else if let Some(eps_t2) = weights.get(inp0) {
                    if eps_t2.numel() == 1 && eps_t2.data[0] < 0.01 {
                        (inp1.clone(), eps_t2.data[0])
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else if let Some(eps_t) = weights.get(inp0) {
                if eps_t.numel() == 1 && eps_t.data[0] < 0.01 {
                    (inp1.clone(), eps_t.data[0])
                } else {
                    continue;
                }
            } else {
                continue;
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
        if !single_consumer(&nodes[var_reduce_idx].outputs[0]) {
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
        if !single_consumer(&nodes[pow_idx].outputs[0]) {
            continue;
        }
        if nodes[pow_idx].inputs.len() < 2 {
            continue;
        }
        let pow_exp_name = &nodes[pow_idx].inputs[1];
        let is_pow2 = if let Some(exp_t) = weights.get(pow_exp_name) {
            exp_t.numel() == 1 && (exp_t.data[0] - 2.0).abs() < 1e-6
        } else {
            false
        };
        if !is_pow2 {
            continue;
        }

        // Step 2: Pow input[0] should come from Sub(X, mean) = diff
        let pow_diff_name = &nodes[pow_idx].inputs[0];
        if pow_diff_name != div_input0 {
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
        if !single_consumer(&nodes[mean_reduce_idx].outputs[0]) {
            continue;
        }

        if nodes[mean_reduce_idx].inputs.is_empty() {
            continue;
        }
        if &nodes[mean_reduce_idx].inputs[0] != x_name {
            continue;
        }

        let axes = nodes[mean_reduce_idx].attrs.ints("axes");
        let axis = if axes.is_empty() { -1i64 } else { axes[0] };

        let var_axes = nodes[var_reduce_idx].attrs.ints("axes");
        if !var_axes.is_empty() && !axes.is_empty() && var_axes != axes {
            continue;
        }

        // Now check for optional Mul(scale) and Add(bias) after the Div
        let mut final_output = node.outputs[0].clone();
        let mut scale_name: Option<String> = None;
        let mut bias_name: Option<String> = None;
        let mut extra_skip = Vec::new();

        if single_consumer(&node.outputs[0]) {
            for (j, next_node) in nodes.iter().enumerate() {
                if skip.contains(&j) || j == i {
                    continue;
                }
                if !matches!(next_node.op, OpKind::Mul) {
                    continue;
                }
                if next_node.inputs.len() < 2 {
                    continue;
                }
                let (is_match, s_name) = if next_node.inputs[0] == node.outputs[0]
                    && weights.contains_key(&next_node.inputs[1])
                {
                    (true, next_node.inputs[1].clone())
                } else if next_node.inputs[1] == node.outputs[0]
                    && weights.contains_key(&next_node.inputs[0])
                {
                    (true, next_node.inputs[0].clone())
                } else {
                    (false, String::new())
                };
                if is_match {
                    scale_name = Some(s_name);
                    final_output = next_node.outputs[0].clone();
                    extra_skip.push(j);

                    if single_consumer(&next_node.outputs[0]) {
                        for (k, add_node) in nodes.iter().enumerate() {
                            if skip.contains(&k) || k == j || k == i {
                                continue;
                            }
                            if !matches!(add_node.op, OpKind::Add) {
                                continue;
                            }
                            if add_node.inputs.len() < 2 {
                                continue;
                            }
                            let (is_add_match, b_name) = if add_node.inputs[0]
                                == next_node.outputs[0]
                                && weights.contains_key(&add_node.inputs[1])
                            {
                                (true, add_node.inputs[1].clone())
                            } else if add_node.inputs[1] == next_node.outputs[0]
                                && weights.contains_key(&add_node.inputs[0])
                            {
                                (true, add_node.inputs[0].clone())
                            } else {
                                (false, String::new())
                            };
                            if is_add_match {
                                bias_name = Some(b_name);
                                final_output = add_node.outputs[0].clone();
                                extra_skip.push(k);
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }

        let mut inputs = vec![x_name.clone()];
        if let Some(ref s) = scale_name {
            inputs.push(s.clone());
        }
        if let Some(ref b) = bias_name {
            inputs.push(b.clone());
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
