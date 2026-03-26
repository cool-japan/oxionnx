//! Fusion optimization passes: MatMul+Add, Conv+BatchNorm, Conv+Relu,
//! LayerNorm pattern, and consecutive Transpose cancellation.

use crate::graph::{Attributes, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// MatMul + Add -> Gemm fusion
/// Pattern: node A = MatMul(X, W), node B = Add(A.output, bias) where bias is 1D in weights
/// Fused: Gemm(X, W, bias) with alpha=1, beta=1
pub fn fuse_matmul_add(nodes: Vec<Node>, weights: &HashMap<String, Tensor>) -> Vec<Node> {
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

        if !matches!(node.op, OpKind::Add) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }

        let matmul_tensor = &node.inputs[0];
        let bias_tensor = &node.inputs[1];

        if consumer_count.get(matmul_tensor).copied().unwrap_or(0) != 1 {
            continue;
        }

        let matmul_idx = match producer.get(matmul_tensor) {
            Some(&idx) => idx,
            None => continue,
        };

        if !matches!(nodes[matmul_idx].op, OpKind::MatMul) {
            continue;
        }

        if let Some(bias_t) = weights.get(bias_tensor) {
            if bias_t.ndim() != 1 {
                continue;
            }
        } else {
            continue;
        }

        let mut attrs = Attributes::default();
        attrs.floats.insert("alpha".to_string(), 1.0);
        attrs.floats.insert("beta".to_string(), 1.0);
        attrs.ints.insert("transA".to_string(), 0);
        attrs.ints.insert("transB".to_string(), 0);

        let fused = Node {
            op: OpKind::Gemm,
            name: format!("{}_fused_gemm", nodes[matmul_idx].name),
            inputs: vec![
                nodes[matmul_idx].inputs[0].clone(),
                nodes[matmul_idx].inputs[1].clone(),
                bias_tensor.clone(),
            ],
            outputs: node.outputs.clone(),
            attrs,
        };

        replacements.insert(matmul_idx, fused);
        skip.insert(i);
    }

    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}

/// Conv + BatchNorm fusion
/// Pattern: node A = Conv(X, W, B), node B = BatchNorm(A.output, scale, bias, mean, var)
/// Fused: Conv with modified weights and bias
pub fn fuse_conv_batchnorm(nodes: Vec<Node>, weights: &mut HashMap<String, Tensor>) -> Vec<Node> {
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
        if !matches!(node.op, OpKind::BatchNorm) {
            continue;
        }
        if node.inputs.len() < 5 {
            continue;
        }

        let conv_tensor = &node.inputs[0];
        let bn_scale_name = &node.inputs[1];
        let bn_bias_name = &node.inputs[2];
        let bn_mean_name = &node.inputs[3];
        let bn_var_name = &node.inputs[4];

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

        let bn_scale = match weights.get(bn_scale_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let bn_bias = match weights.get(bn_bias_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let bn_mean = match weights.get(bn_mean_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let bn_var = match weights.get(bn_var_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let epsilon = node.attrs.floats.get("epsilon").copied().unwrap_or(1e-5);

        let conv_node = &nodes[conv_idx];
        if conv_node.inputs.len() < 2 {
            continue;
        }
        let conv_weight_name = &conv_node.inputs[1];
        let conv_bias_name = conv_node.inputs.get(2).cloned();

        let conv_weight = match weights.get(conv_weight_name) {
            Some(t) => t.clone(),
            None => continue,
        };

        let c_out = bn_scale.data.len();
        if c_out == 0 || conv_weight.data.len() % c_out != 0 {
            continue;
        }
        let weight_per_channel: usize = conv_weight.data.len() / c_out;

        let mut fused_weight = conv_weight.data.clone();
        let mut fused_bias = vec![0.0f32; c_out];

        let conv_bias_data = if let Some(ref name) = conv_bias_name {
            if let Some(b) = weights.get(name) {
                b.data.clone()
            } else {
                vec![0.0f32; c_out]
            }
        } else {
            vec![0.0f32; c_out]
        };

        for c in 0..c_out {
            let inv_std = 1.0 / (bn_var.data[c] + epsilon).sqrt();
            let factor = bn_scale.data[c] * inv_std;

            let start = c * weight_per_channel;
            for w in &mut fused_weight[start..start + weight_per_channel] {
                *w *= factor;
            }

            fused_bias[c] = (conv_bias_data[c] - bn_mean.data[c]) * factor + bn_bias.data[c];
        }

        let fused_weight_name = format!("{}_fused_weight", conv_node.name);
        let fused_bias_name = format!("{}_fused_bias", conv_node.name);
        weights.insert(
            fused_weight_name.clone(),
            Tensor::new(fused_weight, conv_weight.shape.clone()),
        );
        weights.insert(
            fused_bias_name.clone(),
            Tensor::new(fused_bias, vec![c_out]),
        );

        let fused_inputs = vec![
            conv_node.inputs[0].clone(),
            fused_weight_name,
            fused_bias_name,
        ];
        let fused_conv = Node {
            op: OpKind::Conv,
            name: format!("{}_fused_convbn", conv_node.name),
            inputs: fused_inputs,
            outputs: node.outputs.clone(),
            attrs: conv_node.attrs.clone(),
        };

        replacements.insert(conv_idx, fused_conv);
        skip.insert(i);
    }

    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}

/// Consecutive Transpose cancellation
/// If Transpose(perm1) -> Transpose(perm2) composes to identity, remove both.
/// Otherwise, replace with single Transpose(composed_perm).
pub fn cancel_consecutive_transpose(nodes: Vec<Node>) -> Vec<Node> {
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
    let mut redirects: HashMap<String, String> = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        if !matches!(node.op, OpKind::Transpose) {
            continue;
        }

        let input_name = match node.inputs.first() {
            Some(name) if !name.is_empty() => name,
            _ => continue,
        };

        let prev_idx = match producer.get(input_name) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&prev_idx) {
            continue;
        }
        if !matches!(nodes[prev_idx].op, OpKind::Transpose) {
            continue;
        }
        if consumer_count.get(input_name).copied().unwrap_or(0) != 1 {
            continue;
        }

        let perm1 = match nodes[prev_idx].attrs.int_lists.get("perm") {
            Some(p) if !p.is_empty() => p.clone(),
            _ => continue,
        };
        let perm2 = match node.attrs.int_lists.get("perm") {
            Some(p) if !p.is_empty() => p.clone(),
            _ => continue,
        };

        if perm1.len() != perm2.len() {
            continue;
        }

        let len = perm1.len();
        let valid = perm2.iter().all(|&p| p >= 0 && (p as usize) < len);
        if !valid {
            continue;
        }

        let composed: Vec<i64> = perm2.iter().map(|&p| perm1[p as usize]).collect();

        let is_identity = composed.iter().enumerate().all(|(i, &v)| v == i as i64);

        if is_identity {
            skip.insert(prev_idx);
            skip.insert(i);
            if let Some(out_name) = node.outputs.first() {
                redirects.insert(out_name.clone(), nodes[prev_idx].inputs[0].clone());
            }
        } else {
            let mut attrs = Attributes::default();
            attrs.int_lists.insert("perm".to_string(), composed);
            let fused = Node {
                op: OpKind::Transpose,
                name: format!("{}_fused_transpose", nodes[prev_idx].name),
                inputs: nodes[prev_idx].inputs.clone(),
                outputs: node.outputs.clone(),
                attrs,
            };
            replacements.insert(prev_idx, fused);
            skip.insert(i);
        }
    }

    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, mut n)| {
            if let Some(replacement) = replacements.remove(&i) {
                replacement
            } else {
                for inp in &mut n.inputs {
                    if let Some(redirect) = redirects.get(inp) {
                        *inp = redirect.clone();
                    }
                }
                n
            }
        })
        .collect()
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::test_utils::{make_layer_norm_pattern, make_node};

    #[test]
    fn test_fuse_matmul_add() {
        let nodes = vec![
            make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
            make_node(OpKind::Add, "add", vec!["mm_out", "bias"], vec!["add_out"]),
        ];
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));
        weights.insert("bias".to_string(), Tensor::new(vec![0.5, 0.5], vec![2]));

        let result = fuse_matmul_add(nodes, &weights);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Gemm));
        assert_eq!(result[0].outputs[0], "add_out");
        assert_eq!(result[0].inputs.len(), 3);
        assert_eq!(result[0].inputs[0], "x");
        assert_eq!(result[0].inputs[1], "w");
        assert_eq!(result[0].inputs[2], "bias");
    }

    #[test]
    fn test_fuse_matmul_add_single_node() {
        let nodes = vec![make_node(
            OpKind::MatMul,
            "mm",
            vec!["x", "w"],
            vec!["mm_out"],
        )];
        let weights = HashMap::new();
        let result = fuse_matmul_add(nodes, &weights);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_fuse_matmul_add_bias_not_1d() {
        let nodes = vec![
            make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
            make_node(OpKind::Add, "add", vec!["mm_out", "bias"], vec!["add_out"]),
        ];
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));
        weights.insert("bias".to_string(), Tensor::new(vec![0.5; 4], vec![2, 2]));
        let result = fuse_matmul_add(nodes, &weights);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_no_fusion_when_multiple_consumers() {
        let nodes = vec![
            make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
            make_node(OpKind::Add, "add", vec!["mm_out", "bias"], vec!["add_out"]),
            make_node(OpKind::Relu, "relu", vec!["mm_out"], vec!["relu_out"]),
        ];
        let weights = {
            let mut w = HashMap::new();
            w.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));
            w.insert("bias".to_string(), Tensor::new(vec![0.5, 0.5], vec![2]));
            w
        };

        let result = fuse_matmul_add(nodes, &weights);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_cancel_consecutive_transpose_identity() {
        let mut node1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node1.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let mut node2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
        node2.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);

        let nodes = vec![node1, node2];
        let result = cancel_consecutive_transpose(nodes);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_cancel_consecutive_transpose_non_identity() {
        let mut node1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node1
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![2, 0, 1]);
        let mut node2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
        node2
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 2, 0]);

        let nodes = vec![node1, node2];
        let result = cancel_consecutive_transpose(nodes);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_cancel_consecutive_transpose_compose() {
        let mut node1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node1
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 2, 0]);
        let mut node2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
        node2
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 2, 0]);

        let nodes = vec![node1, node2];
        let result = cancel_consecutive_transpose(nodes);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Transpose));
        let perm = result[0].attrs.int_lists.get("perm").expect("perm attr");
        assert_eq!(perm, &vec![2, 0, 1]);
    }

    #[test]
    fn test_cancel_consecutive_transpose_redirect() {
        let mut node1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node1.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let mut node2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
        node2.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let relu = make_node(OpKind::Relu, "relu", vec!["t2_out"], vec!["out"]);

        let nodes = vec![node1, node2, relu];
        let result = cancel_consecutive_transpose(nodes);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "relu");
        assert_eq!(result[0].inputs[0], "x");
    }

    #[test]
    fn test_cancel_single_transpose() {
        let mut node = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let nodes = vec![node];
        let result = cancel_consecutive_transpose(nodes);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_fuse_conv_batchnorm() {
        let conv = make_node(
            OpKind::Conv,
            "conv",
            vec!["x", "conv_w", "conv_b"],
            vec!["conv_out"],
        );
        let mut bn = make_node(
            OpKind::BatchNorm,
            "bn",
            vec!["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            vec!["bn_out"],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

        let nodes = vec![conv, bn];
        let mut weights = HashMap::new();
        weights.insert(
            "conv_w".to_string(),
            Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
        );
        weights.insert("conv_b".to_string(), Tensor::new(vec![0.0], vec![1]));
        weights.insert("bn_scale".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("bn_bias".to_string(), Tensor::new(vec![0.0], vec![1]));
        weights.insert("bn_mean".to_string(), Tensor::new(vec![0.0], vec![1]));
        weights.insert("bn_var".to_string(), Tensor::new(vec![1.0], vec![1]));

        let result = fuse_conv_batchnorm(nodes, &mut weights);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Conv));
        assert_eq!(result[0].outputs[0], "bn_out");
        assert!(weights.contains_key("conv_fused_weight"));
        assert!(weights.contains_key("conv_fused_bias"));
    }

    #[test]
    fn test_fuse_conv_batchnorm_no_conv_bias() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "conv_w"], vec!["conv_out"]);
        let mut bn = make_node(
            OpKind::BatchNorm,
            "bn",
            vec!["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            vec!["bn_out"],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

        let nodes = vec![conv, bn];
        let mut weights = HashMap::new();
        weights.insert(
            "conv_w".to_string(),
            Tensor::new(vec![2.0], vec![1, 1, 1, 1]),
        );
        weights.insert("bn_scale".to_string(), Tensor::new(vec![3.0], vec![1]));
        weights.insert("bn_bias".to_string(), Tensor::new(vec![0.5], vec![1]));
        weights.insert("bn_mean".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("bn_var".to_string(), Tensor::new(vec![4.0], vec![1]));

        let result = fuse_conv_batchnorm(nodes, &mut weights);
        assert_eq!(result.len(), 1);

        let fused_w = weights.get("conv_fused_weight").expect("fused weight");
        let inv_std = 1.0 / (4.0f32 + 1e-5).sqrt();
        let expected_w = 2.0 * 3.0 * inv_std;
        assert!((fused_w.data[0] - expected_w).abs() < 1e-5);

        let fused_b = weights.get("conv_fused_bias").expect("fused bias");
        let expected_b = (0.0 - 1.0) * 3.0 * inv_std + 0.5;
        assert!((fused_b.data[0] - expected_b).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_conv_batchnorm_multiple_consumers() {
        let conv = make_node(
            OpKind::Conv,
            "conv",
            vec!["x", "conv_w", "conv_b"],
            vec!["conv_out"],
        );
        let mut bn = make_node(
            OpKind::BatchNorm,
            "bn",
            vec!["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            vec!["bn_out"],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 1e-5);
        let relu = make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["relu_out"]);

        let nodes = vec![conv, bn, relu];
        let mut weights = HashMap::new();
        weights.insert(
            "conv_w".to_string(),
            Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
        );
        weights.insert("conv_b".to_string(), Tensor::new(vec![0.0], vec![1]));
        weights.insert("bn_scale".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("bn_bias".to_string(), Tensor::new(vec![0.0], vec![1]));
        weights.insert("bn_mean".to_string(), Tensor::new(vec![0.0], vec![1]));
        weights.insert("bn_var".to_string(), Tensor::new(vec![1.0], vec![1]));

        let result = fuse_conv_batchnorm(nodes, &mut weights);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_fuse_conv_relu() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["relu_out"]);

        let nodes = vec![conv, relu];
        let result = fuse_conv_relu(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Conv));
        assert_eq!(result[0].outputs[0], "relu_out");
        assert_eq!(result[0].attrs.s("activation"), "relu");
    }

    #[test]
    fn test_fuse_conv_clip_as_relu() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
        clip.attrs.floats.insert("min".to_string(), 0.0);
        clip.attrs.floats.insert("max".to_string(), f32::INFINITY);

        let nodes = vec![conv, clip];
        let result = fuse_conv_relu(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Conv));
        assert_eq!(result[0].outputs[0], "clip_out");
        assert_eq!(result[0].attrs.s("activation"), "relu");
    }

    #[test]
    fn test_fuse_conv_clip_general() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
        clip.attrs.floats.insert("min".to_string(), 0.0);
        clip.attrs.floats.insert("max".to_string(), 6.0);

        let nodes = vec![conv, clip];
        let result = fuse_conv_relu(nodes);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].attrs.s("activation"), "clip");
        assert_eq!(result[0].attrs.f("activation_min", -1.0), 0.0);
        assert_eq!(result[0].attrs.f("activation_max", -1.0), 6.0);
    }

    #[test]
    fn test_fuse_conv_relu_no_fusion_multiple_consumers() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["conv_out"], vec!["relu_out"]);
        let add = make_node(
            OpKind::Add,
            "add",
            vec!["conv_out", "other"],
            vec!["add_out"],
        );

        let nodes = vec![conv, relu, add];
        let result = fuse_conv_relu(nodes);

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_fuse_layer_norm_basic() {
        let (nodes, weights) = make_layer_norm_pattern(false);
        let result = fuse_layer_norm(nodes, &weights);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::LayerNorm));
        assert_eq!(result[0].inputs[0], "X");
        assert_eq!(result[0].outputs[0], "normalized");
        let eps = result[0].attrs.f("epsilon", 0.0);
        assert!((eps - 1e-5).abs() < 1e-8);
    }

    #[test]
    fn test_fuse_layer_norm_with_scale_bias() {
        let (nodes, weights) = make_layer_norm_pattern(true);
        let result = fuse_layer_norm(nodes, &weights);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::LayerNorm));
        assert_eq!(result[0].inputs.len(), 3);
        assert_eq!(result[0].inputs[0], "X");
        assert_eq!(result[0].inputs[1], "scale");
        assert_eq!(result[0].inputs[2], "bias");
        assert_eq!(result[0].outputs[0], "output");
    }

    #[test]
    fn test_fuse_layer_norm_no_match_wrong_pow() {
        let (nodes, mut weights) = make_layer_norm_pattern(false);
        weights.insert("pow_exp".to_string(), Tensor::new(vec![3.0], vec![1]));

        let original_len = nodes.len();
        let result = fuse_layer_norm(nodes, &weights);

        assert_eq!(result.len(), original_len);
    }
}
