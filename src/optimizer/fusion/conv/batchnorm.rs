//! Conv + BatchNorm fusion and standalone BatchNorm folding passes.

use crate::graph::{Attributes, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

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

/// Standalone BatchNormalization folding (inference mode).
///
/// When all BatchNorm parameters (scale, bias, mean, var) are known constants
/// and the BatchNorm node is *not* preceded by a Conv (that case is handled by
/// `fuse_conv_batchnorm`), fold the normalisation into a Mul + Add pair with
/// pre-computed constant weights:
///
/// ```text
/// factor = scale / sqrt(var + epsilon)
/// shift  = bias - mean * factor
/// y      = factor * x + shift
/// ```
///
/// This eliminates the runtime overhead of computing mean/var lookups and the
/// sqrt/div at inference time.
pub fn fold_batch_norm_inference(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
) -> Vec<Node> {
    if nodes.is_empty() {
        return nodes;
    }

    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.clone(), i);
        }
    }

    let mut skip: HashSet<usize> = HashSet::new();
    let mut new_nodes: Vec<(usize, Vec<Node>)> = Vec::new();

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

        let x_name = &node.inputs[0];
        let bn_scale_name = &node.inputs[1];
        let bn_bias_name = &node.inputs[2];
        let bn_mean_name = &node.inputs[3];
        let bn_var_name = &node.inputs[4];

        // Skip if preceded by Conv (handled by fuse_conv_batchnorm)
        if let Some(&prev_idx) = producer.get(x_name) {
            if matches!(nodes[prev_idx].op, OpKind::Conv) {
                continue;
            }
        }

        // All BN params must be constant weights
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
        let c_out = bn_scale.data.len();

        // Validate shapes are consistent
        if c_out == 0
            || bn_bias.data.len() != c_out
            || bn_mean.data.len() != c_out
            || bn_var.data.len() != c_out
        {
            continue;
        }

        // Compute factor = scale / sqrt(var + eps)  and  shift = bias - mean * factor
        let mut factor_data = Vec::with_capacity(c_out);
        let mut shift_data = Vec::with_capacity(c_out);
        for c in 0..c_out {
            let inv_std = 1.0 / (bn_var.data[c] + epsilon).sqrt();
            let f = bn_scale.data[c] * inv_std;
            factor_data.push(f);
            shift_data.push(bn_bias.data[c] - bn_mean.data[c] * f);
        }

        let factor_name = format!("{}_bn_factor", node.name);
        let shift_name = format!("{}_bn_shift", node.name);
        let mul_out_name = format!("{}_bn_mul_out", node.name);

        weights.insert(factor_name.clone(), Tensor::new(factor_data, vec![c_out]));
        weights.insert(shift_name.clone(), Tensor::new(shift_data, vec![c_out]));

        // Emit Mul(X, factor) → Add(mul_out, shift)
        let mul_node = Node {
            op: OpKind::Mul,
            name: format!("{}_bn_mul", node.name),
            inputs: vec![x_name.clone(), factor_name],
            outputs: vec![mul_out_name.clone()],
            attrs: Attributes::default(),
        };
        let add_node = Node {
            op: OpKind::Add,
            name: format!("{}_bn_add", node.name),
            inputs: vec![mul_out_name, shift_name],
            outputs: node.outputs.clone(),
            attrs: Attributes::default(),
        };

        skip.insert(i);
        new_nodes.push((i, vec![mul_node, add_node]));
    }

    // Build result: replace skipped BN nodes with their Mul+Add replacements
    let mut result = Vec::with_capacity(nodes.len() + new_nodes.len());
    let replacement_map: HashMap<usize, Vec<Node>> = new_nodes.into_iter().collect();

    for (i, node) in nodes.into_iter().enumerate() {
        if let Some(replacement_nodes) = replacement_map.get(&i) {
            result.extend(replacement_nodes.iter().cloned());
        } else if !skip.contains(&i) {
            result.push(node);
        }
    }

    result
}
