//! Conv-related fusion passes:
//! - Conv + BatchNorm folding (weight baking)
//! - Conv + Relu / Clip activation fusion
//! - Conv + Clip(0,6) → Conv with ReLU6 activation
//! - Standalone BatchNorm folding (Mul + Add replacement)

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
/// - Conv output has exactly one consumer (the Add node).
/// - Add output has exactly one consumer (the Relu node).
/// - One of Add's inputs comes from Conv; the other is the residual / skip connection.
pub fn fuse_conv_add_relu(nodes: Vec<Node>) -> Vec<Node> {
    if nodes.len() < 3 {
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
        if !matches!(node.op, OpKind::Relu) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }

        let relu_input = &node.inputs[0];

        // Relu input must have exactly one consumer (this Relu)
        if consumer_count.get(relu_input).copied().unwrap_or(0) != 1 {
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
                    if consumer_count.get(conv_candidate).copied().unwrap_or(0) != 1 {
                        return None;
                    }
                    let idx = producer.get(conv_candidate)?;
                    if skip.contains(idx) {
                        return None;
                    }
                    if !matches!(nodes[*idx].op, OpKind::Conv) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::test_utils::make_node;

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

    // --- fuse_conv_clip_to_conv_relu6 tests ---

    #[test]
    fn test_fuse_conv_clip_to_conv_relu6_basic() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
        clip.attrs.floats.insert("min".to_string(), 0.0);
        clip.attrs.floats.insert("max".to_string(), 6.0);

        let nodes = vec![conv, clip];
        let result = fuse_conv_clip_to_conv_relu6(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Conv));
        assert_eq!(result[0].attrs.s("activation"), "relu6");
        assert_eq!(result[0].attrs.f("activation_min", -1.0), 0.0);
        assert_eq!(result[0].attrs.f("activation_max", -1.0), 6.0);
        assert_eq!(result[0].outputs, vec!["clip_out"]);
    }

    #[test]
    fn test_fuse_conv_clip_to_conv_relu6_wrong_range() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
        clip.attrs.floats.insert("min".to_string(), 0.0);
        clip.attrs.floats.insert("max".to_string(), 1.0); // Not 6.0

        let nodes = vec![conv, clip];
        let result = fuse_conv_clip_to_conv_relu6(nodes);

        // Not ReLU6 range, no fusion
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_fuse_conv_clip_to_conv_relu6_not_conv() {
        // Relu followed by Clip(0,6) — not a Conv, so no fusion
        let relu = make_node(OpKind::Relu, "relu", vec!["x"], vec!["relu_out"]);
        let mut clip = make_node(OpKind::Clip, "clip", vec!["relu_out"], vec!["clip_out"]);
        clip.attrs.floats.insert("min".to_string(), 0.0);
        clip.attrs.floats.insert("max".to_string(), 6.0);

        let nodes = vec![relu, clip];
        let result = fuse_conv_clip_to_conv_relu6(nodes);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_fuse_conv_clip_to_conv_relu6_multiple_consumers() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let mut clip = make_node(OpKind::Clip, "clip", vec!["conv_out"], vec!["clip_out"]);
        clip.attrs.floats.insert("min".to_string(), 0.0);
        clip.attrs.floats.insert("max".to_string(), 6.0);
        let add = make_node(
            OpKind::Add,
            "add",
            vec!["conv_out", "other"],
            vec!["add_out"],
        );

        let nodes = vec![conv, clip, add];
        let result = fuse_conv_clip_to_conv_relu6(nodes);

        // conv_out has 2 consumers, no fusion
        assert_eq!(result.len(), 3);
    }

    // --- fold_batch_norm_inference tests ---

    #[test]
    fn test_fold_batch_norm_inference_basic() {
        let mut bn = make_node(
            OpKind::BatchNorm,
            "bn",
            vec!["x", "scale", "bias", "mean", "var"],
            vec!["bn_out"],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

        let nodes = vec![bn];
        let mut weights = HashMap::new();
        weights.insert("scale".to_string(), Tensor::new(vec![2.0], vec![1]));
        weights.insert("bias".to_string(), Tensor::new(vec![0.5], vec![1]));
        weights.insert("mean".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("var".to_string(), Tensor::new(vec![4.0], vec![1]));

        let result = fold_batch_norm_inference(nodes, &mut weights);

        // BN replaced with Mul + Add
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Mul));
        assert!(matches!(result[1].op, OpKind::Add));

        // Check outputs — final output should match BN's output
        assert_eq!(result[1].outputs, vec!["bn_out"]);
        // Mul takes X as input
        assert_eq!(result[0].inputs[0], "x");

        // Verify precomputed factor and shift
        let inv_std = 1.0 / (4.0f32 + 1e-5).sqrt();
        let expected_factor = 2.0 * inv_std;
        let expected_shift = 0.5 - 1.0 * expected_factor;

        let factor = weights.get("bn_bn_factor").expect("factor weight");
        assert!((factor.data[0] - expected_factor).abs() < 1e-5);

        let shift = weights.get("bn_bn_shift").expect("shift weight");
        assert!((shift.data[0] - expected_shift).abs() < 1e-5);
    }

    #[test]
    fn test_fold_batch_norm_inference_skips_conv_preceded() {
        // When BN is preceded by Conv, fuse_conv_batchnorm handles it
        let conv = make_node(OpKind::Conv, "conv", vec!["inp", "w"], vec!["conv_out"]);
        let mut bn = make_node(
            OpKind::BatchNorm,
            "bn",
            vec!["conv_out", "scale", "bias", "mean", "var"],
            vec!["bn_out"],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

        let nodes = vec![conv, bn];
        let mut weights = HashMap::new();
        weights.insert("scale".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("bias".to_string(), Tensor::new(vec![0.0], vec![1]));
        weights.insert("mean".to_string(), Tensor::new(vec![0.0], vec![1]));
        weights.insert("var".to_string(), Tensor::new(vec![1.0], vec![1]));

        let result = fold_batch_norm_inference(nodes, &mut weights);

        // Should NOT fold — Conv precedes BN
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Conv));
        assert!(matches!(result[1].op, OpKind::BatchNorm));
    }

    #[test]
    fn test_fold_batch_norm_inference_missing_weights() {
        let mut bn = make_node(
            OpKind::BatchNorm,
            "bn",
            vec!["x", "scale", "bias", "mean", "var"],
            vec!["bn_out"],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

        let nodes = vec![bn];
        let mut weights = HashMap::new();
        weights.insert("scale".to_string(), Tensor::new(vec![1.0], vec![1]));
        // Missing bias, mean, var weights

        let result = fold_batch_norm_inference(nodes, &mut weights);

        // Should not fold — missing weights
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::BatchNorm));
    }

    #[test]
    fn test_fold_batch_norm_inference_multi_channel() {
        let mut bn = make_node(
            OpKind::BatchNorm,
            "bn",
            vec!["x", "scale", "bias", "mean", "var"],
            vec!["bn_out"],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 0.001);

        let nodes = vec![bn];
        let mut weights = HashMap::new();
        weights.insert(
            "scale".to_string(),
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        );
        weights.insert(
            "bias".to_string(),
            Tensor::new(vec![0.1, 0.2, 0.3], vec![3]),
        );
        weights.insert(
            "mean".to_string(),
            Tensor::new(vec![0.5, 1.0, 1.5], vec![3]),
        );
        weights.insert("var".to_string(), Tensor::new(vec![1.0, 2.0, 4.0], vec![3]));

        let result = fold_batch_norm_inference(nodes, &mut weights);

        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Mul));
        assert!(matches!(result[1].op, OpKind::Add));

        let factor = weights.get("bn_bn_factor").expect("factor");
        assert_eq!(factor.shape, vec![3]);
        let shift = weights.get("bn_bn_shift").expect("shift");
        assert_eq!(shift.shape, vec![3]);

        // Verify channel 0: scale=1.0, var=1.0, eps=0.001
        let inv_std_0 = 1.0 / (1.0f32 + 0.001).sqrt();
        let expected_f0 = 1.0 * inv_std_0;
        assert!((factor.data[0] - expected_f0).abs() < 1e-5);
    }

    #[test]
    fn test_fold_batch_norm_inference_shape_mismatch() {
        let mut bn = make_node(
            OpKind::BatchNorm,
            "bn",
            vec!["x", "scale", "bias", "mean", "var"],
            vec!["bn_out"],
        );
        bn.attrs.floats.insert("epsilon".to_string(), 1e-5);

        let nodes = vec![bn];
        let mut weights = HashMap::new();
        weights.insert("scale".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
        weights.insert("bias".to_string(), Tensor::new(vec![0.0], vec![1])); // Mismatch!
        weights.insert("mean".to_string(), Tensor::new(vec![0.0, 0.0], vec![2]));
        weights.insert("var".to_string(), Tensor::new(vec![1.0, 1.0], vec![2]));

        let result = fold_batch_norm_inference(nodes, &mut weights);

        // Shape mismatch — should not fold
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::BatchNorm));
    }

    // --- fuse_conv_add_relu tests ---

    #[test]
    fn test_fuse_conv_add_relu_basic() {
        // Conv(x, w, b) → Add(conv_out, residual) → Relu → fused ConvAddRelu
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
        let add = make_node(
            OpKind::Add,
            "add",
            vec!["conv_out", "residual"],
            vec!["add_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);

        let nodes = vec![conv, add, relu];
        let result = fuse_conv_add_relu(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::ConvAddRelu));
        assert_eq!(result[0].inputs, vec!["x", "w", "b", "residual"]);
        assert_eq!(result[0].outputs, vec!["relu_out"]);
    }

    #[test]
    fn test_fuse_conv_add_relu_reversed_add_inputs() {
        // Add(residual, conv_out) — reversed order
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
        let add = make_node(
            OpKind::Add,
            "add",
            vec!["residual", "conv_out"],
            vec!["add_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);

        let nodes = vec![conv, add, relu];
        let result = fuse_conv_add_relu(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::ConvAddRelu));
        assert_eq!(result[0].inputs, vec!["x", "w", "b", "residual"]);
    }

    #[test]
    fn test_fuse_conv_add_relu_no_bias() {
        // Conv without bias input
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let add = make_node(
            OpKind::Add,
            "add",
            vec!["conv_out", "residual"],
            vec!["add_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);

        let nodes = vec![conv, add, relu];
        let result = fuse_conv_add_relu(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::ConvAddRelu));
        // Bias slot should be empty string
        assert_eq!(result[0].inputs[2], "");
        assert_eq!(result[0].inputs[3], "residual");
    }

    #[test]
    fn test_fuse_conv_add_relu_no_fusion_conv_multiple_consumers() {
        // conv_out consumed by both Add and another node — don't fuse
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
        let add = make_node(
            OpKind::Add,
            "add",
            vec!["conv_out", "residual"],
            vec!["add_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);
        let extra = make_node(OpKind::Relu, "extra", vec!["conv_out"], vec!["extra_out"]);

        let nodes = vec![conv, add, relu, extra];
        let result = fuse_conv_add_relu(nodes);

        assert_eq!(result.len(), 4);
        assert!(matches!(result[0].op, OpKind::Conv));
    }

    #[test]
    fn test_fuse_conv_add_relu_no_fusion_add_multiple_consumers() {
        // add_out consumed by both Relu and another node — don't fuse
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
        let add = make_node(
            OpKind::Add,
            "add",
            vec!["conv_out", "residual"],
            vec!["add_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["add_out"], vec!["relu_out"]);
        let extra = make_node(OpKind::Sigmoid, "extra", vec!["add_out"], vec!["extra_out"]);

        let nodes = vec![conv, add, relu, extra];
        let result = fuse_conv_add_relu(nodes);

        assert_eq!(result.len(), 4);
        assert!(matches!(result[0].op, OpKind::Conv));
    }

    #[test]
    fn test_fuse_conv_add_relu_no_fusion_not_relu() {
        // Pattern ends with Sigmoid instead of Relu — don't fuse
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w", "b"], vec!["conv_out"]);
        let add = make_node(
            OpKind::Add,
            "add",
            vec!["conv_out", "residual"],
            vec!["add_out"],
        );
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["add_out"], vec!["sig_out"]);

        let nodes = vec![conv, add, sigmoid];
        let result = fuse_conv_add_relu(nodes);

        // No fusion: pattern requires Relu at the end
        assert_eq!(result.len(), 3);
    }
}
