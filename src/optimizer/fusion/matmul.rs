//! MatMul-related fusion passes:
//! - MatMul + Add → Gemm
//! - LayerNorm pattern fusion
//! - MatMul + Transpose → transposed Gemm
//! - Add(bias) + MatMul → Gemm with pre-computed bias

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

/// MatMul + Transpose → Transposed MatMul fusion.
/// Pattern: MatMul(A, B) → Transpose(perm swaps last two dims)
/// Fused: MatMul(B^T, A^T) with a single Transpose that swaps the last two dims
/// For 2-D inputs this is exact since (A·B)^T = B^T·A^T.
/// We emit a Gemm(B, A) with transA=1, transB=1 to get the transposed result
/// directly, avoiding a separate Transpose node.
pub fn fuse_matmul_transpose(nodes: Vec<Node>) -> Vec<Node> {
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
        if !matches!(node.op, OpKind::Transpose) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }

        // Check that the transpose only swaps the last two dimensions
        let perm = match node.attrs.int_lists.get("perm") {
            Some(p) if p.len() >= 2 => p,
            _ => continue,
        };
        let ndim = perm.len();
        // All dims except the last two must be identity
        let prefix_identity = perm[..ndim - 2]
            .iter()
            .enumerate()
            .all(|(j, &v)| v == j as i64);
        let swaps_last_two =
            perm[ndim - 2] == (ndim - 1) as i64 && perm[ndim - 1] == (ndim - 2) as i64;
        if !prefix_identity || !swaps_last_two {
            continue;
        }

        let matmul_out = &node.inputs[0];
        if consumer_count.get(matmul_out).copied().unwrap_or(0) != 1 {
            continue;
        }

        let matmul_idx = match producer.get(matmul_out) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&matmul_idx) {
            continue;
        }
        if !matches!(nodes[matmul_idx].op, OpKind::MatMul) {
            continue;
        }
        if nodes[matmul_idx].inputs.len() < 2 {
            continue;
        }

        // (A·B)^T = B^T · A^T  →  Gemm(B, A, transA=1, transB=1)
        let a_input = &nodes[matmul_idx].inputs[0];
        let b_input = &nodes[matmul_idx].inputs[1];

        let mut attrs = Attributes::default();
        attrs.ints.insert("transA".to_string(), 1);
        attrs.ints.insert("transB".to_string(), 1);
        attrs.floats.insert("alpha".to_string(), 1.0);
        attrs.floats.insert("beta".to_string(), 0.0);

        let fused = Node {
            op: OpKind::Gemm,
            name: format!("{}_fused_matmul_transpose", nodes[matmul_idx].name),
            inputs: vec![b_input.clone(), a_input.clone()],
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

/// Add(bias) + MatMul → Gemm with bias fusion.
/// Pattern: Add(X, bias_1d) → MatMul(result, W)
/// This is NOT the standard MatMul+Add; here the bias precedes the MatMul.
/// When bias is 1-D (broadcast-added to X), and intermediate has single consumer,
/// we fuse into Gemm(X, W, bias) with alpha=1, beta=1.
///
/// Note: This assumes the bias addition distributes over the matmul, which holds
/// only when the bias is added to the *input* (not the output). The Gemm semantics
/// are: Y = alpha * A @ B + beta * C. So we set C = bias broadcast-multiplied by B,
/// but that's only valid if bias is truly constant and MatMul is linear.
/// Actually, for bias added to input: (X + b) @ W = X @ W + b @ W. The fused
/// bias would be b @ W which requires knowing W. Instead, we only fuse when the
/// bias is a weight and we can compute the fused bias = bias @ W.
pub fn fuse_add_matmul_to_gemm(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
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
        if !matches!(node.op, OpKind::MatMul) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }

        let add_out = &node.inputs[0];
        let w_name = &node.inputs[1];

        // The weight matrix W must be known at compile time
        let w_tensor = match weights.get(w_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        // W must be 2-D: [K, N]
        if w_tensor.shape.len() != 2 {
            continue;
        }

        if consumer_count.get(add_out).copied().unwrap_or(0) != 1 {
            continue;
        }

        let add_idx = match producer.get(add_out) {
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

        // Identify which input of Add is the bias (1-D weight) and which is the data
        let (x_name, bias_name) = {
            let inp0 = &nodes[add_idx].inputs[0];
            let inp1 = &nodes[add_idx].inputs[1];
            if let Some(b) = weights.get(inp1) {
                if b.ndim() == 1 {
                    (inp0.clone(), inp1.clone())
                } else {
                    continue;
                }
            } else if let Some(b) = weights.get(inp0) {
                if b.ndim() == 1 {
                    (inp1.clone(), inp0.clone())
                } else {
                    continue;
                }
            } else {
                continue;
            }
        };

        let bias = match weights.get(&bias_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        // bias shape [K], W shape [K, N] → fused_bias = bias @ W → shape [N]
        let k = w_tensor.shape[0];
        let n = w_tensor.shape[1];
        if bias.shape.len() != 1 || bias.shape[0] != k {
            continue;
        }

        // Compute fused_bias = bias @ W (vector-matrix multiply)
        let mut fused_bias_data = vec![0.0f32; n];
        for (j, fused_val) in fused_bias_data.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += bias.data[ki] * w_tensor.data[ki * n + j];
            }
            *fused_val = sum;
        }

        let fused_bias_name = format!("{}_fused_add_matmul_bias", nodes[add_idx].name);
        weights.insert(
            fused_bias_name.clone(),
            Tensor::new(fused_bias_data, vec![n]),
        );

        let mut attrs = Attributes::default();
        attrs.floats.insert("alpha".to_string(), 1.0);
        attrs.floats.insert("beta".to_string(), 1.0);
        attrs.ints.insert("transA".to_string(), 0);
        attrs.ints.insert("transB".to_string(), 0);

        let fused = Node {
            op: OpKind::Gemm,
            name: format!("{}_fused_add_matmul_gemm", nodes[add_idx].name),
            inputs: vec![x_name, w_name.clone(), fused_bias_name],
            outputs: node.outputs.clone(),
            attrs,
        };

        replacements.insert(add_idx, fused);
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

    // --- fuse_matmul_transpose tests ---

    #[test]
    fn test_fuse_matmul_transpose_2d() {
        let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
        let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 0]);

        let nodes = vec![matmul, transpose];
        let result = fuse_matmul_transpose(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Gemm));
        // Should be Gemm(B, A) with transA=1, transB=1
        assert_eq!(result[0].inputs[0], "b");
        assert_eq!(result[0].inputs[1], "a");
        assert_eq!(result[0].attrs.i("transA", 0), 1);
        assert_eq!(result[0].attrs.i("transB", 0), 1);
        assert_eq!(result[0].outputs[0], "t_out");
    }

    #[test]
    fn test_fuse_matmul_transpose_3d_last_two() {
        let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
        let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![0, 2, 1]);

        let nodes = vec![matmul, transpose];
        let result = fuse_matmul_transpose(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Gemm));
    }

    #[test]
    fn test_fuse_matmul_transpose_no_fusion_wrong_perm() {
        let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
        let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["t_out"]);
        // Perm [2, 0, 1] doesn't just swap last two dims
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![2, 0, 1]);

        let nodes = vec![matmul, transpose];
        let result = fuse_matmul_transpose(nodes);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_fuse_matmul_transpose_no_fusion_multiple_consumers() {
        let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
        let mut transpose = make_node(OpKind::Transpose, "t", vec!["mm_out"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 0]);
        let relu = make_node(OpKind::Relu, "relu", vec!["mm_out"], vec!["relu_out"]);

        let nodes = vec![matmul, transpose, relu];
        let result = fuse_matmul_transpose(nodes);

        assert_eq!(result.len(), 3);
    }

    // --- fuse_add_matmul_to_gemm tests ---

    #[test]
    fn test_fuse_add_matmul_to_gemm() {
        let add = make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]);
        let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

        let nodes = vec![add, matmul];
        let mut weights = HashMap::new();
        // bias: [2], W: [2, 3]
        weights.insert("bias".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
        weights.insert(
            "w".to_string(),
            Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]),
        );

        let result = fuse_add_matmul_to_gemm(nodes, &mut weights);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Gemm));
        assert_eq!(result[0].inputs[0], "x");
        assert_eq!(result[0].inputs[1], "w");
        assert_eq!(result[0].inputs.len(), 3);
        assert_eq!(result[0].outputs[0], "mm_out");
        assert_eq!(result[0].attrs.f("alpha", 0.0), 1.0);
        assert_eq!(result[0].attrs.f("beta", 0.0), 1.0);

        // fused_bias = [1, 2] @ [[1,0,0],[0,1,0]] = [1, 2, 0]
        let fused_bias_name = &result[0].inputs[2];
        let fused_bias = weights
            .get(fused_bias_name)
            .expect("fused bias should exist");
        assert_eq!(fused_bias.shape, vec![3]);
        assert!((fused_bias.data[0] - 1.0).abs() < 1e-6);
        assert!((fused_bias.data[1] - 2.0).abs() < 1e-6);
        assert!((fused_bias.data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_add_matmul_to_gemm_bias_first_input() {
        // Add with bias as first input
        let add = make_node(OpKind::Add, "add", vec!["bias", "x"], vec!["add_out"]);
        let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

        let nodes = vec![add, matmul];
        let mut weights = HashMap::new();
        weights.insert("bias".to_string(), Tensor::new(vec![3.0, 4.0], vec![2]));
        weights.insert(
            "w".to_string(),
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
        );

        let result = fuse_add_matmul_to_gemm(nodes, &mut weights);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Gemm));
        assert_eq!(result[0].inputs[0], "x");
    }

    #[test]
    fn test_fuse_add_matmul_no_fusion_bias_not_1d() {
        let add = make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]);
        let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

        let nodes = vec![add, matmul];
        let mut weights = HashMap::new();
        weights.insert("bias".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));
        weights.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));

        let result = fuse_add_matmul_to_gemm(nodes, &mut weights);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_fuse_add_matmul_no_fusion_w_not_in_weights() {
        let add = make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]);
        let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

        let nodes = vec![add, matmul];
        let mut weights = HashMap::new();
        weights.insert("bias".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
        // w not in weights

        let result = fuse_add_matmul_to_gemm(nodes, &mut weights);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_fuse_add_matmul_no_fusion_shape_mismatch() {
        let add = make_node(OpKind::Add, "add", vec!["x", "bias"], vec!["add_out"]);
        let matmul = make_node(OpKind::MatMul, "mm", vec!["add_out", "w"], vec!["mm_out"]);

        let nodes = vec![add, matmul];
        let mut weights = HashMap::new();
        // bias [3] vs W [2, 2] — K mismatch
        weights.insert(
            "bias".to_string(),
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        );
        weights.insert("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]));

        let result = fuse_add_matmul_to_gemm(nodes, &mut weights);
        assert_eq!(result.len(), 2);
    }
}
