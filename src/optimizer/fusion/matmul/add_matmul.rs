//! Add(bias) + MatMul → Gemm with pre-computed bias fusion pass.

use crate::graph::{Attributes, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

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
