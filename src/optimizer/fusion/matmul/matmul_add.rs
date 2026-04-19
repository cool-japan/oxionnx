//! MatMul + Add → Gemm fusion pass.

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
