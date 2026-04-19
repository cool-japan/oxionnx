//! MatMul + Transpose → transposed Gemm fusion pass.

use crate::graph::{Attributes, Node, OpKind};
use std::collections::{HashMap, HashSet};

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
