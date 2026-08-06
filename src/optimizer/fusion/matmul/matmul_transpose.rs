//! MatMul + Transpose → transposed Gemm fusion pass.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use std::collections::{HashMap, HashSet};

/// MatMul + Transpose → transposed Gemm fusion.
///
/// Pattern: `MatMul(A, B) → Transpose(perm = [1, 0])`.
/// Fused: `Gemm(B, A, transA=1, transB=1)`, which computes `(A·B)^T = B^T·A^T`
/// directly and removes the Transpose node.
///
/// Restricted to the **2-D** case: ONNX `Gemm` is defined only for 2-D operands,
/// and the typed / quantized kernels read `(m, k)` straight out of `a_shape[0..2]`,
/// so a batched (rank ≥ 3) MatMul must keep its explicit Transpose.  A rank-2
/// permutation attribute implies both operands are 2-D.
///
/// The MatMul output must have exactly one consumer (the Transpose) and must not
/// be a declared graph output.
pub fn fuse_matmul_transpose(nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
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
        if !matches!(node.op, OpKind::Transpose) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }

        // Only the 2-D case: `Gemm` cannot express a batched transpose.
        let perm = match node.attrs.int_lists.get("perm") {
            Some(p) if p.len() == 2 => p,
            _ => continue,
        };
        if perm[0] != 1 || perm[1] != 0 {
            continue;
        }

        let matmul_out = &node.inputs[0];
        if !usage.is_fusable_intermediate(matmul_out) {
            continue;
        }

        let matmul_idx = match producer.get(matmul_out) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&matmul_idx) || replacements.contains_key(&matmul_idx) {
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
