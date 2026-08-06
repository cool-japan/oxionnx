//! MatMul + Add → Gemm fusion pass.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Rank of tensor `name`, from shape inference or from the initializer map.
fn known_rank(
    name: &str,
    weights: &HashMap<String, Tensor>,
    known_shapes: &HashMap<String, Vec<usize>>,
) -> Option<usize> {
    if let Some(shape) = known_shapes.get(name) {
        return Some(shape.len());
    }
    weights.get(name).map(|t| t.ndim())
}

/// MatMul + Add → Gemm fusion.
///
/// Pattern: node A = `MatMul(X, W)`, node B = `Add(A.output, bias)` where the
/// bias is a 1-D initializer.  Fused: `Gemm(X, W, bias)` with `alpha = beta = 1`.
///
/// ONNX defines `Gemm` only for a **2-D** `A`: the typed / quantized kernels
/// read `(m, k) = (a_shape[0], a_shape[1])` and emit `[m, n]`, so feeding them a
/// rank-3 activation (the standard transformer projection `X[B, T, C] @ W`)
/// silently produces a wrongly-shaped result, and a rank-1 `A` indexes out of
/// bounds.  The fusion therefore only fires when `X`'s rank is *known* to be 2;
/// when shape inference cannot prove it, the MatMul + Add pair is left alone.
///
/// The MatMul output must have exactly one consumer (the Add) and must not be a
/// declared graph output — the fused node produces only the Add's output name.
pub fn fuse_matmul_add(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
    known_shapes: &HashMap<String, Vec<usize>>,
    output_names: &[String],
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

    let usage = TensorUsage::new(&nodes, output_names);

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

        if !usage.is_fusable_intermediate(matmul_tensor) {
            continue;
        }

        let matmul_idx = match producer.get(matmul_tensor) {
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

        let a_name = &nodes[matmul_idx].inputs[0];
        let b_name = &nodes[matmul_idx].inputs[1];

        // Gemm's contract is 2-D × 2-D; anything else must stay a MatMul.
        if known_rank(a_name, weights, known_shapes) != Some(2)
            || known_rank(b_name, weights, known_shapes) != Some(2)
        {
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
            inputs: vec![a_name.clone(), b_name.clone(), bias_tensor.clone()],
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
