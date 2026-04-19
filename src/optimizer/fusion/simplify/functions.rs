//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::{Attributes, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

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
            Some(p) => p.clone(),
            None => continue,
        };
        let perm2 = match node.attrs.int_lists.get("perm") {
            Some(p) => p.clone(),
            None => continue,
        };
        if perm1.len() != perm2.len() {
            continue;
        }
        let composed: Vec<i64> = perm2
            .iter()
            .map(|&j| {
                let j_usize = j as usize;
                if j_usize < perm1.len() {
                    perm1[j_usize]
                } else {
                    j
                }
            })
            .collect();
        let is_identity = composed.iter().enumerate().all(|(idx, &v)| v == idx as i64);
        if is_identity {
            let original_input = match nodes[prev_idx].inputs.first() {
                Some(name) => name.clone(),
                None => continue,
            };
            skip.insert(prev_idx);
            skip.insert(i);
            if let Some(out_name) = node.outputs.first() {
                redirects.insert(out_name.clone(), original_input);
            }
        } else {
            let mut new_attrs = Attributes::default();
            new_attrs.int_lists.insert("perm".to_string(), composed);
            let original_input = match nodes[prev_idx].inputs.first() {
                Some(name) => name.clone(),
                None => continue,
            };
            let collapsed = Node {
                op: OpKind::Transpose,
                name: format!("{}_collapsed_transpose", nodes[prev_idx].name),
                inputs: vec![original_input],
                outputs: node.outputs.clone(),
                attrs: new_attrs,
            };
            skip.insert(prev_idx);
            replacements.insert(i, collapsed);
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
/// Consecutive Reshape cancellation.
/// If Reshape(X, shape1) → Reshape(_, shape2) and the final shape equals X's
/// original shape (available as a constant), we can eliminate both Reshapes.
/// If shape2 is different from X's original shape but the intermediate has a
/// single consumer, we can still eliminate the first Reshape and keep only the second
/// connected directly to the original input.
pub fn cancel_consecutive_reshape(nodes: Vec<Node>) -> Vec<Node> {
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
    let mut redirects: HashMap<String, String> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        if !matches!(node.op, OpKind::Reshape) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }
        let prev_out = &node.inputs[0];
        if consumer_count.get(prev_out).copied().unwrap_or(0) != 1 {
            continue;
        }
        let prev_idx = match producer.get(prev_out) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&prev_idx) {
            continue;
        }
        if !matches!(nodes[prev_idx].op, OpKind::Reshape) {
            continue;
        }
        if nodes[prev_idx].inputs.is_empty() {
            continue;
        }
        let original_input = nodes[prev_idx].inputs[0].clone();
        let mut new_inputs = vec![original_input.clone()];
        if node.inputs.len() > 1 {
            new_inputs.push(node.inputs[1].clone());
        }
        let shapes_match = if nodes[prev_idx].inputs.len() > 1 && node.inputs.len() > 1 {
            nodes[prev_idx].inputs[1] == node.inputs[1]
        } else {
            false
        };
        if shapes_match {
            skip.insert(prev_idx);
            skip.insert(i);
            if let Some(out_name) = node.outputs.first() {
                redirects.insert(out_name.clone(), original_input);
            }
        } else {
            let collapsed = Node {
                op: OpKind::Reshape,
                name: format!("{}_collapsed_reshape", nodes[prev_idx].name),
                inputs: new_inputs,
                outputs: node.outputs.clone(),
                attrs: node.attrs.clone(),
            };
            skip.insert(prev_idx);
            replacements.insert(i, collapsed);
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
/// SiLU fusion: Sigmoid(X) + Mul(X, Sigmoid(X)) → SiLU(X).
///
/// The SiLU (Sigmoid Linear Unit) activation is `x * sigmoid(x)`.  Many modern
/// transformer architectures (SwiGLU, LLaMA MLP, etc.) emit this as two separate
/// ONNX ops.  Fusing them eliminates one intermediate tensor and lets downstream
/// execution engines use a single fused kernel.
///
/// Conditions:
/// - Sigmoid node has exactly one consumer (the Mul node).
/// - Mul's inputs are the original X and Sigmoid's output (either order).
pub fn fuse_mul_sigmoid_to_silu(nodes: Vec<Node>) -> Vec<Node> {
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
        if !matches!(node.op, OpKind::Mul) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }
        let (sigmoid_out, x_name, sigmoid_idx) = {
            let inp0 = &node.inputs[0];
            let inp1 = &node.inputs[1];
            let try_order =
                |sig_candidate: &str, x_candidate: &str| -> Option<(String, String, usize)> {
                    let sig_idx = match producer.get(sig_candidate) {
                        Some(&idx) => idx,
                        None => return None,
                    };
                    if skip.contains(&sig_idx) {
                        return None;
                    }
                    if !matches!(nodes[sig_idx].op, OpKind::Sigmoid) {
                        return None;
                    }
                    if consumer_count.get(sig_candidate).copied().unwrap_or(0) != 1 {
                        return None;
                    }
                    if nodes[sig_idx].inputs.is_empty() {
                        return None;
                    }
                    if nodes[sig_idx].inputs[0] != *x_candidate {
                        return None;
                    }
                    Some((sig_candidate.to_string(), x_candidate.to_string(), sig_idx))
                };
            match try_order(inp1, inp0).or_else(|| try_order(inp0, inp1)) {
                Some(result) => result,
                None => continue,
            }
        };
        let _ = &sigmoid_out;
        let fused = Node {
            op: OpKind::SiLU,
            name: format!("{}_fused_silu", node.name),
            inputs: vec![x_name],
            outputs: node.outputs.clone(),
            attrs: Attributes::default(),
        };
        replacements.insert(i, fused);
        skip.insert(sigmoid_idx);
    }
    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}
/// Rsqrt fusion: Div(const_1, Sqrt(X)) → Reciprocal(Sqrt(X)).
///
/// When a Div node's first input is a constant scalar 1.0 and its second input
/// comes from Sqrt, replace the Div with Reciprocal.  This eliminates the
/// constant tensor allocation for the `1.0` scalar and replaces a general Div
/// with a cheaper Reciprocal op.  Common in attention score computation
/// (`1.0 / sqrt(d_k)`).
///
/// Conditions:
/// - Div's first input is a constant weight tensor with a single value of 1.0.
/// - Div's second input comes from a Sqrt node with exactly one consumer.
pub fn fuse_div_sqrt_to_rsqrt(nodes: Vec<Node>, weights: &HashMap<String, Tensor>) -> Vec<Node> {
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
    let mut replacements: HashMap<usize, Node> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if !matches!(node.op, OpKind::Div) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }
        let numerator_name = &node.inputs[0];
        let denominator_name = &node.inputs[1];
        let is_const_one = match weights.get(numerator_name) {
            Some(t) => t.numel() == 1 && (t.data[0] - 1.0).abs() < 1e-7,
            None => false,
        };
        if !is_const_one {
            continue;
        }
        let sqrt_idx = match producer.get(denominator_name) {
            Some(&idx) => idx,
            None => continue,
        };
        if !matches!(nodes[sqrt_idx].op, OpKind::Sqrt) {
            continue;
        }
        if consumer_count.get(denominator_name).copied().unwrap_or(0) != 1 {
            continue;
        }
        let fused = Node {
            op: OpKind::Reciprocal,
            name: format!("{}_fused_rsqrt", node.name),
            inputs: vec![denominator_name.clone()],
            outputs: node.outputs.clone(),
            attrs: Attributes::default(),
        };
        replacements.insert(i, fused);
    }
    nodes
        .into_iter()
        .enumerate()
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}
/// Gather + Gather composition on the same axis.
///
/// Pattern: `Gather(Gather(X, indices1, axis=a), indices2, axis=0)`
/// When the outer Gather operates on axis 0 of the inner Gather's result,
/// and both effectively index along the same original axis `a`, compose the
/// index selections: `indices_composed = gather(indices1, indices2)` →
/// single `Gather(X, indices_composed, axis=a)`.
///
/// Conservative: only applies when both Gathers share the same axis value
/// and the inner Gather's output has exactly one consumer.
pub fn fuse_gather_composition(nodes: Vec<Node>, weights: &HashMap<String, Tensor>) -> Vec<Node> {
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
    let skip: HashSet<usize> = HashSet::new();
    let mut replacements: HashMap<usize, Node> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        if !matches!(node.op, OpKind::Gather) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }
        let outer_axis = node.attrs.i("axis", 0);
        let inner_result_name = &node.inputs[0];
        let outer_indices_name = &node.inputs[1];
        if consumer_count.get(inner_result_name).copied().unwrap_or(0) != 1 {
            continue;
        }
        let inner_idx = match producer.get(inner_result_name) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&inner_idx) {
            continue;
        }
        if !matches!(nodes[inner_idx].op, OpKind::Gather) {
            continue;
        }
        if nodes[inner_idx].inputs.len() < 2 {
            continue;
        }
        let inner_axis = nodes[inner_idx].attrs.i("axis", 0);
        if outer_axis != inner_axis {
            continue;
        }
        if outer_axis != 0 && outer_axis != inner_axis {
            continue;
        }
        let orig_data_name = &nodes[inner_idx].inputs[0];
        let inner_indices_name = &nodes[inner_idx].inputs[1];
        let inner_indices = match weights.get(inner_indices_name) {
            Some(t) => t,
            None => continue,
        };
        let outer_indices = match weights.get(outer_indices_name) {
            Some(t) => t,
            None => continue,
        };
        let mut composed_data = Vec::with_capacity(outer_indices.data.len());
        let mut valid = true;
        for &oi in &outer_indices.data {
            let idx = oi as usize;
            if idx >= inner_indices.data.len() {
                valid = false;
                break;
            }
            composed_data.push(inner_indices.data[idx]);
        }
        if !valid {
            continue;
        }
        let composed_name = format!("{}_composed_indices", node.name);
        let composed_shape = outer_indices.shape.clone();
        let mut const_attrs = Attributes::default();
        const_attrs.tensors.insert(
            "value".to_string(),
            Tensor::new(composed_data, composed_shape),
        );
        let const_node = Node {
            op: OpKind::Constant,
            name: format!("{}_const", composed_name),
            inputs: vec![],
            outputs: vec![composed_name.clone()],
            attrs: const_attrs,
        };
        let mut fused_attrs = Attributes::default();
        fused_attrs.ints.insert("axis".to_string(), inner_axis);
        let fused_gather = Node {
            op: OpKind::Gather,
            name: format!("{}_fused_gather", nodes[inner_idx].name),
            inputs: vec![orig_data_name.clone(), composed_name],
            outputs: node.outputs.clone(),
            attrs: fused_attrs,
        };
        replacements.insert(inner_idx, const_node);
        replacements.insert(i, fused_gather);
    }
    nodes
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            if skip.contains(&i) {
                n
            } else {
                replacements.remove(&i).unwrap_or(n)
            }
        })
        .collect()
}
/// Dropout elimination during inference.
///
/// During inference, a Dropout node with `training_mode=false` (or absent,
/// since ONNX defaults to inference) or with `ratio=0` acts as identity.
/// This pass removes such Dropout nodes entirely and redirects consumers
/// to the Dropout's input.
///
/// Also handles the Softmax → Dropout pattern specifically, but applies to
/// any standalone inference-mode Dropout.
pub fn eliminate_dropout_inference(nodes: Vec<Node>) -> Vec<Node> {
    if nodes.is_empty() {
        return nodes;
    }
    let mut skip: HashSet<usize> = HashSet::new();
    let mut redirects: HashMap<String, String> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if !matches!(node.op, OpKind::Dropout) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }
        let training_mode = node.attrs.i("training_mode", 0);
        if training_mode != 0 {
            continue;
        }
        let data_input = &node.inputs[0];
        if let Some(out_name) = node.outputs.first() {
            if !out_name.is_empty() {
                redirects.insert(out_name.clone(), data_input.clone());
            }
        }
        let mask_used = node.outputs.get(1).is_some_and(|mask_name| {
            !mask_name.is_empty()
                && nodes
                    .iter()
                    .any(|n| n.inputs.iter().any(|inp| inp == mask_name))
        });
        if mask_used {
            redirects.remove(node.outputs.first().unwrap_or(&String::new()));
            continue;
        }
        skip.insert(i);
    }
    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(_, mut n)| {
            for inp in &mut n.inputs {
                if let Some(redirect) = redirects.get(inp) {
                    *inp = redirect.clone();
                }
            }
            n
        })
        .collect()
}
/// Transpose + Reshape simplification.
///
/// Pattern: `Reshape(Transpose(X, perm), shape)`
///
/// When a Transpose is immediately followed by a Reshape and the Transpose's
/// permutation is such that the data is still contiguous in memory (i.e., the
/// permutation only reorders trailing dimensions of size 1, or the transpose
/// is effectively a no-op for the given shape), we can eliminate the Transpose
/// and keep only the Reshape.
///
/// Conservative check: we only apply this when the permutation is the identity
/// (effectively a no-op transpose that some exporters emit) or when examining
/// constant shape information shows the transpose doesn't change memory layout.
pub fn simplify_transpose_reshape(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
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
        if !matches!(node.op, OpKind::Reshape) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }
        let reshape_input = &node.inputs[0];
        if consumer_count.get(reshape_input).copied().unwrap_or(0) != 1 {
            continue;
        }
        let transpose_idx = match producer.get(reshape_input) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&transpose_idx) {
            continue;
        }
        if !matches!(nodes[transpose_idx].op, OpKind::Transpose) {
            continue;
        }
        let perm = match nodes[transpose_idx].attrs.int_lists.get("perm") {
            Some(p) => p.clone(),
            None => continue,
        };
        let is_identity = perm.iter().enumerate().all(|(idx, &v)| v == idx as i64);
        let is_contiguous_transpose = if !is_identity {
            if let Some(shape_name) = node.inputs.get(1) {
                if let Some(shape_tensor) = weights.get(shape_name) {
                    let rank = perm.len();
                    if rank == 0 {
                        false
                    } else {
                        let target_dims: Vec<i64> =
                            shape_tensor.data.iter().map(|&v| v as i64).collect();
                        target_dims.len() == 1
                            || (target_dims.len() < rank && target_dims.iter().all(|&d| d >= 0))
                    }
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            true
        };
        if !is_identity && !is_contiguous_transpose {
            continue;
        }
        let original_input = match nodes[transpose_idx].inputs.first() {
            Some(name) => name.clone(),
            None => continue,
        };
        let mut new_inputs = vec![original_input];
        for inp in node.inputs.iter().skip(1) {
            new_inputs.push(inp.clone());
        }
        let simplified = Node {
            op: OpKind::Reshape,
            name: format!("{}_simplified", node.name),
            inputs: new_inputs,
            outputs: node.outputs.clone(),
            attrs: node.attrs.clone(),
        };
        skip.insert(transpose_idx);
        replacements.insert(i, simplified);
    }
    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}
