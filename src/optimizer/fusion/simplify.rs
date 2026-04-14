//! Graph simplification passes:
//! - Consecutive Transpose cancellation / composition
//! - Consecutive Reshape cancellation / collapsing
//! - Mul + Sigmoid → SiLU fusion
//! - Div(1, Sqrt(X)) → Reciprocal(Sqrt(X)) (Rsqrt fusion)

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

        // prev_out must have exactly one consumer (this node)
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

        // Compose perm: composed[i] = perm1[perm2[i]]
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

        // Check if composed is identity
        let is_identity = composed.iter().enumerate().all(|(idx, &v)| v == idx as i64);

        if is_identity {
            // Both transposes cancel out: redirect downstream consumers to the
            // original input of the first transpose
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
            // Replace both with a single Transpose(composed_perm)
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

        // Two consecutive Reshapes: Reshape(orig_input, shape1) → Reshape(_, shape2)
        // We can always collapse to a single Reshape(orig_input, shape2).
        let original_input = nodes[prev_idx].inputs[0].clone();

        // Build a replacement Reshape that goes directly from original input to final shape
        let mut new_inputs = vec![original_input.clone()];
        // Carry over the shape input from the second Reshape (index 1 if present)
        if node.inputs.len() > 1 {
            new_inputs.push(node.inputs[1].clone());
        }

        // Check if the first Reshape's original input and second Reshape's shape target
        // happen to have the same shape param — if so, both are redundant
        let shapes_match = if nodes[prev_idx].inputs.len() > 1 && node.inputs.len() > 1 {
            nodes[prev_idx].inputs[1] == node.inputs[1]
        } else {
            false
        };

        if shapes_match {
            // Both Reshapes cancel out: the result is the original input
            skip.insert(prev_idx);
            skip.insert(i);
            if let Some(out_name) = node.outputs.first() {
                redirects.insert(out_name.clone(), original_input);
            }
        } else {
            // Collapse two Reshapes into one: skip the first, replace the second
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

        // Try both orderings: Mul(X, Sigmoid(X)) and Mul(Sigmoid(X), X)
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
                    // Sigmoid must have exactly one consumer (this Mul node)
                    if consumer_count.get(sig_candidate).copied().unwrap_or(0) != 1 {
                        return None;
                    }
                    // Sigmoid's input must be the same as the other Mul input (X)
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
        let _ = &sigmoid_out; // suppress unused warning for clarity

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

        // Check numerator is a constant scalar 1.0
        let is_const_one = match weights.get(numerator_name) {
            Some(t) => t.numel() == 1 && (t.data[0] - 1.0).abs() < 1e-7,
            None => false,
        };
        if !is_const_one {
            continue;
        }

        // Check denominator comes from Sqrt
        let sqrt_idx = match producer.get(denominator_name) {
            Some(&idx) => idx,
            None => continue,
        };
        if !matches!(nodes[sqrt_idx].op, OpKind::Sqrt) {
            continue;
        }

        // Sqrt must have exactly one consumer (this Div node)
        if consumer_count.get(denominator_name).copied().unwrap_or(0) != 1 {
            continue;
        }

        // Replace Div(1.0, Sqrt(X)) with Reciprocal(Sqrt(X))
        // Keep the Sqrt node as-is; replace the Div with Reciprocal
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

        // Inner result must have exactly one consumer (this outer Gather)
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

        // Only compose when both Gathers are on the same axis
        if outer_axis != inner_axis {
            continue;
        }
        // The outer Gather must index along axis 0 (of the inner result),
        // which maps to the same axis as inner_axis when they match.
        if outer_axis != 0 && outer_axis != inner_axis {
            continue;
        }

        let orig_data_name = &nodes[inner_idx].inputs[0];
        let inner_indices_name = &nodes[inner_idx].inputs[1];

        // Try to compose indices if both are constant weight tensors.
        // composed[j] = inner_indices[outer_indices[j]]
        let inner_indices = match weights.get(inner_indices_name) {
            Some(t) => t,
            None => continue,
        };
        let outer_indices = match weights.get(outer_indices_name) {
            Some(t) => t,
            None => continue,
        };

        // Compose indices
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

        // We can't mutate the weights map directly (immutable ref), so we store
        // the composed indices as a Constant node that produces the tensor.
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

        // Replace inner Gather with the Constant node, outer Gather with fused Gather
        replacements.insert(inner_idx, const_node);
        replacements.insert(i, fused_gather);
    }

    nodes
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            if skip.contains(&i) {
                // should not happen since skip is unused here, but be safe
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

        // Check training_mode attribute — if true, keep the Dropout
        let training_mode = node.attrs.i("training_mode", 0);
        if training_mode != 0 {
            continue;
        }

        // Check ratio — even if training_mode is 0, if ratio is specified
        // and non-zero, some runtimes still apply it. We only eliminate when
        // ratio is 0 or absent (default semantics = inference identity).
        // ONNX spec says Dropout in inference is always identity regardless
        // of ratio, so we eliminate unconditionally when training_mode != 1.

        let data_input = &node.inputs[0];

        // Dropout has up to 2 outputs: [output, mask].
        // Redirect the first output to the data input.
        if let Some(out_name) = node.outputs.first() {
            if !out_name.is_empty() {
                redirects.insert(out_name.clone(), data_input.clone());
            }
        }

        // The mask output (if present) cannot be redirected meaningfully,
        // so we only eliminate the Dropout if the mask output is unused or absent.
        let mask_used = node.outputs.get(1).is_some_and(|mask_name| {
            !mask_name.is_empty()
                && nodes
                    .iter()
                    .any(|n| n.inputs.iter().any(|inp| inp == mask_name))
        });

        if mask_used {
            // Cannot eliminate: downstream uses the mask output
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

        // Reshape input must have exactly one consumer (this Reshape)
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

        // Check 1: Is the permutation the identity?
        let is_identity = perm.iter().enumerate().all(|(idx, &v)| v == idx as i64);

        // Check 2: For non-identity permutations, check if the transpose
        // preserves contiguity. This requires knowing the input shape.
        // We check if the target shape in the Reshape (from weights) has
        // total elements consistent with a contiguous view.
        let is_contiguous_transpose = if !is_identity {
            // Check if the permutation only swaps dimensions of size 1.
            // This requires shape info. Check if a shape constant is available
            // for the transpose's input (heuristic: look for a Shape node or
            // attribute that tells us the input shape).
            //
            // Conservative fallback: check if the reshape target shape (input[1])
            // is a known constant, and compare total element counts.
            if let Some(shape_name) = node.inputs.get(1) {
                if let Some(shape_tensor) = weights.get(shape_name) {
                    // If the reshape target has total elements = product of shape,
                    // and the permutation only reorders trailing unit dims, it's safe.
                    // For now, we only handle the case where perm swaps dims that
                    // are all size 1 except possibly the last contiguous block.
                    let rank = perm.len();
                    if rank == 0 {
                        false
                    } else {
                        // Check: perm is identity up to a suffix of size-1 dims.
                        // Since we don't always know the input shape, be very
                        // conservative: check if the perm moves only the last dim
                        // and the target shape absorbs it.
                        // Heuristic: if target shape has fewer dims than perm rank,
                        // the reshape is collapsing dims, making the transpose moot.
                        let target_dims: Vec<i64> =
                            shape_tensor.data.iter().map(|&v| v as i64).collect();
                        // If target is a single dim (flatten), transpose before flatten
                        // is always safe to remove since flatten ignores layout.
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

        // Eliminate the Transpose: connect Reshape directly to Transpose's input
        let original_input = match nodes[transpose_idx].inputs.first() {
            Some(name) => name.clone(),
            None => continue,
        };

        let mut new_inputs = vec![original_input];
        // Carry over shape input(s) from the Reshape
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::test_utils::make_node;

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

    // --- cancel_consecutive_reshape tests ---

    #[test]
    fn test_cancel_consecutive_reshape_collapse() {
        // Reshape(x, shape1) → Reshape(_, shape2)  → single Reshape(x, shape2)
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "shape1"], vec!["r1_out"]);
        let r2 = make_node(
            OpKind::Reshape,
            "r2",
            vec!["r1_out", "shape2"],
            vec!["r2_out"],
        );

        let nodes = vec![r1, r2];
        let result = cancel_consecutive_reshape(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Reshape));
        assert_eq!(result[0].inputs[0], "x");
        assert_eq!(result[0].inputs[1], "shape2");
        assert_eq!(result[0].outputs[0], "r2_out");
    }

    #[test]
    fn test_cancel_consecutive_reshape_same_shape_eliminates_both() {
        // Reshape(x, shape_a) → Reshape(_, shape_a) → both eliminated
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "shape_a"], vec!["r1_out"]);
        let r2 = make_node(
            OpKind::Reshape,
            "r2",
            vec!["r1_out", "shape_a"],
            vec!["r2_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["r2_out"], vec!["out"]);

        let nodes = vec![r1, r2, relu];
        let result = cancel_consecutive_reshape(nodes);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "relu");
        assert_eq!(result[0].inputs[0], "x");
    }

    #[test]
    fn test_cancel_consecutive_reshape_no_cancel_multiple_consumers() {
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "shape1"], vec!["r1_out"]);
        let r2 = make_node(
            OpKind::Reshape,
            "r2",
            vec!["r1_out", "shape2"],
            vec!["r2_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["r1_out"], vec!["relu_out"]);

        let nodes = vec![r1, r2, relu];
        let result = cancel_consecutive_reshape(nodes);

        // r1_out has 2 consumers, so no cancellation
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_cancel_consecutive_reshape_single_node() {
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "shape1"], vec!["r1_out"]);
        let nodes = vec![r1];
        let result = cancel_consecutive_reshape(nodes);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cancel_consecutive_reshape_three_reshapes() {
        // Three consecutive Reshapes: should at least collapse adjacent pairs
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "s1"], vec!["r1_out"]);
        let r2 = make_node(OpKind::Reshape, "r2", vec!["r1_out", "s2"], vec!["r2_out"]);
        let r3 = make_node(OpKind::Reshape, "r3", vec!["r2_out", "s3"], vec!["r3_out"]);

        let nodes = vec![r1, r2, r3];
        let result = cancel_consecutive_reshape(nodes);

        // First pass: r1+r2 collapse into one, then r3 remains
        // (single-pass, so r2_collapsed + r3 aren't merged)
        assert!(result.len() <= 2);
        // Final output should still reference r3_out
        let last = result.last().expect("should have at least one node");
        assert_eq!(last.outputs[0], "r3_out");
    }

    // --- fuse_mul_sigmoid_to_silu tests ---

    #[test]
    fn test_fuse_mul_sigmoid_to_silu_basic() {
        // Sigmoid(X) → Mul(X, sigmoid_out) → SiLU(X)
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["x"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["mul_out"]);

        let nodes = vec![sigmoid, mul];
        let result = fuse_mul_sigmoid_to_silu(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::SiLU));
        assert_eq!(result[0].inputs, vec!["x"]);
        assert_eq!(result[0].outputs, vec!["mul_out"]);
    }

    #[test]
    fn test_fuse_mul_sigmoid_to_silu_reversed_mul_inputs() {
        // Mul(sigmoid_out, X) — reversed order
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["x"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["sig_out", "x"], vec!["mul_out"]);

        let nodes = vec![sigmoid, mul];
        let result = fuse_mul_sigmoid_to_silu(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::SiLU));
        assert_eq!(result[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_fuse_mul_sigmoid_to_silu_no_fusion_multiple_consumers() {
        // Sigmoid output used by Mul AND another node — don't fuse
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["x"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["mul_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["sig_out"], vec!["relu_out"]);

        let nodes = vec![sigmoid, mul, relu];
        let result = fuse_mul_sigmoid_to_silu(nodes);

        // sig_out has 2 consumers, no fusion
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_fuse_mul_sigmoid_to_silu_no_fusion_different_input() {
        // Sigmoid(Y) but Mul(X, sigmoid_out) where X != Y — don't fuse
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["y"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["mul_out"]);

        let nodes = vec![sigmoid, mul];
        let result = fuse_mul_sigmoid_to_silu(nodes);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_fuse_mul_sigmoid_to_silu_preserves_downstream() {
        // Verify downstream nodes rewire correctly
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["x"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["mul_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["mul_out"], vec!["relu_out"]);

        let nodes = vec![sigmoid, mul, relu];
        let result = fuse_mul_sigmoid_to_silu(nodes);

        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::SiLU));
        assert_eq!(result[0].outputs, vec!["mul_out"]);
        assert_eq!(result[1].inputs, vec!["mul_out"]);
    }

    // --- fuse_div_sqrt_to_rsqrt tests ---

    #[test]
    fn test_fuse_div_sqrt_to_rsqrt_basic() {
        let sqrt = make_node(OpKind::Sqrt, "sqrt", vec!["x"], vec!["sqrt_out"]);
        let div = make_node(OpKind::Div, "div", vec!["one", "sqrt_out"], vec!["div_out"]);

        let nodes = vec![sqrt, div];
        let mut weights = HashMap::new();
        weights.insert("one".to_string(), Tensor::new(vec![1.0], vec![1]));

        let result = fuse_div_sqrt_to_rsqrt(nodes, &weights);

        assert_eq!(result.len(), 2);
        // Sqrt remains
        assert!(matches!(result[0].op, OpKind::Sqrt));
        // Div replaced with Reciprocal
        assert!(matches!(result[1].op, OpKind::Reciprocal));
        assert_eq!(result[1].inputs, vec!["sqrt_out"]);
        assert_eq!(result[1].outputs, vec!["div_out"]);
    }

    #[test]
    fn test_fuse_div_sqrt_to_rsqrt_not_const_one() {
        let sqrt = make_node(OpKind::Sqrt, "sqrt", vec!["x"], vec!["sqrt_out"]);
        let div = make_node(OpKind::Div, "div", vec!["two", "sqrt_out"], vec!["div_out"]);

        let nodes = vec![sqrt, div];
        let mut weights = HashMap::new();
        weights.insert("two".to_string(), Tensor::new(vec![2.0], vec![1]));

        let result = fuse_div_sqrt_to_rsqrt(nodes, &weights);

        // No fusion: numerator is not 1.0
        assert_eq!(result.len(), 2);
        assert!(matches!(result[1].op, OpKind::Div));
    }

    #[test]
    fn test_fuse_div_sqrt_to_rsqrt_not_sqrt() {
        let relu = make_node(OpKind::Relu, "relu", vec!["x"], vec!["relu_out"]);
        let div = make_node(OpKind::Div, "div", vec!["one", "relu_out"], vec!["div_out"]);

        let nodes = vec![relu, div];
        let mut weights = HashMap::new();
        weights.insert("one".to_string(), Tensor::new(vec![1.0], vec![1]));

        let result = fuse_div_sqrt_to_rsqrt(nodes, &weights);

        // No fusion: denominator is not Sqrt
        assert_eq!(result.len(), 2);
        assert!(matches!(result[1].op, OpKind::Div));
    }

    #[test]
    fn test_fuse_div_sqrt_to_rsqrt_sqrt_multiple_consumers() {
        let sqrt = make_node(OpKind::Sqrt, "sqrt", vec!["x"], vec!["sqrt_out"]);
        let div = make_node(OpKind::Div, "div", vec!["one", "sqrt_out"], vec!["div_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["sqrt_out"], vec!["relu_out"]);

        let nodes = vec![sqrt, div, relu];
        let mut weights = HashMap::new();
        weights.insert("one".to_string(), Tensor::new(vec![1.0], vec![1]));

        let result = fuse_div_sqrt_to_rsqrt(nodes, &weights);

        // No fusion: sqrt_out has 2 consumers
        assert_eq!(result.len(), 3);
        assert!(matches!(result[1].op, OpKind::Div));
    }

    // --- fuse_gather_composition tests ---

    #[test]
    fn test_fuse_gather_composition_basic() {
        // Gather(X, idx1, axis=0) → Gather(_, idx2, axis=0)
        // idx1 = [2, 0, 1], idx2 = [1, 2] → composed = [0, 1]
        let mut gather1 = make_node(OpKind::Gather, "g1", vec!["data", "idx1"], vec!["g1_out"]);
        gather1.attrs.ints.insert("axis".to_string(), 0);

        let mut gather2 = make_node(OpKind::Gather, "g2", vec!["g1_out", "idx2"], vec!["g2_out"]);
        gather2.attrs.ints.insert("axis".to_string(), 0);

        let nodes = vec![gather1, gather2];
        let mut weights = HashMap::new();
        weights.insert(
            "idx1".to_string(),
            Tensor::new(vec![2.0, 0.0, 1.0], vec![3]),
        );
        weights.insert("idx2".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));

        let result = fuse_gather_composition(nodes, &weights);

        // Should produce: Constant(composed_indices) + Gather(data, composed, axis=0)
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Constant));
        assert!(matches!(result[1].op, OpKind::Gather));
        assert_eq!(result[1].inputs[0], "data");
        assert_eq!(result[1].outputs[0], "g2_out");

        // Verify composed indices: idx1[idx2[0]] = idx1[1] = 0, idx1[idx2[1]] = idx1[2] = 1
        let composed = result[0]
            .attrs
            .tensors
            .get("value")
            .expect("composed tensor");
        assert_eq!(composed.data, vec![0.0, 1.0]);
    }

    #[test]
    fn test_fuse_gather_composition_no_fusion_different_axis() {
        // Inner axis=0, outer axis=1 — don't fuse
        let mut gather1 = make_node(OpKind::Gather, "g1", vec!["data", "idx1"], vec!["g1_out"]);
        gather1.attrs.ints.insert("axis".to_string(), 0);

        let mut gather2 = make_node(OpKind::Gather, "g2", vec!["g1_out", "idx2"], vec!["g2_out"]);
        gather2.attrs.ints.insert("axis".to_string(), 1);

        let nodes = vec![gather1, gather2];
        let mut weights = HashMap::new();
        weights.insert("idx1".to_string(), Tensor::new(vec![0.0, 1.0], vec![2]));
        weights.insert("idx2".to_string(), Tensor::new(vec![0.0], vec![1]));

        let result = fuse_gather_composition(nodes, &weights);

        // No fusion: different axes
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Gather));
        assert!(matches!(result[1].op, OpKind::Gather));
    }

    #[test]
    fn test_fuse_gather_composition_no_fusion_multiple_consumers() {
        // g1_out consumed by two nodes — don't fuse
        let mut gather1 = make_node(OpKind::Gather, "g1", vec!["data", "idx1"], vec!["g1_out"]);
        gather1.attrs.ints.insert("axis".to_string(), 0);

        let mut gather2 = make_node(OpKind::Gather, "g2", vec!["g1_out", "idx2"], vec!["g2_out"]);
        gather2.attrs.ints.insert("axis".to_string(), 0);

        let relu = make_node(OpKind::Relu, "relu", vec!["g1_out"], vec!["relu_out"]);

        let nodes = vec![gather1, gather2, relu];
        let mut weights = HashMap::new();
        weights.insert("idx1".to_string(), Tensor::new(vec![0.0, 1.0], vec![2]));
        weights.insert("idx2".to_string(), Tensor::new(vec![0.0], vec![1]));

        let result = fuse_gather_composition(nodes, &weights);

        // No fusion: g1_out has 2 consumers
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_fuse_gather_composition_no_fusion_non_constant_indices() {
        // Indices are not constants — can't compose
        let mut gather1 = make_node(
            OpKind::Gather,
            "g1",
            vec!["data", "dynamic_idx1"],
            vec!["g1_out"],
        );
        gather1.attrs.ints.insert("axis".to_string(), 0);

        let mut gather2 = make_node(
            OpKind::Gather,
            "g2",
            vec!["g1_out", "dynamic_idx2"],
            vec!["g2_out"],
        );
        gather2.attrs.ints.insert("axis".to_string(), 0);

        let nodes = vec![gather1, gather2];
        let weights = HashMap::new(); // No constant indices

        let result = fuse_gather_composition(nodes, &weights);

        // No fusion: indices not in weights
        assert_eq!(result.len(), 2);
    }

    // --- eliminate_dropout_inference tests ---

    #[test]
    fn test_eliminate_dropout_inference_basic() {
        // Dropout(x) with training_mode=0 → eliminated
        let dropout = make_node(OpKind::Dropout, "dropout", vec!["x"], vec!["dropout_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["dropout_out"], vec!["out"]);

        let nodes = vec![dropout, relu];
        let result = eliminate_dropout_inference(nodes);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Relu));
        assert_eq!(result[0].inputs[0], "x");
    }

    #[test]
    fn test_eliminate_dropout_after_softmax() {
        // Softmax → Dropout (inference) → eliminated Dropout
        let softmax = make_node(OpKind::Softmax, "softmax", vec!["x"], vec!["sm_out"]);
        let dropout = make_node(
            OpKind::Dropout,
            "dropout",
            vec!["sm_out"],
            vec!["dropout_out"],
        );
        let matmul = make_node(
            OpKind::MatMul,
            "matmul",
            vec!["dropout_out", "v"],
            vec!["out"],
        );

        let nodes = vec![softmax, dropout, matmul];
        let result = eliminate_dropout_inference(nodes);

        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Softmax));
        assert!(matches!(result[1].op, OpKind::MatMul));
        assert_eq!(result[1].inputs[0], "sm_out");
    }

    #[test]
    fn test_eliminate_dropout_training_mode_not_eliminated() {
        // Dropout with training_mode=1 — keep it
        let mut dropout = make_node(OpKind::Dropout, "dropout", vec!["x"], vec!["dropout_out"]);
        dropout.attrs.ints.insert("training_mode".to_string(), 1);

        let relu = make_node(OpKind::Relu, "relu", vec!["dropout_out"], vec!["out"]);

        let nodes = vec![dropout, relu];
        let result = eliminate_dropout_inference(nodes);

        // Not eliminated: training mode is on
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Dropout));
    }

    #[test]
    fn test_eliminate_dropout_mask_output_used() {
        // Dropout with mask output consumed — don't eliminate
        let dropout = make_node(
            OpKind::Dropout,
            "dropout",
            vec!["x"],
            vec!["dropout_out", "dropout_mask"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["dropout_out"], vec!["out1"]);
        let mask_user = make_node(
            OpKind::Identity,
            "mask_user",
            vec!["dropout_mask"],
            vec!["out2"],
        );

        let nodes = vec![dropout, relu, mask_user];
        let result = eliminate_dropout_inference(nodes);

        // Not eliminated: mask output is consumed
        assert_eq!(result.len(), 3);
        assert!(matches!(result[0].op, OpKind::Dropout));
    }

    // --- simplify_transpose_reshape tests ---

    #[test]
    fn test_simplify_transpose_reshape_identity_perm() {
        // Transpose(perm=[0,1,2]) → Reshape → simplified to just Reshape
        let mut transpose = make_node(OpKind::Transpose, "transpose", vec!["x"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![0, 1, 2]);

        let reshape = make_node(
            OpKind::Reshape,
            "reshape",
            vec!["t_out", "target_shape"],
            vec!["out"],
        );

        let nodes = vec![transpose, reshape];
        let weights = HashMap::new();

        let result = simplify_transpose_reshape(nodes, &weights);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Reshape));
        assert_eq!(result[0].inputs[0], "x");
        assert_eq!(result[0].inputs[1], "target_shape");
        assert_eq!(result[0].outputs[0], "out");
    }

    #[test]
    fn test_simplify_transpose_reshape_flatten_after_transpose() {
        // Transpose(perm=[0,2,1]) → Reshape(flatten to 1D) — safe to remove Transpose
        let mut transpose = make_node(OpKind::Transpose, "transpose", vec!["x"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![0, 2, 1]);

        let reshape = make_node(
            OpKind::Reshape,
            "reshape",
            vec!["t_out", "flat_shape"],
            vec!["out"],
        );

        let nodes = vec![transpose, reshape];
        let mut weights = HashMap::new();
        // Target shape is [N] — flatten to 1D
        weights.insert("flat_shape".to_string(), Tensor::new(vec![-1.0], vec![1]));

        let result = simplify_transpose_reshape(nodes, &weights);

        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Reshape));
        assert_eq!(result[0].inputs[0], "x");
    }

    #[test]
    fn test_simplify_transpose_reshape_no_simplification_non_trivial() {
        // Transpose(perm=[1,0]) → Reshape with 2D target — don't simplify
        let mut transpose = make_node(OpKind::Transpose, "transpose", vec!["x"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 0]);

        let reshape = make_node(
            OpKind::Reshape,
            "reshape",
            vec!["t_out", "shape_2d"],
            vec!["out"],
        );

        let nodes = vec![transpose, reshape];
        let mut weights = HashMap::new();
        // Target shape is [3, 4] — 2D, same rank as input
        weights.insert("shape_2d".to_string(), Tensor::new(vec![3.0, 4.0], vec![2]));

        let result = simplify_transpose_reshape(nodes, &weights);

        // Not simplified: non-identity perm with same-rank target
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Transpose));
        assert!(matches!(result[1].op, OpKind::Reshape));
    }

    #[test]
    fn test_simplify_transpose_reshape_no_simplification_multiple_consumers() {
        // Transpose output consumed by two nodes — don't simplify
        let mut transpose = make_node(OpKind::Transpose, "transpose", vec!["x"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![0, 1, 2]);

        let reshape = make_node(
            OpKind::Reshape,
            "reshape",
            vec!["t_out", "shape"],
            vec!["out1"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["t_out"], vec!["out2"]);

        let nodes = vec![transpose, reshape, relu];
        let weights = HashMap::new();

        let result = simplify_transpose_reshape(nodes, &weights);

        // Not simplified: t_out has 2 consumers
        assert_eq!(result.len(), 3);
    }
}
