//! Graph simplification passes: Transpose / Reshape cancellation, SiLU and
//! Rsqrt fusion, Gather composition, inference-mode Dropout elimination and
//! Transpose+Reshape simplification.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::{NameAllocator, TensorUsage};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Read a constant shape operand (a Reshape / Gather index tensor) as `i64`s.
fn const_i64_operand(weights: &HashMap<String, Tensor>, name: &str) -> Option<Vec<i64>> {
    if name.is_empty() {
        return None;
    }
    let tensor = weights.get(name)?;
    Some(tensor.data.iter().map(|&v| v as i64).collect())
}

/// Resolve a Reshape target-shape operand against a known input shape,
/// expanding `-1` (infer) and, unless `allowzero`, `0` (copy the input dim).
///
/// Returns `None` when the target is not resolvable (more than one `-1`, a `0`
/// beyond the input rank, or a non-divisible inferred dimension).
fn resolve_reshape_target(
    dims: &[i64],
    input_shape: &[usize],
    allowzero: bool,
) -> Option<Vec<usize>> {
    let input_numel: usize = input_shape.iter().product();
    let mut out: Vec<usize> = Vec::with_capacity(dims.len());
    let mut infer_at: Option<usize> = None;
    let mut known_product: usize = 1;

    for (idx, &dim) in dims.iter().enumerate() {
        match dim {
            -1 => {
                if infer_at.is_some() {
                    return None;
                }
                infer_at = Some(idx);
                out.push(1);
            }
            0 if !allowzero => {
                let copied = *input_shape.get(idx)?;
                known_product = known_product.checked_mul(copied)?;
                out.push(copied);
            }
            d if d >= 0 => {
                let d = usize::try_from(d).ok()?;
                known_product = known_product.checked_mul(d)?;
                out.push(d);
            }
            _ => return None,
        }
    }

    if let Some(idx) = infer_at {
        if known_product == 0 || input_numel % known_product != 0 {
            return None;
        }
        let inferred = input_numel / known_product;
        *out.get_mut(idx)? = inferred;
    } else if known_product != input_numel {
        return None;
    }

    Some(out)
}

/// Whether a Reshape's target-shape operand keeps its meaning when the Reshape
/// is re-parented onto a *differently shaped* input.
///
/// With `allowzero = 0` (the ONNX default) a `0` in the target copies the
/// corresponding dimension **from the current input**, so re-parenting changes
/// the result.  A dynamic target could contain a zero we cannot see, so it is
/// rejected as well.
fn target_shape_is_reparent_safe(node: &Node, weights: &HashMap<String, Tensor>) -> bool {
    if node.attrs.i("allowzero", 0) != 0 {
        return true;
    }
    match node
        .inputs
        .get(1)
        .and_then(|name| const_i64_operand(weights, name))
    {
        Some(dims) => !dims.contains(&0),
        None => false,
    }
}

/// Consecutive Transpose cancellation.
///
/// `Transpose(perm1) → Transpose(perm2)`: when the composition is the identity
/// both nodes are removed and consumers are re-pointed at the original input;
/// otherwise the pair collapses into a single `Transpose(composed_perm)`.
///
/// The intermediate tensor must have exactly one consumer and must not be a
/// declared graph output.  The identity case additionally *renames* the second
/// Transpose's output away, so it is skipped when that output is a graph output.
pub fn cancel_consecutive_transpose(nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.clone(), i);
        }
    }
    let usage = TensorUsage::new(&nodes, output_names);
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
        if skip.contains(&prev_idx) || replacements.contains_key(&prev_idx) {
            continue;
        }
        if !matches!(nodes[prev_idx].op, OpKind::Transpose) {
            continue;
        }
        if !usage.is_fusable_intermediate(input_name) {
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
        let original_input = match nodes[prev_idx].inputs.first() {
            Some(name) if !name.is_empty() => name.clone(),
            _ => continue,
        };
        if is_identity {
            let out_name = match node.outputs.first() {
                Some(name) if !name.is_empty() => name.clone(),
                _ => continue,
            };
            // Removing both Transposes renames `out_name` to `original_input`;
            // that is only allowed for an internal tensor.
            if usage.is_graph_output(&out_name) {
                continue;
            }
            skip.insert(prev_idx);
            skip.insert(i);
            redirects.insert(out_name, original_input);
        } else {
            let mut new_attrs = Attributes::default();
            new_attrs.int_lists.insert("perm".to_string(), composed);
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
///
/// `Reshape(X, s1) → Reshape(_, s2)`:
/// * both nodes are removed when `s2` provably resolves to `X`'s own shape
///   (this needs `X`'s shape and a constant `s2` — matching `s1 == s2`
///   textually proves nothing: `X = [2, 3]` with `s1 = s2 = [6]` reshapes to
///   `[6]`, not back to `[2, 3]`);
/// * otherwise the pair collapses into a single `Reshape(X, s2)`.
///
/// Re-parenting the second Reshape onto `X` is only sound when its target shape
/// carries no `allowzero = 0` zero (which would copy a dimension from the
/// *intermediate* tensor), and the intermediate must have exactly one consumer
/// and not be a declared graph output.
pub fn cancel_consecutive_reshape(
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
        if !usage.is_fusable_intermediate(prev_out) {
            continue;
        }
        let prev_idx = match producer.get(prev_out) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&prev_idx) || replacements.contains_key(&prev_idx) {
            continue;
        }
        if !matches!(nodes[prev_idx].op, OpKind::Reshape) {
            continue;
        }
        if nodes[prev_idx].inputs.is_empty() {
            continue;
        }
        // A zero in the target shape refers to the tensor being reshaped, so
        // the second Reshape can only be re-parented when it has none.
        if !target_shape_is_reparent_safe(node, weights) {
            continue;
        }
        let original_input = nodes[prev_idx].inputs[0].clone();
        if original_input.is_empty() {
            continue;
        }
        let mut new_inputs = vec![original_input.clone()];
        if node.inputs.len() > 1 {
            new_inputs.push(node.inputs[1].clone());
        }

        // Does the second Reshape restore `X`'s original shape exactly?
        let restores_input_shape = match known_shapes.get(&original_input) {
            Some(x_shape) => node
                .inputs
                .get(1)
                .and_then(|name| const_i64_operand(weights, name))
                .and_then(|dims| {
                    resolve_reshape_target(&dims, x_shape, node.attrs.i("allowzero", 0) != 0)
                })
                .is_some_and(|resolved| resolved == *x_shape),
            None => false,
        };

        if restores_input_shape {
            let out_name = match node.outputs.first() {
                Some(name) if !name.is_empty() => name.clone(),
                _ => continue,
            };
            if usage.is_graph_output(&out_name) {
                continue;
            }
            skip.insert(prev_idx);
            skip.insert(i);
            redirects.insert(out_name, original_input);
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

/// SiLU fusion: `Sigmoid(X)` + `Mul(X, Sigmoid(X))` → `SiLU(X)`.
///
/// The SiLU (Sigmoid Linear Unit) activation is `x * sigmoid(x)`.  Many modern
/// transformer architectures (SwiGLU, LLaMA MLP, …) emit this as two separate
/// ONNX ops.  Fusing them eliminates one intermediate tensor and lets execution
/// engines use a single fused kernel.
///
/// Conditions:
/// - the Sigmoid output has exactly one consumer (the Mul) and is not a graph output;
/// - the Mul's inputs are the original `X` and the Sigmoid output (either order).
pub fn fuse_mul_sigmoid_to_silu(nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
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
        if !matches!(node.op, OpKind::Mul) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }
        let (x_name, sigmoid_idx) = {
            let inp0 = &node.inputs[0];
            let inp1 = &node.inputs[1];
            let try_order = |sig_candidate: &str, x_candidate: &str| -> Option<(String, usize)> {
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
                if !usage.is_fusable_intermediate(sig_candidate) {
                    return None;
                }
                if nodes[sig_idx].inputs.is_empty() {
                    return None;
                }
                if nodes[sig_idx].inputs[0] != *x_candidate {
                    return None;
                }
                Some((x_candidate.to_string(), sig_idx))
            };
            match try_order(inp1, inp0).or_else(|| try_order(inp0, inp1)) {
                Some(result) => result,
                None => continue,
            }
        };
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

/// Rsqrt fusion: `Div(const_1, Sqrt(X))` → `Reciprocal(Sqrt(X))`.
///
/// When a Div node's first input is a constant scalar `1.0` and its second input
/// comes from Sqrt, replace the Div with Reciprocal.  This eliminates the
/// constant tensor allocation for the `1.0` scalar and replaces a general Div
/// with a cheaper Reciprocal op.  Common in attention score computation
/// (`1.0 / sqrt(d_k)`).
///
/// No tensor is removed by this rewrite (the Sqrt output becomes the
/// Reciprocal's input and the Div output name is preserved), so it is safe for
/// graph outputs by construction.
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
            Some(t) => t.numel() == 1 && t.data.first().is_some_and(|&v| (v - 1.0).abs() < 1e-7),
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
/// Pattern: `Gather(Gather(X, indices1, axis=a), indices2, axis=a)` with both
/// index tensors constant and `indices1` **1-D**.  Only then does axis `a` of
/// the inner result correspond one-to-one to `indices1`, and the two selections
/// compose into `indices_composed = indices1[indices2]`, i.e. a single
/// `Gather(X, indices_composed, axis=a)`.
///
/// The composed indices are stored as an initializer (not as a `Constant` node,
/// which would have to be executed on every run).  Negative `indices2` values
/// are normalised against `indices1`'s length, and out-of-range values abort the
/// rewrite.
///
/// Conservative: the inner Gather's result must have exactly one consumer and
/// must not be a declared graph output.
pub fn fuse_gather_composition(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
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
    let mut names = NameAllocator::new(&nodes, weights);
    let mut skip: HashSet<usize> = HashSet::new();
    let mut replacements: HashMap<usize, Node> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        if !matches!(node.op, OpKind::Gather) {
            continue;
        }
        if node.inputs.len() < 2 || node.outputs.is_empty() {
            continue;
        }
        let outer_axis = node.attrs.i("axis", 0);
        let inner_result_name = &node.inputs[0];
        let outer_indices_name = &node.inputs[1];
        if !usage.is_fusable_intermediate(inner_result_name) {
            continue;
        }
        let inner_idx = match producer.get(inner_result_name) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&inner_idx) || replacements.contains_key(&inner_idx) {
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
        let orig_data_name = &nodes[inner_idx].inputs[0];
        let inner_indices_name = &nodes[inner_idx].inputs[1];
        let inner_indices = match weights.get(inner_indices_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        // Only a 1-D inner index vector keeps axis `a` of the inner result
        // aligned with `indices1`.
        if inner_indices.ndim() != 1 {
            continue;
        }
        let outer_indices = match weights.get(outer_indices_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let inner_len = inner_indices.data.len();
        let inner_len_i64 = match i64::try_from(inner_len) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mut composed_data = Vec::with_capacity(outer_indices.data.len());
        let mut valid = true;
        for &oi in &outer_indices.data {
            let raw = oi as i64;
            // ONNX Gather allows negative indices (from the end).
            let normalized = if raw < 0 { raw + inner_len_i64 } else { raw };
            let idx = match usize::try_from(normalized) {
                Ok(v) if v < inner_len => v,
                _ => {
                    valid = false;
                    break;
                }
            };
            match inner_indices.data.get(idx) {
                Some(&v) => composed_data.push(v),
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }
        let name_base = match node.outputs.first() {
            Some(name) if !name.is_empty() => name.clone(),
            _ => continue,
        };
        let composed_name = names.allocate(&name_base, "_composed_indices");
        let composed_shape = outer_indices.shape.clone();
        weights.insert(
            composed_name.clone(),
            Tensor::new(composed_data, composed_shape),
        );
        let mut fused_attrs = Attributes::default();
        fused_attrs.ints.insert("axis".to_string(), inner_axis);
        let fused_gather = Node {
            op: OpKind::Gather,
            name: format!("{}_fused_gather", nodes[inner_idx].name),
            inputs: vec![orig_data_name.clone(), composed_name],
            outputs: node.outputs.clone(),
            attrs: fused_attrs,
        };
        skip.insert(inner_idx);
        replacements.insert(i, fused_gather);
    }
    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}

/// Dropout elimination during inference.
///
/// A `Dropout` node acts as the identity when training mode is off, which is
/// the ONNX default.  Such nodes are removed and their consumers re-pointed at
/// the Dropout's data input.
///
/// The elimination is skipped when
/// * `training_mode` is set (as an attribute, or as input 2 — which must be a
///   compile-time constant `false`; a runtime flag keeps the node),
/// * the mask output is produced and consumed (or exported), or
/// * the data output is a declared graph output, since removing the node would
///   rename it away.
pub fn eliminate_dropout_inference(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
    output_names: &[String],
) -> Vec<Node> {
    if nodes.is_empty() {
        return nodes;
    }
    let usage = TensorUsage::new(&nodes, output_names);
    let mut skip: HashSet<usize> = HashSet::new();
    let mut redirects: HashMap<String, String> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if !matches!(node.op, OpKind::Dropout) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }
        if node.attrs.i("training_mode", 0) != 0 {
            continue;
        }
        // opset 12+: `training_mode` is input 2.  Present ⇒ must be a constant
        // `false`; a runtime flag means the node has to stay.
        if let Some(mode_name) = node.inputs.get(2) {
            let is_constant_inference = mode_name.is_empty()
                || crate::optimizer::graph_utils::const_scalar(weights, mode_name)
                    .is_some_and(|flag| flag == 0.0);
            if !is_constant_inference {
                continue;
            }
        }
        let data_input = &node.inputs[0];
        if data_input.is_empty() {
            continue;
        }
        let out_name = match node.outputs.first() {
            Some(name) if !name.is_empty() => name.clone(),
            _ => continue,
        };
        if usage.is_graph_output(&out_name) {
            continue;
        }
        // A produced mask that anything reads (or that the model exports) keeps
        // the node alive.
        let mask_used = node.outputs.get(1).is_some_and(|mask_name| {
            !mask_name.is_empty()
                && (usage.consumers(mask_name) > 0 || usage.is_graph_output(mask_name))
        });
        if mask_used {
            continue;
        }
        redirects.insert(out_name, data_input.clone());
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

/// Is `perm` a memory no-op for a tensor of shape `shape`?
///
/// A transpose leaves the flat element order untouched exactly when the axes it
/// actually moves all have extent 1 — i.e. when dropping the extent-1 axes from
/// `perm` leaves a strictly increasing sequence.  In that case a following
/// Reshape sees the same buffer and can be re-parented onto the Transpose's
/// input.
fn transpose_is_memory_noop(perm: &[i64], shape: &[usize]) -> bool {
    if perm.len() != shape.len() {
        return false;
    }
    let mut previous: Option<i64> = None;
    for &axis in perm {
        let idx = match usize::try_from(axis) {
            Ok(v) if v < shape.len() => v,
            _ => return false,
        };
        if shape[idx] == 1 {
            continue;
        }
        if let Some(prev) = previous {
            if axis <= prev {
                return false;
            }
        }
        previous = Some(axis);
    }
    true
}

/// Transpose + Reshape simplification.
///
/// Pattern: `Reshape(Transpose(X, perm), shape)`.
///
/// The Transpose can be dropped only when it does not move any data: either
/// `perm` is the identity, or `X`'s shape is known and `perm` only reorders
/// axes of extent 1 (so the flat element order is unchanged).  Anything else —
/// including the classic `Transpose(NCHW → NHWC) → Reshape([N, -1])` — must
/// keep the Transpose, since dropping it silently flattens a different layout.
///
/// When the Transpose does change the shape (extent-1 axes moving around), the
/// Reshape's target must not rely on `allowzero = 0` zero-copying, which refers
/// to the tensor being reshaped.
pub fn simplify_transpose_reshape(
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
        if !matches!(node.op, OpKind::Reshape) {
            continue;
        }
        if node.inputs.is_empty() {
            continue;
        }
        let reshape_input = &node.inputs[0];
        if !usage.is_fusable_intermediate(reshape_input) {
            continue;
        }
        let transpose_idx = match producer.get(reshape_input) {
            Some(&idx) => idx,
            None => continue,
        };
        if skip.contains(&transpose_idx) || replacements.contains_key(&transpose_idx) {
            continue;
        }
        if !matches!(nodes[transpose_idx].op, OpKind::Transpose) {
            continue;
        }
        let perm = match nodes[transpose_idx].attrs.int_lists.get("perm") {
            Some(p) if !p.is_empty() => p.clone(),
            _ => continue,
        };
        let original_input = match nodes[transpose_idx].inputs.first() {
            Some(name) if !name.is_empty() => name.clone(),
            _ => continue,
        };
        let is_identity = perm.iter().enumerate().all(|(idx, &v)| v == idx as i64);
        if !is_identity {
            // Needs the input shape to prove the permutation moves no data …
            let input_shape = match known_shapes.get(&original_input) {
                Some(s) => s.clone(),
                None => continue,
            };
            if !transpose_is_memory_noop(&perm, &input_shape) {
                continue;
            }
            // … and the Reshape target must not depend on the (now different)
            // input dimensions.
            if !target_shape_is_reparent_safe(node, weights) {
                continue;
            }
        }
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
