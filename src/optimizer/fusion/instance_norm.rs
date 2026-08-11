//! Spatial (InstanceNorm-style) normalization-chain fusion.
//!
//! # The pattern, as real exporters emit it
//!
//! AdaIN generators normalise each `(n, c)` plane and then apply a *runtime*
//! per-channel scale and shift computed from a style vector. PyTorch cannot
//! export that as `InstanceNormalization` — that op's `scale`/`B` are operands
//! it expects to be initialisers-shaped, and here they come out of a Gemm head
//! — so the normalisation is exported as bare arithmetic. In the inswapper
//! face-swap decoder (opset 11, 238 nodes) each of the 12 style blocks reads:
//!
//! ```text
//! mean = ReduceMean(X, axes=[2,3], keepdims=1)
//! diff = Sub(X, mean)
//! sq   = Mul(diff, diff)            // NOT Pow(diff, 2)
//! var  = ReduceMean(sq, axes=[2,3], keepdims=1)
//! vare = Add(var, eps)              // eps = 1e-8, one shared initialiser
//! std  = Sqrt(vare)
//! rstd = Div(one, std)              // reciprocal, not a direct divide
//! norm = Mul(diff, rstd)
//! ```
//!
//! followed by `Mul(scale, norm)` and `Add(·, shift)` — which this pass leaves
//! exactly where they are, because `scale`/`shift` are runtime tensors and the
//! fused op carries no affine term (see
//! [`oxionnx_ops::registry::oxi_instance_norm`] for why).
//!
//! Eight nodes collapse into one `OxiInstanceNorm`, and with them seven
//! intermediate tensors — including `diff`, which is materialised in full
//! (`[1, 1024, 32, 32]` in the widest block) and read three times.
//!
//! # Variants handled
//!
//! * **square**: `Mul(diff, diff)` (both operands the same tensor) or
//!   `Pow(diff, 2)`.
//! * **reciprocal**: `Mul(diff, rstd)` where `rstd` is `Reciprocal(std)` or
//!   `Div(one, std)`, in either operand order; or the direct
//!   `Div(diff, std)`.
//!
//! The `Reciprocal` spelling is not optional robustness: this pass runs after
//! [`super::fuse_div_sqrt_to_rsqrt`], which has already rewritten every
//! `Div(1, Sqrt(·))` in the graph by the time control reaches here.
//!
//! # Why this runs *after* `fuse_layer_norm`
//!
//! For a 4-D `[N, C, H, W]` tensor, `axes=[2, 3]` is also the trailing axis run
//! `[-2, -1]`, so a LayerNorm-over-the-last-two-axes chain and a spatial
//! normalisation chain are shape-indistinguishable at the ReduceMean. Running
//! LayerNorm first gives it first refusal — it folds the affine term in, which
//! this pass cannot — and costs nothing here, because `fuse_layer_norm`
//! requires `Pow` and a direct `Div(diff, std)` and therefore provably declines
//! the form the real model has.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::{const_scalar, TensorUsage};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// How the squared deviation is spelled, which decides how many input slots
/// consume `diff`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SquareForm {
    /// `Mul(diff, diff)` — `diff` occupies two of the square node's slots.
    MulSelf,
    /// `Pow(diff, 2)` — one slot.
    Pow,
}

impl SquareForm {
    /// Input slots the square node occupies on `diff`, on top of the one the
    /// normalising node (the final `Mul`/`Div`) takes.
    fn diff_slots(self) -> usize {
        match self {
            Self::MulSelf => 2,
            Self::Pow => 1,
        }
    }
}

/// Check that a `ReduceMean` reduces over exactly the spatial axes
/// (`2..rank`), keeping dimensions so the result broadcasts back over `X`.
///
/// This is the gate that separates a spatial normalisation from a LayerNorm:
/// "the last `k` axes" is not enough, the reduction must start at axis 2 *and*
/// run to the end. Axes may be given as the pre-opset-18 attribute or (opset
/// 18+) as an initialiser in input slot 1, and negative axes are resolved
/// against the input's rank — so a rank the pass cannot establish is a
/// decline, never a guess.
fn reduces_over_spatial_axes(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    known_shapes: &HashMap<String, Vec<usize>>,
) -> bool {
    if node.attrs.i("keepdims", 1) != 1 {
        return false;
    }

    let attr_axes = node.attrs.ints("axes");
    let raw: Vec<i64> = if !attr_axes.is_empty() {
        attr_axes.to_vec()
    } else {
        // opset 18+: `axes` moved from an attribute to input slot 1. An absent
        // axes list means "reduce everything", which is not this pattern.
        let axes_name = match node.inputs.get(1) {
            Some(name) if !name.is_empty() => name,
            _ => return false,
        };
        match weights.get(axes_name) {
            Some(t) if !t.data.is_empty() => t.data.iter().map(|&v| v as i64).collect(),
            _ => return false,
        }
    };

    let rank = match node.inputs.first().and_then(|x_name| {
        known_shapes
            .get(x_name)
            .map(|s| s.len())
            .or_else(|| weights.get(x_name).map(|t| t.ndim()))
    }) {
        Some(r) => r,
        None => return false,
    };
    // [N, C, spatial…]: at least one spatial axis must exist.
    if rank < 3 {
        return false;
    }
    let rank_i64 = match i64::try_from(rank) {
        Ok(r) => r,
        Err(_) => return false,
    };

    let mut resolved: Vec<i64> = Vec::with_capacity(raw.len());
    for axis in raw {
        let normalized = if axis < 0 { axis + rank_i64 } else { axis };
        if normalized < 0 || normalized >= rank_i64 {
            return false;
        }
        resolved.push(normalized);
    }
    resolved.sort_unstable();
    resolved.dedup();
    let expected: Vec<i64> = (2..rank_i64).collect();
    resolved == expected
}

/// Resolve `Add(var, eps)` / `Add(eps, var)` into `(var_tensor_name, eps)`.
///
/// `eps` must be a compile-time scalar; its magnitude is deliberately *not*
/// range-checked, because `(x - mean) / sqrt(var + k)` is what the fused op
/// computes for any `k` the graph supplies. Negative `k` is refused: that is
/// not a normalisation, and declining beats reasoning about which side
/// produces `NaN` first.
fn resolve_epsilon(add: &Node, weights: &HashMap<String, Tensor>) -> Option<(String, f32)> {
    if add.inputs.len() < 2 {
        return None;
    }
    let lhs = &add.inputs[0];
    let rhs = &add.inputs[1];
    let usable = |v: f32| v.is_finite() && v >= 0.0;
    if let Some(eps) = const_scalar(weights, rhs).filter(|&v| usable(v)) {
        // `var` must be a computed tensor, not a second constant.
        if const_scalar(weights, lhs).is_none() {
            return Some((lhs.clone(), eps));
        }
    }
    if let Some(eps) = const_scalar(weights, lhs).filter(|&v| usable(v)) {
        if const_scalar(weights, rhs).is_none() {
            return Some((rhs.clone(), eps));
        }
    }
    None
}

/// One matched chain, ready to be rewritten.
struct Match {
    /// Every node index the fused op subsumes.
    consumed: Vec<usize>,
    /// The chain's input tensor (`X`).
    x_name: String,
    /// The normalised tensor the chain produced; the fused node keeps it.
    out_name: String,
    epsilon: f32,
}

/// Fuse the decomposed spatial-normalization chain into a single
/// `OxiInstanceNorm` node.
///
/// See the module docs for the pattern, the variants covered, and the ordering
/// constraint against [`super::fuse_layer_norm`].
///
/// Every intermediate the rewrite deletes must be consumed exactly by the
/// nodes the pattern accounts for and must not be a declared graph output; the
/// chain's input `X` and its normalised output keep their names, so anything
/// outside the chain is untouched.
///
/// `OxiInstanceNorm` is an optimizer-generated op: callers must only run this
/// pass when the active registry provides a kernel for it (see
/// [`crate::optimizer::optimize_with_input_shapes`]).
pub fn fuse_instance_norm(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
    known_shapes: &HashMap<String, Vec<usize>>,
    output_names: &[String],
) -> Vec<Node> {
    // mean, sub, square, var, add(eps), sqrt, normalize — the shortest form.
    if nodes.len() < 7 {
        return nodes;
    }

    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            if !out.is_empty() {
                producer.insert(out.as_str(), i);
            }
        }
    }
    let usage = TensorUsage::new(&nodes, output_names);

    // Every index any chain has taken, anchors included: a node already
    // rewritten (or deleted) must not be re-read by a second match.
    let mut claimed: HashSet<usize> = HashSet::new();
    // The subset that is actually dropped from the output — `claimed` minus
    // each chain's anchor, which survives carrying the fused node.
    let mut skip: HashSet<usize> = HashSet::new();
    let mut replacements: HashMap<usize, Node> = HashMap::new();

    for i in 0..nodes.len() {
        if claimed.contains(&i) {
            continue;
        }
        let matched = match_chain(&nodes, i, &producer, &usage, weights, known_shapes);
        let matched = match matched {
            Some(m) if m.consumed.iter().all(|idx| !claimed.contains(idx)) => m,
            _ => continue,
        };

        let anchor = match matched.consumed.iter().min() {
            Some(&idx) => idx,
            None => continue,
        };

        let mut attrs = Attributes::default();
        attrs.floats.insert("epsilon".to_string(), matched.epsilon);
        let fused = Node {
            op: OpKind::OxiInstanceNorm,
            name: format!("{}_fused_instancenorm", nodes[anchor].name),
            inputs: vec![matched.x_name],
            outputs: vec![matched.out_name],
            attrs,
        };

        for idx in matched.consumed {
            claimed.insert(idx);
            if idx != anchor {
                skip.insert(idx);
            }
        }
        replacements.insert(anchor, fused);
    }

    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(i, n)| replacements.remove(&i).unwrap_or(n))
        .collect()
}

/// Try to read the whole chain off the node at `norm_idx`, which must be the
/// node producing the normalised output (`Mul(diff, rstd)` or
/// `Div(diff, std)`).
fn match_chain(
    nodes: &[Node],
    norm_idx: usize,
    producer: &HashMap<&str, usize>,
    usage: &TensorUsage,
    weights: &HashMap<String, Tensor>,
    known_shapes: &HashMap<String, Vec<usize>>,
) -> Option<Match> {
    let norm = &nodes[norm_idx];
    if norm.inputs.len() < 2 || norm.outputs.is_empty() {
        return None;
    }
    let out_name = norm.outputs.first().filter(|n| !n.is_empty())?.clone();

    // `consumed` collects every node index the fused op replaces.
    let mut consumed = vec![norm_idx];

    // ── The normalising node ────────────────────────────────────────────────
    // Either `Mul(diff, rstd)` with `rstd` from a reciprocal, or the direct
    // `Div(diff, std)`.
    let (diff_name, std_name) = match norm.op {
        OpKind::Mul => {
            // Whichever operand comes from a reciprocal node is `rstd`; the
            // other is `diff`.
            let (diff, recip_out) = reciprocal_operand(nodes, norm, producer)?;
            let recip_idx = *producer.get(recip_out.as_str())?;
            if !usage.is_fusable_intermediate(&recip_out) {
                return None;
            }
            let recip = &nodes[recip_idx];
            let std = match recip.op {
                // `Reciprocal(std)` — what `fuse_div_sqrt_to_rsqrt` leaves.
                OpKind::Reciprocal => recip.inputs.first()?.clone(),
                // `Div(one, std)` — the raw exporter form.
                OpKind::Div => {
                    if recip.inputs.len() < 2 {
                        return None;
                    }
                    let one = const_scalar(weights, &recip.inputs[0])?;
                    if (one - 1.0).abs() > 1e-6 {
                        return None;
                    }
                    recip.inputs[1].clone()
                }
                _ => return None,
            };
            consumed.push(recip_idx);
            (diff, std)
        }
        OpKind::Div => (norm.inputs[0].clone(), norm.inputs[1].clone()),
        _ => return None,
    };

    // ── Sqrt(var + eps) ─────────────────────────────────────────────────────
    if !usage.is_fusable_intermediate(&std_name) {
        return None;
    }
    let sqrt_idx = *producer.get(std_name.as_str())?;
    if !matches!(nodes[sqrt_idx].op, OpKind::Sqrt) {
        return None;
    }
    let vare_name = nodes[sqrt_idx].inputs.first()?.clone();
    consumed.push(sqrt_idx);

    // ── Add(var, eps) ───────────────────────────────────────────────────────
    if !usage.is_fusable_intermediate(&vare_name) {
        return None;
    }
    let add_idx = *producer.get(vare_name.as_str())?;
    if !matches!(nodes[add_idx].op, OpKind::Add) {
        return None;
    }
    let (var_name, epsilon) = resolve_epsilon(&nodes[add_idx], weights)?;
    consumed.push(add_idx);

    // ── ReduceMean(sq) over the spatial axes ────────────────────────────────
    if !usage.is_fusable_intermediate(&var_name) {
        return None;
    }
    let var_reduce_idx = *producer.get(var_name.as_str())?;
    if !matches!(nodes[var_reduce_idx].op, OpKind::ReduceMean) {
        return None;
    }
    if !reduces_over_spatial_axes(&nodes[var_reduce_idx], weights, known_shapes) {
        return None;
    }
    let sq_name = nodes[var_reduce_idx].inputs.first()?.clone();
    consumed.push(var_reduce_idx);

    // ── The squared deviation ───────────────────────────────────────────────
    if !usage.is_fusable_intermediate(&sq_name) {
        return None;
    }
    let square_idx = *producer.get(sq_name.as_str())?;
    let square = &nodes[square_idx];
    if square.inputs.len() < 2 {
        return None;
    }
    let square_form = match square.op {
        // `Mul(diff, diff)`: both operands must be the *same* tensor, else this
        // is some unrelated product.
        OpKind::Mul if square.inputs[0] == diff_name && square.inputs[1] == diff_name => {
            SquareForm::MulSelf
        }
        OpKind::Pow if square.inputs[0] == diff_name => {
            let exponent = const_scalar(weights, &square.inputs[1])?;
            if (exponent - 2.0).abs() > 1e-6 {
                return None;
            }
            SquareForm::Pow
        }
        _ => return None,
    };
    consumed.push(square_idx);

    // ── Sub(X, mean) ────────────────────────────────────────────────────────
    // `diff` feeds exactly the square node and the normalising node and
    // nothing else — `TensorUsage` counts input *slots*, so `Mul(diff, diff)`
    // contributes two.
    let expected_diff_slots = square_form.diff_slots() + 1;
    if usage.consumers(&diff_name) != expected_diff_slots || usage.is_graph_output(&diff_name) {
        return None;
    }
    let sub_idx = *producer.get(diff_name.as_str())?;
    if !matches!(nodes[sub_idx].op, OpKind::Sub) {
        return None;
    }
    if nodes[sub_idx].inputs.len() < 2 {
        return None;
    }
    let x_name = nodes[sub_idx].inputs[0].clone();
    let mean_name = nodes[sub_idx].inputs[1].clone();
    consumed.push(sub_idx);

    // ── ReduceMean(X) over the spatial axes ─────────────────────────────────
    if !usage.is_fusable_intermediate(&mean_name) {
        return None;
    }
    let mean_reduce_idx = *producer.get(mean_name.as_str())?;
    if !matches!(nodes[mean_reduce_idx].op, OpKind::ReduceMean) {
        return None;
    }
    if nodes[mean_reduce_idx].inputs.first() != Some(&x_name) {
        return None;
    }
    if !reduces_over_spatial_axes(&nodes[mean_reduce_idx], weights, known_shapes) {
        return None;
    }
    consumed.push(mean_reduce_idx);

    // No node may be claimed twice within one chain (a self-referential graph
    // would otherwise make the rewrite drop a node it still needs).
    let unique: HashSet<usize> = consumed.iter().copied().collect();
    if unique.len() != consumed.len() {
        return None;
    }

    Some(Match {
        consumed,
        x_name,
        out_name,
        epsilon,
    })
}

/// For a `Mul` node, split its two operands into `(diff, reciprocal_output)`.
///
/// Returns `None` unless exactly one operand is produced by a `Reciprocal` or
/// `Div` node — if both are, the pattern is ambiguous and the pass declines.
fn reciprocal_operand(
    nodes: &[Node],
    mul: &Node,
    producer: &HashMap<&str, usize>,
) -> Option<(String, String)> {
    let is_reciprocal = |name: &String| -> bool {
        producer
            .get(name.as_str())
            .is_some_and(|&idx| matches!(nodes[idx].op, OpKind::Reciprocal | OpKind::Div))
    };
    let lhs = &mul.inputs[0];
    let rhs = &mul.inputs[1];
    match (is_reciprocal(lhs), is_reciprocal(rhs)) {
        (false, true) => Some((lhs.clone(), rhs.clone())),
        (true, false) => Some((rhs.clone(), lhs.clone())),
        _ => None,
    }
}

#[cfg(test)]
#[path = "instance_norm_tests.rs"]
mod tests;
