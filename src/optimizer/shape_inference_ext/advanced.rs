//! Shape inference for advanced operators.
//!
//! Covers ConvTranspose, Einsum, LSTM, GRU, LinearClassifier, and LinearRegressor.

use crate::graph::Node;
use std::collections::HashMap;

use super::spatial_attrs::{conv_transpose_out_dim, read_pads, read_positive_spatial};
use crate::optimizer::shape_inference::get_input_shape;

/// ConvTranspose shape inference.
///
/// Mirrors `ConvTransposeGeometry::from_attrs` in
/// `oxionnx-ops/src/registry/conv_ops/conv.rs`, which resolves the three-way
/// interaction the ONNX spec defines between `pads`, `auto_pad` and
/// `output_shape`:
///
/// * `output_shape` wins outright — the operator *derives* the padding from it,
///   so the produced extent is the requested one verbatim. Both the 2-entry
///   (spatial) and 4-entry (full `[N, C, oH, oW]`) exporter forms are accepted;
///   only the 2-entry form used to be recognised here, so a full-form
///   `output_shape` fell through to the natural formula and predicted a
///   different extent than the operator produced.
/// * `auto_pad = SAME_UPPER / SAME_LOWER` without `output_shape` targets
///   `out = in * stride` (previously ignored entirely).
/// * `auto_pad = VALID` forces zero padding; `NOTSET` uses `pads` verbatim.
pub(super) fn infer_conv_transpose_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let weight_shape = get_input_shape(node, 1, known)?;

    if input_shape.len() < 3 || weight_shape.len() < 3 {
        return None;
    }

    let n = input_shape[0];
    // For ConvTranspose, weight is [C_in, C_out/group, kH, kW, ...]
    let group = node.attrs.i("group", 1);
    if group < 1 {
        return None;
    }
    let c_out = weight_shape[1].checked_mul(usize::try_from(group).ok()?)?;
    let spatial = &input_shape[2..];
    let spatial_dims = spatial.len();

    let auto_pad = node.attrs.s("auto_pad");
    if !matches!(
        auto_pad,
        "" | "NOTSET" | "VALID" | "SAME_UPPER" | "SAME_LOWER"
    ) {
        // Unknown auto_pad: the operator errors, so refuse to guess a shape.
        return None;
    }

    // `output_shape` is authoritative when present: the operator back-solves
    // the padding from it, so the output extent *is* the requested one.
    let requested_out = read_output_shape_attr(node.attrs.ints("output_shape"), spatial_dims)?;
    if let Some(target) = requested_out {
        let mut out = Vec::with_capacity(2 + spatial_dims);
        out.push(n);
        out.push(c_out);
        out.extend_from_slice(&target);
        return Some(vec![out]);
    }

    let strides = read_positive_spatial(node.attrs.ints("strides"), spatial_dims, 1)?;

    // SAME_UPPER / SAME_LOWER without `output_shape` targets `in * stride`.
    if matches!(auto_pad, "SAME_UPPER" | "SAME_LOWER") {
        let mut out = Vec::with_capacity(2 + spatial_dims);
        out.push(n);
        out.push(c_out);
        for d in 0..spatial_dims {
            out.push(spatial[d].checked_mul(strides[d])?);
        }
        return Some(vec![out]);
    }

    let kernel_shape_attr = node.attrs.ints("kernel_shape");
    let kernel_shape: Vec<usize> = if kernel_shape_attr.is_empty() {
        weight_shape[2..].to_vec()
    } else {
        read_positive_spatial(kernel_shape_attr, spatial_dims, 1)?
    };
    if kernel_shape.len() < spatial_dims {
        return None;
    }

    let dilations = read_positive_spatial(node.attrs.ints("dilations"), spatial_dims, 1)?;
    let output_padding = read_pads(node.attrs.ints("output_padding"), spatial_dims)?;
    // VALID forces zero padding; NOTSET reads `pads` verbatim.
    let pads = if auto_pad == "VALID" {
        vec![0_usize; spatial_dims * 2]
    } else {
        read_pads(node.attrs.ints("pads"), spatial_dims)?
    };

    let mut out_shape = Vec::with_capacity(2 + spatial_dims);
    out_shape.push(n);
    out_shape.push(c_out);
    for d in 0..spatial_dims {
        out_shape.push(conv_transpose_out_dim(
            spatial[d],
            strides[d],
            output_padding[d],
            kernel_shape[d],
            dilations[d],
            pads[d],
            pads[d + spatial_dims],
        )?);
    }

    Some(vec![out_shape])
}

/// Read the optional `output_shape` attribute as the spatial extents.
///
/// `Ok(None)` (as `Some(None)`) when the attribute is absent; the whole
/// inference declines (`None`) when it is present but malformed, exactly as the
/// operator's `read_output_shape_attr` errors.
#[allow(clippy::option_option)]
fn read_output_shape_attr(values: &[i64], spatial_dims: usize) -> Option<Option<Vec<usize>>> {
    let spatial: &[i64] = if values.is_empty() {
        return Some(None);
    } else if values.len() == spatial_dims {
        values
    } else if values.len() == spatial_dims + 2 {
        // Full `[N, C, oH, oW]` form emitted by several exporters.
        &values[2..]
    } else {
        return None;
    };
    let mut out = Vec::with_capacity(spatial_dims);
    for &v in spatial {
        if v < 1 {
            return None;
        }
        out.push(usize::try_from(v).ok()?);
    }
    Some(Some(out))
}

/// One token of an einsum subscript: a named label, or the ellipsis placeholder.
///
/// Mirrors `oxionnx_ops::einsum::parse::Token`.  The two must agree, because a
/// shape this module publishes is what sizes the pre-allocated output slot the
/// executor writes into (`Session::acquire_output_slots`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum EinsumToken {
    Named(u8),
    Ellipsis,
}

/// Tokenize one einsum subscript, declining (`None`) on anything malformed.
///
/// Shape inference is advisory, so a malformed equation is *not* an error here:
/// it simply yields no inference and the executor raises the real diagnostic.
fn tokenize_einsum(spec: &str) -> Option<Vec<EinsumToken>> {
    let bytes = spec.as_bytes();
    let mut tokens = Vec::with_capacity(bytes.len());
    let mut seen_ellipsis = false;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' {
            // A lone or doubled '.' is malformed, and a second '...' in one
            // subscript is rejected by the executor too.
            if seen_ellipsis || i + 2 >= bytes.len() || bytes[i + 1] != b'.' || bytes[i + 2] != b'.'
            {
                return None;
            }
            seen_ellipsis = true;
            tokens.push(EinsumToken::Ellipsis);
            i += 3;
        } else if b.is_ascii_alphabetic() {
            tokens.push(EinsumToken::Named(b));
            i += 1;
        } else {
            return None;
        }
    }
    Some(tokens)
}

/// Merge `dim` into `slot` under numpy's **cross-operand** broadcasting rule:
/// an extent of 1 loses to any other extent, and two different non-1 extents
/// are irreconcilable.
///
/// Returns `false` when the extents cannot be broadcast, which declines the
/// whole inference.
fn merge_einsum_extent(slot: &mut Option<usize>, dim: usize) -> bool {
    match *slot {
        None => *slot = Some(dim),
        Some(current) if current == dim => {}
        Some(1) => *slot = Some(dim),
        Some(_) if dim == 1 => {}
        Some(_) => return false,
    }
    true
}

/// Einsum: parse the equation and resolve the output shape.
///
/// # Why this mirrors the executor rather than approximating it
///
/// The published shape is not merely informational: `dispatch_node` uses it to
/// pre-size the output slot an `Einsum` node writes into, and `write_node_outputs`
/// validates a provider's result against it.  A shape that disagrees with what
/// `oxionnx_ops::einsum` actually produces is therefore a real defect — it is
/// only *masked* today because `EinsumOp::execute_into_slots` replaces the slot
/// wholesale (`*out = result`) when the shapes disagree, so a stale inference
/// self-heals at the cost of the allocation the slot path exists to avoid.
///
/// Two behaviours this used to get wrong, both now aligned with
/// `oxionnx_ops::einsum::parse::parse_equation` (which is itself pinned against
/// numpy):
///
/// * **Broadcasting.** Label extents were first-writer-wins, so `"ij,ij->ij"`
///   over `(2, 1)` and `(2, 3)` inferred `[2, 1]` while the executor (and numpy)
///   produce `[2, 3]`.  Extent 1 now loses to extent *n*, and a genuine conflict
///   declines the inference instead of publishing a wrong shape.  A label
///   repeated **within one operand** is a diagonal and still demands exactly
///   equal extents — numpy rejects `"ii"` on a `(1, 3)` operand rather than
///   broadcasting it.
/// * **Ellipsis.** `'.'` bytes were silently filtered out of both the input and
///   the output subscripts, so every ellipsis equation fell out at the
///   "labels vs rank" guard and inferred nothing.  Ellipsis axes are now bound
///   per operand (right-aligned, broadcast against one another) and spliced back
///   in at the `...` position of the output.
///
/// Implicit output mode (no `->`) is supported for the same reason: the executor
/// supports it, and the output is the ellipsis axes followed by every named
/// label occurring exactly once, in ASCII order.
pub(super) fn infer_einsum_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let equation = node.attrs.s("equation");
    if equation.is_empty() {
        return None;
    }

    // Whitespace is not significant anywhere in an einsum equation.
    let eq: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs) = match eq.find("->") {
        // "->" is two ASCII bytes, so `pos + 2` is always a char boundary.
        Some(pos) => (&eq[..pos], Some(eq[pos + 2..].to_string())),
        None => (eq.as_str(), None),
    };

    let input_specs: Vec<&str> = lhs.split(',').collect();
    if input_specs.len() != node.inputs.len() {
        return None;
    }

    // ── Pass 1: tokenize, collect shapes, size the ellipsis ─────────────────
    let mut input_tokens: Vec<Vec<EinsumToken>> = Vec::with_capacity(input_specs.len());
    let mut input_shapes: Vec<Vec<usize>> = Vec::with_capacity(input_specs.len());
    let mut ellipsis_dims: Vec<usize> = vec![0; input_specs.len()];
    let mut num_ellipsis = 0usize;
    for (i, spec) in input_specs.iter().enumerate() {
        let tokens = tokenize_einsum(spec)?;
        let shape = get_input_shape(node, i, known)?;
        let named = tokens
            .iter()
            .filter(|t| matches!(t, EinsumToken::Named(_)))
            .count();
        if tokens.contains(&EinsumToken::Ellipsis) {
            // The ellipsis absorbs exactly the axes the named labels leave over.
            if shape.len() < named {
                return None;
            }
            ellipsis_dims[i] = shape.len() - named;
            num_ellipsis = num_ellipsis.max(ellipsis_dims[i]);
        } else if named != shape.len() {
            return None;
        }
        input_tokens.push(tokens);
        input_shapes.push(shape);
    }

    // ── Pass 2: allocate label indices (ellipsis axes occupy 0..num_ellipsis) ─
    let mut label_map: HashMap<u8, usize> = HashMap::new();
    let mut label_count = num_ellipsis;
    let mut input_subscripts: Vec<Vec<usize>> = Vec::with_capacity(input_tokens.len());
    for (i, tokens) in input_tokens.iter().enumerate() {
        let mut subs: Vec<usize> = Vec::with_capacity(input_shapes[i].len());
        for token in tokens {
            match *token {
                // Right-aligned inside the widest ellipsis: numpy's
                // leading-dimension broadcast.
                EinsumToken::Ellipsis => {
                    subs.extend((num_ellipsis - ellipsis_dims[i])..num_ellipsis)
                }
                EinsumToken::Named(c) => {
                    let idx = *label_map.entry(c).or_insert_with(|| {
                        let v = label_count;
                        label_count += 1;
                        v
                    });
                    subs.push(idx);
                }
            }
        }
        input_subscripts.push(subs);
    }

    // ── Pass 3: resolve every label's extent ────────────────────────────────
    let mut label_sizes: Vec<Option<usize>> = vec![None; label_count];
    // Reused per operand; `None` means "this label is absent from this operand".
    let mut operand_dim: Vec<Option<usize>> = vec![None; label_count];
    for (i, subs) in input_subscripts.iter().enumerate() {
        operand_dim.iter_mut().for_each(|slot| *slot = None);
        for (axis, &label) in subs.iter().enumerate() {
            let dim = *input_shapes[i].get(axis)?;
            match operand_dim[label] {
                // Repeated within one operand: a diagonal, which requires
                // exactly equal extents (numpy does not broadcast here).
                Some(prev) if prev != dim => return None,
                Some(_) => {}
                None => operand_dim[label] = Some(dim),
            }
        }
        for (label, slot) in operand_dim.iter().enumerate() {
            if let Some(dim) = *slot {
                if !merge_einsum_extent(&mut label_sizes[label], dim) {
                    return None;
                }
            }
        }
    }

    // ── Pass 4: output subscript ────────────────────────────────────────────
    let output_labels: Vec<usize> = match rhs {
        Some(ref rhs_str) => {
            let tokens = tokenize_einsum(rhs_str)?;
            // numpy: "output has more dimensions than subscripts given".  The
            // executor raises it, so inference must not invent a shape.
            if num_ellipsis > 0 && !tokens.contains(&EinsumToken::Ellipsis) {
                return None;
            }
            let mut used = vec![false; label_count];
            let mut out: Vec<usize> = Vec::with_capacity(tokens.len());
            for token in &tokens {
                match *token {
                    EinsumToken::Ellipsis => {
                        for (label, slot) in used.iter_mut().enumerate().take(num_ellipsis) {
                            *slot = true;
                            out.push(label);
                        }
                    }
                    EinsumToken::Named(c) => {
                        let label = *label_map.get(&c)?;
                        // A repeated output label is an error, not a shape.
                        if used[label] {
                            return None;
                        }
                        used[label] = true;
                        out.push(label);
                    }
                }
            }
            out
        }
        None => {
            // Implicit mode: ellipsis axes first, then singly-occurring named
            // labels in ASCII order ('Z' before 'a', matching numpy).
            let mut counts = vec![0usize; label_count];
            for subs in &input_subscripts {
                for &label in subs {
                    counts[label] += 1;
                }
            }
            let mut out: Vec<usize> = (0..num_ellipsis).collect();
            let mut named: Vec<(u8, usize)> = label_map.iter().map(|(&c, &l)| (c, l)).collect();
            named.sort_unstable();
            out.extend(
                named
                    .into_iter()
                    .filter(|&(_, label)| counts[label] == 1)
                    .map(|(_, label)| label),
            );
            out
        }
    };

    let mut out = Vec::with_capacity(output_labels.len());
    for label in output_labels {
        // Every allocated label was written by some operand in pass 3, so this
        // is populated; declining rather than defaulting keeps the failure
        // mode "no inference" instead of "a fabricated extent".
        out.push((*label_sizes.get(label)?)?);
    }

    Some(vec![out])
}

/// LSTM shape inference.
/// Inputs: X [seq_len, batch, input_size], W, R, B, sequence_lens, initial_h, initial_c, P
/// Outputs: Y [seq_len, num_directions, batch, hidden_size],
///          Y_h [num_directions, batch, hidden_size],
///          Y_c [num_directions, batch, hidden_size]
pub(super) fn infer_lstm_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let x_shape = get_input_shape(node, 0, known)?;
    let w_shape = get_input_shape(node, 1, known)?;

    if x_shape.len() != 3 || w_shape.len() != 3 {
        return None;
    }

    let seq_len = x_shape[0];
    let batch = x_shape[1];
    let num_directions = w_shape[0];
    // W shape: [num_directions, 4*hidden_size, input_size]
    let hidden_size_x4 = w_shape[1];
    if hidden_size_x4 % 4 != 0 {
        return None;
    }
    let hidden_size = hidden_size_x4 / 4;

    let y = vec![seq_len, num_directions, batch, hidden_size];
    let y_h = vec![num_directions, batch, hidden_size];
    let y_c = vec![num_directions, batch, hidden_size];

    Some(vec![y, y_h, y_c])
}

/// GRU shape inference.
/// Similar to LSTM but W has 3*hidden_size and only two state outputs.
pub(super) fn infer_gru_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let x_shape = get_input_shape(node, 0, known)?;
    let w_shape = get_input_shape(node, 1, known)?;

    if x_shape.len() != 3 || w_shape.len() != 3 {
        return None;
    }

    let seq_len = x_shape[0];
    let batch = x_shape[1];
    let num_directions = w_shape[0];
    // W shape: [num_directions, 3*hidden_size, input_size]
    let hidden_size_x3 = w_shape[1];
    if hidden_size_x3 % 3 != 0 {
        return None;
    }
    let hidden_size = hidden_size_x3 / 3;

    let y = vec![seq_len, num_directions, batch, hidden_size];
    let y_h = vec![num_directions, batch, hidden_size];

    Some(vec![y, y_h])
}

/// Split an ONNX-ML input shape into `(num_samples, num_features)`.
///
/// Mirrors `batch_dims` in `oxionnx-ops/src/ml/shape.rs`. Two points where the
/// planner used to disagree with the operators:
///
/// * a 1-D `[C]` input is **one** sample with `C` features (the ONNX-ML
///   convention), not `C` samples with one feature — the old reading produced
///   `[C]` labels where the operator produces `[1]`;
/// * for rank > 2 the feature count is the product of *all* trailing dims, not
///   just `shape[1]`.
fn ml_batch_dims(input_shape: &[usize]) -> Option<(usize, usize)> {
    match input_shape.len() {
        0 => Some((1, 1)),
        1 => Some((1, input_shape[0])),
        _ => {
            let mut features = 1usize;
            for &dim in &input_shape[1..] {
                features = features.checked_mul(dim)?;
            }
            Some((input_shape[0], features))
        }
    }
}

/// LinearClassifier: output labels `[N]` and scores `[N, num_classes]`.
///
/// Mirrors `linear_classifier` in `oxionnx-ops/src/ml/linear.rs`, including its
/// binary one-vs-rest expansion: a single coefficient row with two declared
/// class labels is scored into a `[-s, s]` pair, so the score tensor has 2
/// columns rather than the 1 the raw coefficient count suggests.
pub(super) fn infer_linear_classifier_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let (n, num_features) = ml_batch_dims(&input_shape)?;
    if num_features == 0 {
        return None;
    }

    let coefficients = node.attrs.float_lists.get("coefficients")?;
    let num_targets = coefficients.len() / num_features;
    // The operator reads `classlabels_ints` only.
    let class_labels = node.attrs.ints("classlabels_ints");
    let num_classes = if class_labels.is_empty() {
        if num_targets == 0 {
            // The operator reports a ShapeMismatch here.
            return None;
        }
        num_targets
    } else {
        class_labels.len()
    };

    let is_binary_ovr = node.attrs.i("multi_class", 0) == 0 && num_targets == 1 && num_classes == 2;
    let score_cols = if is_binary_ovr { 2 } else { num_targets };

    // Two outputs: labels [N], scores [N, score_cols]
    Some(vec![vec![n], vec![n, score_cols]])
}

/// LinearRegressor: output `[N, num_targets]`.
///
/// Mirrors `linear_regressor` in `oxionnx-ops/src/ml/linear.rs`: an explicit
/// `targets > 0` attribute wins, otherwise the target count is inferred from
/// the coefficient count and clamped to at least 1. The old reading preferred
/// the coefficient count over the attribute and could yield 0 columns for a
/// coefficient list shorter than the feature count.
pub(super) fn infer_linear_regressor_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let (n, num_features) = ml_batch_dims(&input_shape)?;
    if num_features == 0 {
        return None;
    }

    let coefficients = node.attrs.float_lists.get("coefficients")?;

    let targets_attr = node.attrs.i("targets", 0);
    let num_targets = if targets_attr > 0 {
        usize::try_from(targets_attr).ok()?
    } else {
        (coefficients.len() / num_features).max(1)
    };

    Some(vec![vec![n, num_targets]])
}
