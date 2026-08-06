//! Shape inference for construction, pooling, padding, and resize operators.

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::spatial_attrs::{
    pool_out_dim, read_kernel_shape, read_pads, read_positive_spatial, resolve_pads,
};
use crate::optimizer::shape_inference::get_input_shape;

/// ConstantOfShape: output shape from the input tensor's data values.
pub(super) fn infer_constant_of_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    // The input is a 1-D tensor whose values define the output shape
    let shape_name = node.inputs.first()?;
    if shape_name.is_empty() {
        return None;
    }

    // Try from weights first
    if let Some(shape_tensor) = weights.get(shape_name) {
        let out: Vec<usize> = shape_tensor.data.iter().map(|&v| v as usize).collect();
        return Some(vec![out]);
    }

    // If shape data is known as a shape (e.g., it's a 1-D tensor of known size)
    // we can't determine the actual values without the data
    let _shape = known.get(shape_name)?;
    None
}

/// GlobalAveragePool / GlobalMaxPool: [N, C, ...] -> [N, C, 1, 1, ...]
pub(super) fn infer_global_pool_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.len() < 2 {
        return None;
    }

    let mut out = vec![input_shape[0], input_shape[1]];
    out.extend(std::iter::repeat(1).take(input_shape.len() - 2));
    Some(vec![out])
}

/// AveragePool / MaxPool shape inference.
///
/// Mirrors `PoolGeometry::from_attrs` in
/// `oxionnx-ops/src/registry/conv_ops/pooling.rs`: `auto_pad` is resolved
/// exactly as the kernel resolves it (previously ignored outright, so a
/// `SAME_UPPER` pool was predicted at its *unpadded* extent), and `ceil_mode`
/// carries the ONNX-Runtime correction that drops a trailing window starting
/// inside the right-hand padding.
///
/// # Two outputs for `MaxPool` with `Indices`
///
/// `infer_shapes` zips the returned shapes positionally onto `node.outputs`,
/// and the slot fast path in `run/dispatch.rs` requires *every* non-elided
/// output to have a resolved shape. Returning one shape for a two-output
/// `MaxPool` therefore silently forced the node onto the allocating `execute`
/// path forever. The `Indices` output has the same shape as the values output.
pub(super) fn infer_pool_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.len() < 3 {
        return None;
    }

    let n = input_shape[0];
    let c = input_shape[1];
    let spatial = &input_shape[2..];
    let spatial_dims = spatial.len();

    let kernel_shape = read_kernel_shape(node.attrs.ints("kernel_shape"), spatial_dims)?;
    let strides = read_positive_spatial(node.attrs.ints("strides"), spatial_dims, 1)?;
    let dilations = read_positive_spatial(node.attrs.ints("dilations"), spatial_dims, 1)?;
    let explicit_pads = read_pads(node.attrs.ints("pads"), spatial_dims)?;
    let pads = resolve_pads(
        node.attrs.s("auto_pad"),
        spatial,
        &kernel_shape,
        &strides,
        &dilations,
        &explicit_pads,
    )?;

    let ceil_mode = node.attrs.i("ceil_mode", 0) != 0;

    let mut out_shape = Vec::with_capacity(2 + spatial_dims);
    out_shape.push(n);
    out_shape.push(c);
    for d in 0..spatial_dims {
        out_shape.push(pool_out_dim(
            spatial[d],
            pads[d],
            pads[d + spatial_dims],
            kernel_shape[d],
            dilations[d],
            strides[d],
            ceil_mode,
        )?);
    }

    // Only `MaxPool` defines a second (`Indices`) output; `AveragePool` has
    // exactly one, so a stray second output name there is malformed and must
    // not be given a shape (that would put the node on the slot path with a
    // slot the operator never writes).
    let has_indices = node.op == OpKind::MaxPool
        && node
            .outputs
            .get(1)
            .is_some_and(|name: &String| !name.is_empty());
    if has_indices {
        Some(vec![out_shape.clone(), out_shape])
    } else {
        Some(vec![out_shape])
    }
}

/// Expand: broadcast input to target shape.
pub(super) fn infer_expand_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let shape_name = node.inputs.get(1)?;
    if shape_name.is_empty() {
        return None;
    }

    if let Some(shape_tensor) = weights.get(shape_name) {
        let target: Vec<usize> = shape_tensor.data.iter().map(|&v| v as usize).collect();
        let out = Tensor::broadcast_shape(&input_shape, &target).ok()?;
        return Some(vec![out]);
    }

    None
}

/// Tile: repeat input along each axis.
pub(super) fn infer_tile_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let repeats_name = node.inputs.get(1)?;
    if repeats_name.is_empty() {
        return None;
    }

    if let Some(repeats_tensor) = weights.get(repeats_name) {
        let repeats: Vec<usize> = repeats_tensor.data.iter().map(|&v| v as usize).collect();
        if repeats.len() != input_shape.len() {
            return None;
        }
        let out: Vec<usize> = input_shape
            .iter()
            .zip(repeats.iter())
            .map(|(&d, &r)| d * r)
            .collect();
        return Some(vec![out]);
    }

    None
}

/// Pad: add padding to spatial dims.
pub(super) fn infer_pad_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;

    // pads input: [begin_0, begin_1, ..., end_0, end_1, ...]
    let pads_name = node.inputs.get(1)?;
    if pads_name.is_empty() {
        return None;
    }

    if let Some(pads_tensor) = weights.get(pads_name) {
        let pads: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
        let rank = input_shape.len();
        if pads.len() != rank * 2 {
            return None;
        }

        let mut out = Vec::with_capacity(rank);
        for i in 0..rank {
            let padded = input_shape[i] as i64 + pads[i] + pads[i + rank];
            if padded < 0 {
                return None;
            }
            out.push(padded as usize);
        }
        return Some(vec![out]);
    }

    None
}

/// Look up a Resize tensor input that is a constant initializer.
///
/// Mirrors the operator's `non_empty` helper: ONNX lets a trailing optional
/// input be passed as an *empty* tensor when a later one is used, so a
/// zero-element tensor counts as absent, not as "resize to nothing".
fn constant_input<'a>(
    node: &Node,
    idx: usize,
    weights: &'a HashMap<String, Tensor>,
) -> Option<&'a Tensor> {
    let name = node.inputs.get(idx)?;
    if name.is_empty() {
        return None;
    }
    weights.get(name).filter(|t| !t.data.is_empty())
}

/// Resize: infer from the constant `sizes` or `scales` input.
///
/// Mirrors `ResizeOp` + `oxionnx_ops::resize::plan`:
///
/// * `keep_aspect_ratio_policy` is honoured. It used to be ignored entirely, so
///   `sizes = [1,1,6,6]` on a `[1,1,4,5]` input with `not_larger` was predicted
///   as `[1,1,6,6]` while the operator (and the ONNX reference) produce
///   `[1,1,4,5]`; `not_smaller` produces `[1,1,6,8]`.
/// * The opset-10 two-input `(X, scales)` layout is disambiguated the same way
///   the operator does — by length, since `roi` carries `2 * rank` values.
/// * `scales` uses `floor(dim * scale)` evaluated in **f32**, matching the
///   operator (and onnxruntime); f64 would diverge on an exact tie.
///
/// Returns `None` — no prediction — for the cases the operator rejects or that
/// need per-axis information this pass does not model (`axes`, both `scales`
/// and `sizes` supplied, a non-finite or non-positive scale).
pub(super) fn infer_resize_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len();

    // The `axes` attribute resizes a subset of the dimensions (with negative
    // indices allowed). Declining is correct and cheap; guessing is not.
    if !node.attrs.ints("axes").is_empty() {
        return None;
    }

    // inputs: X, roi, scales, sizes (opset 11+); X, scales (opset 10).
    let roi = constant_input(node, 1, weights);
    let mut scales = constant_input(node, 2, weights);
    let sizes = constant_input(node, 3, weights);
    if scales.is_none() && sizes.is_none() {
        if let Some(candidate) = roi {
            if node.inputs.len() <= 2 && candidate.data.len() == rank {
                scales = Some(candidate);
            }
        }
    }

    match (scales, sizes) {
        // The operator rejects both being supplied at once.
        (Some(_), Some(_)) => None,
        (None, None) => None,
        (Some(scales), None) => {
            if scales.data.len() != rank {
                return None;
            }
            let mut out = Vec::with_capacity(rank);
            for (&dim, &s) in input_shape.iter().zip(scales.data.iter()) {
                if !s.is_finite() || s <= 0.0 {
                    return None;
                }
                let width = (dim as f32) * s;
                let floored = width.floor();
                if !(0.0..=usize::MAX as f32).contains(&floored) {
                    return None;
                }
                out.push(floored as usize);
            }
            Some(vec![out])
        }
        (None, Some(sizes)) => {
            if sizes.data.len() != rank {
                return None;
            }
            let mut requested = Vec::with_capacity(rank);
            for &v in &sizes.data {
                if !v.is_finite() || v < 0.0 || v > usize::MAX as f32 {
                    return None;
                }
                requested.push(v as usize);
            }
            apply_keep_aspect_ratio(
                node.attrs.s("keep_aspect_ratio_policy"),
                &input_shape,
                requested,
            )
            .map(|out| vec![out])
        }
    }
}

/// Apply `keep_aspect_ratio_policy` to a requested `sizes` vector.
///
/// Mirrors `apply_sizes` in `oxionnx-ops/src/resize.rs`: `not_larger` takes the
/// smallest per-axis ratio and `not_smaller` the largest, then every axis is
/// rescaled by that common ratio and rounded with halfway cases going *up*.
/// The ratio and the product are evaluated in f32 exactly as the operator does.
fn apply_keep_aspect_ratio(
    policy: &str,
    input_shape: &[usize],
    requested: Vec<usize>,
) -> Option<Vec<usize>> {
    match policy {
        "" | "stretch" => Some(requested),
        "not_larger" | "not_smaller" => {
            let mut common: Option<f32> = None;
            for (&dim, &want) in input_shape.iter().zip(requested.iter()) {
                if dim == 0 {
                    // The operator reports a ShapeMismatch here.
                    return None;
                }
                let ratio = want as f32 / dim as f32;
                common = Some(match (common, policy) {
                    (None, _) => ratio,
                    (Some(cur), "not_larger") => cur.min(ratio),
                    (Some(cur), _) => cur.max(ratio),
                });
            }
            let k = common?;
            let mut out = Vec::with_capacity(input_shape.len());
            for &dim in input_shape {
                // Spec: round_int rounds halfway cases up.
                let scaled = (k * dim as f32 + 0.5).floor();
                if !(0.0..=usize::MAX as f32).contains(&scaled) {
                    return None;
                }
                out.push(scaled as usize);
            }
            Some(out)
        }
        // Unknown policy: the operator errors, so refuse to guess.
        _ => None,
    }
}
