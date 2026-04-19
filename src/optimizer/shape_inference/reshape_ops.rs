//! Shape inference for reshape, transpose, squeeze, unsqueeze, and flatten operators.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::helpers::get_input_shape;

/// Reshape: compute output shape from constant second input.
/// Resolves -1 dimension using total element count.
pub(super) fn infer_reshape_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let total_elements: usize = input_shape.iter().product();

    // Get the target shape from the second input (must be constant)
    let shape_name = node.inputs.get(1)?;
    if shape_name.is_empty() {
        return None;
    }
    let shape_tensor = weights.get(shape_name)?;

    let mut out_shape: Vec<usize> = Vec::with_capacity(shape_tensor.data.len());
    let mut neg_one_idx: Option<usize> = None;

    for (i, &val) in shape_tensor.data.iter().enumerate() {
        let dim = val as i64;
        if dim == -1 {
            if neg_one_idx.is_some() {
                return None; // Multiple -1 dims not allowed
            }
            neg_one_idx = Some(i);
            out_shape.push(0); // placeholder
        } else if dim == 0 {
            // 0 means "copy from input"
            if i < input_shape.len() {
                out_shape.push(input_shape[i]);
            } else {
                return None;
            }
        } else if dim > 0 {
            out_shape.push(dim as usize);
        } else {
            return None; // Invalid dim
        }
    }

    if let Some(idx) = neg_one_idx {
        let known_product: usize = out_shape
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != idx)
            .map(|(_, &v)| v)
            .product();
        if known_product == 0 {
            return None;
        }
        out_shape[idx] = total_elements / known_product;
    }

    Some(vec![out_shape])
}

/// Transpose: permute input dims by perm attribute (default: reverse).
pub(super) fn infer_transpose_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len();

    let perm: Vec<usize> = if let Some(p) = node.attrs.int_lists.get("perm") {
        if p.is_empty() {
            (0..rank).rev().collect()
        } else {
            p.iter().map(|&v| v as usize).collect()
        }
    } else {
        (0..rank).rev().collect()
    };

    if perm.len() != rank {
        return None;
    }

    // Check for invalid permutation indices
    if perm.iter().any(|&p| p >= rank) {
        return None;
    }

    let out: Vec<usize> = perm.iter().map(|&p| input_shape[p]).collect();

    Some(vec![out])
}

/// Squeeze: remove dims at given axes.
pub(super) fn infer_squeeze_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;

    let axes: Vec<i64> = node.attrs.ints("axes").to_vec();
    if axes.is_empty() {
        // Squeeze all dims of size 1
        let out: Vec<usize> = input_shape.iter().copied().filter(|&d| d != 1).collect();
        return Some(vec![out]);
    }

    let normalized: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (a + rank) as usize
            } else {
                a as usize
            }
        })
        .collect();

    let out: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !normalized.contains(i))
        .map(|(_, &d)| d)
        .collect();

    Some(vec![out])
}

/// Unsqueeze: insert dims at given axes.
pub(super) fn infer_unsqueeze_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let axes: Vec<i64> = node.attrs.ints("axes").to_vec();

    if axes.is_empty() {
        return Some(vec![input_shape]);
    }

    let out_rank = input_shape.len() + axes.len();

    let mut normalized: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (a + out_rank as i64) as usize
            } else {
                a as usize
            }
        })
        .collect();
    normalized.sort();

    let mut out = Vec::with_capacity(out_rank);
    let mut src_idx = 0;
    for i in 0..out_rank {
        if normalized.contains(&i) {
            out.push(1);
        } else if src_idx < input_shape.len() {
            out.push(input_shape[src_idx]);
            src_idx += 1;
        } else {
            return None;
        }
    }

    Some(vec![out])
}

/// Flatten: merge dims before/after axis.
pub(super) fn infer_flatten_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;
    let axis_raw = node.attrs.i("axis", 1);
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };

    if axis > input_shape.len() {
        return None;
    }

    let d0: usize = input_shape[..axis].iter().product();
    let d1: usize = input_shape[axis..].iter().product();

    Some(vec![vec![d0, d1]])
}
