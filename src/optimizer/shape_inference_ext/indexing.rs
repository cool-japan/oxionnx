//! Shape inference for indexing and selection operators.
//!
//! Covers GatherND, OneHot, and TopK.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

use crate::optimizer::shape_inference::get_input_shape;

/// GatherND shape inference.
pub(super) fn infer_gather_nd_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let data_shape = get_input_shape(node, 0, known)?;
    let indices_shape = get_input_shape(node, 1, known)?;
    let batch_dims = node.attrs.i("batch_dims", 0) as usize;

    if indices_shape.is_empty() {
        return None;
    }

    let last_idx_dim = *indices_shape.last()?;
    let data_rank = data_shape.len();

    if batch_dims + last_idx_dim > data_rank {
        return None;
    }

    // output shape = indices_shape[:-1] + data_shape[batch_dims + last_idx_dim:]
    let mut out = Vec::new();
    // batch dims from indices
    out.extend_from_slice(&indices_shape[..indices_shape.len() - 1]);
    // remaining data dims
    out.extend_from_slice(&data_shape[batch_dims + last_idx_dim..]);

    Some(vec![out])
}

/// OneHot: indices_shape + [depth]
pub(super) fn infer_onehot_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let indices_shape = get_input_shape(node, 0, known)?;
    let axis = node.attrs.i("axis", -1);

    // depth is input[1]
    let depth_name = node.inputs.get(1)?;
    if depth_name.is_empty() {
        return None;
    }

    let depth = {
        let depth_tensor = weights.get(depth_name)?;
        if depth_tensor.data.is_empty() {
            return None;
        }
        depth_tensor.data[0] as usize
    };

    let rank = indices_shape.len() as i64 + 1;
    let norm_axis = if axis < 0 {
        (axis + rank) as usize
    } else {
        axis as usize
    };

    let mut out = indices_shape;
    if norm_axis > out.len() {
        return None;
    }
    out.insert(norm_axis, depth);

    Some(vec![out])
}

/// TopK: two outputs with the k-dim replaced.
pub(super) fn infer_topk_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;
    let axis_raw = node.attrs.i("axis", -1);
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };

    if axis >= input_shape.len() {
        return None;
    }

    // k comes from input[1]
    let k_name = node.inputs.get(1)?;
    if k_name.is_empty() {
        return None;
    }

    let k = {
        let k_tensor = weights.get(k_name)?;
        if k_tensor.data.is_empty() {
            return None;
        }
        k_tensor.data[0] as usize
    };

    let mut out = input_shape;
    out[axis] = k;
    // TopK has two outputs: values and indices, both same shape
    Some(vec![out.clone(), out])
}
