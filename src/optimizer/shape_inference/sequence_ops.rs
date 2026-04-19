//! Shape inference for concat and split operators.

use crate::graph::Node;
use std::collections::HashMap;

use super::helpers::get_input_shape;

/// Concat: sum along axis dimension.
pub(super) fn infer_concat_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    if node.inputs.is_empty() {
        return None;
    }

    let first_shape = get_input_shape(node, 0, known)?;
    let rank = first_shape.len() as i64;
    let axis_raw = node.attrs.i("axis", 0);
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };

    if axis >= first_shape.len() {
        return None;
    }

    let mut total_axis_dim = first_shape[axis];

    for i in 1..node.inputs.len() {
        let shape = get_input_shape(node, i, known)?;
        if shape.len() != first_shape.len() {
            return None;
        }
        // Check non-axis dims match
        for (d, (&a, &b)) in first_shape.iter().zip(shape.iter()).enumerate() {
            if d != axis && a != b {
                return None;
            }
        }
        total_axis_dim += shape[axis];
    }

    let mut out = first_shape;
    out[axis] = total_axis_dim;
    Some(vec![out])
}

/// Split: divide along axis, compute each output's shape.
pub(super) fn infer_split_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;
    let axis_raw = node.attrs.i("axis", 0);
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };

    if axis >= input_shape.len() {
        return None;
    }

    let num_outputs = node.outputs.len();
    if num_outputs == 0 {
        return None;
    }

    let split_sizes: Vec<i64> = node.attrs.ints("split").to_vec();

    let sizes: Vec<usize> = if split_sizes.is_empty() {
        // Equal split
        let dim = input_shape[axis];
        let chunk = dim / num_outputs;
        let remainder = dim % num_outputs;
        (0..num_outputs)
            .map(|i| if i < remainder { chunk + 1 } else { chunk })
            .collect()
    } else {
        split_sizes.iter().map(|&s| s as usize).collect()
    };

    let mut result = Vec::with_capacity(num_outputs);
    for &sz in &sizes {
        let mut out = input_shape.clone();
        out[axis] = sz;
        result.push(out);
    }

    Some(result)
}
