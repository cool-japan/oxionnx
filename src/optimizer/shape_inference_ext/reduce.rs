//! Shape inference for reduction operators (ReduceMean, ArgMax, etc.).

use crate::graph::Node;
use std::collections::HashMap;

use crate::optimizer::shape_inference::get_input_shape;

/// Reduce ops: apply axes reduction with keepdims.
pub(super) fn infer_reduce_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;
    let keepdims = node.attrs.i("keepdims", 1) != 0;

    let axes_attr: Vec<i64> = node.attrs.ints("axes").to_vec();
    let axes: Vec<usize> = if axes_attr.is_empty() {
        // Default: reduce all axes
        (0..input_shape.len()).collect()
    } else {
        axes_attr
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + rank) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    };

    let mut out = Vec::new();
    for (i, &dim) in input_shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }

    Some(vec![out])
}

/// ArgMax/ArgMin: reduce one axis, output index along that axis.
pub(super) fn infer_arg_reduce_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rank = input_shape.len() as i64;
    let keepdims = node.attrs.i("keepdims", 1) != 0;
    let axis_raw = node.attrs.i("axis", 0);
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };

    if axis >= input_shape.len() {
        return None;
    }

    let mut out = Vec::new();
    for (i, &dim) in input_shape.iter().enumerate() {
        if i == axis {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }

    Some(vec![out])
}
