//! Shape inference for construction, pooling, padding, and resize operators.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

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

/// AveragePool / MaxPool shape inference (similar to Conv).
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
    let spatial_dims = input_shape.len() - 2;

    let kernel_shape_attr: Vec<i64> = node.attrs.ints("kernel_shape").to_vec();
    if kernel_shape_attr.is_empty() {
        return None;
    }
    let kernel_shape: Vec<usize> = kernel_shape_attr.iter().map(|&k| k as usize).collect();

    if kernel_shape.len() != spatial_dims {
        return None;
    }

    let strides_attr: Vec<i64> = node.attrs.ints("strides").to_vec();
    let strides: Vec<usize> = if strides_attr.is_empty() {
        vec![1; spatial_dims]
    } else {
        strides_attr.iter().map(|&s| s as usize).collect()
    };

    let dilations_attr: Vec<i64> = node.attrs.ints("dilations").to_vec();
    let dilations: Vec<usize> = if dilations_attr.is_empty() {
        vec![1; spatial_dims]
    } else {
        dilations_attr.iter().map(|&d| d as usize).collect()
    };

    let pads_attr: Vec<i64> = node.attrs.ints("pads").to_vec();
    let pads: Vec<usize> = if pads_attr.is_empty() {
        vec![0; spatial_dims * 2]
    } else {
        pads_attr.iter().map(|&p| p as usize).collect()
    };

    if pads.len() != spatial_dims * 2 {
        return None;
    }

    let ceil_mode = node.attrs.i("ceil_mode", 0) != 0;

    let mut out_shape = vec![n, c];
    for d in 0..spatial_dims {
        let input_dim = input_shape[d + 2];
        let effective_kernel = (kernel_shape[d] - 1) * dilations[d] + 1;
        let padded = input_dim + pads[d] + pads[d + spatial_dims];
        if padded < effective_kernel {
            return None;
        }
        let out_dim = if ceil_mode {
            (padded - effective_kernel).div_ceil(strides[d]) + 1
        } else {
            (padded - effective_kernel) / strides[d] + 1
        };
        out_shape.push(out_dim);
    }

    Some(vec![out_shape])
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

/// Resize: infer from sizes or scales constant input.
pub(super) fn infer_resize_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;

    // inputs: X, roi, scales, sizes (opset 11+)
    // Check sizes input first (index 3)
    if let Some(sizes_name) = node.inputs.get(3) {
        if !sizes_name.is_empty() {
            if let Some(sizes_tensor) = weights.get(sizes_name) {
                let out: Vec<usize> = sizes_tensor.data.iter().map(|&v| v as usize).collect();
                if out.len() == input_shape.len() {
                    return Some(vec![out]);
                }
            }
        }
    }

    // Check scales input (index 2)
    if let Some(scales_name) = node.inputs.get(2) {
        if !scales_name.is_empty() {
            if let Some(scales_tensor) = weights.get(scales_name) {
                if scales_tensor.data.len() == input_shape.len() {
                    let out: Vec<usize> = input_shape
                        .iter()
                        .zip(scales_tensor.data.iter())
                        .map(|(&d, &s)| (d as f32 * s) as usize)
                        .collect();
                    return Some(vec![out]);
                }
            }
        }
    }

    None
}
