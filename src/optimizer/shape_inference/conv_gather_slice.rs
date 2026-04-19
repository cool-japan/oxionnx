//! Shape inference for Conv, Gather, and Slice operators.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::helpers::get_input_shape;

/// Conv shape inference: [N, C_out, H_out, W_out]
pub(super) fn infer_conv_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let weight_shape = get_input_shape(node, 1, known)?;

    // Input: [N, C, H, W, ...], Weight: [C_out, C_in/group, kH, kW, ...]
    if input_shape.len() < 3 || weight_shape.len() < 3 {
        return None;
    }

    let n = input_shape[0];
    let c_out = weight_shape[0];
    let spatial_dims = input_shape.len() - 2;

    // Get kernel shape from attributes or weight tensor
    let kernel_shape_attr: Vec<i64> = node.attrs.ints("kernel_shape").to_vec();
    let kernel_shape: Vec<usize> = if kernel_shape_attr.is_empty() {
        weight_shape[2..].to_vec()
    } else {
        kernel_shape_attr.iter().map(|&k| k as usize).collect()
    };

    if kernel_shape.len() != spatial_dims {
        return None;
    }

    // Get strides (default: all 1)
    let strides_attr: Vec<i64> = node.attrs.ints("strides").to_vec();
    let strides: Vec<usize> = if strides_attr.is_empty() {
        vec![1; spatial_dims]
    } else {
        strides_attr.iter().map(|&s| s as usize).collect()
    };

    // Get dilations (default: all 1)
    let dilations_attr: Vec<i64> = node.attrs.ints("dilations").to_vec();
    let dilations: Vec<usize> = if dilations_attr.is_empty() {
        vec![1; spatial_dims]
    } else {
        dilations_attr.iter().map(|&d| d as usize).collect()
    };

    // Get pads (default: all 0). Format: [begin_0, begin_1, ..., end_0, end_1, ...]
    let pads_attr: Vec<i64> = node.attrs.ints("pads").to_vec();
    let pads: Vec<usize> = if pads_attr.is_empty() {
        vec![0; spatial_dims * 2]
    } else {
        pads_attr.iter().map(|&p| p as usize).collect()
    };

    if pads.len() != spatial_dims * 2 {
        return None;
    }

    let mut out_shape = vec![n, c_out];
    for d in 0..spatial_dims {
        let input_dim = input_shape[d + 2];
        let effective_kernel = (kernel_shape[d] - 1) * dilations[d] + 1;
        let padded = input_dim + pads[d] + pads[d + spatial_dims];
        if padded < effective_kernel {
            return None;
        }
        let out_dim = (padded - effective_kernel) / strides[d] + 1;
        out_shape.push(out_dim);
    }

    Some(vec![out_shape])
}

/// Gather shape: replace gathered axis dim with indices shape.
pub(super) fn infer_gather_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let data_shape = get_input_shape(node, 0, known)?;
    let indices_shape = get_input_shape(node, 1, known)?;

    let rank = data_shape.len() as i64;
    let axis_raw = node.attrs.i("axis", 0);
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };

    if axis >= data_shape.len() {
        return None;
    }

    let mut out = Vec::new();
    out.extend_from_slice(&data_shape[..axis]);
    out.extend_from_slice(&indices_shape);
    out.extend_from_slice(&data_shape[axis + 1..]);

    Some(vec![out])
}

/// Slice shape: compute sliced dim sizes from constant starts/ends/steps inputs.
pub(super) fn infer_slice_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;

    // inputs: data, starts, ends, [axes], [steps]
    let starts_name = node.inputs.get(1)?;
    let ends_name = node.inputs.get(2)?;

    let starts_tensor = weights.get(starts_name)?;
    let ends_tensor = weights.get(ends_name)?;

    let starts: Vec<i64> = starts_tensor.data.iter().map(|&v| v as i64).collect();
    let ends: Vec<i64> = ends_tensor.data.iter().map(|&v| v as i64).collect();

    let axes: Vec<usize> = if let Some(axes_name) = node.inputs.get(3) {
        if let Some(axes_t) = weights.get(axes_name) {
            axes_t
                .data
                .iter()
                .map(|&v| {
                    let a = v as i64;
                    if a < 0 {
                        (a + input_shape.len() as i64) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        } else {
            (0..starts.len()).collect()
        }
    } else {
        (0..starts.len()).collect()
    };

    let steps: Vec<i64> = if let Some(steps_name) = node.inputs.get(4) {
        if let Some(steps_t) = weights.get(steps_name) {
            steps_t.data.iter().map(|&v| v as i64).collect()
        } else {
            vec![1; starts.len()]
        }
    } else {
        vec![1; starts.len()]
    };

    let mut out = input_shape.clone();

    for (i, &axis) in axes.iter().enumerate() {
        if axis >= input_shape.len() || i >= starts.len() || i >= ends.len() {
            return None;
        }

        let dim_size = input_shape[axis] as i64;
        let step = if i < steps.len() { steps[i] } else { 1 };
        if step == 0 {
            return None;
        }

        let mut start = starts[i];
        let mut end = ends[i];

        // Clamp to valid range
        if start < 0 {
            start += dim_size;
        }
        if end < 0 {
            end += dim_size;
        }

        start = start.clamp(0, dim_size);
        // Allow i64::MAX as "end" meaning full extent
        end = if end > dim_size { dim_size } else { end.max(0) };

        let sliced_dim = if step > 0 {
            if end > start {
                ((end - start + step - 1) / step) as usize
            } else {
                0
            }
        } else if start > end {
            ((start - end + (-step) - 1) / (-step)) as usize
        } else {
            0
        };

        out[axis] = sliced_dim;
    }

    Some(vec![out])
}
