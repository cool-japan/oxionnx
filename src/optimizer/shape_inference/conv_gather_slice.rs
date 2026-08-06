//! Shape inference for Conv, Gather, and Slice operators.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::helpers::get_input_shape;
use crate::optimizer::shape_inference_ext::spatial_attrs::{conv_out_dim, same_pad_split};

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

    // Kernel extents come from W ([C_out, C_in/group, k...]); the optional
    // `kernel_shape` attribute is redundant and must agree with it. If it does
    // not, refuse to infer rather than derive a padding (for `auto_pad`) that
    // differs from the one the kernel will use.
    let weight_kernel: Vec<usize> = weight_shape[2..].to_vec();
    let kernel_shape_attr: Vec<i64> = node.attrs.ints("kernel_shape").to_vec();
    let kernel_shape: Vec<usize> = if kernel_shape_attr.is_empty() {
        weight_kernel
    } else {
        if kernel_shape_attr.iter().any(|&k| k < 1) {
            return None;
        }
        let attr: Vec<usize> = kernel_shape_attr.iter().map(|&k| k as usize).collect();
        if attr != weight_kernel {
            return None;
        }
        attr
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

    // Reject degenerate attributes up front: a zero stride would divide by
    // zero and a zero kernel/dilation would underflow the effective-kernel
    // computation below. Shape inference is best-effort, so bail out.
    if strides.contains(&0) || dilations.contains(&0) || kernel_shape.contains(&0) {
        return None;
    }

    // Get pads (default: all 0). Format: [begin_0, begin_1, ..., end_0, end_1, ...]
    // `auto_pad` (when not NOTSET) takes precedence over the `pads` attribute:
    // SAME_UPPER/SAME_LOWER pad so that out = ceil(in / stride), VALID never pads.
    let pads: Vec<usize> = match node.attrs.s("auto_pad") {
        "" | "NOTSET" => {
            let pads_attr: Vec<i64> = node.attrs.ints("pads").to_vec();
            if pads_attr.is_empty() {
                vec![0; spatial_dims * 2]
            } else {
                if pads_attr.iter().any(|&p| p < 0) {
                    return None;
                }
                pads_attr.iter().map(|&p| p as usize).collect()
            }
        }
        "VALID" => vec![0; spatial_dims * 2],
        mode @ ("SAME_UPPER" | "SAME_LOWER") => {
            let lower = mode == "SAME_LOWER";
            let mut pads = vec![0_usize; spatial_dims * 2];
            for d in 0..spatial_dims {
                let (begin, end) = same_pad_split(
                    input_shape[d + 2],
                    kernel_shape[d],
                    strides[d],
                    dilations[d],
                    lower,
                );
                pads[d] = begin;
                pads[d + spatial_dims] = end;
            }
            pads
        }
        // Unknown auto_pad value: refuse to guess a shape.
        _ => return None,
    };

    if pads.len() != spatial_dims * 2 {
        return None;
    }

    let mut out_shape = vec![n, c_out];
    for d in 0..spatial_dims {
        // Shared with the pooling path (and mirroring the kernel-side
        // `pool_out_dim`): checked arithmetic throughout, so a malformed
        // attribute declines instead of overflow-panicking in a debug build.
        out_shape.push(conv_out_dim(
            input_shape[d + 2],
            pads[d],
            pads[d + spatial_dims],
            kernel_shape[d],
            dilations[d],
            strides[d],
        )?);
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

#[cfg(test)]
mod w1_auto_pad_tests {
    use super::super::infer_shapes;
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    fn conv_node(auto_pad: &str, strides: &[i64], dilations: &[i64]) -> crate::graph::Node {
        let mut conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["y"]);
        conv.attrs
            .int_lists
            .insert("kernel_shape".to_string(), vec![3, 3]);
        if !strides.is_empty() {
            conv.attrs
                .int_lists
                .insert("strides".to_string(), strides.to_vec());
        }
        if !dilations.is_empty() {
            conv.attrs
                .int_lists
                .insert("dilations".to_string(), dilations.to_vec());
        }
        if !auto_pad.is_empty() {
            conv.attrs
                .strings
                .insert("auto_pad".to_string(), auto_pad.to_string());
        }
        conv
    }

    fn infer(node: crate::graph::Node, input: Vec<usize>) -> Option<Vec<usize>> {
        let weights: HashMap<String, Tensor> = HashMap::new();
        let mut input_shapes = HashMap::new();
        input_shapes.insert("x".to_string(), input);
        input_shapes.insert("w".to_string(), vec![16, 3, 3, 3]);
        infer_shapes(&[node], &weights, &input_shapes)
            .get("y")
            .cloned()
    }

    #[test]
    fn same_upper_preserves_spatial_size() {
        // The tf2onnx / Keras default: 3x3 stride-1 SAME on 224x224 stays 224x224.
        let shape = infer(conv_node("SAME_UPPER", &[1, 1], &[]), vec![1, 3, 224, 224]);
        assert_eq!(shape, Some(vec![1, 16, 224, 224]));
    }

    #[test]
    fn same_lower_matches_same_upper_extent() {
        let upper = infer(conv_node("SAME_UPPER", &[2, 2], &[]), vec![1, 3, 7, 7]);
        let lower = infer(conv_node("SAME_LOWER", &[2, 2], &[]), vec![1, 3, 7, 7]);
        // ceil(7 / 2) = 4 regardless of which side takes the odd pixel.
        assert_eq!(upper, Some(vec![1, 16, 4, 4]));
        assert_eq!(lower, Some(vec![1, 16, 4, 4]));
    }

    #[test]
    fn same_upper_with_dilations() {
        // effective kernel = (3-1)*2 + 1 = 5; out = ceil(10/1) = 10.
        let shape = infer(
            conv_node("SAME_UPPER", &[1, 1], &[2, 2]),
            vec![1, 3, 10, 10],
        );
        assert_eq!(shape, Some(vec![1, 16, 10, 10]));
    }

    #[test]
    fn valid_ignores_explicit_pads() {
        let mut node = conv_node("VALID", &[1, 1], &[]);
        node.attrs
            .int_lists
            .insert("pads".to_string(), vec![1, 1, 1, 1]);
        assert_eq!(infer(node, vec![1, 3, 8, 8]), Some(vec![1, 16, 6, 6]));
    }

    #[test]
    fn notset_still_uses_explicit_pads() {
        let mut node = conv_node("NOTSET", &[1, 1], &[]);
        node.attrs
            .int_lists
            .insert("pads".to_string(), vec![1, 1, 1, 1]);
        assert_eq!(infer(node, vec![1, 3, 8, 8]), Some(vec![1, 16, 8, 8]));
    }

    #[test]
    fn unknown_auto_pad_refuses_to_guess() {
        assert_eq!(
            infer(conv_node("SAME", &[1, 1], &[]), vec![1, 3, 8, 8]),
            None
        );
    }

    #[test]
    fn zero_stride_does_not_divide_by_zero() {
        assert_eq!(
            infer(conv_node("", &[0, 0], &[]), vec![1, 3, 8, 8]),
            None,
            "a zero stride must abort inference, not panic"
        );
    }

    #[test]
    fn zero_dilation_does_not_underflow() {
        assert_eq!(
            infer(conv_node("", &[1, 1], &[0, 0]), vec![1, 3, 8, 8]),
            None
        );
    }

    #[test]
    fn negative_pads_abort_inference() {
        let mut node = conv_node("", &[1, 1], &[]);
        node.attrs
            .int_lists
            .insert("pads".to_string(), vec![-1, 0, 0, 0]);
        assert_eq!(infer(node, vec![1, 3, 8, 8]), None);
    }
}
