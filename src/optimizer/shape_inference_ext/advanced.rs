//! Shape inference for advanced operators.
//!
//! Covers ConvTranspose, Einsum, LSTM, GRU, LinearClassifier, and LinearRegressor.

use crate::graph::Node;
use std::collections::HashMap;

use crate::optimizer::shape_inference::get_input_shape;

/// ConvTranspose shape inference.
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
    let group = node.attrs.i("group", 1) as usize;
    let c_out = weight_shape[1] * group;
    let spatial_dims = input_shape.len() - 2;

    // Check for explicit output_shape attribute
    let output_shape_attr: Vec<i64> = node.attrs.ints("output_shape").to_vec();
    if !output_shape_attr.is_empty() && output_shape_attr.len() == spatial_dims {
        let mut out = vec![n, c_out];
        for &s in &output_shape_attr {
            out.push(s as usize);
        }
        return Some(vec![out]);
    }

    let kernel_shape_attr: Vec<i64> = node.attrs.ints("kernel_shape").to_vec();
    let kernel_shape: Vec<usize> = if kernel_shape_attr.is_empty() {
        weight_shape[2..].to_vec()
    } else {
        kernel_shape_attr.iter().map(|&k| k as usize).collect()
    };

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

    let output_padding_attr: Vec<i64> = node.attrs.ints("output_padding").to_vec();
    let output_padding: Vec<usize> = if output_padding_attr.is_empty() {
        vec![0; spatial_dims]
    } else {
        output_padding_attr.iter().map(|&p| p as usize).collect()
    };

    if pads.len() != spatial_dims * 2 {
        return None;
    }

    let mut out_shape = vec![n, c_out];
    for d in 0..spatial_dims {
        // ConvTranspose formula:
        // out = (input - 1) * stride - pad_begin - pad_end + dilation * (kernel - 1) + output_padding + 1
        let out_dim = (input_shape[d + 2] - 1) * strides[d]
            + dilations[d] * (kernel_shape[d] - 1)
            + output_padding[d]
            + 1
            - pads[d]
            - pads[d + spatial_dims];
        out_shape.push(out_dim);
    }

    Some(vec![out_shape])
}

/// Einsum: parse equation to determine output shape.
pub(super) fn infer_einsum_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let equation = node.attrs.s("equation");
    if equation.is_empty() {
        return None;
    }

    // Parse equation: "ij,jk->ik" style
    let parts: Vec<&str> = equation.split("->").collect();
    if parts.len() != 2 {
        return None;
    }

    let input_specs: Vec<&str> = parts[0].split(',').collect();
    let output_spec = parts[1].trim();

    if input_specs.len() != node.inputs.len() {
        return None;
    }

    // Build label -> dimension mapping from inputs
    let mut label_dims: HashMap<u8, usize> = HashMap::new();
    for (i, spec) in input_specs.iter().enumerate() {
        let shape = get_input_shape(node, i, known)?;
        let labels: Vec<u8> = spec
            .trim()
            .bytes()
            .filter(|b| b.is_ascii_alphabetic())
            .collect();
        if labels.len() != shape.len() {
            return None;
        }
        for (j, &label) in labels.iter().enumerate() {
            label_dims.entry(label).or_insert(shape[j]);
        }
    }

    // Build output shape from output spec
    let output_labels: Vec<u8> = output_spec
        .bytes()
        .filter(|b| b.is_ascii_alphabetic())
        .collect();
    let mut out = Vec::new();
    for &label in &output_labels {
        let dim = label_dims.get(&label)?;
        out.push(*dim);
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

/// LinearClassifier: output labels [N] and scores [N, num_classes].
pub(super) fn infer_linear_classifier_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.is_empty() {
        return None;
    }

    let n = input_shape[0];

    // Try to determine number of classes from coefficients attribute
    let coefficients = node.attrs.float_lists.get("coefficients");
    let num_features = if input_shape.len() > 1 {
        input_shape[1]
    } else {
        1
    };

    let num_classes = if let Some(coeffs) = coefficients {
        coeffs.len().checked_div(num_features)?
    } else {
        return None;
    };

    // Two outputs: labels [N], scores [N, num_classes]
    Some(vec![vec![n], vec![n, num_classes]])
}

/// LinearRegressor: output [N, num_targets].
pub(super) fn infer_linear_regressor_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.is_empty() {
        return None;
    }

    let n = input_shape[0];

    let coefficients = node.attrs.float_lists.get("coefficients");
    let num_features = if input_shape.len() > 1 {
        input_shape[1]
    } else {
        1
    };

    let targets = node.attrs.i("targets", 1) as usize;

    let num_targets = if let Some(coeffs) = coefficients {
        coeffs.len().checked_div(num_features).unwrap_or(targets)
    } else {
        targets
    };

    Some(vec![vec![n, num_targets]])
}
