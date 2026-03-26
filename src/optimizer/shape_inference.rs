//! Shape inference pre-pass for ONNX graphs.
//!
//! Propagates tensor shapes through the graph in topological order,
//! enabling downstream optimizations that need shape information.

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Infer output shapes for all nodes in the graph.
///
/// Walks nodes in order (assumed topologically sorted), computing output
/// shapes from known input shapes and op-specific rules. Unknown shapes
/// are silently skipped (best-effort).
pub fn infer_shapes(
    nodes: &[Node],
    weights: &HashMap<String, Tensor>,
    input_shapes: &HashMap<String, Vec<usize>>,
) -> HashMap<String, Vec<usize>> {
    let mut known: HashMap<String, Vec<usize>> = input_shapes.clone();

    // Add weight shapes
    for (name, tensor) in weights {
        known.insert(name.clone(), tensor.shape.clone());
    }

    for node in nodes {
        if let Some(output_shapes) = infer_node_shapes(node, &known, weights) {
            for (out_name, shape) in node.outputs.iter().zip(output_shapes) {
                if !out_name.is_empty() {
                    known.insert(out_name.clone(), shape);
                }
            }
        }
    }

    known
}

/// Diagnostic information about a shape inference issue for a specific node.
#[derive(Debug, Clone)]
pub struct ShapeDiagnostic {
    /// Name of the node where inference failed or was skipped.
    pub node_name: String,
    /// The ONNX operator type of the node.
    pub op_type: String,
    /// Human-readable description of the issue.
    pub message: String,
}

impl std::fmt::Display for ShapeDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Node '{}' ({}): {}",
            self.node_name, self.op_type, self.message
        )
    }
}

/// Infer shapes with error reporting.
///
/// Returns shapes plus any diagnostics about nodes where inference failed
/// or was skipped. This provides more context than `infer_shapes` for
/// debugging shape-related issues.
pub fn infer_shapes_with_diagnostics(
    nodes: &[Node],
    weights: &HashMap<String, Tensor>,
    input_shapes: &HashMap<String, Vec<usize>>,
) -> (HashMap<String, Vec<usize>>, Vec<ShapeDiagnostic>) {
    let mut known: HashMap<String, Vec<usize>> = input_shapes.clone();
    let mut diagnostics = Vec::new();

    // Add weight shapes
    for (name, tensor) in weights {
        known.insert(name.clone(), tensor.shape.clone());
    }

    for node in nodes {
        let op_str = node.op.as_str().to_string();
        match infer_node_shapes(node, &known, weights) {
            Some(output_shapes) => {
                for (out_name, shape) in node.outputs.iter().zip(output_shapes) {
                    if !out_name.is_empty() {
                        known.insert(out_name.clone(), shape);
                    }
                }
            }
            None => {
                // Determine why inference failed
                let missing_inputs: Vec<String> = node
                    .inputs
                    .iter()
                    .filter(|inp| !inp.is_empty() && !known.contains_key(inp.as_str()))
                    .cloned()
                    .collect();

                let message = if !missing_inputs.is_empty() {
                    format!(
                        "Shape inference skipped: missing input shape(s) for [{}]",
                        missing_inputs.join(", ")
                    )
                } else {
                    format!(
                        "Shape inference not supported or failed for op '{}'",
                        op_str
                    )
                };

                diagnostics.push(ShapeDiagnostic {
                    node_name: node.name.clone(),
                    op_type: op_str,
                    message,
                });
            }
        }
    }

    (known, diagnostics)
}

/// Try to infer output shapes for a single node. Returns `None` if any
/// required input shape is unavailable.
fn infer_node_shapes(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Tensor>,
) -> Option<Vec<Vec<usize>>> {
    match node.op {
        // Unary element-wise: output shape = input[0] shape
        OpKind::Identity
        | OpKind::Cast
        | OpKind::Relu
        | OpKind::Sigmoid
        | OpKind::Tanh
        | OpKind::Gelu
        | OpKind::SiLU
        | OpKind::Erf
        | OpKind::Abs
        | OpKind::Log
        | OpKind::Exp
        | OpKind::Neg
        | OpKind::Sqrt
        | OpKind::Ceil
        | OpKind::Floor
        | OpKind::Round
        | OpKind::Sign => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // Normalization ops: output shape = input[0] shape
        OpKind::Softmax
        | OpKind::LayerNorm
        | OpKind::BatchNorm
        | OpKind::GroupNorm
        | OpKind::RMSNorm => {
            let shape = get_input_shape(node, 0, known)?;
            Some(vec![shape])
        }

        // Binary element-wise with broadcasting
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Pow => {
            let a = get_input_shape(node, 0, known)?;
            let b = get_input_shape(node, 1, known)?;
            let out = Tensor::broadcast_shape(&a, &b).ok()?;
            Some(vec![out])
        }

        OpKind::MatMul => infer_matmul_shape(node, known),

        OpKind::Gemm => infer_gemm_shape(node, known),

        OpKind::Reshape => infer_reshape_shape(node, known, weights),

        OpKind::Transpose => infer_transpose_shape(node, known),

        OpKind::Squeeze => infer_squeeze_shape(node, known),

        OpKind::Unsqueeze => infer_unsqueeze_shape(node, known),

        OpKind::Flatten => infer_flatten_shape(node, known),

        OpKind::Concat => infer_concat_shape(node, known),

        OpKind::Split => infer_split_shape(node, known),

        OpKind::Conv => infer_conv_shape(node, known),

        OpKind::Gather => infer_gather_shape(node, known),

        OpKind::Slice => infer_slice_shape(node, known, weights),

        // Delegate to extended shape inference for remaining ops
        _ => super::shape_inference_ext::infer_ext_node_shapes(node, known, weights),
    }
}

/// Get the shape of the i-th input, or None if unavailable.
pub(crate) fn get_input_shape(
    node: &Node,
    idx: usize,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<usize>> {
    let name = node.inputs.get(idx)?;
    if name.is_empty() {
        return None;
    }
    known.get(name).cloned()
}

/// MatMul shape: [..., M, K] x [..., K, N] -> [..., M, N]
/// with batch dimension broadcasting.
fn infer_matmul_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
    let a = get_input_shape(node, 0, known)?;
    let b = get_input_shape(node, 1, known)?;

    if a.is_empty() || b.is_empty() {
        return None;
    }

    // Handle 1-D cases per ONNX MatMul spec
    let (a_shape, a_was_1d) = if a.len() == 1 {
        (vec![1, a[0]], true)
    } else {
        (a.clone(), false)
    };

    let (b_shape, b_was_1d) = if b.len() == 1 {
        (vec![b[0], 1], true)
    } else {
        (b.clone(), false)
    };

    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    let m = a_shape[a_rank - 2];
    let n = b_shape[b_rank - 1];

    // Broadcast batch dimensions
    let a_batch = &a_shape[..a_rank - 2];
    let b_batch = &b_shape[..b_rank - 2];

    let batch = if a_batch.is_empty() && b_batch.is_empty() {
        vec![]
    } else if a_batch.is_empty() {
        b_batch.to_vec()
    } else if b_batch.is_empty() {
        a_batch.to_vec()
    } else {
        Tensor::broadcast_shape(a_batch, b_batch).ok()?
    };

    let mut out = batch;
    if !a_was_1d {
        out.push(m);
    }
    if !b_was_1d {
        out.push(n);
    }
    // If both were 1-D, result is scalar (empty shape) per spec,
    // but we represent as [1] for simplicity
    if a_was_1d && b_was_1d {
        out.push(1);
    }

    Some(vec![out])
}

/// Gemm: Y = alpha * A' * B' + beta * C
/// Output shape is [M, N] considering transA/transB.
fn infer_gemm_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
    let a = get_input_shape(node, 0, known)?;
    let b = get_input_shape(node, 1, known)?;

    if a.len() != 2 || b.len() != 2 {
        return None;
    }

    let trans_a = node.attrs.i("transA", 0) != 0;
    let trans_b = node.attrs.i("transB", 0) != 0;

    let m = if trans_a { a[1] } else { a[0] };
    let n = if trans_b { b[0] } else { b[1] };

    Some(vec![vec![m, n]])
}

/// Reshape: compute output shape from constant second input.
/// Resolves -1 dimension using total element count.
fn infer_reshape_shape(
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
fn infer_transpose_shape(
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
fn infer_squeeze_shape(
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
fn infer_unsqueeze_shape(
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
fn infer_flatten_shape(
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

/// Concat: sum along axis dimension.
fn infer_concat_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
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
fn infer_split_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
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

/// Conv shape inference: [N, C_out, H_out, W_out]
fn infer_conv_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
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
fn infer_gather_shape(node: &Node, known: &HashMap<String, Vec<usize>>) -> Option<Vec<Vec<usize>>> {
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
fn infer_slice_shape(
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
mod tests {
    use super::*;

    fn shapes_map(pairs: &[(&str, Vec<usize>)]) -> HashMap<String, Vec<usize>> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn weights_map(pairs: &[(&str, Vec<f32>, Vec<usize>)]) -> HashMap<String, Tensor> {
        pairs
            .iter()
            .map(|(k, data, shape)| (k.to_string(), Tensor::new(data.clone(), shape.clone())))
            .collect()
    }

    #[test]
    fn test_shape_inference_elementwise() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let nodes = vec![make_node(OpKind::Add, "add", vec!["a", "b"], vec!["c"])];
        let weights = HashMap::new();
        // a: [2, 3], b: [3] -> broadcast to [2, 3]
        let input_shapes = shapes_map(&[("a", vec![2, 3]), ("b", vec![3])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("c"), Some(&vec![2, 3]));
    }

    #[test]
    fn test_shape_inference_matmul() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let nodes = vec![make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["c"])];
        let weights = HashMap::new();
        // [2, 3] x [3, 4] -> [2, 4]
        let input_shapes = shapes_map(&[("a", vec![2, 3]), ("b", vec![3, 4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("c"), Some(&vec![2, 4]));
    }

    #[test]
    fn test_shape_inference_matmul_batched() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let nodes = vec![make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["c"])];
        let weights = HashMap::new();
        // [8, 2, 3] x [8, 3, 4] -> [8, 2, 4]
        let input_shapes = shapes_map(&[("a", vec![8, 2, 3]), ("b", vec![8, 3, 4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("c"), Some(&vec![8, 2, 4]));
    }

    #[test]
    fn test_shape_inference_conv() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        // Conv2D: input [1, 3, 8, 8], weight [16, 3, 3, 3], stride=1, pad=1
        let mut conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["y"]);
        conv.attrs
            .int_lists
            .insert("strides".to_string(), vec![1, 1]);
        conv.attrs
            .int_lists
            .insert("pads".to_string(), vec![1, 1, 1, 1]);
        conv.attrs
            .int_lists
            .insert("kernel_shape".to_string(), vec![3, 3]);

        let nodes = vec![conv];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![1, 3, 8, 8]), ("w", vec![16, 3, 3, 3])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // Output: [1, 16, 8, 8] with pad=1, kernel=3, stride=1
        assert_eq!(result.get("y"), Some(&vec![1, 16, 8, 8]));
    }

    #[test]
    fn test_shape_inference_conv_no_pad() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["y"]);
        conv.attrs
            .int_lists
            .insert("strides".to_string(), vec![2, 2]);
        conv.attrs
            .int_lists
            .insert("pads".to_string(), vec![0, 0, 0, 0]);
        conv.attrs
            .int_lists
            .insert("kernel_shape".to_string(), vec![3, 3]);

        let nodes = vec![conv];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![1, 3, 8, 8]), ("w", vec![16, 3, 3, 3])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // (8 - 3) / 2 + 1 = 3
        assert_eq!(result.get("y"), Some(&vec![1, 16, 3, 3]));
    }

    #[test]
    fn test_shape_inference_reshape() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let nodes = vec![make_node(
            OpKind::Reshape,
            "reshape",
            vec!["x", "shape"],
            vec!["y"],
        )];
        // shape tensor: [2, -1] meaning reshape [2, 3, 4] -> [2, 12]
        let weights = weights_map(&[("shape", vec![2.0, -1.0], vec![2])]);
        let input_shapes = shapes_map(&[("x", vec![2, 3, 4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 12]));
    }

    #[test]
    fn test_shape_inference_transpose() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut t = make_node(OpKind::Transpose, "t", vec!["x"], vec!["y"]);
        t.attrs.int_lists.insert("perm".to_string(), vec![0, 2, 1]);

        let nodes = vec![t];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3, 4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 4, 3]));
    }

    #[test]
    fn test_shape_inference_transpose_default() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let t = make_node(OpKind::Transpose, "t", vec!["x"], vec!["y"]);
        // No perm attribute -> reverse
        let nodes = vec![t];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3, 4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![4, 3, 2]));
    }

    #[test]
    fn test_shape_inference_concat() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut cat = make_node(OpKind::Concat, "cat", vec!["a", "b", "c"], vec!["y"]);
        cat.attrs.ints.insert("axis".to_string(), 1);

        let nodes = vec![cat];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[
            ("a", vec![2, 3, 4]),
            ("b", vec![2, 5, 4]),
            ("c", vec![2, 7, 4]),
        ]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 15, 4]));
    }

    #[test]
    fn test_shape_inference_flatten() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut flat = make_node(OpKind::Flatten, "flat", vec!["x"], vec!["y"]);
        flat.attrs.ints.insert("axis".to_string(), 2);

        let nodes = vec![flat];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3, 4, 5])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // axis=2: [2*3, 4*5] = [6, 20]
        assert_eq!(result.get("y"), Some(&vec![6, 20]));
    }

    #[test]
    fn test_shape_inference_squeeze() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut sq = make_node(OpKind::Squeeze, "sq", vec!["x"], vec!["y"]);
        sq.attrs.int_lists.insert("axes".to_string(), vec![1, 3]);

        let nodes = vec![sq];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 1, 3, 1, 4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![2, 3, 4]));
    }

    #[test]
    fn test_shape_inference_unsqueeze() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut usq = make_node(OpKind::Unsqueeze, "usq", vec!["x"], vec!["y"]);
        usq.attrs.int_lists.insert("axes".to_string(), vec![0, 3]);

        let nodes = vec![usq];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // [2, 3] -> insert at 0, 3 -> [1, 2, 3, 1]
        assert_eq!(result.get("y"), Some(&vec![1, 2, 3, 1]));
    }

    #[test]
    fn test_shape_inference_gemm() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut gemm = make_node(OpKind::Gemm, "gemm", vec!["a", "b", "c"], vec!["y"]);
        gemm.attrs.ints.insert("transB".to_string(), 1);

        let nodes = vec![gemm];
        let weights = HashMap::new();
        // A: [4, 3], B: [5, 3] (transB=1 -> use B[0]=5 as N)
        let input_shapes = shapes_map(&[("a", vec![4, 3]), ("b", vec![5, 3]), ("c", vec![5])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("y"), Some(&vec![4, 5]));
    }

    #[test]
    fn test_shape_inference_split() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut split = make_node(OpKind::Split, "split", vec!["x"], vec!["a", "b", "c"]);
        split.attrs.ints.insert("axis".to_string(), 1);
        split
            .attrs
            .int_lists
            .insert("split".to_string(), vec![2, 3, 5]);

        let nodes = vec![split];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![4, 10, 6])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("a"), Some(&vec![4, 2, 6]));
        assert_eq!(result.get("b"), Some(&vec![4, 3, 6]));
        assert_eq!(result.get("c"), Some(&vec![4, 5, 6]));
    }

    #[test]
    fn test_shape_inference_gather() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        let mut gather = make_node(OpKind::Gather, "gather", vec!["data", "indices"], vec!["y"]);
        gather.attrs.ints.insert("axis".to_string(), 0);

        let nodes = vec![gather];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("data", vec![10, 5]), ("indices", vec![3, 2])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        // Gather axis=0: [10, 5] with indices [3, 2] -> [3, 2, 5]
        assert_eq!(result.get("y"), Some(&vec![3, 2, 5]));
    }

    #[test]
    fn test_shape_inference_chain() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        // Chain: MatMul -> Relu -> Add
        let nodes = vec![
            make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
            make_node(OpKind::Relu, "relu", vec!["mm_out"], vec!["relu_out"]),
            make_node(OpKind::Add, "add", vec!["relu_out", "bias"], vec!["out"]),
        ];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3]), ("w", vec![3, 4]), ("bias", vec![4])]);

        let result = infer_shapes(&nodes, &weights, &input_shapes);
        assert_eq!(result.get("mm_out"), Some(&vec![2, 4]));
        assert_eq!(result.get("relu_out"), Some(&vec![2, 4]));
        assert_eq!(result.get("out"), Some(&vec![2, 4]));
    }

    #[test]
    fn test_shape_diagnostics() {
        use crate::graph::OpKind;
        use crate::optimizer::test_utils::make_node;

        // Create a graph where the second node references an unknown input,
        // and a third node uses an unsupported op (Unknown).
        let nodes = vec![
            make_node(OpKind::Relu, "relu1", vec!["x"], vec!["r1"]),
            make_node(
                OpKind::Add,
                "add_missing",
                vec!["r1", "missing_input"],
                vec!["a1"],
            ),
            make_node(
                OpKind::Unknown("CustomOp".to_string()),
                "custom",
                vec!["r1"],
                vec!["c1"],
            ),
        ];
        let weights = HashMap::new();
        let input_shapes = shapes_map(&[("x", vec![2, 3])]);

        let (shapes, diagnostics) = infer_shapes_with_diagnostics(&nodes, &weights, &input_shapes);

        // relu1 should succeed
        assert_eq!(shapes.get("r1"), Some(&vec![2, 3]));

        // add_missing should fail due to missing input
        assert!(diagnostics.len() >= 2);

        let add_diag = diagnostics.iter().find(|d| d.node_name == "add_missing");
        assert!(add_diag.is_some(), "Expected diagnostic for add_missing");
        let add_diag = add_diag.expect("checked above");
        assert_eq!(add_diag.op_type, "Add");
        assert!(
            add_diag.message.contains("missing_input"),
            "Diagnostic should mention the missing input, got: {}",
            add_diag.message
        );

        // custom should fail due to unsupported op
        let custom_diag = diagnostics.iter().find(|d| d.node_name == "custom");
        assert!(custom_diag.is_some(), "Expected diagnostic for custom");
        let custom_diag = custom_diag.expect("checked above");
        assert_eq!(custom_diag.op_type, "CustomOp");
        assert!(
            custom_diag.message.contains("not supported"),
            "Diagnostic should mention unsupported op, got: {}",
            custom_diag.message
        );
    }
}
