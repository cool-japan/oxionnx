//! Shape inference pre-pass for ONNX graphs.
//!
//! Propagates tensor shapes through the graph in topological order,
//! enabling downstream optimizations that need shape information.

mod conv_gather_slice;
mod helpers;
mod matmul_gemm;
mod reshape_ops;
mod sequence_ops;

pub(crate) use helpers::get_input_shape;

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

        OpKind::MatMul => matmul_gemm::infer_matmul_shape(node, known),

        OpKind::Gemm => matmul_gemm::infer_gemm_shape(node, known),

        OpKind::Reshape => reshape_ops::infer_reshape_shape(node, known, weights),

        OpKind::Transpose => reshape_ops::infer_transpose_shape(node, known),

        OpKind::Squeeze => reshape_ops::infer_squeeze_shape(node, known),

        OpKind::Unsqueeze => reshape_ops::infer_unsqueeze_shape(node, known),

        OpKind::Flatten => reshape_ops::infer_flatten_shape(node, known),

        OpKind::Concat => sequence_ops::infer_concat_shape(node, known),

        OpKind::Split => sequence_ops::infer_split_shape(node, known),

        OpKind::Conv => conv_gather_slice::infer_conv_shape(node, known),

        OpKind::Gather => conv_gather_slice::infer_gather_shape(node, known),

        OpKind::Slice => conv_gather_slice::infer_slice_shape(node, known, weights),

        // Delegate to extended shape inference for remaining ops
        _ => super::shape_inference_ext::infer_ext_node_shapes(node, known, weights),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
