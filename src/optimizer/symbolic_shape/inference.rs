//! Symbolic shape inference engine.
//!
//! Provides [`infer_node_symbolic`] (per-node dispatcher) and
//! [`infer_symbolic_shapes`] (whole-graph propagation).

use std::collections::HashMap;

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::Tensor;

use super::ops::{
    infer_concat_symbolic, infer_flatten_symbolic, infer_gemm_symbolic, infer_matmul_symbolic,
    infer_reshape_symbolic, infer_squeeze_symbolic, infer_transpose_symbolic,
    infer_unsqueeze_symbolic,
};
use super::types::{SymDim, SymbolicShape};
use super::utils::{broadcast_symbolic, from_concrete};

/// Infer symbolic output shapes for a single node.
///
/// Returns `None` when a required input shape is unavailable or the op is
/// not yet supported by the symbolic engine.
fn infer_node_symbolic(
    op: &OpKind,
    inputs: &[Option<&SymbolicShape>],
    attrs: &Attributes,
) -> Option<Vec<SymbolicShape>> {
    match op {
        // --- unary element-wise ---
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
        | OpKind::Sign
        | OpKind::Reciprocal
        | OpKind::Sin
        | OpKind::Cos
        | OpKind::Tan
        | OpKind::Asin
        | OpKind::Acos
        | OpKind::Atan
        | OpKind::Sinh
        | OpKind::Cosh
        | OpKind::Asinh
        | OpKind::Acosh
        | OpKind::Atanh
        | OpKind::HardSigmoid
        | OpKind::HardSwish
        | OpKind::Not
        | OpKind::LeakyRelu
        | OpKind::LogSoftmax
        | OpKind::Softplus
        | OpKind::Softsign
        | OpKind::Mish
        | OpKind::Celu
        | OpKind::Elu
        | OpKind::Selu
        | OpKind::ThresholdedRelu
        | OpKind::Clip
        | OpKind::BitwiseNot
        | OpKind::Hardmax
        | OpKind::Shrink => {
            let s = inputs.first().and_then(|o| *o)?;
            Some(vec![s.clone()])
        }

        // --- normalization (output shape == input[0] shape) ---
        OpKind::Softmax
        | OpKind::LayerNorm
        | OpKind::BatchNorm
        | OpKind::GroupNorm
        | OpKind::RMSNorm
        | OpKind::InstanceNorm
        | OpKind::LpNorm
        | OpKind::MeanVarianceNormalization => {
            let s = inputs.first().and_then(|o| *o)?;
            Some(vec![s.clone()])
        }

        // --- dropout: first output same as input, optional mask ---
        OpKind::Dropout => {
            let s = inputs.first().and_then(|o| *o)?;
            // output 0 = same shape, output 1 = mask (same shape)
            Some(vec![s.clone(), s.clone()])
        }

        // --- binary element-wise with broadcasting ---
        OpKind::Add
        | OpKind::Sub
        | OpKind::Mul
        | OpKind::Div
        | OpKind::Pow
        | OpKind::Mod
        | OpKind::BitwiseAnd
        | OpKind::BitwiseOr
        | OpKind::BitwiseXor => {
            let a = inputs.first().and_then(|o| *o)?;
            let b = inputs.get(1).and_then(|o| *o)?;
            let out = broadcast_symbolic(a, b)?;
            Some(vec![out])
        }

        // --- comparison / logic (binary broadcast -> same shape) ---
        OpKind::Equal
        | OpKind::Greater
        | OpKind::GreaterOrEqual
        | OpKind::Less
        | OpKind::LessOrEqual
        | OpKind::And
        | OpKind::Or
        | OpKind::Xor => {
            let a = inputs.first().and_then(|o| *o)?;
            let b = inputs.get(1).and_then(|o| *o)?;
            let out = broadcast_symbolic(a, b)?;
            Some(vec![out])
        }

        // --- MatMul ---
        OpKind::MatMul => infer_matmul_symbolic(inputs),

        // --- Gemm ---
        OpKind::Gemm => infer_gemm_symbolic(inputs, attrs),

        // --- Reshape ---
        OpKind::Reshape => infer_reshape_symbolic(inputs),

        // --- Transpose ---
        OpKind::Transpose => infer_transpose_symbolic(inputs, attrs),

        // --- Concat ---
        OpKind::Concat => infer_concat_symbolic(inputs, attrs),

        // --- Squeeze ---
        OpKind::Squeeze => infer_squeeze_symbolic(inputs, attrs),

        // --- Unsqueeze ---
        OpKind::Unsqueeze => infer_unsqueeze_symbolic(inputs, attrs),

        // --- Flatten ---
        OpKind::Flatten => infer_flatten_symbolic(inputs, attrs),

        // --- Expand ---
        OpKind::Expand => {
            // Expand follows broadcast rules between input and shape tensor.
            // If shape tensor is known we use broadcast, otherwise pass through.
            let a = inputs.first().and_then(|o| *o)?;
            let b = inputs.get(1).and_then(|o| *o)?;
            let out = broadcast_symbolic(a, b)?;
            Some(vec![out])
        }

        // --- Where (ternary broadcast) ---
        OpKind::Where => {
            let cond = inputs.first().and_then(|o| *o)?;
            let x = inputs.get(1).and_then(|o| *o)?;
            let y = inputs.get(2).and_then(|o| *o)?;
            let tmp = broadcast_symbolic(cond, x)?;
            let out = broadcast_symbolic(&tmp, y)?;
            Some(vec![out])
        }

        // --- Shape op: output is always [rank] ---
        OpKind::Shape => {
            let s = inputs.first().and_then(|o| *o)?;
            Some(vec![vec![SymDim::Known(s.len())]])
        }

        // Unsupported ops return None so the caller can skip gracefully.
        _ => None,
    }
}

/// Infer symbolic shapes for all tensors in the graph.
///
/// Similar to [`super::super::shape_inference::infer_shapes`] but propagates symbolic
/// dimensions. `input_shapes` maps tensor name to symbolic shape for graph
/// inputs. Returns a map from tensor name to symbolic shape.
pub fn infer_symbolic_shapes(
    nodes: &[Node],
    weights: &HashMap<String, Tensor>,
    input_shapes: &HashMap<String, SymbolicShape>,
) -> HashMap<String, SymbolicShape> {
    let mut shapes: HashMap<String, SymbolicShape> = HashMap::new();

    // Seed with weight shapes (all concrete).
    for (name, tensor) in weights {
        shapes.insert(name.clone(), from_concrete(&tensor.shape));
    }

    // Seed with user-provided input shapes.
    for (name, shape) in input_shapes {
        shapes.insert(name.clone(), shape.clone());
    }

    // Propagate through nodes (assumed topologically sorted).
    for node in nodes {
        let op = &node.op;
        let input_syms: Vec<Option<&SymbolicShape>> = node
            .inputs
            .iter()
            .map(|name| {
                if name.is_empty() {
                    None
                } else {
                    shapes.get(name)
                }
            })
            .collect();

        if let Some(out_shapes) = infer_node_symbolic(op, &input_syms, &node.attrs) {
            for (out_name, out_shape) in node.outputs.iter().zip(out_shapes) {
                if !out_name.is_empty() {
                    shapes.insert(out_name.clone(), out_shape);
                }
            }
        }
    }

    shapes
}
