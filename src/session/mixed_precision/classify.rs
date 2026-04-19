//! Operator classification for mixed precision inference.
//!
//! Determines whether an operator can run in f16 or requires f32 for numerical stability.

/// Returns `true` if the operator can safely execute with f16 inputs/outputs
/// without significant precision loss.
pub fn should_use_f16(op_type: &str) -> bool {
    matches!(
        op_type,
        // Element-wise activations & arithmetic
        "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Relu"
            | "LeakyRelu"
            | "Sigmoid"
            | "Tanh"
            | "Gelu"
            | "Silu"
            | "SiLU"
            | "HardSigmoid"
            | "HardSwish"
            | "Abs"
            | "Neg"
            | "Sqrt"
            | "Reciprocal"
            | "Clip"
            | "Erf"
            | "Softsign"
            | "Softplus"
            | "Mish"
            | "Celu"
            | "Elu"
            | "Selu"
            | "ThresholdedRelu"
            | "PRelu"
            // Normalization (compute in f16; gamma/beta stay f32 but output is f16)
            | "LayerNormalization"
            | "LayerNorm"
            | "BatchNormalization"
            | "BatchNorm"
            | "GroupNormalization"
            | "GroupNorm"
            | "RMSNorm"
            | "SimplifiedLayerNormalization"
            | "InstanceNorm"
            | "InstanceNormalization"
            // Softmax (stable computation works in f16)
            | "Softmax"
            | "LogSoftmax"
            // Shape manipulation (zero-copy or simple data movement)
            | "Transpose"
            | "Reshape"
            | "Concat"
            | "Slice"
            | "Split"
            | "Squeeze"
            | "Unsqueeze"
            | "Flatten"
            | "Identity"
            | "Expand"
            | "Tile"
            | "DepthToSpace"
            | "SpaceToDepth"
            // Attention (scores in f16)
            | "Attention"
            | "MultiHeadAttention"
            | "RotaryEmbedding"
            // Dropout (just passthrough or mask)
            | "Dropout"
    )
}

/// Returns `true` if the operator requires f32 for numerical stability
/// (accumulation, precision-sensitive math).
pub fn requires_f32(op_type: &str) -> bool {
    matches!(
        op_type,
        "MatMul"
            | "Gemm"
            | "ReduceSum"
            | "ReduceMean"
            | "ReduceMax"
            | "ReduceMin"
            | "ReduceProd"
            | "ReduceL1"
            | "ReduceL2"
            | "ReduceLogSum"
            | "ReduceLogSumExp"
            | "ReduceSumSquare"
            | "Pow"
            | "Exp"
            | "Log"
            | "Conv"
            | "ConvTranspose"
            | "ConvAddRelu"
            | "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "GlobalMaxPool"
            | "CumSum"
            | "Einsum"
    )
}
