//! Mixed precision inference support.
//!
//! Classifies operators as f16-safe or f32-required, and provides helpers
//! for executing element-wise operations natively in f16 using the `half` crate.

#![allow(dead_code)]

use crate::tensor::Tensor;
use crate::OnnxError;

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

/// Convert a f32 tensor's data to f16 precision (stored as f32 values).
///
/// Each f32 value is rounded to the nearest f16 representable value.
/// This simulates f16 storage precision loss while keeping the f32 container.
pub fn round_to_f16_precision(tensor: &Tensor) -> Tensor {
    let data: Vec<f32> = tensor
        .data
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    Tensor::new(data, tensor.shape.clone())
}

/// Execute an element-wise operation natively in f16 precision.
///
/// Converts inputs to `half::f16`, performs the operation, and returns
/// the result as a standard f32 `Tensor` with f16-rounded values.
///
/// Returns `None` if the op is not supported for native f16 execution,
/// in which case the caller should fall back to the normal f32 path
/// (optionally with f16 rounding of outputs).
pub fn execute_elementwise_f16(
    op_type: &str,
    inputs: &[&Tensor],
) -> Option<Result<Vec<Tensor>, OnnxError>> {
    match op_type {
        "Relu" => Some(execute_relu_f16(inputs)),
        "Add" => Some(execute_add_f16(inputs)),
        "Mul" => Some(execute_mul_f16(inputs)),
        "Sub" => Some(execute_sub_f16(inputs)),
        "Sigmoid" => Some(execute_sigmoid_f16(inputs)),
        "Tanh" => Some(execute_tanh_f16(inputs)),
        "Neg" => Some(execute_neg_f16(inputs)),
        "Abs" => Some(execute_abs_f16(inputs)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// f16-native element-wise implementations
// ---------------------------------------------------------------------------

fn execute_relu_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Relu f16: expected at least 1 input".to_string())
    })?;
    let zero = half::f16::ZERO;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| {
            let h = half::f16::from_f32(v);
            if h < zero { zero } else { h }.to_f32()
        })
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

fn execute_add_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "Add f16: expected 2 inputs".to_string(),
        ));
    }
    let a = inputs[0];
    let b = inputs[1];

    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(|e| OnnxError::ShapeMismatch(format!("Add f16 broadcast: {e}")))?;
    let out_size: usize = out_shape.iter().product();

    let data = if a.shape == b.shape {
        // Fast path: same shape, no broadcasting needed
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&va, &vb)| {
                let ha = half::f16::from_f32(va);
                let hb = half::f16::from_f32(vb);
                (ha + hb).to_f32()
            })
            .collect()
    } else {
        broadcast_binary_f16(
            &a.data,
            &a.shape,
            &b.data,
            &b.shape,
            &out_shape,
            out_size,
            |ha, hb| ha + hb,
        )
    };

    Ok(vec![Tensor::new(data, out_shape)])
}

fn execute_mul_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "Mul f16: expected 2 inputs".to_string(),
        ));
    }
    let a = inputs[0];
    let b = inputs[1];

    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(|e| OnnxError::ShapeMismatch(format!("Mul f16 broadcast: {e}")))?;
    let out_size: usize = out_shape.iter().product();

    let data = if a.shape == b.shape {
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&va, &vb)| {
                let ha = half::f16::from_f32(va);
                let hb = half::f16::from_f32(vb);
                (ha * hb).to_f32()
            })
            .collect()
    } else {
        broadcast_binary_f16(
            &a.data,
            &a.shape,
            &b.data,
            &b.shape,
            &out_shape,
            out_size,
            |ha, hb| ha * hb,
        )
    };

    Ok(vec![Tensor::new(data, out_shape)])
}

fn execute_sub_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    if inputs.len() < 2 {
        return Err(OnnxError::ShapeMismatch(
            "Sub f16: expected 2 inputs".to_string(),
        ));
    }
    let a = inputs[0];
    let b = inputs[1];

    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(|e| OnnxError::ShapeMismatch(format!("Sub f16 broadcast: {e}")))?;
    let out_size: usize = out_shape.iter().product();

    let data = if a.shape == b.shape {
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&va, &vb)| {
                let ha = half::f16::from_f32(va);
                let hb = half::f16::from_f32(vb);
                (ha - hb).to_f32()
            })
            .collect()
    } else {
        broadcast_binary_f16(
            &a.data,
            &a.shape,
            &b.data,
            &b.shape,
            &out_shape,
            out_size,
            |ha, hb| ha - hb,
        )
    };

    Ok(vec![Tensor::new(data, out_shape)])
}

fn execute_sigmoid_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Sigmoid f16: expected at least 1 input".to_string())
    })?;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| {
            // Compute sigmoid in f16: 1 / (1 + exp(-x))
            let h = half::f16::from_f32(v);
            let neg_h = -h;
            let exp_neg = half::f16::from_f32(neg_h.to_f32().exp());
            let one = half::f16::ONE;
            let denom = one + exp_neg;
            half::f16::from_f32(one.to_f32() / denom.to_f32()).to_f32()
        })
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

fn execute_tanh_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Tanh f16: expected at least 1 input".to_string())
    })?;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| {
            let h = half::f16::from_f32(v);
            half::f16::from_f32(h.to_f32().tanh()).to_f32()
        })
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

fn execute_neg_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Neg f16: expected at least 1 input".to_string())
    })?;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| (-half::f16::from_f32(v)).to_f32())
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

fn execute_abs_f16(inputs: &[&Tensor]) -> Result<Vec<Tensor>, OnnxError> {
    let input = inputs.first().ok_or_else(|| {
        OnnxError::ShapeMismatch("Abs f16: expected at least 1 input".to_string())
    })?;
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| {
            let h = half::f16::from_f32(v);
            half::f16::from_f32(h.to_f32().abs()).to_f32()
        })
        .collect();
    Ok(vec![Tensor::new(data, input.shape.clone())])
}

// ---------------------------------------------------------------------------
// Broadcasting helpers for f16 binary ops
// ---------------------------------------------------------------------------

/// Execute a binary op in f16 with broadcasting.
fn broadcast_binary_f16(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
    out_size: usize,
    op: impl Fn(half::f16, half::f16) -> half::f16,
) -> Vec<f32> {
    let a_strides = broadcast_strides(a_shape, out_shape);
    let b_strides = broadcast_strides(b_shape, out_shape);
    let out_strides = compute_row_major_strides(out_shape);
    let mut result = Vec::with_capacity(out_size);
    for i in 0..out_size {
        let a_idx = broadcast_flat_index(i, out_shape, &out_strides, &a_strides);
        let b_idx = broadcast_flat_index(i, out_shape, &out_strides, &b_strides);
        let ha = half::f16::from_f32(a_data[a_idx]);
        let hb = half::f16::from_f32(b_data[b_idx]);
        result.push(op(ha, hb).to_f32());
    }
    result
}

/// Compute C-order (row-major) strides from shape.
fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute effective strides for a tensor shape broadcast to `out_shape`.
/// Dimensions of size 1 get stride 0 (broadcast dimension).
fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let ndim = out_shape.len();
    let offset = ndim.saturating_sub(shape.len());
    let mut strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        if shape[i] == out_shape[i + offset] {
            strides[i + offset] = stride;
            stride = stride.saturating_mul(shape[i]);
        }
        // else: size 1 => stride stays 0 (broadcast)
    }
    strides
}

/// Convert a flat index in the output to a flat index in a broadcast-strided source tensor.
fn broadcast_flat_index(
    flat_idx: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    src_strides: &[usize],
) -> usize {
    let ndim = out_shape.len();
    let mut idx = 0usize;
    let mut remaining = flat_idx;
    for d in 0..ndim {
        let out_stride = out_strides[d];
        let coord = if out_stride > 0 {
            remaining / out_stride
        } else {
            0
        };
        remaining = if out_stride > 0 {
            remaining % out_stride
        } else {
            remaining
        };
        idx += coord * src_strides[d];
    }
    idx
}

/// Determine whether all downstream consumers of a node's outputs are f16-safe.
///
/// Used to decide whether intermediate results should be kept in f16 precision
/// or promoted back to f32.
pub fn next_consumers_all_f16(
    node_outputs: &[String],
    all_nodes: &[crate::graph::Node],
    current_node_idx: usize,
) -> bool {
    for output_name in node_outputs {
        if output_name.is_empty() {
            continue;
        }
        for node in all_nodes.iter().skip(current_node_idx + 1) {
            if node.inputs.contains(output_name) && !should_use_f16(node.op.as_str()) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_use_f16_activations() {
        assert!(should_use_f16("Relu"));
        assert!(should_use_f16("Add"));
        assert!(should_use_f16("Mul"));
        assert!(should_use_f16("Sub"));
        assert!(should_use_f16("Div"));
        assert!(should_use_f16("Sigmoid"));
        assert!(should_use_f16("Tanh"));
        assert!(should_use_f16("Gelu"));
        assert!(should_use_f16("SiLU"));
        assert!(should_use_f16("HardSigmoid"));
        assert!(should_use_f16("HardSwish"));
        assert!(should_use_f16("LeakyRelu"));
    }

    #[test]
    fn test_should_use_f16_normalization() {
        assert!(should_use_f16("LayerNormalization"));
        assert!(should_use_f16("LayerNorm"));
        assert!(should_use_f16("BatchNormalization"));
        assert!(should_use_f16("BatchNorm"));
        assert!(should_use_f16("GroupNormalization"));
        assert!(should_use_f16("GroupNorm"));
        assert!(should_use_f16("Softmax"));
        assert!(should_use_f16("LogSoftmax"));
    }

    #[test]
    fn test_should_use_f16_shape_ops() {
        assert!(should_use_f16("Identity"));
        assert!(should_use_f16("Reshape"));
        assert!(should_use_f16("Transpose"));
        assert!(should_use_f16("Concat"));
        assert!(should_use_f16("Slice"));
        assert!(should_use_f16("Split"));
        assert!(should_use_f16("Squeeze"));
        assert!(should_use_f16("Unsqueeze"));
        assert!(should_use_f16("Flatten"));
        assert!(should_use_f16("Expand"));
    }

    #[test]
    fn test_should_use_f16_attention() {
        assert!(should_use_f16("Attention"));
        assert!(should_use_f16("MultiHeadAttention"));
        assert!(should_use_f16("RotaryEmbedding"));
    }

    #[test]
    fn test_requires_f32_accumulation() {
        assert!(requires_f32("MatMul"));
        assert!(requires_f32("Gemm"));
        assert!(requires_f32("Conv"));
        assert!(requires_f32("ConvTranspose"));
        assert!(requires_f32("Einsum"));
    }

    #[test]
    fn test_requires_f32_reductions() {
        assert!(requires_f32("ReduceSum"));
        assert!(requires_f32("ReduceMean"));
        assert!(requires_f32("ReduceMax"));
        assert!(requires_f32("ReduceMin"));
        assert!(requires_f32("ReduceProd"));
    }

    #[test]
    fn test_requires_f32_precision_sensitive() {
        assert!(requires_f32("Pow"));
        assert!(requires_f32("Exp"));
        assert!(requires_f32("Log"));
    }

    #[test]
    fn test_f16_safe_not_f32_required() {
        // f16-safe ops should NOT be in the requires_f32 set
        assert!(!requires_f32("Relu"));
        assert!(!requires_f32("Add"));
        assert!(!requires_f32("Sigmoid"));
        assert!(!requires_f32("Identity"));
    }

    #[test]
    fn test_f32_required_not_f16_safe() {
        // f32-required ops should NOT be in the should_use_f16 set
        assert!(!should_use_f16("MatMul"));
        assert!(!should_use_f16("Gemm"));
        assert!(!should_use_f16("Conv"));
        assert!(!should_use_f16("Exp"));
        assert!(!should_use_f16("Log"));
        assert!(!should_use_f16("Pow"));
    }

    #[test]
    fn test_round_to_f16_precision() {
        let t = Tensor::new(vec![1.0, 0.1, 0.001, 100.0, -3.125], vec![5]);
        let rounded = round_to_f16_precision(&t);
        assert_eq!(rounded.shape, t.shape);
        // f16 can represent 1.0 and 100.0 exactly
        assert_eq!(rounded.data[0], 1.0);
        assert_eq!(rounded.data[3], 100.0);
        // 0.1 rounded to f16 ~= 0.0999755859375
        assert!((rounded.data[1] - 0.1).abs() < 0.001);
        // 0.001 rounded to f16 ~= 0.00099945068359375
        assert!((rounded.data[2] - 0.001).abs() < 0.0005);
        // -3.125 rounded to f16 exactly
        assert!((rounded.data[4] - (-3.125)).abs() < 0.01);
    }

    #[test]
    fn test_relu_f16() {
        let input = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let result = execute_elementwise_f16("Relu", &[&input])
            .expect("Relu should be supported")
            .expect("Relu should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_add_f16_same_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let result = execute_elementwise_f16("Add", &[&a, &b])
            .expect("Add should be supported")
            .expect("Add should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_mul_f16_same_shape() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let b = Tensor::new(vec![10.0, 10.0, 10.0], vec![3]);
        let result = execute_elementwise_f16("Mul", &[&a, &b])
            .expect("Mul should be supported")
            .expect("Mul should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_sub_f16_same_shape() {
        let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let result = execute_elementwise_f16("Sub", &[&a, &b])
            .expect("Sub should be supported")
            .expect("Sub should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_add_f16_broadcast() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let result = execute_elementwise_f16("Add", &[&a, &b])
            .expect("Add should be supported")
            .expect("Add should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape, vec![2, 3]);
        assert_eq!(result[0].data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_sigmoid_f16() {
        let input = Tensor::new(vec![0.0], vec![1]);
        let result = execute_elementwise_f16("Sigmoid", &[&input])
            .expect("Sigmoid should be supported")
            .expect("Sigmoid should succeed");
        // sigmoid(0) = 0.5
        assert!((result[0].data[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tanh_f16() {
        let input = Tensor::new(vec![0.0], vec![1]);
        let result = execute_elementwise_f16("Tanh", &[&input])
            .expect("Tanh should be supported")
            .expect("Tanh should succeed");
        // tanh(0) = 0.0
        assert!((result[0].data[0]).abs() < 0.001);
    }

    #[test]
    fn test_neg_f16() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]);
        let result = execute_elementwise_f16("Neg", &[&input])
            .expect("Neg should be supported")
            .expect("Neg should succeed");
        assert_eq!(result[0].data, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_abs_f16() {
        let input = Tensor::new(vec![-1.0, 2.0, -3.0], vec![3]);
        let result = execute_elementwise_f16("Abs", &[&input])
            .expect("Abs should be supported")
            .expect("Abs should succeed");
        assert_eq!(result[0].data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unsupported_op_returns_none() {
        let input = Tensor::new(vec![1.0], vec![1]);
        assert!(execute_elementwise_f16("MatMul", &[&input]).is_none());
        assert!(execute_elementwise_f16("Conv", &[&input]).is_none());
        assert!(execute_elementwise_f16("ReduceSum", &[&input]).is_none());
    }

    #[test]
    fn test_f16_precision_loss() {
        // f16 has ~3.3 decimal digits of precision
        // Values like 1024.5 can be represented, but 1024.001 cannot
        let t = Tensor::new(vec![1024.001], vec![1]);
        let rounded = round_to_f16_precision(&t);
        // f16 rounds 1024.001 to 1024.0
        assert_eq!(rounded.data[0], 1024.0);
    }

    #[test]
    fn test_next_consumers_all_f16() {
        use crate::graph::{Attributes, Node, OpKind};

        let nodes = vec![
            Node {
                op: OpKind::Relu,
                name: "relu1".to_string(),
                inputs: vec!["input".to_string()],
                outputs: vec!["relu_out".to_string()],
                attrs: Attributes::default(),
            },
            Node {
                op: OpKind::Add,
                name: "add1".to_string(),
                inputs: vec!["relu_out".to_string(), "bias".to_string()],
                outputs: vec!["add_out".to_string()],
                attrs: Attributes::default(),
            },
            Node {
                op: OpKind::MatMul,
                name: "matmul1".to_string(),
                inputs: vec!["add_out".to_string(), "weight".to_string()],
                outputs: vec!["mm_out".to_string()],
                attrs: Attributes::default(),
            },
        ];

        // relu_out is consumed by Add (f16-safe) => true
        assert!(next_consumers_all_f16(&["relu_out".to_string()], &nodes, 0,));

        // add_out is consumed by MatMul (f32-required) => false
        assert!(!next_consumers_all_f16(&["add_out".to_string()], &nodes, 1,));
    }

    #[test]
    fn test_broadcast_strides_same_shape() {
        let strides = broadcast_strides(&[2, 3], &[2, 3]);
        assert_eq!(strides, vec![3, 1]);
    }

    #[test]
    fn test_broadcast_strides_broadcast_dim() {
        // [1, 3] broadcast to [2, 3]
        let strides = broadcast_strides(&[1, 3], &[2, 3]);
        // dim 0 is broadcast (size 1), so stride is 0
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn test_broadcast_strides_leading_dims() {
        // [3] broadcast to [2, 3]
        let strides = broadcast_strides(&[3], &[2, 3]);
        assert_eq!(strides, vec![0, 1]);
    }
}
