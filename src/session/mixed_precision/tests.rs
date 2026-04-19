//! Tests for the mixed precision module.

use super::broadcast::{broadcast_strides, compute_row_major_strides};
use super::classify::{requires_f32, should_use_f16};
use super::elementwise::execute_elementwise_f16;
use super::precision::{next_consumers_all_f16, round_to_f16_precision};
use crate::tensor::Tensor;

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
