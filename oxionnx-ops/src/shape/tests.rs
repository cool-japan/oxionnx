//! Tests for shape module operations.

use super::*;
use oxionnx_core::OnnxError;
use oxionnx_core::Tensor;

#[test]
fn test_reshape() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0; 6], vec![2, 3]);
    let y = reshape(&x, &[3, 2], false)?;
    assert_eq!(y.shape, vec![3, 2]);
    Ok(())
}

#[test]
fn test_reshape_neg1() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0; 6], vec![6]);
    let y = reshape(&x, &[2, -1], false)?;
    assert_eq!(y.shape, vec![2, 3]);
    Ok(())
}

#[test]
fn test_reshape_allowzero_default_copies_input_dim() -> Result<(), OnnxError> {
    // allowzero=0 (default): a 0 copies the input dimension at the same index.
    // input [2,3,4], shape [2,0,4] -> [2,3,4].
    let resolved = resolve_reshape(&[2, 3, 4], 24, &[2, 0, 4], false)?;
    assert_eq!(resolved, vec![2, 3, 4]);
    Ok(())
}

#[test]
fn test_reshape_allowzero_default_copy_with_infer() -> Result<(), OnnxError> {
    // allowzero=0: 0 copies input dim 0 (=2), -1 infers the rest. input [2,3,4]=24 elems.
    // shape [0,-1] -> [2, 12].
    let resolved = resolve_reshape(&[2, 3, 4], 24, &[0, -1], false)?;
    assert_eq!(resolved, vec![2, 12]);
    Ok(())
}

#[test]
fn test_reshape_allowzero_true_literal_zero() -> Result<(), OnnxError> {
    // allowzero=1: a 0 is a literal zero-size dimension (NOT copied from input).
    // input [2,0,4] (0 elements), shape [2,0,4] -> literal [2,0,4].
    let resolved = resolve_reshape(&[2, 0, 4], 0, &[2, 0, 4], true)?;
    assert_eq!(resolved, vec![2, 0, 4]);
    Ok(())
}

#[test]
fn test_reshape_allowzero_true_neg1_and_zero_is_error() {
    // allowzero=1: combining -1 with an explicit 0 is ambiguous -> Err.
    let err = resolve_reshape(&[2, 0, 4], 0, &[0, -1], true);
    assert!(err.is_err());
}

#[test]
fn test_reshape_allowzero_true_infer_no_zero() -> Result<(), OnnxError> {
    // allowzero=1 with no zero present behaves identically to the default path.
    // input [2,3,4]=24 elems, shape [-1,4] -> [6,4].
    let resolved = resolve_reshape(&[2, 3, 4], 24, &[-1, 4], true)?;
    assert_eq!(resolved, vec![6, 4]);
    Ok(())
}

#[test]
fn test_reshape_neg1_inference_unchanged() -> Result<(), OnnxError> {
    // Regression: existing -1 inference is unchanged. input [2,3,4]=24, shape [-1,4] -> [6,4].
    let x = Tensor::new(vec![1.0; 24], vec![2, 3, 4]);
    let y = reshape(&x, &[-1, 4], false)?;
    assert_eq!(y.shape, vec![6, 4]);
    Ok(())
}

#[test]
fn test_reshape_op_allowzero_default_path() -> Result<(), OnnxError> {
    // Op-level regression: ReshapeOp::execute with allowzero absent (default 0) and a 0
    // in the shape tensor copies the input dim. input [2,3,4], shape [2,0,4] -> [2,3,4].
    use crate::registry::shape_ops::ReshapeOp;
    use oxionnx_core::graph::{Attributes, Node, OpKind};
    use oxionnx_core::{OpContext, Operator};

    let x = Tensor::new(vec![1.0; 24], vec![2, 3, 4]);
    let shape_t = Tensor::new(vec![2.0, 0.0, 4.0], vec![3]);
    let node = Node {
        op: OpKind::Reshape,
        name: "reshape_test".to_string(),
        inputs: vec![],
        outputs: vec![],
        attrs: Attributes::default(),
    };
    let ctx = OpContext {
        node: &node,
        inputs: vec![Some(&x), Some(&shape_t)],
        outer_scope: None,
        weights: None,
        registry: None,
    };
    let out = ReshapeOp.execute(&ctx)?;
    assert_eq!(out[0].shape, vec![2, 3, 4]);
    Ok(())
}

#[test]
fn test_transpose_2d() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let y = transpose(&x, &[1, 0])?;
    assert_eq!(y.shape, vec![3, 2]);
    assert_eq!(y.data[0], 1.0);
    assert_eq!(y.data[1], 4.0);
    assert_eq!(y.data[2], 2.0);
    Ok(())
}

#[test]
fn test_squeeze_unsqueeze() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
    let sq = squeeze(&x, &[]);
    assert_eq!(sq.shape, vec![3]);
    let un = unsqueeze(&sq, &[0, 2]);
    assert_eq!(un.shape, vec![1, 3, 1]);
}

#[test]
fn test_concat() -> Result<(), OnnxError> {
    let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    let b = Tensor::new(vec![3.0, 4.0], vec![1, 2]);
    let c = concat(&[&a, &b], 0)?;
    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_slice() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5]);
    let y = slice(&x, &[1], &[4], None, None)?;
    assert_eq!(y.data, vec![1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn test_pad_constant() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    // pad 1 on all sides: pads = [1, 1, 1, 1]
    let y = pad(&x, &[1, 1, 1, 1], "constant", 0.0);
    assert_eq!(y.shape, vec![4, 4]);
    // center should be [1,2,3,4], rest 0
    assert_eq!(y.data[0], 0.0); // top-left corner
    assert_eq!(y.data[5], 1.0); // (1,1) = first element
    assert_eq!(y.data[6], 2.0); // (1,2)
    assert_eq!(y.data[9], 3.0); // (2,1)
    assert_eq!(y.data[10], 4.0); // (2,2)
}

#[test]
fn test_split_equal() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let chunks = split(&x, 1, &[1, 2])?;
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].shape, vec![2, 1]);
    assert_eq!(chunks[1].shape, vec![2, 2]);
    assert_eq!(chunks[0].data, vec![1.0, 4.0]);
    assert_eq!(chunks[1].data, vec![2.0, 3.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn test_tile() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let y = tile(&x, &[2, 1])?;
    assert_eq!(y.shape, vec![2, 3]);
    assert_eq!(y.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn test_tile_2d() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = tile(&x, &[1, 2])?;
    assert_eq!(y.shape, vec![2, 4]);
    assert_eq!(y.data, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_pad_reflect() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    // pad 1 on left and right of dim 1: pads = [0, 1, 0, 1]
    let y = pad(&x, &[0, 1, 0, 1], "reflect", 0.0);
    assert_eq!(y.shape, vec![2, 5]);
    // row 0: reflect [1,2,3] with pad 1 left and 1 right -> [2, 1, 2, 3, 2]
    assert_eq!(y.data[0], 2.0);
    assert_eq!(y.data[1], 1.0);
    assert_eq!(y.data[2], 2.0);
    assert_eq!(y.data[3], 3.0);
    assert_eq!(y.data[4], 2.0);
}

#[test]
fn test_depth_to_space() {
    // [1, 4, 1, 1] with blocksize=2 -> [1, 1, 2, 2]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1, 1]);
    let out = depth_to_space(&x, 2, "DCR").expect("depth_to_space DCR failed");
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
}

#[test]
fn test_depth_to_space_crd() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1, 1]);
    let out = depth_to_space(&x, 2, "CRD").expect("depth_to_space CRD failed");
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // CRD: [ci*r*r + rh*r + rw] maps directly
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_space_to_depth() {
    // [1, 1, 2, 2] with blocksize=2 -> [1, 4, 1, 1]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let out = space_to_depth(&x, 2).expect("space_to_depth failed");
    assert_eq!(out.shape, vec![1, 4, 1, 1]);
}

#[test]
fn test_depth_to_space_roundtrip() {
    let x = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 4, 2, 2]);
    let d2s = depth_to_space(&x, 2, "CRD").expect("d2s failed");
    let s2d = space_to_depth(&d2s, 2).expect("s2d failed");
    assert_eq!(x.shape, s2d.shape);
    // Values should roundtrip for CRD mode
    for (a, b) in x.data.iter().zip(s2d.data.iter()) {
        assert!((a - b).abs() < 1e-6, "roundtrip mismatch: {a} vs {b}");
    }
}

#[test]
fn test_depth_to_space_errors() {
    let x3d = Tensor::new(vec![1.0; 8], vec![2, 2, 2]);
    assert!(depth_to_space(&x3d, 2, "DCR").is_err());

    let x = Tensor::new(vec![1.0; 6], vec![1, 6, 1, 1]);
    assert!(depth_to_space(&x, 2, "DCR").is_err()); // 6 % 4 != 0
}

#[test]
fn test_space_to_depth_errors() {
    let x3d = Tensor::new(vec![1.0; 8], vec![2, 2, 2]);
    assert!(space_to_depth(&x3d, 2).is_err());

    let x = Tensor::new(vec![1.0; 6], vec![1, 1, 3, 2]);
    assert!(space_to_depth(&x, 2).is_err()); // 3 % 2 != 0
}

#[test]
fn test_reverse_sequence() {
    // [2, 4] tensor, batch_axis=0, time_axis=1
    // batch 0: reverse first 3 elements, batch 1: reverse first 2 elements
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let seq_lens = Tensor::new(vec![3.0, 2.0], vec![2]);
    let out = reverse_sequence(&x, &seq_lens, 0, 1).expect("reverse_sequence failed");
    assert_eq!(out.shape, vec![2, 4]);
    // batch 0: [3,2,1,4], batch 1: [6,5,7,8]
    assert_eq!(out.data, vec![3.0, 2.0, 1.0, 4.0, 6.0, 5.0, 7.0, 8.0]);
}

#[test]
fn test_reverse_sequence_full() {
    // Reverse entire sequence
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let seq_lens = Tensor::new(vec![4.0], vec![1]);
    let out = reverse_sequence(&x, &seq_lens, 0, 1).expect("reverse_sequence failed");
    assert_eq!(out.data, vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_reverse_sequence_errors() {
    let x = Tensor::new(vec![1.0], vec![1]);
    let seq_lens = Tensor::new(vec![1.0], vec![1]);
    assert!(reverse_sequence(&x, &seq_lens, 0, 1).is_err()); // 1D input

    let x2 = Tensor::new(vec![1.0; 4], vec![2, 2]);
    assert!(reverse_sequence(&x2, &seq_lens, 0, 0).is_err()); // same axis
    assert!(reverse_sequence(&x2, &seq_lens, 0, 1).is_err()); // seq_lens len mismatch
}
