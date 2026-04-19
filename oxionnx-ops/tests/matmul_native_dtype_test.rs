//! Tests for MatMulOp native typed dispatch (Phase D.3).
//!
//! Covers F32, I8, I32, F16, BF16 typed dispatch paths.

use oxionnx_core::{
    dtype::{DType, TensorStorage, TypedTensor},
    graph::{Attributes, Node, OpKind},
    operator::{Operator, TypedOpContext},
};
use oxionnx_ops::registry::math_ops::MatMulOp;

// ── Test infrastructure ──────────────────────────────────────────────────────

fn matmul_node() -> Node {
    Node {
        name: "test_matmul".into(),
        op: OpKind::MatMul,
        inputs: vec![],
        outputs: vec![],
        attrs: Attributes::default(),
    }
}

fn make_typed_ctx<'a>(
    node: &'a Node,
    a: &'a TypedTensor,
    b: &'a TypedTensor,
) -> TypedOpContext<'a> {
    TypedOpContext {
        node,
        inputs: vec![Some(a), Some(b)],
        outer_scope: None,
        registry: None,
    }
}

fn f32_to_f16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

fn f32_to_bf16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

fn f16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::bf16::from_bits(b).to_f32())
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// F32 dispatch: verify DType::F32 is in native_dtypes(), and execute_typed
/// gives identical results to execute for a 2×3 @ 3×2 matmul.
#[test]
fn test_matmul_f32_baseline() {
    // Verify native_dtypes contains F32
    assert!(
        MatMulOp.native_dtypes().contains(&DType::F32),
        "DType::F32 must be in native_dtypes()"
    );

    let a_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    // [2,3] @ [3,2] = [2,2]
    let a = TypedTensor::new(TensorStorage::F32(a_vals.clone()), vec![2, 3]);
    let b = TypedTensor::new(TensorStorage::F32(b_vals.clone()), vec![3, 2]);

    let node = matmul_node();
    let ctx = make_typed_ctx(&node, &a, &b);
    let result = MatMulOp
        .execute_typed(&ctx)
        .expect("execute_typed F32 failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].dtype(), DType::F32);
    assert_eq!(result[0].shape, vec![2, 2]);

    // Reference: [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]]
    // row0: 1*7+2*9+3*11=7+18+33=58, 1*8+2*10+3*12=8+20+36=64
    // row1: 4*7+5*9+6*11=28+45+66=139, 4*8+5*10+6*12=32+50+72=154
    let out_vals = result[0].storage.to_f32_vec();
    assert!(
        (out_vals[0] - 58.0).abs() < 1e-4,
        "F32 [0,0] expected 58.0, got {}",
        out_vals[0]
    );
    assert!(
        (out_vals[1] - 64.0).abs() < 1e-4,
        "F32 [0,1] expected 64.0, got {}",
        out_vals[1]
    );
    assert!(
        (out_vals[2] - 139.0).abs() < 1e-4,
        "F32 [1,0] expected 139.0, got {}",
        out_vals[2]
    );
    assert!(
        (out_vals[3] - 154.0).abs() < 1e-4,
        "F32 [1,1] expected 154.0, got {}",
        out_vals[3]
    );
}

/// I8 dispatch: [[1i8, 2, 3]] @ [[4i8], [5], [6]] = [[1*4+2*5+3*6]] = [[32i32]]
#[test]
fn test_matmul_i8_i32_against_reference() {
    assert!(
        MatMulOp.native_dtypes().contains(&DType::I8),
        "DType::I8 must be in native_dtypes()"
    );

    // [1,3] @ [3,1] → [1,1]
    let a = TypedTensor::new(TensorStorage::I8(vec![1i8, 2, 3]), vec![1, 3]);
    let b = TypedTensor::new(TensorStorage::I8(vec![4i8, 5, 6]), vec![3, 1]);

    let node = matmul_node();
    let ctx = make_typed_ctx(&node, &a, &b);
    let result = MatMulOp
        .execute_typed(&ctx)
        .expect("execute_typed I8 failed");

    assert_eq!(result.len(), 1);
    // Output dtype should be I32 (standard quantized matmul accumulator)
    assert_eq!(result[0].dtype(), DType::I32, "I8@I8 output must be I32");
    assert_eq!(result[0].shape, vec![1, 1]);

    if let TensorStorage::I32(ref data) = result[0].storage {
        assert_eq!(data[0], 32i32, "1*4+2*5+3*6=32");
    } else {
        panic!("Expected I32 storage, got {:?}", result[0].dtype());
    }
}

/// I8 dispatch: larger 2×3 @ 3×2 with known reference.
#[test]
fn test_matmul_i8_i32_2x3_3x2() {
    // [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,0]]
    // row0: 1+0+3=4, 0+2+0=2
    // row1: 4+0+6=10, 0+5+0=5
    let a = TypedTensor::new(TensorStorage::I8(vec![1i8, 2, 3, 4, 5, 6]), vec![2, 3]);
    let b = TypedTensor::new(TensorStorage::I8(vec![1i8, 0, 0, 1, 1, 0]), vec![3, 2]);

    let node = matmul_node();
    let ctx = make_typed_ctx(&node, &a, &b);
    let result = MatMulOp
        .execute_typed(&ctx)
        .expect("execute_typed I8 2x3@3x2 failed");

    assert_eq!(result[0].dtype(), DType::I32);
    assert_eq!(result[0].shape, vec![2, 2]);

    if let TensorStorage::I32(ref data) = result[0].storage {
        assert_eq!(data[0], 4i32, "[0,0]");
        assert_eq!(data[1], 2i32, "[0,1]");
        assert_eq!(data[2], 10i32, "[1,0]");
        assert_eq!(data[3], 5i32, "[1,1]");
    } else {
        panic!("Expected I32 storage");
    }
}

/// I32 dispatch: verify I32@I32→I32 correctness.
#[test]
fn test_matmul_i32_i32() {
    assert!(
        MatMulOp.native_dtypes().contains(&DType::I32),
        "DType::I32 must be in native_dtypes()"
    );

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = TypedTensor::new(TensorStorage::I32(vec![1i32, 2, 3, 4]), vec![2, 2]);
    let b = TypedTensor::new(TensorStorage::I32(vec![5i32, 6, 7, 8]), vec![2, 2]);

    let node = matmul_node();
    let ctx = make_typed_ctx(&node, &a, &b);
    let result = MatMulOp
        .execute_typed(&ctx)
        .expect("execute_typed I32 failed");

    assert_eq!(result[0].dtype(), DType::I32);
    assert_eq!(result[0].shape, vec![2, 2]);

    if let TensorStorage::I32(ref data) = result[0].storage {
        assert_eq!(data[0], 19i32, "[0,0]");
        assert_eq!(data[1], 22i32, "[0,1]");
        assert_eq!(data[2], 43i32, "[1,0]");
        assert_eq!(data[3], 50i32, "[1,1]");
    } else {
        panic!("Expected I32 storage");
    }
}

/// F16 dispatch: numerical parity against f32 reference (allow relative error < 1e-2).
#[test]
fn test_matmul_f16_numerical_parity() {
    assert!(
        MatMulOp.native_dtypes().contains(&DType::F16),
        "DType::F16 must be in native_dtypes()"
    );

    // [2,3] @ [3,2] = [2,2] with small values to keep f16 precision acceptable
    let a_f32 = vec![1.0f32, 0.5, -1.0, 2.0, -0.5, 0.25];
    let b_f32 = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, -1.0];

    // Reference result in f32
    let a_ref = oxionnx_core::Tensor::new(a_f32.clone(), vec![2, 3]);
    let b_ref = oxionnx_core::Tensor::new(b_f32.clone(), vec![3, 2]);
    let ref_f32 = oxionnx_ops::math::matmul(&a_ref, &b_ref).expect("f32 reference matmul failed");

    let a = TypedTensor::new(TensorStorage::F16(f32_to_f16_bits(&a_f32)), vec![2, 3]);
    let b = TypedTensor::new(TensorStorage::F16(f32_to_f16_bits(&b_f32)), vec![3, 2]);

    let node = matmul_node();
    let ctx = make_typed_ctx(&node, &a, &b);
    let result = MatMulOp
        .execute_typed(&ctx)
        .expect("execute_typed F16 failed");

    assert_eq!(result[0].dtype(), DType::F16);
    assert_eq!(result[0].shape, vec![2, 2]);

    if let TensorStorage::F16(ref bits) = result[0].storage {
        let got = f16_bits_to_f32(bits);
        for (i, (&g, &r)) in got.iter().zip(ref_f32.data.iter()).enumerate() {
            let rel_err = if r.abs() > 1e-6 {
                (g - r).abs() / r.abs()
            } else {
                (g - r).abs()
            };
            assert!(
                rel_err < 1e-2,
                "F16 output[{i}]: got {g}, reference {r}, rel_err {rel_err}"
            );
        }
    } else {
        panic!("Expected F16 storage");
    }
}

/// BF16 dispatch: numerical parity against f32 reference (allow relative error < 1e-2).
#[test]
fn test_matmul_bf16_numerical_parity() {
    assert!(
        MatMulOp.native_dtypes().contains(&DType::BF16),
        "DType::BF16 must be in native_dtypes()"
    );

    // [2,3] @ [3,2]
    let a_f32 = vec![1.0f32, 0.5, -1.0, 2.0, -0.5, 0.25];
    let b_f32 = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, -1.0];

    // Reference result in f32
    let a_ref = oxionnx_core::Tensor::new(a_f32.clone(), vec![2, 3]);
    let b_ref = oxionnx_core::Tensor::new(b_f32.clone(), vec![3, 2]);
    let ref_f32 = oxionnx_ops::math::matmul(&a_ref, &b_ref).expect("f32 reference matmul failed");

    let a = TypedTensor::new(TensorStorage::BF16(f32_to_bf16_bits(&a_f32)), vec![2, 3]);
    let b = TypedTensor::new(TensorStorage::BF16(f32_to_bf16_bits(&b_f32)), vec![3, 2]);

    let node = matmul_node();
    let ctx = make_typed_ctx(&node, &a, &b);
    let result = MatMulOp
        .execute_typed(&ctx)
        .expect("execute_typed BF16 failed");

    assert_eq!(result[0].dtype(), DType::BF16);
    assert_eq!(result[0].shape, vec![2, 2]);

    if let TensorStorage::BF16(ref bits) = result[0].storage {
        let got = bf16_bits_to_f32(bits);
        for (i, (&g, &r)) in got.iter().zip(ref_f32.data.iter()).enumerate() {
            let rel_err = if r.abs() > 1e-6 {
                (g - r).abs() / r.abs()
            } else {
                (g - r).abs()
            };
            assert!(
                rel_err < 1e-2,
                "BF16 output[{i}]: got {g}, reference {r}, rel_err {rel_err}"
            );
        }
    } else {
        panic!("Expected BF16 storage");
    }
}

/// All declared native_dtypes are present.
#[test]
fn test_native_dtypes_coverage() {
    let dtypes = MatMulOp.native_dtypes();
    for &expected in &[DType::F32, DType::F16, DType::BF16, DType::I8, DType::I32] {
        assert!(
            dtypes.contains(&expected),
            "{:?} must be in native_dtypes()",
            expected
        );
    }
}

/// Batched I8 matmul: batch=2 [2,2,3] @ [2,3,2] each slice independent.
#[test]
fn test_matmul_i8_batched() {
    // Two independent 2x3 @ 3x2 slices
    // Slice 0: [[1,0,0],[0,1,0]] @ [[1,2],[3,4],[5,6]] = [[1,2],[3,4]]
    // Slice 1: [[1,1,0],[0,0,1]] @ [[1,2],[3,4],[5,6]] = [[4,6],[5,6]]
    let a_data: Vec<i8> = vec![1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1];
    let b_data: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6];
    let a = TypedTensor::new(TensorStorage::I8(a_data), vec![2, 2, 3]);
    let b = TypedTensor::new(TensorStorage::I8(b_data), vec![2, 3, 2]);

    let node = matmul_node();
    let ctx = make_typed_ctx(&node, &a, &b);
    let result = MatMulOp
        .execute_typed(&ctx)
        .expect("batched I8 matmul failed");

    assert_eq!(result[0].shape, vec![2, 2, 2]);
    assert_eq!(result[0].dtype(), DType::I32);

    if let TensorStorage::I32(ref data) = result[0].storage {
        // Slice 0: [0][0,0]=1, [0][0,1]=2, [0][1,0]=3, [0][1,1]=4
        assert_eq!(data[0], 1i32, "batch0[0,0]");
        assert_eq!(data[1], 2i32, "batch0[0,1]");
        assert_eq!(data[2], 3i32, "batch0[1,0]");
        assert_eq!(data[3], 4i32, "batch0[1,1]");
        // Slice 1: [[1,1,0],[0,0,1]] @ [[1,2],[3,4],[5,6]]
        // row0: 1+3=4, 2+4=6  row1: 0+0+5=5, 0+0+6=6
        assert_eq!(data[4], 4i32, "batch1[0,0]");
        assert_eq!(data[5], 6i32, "batch1[0,1]");
        assert_eq!(data[6], 5i32, "batch1[1,0]");
        assert_eq!(data[7], 6i32, "batch1[1,1]");
    } else {
        panic!("Expected I32 storage");
    }
}
