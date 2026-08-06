//! Tests for `QLinearConvOp` / `QLinearMatMulOp` native typed dispatch
//! (`native_dtypes() = [I8, U8, I32, F32]`).
//!
//! `execute_typed` is the one path in this dtype-erased runtime that can
//! resolve `SatRange::infer`'s documented union-range ambiguity *exactly*:
//! when `y_zero_point` arrives as a genuinely typed `I8`/`U8` `TypedTensor`
//! (a `Session::run_typed` graph input, or an upstream operator's native
//! typed output) rather than as an f32-lane `Tensor`, its declared dtype
//! pins the output saturation range instead of falling back to `infer`'s
//! value-based cascade. See `oxionnx-ops/src/registry/quant_ops/mod.rs`'s
//! `SatRange::for_dtype` doc comment.
//!
//! The companion end-to-end test
//! `tests/w2_quantized_ops_e2e.rs::qlinear_matmul_ambiguous_zero_points_use_the_union_range`
//! (workspace root, not this crate) exercises the *untyped* `Session::run` /
//! `Operator::execute` path on this exact fixture and needs no change: that
//! path never carries a dtype tag, so it still — correctly — falls back to
//! the union range. This file pins the complementary, exact behaviour
//! `execute_typed` newly provides, and reuses that same test's fixture so
//! the three outcomes can be compared directly.

use oxionnx_core::{
    dtype::{DType, TensorStorage, TypedTensor},
    graph::{Attributes, Node, OpKind},
    operator::{Operator, TypedOpContext},
};
use oxionnx_ops::registry::quant_ops::{QLinearConvOp, QLinearMatMulOp};

// ── shared fixture ───────────────────────────────────────────────────────────
//
// `a=[[10,0],[0,10]]`, `b=[[10,-15],[0,0]]`, every scale `1.0`, every zero
// point `0` except `y_zero_point`. `acc = a @ b = [[100,-150],[0,0]]`, so
// pre-saturation values are `acc + y_zero_point = [200,-50,100,100]`. Three
// candidate ranges disagree on the two extremes:
//
// | range           | `[0,0]` | `[0,1]` | `[1,0]` | `[1,1]` |
// |-----------------|---------|---------|---------|---------|
// | union `[-128,255]` | 200 | -50 | 100 | 100 |
// | int8  `[-128,127]` | 127 | -50 | 100 | 100 |
// | uint8 `[0,255]`     | 200 |   0 | 100 | 100 |
//
// The union row is exactly `w2_quantized_ops_e2e.rs`'s pinned f32-path value.

fn f32_t(data: Vec<f32>, shape: Vec<usize>) -> TypedTensor {
    TypedTensor::new(TensorStorage::F32(data), shape)
}
fn i8_t(data: Vec<i8>, shape: Vec<usize>) -> TypedTensor {
    TypedTensor::new(TensorStorage::I8(data), shape)
}
fn u8_t(data: Vec<u8>, shape: Vec<usize>) -> TypedTensor {
    TypedTensor::new(TensorStorage::U8(data), shape)
}

/// Run `QLinearMatMulOp::execute_typed` on the shared fixture, varying only
/// `a`, `b` and `y_zero_point`'s storage (every scale is an F32 scalar
/// `1.0`, `a_zero_point`/`b_zero_point` are F32 scalar `0.0` — their dtype
/// never matters, only their *value*, since `SatRange::infer`'s fallback
/// reads decoded `i32` lanes, not a `TypedTensor::dtype()`).
fn run_qlinear_matmul(a: TypedTensor, b: TypedTensor, y_zero_point: TypedTensor) -> TypedTensor {
    let a_scale = f32_t(vec![1.0], vec![1]);
    let a_zp = f32_t(vec![0.0], vec![1]);
    let b_scale = f32_t(vec![1.0], vec![1]);
    let b_zp = f32_t(vec![0.0], vec![1]);
    let y_scale = f32_t(vec![1.0], vec![1]);

    let node = Node {
        name: "qlinear_matmul_test".into(),
        op: OpKind::QLinearMatMul,
        inputs: vec![],
        outputs: vec![],
        attrs: Attributes::default(),
    };
    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![
            Some(&a),
            Some(&a_scale),
            Some(&a_zp),
            Some(&b),
            Some(&b_scale),
            Some(&b_zp),
            Some(&y_scale),
            Some(&y_zero_point),
        ],
        outer_scope: None,
        registry: None,
    };
    let mut results = QLinearMatMulOp
        .execute_typed(&ctx)
        .expect("execute_typed QLinearMatMul");
    assert_eq!(results.len(), 1);
    results.remove(0)
}

/// Run `QLinearConvOp::execute_typed` on a 1x1-kernel fixture chosen to
/// reduce to the same arithmetic as the matmul fixture's first row: `x =
/// [10,-15]` (shape `[1,1,1,2]`), `w = [10]` (shape `[1,1,1,1]`), stride 1,
/// no padding, no bias, every scale `1.0`, `x_zero_point`/`w_zero_point`
/// `0`. `acc = [x[0]*w[0], x[1]*w[0]] = [100,-150]`, so pre-saturation
/// values are `acc + y_zero_point = [200,-50]` — the first row of the
/// matmul fixture's table above.
fn run_qlinear_conv(x: TypedTensor, w: TypedTensor, y_zero_point: TypedTensor) -> TypedTensor {
    let x_scale = f32_t(vec![1.0], vec![1]);
    let x_zp = f32_t(vec![0.0], vec![1]);
    let w_scale = f32_t(vec![1.0], vec![1]);
    let w_zp = f32_t(vec![0.0], vec![1]);
    let y_scale = f32_t(vec![1.0], vec![1]);

    let node = Node {
        name: "qlinear_conv_test".into(),
        op: OpKind::QLinearConv,
        inputs: vec![],
        outputs: vec![],
        attrs: Attributes::default(),
    };
    let ctx = TypedOpContext {
        node: &node,
        // No bias: 8 slots, `ctx.input(8)` safely reads back `None`.
        inputs: vec![
            Some(&x),
            Some(&x_scale),
            Some(&x_zp),
            Some(&w),
            Some(&w_scale),
            Some(&w_zp),
            Some(&y_scale),
            Some(&y_zero_point),
        ],
        outer_scope: None,
        registry: None,
    };
    let mut results = QLinearConvOp
        .execute_typed(&ctx)
        .expect("execute_typed QLinearConv");
    assert_eq!(results.len(), 1);
    results.remove(0)
}

// ── native_dtypes() coverage ─────────────────────────────────────────────────

#[test]
fn test_qlinear_conv_native_dtypes_coverage() {
    let dtypes = QLinearConvOp.native_dtypes();
    for expected in [DType::I8, DType::U8, DType::I32, DType::F32] {
        assert!(dtypes.contains(&expected), "{expected:?} must be native");
    }
}

#[test]
fn test_qlinear_matmul_native_dtypes_coverage() {
    let dtypes = QLinearMatMulOp.native_dtypes();
    for expected in [DType::I8, DType::U8, DType::I32, DType::F32] {
        assert!(dtypes.contains(&expected), "{expected:?} must be native");
    }
}

// ── QLinearMatMul: dtype-aware saturation ────────────────────────────────────

/// `y_zero_point` tagged `F32` (the same shape every value has on the
/// untyped `execute` path) still can't disambiguate `int8` from `uint8`, so
/// `execute_typed` must fall back to `SatRange::infer` and land on the union
/// — bit-identical to `Operator::execute`'s result on the equivalent f32
/// input (see `w2_quantized_ops_e2e.rs`'s pinned test).
#[test]
fn test_qlinear_matmul_f32_y_zero_point_falls_back_to_union_range() {
    let a = f32_t(vec![10.0, 0.0, 0.0, 10.0], vec![2, 2]);
    let b = f32_t(vec![10.0, -15.0, 0.0, 0.0], vec![2, 2]);
    let y_zp = f32_t(vec![100.0], vec![1]);

    let out = run_qlinear_matmul(a, b, y_zp);
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.storage.to_f32_vec(), vec![200.0, -50.0, 100.0, 100.0]);
}

/// `y_zero_point` tagged `I8` resolves the ambiguity exactly: `200` clips to
/// `127`, `-50` does not clip (it is inside `[-128,127]`).
#[test]
fn test_qlinear_matmul_i8_y_zero_point_saturates_to_int8_exactly() {
    let a = i8_t(vec![10, 0, 0, 10], vec![2, 2]);
    let b = i8_t(vec![10, -15, 0, 0], vec![2, 2]);
    let y_zp = i8_t(vec![100], vec![1]);

    let out = run_qlinear_matmul(a, b, y_zp);
    assert_eq!(out.storage.to_f32_vec(), vec![127.0, -50.0, 100.0, 100.0]);
}

/// `y_zero_point` tagged `U8` resolves the ambiguity the other way: `-50`
/// clips to `0`, `200` does not clip (it is inside `[0,255]`). `b` must
/// stay signed (`-15` is not representable as `u8`) — legal per spec, since
/// `a`'s dtype (`T1`), `b`'s dtype (`T2`) and `y`'s dtype (`T3`) are
/// independent type parameters.
#[test]
fn test_qlinear_matmul_u8_y_zero_point_saturates_to_uint8_exactly() {
    let a = u8_t(vec![10, 0, 0, 10], vec![2, 2]);
    let b = i8_t(vec![10, -15, 0, 0], vec![2, 2]);
    let y_zp = u8_t(vec![100], vec![1]);

    let out = run_qlinear_matmul(a, b, y_zp);
    assert_eq!(out.storage.to_f32_vec(), vec![200.0, 0.0, 100.0, 100.0]);
}

// ── QLinearConv: dtype-aware saturation ──────────────────────────────────────

#[test]
fn test_qlinear_conv_f32_y_zero_point_falls_back_to_union_range() {
    let x = f32_t(vec![10.0, -15.0], vec![1, 1, 1, 2]);
    let w = f32_t(vec![10.0], vec![1, 1, 1, 1]);
    let y_zp = f32_t(vec![100.0], vec![1]);

    let out = run_qlinear_conv(x, w, y_zp);
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.shape, vec![1, 1, 1, 2]);
    assert_eq!(out.storage.to_f32_vec(), vec![200.0, -50.0]);
}

#[test]
fn test_qlinear_conv_i8_y_zero_point_saturates_to_int8_exactly() {
    let x = i8_t(vec![10, -15], vec![1, 1, 1, 2]);
    let w = i8_t(vec![10], vec![1, 1, 1, 1]);
    let y_zp = i8_t(vec![100], vec![1]);

    let out = run_qlinear_conv(x, w, y_zp);
    assert_eq!(out.storage.to_f32_vec(), vec![127.0, -50.0]);
}

#[test]
fn test_qlinear_conv_u8_y_zero_point_saturates_to_uint8_exactly() {
    let x = i8_t(vec![10, -15], vec![1, 1, 1, 2]);
    let w = i8_t(vec![10], vec![1, 1, 1, 1]);
    let y_zp = u8_t(vec![100], vec![1]);

    let out = run_qlinear_conv(x, w, y_zp);
    assert_eq!(out.storage.to_f32_vec(), vec![200.0, 0.0]);
}
