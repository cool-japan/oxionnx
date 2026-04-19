//! Elementwise and unary operator integration tests: arithmetic (Add, Sub, Mul,
//! Div, Pow), unary math (Neg, Sqrt, Abs, Exp, Log, Reciprocal, Erf, Sin, Cos),
//! clamp/clip, bitwise ops, and Size.

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_tensor_approx, run_single_op};

// ═══════════════════════════════════════════════════════════════════════════════
// Arithmetic binary ops
// ═══════════════════════════════════════════════════════════════════════════════

// 19. test_add_scalar - scalar + tensor broadcasting
#[test]
fn test_add_scalar() {
    // scalar [1] + tensor [2,3] => broadcast to [2,3]
    let scalar = Tensor::new(vec![10.0], vec![1]);
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let outputs = run_single_op(
        OpKind::Add,
        vec![("a", scalar), ("b", tensor)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_tensor_approx(out, &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0], 1e-5);
}

// test_sub_elementwise
#[test]
fn test_sub_elementwise() {
    let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Sub,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[9.0, 18.0, 27.0], 1e-5);
}

// test_mul_broadcast
#[test]
fn test_mul_broadcast() {
    // [2,3] * [1,3] => broadcast to [2,3]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
    let outputs = run_single_op(
        OpKind::Mul,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_tensor_approx(out, &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0], 1e-5);
}

// test_div_elementwise
#[test]
fn test_div_elementwise() {
    let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let b = Tensor::new(vec![2.0, 4.0, 5.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Div,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[5.0, 5.0, 6.0], 1e-5);
}

// test_pow
#[test]
fn test_pow() {
    let a = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
    let b = Tensor::new(vec![3.0, 2.0, 0.5], vec![3]);
    let outputs = run_single_op(
        OpKind::Pow,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    // 2^3=8, 3^2=9, 4^0.5=2
    assert_tensor_approx(out, &[8.0, 9.0, 2.0], 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unary math ops
// ═══════════════════════════════════════════════════════════════════════════════

// test_neg
#[test]
fn test_neg() {
    let x = Tensor::new(vec![1.0, -2.0, 0.0, 3.5], vec![4]);
    let outputs = run_single_op(
        OpKind::Neg,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[-1.0, 2.0, 0.0, -3.5], 1e-5);
}

// test_sqrt
#[test]
fn test_sqrt() {
    let x = Tensor::new(vec![0.0, 1.0, 4.0, 9.0, 16.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Sqrt,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-5);
}

// test_abs
#[test]
fn test_abs() {
    let x = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);
    let outputs = run_single_op(
        OpKind::Abs,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[3.0, 1.0, 0.0, 1.0, 3.0], 1e-5);
}

// test_exp
#[test]
fn test_exp() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Exp,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[1.0, 1.0_f32.exp(), 2.0_f32.exp()], 1e-4);
}

// test_log
#[test]
fn test_log() {
    let x = Tensor::new(vec![1.0, std::f32::consts::E, 10.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Log,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[0.0, 1.0, 10.0_f32.ln()], 1e-4);
}

// test_reciprocal
#[test]
fn test_reciprocal() {
    let x = Tensor::new(vec![1.0, 2.0, 4.0, 0.5], vec![4]);
    let outputs = run_single_op(
        OpKind::Reciprocal,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[1.0, 0.5, 0.25, 2.0], 1e-5);
}

// test_erf
#[test]
fn test_erf() {
    // erf(0) = 0, erf(inf) = 1, erf(-inf) = -1
    let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
    let outputs = run_single_op(
        OpKind::Erf,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert!(out.data[0].abs() < 1e-5, "erf(0) = {}", out.data[0]);
    // erf(1) ~ 0.8427
    assert!(
        (out.data[1] - 0.8427).abs() < 0.01,
        "erf(1) = {}",
        out.data[1]
    );
    // erf(-1) ~ -0.8427
    assert!(
        (out.data[2] + 0.8427).abs() < 0.01,
        "erf(-1) = {}",
        out.data[2]
    );
}

// test_sin_cos
#[test]
fn test_sin_cos() {
    let x = Tensor::new(
        vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
        vec![3],
    );

    let sin_out = run_single_op(
        OpKind::Sin,
        vec![("x", x.clone())],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let s = sin_out.get("out").unwrap();
    assert_tensor_approx(s, &[0.0, 1.0, 0.0], 1e-5);

    let cos_out = run_single_op(
        OpKind::Cos,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let c = cos_out.get("out").unwrap();
    assert_tensor_approx(c, &[1.0, 0.0, -1.0], 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Clip
// ═══════════════════════════════════════════════════════════════════════════════

// test_clip
#[test]
fn test_clip() {
    // Clip(x, min=2, max=5) on [1, 3, 6]
    let x = Tensor::new(vec![1.0, 3.0, 6.0], vec![3]);
    let min_t = Tensor::new(vec![2.0], vec![1]);
    let max_t = Tensor::new(vec![5.0], vec![1]);

    let outputs = run_single_op(
        OpKind::Clip,
        vec![("x", x)],
        vec![("min", min_t), ("max", max_t)],
        vec!["x"],
        vec!["x", "min", "max"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_tensor_approx(out, &[2.0, 3.0, 5.0], 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bitwise ops
// ═══════════════════════════════════════════════════════════════════════════════

// test_bitwise_and
#[test]
fn test_bitwise_and() {
    // [5, 3] & [3, 6] = [1, 2]
    let a = Tensor::new(vec![5.0, 3.0], vec![2]);
    let b = Tensor::new(vec![3.0, 6.0], vec![2]);
    let outputs = run_single_op(
        OpKind::BitwiseAnd,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.data[0] as u32, 5u32 & 3u32); // 1
    assert_eq!(out.data[1] as u32, 3u32 & 6u32); // 2
}

// test_bitwise_or
#[test]
fn test_bitwise_or() {
    // [5, 3] | [3, 6] = [7, 7]
    let a = Tensor::new(vec![5.0, 3.0], vec![2]);
    let b = Tensor::new(vec![3.0, 6.0], vec![2]);
    let outputs = run_single_op(
        OpKind::BitwiseOr,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.data[0] as u32, 5u32 | 3u32); // 7
    assert_eq!(out.data[1] as u32, 3u32 | 6u32); // 7
}

// test_bitwise_xor
#[test]
fn test_bitwise_xor() {
    // [5, 3] ^ [3, 6] = [6, 5]
    let a = Tensor::new(vec![5.0, 3.0], vec![2]);
    let b = Tensor::new(vec![3.0, 6.0], vec![2]);
    let outputs = run_single_op(
        OpKind::BitwiseXor,
        vec![("a", a), ("b", b)],
        vec![],
        vec!["a", "b"],
        vec!["a", "b"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.data[0] as u32, 5u32 ^ 3u32); // 6
    assert_eq!(out.data[1] as u32, 3u32 ^ 6u32); // 5
}

// test_bitwise_not
#[test]
fn test_bitwise_not() {
    // !0u32 = 4294967295 (u32::MAX)
    let x = Tensor::new(vec![0.0], vec![1]);
    let outputs = run_single_op(
        OpKind::BitwiseNot,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(
        out.data[0] as u32, !0u32,
        "bitwise_not(0) should be u32::MAX as u32"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Misc
// ═══════════════════════════════════════════════════════════════════════════════

// test_size_op
#[test]
fn test_size_op() {
    // input shape [2, 3] → num elements = 6
    let x = Tensor::new(vec![1.0; 6], vec![2, 3]);
    let outputs = run_single_op(
        OpKind::Size,
        vec![("x", x)],
        vec![],
        vec!["x"],
        vec!["x"],
        "out",
        Attributes::default(),
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.data[0], 6.0, "Size of [2,3] tensor should be 6");
}
