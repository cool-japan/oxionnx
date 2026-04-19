//! Conformance tests 1–15: Math and reduction operators.

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_close, assert_shape, run_op};

// ═══════════════════════════════════════════════════════════════════════════════
// 1–10: Math conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 1. conformance_add_broadcast — [2,3] + [3] = broadcast add
#[test]
fn conformance_add_broadcast() {
    // A = [[1,2,3],[4,5,6]], B = [10,20,30]
    // Expected: [[11,22,33],[14,25,36]]
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let out = run_op(
        OpKind::Add,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "add_broadcast");
    assert_close(
        &t.data,
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        1e-5,
        "add_broadcast",
    );
}

/// 2. conformance_sub — [4] - [4]
#[test]
fn conformance_sub() {
    let a = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let b = Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![4]);
    let out = run_op(
        OpKind::Sub,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[4], "sub");
    assert_close(&t.data, &[9.0, 17.0, 25.0, 33.0], 1e-5, "sub");
}

/// 3. conformance_mul_scalar — [3,3] * scalar
#[test]
fn conformance_mul_scalar() {
    let a = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    );
    let b = Tensor::new(vec![3.0], vec![1]);
    let out = run_op(
        OpKind::Mul,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[3, 3], "mul_scalar");
    assert_close(
        &t.data,
        &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0],
        1e-5,
        "mul_scalar",
    );
}

/// 4. conformance_div — element-wise division
#[test]
fn conformance_div() {
    let a = Tensor::new(vec![10.0, 21.0, 36.0, 4.0], vec![4]);
    let b = Tensor::new(vec![2.0, 3.0, 4.0, 8.0], vec![4]);
    let out = run_op(
        OpKind::Div,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[5.0, 7.0, 9.0, 0.5], 1e-5, "div");
}

/// 5. conformance_pow — x^2
#[test]
fn conformance_pow() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let b = Tensor::new(vec![2.0], vec![1]);
    let out = run_op(
        OpKind::Pow,
        vec!["a", "b"],
        vec!["out"],
        vec!["a", "b"],
        vec![("a", a), ("b", b)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[1.0, 4.0, 9.0, 16.0, 25.0], 1e-5, "pow");
}

/// 6. conformance_sqrt — sqrt of known values
#[test]
fn conformance_sqrt() {
    let x = Tensor::new(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0], vec![6]);
    let out = run_op(
        OpKind::Sqrt,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 1e-5, "sqrt");
}

/// 7. conformance_exp — exp(0)=1, exp(1)≈2.718
#[test]
fn conformance_exp() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]);
    let out = run_op(
        OpKind::Exp,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    let expected = [1.0, std::f32::consts::E, (-1.0_f32).exp(), (2.0_f32).exp()];
    assert_close(&t.data, &expected, 1e-5, "exp");
}

/// 8. conformance_log — log(1)=0, log(e)=1
#[test]
fn conformance_log() {
    let x = Tensor::new(
        vec![
            1.0,
            std::f32::consts::E,
            std::f32::consts::E * std::f32::consts::E,
            10.0,
        ],
        vec![4],
    );
    let out = run_op(
        OpKind::Log,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    let expected = [0.0, 1.0, 2.0, (10.0_f32).ln()];
    assert_close(&t.data, &expected, 1e-5, "log");
}

/// 9. conformance_abs — absolute value of negatives
#[test]
fn conformance_abs() {
    let x = Tensor::new(vec![-3.0, -1.5, 0.0, 2.5, -7.0], vec![5]);
    let out = run_op(
        OpKind::Abs,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[3.0, 1.5, 0.0, 2.5, 7.0], 1e-5, "abs");
}

/// 10. conformance_neg — negation
#[test]
fn conformance_neg() {
    let x = Tensor::new(vec![1.0, -2.0, 0.0, 3.5, -0.5], vec![5]);
    let out = run_op(
        OpKind::Neg,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[-1.0, 2.0, 0.0, -3.5, 0.5], 1e-5, "neg");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11–15: Reduction conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 11. conformance_reduce_mean_keepdims — ReduceMean axis=1, keepdims=1
#[test]
fn conformance_reduce_mean_keepdims() {
    // x = [[1,2,3],[4,5,6]] shape [2,3]
    // mean axis=1 keepdims => [[2],[5]] shape [2,1]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 1);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = run_op(
        OpKind::ReduceMean,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 1], "reduce_mean_keepdims");
    assert_close(&t.data, &[2.0, 5.0], 1e-5, "reduce_mean_keepdims");
}

/// 12. conformance_reduce_sum_no_keepdims — ReduceSum axis=0, keepdims=0
#[test]
fn conformance_reduce_sum_no_keepdims() {
    // x = [[1,2,3],[4,5,6]] shape [2,3]
    // sum axis=0 => [5,7,9] shape [3]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![0]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = run_op(
        OpKind::ReduceSum,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[5.0, 7.0, 9.0], 1e-5, "reduce_sum_no_keepdims");
}

/// 13. conformance_reduce_max — ReduceMax axis=1
#[test]
fn conformance_reduce_max() {
    // x = [[3,1,2],[6,4,5]] shape [2,3]
    // max axis=1, keepdims=0 => [3, 6]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], vec![2, 3]);
    let out = run_op(
        OpKind::ReduceMax,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[3.0, 6.0], 1e-5, "reduce_max");
}

/// 14. conformance_reduce_min — ReduceMin axis=1
#[test]
fn conformance_reduce_min() {
    // x = [[3,1,2],[6,4,5]] shape [2,3]
    // min axis=1, keepdims=0 => [1, 4]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("axes".to_string(), vec![1]);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], vec![2, 3]);
    let out = run_op(
        OpKind::ReduceMin,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[1.0, 4.0], 1e-5, "reduce_min");
}

/// 15. conformance_argmax — ArgMax axis=1
#[test]
fn conformance_argmax() {
    // x = [[3,1,2],[6,4,5]] shape [2,3]
    // argmax axis=1, keepdims=0 => [0, 0] (index of max in each row)
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 1);
    attrs.ints.insert("keepdims".to_string(), 0);

    let x = Tensor::new(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], vec![2, 3]);
    let out = run_op(
        OpKind::ArgMax,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    // argmax returns indices: row0 max=3 at idx 0, row1 max=6 at idx 0
    assert_close(&t.data, &[0.0, 0.0], 1e-5, "argmax");
}
