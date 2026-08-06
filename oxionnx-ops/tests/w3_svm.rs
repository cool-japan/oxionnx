//! Wave-3 `T6-tests-ops`: `SVMClassifier`/`SVMRegressor` `poly`/`sigmoid`
//! kernels, from finding [a11-22].
//!
//! `oxionnx-ops/src/ml_svm/tests.rs` (pre-existing, colocated with the
//! implementation) already covers multi-class one-vs-one voting (3-class and
//! 4-class), Platt-scaling probability outputs (both the 2-class direct form
//! and the >2-class pairwise-coupling form), and malformed-model typed
//! errors — all far beyond the "3 tests total" the audit finding describes,
//! so none of that is repeated here. What is still genuinely missing: the
//! `poly` and `sigmoid` kernel types (`KernelType` in
//! `oxionnx-ops/src/ml_svm.rs` supports four kernels; only `LINEAR` and `RBF`
//! have ever been exercised by a test, confirmed by a corpus-wide grep for
//! `"POLY"`/`"SIGMOID"` under `ml_svm`).
//!
//! Reference kernel values are hand-traced (`poly`: `(gamma*dot+coef0)^degree`;
//! `sigmoid`: `tanh(gamma*dot+coef0)`, the latter cross-checked against NumPy
//! float32 — see the session's final report for the generating snippet).

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::Tensor;
use oxionnx_ops::registry::ml_ops::{SVMClassifierOp, SVMRegressorOp};

fn dummy_node(op: OpKind) -> Node {
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn ctx<'a>(node: &'a Node, x: &'a Tensor) -> OpContext<'a> {
    OpContext {
        node,
        inputs: vec![Some(x)],
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn run_classifier(x: &Tensor, attrs: Attributes) -> Result<Vec<Tensor>, oxionnx_core::OnnxError> {
    let mut node = dummy_node(OpKind::SVMClassifier);
    node.attrs = attrs;
    SVMClassifierOp.execute(&ctx(&node, x))
}

fn run_regressor(x: &Tensor, attrs: Attributes) -> Result<Vec<Tensor>, oxionnx_core::OnnxError> {
    let mut node = dummy_node(OpKind::SVMRegressor);
    node.attrs = attrs;
    SVMRegressorOp.execute(&ctx(&node, x))
}

fn assert_close(got: f32, want: f32, tol: f32, label: &str) {
    assert!(
        (got - want).abs() < tol,
        "{label}: got {got}, want {want} (delta {})",
        (got - want).abs()
    );
}

/// One support vector `[1, 0]`, `gamma=1, coef0=1, degree=2`: kernel(x, sv) =
/// `(dot(x, sv) + 1)^2`. `x1=[2,0]`: dot=2, kernel=(2+1)^2=9, decision =
/// `1*9 - 0.5 = 8.5` (positive -> class 0). `x2=[-1,0]`: dot=-1,
/// kernel=(-1+1)^2=0, decision=`1*0 - 0.5 = -0.5` (negative -> class 1).
#[test]
fn poly_kernel_classifier_decision_values() {
    let x = Tensor::new(vec![2.0, 0.0, -1.0, 0.0], vec![2, 2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "POLY".into());
    attrs
        .float_lists
        .insert("kernel_params".into(), vec![1.0, 1.0, 2.0]); // gamma, coef0, degree
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![1.0, 0.0]);
    attrs.float_lists.insert("coefficients".into(), vec![1.0]);
    attrs.float_lists.insert("rho".into(), vec![-0.5]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);

    let result = run_classifier(&x, attrs).expect("svm_classifier poly");
    let labels = &result[0];
    assert_close(labels.data[0], 0.0, 1e-6, "x1 -> class 0");
    assert_close(labels.data[1], 1.0, 1e-6, "x2 -> class 1");

    let scores = &result[1];
    assert_close(scores.data[0], -8.5, 1e-4, "x1 score[0]");
    assert_close(scores.data[1], 8.5, 1e-4, "x1 score[1]");
    assert_close(scores.data[2], 0.5, 1e-4, "x2 score[0]");
    assert_close(scores.data[3], -0.5, 1e-4, "x2 score[1]");
}

/// Same poly kernel, single-target regressor. `x1=[2,0]` -> kernel=9,
/// output=`1*9 + 0 = 9`. `x2=[-1,0]` -> kernel=0, output=0.
#[test]
fn poly_kernel_regressor_matches_hand_kernel() {
    let x = Tensor::new(vec![2.0, 0.0, -1.0, 0.0], vec![2, 2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "POLY".into());
    attrs
        .float_lists
        .insert("kernel_params".into(), vec![1.0, 1.0, 2.0]);
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![1.0, 0.0]);
    attrs.float_lists.insert("coefficients".into(), vec![1.0]);
    attrs.float_lists.insert("rho".into(), vec![0.0]);

    let result = run_regressor(&x, attrs).expect("svm_regressor poly");
    assert_eq!(result[0].shape, vec![2, 1]);
    assert_close(result[0].data[0], 9.0, 1e-4, "x1 output");
    assert_close(result[0].data[1], 0.0, 1e-4, "x2 output");
}

/// One support vector `[1, 1]`, `gamma=0.5, coef0=0.1`: kernel(x, sv) =
/// `tanh(0.5*dot(x, sv) + 0.1)`. `x1=[1,1]`: dot=2, kernel=tanh(1.1)=
/// 0.800499, decision=`2*0.800499 = 1.600998` (positive -> class 0).
/// `x2=[-1,-1]`: dot=-2, kernel=tanh(-0.9)=-0.716298,
/// decision=`2*-0.716298 = -1.432596` (negative -> class 1).
#[test]
fn sigmoid_kernel_classifier_decision_values() {
    let x = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "SIGMOID".into());
    attrs
        .float_lists
        .insert("kernel_params".into(), vec![0.5, 0.1, 3.0]); // gamma, coef0, degree (unused)
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![1.0, 1.0]);
    attrs.float_lists.insert("coefficients".into(), vec![2.0]);
    attrs.float_lists.insert("rho".into(), vec![0.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);

    let result = run_classifier(&x, attrs).expect("svm_classifier sigmoid");
    let labels = &result[0];
    assert_close(labels.data[0], 0.0, 1e-6, "x1 -> class 0");
    assert_close(labels.data[1], 1.0, 1e-6, "x2 -> class 1");

    let scores = &result[1];
    assert_close(scores.data[1], 1.600_998, 1e-4, "x1 decision");
    assert_close(scores.data[2], 1.432_596, 1e-4, "x2 score[0] == -decision");
    assert_close(scores.data[3], -1.432_596, 1e-4, "x2 decision");
}

/// Same sigmoid kernel, single-target regressor with a nonzero `rho`.
/// `x1=[1,1]` -> `2*0.800499 + 0.5 = 2.100998`. `x2=[-1,-1]` ->
/// `2*-0.716298 + 0.5 = -0.932596`.
#[test]
fn sigmoid_kernel_regressor_matches_hand_kernel() {
    let x = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "SIGMOID".into());
    attrs
        .float_lists
        .insert("kernel_params".into(), vec![0.5, 0.1, 3.0]);
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![1.0, 1.0]);
    attrs.float_lists.insert("coefficients".into(), vec![2.0]);
    attrs.float_lists.insert("rho".into(), vec![0.5]);

    let result = run_regressor(&x, attrs).expect("svm_regressor sigmoid");
    assert_close(result[0].data[0], 2.100_998, 1e-4, "x1 output");
    assert_close(result[0].data[1], -0.932_596, 1e-4, "x2 output");
}

/// `KernelType::parse` (`oxionnx-ops/src/ml_svm.rs`) now rejects an
/// unrecognized `kernel_type` string with a typed [`OnnxError::InvalidModel`]
/// instead of silently computing a *plain dot product* (falling through to
/// `Linear`) — the same "bad enum falls through to a default variant" shape
/// as the `Cast`/`Resize`/RNN-`direction` findings in `w3_malformed_attrs.rs`,
/// fixed the same way. `kernel_type="TOTALLY_BOGUS"` with the same linear
/// setup used elsewhere in this crate's SVM tests (`x=[3,4]`, `sv=[1,0]`,
/// which would have silently produced `decision=3*1+0=3`, positive -> class
/// 0, had the bogus string been treated as `LINEAR`) must now be a typed
/// error rather than a computed result.
#[test]
fn unknown_kernel_type_is_a_typed_error() {
    let x = Tensor::new(vec![3.0, 4.0], vec![1, 2]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("kernel_type".into(), "TOTALLY_BOGUS".into());
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![1.0, 0.0]);
    attrs.float_lists.insert("coefficients".into(), vec![1.0]);
    attrs.float_lists.insert("rho".into(), vec![0.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);

    let err = run_classifier(&x, attrs).expect_err("bogus kernel_type must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("kernel_type") && message.contains("TOTALLY_BOGUS"),
        "expected the error to name the offending attribute and value, got: {message}"
    );
}
