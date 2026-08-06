//! Tests for the ONNX-ML SVM operators.
//!
//! Score and label conventions follow onnxruntime's `svmclassifier.cc`:
//! a pairwise decision value is `sum(coef * kernel) + rho` (`decision > 0`
//! votes for the *lower* class index), and skl2onnx negates
//! `coefficients`/`rho` when exporting a binary SVC so that this convention
//! reproduces scikit-learn's prediction. Votes decide the predicted label
//! only when no Platt scaling (`prob_a`/`prob_b`) is present; when it is,
//! the label is instead the argmax of the pairwise-coupled probabilities
//! (libsvm's `svm_predict_probability` semantics), which can disagree with
//! the vote winner — see
//! `multiclass_probability_argmax_can_disagree_with_votes` below.

use super::*;
use oxionnx_core::graph::{Attributes, Node, OpKind};

fn make_context(
    op: OpKind,
    inputs: Vec<Option<&Tensor>>,
    attrs: Attributes,
) -> (Node, Vec<Option<&Tensor>>) {
    let node = Node {
        op,
        name: "test_node".to_string(),
        inputs: vec![],
        outputs: vec![],
        attrs,
    };
    (node, inputs)
}

fn ctx_from<'a>(node: &'a Node, inputs: &'a [Option<&'a Tensor>]) -> OpContext<'a> {
    OpContext {
        node,
        inputs: inputs.to_vec(),
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn run_classifier(x: &Tensor, attrs: Attributes) -> Result<Vec<Tensor>, OnnxError> {
    let (node, inputs) = make_context(OpKind::SVMClassifier, vec![Some(x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    svm_classifier(&ctx)
}

fn run_regressor(x: &Tensor, attrs: Attributes) -> Result<Vec<Tensor>, OnnxError> {
    let (node, inputs) = make_context(OpKind::SVMRegressor, vec![Some(x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    svm_regressor(&ctx)
}

/// Binary SVC: SV0 = [1, 0] (class 0), SV1 = [0, 1] (class 1),
/// coefficients [1, -1], so `decision = x[0] - x[1] + rho`.
fn binary_attrs(rho: f32) -> Attributes {
    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "LINEAR".into());
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![1.0, 0.0, 0.0, 1.0]);
    attrs
        .float_lists
        .insert("coefficients".into(), vec![1.0, -1.0]);
    attrs.float_lists.insert("rho".into(), vec![rho]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);
    attrs.strings.insert("post_transform".into(), "NONE".into());
    attrs
}

/// Three-class SVC with one support vector per class.
///
/// SVs: [1,0] (class 0), [0,1] (class 1), [-1,0] (class 2).
/// coefficients rows: [0.5, -0.5, 0.25] and [0.3, 0.7, -0.4].
/// For x = [1, 1] the kernels are [1, 1, -1], giving pairwise decisions
/// (0,1) = 0.1, (0,2) = 0.25 and (1,2) = 1.15.
fn three_class_attrs() -> Attributes {
    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "LINEAR".into());
    attrs.float_lists.insert(
        "support_vectors".into(),
        vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
    );
    attrs
        .int_lists
        .insert("vectors_per_class".into(), vec![1, 1, 1]);
    attrs
        .float_lists
        .insert("coefficients".into(), vec![0.5, -0.5, 0.25, 0.3, 0.7, -0.4]);
    attrs.float_lists.insert("rho".into(), vec![0.1, 0.2, 0.05]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![10, 20, 30]);
    attrs.strings.insert("post_transform".into(), "NONE".into());
    attrs
}

// ── Decision values, rho sign and votes ────────────────────────────────────

#[test]
fn binary_linear_decision_votes_for_the_lower_class_index() {
    let x = Tensor::new(
        vec![
            2.0, 1.0, // decision = 2 - 1 = 1 > 0 => class 0
            1.0, 3.0, // decision = 1 - 3 = -2 < 0 => class 1
        ],
        vec![2, 2],
    );

    let result = run_classifier(&x, binary_attrs(0.0)).expect("svm_classifier");
    assert_eq!(result.len(), 2);

    let labels = &result[0];
    assert_eq!(labels.shape, vec![2]);
    assert!((labels.data[0] - 0.0).abs() < 1e-5, "{:?}", labels.data);
    assert!((labels.data[1] - 1.0).abs() < 1e-5, "{:?}", labels.data);

    let scores = &result[1];
    assert_eq!(scores.shape, vec![2, 2]);
    assert!((scores.data[0] - (-1.0)).abs() < 1e-5);
    assert!((scores.data[1] - 1.0).abs() < 1e-5);
    assert!((scores.data[2] - 2.0).abs() < 1e-5);
    assert!((scores.data[3] - (-2.0)).abs() < 1e-5);
}

#[test]
fn rho_is_added_to_the_decision_value() {
    // onnxruntime computes `sum += rho_[classifier_idx]`; the ONNX `rho`
    // attribute already carries scikit-learn's intercept sign.
    let x = Tensor::new(vec![2.0, 1.0], vec![1, 2]);

    let result = run_classifier(&x, binary_attrs(0.5)).expect("svm_classifier");
    let scores = &result[1];
    // decision = (2 - 1) + 0.5 = 1.5 (subtracting rho would give 0.5).
    assert!((scores.data[1] - 1.5).abs() < 1e-5, "{:?}", scores.data);
    assert!((scores.data[0] - (-1.5)).abs() < 1e-5);
}

#[test]
fn rbf_kernel_decision() {
    // One support vector at the origin, coefficient 1, gamma 1, rho -0.5:
    // decision = exp(-||x||^2) - 0.5.
    let x = Tensor::new(
        vec![
            0.0, 0.0, // exp(0) - 0.5 = 0.5 > 0 => class 0
            10.0, 10.0, // exp(-200) - 0.5 ~= -0.5 < 0 => class 1
            0.5, 0.5, // exp(-0.5) - 0.5 ~= 0.1065 > 0 => class 0
        ],
        vec![3, 2],
    );

    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "RBF".into());
    attrs
        .float_lists
        .insert("kernel_params".into(), vec![1.0, 0.0, 3.0]);
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![0.0, 0.0]);
    attrs.float_lists.insert("coefficients".into(), vec![1.0]);
    attrs.float_lists.insert("rho".into(), vec![-0.5]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);
    attrs.strings.insert("post_transform".into(), "NONE".into());

    let result = run_classifier(&x, attrs).expect("svm_classifier rbf");
    let labels = &result[0];
    assert!((labels.data[0] - 0.0).abs() < 1e-5);
    assert!((labels.data[1] - 1.0).abs() < 1e-5);
    assert!((labels.data[2] - 0.0).abs() < 1e-5);

    let scores = &result[1];
    assert!((scores.data[1] - 0.5).abs() < 1e-5, "{:?}", scores.data);
    assert!((scores.data[0] - (-0.5)).abs() < 1e-5);
    assert!(
        (scores.data[5] - 0.106_530_66).abs() < 1e-5,
        "{:?}",
        scores.data
    );
}

// ── [a3-17] multiclass scores and Platt scaling ────────────────────────────

#[test]
fn multiclass_scores_are_pairwise_decision_values_not_votes() {
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let result = run_classifier(&x, three_class_attrs()).expect("svm_classifier");
    let scores = &result[1];
    assert_eq!(scores.shape, vec![1, 3]);
    assert!((scores.data[0] - 0.1).abs() < 1e-5, "{:?}", scores.data);
    assert!((scores.data[1] - 0.25).abs() < 1e-5, "{:?}", scores.data);
    assert!((scores.data[2] - 1.15).abs() < 1e-5, "{:?}", scores.data);

    // Votes: (0,1) -> 0, (0,2) -> 0, (1,2) -> 1, so class 0 wins with 2 votes.
    assert!(
        (result[0].data[0] - 10.0).abs() < 1e-5,
        "{:?}",
        result[0].data
    );
}

#[test]
fn multiclass_probabilities_use_pairwise_coupling() {
    // Reference values computed with libsvm's `multiclass_probability`
    // (Wu/Lin/Weng method 2) in float32 over the pairwise Platt probabilities
    // sigmoid(d * a + b) for d = [0.1, 0.25, 1.15],
    // prob_a = [-1.5, -2.0, -1.0], prob_b = [0.1, -0.2, 0.3].
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let mut attrs = three_class_attrs();
    attrs
        .float_lists
        .insert("prob_a".into(), vec![-1.5, -2.0, -1.0]);
    attrs
        .float_lists
        .insert("prob_b".into(), vec![0.1, -0.2, 0.3]);

    let result = run_classifier(&x, attrs).expect("svm_classifier");
    let scores = &result[1];
    assert_eq!(scores.shape, vec![1, 3]);

    let expected = [0.409_328_8f32, 0.403_288_54, 0.187_382_71];
    for (i, &want) in expected.iter().enumerate() {
        assert!(
            (scores.data[i] - want).abs() < 1e-5,
            "class {i}: expected {want}, got {} ({:?})",
            scores.data[i],
            scores.data
        );
    }
    let sum: f32 = scores.data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "probabilities must sum to 1: {sum}"
    );

    assert!((result[0].data[0] - 10.0).abs() < 1e-5);
}

/// The predicted label must come from the argmax of the pairwise-coupled
/// *probabilities* (libsvm's `svm_predict_probability` semantics, which
/// onnxruntime follows), not from the raw one-versus-one vote count, once
/// `prob_a`/`prob_b` are present.
///
/// Reuses `three_class_attrs()`'s decisions `d = [0.1, 0.25, 1.15]` (pairs
/// `(0,1)`, `(0,2)`, `(1,2)`), whose *votes* are unambiguous: `d > 0` on every
/// pair, so pair `(0,1)` and `(0,2)` both vote for the lower index (class 0),
/// and `(1,2)` votes for class 1. Vote tally = `[2, 1, 0]` — class 0 (label
/// 10) is the vote winner.
///
/// `prob_a = [-1, -1, -1]`, `prob_b = [4, 4, 0]` deliberately pushes both of
/// class 0's pairwise probabilities near zero (a large positive intercept on
/// exactly the two pairs class 0 participates in overwhelms its small
/// positive decision values), while leaving pair `(1,2)` unscaled (`a = -1,
/// b = 0`, using `d = 1.15` on its own) so class 1 keeps a decisive pairwise
/// edge over class 2. Pairwise coupling therefore drives class 0's overall
/// probability far below classes 1 and 2's, flipping the predicted label to
/// class 1 (label 20) even though class 0 still wins the raw vote 2-to-1.
///
/// Reference values computed with libsvm's `multiclass_probability` (Wu/Lin/
/// Weng method 2) in float32, replicating this crate's algorithm exactly
/// (same iteration order, same `eps = 0.005 / k`, same `[1e-7, 1 - 1e-7]`
/// clamp) — see the sibling script this test's constants were generated
/// from.
#[test]
fn multiclass_probability_argmax_can_disagree_with_votes() {
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let mut attrs = three_class_attrs();
    attrs
        .float_lists
        .insert("prob_a".into(), vec![-1.0, -1.0, -1.0]);
    attrs
        .float_lists
        .insert("prob_b".into(), vec![4.0, 4.0, 0.0]);

    let result = run_classifier(&x, attrs).expect("svm_classifier");
    let scores = &result[1];
    assert_eq!(scores.shape, vec![1, 3]);

    let expected = [0.010_618_734f32, 0.751_237_2, 0.238_144_07];
    for (i, &want) in expected.iter().enumerate() {
        assert!(
            (scores.data[i] - want).abs() < 1e-5,
            "class {i}: expected {want}, got {} ({:?})",
            scores.data[i],
            scores.data
        );
    }
    let sum: f32 = scores.data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "probabilities must sum to 1: {sum}"
    );

    // The probability argmax is class 1 (index 1 -> label 20): confirm this
    // genuinely disagrees with the vote winner, not merely that the fixed
    // code happens to produce label 20 for some other reason.
    let (vote_winner_idx, _) = [2u32, 1, 0]
        .iter()
        .enumerate()
        .max_by_key(|&(_, &v)| v)
        .expect("three classes");
    assert_eq!(
        vote_winner_idx, 0,
        "sanity: pairwise decisions [0.1, 0.25, 1.15] must vote class 0 the winner (2 votes)"
    );
    let (prob_winner_idx, _) = expected
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("no NaNs"))
        .expect("three classes");
    assert_eq!(
        prob_winner_idx, 1,
        "sanity: the coupled-probability argmax must be class 1, not the vote winner (class 0)"
    );
    assert_ne!(
        vote_winner_idx, prob_winner_idx,
        "this test is only meaningful if votes and probability argmax disagree"
    );

    assert!(
        (result[0].data[0] - 20.0).abs() < 1e-5,
        "predicted label must follow the probability argmax (class 1 -> label 20), \
         not the vote winner (class 0 -> label 10); got {:?}",
        result[0].data
    );
}

#[test]
fn binary_probabilities_are_platt_scaled() {
    // decision = 1.0, prob_a = -1, prob_b = 0
    // => P(class 0) = 1 / (1 + exp(-1)) = 0.73105857.
    let x = Tensor::new(vec![2.0, 1.0], vec![1, 2]);

    let mut attrs = binary_attrs(0.0);
    attrs.float_lists.insert("prob_a".into(), vec![-1.0]);
    attrs.float_lists.insert("prob_b".into(), vec![0.0]);

    let result = run_classifier(&x, attrs).expect("svm_classifier");
    let scores = &result[1];
    assert_eq!(scores.shape, vec![1, 2]);
    assert!(
        (scores.data[0] - 0.731_058_6).abs() < 1e-5,
        "{:?}",
        scores.data
    );
    assert!(
        (scores.data[1] - 0.268_941_43).abs() < 1e-5,
        "{:?}",
        scores.data
    );
    assert!((result[0].data[0] - 0.0).abs() < 1e-5);
}

#[test]
fn four_class_scores_have_one_column_per_classifier_pair() {
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "LINEAR".into());
    attrs.float_lists.insert(
        "support_vectors".into(),
        vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
    );
    attrs
        .int_lists
        .insert("vectors_per_class".into(), vec![1, 1, 1, 1]);
    attrs
        .float_lists
        .insert("coefficients".into(), vec![0.1; 12]);
    attrs.float_lists.insert("rho".into(), vec![0.0; 6]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1, 2, 3]);

    let result = run_classifier(&x, attrs).expect("svm_classifier");
    assert_eq!(result[1].shape, vec![1, 6]);
    assert_eq!(result[0].shape, vec![1]);
}

// ── [a3-15] 1-D input is one sample ────────────────────────────────────────

#[test]
fn one_dimensional_input_is_a_single_sample() {
    // [2.0, 1.0] as shape [2] is ONE sample with two features, not two
    // samples of one feature each.
    let x = Tensor::new(vec![2.0, 1.0], vec![2]);

    let result = run_classifier(&x, binary_attrs(0.0)).expect("svm_classifier");
    assert_eq!(result[0].shape, vec![1]);
    assert_eq!(result[1].shape, vec![1, 2]);
    assert!(
        (result[1].data[1] - 1.0).abs() < 1e-5,
        "{:?}",
        result[1].data
    );
}

#[test]
fn regressor_accepts_one_dimensional_input() {
    let x = Tensor::new(vec![2.0, 4.0], vec![2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "LINEAR".into());
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![1.0, 0.0, 0.0, 1.0]);
    attrs
        .float_lists
        .insert("coefficients".into(), vec![0.5, 0.5]);
    attrs.float_lists.insert("rho".into(), vec![1.0]);
    attrs.strings.insert("post_transform".into(), "NONE".into());

    let result = run_regressor(&x, attrs).expect("svm_regressor");
    assert_eq!(result[0].shape, vec![1, 1]);
    // 0.5 * 2 + 0.5 * 4 + 1 = 4
    assert!(
        (result[0].data[0] - 4.0).abs() < 1e-5,
        "{:?}",
        result[0].data
    );
}

// ── Malformed models produce typed errors ──────────────────────────────────

#[test]
fn missing_support_vectors_is_a_typed_error() {
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let mut attrs = binary_attrs(0.0);
    attrs.float_lists.remove("support_vectors");

    assert!(matches!(
        run_classifier(&x, attrs),
        Err(OnnxError::Unsupported(_))
    ));
}

#[test]
fn short_rho_is_a_typed_error() {
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let mut attrs = three_class_attrs();
    attrs.float_lists.insert("rho".into(), vec![0.1]);

    assert!(matches!(
        run_classifier(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn vectors_per_class_mismatch_is_a_typed_error() {
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let mut attrs = three_class_attrs();
    attrs
        .int_lists
        .insert("vectors_per_class".into(), vec![1, 2]);

    assert!(matches!(
        run_classifier(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn missing_vectors_per_class_for_multiclass_is_a_typed_error() {
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let mut attrs = three_class_attrs();
    attrs.int_lists.remove("vectors_per_class");

    assert!(matches!(
        run_classifier(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn prob_a_shorter_than_the_pair_count_is_a_typed_error() {
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let mut attrs = three_class_attrs();
    attrs.float_lists.insert("prob_a".into(), vec![-1.0]);
    attrs.float_lists.insert("prob_b".into(), vec![0.0]);

    assert!(matches!(
        run_classifier(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

// ── SVMRegressor ───────────────────────────────────────────────────────────

#[test]
fn test_svm_regressor_linear() {
    // Output = 0.5 * x[0] + 0.5 * x[1] + 1.0
    let x = Tensor::new(vec![2.0, 4.0, 0.0, 0.0], vec![2, 2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("kernel_type".into(), "LINEAR".into());
    attrs
        .float_lists
        .insert("support_vectors".into(), vec![1.0, 0.0, 0.0, 1.0]);
    attrs
        .float_lists
        .insert("coefficients".into(), vec![0.5, 0.5]);
    attrs.float_lists.insert("rho".into(), vec![1.0]);
    attrs.strings.insert("post_transform".into(), "NONE".into());

    let result = run_regressor(&x, attrs).expect("svm_regressor");
    assert_eq!(result.len(), 1);
    let y = &result[0];
    assert_eq!(y.shape, vec![2, 1]);
    assert!((y.data[0] - 4.0).abs() < 1e-5);
    assert!((y.data[1] - 1.0).abs() < 1e-5);
}
