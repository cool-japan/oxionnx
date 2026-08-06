//! Tests for the ONNX-ML tree ensemble operators.

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

/// Single tree, one feature: `x[0] <= threshold ? node 1 : node 2`.
fn stump_attrs(threshold: f32, true_weight: f32, false_weight: f32) -> Attributes {
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("nodes_treeids".into(), vec![0, 0, 0]);
    attrs
        .int_lists
        .insert("nodes_nodeids".into(), vec![0, 1, 2]);
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![0, 0, 0]);
    attrs
        .float_lists
        .insert("nodes_values".into(), vec![threshold, 0.0, 0.0]);
    attrs
        .int_lists
        .insert("nodes_truenodeids".into(), vec![1, 0, 0]);
    attrs
        .int_lists
        .insert("nodes_falsenodeids".into(), vec![2, 0, 0]);
    attrs.string_lists.insert(
        "nodes_modes".into(),
        vec!["BRANCH_LEQ".into(), "LEAF".into(), "LEAF".into()],
    );
    attrs.int_lists.insert("target_treeids".into(), vec![0, 0]);
    attrs.int_lists.insert("target_nodeids".into(), vec![1, 2]);
    attrs.int_lists.insert("target_ids".into(), vec![0, 0]);
    attrs
        .float_lists
        .insert("target_weights".into(), vec![true_weight, false_weight]);
    attrs.ints.insert("n_targets".into(), 1);
    attrs.strings.insert("post_transform".into(), "NONE".into());
    attrs
}

/// Two single-node trees whose leaves return `w0` and `w1`.
fn two_leaf_trees(w0: f32, w1: f32) -> Attributes {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("nodes_treeids".into(), vec![0, 1]);
    attrs.int_lists.insert("nodes_nodeids".into(), vec![0, 0]);
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![0, 0]);
    attrs
        .float_lists
        .insert("nodes_values".into(), vec![0.0, 0.0]);
    attrs
        .int_lists
        .insert("nodes_truenodeids".into(), vec![0, 0]);
    attrs
        .int_lists
        .insert("nodes_falsenodeids".into(), vec![0, 0]);
    attrs
        .string_lists
        .insert("nodes_modes".into(), vec!["LEAF".into(), "LEAF".into()]);
    attrs.int_lists.insert("target_treeids".into(), vec![0, 1]);
    attrs.int_lists.insert("target_nodeids".into(), vec![0, 0]);
    attrs.int_lists.insert("target_ids".into(), vec![0, 0]);
    attrs
        .float_lists
        .insert("target_weights".into(), vec![w0, w1]);
    attrs.ints.insert("n_targets".into(), 1);
    attrs.strings.insert("post_transform".into(), "NONE".into());
    attrs
}

fn run_regressor(x: &Tensor, attrs: Attributes) -> Result<Vec<Tensor>, OnnxError> {
    let (node, inputs) = make_context(OpKind::TreeEnsembleRegressor, vec![Some(x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    tree_ensemble_regressor(&ctx)
}

fn run_classifier(x: &Tensor, attrs: Attributes) -> Result<Vec<Tensor>, OnnxError> {
    let (node, inputs) = make_context(OpKind::TreeEnsembleClassifier, vec![Some(x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    tree_ensemble_classifier(&ctx)
}

// ── Baseline behaviour ─────────────────────────────────────────────────────

#[test]
fn test_tree_ensemble_classifier_2tree_2class() {
    // Tree 0: x[0] <= 0.5 -> class 0 else class 1
    // Tree 1: x[1] <= 0.5 -> class 0 else class 1
    let x = Tensor::new(
        vec![
            0.0, 0.0, // both features below the split => class 0 (2 votes)
            1.0, 1.0, // both above => class 1 (2 votes)
            0.0, 1.0, // split votes => tie, argmax picks the first
        ],
        vec![3, 2],
    );

    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("nodes_treeids".into(), vec![0, 0, 0, 1, 1, 1]);
    attrs
        .int_lists
        .insert("nodes_nodeids".into(), vec![0, 1, 2, 0, 1, 2]);
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![0, 0, 0, 1, 0, 0]);
    attrs
        .float_lists
        .insert("nodes_values".into(), vec![0.5, 0.0, 0.0, 0.5, 0.0, 0.0]);
    attrs
        .int_lists
        .insert("nodes_truenodeids".into(), vec![1, 0, 0, 1, 0, 0]);
    attrs
        .int_lists
        .insert("nodes_falsenodeids".into(), vec![2, 0, 0, 2, 0, 0]);
    attrs.string_lists.insert(
        "nodes_modes".into(),
        vec![
            "BRANCH_LEQ".into(),
            "LEAF".into(),
            "LEAF".into(),
            "BRANCH_LEQ".into(),
            "LEAF".into(),
            "LEAF".into(),
        ],
    );
    attrs
        .int_lists
        .insert("class_treeids".into(), vec![0, 0, 1, 1]);
    attrs
        .int_lists
        .insert("class_nodeids".into(), vec![1, 2, 1, 2]);
    attrs.int_lists.insert("class_ids".into(), vec![0, 1, 0, 1]);
    attrs
        .float_lists
        .insert("class_weights".into(), vec![1.0, 1.0, 1.0, 1.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);
    attrs.strings.insert("post_transform".into(), "NONE".into());

    let result = run_classifier(&x, attrs).expect("classifier");
    assert_eq!(result.len(), 2);

    let labels = &result[0];
    assert_eq!(labels.shape, vec![3]);
    assert!((labels.data[0] - 0.0).abs() < 1e-5);
    assert!((labels.data[1] - 1.0).abs() < 1e-5);
    assert!((labels.data[2] - 0.0).abs() < 1e-5);

    let scores = &result[1];
    assert_eq!(scores.shape, vec![3, 2]);
    assert!((scores.data[0] - 2.0).abs() < 1e-5);
    assert!((scores.data[1] - 0.0).abs() < 1e-5);
    assert!((scores.data[2] - 0.0).abs() < 1e-5);
    assert!((scores.data[3] - 2.0).abs() < 1e-5);
}

#[test]
fn test_tree_ensemble_regressor_single_tree() {
    let x = Tensor::new(vec![0.5, 0.0, 2.0, 0.0], vec![2, 2]);

    let mut attrs = stump_attrs(1.0, 10.0, 20.0);
    // Two features in this model; the stump only splits on feature 0.
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![0, 0, 0]);

    let result = run_regressor(&x, attrs).expect("regressor");
    assert_eq!(result.len(), 1);
    let y = &result[0];
    assert_eq!(y.shape, vec![2, 1]);
    assert!((y.data[0] - 10.0).abs() < 1e-5);
    assert!((y.data[1] - 20.0).abs() < 1e-5);
}

// ── [a3-1] traversal cannot loop forever ───────────────────────────────────

#[test]
fn cyclic_node_table_is_rejected_instead_of_hanging() {
    // Every node is a branch and the "leaves" point back at node 0 — the exact
    // shape a dropped `nodes_modes` attribute used to produce. Traversal must
    // terminate with a typed error rather than spinning forever.
    let x = Tensor::new(vec![0.0], vec![1, 1]);

    let mut attrs = stump_attrs(0.5, 10.0, 20.0);
    attrs.string_lists.insert(
        "nodes_modes".into(),
        vec![
            "BRANCH_LEQ".into(),
            "BRANCH_LEQ".into(),
            "BRANCH_LEQ".into(),
        ],
    );

    match run_regressor(&x, attrs) {
        Err(OnnxError::InvalidModel(msg)) => assert!(msg.contains("cycle"), "{msg}"),
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn self_looping_classifier_node_is_rejected() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);

    let mut attrs = Attributes::default();
    attrs.int_lists.insert("nodes_treeids".into(), vec![0, 0]);
    attrs.int_lists.insert("nodes_nodeids".into(), vec![0, 1]);
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![0, 0]);
    attrs
        .float_lists
        .insert("nodes_values".into(), vec![0.5, 0.5]);
    // Node 1 points at itself on both branches.
    attrs
        .int_lists
        .insert("nodes_truenodeids".into(), vec![1, 1]);
    attrs
        .int_lists
        .insert("nodes_falsenodeids".into(), vec![1, 1]);
    attrs.string_lists.insert(
        "nodes_modes".into(),
        vec!["BRANCH_LEQ".into(), "BRANCH_LEQ".into()],
    );
    attrs.int_lists.insert("class_treeids".into(), vec![0]);
    attrs.int_lists.insert("class_nodeids".into(), vec![1]);
    attrs.int_lists.insert("class_ids".into(), vec![0]);
    attrs.float_lists.insert("class_weights".into(), vec![1.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);

    assert!(matches!(
        run_classifier(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn missing_nodes_modes_is_rejected() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);

    let mut attrs = stump_attrs(0.5, 10.0, 20.0);
    attrs.string_lists.remove("nodes_modes");

    match run_regressor(&x, attrs) {
        Err(OnnxError::InvalidModel(msg)) => assert!(msg.contains("nodes_modes"), "{msg}"),
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn unknown_mode_string_is_rejected() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);

    let mut attrs = stump_attrs(0.5, 10.0, 20.0);
    attrs.string_lists.insert(
        "nodes_modes".into(),
        vec!["BRANCH_MAYBE".into(), "LEAF".into(), "LEAF".into()],
    );

    assert!(matches!(
        run_regressor(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

// ── [a10-14] parallel arrays are bounds-checked up front ───────────────────

#[test]
fn short_featureids_array_is_rejected_without_panicking() {
    let x = Tensor::new(vec![0.0, 1.0], vec![1, 2]);

    let mut attrs = stump_attrs(0.5, 10.0, 20.0);
    // One entry short of `nodes_modes`: used to index out of bounds mid-walk.
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![0, 0]);

    match run_regressor(&x, attrs) {
        Err(OnnxError::InvalidModel(msg)) => assert!(msg.contains("nodes_featureids"), "{msg}"),
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

#[test]
fn short_truenodeids_array_is_rejected() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);

    let mut attrs = stump_attrs(0.5, 10.0, 20.0);
    attrs
        .int_lists
        .insert("nodes_truenodeids".into(), vec![1, 0]);

    assert!(matches!(
        run_regressor(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn short_class_weights_array_is_rejected() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);

    let mut attrs = Attributes::default();
    attrs.int_lists.insert("nodes_treeids".into(), vec![0]);
    attrs.int_lists.insert("nodes_nodeids".into(), vec![0]);
    attrs.int_lists.insert("nodes_featureids".into(), vec![0]);
    attrs.float_lists.insert("nodes_values".into(), vec![0.0]);
    attrs.int_lists.insert("nodes_truenodeids".into(), vec![0]);
    attrs.int_lists.insert("nodes_falsenodeids".into(), vec![0]);
    attrs
        .string_lists
        .insert("nodes_modes".into(), vec!["LEAF".into()]);
    attrs.int_lists.insert("class_treeids".into(), vec![0, 0]);
    attrs.int_lists.insert("class_nodeids".into(), vec![0, 0]);
    attrs.int_lists.insert("class_ids".into(), vec![0, 1]);
    // Only one weight for two class entries.
    attrs.float_lists.insert("class_weights".into(), vec![1.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);

    assert!(matches!(
        run_classifier(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

// ── [a3-8] NaN routing via nodes_missing_value_tracks_true ─────────────────

#[test]
fn nan_takes_true_branch_when_missing_tracks_true() {
    let x = Tensor::new(vec![f32::NAN], vec![1, 1]);

    let mut attrs = stump_attrs(0.5, 10.0, 20.0);
    attrs
        .int_lists
        .insert("nodes_missing_value_tracks_true".into(), vec![1, 0, 0]);

    let y = run_regressor(&x, attrs).expect("regressor");
    // The root tracks missing values to the true branch => leaf weight 10.
    assert!((y[0].data[0] - 10.0).abs() < 1e-5, "{:?}", y[0].data);
}

#[test]
fn nan_takes_false_branch_without_the_flag() {
    let x = Tensor::new(vec![f32::NAN], vec![1, 1]);

    let attrs = stump_attrs(0.5, 10.0, 20.0);
    let y = run_regressor(&x, attrs).expect("regressor");
    assert!((y[0].data[0] - 20.0).abs() < 1e-5, "{:?}", y[0].data);
}

#[test]
fn non_nan_values_ignore_the_missing_flag() {
    // The flag must not change routing for ordinary values.
    let x = Tensor::new(vec![9.0], vec![1, 1]);

    let mut attrs = stump_attrs(0.5, 10.0, 20.0);
    attrs
        .int_lists
        .insert("nodes_missing_value_tracks_true".into(), vec![1, 0, 0]);

    let y = run_regressor(&x, attrs).expect("regressor");
    assert!((y[0].data[0] - 20.0).abs() < 1e-5);
}

// ── [a3-18] aggregate_function ─────────────────────────────────────────────

#[test]
fn aggregate_functions_reduce_per_tree_contributions() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);

    for (function, expected) in [
        ("SUM", 30.0f32),
        ("AVERAGE", 15.0),
        ("MIN", 10.0),
        ("MAX", 20.0),
    ] {
        let mut attrs = two_leaf_trees(10.0, 20.0);
        attrs
            .strings
            .insert("aggregate_function".into(), function.into());
        let y = run_regressor(&x, attrs).expect("regressor");
        assert!(
            (y[0].data[0] - expected).abs() < 1e-5,
            "{function}: expected {expected}, got {}",
            y[0].data[0]
        );
    }
}

#[test]
fn absent_aggregate_function_defaults_to_sum() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);
    let y = run_regressor(&x, two_leaf_trees(10.0, 20.0)).expect("regressor");
    assert!((y[0].data[0] - 30.0).abs() < 1e-5);
}

#[test]
fn unknown_aggregate_function_is_rejected() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);
    let mut attrs = two_leaf_trees(10.0, 20.0);
    attrs
        .strings
        .insert("aggregate_function".into(), "MEDIAN".into());
    assert!(matches!(
        run_regressor(&x, attrs),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn base_values_are_added_after_averaging() {
    // onnxruntime divides the tree sum by the tree count and only then adds
    // base_values: (10 + 20) / 2 + 1 = 16.
    let x = Tensor::new(vec![0.0], vec![1, 1]);
    let mut attrs = two_leaf_trees(10.0, 20.0);
    attrs
        .strings
        .insert("aggregate_function".into(), "AVERAGE".into());
    attrs.float_lists.insert("base_values".into(), vec![1.0]);

    let y = run_regressor(&x, attrs).expect("regressor");
    assert!((y[0].data[0] - 16.0).abs() < 1e-5, "{:?}", y[0].data);
}

#[test]
fn min_aggregate_adds_base_value_after_the_reduction() {
    // MIN(10, 20) + 1 = 11 (the base value never participates in the min).
    let x = Tensor::new(vec![0.0], vec![1, 1]);
    let mut attrs = two_leaf_trees(10.0, 20.0);
    attrs
        .strings
        .insert("aggregate_function".into(), "MIN".into());
    attrs.float_lists.insert("base_values".into(), vec![1.0]);

    let y = run_regressor(&x, attrs).expect("regressor");
    assert!((y[0].data[0] - 11.0).abs() < 1e-5, "{:?}", y[0].data);
}

#[test]
fn min_aggregate_over_negative_weights_is_not_zero_clamped() {
    // Both trees are negative: MIN must be -7, not the 0.0 initial value.
    let x = Tensor::new(vec![0.0], vec![1, 1]);
    let mut attrs = two_leaf_trees(-3.0, -7.0);
    attrs
        .strings
        .insert("aggregate_function".into(), "MIN".into());

    let y = run_regressor(&x, attrs).expect("regressor");
    assert!((y[0].data[0] - (-7.0)).abs() < 1e-5, "{:?}", y[0].data);
}

#[test]
fn max_aggregate_over_negative_weights_is_not_zero_clamped() {
    let x = Tensor::new(vec![0.0], vec![1, 1]);
    let mut attrs = two_leaf_trees(-3.0, -7.0);
    attrs
        .strings
        .insert("aggregate_function".into(), "MAX".into());

    let y = run_regressor(&x, attrs).expect("regressor");
    assert!((y[0].data[0] - (-3.0)).abs() < 1e-5, "{:?}", y[0].data);
}

// ── [a3-15] 1-D input is a single sample ───────────────────────────────────

#[test]
fn one_dimensional_input_is_one_sample_with_c_features() {
    // x = [0.0, 9.0] as [2]: ONE sample whose feature 1 is 9.0.
    let x = Tensor::new(vec![0.0, 9.0], vec![2]);

    let mut attrs = stump_attrs(0.5, 10.0, 20.0);
    // Split on feature 1 (= 9.0) so a mis-shaped [2,1] read would answer 10.
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![1, 0, 0]);

    let result = run_regressor(&x, attrs).expect("regressor");
    let y = &result[0];
    assert_eq!(y.shape, vec![1, 1]);
    assert!((y.data[0] - 20.0).abs() < 1e-5, "{:?}", y.data);
}

#[test]
fn classifier_accepts_one_dimensional_input() {
    let x = Tensor::new(vec![0.0, 9.0], vec![2]);

    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("nodes_treeids".into(), vec![0, 0, 0]);
    attrs
        .int_lists
        .insert("nodes_nodeids".into(), vec![0, 1, 2]);
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![1, 0, 0]);
    attrs
        .float_lists
        .insert("nodes_values".into(), vec![0.5, 0.0, 0.0]);
    attrs
        .int_lists
        .insert("nodes_truenodeids".into(), vec![1, 0, 0]);
    attrs
        .int_lists
        .insert("nodes_falsenodeids".into(), vec![2, 0, 0]);
    attrs.string_lists.insert(
        "nodes_modes".into(),
        vec!["BRANCH_LEQ".into(), "LEAF".into(), "LEAF".into()],
    );
    attrs.int_lists.insert("class_treeids".into(), vec![0, 0]);
    attrs.int_lists.insert("class_nodeids".into(), vec![1, 2]);
    attrs.int_lists.insert("class_ids".into(), vec![0, 1]);
    attrs
        .float_lists
        .insert("class_weights".into(), vec![1.0, 1.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);

    let result = run_classifier(&x, attrs).expect("classifier");
    assert_eq!(result[0].shape, vec![1]);
    assert_eq!(result[1].shape, vec![1, 2]);
    // Feature 1 is 9.0 > 0.5 => false branch => class 1.
    assert!((result[0].data[0] - 1.0).abs() < 1e-5);
}
