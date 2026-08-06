//! Wave-3 `T6-tests-ops`: `TreeEnsembleClassifier`/`TreeEnsembleRegressor`
//! `post_transform` variants and multi-class (>2) ensembles, from finding
//! [a11-12].
//!
//! `oxionnx-ops/src/ml_tree/tests.rs` (pre-existing, colocated with the
//! implementation) already covers `aggregate_function` (SUM/AVERAGE/MIN/MAX),
//! `nodes_missing_value_tracks_true` NaN routing, cyclic/malformed node
//! tables, and 1-D single-sample inputs — all far beyond the "2 tests total"
//! the audit finding describes, so none of that is repeated here. What is
//! still genuinely uncovered anywhere in the tree is: a classifier with more
//! than two classes where more than one tree contributes to the *same* class
//! (ensemble summation across trees, not just across features), and every
//! `post_transform` value except the implicit `NONE` used by every existing
//! test (a corpus-wide grep for `SOFTMAX`/`LOGISTIC`/`SOFTMAX_ZERO`/`PROBIT`
//! under `ml_tree`/`ml_svm` finds zero matches; `SOFTMAX` alone is exercised,
//! but only via `LinearClassifier` in `oxionnx-ops/src/ml/tests.rs`, a
//! different operator with independent wiring).
//!
//! Reference values are hand-traced sums for the plain-SUM cases, and NumPy /
//! Python's `statistics.NormalDist` (true softmax and true probit,
//! respectively — not the crate's own approximation) for the transformed
//! ones; see the session's final report for the generating snippet.

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::Tensor;
use oxionnx_ops::registry::ml_ops::{TreeEnsembleClassifierOp, TreeEnsembleRegressorOp};

// ── Test infrastructure ──────────────────────────────────────────────────────

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
    let mut node = dummy_node(OpKind::TreeEnsembleClassifier);
    node.attrs = attrs;
    TreeEnsembleClassifierOp.execute(&ctx(&node, x))
}

fn run_regressor(x: &Tensor, attrs: Attributes) -> Result<Vec<Tensor>, oxionnx_core::OnnxError> {
    let mut node = dummy_node(OpKind::TreeEnsembleRegressor);
    node.attrs = attrs;
    TreeEnsembleRegressorOp.execute(&ctx(&node, x))
}

fn assert_close(got: f32, want: f32, tol: f32, label: &str) {
    assert!(
        (got - want).abs() < tol,
        "{label}: got {got}, want {want} (delta {})",
        (got - want).abs()
    );
}

/// Four independent single-leaf trees (no branching — `nodes_values`/
/// `nodes_featureids` are irrelevant, only `class_ids`/`class_weights`
/// matter). Trees 0 and 1 **both** contribute to class 0, exercising ensemble
/// summation across trees into the same class, alongside a genuine 3-class
/// output (`classlabels_int64s` has 3 entries).
fn four_tree_three_class_attrs() -> Attributes {
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("nodes_treeids".into(), vec![0, 1, 2, 3]);
    attrs
        .int_lists
        .insert("nodes_nodeids".into(), vec![0, 0, 0, 0]);
    attrs
        .int_lists
        .insert("nodes_featureids".into(), vec![0, 0, 0, 0]);
    attrs
        .float_lists
        .insert("nodes_values".into(), vec![0.0, 0.0, 0.0, 0.0]);
    attrs
        .int_lists
        .insert("nodes_truenodeids".into(), vec![0, 0, 0, 0]);
    attrs
        .int_lists
        .insert("nodes_falsenodeids".into(), vec![0, 0, 0, 0]);
    attrs.string_lists.insert(
        "nodes_modes".into(),
        vec!["LEAF".into(), "LEAF".into(), "LEAF".into(), "LEAF".into()],
    );
    attrs
        .int_lists
        .insert("class_treeids".into(), vec![0, 1, 2, 3]);
    attrs
        .int_lists
        .insert("class_nodeids".into(), vec![0, 0, 0, 0]);
    // Trees 0 and 1 both vote weight into class 0; trees 2 and 3 give the
    // other two classes their sole contribution.
    attrs.int_lists.insert("class_ids".into(), vec![0, 0, 1, 2]);
    attrs
        .float_lists
        .insert("class_weights".into(), vec![1.0, 0.5, 1.0, 1.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1, 2]);
    attrs
}

fn one_sample() -> Tensor {
    Tensor::new(vec![0.0], vec![1, 1])
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-class (>2), multi-tree-per-class ensemble summation
// ═══════════════════════════════════════════════════════════════════════════

/// class0 = tree0(1.0) + tree1(0.5) = 1.5, class1 = 1.0, class2 = 1.0. The
/// highest-scoring class (0) wins the label, and its score is the *sum* of
/// both contributing trees, not either one alone.
#[test]
fn three_class_four_trees_sums_contributions_per_class() {
    let mut attrs = four_tree_three_class_attrs();
    attrs.strings.insert("post_transform".into(), "NONE".into());

    let result = run_classifier(&one_sample(), attrs).expect("classifier");
    let scores = &result[1];
    assert_eq!(scores.shape, vec![1, 3]);
    assert_close(scores.data[0], 1.5, 1e-6, "class0 (two trees)");
    assert_close(scores.data[1], 1.0, 1e-6, "class1");
    assert_close(scores.data[2], 1.0, 1e-6, "class2");

    assert_close(result[0].data[0], 0.0, 1e-6, "predicted label is class 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// post_transform wiring
// ═══════════════════════════════════════════════════════════════════════════

/// `post_transform=SOFTMAX` on the same 3-class ensemble. Reference:
/// `scipy`-free NumPy softmax of `[1.5, 1.0, 1.0]` (max-subtraction form,
/// matching `apply_post_transform`'s own algorithm) = `[0.45186275,
/// 0.27406862, 0.27406862]`. Softmax is monotonic, so the label is unchanged.
#[test]
fn classifier_post_transform_softmax_wiring() {
    let mut attrs = four_tree_three_class_attrs();
    attrs
        .strings
        .insert("post_transform".into(), "SOFTMAX".into());

    let result = run_classifier(&one_sample(), attrs).expect("classifier");
    let scores = &result[1];
    assert_close(scores.data[0], 0.451_862_75, 1e-5, "softmax class0");
    assert_close(scores.data[1], 0.274_068_62, 1e-5, "softmax class1");
    assert_close(scores.data[2], 0.274_068_62, 1e-5, "softmax class2");
    let sum: f32 = scores.data.iter().sum();
    assert_close(sum, 1.0, 1e-5, "softmax rows sum to one");
    assert_close(
        result[0].data[0],
        0.0,
        1e-6,
        "argmax unchanged by a monotonic transform",
    );
}

/// `post_transform=LOGISTIC` applies elementwise sigmoid, with no
/// normalization constraint (unlike SOFTMAX, the row need not sum to 1).
/// Reference: `sigmoid(1.5) = 0.8175745`, `sigmoid(1.0) = 0.7310586`.
#[test]
fn classifier_post_transform_logistic_wiring() {
    let mut attrs = four_tree_three_class_attrs();
    attrs
        .strings
        .insert("post_transform".into(), "LOGISTIC".into());

    let result = run_classifier(&one_sample(), attrs).expect("classifier");
    let scores = &result[1];
    assert_close(scores.data[0], 0.817_574_5, 1e-5, "logistic class0");
    assert_close(scores.data[1], 0.731_058_6, 1e-5, "logistic class1");
    assert_close(scores.data[2], 0.731_058_6, 1e-5, "logistic class2");
}

/// `post_transform=SOFTMAX_ZERO`'s discriminator against plain SOFTMAX: a
/// class that received **no** leaf contribution stays at raw `0.0` and is
/// left untouched by the transform (excluded from the max-subtraction and the
/// normalizing sum), rather than receiving a small nonzero probability the
/// way ordinary SOFTMAX would. Two trees give class0=2.0, class1=-1.0; class2
/// is never mentioned by any `class_ids` entry, so it stays at its `0.0`
/// initial value. Reference: softmax of the two *nonzero* entries alone,
/// `[2.0, -1.0]` -> `[0.95257413, 0.04742587]`.
#[test]
fn classifier_post_transform_softmax_zero_leaves_untouched_classes_at_zero() {
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
    attrs.int_lists.insert("class_treeids".into(), vec![0, 1]);
    attrs.int_lists.insert("class_nodeids".into(), vec![0, 0]);
    attrs.int_lists.insert("class_ids".into(), vec![0, 1]); // class2 never appears
    attrs
        .float_lists
        .insert("class_weights".into(), vec![2.0, -1.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1, 2]);
    attrs
        .strings
        .insert("post_transform".into(), "SOFTMAX_ZERO".into());

    let result = run_classifier(&one_sample(), attrs).expect("classifier");
    let scores = &result[1];
    assert_eq!(scores.shape, vec![1, 3]);
    assert_eq!(
        scores.data[2], 0.0,
        "a class with no leaf contribution stays exactly 0.0, not a small nonzero probability"
    );
    assert_close(scores.data[0], 0.952_574_1, 1e-5, "softmax_zero class0");
    assert_close(scores.data[1], 0.047_425_87, 1e-5, "softmax_zero class1");
    let nonzero_sum = scores.data[0] + scores.data[1];
    assert_close(
        nonzero_sum,
        1.0,
        1e-5,
        "probability mass is conserved within the originally-nonzero group alone",
    );
}

/// `argmax_labels` now reads `all_scores` *before* `apply_post_transform`
/// mutates them in place (`oxionnx-ops/src/ml_tree.rs`, end of
/// `tree_ensemble_classifier`), matching onnxruntime's
/// `TreeAggregatorClassifier::FinalizeScores`: `get_max_weight` picks the
/// label from the raw aggregated predictions, and only afterwards does
/// `write_scores` apply `post_transform` to the score output. For a
/// monotonic transform (SOFTMAX, LOGISTIC, PROBIT on in-domain inputs) the
/// argmax is the same either way, so the ordering was never visible there —
/// but `SOFTMAX_ZERO` is deliberately *not* rank-order-preserving across
/// untouched-vs-touched classes: a class that only ever gets a raw score of
/// `0.0` (no tree contributes) can outrank a class that went negative on raw
/// votes, yet receives none of the probability mass once transformed. Here
/// class0 stays at raw `0.0` (untouched) and class1 gets `-5.0`: raw votes
/// favor class0 (`0.0 > -5.0`), and — now that labels come from the raw
/// scores — so does the predicted label, even though the transformed score
/// output still hands class1 the entire probability mass (`[0.0, 1.0]`).
#[test]
fn softmax_zero_does_not_flip_the_predicted_label() {
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
    attrs.int_lists.insert("class_treeids".into(), vec![0]);
    attrs.int_lists.insert("class_nodeids".into(), vec![0]);
    attrs.int_lists.insert("class_ids".into(), vec![1]); // class0 never appears -> stays 0.0
    attrs.float_lists.insert("class_weights".into(), vec![-5.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);
    attrs
        .strings
        .insert("post_transform".into(), "SOFTMAX_ZERO".into());

    let result = run_classifier(&one_sample(), attrs).expect("classifier");
    assert_eq!(
        result[1].data,
        vec![0.0, 1.0],
        "transformed scores: class0 untouched at 0.0, class1 gets all the mass"
    );
    assert_close(
        result[0].data[0],
        0.0,
        1e-6,
        "the label is class 0: the RAW scores (0.0 for class0 vs -5.0 for \
         class1) favor class 0, and the label is decided before \
         SOFTMAX_ZERO's asymmetric transform runs",
    );
}

/// `post_transform=PROBIT` on raw scores that already sit in the (0, 1)
/// domain PROBIT is meaningful for. Reference is the *true* probit function
/// (`statistics.NormalDist().inv_cdf`, not the crate's Abramowitz & Stegun
/// rational approximation): `probit(0.7) = 0.5244005127080407`,
/// `probit(0.3) = -0.5244005127080408`. Tolerance 2e-3 comfortably covers the
/// approximation's documented error (~4e-4 at these two points) plus f32
/// rounding; this is a wiring check (does TreeEnsembleClassifier apply PROBIT
/// at all), not an accuracy audit of the approximation itself.
#[test]
fn classifier_post_transform_probit_wiring() {
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
    attrs.int_lists.insert("class_treeids".into(), vec![0, 1]);
    attrs.int_lists.insert("class_nodeids".into(), vec![0, 0]);
    attrs.int_lists.insert("class_ids".into(), vec![0, 1]);
    attrs
        .float_lists
        .insert("class_weights".into(), vec![0.7, 0.3]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);
    attrs
        .strings
        .insert("post_transform".into(), "PROBIT".into());

    let result = run_classifier(&one_sample(), attrs).expect("classifier");
    let scores = &result[1];
    assert_close(scores.data[0], 0.524_400_5, 2e-3, "true probit(0.7)");
    assert_close(scores.data[1], -0.524_400_5, 2e-3, "true probit(0.3)");
}

/// `probit_inplace` (`oxionnx-ops/src/ml/post_transform.rs`) clamps its input
/// to `[1e-7, 1-1e-7]` before transforming — a defensive measure against
/// `ln(0)`, applied unconditionally even when the raw class score is nowhere
/// near a probability (a plain vote-count sum can be any real number).
/// class0's raw score of `2.0` clamps to the upper bound and class1's
/// untouched `0.0` clamps to the lower bound; by the symmetry of the probit
/// function and the symmetry of the two clamp bounds around 0.5,
/// `probit(clamp_hi)` and `probit(clamp_lo)` must be negatives of each other.
///
/// This used to fail: the clamped-high branch (`p >= 0.5`) recomputed
/// `1.0 - p` *after* `p` had already been rounded to the nearest f32 by the
/// clamp — and the nearest f32 to `1.0 - 1e-7` is `1.0 - f32::EPSILON`
/// (`0.999_999_88`, one ULP below 1.0), not `0.999_999_9`. Recomputing
/// `1.0 - p` from that rounded value recovered `1.192_092_9e-7`
/// (`f32::EPSILON`) instead of the intended `1e-7` — a ~19% relative error
/// from catastrophic cancellation, well outside the approximation's own
/// error budget, while the clamped-low branch (which never subtracts from
/// 1.0) landed within ~2.6e-4 of the true value. `probit_inplace` now derives
/// the tail probability `q = min(p, 1-p)` from the pre-clamp value and clamps
/// `q` itself symmetrically, so both tails funnel through the exact same
/// `q = 1e-7` and the two results are bit-exact negatives of one another
/// (not merely "close").
#[test]
fn probit_clamp_high_and_low_branches_are_exactly_symmetric() {
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
    attrs.int_lists.insert("class_treeids".into(), vec![0]);
    attrs.int_lists.insert("class_nodeids".into(), vec![0]);
    attrs.int_lists.insert("class_ids".into(), vec![0]); // class1 never appears -> stays 0.0
    attrs.float_lists.insert("class_weights".into(), vec![2.0]);
    attrs
        .int_lists
        .insert("classlabels_int64s".into(), vec![0, 1]);
    attrs
        .strings
        .insert("post_transform".into(), "PROBIT".into());

    let result = run_classifier(&one_sample(), attrs).expect("classifier");
    let scores = &result[1];
    // Fixed value, pinned tightly so a regression is visible. Both branches
    // now funnel through the identical `q = 1e-7` tail probability, so this
    // is also exactly `-probit(clamp(0.0))` (see the `assert_eq!` below) --
    // an order of magnitude closer to the true value (-5.199337...) than the
    // old cancellation-affected `5.166_328`.
    assert_close(
        scores.data[0],
        5.199_082,
        1e-4,
        "probit(clamp(2.0)): matches the AS approximation closely, post-fix",
    );
    assert_close(
        scores.data[1],
        -5.199_082,
        1e-4,
        "probit(clamp(0.0)): matches the AS approximation closely",
    );
    // The discriminator: the two clamp bounds are now symmetric enough that
    // the results are bit-exact negatives, not merely within a tolerance.
    assert_eq!(
        scores.data[0], -scores.data[1],
        "clamp-high and clamp-low branches must be exact mirror images"
    );
}

/// `TreeEnsembleRegressor` gets the same `post_transform` plumbing as the
/// classifier (`apply_post_transform` is shared). A single-tree, single-target
/// ensemble summing to a raw output of `2.0`, transformed by LOGISTIC:
/// `sigmoid(2.0) = 0.8807971`.
#[test]
fn regressor_post_transform_logistic_wiring() {
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
    attrs.int_lists.insert("target_treeids".into(), vec![0]);
    attrs.int_lists.insert("target_nodeids".into(), vec![0]);
    attrs.int_lists.insert("target_ids".into(), vec![0]);
    attrs.float_lists.insert("target_weights".into(), vec![2.0]);
    attrs.ints.insert("n_targets".into(), 1);
    attrs
        .strings
        .insert("post_transform".into(), "LOGISTIC".into());

    let result = run_regressor(&one_sample(), attrs).expect("regressor");
    assert_eq!(result[0].shape, vec![1, 1]);
    assert_close(result[0].data[0], 0.880_797_1, 1e-5, "sigmoid(2.0)");
}
