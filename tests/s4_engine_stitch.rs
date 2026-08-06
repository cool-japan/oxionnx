//! Stitch-wave regression tests for the planner ↔ kernel contract.
//!
//! Wave 1 rewrote several operators against the ONNX spec but could not touch
//! `src/optimizer/shape_inference_ext/**`, which predicts those operators'
//! output shapes.  Every test here is a **plan-vs-actual** check: it runs a real
//! session and asserts that the shape the planner recorded in
//! `Session::resolved_shapes()` is exactly the shape the operator produced —
//! *and* that both equal an independently derived expected value, so a shared
//! misreading of the spec cannot pass.
//!
//! Why plan-vs-actual is not merely a unit test of the inference function: a
//! disagreement is silent in production.  `execute_into_slots` resizes a slot it
//! was handed at the wrong size, so the run still returns the right numbers
//! while the buffer-pool reuse is wasted; and the same map is what
//! `run/dispatch.rs` validates provider results against and what the optimizer's
//! fusion gates read.
//!
//! | area                            | what was wrong before                    |
//! |---------------------------------|------------------------------------------|
//! | `infer_pool_shape`              | `auto_pad` ignored; no `ceil_mode` fix   |
//! | `infer_pool_shape`              | no shape for `MaxPool`'s `Indices`       |
//! | `infer_conv_transpose_shape`    | 4-entry `output_shape` and `auto_pad`    |
//! | `infer_resize_shape`            | `keep_aspect_ratio_policy` ignored       |
//! | `OpKind::DFT`                   | `axis` ignored, output rank hardcoded    |
//! | `infer_linear_*_shape`          | 1-D input read as N samples, not 1       |
//! | `Session::build_from_graph`     | optimizer saw no input shapes at all     |

use std::collections::HashMap;

use oxionnx::{
    Attributes, Graph, Node, OpKind, OptLevel, Session, SessionBuilder, Tensor, TensorInfo,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

fn node_with_attrs(
    op: OpKind,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Attributes,
) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs,
    }
}

fn ints(attrs: &mut Attributes, name: &str, values: &[i64]) {
    attrs.int_lists.insert(name.to_string(), values.to_vec());
}

fn int(attrs: &mut Attributes, name: &str, value: i64) {
    attrs.ints.insert(name.to_string(), value);
}

fn string(attrs: &mut Attributes, name: &str, value: &str) {
    attrs.strings.insert(name.to_string(), value.to_string());
}

fn floats(attrs: &mut Attributes, name: &str, values: &[f32]) {
    attrs.float_lists.insert(name.to_string(), values.to_vec());
}

/// `[0, 1, 2, …]` of the given length — the actual values never matter here,
/// only the shapes do.
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| i as f32).collect(), shape.to_vec())
}

/// Build a single-node session whose only input is a real graph input (so
/// constant folding cannot evaluate the node away), run it, and return the
/// outputs together with the session.
fn run_single(
    node: Node,
    input_name: &str,
    input: Tensor,
    output_names: &[&str],
    weights: HashMap<String, Tensor>,
) -> (Session, HashMap<String, Tensor>) {
    let graph = Graph {
        nodes: vec![node],
        input_names: names(&[input_name]),
        output_names: names(output_names),
        ..Default::default()
    };
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, weights)
        .expect("session build failed");
    let outputs = session
        .run_one(input_name, input)
        .expect("session run failed");
    (session, outputs)
}

/// The core assertion: expected == produced == planned.
fn assert_plan_matches_actual(
    session: &Session,
    outputs: &HashMap<String, Tensor>,
    name: &str,
    expected: &[usize],
) {
    let actual = outputs
        .get(name)
        .unwrap_or_else(|| panic!("output '{name}' missing from the run"));
    assert_eq!(
        actual.shape, expected,
        "operator produced the wrong shape for '{name}'"
    );
    let planned = session.resolved_shapes();
    let planned = planned
        .get(name)
        .unwrap_or_else(|| panic!("shape inference produced no shape for '{name}'"));
    assert_eq!(
        planned, expected,
        "shape inference disagrees with the operator for '{name}'"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// infer_pool_shape — auto_pad
// ─────────────────────────────────────────────────────────────────────────────

/// `MaxPool` over `[1, 1, 5, 5]`, kernel 3, stride 2, `auto_pad = SAME_UPPER`.
///
/// The spec's SAME target extent is `ceil(in / stride) = ceil(5 / 2) = 3`, which
/// the kernel reaches by padding `[1, 1]` per axis.  Inference ignored `auto_pad`
/// entirely and predicted the *unpadded* `(5 - 3) / 2 + 1 = 2`.
#[test]
fn max_pool_auto_pad_same_upper_is_planned_at_the_padded_extent() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "kernel_shape", &[3, 3]);
    ints(&mut attrs, "strides", &[2, 2]);
    string(&mut attrs, "auto_pad", "SAME_UPPER");
    let node = node_with_attrs(OpKind::MaxPool, "pool", &["x"], &["y"], attrs);

    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 5, 5]), &["y"], HashMap::new());
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 3, 3]);
}

/// `AveragePool` with `auto_pad = SAME_LOWER`: the odd pad pixel goes to the
/// *beginning*, but the extent is the same `ceil(in / stride)`.
#[test]
fn average_pool_auto_pad_same_lower_is_planned_at_the_padded_extent() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "kernel_shape", &[3, 3]);
    ints(&mut attrs, "strides", &[2, 2]);
    string(&mut attrs, "auto_pad", "SAME_LOWER");
    let node = node_with_attrs(OpKind::AveragePool, "pool", &["x"], &["y"], attrs);

    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 5, 5]), &["y"], HashMap::new());
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 3, 3]);
}

/// The discriminating `ceil_mode` case: `in = 5, k = 3, s = 3, pads = [0,0,2,2]`.
///
/// The bare ceil formula gives `ceil((5 + 2 - 3) / 3) + 1 = 3`, but a window
/// that *starts* inside the right-hand padding is illegal, so the operator (and
/// onnxruntime) drop it and produce 2.  Inference had the formula without the
/// correction and over-predicted by one on each axis.
#[test]
fn max_pool_ceil_mode_window_in_right_padding_is_dropped_by_the_planner_too() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "kernel_shape", &[3, 3]);
    ints(&mut attrs, "strides", &[3, 3]);
    ints(&mut attrs, "pads", &[0, 0, 2, 2]);
    int(&mut attrs, "ceil_mode", 1);
    let node = node_with_attrs(OpKind::MaxPool, "pool", &["x"], &["y"], attrs);

    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 5, 5]), &["y"], HashMap::new());
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 2, 2]);
}

/// `ceil_mode` where the extra window *is* legal: `in = 5, k = 2, s = 2`
/// yields `ceil((5 - 2) / 2) + 1 = 3` and the last window starts at 4, inside
/// the input — so the correction must not fire.
#[test]
fn max_pool_ceil_mode_keeps_a_window_that_starts_inside_the_input() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "kernel_shape", &[2, 2]);
    ints(&mut attrs, "strides", &[2, 2]);
    int(&mut attrs, "ceil_mode", 1);
    let node = node_with_attrs(OpKind::MaxPool, "pool", &["x"], &["y"], attrs);

    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 5, 5]), &["y"], HashMap::new());
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 3, 3]);
}

/// `MaxPool` with the optional `Indices` output.
///
/// `infer_shapes` zips the returned shapes positionally onto `node.outputs`, so
/// returning a single shape left `Indices` unplanned — and the slot fast path in
/// `run/dispatch.rs` requires *every* non-elided output to be planned, so the
/// node fell off it permanently.  Both outputs have the same shape.
#[test]
fn max_pool_indices_output_is_planned_so_the_node_stays_on_the_slot_path() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "kernel_shape", &[2, 2]);
    ints(&mut attrs, "strides", &[2, 2]);
    let node = node_with_attrs(OpKind::MaxPool, "pool", &["x"], &["y", "idx"], attrs);

    let (session, outputs) = run_single(
        node,
        "x",
        ramp(&[1, 1, 4, 4]),
        &["y", "idx"],
        HashMap::new(),
    );
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 2, 2]);
    assert_plan_matches_actual(&session, &outputs, "idx", &[1, 1, 2, 2]);
}

/// A `strides` attribute shorter than the spatial rank.
///
/// The operator fills the missing axes with the default 1 (`read_positive_pair`
/// reads `values.get(axis)` per axis).  Inference instead collected exactly what
/// the attribute held and then indexed `strides[1]` — an out-of-bounds **panic**
/// inside `Session::run`, since a graph input's shape is only known at run time.
#[test]
fn a_short_strides_attribute_defaults_the_missing_axis_instead_of_panicking() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "kernel_shape", &[2, 2]);
    ints(&mut attrs, "strides", &[2]);
    let node = node_with_attrs(OpKind::MaxPool, "pool", &["x"], &["y"], attrs);

    // stride 2 on H -> (4 - 2) / 2 + 1 = 2; default stride 1 on W -> 3.
    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 4, 4]), &["y"], HashMap::new());
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 2, 3]);
}

/// A negative `pads` entry is malformed.  It used to be cast with `as usize`,
/// wrapping to ~2^64, and the following `input_dim + pads[..]` overflow-panicked
/// in any build with `debug_assertions` — i.e. under `cargo test`.  Inference
/// must simply decline.
#[test]
fn a_negative_pad_makes_the_planner_decline_rather_than_panic() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "kernel_shape", &[2, 2]);
    ints(&mut attrs, "pads", &[-1, 0, 0, 0]);
    let node = node_with_attrs(OpKind::MaxPool, "pool", &["x"], &["y"], attrs);

    let graph = Graph {
        nodes: vec![node],
        input_names: names(&["x"]),
        output_names: names(&["y"]),
        ..Default::default()
    };
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, HashMap::new())
        .expect("building a session must not panic on a malformed attribute");
    // The operator rejects it too; the point is that neither path panics.
    let err = session.run_one("x", ramp(&[1, 1, 4, 4]));
    assert!(
        err.is_err(),
        "a negative pad is malformed and must be a typed error"
    );
    assert!(
        !session.resolved_shapes().contains_key("y"),
        "no shape may be predicted for a node the operator rejects"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// infer_conv_transpose_shape — output_shape / auto_pad
// ─────────────────────────────────────────────────────────────────────────────

/// `ConvTranspose` with the **4-entry** `output_shape` form (`[N, C, oH, oW]`,
/// which several exporters emit).
///
/// The operator accepts it and back-solves the padding, so the extent is the
/// requested 6.  Inference only recognised the 2-entry form, fell through to the
/// natural formula, and predicted `2 * (3 - 1) + 3 = 7`.
#[test]
fn conv_transpose_full_form_output_shape_is_honoured_verbatim() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "strides", &[2, 2]);
    ints(&mut attrs, "output_shape", &[1, 2, 6, 6]);
    let node = node_with_attrs(OpKind::ConvTranspose, "ct", &["x", "w"], &["y"], attrs);

    let mut weights = HashMap::new();
    // W is [C_in, C_out/group, kH, kW].
    weights.insert("w".to_string(), ramp(&[1, 2, 3, 3]));

    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 3, 3]), &["y"], weights);
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 2, 6, 6]);
}

/// `ConvTranspose` with `auto_pad = SAME_UPPER` and no `output_shape`: the spec
/// targets `out = in * stride = 6`.  Inference ignored `auto_pad` and predicted
/// the un-cropped 7.
#[test]
fn conv_transpose_auto_pad_same_upper_targets_input_times_stride() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "strides", &[2, 2]);
    string(&mut attrs, "auto_pad", "SAME_UPPER");
    let node = node_with_attrs(OpKind::ConvTranspose, "ct", &["x", "w"], &["y"], attrs);

    let mut weights = HashMap::new();
    weights.insert("w".to_string(), ramp(&[1, 1, 3, 3]));

    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 3, 3]), &["y"], weights);
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 6, 6]);
}

/// The plain `NOTSET` path must keep working: `stride * (in - 1) + k = 7`.
#[test]
fn conv_transpose_without_auto_pad_uses_the_natural_extent() {
    let mut attrs = Attributes::default();
    ints(&mut attrs, "strides", &[2, 2]);
    let node = node_with_attrs(OpKind::ConvTranspose, "ct", &["x", "w"], &["y"], attrs);

    let mut weights = HashMap::new();
    weights.insert("w".to_string(), ramp(&[1, 1, 3, 3]));

    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 3, 3]), &["y"], weights);
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 7, 7]);
}

// ─────────────────────────────────────────────────────────────────────────────
// infer_resize_shape — keep_aspect_ratio_policy
// ─────────────────────────────────────────────────────────────────────────────

fn resize_node(policy: &str) -> Node {
    let mut attrs = Attributes::default();
    string(&mut attrs, "mode", "nearest");
    string(&mut attrs, "keep_aspect_ratio_policy", policy);
    node_with_attrs(
        OpKind::Resize,
        "resize",
        &["x", "", "", "sizes"],
        &["y"],
        attrs,
    )
}

/// `sizes = [1,1,6,6]` on a `[1,1,4,5]` input with `not_larger`.
///
/// Ratios are `1, 1, 6/4 = 1.5, 6/5 = 1.2`; `not_larger` takes the smallest
/// (1.0) and rescales every axis by it, so the output is the *input* shape.
/// Inference used to return the requested `[1,1,6,6]` verbatim.
#[test]
fn resize_not_larger_policy_is_planned_like_the_operator() {
    let mut weights = HashMap::new();
    weights.insert(
        "sizes".to_string(),
        Tensor::new(vec![1.0, 1.0, 6.0, 6.0], vec![4]),
    );

    let (session, outputs) = run_single(
        resize_node("not_larger"),
        "x",
        ramp(&[1, 1, 4, 5]),
        &["y"],
        weights,
    );
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 4, 5]);
}

/// The same request with `not_smaller` takes the *largest* ratio (1.5), so every
/// axis grows: `round(1*1.5) = 2`, `round(4*1.5) = 6`, `round(5*1.5) = 8`
/// (halfway cases round up).
#[test]
fn resize_not_smaller_policy_is_planned_like_the_operator() {
    let mut weights = HashMap::new();
    weights.insert(
        "sizes".to_string(),
        Tensor::new(vec![1.0, 1.0, 6.0, 6.0], vec![4]),
    );

    let (session, outputs) = run_single(
        resize_node("not_smaller"),
        "x",
        ramp(&[1, 1, 4, 5]),
        &["y"],
        weights,
    );
    assert_plan_matches_actual(&session, &outputs, "y", &[2, 2, 6, 8]);
}

/// The default (`stretch`) policy must be unaffected: the requested sizes are
/// used verbatim.
#[test]
fn resize_default_stretch_policy_still_uses_sizes_verbatim() {
    let mut weights = HashMap::new();
    weights.insert(
        "sizes".to_string(),
        Tensor::new(vec![1.0, 1.0, 6.0, 6.0], vec![4]),
    );

    let (session, outputs) = run_single(resize_node(""), "x", ramp(&[1, 1, 4, 5]), &["y"], weights);
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 6, 6]);
}

/// The `scales` path uses `floor(dim * scale)` in f32, matching the operator.
#[test]
fn resize_scales_path_is_planned_like_the_operator() {
    let mut attrs = Attributes::default();
    string(&mut attrs, "mode", "nearest");
    let node = node_with_attrs(
        OpKind::Resize,
        "resize",
        &["x", "", "scales"],
        &["y"],
        attrs,
    );

    let mut weights = HashMap::new();
    // 4 * 1.5 = 6, 5 * 1.5 = 7.5 -> floor -> 7
    weights.insert(
        "scales".to_string(),
        Tensor::new(vec![1.0, 1.0, 1.5, 1.5], vec![4]),
    );

    let (session, outputs) = run_single(node, "x", ramp(&[1, 1, 4, 5]), &["y"], weights);
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1, 6, 7]);
}

// ─────────────────────────────────────────────────────────────────────────────
// DFT — the `axis` attribute
// ─────────────────────────────────────────────────────────────────────────────

/// A real signal `[3, 5, 1]` transformed along **axis 0** with `onesided = 1`.
///
/// The component axis (trailing 1) is stripped, the transform replaces
/// `outer[0] = 3` with `3 / 2 + 1 = 2`, and an explicit component axis of size 2
/// is appended: `[2, 5, 2]`.  Inference hardcoded `[in[0], in[1] / 2 + 1, 2]`
/// and predicted `[3, 3, 2]` — right rank, wrong extents, on the wrong axis.
#[test]
fn dft_axis_attribute_is_honoured_by_the_planner() {
    let mut attrs = Attributes::default();
    int(&mut attrs, "axis", 0);
    int(&mut attrs, "onesided", 1);
    let node = node_with_attrs(OpKind::DFT, "dft", &["x"], &["y"], attrs);

    let (session, outputs) = run_single(node, "x", ramp(&[3, 5, 1]), &["y"], HashMap::new());
    assert_plan_matches_actual(&session, &outputs, "y", &[2, 5, 2]);
}

/// The spec default `axis = -2` on a rank-4 complex input `[2, 4, 3, 2]`:
/// logical rank is 4, `-2` normalises to axis 2, which is left unchanged when
/// `onesided = 0`, so the output keeps the input's shape.  Inference predicted a
/// rank-**3** `[2, 4, 2]`.
#[test]
fn dft_default_axis_on_a_rank_4_input_keeps_the_full_outer_shape() {
    let node = node_with_attrs(OpKind::DFT, "dft", &["x"], &["y"], Attributes::default());

    let (session, outputs) = run_single(node, "x", ramp(&[2, 4, 3, 2]), &["y"], HashMap::new());
    assert_plan_matches_actual(&session, &outputs, "y", &[2, 4, 3, 2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX-ML linear operators — a 1-D input is ONE sample
// ─────────────────────────────────────────────────────────────────────────────

/// `LinearClassifier` on a 1-D `[3]` input.
///
/// The ONNX-ML convention (and `batch_dims` in `oxionnx-ops/src/ml/shape.rs`) is
/// that `[C]` is one sample with `C` features, so the operator emits 1 label and
/// a `[1, 2]` score row.  Inference read it as 3 samples with 1 feature each and
/// predicted `[3]` / `[3, 6]`.
#[test]
fn linear_classifier_reads_a_1d_input_as_one_sample() {
    let mut attrs = Attributes::default();
    // 2 classes x 3 features.
    floats(&mut attrs, "coefficients", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    floats(&mut attrs, "intercepts", &[0.0, 0.0]);
    ints(&mut attrs, "classlabels_ints", &[0, 1]);
    let node = node_with_attrs(
        OpKind::LinearClassifier,
        "lc",
        &["x"],
        &["label", "scores"],
        attrs,
    );

    let (session, outputs) = run_single(
        node,
        "x",
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        &["label", "scores"],
        HashMap::new(),
    );
    assert_plan_matches_actual(&session, &outputs, "label", &[1]);
    assert_plan_matches_actual(&session, &outputs, "scores", &[1, 2]);
}

/// Binary one-vs-rest: a *single* coefficient row with two declared class labels
/// is expanded to `[-s, s]`, so the score tensor has 2 columns even though the
/// raw coefficient count implies 1.
#[test]
fn linear_classifier_binary_one_vs_rest_is_planned_with_two_score_columns() {
    let mut attrs = Attributes::default();
    // 1 row x 2 features.
    floats(&mut attrs, "coefficients", &[1.0, -1.0]);
    floats(&mut attrs, "intercepts", &[0.5]);
    ints(&mut attrs, "classlabels_ints", &[0, 1]);
    let node = node_with_attrs(
        OpKind::LinearClassifier,
        "lc",
        &["x"],
        &["label", "scores"],
        attrs,
    );

    let (session, outputs) = run_single(
        node,
        "x",
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        &["label", "scores"],
        HashMap::new(),
    );
    assert_plan_matches_actual(&session, &outputs, "label", &[2]);
    assert_plan_matches_actual(&session, &outputs, "scores", &[2, 2]);
}

/// `LinearRegressor` on a 1-D `[4]` input: one sample, four features, one
/// target.  Inference predicted `[4, 4]`.
#[test]
fn linear_regressor_reads_a_1d_input_as_one_sample() {
    let mut attrs = Attributes::default();
    floats(&mut attrs, "coefficients", &[1.0, 1.0, 1.0, 1.0]);
    floats(&mut attrs, "intercepts", &[0.0]);
    let node = node_with_attrs(OpKind::LinearRegressor, "lr", &["x"], &["y"], attrs);

    let (session, outputs) = run_single(
        node,
        "x",
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
        &["y"],
        HashMap::new(),
    );
    assert_plan_matches_actual(&session, &outputs, "y", &[1, 1]);
}

/// An explicit `targets` attribute wins over the coefficient-count inference,
/// as it does in the operator.
#[test]
fn linear_regressor_honours_an_explicit_targets_attribute() {
    let mut attrs = Attributes::default();
    // 2 targets x 2 features.
    floats(&mut attrs, "coefficients", &[1.0, 0.0, 0.0, 1.0]);
    floats(&mut attrs, "intercepts", &[0.0, 0.0]);
    int(&mut attrs, "targets", 2);
    let node = node_with_attrs(OpKind::LinearRegressor, "lr", &["x"], &["y"], attrs);

    let (session, outputs) = run_single(
        node,
        "x",
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        &["y"],
        HashMap::new(),
    );
    assert_plan_matches_actual(&session, &outputs, "y", &[2, 2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Session::build_from_graph — the optimizer is seeded with the declared inputs
// ─────────────────────────────────────────────────────────────────────────────

fn matmul_add_graph(input_info: Option<TensorInfo>) -> (Graph, HashMap<String, Tensor>) {
    let graph = Graph {
        nodes: vec![
            Node {
                op: OpKind::MatMul,
                name: "mm".to_string(),
                inputs: names(&["x", "w"]),
                outputs: names(&["t"]),
                attrs: Attributes::default(),
            },
            Node {
                op: OpKind::Add,
                name: "add".to_string(),
                inputs: names(&["t", "b"]),
                outputs: names(&["y"]),
                attrs: Attributes::default(),
            },
        ],
        input_names: names(&["x"]),
        output_names: names(&["y"]),
        input_infos: input_info.into_iter().collect(),
        ..Default::default()
    };
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), ramp(&[3, 4]));
    weights.insert("b".to_string(), ramp(&[4]));
    (graph, weights)
}

fn static_input_info(name: &str, dims: &[usize]) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        shape: dims.iter().map(|&d| Some(d)).collect(),
        dim_params: vec![None; dims.len()],
        ..Default::default()
    }
}

fn has_gemm(session: &Session) -> bool {
    session.nodes().iter().any(|n| n.op_type == "Gemm")
}

/// `MatMul + Add → Gemm` is gated on a provably rank-2 activation.  Load-time
/// shape inference started from an *empty* input-shape map, so it never learned
/// the rank of a graph input and the fusion could not fire on any real model.
/// Seeding it from the declared `input_infos` restores that.
#[test]
fn a_statically_shaped_declared_input_unlocks_the_gemm_fusion() {
    let (graph, weights) = matmul_add_graph(Some(static_input_info("x", &[2, 3])));
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, weights)
        .expect("session build failed");

    assert!(
        has_gemm(&session),
        "MatMul+Add over a declared [2, 3] input must fuse to Gemm"
    );

    let outputs = session
        .run_one("x", ramp(&[2, 3]))
        .expect("the fused graph must still run");
    // y = x @ w + b, computed with numpy:
    //   x = [[0,1,2],[3,4,5]], w = [[0,1,2,3],[4,5,6,7],[8,9,10,11]], b = [0,1,2,3]
    //   x @ w = [[20,23,26,29],[56,68,80,92]]
    assert_eq!(
        outputs.get("y").expect("y").data,
        vec![20.0, 24.0, 28.0, 32.0, 56.0, 69.0, 82.0, 95.0]
    );
}

/// The seeding must be **static-only**.  An input with a symbolic axis is not
/// seeded at all: substituting a placeholder would feed a fabricated dimension
/// to the passes that *size synthesised constants* from the inferred shapes.
/// The graph must still build and run correctly, just unfused.
#[test]
fn a_symbolic_input_dimension_is_never_given_a_placeholder() {
    let dynamic = TensorInfo {
        name: "x".to_string(),
        shape: vec![None, Some(3)],
        dim_params: vec![Some("batch".to_string()), None],
        ..Default::default()
    };
    let (graph, weights) = matmul_add_graph(Some(dynamic));
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, weights)
        .expect("session build failed");

    assert!(
        !has_gemm(&session),
        "a symbolic batch axis must not be materialised into a concrete shape"
    );

    // Both batch sizes must run, which is the reason the axis stayed symbolic.
    let one = session.run_one("x", ramp(&[1, 3])).expect("batch 1");
    assert_eq!(one.get("y").expect("y").shape, vec![1, 4]);
    let four = session.run_one("x", ramp(&[4, 3])).expect("batch 4");
    assert_eq!(four.get("y").expect("y").shape, vec![4, 4]);
}

// ─────────────────────────────────────────────────────────────────────────────
// ShapePlanCache — alternating input shapes stop re-inferring every run
// ─────────────────────────────────────────────────────────────────────────────

/// The single-slot memo holds one plan, so a server alternating two batch sizes
/// missed it on *every* run and paid a full `infer_shapes` pass each time.  The
/// plan cache keeps the last few; this pins the observable half of that — the
/// shapes must stay correct across the alternation, and
/// `Session::resolved_shapes()` must always describe the run that just
/// finished, whether it was served from the cache or recomputed.
#[test]
fn alternating_batch_sizes_stay_correct_through_the_shape_plan_cache() {
    let dynamic = TensorInfo {
        name: "x".to_string(),
        shape: vec![None, Some(3)],
        dim_params: vec![Some("batch".to_string()), None],
        ..Default::default()
    };
    let (graph, weights) = matmul_add_graph(Some(dynamic));
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, weights)
        .expect("session build failed");

    for _ in 0..4 {
        for &batch in &[1_usize, 8, 1, 8, 3] {
            let outputs = session
                .run_one("x", ramp(&[batch, 3]))
                .expect("every batch size must run");
            let y = outputs.get("y").expect("y");
            assert_eq!(
                y.shape,
                vec![batch, 4],
                "wrong output shape for batch {batch}"
            );
            assert_eq!(
                session.resolved_shapes().get("y"),
                Some(&vec![batch, 4]),
                "resolved_shapes must describe the run that just finished (batch {batch})"
            );
        }
    }
}

/// The same alternation from several threads at once: `Session` is `Send +
/// Sync` and is routinely shared behind a web handler, so the cache must never
/// hand one run another run's shapes.
#[test]
fn concurrent_alternating_runs_never_see_another_run_s_plan() {
    let dynamic = TensorInfo {
        name: "x".to_string(),
        shape: vec![None, Some(3)],
        dim_params: vec![Some("batch".to_string()), None],
        ..Default::default()
    };
    let (graph, weights) = matmul_add_graph(Some(dynamic));
    let session = std::sync::Arc::new(
        SessionBuilder::new()
            .with_optimization_level(OptLevel::All)
            .build_from_graph(graph, weights)
            .expect("session build failed"),
    );

    let handles: Vec<_> = [1_usize, 2, 8, 16, 32]
        .into_iter()
        .map(|batch| {
            let session = std::sync::Arc::clone(&session);
            std::thread::spawn(move || {
                for _ in 0..40 {
                    let outputs = session
                        .run_one("x", ramp(&[batch, 3]))
                        .expect("concurrent run failed");
                    assert_eq!(outputs.get("y").expect("y").shape, vec![batch, 4]);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("worker thread panicked");
    }
}
