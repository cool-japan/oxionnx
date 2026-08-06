//! Wave-2 session-level end-to-end tests for `W2-registry-lane-B`: `Det`,
//! `Col2Im`, `CenterCropPad`, the `Random*`/`Multinomial` generator family,
//! `NegativeLogLikelihoodLoss` and `SoftmaxCrossEntropyLoss`, plus
//! verification coverage for `GatherND`'s `batch_dims`, `IsInf`'s
//! `detect_negative`/`detect_positive`, and the out-of-scope
//! `Sequence*`/`Optional*` operator family (no sequence/optional value type
//! exists in this runtime -- see the module doc at the bottom of this file).
//!
//! These run through the *real* `Session` (graph build, shape/type
//! resolution, output-slot allocation) rather than calling an `Operator`
//! directly, so they catch anything the plain per-operator unit tests
//! (colocated with each op under `oxionnx-ops/src/registry/`) cannot: in
//! particular, whether a **rank-0** operator output (`Det` on a 2-D input)
//! survives graph execution instead of being silently promoted to `[1]`
//! somewhere in the session pipeline.
//!
//! Reference values are `onnx.reference` (`onnx` 1.21.0) via
//! `ReferenceEvaluator`, or `numpy`/`math` closed forms where noted; the
//! generating scripts are inline docs on the per-op unit tests under
//! `oxionnx-ops/src/registry/{det,col2im,center_crop_pad,random_ops,loss_ops}`.

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

// ── helpers (mirrors tests/w2_vision_ops_e2e.rs) ────────────────────────────

fn run_op(
    op: OpKind,
    node_inputs: &[&str],
    node_outputs: &[&str],
    feeds: Vec<(&str, Tensor)>,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let graph = Graph {
        nodes: vec![Node {
            op,
            name: "op0".to_string(),
            inputs: node_inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: node_outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs,
        }],
        input_names: feeds.iter().map(|(n, _)| (*n).to_string()).collect(),
        output_names: node_outputs.iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let feed: HashMap<&str, Tensor> = feeds.into_iter().collect();
    session.run(&feed).expect("run")
}

fn try_run_op(
    op: OpKind,
    node_inputs: &[&str],
    node_outputs: &[&str],
    feeds: Vec<(&str, Tensor)>,
    attrs: Attributes,
) -> Result<HashMap<String, Tensor>, oxionnx::OnnxError> {
    let graph = Graph {
        nodes: vec![Node {
            op,
            name: "op0".to_string(),
            inputs: node_inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: node_outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs,
        }],
        input_names: feeds.iter().map(|(n, _)| (*n).to_string()).collect(),
        output_names: node_outputs.iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");
    let feed: HashMap<&str, Tensor> = feeds.into_iter().collect();
    session.run(&feed)
}

// ── registry / OpKind wiring ────────────────────────────────────────────────

#[test]
fn lane_b_ops_are_registered() {
    let registry = oxionnx_ops::default_registry();
    for name in [
        "Det",
        "Col2Im",
        "CenterCropPad",
        "RandomNormal",
        "RandomUniform",
        "RandomNormalLike",
        "RandomUniformLike",
        "Multinomial",
        "NegativeLogLikelihoodLoss",
        "SoftmaxCrossEntropyLoss",
    ] {
        assert!(registry.contains(name), "{name} must be registered");
        let kind = OpKind::parse(name);
        assert_ne!(
            kind,
            OpKind::Unknown(name.to_string()),
            "{name} must have its own OpKind variant"
        );
    }
}

// ── Det: rank-0 through the real session path ───────────────────────────────

/// The critical check for this whole file: `Det` of a plain 2-D input is an
/// ONNX **scalar** (rank 0, shape `[]`). The per-operator unit test already
/// pins that `DetOp::execute` returns `Tensor::rank0`; this test additionally
/// exercises the full `Session` (shape resolution, output-slot allocation,
/// final-output collection) to make sure nothing along that path promotes it
/// to the legacy `[1]` representation. Reference: `numpy.linalg.det` /
/// `onnx.reference` both give `-3.0`, shape `()`.
#[test]
fn det_rank0_output_survives_the_session_path() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    );
    let out = run_op(
        OpKind::Det,
        &["x"],
        &["y"],
        vec![("x", x)],
        Attributes::default(),
    );
    assert_eq!(
        out["y"].shape,
        Vec::<usize>::new(),
        "Det of a 2-D input must be rank-0 end-to-end, not promoted to [1]"
    );
    assert!((out["y"].data[0] - (-3.0)).abs() < 1e-4);
}

/// Batched `Det` (`[2, 3, 3]` input) produces a rank-1 `[2]` output, and both
/// the value and the rank must survive the session, not just `execute`.
#[test]
fn det_batched_output_survives_the_session_path() {
    let x = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, //
            2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0,
        ],
        vec![2, 3, 3],
    );
    let out = run_op(
        OpKind::Det,
        &["x"],
        &["y"],
        vec![("x", x)],
        Attributes::default(),
    );
    assert_eq!(out["y"].shape, vec![2]);
    assert!((out["y"].data[0] - (-3.0)).abs() < 1e-4);
    assert!((out["y"].data[1] - 24.0).abs() < 1e-3);
}

// ── Col2Im ──────────────────────────────────────────────────────────────────

/// Reference: `onnx.reference` `Col2Im` (opset 21) -- overlapping-block
/// accumulation case, run through the full session (`image_shape` and
/// `block_shape` arrive as ordinary graph inputs, exactly as a real model
/// would supply them, usually as initializers).
#[test]
fn col2im_overlapping_blocks_through_session() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".into(), vec![1, 1]);
    let out = run_op(
        OpKind::Col2Im,
        &["x", "image_shape", "block_shape"],
        &["y"],
        vec![
            (
                "x",
                Tensor::new((1..=16).map(|v| v as f32).collect(), vec![1, 4, 4]),
            ),
            ("image_shape", Tensor::new(vec![3.0, 3.0], vec![2])),
            ("block_shape", Tensor::new(vec![2.0, 2.0], vec![2])),
        ],
        attrs,
    );
    assert_eq!(out["y"].shape, vec![1, 1, 3, 3]);
    assert_eq!(
        out["y"].data,
        vec![1.0, 7.0, 6.0, 12.0, 34.0, 22.0, 11.0, 27.0, 16.0]
    );
}

// ── CenterCropPad ────────────────────────────────────────────────────────────

/// Mixed crop+pad (odd amounts on each axis) through the full session.
/// Reference: `onnx.reference` `CenterCropPad` (opset 21).
#[test]
fn center_crop_pad_mixed_through_session() {
    let out = run_op(
        OpKind::CenterCropPad,
        &["x", "shape"],
        &["y"],
        vec![
            (
                "x",
                Tensor::new((1..=20).map(|v| v as f32).collect(), vec![4, 5]),
            ),
            ("shape", Tensor::new(vec![3.0, 8.0], vec![2])),
        ],
        Attributes::default(),
    );
    assert_eq!(out["y"].shape, vec![3, 8]);
    assert_eq!(
        out["y"].data,
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, //
            0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, //
            0.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0,
        ]
    );
}

// ── Random* / Multinomial ───────────────────────────────────────────────────

#[test]
fn random_uniform_through_session_respects_shape_and_range() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("shape".into(), vec![3, 5]);
    attrs.floats.insert("low".into(), -1.0);
    attrs.floats.insert("high".into(), 1.0);
    attrs.floats.insert("seed".into(), 21.0);
    let out = run_op(OpKind::RandomUniform, &[], &["y"], vec![], attrs);
    assert_eq!(out["y"].shape, vec![3, 5]);
    for &v in &out["y"].data {
        assert!((-1.0..1.0).contains(&v));
    }
}

#[test]
fn random_normal_like_through_session_matches_input_shape() {
    let out = run_op(
        OpKind::RandomNormalLike,
        &["x"],
        &["y"],
        vec![("x", Tensor::zeros(&[2, 3]))],
        Attributes::default(),
    );
    assert_eq!(out["y"].shape, vec![2, 3]);
}

#[test]
fn multinomial_through_session() {
    let mut attrs = Attributes::default();
    attrs.ints.insert("sample_size".into(), 10);
    attrs.floats.insert("seed".into(), 4.0);
    let out = run_op(
        OpKind::Multinomial,
        &["x"],
        &["y"],
        vec![("x", Tensor::new(vec![20.0, -20.0], vec![1, 2]))],
        attrs,
    );
    assert_eq!(out["y"].shape, vec![1, 10]);
    assert!(out["y"].data.iter().all(|&v| v == 0.0));
}

// ── NegativeLogLikelihoodLoss / SoftmaxCrossEntropyLoss ─────────────────────

/// Reference: `onnx.reference` `NegativeLogLikelihoodLoss` (opset 21), mean
/// reduction (the rank-0 output case) through the full session.
#[test]
fn nll_loss_mean_through_session() {
    let out = run_op(
        OpKind::NegativeLogLikelihoodLoss,
        &["input", "target"],
        &["y"],
        vec![
            (
                "input",
                Tensor::new(
                    vec![
                        -1.2, -0.5, -2.1, -3.0, //
                        -0.1, -2.5, -1.8, -0.9, //
                        -3.3, -0.2, -1.1, -2.7,
                    ],
                    vec![3, 4],
                ),
            ),
            ("target", Tensor::new(vec![1.0, 0.0, 2.0], vec![3])),
        ],
        Attributes::default(),
    );
    assert_eq!(out["y"].shape, Vec::<usize>::new());
    assert!((out["y"].data[0] - 0.566_666_66).abs() < 1e-5);
}

/// Reference: `onnx.reference` `SoftmaxCrossEntropyLoss` (opset 21), through
/// the full session, with both outputs requested.
#[test]
fn softmax_cross_entropy_loss_two_outputs_through_session() {
    let out = run_op(
        OpKind::SoftmaxCrossEntropyLoss,
        &["scores", "labels"],
        &["y", "log_prob"],
        vec![
            (
                "scores",
                Tensor::new(
                    vec![
                        1.0, 2.0, 0.5, //
                        0.2, 0.1, 3.0, //
                        1.5, 1.5, 1.5,
                    ],
                    vec![3, 3],
                ),
            ),
            ("labels", Tensor::new(vec![1.0, 2.0, 0.0], vec![3])),
        ],
        Attributes::default(),
    );
    assert!((out["y"].data[0] - 0.557_527_5).abs() < 1e-5);
    assert_eq!(out["log_prob"].shape, vec![3, 3]);
}

/// `reduction="none"` combined with the optional second (`log_prob`) output:
/// the two outputs then have **different ranks** (`y` is rank-1 `[N]`,
/// `log_prob` stays rank-2 `[N, C]`), which is the one cell of the
/// reduction x output-count matrix the other two tests here don't cover --
/// and the one most likely to trip up session-level output-slot
/// pre-allocation (which sizes each slot independently per declared output).
/// Reference: `onnx.reference`.
#[test]
fn softmax_cross_entropy_loss_none_reduction_with_two_outputs_through_session() {
    let out = run_op(
        OpKind::SoftmaxCrossEntropyLoss,
        &["scores", "labels"],
        &["y", "log_prob"],
        vec![
            (
                "scores",
                Tensor::new(
                    vec![
                        1.0, 2.0, 0.5, //
                        0.2, 0.1, 3.0, //
                        1.5, 1.5, 1.5,
                    ],
                    vec![3, 3],
                ),
            ),
            ("labels", Tensor::new(vec![1.0, 2.0, 0.0], vec![3])),
        ],
        {
            let mut a = Attributes::default();
            a.strings.insert("reduction".into(), "none".into());
            a
        },
    );
    assert_eq!(out["y"].shape, vec![3], "unreduced loss is rank-1 [N]");
    let expected_y = [0.464_368_82_f32, 0.109_601_45, 1.098_612_3];
    for (a, e) in out["y"].data.iter().zip(expected_y) {
        assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
    }
    assert_eq!(
        out["log_prob"].shape,
        vec![3, 3],
        "log_prob stays rank-2 [N, C]"
    );
    let expected_log_prob = [
        -1.464_368_8_f32,
        -0.464_368_82,
        -1.964_368_8,
        -2.909_601_4,
        -3.009_601_6,
        -0.109_601_45,
        -1.098_612_3,
        -1.098_612_3,
        -1.098_612_3,
    ];
    for (a, e) in out["log_prob"].data.iter().zip(expected_log_prob) {
        assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
    }
}

// ── Verification: GatherND batch_dims (existing implementation) ────────────

/// `batch_dims = 1` on a `[2, 2, 2]` tensor. Reference: `onnx.reference`
/// `GatherND` (opset 21) -- also matches the ONNX spec's own worked example
/// for this exact input.
#[test]
fn gathernd_batch_dims_1_matches_onnx_reference() {
    let out = run_op(
        OpKind::GatherND,
        &["data", "indices"],
        &["y"],
        vec![
            (
                "data",
                Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 2, 2]),
            ),
            ("indices", Tensor::new(vec![1.0, 0.0], vec![2, 1])),
        ],
        {
            let mut a = Attributes::default();
            a.ints.insert("batch_dims".into(), 1);
            a
        },
    );
    assert_eq!(out["y"].shape, vec![2, 2]);
    assert_eq!(out["y"].data, vec![2.0, 3.0, 4.0, 5.0]);
}

/// `batch_dims = 2` (both leading dims are batch dims, `K = 1` indexes the
/// single remaining axis). Reference: `onnx.reference`.
#[test]
fn gathernd_batch_dims_2_matches_onnx_reference() {
    let out = run_op(
        OpKind::GatherND,
        &["data", "indices"],
        &["y"],
        vec![
            (
                "data",
                Tensor::new((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]),
            ),
            (
                "indices",
                Tensor::new(vec![2.0, 0.0, 1.0, 2.0], vec![2, 2, 1]),
            ),
        ],
        {
            let mut a = Attributes::default();
            a.ints.insert("batch_dims".into(), 2);
            a
        },
    );
    assert_eq!(out["y"].shape, vec![2, 2]);
    assert_eq!(out["y"].data, vec![3.0, 4.0, 8.0, 12.0]);
}

// ── Verification: IsInf detect_negative / detect_positive ──────────────────

#[test]
fn isinf_detect_flags_all_four_combinations_match_onnx_reference() {
    let x = Tensor::new(
        vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.5, -3.2, f32::NAN],
        vec![6],
    );
    let cases: [(i64, i64, [f32; 6]); 4] = [
        (1, 1, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        (1, 0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (0, 1, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        (0, 0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    for (dp, dn, expected) in cases {
        let mut attrs = Attributes::default();
        attrs.ints.insert("detect_positive".into(), dp);
        attrs.ints.insert("detect_negative".into(), dn);
        let out = run_op(OpKind::IsInf, &["x"], &["y"], vec![("x", x.clone())], attrs);
        assert_eq!(
            out["y"].data,
            expected.to_vec(),
            "detect_positive={dp} detect_negative={dn}"
        );
    }
}

// ── Verification: Mish / Celu / Gelu(approximate) ───────────────────────────

/// Closed-form reference values computed directly from the ONNX spec formula
/// via Python `math` (`mish(x) = x * tanh(softplus(x))`), independent of this
/// engine's implementation.
#[test]
fn mish_matches_closed_form_reference() {
    let xs = vec![-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5, 5.0];
    let expected = [
        -0.145_647_46,
        -0.303_401_46,
        -0.056_788_58,
        0.0,
        0.063_179_42,
        0.865_098_4,
        2.471_392_3,
        4.999_552,
    ];
    let out = run_op(
        OpKind::Mish,
        &["x"],
        &["y"],
        vec![("x", Tensor::new(xs.clone(), vec![xs.len()]))],
        Attributes::default(),
    );
    for (a, e) in out["y"].data.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
    }
}

/// `Gelu` with `approximate="none"` (the opset-20 default) must use the
/// *exact* erf-based formula, not the tanh approximation. Closed-form
/// reference via Python `math.erf`.
#[test]
fn gelu_exact_matches_closed_form_reference() {
    let xs = vec![-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5, 5.0];
    let expected = [
        -0.004_049_69,
        -0.158_655_25,
        -0.046_017_22,
        0.0,
        0.053_982_78,
        0.841_344_8,
        2.484_475_9,
        4.999_998_6,
    ];
    let out = run_op(
        OpKind::Gelu,
        &["x"],
        &["y"],
        vec![("x", Tensor::new(xs.clone(), vec![xs.len()]))],
        Attributes::default(),
    );
    for (a, e) in out["y"].data.iter().zip(expected) {
        assert!((a - e).abs() < 2e-6, "got {a}, expected {e}");
    }
}

/// `Gelu` with `approximate="tanh"` must switch to the tanh approximation
/// `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))` -- a visibly different
/// curve from the exact erf form (e.g. at `x = -3`: `-0.00364` vs. `-0.00405`
/// exact), so this also discriminates "attribute read but ignored" from
/// "attribute correctly dispatched". Closed-form reference via Python `math`.
#[test]
fn gelu_tanh_approximate_matches_closed_form_reference() {
    let xs = vec![-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5, 5.0];
    let expected = [
        -0.003_637_39,
        -0.158_808_01,
        -0.046_017_25,
        0.0,
        0.053_982_75,
        0.841_192,
        2.484_915_7,
        5.0,
    ];
    let mut attrs = Attributes::default();
    attrs.strings.insert("approximate".into(), "tanh".into());
    let out = run_op(
        OpKind::Gelu,
        &["x"],
        &["y"],
        vec![("x", Tensor::new(xs.clone(), vec![xs.len()]))],
        attrs,
    );
    for (a, e) in out["y"].data.iter().zip(expected) {
        assert!((a - e).abs() < 2e-5, "got {a}, expected {e}");
    }
}

/// `Celu` with a non-default `alpha`. Closed-form reference:
/// `max(0,x) + min(0, alpha*(exp(x/alpha)-1))`.
#[test]
fn celu_alpha_2_matches_closed_form_reference() {
    let xs = vec![-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5, 5.0];
    let expected = [
        -1.553_739_7,
        -0.786_938_68,
        -0.097_541_15,
        0.0,
        0.1,
        1.0,
        2.5,
        5.0,
    ];
    let mut attrs = Attributes::default();
    attrs.floats.insert("alpha".into(), 2.0);
    let out = run_op(
        OpKind::Celu,
        &["x"],
        &["y"],
        vec![("x", Tensor::new(xs.clone(), vec![xs.len()]))],
        attrs,
    );
    for (a, e) in out["y"].data.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
    }
}

// ── Out of scope: Sequence* / Optional* / string ops ────────────────────────
//
// This runtime's `Tensor` is a flat `Vec<f32>` + shape with no value type
// able to hold a variable-length sequence or an optional -- see brief item
// a3-21. `SequenceConstruct`, `OptionalHasElement` and the rest of that
// family are therefore explicitly out of scope for this lane; what is in
// scope is making sure a model that references one fails cleanly (a typed
// `OnnxError`) rather than panicking.
//
// The actual error variant, verified by reading `src/session/run/mod.rs`
// rather than assumed: `OpKind::parse` has no arm for either name, so both
// fall through to `OpKind::Unknown(name)` -- but the run loop's gate is
// deliberately *not* "reject `OpKind::Unknown`" (a registered custom op can
// legitimately be `Unknown` to this enum; see that file's doc comment on
// `unsupported_op_error`). Instead every execution path looks the op_type up
// in the registry and raises `OnnxError::UnsupportedOp` when it is absent --
// so that, not `UnknownOp`, is what a `Sequence*`/`Optional*` node produces.

#[test]
fn sequence_construct_is_a_clean_unsupported_op_error_not_a_panic() {
    let err = try_run_op(
        OpKind::parse("SequenceConstruct"),
        &["x"],
        &["y"],
        vec![("x", Tensor::new(vec![1.0], vec![1]))],
        Attributes::default(),
    )
    .expect_err("SequenceConstruct must fail cleanly, not panic");
    assert!(
        matches!(err, oxionnx::OnnxError::UnsupportedOp(_)),
        "expected OnnxError::UnsupportedOp, got: {err:?}"
    );
}

#[test]
fn optional_has_element_is_a_clean_unsupported_op_error_not_a_panic() {
    let err = try_run_op(
        OpKind::parse("OptionalHasElement"),
        &["x"],
        &["y"],
        vec![("x", Tensor::new(vec![1.0], vec![1]))],
        Attributes::default(),
    )
    .expect_err("OptionalHasElement must fail cleanly, not panic");
    assert!(
        matches!(err, oxionnx::OnnxError::UnsupportedOp(_)),
        "expected OnnxError::UnsupportedOp, got: {err:?}"
    );
}
