//! Opset-boundary semantics for the `Softmax` / `LogSoftmax` / `Hardmax` family
//! (stitch-wave item S5, from Wave-1 findings a1-10 / a5-6 / a11-9).
//!
//! ONNX redefined all three ops at **opset 13**:
//!
//! * opsets 1–12: `axis` defaults to **1** and names the point at which the input
//!   is coerced to 2D — `[prod(shape[..axis]), prod(shape[axis..])]` — so the
//!   reduction runs across the whole flattened trailing block.
//! * opsets 13+: `axis` defaults to **-1** and names the one axis reduced, per
//!   slice, with no coercion.
//!
//! The regimes coincide for rank ≤ 2 and for a trailing axis, so every test here
//! uses a **rank-3 `[2,3,4]` tensor with the middle axis**, where they genuinely
//! disagree. Reference values come from NumPy (float64 accumulation), not from
//! the implementation under test:
//!
//! ```text
//! x = np.array([...]).reshape(2,3,4)
//! pre13  = softmax(x.reshape(2, 12), axis=1).reshape(2,3,4)   # coerce at axis 1
//! post13 = softmax(x, axis=1)                                 # reduce axis 1
//! ```

use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::{OpContext, Operator, OperatorRegistry, DEFAULT_OPSET},
    Tensor,
};
use oxionnx_ops::registry::nn_ops::{HardmaxOp, LogSoftmaxOp, SoftmaxOp};

// ── Test infrastructure ─────────────────────────────────────────────────────

/// The shared rank-3 input, row-major `[2,3,4]`.
const X: [f32; 24] = [
    0.1, -0.2, 0.3, 0.4, //
    1.0, 0.5, -0.5, 0.0, //
    -1.0, 2.0, 0.25, -0.75, //
    0.7, 0.7, 0.7, 0.7, //
    -2.0, 1.5, 0.0, 0.5, //
    3.0, -3.0, 1.0, -1.0,
];
const SHAPE: [usize; 3] = [2, 3, 4];

fn input() -> Tensor {
    Tensor::new(X.to_vec(), SHAPE.to_vec())
}

/// A registry bound to `opset`. Empty of operators: the ops under test are
/// invoked directly, and `OpContext::opset()` only reads the bound version.
fn registry_at(opset: i64) -> OperatorRegistry {
    let registry = OperatorRegistry::new();
    registry.set_model_opset(opset);
    registry
}

fn node_with(op: OpKind, axis: Option<i64>) -> Node {
    let mut attrs = Attributes::default();
    if let Some(value) = axis {
        attrs.ints.insert("axis".into(), value);
    }
    Node {
        name: "test".into(),
        op,
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        attrs,
    }
}

fn ctx<'a>(node: &'a Node, x: &'a Tensor, registry: Option<&'a OperatorRegistry>) -> OpContext<'a> {
    OpContext {
        node,
        inputs: vec![Some(x)],
        outer_scope: None,
        weights: None,
        registry,
    }
}

/// Run `op` through **both** dispatch paths and assert they agree.
///
/// `execute_into_slots` — not `execute` — is what a real session run reaches
/// whenever the output shape is known (see `Session::dispatch_node`), and all
/// three ops here declare `supports_output_slots()`. `Hardmax` in particular
/// carries a hand-rolled slot kernel independent of `nn::hardmax`, so opset
/// dispatch has to be proven on both.
fn run_both(op: &dyn Operator, node: &Node, x: &Tensor, opset: Option<i64>) -> Tensor {
    let registry = opset.map(registry_at);
    let direct = op
        .execute(&ctx(node, x, registry.as_ref()))
        .expect("execute must succeed");
    assert_eq!(direct.len(), 1, "family ops produce exactly one output");

    let mut slots = vec![Tensor::new(Vec::new(), vec![0])];
    op.execute_into_slots(&ctx(node, x, registry.as_ref()), &mut slots)
        .expect("execute_into_slots must succeed");
    assert_eq!(slots[0].shape, direct[0].shape, "slot vs execute shape");
    assert_eq!(slots[0].data, direct[0].data, "slot vs execute data");

    direct.into_iter().next().unwrap_or_else(|| unreachable!())
}

fn assert_close(got: &Tensor, want: &[f32], label: &str) {
    assert_eq!(got.shape, SHAPE.to_vec(), "{label}: shape is preserved");
    assert_eq!(got.data.len(), want.len(), "{label}: element count");
    for (i, (&g, &w)) in got.data.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-6,
            "{label}[{i}]: got {g}, expected {w} (delta {})",
            (g - w).abs()
        );
    }
}

// ── NumPy reference values ──────────────────────────────────────────────────

/// `softmax(x.reshape(2,12), axis=1).reshape(2,3,4)` — the opset ≤ 12 contract at
/// `axis=1`, which is also its default.
const PRE13_SOFTMAX_AXIS1: [f32; 24] = [
    0.054_569_75,
    0.040_426_264,
    0.066_651_64,
    0.073_661_456,
    0.134_219_92,
    0.081_408_5,
    0.029_948_513,
    0.049_376_75,
    0.018_164_692,
    0.364_847_58,
    0.063_401,
    0.023_323_925,
    0.052_247_94,
    0.052_247_94,
    0.052_247_94,
    0.052_247_94,
    0.003_511_349_6,
    0.116_279_93,
    0.025_945_56,
    0.042_776_995,
    0.521_130_5,
    0.001_291_753_3,
    0.070_527_34,
    0.009_544_838,
];

/// `softmax(x, axis=1)` — the opset ≥ 13 contract at the same explicit `axis=1`.
const POST13_SOFTMAX_AXIS1: [f32; 24] = [
    0.263_680_1,
    0.083_064_99,
    0.416_569_75,
    0.503_282_2,
    0.648_548_4,
    0.167_272_35,
    0.187_176_85,
    0.337_360_15,
    0.087_771_48,
    0.749_662_66,
    0.396_253_4,
    0.159_357_65,
    0.090_568_32,
    0.307_667_27,
    0.351_315_52,
    0.499_646_7,
    0.006_086_691,
    0.684_726_1,
    0.174_458_13,
    0.409_076_1,
    0.903_345,
    0.007_606_62,
    0.474_226_35,
    0.091_277_22,
];

/// `softmax(x, axis=-1)` — the opset ≥ 13 **default**.
const POST13_SOFTMAX_DEFAULT: [f32; 24] = [
    0.231_906_66,
    0.171_800_68,
    0.283_251_43,
    0.313_041_24,
    0.455_054_23,
    0.276_004_35,
    0.101_536_32,
    0.167_405_1,
    0.038_669_902,
    0.776_705_74,
    0.134_971_22,
    0.049_653_137,
    0.25,
    0.25,
    0.25,
    0.25,
    0.018_626_483,
    0.616_824_4,
    0.137_632_12,
    0.226_917_01,
    0.864_954_9,
    0.002_144_008_8,
    0.117_058_91,
    0.015_842_201,
];

/// `log_softmax(x.reshape(2,12), axis=1).reshape(2,3,4)` — opset ≤ 12, `axis=1`.
const PRE13_LOG_SOFTMAX_AXIS1: [f32; 24] = [
    -2.908_275_6,
    -3.208_275_6,
    -2.708_275_6,
    -2.608_275_6,
    -2.008_275_6,
    -2.508_275_6,
    -3.508_275_6,
    -3.008_275_6,
    -4.008_275_6,
    -1.008_275_6,
    -2.758_275_6,
    -3.758_275_6,
    -2.951_754_8,
    -2.951_754_8,
    -2.951_754_8,
    -2.951_754_8,
    -5.651_755,
    -2.151_754_8,
    -3.651_754_8,
    -3.151_754_8,
    -0.651_754_8,
    -6.651_755,
    -2.651_754_8,
    -4.651_755,
];

/// `log_softmax(x, axis=1)` — opset ≥ 13, same explicit axis.
const POST13_LOG_SOFTMAX_AXIS1: [f32; 24] = [
    -1.333_018_6,
    -2.488_132,
    -0.875_701_4,
    -0.686_604_2,
    -0.433_018_63,
    -1.788_132,
    -1.675_701_4,
    -1.086_604_2,
    -2.433_018_6,
    -0.288_131_96,
    -0.925_701_4,
    -1.836_604_2,
    -2.401_651,
    -1.178_736_4,
    -1.046_070_5,
    -0.693_854_1,
    -5.101_650_8,
    -0.378_736_36,
    -1.746_070_5,
    -0.893_854_1,
    -0.101_650_75,
    -4.878_736_4,
    -0.746_070_5,
    -2.393_854_1,
];

// ═══════════════════════════════════════════════════════════════════════════
// Subgraph propagation
// ═══════════════════════════════════════════════════════════════════════════

/// A model declares one opset for its whole graph, control-flow bodies included.
///
/// `If`/`Loop`/`Scan` hand `ctx.registry` straight to `execute_subgraph`, which
/// rebuilds an `OpContext` around that same registry — so binding the opset to
/// the registry propagates into subgraph bodies with no per-op plumbing. This
/// test is what keeps that free ride honest: a future refactor that built a
/// fresh registry for subgraph execution would silently reset the body to
/// `DEFAULT_OPSET`, and only this assertion would notice.
#[test]
fn subgraph_bodies_inherit_the_outer_opset() {
    use oxionnx_core::graph::Graph;
    use oxionnx_ops::control_flow::IfOp;
    use std::collections::HashMap;

    let mut body_attrs = Attributes::default();
    body_attrs.ints.insert("axis".into(), 1);
    let branch = Graph {
        nodes: vec![Node {
            name: "inner_softmax".into(),
            op: OpKind::Softmax,
            inputs: vec!["captured_x".into()],
            outputs: vec!["inner_y".into()],
            attrs: body_attrs,
        }],
        input_names: vec![],
        output_names: vec!["inner_y".into()],
        ..Default::default()
    };

    let mut attrs = Attributes::default();
    attrs.graphs.insert("then_branch".into(), branch.clone());
    attrs.graphs.insert("else_branch".into(), branch);
    let if_node = Node {
        name: "if_node".into(),
        op: OpKind::If,
        inputs: vec!["cond".into()],
        outputs: vec!["y".into()],
        attrs,
    };

    // The body reads its input by implicit outer-scope capture, as ONNX subgraphs do.
    let mut outer_scope: HashMap<String, Tensor> = HashMap::new();
    outer_scope.insert("captured_x".into(), input());
    let cond = Tensor::scalar(1.0);

    for (opset, want, label) in [
        (11_i64, &PRE13_SOFTMAX_AXIS1, "opset 11 body"),
        (13, &POST13_SOFTMAX_AXIS1, "opset 13 body"),
    ] {
        // The real registry: the subgraph looks its own operators up in it.
        let registry = oxionnx_ops::default_registry();
        registry.set_model_opset(opset);
        let ctx = OpContext {
            node: &if_node,
            inputs: vec![Some(&cond)],
            outer_scope: Some(&outer_scope),
            weights: None,
            registry: Some(&registry),
        };
        let out = IfOp.execute(&ctx).expect("If must execute its branch");
        assert_eq!(out.len(), 1, "the branch produces one output");
        assert_close(&out[0], want, label);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax
// ═══════════════════════════════════════════════════════════════════════════

/// [a1-10 / a5-6] Explicit `axis=1` means two different things either side of
/// opset 13. This is the discriminator: same node, same tensor, same axis
/// attribute — the model's declared opset alone decides the answer.
#[test]
fn softmax_explicit_axis1_follows_the_declared_opset() {
    let x = input();
    let node = node_with(OpKind::Softmax, Some(1));

    let pre = run_both(&SoftmaxOp, &node, &x, Some(11));
    assert_close(
        &pre,
        &PRE13_SOFTMAX_AXIS1,
        "opset 11, axis=1 (coerced to 2D)",
    );

    let post = run_both(&SoftmaxOp, &node, &x, Some(13));
    assert_close(&post, &POST13_SOFTMAX_AXIS1, "opset 13, axis=1 (per-slice)");

    // Sanity: the two regimes really are distinguishable on this input.
    assert!(
        (pre.data[0] - post.data[0]).abs() > 0.1,
        "the regimes must differ on a rank-3 middle axis"
    );
}

/// The *default* axis flips 1 → -1 at the same boundary. A version dispatch that
/// branched the coercion but kept one default would pass the test above and fail
/// this one.
#[test]
fn softmax_default_axis_flips_at_opset13() {
    let x = input();
    let node = node_with(OpKind::Softmax, None);

    let pre = run_both(&SoftmaxOp, &node, &x, Some(11));
    assert_close(&pre, &PRE13_SOFTMAX_AXIS1, "opset 11 default axis = 1");

    let post = run_both(&SoftmaxOp, &node, &x, Some(13));
    assert_close(&post, &POST13_SOFTMAX_DEFAULT, "opset 13 default axis = -1");
}

/// Every opset below 13 shares the legacy contract, and every opset at or above
/// it shares the current one — the dispatch is a boundary, not a lookup table.
#[test]
fn softmax_boundary_is_exactly_13() {
    let x = input();
    let node = node_with(OpKind::Softmax, Some(1));

    for opset in [1, 7, 11, 12] {
        let got = run_both(&SoftmaxOp, &node, &x, Some(opset));
        assert_close(&got, &PRE13_SOFTMAX_AXIS1, "legacy contract");
    }
    for opset in [13, 18, 21, 23] {
        let got = run_both(&SoftmaxOp, &node, &x, Some(opset));
        assert_close(&got, &POST13_SOFTMAX_AXIS1, "current contract");
    }
}

/// The coercion is a no-op when `axis` is already the last one, so both regimes
/// must agree there — this is why the 2D classification tail of most models
/// never exposed the bug.
#[test]
fn softmax_regimes_agree_on_the_trailing_axis() {
    let x = input();
    let pre = run_both(
        &SoftmaxOp,
        &node_with(OpKind::Softmax, Some(2)),
        &x,
        Some(11),
    );
    assert_close(&pre, &POST13_SOFTMAX_DEFAULT, "opset 11, axis=2");

    let post = run_both(
        &SoftmaxOp,
        &node_with(OpKind::Softmax, Some(-1)),
        &x,
        Some(21),
    );
    assert_close(&post, &POST13_SOFTMAX_DEFAULT, "opset 21, axis=-1");
}

/// A context with no registry — a unit test, or the constant folder — reports
/// `DEFAULT_OPSET` and therefore keeps the current semantics. This is the
/// backward-compatibility guarantee for the ~70 `OpContext` literals in the
/// workspace that predate opset plumbing.
#[test]
fn absent_registry_keeps_current_semantics() {
    // `DEFAULT_OPSET` sits above the boundary, so "unbound" and "bound to the
    // default" must both land on the current contract — asserted below against
    // the NumPy reference rather than against the constant itself.
    let x = input();
    let node = node_with(OpKind::Softmax, None);

    let unbound = run_both(&SoftmaxOp, &node, &x, None);
    assert_close(&unbound, &POST13_SOFTMAX_DEFAULT, "no registry");

    let bound = run_both(&SoftmaxOp, &node, &x, Some(DEFAULT_OPSET));
    assert_eq!(unbound.data, bound.data, "no registry == DEFAULT_OPSET");
}

// ═══════════════════════════════════════════════════════════════════════════
// LogSoftmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_softmax_explicit_axis1_follows_the_declared_opset() {
    let x = input();
    let node = node_with(OpKind::LogSoftmax, Some(1));

    let pre = run_both(&LogSoftmaxOp, &node, &x, Some(11));
    assert_close(&pre, &PRE13_LOG_SOFTMAX_AXIS1, "opset 11, axis=1");

    let post = run_both(&LogSoftmaxOp, &node, &x, Some(13));
    assert_close(&post, &POST13_LOG_SOFTMAX_AXIS1, "opset 13, axis=1");
}

#[test]
fn log_softmax_default_axis_flips_at_opset13() {
    let x = input();
    let node = node_with(OpKind::LogSoftmax, None);

    let pre = run_both(&LogSoftmaxOp, &node, &x, Some(11));
    assert_close(&pre, &PRE13_LOG_SOFTMAX_AXIS1, "opset 11 default axis = 1");

    // opset 13+ default is -1: log of the per-row softmax over the last axis.
    let post = run_both(&LogSoftmaxOp, &node, &x, Some(13));
    let want: Vec<f32> = POST13_SOFTMAX_DEFAULT.iter().map(|v| v.ln()).collect();
    assert_close(&post, &want, "opset 13 default axis = -1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Hardmax
// ═══════════════════════════════════════════════════════════════════════════

/// Hardmax makes the contract change structural rather than numeric: the count
/// of ones in the output *is* the number of independent reductions performed.
///
/// * opset 11, `axis=1`: coerced to `[2,12]` → **2** winners.
/// * opset 13, `axis=1`: one winner per `(batch, ·, k)` slice → 2·4 = **8**.
/// * opset 13, default `axis=-1`: one per `(batch, j, ·)` slice → 2·3 = **6**.
#[test]
fn hardmax_reduction_count_follows_the_declared_opset() {
    let x = input();
    let axis1 = node_with(OpKind::Hardmax, Some(1));

    let pre = run_both(&HardmaxOp, &axis1, &x, Some(11));
    assert_eq!(
        pre.data.iter().sum::<f32>(),
        2.0,
        "opset 11: one per coerced row"
    );
    // The winners: flat index 9 (value 2.0, row 0) and 20 (value 3.0, row 1).
    let mut pre_ones: Vec<usize> = Vec::new();
    for (i, &v) in pre.data.iter().enumerate() {
        if v == 1.0 {
            pre_ones.push(i);
        }
    }
    assert_eq!(pre_ones, vec![9, 20], "argmax of each coerced row");

    let post = run_both(&HardmaxOp, &axis1, &x, Some(13));
    assert_eq!(
        post.data.iter().sum::<f32>(),
        8.0,
        "opset 13: one per axis-1 slice"
    );

    let default_post = run_both(&HardmaxOp, &node_with(OpKind::Hardmax, None), &x, Some(13));
    assert_eq!(
        default_post.data.iter().sum::<f32>(),
        6.0,
        "opset 13 default axis=-1: one per row of 4"
    );

    let default_pre = run_both(&HardmaxOp, &node_with(OpKind::Hardmax, None), &x, Some(11));
    assert_eq!(default_pre.data, pre.data, "opset 11 default axis = 1");
}

/// Ties resolve to the lowest index in both regimes (the kernels compare with a
/// strict `>`), so the all-equal row of the input has exactly one winner.
#[test]
fn hardmax_ties_pick_the_first_index() {
    // [1,2,3]: row 1 is all 0.5, so its winner must be index 0 of that row.
    let x = Tensor::new(vec![0.1, 0.2, 0.3, 0.5, 0.5, 0.5], vec![2, 3]);
    let node = node_with(OpKind::Hardmax, Some(1));

    for opset in [11, 13] {
        let registry = registry_at(opset);
        let out = HardmaxOp
            .execute(&ctx(&node, &x, Some(&registry)))
            .expect("hardmax must succeed");
        assert_eq!(
            out[0].data,
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "opset {opset}: rank-2 is regime-independent, ties pick index 0"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Malformed input
// ═══════════════════════════════════════════════════════════════════════════

/// The pre-opset-13 default `axis=1` is out of the spec's `[-r, r-1]` range for a
/// rank-1 tensor. That must be a typed error naming the axis, never a panic and
/// never a silently wrapped index.
#[test]
fn pre13_default_axis_on_rank1_is_a_typed_error() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let registry = registry_at(11);
    let node = node_with(OpKind::Softmax, None);

    let err = SoftmaxOp
        .execute(&ctx(&node, &x, Some(&registry)))
        .expect_err("rank-1 with the legacy default axis is out of range");
    let text = format!("{err}");
    assert!(
        text.contains("axis 1") && text.contains("1D"),
        "error must name the offending axis and rank, got: {text}"
    );

    // The slot path must reject it the same way rather than writing garbage.
    let mut slots = vec![Tensor::new(Vec::new(), vec![0])];
    SoftmaxOp
        .execute_into_slots(&ctx(&node, &x, Some(&registry)), &mut slots)
        .expect_err("slot path must reject it too");
}

/// Out-of-range axes are rejected in both regimes and by all three ops, with no
/// panic and no output written.
#[test]
fn out_of_range_axis_is_rejected_in_both_regimes() {
    let x = input();
    let ops: [(&dyn Operator, OpKind); 3] = [
        (&SoftmaxOp, OpKind::Softmax),
        (&LogSoftmaxOp, OpKind::LogSoftmax),
        (&HardmaxOp, OpKind::Hardmax),
    ];
    for (op, kind) in ops {
        for opset in [11, 21] {
            let registry = registry_at(opset);
            for axis in [3_i64, -4, i64::MIN, i64::MAX] {
                let node = node_with(kind.clone(), Some(axis));
                op.execute(&ctx(&node, &x, Some(&registry)))
                    .expect_err("axis outside [-3, 2] must be an error");
                let mut slots = vec![Tensor::new(Vec::new(), vec![0])];
                op.execute_into_slots(&ctx(&node, &x, Some(&registry)), &mut slots)
                    .expect_err("slot path must reject it too");
            }
        }
    }
}

/// A zero-sized tensor coerces to a zero-length row in the legacy regime; the
/// kernels must produce an empty output instead of indexing out of bounds.
#[test]
fn empty_tensor_is_handled_in_both_regimes() {
    let x = Tensor::new(Vec::new(), vec![2, 0, 4]);
    for opset in [11, 21] {
        let registry = registry_at(opset);
        for (op, kind) in [
            (&SoftmaxOp as &dyn Operator, OpKind::Softmax),
            (&HardmaxOp as &dyn Operator, OpKind::Hardmax),
        ] {
            let node = node_with(kind, Some(1));
            let out = op
                .execute(&ctx(&node, &x, Some(&registry)))
                .expect("empty input must not panic");
            assert!(out[0].data.is_empty(), "opset {opset}: empty in, empty out");

            let mut slots = vec![Tensor::new(Vec::new(), vec![0])];
            op.execute_into_slots(&ctx(&node, &x, Some(&registry)), &mut slots)
                .expect("empty input must not panic on the slot path");
            assert!(slots[0].data.is_empty());
        }
    }
}
