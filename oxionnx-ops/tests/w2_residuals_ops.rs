//! Regression tests for the W2-residuals sweep.
//!
//! `oxionnx-ops/src/registry/misc_ops.rs` (owned this wave):
//!
//! 1. The `.max(1)` zero-dim bug in `ConstantOfShape`'s output-slot path, which
//!    used to clamp a legitimate zero-size target (e.g. shape input `[0, 3]`)
//!    up to a one-element buffer while leaving `shape == [0, 3]` -- a tensor
//!    whose `data.len()` no longer matched `shape.iter().product()`. `execute`
//!    (via `comparison::constant_of_shape`, which never had the clamp) already
//!    got this right; this file pins that `execute_into_slots` now agrees.
//! 2. Present-but-empty attribute/optional-input tensors (a malformed model,
//!    but not one the engine may panic on) on `ConstantOfShape`'s `value`
//!    attribute, `Trilu`'s optional `k` input, and `NonMaxSuppression`'s three
//!    optional scalar inputs -- all previously indexed `t.data[0]` directly.
//!
//! `oxionnx-ops/src/registry/nn_ops/parameterized.rs` (unowned this wave;
//! touched only because opset-plumbing verification turned up a genuine
//! panic -- see the task brief's "fix only if genuinely broken" clause):
//!
//! 3. `HardmaxOp::execute_into_slots`'s opset-13+ branch indexed `input.data`
//!    using `outer`/`inner` bounds `.max(1)`-clamped for the legitimate
//!    empty-product case, which also swallowed a genuine zero-size dim
//!    elsewhere in the shape -- an out-of-bounds panic on a legal (e.g.
//!    dynamic batch 0) input that `execute()` already guarded against.

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator, OperatorRegistry};
use oxionnx_core::Tensor;
use oxionnx_ops::registry::misc_ops::{ConstantOfShapeOp, NonMaxSuppressionOp, TriluOp};
use oxionnx_ops::registry::nn_ops::HardmaxOp;

fn dummy_node(op: OpKind) -> Node {
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

/// A registry bound to `opset`, for tests that need `OpContext::opset()` to
/// report something other than `DEFAULT_OPSET`.
fn registry_at(opset: i64) -> OperatorRegistry {
    let registry = OperatorRegistry::new();
    registry.set_model_opset(opset);
    registry
}

fn ctx_with_registry<'a>(
    node: &'a Node,
    inputs: Vec<Option<&'a Tensor>>,
    registry: &'a OperatorRegistry,
) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: Some(registry),
    }
}

fn node_with_axis(op: OpKind, axis: i64) -> Node {
    let mut node = dummy_node(op);
    node.attrs.ints.insert("axis".to_string(), axis);
    node
}

// ── ConstantOfShape: the `.max(1)` zero-dim regression ─────────────────────

/// A `shape` input containing an explicit zero dim (`[0, 3]`) must produce a
/// genuinely empty tensor -- shape `[0, 3]`, zero elements -- through *both*
/// dispatch paths, with the same `data.len() == shape.product()` invariant
/// `execute` already satisfied.
#[test]
fn constant_of_shape_zero_dim_execute_and_slots_agree() {
    let node = dummy_node(OpKind::ConstantOfShape);
    let shape_input = Tensor::new(vec![0.0, 3.0], vec![2]);
    let ctx = make_ctx(&node, vec![Some(&shape_input)]);

    let direct = ConstantOfShapeOp
        .execute(&ctx)
        .expect("execute failed")
        .into_iter()
        .next()
        .expect("one output");
    assert_eq!(direct.shape, vec![0, 3]);
    assert!(
        direct.data.is_empty(),
        "zero-size shape must produce zero elements"
    );

    // Pre-populate the slot with stale nonzero-length data, mimicking a pooled
    // buffer reused from a prior (nonempty) call.
    let mut slots = vec![Tensor::new(vec![9.0; 3], vec![3])];
    ConstantOfShapeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_eq!(
        slots[0].shape,
        vec![0, 3],
        "slot path must match the direct path's shape"
    );
    assert!(
        slots[0].data.is_empty(),
        "slot path's data.len() must become 0, not stay clamped at a stale nonzero length"
    );
    assert_eq!(
        slots[0].data.len(),
        slots[0].shape.iter().product::<usize>(),
        "data.len() must equal shape.product() -- the invariant `.max(1)` broke"
    );
}

/// The scalar case (`shape` input is an empty 1-D tensor, i.e. rank-0 output)
/// is the case the removed `.max(1)` was presumably guarding: confirm the
/// empty-product identity (`[].iter().product() == 1`) already handles it
/// correctly on its own, with no clamp needed.
#[test]
fn constant_of_shape_scalar_shape_execute_and_slots_agree() {
    let node = dummy_node(OpKind::ConstantOfShape);
    let shape_input = Tensor::new(Vec::new(), vec![0]); // empty 1-D "shape" tensor => scalar output
    let ctx = make_ctx(&node, vec![Some(&shape_input)]);

    let direct = ConstantOfShapeOp
        .execute(&ctx)
        .expect("execute failed")
        .into_iter()
        .next()
        .expect("one output");
    assert_eq!(direct.shape, Vec::<usize>::new());
    assert_eq!(
        direct.data.len(),
        1,
        "scalar output has exactly one element"
    );

    let mut slots = vec![Tensor::new(vec![0.0], vec![1])];
    ConstantOfShapeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots failed");
    assert_eq!(slots[0].shape, Vec::<usize>::new());
    assert_eq!(slots[0].data.len(), 1);
    assert_eq!(slots[0].data, direct.data);
}

/// A present-but-empty `value` attribute tensor (malformed model) must fall
/// back to the 0.0 default instead of indexing an empty `data` slice.
#[test]
fn constant_of_shape_empty_value_attr_does_not_panic() {
    let mut node = dummy_node(OpKind::ConstantOfShape);
    node.attrs
        .tensors
        .insert("value".to_string(), Tensor::new(Vec::new(), vec![0]));
    let shape_input = Tensor::new(vec![2.0], vec![1]);
    let ctx = make_ctx(&node, vec![Some(&shape_input)]);

    let direct = ConstantOfShapeOp
        .execute(&ctx)
        .expect("execute must not panic on an empty value tensor")
        .into_iter()
        .next()
        .expect("one output");
    assert_eq!(direct.data, vec![0.0, 0.0]);

    let mut slots = vec![Tensor::new(vec![9.0, 9.0], vec![2])];
    ConstantOfShapeOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots must not panic on an empty value tensor");
    assert_eq!(slots[0].data, vec![0.0, 0.0]);
}

// ── Trilu: present-but-empty optional `k` input ─────────────────────────────

/// A present-but-empty optional `k` input (malformed model) must fall back to
/// the `k = 0` default instead of indexing an empty `data` slice.
#[test]
fn trilu_empty_k_input_does_not_panic() {
    let node = dummy_node(OpKind::Trilu);
    let x = Tensor::new((1..=9).map(|v| v as f32).collect(), vec![3, 3]);
    let empty_k = Tensor::new(Vec::new(), vec![0]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&empty_k)]);

    let out = TriluOp
        .execute(&ctx)
        .expect("execute must not panic on an empty k input")
        .into_iter()
        .next()
        .expect("one output");
    // k defaults to 0, upper defaults to true: standard upper-triangular.
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
}

// ── NonMaxSuppression: present-but-empty optional scalar inputs ────────────

/// Present-but-empty optional scalar inputs (`max_output_boxes_per_class`,
/// `iou_threshold`, `score_threshold` -- a malformed model) must fall back to
/// their spec defaults instead of indexing an empty `data` slice.
/// `max_output_boxes_per_class`'s spec default is 0, which selects nothing --
/// distinguishing "fell back to the default and ran" from "silently kept
/// stale/garbage data" still requires the call to complete without panicking.
#[test]
fn nms_empty_optional_inputs_do_not_panic() {
    let node = dummy_node(OpKind::NonMaxSuppression);
    let boxes = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![1, 1, 4]);
    let scores = Tensor::new(vec![0.9], vec![1, 1, 1]);
    let empty = Tensor::new(Vec::new(), vec![0]);
    let ctx = make_ctx(
        &node,
        vec![
            Some(&boxes),
            Some(&scores),
            Some(&empty), // max_output_boxes_per_class
            Some(&empty), // iou_threshold
            Some(&empty), // score_threshold
        ],
    );

    let out = NonMaxSuppressionOp
        .execute(&ctx)
        .expect("execute must not panic on empty optional scalar inputs")
        .into_iter()
        .next()
        .expect("one output");
    assert_eq!(out.shape, vec![0, 3]);
    assert!(
        out.data.is_empty(),
        "max_output_boxes_per_class default of 0 selects nothing"
    );
}

// ── Hardmax: zero-size dim outside the reduction axis (parameterized.rs) ───

/// `HardmaxOp::execute_into_slots`'s opset-13+ (non-coerce) branch computed
/// its `outer`/`inner` loop bounds with a `.max(1)` clamp needed for the
/// legitimate empty-product case (`ax` flush against a shape boundary, or a
/// rank-1 tensor) -- but the same clamp could not tell that apart from a
/// genuine zero-size dim *elsewhere* in the shape. `[0,3,4]` puts the zero in
/// `outer`'s product (`shape[..ax]`); `[2,3,0]` puts it in `inner`'s
/// (`shape[ax+1..]`); both with `ax=1`, so in neither case is the zero the
/// axis being reduced (that case -- e.g. `[2,0,4]` -- was already covered by
/// `opset_softmax_family.rs::empty_tensor_is_handled_in_both_regimes`, and
/// passed, because `axis_len == 0` alone already short-circuits the write via
/// the existing `if axis_len > 0` guard without ever indexing `input.data`).
/// Clamping `outer`/`inner` away from their true value of 0 sent the loop
/// indexing into `input.data`, which is correctly zero-length for a
/// zero-numel tensor: an out-of-bounds panic on a legal (e.g. dynamic batch
/// 0) input, not a malformed one.
///
/// Both opsets are covered deliberately: opset 11 selects the coerce branch,
/// which was never clamped and never panicked -- included to document that
/// only the non-coerce branch needed the fix, and that both branches agree
/// with `execute()` on shape *and* data, not merely "didn't panic".
#[test]
fn hardmax_slot_path_zero_size_dim_outside_axis_matches_execute() {
    for opset in [11, 21] {
        let registry = registry_at(opset);
        for shape in [vec![0usize, 3, 4], vec![2, 3, 0]] {
            let node = node_with_axis(OpKind::Hardmax, 1);
            let x = Tensor::new(Vec::new(), shape.clone());

            let direct = HardmaxOp
                .execute(&ctx_with_registry(&node, vec![Some(&x)], &registry))
                .unwrap_or_else(|e| panic!("{shape:?} opset {opset}: execute Err({e:?})"))
                .into_iter()
                .next()
                .expect("one output");

            // Pre-populate the slot with stale nonempty data, mimicking a
            // pooled buffer reused from a prior (nonempty) call.
            let mut slots = vec![Tensor::new(vec![9.0; 5], vec![5])];
            HardmaxOp
                .execute_into_slots(
                    &ctx_with_registry(&node, vec![Some(&x)], &registry),
                    &mut slots,
                )
                .unwrap_or_else(|e| {
                    panic!("{shape:?} opset {opset}: execute_into_slots Err({e:?})")
                });

            assert_eq!(
                slots[0].shape, direct.shape,
                "{shape:?} opset {opset}: slot path shape must match execute()"
            );
            assert_eq!(
                slots[0].shape, shape,
                "{shape:?} opset {opset}: both paths must preserve the full input shape"
            );
            assert!(
                direct.data.is_empty(),
                "{shape:?} opset {opset}: execute() must produce zero-numel output"
            );
            assert!(
                slots[0].data.is_empty(),
                "{shape:?} opset {opset}: slot path must produce zero-numel output, \
                 not stay clamped at a stale nonzero length"
            );
        }
    }
}
