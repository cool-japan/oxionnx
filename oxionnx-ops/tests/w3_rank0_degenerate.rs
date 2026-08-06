//! Wave-3 `T3-rank0-migration`, the cases the rank-0 migration exposed or fixed
//! that are *not* about a well-formed scalar: zero-size dimensions, malformed
//! tensor buffers, and the ops whose scalar outputs moved from `[1]` to `[]`
//! but which `oxionnx-ops/tests/w2_rank0.rs` does not reach.
//!
//! Three separate classes live here, and they are separate on purpose:
//!
//! 1. **`[a0-22]` zero-size dimensions in `argminmax.rs`.** The `.max(1)` clamp
//!    on the `outer`/`inner` products turned a legitimate `0` into a `1` and
//!    then indexed an empty buffer. Every test in this section panicked with
//!    index-out-of-bounds before the fix — they are fail-without/pass-with
//!    proofs, not shape assertions.
//! 2. **`[a6-*]` the `Tensor` data/shape invariant gap in `transpose`.** The
//!    odometer's loop bound came from `x.numel()` (which this crate defines as
//!    `data.len()`) rather than the shape product, so an over-long buffer drove
//!    extra laps.
//! 3. **Scalar-output ops** whose `execute` and `execute_into_slots` paths must
//!    agree on rank 0.
//!
//! Reference values are NumPy's, computed with `python3`:
//!
//! ```text
//! np.argmax(np.zeros((0,3)), axis=1)               -> shape (0,), []
//! np.argmax(np.zeros((0,3)), axis=1, keepdims=True) -> shape (0,1), []
//! np.argmax(np.zeros((0,3)), axis=0)               -> ValueError: attempt to get argmax of an empty sequence
//! np.argmax(np.zeros((3,0)), axis=1)               -> ValueError (same)
//! np.argmax(np.zeros((0,0)), axis=1)               -> ValueError (same)
//! np.cumsum(np.zeros((0,3)), axis=1).shape         -> (0, 3)
//! np.cumsum(np.zeros((3,0)), axis=1).shape         -> (3, 0)
//! ```

use oxionnx_core::graph::{Attributes, Graph, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::{OnnxError, Tensor};
use oxionnx_ops::control_flow::LoopOp;
use oxionnx_ops::registry::misc_ops::{ConstantOp, ShapeOp};
use oxionnx_ops::{math, shape};
use std::collections::HashMap;

// ── Test infrastructure ──────────────────────────────────────────────────────

fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn dummy_node(op_type: &str) -> Node {
    Node {
        name: "test".into(),
        op: OpKind::parse(op_type),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn plain_node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.into(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

/// The empty shape, spelled once so the assertions below stay readable.
fn rank0_shape() -> Vec<usize> {
    Vec::new()
}

/// A tensor with a zero-size dimension: `data` is genuinely empty, and the shape
/// says so. This is a *well-formed* tensor, not a malformed one — the empty
/// product rule means `shape.iter().product() == 0 == data.len()`.
fn zero_size(shape: &[usize]) -> Tensor {
    Tensor::new(Vec::new(), shape.to_vec())
}

// ═══════════════════════════════════════════════════════════════════════════
// 1 — [a0-22] zero-size dimensions in ArgMax / ArgMin / CumSum
// ═══════════════════════════════════════════════════════════════════════════

/// A zero-size dimension *outside* the reduction axis yields an empty result,
/// not a panic.
///
/// `arg_max([0,3], axis=1)` reduces the length-3 axis for each of the zero rows,
/// i.e. zero times, giving shape `[0]`. Before the fix the `.max(1)` clamp on
/// `outer = product(shape[..axis])` turned that `0` into a `1`, so the loop ran
/// once and read `x.data[0]` out of an empty buffer — an index-out-of-bounds
/// panic for both `keepdims` settings.
///
/// NumPy: `np.argmax(np.zeros((0,3)), axis=1)` has shape `(0,)`.
#[test]
fn arg_reduce_with_a_zero_size_outer_dim_returns_empty_instead_of_panicking() {
    let x = zero_size(&[0, 3]);

    let out = math::arg_max(&x, 1, false, false).expect("must not panic or error");
    assert_eq!(out.shape, vec![0], "zero rows to report an index for");
    assert_eq!(out.data, Vec::<f32>::new());

    let out = math::arg_max(&x, 1, true, false).expect("keepdims must behave the same");
    assert_eq!(out.shape, vec![0, 1]);
    assert_eq!(out.data, Vec::<f32>::new());

    // ArgMin shares the kernel, so it shares the fix.
    let out = math::arg_min(&x, 1, false, false).expect("arg_min too");
    assert_eq!(out.shape, vec![0]);
    assert_eq!(out.data, Vec::<f32>::new());
}

/// A zero-size dimension in the *trailing* slice is the same class, reached
/// through `inner = product(shape[axis+1..])` instead.
///
/// `arg_max([2,3,0], axis=1)` has `inner == 0`, so there is nothing to reduce
/// and the output is shape `[2,0]`.
#[test]
fn arg_reduce_with_a_zero_size_inner_dim_returns_empty_instead_of_panicking() {
    let x = zero_size(&[2, 3, 0]);

    let out = math::arg_max(&x, 1, false, false).expect("must not panic or error");
    assert_eq!(out.shape, vec![2, 0]);
    assert_eq!(out.data, Vec::<f32>::new());

    let out = math::arg_max(&x, 1, true, false).expect("keepdims must behave the same");
    assert_eq!(out.shape, vec![2, 1, 0]);
    assert_eq!(out.data, Vec::<f32>::new());
}

/// A zero-length **reduction axis** is a different case, and unclamping the
/// products does not rescue it: `outer`/`inner` can both be legitimately
/// non-zero while `axis_len == 0`, and the seed read then indexes an empty
/// buffer regardless.
///
/// There is no answer to report — the index of the extremum of an empty sequence
/// does not exist — so this is a typed error rather than a fabricated `0`.
/// NumPy raises `ValueError: attempt to get argmax of an empty sequence` for
/// every one of these shapes, including `(0,0)` where the output would itself be
/// empty.
#[test]
fn arg_reduce_over_a_zero_length_axis_is_an_error_not_a_panic_and_not_a_fake_index() {
    for (shape, axis) in [
        (vec![3usize, 0], 1i64),
        (vec![0, 3], 0),
        (vec![0, 0], 1),
        (vec![0, 0], 0),
        (vec![2, 0, 4], 1),
    ] {
        let x = zero_size(&shape);
        for keepdims in [false, true] {
            let err = math::arg_max(&x, axis, keepdims, false).expect_err(&format!(
                "arg_max over an empty axis must error: shape {shape:?} axis {axis} keepdims {keepdims}"
            ));
            assert!(
                err.contains("length 0"),
                "unexpected message for {shape:?} axis {axis}: {err}"
            );
            math::arg_min(&x, axis, keepdims, false)
                .expect_err("arg_min shares the kernel and the guard");
        }
    }
}

/// A negative axis resolves before the guard, so `[3,0]` with `axis=-1` is the
/// same empty-axis error rather than an out-of-range one.
#[test]
fn arg_reduce_empty_axis_guard_applies_after_negative_axis_normalization() {
    let err = math::arg_max(&zero_size(&[3, 0]), -1, false, false)
        .expect_err("axis -1 is axis 1 here, which has length 0");
    assert!(err.contains("length 0"), "unexpected message: {err}");
}

/// Non-degenerate inputs are unaffected by dropping the clamps: a rank-1 input
/// has an *empty* leading slice, whose product is the empty product `1` — the
/// value the clamp used to supply — so the common path is untouched.
#[test]
fn dropping_the_clamp_does_not_change_ordinary_shapes() {
    let v = Tensor::new(vec![3.0, 9.0, 4.0], vec![3]);
    let out = math::arg_max(&v, 0, false, false).expect("1-D argmax");
    assert_eq!(out.shape, rank0_shape(), "rank 1 -> rank 0");
    assert_eq!(out.data, vec![1.0]);

    // A rank-1 reduction with keepdims, and a 2-D reduction on each axis.
    let m = Tensor::new(vec![1.0, 5.0, 2.0, 8.0, 3.0, 0.0], vec![2, 3]);
    assert_eq!(
        math::arg_max(&m, 1, false, false).expect("axis 1").data,
        vec![1.0, 0.0]
    );
    assert_eq!(
        math::arg_max(&m, 0, false, false).expect("axis 0").data,
        vec![1.0, 0.0, 0.0]
    );
    assert_eq!(
        math::arg_max(&m, 0, true, false).expect("keepdims").shape,
        vec![1, 3]
    );
}

/// `CumSum` carries the identical clamp in both its allocating and its `_into`
/// form. `cumsum([0,3], axis=1)` panicked for the same reason `arg_max` did.
///
/// Unlike `arg_reduce` a zero-length *axis* needs no guard here: an empty prefix
/// sum is correctly the empty result, and the inner loop simply does not run.
/// NumPy: both `np.cumsum(np.zeros((0,3)), axis=1)` and
/// `np.cumsum(np.zeros((3,0)), axis=1)` return the input shape unchanged.
#[test]
fn cumsum_with_zero_size_dims_returns_the_input_shape_instead_of_panicking() {
    for (shape, axis) in [
        (vec![0usize, 3], 1i64),
        (vec![3, 0], 1),
        (vec![0, 3], 0),
        (vec![2, 0, 4], 1),
    ] {
        let x = zero_size(&shape);
        let out = math::cumsum(&x, axis, false, false)
            .unwrap_or_else(|e| panic!("cumsum {shape:?} axis {axis} failed: {e}"));
        assert_eq!(out.shape, shape, "CumSum preserves the input shape");
        assert_eq!(out.data, Vec::<f32>::new());

        // `exclusive` / `reverse` take different inner loops; both must be safe.
        for (exclusive, reverse) in [(true, false), (false, true), (true, true)] {
            let out = math::cumsum(&x, axis, exclusive, reverse)
                .unwrap_or_else(|e| panic!("cumsum variant failed: {e}"));
            assert_eq!(out.shape, shape);
        }
    }
}

/// Ordinary `CumSum` results are unchanged by dropping the clamp.
/// NumPy: `np.cumsum([[1,2,3],[4,5,6]], axis=1)` is `[[1,3,6],[4,9,15]]`.
#[test]
fn cumsum_ordinary_results_are_unchanged() {
    let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = math::cumsum(&m, 1, false, false).expect("cumsum");
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.data, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);

    let v = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let out = math::cumsum(&v, 0, true, false).expect("exclusive");
    assert_eq!(out.data, vec![0.0, 1.0, 3.0], "exclusive prefix sum");
}

/// The registry's slot path reaches `arg_reduce_into` / `cumsum_into`, which
/// carry their own copies of the clamp and the guard. Same shapes, same
/// outcomes — a fix applied to only the allocating twin would show up here.
#[test]
fn the_slot_path_shares_the_zero_size_behaviour() {
    let registry = oxionnx_ops::default_registry();

    let arg_max = registry.get("ArgMax").expect("ArgMax registered");
    let mut node = dummy_node("ArgMax");
    node.attrs.ints.insert("axis".into(), 1);
    node.attrs.ints.insert("keepdims".into(), 0);

    // Zero outer dim: empty result, no panic.
    let x = zero_size(&[0, 3]);
    let mut slots = vec![Tensor::new(vec![0.0], vec![1])];
    arg_max
        .execute_into_slots(&make_ctx(&node, vec![Some(&x)]), &mut slots)
        .expect("slot path must not panic or error");
    assert_eq!(slots[0].shape, vec![0]);
    // The slot must be *well-formed*, not merely correctly shaped. `arg_output_shape`
    // used to floor its length hint with `.max(1)`, leaving one stale element behind a
    // shape that says zero — and since `Tensor::numel()` is `data.len()` here, a
    // following `Size` node would then have reported 1 element for an empty tensor.
    assert_eq!(
        slots[0].data.len(),
        0,
        "a zero-size output must leave no elements in the slot"
    );
    assert_eq!(slots[0].numel(), 0, "numel must agree with the shape");

    // Zero-length reduction axis: typed error, no panic.
    let y = zero_size(&[3, 0]);
    let mut slots = vec![Tensor::new(vec![0.0], vec![1])];
    let err = arg_max
        .execute_into_slots(&make_ctx(&node, vec![Some(&y)]), &mut slots)
        .expect_err("slot path must report the empty axis too");
    assert!(format!("{err}").contains("length 0"), "unexpected: {err}");

    // CumSum's slot path, same two shapes; neither is an error for CumSum.
    let cumsum = registry.get("CumSum").expect("CumSum registered");
    let axis = Tensor::new(vec![1.0], vec![1]);
    let cs_node = dummy_node("CumSum");
    for shape in [vec![0usize, 3], vec![3, 0]] {
        let t = zero_size(&shape);
        let mut slots = vec![Tensor::new(Vec::new(), shape.clone())];
        cumsum
            .execute_into_slots(&make_ctx(&cs_node, vec![Some(&t), Some(&axis)]), &mut slots)
            .expect("CumSum slot path must not panic");
        assert_eq!(slots[0].shape, shape);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2 — transpose's loop bound: shape product, not buffer length
// ═══════════════════════════════════════════════════════════════════════════

/// `transpose`'s odometer must run `shape.iter().product() / run_len` outer
/// steps, not `x.numel() / run_len`.
///
/// Those differ because this crate's `Tensor::numel()` returns `data.len()`,
/// while `Tensor::new` only checks `data.len() == shape.product()` through a
/// `debug_assert!` — so a release build can carry a tensor whose buffer is
/// longer than its shape describes. Driving the loop from the buffer length made
/// the odometer wrap and re-walk the same valid output range with the trailing
/// "extra" data, leaving a last-write-wins result decided by the final lap.
///
/// The tensor here is built with a struct literal precisely to bypass that
/// `debug_assert`. The assertion is that the oversized buffer's tail is *ignored*
/// — the result must equal a well-formed transpose of just the valid prefix.
/// This mirrors `reduce.rs`'s `oversized_data_buffer_is_capped_at_shape_product`.
#[test]
fn transpose_is_bounded_by_the_shape_product_not_the_buffer_length() {
    // Valid prefix describes [2,3]; the buffer carries 6 extra elements.
    let valid: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut oversized = valid.clone();
    oversized.extend_from_slice(&[90.0, 91.0, 92.0, 93.0, 94.0, 95.0]);

    // A perm that reaches the general odometer path rather than the trailing-swap
    // fast path: 3-D with the *leading* two axes exchanged.
    let shape = vec![1usize, 2, 3];
    let perm = [1, 0, 2];

    let baseline =
        shape::transpose(&Tensor::new(valid, shape.clone()), &perm).expect("well-formed transpose");
    let malformed = Tensor {
        data: oversized,
        shape: shape.clone(),
    };
    let actual = shape::transpose(&malformed, &perm).expect("must not panic");

    assert_eq!(actual.shape, baseline.shape, "shape is shape-derived");
    assert_eq!(
        actual.data, baseline.data,
        "the valid prefix must transpose exactly as a well-formed tensor does; \
         the oversized tail must neither drive extra laps that overwrite it nor \
         be carried into the result"
    );
    assert!(
        !actual.data.contains(&90.0),
        "no value from the oversized tail may reach the output"
    );
    // The result is itself well-formed: `data.len()` equals the shape product,
    // so the malformed input does not propagate into a malformed output.
    assert_eq!(
        actual.data.len(),
        actual.shape.iter().product::<usize>(),
        "output must be well-formed even when the input was not"
    );
}

/// The opposite mismatch — a buffer *shorter* than the shape describes — has no
/// correct transpose: a source element the output requires simply does not
/// exist. It is reported as a typed error rather than indexing past the end of
/// `x.data` (a panic) or zero-filling part of the result (a silent wrong
/// answer).
///
/// Both the general odometer and the tiled trailing-swap fast path are covered,
/// because the guard has to sit ahead of the branch between them.
#[test]
fn transpose_with_an_undersized_buffer_is_an_error_not_a_panic() {
    for perm in [vec![1usize, 0, 2], vec![0, 2, 1]] {
        let malformed = Tensor {
            data: vec![0.0, 1.0, 2.0],
            shape: vec![1, 2, 3],
        };
        let err = shape::transpose(&malformed, &perm)
            .expect_err("an undersized buffer cannot be transposed");
        assert!(
            err.contains("3 elements") && err.contains("6"),
            "message should name both counts: {err}"
        );
    }
}

/// A size-0 axis inside the trailing identity run makes `run_len == 0`, which is
/// the divisor. A well-formed tensor is caught by the `numel() == 0` early
/// return; a malformed one with a non-empty buffer reaches the division, where
/// `0 / 0` would panic.
#[test]
fn transpose_with_a_zero_size_axis_does_not_divide_by_zero() {
    // Well-formed: empty buffer, zero-size axis in the trailing run.
    let out = shape::transpose(&zero_size(&[2, 3, 0]), &[1, 0, 2]).expect("well-formed");
    assert_eq!(out.shape, vec![3, 2, 0]);
    assert_eq!(out.data, Vec::<f32>::new());

    // Malformed: shape product 0 but a non-empty buffer.
    let malformed = Tensor {
        data: vec![7.0, 8.0],
        shape: vec![2, 3, 0],
    };
    let out = shape::transpose(&malformed, &[1, 0, 2]).expect("must not panic");
    assert_eq!(out.shape, vec![3, 2, 0]);
    assert_eq!(out.data, Vec::<f32>::new(), "no elements to carry");
}

/// The ordinary transpose results the migration must not disturb, including the
/// tiled trailing-swap fast path and the rank-0 identity.
#[test]
fn transpose_ordinary_results_are_unchanged() {
    // Trailing swap (tiled path): [2,3] -> [3,2].
    let m = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    let t = shape::transpose(&m, &[1, 0]).expect("2-D transpose");
    assert_eq!(t.shape, vec![3, 2]);
    assert_eq!(t.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

    // General odometer path: [1,2,3] -> [2,1,3] with perm [1,0,2].
    let g = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 2, 3]);
    let t = shape::transpose(&g, &[1, 0, 2]).expect("3-D transpose");
    assert_eq!(t.shape, vec![2, 1, 3]);
    assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    // Rank 0 with an empty perm is the identity.
    let z = shape::transpose(&Tensor::rank0(7.0), &[]).expect("rank-0 transpose");
    assert_eq!(z.shape, rank0_shape());
    assert_eq!(z.data, vec![7.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3 — scalar-output ops: execute and execute_into_slots must agree on rank 0
// ═══════════════════════════════════════════════════════════════════════════

/// `Constant` with `value_float` / `value_int` (and the no-attribute fallback
/// that stands in for them) emits rank 0 on **both** paths.
///
/// Opset-21 documents these attributes as "the value for the sole element for
/// the scalar ... output tensor". The slot path is asserted from a deliberately
/// mis-shaped incoming slot, because that is what a reused buffer looks like:
/// the op must overwrite `shape`, not leave the caller's `[1]` in place.
#[test]
fn constant_scalar_attributes_are_rank0_on_both_paths() -> Result<(), OnnxError> {
    let mut float_node = dummy_node("Constant");
    float_node.attrs.floats.insert("value_float".into(), 3.25);

    let mut int_node = dummy_node("Constant");
    int_node.attrs.ints.insert("value_int".into(), 42);

    let bare_node = dummy_node("Constant");

    for (node, expected, label) in [
        (&float_node, 3.25_f32, "value_float"),
        (&int_node, 42.0, "value_int"),
        (&bare_node, 0.0, "no value attribute"),
    ] {
        let out = ConstantOp.execute(&make_ctx(node, Vec::new()))?;
        assert_eq!(out[0].shape, rank0_shape(), "{label}: execute");
        assert_eq!(out[0].data, vec![expected], "{label}: execute value");

        // A stale `[1]`-shaped slot, as a reused buffer would be.
        let mut slots = vec![Tensor::new(vec![-1.0], vec![1])];
        ConstantOp.execute_into_slots(&make_ctx(node, Vec::new()), &mut slots)?;
        assert_eq!(slots[0].shape, rank0_shape(), "{label}: slot path");
        assert_eq!(slots[0].data, vec![expected], "{label}: slot value");
    }
    Ok(())
}

/// The `value` **tensor** attribute is the arm that did not change: it carries
/// its own shape, which is passed through untouched on both paths.
#[test]
fn constant_with_a_tensor_value_keeps_its_shape_on_both_paths() -> Result<(), OnnxError> {
    let mut node = dummy_node("Constant");
    node.attrs
        .tensors
        .insert("value".into(), Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));

    let out = ConstantOp.execute(&make_ctx(&node, Vec::new()))?;
    assert_eq!(out[0].shape, vec![3]);

    // A stale rank-0 slot (which holds exactly one element -- the empty shape's
    // product is the empty product 1), to prove the tensor arm overwrites it.
    let mut slots = vec![Tensor::rank0(-1.0)];
    ConstantOp.execute_into_slots(&make_ctx(&node, Vec::new()), &mut slots)?;
    assert_eq!(slots[0].shape, vec![3]);
    assert_eq!(slots[0].data, vec![1.0, 2.0, 3.0]);
    Ok(())
}

/// `DynamicQuantizeLinear`'s `y_scale` and `y_zero_point` are declared scalars
/// ("it's a scalar, which means a per-tensor/layer quantization"), so they are
/// rank 0 regardless of the input's rank; only `y` carries the input shape.
///
/// The numeric values are the op's own documented formula on this input:
/// `max_x = max(0, 2.0) = 2.0`, `min_x = min(0, -3.0) = -3.0`, so
/// `y_scale = 5/255` and `y_zero_point = round(0 - (-3.0)/(5/255)) = 153`.
#[test]
fn dynamic_quantize_linear_emits_rank0_scale_and_zero_point() -> Result<(), OnnxError> {
    let registry = oxionnx_ops::default_registry();
    let op = registry
        .get("DynamicQuantizeLinear")
        .ok_or_else(|| OnnxError::Internal("DynamicQuantizeLinear not registered".into()))?;
    let node = dummy_node("DynamicQuantizeLinear");

    for shape in [vec![6usize], vec![2, 3]] {
        let x = Tensor::new(vec![0.0, 2.0, -3.0, -2.5, 1.34, 0.5], shape.clone());
        let out = op.execute(&make_ctx(&node, vec![Some(&x)]))?;
        assert_eq!(out.len(), 3, "y, y_scale, y_zero_point");
        assert_eq!(out[0].shape, shape, "y keeps the input shape");
        assert_eq!(out[1].shape, rank0_shape(), "y_scale is a scalar");
        assert_eq!(out[2].shape, rank0_shape(), "y_zero_point is a scalar");
        assert!(
            (out[1].data[0] - 5.0 / 255.0).abs() < 1e-9,
            "y_scale value: {:?}",
            out[1].data
        );
        assert_eq!(out[2].data, vec![153.0], "y_zero_point value");

        // A following `Shape` node is where the rank is actually observable.
        let shape_node = dummy_node("Shape");
        let dims = ShapeOp.execute(&make_ctx(&shape_node, vec![Some(&out[1])]))?;
        assert_eq!(
            dims[0].shape,
            vec![0],
            "Shape of y_scale is a length-0 vector"
        );
        assert_eq!(dims[0].data, Vec::<f32>::new());
    }
    Ok(())
}

/// `Loop` hands its body `iteration_num` and `cond` as declared **rank-0**
/// scalars.
///
/// The body below makes both observable without reading `data[0]` (which works
/// at either rank and so proves nothing): one scan output is `Shape(iter_num)`,
/// which is a length-0 vector exactly when `iter_num` is rank 0, and the other
/// is `Identity(iter_num)` itself. Three iterations stack a rank-0 value into
/// `[3]` — the ONNX rule is `[iters] ++ per_iteration_shape`, and
/// `per_iteration_shape` is `[]` — where a `[1]`-shaped `iteration_num` would
/// have given `[3, 1]`.
#[test]
fn loop_feeds_iteration_num_and_cond_as_rank0() -> Result<(), OnnxError> {
    let registry = oxionnx_ops::default_registry();

    let body = Graph {
        nodes: vec![
            plain_node(OpKind::Identity, "cond_pass", &["cond_in"], &["cond_out"]),
            plain_node(OpKind::Identity, "acc_pass", &["acc_in"], &["acc_out"]),
            plain_node(OpKind::Shape, "iter_dims", &["iter_num"], &["dims_scan"]),
            plain_node(OpKind::Identity, "iter_pass", &["iter_num"], &["iter_scan"]),
            plain_node(OpKind::Shape, "cond_dims", &["cond_in"], &["cond_scan"]),
        ],
        input_names: vec!["iter_num".into(), "cond_in".into(), "acc_in".into()],
        output_names: vec![
            "cond_out".into(),
            "acc_out".into(),
            "dims_scan".into(),
            "iter_scan".into(),
            "cond_scan".into(),
        ],
        ..Default::default()
    };

    let mut attrs = Attributes::default();
    attrs.graphs.insert("body".into(), body);
    let node = Node {
        op: OpKind::Loop,
        name: "loop_node".into(),
        inputs: vec!["max_trip".into(), "init_cond".into(), "init_acc".into()],
        outputs: vec![
            "final_acc".into(),
            "dims_out".into(),
            "iters_out".into(),
            "conds_out".into(),
        ],
        attrs,
    };

    let max_trip = Tensor::scalar(3.0);
    let init_cond = Tensor::scalar(1.0);
    let init_acc = Tensor::scalar(0.0);
    let outer_scope: HashMap<String, Tensor> = HashMap::new();
    let ctx = OpContext {
        node: &node,
        inputs: vec![Some(&max_trip), Some(&init_cond), Some(&init_acc)],
        outer_scope: Some(&outer_scope),
        weights: None,
        registry: Some(&registry),
    };

    let out = LoopOp.execute(&ctx)?;
    assert_eq!(out.len(), 4, "1 carried dep + 3 scan outputs");

    // `Shape(iter_num)` is a length-0 vector per iteration; 3 of them stack to
    // [3, 0]. A `[1]`-shaped iteration_num would have given [3, 1] with data
    // [1,1,1].
    assert_eq!(
        out[1].shape,
        vec![3, 0],
        "Shape(iteration_num) reports no axes"
    );
    assert_eq!(out[1].data, Vec::<f32>::new());

    // `iteration_num` itself: rank-0 values stacked along a new leading axis.
    assert_eq!(
        out[2].shape,
        vec![3],
        "3 rank-0 iteration indices stack to [3]"
    );
    assert_eq!(out[2].data, vec![0.0, 1.0, 2.0]);

    // `cond` is declared a scalar too, and takes the same path.
    assert_eq!(out[3].shape, vec![3, 0], "Shape(cond) reports no axes");
    Ok(())
}
