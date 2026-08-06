//! Wave-3 `T6-tests-ops`: axes-as-INPUT / split-as-INPUT forms (opset 13/18)
//! exercised end-to-end through a real `Session::run`, from finding [a11-8]
//! and [a11-19].
//!
//! `ReduceSum`/`ReduceMean` (opset 18), `Squeeze`/`Unsqueeze` (opset 13, the
//! *required* tensor-input form), and `Split` (both the opset-13 `split`
//! tensor input and the opset-18 `num_outputs` attribute) all correctly branch
//! on a second graph input in their `Operator::execute`/`execute_into_slots`,
//! and `oxionnx-ops/tests/output_slots_f14_test.rs` already proves the
//! Reduce* family agrees between `execute` and `execute_into_slots` when axes
//! arrive as a tensor. What none of that covers is the *session* seam: graph
//! construction, named-input feeding, topological/shape resolution, and (for
//! `Split`) real output-slot allocation for a node with more than one output.
//! `Squeeze`/`Unsqueeze` axes-as-input and `Split`'s `split`-as-input /
//! `num_outputs` forms have **zero** coverage at any layer before this file
//! (a corpus-wide grep for `Some(&axes)` / `num_outputs` outside this file's
//! new additions confirms it — see the session's final report for the exact
//! commands). Only `Unsqueeze` with a *single* axis via input was previously
//! covered (`tests/op_shape_tests.rs::test_squeeze_unsqueeze`, despite its
//! name); this file adds `Squeeze` and the multi-axis `Unsqueeze` case.
//!
//! Every expected value below is a hand-traced arithmetic result (sums/means
//! of small integer sequences, or pure reshapes), not copied from the
//! implementation under test.

mod common;

use common::run_op;
use oxionnx::{Attributes, OpKind, Tensor};

fn int_attr(pairs: &[(&str, i64)]) -> Attributes {
    let mut attrs = Attributes::default();
    for &(k, v) in pairs {
        attrs.ints.insert(k.to_string(), v);
    }
    attrs
}

// ── ReduceSum / ReduceMean: axes as a second graph input (opset 18) ─────────

/// `ReduceSum(axes=[1], keepdims=1)` with axes fed as a runtime tensor input,
/// not the legacy `axes` attribute. x = [[0,1,2],[3,4,5]] (shape [2,3]);
/// row sums are 3 and 12; `keepdims=1` keeps the reduced axis at size 1.
#[test]
fn reduce_sum_axes_as_input_e2e_with_keepdims() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    let axes = Tensor::new(vec![1.0], vec![1]);

    let out = run_op(
        OpKind::ReduceSum,
        vec!["x", "axes"],
        vec!["y"],
        vec!["x", "axes"],
        vec![("x", x), ("axes", axes)],
        vec![],
        int_attr(&[("keepdims", 1)]),
    );

    let y = out.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![2, 1], "keepdims=1 keeps the reduced axis");
    assert_eq!(y.data, vec![3.0, 12.0]);
}

/// The same op family, `keepdims=0`: the reduced axis disappears entirely.
/// Mean of each row of the same input is 1 and 4.
#[test]
fn reduce_mean_axes_as_input_e2e_no_keepdims() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    let axes = Tensor::new(vec![1.0], vec![1]);

    let out = run_op(
        OpKind::ReduceMean,
        vec!["x", "axes"],
        vec!["y"],
        vec!["x", "axes"],
        vec![("x", x), ("axes", axes)],
        vec![],
        int_attr(&[("keepdims", 0)]),
    );

    let y = out.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![2], "keepdims=0 drops the reduced axis");
    assert_eq!(y.data, vec![1.0, 4.0]);
}

/// The tensor-input axes value can be negative, exactly like the attribute
/// form: `axes=[-1]` on a rank-2 input names the same (last) axis as `[1]`.
/// This exercises the negative-axis normalization on the *input-tensor*
/// branch specifically, a different code path from the attribute branch that
/// the pre-existing tests all use.
#[test]
fn reduce_sum_axes_as_input_negative_axis_e2e() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    let axes = Tensor::new(vec![-1.0], vec![1]);

    let out = run_op(
        OpKind::ReduceSum,
        vec!["x", "axes"],
        vec!["y"],
        vec!["x", "axes"],
        vec![("x", x), ("axes", axes)],
        vec![],
        int_attr(&[("keepdims", 0)]),
    );

    let y = out.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![2]);
    assert_eq!(
        y.data,
        vec![3.0, 12.0],
        "axes=[-1] == axes=[1] on a rank-2 input"
    );
}

// ── Squeeze / Unsqueeze: axes as a second graph input (opset 13, required) ──

/// `Squeeze` with `axes` supplied as a tensor input rather than an attribute,
/// removing a *proper subset* of the size-1 axes (not every axis, so this
/// stays well clear of the pending [a0-21] rank-0 promotion edit tracked in
/// `oxionnx-ops/tests/w2_rank0.rs`). x shape [1,3,1], axes=[0,2] -> [3].
#[test]
fn squeeze_axes_as_input_e2e_partial() {
    let x = Tensor::new(vec![5.0, 6.0, 7.0], vec![1, 3, 1]);
    let axes = Tensor::new(vec![0.0, 2.0], vec![2]);

    let out = run_op(
        OpKind::Squeeze,
        vec!["x", "axes"],
        vec!["y"],
        vec!["x", "axes"],
        vec![("x", x), ("axes", axes)],
        vec![],
        Attributes::default(),
    );

    let y = out.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![3]);
    assert_eq!(y.data, vec![5.0, 6.0, 7.0]);
}

/// `Unsqueeze` with **two** axes supplied as a tensor input. The only
/// pre-existing session-level coverage (`tests/op_shape_tests.rs`) used a
/// single axis; inserting more than one dimension at once is a materially
/// different code path (output rank = input rank + len(axes), axes normalized
/// against the *output* rank). x shape [3], axes=[0,2] -> [1,3,1].
#[test]
fn unsqueeze_multi_axes_as_input_e2e() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let axes = Tensor::new(vec![0.0, 2.0], vec![2]);

    let out = run_op(
        OpKind::Unsqueeze,
        vec!["x", "axes"],
        vec!["y"],
        vec!["x", "axes"],
        vec![("x", x), ("axes", axes)],
        vec![],
        Attributes::default(),
    );

    let y = out.get("y").expect("output 'y'");
    assert_eq!(y.shape, vec![1, 3, 1]);
    assert_eq!(y.data, vec![1.0, 2.0, 3.0]);
}

// ── Split: sizes-as-input (opset 13) and num_outputs attribute (opset 18) ───

/// `Split` with the `split` sizes tensor supplied as the second input
/// (opset-13 form), not the legacy `split` int-list attribute — the form
/// every modern (opset>=13) exporter emits. x = 0..8 (shape [8]),
/// split=[3,5], axis=0 -> chunks [0,1,2] and [3,4,5,6,7].
#[test]
fn split_sizes_as_input_e2e() {
    let x = Tensor::new((0..8).map(|i| i as f32).collect(), vec![8]);
    let split = Tensor::new(vec![3.0, 5.0], vec![2]);

    let out = run_op(
        OpKind::Split,
        vec!["x", "split"],
        vec!["a", "b"],
        vec!["x", "split"],
        vec![("x", x), ("split", split)],
        vec![],
        int_attr(&[("axis", 0)]),
    );

    let a = out.get("a").expect("output 'a'");
    let b = out.get("b").expect("output 'b'");
    assert_eq!(a.shape, vec![3]);
    assert_eq!(a.data, vec![0.0, 1.0, 2.0]);
    assert_eq!(b.shape, vec![5]);
    assert_eq!(b.data, vec![3.0, 4.0, 5.0, 6.0, 7.0]);
}

/// `Split` with neither a `split` input nor attribute, driven purely by the
/// opset-18 `num_outputs` attribute — through the real session, which is what
/// forces genuine multi-output *slot allocation* (the direct-`execute` unit
/// test in `oxionnx-ops/tests/w1_shape_ops.rs` cannot reach that path). Axis
/// length 5, num_outputs=4 -> ONNX's equal-split-with-smaller-last-chunk rule
/// gives sizes [2,2,1,0]: four outputs, the last legitimately empty.
#[test]
fn split_num_outputs_attribute_e2e_with_trailing_zero_chunk() {
    let x = Tensor::new((0..5).map(|i| i as f32).collect(), vec![5]);

    let out = run_op(
        OpKind::Split,
        vec!["x"],
        vec!["a", "b", "c", "d"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        int_attr(&[("axis", 0), ("num_outputs", 4)]),
    );

    let sizes: Vec<usize> = ["a", "b", "c", "d"]
        .iter()
        .map(|name| out.get(*name).expect("output").shape[0])
        .collect();
    assert_eq!(
        sizes,
        vec![2, 2, 1, 0],
        "must produce exactly num_outputs tensors"
    );
    assert_eq!(out["a"].data, vec![0.0, 1.0]);
    assert_eq!(out["b"].data, vec![2.0, 3.0]);
    assert_eq!(out["c"].data, vec![4.0]);
    assert!(out["d"].data.is_empty());
}
