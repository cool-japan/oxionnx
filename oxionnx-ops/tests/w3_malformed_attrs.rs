//! Wave-3 `T6-tests-ops`: malformed-attribute negative tests across ops with
//! no prior negative-attribute coverage, from finding [a11-20].
//!
//! `Slice`/`Concat`/`Split`/`Squeeze`/`Transpose`/`Expand`/`Reshape` and the
//! `Softmax` family already have out-of-range-axis / malformed-list
//! regression tests (`oxionnx-ops/tests/w1_shape_ops.rs`,
//! `oxionnx-ops/tests/opset_softmax_family.rs`) — none of that is repeated
//! here. This file picks ops with **zero** prior negative-attribute
//! coverage: out-of-range `axis` on `Gather`/`GatherElements`/
//! `ScatterElements`/`ArgMax`/`ArgMin`/`CumSum`/`TopK`/`OneHot`/`Compress`,
//! `blocksize<=0` on `DepthToSpace`/`SpaceToDepth`, a length-mismatched
//! `repeats` on `Tile`, and unrecognized-enum-string attributes on
//! `Cast`/`Resize`/`LSTM`/`GRU`.
//!
//! Each malformed case gets its own `#[test]` rather than one table-driven
//! loop over all of them: this file cannot edit `src/`, so a case that turns
//! out to genuinely accept the malformed input (or panic on it) has to be
//! individually `#[ignore]`d with the finding written into its doc comment —
//! a shared loop would force ignoring all fifteen the moment one fails.
//!
//! Every test constructs an otherwise-valid node and asserts
//! `Operator::execute(...).is_err()` directly on the `Result` (no
//! `catch_unwind`, per this domain's brief) — a case that instead panics
//! shows up as a failed/aborted test in the harness output, which is exactly
//! as diagnostic and does not need `catch_unwind` to be caught.

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::Tensor;
use oxionnx_ops::registry::conv_ops::ResizeOp;
use oxionnx_ops::registry::indexing_ops::{
    CompressOp, GatherElementsOp, GatherOp, OneHotOp, ScatterElementsOp,
};
use oxionnx_ops::registry::math_ops::{ArgMaxOp, ArgMinOp, CumSumOp, TopKOp};
use oxionnx_ops::registry::misc_ops::CastOp;
use oxionnx_ops::registry::rnn_ops::{GRUOp, LSTMOp, RNNOp};
use oxionnx_ops::registry::shape_ops::{DepthToSpaceOp, SpaceToDepthOp, TileOp};

// ── Test infrastructure ──────────────────────────────────────────────────────

fn node_with(op: OpKind, int_attrs: &[(&str, i64)], str_attrs: &[(&str, &str)]) -> Node {
    let mut attrs = Attributes::default();
    for &(k, v) in int_attrs {
        attrs.ints.insert(k.to_string(), v);
    }
    for &(k, v) in str_attrs {
        attrs.strings.insert(k.to_string(), v.to_string());
    }
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs,
    }
}

fn ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Out-of-range `axis`, ops with no prior coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gather_rejects_out_of_range_axis() {
    let node = node_with(OpKind::Gather, &[("axis", 99)], &[]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let idx = Tensor::new(vec![0.0], vec![1]);
    assert!(GatherOp
        .execute(&ctx(&node, vec![Some(&x), Some(&idx)]))
        .is_err());
}

#[test]
fn gather_elements_rejects_out_of_range_axis() {
    let node = node_with(OpKind::GatherElements, &[("axis", 99)], &[]);
    let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let idx = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
    assert!(GatherElementsOp
        .execute(&ctx(&node, vec![Some(&data), Some(&idx)]))
        .is_err());
}

#[test]
fn scatter_elements_rejects_out_of_range_axis() {
    let node = node_with(OpKind::ScatterElements, &[("axis", 99)], &[]);
    let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let idx = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
    let updates = Tensor::new(vec![9.0, 9.0, 9.0, 9.0], vec![2, 2]);
    assert!(ScatterElementsOp
        .execute(&ctx(&node, vec![Some(&data), Some(&idx), Some(&updates)]))
        .is_err());
}

#[test]
fn arg_max_rejects_out_of_range_axis() {
    let node = node_with(OpKind::ArgMax, &[("axis", 99), ("keepdims", 0)], &[]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert!(ArgMaxOp.execute(&ctx(&node, vec![Some(&x)])).is_err());
}

#[test]
fn arg_min_rejects_out_of_range_axis() {
    let node = node_with(OpKind::ArgMin, &[("axis", 99), ("keepdims", 0)], &[]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert!(ArgMinOp.execute(&ctx(&node, vec![Some(&x)])).is_err());
}

/// [a11-20, fixed] `execute_into_slots` takes a different path to the output
/// shape than `execute` above: it calls `math::arg_output_shape` to size the
/// pre-allocated slot *before* `arg_reduce_into` ever validates `axis`.
/// `arg_output_shape` used to index (`s[ax] = 1`) / `Vec::remove`
/// (`s.remove(ax)`) the shape with an unchecked `axis`, so an out-of-range
/// axis reaching this path panicked instead of erroring even though
/// `arg_max_rejects_out_of_range_axis` above (which never calls
/// `arg_output_shape`) already passed. Fixed by validating `axis` inside
/// `arg_output_shape` itself, before either operation, and by the
/// `execute_into_slots` call sites (`registry/math_ops/reduce.rs`) now
/// propagating that check's error before touching the slot at all.
#[test]
fn arg_max_slots_rejects_out_of_range_axis() {
    let node = node_with(OpKind::ArgMax, &[("axis", 99), ("keepdims", 0)], &[]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mut slots = vec![Tensor::new(vec![0.0], vec![1])];
    assert!(ArgMaxOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .is_err());
}

/// Same root cause as `arg_max_slots_rejects_out_of_range_axis` above, for `ArgMin`.
#[test]
fn arg_min_slots_rejects_out_of_range_axis() {
    let node = node_with(OpKind::ArgMin, &[("axis", 99), ("keepdims", 0)], &[]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mut slots = vec![Tensor::new(vec![0.0], vec![1])];
    assert!(ArgMinOp
        .execute_into_slots(&ctx(&node, vec![Some(&x)]), &mut slots)
        .is_err());
}

/// `CumSum`'s `axis` is a required *input* (not an attribute) per spec — the
/// malformed value is fed as input 1, not through `node_with`'s attrs.
#[test]
fn cumsum_rejects_out_of_range_axis_input() {
    let node = node_with(OpKind::CumSum, &[], &[]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let axis = Tensor::new(vec![99.0], vec![]);
    assert!(CumSumOp
        .execute(&ctx(&node, vec![Some(&x), Some(&axis)]))
        .is_err());
}

#[test]
fn topk_rejects_out_of_range_axis() {
    let node = node_with(OpKind::TopK, &[("axis", 99)], &[]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]);
    let k = Tensor::new(vec![1.0], vec![1]);
    assert!(TopKOp
        .execute(&ctx(&node, vec![Some(&x), Some(&k)]))
        .is_err());
}

#[test]
fn one_hot_rejects_out_of_range_axis() {
    let node = node_with(OpKind::OneHot, &[("axis", 99)], &[]);
    let indices = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let depth = Tensor::new(vec![4.0], vec![1]);
    let values = Tensor::new(vec![0.0, 1.0], vec![2]);
    assert!(OneHotOp
        .execute(&ctx(
            &node,
            vec![Some(&indices), Some(&depth), Some(&values)]
        ))
        .is_err());
}

#[test]
fn compress_rejects_out_of_range_axis() {
    let node = node_with(OpKind::Compress, &[("axis", 99)], &[]);
    let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let condition = Tensor::new(vec![1.0, 0.0], vec![2]);
    assert!(CompressOp
        .execute(&ctx(&node, vec![Some(&data), Some(&condition)]))
        .is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wrong-length lists
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tile_rejects_repeats_length_mismatch() {
    let node = node_with(OpKind::Tile, &[], &[]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    // 3 entries for a rank-2 tensor.
    let repeats = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(TileOp
        .execute(&ctx(&node, vec![Some(&x), Some(&repeats)]))
        .is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// blocksize <= 0: a genuine panic risk (integer modulo/division by zero),
// not just a validation gap
// ═══════════════════════════════════════════════════════════════════════════

/// [a11-20, fixed] `shape::depth_to_space` (`oxionnx-ops/src/shape/spatial.rs`)
/// used to compute `c_total % (r * r)` with `r = blocksize` **before**
/// checking `r > 0`. `blocksize=0` made that a modulo-by-zero, which is a
/// hard Rust panic ("attempt to calculate the remainder with a divisor of
/// zero"), not a typed `OnnxError` — exactly the "malformed attribute must
/// error, never panic" property this file exists to check. Fixed by
/// validating `blocksize > 0` before the modulo, in both `depth_to_space` and
/// `space_to_depth`.
#[test]
fn depth_to_space_rejects_zero_blocksize_without_panicking() {
    let node = node_with(OpKind::DepthToSpace, &[("blocksize", 0)], &[]);
    let x = Tensor::new(vec![0.0; 16], vec![1, 4, 2, 2]);
    assert!(DepthToSpaceOp.execute(&ctx(&node, vec![Some(&x)])).is_err());
}

/// [a11-20, fixed] Same root cause as `depth_to_space` above, in
/// `space_to_depth`'s `h % r` / `w % r` divisibility check.
#[test]
fn space_to_depth_rejects_zero_blocksize_without_panicking() {
    let node = node_with(OpKind::SpaceToDepth, &[("blocksize", 0)], &[]);
    let x = Tensor::new(vec![0.0; 16], vec![1, 1, 4, 4]);
    assert!(SpaceToDepthOp.execute(&ctx(&node, vec![Some(&x)])).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Bad enum values
// ═══════════════════════════════════════════════════════════════════════════

/// `Resize`'s mode dispatch (`oxionnx-ops/src/resize.rs::parse_interp`) *is*
/// a typed-error catch-all now (`other => Err(OnnxError::Unsupported(...))`)
/// — this pins that as a passing regression test. Historically (finding
/// [a11-4]) an unrecognized mode silently fell back to nearest-neighbor
/// instead; that is no longer the case.
#[test]
fn resize_rejects_unknown_mode() {
    let node = node_with(OpKind::Resize, &[], &[("mode", "not_a_real_mode_xyz")]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let scales = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4]);
    // Resize input order: X, roi(opt), scales(opt), sizes(opt).
    assert!(ResizeOp
        .execute(&ctx(&node, vec![Some(&x), None, Some(&scales), None]))
        .is_err());
}

/// [a11-20, fixed] `CastOp::execute` (`oxionnx-ops/src/registry/misc_ops.rs`)
/// used to match the ONNX `to` enum with an explicit arm per supported
/// `TensorProto.DataType` value and a catch-all `_ => x.data.clone()` for
/// everything else — so an out-of-range `to` (no `TensorProto.DataType`
/// goes anywhere near `999999`) silently became a no-op cast instead of a
/// typed error. Fixed by routing `to` through `DType::from_onnx` — already
/// used by the typed dispatch path two lines below this one — and erroring
/// when that returns `None`, before the match; `execute_typed`'s identical
/// `.unwrap_or(DType::F32)` fallback got the same fix.
#[test]
fn cast_rejects_unknown_to_dtype() {
    let node = node_with(OpKind::Cast, &[("to", 999_999)], &[]);
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    assert!(CastOp.execute(&ctx(&node, vec![Some(&x)])).is_err());
}

/// [a11-20, fixed] Neither `LSTMOp` nor the shared `direction` plumbing in
/// `oxionnx-ops/src/rnn/lstm.rs` used to validate the `direction` string
/// against `{"forward", "reverse", "bidirectional"}` — every comparison was
/// `direction == "bidirectional"` / `direction == "reverse"`, so any other
/// string (typo, wrong case, `"sideways"`) silently fell through to plain
/// forward-only execution instead of reporting an error. Fixed by
/// `rnn::common::validate_direction`, called once at the top of
/// `lstm_into_seq_major` — the single core `execute`, `execute_into_slots`,
/// and the F16/BF16 typed dispatch path (`rnn_typed::lstm_f16`/`lstm_bf16`,
/// which call the plain `rnn::lstm` entry point) all funnel through.
#[test]
fn lstm_rejects_unknown_direction() {
    let node = node_with(
        OpKind::LSTM,
        &[("hidden_size", 4)],
        &[("direction", "sideways")],
    );
    let x = Tensor::new(vec![0.0; 2 * 3], vec![2, 1, 3]); // [seq, batch, input_size]
    let w = Tensor::new(vec![0.1; 16 * 3], vec![1, 16, 3]); // [num_dir, 4*hidden, input_size]
    let r = Tensor::new(vec![0.1; 16 * 4], vec![1, 16, 4]); // [num_dir, 4*hidden, hidden]
    assert!(LSTMOp
        .execute(&ctx(
            &node,
            vec![Some(&x), Some(&w), Some(&r), None, None, None, None, None]
        ))
        .is_err());
}

/// [a11-20, fixed] Same root cause as `LSTM` above, in
/// `oxionnx-ops/src/rnn/gru.rs` — fixed by the same
/// `rnn::common::validate_direction` call, at the top of
/// `gru_into_seq_major`.
#[test]
fn gru_rejects_unknown_direction() {
    let node = node_with(
        OpKind::GRU,
        &[("hidden_size", 4)],
        &[("direction", "sideways")],
    );
    let x = Tensor::new(vec![0.0; 2 * 3], vec![2, 1, 3]); // [seq, batch, input_size]
    let w = Tensor::new(vec![0.1; 12 * 3], vec![1, 12, 3]); // [num_dir, 3*hidden, input_size]
    let r = Tensor::new(vec![0.1; 12 * 4], vec![1, 12, 4]); // [num_dir, 3*hidden, hidden]
    assert!(GRUOp
        .execute(&ctx(
            &node,
            vec![Some(&x), Some(&w), Some(&r), None, None, None]
        ))
        .is_err());
}

/// [G1-ops-close] Same root cause as `LSTM`/`GRU` above, in the plain
/// `RNN` op's kernel (`oxionnx-ops/src/rnn/simple_rnn.rs::simple_rnn_ext`),
/// which was never updated when `rnn::common::validate_direction` was added
/// for `LSTM`/`GRU` -- `num_dir`/`is_reverse` there compared `direction`
/// directly against `"reverse"`/`"bidirectional"` with no upfront
/// validation, so an unrecognized string silently ran as plain forward
/// execution instead of erroring. Fixed by the same `validate_direction`
/// call, once, immediately before `num_dir` is computed.
#[test]
fn rnn_rejects_unknown_direction() {
    let node = node_with(
        OpKind::RNN,
        &[("hidden_size", 4)],
        &[("direction", "sideways")],
    );
    let x = Tensor::new(vec![0.0; 2 * 3], vec![2, 1, 3]); // [seq, batch, input_size]
    let w = Tensor::new(vec![0.1; 4 * 3], vec![1, 4, 3]); // [num_dir, hidden, input_size]
    let r = Tensor::new(vec![0.1; 4 * 4], vec![1, 4, 4]); // [num_dir, hidden, hidden]
    assert!(RNNOp
        .execute(&ctx(
            &node,
            vec![Some(&x), Some(&w), Some(&r), None, None, None]
        ))
        .is_err());
}
