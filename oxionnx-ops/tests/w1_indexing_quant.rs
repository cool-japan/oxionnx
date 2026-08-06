//! Wave-1 correctness tests for the `D-indexing-quant` domain: real N-D
//! `Where` broadcasting, `ScatterElements`/`ScatterND` reduction +
//! bounds-checking + negative-index handling, `Gather`/`GatherElements`
//! bounds-checking consistency, `OneHot` depth validation, and
//! `QuantizeLinear`/`DequantizeLinear` per-axis scale/zero-point + dtype-
//! aware saturation + round-ties-to-even.
//!
//! Reference values are cross-checked with `numpy` (float32) — see the
//! computation notes inline; the exact commands used are reproduced in
//! comments so the numbers are independently re-derivable.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::indexing;
use oxionnx_ops::registry::indexing_ops::{
    DequantizeLinearOp, GatherOp, OneHotOp, QuantizeLinearOp, ScatterElementsOp, ScatterNDOp,
};

// ── Test infrastructure (mirrors tests/output_slots_indexing_test.rs) ───────

fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn dummy_node(op: OpKind) -> Node {
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn node_with_attrs(op: OpKind, ints: &[(&str, i64)], strings: &[(&str, &str)]) -> Node {
    let mut n = dummy_node(op);
    for &(k, v) in ints {
        n.attrs.ints.insert(k.to_string(), v);
    }
    for &(k, v) in strings {
        n.attrs.strings.insert(k.to_string(), v.to_string());
    }
    n
}

fn assert_close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < 1e-5,
            "{label}[{i}]: got {av}, expected {bv}"
        );
    }
}

// ── [a0-0] Where: real N-D broadcasting ──────────────────────────────────────

#[test]
fn where_op_broadcasts_non_trailing_axis() {
    // Brief's concrete counter-example: modulo indexing gives [1,0,3,0,5,0];
    // real broadcasting must give [[1,2,3],[0,0,0]].
    let cond = Tensor::new(vec![1.0, 0.0], vec![2, 1]);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let y = Tensor::new(vec![0.0], vec![1]);
    let out = indexing::where_op(&cond, &x, &y).expect("where_op failed");
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
}

#[test]
fn where_op_attention_mask_pattern() {
    // Realistic instance named in the brief: Where(mask[B,1,1,S],
    // scores[B,H,S,S], neg) — mask broadcasts over H and the query axis S,
    // varying only over batch and the key axis. Reference computed with:
    //   np.where(mask.reshape(2,1,1,2) != 0, scores.reshape(2,2,2,2), -100)
    let scores = Tensor::new((0..16).map(|i| i as f32).collect(), vec![2, 2, 2, 2]);
    let mask = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 1, 1, 2]);
    let neg = Tensor::new(vec![-100.0], vec![1]);
    let out = indexing::where_op(&mask, &scores, &neg).expect("where_op failed");
    assert_eq!(out.shape, vec![2, 2, 2, 2]);
    assert_close(
        &out.data,
        &[
            0.0, -100.0, 2.0, -100.0, 4.0, -100.0, 6.0, -100.0, -100.0, 9.0, -100.0, 11.0, -100.0,
            13.0, -100.0, 15.0,
        ],
        "attention mask where",
    );
}

#[test]
fn where_op_empty_operand_does_not_panic() {
    // A zero-sized operand must not hit the old `i % numel()` division by
    // zero; broadcast_shape forces the output to be zero-sized too, so the
    // loop body (and thus any indexing) never runs.
    let cond = Tensor::new(vec![], vec![0]);
    let x = Tensor::new(vec![], vec![0]);
    let y = Tensor::new(vec![0.0], vec![1]);
    let out = indexing::where_op(&cond, &x, &y).expect("where_op on empty operand failed");
    assert_eq!(out.data.len(), 0);
}

// ── [a0-10]/[a5-4]/[a11-6] Scatter reduction ─────────────────────────────────

#[test]
fn scatter_nd_reduction_add_accumulates_duplicate_indices() {
    // Brief's concrete example: data=[0,0,0,0], indices=[[1],[1]],
    // updates=[[5],[3]], reduction='add' -> [0,8,0,0] (not [0,3,0,0]).
    let data = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let indices = Tensor::new(vec![1.0, 1.0], vec![2, 1]);
    let updates = Tensor::new(vec![5.0, 3.0], vec![2, 1]);
    let out =
        indexing::scatter_nd_reduce(&data, &indices, &updates, indexing::ScatterReduction::Add)
            .expect("scatter_nd add failed");
    assert_eq!(out.data, vec![0.0, 8.0, 0.0, 0.0]);

    // No-reduction default must still overwrite (last write wins), matching
    // pre-opset-16 behavior — regression guard against the reduction fix
    // accidentally always accumulating.
    let out_none = indexing::scatter_nd(&data, &indices, &updates).expect("scatter_nd failed");
    assert_eq!(out_none.data, vec![0.0, 3.0, 0.0, 0.0]);
}

#[test]
fn scatter_elements_reduction_add_accumulates_duplicate_indices() {
    // Brief's concrete example: data=[1,1,1], indices=[0,0], updates=[5,3],
    // reduction='add' -> [9,1,1] (not [3,1,1]).
    let data = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
    let indices = Tensor::new(vec![0.0, 0.0], vec![2]);
    let updates = Tensor::new(vec![5.0, 3.0], vec![2]);
    let out = indexing::scatter_elements_reduce(
        &data,
        &indices,
        &updates,
        0,
        indexing::ScatterReduction::Add,
    )
    .expect("scatter_elements add failed");
    assert_eq!(out.data, vec![9.0, 1.0, 1.0]);
}

#[test]
fn scatter_elements_reduction_mul_max_min() {
    let data = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]);
    let indices = Tensor::new(vec![0.0, 0.0], vec![2]);
    let updates = Tensor::new(vec![3.0, 4.0], vec![2]);
    let mul = indexing::scatter_elements_reduce(
        &data,
        &indices,
        &updates,
        0,
        indexing::ScatterReduction::Mul,
    )
    .expect("mul failed");
    assert_eq!(mul.data[0], 2.0 * 3.0 * 4.0);

    let max = indexing::scatter_elements_reduce(
        &data,
        &indices,
        &updates,
        0,
        indexing::ScatterReduction::Max,
    )
    .expect("max failed");
    assert_eq!(max.data[0], 4.0);

    let min = indexing::scatter_elements_reduce(
        &data,
        &indices,
        &updates,
        0,
        indexing::ScatterReduction::Min,
    )
    .expect("min failed");
    assert_eq!(min.data[0], 2.0);
}

#[test]
fn scatter_nd_op_reads_reduction_attribute_end_to_end() {
    // Validates the registry wiring: ScatterNDOp::execute must read the
    // ONNX `reduction` node attribute (a string attribute), not just axis.
    let data = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let indices = Tensor::new(vec![1.0, 1.0], vec![2, 1]);
    let updates = Tensor::new(vec![5.0, 3.0], vec![2, 1]);
    let node = node_with_attrs(OpKind::ScatterND, &[], &[("reduction", "add")]);
    let ctx = make_ctx(&node, vec![Some(&data), Some(&indices), Some(&updates)]);
    let out = ScatterNDOp
        .execute(&ctx)
        .expect("ScatterNDOp execute failed");
    assert_eq!(out[0].data, vec![0.0, 8.0, 0.0, 0.0]);

    // Also exercise the output-slot dispatch path with the same attribute.
    let mut slots = vec![Tensor::new(vec![0.0; 4], vec![4])];
    ScatterNDOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ScatterNDOp execute_into_slots failed");
    assert_eq!(slots[0].data, vec![0.0, 8.0, 0.0, 0.0]);
}

#[test]
fn scatter_elements_op_reads_reduction_attribute_end_to_end() {
    let data = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
    let indices = Tensor::new(vec![0.0, 0.0], vec![2]);
    let updates = Tensor::new(vec![5.0, 3.0], vec![2]);
    let node = node_with_attrs(
        OpKind::ScatterElements,
        &[("axis", 0)],
        &[("reduction", "add")],
    );
    let ctx = make_ctx(&node, vec![Some(&data), Some(&indices), Some(&updates)]);
    let out = ScatterElementsOp
        .execute(&ctx)
        .expect("ScatterElementsOp execute failed");
    assert_eq!(out[0].data, vec![9.0, 1.0, 1.0]);

    let mut slots = vec![Tensor::new(vec![0.0; 3], vec![3])];
    ScatterElementsOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("ScatterElementsOp execute_into_slots failed");
    assert_eq!(slots[0].data, vec![9.0, 1.0, 1.0]);
}

#[test]
fn scatter_reduction_rejects_unknown_string() {
    assert!(indexing::ScatterReduction::parse("bogus").is_err());
    assert!(indexing::ScatterReduction::parse("").is_ok());
    assert!(indexing::ScatterReduction::parse("none").is_ok());
}

// ── [a0-11] ScatterND negative indices count from the end ───────────────────

#[test]
fn scatter_nd_negative_index_counts_from_end() {
    // Brief's concrete example: data shape [4,2] zeros, indices=[[-1]],
    // updates=[[9,9]] must write row 3 (last row), not row 0.
    let data = Tensor::new(vec![0.0; 8], vec![4, 2]);
    let indices = Tensor::new(vec![-1.0], vec![1, 1]);
    let updates = Tensor::new(vec![9.0, 9.0], vec![1, 2]);
    let out = indexing::scatter_nd(&data, &indices, &updates).expect("scatter_nd failed");
    assert_eq!(
        out.data,
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0],
        "negative index must land on the last row, not row 0"
    );
}

// ── [a10-1] ScatterElements bounds-checks before writing ─────────────────────

#[test]
fn scatter_elements_out_of_range_index_errors_not_panics() {
    let data = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let indices = Tensor::new(vec![999.0], vec![1]);
    let updates = Tensor::new(vec![1.0], vec![1]);
    let err = indexing::scatter_elements(&data, &indices, &updates, 0);
    assert!(
        err.is_err(),
        "out-of-range scatter index must error, not panic"
    );

    // Same guarantee through the output-slot dispatch path used by the
    // memory planner.
    let node = node_with_attrs(OpKind::ScatterElements, &[("axis", 0)], &[]);
    let ctx = make_ctx(&node, vec![Some(&data), Some(&indices), Some(&updates)]);
    let mut slots = vec![Tensor::new(vec![0.0; 4], vec![4])];
    let result = ScatterElementsOp.execute_into_slots(&ctx, &mut slots);
    assert!(
        result.is_err(),
        "execute_into_slots must also error, not panic"
    );
}

// ── [a10-2] ScatterND bounds-checks before writing ───────────────────────────

#[test]
fn scatter_nd_out_of_range_component_errors_not_panics() {
    let data = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let indices = Tensor::new(vec![999.0], vec![1, 1]);
    let updates = Tensor::new(vec![1.0], vec![1, 1]);
    let err = indexing::scatter_nd(&data, &indices, &updates);
    assert!(
        err.is_err(),
        "out-of-range scatter_nd index must error, not panic"
    );

    let node = dummy_node(OpKind::ScatterND);
    let ctx = make_ctx(&node, vec![Some(&data), Some(&indices), Some(&updates)]);
    let mut slots = vec![Tensor::new(vec![0.0; 4], vec![4])];
    let result = ScatterNDOp.execute_into_slots(&ctx, &mut slots);
    assert!(
        result.is_err(),
        "execute_into_slots must also error, not panic"
    );
}

#[test]
fn scatter_nd_zero_index_depth_errors_not_panics() {
    // `indices.shape.last() == 0` used to divide by zero computing
    // `n_idx = indices.numel() / k`.
    let data = Tensor::new(vec![0.0, 0.0], vec![2]);
    let indices = Tensor::new(vec![], vec![1, 0]);
    // updates shape = indices.shape[:-1] + data.shape[k:] = [1] + [2] = [1,2]
    let updates = Tensor::new(vec![0.0, 0.0], vec![1, 2]);
    let err = indexing::scatter_nd(&data, &indices, &updates);
    assert!(
        err.is_err(),
        "zero index depth must error, not divide by zero"
    );
}

// ── [a0-18] Gather: execute vs execute_into_slots must agree ────────────────

#[test]
fn gather_out_of_range_index_errors_on_both_dispatch_paths() {
    let table = Tensor::new((0..20).map(|i| i as f32).collect(), vec![5, 4]);
    let bad_indices = Tensor::new(vec![2.0, 999.0], vec![2]);

    let direct = indexing::gather(&table, &bad_indices, 0);
    assert!(direct.is_err(), "gather() must error on out-of-range index");

    let node = node_with_attrs(OpKind::Gather, &[("axis", 0)], &[]);
    let ctx = make_ctx(&node, vec![Some(&table), Some(&bad_indices)]);
    let mut slots = vec![Tensor::new(vec![0.0; 8], vec![2, 4])];
    let via_slots = GatherOp.execute_into_slots(&ctx, &mut slots);
    assert!(
        via_slots.is_err(),
        "execute_into_slots must also error instead of silently clamping to the last row"
    );
}

#[test]
fn gather_deeply_negative_index_errors_on_both_dispatch_paths() {
    // -100 on a 10-row table must error, not clamp to row 9.
    let table = Tensor::new((0..20).map(|i| i as f32).collect(), vec![10, 2]);
    let bad_indices = Tensor::new(vec![-100.0], vec![1]);

    assert!(indexing::gather(&table, &bad_indices, 0).is_err());

    let node = node_with_attrs(OpKind::Gather, &[("axis", 0)], &[]);
    let ctx = make_ctx(&node, vec![Some(&table), Some(&bad_indices)]);
    let mut slots = vec![Tensor::new(vec![0.0; 2], vec![1, 2])];
    assert!(GatherOp.execute_into_slots(&ctx, &mut slots).is_err());
}

// ── [a0-19] GatherElements bounds-checks before reading ──────────────────────

#[test]
fn gather_elements_out_of_range_index_errors_not_panics() {
    // Brief's concrete example: data shape [2,2], indices=[[5,0],[0,0]],
    // axis=0 -> data_flat = 5*2+0 = 10 on a 4-element buffer.
    let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let indices = Tensor::new(vec![5.0, 0.0, 0.0, 0.0], vec![2, 2]);
    let err = indexing::gather_elements(&data, &indices, 0);
    assert!(
        err.is_err(),
        "out-of-range GatherElements index must error, not panic"
    );
}

#[test]
fn gather_elements_axis_out_of_range_errors() {
    let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let indices = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
    assert!(indexing::gather_elements(&data, &indices, 5).is_err());
}

// ── bonus: gather_nd negative batch_dims must not overflow-panic ────────────

#[test]
fn gather_nd_negative_batch_dims_errors_not_panics() {
    // batch_dims is spec'd non-negative; a malformed model setting it to -1
    // used to cast to `usize::MAX` and panic (either an arithmetic overflow
    // in debug builds, or an immediate index-out-of-bounds while building
    // the output shape) instead of returning a typed error.
    let data = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]);
    let indices = Tensor::new(vec![0.0], vec![1, 1]);
    let err = indexing::gather_nd(&data, &indices, -1);
    assert!(err.is_err(), "negative batch_dims must error, not panic");
}

// ── [a10-16] OneHot depth validation ─────────────────────────────────────────

#[test]
fn one_hot_huge_depth_errors_not_panics() {
    let indices = Tensor::new(vec![0.0, 1.0], vec![2]);
    let err = indexing::one_hot(&indices, 1_000_000_000_000, (0.0, 1.0), -1);
    assert!(
        err.is_err(),
        "absurd depth must error, not attempt a petabyte allocation"
    );
}

#[test]
fn one_hot_op_huge_depth_input_errors_end_to_end() {
    // Mirrors the exact adversarial-input shape from the brief: `depth` is
    // a *runtime input tensor*, not an attribute, holding e.g. 1e15.
    let indices = Tensor::new(vec![0.0], vec![1]);
    let depth_t = Tensor::new(vec![1.0e15], vec![1]);
    let values_t = Tensor::new(vec![0.0, 1.0], vec![2]);
    let node = dummy_node(OpKind::OneHot);
    let ctx = make_ctx(&node, vec![Some(&indices), Some(&depth_t), Some(&values_t)]);
    let result = OneHotOp.execute(&ctx);
    assert!(
        result.is_err(),
        "OneHotOp must error, not panic, on an absurd depth input"
    );
}

// ── [a3-2]/[a11-0] QuantizeLinear/DequantizeLinear per-axis scale/zero-point ─

#[test]
fn quantize_linear_per_channel_axis0_not_last_axis() {
    // Brief's corrected example (round-ties-to-even, see comment below):
    // x=[[1,2,3],[4,5,6]] shape [2,3], per-channel scale=[1.0,10.0], axis=0,
    // zero_point omitted (-> uint8 default, zp=0).
    //   row0 (scale=1.0): [1,2,3]/1.0 = [1,2,3] -> round -> [1,2,3]
    //   row1 (scale=10.0): [4,5,6]/10.0 = [0.4,0.5,0.6]
    //     -> round-ties-to-even -> [0,0,1]   (0.5 ties to the EVEN 0, not 1)
    // Computed and cross-checked with:
    //   np.round(np.float32([4,5,6])/np.float32(10.0)) == [0.,0.,1.]
    // NOTE: this differs from finding a11-0's own illustrative numbers
    // ([1,2,3,0,1,1]), which described round-half-AWAY-from-zero for the
    // 0.5 tie. That contradicts a3-2's explicit round-ties-to-even
    // requirement (the actual ONNX QuantizeLinear spec, matching numpy's
    // `round`), so the tie case here follows the spec-correct value.
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let scale = Tensor::new(vec![1.0, 10.0], vec![2]);
    let out = indexing::quantize_linear_axis(&x, &scale, None, 0).expect("quantize failed");
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 0.0, 0.0, 1.0]);
}

#[test]
fn quantize_linear_per_channel_axis1_non_leading() {
    // Same data quantized along axis=1 (the columns) instead of axis=0, to
    // prove the fix is genuinely stride-based and not coincidentally
    // correct only for axis=0 / the last axis.
    //   scale = [1.0, 2.0, 4.0] (one per column)
    // Reference: np.round(x/scale) elementwise, ties-to-even, zp=0:
    //   [[1,1,1],[4,2,2]]   (5/2=2.5 -> 2; 6/4=1.5 -> 2, both ties-to-even)
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let scale = Tensor::new(vec![1.0, 2.0, 4.0], vec![3]);
    let out = indexing::quantize_linear_axis(&x, &scale, None, 1).expect("quantize failed");
    assert_eq!(out.data, vec![1.0, 1.0, 1.0, 4.0, 2.0, 2.0]);
}

#[test]
fn quantize_linear_per_channel_zero_point() {
    // Per-channel zero_point must be indexed the same way as scale, not
    // just `zero_point.data[0]`. axis=0, scale=[1,1], zero_point=[2,100].
    // Reference: row0 = round(x/1)+2, row1 = round(x/1)+100.
    let x = Tensor::new(vec![5.0, 15.0, 7.0, 3.0], vec![2, 2]);
    let scale = Tensor::new(vec![1.0, 1.0], vec![2]);
    let zp = Tensor::new(vec![2.0, 100.0], vec![2]);
    let out = indexing::quantize_linear_axis(&x, &scale, Some(&zp), 0).expect("quantize failed");
    assert_eq!(out.data, vec![7.0, 17.0, 107.0, 103.0]);
}

#[test]
fn dequantize_linear_per_channel_axis0() {
    // q=[[10,20],[30,40]], scale=[2.0,0.5], zero_point=[1,2], axis=0.
    // Reference: y = (q - zp) * scale, per row -> [[18,38],[14,19]].
    let q = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let scale = Tensor::new(vec![2.0, 0.5], vec![2]);
    let zp = Tensor::new(vec![1.0, 2.0], vec![2]);
    let out =
        indexing::dequantize_linear_axis(&q, &scale, Some(&zp), 0).expect("dequantize failed");
    assert_eq!(out.data, vec![18.0, 38.0, 14.0, 19.0]);
}

#[test]
fn quantize_dequantize_op_reads_axis_attribute_end_to_end() {
    // Validates that QuantizeLinearOp/DequantizeLinearOp actually read the
    // `axis` node attribute (default 1) instead of hardcoding modulo
    // indexing.
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let scale = Tensor::new(vec![1.0, 10.0], vec![2]);
    let node = node_with_attrs(OpKind::QuantizeLinear, &[("axis", 0)], &[]);
    let ctx = make_ctx(&node, vec![Some(&x), Some(&scale), None]);
    let out = QuantizeLinearOp
        .execute(&ctx)
        .expect("QuantizeLinearOp failed");
    assert_eq!(out[0].data, vec![1.0, 2.0, 3.0, 0.0, 0.0, 1.0]);

    let dnode = node_with_attrs(OpKind::DequantizeLinear, &[("axis", 0)], &[]);
    let dctx = make_ctx(&dnode, vec![Some(&out[0]), Some(&scale), None]);
    let deq = DequantizeLinearOp
        .execute(&dctx)
        .expect("DequantizeLinearOp failed");
    // Row0 (scale 1.0): unchanged; row1 (scale 10.0) reconstructs to a
    // coarser multiple of 10 (lossy, as expected of quantization).
    assert_eq!(deq[0].data, vec![1.0, 2.0, 3.0, 0.0, 0.0, 10.0]);
}

// ── [a11-5] QuantizeLinear dtype-aware saturation range ──────────────────────

#[test]
fn quantize_linear_uint8_default_when_zero_point_omitted() {
    // Brief's concrete example: x=200, scale=1, zero_point omitted -> spec
    // output is 200 (valid uint8), NOT clamped to 127 as a hardcoded int8
    // range would.
    let x = Tensor::new(vec![200.0], vec![1]);
    let scale = Tensor::new(vec![1.0], vec![1]);
    let out = indexing::quantize_linear(&x, &scale, None).expect("quantize failed");
    assert_eq!(out.data, vec![200.0]);
}

#[test]
fn quantize_linear_symmetric_int8_weights_with_explicit_zero_point_zero() {
    // `Tensor` carries no dtype tag, so an EXPLICITLY-provided zero_point
    // of 0 is deliberately treated as int8 (not the uint8 default used when
    // zero_point is absent). Rationale: zero_point=0 is, in practice, the
    // signature of *symmetric int8* quantization — the standard scheme for
    // weight quantization (ORT static quantization, PyTorch FX, TensorRT
    // all default weights to signed symmetric int8 with zero_point 0).
    // Treating an explicit zero_point=0 as uint8 would silently zero out
    // every negative quantized weight instead of preserving it, which is a
    // far more damaging failure than the reverse. This is a deliberate,
    // reasoned deviation from finding a11-5's "(or 0)" parenthetical — see
    // the `saturation_range` doc comment in indexing/quantize.rs.
    let weight = Tensor::new(vec![-0.5, 0.3], vec![2]);
    let scale = Tensor::new(vec![0.01], vec![1]);
    let zp = Tensor::new(vec![0.0], vec![1]);
    let out = indexing::quantize_linear(&weight, &scale, Some(&zp)).expect("quantize failed");
    // round(-0.5/0.01) + 0 = -50 (must stay -50, not clamp to 0 as a
    // uint8-default interpretation would).
    assert_eq!(out.data, vec![-50.0, 30.0]);
}

#[test]
fn quantize_linear_uint8_when_explicit_zero_point_exceeds_int8_range() {
    // A provided zero-point > 127 can only be valid for uint8 — this case
    // remains unambiguous regardless of the int8-leaning default above.
    let x = Tensor::new(vec![200.0], vec![1]);
    let scale = Tensor::new(vec![1.0], vec![1]);
    let zp = Tensor::new(vec![128.0], vec![1]);
    let out = indexing::quantize_linear(&x, &scale, Some(&zp)).expect("quantize failed");
    // round(200/1) + 128 = 328, clamped to uint8 max 255.
    assert_eq!(out.data, vec![255.0]);
}

#[test]
fn quantize_linear_int8_range_when_zero_point_negative() {
    // A negative zero-point component can only be valid for a signed int8
    // output; the saturation range must switch to [-128, 127].
    let x = Tensor::new(vec![200.0], vec![1]);
    let scale = Tensor::new(vec![1.0], vec![1]);
    let zp = Tensor::new(vec![-1.0], vec![1]);
    let out = indexing::quantize_linear(&x, &scale, Some(&zp)).expect("quantize failed");
    // round(200/1) + (-1) = 199, clamped to int8 max 127.
    assert_eq!(out.data, vec![127.0]);
}

#[test]
fn quantize_linear_uint8_clamps_negative_input_to_zero_not_negative() {
    // Corrected roundtrip reference (also asserted in indexing::tests, kept
    // here as an independent integration-level check):
    // x=[0,1,-1,2], scale=0.01, zero_point omitted -> uint8 default.
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]);
    let scale = Tensor::new(vec![0.01], vec![1]);
    let out = indexing::quantize_linear(&x, &scale, None).expect("quantize failed");
    assert_eq!(out.data, vec![0.0, 100.0, 0.0, 200.0]);
}

// ── round-ties-to-even (spec-mandated, not Rust's default round()) ──────────

#[test]
fn quantize_linear_rounds_ties_to_even() {
    // 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4 (all ties resolve to the
    // nearest EVEN integer, matching numpy.round / the ONNX spec).
    let x = Tensor::new(vec![0.5, 1.5, 2.5, 3.5], vec![4]);
    let scale = Tensor::new(vec![1.0], vec![1]);
    let out = indexing::quantize_linear(&x, &scale, None).expect("quantize failed");
    assert_eq!(out.data, vec![0.0, 2.0, 2.0, 4.0]);
}
