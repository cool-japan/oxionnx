//! Wave-3: `Einsum` shape inference agrees with numpy **and** with the executor.
//!
//! # Why this file exists
//!
//! `infer_einsum_shape` (src/optimizer/shape_inference_ext/advanced.rs) publishes
//! the shape the engine pre-allocates an `Einsum` node's output slot from
//! (`Session::acquire_output_slots`) and validates provider results against
//! (`Session::write_node_outputs`).  It used to disagree with the executor in two
//! ways:
//!
//! * label extents were **first-writer-wins**, so `"ij,ij->ij"` over `(2, 1)` and
//!   `(2, 3)` inferred `[2, 1]` where numpy and `oxionnx_ops::einsum` produce
//!   `[2, 3]` (this became reachable when Wave-2 taught einsum to broadcast named
//!   size-1 labels);
//! * `'.'` bytes were filtered out of every subscript, so no ellipsis equation —
//!   i.e. essentially every attention model — inferred anything at all.
//!
//! # The oracle
//!
//! Every expected shape below is `numpy.einsum(...).shape` from numpy 2.4.2, and
//! every case is asserted **twice**: once against `infer_shapes` (what the
//! optimizer publishes) and once against a real `Session::run` of the same node
//! (what the engine actually produces).  The second assertion is what makes this
//! more than a transcription test — inference and execution cannot drift apart
//! without failing it.

use oxionnx::optimizer::shape_inference::infer_shapes;
use oxionnx::{Attributes, Graph, Node, OpKind, Session, Tensor};
use std::collections::HashMap;

/// Build a one-node `Einsum` graph over `n` inputs named `in0..in{n-1}`.
fn einsum_node(equation: &str, n_inputs: usize) -> Node {
    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("equation".to_string(), equation.to_string());
    Node {
        op: OpKind::Einsum,
        name: "ein".to_string(),
        inputs: (0..n_inputs).map(|i| format!("in{i}")).collect(),
        outputs: vec!["y".to_string()],
        attrs,
    }
}

/// The shape `infer_shapes` publishes for `y`, or `None` when it declines.
fn inferred(equation: &str, shapes: &[&[usize]]) -> Option<Vec<usize>> {
    let node = einsum_node(equation, shapes.len());
    let input_shapes: HashMap<String, Vec<usize>> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| (format!("in{i}"), s.to_vec()))
        .collect();
    infer_shapes(&[node], &HashMap::new(), &input_shapes)
        .get("y")
        .cloned()
}

/// The shape the **engine** actually produces for the same node.
///
/// Values are `1.0, 2.0, ...` rather than zeros so a wrongly-sized slot cannot
/// be mistaken for a correct one by accident.
fn executed(equation: &str, shapes: &[&[usize]]) -> Result<Vec<usize>, String> {
    let node = einsum_node(equation, shapes.len());
    let graph = Graph {
        nodes: vec![node],
        input_names: (0..shapes.len()).map(|i| format!("in{i}")).collect(),
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let session = Session::from_graph(graph, HashMap::new()).map_err(|e| e.to_string())?;

    let tensors: Vec<Tensor> = shapes
        .iter()
        .map(|s| {
            let numel: usize = s.iter().product();
            Tensor::new((0..numel).map(|v| v as f32 + 1.0).collect(), s.to_vec())
        })
        .collect();
    let inputs: HashMap<&str, Tensor> = (0..shapes.len())
        .map(|i| (INPUT_NAMES[i], tensors[i].clone()))
        .collect();

    let out = session.run(&inputs).map_err(|e| e.to_string())?;
    out.get("y")
        .map(|t| t.shape.clone())
        .ok_or_else(|| "no output 'y'".to_string())
}

/// `run()` keys are `&str`, so the names must outlive the map.
const INPUT_NAMES: [&str; 3] = ["in0", "in1", "in2"];

/// Assert both the published shape and the executed shape equal the numpy one.
#[track_caller]
fn agrees_with_numpy(equation: &str, shapes: &[&[usize]], numpy_shape: &[usize]) {
    assert_eq!(
        inferred(equation, shapes).as_deref(),
        Some(numpy_shape),
        "inference for '{equation}' over {shapes:?} disagrees with numpy",
    );
    assert_eq!(
        executed(equation, shapes).as_deref(),
        Ok(numpy_shape),
        "execution of '{equation}' over {shapes:?} disagrees with numpy",
    );
}

// ── The regression this file was written for ────────────────────────────────

/// numpy: `einsum('ij,ij->ij', zeros((2,1)), zeros((2,3))).shape == (2, 3)`.
///
/// The old first-writer-wins merge published `[2, 1]` — an under-sized output
/// slot for the tensor the executor goes on to write.
#[test]
fn a_size_one_label_loses_to_the_broadcast_extent() {
    agrees_with_numpy("ij,ij->ij", &[&[2, 1], &[2, 3]], &[2, 3]);
    // …and in the other operand order, which the old code got right by luck.
    agrees_with_numpy("ij,ij->ij", &[&[2, 3], &[2, 1]], &[2, 3]);
}

/// A batch axis of 1 on the left broadcasts against 6 on the right.
#[test]
fn a_broadcast_batch_axis_is_resolved_to_the_wider_extent() {
    agrees_with_numpy("bij,bjk->bik", &[&[1, 5, 7], &[6, 7, 3]], &[6, 5, 3]);
}

/// Two different non-1 extents for one label cannot be broadcast.  Inference
/// must **decline**, never publish one of them.
#[test]
fn irreconcilable_extents_decline_rather_than_guess() {
    assert_eq!(inferred("ij,ij->ij", &[&[2, 2], &[2, 3]]), None);
    assert!(
        executed("ij,ij->ij", &[&[2, 2], &[2, 3]]).is_err(),
        "the executor rejects it too, so declining is the honest answer",
    );
}

// ── Ellipsis ────────────────────────────────────────────────────────────────

/// The attention contraction, the reason ellipsis support matters at all.
#[test]
fn an_ellipsis_batch_contraction_is_inferred() {
    agrees_with_numpy(
        "...ij,...jk->...ik",
        &[&[2, 4, 5, 7], &[2, 4, 7, 3]],
        &[2, 4, 5, 3],
    );
}

/// An ellipsis binding **zero** axes is legal and infers the plain matmul shape.
#[test]
fn an_ellipsis_over_no_leading_axes_is_still_valid() {
    agrees_with_numpy("...ij,...jk->...ik", &[&[5, 7], &[7, 3]], &[5, 3]);
}

/// Operands may bind *different* numbers of ellipsis axes; they right-align and
/// broadcast, exactly as numpy's leading-dimension rule does.
#[test]
fn ellipsis_axes_right_align_across_operands() {
    agrees_with_numpy(
        "...ij,...jk->...ik",
        &[&[2, 4, 5, 7], &[7, 3]],
        &[2, 4, 5, 3],
    );
}

/// Size-1 ellipsis axes broadcast against each other in both directions.
#[test]
fn size_one_ellipsis_axes_broadcast_in_both_directions() {
    agrees_with_numpy("i...,i...->i...", &[&[3, 1, 4], &[3, 5, 1]], &[3, 5, 4]);
}

/// An ellipsis that survives into the output while every named label is summed
/// away.
#[test]
fn an_ellipsis_only_output_reduces_the_named_axis() {
    agrees_with_numpy("...i->...", &[&[2, 3, 4]], &[2, 3]);
}

/// numpy: *"output has more dimensions than subscripts given, but no '...'
/// ellipsis provided"*.  Inference must decline rather than publish `[5, 3]`.
#[test]
fn an_output_missing_its_ellipsis_declines() {
    assert_eq!(inferred("...ij,...jk->ik", &[&[2, 5, 7], &[7, 3]]), None);
    assert!(executed("...ij,...jk->ik", &[&[2, 5, 7], &[7, 3]]).is_err());
}

// ── Diagonals, transposes, implicit output ──────────────────────────────────

/// A label repeated **within one operand** is a diagonal, and numpy demands
/// exactly equal extents there — it does *not* broadcast `(1, 3)` to `ii`.
#[test]
fn a_diagonal_requires_equal_extents_and_does_not_broadcast() {
    agrees_with_numpy("ii->i", &[&[4, 4]], &[4]);
    assert_eq!(
        inferred("ii->i", &[&[1, 3]]),
        None,
        "numpy rejects this; inference must not invent [3] or [1]",
    );
}

/// A pure transpose keeps every extent and only reorders them.
#[test]
fn a_transpose_equation_reorders_the_extents() {
    agrees_with_numpy("ij->ji", &[&[2, 5]], &[5, 2]);
}

/// Implicit output mode (no `->`): the executor supports it, so inference must
/// too — the output is the singly-occurring labels in ASCII order.
#[test]
fn implicit_output_mode_is_inferred() {
    agrees_with_numpy("ij,jk", &[&[2, 3], &[3, 4]], &[2, 4]);
}

/// Implicit mode with an ellipsis: ellipsis axes come first, then the
/// singly-occurring named labels.
#[test]
fn implicit_output_mode_puts_ellipsis_axes_first() {
    agrees_with_numpy("...ij,...jk", &[&[2, 5, 7], &[7, 3]], &[2, 5, 3]);
}

/// A full reduction to a scalar: numpy gives shape `()`, i.e. rank 0.
#[test]
fn a_full_reduction_infers_rank_zero() {
    assert_eq!(
        inferred("ij->", &[&[2, 3]]),
        Some(Vec::new()),
        "numpy.einsum('ij->', ...).shape is ()",
    );
}

/// A repeated *output* label is an error in numpy, not a shape.
#[test]
fn a_repeated_output_label_declines() {
    assert_eq!(inferred("ij->ii", &[&[2, 3]]), None);
}

/// An output label that appears in no input subscript cannot be sized.
#[test]
fn an_unknown_output_label_declines() {
    assert_eq!(inferred("ij->ik", &[&[2, 3]]), None);
}

// ── Malformed input is declined, never inferred ─────────────────────────────

/// A subscript whose label count disagrees with its operand's rank.
#[test]
fn a_rank_mismatch_declines() {
    assert_eq!(inferred("ij,jk->ik", &[&[2, 3, 4], &[3, 4]]), None);
}

/// Two dots are not an ellipsis, and a digit is not a label.
#[test]
fn malformed_subscripts_decline() {
    assert_eq!(inferred("..ij,jk->ik", &[&[2, 3], &[3, 4]]), None);
    assert_eq!(inferred("i1,jk->ik", &[&[2, 3], &[3, 4]]), None);
    assert_eq!(
        inferred("...i...,j->ij", &[&[2, 3], &[4]]),
        None,
        "two ellipses in one subscript",
    );
}

/// The equation's operand count must match the node's input count.
#[test]
fn an_operand_count_mismatch_declines() {
    assert_eq!(inferred("ij,jk,kl->il", &[&[2, 3], &[3, 4]]), None);
}

/// Whitespace is insignificant — an exporter that emits `"ij, jk -> ik"` must
/// still be understood.
#[test]
fn whitespace_in_the_equation_is_ignored() {
    agrees_with_numpy(" ij , jk -> ik ", &[&[2, 3], &[3, 4]], &[2, 4]);
}
