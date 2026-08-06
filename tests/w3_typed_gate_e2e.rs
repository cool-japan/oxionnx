//! End-to-end proof that `Session::run_typed`'s `native_dtypes()` gate routes
//! an `I64` tensor through the *exact* integer path for the unary ops that
//! promise it in `native_dtypes()`, rather than silently taking the lossy f32
//! round-trip.
//!
//! `f32`'s mantissa has 24 significant bits, so every whole number through
//! `2^24` (16_777_216) round-trips through f32 exactly, and the first one
//! that cannot is `2^24 + 1` = 16_777_217 -- an f32 cast would silently turn
//! it into 16_777_216. `oxionnx_ops::registry::math_ops`'s
//! `unary_op_inplace_exact_int!` family (`Neg`, `Ceil`, `Floor`, `Round`,
//! `Sign`) and `nn_ops::AbsOp` (`Abs`) each declare `I64` in
//! `native_dtypes()` *and* back that promise with real `i64` arithmetic in
//! `execute_typed()` -- see `unary_op_inplace_exact_int!`'s module-level doc
//! comment (`oxionnx-ops/src/registry/math_ops/macros.rs`) for the fix this
//! file is the end-to-end counterpart of.
//!
//! `macros.rs`'s own unit tests already prove the op-level arithmetic is
//! exact by calling `execute_typed()` directly on each op; what they
//! explicitly do *not* cover (see that file's test module doc comment) is
//! `Session::run_typed`'s gate itself -- the `all_native` check in
//! `src/session/run/typed.rs` that decides whether a node's `execute_typed`
//! is reached at all, instead of the surgical f32-cast branch every operator
//! without native dispatch takes. This file closes that gap: it drives a real
//! `Session::run_typed` call end to end and checks the *value* that comes out
//! the other side, not just that `execute_typed()` in isolation is exact.

use oxionnx::{DType, Session, TensorStorage, TypedTensor};
use oxionnx_core::{Attributes, Graph, Node, OpKind};
use std::collections::HashMap;

/// `2^24 + 1` -- see the module doc above.
const ABOVE_F32_EXACT_RANGE: i64 = 16_777_217;

/// Build a single-node `x -op-> y` graph, run it through `Session::run_typed`
/// with an `I64` input holding `input`, and return the `y` output.
///
/// The graph deliberately declares no `output_infos` entry for `y`. Had one
/// been declared as `I64`, `run_internal_typed`'s post-run reconciliation
/// (`TypedTensor::from_f32_vec`) would convert an f32-fallback result *back*
/// to `I64` after the fact -- laundering a gate that silently fell back to
/// f32 into a result whose `dtype()` still reads `I64`, exactly the failure
/// mode this test exists to catch. Leaving `output_infos` empty means a gate
/// regression surfaces honestly as `TensorStorage::F32` carrying a rounded
/// value, not a falsely-relabeled `I64`.
fn run_single_op_i64(op: OpKind, input: i64) -> TypedTensor {
    let graph = Graph {
        name: String::new(),
        nodes: vec![Node {
            op,
            name: "n".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![],
    };
    let session = Session::from_graph(graph, HashMap::new()).expect("session creation");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x",
        TypedTensor::new(TensorStorage::I64(vec![input]), vec![1]),
    );

    let outputs = session.run_typed(&inputs).expect("run_typed");
    outputs.get("y").cloned().expect("output 'y' present")
}

/// The load-bearing assertion: the output must be genuinely `I64` storage
/// (not an f32-fallback result, even one later relabeled) holding exactly
/// `expected` -- not `expected` rounded through an f32 round-trip.
fn assert_exact_i64(op_name: &str, out: &TypedTensor, expected: i64) {
    match &out.storage {
        TensorStorage::I64(v) => {
            assert_eq!(
                v.as_slice(),
                &[expected],
                "{op_name}: native_dtypes gate did not preserve the exact i64 value \
                 (an f32 round-trip would have produced {}, not {expected})",
                (expected as f32) as i64
            );
        }
        other => panic!(
            "{op_name}: expected I64 storage (native_dtypes gate should route to \
             execute_typed's exact integer path), got {other:?} -- the gate fell back \
             to the lossy f32 path"
        ),
    }
    assert_eq!(
        out.dtype(),
        DType::I64,
        "{op_name}: TypedTensor::dtype() must agree with its own storage variant"
    );
}

#[test]
fn neg_run_typed_preserves_i64_above_two_pow_24() {
    let out = run_single_op_i64(OpKind::Neg, ABOVE_F32_EXACT_RANGE);
    assert_exact_i64("Neg", &out, -ABOVE_F32_EXACT_RANGE);
}

#[test]
fn ceil_run_typed_preserves_i64_above_two_pow_24() {
    let out = run_single_op_i64(OpKind::Ceil, ABOVE_F32_EXACT_RANGE);
    assert_exact_i64("Ceil", &out, ABOVE_F32_EXACT_RANGE);
}

#[test]
fn floor_run_typed_preserves_i64_above_two_pow_24() {
    let out = run_single_op_i64(OpKind::Floor, ABOVE_F32_EXACT_RANGE);
    assert_exact_i64("Floor", &out, ABOVE_F32_EXACT_RANGE);
}

#[test]
fn round_run_typed_preserves_i64_above_two_pow_24() {
    let out = run_single_op_i64(OpKind::Round, ABOVE_F32_EXACT_RANGE);
    assert_exact_i64("Round", &out, ABOVE_F32_EXACT_RANGE);
}

#[test]
fn sign_run_typed_preserves_i64_above_two_pow_24() {
    let out = run_single_op_i64(OpKind::Sign, ABOVE_F32_EXACT_RANGE);
    assert_exact_i64("Sign", &out, 1);
}

/// `AbsOp` (`oxionnx-ops/src/registry/nn_ops/activations.rs`) has its own
/// exact `I32`/`I64` arms in `execute_typed()` (via `wrapping_abs`, mirroring
/// `NegOp`'s `wrapping_neg` for the same `i32::MIN`/`i64::MIN`-has-no-positive-
/// representation reason), so `Session::run_typed`'s gate must route it
/// through that exact path here too, not just the `unary_op_inplace_exact_int!`
/// family above.
#[test]
fn abs_run_typed_preserves_i64_above_two_pow_24() {
    let out = run_single_op_i64(OpKind::Abs, ABOVE_F32_EXACT_RANGE);
    assert_exact_i64("Abs", &out, ABOVE_F32_EXACT_RANGE);
}
