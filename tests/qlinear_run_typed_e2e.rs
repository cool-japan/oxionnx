//! End-to-end proof that `QLinearMatMulOp`'s `execute_typed`
//! (T2-quant-stitch, item 2) is actually *reached* through
//! `Session::run_typed`, not merely correct when called directly.
//!
//! `oxionnx-ops/tests/qlinear_native_dtype_test.rs` builds a `TypedOpContext`
//! by hand and calls `execute_typed` directly — that proves the operator
//! does the right thing when invoked. It does not prove the session ever
//! invokes it: `Session::run_typed`'s `all_native` gate
//! (`src/session/run/typed.rs`) only dispatches through `execute_typed` when
//! every input's dtype is in the operator's `native_dtypes()`, and every
//! test in that file bypasses the gate entirely. This file closes that gap
//! by going through `Session::run_typed` for real.
//!
//! Every input here is a *graph input* (not a model initializer): a
//! `TypedTensor` fed to `run_typed` carries its caller-declared dtype into
//! the run as `InputSource::Intermediate`. A model initializer would instead
//! resolve through `Session::weights`, which is f32-only (see
//! `SatRange::for_dtype`'s doc comment in `oxionnx-ops`) — so this is also
//! the one shape of graph the exact dtype-aware saturation fix can actually
//! help today; a `y_zero_point` baked into the model as an initializer still
//! falls back to the union range, unchanged by this fix.

use std::collections::HashMap;

use oxionnx::{
    Attributes, DType, Graph, Node, OpKind, OptLevel, Session, TensorStorage, TypedTensor,
};

fn node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: "op0".to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

fn f32_t(data: Vec<f32>, shape: Vec<usize>) -> TypedTensor {
    TypedTensor::new(TensorStorage::F32(data), shape)
}

/// Build and run the shared fixture (same numbers as
/// `oxionnx-ops/tests/qlinear_native_dtype_test.rs` and
/// `tests/w2_quantized_ops_e2e.rs::qlinear_matmul_ambiguous_zero_points_use_the_union_range`):
/// `a=[[10,0],[0,10]]`, `b=[[10,-15],[0,0]]`, every scale `1.0`, every zero
/// point `0` except `y_zero_point` (supplied by the caller). `acc = a @ b =
/// [[100,-150],[0,0]]`, so pre-saturation values are `acc + y_zero_point =
/// [200,-50,100,100]`.
fn run_via_session(y_zero_point: TypedTensor) -> TypedTensor {
    let input_names = ["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"];
    let graph = Graph {
        nodes: vec![node(OpKind::QLinearMatMul, &input_names, &["y"])],
        input_names: input_names.iter().map(|s| (*s).to_string()).collect(),
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build session");

    let a = TypedTensor::new(TensorStorage::U8(vec![10, 0, 0, 10]), vec![2, 2]);
    let b = TypedTensor::new(TensorStorage::I8(vec![10, -15, 0, 0]), vec![2, 2]);

    let feed: HashMap<&str, TypedTensor> = [
        ("a", a),
        ("as", f32_t(vec![1.0], vec![1])),
        ("azp", f32_t(vec![0.0], vec![1])),
        ("b", b),
        ("bs", f32_t(vec![1.0], vec![1])),
        ("bzp", f32_t(vec![0.0], vec![1])),
        ("ys", f32_t(vec![1.0], vec![1])),
        ("yzp", y_zero_point),
    ]
    .into_iter()
    .collect();

    let mut outputs = session.run_typed(&feed).expect("run_typed");
    outputs.remove("y").expect("graph must produce output 'y'")
}

/// `y_zero_point` fed as a genuinely typed `U8` graph input must resolve the
/// saturation range exactly: `-50` clips to `0`, `200` does not (it is
/// inside `[0,255]`). If `Session::run_typed`'s `all_native` gate failed to
/// open for `QLinearMatMul` (e.g. a `native_dtypes()` regression), this
/// would silently fall back to the untyped f32 path and produce the union
/// row `[200, -50, 100, 100]` instead — the assertion below distinguishes
/// the two.
#[test]
fn qlinear_matmul_run_typed_resolves_uint8_exactly_not_the_union() {
    let y_zp = TypedTensor::new(TensorStorage::U8(vec![100]), vec![1]);
    let y = run_via_session(y_zp);

    // No `output_infos` on this hand-built `Graph`, so the session's output
    // dtype reconciliation is a no-op and the result stays F32-tagged; only
    // the *values* prove which saturation range was used.
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(
        y.storage.to_f32_vec(),
        vec![200.0, 0.0, 100.0, 100.0],
        "uint8 row expected — got the union row, meaning the all_native gate \
         did not dispatch to QLinearMatMulOp::execute_typed"
    );
}

/// `y_zero_point` fed as a typed `I8` graph input resolves the other way:
/// `200` clips to `127`, `-50` does not.
#[test]
fn qlinear_matmul_run_typed_resolves_int8_exactly_not_the_union() {
    let y_zp = TypedTensor::new(TensorStorage::I8(vec![100]), vec![1]);
    let y = run_via_session(y_zp);

    assert_eq!(
        y.storage.to_f32_vec(),
        vec![127.0, -50.0, 100.0, 100.0],
        "int8 row expected — got the union row, meaning the all_native gate \
         did not dispatch to QLinearMatMulOp::execute_typed"
    );
}

/// `y_zero_point` fed as `F32` — what a model initializer always presents as
/// (`Session::weights` is f32-only) — cannot disambiguate, so `run_typed`
/// must fall back to the same union range the untyped `Session::run` path
/// already pins in `w2_quantized_ops_e2e.rs`. This is not a bug: it is the
/// documented boundary of what this fix reaches (see this file's module
/// doc comment).
#[test]
fn qlinear_matmul_run_typed_f32_y_zero_point_still_falls_back_to_union() {
    let y_zp = f32_t(vec![100.0], vec![1]);
    let y = run_via_session(y_zp);

    assert_eq!(y.storage.to_f32_vec(), vec![200.0, -50.0, 100.0, 100.0]);
}
