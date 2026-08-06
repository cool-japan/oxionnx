//! End-to-end proof that a model's declared `ai.onnx` opset reaches its
//! operators (stitch-wave item S5, from Wave-1 findings a1-10 / a5-6 / a11-9).
//!
//! `ModelMetadata::opset_imports` was parsed and exposed but read by nothing in
//! the execution path, so `Softmax`/`LogSoftmax`/`Hardmax` ran the opset-13+
//! contract for every model regardless of what it declared. The per-operator
//! semantics are covered by `oxionnx-ops/tests/opset_softmax_family.rs`; what is
//! tested here is the *seam* those tests cannot reach — that a real
//! `Session::from_bytes` binds the parsed opset to the registry every execution
//! path builds its `OpContext` from.
//!
//! Reference values are NumPy's (float64 accumulation) for a rank-3 `[2,3,4]`
//! input, where the two regimes genuinely disagree:
//!
//! ```text
//! pre13  = softmax(x.reshape(2, 12), axis=1).reshape(2,3,4)
//! post13 = softmax(x, axis=1)
//! ```

use oxionnx::{ProviderKind, Session, SessionBuilder};
use oxionnx_core::Tensor;
use std::collections::HashMap;

// ── Minimal ModelProto encoder (mirrors tests/metadata_test.rs) ─────────────

fn encode_varint(mut val: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
    buf
}

fn encode_varint_field(field: u32, val: u64) -> Vec<u8> {
    let mut buf = encode_varint((field << 3) as u64); // wire type 0
    buf.extend(encode_varint(val));
    buf
}

fn encode_bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = encode_varint(((field << 3) | 2) as u64); // wire type 2
    buf.extend(encode_varint(data.len() as u64));
    buf.extend_from_slice(data);
    buf
}

/// `ValueInfoProto` for a float32 tensor with a fully static shape.
fn value_info(name: &str, shape: &[u64]) -> Vec<u8> {
    let mut shape_bytes = Vec::new();
    for dim in shape {
        let dim_msg = encode_varint_field(1, *dim); // Dimension.dim_value
        shape_bytes.extend(encode_bytes_field(1, &dim_msg)); // TensorShapeProto.dim
    }
    let mut tensor_type = encode_varint_field(1, 1); // elem_type = FLOAT
    tensor_type.extend(encode_bytes_field(2, &shape_bytes)); // shape
    let type_proto = encode_bytes_field(1, &tensor_type); // TypeProto.tensor_type

    let mut vi = encode_bytes_field(1, name.as_bytes()); // name
    vi.extend(encode_bytes_field(2, &type_proto)); // type
    vi
}

/// A `NodeProto`: `<output> = <op_type>(x)` with an optional `axis` attribute.
fn unary_node(op_type: &str, name: &str, output: &str, axis: Option<i64>) -> Vec<u8> {
    let mut node = encode_bytes_field(1, b"x"); // input
    node.extend(encode_bytes_field(2, output.as_bytes())); // output
    node.extend(encode_bytes_field(3, name.as_bytes())); // name
    node.extend(encode_bytes_field(4, op_type.as_bytes())); // op_type
    if let Some(value) = axis {
        let mut attr = encode_bytes_field(1, b"axis"); // AttributeProto.name
        attr.extend(encode_varint_field(3, value as u64)); // AttributeProto.i
        attr.extend(encode_varint_field(20, 2)); // AttributeProto.type = INT
        node.extend(encode_bytes_field(5, &attr)); // NodeProto.attribute
    }
    node
}

/// A single-node `GraphProto`: `y = <op_type>(x)` with an optional `axis`.
fn one_node_graph(op_type: &str, axis: Option<i64>, shape: &[u64]) -> Vec<u8> {
    let mut graph = encode_bytes_field(1, &unary_node(op_type, "the_node", "y", axis));
    graph.extend(encode_bytes_field(2, b"opset_probe")); // name
    graph.extend(encode_bytes_field(11, &value_info("x", shape))); // input
    graph.extend(encode_bytes_field(12, &value_info("y", shape))); // output
    graph
}

/// A `GraphProto` with **two independent** `Softmax` nodes over the same input.
///
/// Both sit at topological depth 0, so `run_parallel_inner` groups them into one
/// multi-node level and executes them through its `par_iter` — the one
/// `OpContext` construction site that a single-node graph never reaches (a
/// one-node level is routed through `dispatch_serially` instead).
fn two_softmax_graph(axis: i64, shape: &[u64]) -> Vec<u8> {
    let mut graph = encode_bytes_field(1, &unary_node("Softmax", "left", "y1", Some(axis)));
    graph.extend(encode_bytes_field(
        1,
        &unary_node("Softmax", "right", "y2", Some(axis)),
    ));
    graph.extend(encode_bytes_field(2, b"opset_probe_parallel"));
    graph.extend(encode_bytes_field(11, &value_info("x", shape)));
    graph.extend(encode_bytes_field(12, &value_info("y1", shape)));
    graph.extend(encode_bytes_field(12, &value_info("y2", shape)));
    graph
}

/// A `ModelProto` declaring exactly one opset import.
fn model_with_opset(domain: &str, version: u64, graph: &[u8]) -> Vec<u8> {
    let mut model = encode_varint_field(1, 8); // ir_version
    model.extend(encode_bytes_field(7, graph)); // graph

    let mut opset = Vec::new();
    if !domain.is_empty() {
        opset.extend(encode_bytes_field(1, domain.as_bytes())); // OperatorSetId.domain
    }
    opset.extend(encode_varint_field(2, version)); // OperatorSetId.version
    model.extend(encode_bytes_field(8, &opset)); // opset_import
    model
}

/// A `ModelProto` with **no** `opset_import` at all.
fn model_without_opset(graph: &[u8]) -> Vec<u8> {
    let mut model = encode_varint_field(1, 8);
    model.extend(encode_bytes_field(7, graph));
    model
}

// ── Fixtures ────────────────────────────────────────────────────────────────

const X: [f32; 24] = [
    0.1, -0.2, 0.3, 0.4, //
    1.0, 0.5, -0.5, 0.0, //
    -1.0, 2.0, 0.25, -0.75, //
    0.7, 0.7, 0.7, 0.7, //
    -2.0, 1.5, 0.0, 0.5, //
    3.0, -3.0, 1.0, -1.0,
];
const DIMS: [u64; 3] = [2, 3, 4];

/// `softmax(x.reshape(2,12), axis=1).reshape(2,3,4)` — the opset ≤ 12 contract.
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

/// `softmax(x, axis=1)` — the opset ≥ 13 contract at the same explicit axis.
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

fn input() -> Tensor {
    Tensor::new(X.to_vec(), vec![2, 3, 4])
}

fn run(session: &Session) -> Tensor {
    let x = input();
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert("x", x);
    let outputs = session.run(&inputs).expect("inference must succeed");
    outputs.get("y").cloned().expect("output 'y'")
}

fn assert_close(got: &Tensor, want: &[f32], label: &str) {
    assert_eq!(got.shape, vec![2, 3, 4], "{label}: shape is preserved");
    assert_eq!(got.data.len(), want.len(), "{label}: element count");
    for (i, (&g, &w)) in got.data.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-6, "{label}[{i}]: got {g}, expected {w}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════

/// The whole point: two byte-identical graphs that differ only in the declared
/// opset must compute different results.
#[test]
fn declared_opset_selects_the_softmax_contract() {
    let graph = one_node_graph("Softmax", Some(1), &DIMS);

    let legacy =
        Session::from_bytes(&model_with_opset("", 11, &graph)).expect("opset-11 model must load");
    assert_eq!(
        legacy.metadata().opset_imports,
        vec![(String::new(), 11)],
        "the import list is what the run path reads"
    );
    assert_close(&run(&legacy), &PRE13_SOFTMAX_AXIS1, "opset 11");

    let current =
        Session::from_bytes(&model_with_opset("", 13, &graph)).expect("opset-13 model must load");
    assert_close(&run(&current), &POST13_SOFTMAX_AXIS1, "opset 13");
}

/// The default domain is spelled `""` by every real exporter and `"ai.onnx"` by
/// the spec; both must resolve.
#[test]
fn ai_onnx_domain_spelling_is_recognised() {
    let graph = one_node_graph("Softmax", Some(1), &DIMS);
    let session =
        Session::from_bytes(&model_with_opset("ai.onnx", 11, &graph)).expect("model must load");
    assert_close(
        &run(&session),
        &PRE13_SOFTMAX_AXIS1,
        "domain 'ai.onnx', opset 11",
    );
}

/// A non-default domain says nothing about `ai.onnx`, so the engine falls back to
/// its default — it must not be read as "opset 2, therefore legacy".
#[test]
fn foreign_domain_opset_does_not_select_the_legacy_contract() {
    let graph = one_node_graph("Softmax", Some(1), &DIMS);
    let session =
        Session::from_bytes(&model_with_opset("ai.onnx.ml", 2, &graph)).expect("model must load");
    assert_close(
        &run(&session),
        &POST13_SOFTMAX_AXIS1,
        "only ai.onnx imports bind the opset",
    );
}

/// A model that declares no opset at all keeps current semantics.
#[test]
fn missing_opset_import_keeps_current_semantics() {
    let graph = one_node_graph("Softmax", Some(1), &DIMS);
    let session = Session::from_bytes(&model_without_opset(&graph)).expect("model must load");
    assert!(session.metadata().opset_imports.is_empty());
    assert_close(&run(&session), &POST13_SOFTMAX_AXIS1, "no opset_import");
}

/// The opset must reach the operator on **every** execution path, not just the
/// default sequential one: parallel execution builds its own `OpContext`s, and
/// the memory-pool path routes through `execute_into_slots` instead of
/// `execute`.
///
/// Every session here is pinned to the CPU. These assertions are about *CPU*
/// opset semantics, and with accelerator features compiled in the default
/// placement would leave the choice to a payload-size threshold — which would
/// make the test quietly feature- and hardware-dependent. (The accelerator
/// backends still hard-code the opset-13 axis; see the deferred notes.)
#[test]
fn every_execution_path_sees_the_opset() {
    let graph = one_node_graph("Softmax", Some(1), &DIMS);
    let bytes = model_with_opset("", 11, &graph);

    let parallel = SessionBuilder::new()
        .with_parallel_execution(true)
        .with_provider_kinds([ProviderKind::Cpu])
        .load_from_bytes(&bytes)
        .expect("parallel session must load");
    assert_close(&run(&parallel), &PRE13_SOFTMAX_AXIS1, "parallel path");

    // A *multi-node* level is what forces `run_parallel_inner`'s `par_iter`
    // branch and its own `OpContext` literal; a one-node level is dispatched
    // serially through `dispatch_node` instead.
    let par_bytes = model_with_opset("", 11, &two_softmax_graph(1, &DIMS));
    let par_multi = SessionBuilder::new()
        .with_parallel_execution(true)
        .with_provider_kinds([ProviderKind::Cpu])
        .load_from_bytes(&par_bytes)
        .expect("multi-node parallel session must load");
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert("x", input());
    let outputs = par_multi.run(&inputs).expect("parallel inference");
    for name in ["y1", "y2"] {
        let got = outputs.get(name).expect("both outputs are produced");
        assert_close(got, &PRE13_SOFTMAX_AXIS1, "rayon par_iter path");
    }

    // The memory pool resolves output shapes up front, which is what unlocks the
    // slot-writing dispatch inside `Session::dispatch_node`.
    let pooled = SessionBuilder::new()
        .with_memory_pool(true)
        .with_provider_kinds([ProviderKind::Cpu])
        .load_from_bytes(&bytes)
        .expect("pooled session must load");
    assert_close(&run(&pooled), &PRE13_SOFTMAX_AXIS1, "slot-writing path");

    let unoptimized = SessionBuilder::new()
        .with_optimization_level(oxionnx::OptLevel::None)
        .with_provider_kinds([ProviderKind::Cpu])
        .load_from_bytes(&bytes)
        .expect("unoptimized session must load");
    assert_close(&run(&unoptimized), &PRE13_SOFTMAX_AXIS1, "no optimization");
}

/// `run_typed` builds its contexts independently of `run`; the same model must
/// produce the same answer through it.
#[test]
fn typed_execution_path_sees_the_opset() {
    use oxionnx_core::{TensorStorage, TypedTensor};

    let graph = one_node_graph("Softmax", Some(1), &DIMS);
    let session = Session::from_bytes(&model_with_opset("", 11, &graph)).expect("must load");

    let mut inputs: HashMap<&str, TypedTensor> = HashMap::new();
    inputs.insert(
        "x",
        TypedTensor::new(TensorStorage::F32(X.to_vec()), vec![2, 3, 4]),
    );
    let outputs = session.run_typed(&inputs).expect("typed inference");
    let y = outputs.get("y").expect("output 'y'");
    let got = Tensor::new(y.storage.to_f32_vec(), y.shape.clone());
    assert_close(&got, &PRE13_SOFTMAX_AXIS1, "typed path, opset 11");
}

/// LogSoftmax and Hardmax are on the same boundary and must move with it.
#[test]
fn log_softmax_and_hardmax_follow_the_declared_opset() {
    let log_graph = one_node_graph("LogSoftmax", Some(1), &DIMS);
    let legacy = Session::from_bytes(&model_with_opset("", 11, &log_graph)).expect("must load");
    let want: Vec<f32> = PRE13_SOFTMAX_AXIS1.iter().map(|v| v.ln()).collect();
    assert_close(&run(&legacy), &want, "LogSoftmax at opset 11");

    // Hardmax's contract change is structural: the number of ones is the number
    // of independent reductions — 2 coerced rows at opset 11, 2·4 slices at 13.
    let hard_graph = one_node_graph("Hardmax", Some(1), &DIMS);
    let hard_legacy = Session::from_bytes(&model_with_opset("", 11, &hard_graph)).expect("load");
    assert_eq!(
        run(&hard_legacy).data.iter().sum::<f32>(),
        2.0,
        "Hardmax at opset 11 elects one winner per coerced row"
    );
    let hard_current = Session::from_bytes(&model_with_opset("", 13, &hard_graph)).expect("load");
    assert_eq!(
        run(&hard_current).data.iter().sum::<f32>(),
        8.0,
        "Hardmax at opset 13 elects one winner per axis-1 slice"
    );
}
