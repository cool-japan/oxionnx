#![cfg(feature = "mmap")]
//! Wave-3 (T7-tests-engine): `SessionBuilder::load_mmap` end-to-end integration.
//!
//! Before this file, `load_mmap` had no test that ever called `Session::run`
//! on the result: `oxionnx-proto/src/mmap_loader.rs`'s unit tests stop at
//! `MmapModel::open`/`into_parts` (never build a `Session`), and no test
//! anywhere in the root crate calls `SessionBuilder::load_mmap` at all. These
//! tests hand-encode a small but real ONNX protobuf model (mirroring the wire
//! format documented in `oxionnx-proto/src/parser.rs`'s
//! `parse_value_info_proto` doc comment, and the encoding helpers in
//! `mmap_loader.rs`'s own `#[cfg(test)]` module), write it to a temp file via
//! `std::env::temp_dir()`, and drive it through `Session::run`.
//!
//! Each test compares against **two** independent baselines, not just against
//! itself: `SessionBuilder::load_from_bytes` on the identical bytes (so a
//! divergence between the two loading paths is caught), and a hand-computed
//! expected value (so two identically-broken loading paths agreeing with each
//! other cannot pass silently).
//!
//! ## Known gap, not covered here (see the deferred report)
//!
//! `SessionBuilder::load_mmap` (src/session/builder.rs) passes
//! `ModelMetadata::default()` instead of the model's real parsed metadata, so
//! `session.metadata()` after `load_mmap` does NOT match the metadata
//! `load_from_bytes` reports for the same bytes (producer_name, ir_version,
//! opset_imports are all silently lost). This file does not assert on
//! `session.metadata()` in either direction: asserting equality would fail
//! against today's real behavior, and asserting today's (wrong) default would
//! pin a bug this test-only lane cannot fix and that another lane may already
//! be fixing. The gap and its exact fix location are reported separately, not
//! encoded as a test expectation here.

use oxionnx::{Session, SessionBuilder, Tensor};
use std::collections::HashMap;

// ── minimal ONNX protobuf encoder ───────────────────────────────────────────
//
// Mirrors the wire-format helpers in oxionnx-proto/src/mmap_loader.rs's own
// #[cfg(test)] module (encode_varint / encode_varint_field / encode_bytes_field
// / build_minimal_model), extended with a NodeProto and full ValueInfoProto
// (dtype + shape) so the model actually computes something instead of holding
// a bare initializer.

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
    let tag = field << 3;
    let mut buf = encode_varint(tag as u64);
    buf.extend(encode_varint(val));
    buf
}

fn encode_bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
    let tag = (field << 3) | 2;
    let mut buf = encode_varint(tag as u64);
    buf.extend(encode_varint(data.len() as u64));
    buf.extend(data);
    buf
}

/// `TensorProto` with an explicit (possibly multi-dimensional) shape.
fn tensor_proto(name: &str, floats: &[f32], shape: &[usize]) -> Vec<u8> {
    let mut t = Vec::new();
    let mut dims_packed = Vec::new();
    for &d in shape {
        dims_packed.extend(encode_varint(d as u64));
    }
    t.extend(encode_bytes_field(1, &dims_packed)); // dims (packed repeated int64)
    t.extend(encode_varint_field(2, 1)); // data_type = 1 (FLOAT)
    t.extend(encode_bytes_field(8, name.as_bytes())); // name
    let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    t.extend(encode_bytes_field(9, &raw)); // raw_data
    t
}

/// `ValueInfoProto { name, type: TypeProto { tensor_type: { elem_type, shape } } }`,
/// per the field layout documented in `oxionnx-proto/src/parser.rs::parse_value_info_proto`.
fn value_info(name: &str, shape: &[usize]) -> Vec<u8> {
    let mut tensor_type = encode_varint_field(1, 1); // elem_type = FLOAT
    let mut shape_bytes = Vec::new();
    for &d in shape {
        let dimension = encode_varint_field(1, d as u64); // Dimension.dim_value
        shape_bytes.extend(encode_bytes_field(1, &dimension));
    }
    tensor_type.extend(encode_bytes_field(2, &shape_bytes));
    let type_proto = encode_bytes_field(1, &tensor_type);
    let mut vi = encode_bytes_field(1, name.as_bytes());
    vi.extend(encode_bytes_field(2, &type_proto));
    vi
}

fn node_proto(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
    let mut n = Vec::new();
    for i in inputs {
        n.extend(encode_bytes_field(1, i.as_bytes()));
    }
    for o in outputs {
        n.extend(encode_bytes_field(2, o.as_bytes()));
    }
    n.extend(encode_bytes_field(3, name.as_bytes())); // name
    n.extend(encode_bytes_field(4, op_type.as_bytes())); // op_type
    n
}

/// Build a minimal-but-real ONNX model computing `y = x + w`, where `w` is a
/// graph initializer of the given shape and `x` is a same-shaped graph input.
fn build_add_model(w: &[f32], shape: &[usize]) -> Vec<u8> {
    let mut graph = Vec::new();
    graph.extend(encode_bytes_field(2, b"w3_mmap_test_graph")); // GraphProto.name
    graph.extend(encode_bytes_field(
        1,
        &node_proto("Add", "add1", &["x", "w"], &["y"]),
    )); // node
    graph.extend(encode_bytes_field(5, &tensor_proto("w", w, shape))); // initializer
    graph.extend(encode_bytes_field(11, &value_info("x", shape))); // input
    graph.extend(encode_bytes_field(12, &value_info("y", shape))); // output

    let opset = encode_varint_field(2, 13); // OpsetImport.version = 13
    let mut model = encode_varint_field(1, 7); // ModelProto.ir_version
    model.extend(encode_bytes_field(2, b"oxionnx-w3-test")); // producer_name
    model.extend(encode_bytes_field(8, &opset)); // opset_import
    model.extend(encode_bytes_field(7, &graph)); // graph
    model
}

/// Write bytes to a fresh file under `std::env::temp_dir()` and return its path.
fn write_temp_onnx(name: &str, data: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("oxionnx_w3_mmap_tests");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(name);
    std::fs::write(&path, data).expect("write temp file");
    path
}

fn run_add(session: &Session, x: Vec<f32>, shape: Vec<usize>) -> Tensor {
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(x, shape));
    session
        .run(&inputs)
        .expect("session.run should succeed")
        .remove("y")
        .expect("output 'y' should be present")
}

// ── tests ────────────────────────────────────────────────────────────────

/// The central claim: a model loaded via `load_mmap` from a real temp file
/// runs correctly, agrees bit-for-bit with `load_from_bytes` on the identical
/// bytes, AND matches a value computed independently of either code path.
#[test]
fn load_mmap_and_load_from_bytes_agree_and_match_hand_computed_values() {
    let w = vec![10.0f32, 20.0, 30.0, 40.0];
    let shape = vec![4usize];
    let model_bytes = build_add_model(&w, &shape);
    let path = write_temp_onnx("add_model.onnx", &model_bytes);

    let session_mmap = SessionBuilder::new()
        .load_mmap(&path)
        .expect("load_mmap should succeed on a well-formed model");
    let session_bytes = SessionBuilder::new()
        .load_from_bytes(&model_bytes)
        .expect("load_from_bytes should succeed on the same bytes");

    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let out_mmap = run_add(&session_mmap, x.clone(), shape.clone());
    let out_bytes = run_add(&session_bytes, x, shape.clone());

    // Hand-computed: x + w = [1+10, 2+20, 3+30, 4+40].
    let expected = vec![11.0f32, 22.0, 33.0, 44.0];

    assert_eq!(out_mmap.shape, shape);
    assert_eq!(out_bytes.shape, shape);
    assert_eq!(
        out_mmap.data, expected,
        "load_mmap result vs. hand-computed"
    );
    assert_eq!(
        out_bytes.data, expected,
        "load_from_bytes result vs. hand-computed"
    );
    assert_eq!(
        out_mmap.data, out_bytes.data,
        "load_mmap and load_from_bytes must agree bit-for-bit on identical bytes"
    );

    let _ = std::fs::remove_file(&path);
}

/// The initializer ("weight") tensor itself must survive the mmap path with
/// its shape intact, not just its flattened values — exercised here with a
/// genuinely 2-D weight (mmap's raison d'etre is large multi-dimensional
/// weight tensors, so a 1-D-only proof would under-cover the claim).
#[test]
fn load_mmap_preserves_multi_dimensional_weight_shape_and_values() {
    let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2, 3]
    let shape = vec![2usize, 3usize];
    let model_bytes = build_add_model(&w, &shape);
    let path = write_temp_onnx("add_model_2d.onnx", &model_bytes);

    let session = SessionBuilder::new()
        .load_mmap(&path)
        .expect("load_mmap should succeed");

    assert_eq!(
        session.weights().get("w").map(|t| t.shape.clone()),
        Some(shape.clone()),
        "the mmap-loaded weight must keep its declared 2-D shape"
    );
    assert_eq!(
        session.weights().get("w").map(|t| t.data.clone()),
        Some(w.clone()),
        "the mmap-loaded weight's values must be exact (small integers, exact in f32)"
    );

    let x = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let out = run_add(&session, x, shape.clone());
    let expected = vec![11.0f32, 22.0, 33.0, 44.0, 55.0, 66.0];
    assert_eq!(out.shape, shape);
    assert_eq!(out.data, expected);

    let _ = std::fs::remove_file(&path);
}

/// A session built via `load_mmap` is not a one-shot object: the file (and any
/// mapping of it) can be safely gone from the filesystem by the time `run` is
/// called a second or third time, because parsing fully materializes weights
/// into owned `Tensor` storage — it does not keep borrowing the mapped pages.
#[test]
fn load_mmap_session_runs_correctly_multiple_times_after_the_file_is_removed() {
    let w = vec![100.0f32, 200.0];
    let shape = vec![2usize];
    let model_bytes = build_add_model(&w, &shape);
    let path = write_temp_onnx("add_model_multi_run.onnx", &model_bytes);

    let session = SessionBuilder::new()
        .load_mmap(&path)
        .expect("load_mmap should succeed");

    // Remove the backing file/mapping before any run: if weight loading were
    // not fully materialized, this would be the moment it breaks.
    std::fs::remove_file(&path).expect("remove temp file");

    for i in 0..3 {
        let factor = i as f32;
        let out = run_add(&session, vec![factor, factor * 2.0], shape.clone());
        assert_eq!(
            out.data,
            vec![factor + 100.0, factor * 2.0 + 200.0],
            "run {i} produced the wrong result after the source file was removed"
        );
    }
}

/// `load_mmap` on a path that does not exist must return a typed `OnnxError`,
/// never panic — the `SessionBuilder`-level equivalent of
/// `mmap_loader.rs`'s `test_mmap_open_nonexistent`, which only exercises the
/// lower-level `MmapModel::open` directly and never goes through the public
/// session-building API a real caller uses.
#[test]
fn load_mmap_reports_a_typed_error_for_a_missing_file() {
    let missing = std::env::temp_dir().join("oxionnx_w3_mmap_tests_definitely_absent.onnx");
    let _ = std::fs::remove_file(&missing); // best-effort: ensure it is absent

    let result = SessionBuilder::new().load_mmap(&missing);
    assert!(
        result.is_err(),
        "load_mmap on a nonexistent path must return Err, not panic or succeed"
    );
}
