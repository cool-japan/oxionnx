//! Integration tests for Phase A metadata features:
//!   - A.1/A.2: producer_name, producer_version, domain, model_version, doc_string, metadata_props
//!   - A.3/A.4: dim_param parsing and TensorInfo::symbolic_shape()
//!   - A.5: Session::metadata()

use oxionnx::Session;
use oxionnx_core::graph::Dim;

// ─────────────────────────────────────────────────────────────────────────────
// Proto-encoding helpers (mirrors the helpers in parser.rs unit tests)
// ─────────────────────────────────────────────────────────────────────────────

fn encode_varint(mut val: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
    buf
}

fn encode_varint_field(field: u32, val: u64) -> Vec<u8> {
    let tag = field << 3; // wire type 0
    let mut buf = encode_varint(tag as u64);
    buf.extend(encode_varint(val));
    buf
}

fn encode_bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
    let tag = (field << 3) | 2; // wire type 2 = length-delimited
    let mut buf = encode_varint(tag as u64);
    buf.extend(encode_varint(data.len() as u64));
    buf.extend_from_slice(data);
    buf
}

/// Build a minimal `GraphProto` with a single named input that has the given
/// shape encoded as `ValueInfoProto` bytes.
///
/// `dims`: each element is (static_value: Option<u64>, param_name: Option<&str>).
fn build_graph_with_input(
    input_name: &str,
    elem_type: u32,
    dims: &[(Option<u64>, Option<&str>)],
) -> Vec<u8> {
    // Build Dimension messages
    let mut shape_bytes = Vec::new();
    for (dim_val, dim_param) in dims {
        let mut dim_msg = Vec::new();
        if let Some(v) = dim_val {
            dim_msg.extend(encode_varint_field(1, *v)); // field 1 = dim_value
        }
        if let Some(p) = dim_param {
            dim_msg.extend(encode_bytes_field(2, p.as_bytes())); // field 2 = dim_param
        }
        shape_bytes.extend(encode_bytes_field(1, &dim_msg)); // TensorShapeProto field 1 = dim
    }

    // TensorTypeProto: field 1 = elem_type, field 2 = shape
    let mut tensor_type = encode_varint_field(1, elem_type as u64);
    tensor_type.extend(encode_bytes_field(2, &shape_bytes));

    // TypeProto: field 1 = tensor_type
    let type_proto = encode_bytes_field(1, &tensor_type);

    // ValueInfoProto: field 1 = name, field 2 = type
    let mut vi_bytes = encode_bytes_field(1, input_name.as_bytes());
    vi_bytes.extend(encode_bytes_field(2, &type_proto));

    // GraphProto: field 11 = input (ValueInfoProto)
    encode_bytes_field(11, &vi_bytes)
}

/// Build a minimal ModelProto bytes from:
/// - `fields`: arbitrary additional field bytes appended after ir_version
/// - `graph_bytes`: bytes of an embedded GraphProto (field 7)
fn build_model(ir_version: u64, extra_fields: Vec<u8>, graph_bytes: Option<Vec<u8>>) -> Vec<u8> {
    let mut model = encode_varint_field(1, ir_version); // field 1 = ir_version
    model.extend(extra_fields);
    if let Some(g) = graph_bytes {
        model.extend(encode_bytes_field(7, &g)); // field 7 = graph
    }
    // Add a default opset import (field 8): version=11
    let opset = encode_varint_field(2, 11);
    model.extend(encode_bytes_field(8, &opset));
    model
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: producer_name is parsed and accessible via Session::metadata()
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parse_producer_name() {
    // Build a ModelProto with:
    //   field 1 = ir_version = 8
    //   field 2 = producer_name = "test_framework"
    //   field 3 = producer_version = "1.2.3"
    let mut extra = encode_bytes_field(2, b"test_framework");
    extra.extend(encode_bytes_field(3, b"1.2.3"));

    let model_bytes = build_model(8, extra, None);

    let session = Session::from_bytes(&model_bytes).expect("should parse model");
    let meta = session.metadata();

    assert_eq!(meta.producer_name, "test_framework");
    assert_eq!(meta.producer_version, "1.2.3");
    assert_eq!(meta.ir_version, 8);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: dim_param is captured and symbolic_shape() returns Dim::Symbol
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dim_param_symbolic() {
    // Build a model with one input: shape = [batch_size (param), 768 (static)]
    let graph_bytes = build_graph_with_input(
        "input_ids",
        1, // float32
        &[
            (None, Some("batch_size")), // symbolic dim
            (Some(768), None),          // static dim
        ],
    );

    let model_bytes = build_model(7, Vec::new(), Some(graph_bytes));

    let session = Session::from_bytes(&model_bytes).expect("should parse model with dim_param");

    // The session input_info should have at least one entry with symbolic_shape
    let infos = session.input_info();
    assert!(!infos.is_empty(), "expected at least one input_info entry");

    // Find the input named "input_ids"
    let info = infos
        .iter()
        .find(|i| i.name == "input_ids")
        .expect("input_ids should be in input_infos");

    assert_eq!(info.shape.len(), 2, "expected 2 dimensions");

    let sym_shape = info.symbolic_shape();
    assert_eq!(sym_shape.len(), 2);

    // First dim: symbolic "batch_size"
    assert_eq!(
        sym_shape[0],
        Dim::Symbol("batch_size".to_string()),
        "first dim should be Dim::Symbol(batch_size)"
    );

    // Second dim: static 768
    assert_eq!(
        sym_shape[1],
        Dim::Static(768),
        "second dim should be Dim::Static(768)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: metadata_props are parsed into ModelMetadata::custom_metadata
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metadata_props() {
    // Build two StringStringEntryProto messages for metadata_props (field 14):
    //   "license" = "Apache-2.0"
    //   "framework" = "onnx-test"
    let mut entry1 = encode_bytes_field(1, b"license");
    entry1.extend(encode_bytes_field(2, b"Apache-2.0"));

    let mut entry2 = encode_bytes_field(1, b"framework");
    entry2.extend(encode_bytes_field(2, b"onnx-test"));

    let mut extra = encode_bytes_field(14, &entry1);
    extra.extend(encode_bytes_field(14, &entry2));

    let model_bytes = build_model(7, extra, None);

    let session =
        Session::from_bytes(&model_bytes).expect("should parse model with metadata_props");
    let meta = session.metadata();

    assert_eq!(
        meta.custom_metadata.get("license").map(String::as_str),
        Some("Apache-2.0"),
        "license metadata_prop should be Apache-2.0"
    );
    assert_eq!(
        meta.custom_metadata.get("framework").map(String::as_str),
        Some("onnx-test"),
        "framework metadata_prop should be onnx-test"
    );
}
