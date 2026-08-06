//! Wave-1 regression tests for the ONNX model/tensor loading path.
//!
//! Covers dtype-complete initializer decoding, shape/payload validation,
//! STRINGS attributes, and external-data sandboxing.

use std::path::{Path, PathBuf};

use oxionnx_proto::model::{load, load_with_path};
use oxionnx_proto::reader::{dtype_code, resolve_external_path, ReaderError};
use oxionnx_proto::types::{TensorProto, ValueInfoProto};

// ─────────────────────────────────────────────────────────────────
// protobuf encoding helpers
// ─────────────────────────────────────────────────────────────────

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
    let mut buf = encode_varint(u64::from(field << 3));
    buf.extend(encode_varint(val));
    buf
}

fn encode_bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = encode_varint(u64::from((field << 3) | 2));
    buf.extend(encode_varint(data.len() as u64));
    buf.extend(data);
    buf
}

/// Wrap `graph_bytes` into a ModelProto (ir_version=7, opset=13).
fn model_with_graph(graph_bytes: &[u8]) -> Vec<u8> {
    let opset = encode_varint_field(2, 13);
    let mut model_bytes = encode_varint_field(1, 7);
    model_bytes.extend(encode_bytes_field(8, &opset));
    model_bytes.extend(encode_bytes_field(7, graph_bytes));
    model_bytes
}

fn model_with_initializer(tensor_bytes: &[u8]) -> Vec<u8> {
    model_with_graph(&encode_bytes_field(5, tensor_bytes))
}

/// TensorProto carrying `external_data` entries; `data_location` is emitted
/// only when `with_data_location` is set.
fn external_tensor_proto(
    name: &str,
    dims: &[u64],
    data_type: u64,
    entries: &[(&str, &str)],
    with_data_location: bool,
) -> Vec<u8> {
    let mut tensor_bytes = Vec::new();
    let mut dims_packed = Vec::new();
    for d in dims {
        dims_packed.extend(encode_varint(*d));
    }
    tensor_bytes.extend(encode_bytes_field(1, &dims_packed));
    tensor_bytes.extend(encode_varint_field(2, data_type));
    tensor_bytes.extend(encode_bytes_field(8, name.as_bytes()));
    for (key, value) in entries {
        let mut entry = encode_bytes_field(1, key.as_bytes());
        entry.extend(encode_bytes_field(2, value.as_bytes()));
        tensor_bytes.extend(encode_bytes_field(13, &entry));
    }
    if with_data_location {
        tensor_bytes.extend(encode_varint_field(14, 1));
    }
    tensor_bytes
}

/// Fresh, empty scratch directory under the system temp dir.
fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxionnx_w1_proto_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn raw_tensor(name: &str, dims: &[i64], data_type: i32, raw: Vec<u8>) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type,
        raw_data: raw,
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────
// [a2-5] / [a5-1] every ONNX dtype decodes from raw_data
// ─────────────────────────────────────────────────────────────────

#[test]
fn raw_data_decodes_every_numeric_dtype() {
    // (dtype, raw bytes, expected f32 values)
    let cases: Vec<(i32, Vec<u8>, Vec<f32>)> = vec![
        (
            dtype_code::FLOAT32,
            vec![0, 0, 128, 63, 0, 0, 0, 192],
            vec![1.0, -2.0],
        ),
        // uint8 / int8: the quantized-model case that used to load as zeros.
        (
            dtype_code::UINT8,
            vec![0, 1, 127, 255],
            vec![0.0, 1.0, 127.0, 255.0],
        ),
        (
            dtype_code::INT8,
            vec![0x00, 0x01, 0x7F, 0x80, 0xFF],
            vec![0.0, 1.0, 127.0, -128.0, -1.0],
        ),
        (dtype_code::BOOL, vec![0, 1, 2], vec![0.0, 1.0, 1.0]),
        (
            dtype_code::INT16,
            vec![0, 128, 255, 127],
            vec![-32768.0, 32767.0],
        ),
        (dtype_code::UINT16, vec![0, 0, 255, 255], vec![0.0, 65535.0]),
        (
            dtype_code::INT32,
            vec![0, 0, 0, 128, 5, 0, 0, 0],
            vec![-2147483648.0, 5.0],
        ),
        (dtype_code::UINT32, vec![0, 94, 208, 178], vec![3e9]),
        (
            dtype_code::INT64,
            vec![
                255, 255, 255, 255, 255, 255, 255, 255, 64, 66, 15, 0, 0, 0, 0, 0,
            ],
            vec![-1.0, 1e6],
        ),
        (
            dtype_code::UINT64,
            vec![57, 48, 0, 0, 0, 0, 0, 0],
            vec![12345.0],
        ),
        (
            dtype_code::DOUBLE,
            vec![0, 0, 0, 0, 0, 0, 248, 63, 0, 0, 0, 0, 0, 0, 2, 192],
            vec![1.5, -2.25],
        ),
        // float16 1.0 = 0x3C00, -0.5 = 0xB800
        (
            dtype_code::FLOAT16,
            vec![0x00, 0x3C, 0x00, 0xB8],
            vec![1.0, -0.5],
        ),
        // bfloat16 1.0 = 0x3F80, -2.5 = 0xC020
        (
            dtype_code::BFLOAT16,
            vec![0x80, 0x3F, 0x20, 0xC0],
            vec![1.0, -2.5],
        ),
    ];

    for (dtype, raw, expected) in cases {
        let tp = raw_tensor("w", &[expected.len() as i64], dtype, raw);
        let tensor = tp
            .try_to_tensor()
            .unwrap_or_else(|e| panic!("dtype {dtype} must decode: {e}"));
        assert_eq!(tensor.shape, vec![expected.len()], "dtype {dtype} shape");
        assert_eq!(tensor.data, expected, "dtype {dtype} values");
        // The infallible shim must agree with the fallible API.
        assert_eq!(tp.to_tensor().data, expected, "dtype {dtype} via to_tensor");
    }
}

#[test]
fn raw_data_unsupported_dtype_is_an_error_not_zeros() {
    // STRING (8) has no f32 representation: it must not silently become zeros.
    let tp = raw_tensor("s", &[2], dtype_code::STRING, vec![1, 2, 3, 4]);
    let err = tp.try_to_tensor().expect_err("STRING must not decode");
    assert!(
        matches!(err, ReaderError::UnsupportedDtype { dtype: 8, .. }),
        "got {err:?}"
    );
    // …and the same for a dtype code that does not exist at all.
    let tp = raw_tensor("f8", &[1], 17, vec![0]);
    assert!(tp.try_to_tensor().is_err());
}

#[test]
fn ragged_raw_data_is_rejected() {
    // 6 bytes cannot be a whole number of float32 elements.
    let tp = raw_tensor("w", &[2], dtype_code::FLOAT32, vec![0; 6]);
    let err = tp.try_to_tensor().expect_err("ragged payload must fail");
    assert!(
        matches!(err, ReaderError::RaggedRawData { elem_size: 4, .. }),
        "got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────
// [a5-1] typed repeated fields are decoded per dtype, not blindly cast
// ─────────────────────────────────────────────────────────────────

#[test]
fn int32_data_is_reinterpreted_per_dtype() {
    // FLOAT16 stored bit-wise in int32_data: 0x3C00 is 1.0, not 15360.0.
    let tp = TensorProto {
        name: "half".to_string(),
        dims: vec![3],
        data_type: dtype_code::FLOAT16,
        int32_data: vec![0x3C00, 0xB800, 0x3E00],
        ..Default::default()
    };
    assert_eq!(
        tp.try_to_tensor().expect("f16 typed field").data,
        vec![1.0, -0.5, 1.5]
    );

    // BFLOAT16 0x3F80 is 1.0, 0xC020 is -2.5.
    let tp = TensorProto {
        name: "bhalf".to_string(),
        dims: vec![2],
        data_type: dtype_code::BFLOAT16,
        int32_data: vec![0x3F80, 0xC020],
        ..Default::default()
    };
    assert_eq!(
        tp.try_to_tensor().expect("bf16 typed field").data,
        vec![1.0, -2.5]
    );

    // INT8 / UINT8 / BOOL sub-word packing.
    let tp = TensorProto {
        name: "i8".to_string(),
        dims: vec![3],
        data_type: dtype_code::INT8,
        int32_data: vec![-1, 127, 0xFF],
        ..Default::default()
    };
    assert_eq!(
        tp.try_to_tensor().expect("i8 typed field").data,
        vec![-1.0, 127.0, -1.0]
    );

    let tp = TensorProto {
        name: "b".to_string(),
        dims: vec![3],
        data_type: dtype_code::BOOL,
        int32_data: vec![0, 1, 5],
        ..Default::default()
    };
    assert_eq!(
        tp.try_to_tensor().expect("bool typed field").data,
        vec![0.0, 1.0, 1.0]
    );
}

#[test]
fn typed_fields_decode_int64_double_and_uint64() {
    let tp = TensorProto {
        name: "i64".to_string(),
        dims: vec![2],
        data_type: dtype_code::INT64,
        int64_data: vec![-3, 7],
        ..Default::default()
    };
    assert_eq!(tp.try_to_tensor().expect("i64").data, vec![-3.0, 7.0]);

    let tp = TensorProto {
        name: "f64".to_string(),
        dims: vec![2],
        data_type: dtype_code::DOUBLE,
        double_data: vec![0.25, -8.5],
        ..Default::default()
    };
    assert_eq!(tp.try_to_tensor().expect("f64").data, vec![0.25, -8.5]);

    let tp = TensorProto {
        name: "u64".to_string(),
        dims: vec![2],
        data_type: dtype_code::UINT64,
        uint64_data: vec![0, 65536],
        ..Default::default()
    };
    assert_eq!(tp.try_to_tensor().expect("u64").data, vec![0.0, 65536.0]);
}

#[test]
fn missing_payload_is_an_error_but_empty_tensors_are_allowed() {
    // dims say 4 elements, no field carries data → typed error, not zeros.
    let tp = TensorProto {
        name: "w".to_string(),
        dims: vec![2, 2],
        data_type: dtype_code::FLOAT32,
        ..Default::default()
    };
    let err = tp.try_to_tensor().expect_err("missing payload must fail");
    assert!(
        matches!(err, ReaderError::MissingTensorData { expected: 4, .. }),
        "got {err:?}"
    );

    // A genuinely empty tensor (dim 0) is legal and carries no payload.
    let tp = TensorProto {
        name: "empty".to_string(),
        dims: vec![0, 4],
        data_type: dtype_code::FLOAT32,
        ..Default::default()
    };
    let tensor = tp.try_to_tensor().expect("empty tensor is valid");
    assert_eq!(tensor.shape, vec![0, 4]);
    assert!(tensor.data.is_empty());
}

// ─────────────────────────────────────────────────────────────────
// [a2-9] / [a10-4] dims validation
// ─────────────────────────────────────────────────────────────────

#[test]
fn negative_dims_are_rejected_without_panicking() {
    let tp = raw_tensor("w", &[-1], dtype_code::FLOAT32, vec![0, 0, 128, 63]);
    let err = tp.try_to_tensor().expect_err("negative dim must fail");
    assert!(
        matches!(err, ReaderError::InvalidDim { dim: -1, .. }),
        "got {err:?}"
    );
    // The infallible shim must degrade, never allocate usize::MAX floats.
    let fallback = tp.to_tensor();
    assert!(fallback.data.is_empty());

    // …and the model loader surfaces it instead of crashing.
    let mut tensor_bytes = Vec::new();
    // dims = [-1] as a 10-byte two's-complement varint
    tensor_bytes.extend(encode_bytes_field(1, &encode_varint(u64::MAX)));
    tensor_bytes.extend(encode_varint_field(2, 1));
    tensor_bytes.extend(encode_bytes_field(8, b"w"));
    tensor_bytes.extend(encode_bytes_field(9, &[0, 0, 128, 63]));
    let err = load(&model_with_initializer(&tensor_bytes)).expect_err("load must fail");
    assert!(err.to_lowercase().contains("dim"), "got: {err}");
}

#[test]
fn overflowing_dims_product_is_rejected() {
    let big = 1i64 << 32;
    let tp = raw_tensor("w", &[big, big], dtype_code::FLOAT32, vec![0, 0, 128, 63]);
    let err = tp.try_to_tensor().expect_err("overflowing dims must fail");
    assert!(
        // 32-bit hosts reject the individual dim before the product overflows.
        matches!(
            err,
            ReaderError::ShapeOverflow { .. } | ReaderError::InvalidDim { .. }
        ),
        "got {err:?}"
    );
    assert!(tp.to_tensor().data.is_empty());
}

// ─────────────────────────────────────────────────────────────────
// [a2-10] payload length vs dims product
// ─────────────────────────────────────────────────────────────────

#[test]
fn truncated_payload_is_rejected() {
    // dims=[64,64] (4096 elements) but only 2 floats of raw_data.
    let tp = raw_tensor(
        "w",
        &[64, 64],
        dtype_code::FLOAT32,
        vec![0, 0, 128, 63, 0, 0, 0, 64],
    );
    let err = tp.try_to_tensor().expect_err("truncated payload must fail");
    match err {
        ReaderError::ElementCountMismatch {
            expected, found, ..
        } => {
            assert_eq!(expected, 4096);
            assert_eq!(found, 2);
        }
        other => panic!("expected ElementCountMismatch, got {other:?}"),
    }

    // Too many elements is equally invalid.
    let tp = TensorProto {
        name: "w".to_string(),
        dims: vec![2],
        data_type: dtype_code::FLOAT32,
        float_data: vec![1.0, 2.0, 3.0],
        ..Default::default()
    };
    assert!(tp.try_to_tensor().is_err());
}

/// The infallible `to_tensor()` shim keeps its legacy contract for callers
/// that cannot propagate an error (the streaming parser): a valid but
/// undecodable tensor still reports its declared shape, while a hostile shape
/// degrades to an empty tensor instead of a huge allocation.
#[test]
fn to_tensor_shim_degrades_within_bounds() {
    let tp = raw_tensor("s", &[2], dtype_code::STRING, vec![1, 2, 3, 4]);
    let placeholder = tp.to_tensor();
    assert_eq!(placeholder.shape, vec![2]);
    assert_eq!(placeholder.data, vec![0.0, 0.0]);

    // Above the placeholder cap nothing is allocated.
    let tp = TensorProto {
        name: "huge".to_string(),
        dims: vec![1 << 40],
        data_type: dtype_code::FLOAT32,
        ..Default::default()
    };
    assert!(tp.to_tensor().data.is_empty());
}

// ─────────────────────────────────────────────────────────────────
// [a2-6] / [a5-2] STRINGS attributes reach Attributes::string_lists
// ─────────────────────────────────────────────────────────────────

/// Build a one-node graph whose node carries a repeated-string attribute.
/// `attr_type` is emitted only when `Some` (to exercise the type-0 inference).
fn model_with_strings_attribute(name: &str, values: &[&str], attr_type: Option<u64>) -> Vec<u8> {
    let mut attr = encode_bytes_field(1, name.as_bytes());
    for value in values {
        // AttributeProto.strings is field 9 (repeated bytes).
        attr.extend(encode_bytes_field(9, value.as_bytes()));
    }
    if let Some(t) = attr_type {
        attr.extend(encode_varint_field(20, t));
    }

    let mut node = Vec::new();
    node.extend(encode_bytes_field(1, b"X"));
    node.extend(encode_bytes_field(2, b"Y"));
    node.extend(encode_bytes_field(3, b"tree"));
    node.extend(encode_bytes_field(4, b"TreeEnsembleRegressor"));
    node.extend(encode_bytes_field(5, &attr));

    model_with_graph(&encode_bytes_field(1, &node))
}

#[test]
fn strings_attribute_is_converted_to_string_lists() {
    let modes = ["BRANCH_LEQ", "LEAF", "BRANCH_GT"];
    // attr_type = 8 (STRINGS)
    let (graph, _weights) = load(&model_with_strings_attribute(
        "nodes_modes",
        &modes,
        Some(8),
    ))
    .expect("load");
    let attrs = &graph.nodes[0].attrs;
    assert_eq!(
        attrs.string_lists.get("nodes_modes").map(Vec::as_slice),
        Some(
            &[
                "BRANCH_LEQ".to_string(),
                "LEAF".to_string(),
                "BRANCH_GT".to_string()
            ][..]
        ),
        "STRINGS attribute must populate string_lists",
    );
}

#[test]
fn strings_attribute_is_inferred_when_attr_type_is_unset() {
    let (graph, _weights) = load(&model_with_strings_attribute(
        "activations",
        &["Sigmoid", "Tanh"],
        None,
    ))
    .expect("load");
    let attrs = &graph.nodes[0].attrs;
    assert_eq!(
        attrs.string_lists.get("activations").map(Vec::as_slice),
        Some(&["Sigmoid".to_string(), "Tanh".to_string()][..]),
    );
}

// ─────────────────────────────────────────────────────────────────
// [a2-7] external-data location sandboxing
// ─────────────────────────────────────────────────────────────────

#[test]
fn external_location_cannot_escape_the_model_directory() {
    let dir = scratch_dir("sandbox");

    for bad in ["/etc/passwd", "../../../../etc/passwd", "sub/../../escape"] {
        let tensor_bytes = external_tensor_proto(
            "w",
            &[2],
            1,
            &[("location", bad), ("offset", "0"), ("length", "8")],
            true,
        );
        let err = load_with_path(&model_with_initializer(&tensor_bytes), &dir)
            .expect_err(&format!("location {bad:?} must be rejected"));
        assert!(
            err.contains("rejected"),
            "location {bad:?} gave unexpected error: {err}"
        );
    }

    // A symlink pointing outside the model directory is refused as well.
    let outside = scratch_dir("sandbox_outside");
    std::fs::write(outside.join("secret.bin"), [0u8; 8]).expect("write secret");
    #[cfg(unix)]
    {
        let link = dir.join("weights.bin");
        std::os::unix::fs::symlink(outside.join("secret.bin"), &link).expect("symlink");
        let err = resolve_external_path(&dir, "weights.bin", "w")
            .expect_err("symlink escape must be rejected");
        assert!(
            matches!(err, ReaderError::InvalidExternalLocation { .. }),
            "got {err:?}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&outside);
}

/// The second external-data entry point — the mmap `OnnxReader` — must apply
/// the same sandbox before memory-mapping a sidecar.
#[cfg(feature = "mmap")]
#[test]
fn mmap_reader_rejects_absolute_external_location() {
    use oxionnx_proto::reader::OnnxReader;

    let dir = scratch_dir("mmap_sandbox");
    let tensor_bytes = external_tensor_proto(
        "w",
        &[2],
        1,
        &[
            ("location", "/etc/passwd"),
            ("offset", "0"),
            ("length", "8"),
        ],
        true,
    );
    let model_path = dir.join("model.onnx");
    std::fs::write(&model_path, model_with_initializer(&tensor_bytes)).expect("write model");

    let mut reader = OnnxReader::open(&model_path).expect("open model");
    let tensor = reader
        .find_initializer("w")
        .cloned()
        .expect("initializer present");
    let err = reader
        .initializer_bytes(&tensor)
        .expect_err("absolute location must be rejected");
    assert!(
        matches!(err, ReaderError::InvalidExternalLocation { .. }),
        "got {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────
// [a2-8] external-data offset/length arithmetic
// ─────────────────────────────────────────────────────────────────

fn write_sidecar(dir: &Path, bytes: &[u8]) {
    std::fs::write(dir.join("weights.bin"), bytes).expect("write sidecar");
}

#[test]
fn external_offset_beyond_file_is_an_error_not_a_panic() {
    let dir = scratch_dir("reversed_range");
    write_sidecar(&dir, &[0u8; 100]);

    // offset past EOF, no "length" entry: end = file_len < start.
    let tensor_bytes = external_tensor_proto(
        "w",
        &[2],
        1,
        &[("location", "weights.bin"), ("offset", "1000")],
        true,
    );
    let err = load_with_path(&model_with_initializer(&tensor_bytes), &dir)
        .expect_err("reversed range must fail");
    assert!(err.contains("invalid"), "got: {err}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn external_offset_plus_length_overflow_is_an_error() {
    let dir = scratch_dir("overflow_range");
    write_sidecar(&dir, &[0u8; 100]);

    let tensor_bytes = external_tensor_proto(
        "w",
        &[2],
        1,
        &[
            ("location", "weights.bin"),
            ("offset", "18446744073709551615"),
            ("length", "10"),
        ],
        true,
    );
    let err = load_with_path(&model_with_initializer(&tensor_bytes), &dir)
        .expect_err("overflowing range must fail");
    assert!(
        err.contains("not addressable") || err.contains("overflows") || err.contains("invalid"),
        "got: {err}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn external_length_past_eof_is_an_error() {
    let dir = scratch_dir("past_eof");
    write_sidecar(&dir, &[0u8; 16]);

    let tensor_bytes = external_tensor_proto(
        "w",
        &[8],
        1,
        &[
            ("location", "weights.bin"),
            ("offset", "8"),
            ("length", "32"),
        ],
        true,
    );
    let err = load_with_path(&model_with_initializer(&tensor_bytes), &dir)
        .expect_err("range past EOF must fail");
    assert!(err.contains("invalid"), "got: {err}");

    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────
// [a2-5] external data shares the inline dtype table
// ─────────────────────────────────────────────────────────────────

#[test]
fn external_uint8_weights_decode_instead_of_zeroing() {
    let dir = scratch_dir("ext_uint8");
    write_sidecar(&dir, &[0, 1, 254, 255]);

    let tensor_bytes = external_tensor_proto(
        "qw",
        &[2, 2],
        u64::try_from(dtype_code::UINT8).expect("dtype code"),
        &[
            ("location", "weights.bin"),
            ("offset", "0"),
            ("length", "4"),
        ],
        true,
    );
    let (_graph, weights) =
        load_with_path(&model_with_initializer(&tensor_bytes), &dir).expect("load");
    let tensor = weights.get("qw").expect("weight present");
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.data, vec![0.0, 1.0, 254.0, 255.0]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn external_payload_must_match_dims() {
    let dir = scratch_dir("ext_mismatch");
    write_sidecar(&dir, &[0u8; 8]);

    // dims=[64,64] but only 8 bytes are referenced.
    let tensor_bytes = external_tensor_proto(
        "w",
        &[64, 64],
        1,
        &[
            ("location", "weights.bin"),
            ("offset", "0"),
            ("length", "8"),
        ],
        true,
    );
    let err = load_with_path(&model_with_initializer(&tensor_bytes), &dir)
        .expect_err("payload/dims mismatch must fail");
    assert!(err.contains("expected 4096 elements"), "got: {err}");

    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────
// [a2-21] one external-data predicate for both APIs
// ─────────────────────────────────────────────────────────────────

#[test]
fn external_data_without_data_location_is_still_external() {
    let dir = scratch_dir("no_data_location");
    let floats: [f32; 4] = [1.5, -2.5, 3.5, -4.5];
    let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    write_sidecar(&dir, &raw);

    // external_data entries present, data_location omitted.
    let tensor_bytes = external_tensor_proto(
        "w",
        &[2, 2],
        1,
        &[
            ("location", "weights.bin"),
            ("offset", "0"),
            ("length", "16"),
        ],
        false,
    );
    let model_bytes = model_with_initializer(&tensor_bytes);

    let (_graph, weights) = load_with_path(&model_bytes, &dir).expect("load_with_path");
    let tensor = weights.get("w").expect("weight present");
    assert_eq!(tensor.data, floats.to_vec());

    // The pathless API must refuse it rather than yield an all-zero tensor.
    let err = load(&model_bytes).expect_err("pathless load must refuse external data");
    assert!(
        err.contains("External data requires load_with_path()"),
        "got: {err}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────
// [a5-8] unknown graph-I/O element types
// ─────────────────────────────────────────────────────────────────

#[test]
fn unrecognized_elem_type_is_reported_through_the_fallible_api() {
    let value_info = ValueInfoProto {
        name: "labels".to_string(),
        elem_type: 8, // STRING
        shape: vec![Some(1)],
        ..Default::default()
    };
    let err = value_info
        .try_to_tensor_info()
        .expect_err("STRING has no DType");
    assert!(
        matches!(err, ReaderError::UnsupportedDtype { dtype: 8, .. }),
        "got {err:?}"
    );

    // Recognized types still map exactly.
    let value_info = ValueInfoProto {
        name: "x".to_string(),
        elem_type: dtype_code::BOOL,
        shape: vec![Some(2)],
        ..Default::default()
    };
    let info = value_info.try_to_tensor_info().expect("bool maps");
    assert_eq!(info.dtype, oxionnx_core::dtype::DType::Bool);
    assert_eq!(info.shape, vec![Some(2)]);
}
