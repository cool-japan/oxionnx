//! Minimal ONNX protobuf parser — zero external dependencies.
//!
//! Supports the subset of proto3 wire format needed for ONNX model files:
//! - Varint (wire type 0): i32, i64, bool, enum
//! - 64-bit fixed (wire type 1): f64
//! - Length-delimited (wire type 2): strings, bytes, nested messages, packed arrays
//! - 32-bit fixed (wire type 5): f32

use crate::types::*;

#[derive(Debug)]
enum WireValue<'a> {
    Varint(u64),
    Fixed64([u8; 8]),
    Bytes(&'a [u8]),
    Fixed32([u8; 4]),
}

pub fn read_varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), String> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        if pos >= buf.len() {
            return Err("varint: unexpected EOF".into());
        }
        let byte = buf[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint: overflow".into());
        }
    }
    Ok((result, pos))
}

fn read_field<'a>(buf: &'a [u8], pos: usize) -> Result<(u32, WireValue<'a>, usize), String> {
    let (tag, pos) = read_varint(buf, pos)?;
    let field_no = (tag >> 3) as u32;
    let wire_type = (tag & 0x7) as u8;
    match wire_type {
        0 => {
            let (v, pos) = read_varint(buf, pos)?;
            Ok((field_no, WireValue::Varint(v), pos))
        }
        1 => {
            if pos + 8 > buf.len() {
                return Err("fixed64: EOF".into());
            }
            let arr = [
                buf[pos],
                buf[pos + 1],
                buf[pos + 2],
                buf[pos + 3],
                buf[pos + 4],
                buf[pos + 5],
                buf[pos + 6],
                buf[pos + 7],
            ];
            Ok((field_no, WireValue::Fixed64(arr), pos + 8))
        }
        2 => {
            let (len, pos) = read_varint(buf, pos)?;
            let len = len as usize;
            if pos + len > buf.len() {
                return Err(format!(
                    "length-delimited: EOF (need {len}, have {})",
                    buf.len() - pos
                ));
            }
            Ok((field_no, WireValue::Bytes(&buf[pos..pos + len]), pos + len))
        }
        5 => {
            if pos + 4 > buf.len() {
                return Err("fixed32: EOF".into());
            }
            let arr = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
            Ok((field_no, WireValue::Fixed32(arr), pos + 4))
        }
        wt => Err(format!("unknown wire type {wt} at pos {pos}")),
    }
}

// ─────────────────────────────────────────────────────────────────

pub fn parse_tensor_proto(buf: &[u8]) -> Result<TensorProto, String> {
    let mut t = TensorProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Varint(v)) => {
                // dims: individual int64 varint
                t.dims.push(v as i64);
            }
            (1, WireValue::Bytes(b)) => {
                // dims: packed int64
                let mut p = 0;
                while p < b.len() {
                    let (v, np) = read_varint(b, p)?;
                    t.dims.push(v as i64);
                    p = np;
                }
            }
            (2, WireValue::Varint(v)) => t.data_type = v as i32,
            (8, WireValue::Bytes(b)) => t.name = String::from_utf8_lossy(b).into_owned(),
            (4, WireValue::Bytes(b)) => {
                // float_data: packed float32
                for chunk in b.chunks_exact(4) {
                    t.float_data
                        .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (5, WireValue::Bytes(b)) => {
                // int32_data: packed int32
                for chunk in b.chunks_exact(4) {
                    t.int32_data
                        .push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (6, WireValue::Bytes(b)) => {
                // int64_data: packed int64
                for chunk in b.chunks_exact(8) {
                    t.int64_data.push(i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]));
                }
            }
            (7, WireValue::Bytes(b)) => {
                // double_data: packed float64
                for chunk in b.chunks_exact(8) {
                    t.double_data.push(f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]));
                }
            }
            (7, WireValue::Fixed64(b)) => {
                // double_data: individual (unpacked) float64
                t.double_data.push(f64::from_le_bytes(b));
            }
            (9, WireValue::Bytes(b)) => t.raw_data = b.to_vec(),
            (13, WireValue::Bytes(b)) => {
                // ONNX spec: field 13 = repeated StringStringEntryProto external_data.
                // StringStringEntryProto: field 1 = key, field 2 = value.
                let mut key = String::new();
                let mut val = String::new();
                let mut p = 0;
                while p < b.len() {
                    let (f, v2, np) = read_field(b, p)?;
                    p = np;
                    match (f, v2) {
                        (1, WireValue::Bytes(s)) => key = String::from_utf8_lossy(s).into_owned(),
                        (2, WireValue::Bytes(s)) => val = String::from_utf8_lossy(s).into_owned(),
                        _ => {}
                    }
                }
                t.external_data.push((key, val));
            }
            (14, WireValue::Varint(v)) => t.data_location = v as i32,
            _ => {} // ignore unknown fields
        }
    }
    Ok(t)
}

// ─────────────────────────────────────────────────────────────────

pub fn parse_attribute(buf: &[u8]) -> Result<AttributeProto, String> {
    let mut attr = AttributeProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Bytes(b)) => attr.name = String::from_utf8_lossy(b).into_owned(),
            (2, WireValue::Fixed32(b)) => attr.value.f = f32::from_le_bytes(b),
            (3, WireValue::Varint(v)) => attr.value.i = v as i64,
            (4, WireValue::Bytes(b)) => attr.value.s = String::from_utf8_lossy(b).into_owned(),
            (5, WireValue::Bytes(b)) => attr.value.t = Some(parse_tensor_proto(b)?),
            (6, WireValue::Bytes(b)) => {
                attr.value.g = Some(Box::new(parse_graph(b)?));
            }
            (7, WireValue::Bytes(b)) => {
                // floats: packed
                for chunk in b.chunks_exact(4) {
                    attr.value
                        .floats
                        .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (8, WireValue::Varint(v)) => {
                // ints: individual varint
                attr.value.ints.push(v as i64);
            }
            (8, WireValue::Bytes(b)) => {
                // ints: packed varint
                let mut p = 0;
                while p < b.len() {
                    let (v, np) = read_varint(b, p)?;
                    attr.value.ints.push(v as i64);
                    p = np;
                }
            }
            (9, WireValue::Bytes(b)) => {
                let mut p = 0;
                while p < b.len() {
                    let (field2, val2, next2) = read_field(b, p)?;
                    p = next2;
                    if field2 == 0 {
                        if let WireValue::Bytes(s) = val2 {
                            attr.value
                                .strings
                                .push(String::from_utf8_lossy(s).into_owned());
                        }
                    }
                }
            }
            (20, WireValue::Varint(v)) => attr.value.attr_type = v as i32,
            _ => {}
        }
    }
    Ok(attr)
}

// ─────────────────────────────────────────────────────────────────

pub fn parse_node(buf: &[u8]) -> Result<NodeProto, String> {
    let mut node = NodeProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Bytes(b)) => node.inputs.push(String::from_utf8_lossy(b).into_owned()),
            (2, WireValue::Bytes(b)) => node.outputs.push(String::from_utf8_lossy(b).into_owned()),
            (3, WireValue::Bytes(b)) => node.name = String::from_utf8_lossy(b).into_owned(),
            (4, WireValue::Bytes(b)) => node.op_type = String::from_utf8_lossy(b).into_owned(),
            (5, WireValue::Bytes(b)) => node.attributes.push(parse_attribute(b)?),
            (7, WireValue::Bytes(b)) => node.domain = String::from_utf8_lossy(b).into_owned(),
            _ => {}
        }
    }
    Ok(node)
}

// ─────────────────────────────────────────────────────────────────

pub fn parse_graph(buf: &[u8]) -> Result<GraphProto, String> {
    let mut graph = GraphProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Bytes(b)) => graph.nodes.push(parse_node(b)?),
            (2, WireValue::Bytes(b)) => graph.name = String::from_utf8_lossy(b).into_owned(),
            (5, WireValue::Bytes(b)) => graph.initializers.push(parse_tensor_proto(b)?),
            (11, WireValue::Bytes(b)) => {
                // ValueInfoProto for graph inputs — parse full info and extract name
                let vi = parse_value_info_proto(b);
                graph.inputs.push(vi.name.clone());
                graph.input_value_infos.push(vi);
            }
            (12, WireValue::Bytes(b)) => {
                // ValueInfoProto for graph outputs
                let vi = parse_value_info_proto(b);
                graph.outputs.push(vi.name.clone());
                graph.output_value_infos.push(vi);
            }
            _ => {}
        }
    }
    Ok(graph)
}

/// Parse a full `ValueInfoProto`, extracting name, elem_type, and shape.
///
/// ONNX proto layout (field numbers):
/// ```text
/// ValueInfoProto {
///   1: name (string)
///   2: type (TypeProto) {
///     1: tensor_type (Tensor) {
///       1: elem_type (varint)
///       2: shape (TensorShapeProto) {
///         1: dim[] (Dimension) {
///           1: dim_value (varint) -- static size
///           2: dim_param (string) -- symbolic name (dynamic)
///         }
///       }
///     }
///   }
/// }
/// ```
pub fn parse_value_info_proto(buf: &[u8]) -> ValueInfoProto {
    let mut vi = ValueInfoProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        match read_field(buf, pos) {
            Ok((1, WireValue::Bytes(b), next)) => {
                vi.name = String::from_utf8_lossy(b).into_owned();
                pos = next;
            }
            Ok((2, WireValue::Bytes(b), next)) => {
                // TypeProto — look for tensor_type (field 1)
                parse_type_proto_into(b, &mut vi);
                pos = next;
            }
            Ok((_, _, next)) => {
                pos = next;
            }
            Err(_) => break,
        }
    }
    vi
}

/// Parse TypeProto bytes and fill elem_type / shape into `vi`.
fn parse_type_proto_into(buf: &[u8], vi: &mut ValueInfoProto) {
    let mut pos = 0;
    while pos < buf.len() {
        match read_field(buf, pos) {
            Ok((1, WireValue::Bytes(b), next)) => {
                // tensor_type: Tensor { 1: elem_type, 2: shape }
                parse_tensor_type_into(b, vi);
                pos = next;
            }
            Ok((_, _, next)) => {
                pos = next;
            }
            Err(_) => break,
        }
    }
}

/// Parse TensorTypeProto bytes and fill elem_type / shape into `vi`.
fn parse_tensor_type_into(buf: &[u8], vi: &mut ValueInfoProto) {
    let mut pos = 0;
    while pos < buf.len() {
        match read_field(buf, pos) {
            Ok((1, WireValue::Varint(v), next)) => {
                vi.elem_type = v as i32;
                pos = next;
            }
            Ok((2, WireValue::Bytes(b), next)) => {
                // TensorShapeProto — repeated Dimension (field 1)
                parse_shape_proto_into(b, vi);
                pos = next;
            }
            Ok((_, _, next)) => {
                pos = next;
            }
            Err(_) => break,
        }
    }
}

/// Parse TensorShapeProto bytes and append dimensions into `vi.shape` and `vi.dim_params`.
fn parse_shape_proto_into(buf: &[u8], vi: &mut ValueInfoProto) {
    let mut pos = 0;
    while pos < buf.len() {
        match read_field(buf, pos) {
            Ok((1, WireValue::Bytes(b), next)) => {
                // Dimension: field 1 = dim_value (varint), field 2 = dim_param (string)
                let (dim_value, dim_param) = parse_dimension(b);
                vi.shape.push(dim_value);
                vi.dim_params.push(dim_param);
                pos = next;
            }
            Ok((_, _, next)) => {
                pos = next;
            }
            Err(_) => break,
        }
    }
}

/// Parse a single Dimension message and return (static_size, symbolic_param_name).
///
/// ONNX Dimension fields:
///   1: dim_value (varint) — concrete size; 0 means dynamic
///   2: dim_param (string) — symbolic name (e.g. "batch_size")
fn parse_dimension(buf: &[u8]) -> (Option<i64>, Option<String>) {
    let mut dim_value: Option<i64> = None;
    let mut dim_param: Option<String> = None;
    let mut pos = 0;
    while pos < buf.len() {
        match read_field(buf, pos) {
            Ok((1, WireValue::Varint(v), next)) => {
                // dim_value: 0 = dynamic/batch, >0 = static
                dim_value = if v == 0 { None } else { Some(v as i64) };
                pos = next;
            }
            Ok((2, WireValue::Bytes(s), next)) => {
                // dim_param: symbolic dimension name
                let name = String::from_utf8_lossy(s).into_owned();
                if !name.is_empty() {
                    dim_param = Some(name);
                }
                pos = next;
            }
            Ok((_, _, next)) => {
                pos = next;
            }
            Err(_) => break,
        }
    }
    (dim_value, dim_param)
}

// ─────────────────────────────────────────────────────────────────

/// Parse an ONNX model file (`.onnx`) from its raw bytes.
pub fn parse_model(buf: &[u8]) -> Result<ModelProto, String> {
    let mut model = ModelProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Varint(v)) => model.ir_version = v as i64,
            (2, WireValue::Bytes(b)) => {
                model.producer_name = String::from_utf8_lossy(b).into_owned();
            }
            (3, WireValue::Bytes(b)) => {
                model.producer_version = String::from_utf8_lossy(b).into_owned();
            }
            (4, WireValue::Bytes(b)) => {
                model.domain = String::from_utf8_lossy(b).into_owned();
            }
            (5, WireValue::Varint(v)) => model.model_version = v as i64,
            (6, WireValue::Bytes(b)) => {
                model.doc_string = String::from_utf8_lossy(b).into_owned();
            }
            (7, WireValue::Bytes(b)) => model.graph = parse_graph(b)?,
            (8, WireValue::Bytes(b)) => {
                // OperatorSetIdProto — field 1 = domain (string), field 2 = version (varint)
                let mut domain = String::new();
                let mut version = 0i64;
                let mut p = 0;
                while p < b.len() {
                    if let Ok((f, val, np)) = read_field(b, p) {
                        match (f, val) {
                            (1, WireValue::Bytes(s)) => {
                                domain = String::from_utf8_lossy(s).into_owned();
                            }
                            (2, WireValue::Varint(v)) => {
                                version = v as i64;
                            }
                            _ => {}
                        }
                        p = np;
                    } else {
                        break;
                    }
                }
                // Backwards compat: set opset_version from default domain
                if domain.is_empty() {
                    model.opset_version = version;
                }
                model
                    .opset_imports
                    .push(crate::types::OpsetImport { domain, version });
            }
            (14, WireValue::Bytes(b)) => {
                let (key, val) = parse_string_string_entry(b)?;
                model.metadata_props.push((key, val));
            }
            (20, WireValue::Bytes(b)) => {
                model.training_info.push(parse_training_info(b)?);
            }
            _ => {}
        }
    }
    Ok(model)
}

/// Parse a TrainingInfoProto message.
///
/// ONNX TrainingInfoProto fields:
///   1 = initialization (GraphProto) — optional
///   2 = algorithm (GraphProto) — the training graph
///   3 = initialization_binding (repeated StringStringEntryProto)
///   4 = update_binding (repeated StringStringEntryProto)
fn parse_training_info(buf: &[u8]) -> Result<crate::types::TrainingInfo, String> {
    let mut info = crate::types::TrainingInfo::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Bytes(b)) => {
                // initialization: GraphProto computing initial values for trainable tensors
                let graph = parse_graph(b)?;
                info.initialization_graph = Some(graph);
            }
            (2, WireValue::Bytes(b)) => {
                // algorithm / training graph
                let graph = parse_graph(b)?;
                // Try to extract algorithm name from the graph name
                if !graph.name.is_empty() {
                    info.algorithm = graph.name.clone();
                }
                info.training_graph = Some(graph);
            }
            (3, WireValue::Bytes(b)) => {
                // initialization_binding: StringStringEntryProto
                let (key, val) = parse_string_string_entry(b)?;
                info.initialization_bindings.push((key, val));
            }
            (4, WireValue::Bytes(b)) => {
                // update_binding: StringStringEntryProto
                let (key, val) = parse_string_string_entry(b)?;
                info.update_bindings.push((key, val));
            }
            _ => {}
        }
    }
    Ok(info)
}

/// Parse a StringStringEntryProto (field 1 = key, field 2 = value).
fn parse_string_string_entry(buf: &[u8]) -> Result<(String, String), String> {
    let mut key = String::new();
    let mut val = String::new();
    let mut pos = 0;
    while pos < buf.len() {
        let (f, v, next) = read_field(buf, pos)?;
        pos = next;
        match (f, v) {
            (1, WireValue::Bytes(s)) => key = String::from_utf8_lossy(s).into_owned(),
            (2, WireValue::Bytes(s)) => val = String::from_utf8_lossy(s).into_owned(),
            _ => {}
        }
    }
    Ok((key, val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_varint_single_byte() {
        let buf = [0x05u8];
        let (v, pos) = read_varint(&buf, 0).expect("varint should parse");
        assert_eq!(v, 5);
        assert_eq!(pos, 1);
    }

    #[test]
    fn test_read_varint_multibyte() {
        // 300 = 0b1_0010_1100 → 0b1010_1100 0b0000_0010
        let buf = [0xACu8, 0x02];
        let (v, pos) = read_varint(&buf, 0).expect("varint should parse");
        assert_eq!(v, 300);
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_parse_empty_model() {
        // An empty protobuf buffer = empty model (all defaults)
        let model = parse_model(&[]).expect("empty model should parse");
        assert_eq!(model.ir_version, 0);
        assert!(model.graph.nodes.is_empty());
    }

    /// Helper: encode a varint into a Vec<u8>.
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

    /// Helper: encode a protobuf field tag + varint value.
    fn encode_varint_field(field: u32, val: u64) -> Vec<u8> {
        let tag = field << 3; // wire type 0 = varint
        let mut buf = encode_varint(tag as u64);
        buf.extend(encode_varint(val));
        buf
    }

    /// Helper: encode a protobuf field tag + length-delimited bytes.
    fn encode_bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
        let tag = (field << 3) | 2; // wire type 2 = length-delimited
        let mut buf = encode_varint(tag as u64);
        buf.extend(encode_varint(data.len() as u64));
        buf.extend(data);
        buf
    }

    #[test]
    fn test_parse_opset_imports() {
        // Build a ModelProto with two opset imports:
        //   1) default domain "", version 13
        //   2) domain "ai.onnx.ml", version 2

        // OperatorSetIdProto for default domain: field 2 = 13 (no field 1)
        let opset1 = encode_varint_field(2, 13);
        // OperatorSetIdProto for ai.onnx.ml: field 1 = "ai.onnx.ml", field 2 = 2
        let mut opset2 = encode_bytes_field(1, b"ai.onnx.ml");
        opset2.extend(encode_varint_field(2, 2));

        // ModelProto: field 1 = ir_version, field 8 = opset_import (repeated)
        let mut model_bytes = encode_varint_field(1, 7); // ir_version=7
        model_bytes.extend(encode_bytes_field(8, &opset1));
        model_bytes.extend(encode_bytes_field(8, &opset2));

        let model = parse_model(&model_bytes).expect("should parse model with opsets");
        assert_eq!(model.ir_version, 7);
        assert_eq!(model.opset_imports.len(), 2);

        assert_eq!(model.opset_imports[0].domain, "");
        assert_eq!(model.opset_imports[0].version, 13);

        assert_eq!(model.opset_imports[1].domain, "ai.onnx.ml");
        assert_eq!(model.opset_imports[1].version, 2);

        // Backwards compat: opset_version should be set from default domain
        assert_eq!(model.opset_version, 13);
    }

    #[test]
    fn test_opset_version_backwards_compat() {
        // Single opset import, default domain, version 11
        let opset = encode_varint_field(2, 11);
        let mut model_bytes = encode_varint_field(1, 6);
        model_bytes.extend(encode_bytes_field(8, &opset));

        let model = parse_model(&model_bytes).expect("should parse");
        assert_eq!(model.opset_version, 11);
        assert_eq!(model.opset_imports.len(), 1);
        assert_eq!(model.opset_imports[0].domain, "");
        assert_eq!(model.opset_imports[0].version, 11);
    }

    #[test]
    fn test_external_data_fields_parsed() {
        // Build a TensorProto with data_location=1 and two external_data entries

        // StringStringEntryProto for "location" = "weights.bin"
        let mut entry1 = encode_bytes_field(1, b"location");
        entry1.extend(encode_bytes_field(2, b"weights.bin"));

        // StringStringEntryProto for "offset" = "1024"
        let mut entry2 = encode_bytes_field(1, b"offset");
        entry2.extend(encode_bytes_field(2, b"1024"));

        // StringStringEntryProto for "length" = "4096"
        let mut entry3 = encode_bytes_field(1, b"length");
        entry3.extend(encode_bytes_field(2, b"4096"));

        // TensorProto: dims=[2, 4], data_type=1, name="weight", data_location=1, external_data entries
        let mut tensor_bytes = Vec::new();
        // dims packed (field 1, wire type 2)
        let mut dims_packed = encode_varint(2);
        dims_packed.extend(encode_varint(4));
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed));
        // data_type = 1
        tensor_bytes.extend(encode_varint_field(2, 1));
        // name = "weight"
        tensor_bytes.extend(encode_bytes_field(8, b"weight"));
        // external_data entries (field 13, repeated StringStringEntryProto)
        tensor_bytes.extend(encode_bytes_field(13, &entry1));
        tensor_bytes.extend(encode_bytes_field(13, &entry2));
        tensor_bytes.extend(encode_bytes_field(13, &entry3));
        // data_location = 1 (field 14, enum)
        tensor_bytes.extend(encode_varint_field(14, 1));

        let tp = parse_tensor_proto(&tensor_bytes).expect("should parse tensor proto");
        assert_eq!(tp.name, "weight");
        assert_eq!(tp.data_location, 1);
        assert_eq!(tp.dims, vec![2, 4]);
        assert_eq!(tp.data_type, 1);
        assert_eq!(tp.external_data.len(), 3);
        assert_eq!(
            tp.external_data[0],
            ("location".to_string(), "weights.bin".to_string())
        );
        assert_eq!(
            tp.external_data[1],
            ("offset".to_string(), "1024".to_string())
        );
        assert_eq!(
            tp.external_data[2],
            ("length".to_string(), "4096".to_string())
        );
    }

    #[test]
    fn test_training_info_empty() {
        // Empty model bytes should have no training info
        let model = parse_model(&[]).expect("empty model should parse");
        assert!(model.training_info.is_empty());

        // Model with only ir_version, no training_info field
        let model_bytes = encode_varint_field(1, 7);
        let model = parse_model(&model_bytes).expect("should parse");
        assert!(model.training_info.is_empty());
    }

    #[test]
    fn test_training_info_parsing() {
        // Build a TrainingInfoProto with bindings
        // field 3 = initialization_binding (StringStringEntryProto)
        let mut init_binding = encode_bytes_field(1, b"weight");
        init_binding.extend(encode_bytes_field(2, b"weight_init"));

        // field 4 = update_binding
        let mut update_binding = encode_bytes_field(1, b"weight");
        update_binding.extend(encode_bytes_field(2, b"weight_grad"));

        // field 2 = algorithm graph with name "Adam"
        let graph_bytes = encode_bytes_field(2, b"Adam"); // GraphProto field 2 = name

        let mut training_bytes = Vec::new();
        training_bytes.extend(encode_bytes_field(2, &graph_bytes)); // training graph
        training_bytes.extend(encode_bytes_field(3, &init_binding));
        training_bytes.extend(encode_bytes_field(4, &update_binding));

        // ModelProto with field 20 = training_info
        let mut model_bytes = encode_varint_field(1, 8);
        model_bytes.extend(encode_bytes_field(20, &training_bytes));

        let model = parse_model(&model_bytes).expect("should parse model with training info");
        assert_eq!(model.training_info.len(), 1);

        let info = &model.training_info[0];
        assert_eq!(info.algorithm, "Adam");
        assert!(info.training_graph.is_some());
        assert_eq!(info.initialization_bindings.len(), 1);
        assert_eq!(info.initialization_bindings[0].0, "weight");
        assert_eq!(info.initialization_bindings[0].1, "weight_init");
        assert_eq!(info.update_bindings.len(), 1);
        assert_eq!(info.update_bindings[0].0, "weight");
        assert_eq!(info.update_bindings[0].1, "weight_grad");
        // No initialization graph (field 1) was supplied in this message.
        assert!(info.initialization_graph.is_none());
    }

    #[test]
    fn test_training_info_initialization_graph_parsed() {
        // Build a TrainingInfoProto whose field 1 (initialization) is a GraphProto
        // containing one node (Constant) and a graph name.

        // ── encode a Constant NodeProto for the initialization graph ──────────
        let mut const_node = Vec::new();
        const_node.extend(encode_bytes_field(2, b"weight")); // output (field 2)
        const_node.extend(encode_bytes_field(3, b"init_const")); // name (field 3)
        const_node.extend(encode_bytes_field(4, b"Constant")); // op_type (field 4)

        // ── encode the initialization GraphProto: field 1 = node, field 2 = name ──
        let mut init_graph = Vec::new();
        init_graph.extend(encode_bytes_field(1, &const_node)); // node (field 1)
        init_graph.extend(encode_bytes_field(2, b"init_graph")); // name (field 2)

        // field 2 = algorithm/training graph with name "SGD"
        let train_graph = encode_bytes_field(2, b"SGD"); // GraphProto field 2 = name

        let mut training_bytes = Vec::new();
        training_bytes.extend(encode_bytes_field(1, &init_graph)); // initialization graph
        training_bytes.extend(encode_bytes_field(2, &train_graph)); // training graph

        // ModelProto with field 20 = training_info
        let mut model_bytes = encode_varint_field(1, 8);
        model_bytes.extend(encode_bytes_field(20, &training_bytes));

        let model =
            parse_model(&model_bytes).expect("should parse model with initialization graph");
        assert_eq!(model.training_info.len(), 1);

        let info = &model.training_info[0];
        assert_eq!(info.algorithm, "SGD");
        assert!(info.training_graph.is_some());

        let init = info
            .initialization_graph
            .as_ref()
            .expect("initialization graph must be parsed");
        assert_eq!(init.name, "init_graph");
        assert_eq!(init.nodes.len(), 1);
        assert_eq!(init.nodes[0].op_type, "Constant");
    }

    #[test]
    fn test_training_info_without_initialization_graph() {
        // A TrainingInfoProto with only a training graph (field 2) and no field 1
        // must yield initialization_graph == None.
        let train_graph = encode_bytes_field(2, b"Adam"); // GraphProto field 2 = name

        let mut training_bytes = Vec::new();
        training_bytes.extend(encode_bytes_field(2, &train_graph)); // training graph only

        let mut model_bytes = encode_varint_field(1, 8);
        model_bytes.extend(encode_bytes_field(20, &training_bytes));

        let model = parse_model(&model_bytes).expect("should parse model");
        assert_eq!(model.training_info.len(), 1);
        assert!(model.training_info[0].initialization_graph.is_none());
    }

    #[test]
    fn test_subgraph_attribute_parsed() {
        // Build a minimal If node with a then_branch subgraph.
        // The subgraph has one node (Relu) with input "X" and output "Y".

        // ── encode a Relu NodeProto ──────────────────────────────────────────
        let mut relu_node = Vec::new();
        relu_node.extend(encode_bytes_field(1, b"X")); // input
        relu_node.extend(encode_bytes_field(2, b"Y")); // output
        relu_node.extend(encode_bytes_field(3, b"relu_node")); // name
        relu_node.extend(encode_bytes_field(4, b"Relu")); // op_type

        // ── encode a GraphProto (then_graph): field 1 = node, field 2 = name ──
        let mut then_graph = Vec::new();
        then_graph.extend(encode_bytes_field(1, &relu_node)); // node (field 1)
        then_graph.extend(encode_bytes_field(2, b"then_graph")); // name (field 2)

        // ── encode an AttributeProto: name="then_branch", g=then_graph (field 6), attr_type=5 ──
        let mut then_attr = Vec::new();
        then_attr.extend(encode_bytes_field(1, b"then_branch")); // name (field 1)
        then_attr.extend(encode_bytes_field(6, &then_graph)); // g: GraphProto (field 6)
        then_attr.extend(encode_varint_field(20, 5)); // attr_type = GRAPH (field 20)

        // ── encode an If NodeProto ────────────────────────────────────────────
        let mut if_node = Vec::new();
        if_node.extend(encode_bytes_field(1, b"cond")); // input
        if_node.extend(encode_bytes_field(2, b"result")); // output
        if_node.extend(encode_bytes_field(3, b"if_node")); // name
        if_node.extend(encode_bytes_field(4, b"If")); // op_type
        if_node.extend(encode_bytes_field(5, &then_attr)); // attribute

        let node = parse_node(&if_node).expect("parse_node failed");
        assert_eq!(node.op_type, "If");
        assert_eq!(node.attributes.len(), 1);

        let attr = &node.attributes[0];
        assert_eq!(attr.name, "then_branch");
        assert!(
            attr.value.g.is_some(),
            "then_branch subgraph must be parsed into g"
        );

        let subgraph = attr.value.g.as_ref().unwrap();
        assert_eq!(subgraph.nodes.len(), 1);
        assert_eq!(subgraph.nodes[0].op_type, "Relu");
    }
}
