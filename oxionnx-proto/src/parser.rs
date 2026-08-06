//! Minimal ONNX protobuf parser — zero external dependencies.
//!
//! Supports the protobuf wire format needed for ONNX model files:
//! - Varint (wire type 0): i32, i64, bool, enum
//! - 64-bit fixed (wire type 1): f64
//! - Length-delimited (wire type 2): strings, bytes, nested messages, packed arrays
//! - Groups (wire types 3/4): skipped, as required for unknown-field tolerance
//! - 32-bit fixed (wire type 5): f32
//!
//! ONNX's schema is proto2, so every repeated numeric field is accepted in both its
//! packed and unpacked form. All offset arithmetic is checked: a malformed or
//! truncated file yields an `Err`, never a panic, and nesting is depth-bounded by
//! [`MAX_NESTING_DEPTH`].

use crate::types::*;

/// Maximum protobuf message nesting depth accepted by the parser.
///
/// ONNX subgraph attributes (`If`/`Loop`/`Scan` bodies) nest recursively, and each
/// level costs only a handful of wire bytes. Without a bound, a small hostile file
/// can drive the parser (and the recursive `Drop` of the resulting tree) past the
/// thread stack and abort the process. 64 is far above anything a real model uses.
pub const MAX_NESTING_DEPTH: u32 = 64;

/// Largest legal protobuf field number (2^29 - 1).
const MAX_FIELD_NUMBER: u64 = 0x1FFF_FFFF;

#[derive(Debug)]
enum WireValue<'a> {
    Varint(u64),
    Fixed64([u8; 8]),
    Bytes(&'a [u8]),
    Fixed32([u8; 4]),
    /// A group (wire types 3/4). Groups are a deprecated protobuf encoding that no
    /// ONNX message uses, but a conformant parser must skip them rather than abort.
    Group,
}

/// Read a base-128 varint, rejecting encodings that do not fit in 64 bits.
///
/// The 10th byte may only carry bit 63 (`0x00` or `0x01`); anything larger would
/// silently discard bits, letting two distinct byte sequences decode to the same
/// value — a way for a corrupt length to masquerade as a valid one.
pub fn read_varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), String> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        if pos >= buf.len() {
            return Err("varint: unexpected EOF".into());
        }
        let byte = buf[pos];
        pos += 1;
        if shift == 63 {
            // 10th byte: only bit 63 is representable, and no 11th byte may follow.
            if byte > 0x01 {
                return Err("varint: overflow (value exceeds 64 bits)".into());
            }
            return Ok((result | ((byte as u64) << 63), pos));
        }
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
    }
}

/// Compute `pos + len`, rejecting overflow and any end past the buffer.
#[inline]
fn checked_end(pos: usize, len: usize, buf_len: usize, what: &str) -> Result<usize, String> {
    match pos.checked_add(len) {
        Some(end) if end <= buf_len => Ok(end),
        _ => Err(format!(
            "{what}: EOF (need {len} bytes at offset {pos}, buffer is {buf_len})"
        )),
    }
}

/// Convert a wire-format length to `usize`, rejecting values that cannot be
/// addressed on this target (relevant on 32-bit targets such as wasm32).
#[inline]
fn len_to_usize(len: u64, what: &str) -> Result<usize, String> {
    usize::try_from(len).map_err(|_| format!("{what}: length {len} exceeds addressable memory"))
}

fn read_field<'a>(buf: &'a [u8], pos: usize) -> Result<(u32, WireValue<'a>, usize), String> {
    read_field_at(buf, pos, 0)
}

fn read_field_at<'a>(
    buf: &'a [u8],
    pos: usize,
    depth: u32,
) -> Result<(u32, WireValue<'a>, usize), String> {
    let (field_no, wire_type, pos) = read_tag(buf, pos)?;
    match wire_type {
        0 => {
            let (v, pos) = read_varint(buf, pos)?;
            Ok((field_no, WireValue::Varint(v), pos))
        }
        1 => {
            let end = checked_end(pos, 8, buf.len(), "fixed64")?;
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&buf[pos..end]);
            Ok((field_no, WireValue::Fixed64(arr), end))
        }
        2 => {
            let (len, pos) = read_varint(buf, pos)?;
            let len = len_to_usize(len, "length-delimited")?;
            let end = checked_end(pos, len, buf.len(), "length-delimited")?;
            Ok((field_no, WireValue::Bytes(&buf[pos..end]), end))
        }
        3 => {
            let end = skip_group(buf, pos, field_no, depth)?;
            Ok((field_no, WireValue::Group, end))
        }
        4 => Err(format!(
            "unexpected end-group tag for field {field_no} at pos {pos}"
        )),
        5 => {
            let end = checked_end(pos, 4, buf.len(), "fixed32")?;
            let mut arr = [0u8; 4];
            arr.copy_from_slice(&buf[pos..end]);
            Ok((field_no, WireValue::Fixed32(arr), end))
        }
        wt => Err(format!("unknown wire type {wt} at pos {pos}")),
    }
}

/// Read a field tag, returning `(field_no, wire_type, next_pos)`.
///
/// Shared with the streaming parser so both paths validate field numbers identically.
pub(crate) fn read_tag(buf: &[u8], pos: usize) -> Result<(u32, u8, usize), String> {
    let (tag, next) = read_varint(buf, pos)?;
    let raw_field_no = tag >> 3;
    if raw_field_no == 0 || raw_field_no > MAX_FIELD_NUMBER {
        return Err(format!("invalid field number {raw_field_no}"));
    }
    Ok((raw_field_no as u32, (tag & 0x7) as u8, next))
}

/// Read a length-delimited payload whose length prefix starts at `pos`.
///
/// Returns the payload slice and the offset just past it. Every advance is checked,
/// so a hostile length can neither overflow `pos + len` nor escape the buffer.
pub(crate) fn read_len_delim_at<'a>(
    buf: &'a [u8],
    pos: usize,
    what: &str,
) -> Result<(&'a [u8], usize), String> {
    let (len, after_len) = read_varint(buf, pos)?;
    let len = len_to_usize(len, what)?;
    let end = checked_end(after_len, len, buf.len(), what)?;
    Ok((&buf[after_len..end], end))
}

/// Skip a single field value in an in-memory buffer.
///
/// `pos` must point just past the field tag; the returned offset is just past the value.
pub(crate) fn skip_field_value_in_buf(
    buf: &[u8],
    pos: usize,
    field_no: u32,
    wire_type: u8,
) -> Result<usize, String> {
    match wire_type {
        0 => Ok(read_varint(buf, pos)?.1),
        1 => checked_end(pos, 8, buf.len(), "fixed64"),
        2 => {
            let (len, after_len) = read_varint(buf, pos)?;
            let len = len_to_usize(len, "length-delimited")?;
            checked_end(after_len, len, buf.len(), "length-delimited")
        }
        3 => skip_group(buf, pos, field_no, 0),
        4 => Err(format!(
            "unexpected end-group tag for field {field_no} at pos {pos}"
        )),
        5 => checked_end(pos, 4, buf.len(), "fixed32"),
        wt => Err(format!("unknown wire type {wt} at pos {pos}")),
    }
}

/// Skip a group body, returning the offset just past its matching end-group tag.
fn skip_group(
    buf: &[u8],
    mut pos: usize,
    group_field_no: u32,
    depth: u32,
) -> Result<usize, String> {
    if depth >= MAX_NESTING_DEPTH {
        return Err(format!(
            "group nesting exceeds maximum depth {MAX_NESTING_DEPTH}"
        ));
    }
    loop {
        if pos >= buf.len() {
            return Err(format!(
                "group {group_field_no}: unterminated (EOF before end-group tag)"
            ));
        }
        let (tag, after_tag) = read_varint(buf, pos)?;
        if (tag & 0x7) as u8 == 4 {
            let end_field_no = tag >> 3;
            if end_field_no != u64::from(group_field_no) {
                return Err(format!(
                    "group {group_field_no}: mismatched end-group tag for field {end_field_no}"
                ));
            }
            return Ok(after_tag);
        }
        let (_, _, next) = read_field_at(buf, pos, depth + 1)?;
        pos = next;
    }
}

// ─────────────────────────────────────────────────────────────────

/// Decode a packed run of varints, invoking `push` for each decoded value.
fn for_each_packed_varint<F: FnMut(u64)>(b: &[u8], mut push: F) -> Result<(), String> {
    let mut p = 0;
    while p < b.len() {
        let (v, np) = read_varint(b, p)?;
        push(v);
        p = np;
    }
    Ok(())
}

/// Parse a `TensorProto`.
///
/// Field numbers follow onnx.proto exactly:
/// 1 = dims (packed int64), 2 = data_type, 3 = segment, 4 = float_data (packed float),
/// 5 = int32_data (packed varint), 6 = string_data (repeated bytes),
/// 7 = int64_data (packed varint), 8 = name, 9 = raw_data,
/// 10 = double_data (packed fixed64), 11 = uint64_data (packed varint),
/// 13 = external_data, 14 = data_location.
///
/// Every repeated numeric field is accepted in both packed and unpacked form, as
/// required by the protobuf spec (ONNX's own .proto is proto2, where `floats`-style
/// fields are emitted unpacked unless explicitly marked `[packed = true]`).
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
                for_each_packed_varint(b, |v| t.dims.push(v as i64))?;
            }
            (2, WireValue::Varint(v)) => t.data_type = v as i32,
            (3, WireValue::Bytes(b)) => {
                // segment: this TensorProto holds only a slice of the logical tensor.
                // Loading it as if it were complete yields silently wrong data, so
                // refuse it explicitly instead.
                let (begin, end) = parse_tensor_segment(b)?;
                if begin != 0 || end != 0 {
                    return Err(format!(
                        "TensorProto '{}': segmented tensors are not supported (segment begin={begin}, end={end})",
                        t.name
                    ));
                }
            }
            (8, WireValue::Bytes(b)) => t.name = String::from_utf8_lossy(b).into_owned(),
            (4, WireValue::Bytes(b)) => {
                // float_data: packed float32
                t.float_data.reserve(b.len() / 4);
                for chunk in b.chunks_exact(4) {
                    t.float_data
                        .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (4, WireValue::Fixed32(b)) => {
                // float_data: individual (unpacked) float32
                t.float_data.push(f32::from_le_bytes(b));
            }
            (5, WireValue::Bytes(b)) => {
                // int32_data: packed varint (proto int32 → varint, not fixed32)
                for_each_packed_varint(b, |v| t.int32_data.push(v as i32))?;
            }
            (5, WireValue::Varint(v)) => t.int32_data.push(v as i32),
            (6, WireValue::Bytes(b)) => {
                // string_data: repeated bytes — one raw value per occurrence
                t.string_data.push(b.to_vec());
            }
            (7, WireValue::Bytes(b)) => {
                // int64_data: packed varint
                for_each_packed_varint(b, |v| t.int64_data.push(v as i64))?;
            }
            (7, WireValue::Varint(v)) => t.int64_data.push(v as i64),
            (10, WireValue::Bytes(b)) => {
                // double_data: packed float64
                t.double_data.reserve(b.len() / 8);
                for chunk in b.chunks_exact(8) {
                    t.double_data.push(f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]));
                }
            }
            (10, WireValue::Fixed64(b)) => {
                // double_data: individual (unpacked) float64
                t.double_data.push(f64::from_le_bytes(b));
            }
            (11, WireValue::Bytes(b)) => {
                // uint64_data: packed varint
                for_each_packed_varint(b, |v| t.uint64_data.push(v))?;
            }
            (11, WireValue::Varint(v)) => t.uint64_data.push(v),
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

/// Parse a `TensorProto.Segment` sub-message, returning `(begin, end)`.
fn parse_tensor_segment(buf: &[u8]) -> Result<(i64, i64), String> {
    let mut begin = 0i64;
    let mut end = 0i64;
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Varint(v)) => begin = v as i64,
            (2, WireValue::Varint(v)) => end = v as i64,
            _ => {}
        }
    }
    Ok((begin, end))
}

// ─────────────────────────────────────────────────────────────────

/// Parse an `AttributeProto`.
pub fn parse_attribute(buf: &[u8]) -> Result<AttributeProto, String> {
    parse_attribute_at(buf, 0)
}

fn parse_attribute_at(buf: &[u8], depth: u32) -> Result<AttributeProto, String> {
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
                attr.value.g = Some(Box::new(parse_graph_at(b, depth + 1)?));
            }
            (7, WireValue::Bytes(b)) => {
                // floats: packed
                attr.value.floats.reserve(b.len() / 4);
                for chunk in b.chunks_exact(4) {
                    attr.value
                        .floats
                        .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (7, WireValue::Fixed32(b)) => {
                // floats: individual (unpacked) float32 — ONNX's proto2 schema does not
                // mark `floats` as packed, so real writers emit one tag per element.
                attr.value.floats.push(f32::from_le_bytes(b));
            }
            (8, WireValue::Varint(v)) => {
                // ints: individual varint
                attr.value.ints.push(v as i64);
            }
            (8, WireValue::Bytes(b)) => {
                // ints: packed varint
                for_each_packed_varint(b, |v| attr.value.ints.push(v as i64))?;
            }
            (9, WireValue::Bytes(b)) => {
                // AttributeProto.strings is `repeated bytes strings = 9`: each field-9
                // occurrence carries one raw string value directly — it is NOT a nested
                // message. Push the bytes verbatim. Re-parsing them as protobuf mis-decodes
                // any string whose leading bytes resemble a tag + length prefix, e.g.
                // ONNX-ML TreeEnsemble "nodes_modes" entries like "BRANCH_GTE" (0x42 0x52…
                // decodes as field 8 / wire type 2 / length 82, tripping a spurious EOF).
                attr.value
                    .strings
                    .push(String::from_utf8_lossy(b).into_owned());
            }
            (10, WireValue::Bytes(b)) => {
                // tensors (AttributeType::TENSORS)
                attr.value.tensors.push(parse_tensor_proto(b)?);
            }
            (11, WireValue::Bytes(b)) => {
                // graphs (AttributeType::GRAPHS)
                attr.value.graphs.push(parse_graph_at(b, depth + 1)?);
            }
            (14, WireValue::Bytes(_)) | (15, WireValue::Bytes(_)) => {
                return Err(format!(
                    "attribute '{}': TYPE_PROTO attributes (tp / type_protos) are not supported",
                    attr.name
                ));
            }
            (20, WireValue::Varint(v)) => attr.value.attr_type = v as i32,
            (21, WireValue::Bytes(b)) => {
                // ref_attr_name: this attribute takes its value from the enclosing
                // function's attribute of that name.
                attr.value.ref_attr_name = String::from_utf8_lossy(b).into_owned();
            }
            (22, WireValue::Bytes(_)) | (23, WireValue::Bytes(_)) => {
                return Err(format!(
                    "attribute '{}': sparse tensor attributes are not supported",
                    attr.name
                ));
            }
            _ => {}
        }
    }
    Ok(attr)
}

// ─────────────────────────────────────────────────────────────────

/// Parse a `NodeProto`.
pub fn parse_node(buf: &[u8]) -> Result<NodeProto, String> {
    parse_node_at(buf, 0)
}

fn parse_node_at(buf: &[u8], depth: u32) -> Result<NodeProto, String> {
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
            (5, WireValue::Bytes(b)) => node.attributes.push(parse_attribute_at(b, depth)?),
            (7, WireValue::Bytes(b)) => node.domain = String::from_utf8_lossy(b).into_owned(),
            (8, WireValue::Bytes(b)) => node.overload = String::from_utf8_lossy(b).into_owned(),
            _ => {}
        }
    }
    Ok(node)
}

// ─────────────────────────────────────────────────────────────────

/// Parse a `GraphProto`.
pub fn parse_graph(buf: &[u8]) -> Result<GraphProto, String> {
    parse_graph_at(buf, 0)
}

fn parse_graph_at(buf: &[u8], depth: u32) -> Result<GraphProto, String> {
    if depth > MAX_NESTING_DEPTH {
        return Err(format!(
            "graph nesting exceeds maximum depth {MAX_NESTING_DEPTH}"
        ));
    }
    let mut graph = GraphProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Bytes(b)) => graph.nodes.push(parse_node_at(b, depth)?),
            (2, WireValue::Bytes(b)) => graph.name = String::from_utf8_lossy(b).into_owned(),
            (5, WireValue::Bytes(b)) => graph.initializers.push(parse_tensor_proto(b)?),
            (11, WireValue::Bytes(b)) => {
                // ValueInfoProto for graph inputs — parse full info and extract name
                let vi = parse_value_info_proto(b)?;
                graph.inputs.push(vi.name.clone());
                graph.input_value_infos.push(vi);
            }
            (12, WireValue::Bytes(b)) => {
                // ValueInfoProto for graph outputs
                let vi = parse_value_info_proto(b)?;
                graph.outputs.push(vi.name.clone());
                graph.output_value_infos.push(vi);
            }
            (13, WireValue::Bytes(b)) => {
                // value_info: shape/dtype metadata for intermediate values
                graph.value_infos.push(parse_value_info_proto(b)?);
            }
            (15, WireValue::Bytes(_)) => {
                // sparse_initializer: dropping it would leave the graph referencing a
                // tensor that never materialises, so fail with a diagnosable message.
                return Err(format!(
                    "graph '{}': sparse_initializer is not supported",
                    graph.name
                ));
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
pub fn parse_value_info_proto(buf: &[u8]) -> Result<ValueInfoProto, String> {
    let mut vi = ValueInfoProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Bytes(b)) => {
                vi.name = String::from_utf8_lossy(b).into_owned();
            }
            (2, WireValue::Bytes(b)) => {
                // TypeProto — look for tensor_type (field 1)
                parse_type_proto_into(b, &mut vi)?;
            }
            _ => {}
        }
    }
    Ok(vi)
}

/// Parse TypeProto bytes and fill elem_type / shape into `vi`.
fn parse_type_proto_into(buf: &[u8], vi: &mut ValueInfoProto) -> Result<(), String> {
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        if let (1, WireValue::Bytes(b)) = (field, value) {
            // tensor_type: Tensor { 1: elem_type, 2: shape }
            parse_tensor_type_into(b, vi)?;
        }
    }
    Ok(())
}

/// Parse TensorTypeProto bytes and fill elem_type / shape into `vi`.
fn parse_tensor_type_into(buf: &[u8], vi: &mut ValueInfoProto) -> Result<(), String> {
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Varint(v)) => vi.elem_type = v as i32,
            (2, WireValue::Bytes(b)) => {
                // TensorShapeProto — repeated Dimension (field 1)
                parse_shape_proto_into(b, vi)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Parse TensorShapeProto bytes and append dimensions into `vi.shape` and `vi.dim_params`.
fn parse_shape_proto_into(buf: &[u8], vi: &mut ValueInfoProto) -> Result<(), String> {
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        if let (1, WireValue::Bytes(b)) = (field, value) {
            // Dimension: field 1 = dim_value (varint), field 2 = dim_param (string)
            let (dim_value, dim_param) = parse_dimension(b)?;
            vi.shape.push(dim_value);
            vi.dim_params.push(dim_param);
        }
    }
    Ok(())
}

/// Parse a single Dimension message and return (static_size, symbolic_param_name).
///
/// ONNX Dimension fields (a `oneof`):
///   1: dim_value (varint) — concrete size, which may legitimately be 0
///   2: dim_param (string) — symbolic name (e.g. "batch_size")
///
/// A dimension with neither field set is unknown; only that case yields `None`,
/// so an explicit zero-sized dimension is no longer confused with a symbolic one.
fn parse_dimension(buf: &[u8]) -> Result<(Option<i64>, Option<String>), String> {
    let mut dim_value: Option<i64> = None;
    let mut dim_param: Option<String> = None;
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Varint(v)) => {
                dim_value = Some(v as i64);
            }
            (2, WireValue::Bytes(s)) => {
                // dim_param: symbolic dimension name
                let name = String::from_utf8_lossy(s).into_owned();
                if !name.is_empty() {
                    dim_param = Some(name);
                }
            }
            _ => {}
        }
    }
    Ok((dim_value, dim_param))
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
                let import = parse_opset_import(b)?;
                // Backwards compat: set opset_version from default domain
                if import.domain.is_empty() {
                    model.opset_version = import.version;
                }
                model.opset_imports.push(import);
            }
            (14, WireValue::Bytes(b)) => {
                let (key, val) = parse_string_string_entry(b)?;
                model.metadata_props.push((key, val));
            }
            (20, WireValue::Bytes(b)) => {
                model.training_info.push(parse_training_info(b)?);
            }
            (25, WireValue::Bytes(b)) => {
                model.functions.push(parse_function_proto(b)?);
            }
            _ => {}
        }
    }
    Ok(model)
}

// ─── Model-local functions (ModelProto field 25) ──────────────────────────

/// A parsed ONNX `FunctionProto` — one entry of a model's *local function
/// library* (`ModelProto.functions`, wire field 25).
///
/// A local function is a named, reusable subgraph: nodes elsewhere in the model
/// call it by `(domain, op_type)` and the runtime is expected to substitute the
/// body. PyTorch `dynamo` export and opset-18+ ONNX both emit these routinely,
/// and a loader that ignores field 25 turns every such call into an unknown
/// operator even though the model carries a complete body for it.
///
/// Wire layout (`onnx.proto`):
/// ```text
/// FunctionProto {
///    1: name (string)
///    4: input (repeated string)     — formal input names
///    5: output (repeated string)    — formal output names
///    6: attribute (repeated string) — attribute names with no default
///    7: node (repeated NodeProto)   — the body
///    8: doc_string (string)
///    9: opset_import (repeated OperatorSetIdProto)
///   10: domain (string)
///   11: attribute_proto (repeated AttributeProto) — attributes *with* defaults
///   13: overload (string)
/// }
/// ```
#[derive(Debug, Default, Clone)]
pub struct FunctionProto {
    /// Function name; combined with [`domain`](Self::domain) it is what a call
    /// node's `op_type` / `domain` pair matches against.
    pub name: String,
    /// Function domain (`""` = the default ONNX domain).
    pub domain: String,
    /// Formal input names, positionally matched to a call node's inputs.
    pub inputs: Vec<String>,
    /// Formal output names, positionally matched to a call node's outputs.
    pub outputs: Vec<String>,
    /// `attribute` (field 6): attribute names that have **no** declared default.
    /// A body reference to one of these that the call site does not supply
    /// leaves the attribute unset (i.e. the operator's own default applies).
    pub attribute_names: Vec<String>,
    /// `attribute_proto` (field 11): attributes that carry a default value,
    /// used when the call site omits them. This is a *different* field from
    /// `attribute` (field 6), which is only a list of names.
    pub attribute_defaults: Vec<AttributeProto>,
    /// The function body.
    pub nodes: Vec<NodeProto>,
    /// Operator sets the body is written against.
    pub opset_imports: Vec<OpsetImport>,
    /// `overload` (field 13): disambiguates same-name functions (IR ≥ 10).
    pub overload: String,
}

/// Parse one `FunctionProto` message.
pub fn parse_function_proto(buf: &[u8]) -> Result<FunctionProto, String> {
    let mut func = FunctionProto::default();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Bytes(b)) => func.name = String::from_utf8_lossy(b).into_owned(),
            (4, WireValue::Bytes(b)) => func.inputs.push(String::from_utf8_lossy(b).into_owned()),
            (5, WireValue::Bytes(b)) => func.outputs.push(String::from_utf8_lossy(b).into_owned()),
            (6, WireValue::Bytes(b)) => func
                .attribute_names
                .push(String::from_utf8_lossy(b).into_owned()),
            (7, WireValue::Bytes(b)) => func.nodes.push(parse_node(b)?),
            (9, WireValue::Bytes(b)) => func.opset_imports.push(parse_opset_import(b)?),
            (10, WireValue::Bytes(b)) => func.domain = String::from_utf8_lossy(b).into_owned(),
            (11, WireValue::Bytes(b)) => func.attribute_defaults.push(parse_attribute(b)?),
            (13, WireValue::Bytes(b)) => func.overload = String::from_utf8_lossy(b).into_owned(),
            _ => {}
        }
    }
    Ok(func)
}

/// Parse a model's local function library (`ModelProto.functions`, field 25).
///
/// Takes the same buffer as [`parse_model`] and rescans only its *top-level*
/// fields, which is cheap: a length-delimited field is skipped by its length,
/// so the (large) `graph` submessage is never walked.
///
/// [`parse_model`] now collects the same library into [`ModelProto::functions`]
/// in its one pass over `buf`, so `crate::model::load*` reads that field
/// instead of calling this a second time — do not reintroduce the second scan
/// there. This function remains for a caller that holds only raw bytes and
/// wants just the function list, e.g. without paying for a full `parse_model`.
pub fn parse_model_functions(buf: &[u8]) -> Result<Vec<FunctionProto>, String> {
    let mut functions = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        if let (25, WireValue::Bytes(b)) = (field, value) {
            functions.push(parse_function_proto(b)?);
        }
    }
    Ok(functions)
}

/// Parse an `OperatorSetIdProto` (field 1 = domain, field 2 = version).
fn parse_opset_import(buf: &[u8]) -> Result<OpsetImport, String> {
    let mut domain = String::new();
    let mut version = 0i64;
    let mut pos = 0;
    while pos < buf.len() {
        let (field, value, next) = read_field(buf, pos)?;
        pos = next;
        match (field, value) {
            (1, WireValue::Bytes(s)) => domain = String::from_utf8_lossy(s).into_owned(),
            (2, WireValue::Varint(v)) => version = v as i64,
            _ => {}
        }
    }
    Ok(OpsetImport { domain, version })
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

    #[test]
    fn test_issue_3_strings_attribute_repeated_bytes() {
        // Regression for https://github.com/cool-japan/oxionnx/issues/3
        //
        // AttributeProto.strings is `repeated bytes strings = 9`. Each field-9 entry is a
        // single raw string, NOT a nested message. ONNX-ML TreeEnsemble models store the
        // "nodes_modes" attribute this way, with values such as "BRANCH_GTE" / "LEAF".
        //
        // "BRANCH_GTE" begins with bytes 0x42 ('B') 0x52 ('R'): if the entry is (wrongly)
        // re-parsed as protobuf, 0x42 decodes as field 8 / wire type 2 and 0x52 as a length
        // prefix of 82, which overruns the 10-byte string and raised
        // `length-delimited: EOF (need 82, have 8)` — exactly the failure reported in #3.
        //
        // Build a NodeProto (mirroring the real model) whose attribute carries these
        // string values, and assert they are decoded verbatim rather than re-parsed.
        let mut modes_attr = Vec::new();
        modes_attr.extend(encode_bytes_field(1, b"nodes_modes")); // name (field 1)
        modes_attr.extend(encode_bytes_field(9, b"BRANCH_GTE")); // strings (field 9)
        modes_attr.extend(encode_bytes_field(9, b"LEAF")); // strings (field 9)
        modes_attr.extend(encode_bytes_field(9, b"BRANCH_LEQ")); // strings (field 9)
        modes_attr.extend(encode_varint_field(20, 8)); // attr_type = STRINGS (field 20)

        // Verify direct attribute parsing does not error and preserves every string.
        let attr = parse_attribute(&modes_attr).expect("strings attribute must parse");
        assert_eq!(attr.name, "nodes_modes");
        assert_eq!(
            attr.value.strings,
            vec![
                "BRANCH_GTE".to_string(),
                "LEAF".to_string(),
                "BRANCH_LEQ".to_string(),
            ]
        );

        // Verify the full node path (parse_node -> parse_attribute) also succeeds.
        let mut node_bytes = Vec::new();
        node_bytes.extend(encode_bytes_field(1, b"input")); // input
        node_bytes.extend(encode_bytes_field(2, b"variable")); // output
        node_bytes.extend(encode_bytes_field(4, b"TreeEnsembleRegressor")); // op_type
        node_bytes.extend(encode_bytes_field(5, &modes_attr)); // attribute

        let node = parse_node(&node_bytes).expect("node with strings attribute must parse");
        assert_eq!(node.op_type, "TreeEnsembleRegressor");
        assert_eq!(node.attributes.len(), 1);
        assert_eq!(
            node.attributes[0].value.strings,
            vec![
                "BRANCH_GTE".to_string(),
                "LEAF".to_string(),
                "BRANCH_LEQ".to_string(),
            ]
        );
    }
}
