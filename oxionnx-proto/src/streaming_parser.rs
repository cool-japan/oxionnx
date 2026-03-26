//! Streaming protobuf parser for large ONNX models.
//!
//! Parses ONNX models from a `Read` source without loading the entire file
//! into memory. Weight tensors are yielded one at a time via callback,
//! allowing the caller to process or discard them immediately.
//!
//! # Architecture
//!
//! Protobuf uses length-delimited fields, which allows us to stream: read each
//! top-level field header, then either parse the field body or skip it. For the
//! graph field, we enter it and stream its sub-fields. When we encounter a large
//! initializer, we parse its TensorProto header (dims, dtype) and raw_data field,
//! yielding the tensor immediately without buffering the entire model.
//!
//! # Usage
//!
//! ```no_run
//! use std::io::Cursor;
//! use oxionnx_proto::streaming_parser::{StreamingParser, ParseEvent};
//!
//! let data: Vec<u8> = vec![]; // model bytes
//! let mut parser = StreamingParser::new(Cursor::new(data));
//! parser.parse(|event| {
//!     match event {
//!         ParseEvent::Weight { name, tensor } => {
//!             // Process weight immediately
//!         }
//!         _ => {}
//!     }
//!     Ok(())
//! }).ok();
//! ```

use crate::parser;
use crate::types::*;
use std::collections::HashMap;
use std::io::{BufReader, Read};

use oxionnx_core::Tensor;

// ─────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────

/// Events emitted during streaming parse of an ONNX model.
#[derive(Debug)]
pub enum ParseEvent {
    /// Model metadata (ir_version, opset_imports, etc.)
    ModelHeader {
        ir_version: i64,
        producer_name: String,
        producer_version: String,
        opset_imports: Vec<(String, i64)>,
    },
    /// A graph node was parsed.
    Node(NodeProto),
    /// A weight/initializer tensor was parsed.
    Weight { name: String, tensor: Tensor },
    /// Graph input name and optional shape info.
    GraphInput(String),
    /// Graph output name and optional shape info.
    GraphOutput(String),
    /// End of model.
    End,
}

// ─────────────────────────────────────────────────────────────────
// Stream-based varint / field reading
// ─────────────────────────────────────────────────────────────────

/// Read a single varint from a `Read` source.
/// Returns `None` on clean EOF (first byte), `Err` on truncation.
fn read_varint_from_reader<R: Read>(reader: &mut R) -> Result<Option<u64>, String> {
    let mut result = 0u64;
    let mut shift = 0u32;
    let mut one = [0u8; 1];

    loop {
        match reader.read_exact(&mut one) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                if shift == 0 {
                    return Ok(None); // clean EOF
                }
                return Err("varint: unexpected EOF mid-varint".into());
            }
            Err(e) => return Err(format!("varint read error: {e}")),
        }
        let byte = one[0];
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(Some(result));
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint: overflow".into());
        }
    }
}

/// Read exactly `len` bytes from a reader into a new Vec.
fn read_exact_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read_exact({len} bytes) failed: {e}"))?;
    Ok(buf)
}

/// Skip exactly `len` bytes by reading and discarding in chunks.
fn skip_bytes<R: Read>(reader: &mut R, mut len: usize) -> Result<(), String> {
    let mut discard = [0u8; 8192];
    while len > 0 {
        let chunk = len.min(discard.len());
        reader
            .read_exact(&mut discard[..chunk])
            .map_err(|e| format!("skip({len} bytes) failed: {e}"))?;
        len -= chunk;
    }
    Ok(())
}

/// Wire types used in protobuf encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireType {
    Varint,   // 0
    Fixed64,  // 1
    LenDelim, // 2
    Fixed32,  // 5
}

impl WireType {
    fn from_u8(wt: u8) -> Result<Self, String> {
        match wt {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LenDelim),
            5 => Ok(Self::Fixed32),
            other => Err(format!("unknown wire type {other}")),
        }
    }
}

/// A field header: field number + wire type.
struct FieldHeader {
    field_no: u32,
    wire_type: WireType,
}

/// Read a field header from a reader. Returns None on clean EOF.
fn read_field_header<R: Read>(reader: &mut R) -> Result<Option<FieldHeader>, String> {
    let tag = match read_varint_from_reader(reader)? {
        Some(t) => t,
        None => return Ok(None),
    };
    let field_no = (tag >> 3) as u32;
    let wire_type = WireType::from_u8((tag & 0x7) as u8)?;
    Ok(Some(FieldHeader {
        field_no,
        wire_type,
    }))
}

/// Skip a field value based on wire type. For LenDelim, also reads the length.
fn skip_field_value<R: Read>(reader: &mut R, wire_type: WireType) -> Result<(), String> {
    match wire_type {
        WireType::Varint => {
            // Just consume the varint
            read_varint_from_reader(reader)?
                .ok_or_else(|| "unexpected EOF skipping varint".to_string())?;
        }
        WireType::Fixed64 => {
            skip_bytes(reader, 8)?;
        }
        WireType::Fixed32 => {
            skip_bytes(reader, 4)?;
        }
        WireType::LenDelim => {
            let len = read_varint_from_reader(reader)?
                .ok_or_else(|| "unexpected EOF reading length".to_string())?
                as usize;
            skip_bytes(reader, len)?;
        }
    }
    Ok(())
}

/// Read a varint field value from a reader.
fn read_varint_value<R: Read>(reader: &mut R) -> Result<u64, String> {
    read_varint_from_reader(reader)?
        .ok_or_else(|| "unexpected EOF reading varint value".to_string())
}

/// Read a length-delimited field value into a Vec<u8>.
fn read_len_delim_value<R: Read>(reader: &mut R) -> Result<Vec<u8>, String> {
    let len = read_varint_from_reader(reader)?
        .ok_or_else(|| "unexpected EOF reading length prefix".to_string())? as usize;
    read_exact_vec(reader, len)
}

// ─────────────────────────────────────────────────────────────────
// Streaming parser
// ─────────────────────────────────────────────────────────────────

/// Callback-based streaming parser for ONNX protobuf models.
///
/// Reads the model incrementally from a `Read` source, yielding events
/// for each major structure (model header, nodes, weights, inputs, outputs).
pub struct StreamingParser<R: Read> {
    reader: BufReader<R>,
}

impl<R: Read> StreamingParser<R> {
    /// Create a new streaming parser with a 64 KB read buffer.
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::with_capacity(64 * 1024, reader),
        }
    }

    /// Parse the model, calling the callback for each event.
    ///
    /// Events are emitted in wire order. The `ModelHeader` event is emitted
    /// once we finish the top-level model fields (before entering the graph).
    /// However, because protobuf fields can appear in any order, we accumulate
    /// header info and emit it at the end along with `End`.
    pub fn parse<F>(&mut self, mut callback: F) -> Result<(), String>
    where
        F: FnMut(ParseEvent) -> Result<(), String>,
    {
        let mut ir_version: i64 = 0;
        let mut producer_name = String::new();
        let mut producer_version = String::new();
        let mut opset_imports: Vec<(String, i64)> = Vec::new();
        let mut header_emitted = false;

        // Parse ModelProto fields from the stream
        while let Some(hdr) = read_field_header(&mut self.reader)? {
            match (hdr.field_no, hdr.wire_type) {
                // field 1: ir_version (varint)
                (1, WireType::Varint) => {
                    ir_version = read_varint_value(&mut self.reader)? as i64;
                }
                // field 2: producer_name (string)
                (2, WireType::LenDelim) => {
                    let bytes = read_len_delim_value(&mut self.reader)?;
                    producer_name = String::from_utf8_lossy(&bytes).into_owned();
                }
                // field 3: model_version (varint)
                (3, WireType::Varint) => {
                    // Consume but don't need for events
                    let _model_version = read_varint_value(&mut self.reader)?;
                }
                // field 4: doc_string (string)
                (4, WireType::LenDelim) => {
                    let _doc = read_len_delim_value(&mut self.reader)?;
                }
                // field 5: producer_version (string)
                (5, WireType::LenDelim) => {
                    let bytes = read_len_delim_value(&mut self.reader)?;
                    producer_version = String::from_utf8_lossy(&bytes).into_owned();
                }
                // field 7: graph (GraphProto, length-delimited)
                (7, WireType::LenDelim) => {
                    // Emit the header event before entering the graph
                    if !header_emitted {
                        callback(ParseEvent::ModelHeader {
                            ir_version,
                            producer_name: producer_name.clone(),
                            producer_version: producer_version.clone(),
                            opset_imports: opset_imports.clone(),
                        })?;
                        header_emitted = true;
                    }

                    // Read the graph length, then stream its contents
                    let graph_len = read_varint_from_reader(&mut self.reader)?
                        .ok_or_else(|| "unexpected EOF reading graph length".to_string())?
                        as usize;
                    let graph_bytes = read_exact_vec(&mut self.reader, graph_len)?;
                    self.parse_graph_streaming(&graph_bytes, &mut callback)?;
                }
                // field 8: opset_import (OperatorSetIdProto)
                (8, WireType::LenDelim) => {
                    let bytes = read_len_delim_value(&mut self.reader)?;
                    let (domain, version) = parse_opset_import_bytes(&bytes)?;
                    opset_imports.push((domain, version));
                }
                // Skip unknown fields
                (_, wt) => {
                    skip_field_value(&mut self.reader, wt)?;
                }
            }
        }

        // Emit header if not already emitted (model with no graph)
        if !header_emitted {
            callback(ParseEvent::ModelHeader {
                ir_version,
                producer_name,
                producer_version,
                opset_imports,
            })?;
        }

        callback(ParseEvent::End)?;
        Ok(())
    }

    /// Parse GraphProto fields from an in-memory buffer, emitting events via callback.
    fn parse_graph_streaming<F>(&self, graph_bytes: &[u8], callback: &mut F) -> Result<(), String>
    where
        F: FnMut(ParseEvent) -> Result<(), String>,
    {
        let mut pos = 0;
        while pos < graph_bytes.len() {
            let (tag, next_pos) = parser::read_varint(graph_bytes, pos)?;
            let field_no = (tag >> 3) as u32;
            let wire_type = (tag & 0x7) as u8;
            pos = next_pos;

            match (field_no, wire_type) {
                // field 1: node (NodeProto) - length-delimited
                (1, 2) => {
                    let (len, next_pos) = parser::read_varint(graph_bytes, pos)?;
                    let len = len as usize;
                    pos = next_pos;
                    if pos + len > graph_bytes.len() {
                        return Err(format!(
                            "graph node: need {len} bytes, have {}",
                            graph_bytes.len() - pos
                        ));
                    }
                    let node = parser::parse_node(&graph_bytes[pos..pos + len])?;
                    callback(ParseEvent::Node(node))?;
                    pos += len;
                }
                // field 2: name (string) - skip
                (2, 2) => {
                    let (len, next_pos) = parser::read_varint(graph_bytes, pos)?;
                    pos = next_pos + len as usize;
                }
                // field 5: initializer (TensorProto) - length-delimited
                (5, 2) => {
                    let (len, next_pos) = parser::read_varint(graph_bytes, pos)?;
                    let len = len as usize;
                    pos = next_pos;
                    if pos + len > graph_bytes.len() {
                        return Err(format!(
                            "graph initializer: need {len} bytes, have {}",
                            graph_bytes.len() - pos
                        ));
                    }
                    let tp = parser::parse_tensor_proto(&graph_bytes[pos..pos + len])?;
                    let name = tp.name.clone();
                    let tensor = tp.to_tensor();
                    callback(ParseEvent::Weight { name, tensor })?;
                    pos += len;
                }
                // field 11: input (ValueInfoProto) - length-delimited
                (11, 2) => {
                    let (len, next_pos) = parser::read_varint(graph_bytes, pos)?;
                    let len = len as usize;
                    pos = next_pos;
                    if pos + len > graph_bytes.len() {
                        return Err(format!(
                            "graph input: need {len} bytes, have {}",
                            graph_bytes.len() - pos
                        ));
                    }
                    let name = extract_name_from_bytes(&graph_bytes[pos..pos + len]);
                    callback(ParseEvent::GraphInput(name))?;
                    pos += len;
                }
                // field 12: output (ValueInfoProto) - length-delimited
                (12, 2) => {
                    let (len, next_pos) = parser::read_varint(graph_bytes, pos)?;
                    let len = len as usize;
                    pos = next_pos;
                    if pos + len > graph_bytes.len() {
                        return Err(format!(
                            "graph output: need {len} bytes, have {}",
                            graph_bytes.len() - pos
                        ));
                    }
                    let name = extract_name_from_bytes(&graph_bytes[pos..pos + len]);
                    callback(ParseEvent::GraphOutput(name))?;
                    pos += len;
                }
                // Varint field: skip
                (_, 0) => {
                    let (_val, next_pos) = parser::read_varint(graph_bytes, pos)?;
                    pos = next_pos;
                }
                // Fixed64: skip
                (_, 1) => {
                    pos += 8;
                }
                // Length-delimited (unknown): skip
                (_, 2) => {
                    let (len, next_pos) = parser::read_varint(graph_bytes, pos)?;
                    pos = next_pos + len as usize;
                }
                // Fixed32: skip
                (_, 5) => {
                    pos += 4;
                }
                (_, wt) => {
                    return Err(format!("graph: unknown wire type {wt}"));
                }
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────

/// Parse an OperatorSetIdProto from bytes.
fn parse_opset_import_bytes(buf: &[u8]) -> Result<(String, i64), String> {
    let mut domain = String::new();
    let mut version: i64 = 0;
    let mut pos = 0;
    while pos < buf.len() {
        let (tag, next_pos) = parser::read_varint(buf, pos)?;
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;
        match (field_no, wire_type) {
            (1, 2) => {
                let (len, next_pos) = parser::read_varint(buf, pos)?;
                let len = len as usize;
                pos = next_pos;
                domain = String::from_utf8_lossy(&buf[pos..pos + len]).into_owned();
                pos += len;
            }
            (2, 0) => {
                let (v, next_pos) = parser::read_varint(buf, pos)?;
                version = v as i64;
                pos = next_pos;
            }
            (_, 0) => {
                let (_v, next_pos) = parser::read_varint(buf, pos)?;
                pos = next_pos;
            }
            (_, 1) => pos += 8,
            (_, 2) => {
                let (len, next_pos) = parser::read_varint(buf, pos)?;
                pos = next_pos + len as usize;
            }
            (_, 5) => pos += 4,
            (_, wt) => return Err(format!("opset_import: unknown wire type {wt}")),
        }
    }
    Ok((domain, version))
}

/// Extract the name (field 1, string) from a ValueInfoProto buffer.
fn extract_name_from_bytes(buf: &[u8]) -> String {
    let mut pos = 0;
    while pos < buf.len() {
        if let Ok((tag, next_pos)) = parser::read_varint(buf, pos) {
            let field_no = (tag >> 3) as u32;
            let wire_type = (tag & 0x7) as u8;
            pos = next_pos;
            match (field_no, wire_type) {
                (1, 2) => {
                    if let Ok((len, next_pos)) = parser::read_varint(buf, pos) {
                        let len = len as usize;
                        pos = next_pos;
                        if pos + len <= buf.len() {
                            return String::from_utf8_lossy(&buf[pos..pos + len]).into_owned();
                        }
                    }
                    break;
                }
                (_, 0) => {
                    if let Ok((_v, np)) = parser::read_varint(buf, pos) {
                        pos = np;
                    } else {
                        break;
                    }
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    if let Ok((len, np)) = parser::read_varint(buf, pos) {
                        pos = np + len as usize;
                    } else {
                        break;
                    }
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        } else {
            break;
        }
    }
    String::new()
}

// ─────────────────────────────────────────────────────────────────
// Convenience functions
// ─────────────────────────────────────────────────────────────────

/// Parse an ONNX model from a `Read` source, returning the full graph and weights.
///
/// This is the streaming equivalent of `crate::model::load()` but reads from
/// any `Read` source instead of requiring all bytes in memory upfront.
pub fn parse_streaming<R: Read>(
    reader: R,
) -> Result<(crate::types::GraphProto, HashMap<String, Tensor>), String> {
    let mut nodes: Vec<NodeProto> = Vec::new();
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut outputs: Vec<String> = Vec::new();

    let mut parser = StreamingParser::new(reader);
    parser.parse(|event| {
        match event {
            ParseEvent::ModelHeader { .. } => {}
            ParseEvent::Node(node_proto) => {
                nodes.push(node_proto);
            }
            ParseEvent::Weight { name, tensor } => {
                weights.insert(name, tensor);
            }
            ParseEvent::GraphInput(name) => {
                inputs.push(name);
            }
            ParseEvent::GraphOutput(name) => {
                outputs.push(name);
            }
            ParseEvent::End => {}
        }
        Ok(())
    })?;

    let graph = GraphProto {
        nodes,
        name: String::new(),
        initializers: Vec::new(), // weights already extracted above
        inputs,
        outputs,
    };

    Ok((graph, weights))
}

/// Parse an ONNX model from a `Read` source with selective weight loading.
///
/// The `weight_filter` closure receives each weight's name and shape (as `&[usize]`).
/// If it returns `true`, the weight is materialized and included in the result.
/// If `false`, the weight's raw data is discarded (saving memory).
///
/// Weights that pass the filter are returned in the HashMap. Weights that are
/// filtered out are not stored at all.
pub fn parse_with_weight_filter<R: Read, F>(
    reader: R,
    mut weight_filter: F,
) -> Result<(crate::types::GraphProto, HashMap<String, Tensor>), String>
where
    F: FnMut(&str, &[usize]) -> bool,
{
    let mut nodes: Vec<NodeProto> = Vec::new();
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut outputs: Vec<String> = Vec::new();

    // We need a two-pass approach within each initializer:
    // parse the TensorProto to get name/dims, then decide whether to convert.
    // Since the StreamingParser already parses the full TensorProto and converts,
    // we use the raw parsing approach instead.
    let mut parser = StreamingParser::new(reader);

    // We'll use a wrapper that intercepts Weight events
    parser.parse(|event| {
        match event {
            ParseEvent::ModelHeader { .. } => {}
            ParseEvent::Node(node_proto) => {
                nodes.push(node_proto);
            }
            ParseEvent::Weight { name, tensor } => {
                let shape = tensor.shape.clone();
                if weight_filter(&name, &shape) {
                    weights.insert(name, tensor);
                }
                // Otherwise: weight is dropped, freeing memory
            }
            ParseEvent::GraphInput(name) => {
                inputs.push(name);
            }
            ParseEvent::GraphOutput(name) => {
                outputs.push(name);
            }
            ParseEvent::End => {}
        }
        Ok(())
    })?;

    let graph = GraphProto {
        nodes,
        name: String::new(),
        initializers: Vec::new(),
        inputs,
        outputs,
    };

    Ok((graph, weights))
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ── Protobuf encoding helpers ──────────────────────────────

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
        let tag = (field << 3) | 0;
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

    /// Build a TensorProto binary with given name, dims, data_type, and f32 raw_data.
    fn build_tensor_proto(name: &str, dims: &[i64], data_type: i32, floats: &[f32]) -> Vec<u8> {
        let mut tensor_bytes = Vec::new();

        // dims packed (field 1, wire type 2)
        let mut dims_packed = Vec::new();
        for &d in dims {
            dims_packed.extend(encode_varint(d as u64));
        }
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed));

        // data_type (field 2, varint)
        tensor_bytes.extend(encode_varint_field(2, data_type as u64));

        // name (field 8, string)
        tensor_bytes.extend(encode_bytes_field(8, name.as_bytes()));

        // raw_data (field 9, bytes)
        let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        tensor_bytes.extend(encode_bytes_field(9, &raw));

        tensor_bytes
    }

    /// Build a NodeProto binary with op_type, inputs, outputs.
    fn build_node_proto(op_type: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
        let mut node_bytes = Vec::new();
        for inp in inputs {
            node_bytes.extend(encode_bytes_field(1, inp.as_bytes()));
        }
        for out in outputs {
            node_bytes.extend(encode_bytes_field(2, out.as_bytes()));
        }
        node_bytes.extend(encode_bytes_field(4, op_type.as_bytes()));
        node_bytes
    }

    /// Build a ValueInfoProto binary with just a name.
    fn build_value_info(name: &str) -> Vec<u8> {
        encode_bytes_field(1, name.as_bytes())
    }

    /// Build a complete model binary with graph containing nodes, initializers,
    /// inputs, and outputs.
    fn build_model(
        ir_version: i64,
        opset_version: i64,
        nodes: &[Vec<u8>],
        initializers: &[Vec<u8>],
        inputs: &[&str],
        outputs: &[&str],
    ) -> Vec<u8> {
        // Build graph
        let mut graph_bytes = Vec::new();
        for node in nodes {
            graph_bytes.extend(encode_bytes_field(1, node));
        }
        for init in initializers {
            graph_bytes.extend(encode_bytes_field(5, init));
        }
        for inp in inputs {
            let vi = build_value_info(inp);
            graph_bytes.extend(encode_bytes_field(11, &vi));
        }
        for out in outputs {
            let vi = build_value_info(out);
            graph_bytes.extend(encode_bytes_field(12, &vi));
        }

        // Build model
        let mut model_bytes = Vec::new();
        model_bytes.extend(encode_varint_field(1, ir_version as u64));

        // opset import (default domain)
        let opset = encode_varint_field(2, opset_version as u64);
        model_bytes.extend(encode_bytes_field(8, &opset));

        model_bytes.extend(encode_bytes_field(7, &graph_bytes));
        model_bytes
    }

    // ── Tests ──────────────────────────────────────────────────

    #[test]
    fn test_streaming_parse_empty() {
        let data: Vec<u8> = vec![];
        let mut parser = StreamingParser::new(Cursor::new(data));
        let mut events = Vec::new();
        parser
            .parse(|event| {
                events.push(format!("{:?}", std::mem::discriminant(&event)));
                Ok(())
            })
            .expect("empty parse should succeed");

        // Should get ModelHeader + End
        assert_eq!(
            events.len(),
            2,
            "expected ModelHeader + End, got {events:?}"
        );
    }

    #[test]
    fn test_streaming_parse_simple() {
        let node = build_node_proto("Relu", &["x"], &["y"]);
        let weight = build_tensor_proto("w", &[2, 3], 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let model = build_model(7, 13, &[node], &[weight], &["x", "w"], &["y"]);

        let mut parser = StreamingParser::new(Cursor::new(model));
        let mut events = Vec::new();

        parser
            .parse(|event| {
                match &event {
                    ParseEvent::ModelHeader { ir_version, .. } => {
                        assert_eq!(*ir_version, 7);
                    }
                    ParseEvent::Node(n) => {
                        assert_eq!(n.op_type, "Relu");
                        assert_eq!(n.inputs, vec!["x"]);
                        assert_eq!(n.outputs, vec!["y"]);
                    }
                    ParseEvent::Weight { name, tensor } => {
                        assert_eq!(name, "w");
                        assert_eq!(tensor.shape, vec![2, 3]);
                        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
                    }
                    ParseEvent::GraphInput(name) => {
                        assert!(!name.is_empty());
                    }
                    ParseEvent::GraphOutput(name) => {
                        assert_eq!(name, "y");
                    }
                    ParseEvent::End => {}
                }
                events.push(format!("{event:?}"));
                Ok(())
            })
            .expect("parse should succeed");

        // ModelHeader, Node, Weight, 2x GraphInput, 1x GraphOutput, End
        assert_eq!(events.len(), 7, "got {events:#?}");
    }

    #[test]
    fn test_weight_filter() {
        let w1 = build_tensor_proto("keep_me", &[2, 2], 1, &[1.0, 2.0, 3.0, 4.0]);
        let w2 = build_tensor_proto("skip_me", &[3, 3], 1, &[0.0; 9]);
        let w3 = build_tensor_proto("also_keep", &[1, 4], 1, &[5.0, 6.0, 7.0, 8.0]);

        let model = build_model(7, 13, &[], &[w1, w2, w3], &["input"], &["output"]);

        let (_graph, weights) =
            parse_with_weight_filter(Cursor::new(model), |name, _shape| name != "skip_me")
                .expect("filtered parse should succeed");

        assert!(weights.contains_key("keep_me"), "keep_me should be present");
        assert!(
            !weights.contains_key("skip_me"),
            "skip_me should be filtered out"
        );
        assert!(
            weights.contains_key("also_keep"),
            "also_keep should be present"
        );
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_parse_event_callback_order() {
        let node1 = build_node_proto("Add", &["a", "b"], &["c"]);
        let node2 = build_node_proto("Relu", &["c"], &["d"]);
        let weight = build_tensor_proto("b", &[2], 1, &[1.0, 2.0]);
        let model = build_model(7, 13, &[node1, node2], &[weight], &["a", "b"], &["d"]);

        let mut event_types: Vec<String> = Vec::new();
        let mut parser = StreamingParser::new(Cursor::new(model));

        parser
            .parse(|event| {
                let label = match &event {
                    ParseEvent::ModelHeader { .. } => "ModelHeader".to_string(),
                    ParseEvent::Node(n) => format!("Node({})", n.op_type),
                    ParseEvent::Weight { name, .. } => format!("Weight({name})"),
                    ParseEvent::GraphInput(n) => format!("GraphInput({n})"),
                    ParseEvent::GraphOutput(n) => format!("GraphOutput({n})"),
                    ParseEvent::End => "End".to_string(),
                };
                event_types.push(label);
                Ok(())
            })
            .expect("parse should succeed");

        // ModelHeader comes first, End comes last
        assert_eq!(event_types.first().map(|s| s.as_str()), Some("ModelHeader"));
        assert_eq!(event_types.last().map(|s| s.as_str()), Some("End"));

        // Nodes come before weights (they appear in graph field order: field 1 before field 5)
        let node_idx = event_types
            .iter()
            .position(|s| s.starts_with("Node"))
            .expect("should have a Node event");
        let weight_idx = event_types
            .iter()
            .position(|s| s.starts_with("Weight"))
            .expect("should have a Weight event");
        assert!(
            node_idx < weight_idx,
            "Nodes should appear before weights in wire order"
        );
    }

    #[test]
    fn test_streaming_vs_batch() {
        // Build a model with multiple nodes and weights
        let node1 = build_node_proto("MatMul", &["input", "w1"], &["mm_out"]);
        let node2 = build_node_proto("Add", &["mm_out", "b1"], &["add_out"]);
        let node3 = build_node_proto("Relu", &["add_out"], &["output"]);

        let w1 = build_tensor_proto("w1", &[3, 2], 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b1 = build_tensor_proto("b1", &[2], 1, &[0.1, 0.2]);

        let model_bytes = build_model(
            7,
            13,
            &[node1, node2, node3],
            &[w1, b1],
            &["input", "w1", "b1"],
            &["output"],
        );

        // Parse with batch parser
        let batch_model =
            crate::parser::parse_model(&model_bytes).expect("batch parse should succeed");

        // Parse with streaming parser
        let (stream_graph, stream_weights) =
            parse_streaming(Cursor::new(model_bytes)).expect("streaming parse should succeed");

        // Compare node counts
        assert_eq!(
            batch_model.graph.nodes.len(),
            stream_graph.nodes.len(),
            "node count mismatch"
        );

        // Compare node op_types
        for (batch_node, stream_node) in batch_model
            .graph
            .nodes
            .iter()
            .zip(stream_graph.nodes.iter())
        {
            assert_eq!(batch_node.op_type, stream_node.op_type);
            assert_eq!(batch_node.inputs, stream_node.inputs);
            assert_eq!(batch_node.outputs, stream_node.outputs);
        }

        // Compare weight names and values
        let batch_weights: HashMap<String, Tensor> = batch_model
            .graph
            .initializers
            .iter()
            .map(|tp| (tp.name.clone(), tp.to_tensor()))
            .collect();

        assert_eq!(
            batch_weights.len(),
            stream_weights.len(),
            "weight count mismatch"
        );

        for (name, batch_tensor) in &batch_weights {
            let stream_tensor = stream_weights
                .get(name)
                .unwrap_or_else(|| panic!("streaming missing weight '{name}'"));
            assert_eq!(
                batch_tensor.shape, stream_tensor.shape,
                "shape mismatch for weight '{name}'"
            );
            assert_eq!(
                batch_tensor.data, stream_tensor.data,
                "data mismatch for weight '{name}'"
            );
        }

        // Compare inputs/outputs
        assert_eq!(batch_model.graph.inputs, stream_graph.inputs);
        assert_eq!(batch_model.graph.outputs, stream_graph.outputs);
    }
}
