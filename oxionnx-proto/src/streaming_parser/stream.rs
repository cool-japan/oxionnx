//! `StreamingParser` struct and its main parse logic.

use crate::parser;
use std::io::{BufReader, Read};

use super::helpers::parse_opset_import_bytes;
use super::types::ParseEvent;
use super::wire::{
    len_to_usize, read_exact_vec, read_field_header, read_len_delim_value, read_varint_from_reader,
    read_varint_value, skip_field_value, WireType,
};

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
        let mut model_version: i64 = 0;
        let mut producer_name = String::new();
        let mut producer_version = String::new();
        let mut opset_imports: Vec<(String, i64)> = Vec::new();
        let mut header_emitted = false;

        // Parse ModelProto fields from the stream. Field numbers follow onnx.proto
        // and must stay identical to the eager parser in `crate::parser`:
        //   1 = ir_version, 2 = producer_name, 3 = producer_version, 4 = domain,
        //   5 = model_version, 6 = doc_string, 7 = graph, 8 = opset_import.
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
                // field 3: producer_version (string)
                (3, WireType::LenDelim) => {
                    let bytes = read_len_delim_value(&mut self.reader)?;
                    producer_version = String::from_utf8_lossy(&bytes).into_owned();
                }
                // field 4: domain (string)
                (4, WireType::LenDelim) => {
                    let _domain = read_len_delim_value(&mut self.reader)?;
                }
                // field 5: model_version (varint)
                (5, WireType::Varint) => {
                    model_version = read_varint_value(&mut self.reader)? as i64;
                }
                // field 6: doc_string (string)
                (6, WireType::LenDelim) => {
                    let _doc = read_len_delim_value(&mut self.reader)?;
                }
                // field 7: graph (GraphProto, length-delimited)
                (7, WireType::LenDelim) => {
                    // Emit the header event before entering the graph
                    if !header_emitted {
                        callback(ParseEvent::ModelHeader {
                            ir_version,
                            model_version,
                            producer_name: producer_name.clone(),
                            producer_version: producer_version.clone(),
                            opset_imports: opset_imports.clone(),
                        })?;
                        header_emitted = true;
                    }

                    // Read the graph length, then stream its contents
                    let graph_len = read_varint_from_reader(&mut self.reader)?
                        .ok_or_else(|| "unexpected EOF reading graph length".to_string())?;
                    let graph_len = len_to_usize(graph_len, "graph")?;
                    let graph_bytes = read_exact_vec(&mut self.reader, graph_len)?;
                    self.parse_graph_streaming(&graph_bytes, &mut callback)?;
                }
                // field 8: opset_import (OperatorSetIdProto)
                //
                // Emitted as its own event because field 8 sorts after field 7 (graph),
                // so a header emitted before the graph cannot carry these.
                (8, WireType::LenDelim) => {
                    let bytes = read_len_delim_value(&mut self.reader)?;
                    let (domain, version) = parse_opset_import_bytes(&bytes)?;
                    // `ModelHeader` stays the first event of the stream.
                    if !header_emitted {
                        callback(ParseEvent::ModelHeader {
                            ir_version,
                            model_version,
                            producer_name: producer_name.clone(),
                            producer_version: producer_version.clone(),
                            opset_imports: opset_imports.clone(),
                        })?;
                        header_emitted = true;
                    }
                    callback(ParseEvent::OpsetImport {
                        domain: domain.clone(),
                        version,
                    })?;
                    opset_imports.push((domain, version));
                }
                // field 25: functions (FunctionProto) — the model-local function
                // library. Sorts after field 7 (graph) like opset_import above, but
                // is handled with the same order-tolerant guard rather than an
                // assumption about wire order.
                (25, WireType::LenDelim) => {
                    if !header_emitted {
                        callback(ParseEvent::ModelHeader {
                            ir_version,
                            model_version,
                            producer_name: producer_name.clone(),
                            producer_version: producer_version.clone(),
                            opset_imports: opset_imports.clone(),
                        })?;
                        header_emitted = true;
                    }
                    let bytes = read_len_delim_value(&mut self.reader)?;
                    callback(ParseEvent::LocalFunction(parser::parse_function_proto(
                        &bytes,
                    )?))?;
                }
                // Skip unknown fields
                (_, wt) => {
                    skip_field_value(&mut self.reader, wt, hdr.field_no)?;
                }
            }
        }

        // Emit header if not already emitted (model with no graph)
        if !header_emitted {
            callback(ParseEvent::ModelHeader {
                ir_version,
                model_version,
                producer_name,
                producer_version,
                opset_imports,
            })?;
        }

        callback(ParseEvent::End)?;
        Ok(())
    }

    /// Parse GraphProto fields from an in-memory buffer, emitting events via callback.
    ///
    /// Every offset advance goes through `parser::read_len_delim_at` /
    /// `parser::skip_field_value_in_buf`, which use checked arithmetic — a hostile
    /// length can neither overflow `pos + len` (panicking in debug, wrapping into an
    /// infinite re-read loop in release) nor slice past the buffer.
    fn parse_graph_streaming<F>(&self, graph_bytes: &[u8], callback: &mut F) -> Result<(), String>
    where
        F: FnMut(ParseEvent) -> Result<(), String>,
    {
        let mut pos = 0;
        while pos < graph_bytes.len() {
            let (field_no, wire_type, next_pos) = parser::read_tag(graph_bytes, pos)?;
            pos = next_pos;

            match (field_no, wire_type) {
                // field 1: node (NodeProto) - length-delimited
                (1, 2) => {
                    let (body, next) = parser::read_len_delim_at(graph_bytes, pos, "graph node")?;
                    pos = next;
                    callback(ParseEvent::Node(parser::parse_node(body)?))?;
                }
                // field 2: name (string)
                (2, 2) => {
                    let (body, next) = parser::read_len_delim_at(graph_bytes, pos, "graph name")?;
                    pos = next;
                    callback(ParseEvent::GraphName(
                        String::from_utf8_lossy(body).into_owned(),
                    ))?;
                }
                // field 5: initializer (TensorProto) - length-delimited
                (5, 2) => {
                    let (body, next) =
                        parser::read_len_delim_at(graph_bytes, pos, "graph initializer")?;
                    pos = next;
                    let tp = parser::parse_tensor_proto(body)?;
                    let name = tp.name.clone();
                    // Fallible, dtype-aware decode: a streaming load must report a
                    // malformed initializer as an error, the same as the eager
                    // `load()`/`load_with_path()` path — never substitute a
                    // logged placeholder (the infallible `to_tensor()` shim).
                    let tensor = tp.try_to_tensor().map_err(|e| e.to_string())?;
                    callback(ParseEvent::Weight { name, tensor })?;
                }
                // field 11: input (ValueInfoProto) - length-delimited
                (11, 2) => {
                    let (body, next) = parser::read_len_delim_at(graph_bytes, pos, "graph input")?;
                    pos = next;
                    callback(ParseEvent::GraphInput(parser::parse_value_info_proto(
                        body,
                    )?))?;
                }
                // field 12: output (ValueInfoProto) - length-delimited
                (12, 2) => {
                    let (body, next) = parser::read_len_delim_at(graph_bytes, pos, "graph output")?;
                    pos = next;
                    callback(ParseEvent::GraphOutput(parser::parse_value_info_proto(
                        body,
                    )?))?;
                }
                // field 13: value_info (ValueInfoProto) - length-delimited
                (13, 2) => {
                    let (body, next) =
                        parser::read_len_delim_at(graph_bytes, pos, "graph value_info")?;
                    pos = next;
                    callback(ParseEvent::ValueInfo(parser::parse_value_info_proto(body)?))?;
                }
                // field 15: sparse_initializer — refuse rather than drop the weight
                (15, 2) => {
                    return Err("graph: sparse_initializer is not supported".to_string());
                }
                // Any other field (including groups): skip with checked arithmetic.
                (_, wt) => {
                    pos = parser::skip_field_value_in_buf(graph_bytes, pos, field_no, wt)?;
                }
            }
        }
        Ok(())
    }
}
