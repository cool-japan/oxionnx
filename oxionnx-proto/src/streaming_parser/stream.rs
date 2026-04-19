//! `StreamingParser` struct and its main parse logic.

use crate::parser;
use std::io::{BufReader, Read};

use super::helpers::{extract_name_from_bytes, parse_opset_import_bytes};
use super::types::ParseEvent;
use super::wire::{
    read_exact_vec, read_field_header, read_len_delim_value, read_varint_from_reader,
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
