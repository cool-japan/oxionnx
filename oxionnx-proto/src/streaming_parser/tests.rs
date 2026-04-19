//! Tests for the streaming ONNX parser.

#![cfg(test)]

use std::collections::HashMap;
use std::io::Cursor;

use oxionnx_core::Tensor;

use crate::parser as batch_parser;

use super::convenience::{parse_streaming, parse_with_weight_filter};
use super::stream::StreamingParser;
use super::types::ParseEvent;

// ── Protobuf encoding helpers ──────────────────────────────────

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

// ── Tests ──────────────────────────────────────────────────────

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
    let batch_model = batch_parser::parse_model(&model_bytes).expect("batch parse should succeed");

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
        assert!(
            stream_weights.contains_key(name),
            "streaming missing weight '{name}'"
        );
        let stream_tensor = &stream_weights[name];
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
