//! Convenience functions for whole-model streaming parse.

use std::collections::HashMap;
use std::io::Read;

use oxionnx_core::Tensor;

use crate::types::{GraphProto, NodeProto};

use super::stream::StreamingParser;
use super::types::ParseEvent;

/// Parse an ONNX model from a `Read` source, returning the full graph and weights.
///
/// This is the streaming equivalent of `crate::model::load()` but reads from
/// any `Read` source instead of requiring all bytes in memory upfront.
pub fn parse_streaming<R: Read>(
    reader: R,
) -> Result<(GraphProto, HashMap<String, Tensor>), String> {
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
        ..Default::default()
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
) -> Result<(GraphProto, HashMap<String, Tensor>), String>
where
    F: FnMut(&str, &[usize]) -> bool,
{
    let mut nodes: Vec<NodeProto> = Vec::new();
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut outputs: Vec<String> = Vec::new();

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
        ..Default::default()
    };

    Ok((graph, weights))
}
