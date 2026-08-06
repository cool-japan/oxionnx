//! Convenience functions for whole-model streaming parse.

use std::collections::HashMap;
use std::io::Read;

use oxionnx_core::Tensor;

use crate::parser::FunctionProto;
use crate::types::{GraphProto, NodeProto, ValueInfoProto};

use super::stream::StreamingParser;
use super::types::ParseEvent;

/// Parse an ONNX model from a `Read` source, returning the full graph and weights.
///
/// This is the streaming equivalent of `crate::model::load()` but reads from
/// any `Read` source instead of requiring all bytes in memory upfront. The graph
/// carries the same name and input/output `ValueInfoProto` metadata as the eager
/// parser, so sessions built from a reader keep their input dtype/shape info.
///
/// Like the eager path, any model-local function call (`ModelProto.functions`,
/// wire field 25) is resolved by inlining before this returns: the caller never
/// sees a node that refers to a function body. Node and function events are both
/// buffered for the whole stream because a real ONNX writer emits `functions`
/// after `graph` on the wire — inlining node-by-node as events arrive would run
/// ahead of the library it needs.
pub fn parse_streaming<R: Read>(
    reader: R,
) -> Result<(GraphProto, HashMap<String, Tensor>), String> {
    let mut nodes: Vec<NodeProto> = Vec::new();
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    let mut name = String::new();
    let mut input_value_infos: Vec<ValueInfoProto> = Vec::new();
    let mut output_value_infos: Vec<ValueInfoProto> = Vec::new();
    let mut value_infos: Vec<ValueInfoProto> = Vec::new();
    let mut functions: Vec<FunctionProto> = Vec::new();

    let mut parser = StreamingParser::new(reader);
    parser.parse(|event| {
        match event {
            ParseEvent::ModelHeader { .. } | ParseEvent::OpsetImport { .. } => {}
            ParseEvent::GraphName(graph_name) => {
                name = graph_name;
            }
            ParseEvent::Node(node_proto) => {
                nodes.push(node_proto);
            }
            ParseEvent::Weight {
                name: weight_name,
                tensor,
            } => {
                weights.insert(weight_name, tensor);
            }
            ParseEvent::GraphInput(vi) => {
                input_value_infos.push(vi);
            }
            ParseEvent::GraphOutput(vi) => {
                output_value_infos.push(vi);
            }
            ParseEvent::ValueInfo(vi) => {
                value_infos.push(vi);
            }
            ParseEvent::LocalFunction(f) => {
                functions.push(f);
            }
            ParseEvent::End => {}
        }
        Ok(())
    })?;

    let inputs = input_value_infos.iter().map(|vi| vi.name.clone()).collect();
    let outputs = output_value_infos
        .iter()
        .map(|vi| vi.name.clone())
        .collect();

    let mut graph = GraphProto {
        nodes,
        name,
        initializers: Vec::new(), // weights already extracted above
        inputs,
        outputs,
        input_value_infos,
        output_value_infos,
        value_infos,
    };
    crate::model::inline_local_functions(&mut graph, &functions)?;

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
    let mut name = String::new();
    let mut input_value_infos: Vec<ValueInfoProto> = Vec::new();
    let mut output_value_infos: Vec<ValueInfoProto> = Vec::new();
    let mut value_infos: Vec<ValueInfoProto> = Vec::new();
    let mut functions: Vec<FunctionProto> = Vec::new();

    let mut parser = StreamingParser::new(reader);

    // We'll use a wrapper that intercepts Weight events
    parser.parse(|event| {
        match event {
            ParseEvent::ModelHeader { .. } | ParseEvent::OpsetImport { .. } => {}
            ParseEvent::GraphName(graph_name) => {
                name = graph_name;
            }
            ParseEvent::Node(node_proto) => {
                nodes.push(node_proto);
            }
            ParseEvent::Weight {
                name: weight_name,
                tensor,
            } => {
                let shape = tensor.shape.clone();
                if weight_filter(&weight_name, &shape) {
                    weights.insert(weight_name, tensor);
                }
                // Otherwise: weight is dropped, freeing memory
            }
            ParseEvent::GraphInput(vi) => {
                input_value_infos.push(vi);
            }
            ParseEvent::GraphOutput(vi) => {
                output_value_infos.push(vi);
            }
            ParseEvent::ValueInfo(vi) => {
                value_infos.push(vi);
            }
            ParseEvent::LocalFunction(f) => {
                functions.push(f);
            }
            ParseEvent::End => {}
        }
        Ok(())
    })?;

    let inputs = input_value_infos.iter().map(|vi| vi.name.clone()).collect();
    let outputs = output_value_infos
        .iter()
        .map(|vi| vi.name.clone())
        .collect();

    let mut graph = GraphProto {
        nodes,
        name,
        initializers: Vec::new(),
        inputs,
        outputs,
        input_value_infos,
        output_value_infos,
        value_infos,
    };
    crate::model::inline_local_functions(&mut graph, &functions)?;

    Ok((graph, weights))
}
