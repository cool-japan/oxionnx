//! Public event types emitted by the streaming parser.

use crate::parser::FunctionProto;
use crate::types::{NodeProto, ValueInfoProto};
use oxionnx_core::Tensor;

/// Events emitted during streaming parse of an ONNX model.
#[derive(Debug)]
pub enum ParseEvent {
    /// Model metadata, emitted before any graph content so consumers can size their
    /// work up front.
    ///
    /// `opset_imports` holds only the imports that appeared *before* the graph on the
    /// wire. Protobuf writers emit fields in field-number order and `opset_import` is
    /// field 8 while `graph` is field 7, so for essentially every real ONNX file this
    /// list is empty — use the [`ParseEvent::OpsetImport`] events instead, which are
    /// emitted for every entry in wire order.
    ModelHeader {
        ir_version: i64,
        model_version: i64,
        producer_name: String,
        producer_version: String,
        opset_imports: Vec<(String, i64)>,
    },
    /// One `opset_import` entry, emitted in wire order as it is encountered.
    OpsetImport { domain: String, version: i64 },
    /// The graph's name (GraphProto field 2).
    GraphName(String),
    /// A graph node was parsed.
    Node(NodeProto),
    /// A weight/initializer tensor was parsed.
    Weight { name: String, tensor: Tensor },
    /// A graph input, with its full type/shape metadata.
    GraphInput(ValueInfoProto),
    /// A graph output, with its full type/shape metadata.
    GraphOutput(ValueInfoProto),
    /// Shape/dtype metadata for an intermediate value (GraphProto field 13).
    ValueInfo(ValueInfoProto),
    /// A model-local function declaration (`ModelProto.functions`, wire field
    /// 25). Real ONNX writers emit this after the graph (field 7), so by the
    /// time this event arrives every [`ParseEvent::Node`] a call site of it
    /// could match has already been seen — a consumer that wants to resolve
    /// calls (as `crate::model::inline_local_functions` does) should buffer
    /// nodes and functions separately and inline once the stream ends, the
    /// same way `crate::streaming_parser::parse_streaming` does.
    LocalFunction(FunctionProto),
    /// End of model.
    End,
}
