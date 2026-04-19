//! Public event types emitted by the streaming parser.

use crate::types::NodeProto;
use oxionnx_core::Tensor;

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
