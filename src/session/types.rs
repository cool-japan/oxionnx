use std::collections::HashMap;

/// Optimization level for graph optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations.
    None,
    /// Basic: dead node elimination only.
    Basic,
    /// Extended: dead node elimination + operator fusions.
    Extended,
    /// All: constant folding + dead node elimination + fusions.
    All,
}

/// Profiling information for a single executed node.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    /// Name of the node in the graph.
    pub node_name: String,
    /// The ONNX op type (e.g. "MatMul", "Relu").
    pub op_type: String,
    /// Wall-clock execution duration.
    pub duration: std::time::Duration,
    /// Shapes of each output tensor produced by this node.
    pub output_shapes: Vec<Vec<usize>>,
}

/// Metadata extracted from the ONNX model file.
///
/// Corresponds to top-level fields in the ONNX `ModelProto`.
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// Tool that generated the model (e.g., "pytorch", "tf2onnx").
    pub producer_name: String,
    /// Version of the generating tool.
    pub producer_version: String,
    /// Model namespace/domain.
    pub domain: String,
    /// Name of the root computation graph.
    pub graph_name: String,
    /// ONNX IR version used by the model.
    pub ir_version: i64,
    /// Opset domain + version pairs declared by the model.
    pub opset_imports: Vec<(String, i64)>,
    /// User-defined key-value metadata (from `metadata_props`).
    pub custom_metadata: HashMap<String, String>,
}

/// Summary information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Number of computation nodes in the (optimized) graph.
    pub node_count: usize,
    /// Total number of scalar parameters stored as weights.
    pub parameter_count: usize,
    /// Estimated weight memory in bytes (assuming f32).
    pub weight_bytes: usize,
    /// Histogram of operator types: op_name -> count.
    pub op_histogram: HashMap<String, usize>,
}

/// Convert a `RawModelMeta` (proto layer) into a public `ModelMetadata`.
pub(crate) fn raw_meta_to_model_metadata(raw: oxionnx_proto::model::RawModelMeta) -> ModelMetadata {
    ModelMetadata {
        producer_name: raw.producer_name,
        producer_version: raw.producer_version,
        domain: raw.domain,
        graph_name: raw.graph_name,
        ir_version: raw.ir_version,
        opset_imports: raw.opset_imports,
        custom_metadata: raw.metadata_props.into_iter().collect(),
    }
}
