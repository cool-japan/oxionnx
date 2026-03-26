#![deny(unsafe_code)]
//! oxionnx-proto — Pure Rust ONNX protobuf parser.

#[cfg(feature = "mmap")]
#[allow(unsafe_code)]
pub mod mmap_loader;

pub mod model;
pub mod parser;
pub mod schema;
pub mod streaming_parser;
pub mod types;

pub use model::{build_graph, extract_training_info, load, load_with_path, SUPPORTED_OPSET_RANGE};
pub use parser::parse_model;
pub use schema::{default_schemas, validate_schemas, OpSchema, SchemaViolation};
pub use streaming_parser::{
    parse_streaming, parse_with_weight_filter, ParseEvent, StreamingParser,
};
pub use types::*;
