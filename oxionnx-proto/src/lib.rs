#![deny(unsafe_code)]
//! oxionnx-proto — Pure Rust ONNX protobuf parser.

#[cfg(feature = "mmap")]
#[allow(unsafe_code)]
pub mod mmap_loader;

pub mod model;
pub mod parser;
#[allow(unsafe_code)]
pub mod reader;
pub mod schema;
pub mod streaming_parser;
pub mod types;

pub use model::{build_graph, extract_training_info, load, load_with_path, SUPPORTED_OPSET_RANGE};
pub use parser::parse_model;
#[cfg(feature = "mmap")]
pub use reader::OnnxReader;
pub use reader::{
    attr_float, attr_int, attr_ints, attr_string, attr_type, bytes_to_f32, dtype_code,
    external_entry, is_external, ReaderError,
};
pub use schema::{default_schemas, validate_schemas, OpSchema, SchemaViolation};
pub use streaming_parser::{
    parse_streaming, parse_with_weight_filter, ParseEvent, StreamingParser,
};
pub use types::*;
