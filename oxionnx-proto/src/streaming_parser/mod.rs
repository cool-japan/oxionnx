//! Streaming protobuf parser for large ONNX models.
//!
//! Parses ONNX models from a `Read` source without loading the entire file
//! into memory. Weight tensors are yielded one at a time via callback,
//! allowing the caller to process or discard them immediately.
//!
//! # Architecture
//!
//! Protobuf uses length-delimited fields, which allows us to stream: read each
//! top-level field header, then either parse the field body or skip it. For the
//! graph field, we enter it and stream its sub-fields. When we encounter a large
//! initializer, we parse its TensorProto header (dims, dtype) and raw_data field,
//! yielding the tensor immediately without buffering the entire model.
//!
//! # Usage
//!
//! ```no_run
//! use std::io::Cursor;
//! use oxionnx_proto::streaming_parser::{StreamingParser, ParseEvent};
//!
//! let data: Vec<u8> = vec![]; // model bytes
//! let mut parser = StreamingParser::new(Cursor::new(data));
//! parser.parse(|event| {
//!     match event {
//!         ParseEvent::Weight { name, tensor } => {
//!             // Process weight immediately
//!         }
//!         _ => {}
//!     }
//!     Ok(())
//! }).ok();
//! ```

pub(super) mod convenience;
pub(super) mod helpers;
pub(super) mod stream;
pub(super) mod types;
pub(super) mod wire;

#[cfg(test)]
mod tests;

pub use convenience::{parse_streaming, parse_with_weight_filter};
pub use stream::StreamingParser;
pub use types::ParseEvent;
