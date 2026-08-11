#![cfg_attr(not(feature = "simd"), deny(unsafe_code))]
//! oxionnx-ops — ONNX operator implementations.

#[cfg(feature = "simd")]
pub mod simd_ops;

pub mod attention;
pub mod bitwise;
pub mod comparison;
pub mod control_flow;
pub mod conv;
pub(crate) mod conv_typed;
pub mod dsp;
pub mod einsum;
pub mod flash;
pub mod indexing;
pub mod kv_cache;
pub mod math;
pub(crate) mod math_typed;
pub mod ml;
pub mod ml_svm;
pub mod ml_tree;
pub mod nms;
pub mod nn;
pub mod quantized;
pub mod registry;
pub mod resize;
pub mod rnn;
pub(crate) mod rnn_typed;
pub mod shape;
pub mod spatial;
/// Cross-platform wall-clock time (`SystemTime` panics at runtime on
/// wasm32-unknown-unknown; see the module docs). Internal only.
mod time_compat;
pub mod typed_ops;

pub use registry::default_registry;
