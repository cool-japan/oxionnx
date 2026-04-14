#![cfg_attr(not(feature = "simd"), deny(unsafe_code))]
//! oxionnx-ops — ONNX operator implementations.

#[cfg(feature = "simd")]
pub mod simd_ops;

pub mod attention;
pub mod bitwise;
pub mod comparison;
pub mod control_flow;
pub mod conv;
pub mod dsp;
pub mod einsum;
pub mod flash;
pub mod indexing;
pub mod kv_cache;
pub mod math;
pub mod ml;
pub mod ml_svm;
pub mod ml_tree;
pub mod nms;
pub mod nn;
pub mod quantized;
pub mod registry;
pub mod resize;
pub mod rnn;
pub mod shape;
pub mod spatial;
pub mod typed_ops;

pub use registry::default_registry;
