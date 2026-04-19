//! Tensor types and utilities for the oxionnx ONNX inference engine.

pub mod broadcastiter_traits;
pub mod functions;
pub mod tensorviewiter_traits;
pub mod types;

// Re-export all public items so the original `tensor::*` usage is preserved.
pub use functions::*;
pub use types::*;
