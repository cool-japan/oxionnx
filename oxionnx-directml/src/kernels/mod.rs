//! DirectML compute kernels.
//!
//! Each sub-module owns the GPU-side implementation for a family of ONNX ops.
//! All kernels return `Err(DirectMLError::DispatchFailed(_))` in this scaffold
//! wave; the dispatch layer converts those errors to `Ok(None)` so inference
//! falls back to CPU transparently.

pub mod elementwise;
pub mod matmul;
