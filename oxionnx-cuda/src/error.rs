//! Error types for CUDA-accelerated ONNX dispatch.

use thiserror::Error;

/// Errors that can arise during CUDA-accelerated op dispatch.
///
/// Most of these wrap underlying OxiCUDA driver, BLAS, DNN, PTX, or launch
/// errors. The caller in `session.rs` maps these to `OnnxError::Internal`.
#[derive(Debug, Error)]
pub enum CudaDispatchError {
    /// CUDA driver-level error (init, context, module load, etc.).
    #[error("CUDA driver error: {0}")]
    Driver(#[from] oxicuda_driver::CudaError),

    /// BLAS operation error (GEMM, etc.).
    #[error("CUDA BLAS error: {0}")]
    Blas(String),

    /// DNN operation error (Conv, etc.).
    #[error("CUDA DNN error: {0}")]
    Dnn(String),

    /// PTX code generation error.
    #[error("PTX generation error: {0}")]
    Ptx(String),

    /// Unsupported configuration for this op (falls back to CPU).
    #[error("Unsupported CUDA config for op '{op}': {reason}")]
    Unsupported {
        /// The ONNX operator name.
        op: &'static str,
        /// Human-readable reason.
        reason: String,
    },

    /// Tensor shape is incompatible with the expected CUDA kernel contract.
    #[error("Shape error for op '{op}': {msg}")]
    Shape {
        /// The ONNX operator name.
        op: &'static str,
        /// Description of the shape problem.
        msg: String,
    },
}

impl From<CudaDispatchError> for oxionnx_core::OnnxError {
    fn from(e: CudaDispatchError) -> Self {
        Self::Internal(e.to_string())
    }
}
