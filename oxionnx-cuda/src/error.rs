//! Error types for CUDA-accelerated ONNX dispatch.

use thiserror::Error;

/// Errors that can arise during CUDA-accelerated op dispatch.
///
/// Most of these wrap underlying OxiCUDA driver, BLAS, DNN, PTX, or launch
/// errors. The caller in `session.rs` maps these to `OnnxError::Internal`.
///
/// `#[non_exhaustive]`: new CUDA failure modes may be added as more kernels
/// and dispatch paths are implemented. Downstream `match`es need a wildcard
/// arm; existing variants remain constructible as before.
#[derive(Debug, Error)]
#[non_exhaustive]
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

    /// A CUDA kernel's output failed shadow verification against the CPU
    /// oracle (`OXIONNX_CUDA_VERIFY=1`, see [`crate::reference`]).
    ///
    /// The GPU has been *proved* wrong on this exact input; its output is
    /// discarded rather than returned. Only reachable when
    /// `OXIONNX_CUDA_STRICT=1` is also set — under the default
    /// (`Fallback`) failure policy a mismatch is logged at `error!` and the
    /// node falls back to `Ok(None)` instead of surfacing this variant.
    #[error("CUDA VERIFY MISMATCH: {0}")]
    Verify(String),
}

impl From<CudaDispatchError> for oxionnx_core::OnnxError {
    fn from(e: CudaDispatchError) -> Self {
        Self::Internal(e.to_string())
    }
}
