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
    #[error("{VERIFY_MISMATCH_MARKER}{0}")]
    Verify(String),
}

/// The literal prefix every [`CudaDispatchError::Verify`] message carries.
///
/// It exists because the identity of a verify mismatch has to survive the
/// conversion to [`oxionnx_core::OnnxError`], which is a lossy
/// `Internal(String)` — and the session runner has to be able to tell a
/// *proved-wrong kernel* apart from an ordinary dispatch failure, because the
/// two get opposite treatment: a dispatch failure falls back to the CPU, a
/// verify mismatch under `OXIONNX_CUDA_STRICT=1` fails the run. See
/// [`is_verify_mismatch`].
///
/// Interpolated directly into the `#[error]` format string above, so the
/// marker and the message cannot drift apart; `verify_display_starts_with_the_marker`
/// pins that.
pub const VERIFY_MISMATCH_MARKER: &str = "CUDA VERIFY MISMATCH: ";

impl CudaDispatchError {
    /// Is this a shadow-verification mismatch — i.e. did a CUDA kernel get
    /// *proved wrong* on this input, as opposed to failing to run?
    #[must_use]
    pub fn is_verify_mismatch(&self) -> bool {
        matches!(self, Self::Verify(_))
    }
}

/// Does this [`OnnxError`](oxionnx_core::OnnxError) carry a CUDA
/// shadow-verification mismatch?
///
/// The session runner's dispatch policy hinges on this distinction. A CUDA
/// dispatch that *fails* (driver error, PTX error, an unsupported
/// configuration reported as an error rather than a decline) is recoverable:
/// the node has not been executed, and the CPU operator one frame up computes
/// it correctly. A dispatch that comes back as a verify mismatch is not: under
/// `OXIONNX_CUDA_STRICT=1` the user has asked for a *proved-wrong GPU* to end
/// the run rather than be papered over, and `oxionnx-cuda` can only signal
/// that by returning `Err`. Collapsing both into a CPU fallback is what made
/// `OXIONNX_CUDA_STRICT=1` exit `0` on a run whose kernels it had just caught
/// disagreeing with the oracle.
///
/// Matching on the message prefix rather than the enum is forced by the
/// `From<CudaDispatchError> for OnnxError` conversion below being
/// `Internal(String)`: by the time the runner sees the error there is no enum
/// left. [`VERIFY_MISMATCH_MARKER`] is what makes that recoverable, and it is
/// the *same* constant the `#[error]` attribute formats with.
#[must_use]
pub fn is_verify_mismatch(err: &oxionnx_core::OnnxError) -> bool {
    match err {
        oxionnx_core::OnnxError::Internal(msg) => msg.contains(VERIFY_MISMATCH_MARKER),
        _ => false,
    }
}

impl From<CudaDispatchError> for oxionnx_core::OnnxError {
    fn from(e: CudaDispatchError) -> Self {
        Self::Internal(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole `is_verify_mismatch` mechanism rests on the `Display` of
    /// `Verify` beginning with the marker. Interpolating the constant into the
    /// `#[error]` attribute makes that true by construction; this makes it
    /// true by test as well, so a future edit to either cannot quietly break
    /// `OXIONNX_CUDA_STRICT`.
    #[test]
    fn verify_display_starts_with_the_marker() {
        let rendered = CudaDispatchError::Verify("element 7: GPU=1, CPU=2".into()).to_string();
        assert!(
            rendered.starts_with(VERIFY_MISMATCH_MARKER),
            "Verify's Display must start with VERIFY_MISMATCH_MARKER, got {rendered:?}"
        );
        assert!(rendered.ends_with("element 7: GPU=1, CPU=2"));
    }

    #[test]
    fn a_verify_error_is_recognised_after_the_lossy_onnx_error_conversion() {
        let err: oxionnx_core::OnnxError =
            CudaDispatchError::Verify("element 7: GPU=1, CPU=2".into()).into();
        assert!(is_verify_mismatch(&err));
    }

    #[test]
    fn an_ordinary_dispatch_failure_is_not_a_verify_mismatch() {
        for err in [
            CudaDispatchError::Blas("gemm failed".into()),
            CudaDispatchError::Dnn("conv failed".into()),
            CudaDispatchError::Ptx("bad ptx".into()),
            CudaDispatchError::Unsupported {
                op: "Conv",
                reason: "asymmetric pads".into(),
            },
            CudaDispatchError::Shape {
                op: "MatMul",
                msg: "K mismatch".into(),
            },
        ] {
            assert!(!err.is_verify_mismatch(), "{err} must not be a mismatch");
            let onnx: oxionnx_core::OnnxError = err.into();
            assert!(!is_verify_mismatch(&onnx));
        }
    }

    #[test]
    fn a_non_internal_onnx_error_is_never_a_verify_mismatch() {
        assert!(!is_verify_mismatch(
            &oxionnx_core::OnnxError::ShapeMismatch("CUDA VERIFY MISMATCH: not really".into())
        ));
    }
}
