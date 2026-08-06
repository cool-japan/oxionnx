//! Error surface for the DirectML execution provider.
//!
//! ## The one distinction that matters: [`DirectMLError::Declined`]
//!
//! A *declined* op is not a failure.  It means "this backend cannot express this
//! node, but the node itself is perfectly well-formed": rank above
//! [`crate::layout::DML_RANK`], an element count above `u32::MAX`, a dispatch grid
//! above D3D12's per-dimension limit, an empty (`numel() == 0`) tensor, a MatMul
//! that is not 2-D × 2-D, an elementwise op whose operands need broadcasting.
//! The router turns [`DirectMLError::Declined`] into `Ok(None)`, which is a
//! *correct, silent CPU fallback*.
//!
//! [`DirectMLError::ShapeMismatch`], by contrast, means the *CPU operator would
//! fail on the same inputs* — the model is malformed.
//!
//! Every other variant is a genuine GPU / driver / compiler failure.  The router
//! still falls back to CPU on those, so inference never breaks, but it logs them,
//! because they mean something is actually wrong.
//!
//! Keeping these three categories apart is what lets a caller distinguish
//! "we chose not to" from "we could not" from "your model is broken".

use thiserror::Error;

/// Convenience alias used by every fallible function in this crate.
pub type Result<T> = core::result::Result<T, DirectMLError>;

/// Errors specific to the DirectML execution provider.
///
/// `#[non_exhaustive]`: new backend / dispatch failure modes may be added as
/// this provider's op coverage grows. Downstream `match`es need a wildcard
/// arm; existing variants remain constructible as before.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DirectMLError {
    /// D3D12 device initialization failed.
    #[error("D3D12 device initialization failed: {0}")]
    DeviceInitFailed(String),

    /// A DirectML operator is not supported by this provider.
    #[error("DirectML operator not supported: {0}")]
    UnsupportedOp(String),

    /// A DirectML GPU dispatch failed.
    #[error("DirectML dispatch error: {0}")]
    DispatchFailed(String),

    /// A buffer transfer between CPU and GPU failed.
    #[error("Buffer transfer error: {0}")]
    TransferError(String),

    /// Shape / rank validation failed *before* any GPU work was queued, on inputs
    /// the CPU operator would reject too.  This describes a malformed model, not
    /// a backend limitation.
    #[error("DirectML shape error: {0}")]
    ShapeMismatch(String),

    /// The node is well-formed but this backend *declines* it.
    ///
    /// The router maps this to `Ok(None)` — a correct, silent CPU fallback, not a
    /// failure.  See this module's documentation for the full list of reasons.
    #[error("DirectML backend declined op: {0}")]
    Declined(String),

    /// `D3DCompile` rejected the HLSL; carries the compiler's error blob.
    #[error("HLSL compilation failed: {0}")]
    ShaderCompile(String),

    /// A Win32/COM call returned a failing `HRESULT`.
    #[error("Win32 error {hresult:#010x} in {context}: {message}")]
    Win32 {
        /// Static name of the call site that failed, e.g. `"D3D12CreateDevice"`.
        context: &'static str,
        /// The raw `HRESULT`, reinterpreted as `u32` so that it renders as
        /// `0x887a0005` rather than as a negative `i32`.
        hresult: u32,
        /// The system's message for this `HRESULT`.
        message: String,
    },

    /// The context's backend mutex was poisoned by a panic on another thread.
    #[error("DirectML context lock poisoned")]
    LockPoisoned,
}

impl From<DirectMLError> for oxionnx_core::OnnxError {
    fn from(e: DirectMLError) -> Self {
        Self::Internal(e.to_string())
    }
}

/// Attach a static call-site name to every failing `HRESULT`.
///
/// **Every** Win32/COM call in this crate goes through `.ctx("…")`.  There is
/// deliberately *no* `From<windows::core::Error> for DirectMLError` impl, precisely
/// so that a bare `?` cannot silently discard the call site — a lone `HRESULT`
/// (`0x80070057`, "the parameter is incorrect") is close to useless when the
/// caller has made forty of them.
///
/// ```ignore
/// let device: ID3D12Device = create().ctx("D3D12CreateDevice")?;
/// ```
#[cfg(target_os = "windows")]
pub trait HrExt<T> {
    /// Convert a `windows::core::Result` into a [`Result`], tagging the failure
    /// with the name of the call that produced it.
    ///
    /// # Errors
    /// [`DirectMLError::Win32`] when `self` is `Err`.
    fn ctx(self, context: &'static str) -> Result<T>;
}

#[cfg(target_os = "windows")]
impl<T> HrExt<T> for windows::core::Result<T> {
    fn ctx(self, context: &'static str) -> Result<T> {
        self.map_err(|e| DirectMLError::Win32 {
            context,
            // An HRESULT is a bit pattern, not a magnitude: 0x887A0005 is *the*
            // value, and printing it as -2005270523 helps nobody.
            #[allow(clippy::cast_sign_loss)]
            hresult: e.code().0 as u32,
            message: e.message(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::DirectMLError;

    #[test]
    fn display_carries_the_payload() {
        let e = DirectMLError::DispatchFailed("test error".into());
        assert!(format!("{e}").contains("test error"));

        let e = DirectMLError::DeviceInitFailed("no d3d12".into());
        assert!(format!("{e}").contains("no d3d12"));

        let e = DirectMLError::Declined("rank 5".into());
        assert!(format!("{e}").contains("declined"));

        let e = DirectMLError::ShapeMismatch("k mismatch".into());
        assert!(format!("{e}").contains("k mismatch"));

        let e = DirectMLError::ShaderCompile("error X3000".into());
        assert!(format!("{e}").contains("error X3000"));
    }

    #[test]
    fn win32_renders_hresult_as_unsigned_hex() {
        let e = DirectMLError::Win32 {
            context: "D3D12CreateDevice",
            hresult: 0x887a_0005,
            message: "device removed".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("0x887a0005"), "got: {s}");
        assert!(s.contains("D3D12CreateDevice"), "got: {s}");
        assert!(s.contains("device removed"), "got: {s}");
    }

    #[test]
    fn converts_into_onnx_internal_error() {
        let e = DirectMLError::Declined("empty tensor".into());
        let o: oxionnx_core::OnnxError = e.into();
        assert!(matches!(o, oxionnx_core::OnnxError::Internal(_)));
        assert!(format!("{o}").contains("empty tensor"));
    }

    #[test]
    fn lock_poisoned_has_a_message() {
        assert!(!format!("{}", DirectMLError::LockPoisoned).is_empty());
    }
}
