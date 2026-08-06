//! Backend selection — **the only file in this crate with a platform swap**.
//!
//! This mirrors `oxionnx-coreml`'s `package/mod.rs` exactly: the platform-neutral
//! types live here, and a single `#[cfg]` swaps in either the Windows
//! implementation or a permanently-declining stub, **both exposing a duck-typed
//! identical API**.  Nothing else in the crate is `#[cfg]`-aware, which is what lets
//! `dispatch.rs`, `kernels/*`, `plan.rs`, `layout.rs` and `reference.rs` be compiled —
//! and tested — on every target.
//!
//! # The `Backend` contract, identical on both sides
//!
//! ```ignore
//! impl Backend {
//!     /// Acquire the best available backend.  **Never panics.**  `None` on every
//!     /// non-Windows target, and on Windows when no D3D12 adapter works.
//!     pub(crate) fn try_new() -> Option<Self>;
//!
//!     /// Which backend this is.  Stub: always `BackendKind::Unavailable`.
//!     pub(crate) fn kind(&self) -> BackendKind;
//!
//!     /// DXGI adapter description.  Stub: `"none"`.
//!     pub(crate) fn adapter_name(&self) -> String;
//!
//!     /// Execute a planned MatMul/Gemm.  Returns the dense `[batch, m, n]` output.
//!     ///
//!     /// # Errors
//!     /// `DirectMLError::Declined` when this backend cannot express the plan — the
//!     /// router turns that into `Ok(None)` (CPU fallback), *not* an error.  Any
//!     /// other variant is a genuine GPU failure.  The stub always declines.
//!     pub(crate) fn matmul(&self, plan: &MatMulPlan, a: &[f32], b: &[f32], c: Option<&[f32]>)
//!         -> Result<Vec<f32>>;
//!
//!     pub(crate) fn binary(&self, plan: &ElementwisePlan, op: BinaryOp, a: &[f32], b: &[f32])
//!         -> Result<Vec<f32>>;
//!
//!     pub(crate) fn unary(&self, plan: &ElementwisePlan, op: UnaryOp, a: &[f32])
//!         -> Result<Vec<f32>>;
//! }
//! ```
//!
//! # Zero overhead off Windows
//!
//! `lib.rs::try_directml_dispatch` opens with `if !ctx.is_active() { return Ok(None) }`,
//! and off Windows `is_active()` is a monomorphic `false` — LLVM folds the entire body
//! away.  The session already guards with `self.dml.is_some()`, and `try_new()` returns
//! `None`, so the function is never even reached.

/// Which execution backend a [`crate::DirectMLContext`] resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Genuine DirectML operators (`IDMLDevice` + `DML_*_OPERATOR_DESC`).
    DirectMl,
    /// D3D12 compute shaders compiled from [`crate::hlsl`] at run time via
    /// `D3DCompile`.  Selected when D3D12 works but `DMLCreateDevice` does not —
    /// `DirectML.dll` is not present on every supported Windows SKU.
    Hlsl,
    /// No GPU backend: every non-Windows target, and Windows without a working
    /// D3D12 adapter.
    Unavailable,
}

impl BackendKind {
    /// Stable tag for logs and [`crate::SelfCheckReport`].
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirectMl => "DirectML",
            Self::Hlsl => "HLSL",
            Self::Unavailable => "unavailable",
        }
    }

    /// `true` for [`Self::DirectMl`] and [`Self::Hlsl`].
    #[must_use]
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::DirectMl | Self::Hlsl)
    }
}

#[cfg(target_os = "windows")]
pub(crate) mod d3d12;
#[cfg(target_os = "windows")]
pub(crate) mod dml;
#[cfg(target_os = "windows")]
mod windows_backend;
#[cfg(target_os = "windows")]
pub(crate) use windows_backend::Backend;

#[cfg(not(target_os = "windows"))]
mod stub_backend;
#[cfg(not(target_os = "windows"))]
pub(crate) use stub_backend::Backend;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::BackendKind;

    #[test]
    fn is_gpu_is_false_only_for_unavailable() {
        assert!(BackendKind::DirectMl.is_gpu());
        assert!(BackendKind::Hlsl.is_gpu());
        assert!(!BackendKind::Unavailable.is_gpu());
    }

    #[test]
    fn as_str_is_stable() {
        assert_eq!(BackendKind::DirectMl.as_str(), "DirectML");
        assert_eq!(BackendKind::Hlsl.as_str(), "HLSL");
        assert_eq!(BackendKind::Unavailable.as_str(), "unavailable");
    }
}
