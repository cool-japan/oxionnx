//! Per-op kernels: `Tensor` in, [`crate::plan`] validation, `Backend` dispatch,
//! `Tensor` out.
//!
//! **Platform-neutral.**  These modules contain no `#[cfg]` and no FFI: they build a
//! plan, hand it plus the raw f32 slices to whichever [`crate::backend::Backend`] the
//! context resolved to, and wrap the returned buffer back into a `Tensor`.  On Linux
//! the backend is the declining stub, so the entire `Ok(Some)` / `Ok(None)` / `Err`
//! routing contract is exercisable in CI without a GPU.
//!
//! The kernels never compute a dimension themselves — everything shape-derived comes
//! pre-validated and pre-range-checked off a [`crate::plan::MatMulPlan`] or
//! [`crate::plan::ElementwisePlan`].

pub(crate) mod conv;
pub(crate) mod elementwise;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub(crate) mod softmax;
