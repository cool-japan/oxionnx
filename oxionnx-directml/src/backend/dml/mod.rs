//! The genuine DirectML backend: `IDMLDevice` and real `DML_*_OPERATOR_DESC`s.
//!
//! Compiled on Windows only, and reached only when `DMLCreateDevice` resolves at run
//! time (see [`device::DmlDevice`]).  When it does not — `DirectML.dll` is absent on
//! some Windows SKUs — [`crate::backend`] falls back to
//! [`crate::backend::d3d12::hlsl_backend::HlslEngine`], which needs nothing but
//! D3D12 itself.
//!
//! # The two things this subtree exists to contain
//!
//! 1. **Raw pointer chains.**  `DML_*_OPERATOR_DESC → DML_TENSOR_DESC →
//!    DML_BUFFER_TENSOR_DESC → Sizes[]` is four levels of `*const`, none of which
//!    carries a Rust lifetime.  The borrow checker will not save you.  [`tensor`]
//!    documents the rule that keeps it sound: descriptors are **stack locals inside a
//!    single function** and die before it returns; no function in this subtree may
//!    return one by value.
//! 2. **`ManuallyDrop` COM fields.**  `DML_BUFFER_BINDING::Buffer` and
//!    `DML_BINDING_TABLE_DESC::Dispatchable` are `ManuallyDrop<Option<…>>`: never
//!    dropped → a leaked COM reference per dispatch; dropped from a borrow → a
//!    double-release and a use-after-free.  Neither is caught by rustc, clippy, Miri
//!    or any test that can run without a GPU.  [`binding`] owns both, and nothing
//!    outside it may construct either type.
//!
//! All shape and stride math — including `TotalTensorSizeInBytes`, which is *not*
//! `product(sizes) * 4` — is computed in [`crate::layout`] and merely copied here.

pub(crate) mod binding;
pub(crate) mod device;
pub(crate) mod dml_backend;
pub(crate) mod op;
pub(crate) mod tensor;
