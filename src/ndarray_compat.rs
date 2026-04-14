//! ndarray interoperability helpers for the `ort`-to-oxionnx migration.
//!
//! Enabled with the `ndarray` Cargo feature.  When active, [`crate::Tensor`]
//! gains several conversion methods and `ort`-compatible extraction helpers:
//!
//! | Method | Description |
//! |--------|-------------|
//! | `Tensor::from_ndarray(arr)` | `ArrayBase<S, D>` → `Tensor` |
//! | `Tensor::from_ndarray_view(view)` | `ArrayView<'_, f32, D>` → `Tensor` |
//! | `Tensor::to_ndarray()` | `Tensor` → `ArrayD<f32>` (copy) |
//! | `Tensor::as_ndarray_view()` | `Tensor` → `Result<ArrayViewD<f32>, OnnxError>` (borrow) |
//! | `Tensor::try_extract_tensor::<T>()` | `ort` compat: `(&[usize], &[f32])` |
//! | `Tensor::try_extract_array::<T>()` | `ort` compat: `ArrayViewD<f32>` |
//!
//! All methods are implemented directly on [`crate::Tensor`] in
//! `oxionnx-core/src/tensor.rs` under `#[cfg(feature = "ndarray")]`.
//! This module exists purely as a documentation and re-export entry point.
