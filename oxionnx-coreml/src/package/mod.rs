//! `MlPackageModel` — load a pre-converted `.mlpackage` (or `.mlmodelc`) and
//! run inference through Apple's CoreML runtime.
//!
//! The runtime is a thin wrapper over `MLModel::predictionFromFeatures_error`.
//! It deliberately exposes only a graph-level interface — there is no
//! per-op dispatch surface, because CoreML's whole-graph scheduler is what
//! delivers the 17–25× speedups on Apple Silicon (per the OxiFace
//! ArcFace / SCRFD / InSwapper sub-gates).
//!
//! ## Threading
//!
//! `MLModel` is documented as thread-safe by Apple — multiple threads may
//! call `predictionFromFeatures_error:` concurrently on the same instance.
//! We expose this as `predict(&self, ...)` and provide manual `Send` + `Sync`
//! impls for [`MlPackageModel`] (the `Retained<MLModel>` field does not
//! auto-derive these, since `objc2` cannot tell from the type alone).
//!
//! ## Compilation
//!
//! A `.mlpackage` must be compiled to a `.mlmodelc` before CoreML can
//! load it.  `MLModel::compileModelAtURL_error:` does that into a fresh
//! UUID-named `$TMPDIR` directory on *every* call and never reuses or
//! deletes it, so the naive loop compiles on every process start and
//! leaks the result forever.  `compile_cache` fixes both halves: the
//! compile happens once per (bundle, content) pair and its output is
//! moved into a stable, content-keyed cache directory.  See
//! [`MlPackageModel::ensure_compiled`](crate::MlPackageModel::ensure_compiled)
//! for the cache location and `compile_cache`'s own module
//! documentation for keying, concurrency and degradation.
//!
//! ## I/O details
//!
//! * Inputs are projected into `MLMultiArray` instances (Float32)
//!   **without copying**.  `multi_array_from_f32` hands CoreML a raw
//!   pointer directly into the caller's [`oxionnx_core::Tensor::data`]
//!   slice via `initWithDataPointer_shape_dataType_strides_deallocator_error`
//!   with `deallocator: None` — CoreML is told it does not own this memory
//!   and must never free it.  The pointer stays valid because the
//!   constructed `MLMultiArray` never escapes the synchronous `predict` /
//!   `predict_raw` call that built it (see that function's `SAFETY:`
//!   comment for the full invariant).  A separate `multi_array_from_owned`
//!   builds an ownership-transferring variant (via a `block2::RcBlock`
//!   deallocator) for buffers that must outlive their constructing frame.
//! * Outputs are read via `getBytesWithHandler:` (the modern, non-deprecated
//!   path) into a tightly packed, C-contiguous buffer, regardless of
//!   whatever strides CoreML's own allocation used internally.  `predict`
//!   accepts `Float32` and `Float16` outputs, up-converting the latter to
//!   `f32` **in the same pass** as the stride walk — no intermediate byte
//!   buffer.  `predict_raw` returns the same bytes dtype-preserving (see
//!   `RawArray`) — no up-conversion — for pipelines that want the exact
//!   bytes an upstream CoreML model produced.  Both readers live in
//!   `array_read`; see that module for the layout planner and the SCRFD
//!   stride-padding story that motivates it.

use std::collections::HashMap;
use std::path::Path;

use crate::compute::{ComputePlanSummary, MlComputeUnits};
use crate::error::{CoreMLError, Result};
use oxionnx_core::Tensor;

/// Element dtype tag for [`RawArray`] — a portable mirror of the subset of
/// `MLMultiArrayDataType` this runtime can describe without depending on
/// `objc2-core-ml`, so the type is usable from non-macOS builds too.
///
/// Not every variant is necessarily *readable* by [`MlPackageModel`]'s
/// `predict_raw` today — unsupported source dtypes surface as
/// [`CoreMLError::UnsupportedOutputDtype`] rather than silently
/// mis-decoding; see that method's documentation for the current
/// coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlArrayDtype {
    /// IEEE 754 binary32 (`MLMultiArrayDataTypeFloat32` /
    /// `MLMultiArrayDataTypeFloat`).
    F32,
    /// IEEE 754 binary16 (`MLMultiArrayDataTypeFloat16`) — 2 bytes per
    /// element, stored exactly as CoreML laid it out; never up-converted
    /// to `f32` by [`RawArray`]-producing APIs.
    F16,
    /// IEEE 754 binary64 (`MLMultiArrayDataTypeDouble` /
    /// `MLMultiArrayDataTypeFloat64`).
    F64,
    /// Signed 32-bit integer (`MLMultiArrayDataTypeInt32`).
    I32,
    /// Signed 8-bit integer (`MLMultiArrayDataTypeInt8`).
    I8,
}

/// A CoreML `MLMultiArray`'s contents extracted verbatim — dtype
/// preserved, no up-conversion — as opposed to
/// [`predict`](crate::MlPackageModel::predict), which always returns
/// `f32` [`Tensor`]s.
///
/// `data` is always a tightly packed, C-contiguous ("row-major") run of
/// `shape.iter().product()` elements of `dtype`, regardless of whatever
/// strides CoreML's own buffer used internally — the same stride-aware
/// copy `predict` relies on normalizes the layout during extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawArray {
    /// Element counts, outermost dimension first.
    pub shape: Vec<usize>,
    /// Element dtype `data` is encoded as.
    pub dtype: MlArrayDtype,
    /// Tightly packed, C-contiguous, native-endian element bytes —
    /// `shape.iter().product::<usize>() * dtype`'s element width, in
    /// total length.
    pub data: Vec<u8>,
}

/// The element payload of an `MLSequence` — the contents of
/// [`FeatureOutput::Sequence`].
///
/// `MLSequence` (CoreML's ordered, homogeneously-typed collection) only
/// ever holds `Int64` or `String` elements — its own `type` property
/// (an `MLFeatureType`) is always one of those two values — so there is
/// no third variant to add here.
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceValue {
    /// An ordered run of 64-bit integers (`MLFeatureTypeInt64` elements).
    Int64(Vec<i64>),
    /// An ordered run of strings (`MLFeatureTypeString` elements).
    String(Vec<String>),
}

/// A single CoreML output feature, typed by its `MLFeatureType`.
///
/// [`MlPackageModel::predict`](crate::MlPackageModel::predict) only ever
/// surfaces [`FeatureOutput::MultiArray`]-shaped outputs directly as a
/// plain [`Tensor`], for backward compatibility and because every
/// OxiFace sub-gate model (ArcFace / SCRFD / InSwapper) is multi-array
/// only.
/// [`MlPackageModel::predict_features`](crate::MlPackageModel::predict_features)
/// surfaces *every* declared output through this richer enum instead,
/// covering all `MLFeatureType` variants that have a portable
/// representation.  `MLFeatureTypeInvalid` and `MLFeatureTypeState` are
/// not representable here — see `predict_features`'s documentation for
/// why (and what error surfaces for them instead).
///
/// Does not derive `PartialEq` (unlike [`SequenceValue`]): its
/// `MultiArray`/`Image` variants embed [`Tensor`], which does not
/// implement `PartialEq`. Compare via `match` on the
/// specific variant instead.
#[derive(Debug, Clone)]
pub enum FeatureOutput {
    /// `MLFeatureTypeMultiArray` — an n-dimensional numeric array, decoded
    /// exactly like [`predict`](crate::MlPackageModel::predict)'s outputs
    /// (dtype-converted to `f32`, stride-normalized to C-contiguous).
    MultiArray(Tensor),
    /// `MLFeatureTypeImage` (`CVPixelBufferRef`) — decoded into a `Tensor`
    /// for a bounded set of standard pixel formats; see
    /// [`MlPackageModel::predict_features`](crate::MlPackageModel::predict_features)
    /// for exactly which formats are supported, and each format's
    /// resulting shape/layout.
    Image(Tensor),
    /// `MLFeatureTypeSequence` — an ordered, homogeneously-typed run of
    /// `Int64` or `String` values.
    Sequence(SequenceValue),
    /// `MLFeatureTypeDictionary` — sparse numeric weights (e.g. class
    /// probabilities) keyed by a stringified dictionary key.  CoreML
    /// guarantees dictionary keys are always `NSNumber` or `NSString`;
    /// both stringify losslessly (numbers via their decimal form).
    Dictionary(HashMap<String, f64>),
    /// `MLFeatureTypeString` — a single string value.
    String(String),
    /// `MLFeatureTypeInt64` — a single 64-bit integer value.
    Int64(i64),
    /// `MLFeatureTypeDouble` — a single double-precision value.
    Double(f64),
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]
mod array_read;
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]
mod compile_cache;
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]
mod macos_impl;
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]
pub use macos_impl::MlPackageModel;

// ──────────────────────────────────────────────────────────────────────────── //
// Non-Apple stub — preserves the API surface so dependent crates compile     //
// portably.  Every method short-circuits to `UnsupportedPlatform`.            //
// ──────────────────────────────────────────────────────────────────────────── //

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
)))]
mod stub_impl;
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
)))]
pub use stub_impl::MlPackageModel;

// ──────────────────────────────────────────────────────────────────────────── //
// Tests — every test that hits the framework is `#[ignore]` because the      //
// bundles live outside the source tree.  Manual run command:                 //
//                                                                            //
//     cargo test -p oxionnx-coreml -- --ignored --test-threads=1             //
// ──────────────────────────────────────────────────────────────────────────── //

#[cfg(all(
    test,
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
mod tests;

#[cfg(all(
    test,
    not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    ))
))]
mod stub_tests;
