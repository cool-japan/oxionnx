//! # oxionnx-coreml
//!
//! Apple **CoreML** execution provider for the OxiONNX inference engine.
//!
//! This crate is a *thin runtime* around `MLModel`: it loads pre-converted
//! `.mlpackage` (or already-compiled `.mlmodelc`) bundles and runs whole-graph
//! inference through Apple's scheduler, which transparently dispatches each
//! op to the best available compute unit (Apple Neural Engine, integrated
//! GPU, or CPU).
//!
//! The crate intentionally exposes **no per-op kernel surface**.  CoreML's
//! whole-graph optimiser is what unlocks the dramatic speedups OxiFace's
//! sub-gates measured on Apple Silicon M3:
//!
//! | Model       | Speedup vs CPU | ANE engagement |
//! | :---------- | :------------- | :------------- |
//! | ArcFace     | 17.4×          | 97 %           |
//! | SCRFD       | 25.3×          | 97 %           |
//! | InSwapper   | 7.91×          | 100 %          |
//!
//! Per-op dispatch through `try_*_dispatch` would give back every property
//! that produced these numbers (graph fusion, kernel selection, memory
//! reuse), so the design treats CoreML as a sibling whole-model session
//! rather than a node-level execution provider.
//!
//! ## Getting started
//!
//! ```no_run
//! use std::collections::HashMap;
//! use oxionnx_coreml::{MlPackageModel, MlComputeUnits};
//! use oxionnx_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = MlPackageModel::load(
//!     "/path/to/arcface.mlpackage",
//!     MlComputeUnits::All,
//! )?;
//!
//! let mut inputs = HashMap::new();
//! let n = 3 * 112 * 112;
//! let input_name = model
//!     .input_names()
//!     .into_iter()
//!     .next()
//!     .ok_or("no input name")?;
//! inputs.insert(
//!     input_name,
//!     Tensor::new(vec![0.0_f32; n], vec![1, 3, 112, 112]),
//! );
//!
//! let outputs = model.predict(&inputs)?;
//! let summary = model.compute_plan_summary()?;
//! println!("ANE engagement: {:.1}%", summary.ane_fraction() * 100.0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Platform support
//!
//! The `objc2`-backed runtime is gated in for every Apple platform CoreML
//! itself ships on — macOS, iOS, tvOS, and visionOS — behind a single
//! `cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos"))`.
//! macOS and iOS builds (`aarch64-apple-ios`) are compile-validated by this
//! crate directly; tvOS/visionOS are Rust tier-3 targets (nightly +
//! `-Zbuild-std`) sharing the identical CoreML API surface, so the gate
//! includes them, but they are not independently compile-validated here —
//! see `TODO.md` for the exact validation matrix.  On every other target
//! (Linux, Windows, wasm, …) the crate compiles to a stub that surfaces
//! [`CoreMLError::UnsupportedPlatform`] from every entry point — so
//! downstream code that conditionally enables this provider does not need
//! to fence its own code paths.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_safety_doc)]

pub mod compute;
pub mod error;
pub mod package;

pub use compute::{ComputePlanSummary, MlComputeUnits};
pub use error::{CoreMLError, Result};
pub use package::{FeatureOutput, MlArrayDtype, MlPackageModel, RawArray, SequenceValue};
