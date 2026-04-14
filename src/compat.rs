//! Compatibility layer for easy migration from the `ort` crate.
//!
//! This module provides type aliases and re-exports that mirror the public API
//! of `ort` 2.x so that downstream crates (torsh, oxigdal, oxify) can switch
//! from `ort` to `oxionnx` with minimal code changes.

/// `ort`-compatible alias for [`crate::OptLevel`].
///
/// In `ort` this is `GraphOptimizationLevel`; here it maps directly to our
/// internal optimization enum.
pub use crate::OptLevel as GraphOptimizationLevel;

/// `ort`-compatible type alias for the output map returned by `Session::run`.
///
/// In `ort` 2.x, `Session::run` returns `SessionOutputs<'_>` which is
/// essentially a `HashMap<String, Value>`.  Here we use `HashMap<String, Tensor>`
/// as the equivalent.
pub type SessionOutputs<'a> = std::collections::HashMap<String, crate::Tensor>;
