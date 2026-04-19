//! Symbolic shape propagation for dynamic dimensions.
//!
//! Extends the concrete shape inference in [`super::shape_inference`] to handle
//! dimensions that remain unresolved until runtime (e.g. batch size, sequence
//! length). Each dimension is represented as [`SymDim`] — either a known
//! `usize` or a named symbol.
//!
//! ## Module structure
//!
//! - `types`     — [`SymDim`], [`SymbolicShape`], [`SymbolEnv`]
//! - `utils`     — resolution, construction, and broadcasting helpers
//! - `ops`       — per-operator symbolic shape helpers
//! - `inference` — single-node dispatcher and graph-level propagation
#![allow(rustdoc::private_intra_doc_links)]

mod inference;
mod ops;
mod types;
mod utils;

// Re-export the full public API so callers see no change.
pub use inference::infer_symbolic_shapes;
pub use types::{SymDim, SymbolEnv, SymbolicShape};
pub use utils::{broadcast_symbolic, from_concrete, resolve_shape, symbolic_numel};

#[cfg(test)]
mod tests;
