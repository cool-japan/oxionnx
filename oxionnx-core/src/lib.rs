//! oxionnx-core — Core types for the oxionnx ONNX inference engine.
//!
//! ## `no_std` support
//!
//! This crate builds with `--no-default-features` (`alloc`-only, no `std`).
//! Every module resolves `Vec`/`String`/`format!`/etc. through `alloc`, and
//! `HashMap`/`HashSet` fall back to `hashbrown` (already a non-optional
//! dependency) when `std` is off. Verify locally with:
//!
//! ```text
//! cargo check -p oxionnx-core --no-default-features
//! cargo check -p oxionnx-core --no-default-features --features ndarray
//! ```
//!
//! Note this proves *source-level* alloc-only correctness, not a bare-metal
//! link: `half` and `hashbrown` are pulled in with their own default
//! (std-enabled) features, since flipping those off is a separate,
//! independently-riskier change outside this crate's own feature surface.
//! Do **not** run `cargo check --no-default-features --all-targets` (or
//! `--tests`) — `oxionnx-core/tests/*.rs` integration tests use `#[test]`,
//! which needs the standard test harness (`std`) to link regardless of this
//! crate's own feature flags; that failure is a test-harness limitation, not
//! a regression in the library's `no_std` support.
//!
//! No in-workspace crate builds `oxionnx-core` with `default-features =
//! false` today (verified by grepping every member `Cargo.toml`), so the
//! `std` (default) build remains the only path exercised by the workspace's
//! own tests; the check above is the standing regression guard for the
//! advertised `no_std` support itself and for external `no_std` consumers.

#![cfg_attr(not(feature = "std"), no_std)]

// `alloc` is linked unconditionally (not just when the `std` feature is off).
// This lets every module in this crate spell allocation-backed types as
// `alloc::vec::Vec` / `alloc::string::String` / etc. *unconditionally*
// (no per-import `#[cfg(...)]`), because `alloc::vec::Vec` and
// `std::vec::Vec` are the exact same type either way -- `std` itself is
// built on top of `alloc` and re-exports these items rather than
// redefining them. Modules that need genuinely `std`-only functionality
// (e.g. `std::error::Error`, filesystem/threading APIs) still gate those
// specific items behind `#[cfg(feature = "std")]`.
extern crate alloc;

pub mod dtype;
pub mod error;
pub mod graph;
pub mod operator;
pub mod operator_slots;
pub mod operator_typed;
pub mod tensor;

pub use dtype::{promote, DType, TensorStorage, TypedTensor};
pub use error::OnnxError;
pub use graph::{Attributes, Dim, Graph, Node, NodeInfo, OpKind, TensorInfo};
pub use operator::{OpContext, Operator, OperatorRegistry, TypedOpContext};
pub use operator_slots::default_into_slots;
pub use operator_typed::default_typed_via_f32;
pub use tensor::{
    compute_strides, convert_layout, nchw_to_nhwc, nhwc_to_nchw, BroadcastIter, Tensor,
    TensorLayout,
};
