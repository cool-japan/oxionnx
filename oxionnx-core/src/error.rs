//! Typed error returned by all public oxionnx methods.

// `core::fmt` (not `std::fmt`): this crate supports `no_std` (`alloc`-only)
// builds via the `std` feature (default-on). `core::fmt::{Display, Debug,
// Formatter, Result}` are the exact same items `std::fmt` re-exports, so
// this works identically in both std and no_std builds -- no `#[cfg(...)]`
// needed for this import specifically.
use core::fmt;

// `alloc::string::String`: every `OnnxError` variant carries a message
// string. `alloc` is always linked by the crate root (see lib.rs), so this
// resolves to the same `String` as `std::string::String` in std builds too.
use alloc::string::String;

/// Typed error returned by all public `Session` methods.
///
/// `#[non_exhaustive]`: new failure modes are added to this enum as the
/// engine grows (e.g. new dispatch backends, new validation checks).
/// Downstream `match`es must carry a wildcard arm so those additions are
/// non-breaking; construction of existing variants is unaffected.
#[derive(Debug)]
#[non_exhaustive]
pub enum OnnxError {
    /// ONNX protobuf parsing or decoding failure.
    Parse(String),
    /// An operator referenced by the model is not in the registry.
    UnknownOp(String),
    /// Tensor dimensions are incompatible for the requested operation.
    ShapeMismatch(String),
    /// A required tensor (input or weight) was not found in the value map.
    TensorNotFound(String),
    /// A feature or data type is recognized but not yet implemented.
    Unsupported(String),
    /// An ONNX operator is recognized but not yet implemented.
    UnsupportedOp(String),
    /// The model structure is invalid (e.g. cyclic graph, missing outputs).
    InvalidModel(String),
    /// Catch-all for unexpected internal failures.
    Internal(String),
    /// Inference was cancelled via a `CancellationToken`.
    Cancelled(String),
    /// A tensor dtype is incompatible with the requested operation or dispatch path.
    DTypeMismatch(String),
    /// An arithmetic error occurred during computation (e.g. integer division by zero).
    Arithmetic(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(s) => write!(f, "Parse error: {s}"),
            Self::UnknownOp(s) => write!(f, "Unknown op: {s}"),
            Self::ShapeMismatch(s) => write!(f, "Shape mismatch: {s}"),
            Self::TensorNotFound(s) => write!(f, "Tensor not found: {s}"),
            Self::Unsupported(s) => write!(f, "Unsupported: {s}"),
            Self::UnsupportedOp(s) => write!(f, "Unsupported op: {s}"),
            Self::InvalidModel(s) => write!(f, "Invalid model: {s}"),
            Self::Internal(s) => write!(f, "Internal error: {s}"),
            Self::Cancelled(s) => write!(f, "Cancelled: {s}"),
            Self::DTypeMismatch(s) => write!(f, "DType mismatch: {s}"),
            Self::Arithmetic(s) => write!(f, "Arithmetic error: {s}"),
        }
    }
}

// `core::error::Error` only became stable in Rust 1.81; this crate's MSRV is
// 1.75 (see workspace `rust-version`), so the trait impl is only provided
// under the `std` feature rather than attempting a `core::error::Error`
// impl that would raise the effective MSRV for `no_std` users.
#[cfg(feature = "std")]
impl std::error::Error for OnnxError {}

impl From<String> for OnnxError {
    fn from(s: String) -> Self {
        Self::Internal(s)
    }
}
