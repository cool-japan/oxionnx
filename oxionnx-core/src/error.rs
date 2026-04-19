//! Typed error returned by all public oxionnx methods.

use std::fmt;

/// Typed error returned by all public `Session` methods.
#[derive(Debug)]
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

impl std::error::Error for OnnxError {}

impl From<String> for OnnxError {
    fn from(s: String) -> Self {
        Self::Internal(s)
    }
}
