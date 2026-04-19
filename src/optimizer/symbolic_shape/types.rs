//! Core symbolic dimension types.
//!
//! Provides [`SymDim`], [`SymbolicShape`], and [`SymbolEnv`] — the fundamental
//! building blocks for symbolic shape propagation.

use std::collections::HashMap;

/// A dimension that may be either a concrete value or a symbolic name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymDim {
    /// A known concrete dimension.
    Known(usize),
    /// A symbolic/dynamic dimension (e.g., `"batch_size"`, `"seq_len"`).
    Symbol(String),
}

impl SymDim {
    /// Return the concrete value if known.
    pub fn as_known(&self) -> Option<usize> {
        match self {
            Self::Known(v) => Some(*v),
            Self::Symbol(_) => None,
        }
    }

    /// Return the symbol name if symbolic.
    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            Self::Known(_) => None,
            Self::Symbol(s) => Some(s),
        }
    }

    /// Check if the dimension is known (concrete).
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }
}

impl std::fmt::Display for SymDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Known(v) => write!(f, "{v}"),
            Self::Symbol(s) => write!(f, "{s}"),
        }
    }
}

/// A shape where each dimension may be concrete or symbolic.
pub type SymbolicShape = Vec<SymDim>;

/// Environment mapping symbol names to concrete values.
pub type SymbolEnv = HashMap<String, usize>;
