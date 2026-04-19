//! Utility functions for symbolic shape manipulation.
//!
//! Provides resolution, construction, and broadcasting helpers for
//! [`SymbolicShape`] values.

use super::types::{SymDim, SymbolEnv, SymbolicShape};

/// Resolve a symbolic shape to concrete dimensions using the given environment.
/// Returns `None` if any symbol cannot be resolved.
pub fn resolve_shape(shape: &[SymDim], env: &SymbolEnv) -> Option<Vec<usize>> {
    shape
        .iter()
        .map(|d| match d {
            SymDim::Known(v) => Some(*v),
            SymDim::Symbol(s) => env.get(s).copied(),
        })
        .collect()
}

/// Convert a concrete shape to a symbolic shape (all dimensions known).
pub fn from_concrete(shape: &[usize]) -> SymbolicShape {
    shape.iter().map(|&d| SymDim::Known(d)).collect()
}

/// Compute the total number of elements if all dimensions are known.
/// Returns `None` when any dimension is symbolic.
pub fn symbolic_numel(shape: &[SymDim]) -> Option<usize> {
    let mut total = 1usize;
    for d in shape {
        match d {
            SymDim::Known(v) => {
                total = total.checked_mul(*v)?;
            }
            SymDim::Symbol(_) => return None,
        }
    }
    Some(total)
}

/// Try to broadcast two symbolic shapes following NumPy broadcasting rules.
///
/// Returns `None` when the shapes are provably incompatible or when two
/// different symbolic dimensions appear in the same position without one of
/// them being the literal 1.
pub fn broadcast_symbolic(a: &[SymDim], b: &[SymDim]) -> Option<SymbolicShape> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    let a_pad = n - a.len();
    let b_pad = n - b.len();
    let one = SymDim::Known(1);
    for i in 0..n {
        let ai = if i < a_pad { &one } else { &a[i - a_pad] };
        let bi = if i < b_pad { &one } else { &b[i - b_pad] };
        match (ai, bi) {
            (SymDim::Known(1), other) | (other, SymDim::Known(1)) => out.push(other.clone()),
            (SymDim::Known(a_val), SymDim::Known(b_val)) => {
                if a_val != b_val {
                    return None;
                }
                out.push(SymDim::Known(*a_val));
            }
            (SymDim::Symbol(s1), SymDim::Symbol(s2)) if s1 == s2 => {
                out.push(SymDim::Symbol(s1.clone()));
            }
            // A symbol paired with itself-as-known(1) was already handled above.
            // Two different symbols or symbol vs concrete > 1 cannot be resolved.
            _ => return None,
        }
    }
    Some(out)
}
