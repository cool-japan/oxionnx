//! Neural network operator implementations: activations and normalizations.

pub mod activations;
pub mod normalization;

#[cfg(test)]
mod tests;

// Re-export all public items so callers use `crate::nn::relu`, `crate::nn::softmax`, etc.
pub use activations::*;
pub use normalization::*;
