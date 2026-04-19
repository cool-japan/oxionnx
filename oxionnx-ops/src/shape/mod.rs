//! Shape manipulation operations for tensors.
//!
//! This module provides ONNX-compatible shape operations including:
//! - Basic shape transforms: reshape, flatten, transpose, squeeze, unsqueeze
//! - Sequence operations: concat, slice, pad, split, tile
//! - Spatial rearrangements: depth_to_space, space_to_depth, reverse_sequence

pub mod basic;
pub mod sequence;
pub mod spatial;

#[cfg(test)]
#[path = "tests.rs"]
mod shape_tests;

pub use basic::{flatten, reshape, squeeze, transpose, unsqueeze};
pub use sequence::{concat, pad, slice, split, tile};
pub use spatial::{depth_to_space, reverse_sequence, space_to_depth};
