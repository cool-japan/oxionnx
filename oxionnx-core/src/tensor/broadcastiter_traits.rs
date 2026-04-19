//! # BroadcastIter - Trait Implementations
//!
//! This module contains trait implementations for `BroadcastIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//! - `ExactSizeIterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BroadcastIter;

impl<'a> Iterator for BroadcastIter<'a> {
    type Item = (f32, f32);
    fn next(&mut self) -> Option<(f32, f32)> {
        if self.idx >= self.total {
            return None;
        }
        let mut a_flat = 0usize;
        let mut b_flat = 0usize;
        let mut remaining = self.idx;
        for dim in 0..self.output_shape.len() {
            let coord = remaining / self.output_strides[dim];
            remaining %= self.output_strides[dim];
            a_flat += coord * self.a_strides[dim];
            b_flat += coord * self.b_strides[dim];
        }
        self.idx += 1;
        Some((self.a_data[a_flat], self.b_data[b_flat]))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total - self.idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BroadcastIter<'_> {}
