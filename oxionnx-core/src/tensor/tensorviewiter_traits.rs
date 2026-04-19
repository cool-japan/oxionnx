//! # TensorViewIter - Trait Implementations
//!
//! This module contains trait implementations for `TensorViewIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//! - `ExactSizeIterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::compute_strides;
use super::types::TensorViewIter;

impl Iterator for TensorViewIter<'_> {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.exhausted {
            return None;
        }
        let val = self.get_at(&self.indices);
        let ndim = self.shape.len();
        let mut carry = true;
        for i in (0..ndim).rev() {
            if carry {
                self.indices[i] += 1;
                if self.indices[i] < self.shape[i] {
                    carry = false;
                } else {
                    self.indices[i] = 0;
                }
            }
        }
        if carry {
            self.exhausted = true;
        }
        val
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.exhausted {
            return (0, Some(0));
        }
        let total: usize = self.shape.iter().product();
        let mut consumed = 0usize;
        let logical_strides = compute_strides(&self.shape);
        for (i, &idx) in self.indices.iter().enumerate() {
            consumed += idx * logical_strides[i];
        }
        let remaining = total.saturating_sub(consumed);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TensorViewIter<'_> {}
