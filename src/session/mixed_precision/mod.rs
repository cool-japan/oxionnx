//! Mixed precision inference support.
//!
//! Classifies operators as f16-safe or f32-required, and provides helpers
//! for executing element-wise operations natively in f16 using the `half` crate.

#![allow(dead_code, unused_imports)]

mod broadcast;
mod classify;
mod elementwise;
mod precision;

pub use classify::{requires_f32, should_use_f16};
pub use elementwise::execute_elementwise_f16;
pub use precision::{next_consumers_all_f16, round_to_f16_precision};

#[cfg(test)]
mod tests;
