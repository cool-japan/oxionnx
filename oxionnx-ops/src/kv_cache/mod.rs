//! Key-Value cache for autoregressive (incremental) inference.
//!
//! Split into submodules:
//! - `types`: [`KvCache`] struct definition and implementation
//! - `tests`: integration tests (cfg(test) only)

pub mod types;

#[cfg(test)]
mod tests;

// Re-export the public type
pub use types::KvCache;
