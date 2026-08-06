//! GPU context module — wgpu device/queue, compute pipelines, and the buffer pool.

pub mod functions;
pub mod tracker_pool;
pub mod types;

pub use tracker_pool::{GpuBufferPool, DEFAULT_POOL_BYTE_BUDGET};
pub use types::GpuContext;
