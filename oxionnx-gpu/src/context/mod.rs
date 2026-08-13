//! GPU context module — wgpu device/queue, compute pipelines, and the buffer pool.

pub mod activation;
pub mod budget;
pub mod functions;
pub mod init_error;
pub mod resident;
pub mod tracker_pool;
pub mod tuning;
pub mod types;
pub mod weight_format;

pub use activation::{
    skips_size_threshold, DeviceTensor, GpuOutput, OutputPlacement, TensorSource,
};
pub use budget::{GpuMemoryBudget, TrackedBuffer, DEFAULT_LIVE_BYTE_BUDGET};
pub use init_error::{GpuInitDiagnostic, GpuInitError};
pub use resident::{ResidentCounters, WeightKeys};
pub use tracker_pool::{GpuBufferPool, DEFAULT_POOL_BYTE_BUDGET};
pub use tuning::{GemmWeightTraffic, GpuPerfClass, GpuTuning};
pub use types::GpuContext;
pub use weight_format::{WeightBytes, WeightFormat};
