use crate::execution_providers::{OpPlacement, ProviderKind};
use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use oxionnx_core::OperatorRegistry;
use std::collections::HashMap;
use std::sync::Mutex;

mod accessors;
mod builder;
#[cfg(feature = "gpu")]
mod gpu_dispatch;
mod loading;
pub(crate) mod mixed_precision;
mod run;
mod tests;
pub mod types;

pub use types::{ModelInfo, ModelMetadata, NodeProfile, OptLevel};

#[cfg(feature = "gpu")]
pub use gpu_dispatch::GpuExecutionProvider;

pub use builder::SessionBuilder;

/// A loaded ONNX model ready for inference.
pub struct Session {
    pub(crate) sorted_nodes: Vec<crate::graph::Node>,
    pub(crate) weights: HashMap<String, Tensor>,
    pub(crate) input_names: Vec<String>,
    pub(crate) output_names: Vec<String>,
    /// Detailed metadata for graph inputs (from ValueInfoProto).
    pub(crate) input_infos: Vec<oxionnx_core::TensorInfo>,
    /// Detailed metadata for graph outputs (from ValueInfoProto).
    pub(crate) output_infos: Vec<oxionnx_core::TensorInfo>,
    /// Model-level metadata (producer, IR version, opset imports, etc.).
    pub(crate) metadata: ModelMetadata,
    pub(crate) registry: OperatorRegistry,
    pub(crate) profiling_data: Option<Mutex<Vec<NodeProfile>>>,
    pub(crate) pool: Option<Mutex<SizeClassPool>>,
    pub(crate) shape_cache: Option<HashMap<String, Vec<usize>>>,
    /// Whether to use rayon-based parallel execution for independent nodes.
    pub(crate) parallel: bool,
    /// Whether to use mixed-precision inference (f16 activations, f32 accumulation).
    pub(crate) mixed_precision: bool,
    /// Operator placement strategy for CPU/GPU routing.
    pub(crate) op_placement: OpPlacement,
    /// Ordered list of execution provider backends to attempt, in priority order.
    ///
    /// When non-empty, the dispatch loop in `run_sequential_inner` iterates this
    /// list for every node and uses the first provider that returns a result.
    /// CPU is always the implicit terminal fallback.
    ///
    /// When empty, the legacy feature-flag heuristic dispatch is used.
    pub(crate) providers: Vec<ProviderKind>,
    /// Current dynamic dimension bindings, updated on each `run()` call.
    /// Maps symbolic dimension names (e.g. "batch_size") to concrete values.
    pub(crate) dynamic_dims: Mutex<HashMap<String, usize>>,
    /// Resolved intermediate tensor shapes, invalidated when `dynamic_dims` change.
    pub(crate) resolved_shapes: Mutex<HashMap<String, Vec<usize>>>,
    /// Per-session rayon thread pool for parallel execution.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) thread_pool: Option<rayon::ThreadPool>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu: Option<crate::gpu::GpuContext>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Option<oxionnx_cuda::CudaContext>,
    #[cfg(feature = "directml")]
    pub(crate) dml: Option<oxionnx_directml::DirectMLContext>,
}
