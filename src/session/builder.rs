use crate::execution_providers::OpPlacement;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;
use std::path::Path;

use super::types::{raw_meta_to_model_metadata, ModelMetadata, OptLevel};
use super::Session;
use oxionnx_core::OperatorRegistry;

/// Builder for configuring and creating a Session.
pub struct SessionBuilder {
    pub(crate) opt_level: OptLevel,
    pub(crate) registry: Option<OperatorRegistry>,
    pub(crate) enable_profiling: bool,
    pub(crate) enable_memory_pool: bool,
    pub(crate) parallel: bool,
    pub(crate) mixed_precision: bool,
    pub(crate) num_threads: Option<usize>,
    pub(crate) op_placement: OpPlacement,
}

impl SessionBuilder {
    /// Create a new builder with default settings (all optimizations, no profiling, no pool,
    /// sequential execution).
    pub fn new() -> Self {
        Self {
            opt_level: OptLevel::All,
            registry: None,
            enable_profiling: false,
            enable_memory_pool: true,
            parallel: false,
            mixed_precision: false,
            num_threads: None,
            op_placement: OpPlacement::default(),
        }
    }

    /// Set the optimization level for graph optimization passes.
    pub fn with_optimization_level(mut self, level: OptLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Set a custom operator registry.
    pub fn with_registry(mut self, registry: OperatorRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Enable per-node profiling during `run()`.
    pub fn with_profiling(mut self) -> Self {
        self.enable_profiling = true;
        self
    }

    /// Enable the activation memory pool for buffer reuse during inference.
    pub fn with_memory_pool(mut self, enabled: bool) -> Self {
        self.enable_memory_pool = enabled;
        self
    }

    /// Enable or disable multi-threaded parallel execution of independent nodes.
    /// When enabled, nodes at the same topological depth are executed concurrently
    /// using rayon. On `wasm32` targets, this flag is ignored and execution is
    /// always sequential. Default: `false`.
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Enable mixed-precision inference (f16 activations, f32 accumulation).
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Set operator placement strategy for routing ops to CPU/GPU.
    pub fn with_op_placement(mut self, placement: OpPlacement) -> Self {
        self.op_placement = placement;
        self
    }

    /// Load an ONNX model from a file path.
    /// Supports models with external data by resolving paths relative to the file's directory.
    pub fn load(self, path: &Path) -> Result<Session, OnnxError> {
        let bytes = std::fs::read(path).map_err(|e| {
            OnnxError::Parse(format!("Cannot read ONNX file {}: {e}", path.display()))
        })?;
        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (raw_meta, graph, weights) =
            crate::model::load_with_metadata_and_path(&bytes, base_path)
                .map_err(OnnxError::Parse)?;
        let metadata = raw_meta_to_model_metadata(raw_meta);
        Session::build_from_graph(
            graph,
            weights,
            metadata,
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
        )
    }

    /// Load an ONNX model from a file using memory mapping.
    ///
    /// The file is memory-mapped instead of being read entirely into a `Vec<u8>`.
    /// This lets the OS virtual-memory subsystem page out weight data that is not
    /// actively used, reducing resident memory for large models.
    #[cfg(feature = "mmap")]
    pub fn load_mmap(self, path: &Path) -> Result<Session, OnnxError> {
        let mmap_model =
            oxionnx_proto::mmap_loader::MmapModel::open(path).map_err(OnnxError::Parse)?;
        let (graph, weights) = mmap_model.into_parts();
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        Session::build_from_graph(
            graph,
            weights,
            ModelMetadata::default(),
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
        )
    }

    /// Load an ONNX model from raw bytes.
    pub fn load_from_bytes(self, bytes: &[u8]) -> Result<Session, OnnxError> {
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (raw_meta, graph, weights) =
            crate::model::load_with_metadata(bytes).map_err(OnnxError::Parse)?;
        let metadata = raw_meta_to_model_metadata(raw_meta);
        Session::build_from_graph(
            graph,
            weights,
            metadata,
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
        )
    }

    /// Load an ONNX model from a `Read` source (streaming).
    ///
    /// Parses the model incrementally from the reader without loading the entire
    /// file into memory at once. Useful for multi-GB models.
    pub fn load_from_reader<R: std::io::Read>(self, reader: R) -> Result<Session, OnnxError> {
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (graph_proto, weights) =
            oxionnx_proto::parse_streaming(reader).map_err(OnnxError::Parse)?;
        let graph = oxionnx_proto::build_graph(&graph_proto, &weights).map_err(OnnxError::Parse)?;
        Session::build_from_graph(
            graph,
            weights,
            ModelMetadata::default(),
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
        )
    }

    /// Load an ONNX model with selective weight loading.
    ///
    /// The `weight_filter` closure receives each weight's name and shape.
    /// If it returns `true`, the weight is loaded; if `false`, it is skipped.
    /// This is useful for loading only needed layers from a large model.
    pub fn load_filtered<F>(self, path: &Path, weight_filter: F) -> Result<Session, OnnxError>
    where
        F: FnMut(&str, &[usize]) -> bool,
    {
        let file = std::fs::File::open(path).map_err(|e| {
            OnnxError::Parse(format!("Cannot read ONNX file {}: {e}", path.display()))
        })?;
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (graph_proto, weights) = oxionnx_proto::parse_with_weight_filter(file, weight_filter)
            .map_err(OnnxError::Parse)?;
        let graph = oxionnx_proto::build_graph(&graph_proto, &weights).map_err(OnnxError::Parse)?;
        Session::build_from_graph(
            graph,
            weights,
            ModelMetadata::default(),
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
        )
    }

    /// Build a Session from a pre-parsed Graph and weights.
    pub fn build_from_graph(
        self,
        graph: Graph,
        weights: HashMap<String, Tensor>,
    ) -> Result<Session, OnnxError> {
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        Session::build_from_graph(
            graph,
            weights,
            ModelMetadata::default(),
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
        )
    }

    // ── ort-compatibility aliases ────────────────────────────────────────────

    /// `ort`-compatible alias for [`SessionBuilder::load`].
    ///
    /// Allows callers migrating from `ort` to use `commit_from_file` without
    /// changing call-sites.
    pub fn commit_from_file(self, path: impl AsRef<std::path::Path>) -> Result<Session, OnnxError> {
        self.load(path.as_ref())
    }

    /// `ort`-compatible alias for [`SessionBuilder::load_from_bytes`].
    pub fn commit_from_memory(self, bytes: &[u8]) -> Result<Session, OnnxError> {
        self.load_from_bytes(bytes)
    }

    /// Set the number of threads for intra-op parallelism.
    ///
    /// When set, a per-session rayon thread pool is created with this many threads.
    /// If not set (or on `wasm32`), the global rayon pool is used.
    /// Also enables parallel execution automatically.
    pub fn with_intra_threads(mut self, n: usize) -> Self {
        self.num_threads = Some(n);
        self.parallel = true;
        self
    }

    /// Set the number of threads for inter-op parallelism.
    ///
    /// Currently an alias for [`SessionBuilder::with_intra_threads`].
    pub fn with_inter_threads(mut self, n: usize) -> Self {
        self.num_threads = Some(n);
        self.parallel = true;
        self
    }

    /// `ort`-compatible no-op for execution provider selection.
    ///
    /// oxionnx's runtime backend is determined at compile time via feature flags
    /// (`gpu`, `cuda`).  This method accepts any iterator of
    /// [`crate::ExecutionProviderDispatch`] values so that ort-style provider-registration
    /// code compiles without modification.
    pub fn with_execution_providers<I>(self, _providers: I) -> Self
    where
        I: IntoIterator<Item = crate::execution_providers::ExecutionProviderDispatch>,
    {
        self
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}
