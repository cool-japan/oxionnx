use crate::execution_providers::OpPlacement;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::OperatorRegistry;
use std::collections::HashMap;
use std::path::Path;

use super::types::{raw_meta_to_model_metadata, ModelMetadata, OptLevel};
use super::Session;

impl Session {
    /// Load an ONNX model from a `.onnx` file.
    /// Supports models with external data by resolving paths relative to the file's directory.
    pub fn from_file(path: &Path) -> Result<Self, OnnxError> {
        let bytes = std::fs::read(path).map_err(|e| {
            OnnxError::Parse(format!("Cannot read ONNX file {}: {e}", path.display()))
        })?;
        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let registry = oxionnx_ops::default_registry();
        let (raw_meta, graph, weights) =
            crate::model::load_with_metadata_and_path(&bytes, base_path)
                .map_err(OnnxError::Parse)?;
        let metadata = raw_meta_to_model_metadata(raw_meta);
        Self::build_from_graph(
            graph,
            weights,
            metadata,
            registry,
            OptLevel::All,
            false,
            false,
            false,
            false,
            None,
            OpPlacement::default(),
        )
    }

    /// Load an ONNX model from raw bytes, using the default operator registry.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        Self::from_bytes_with_registry(bytes, oxionnx_ops::default_registry())
    }

    /// Load an ONNX model from a `.onnx` file, with a custom operator registry.
    /// Supports models with external data by resolving paths relative to the file's directory.
    pub fn from_file_with_registry(
        path: &Path,
        registry: OperatorRegistry,
    ) -> Result<Self, OnnxError> {
        let bytes = std::fs::read(path).map_err(|e| {
            OnnxError::Parse(format!("Cannot read ONNX file {}: {e}", path.display()))
        })?;
        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let (raw_meta, graph, weights) =
            crate::model::load_with_metadata_and_path(&bytes, base_path)
                .map_err(OnnxError::Parse)?;
        let metadata = raw_meta_to_model_metadata(raw_meta);
        Self::build_from_graph(
            graph,
            weights,
            metadata,
            registry,
            OptLevel::All,
            false,
            false,
            false,
            false,
            None,
            OpPlacement::default(),
        )
    }

    /// Load an ONNX model from raw bytes, with a custom operator registry.
    pub fn from_bytes_with_registry(
        bytes: &[u8],
        registry: OperatorRegistry,
    ) -> Result<Self, OnnxError> {
        let (raw_meta, graph, weights) =
            crate::model::load_with_metadata(bytes).map_err(OnnxError::Parse)?;
        let metadata = raw_meta_to_model_metadata(raw_meta);
        Self::build_from_graph(
            graph,
            weights,
            metadata,
            registry,
            OptLevel::All,
            false,
            false,
            false,
            false,
            None,
            OpPlacement::default(),
        )
    }

    /// Create a Session directly from a Graph and weights.
    /// Useful for testing and programmatic graph construction.
    pub fn from_graph(graph: Graph, weights: HashMap<String, Tensor>) -> Result<Self, OnnxError> {
        Self::from_graph_with_registry(graph, weights, oxionnx_ops::default_registry())
    }

    /// Create a Session from a Graph and weights with a custom operator registry.
    pub fn from_graph_with_registry(
        graph: Graph,
        weights: HashMap<String, Tensor>,
        registry: OperatorRegistry,
    ) -> Result<Self, OnnxError> {
        Self::build_from_graph(
            graph,
            weights,
            ModelMetadata::default(),
            registry,
            OptLevel::All,
            false,
            false,
            false,
            false,
            None,
            OpPlacement::default(),
        )
    }

    /// Internal: build a session from a graph, applying the given optimization level.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_from_graph(
        graph: Graph,
        weights: HashMap<String, Tensor>,
        metadata: ModelMetadata,
        registry: OperatorRegistry,
        opt_level: OptLevel,
        enable_profiling: bool,
        enable_memory_pool: bool,
        parallel: bool,
        mixed_precision: bool,
        num_threads: Option<usize>,
        op_placement: OpPlacement,
    ) -> Result<Self, OnnxError> {
        use crate::memory::SizeClassPool;
        use std::sync::Mutex;

        let mut weights = weights;
        let input_names = graph.input_names.clone();
        let output_names = graph.output_names.clone();
        let input_infos = graph.input_infos.clone();
        let output_infos = graph.output_infos.clone();

        let optimized_nodes = match opt_level {
            OptLevel::None => graph.nodes,
            OptLevel::Basic | OptLevel::Extended | OptLevel::All => crate::optimizer::optimize(
                graph.nodes,
                &mut weights,
                &graph.output_names,
                &registry,
            ),
        };

        // Build a temporary graph for topological sort
        let opt_graph = Graph {
            nodes: optimized_nodes,
            input_names: input_names.clone(),
            output_names: output_names.clone(),
            ..Default::default()
        };

        let known: Vec<String> = weights
            .keys()
            .cloned()
            .chain(input_names.iter().cloned())
            .collect();
        let order = opt_graph.topological_sort(&known);

        let sorted_nodes: Vec<crate::graph::Node> =
            order.iter().map(|&i| opt_graph.nodes[i].clone()).collect();

        let profiling_data = if enable_profiling {
            Some(Mutex::new(Vec::new()))
        } else {
            Option::None
        };

        // Optionally run shape inference and set up buffer pool
        let (pool, shape_cache) = if enable_memory_pool {
            let input_shapes: HashMap<String, Vec<usize>> = HashMap::new();
            let shapes = crate::optimizer::shape_inference::infer_shapes(
                &sorted_nodes,
                &weights,
                &input_shapes,
            );
            (Some(Mutex::new(SizeClassPool::new())), Some(shapes))
        } else {
            (None, None)
        };

        #[cfg(feature = "cuda")]
        let cuda = oxionnx_cuda::CudaContext::try_new();

        #[cfg(feature = "directml")]
        let dml = oxionnx_directml::DirectMLContext::try_new();

        #[cfg(feature = "gpu")]
        let gpu = crate::gpu::GpuContext::try_new();

        if mixed_precision {
            tracing::info!("Mixed-precision inference enabled (f16 activations, f32 accumulation)");
        }

        #[cfg(not(target_arch = "wasm32"))]
        let thread_pool = num_threads
            .map(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build())
            .transpose()
            .map_err(|e| OnnxError::Internal(format!("thread pool: {e}")))?;

        Ok(Self {
            sorted_nodes,
            weights,
            input_names,
            output_names,
            input_infos,
            output_infos,
            metadata,
            registry,
            profiling_data,
            pool,
            shape_cache,
            parallel,
            mixed_precision,
            op_placement,
            dynamic_dims: Mutex::new(HashMap::new()),
            resolved_shapes: Mutex::new(HashMap::new()),
            #[cfg(not(target_arch = "wasm32"))]
            thread_pool,
            #[cfg(feature = "cuda")]
            cuda,
            #[cfg(feature = "directml")]
            dml,
            #[cfg(feature = "gpu")]
            gpu,
        })
    }

    /// Return a builder for configuring and creating a Session.
    pub fn builder() -> super::SessionBuilder {
        super::SessionBuilder::new()
    }
}
