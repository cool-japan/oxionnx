use crate::execution_providers::{OpPlacement, ProviderKind};
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::OperatorRegistry;
use std::collections::HashMap;
use std::path::Path;

use super::types::{raw_meta_to_model_metadata, ModelMetadata, OptLevel};
use super::Session;

// ── Load-path instrumentation ───────────────────────────────────────────────

/// The `tracing` target every load-path span and event is emitted under.
///
/// A single target so a caller can turn the whole of model loading up to
/// `debug` (`RUST_LOG=oxionnx::session::load=debug`) without also enabling the
/// per-node inference logs, which are far chattier and answer a different
/// question.
pub(crate) const LOAD_TARGET: &str = "oxionnx::session::load";

/// Wrap a model **parse** in a span and a duration event.
///
/// # Why loading is instrumented at all
///
/// "Why did my service take 4 s to start?" is answered by exactly four numbers —
/// parse, optimize, sort, plan — and until now the engine reported none of them.
/// They are also the four stages whose cost is *paid once* and is therefore
/// invisible in every inference benchmark, which is precisely how a slow
/// optimizer pass hides.
///
/// `kind` names the entry point (`"file"`, `"bytes"`, `"mmap"`, `"reader"`,
/// `"filtered"`, `"cache"`) because their costs are not comparable: `mmap` does
/// almost no work here and defers it to the page fault, while `reader` streams.
///
/// The closure is timed whether it succeeds or fails — a parse that fails after
/// 30 s is exactly the case worth seeing.
pub(crate) fn parse_stage<T, E>(
    kind: &'static str,
    byte_len: usize,
    parse: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let span = tracing::debug_span!(target: LOAD_TARGET, "parse", kind, bytes = byte_len);
    let _entered = span.enter();
    let started = std::time::Instant::now();
    let result = parse();
    tracing::debug!(
        target: LOAD_TARGET,
        stage = "parse",
        kind,
        bytes = byte_len,
        elapsed_us = started.elapsed().as_micros() as u64,
        ok = result.is_ok(),
        "model parse complete",
    );
    result
}

impl Session {
    /// Load an ONNX model from a `.onnx` file.
    /// Supports models with external data by resolving paths relative to the file's directory.
    pub fn from_file(path: &Path) -> Result<Self, OnnxError> {
        let bytes = std::fs::read(path).map_err(|e| {
            OnnxError::Parse(format!("Cannot read ONNX file {}: {e}", path.display()))
        })?;
        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let registry = oxionnx_ops::default_registry();
        let (raw_meta, graph, weights) = parse_stage("file", bytes.len(), || {
            crate::model::load_with_metadata_and_path(&bytes, base_path)
        })
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
            Vec::new(),
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
        let (raw_meta, graph, weights) = parse_stage("file", bytes.len(), || {
            crate::model::load_with_metadata_and_path(&bytes, base_path)
        })
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
            Vec::new(),
        )
    }

    /// Load an ONNX model from raw bytes, with a custom operator registry.
    pub fn from_bytes_with_registry(
        bytes: &[u8],
        registry: OperatorRegistry,
    ) -> Result<Self, OnnxError> {
        let (raw_meta, graph, weights) = parse_stage("bytes", bytes.len(), || {
            crate::model::load_with_metadata(bytes)
        })
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
            Vec::new(),
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
            Vec::new(),
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
        providers: Vec<ProviderKind>,
    ) -> Result<Self, OnnxError> {
        use crate::memory::SizeClassPool;
        use std::sync::Mutex;

        // One span for the whole build; the four stage events below hang off it,
        // so a caller sees "this model took 4.1 s to load, of which 3.7 s was
        // constant folding" rather than a single opaque number.  See
        // [`parse_stage`] for why the load path is instrumented.
        let build_span = tracing::info_span!(
            target: LOAD_TARGET,
            "build",
            nodes = graph.nodes.len(),
            weights = weights.len(),
            opt_level = ?opt_level,
        );
        let _entered = build_span.enter();
        let build_started = std::time::Instant::now();

        let mut weights = weights;
        let input_names = graph.input_names.clone();
        let output_names = graph.output_names.clone();
        let input_infos = graph.input_infos.clone();
        let output_infos = graph.output_infos.clone();

        // Shape inference is seeded with whatever the model states about its
        // own inputs, so the rank-gated fusions can actually fire.
        let declared_input_shapes = static_input_shapes(&input_infos);

        // Each level runs exactly the pass set its documentation promises:
        // `Basic` = dead-node elimination, `Extended` = dead-node elimination
        // plus the fusions, `All` = the full pipeline (shape materialisation,
        // constant folding, CSE and fusions).
        let nodes_before_optimize = graph.nodes.len();
        let optimize_started = std::time::Instant::now();
        let optimized_nodes = match opt_level {
            OptLevel::None => graph.nodes,
            OptLevel::Basic | OptLevel::Extended | OptLevel::All => {
                let pass_level = match opt_level {
                    OptLevel::Basic => crate::optimizer::PassLevel::Basic,
                    OptLevel::Extended => crate::optimizer::PassLevel::Extended,
                    _ => crate::optimizer::PassLevel::All,
                };
                let _span =
                    tracing::debug_span!(target: LOAD_TARGET, "optimize", level = ?pass_level)
                        .entered();
                crate::optimizer::optimize_with_input_shapes(
                    graph.nodes,
                    &mut weights,
                    &graph.output_names,
                    &registry,
                    pass_level,
                    &declared_input_shapes,
                )
            }
        };
        tracing::debug!(
            target: LOAD_TARGET,
            stage = "optimize",
            nodes_before = nodes_before_optimize,
            nodes_after = optimized_nodes.len(),
            elapsed_us = optimize_started.elapsed().as_micros() as u64,
            "graph optimization complete",
        );

        // Build a temporary graph for topological sort
        let opt_graph = Graph {
            nodes: optimized_nodes,
            input_names: input_names.clone(),
            output_names: output_names.clone(),
            ..Default::default()
        };

        let sort_started = std::time::Instant::now();
        let known: Vec<String> = weights
            .keys()
            .cloned()
            .chain(input_names.iter().cloned())
            .collect();
        let order = opt_graph.topological_sort(&known);

        let sorted_nodes: Vec<crate::graph::Node> =
            order.iter().map(|&i| opt_graph.nodes[i].clone()).collect();
        tracing::debug!(
            target: LOAD_TARGET,
            stage = "sort",
            nodes = sorted_nodes.len(),
            elapsed_us = sort_started.elapsed().as_micros() as u64,
            "topological sort complete",
        );

        let profiling_data = if enable_profiling {
            Some(Mutex::new(Vec::new()))
        } else {
            Option::None
        };

        // Optionally run shape inference and set up buffer pool.  The same
        // static-input seed the optimizer used: without it this pass could not
        // size a single activation buffer, so the pool's size classes were
        // derived from weights alone.
        let plan_started = std::time::Instant::now();
        let (pool, shape_cache) = if enable_memory_pool {
            let _span = tracing::debug_span!(target: LOAD_TARGET, "shape_inference").entered();
            let shapes = crate::optimizer::shape_inference::infer_shapes(
                &sorted_nodes,
                &weights,
                &declared_input_shapes,
            );
            (Some(Mutex::new(SizeClassPool::new())), Some(shapes))
        } else {
            (None, None)
        };

        // The static per-run plan: the reference-count seed and, natively, the
        // depth-grouped cost-sorted schedule.  Built here — after optimization
        // and the topological sort — because both depend on the *final* node
        // list; constant folding removes consumers, and a seed built from the
        // unoptimized graph would over-count them.  See [`StaticRunPlan`].
        let run_plan = super::run::plan::StaticRunPlan::build(
            &sorted_nodes,
            &weights,
            &output_names,
            shape_cache.as_ref(),
        );
        tracing::debug!(
            target: LOAD_TARGET,
            stage = "plan",
            counted_tensors = run_plan.base_ref_counts.len(),
            inferred_shapes = shape_cache.as_ref().map_or(0, HashMap::len),
            elapsed_us = plan_started.elapsed().as_micros() as u64,
            "execution plan complete",
        );

        // Seed the shape-plan cache with the build-time inference when the model
        // declares every one of its inputs statically.  `infer_shapes` is a pure
        // function of (nodes, weights, input shapes), and the shapes above were
        // inferred from exactly `declared_input_shapes` — so for a caller that
        // feeds the declared shapes (the overwhelmingly common case for a fully
        // static model) this is *the* answer `resolve_run_shapes` would compute,
        // and the first inference no longer pays a full inference pass.
        //
        // Guarded on covering every declared input: the plan cache is keyed on
        // the whole input-shape map compared for equality, so a partial seed
        // could never be hit anyway, and storing one would only evict a real
        // entry.
        let shape_plans = super::ShapePlanCache::default();
        if let Some(ref shapes) = shape_cache {
            if declared_input_shapes.len() == input_names.len() && !input_names.is_empty() {
                shape_plans.store(&declared_input_shapes, shapes);
            }
        }

        #[cfg(feature = "cuda")]
        let cuda = oxionnx_cuda::CudaContext::try_new();

        #[cfg(feature = "directml")]
        let dml = oxionnx_directml::DirectMLContext::try_new();

        // Routes through `gpu_owner`'s dedicated thread rather than calling
        // `crate::gpu::GpuContext::try_new()` inline here on the caller's
        // own thread — see that module's docs for the cross-thread
        // create/destroy driver crash this closes.
        #[cfg(feature = "gpu")]
        let gpu = super::gpu_owner::try_new();

        if mixed_precision {
            tracing::info!("Mixed-precision inference enabled (f16 activations, f32 accumulation)");
        }

        #[cfg(not(target_arch = "wasm32"))]
        let thread_pool = num_threads
            .map(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build())
            .transpose()
            .map_err(|e| OnnxError::Internal(format!("thread pool: {e}")))?;

        // `num_threads` configures the rayon thread pool, which is native-only.
        // On wasm32 there is no thread pool, so the argument is intentionally unused.
        #[cfg(target_arch = "wasm32")]
        let _ = num_threads;

        tracing::debug!(
            target: LOAD_TARGET,
            stage = "build",
            nodes = sorted_nodes.len(),
            elapsed_us = build_started.elapsed().as_micros() as u64,
            "session build complete",
        );

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
            run_plan,
            parallel,
            mixed_precision,
            op_placement,
            providers,
            dynamic_dims: Mutex::new(HashMap::new()),
            resolved_shapes: Mutex::new(HashMap::new()),
            shape_plans,
            cancellation: None,
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

/// The declared shapes of the graph inputs that are **fully static**.
///
/// Load-time shape inference used to start from an empty map, so it never
/// learned the rank of a single activation: every shape it knew originated in a
/// constant weight.  That is why the rank-2 gates added to `MatMul + Add → Gemm`
/// and `Add + MatMul → Gemm` (needed for soundness — a rank-3 MatMul is not a
/// Gemm) took most of those fusions out of service on real models, and why the
/// memory pool's `shape_cache` could not size an activation buffer.
///
/// # Why an input with any symbolic dimension is skipped entirely
///
/// The inferred shapes do not only gate fusions — `fold_batch_norm_inference`
/// *sizes synthesised constants* from them, and `simplify_transpose_reshape` /
/// `cancel_consecutive_reshape` decide from them.  Substituting a placeholder
/// (say 1) for a `batch` axis to recover the rank would feed those passes a
/// fabricated dimension and turn a coverage gap into a correctness bug.  A
/// dimension is used only when the model states it outright.
fn static_input_shapes(input_infos: &[oxionnx_core::TensorInfo]) -> HashMap<String, Vec<usize>> {
    let mut out = HashMap::new();
    for info in input_infos {
        if info.name.is_empty() || info.shape.is_empty() {
            continue;
        }
        let mut dims = Vec::with_capacity(info.shape.len());
        for dim in &info.shape {
            match dim {
                Some(d) => dims.push(*d),
                // One symbolic/unknown axis disqualifies the whole input.
                None => {
                    dims.clear();
                    break;
                }
            }
        }
        if dims.len() == info.shape.len() {
            out.insert(info.name.clone(), dims);
        }
    }
    out
}
