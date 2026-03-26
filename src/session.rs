use crate::graph::{Graph, Node, OpKind};
use crate::memory::BufferPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{OpContext, Operator, OperatorRegistry};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use rayon::prelude::*;

/// Optimization level for graph optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations.
    None,
    /// Basic: dead node elimination only.
    Basic,
    /// Extended: dead node elimination + operator fusions.
    Extended,
    /// All: constant folding + dead node elimination + fusions.
    All,
}

/// Profiling information for a single executed node.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    /// Name of the node in the graph.
    pub node_name: String,
    /// The ONNX op type (e.g. "MatMul", "Relu").
    pub op_type: String,
    /// Wall-clock execution duration.
    pub duration: std::time::Duration,
    /// Shapes of each output tensor produced by this node.
    pub output_shapes: Vec<Vec<usize>>,
}

/// Summary information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Number of computation nodes in the (optimized) graph.
    pub node_count: usize,
    /// Total number of scalar parameters stored as weights.
    pub parameter_count: usize,
    /// Estimated weight memory in bytes (assuming f32).
    pub weight_bytes: usize,
    /// Histogram of operator types: op_name -> count.
    pub op_histogram: HashMap<String, usize>,
}

/// A loaded ONNX model ready for inference.
pub struct Session {
    sorted_nodes: Vec<Node>,
    weights: HashMap<String, Tensor>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    registry: OperatorRegistry,
    profiling_data: Option<Mutex<Vec<NodeProfile>>>,
    pool: Option<Mutex<BufferPool>>,
    shape_cache: Option<HashMap<String, Vec<usize>>>,
    /// Whether to use rayon-based parallel execution for independent nodes.
    #[allow(dead_code)]
    parallel: bool,
    /// Whether to use mixed-precision inference (f16 activations, f32 accumulation).
    #[allow(dead_code)]
    mixed_precision: bool,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::GpuContext>,
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
        let (graph, weights) =
            crate::model::load_with_path(&bytes, base_path).map_err(OnnxError::Parse)?;
        Self::build_from_graph(graph, weights, registry, OptLevel::All, false, false, false)
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
        let (graph, weights) =
            crate::model::load_with_path(&bytes, base_path).map_err(OnnxError::Parse)?;
        Self::build_from_graph(graph, weights, registry, OptLevel::All, false, false, false)
    }

    /// Load an ONNX model from raw bytes, with a custom operator registry.
    pub fn from_bytes_with_registry(
        bytes: &[u8],
        registry: OperatorRegistry,
    ) -> Result<Self, OnnxError> {
        let (graph, weights) = crate::model::load(bytes).map_err(OnnxError::Parse)?;
        Self::build_from_graph(graph, weights, registry, OptLevel::All, false, false, false)
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
        Self::build_from_graph(graph, weights, registry, OptLevel::All, false, false, false)
    }

    /// Internal: build a session from a graph, applying the given optimization level.
    fn build_from_graph(
        graph: Graph,
        weights: HashMap<String, Tensor>,
        registry: OperatorRegistry,
        opt_level: OptLevel,
        enable_profiling: bool,
        enable_memory_pool: bool,
        parallel: bool,
    ) -> Result<Self, OnnxError> {
        let mut weights = weights;
        let input_names = graph.input_names.clone();
        let output_names = graph.output_names.clone();

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
        };

        let known: Vec<String> = weights
            .keys()
            .cloned()
            .chain(input_names.iter().cloned())
            .collect();
        let order = opt_graph.topological_sort(&known);

        let sorted_nodes: Vec<Node> = order.iter().map(|&i| opt_graph.nodes[i].clone()).collect();

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
            (Some(Mutex::new(BufferPool::new())), Some(shapes))
        } else {
            (None, None)
        };

        #[cfg(feature = "gpu")]
        let gpu = crate::gpu::GpuContext::try_new();

        Ok(Self {
            sorted_nodes,
            weights,
            input_names,
            output_names,
            registry,
            profiling_data,
            pool,
            shape_cache,
            parallel,
            mixed_precision: false,
            #[cfg(feature = "gpu")]
            gpu,
        })
    }

    /// Return a builder for configuring and creating a Session.
    pub fn builder() -> SessionBuilder {
        SessionBuilder::new()
    }

    /// Register an additional (or replacement) operator at runtime.
    pub fn register_op(&mut self, op: Box<dyn Operator>) {
        self.registry.register(op);
    }

    /// Return the names of the model's graph inputs (excluding initializers/weights).
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Return the names of the model's graph outputs.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Return a reference to the model's weight tensors.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }

    /// Compute the topological depth for each node in `sorted_nodes`.
    /// Depth 0 = all inputs come from model inputs / weights (no graph predecessors).
    /// For others, depth = max(depth of predecessor nodes) + 1.
    fn compute_node_depths(sorted_nodes: &[Node], weights: &HashMap<String, Tensor>) -> Vec<usize> {
        let mut tensor_depth: HashMap<&str, usize> = HashMap::new();
        let mut depths = Vec::with_capacity(sorted_nodes.len());

        for node in sorted_nodes {
            let mut max_pred_depth: Option<usize> = None;
            for inp in &node.inputs {
                if inp.is_empty() || weights.contains_key(inp) {
                    continue;
                }
                if let Some(&d) = tensor_depth.get(inp.as_str()) {
                    max_pred_depth = Some(match max_pred_depth {
                        Some(cur) => cur.max(d),
                        None => d,
                    });
                }
            }
            let depth = match max_pred_depth {
                Some(d) => d + 1,
                None => 0,
            };
            depths.push(depth);
            for out in &node.outputs {
                if !out.is_empty() {
                    tensor_depth.insert(out.as_str(), depth);
                }
            }
        }
        depths
    }

    /// Group node indices by their topological depth.
    fn group_by_depth(depths: &[usize]) -> Vec<Vec<usize>> {
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        let mut groups = vec![Vec::new(); max_depth + 1];
        for (i, &d) in depths.iter().enumerate() {
            groups[d].push(i);
        }
        groups
    }

    /// Execute a single node, with optional in-place optimization.
    fn execute_node_with_inplace(
        node: &Node,
        operator: &dyn Operator,
        intermediates: &mut HashMap<String, Tensor>,
        weights: &HashMap<String, Tensor>,
        ref_counts: &HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) -> Result<(Vec<Tensor>, std::time::Duration), OnnxError> {
        // Check if in-place execution is possible for the first input
        let can_inplace = operator.supports_inplace()
            && !node.inputs.is_empty()
            && !node.inputs[0].is_empty()
            && !weights.contains_key(&node.inputs[0])
            && !output_set.contains(node.inputs[0].as_str())
            && ref_counts.get(&node.inputs[0]).copied().unwrap_or(0) == 1;

        let start = std::time::Instant::now();

        let results = if can_inplace {
            let owned_input = intermediates.remove(&node.inputs[0]);
            let resolved_inputs: Vec<Option<&Tensor>> = node
                .inputs
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    if name.is_empty() || i == 0 {
                        None
                    } else {
                        intermediates.get(name).or_else(|| weights.get(name))
                    }
                })
                .collect();
            let ctx = OpContext {
                node,
                inputs: resolved_inputs,
                outer_scope: None,
                registry: None,
            };
            match owned_input {
                Some(tensor) => operator.execute_inplace(tensor, &ctx)?,
                None => operator.execute(&ctx)?,
            }
        } else {
            let resolved_inputs: Vec<Option<&Tensor>> = node
                .inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        intermediates.get(name).or_else(|| weights.get(name))
                    }
                })
                .collect();
            let ctx = OpContext {
                node,
                inputs: resolved_inputs,
                outer_scope: None,
                registry: None,
            };
            operator.execute(&ctx)?
        };

        let elapsed = start.elapsed();
        Ok((results, elapsed))
    }

    /// Decrement reference counts for a node's inputs and free tensors that are
    /// no longer needed, optionally returning buffers to the memory pool.
    fn decrement_refs(
        &self,
        node: &Node,
        intermediates: &mut HashMap<String, Tensor>,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) {
        for inp in &node.inputs {
            if inp.is_empty() || self.weights.contains_key(inp) {
                continue;
            }
            if let Some(count) = ref_counts.get_mut(inp) {
                *count = count.saturating_sub(1);
                if *count == 0 && !output_set.contains(inp.as_str()) {
                    if let Some(mut tensor) = intermediates.remove(inp) {
                        if let Some(ref pool_mutex) = self.pool {
                            if let Ok(mut pool) = pool_mutex.lock() {
                                let buf = std::mem::take(&mut tensor.data);
                                pool.return_buffer(buf);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run inference with the given named inputs.
    /// Returns all graph output tensors by name.
    ///
    /// Weights are borrowed (not cloned) to avoid copying hundreds of MB
    /// of model parameters on every inference call.
    ///
    /// When parallel execution is enabled, independent nodes at the same
    /// topological depth are executed concurrently via rayon.
    pub fn run(
        &self,
        inputs: &HashMap<&str, Tensor>,
    ) -> Result<HashMap<String, Tensor>, OnnxError> {
        let output_set: std::collections::HashSet<&str> =
            self.output_names.iter().map(|s| s.as_str()).collect();
        let mut ref_counts: HashMap<String, usize> = HashMap::new();
        for node in &self.sorted_nodes {
            for inp in &node.inputs {
                if !inp.is_empty() && !self.weights.contains_key(inp) {
                    *ref_counts.entry(inp.clone()).or_insert(0) += 1;
                }
            }
        }
        for name in &self.output_names {
            *ref_counts.entry(name.clone()).or_insert(0) += 1;
        }

        let mut intermediates: HashMap<String, Tensor> =
            HashMap::with_capacity(self.sorted_nodes.len());
        for (&name, tensor) in inputs {
            intermediates.insert(name.to_string(), tensor.clone());
        }

        let use_parallel = self.parallel && cfg!(not(target_arch = "wasm32"));

        if use_parallel {
            self.run_parallel_inner(&mut intermediates, &mut ref_counts, &output_set)?;
        } else {
            self.run_sequential_inner(&mut intermediates, &mut ref_counts, &output_set)?;
        }

        let mut outputs = HashMap::new();
        for name in &self.output_names {
            if let Some(t) = intermediates.remove(name) {
                outputs.insert(name.clone(), t);
            }
        }
        Ok(outputs)
    }

    /// Sequential execution path with in-place optimization.
    fn run_sequential_inner(
        &self,
        intermediates: &mut HashMap<String, Tensor>,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) -> Result<(), OnnxError> {
        for node in &self.sorted_nodes {
            if let OpKind::Unknown(_) = &node.op {
                continue;
            }

            #[cfg(feature = "gpu")]
            {
                if let Some(gpu_ctx) = &self.gpu {
                    if let Some(results) =
                        try_gpu_dispatch(node, &self.weights, intermediates, gpu_ctx)?
                    {
                        for (name, tensor) in node.outputs.iter().zip(results) {
                            if !name.is_empty() {
                                intermediates.insert(name.clone(), tensor);
                            }
                        }
                        self.decrement_refs(node, intermediates, ref_counts, output_set);
                        continue;
                    }
                    // GPU dispatch returned None — falling back to CPU for this op
                    if GpuExecutionProvider::is_supported(node.op.as_str()) {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[oxionnx] GPU fallback: op '{}' (node '{}') fell back to CPU",
                            node.op.as_str(),
                            node.name,
                        );
                    }
                }
            }

            let op_name = node.op.as_str();
            let operator = self.registry.get(op_name).ok_or_else(|| {
                OnnxError::UnknownOp(format!("No operator registered for '{}'", op_name))
            })?;

            let (results, elapsed) = Self::execute_node_with_inplace(
                node,
                operator,
                intermediates,
                &self.weights,
                ref_counts,
                output_set,
            )?;

            if let Some(ref profiling) = self.profiling_data {
                if let Ok(mut data) = profiling.lock() {
                    data.push(NodeProfile {
                        node_name: node.name.clone(),
                        op_type: node.op.as_str().to_string(),
                        duration: elapsed,
                        output_shapes: results.iter().map(|t| t.shape.clone()).collect(),
                    });
                }
            }

            for (name, tensor) in node.outputs.iter().zip(results) {
                if !name.is_empty() {
                    intermediates.insert(name.clone(), tensor);
                }
            }

            self.decrement_refs(node, intermediates, ref_counts, output_set);
        }
        Ok(())
    }

    /// Parallel execution: group nodes by topological depth and execute each
    /// depth level concurrently using rayon.
    #[cfg(not(target_arch = "wasm32"))]
    fn run_parallel_inner(
        &self,
        intermediates: &mut HashMap<String, Tensor>,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) -> Result<(), OnnxError> {
        let depths = Self::compute_node_depths(&self.sorted_nodes, &self.weights);
        let groups = Self::group_by_depth(&depths);

        for group in &groups {
            if group.is_empty() {
                continue;
            }

            if group.len() == 1 {
                // Single node — execute sequentially (no rayon overhead)
                let node = &self.sorted_nodes[group[0]];
                if let OpKind::Unknown(_) = &node.op {
                    continue;
                }
                let op_name = node.op.as_str();
                let operator = self.registry.get(op_name).ok_or_else(|| {
                    OnnxError::UnknownOp(format!("No operator registered for '{}'", op_name))
                })?;

                let (results, elapsed) = Self::execute_node_with_inplace(
                    node,
                    operator,
                    intermediates,
                    &self.weights,
                    ref_counts,
                    output_set,
                )?;

                if let Some(ref profiling) = self.profiling_data {
                    if let Ok(mut data) = profiling.lock() {
                        data.push(NodeProfile {
                            node_name: node.name.clone(),
                            op_type: node.op.as_str().to_string(),
                            duration: elapsed,
                            output_shapes: results.iter().map(|t| t.shape.clone()).collect(),
                        });
                    }
                }

                for (name, tensor) in node.outputs.iter().zip(results) {
                    if !name.is_empty() {
                        intermediates.insert(name.clone(), tensor);
                    }
                }
                self.decrement_refs(node, intermediates, ref_counts, output_set);
            } else {
                // Multiple nodes at this depth — execute in parallel via rayon
                let nodes_at_depth: Vec<&Node> =
                    group.iter().map(|&i| &self.sorted_nodes[i]).collect();

                // Collect operators and pre-resolve inputs
                let work_items: Vec<(&Node, &dyn Operator, Vec<Option<&Tensor>>)> = nodes_at_depth
                    .iter()
                    .filter(|n| !matches!(n.op, OpKind::Unknown(_)))
                    .map(|n| {
                        let op = self.registry.get(n.op.as_str()).ok_or_else(|| {
                            OnnxError::UnknownOp(format!(
                                "No operator registered for '{}'",
                                n.op.as_str()
                            ))
                        });
                        let inputs: Vec<Option<&Tensor>> = n
                            .inputs
                            .iter()
                            .map(|name| {
                                if name.is_empty() {
                                    None
                                } else {
                                    intermediates.get(name).or_else(|| self.weights.get(name))
                                }
                            })
                            .collect();
                        op.map(|o| (*n, o, inputs))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Execute in parallel — each produces (node_name, results, duration)
                type ParResult<'a> = Result<(&'a str, Vec<Tensor>, std::time::Duration), OnnxError>;
                let par_results: Vec<ParResult<'_>> = work_items
                    .par_iter()
                    .map(|(node, operator, inputs)| {
                        let ctx = OpContext {
                            node,
                            inputs: inputs.clone(),
                            outer_scope: None,
                            registry: None,
                        };
                        let start = std::time::Instant::now();
                        let res = operator.execute(&ctx)?;
                        let elapsed = start.elapsed();
                        Ok((node.name.as_str(), res, elapsed))
                    })
                    .collect();

                // Insert all outputs sequentially
                for result in par_results {
                    let (node_name, tensors, elapsed) = result?;
                    if let Some(node) = nodes_at_depth.iter().find(|n| n.name == node_name) {
                        if let Some(ref profiling) = self.profiling_data {
                            if let Ok(mut data) = profiling.lock() {
                                data.push(NodeProfile {
                                    node_name: node.name.clone(),
                                    op_type: node.op.as_str().to_string(),
                                    duration: elapsed,
                                    output_shapes: tensors
                                        .iter()
                                        .map(|t| t.shape.clone())
                                        .collect(),
                                });
                            }
                        }
                        for (name, tensor) in node.outputs.iter().zip(tensors) {
                            if !name.is_empty() {
                                intermediates.insert(name.clone(), tensor);
                            }
                        }
                    }
                }

                // Decrement ref counts for all nodes in this group
                for node in &nodes_at_depth {
                    self.decrement_refs(node, intermediates, ref_counts, output_set);
                }
            }
        }
        Ok(())
    }

    /// Fallback on wasm32: parallel is not supported, delegate to sequential.
    #[cfg(target_arch = "wasm32")]
    fn run_parallel_inner(
        &self,
        intermediates: &mut HashMap<String, Tensor>,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) -> Result<(), OnnxError> {
        self.run_sequential_inner(intermediates, ref_counts, output_set)
    }

    /// Convenience wrapper: run with a single input.
    pub fn run_one(&self, name: &str, input: Tensor) -> Result<HashMap<String, Tensor>, OnnxError> {
        let mut inputs = HashMap::new();
        inputs.insert(name, input);
        self.run(&inputs)
    }

    /// Text correction helper: tokenize -> run -> detokenize.
    /// Character-level tokenization (Unicode codepoint IDs).
    pub fn correct_text(&self, text: &str) -> Result<String, OnnxError> {
        let ids: Vec<f32> = text.chars().map(|c| c as u32 as f32).collect();
        let n = ids.len();
        let input = Tensor::new(ids, vec![1, n]);

        let outputs = self.run_one("input_ids", input)?;

        if let Some(out) = outputs.values().next() {
            let chars: String = out
                .data
                .iter()
                .filter_map(|&v| char::from_u32(v as u32).filter(|&c| c != '\0'))
                .collect();
            Ok(chars)
        } else {
            Ok(text.to_string())
        }
    }

    /// Retrieve profiling results collected during `run()` calls.
    /// Returns `None` if profiling was not enabled.
    pub fn profiling_results(&self) -> Option<Vec<NodeProfile>> {
        self.profiling_data
            .as_ref()
            .and_then(|m| m.lock().ok().map(|d| d.clone()))
    }

    /// Return summary information about the loaded model.
    pub fn model_info(&self) -> ModelInfo {
        let parameter_count: usize = self.weights.values().map(|t| t.numel()).sum();
        let mut op_histogram = HashMap::new();
        for node in &self.sorted_nodes {
            *op_histogram
                .entry(node.op.as_str().to_string())
                .or_insert(0) += 1;
        }
        ModelInfo {
            node_count: self.sorted_nodes.len(),
            parameter_count,
            weight_bytes: parameter_count * 4, // f32
            op_histogram,
        }
    }

    /// Returns estimated peak memory usage in bytes for intermediate tensors.
    ///
    /// Uses the cached shape map (from shape inference at build time) to compute
    /// a memory plan. Returns `None` if the memory pool was not enabled or if
    /// shape inference could not determine any tensor shapes.
    pub fn estimated_memory_bytes(&self) -> Option<usize> {
        let shape_map = self.shape_cache.as_ref()?;
        let plan =
            crate::memory::MemoryPlan::compute(&self.sorted_nodes, &self.output_names, shape_map);
        if plan.peak_memory_elements == 0 {
            return None;
        }
        Some(plan.peak_memory_elements * 4) // sizeof f32
    }

    /// Export the computation graph as a DOT (Graphviz) string.
    pub fn export_dot(&self) -> String {
        let mut dot = String::from("digraph model {\n  rankdir=TB;\n  node [shape=box];\n");

        // Weight nodes (ellipse)
        for name in self.weights.keys() {
            dot.push_str(&format!(
                "  \"{}\" [shape=ellipse, style=filled, fillcolor=lightblue];\n",
                name
            ));
        }

        // Op nodes
        for node in &self.sorted_nodes {
            let label = format!("{}\\n({})", node.name, node.op.as_str());
            dot.push_str(&format!("  \"{}\" [label=\"{}\"];\n", node.name, label));

            // Edges from inputs to this node
            for inp in &node.inputs {
                if !inp.is_empty() {
                    dot.push_str(&format!("  \"{}\" -> \"{}\";\n", inp, node.name));
                }
            }
            // Edges from this node to outputs
            for out in &node.outputs {
                if !out.is_empty() {
                    dot.push_str(&format!("  \"{}\" -> \"{}\";\n", node.name, out));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }
}

/// Builder for configuring and creating a Session.
pub struct SessionBuilder {
    opt_level: OptLevel,
    registry: Option<OperatorRegistry>,
    enable_profiling: bool,
    enable_memory_pool: bool,
    parallel: bool,
    mixed_precision: bool,
}

impl SessionBuilder {
    /// Create a new builder with default settings (all optimizations, no profiling, no pool,
    /// sequential execution).
    pub fn new() -> Self {
        Self {
            opt_level: OptLevel::All,
            registry: None,
            enable_profiling: false,
            enable_memory_pool: false,
            parallel: false,
            mixed_precision: false,
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

    /// Load an ONNX model from a file path.
    /// Supports models with external data by resolving paths relative to the file's directory.
    pub fn load(self, path: &Path) -> Result<Session, OnnxError> {
        let bytes = std::fs::read(path).map_err(|e| {
            OnnxError::Parse(format!("Cannot read ONNX file {}: {e}", path.display()))
        })?;
        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (graph, weights) =
            crate::model::load_with_path(&bytes, base_path).map_err(OnnxError::Parse)?;
        Session::build_from_graph(
            graph,
            weights,
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
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
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
        )
    }

    /// Load an ONNX model from raw bytes.
    pub fn load_from_bytes(self, bytes: &[u8]) -> Result<Session, OnnxError> {
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (graph, weights) = crate::model::load(bytes).map_err(OnnxError::Parse)?;
        Session::build_from_graph(
            graph,
            weights,
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
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
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
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
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
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
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
        )
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Provides metadata about GPU-accelerated operator support.
#[cfg(feature = "gpu")]
pub struct GpuExecutionProvider;

#[cfg(feature = "gpu")]
impl GpuExecutionProvider {
    /// Return the list of operator types that have GPU acceleration.
    pub fn supported_ops() -> &'static [&'static str] {
        &[
            "MatMul",
            "Conv",
            "Softmax",
            "Relu",
            "Sigmoid",
            "Gelu",
            "ReduceSum",
            "ReduceMax",
        ]
    }

    /// Check whether a given operator type is GPU-accelerated.
    pub fn is_supported(op_type: &str) -> bool {
        Self::supported_ops().contains(&op_type)
    }
}

/// GPU dispatch for ops with hardware acceleration (MatMul, Conv).
/// Returns `Ok(Some(results))` if GPU handled it, `Ok(None)` to fall back to CPU.
#[cfg(feature = "gpu")]
fn try_gpu_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    gpu: &crate::gpu::GpuContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    let resolve = |name: &str| -> Option<&Tensor> {
        if name.is_empty() {
            None
        } else {
            intermediates.get(name).or_else(|| weights.get(name))
        }
    };

    match &node.op {
        OpKind::MatMul => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                let an = a.ndim();
                let bn = b.ndim();
                if an >= 2 && bn >= 2 {
                    let m = a.shape[an - 2];
                    let k = a.shape[an - 1];
                    let n = b.shape[bn - 1];
                    let batch_size: usize = a.shape[..an - 2].iter().product::<usize>().max(1);
                    if batch_size == 1 {
                        if let Some(result) = crate::gpu::gpu_matmul(gpu, &a.data, &b.data, m, k, n)
                        {
                            return Ok(Some(vec![Tensor::new(result, vec![m, n])]));
                        }
                    }
                }
            }
            Ok(None)
        }
        OpKind::Conv => {
            let input = resolve(&node.inputs[0]);
            let weight = resolve(&node.inputs[1]);
            let bias = node.inputs.get(2).and_then(|n| resolve(n));
            if let (Some(input), Some(weight)) = (input, weight) {
                let attrs = &node.attrs;
                let strides_v = attrs.ints("strides");
                let strides = [
                    strides_v.first().copied().unwrap_or(1) as usize,
                    strides_v.get(1).copied().unwrap_or(1) as usize,
                ];
                let pads_v = attrs.ints("pads");
                let pads = [
                    pads_v.first().copied().unwrap_or(0) as usize,
                    pads_v.get(1).copied().unwrap_or(0) as usize,
                    pads_v.get(2).copied().unwrap_or(0) as usize,
                    pads_v.get(3).copied().unwrap_or(0) as usize,
                ];
                let dilations_v = attrs.ints("dilations");
                let dilations = [
                    dilations_v.first().copied().unwrap_or(1) as usize,
                    dilations_v.get(1).copied().unwrap_or(1) as usize,
                ];
                let group = attrs.i("group", 1) as usize;
                if let Some(result) = crate::gpu::gpu_conv2d(
                    gpu, input, weight, bias, strides, pads, dilations, group,
                ) {
                    return Ok(Some(vec![result]));
                }
            }
            Ok(None)
        }
        OpKind::Softmax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_softmax(gpu, &input.data, &input.shape) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Relu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_relu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Sigmoid => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_sigmoid(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::Gelu => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                if let Some(result) = crate::gpu::gpu_gelu(gpu, &input.data) {
                    return Ok(Some(vec![Tensor::new(result, input.shape.clone())]));
                }
            }
            Ok(None)
        }
        OpKind::ReduceSum => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes = node.attrs.ints("axes");
                if axes.len() == 1 {
                    let axis = axes[0] as usize;
                    if let Some(result) =
                        crate::gpu::gpu_reduce_sum(gpu, &input.data, axis, &input.shape)
                    {
                        let mut out_shape = input.shape.clone();
                        if axis < out_shape.len() {
                            out_shape[axis] = 1;
                        }
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        OpKind::ReduceMax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes = node.attrs.ints("axes");
                if axes.len() == 1 {
                    let axis = axes[0] as usize;
                    if let Some(result) =
                        crate::gpu::gpu_reduce_max(gpu, &input.data, axis, &input.shape)
                    {
                        let mut out_shape = input.shape.clone();
                        if axis < out_shape.len() {
                            out_shape[axis] = 1;
                        }
                        return Ok(Some(vec![Tensor::new(result, out_shape)]));
                    }
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// Build equal-sized split chunks for a given axis length and count.
#[cfg(test)]
fn equal_split(axis_len: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let chunk = axis_len.div_ceil(n);
    (0..n)
        .map(|i| {
            let start = i * chunk;
            (start + chunk).min(axis_len).saturating_sub(start)
        })
        .filter(|&s| s > 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Attributes;

    #[test]
    fn test_session_from_empty_bytes() {
        let session = Session::from_bytes(&[]).expect("should load empty model");
        let inputs = HashMap::new();
        let outputs = session.run(&inputs).expect("should run empty model");
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_equal_split_helper() {
        assert_eq!(equal_split(6, 3), vec![2, 2, 2]);
        assert_eq!(equal_split(7, 3), vec![3, 3, 1]);
        assert_eq!(equal_split(4, 1), vec![4]);
    }

    #[test]
    fn test_from_graph_identity() {
        // Build a simple Identity graph: input -> Identity -> output
        let node = Node {
            op: OpKind::Identity,
            name: "id_node".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
        };
        let weights = HashMap::new();

        let session = Session::from_graph(graph, weights).expect("from_graph should succeed");
        assert_eq!(session.input_names(), &["input".to_string()]);
        assert_eq!(session.output_names(), &["output".to_string()]);

        let input_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let outputs = session
            .run_one("input", input_tensor.clone())
            .expect("run should succeed");
        let out = outputs.get("output").expect("output should exist");
        assert_eq!(out.data, input_tensor.data);
        assert_eq!(out.shape, input_tensor.shape);
    }

    #[test]
    fn test_builder_load_from_empty_bytes() {
        let result = Session::builder()
            .with_optimization_level(OptLevel::None)
            .load_from_bytes(&[]);
        assert!(result.is_ok());
        let session = result.expect("builder should load empty model");
        assert!(session.input_names().is_empty());
    }

    #[test]
    fn test_model_info() {
        let node1 = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["r1".to_string()],
            attrs: Attributes::default(),
        };
        let node2 = Node {
            op: OpKind::Relu,
            name: "relu2".to_string(),
            inputs: vec!["r1".to_string()],
            outputs: vec!["out".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node1, node2],
            input_names: vec!["x".to_string()],
            output_names: vec!["out".to_string()],
        };
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0; 12], vec![3, 4]));

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build_from_graph");
        let info = session.model_info();
        assert_eq!(info.node_count, 2);
        assert_eq!(info.parameter_count, 12);
        assert_eq!(info.weight_bytes, 48); // 12 * 4
        assert_eq!(info.op_histogram.get("Relu").copied().unwrap_or(0), 2);
    }

    #[test]
    fn test_export_dot() {
        let node = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["out".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["out".to_string()],
        };
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0], vec![1]));

        let session = Session::from_graph(graph, weights).expect("from_graph");
        let dot = session.export_dot();

        assert!(dot.starts_with("digraph model {"));
        assert!(dot.ends_with("}\n"));
        assert!(dot.contains("relu1"));
        assert!(dot.contains("Relu"));
        // Weight node should appear as ellipse
        assert!(dot.contains("\"w\""));
        assert!(dot.contains("ellipse"));
        // Edges
        assert!(dot.contains("\"x\" -> \"relu1\""));
        assert!(dot.contains("\"relu1\" -> \"out\""));
    }

    #[test]
    fn test_profiling() {
        let node = Node {
            op: OpKind::Identity,
            name: "id_node".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
        };
        let weights = HashMap::new();

        let session = Session::builder()
            .with_profiling()
            .build_from_graph(graph, weights)
            .expect("build should succeed");

        // Before running, profiling data should be empty
        let initial = session.profiling_results().expect("profiling enabled");
        assert!(initial.is_empty());

        let input_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let _outputs = session
            .run_one("input", input_tensor)
            .expect("run should succeed");

        let profiles = session.profiling_results().expect("profiling enabled");
        assert!(!profiles.is_empty());
        assert_eq!(profiles[0].node_name, "id_node");
        assert_eq!(profiles[0].op_type, "Identity");
        assert_eq!(profiles[0].output_shapes, vec![vec![1, 3]]);
    }

    #[test]
    fn test_profiling_disabled_returns_none() {
        let session = Session::from_bytes(&[]).expect("load empty model");
        assert!(session.profiling_results().is_none());
    }

    #[test]
    fn test_builder_default() {
        let builder = SessionBuilder::default();
        assert_eq!(builder.opt_level, OptLevel::All);
        assert!(!builder.enable_profiling);
        assert!(!builder.enable_memory_pool);
        assert!(builder.registry.is_none());
    }

    #[test]
    fn test_opt_level_variants() {
        assert_ne!(OptLevel::None, OptLevel::Basic);
        assert_ne!(OptLevel::Basic, OptLevel::Extended);
        assert_ne!(OptLevel::Extended, OptLevel::All);
        // Clone + Copy
        let level = OptLevel::All;
        let cloned = level;
        assert_eq!(level, cloned);
    }

    // ── Parallel execution tests ────────────────────────────────────────

    /// Build a graph with two independent branches at the same depth:
    ///   input -> Relu(branch_a) -> output_a
    ///   input -> Relu(branch_b) -> output_b
    /// Both Relu nodes are at depth 0 and should run in parallel.
    #[test]
    fn test_parallel_execution_basic() {
        let node_a = Node {
            op: OpKind::Relu,
            name: "relu_a".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_a".to_string()],
            attrs: Attributes::default(),
        };
        let node_b = Node {
            op: OpKind::Relu,
            name: "relu_b".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_b".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node_a, node_b],
            input_names: vec!["input".to_string()],
            output_names: vec!["out_a".to_string(), "out_b".to_string()],
        };

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(true)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let input = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
        let outputs = session.run_one("input", input).expect("run");

        let expected = vec![0.0, 2.0, 0.0, 4.0];
        let out_a = outputs.get("out_a").expect("out_a");
        let out_b = outputs.get("out_b").expect("out_b");
        assert_eq!(out_a.data, expected);
        assert_eq!(out_b.data, expected);
    }

    /// All nodes sequential (linear chain). Parallel mode should not break anything.
    #[test]
    fn test_parallel_single_node_levels() {
        let node1 = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["mid".to_string()],
            attrs: Attributes::default(),
        };
        let node2 = Node {
            op: OpKind::Relu,
            name: "relu2".to_string(),
            inputs: vec!["mid".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node1, node2],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
        };

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(true)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let input = Tensor::new(vec![-1.0, 5.0, -2.0], vec![1, 3]);
        let outputs = session.run_one("input", input).expect("run");
        let out = outputs.get("output").expect("output");
        assert_eq!(out.data, vec![0.0, 5.0, 0.0]);
    }

    // ── In-place execution tests ────────────────────────────────────────

    /// Verify ReLU produces correct output (in-place path used when ref_count==1).
    #[test]
    fn test_inplace_relu() {
        let node = Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
        };

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let input = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);
        let outputs = session.run_one("x", input).expect("run");
        let y = outputs.get("y").expect("y");
        assert_eq!(y.data, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
    }

    /// Verify element-wise Add works in-place when shapes match.
    #[test]
    fn test_inplace_add_same_shape() {
        // x -> Add(x, w) -> y   where x and w have same shape
        let node = Node {
            op: OpKind::Add,
            name: "add".to_string(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
        };

        let mut weights = HashMap::new();
        weights.insert(
            "w".to_string(),
            Tensor::new(vec![10.0, 20.0, 30.0], vec![3]),
        );

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let outputs = session.run_one("x", input).expect("run");
        let y = outputs.get("y").expect("y");
        assert_eq!(y.data, vec![11.0, 22.0, 33.0]);
    }

    /// Verify broadcast Add falls back to regular path (shapes differ).
    #[test]
    fn test_inplace_fallback_broadcast() {
        // x [2,3] + w [3] -> y [2,3]   (broadcasting needed, inplace should fallback)
        let node = Node {
            op: OpKind::Add,
            name: "add".to_string(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
        };

        let mut weights = HashMap::new();
        weights.insert(
            "w".to_string(),
            Tensor::new(vec![10.0, 20.0, 30.0], vec![3]),
        );

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let outputs = session.run_one("x", input).expect("run");
        let y = outputs.get("y").expect("y");
        assert_eq!(y.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        assert_eq!(y.shape, vec![2, 3]);
    }

    /// A tensor consumed by 2 nodes should NOT be modified in-place.
    #[test]
    fn test_inplace_respects_refcount() {
        // input -> relu_a -> out_a
        // input -> relu_b -> out_b
        // "input" has refcount 2, so neither relu should modify it in-place.
        let node_a = Node {
            op: OpKind::Relu,
            name: "relu_a".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_a".to_string()],
            attrs: Attributes::default(),
        };
        let node_b = Node {
            op: OpKind::Relu,
            name: "relu_b".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_b".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node_a, node_b],
            input_names: vec!["input".to_string()],
            output_names: vec!["out_a".to_string(), "out_b".to_string()],
        };

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let input = Tensor::new(vec![-2.0, 3.0, -1.0, 5.0], vec![2, 2]);
        let outputs = session.run_one("input", input).expect("run");

        let expected = vec![0.0, 3.0, 0.0, 5.0];
        let out_a = outputs.get("out_a").expect("out_a");
        let out_b = outputs.get("out_b").expect("out_b");
        assert_eq!(out_a.data, expected);
        assert_eq!(out_b.data, expected);
    }

    /// Test depth computation helper directly.
    #[test]
    fn test_compute_node_depths() {
        // A linear chain: input -> relu1 -> relu2 -> output
        let node1 = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["mid".to_string()],
            attrs: Attributes::default(),
        };
        let node2 = Node {
            op: OpKind::Relu,
            name: "relu2".to_string(),
            inputs: vec!["mid".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let nodes = vec![node1, node2];
        let weights = HashMap::new();
        let depths = Session::compute_node_depths(&nodes, &weights);
        assert_eq!(depths, vec![0, 1]);
    }

    /// Test depth computation with independent branches.
    #[test]
    fn test_compute_node_depths_parallel_branches() {
        // input -> relu_a -> out_a  (depth 0)
        // input -> relu_b -> out_b  (depth 0)
        let node_a = Node {
            op: OpKind::Relu,
            name: "relu_a".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_a".to_string()],
            attrs: Attributes::default(),
        };
        let node_b = Node {
            op: OpKind::Relu,
            name: "relu_b".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_b".to_string()],
            attrs: Attributes::default(),
        };
        let nodes = vec![node_a, node_b];
        let weights = HashMap::new();
        let depths = Session::compute_node_depths(&nodes, &weights);
        assert_eq!(depths, vec![0, 0]);
    }

    #[test]
    fn test_group_by_depth() {
        let depths = vec![0, 0, 1, 2, 1];
        let groups = Session::group_by_depth(&depths);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![0, 1]);
        assert_eq!(groups[1], vec![2, 4]);
        assert_eq!(groups[2], vec![3]);
    }
}
