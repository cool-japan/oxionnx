use crate::execution_providers::ProviderKind;
use crate::memory::PoolStats;
use crate::tensor::Tensor;
use oxionnx_core::Operator;
use std::collections::HashMap;

use super::types::{ModelInfo, ModelMetadata, NodeProfile};
use super::Session;

impl Session {
    /// Register an additional (or replacement) operator at runtime.
    ///
    /// The operator is keyed by its own [`Operator::op_type`], so registering a
    /// name that already exists replaces the existing implementation.
    ///
    /// # Cancellation
    ///
    /// On a session built with
    /// [`with_session_cancellation`](crate::SessionBuilder::with_session_cancellation),
    /// the operator is wrapped so that it, too, consults the session's
    /// [`CancellationToken`](crate::CancellationToken) before it executes — a
    /// late registration is a cancellation point exactly like every operator the
    /// session was built with.  It used to go straight into the already-wrapped
    /// registry unguarded, so a model whose long-running node was a
    /// late-registered custom operator could not be stopped at that node at all.
    ///
    /// The wrapping is transparent: the registry key and every dispatch
    /// predicate (`supports_inplace`, `supports_output_slots`, `native_dtypes`)
    /// are the inner operator's, so the in-place, output-slot and typed fast
    /// paths stay exactly as available as they were.
    pub fn register_op(&mut self, op: Box<dyn Operator>) {
        let op = match self.cancellation.as_ref() {
            Some(token) => super::cancellation::wrap_owned_op(op, token),
            None => op,
        };
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

    /// Return detailed metadata for each graph input (name, dtype, shape).
    ///
    /// Populated from `ValueInfoProto` when the model encodes type information.
    /// Returns an empty slice when the loaded model omits type annotations.
    pub fn input_info(&self) -> &[oxionnx_core::TensorInfo] {
        &self.input_infos
    }

    /// Return detailed metadata for each graph output (name, dtype, shape).
    ///
    /// Populated from `ValueInfoProto` when the model encodes type information.
    /// Returns an empty slice when the loaded model omits type annotations.
    pub fn output_info(&self) -> &[oxionnx_core::TensorInfo] {
        &self.output_infos
    }

    /// Enumerate the model's compute nodes in deterministic topological
    /// (graph execution) order.
    ///
    /// Each returned [`NodeInfo`](oxionnx_core::NodeInfo) is a read-only
    /// snapshot of one operator: its `name`, `op_type` (the canonical ONNX op
    /// string such as `"Split"` or `"Conv"`), `inputs`, `outputs`, and a
    /// deterministic `attributes` summary (`name -> value` pairs, e.g.
    /// `axis -> 2`, `split -> [32, 32, 64]`).
    ///
    /// The order matches `model_info()` / `export_dot()` (the internal
    /// topologically-sorted node list), so repeated calls are stable.
    ///
    /// # Example
    ///
    /// Print every node's wiring — directly answering "get nodes from the
    /// model to print them as input/output fields":
    ///
    /// ```
    /// use oxionnx::{Session, Graph, Node, OpKind, Attributes};
    /// use std::collections::HashMap;
    ///
    /// // Build a tiny two-node graph: x -> Relu -> r -> Identity -> out
    /// let relu = Node {
    ///     op: OpKind::Relu,
    ///     name: "relu1".to_string(),
    ///     inputs: vec!["x".to_string()],
    ///     outputs: vec!["r".to_string()],
    ///     attrs: Attributes::default(),
    /// };
    /// let id = Node {
    ///     op: OpKind::Identity,
    ///     name: "id1".to_string(),
    ///     inputs: vec!["r".to_string()],
    ///     outputs: vec!["out".to_string()],
    ///     attrs: Attributes::default(),
    /// };
    /// let graph = Graph {
    ///     nodes: vec![relu, id],
    ///     input_names: vec!["x".to_string()],
    ///     output_names: vec!["out".to_string()],
    ///     ..Default::default()
    /// };
    ///
    /// let session = Session::from_graph(graph, HashMap::new())?;
    /// for node in session.nodes() {
    ///     println!(
    ///         "{} ({}): inputs={:?} outputs={:?}",
    ///         node.name, node.op_type, node.inputs, node.outputs
    ///     );
    ///     for (attr_name, attr_value) in &node.attributes {
    ///         println!("    {attr_name} = {attr_value}");
    ///     }
    /// }
    ///
    /// let nodes = session.nodes();
    /// assert_eq!(nodes.len(), 2);
    /// assert_eq!(nodes[0].op_type, "Relu");
    /// assert_eq!(nodes[0].outputs, vec!["r".to_string()]);
    /// # Ok::<(), oxionnx::Error>(())
    /// ```
    pub fn nodes(&self) -> Vec<oxionnx_core::NodeInfo> {
        self.sorted_nodes
            .iter()
            .map(oxionnx_core::NodeInfo::from_node)
            .collect()
    }

    /// Return a reference to the model's weight tensors.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }

    /// Return the model metadata (producer, IR version, opset imports, custom properties).
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
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

    /// Return statistics from the size-class memory pool.
    ///
    /// Returns `None` if the memory pool was not enabled at session build time.
    pub fn pool_stats(&self) -> Option<PoolStats> {
        self.pool
            .as_ref()
            .and_then(|m| m.lock().ok().map(|p| p.stats().clone()))
    }

    /// Drop every idle GPU buffer this session has pooled, returning the memory
    /// to the driver.  Returns the number of buffers released.
    ///
    /// # Why this is a method and not a step at the end of `run()`
    ///
    /// Wave-2's GPU lane proposed calling `pool.clear()` unconditionally when a
    /// run finishes.  That is the wrong default: a `Session` exists to be run
    /// **repeatedly**, and clearing the pool between runs means every inference
    /// re-creates its device buffers — the pool would then only ever serve a
    /// single run, which is precisely the case it was built to stop paying for.
    /// The same lane's own report downgrades the item to "cosmetic, not
    /// load-bearing": the pool is already bounded by a 256 MiB byte budget with
    /// LRU eviction, so nothing is unbounded without the call.
    ///
    /// What was genuinely missing is the *ability* to release that memory at a
    /// point the caller chooses — after the last inference of a batch job, when
    /// a long-lived service goes idle, or before handing the GPU to another
    /// process.  That is what this is.
    ///
    /// Returns `0` when the session has no live GPU context (no device, or a
    /// CPU-only session), and also when the pool mutex is poisoned — a
    /// best-effort release must not turn one panicking thread into a permanent
    /// error for every later caller.
    #[cfg(feature = "gpu")]
    #[must_use = "the count says how many buffers were actually released"]
    pub fn release_gpu_pool(&self) -> usize {
        let Some(ref gpu) = self.gpu else {
            return 0;
        };
        let Ok(mut pool) = gpu.pool.lock() else {
            return 0;
        };
        let released = pool.available_count();
        pool.clear();
        released
    }

    /// Return the ordered execution provider list configured for this session.
    ///
    /// Returns an empty slice when no explicit list was set (legacy heuristic
    /// / compile-time feature-flag dispatch is used in that case).
    #[must_use]
    pub fn provider_kinds(&self) -> &[ProviderKind] {
        &self.providers
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
