use crate::execution_providers::ProviderKind;
use crate::memory::PoolStats;
use crate::tensor::Tensor;
use oxionnx_core::Operator;
use std::collections::HashMap;

use super::types::{ModelInfo, ModelMetadata, NodeProfile};
use super::Session;

impl Session {
    /// A snapshot of this session's CUDA device-cache and transfer counters.
    ///
    /// `None` when the session has no CUDA context (no device, or CUDA was not
    /// activated). The counters are cumulative for the context's lifetime, so
    /// "what did this frame move" is
    /// [`CacheCounters::since`](oxionnx_cuda::residency::CacheCounters::since)
    /// between two snapshots.
    ///
    /// # What to watch
    ///
    /// * `weight_bytes_uploaded` — must be **zero** across a steady-state
    ///   frame, or some initializer is crossing the bus every frame.
    /// * `host_to_device_bytes` / `device_to_host_bytes` — the whole bus cost of
    ///   a frame, counted at the copies themselves rather than inferred from
    ///   shapes.
    /// * `stream_syncs` — blocking host↔device rendezvous. Before activation
    ///   residency this was one per CUDA-claimed node; after it, one per
    ///   host-visible result.
    /// * `resident_activation_binds` / `device_handoffs` — operands bound
    ///   without an upload, and results kept without a read-back. These are the
    ///   two halves of what residency actually did.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_cache_counters(&self) -> Option<oxionnx_cuda::residency::CacheCounters> {
        self.cuda
            .as_ref()
            .map(oxionnx_cuda::CudaContext::cache_counters)
    }

    /// Whether this session's CUDA context issues every launch and copy on one
    /// driver queue.
    ///
    /// `None` when there is no CUDA context. `Some(false)` means the context
    /// was built with a split BLAS stream, in which case activation residency
    /// is switched off for its runs — see
    /// `session::run::sequential::CUDA_RESIDENCY_ENV_VAR`.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_streams_unified(&self) -> Option<bool> {
        self.cuda
            .as_ref()
            .map(oxionnx_cuda::CudaContext::streams_unified)
    }
}

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

    /// [w2-f16] Whether this session's GPU device can run the half-precision
    /// kernels at all.
    ///
    /// `false` when the session has no device, or when the adapter did not
    /// report `shader-f16`. A caller offering the mode as a user-facing option
    /// should ask this first; a caller that just wants it on can call
    /// [`Self::set_f16_compute`] unconditionally and read the answer back.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn f16_compute_supported(&self) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|gpu| gpu.f16_compute_supported())
    }

    /// [w2-f16] Ask the convolution and GEMM kernels to compute in half
    /// precision, and get back the state that actually took effect.
    ///
    /// **Off by default.** The mode changes results — weights are narrowed to
    /// `f16` on the device and products are evaluated at half precision, though
    /// every accumulator stays `f32` — so it is strictly opt-in. See
    /// `oxionnx_gpu::context::weight_format` for the enumerated rounding points
    /// and the measured PSNR.
    ///
    /// Returns the **effective** state, which is `false` whenever the session
    /// has no device or the device lacks the feature. So
    /// `session.set_f16_compute(true) == false` is a complete answer: nothing
    /// changed, and every kernel is still on its `f32` path.
    ///
    /// Safe to call between runs on a live session; the residency cache keys
    /// weights by format, so a flip cannot serve a kernel bytes in a width its
    /// shader does not read.
    #[cfg(feature = "gpu")]
    pub fn set_f16_compute(&self, enabled: bool) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|gpu| gpu.set_f16_compute(enabled))
    }

    /// [w2-f16] Whether dispatches are currently taking the half-precision
    /// path.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn f16_compute_enabled(&self) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|gpu| gpu.f16_compute_enabled())
    }

    /// Device bytes this session's weights are holding for its whole lifetime.
    ///
    /// The initializers the GPU path has uploaded and kept — a subset of the
    /// device's live bytes that no run will release. `0` without a device.
    ///
    /// [w2-f16] Exposed so a caller can *see* the footprint change when it
    /// turns half precision on: an initializer resident in both formats is
    /// counted once per format, because that is what the device is actually
    /// holding.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_resident_bytes(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |gpu| gpu.resident_bytes())
    }

    /// Whether kernels may leave a result in a device buffer for the next node
    /// to consume in place, instead of reading it back and uploading it again.
    ///
    /// `false` without a device. On by default when there is one.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn activation_residency_enabled(&self) -> bool {
        self.gpu
            .as_ref()
            .is_some_and(|gpu| gpu.activation_residency_enabled())
    }

    /// Turn run-scoped activation residency on or off, and get back the state
    /// that actually took effect.
    ///
    /// The switch exists so the two paths can be compared *within one session*:
    /// with it off every entry point behaves exactly as it did before run-scoped
    /// activations existed, so an A/B is a measurement of the mechanism rather
    /// than of two differently-built sessions. The counterpart of
    /// [`Self::set_f16_compute`] for the other half of the GPU data path, and
    /// like it, safe to call between runs on a live session.
    ///
    /// Returns `false` when the session has no device, in which case nothing
    /// changed and there was nothing to change.
    #[cfg(feature = "gpu")]
    pub fn set_activation_residency(&self, enabled: bool) -> bool {
        match self.gpu.as_ref() {
            Some(gpu) => {
                gpu.set_activation_residency(enabled);
                gpu.activation_residency_enabled()
            }
            None => false,
        }
    }

    /// Device bytes this session's context currently has allocated, idle pooled
    /// buffers included.
    ///
    /// `0` without a device.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_live_bytes(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |gpu| gpu.live_gpu_bytes())
    }

    /// Device bytes held by idle buffers waiting to be reused — a subset of
    /// [`Self::gpu_live_bytes`] that no dispatch is reading.
    ///
    /// Run-over-run growth here is the memory-boundedness question that decides
    /// whether recycling is affordable, so it is exposed rather than inferred.
    /// `0` without a device.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_pooled_bytes(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |gpu| gpu.pooled_gpu_bytes())
    }

    /// Idle buffers this session's context is pooling, and the count bound it
    /// keeps them under. `(0, 0)` without a device.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_pooled_buffers(&self) -> (usize, usize) {
        self.gpu.as_ref().map_or((0, 0), |gpu| gpu.pooled_buffers())
    }

    /// The byte retention bound this session's context keeps its idle pooled
    /// buffers under. `0` without a device.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_pool_byte_budget(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |gpu| gpu.pool_byte_budget())
    }

    /// Buffer requests this session's context served from an idle pooled entry,
    /// cumulative. `0` without a device.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_pool_reuses(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |gpu| gpu.pool_reuses())
    }

    /// Buffer requests this session's context had to ask the driver for,
    /// cumulative. `0` without a device.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_pool_allocations(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |gpu| gpu.pool_allocations())
    }

    /// Cumulative host→device bytes this session's context has uploaded —
    /// weights, activations and parameter blocks alike.
    ///
    /// Never falls and never resets, so one run's upload volume is the
    /// difference between two readings taken around it. That difference is the
    /// only complete account of the outbound half of a run's bus traffic:
    /// [`GpuRunStats::weight_upload_bytes`](crate::session::gpu_residency::GpuRunStats::weight_upload_bytes)
    /// covers initializers only and
    /// [`activation_upload_bytes`](crate::session::gpu_residency::GpuRunStats::activation_upload_bytes)
    /// covers only the operands promoted ahead of a dispatch — both are subsets
    /// of this, so adding them to it would double-count.
    ///
    /// `0` without a device.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_uploaded_bytes(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |gpu| gpu.uploaded_bytes())
    }

    /// The first unrecoverable device error this session's GPU context saw, if
    /// any.
    ///
    /// `Some` means every later GPU entry point declined and the graph finished
    /// on CPU operators — so a measurement taken across such a run is timing a
    /// fallback, not the device. `None` without a device, and `None` on a
    /// healthy one.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_device_error(&self) -> Option<String> {
        let gpu = self.gpu.as_ref()?;
        if !gpu.is_degraded() {
            return None;
        }
        // Degraded with an empty error slot should not read as healthy: the
        // caller asked whether the device is usable, and it is not.
        Some(
            gpu.last_error()
                .unwrap_or_else(|| "device degraded without a recorded reason".to_string()),
        )
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
