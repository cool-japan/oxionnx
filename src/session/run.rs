use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{Dim, OpContext, Operator, TensorInfo};
use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use rayon::prelude::*;

use super::types::NodeProfile;
use super::Session;

impl Session {
    /// Build a map of symbolic dimension names to concrete values from input tensors.
    ///
    /// For each model input that has symbolic dimensions (e.g. "batch_size", "seq_len"),
    /// the corresponding axis of the actual input tensor provides the concrete value.
    /// Returns a `HashMap<String, usize>` mapping each symbol to its resolved size.
    pub fn resolve_dynamic_shapes(
        input_infos: &[TensorInfo],
        inputs: &HashMap<&str, &Tensor>,
    ) -> Result<HashMap<String, usize>, OnnxError> {
        let mut dim_map: HashMap<String, usize> = HashMap::new();

        for info in input_infos {
            let tensor = match inputs.get(info.name.as_str()) {
                Some(t) => t,
                None => continue, // input not provided; skip
            };

            let symbolic = info.symbolic_shape();
            for (axis, dim) in symbolic.iter().enumerate() {
                if let Dim::Symbol(ref sym) = dim {
                    if axis >= tensor.shape.len() {
                        return Err(OnnxError::ShapeMismatch(format!(
                            "Input '{}': symbolic dim '{}' at axis {} but tensor rank is {}",
                            info.name,
                            sym,
                            axis,
                            tensor.shape.len()
                        )));
                    }
                    let actual = tensor.shape[axis];
                    if let Some(&existing) = dim_map.get(sym) {
                        if existing != actual {
                            return Err(OnnxError::ShapeMismatch(format!(
                                "Symbolic dimension '{}' has conflicting values: \
                                 {} (from earlier input) vs {} (from input '{}')",
                                sym, existing, actual, info.name
                            )));
                        }
                    } else {
                        dim_map.insert(sym.clone(), actual);
                    }
                }
            }
        }

        Ok(dim_map)
    }

    /// Validate input tensor shapes against model input metadata.
    ///
    /// Checks:
    /// 1. Rank (number of dimensions) matches expected rank.
    /// 2. Static dimensions match exactly.
    /// 3. Symbolic dimensions are consistent across all inputs (same symbol → same value).
    pub fn validate_input_shapes(
        input_infos: &[TensorInfo],
        inputs: &HashMap<&str, &Tensor>,
    ) -> Result<(), OnnxError> {
        let mut sym_values: HashMap<String, usize> = HashMap::new();

        for info in input_infos {
            let tensor = match inputs.get(info.name.as_str()) {
                Some(t) => t,
                None => continue,
            };

            let symbolic = info.symbolic_shape();
            if symbolic.is_empty() {
                continue; // no shape info to validate
            }

            // Check rank
            if tensor.shape.len() != symbolic.len() {
                return Err(OnnxError::ShapeMismatch(format!(
                    "Input '{}': expected rank {} but got rank {}",
                    info.name,
                    symbolic.len(),
                    tensor.shape.len()
                )));
            }

            // Check each dimension
            for (axis, dim) in symbolic.iter().enumerate() {
                let actual = tensor.shape[axis];
                match dim {
                    Dim::Static(expected) => {
                        if actual != *expected {
                            return Err(OnnxError::ShapeMismatch(format!(
                                "Input '{}': axis {} expected static dim {} but got {}",
                                info.name, axis, expected, actual
                            )));
                        }
                    }
                    Dim::Symbol(ref sym) => {
                        if let Some(&prev) = sym_values.get(sym.as_str()) {
                            if prev != actual {
                                return Err(OnnxError::ShapeMismatch(format!(
                                    "Symbolic dimension '{}' is inconsistent: \
                                     {} vs {} (input '{}' axis {})",
                                    sym, prev, actual, info.name, axis
                                )));
                            }
                        } else {
                            sym_values.insert(sym.clone(), actual);
                        }
                    }
                    Dim::Unknown => { /* anything goes */ }
                }
            }
        }

        Ok(())
    }

    /// Update the session's dynamic dimension cache and re-resolve intermediate
    /// shapes if the input shapes have changed since the last call.
    fn update_dynamic_dims(&self, inputs: &HashMap<&str, &Tensor>) -> Result<(), OnnxError> {
        if self.input_infos.is_empty() {
            return Ok(());
        }

        let new_dims = Self::resolve_dynamic_shapes(&self.input_infos, inputs)?;
        if new_dims.is_empty() {
            return Ok(());
        }

        // Check if dims changed
        let dims_changed = {
            let current = self
                .dynamic_dims
                .lock()
                .map_err(|e| OnnxError::Internal(format!("dynamic_dims lock: {e}")))?;
            *current != new_dims
        };

        if dims_changed {
            // Update dynamic dims
            {
                let mut dd = self
                    .dynamic_dims
                    .lock()
                    .map_err(|e| OnnxError::Internal(format!("dynamic_dims lock: {e}")))?;
                *dd = new_dims;
            }

            // Re-resolve intermediate shapes using actual input shapes
            let input_shapes: HashMap<String, Vec<usize>> = inputs
                .iter()
                .map(|(name, tensor)| (name.to_string(), tensor.shape.clone()))
                .collect();
            let new_shapes = crate::optimizer::shape_inference::infer_shapes(
                &self.sorted_nodes,
                &self.weights,
                &input_shapes,
            );

            let mut rs = self
                .resolved_shapes
                .lock()
                .map_err(|e| OnnxError::Internal(format!("resolved_shapes lock: {e}")))?;
            *rs = new_shapes;
        }

        Ok(())
    }

    /// Return the current dynamic dimension bindings.
    pub fn dynamic_dims(&self) -> HashMap<String, usize> {
        self.dynamic_dims
            .lock()
            .map(|d| d.clone())
            .unwrap_or_default()
    }

    /// Return the current resolved intermediate tensor shapes.
    pub fn resolved_shapes(&self) -> HashMap<String, Vec<usize>> {
        self.resolved_shapes
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
    /// Compute the topological depth for each node in `sorted_nodes`.
    /// Depth 0 = all inputs come from model inputs / weights (no graph predecessors).
    /// For others, depth = max(depth of predecessor nodes) + 1.
    pub(crate) fn compute_node_depths(
        sorted_nodes: &[Node],
        weights: &HashMap<String, Tensor>,
    ) -> Vec<usize> {
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
    pub(crate) fn group_by_depth(depths: &[usize]) -> Vec<Vec<usize>> {
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        let mut groups = vec![Vec::new(); max_depth + 1];
        for (i, &d) in depths.iter().enumerate() {
            groups[d].push(i);
        }
        groups
    }

    /// Execute a single node, with optional in-place optimization.
    pub(crate) fn execute_node_with_inplace(
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
    pub(crate) fn decrement_refs(
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
                                pool.release(buf);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Core inference engine shared by `run` and `run_with_binding`.
    ///
    /// Accepts borrowed tensors to avoid the per-call clone that `run`
    /// would otherwise perform for all inputs.
    pub(crate) fn run_internal(
        &self,
        inputs: &HashMap<&str, &Tensor>,
    ) -> Result<HashMap<String, Tensor>, OnnxError> {
        // Validate input shapes against model metadata (rank, static dims, symbolic consistency)
        if !self.input_infos.is_empty() {
            Self::validate_input_shapes(&self.input_infos, inputs)?;
        }

        // Update dynamic dimension bindings and re-resolve intermediate shapes if needed
        self.update_dynamic_dims(inputs)?;

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
        // Clone input tensor data into intermediates (one clone per input, not per op)
        for (name, tensor) in inputs {
            intermediates.insert(name.to_string(), (*tensor).clone());
        }

        let use_parallel = self.parallel && cfg!(not(target_arch = "wasm32"));

        if self.mixed_precision {
            tracing::trace!("Running inference with mixed-precision mode");
        }

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
        let input_refs: HashMap<&str, &Tensor> = inputs.iter().map(|(k, v)| (*k, v)).collect();
        self.run_internal(&input_refs)
    }

    /// Run inference using pre-allocated I/O buffers.
    ///
    /// Avoids input tensor allocation on repeated calls. Output buffers
    /// pre-allocated via [`crate::IoBinding::bind_output`] are reused when the shape
    /// matches; otherwise they are replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying graph execution fails.
    pub fn run_with_binding(&self, binding: &mut crate::IoBinding) -> Result<(), OnnxError> {
        let input_refs: HashMap<&str, &Tensor> = binding
            .inputs()
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();

        let outputs = self.run_internal(&input_refs)?;

        let buf_map = binding.outputs_mut();
        for (name, tensor) in outputs {
            if let Some(buf) = buf_map.get_mut(&name) {
                if buf.data.len() == tensor.data.len() && buf.shape == tensor.shape {
                    buf.data.copy_from_slice(&tensor.data);
                } else {
                    *buf = tensor;
                }
            } else {
                buf_map.insert(name, tensor);
            }
        }
        Ok(())
    }

    /// Estimate the output tensor size in bytes for a node, using resolved
    /// shapes when available or falling back to input tensor sizes.
    pub(crate) fn estimate_output_bytes(
        node: &Node,
        intermediates: &HashMap<String, Tensor>,
        weights: &HashMap<String, Tensor>,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> usize {
        // Try resolved shapes for the first output
        if let Some(first_out) = node.outputs.first() {
            if let Some(shape) = resolved_shapes.get(first_out) {
                let elems: usize = shape.iter().product();
                // f32 → 4 bytes per element
                return elems.saturating_mul(4);
            }
        }
        // Fallback: use the first input tensor size as a proxy
        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            if let Some(t) = intermediates.get(inp).or_else(|| weights.get(inp)) {
                return t.data.len().saturating_mul(4);
            }
        }
        0
    }

    /// Sequential execution path with in-place optimization.
    pub(crate) fn run_sequential_inner(
        &self,
        intermediates: &mut HashMap<String, Tensor>,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) -> Result<(), OnnxError> {
        use crate::execution_providers::{decide_placement, ProviderKind};

        let resolved = self
            .resolved_shapes
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default();

        for node in &self.sorted_nodes {
            if let OpKind::Unknown(_) = &node.op {
                continue;
            }

            // Determine operator placement based on the configured strategy
            let output_bytes =
                Self::estimate_output_bytes(node, intermediates, &self.weights, &resolved);
            let placement = decide_placement(&node.op, output_bytes, &self.op_placement);

            // CUDA dispatch (only when placement allows)
            #[cfg(feature = "cuda")]
            {
                let try_cuda = self.cuda.is_some()
                    && !matches!(
                        self.op_placement,
                        crate::execution_providers::OpPlacement::CpuOnly
                    );
                if try_cuda {
                    if let Some(cuda_ctx) = &self.cuda {
                        let cuda_start = std::time::Instant::now();
                        match oxionnx_cuda::try_cuda_dispatch(
                            node,
                            &self.weights,
                            intermediates,
                            cuda_ctx,
                        ) {
                            Ok(Some(results)) => {
                                let cuda_elapsed = cuda_start.elapsed();
                                if let Some(ref profiling) = self.profiling_data {
                                    if let Ok(mut data) = profiling.lock() {
                                        data.push(NodeProfile {
                                            node_name: node.name.clone(),
                                            op_type: node.op.as_str().to_string(),
                                            duration: cuda_elapsed,
                                            output_shapes: results
                                                .iter()
                                                .map(|t| t.shape.clone())
                                                .collect(),
                                        });
                                    }
                                }
                                for (name, tensor) in node.outputs.iter().zip(results) {
                                    if !name.is_empty() {
                                        intermediates.insert(name.clone(), tensor);
                                    }
                                }
                                self.decrement_refs(node, intermediates, ref_counts, output_set);
                                continue;
                            }
                            Ok(None) => {
                                // Op not supported on CUDA — fall through to CPU
                            }
                            Err(_e) => {
                                // CUDA dispatch failed — fall back to CPU gracefully
                                #[cfg(debug_assertions)]
                                tracing::debug!(
                                    op = %node.op.as_str(),
                                    node = %node.name,
                                    err = %_e,
                                    "CUDA dispatch error, falling back to CPU",
                                );
                            }
                        }
                    }
                }
            }

            // GPU dispatch (only when placement routes to GPU)
            #[cfg(feature = "gpu")]
            {
                use super::gpu_dispatch::{try_gpu_dispatch, GpuExecutionProvider};
                let try_gpu = matches!(placement, ProviderKind::Gpu);
                if try_gpu {
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
                            tracing::debug!(
                                op = %node.op.as_str(),
                                node = %node.name,
                                "GPU fallback: fell back to CPU",
                            );
                        }
                    }
                }
            }

            let op_name = node.op.as_str();

            // Mixed precision: try native f16 execution for f16-safe element-wise ops
            if self.mixed_precision && super::mixed_precision::should_use_f16(op_name) {
                let input_refs: Vec<&Tensor> = node
                    .inputs
                    .iter()
                    .filter_map(|name| {
                        if name.is_empty() {
                            None
                        } else {
                            intermediates.get(name).or_else(|| self.weights.get(name))
                        }
                    })
                    .collect();

                let start = std::time::Instant::now();
                if let Some(f16_result) =
                    super::mixed_precision::execute_elementwise_f16(op_name, &input_refs)
                {
                    let results = f16_result?;
                    let elapsed = start.elapsed();

                    if let Some(ref profiling) = self.profiling_data {
                        if let Ok(mut data) = profiling.lock() {
                            data.push(NodeProfile {
                                node_name: node.name.clone(),
                                op_type: format!("{op_name}(f16)"),
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
                    continue;
                }
                // No native f16 path — fall through to normal execution with f16 rounding
            }

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

            // Mixed precision: round outputs to f16 for f16-safe ops without native f16 path.
            // This simulates f16 storage precision for ops that ran in f32.
            let results = if self.mixed_precision && super::mixed_precision::should_use_f16(op_name)
            {
                results
                    .into_iter()
                    .map(|t| super::mixed_precision::round_to_f16_precision(&t))
                    .collect()
            } else {
                results
            };

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
    pub(crate) fn run_parallel_inner(
        &self,
        intermediates: &mut HashMap<String, Tensor>,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) -> Result<(), OnnxError> {
        let depths = Self::compute_node_depths(&self.sorted_nodes, &self.weights);
        let mut groups = Self::group_by_depth(&depths);

        // Sort nodes within each level by critical-path cost (descending).
        // This ensures the heaviest work starts first, reducing tail latency.
        let critical_costs = crate::optimizer::cost_model::compute_critical_path_costs(
            &self.sorted_nodes,
            self.shape_cache.as_ref(),
        );
        for group in &mut groups {
            group.sort_by(|&a, &b| critical_costs[b].cmp(&critical_costs[a]));
        }

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
                let par_execute = || -> Vec<ParResult<'_>> {
                    work_items
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
                        .collect()
                };
                let par_results: Vec<ParResult<'_>> = if let Some(ref pool) = self.thread_pool {
                    pool.install(par_execute)
                } else {
                    par_execute()
                };

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
    pub(crate) fn run_parallel_inner(
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

    /// Run inference with multi-dtype inputs and outputs.
    ///
    /// Accepts [`oxionnx_core::TypedTensor`] inputs with any supported dtype (i64, f16, bf16, i32,
    /// bool, …) by converting them to f32 internally. Output dtypes are recovered
    /// from [`Session::output_info`].
    ///
    /// # Precision caveat
    /// The internal f32 representation has ~24 bits of significand. Integer
    /// tensors whose absolute values exceed 2^24 (~16.7 million) may lose
    /// precision when converted through f32. This is generally acceptable for
    /// token IDs and sequence lengths in transformer models, but should be
    /// considered for other use cases.
    pub fn run_typed(
        &self,
        inputs: &HashMap<&str, oxionnx_core::TypedTensor>,
    ) -> Result<HashMap<String, oxionnx_core::TypedTensor>, OnnxError> {
        // 1. Convert TypedTensor → f32 Tensor
        let f32_inputs: HashMap<&str, Tensor> = inputs
            .iter()
            .map(|(&name, tt)| {
                let data = tt.storage.to_f32_vec();
                (name, Tensor::new(data, tt.shape.clone()))
            })
            .collect();

        // 2. Run existing f32 inference
        let f32_outputs = self.run(&f32_inputs)?;

        // 3. Recover output dtype from output_info, convert back
        let typed_outputs = f32_outputs
            .into_iter()
            .map(|(name, tensor)| {
                let dtype = self
                    .output_info()
                    .iter()
                    .find(|info| info.name == name)
                    .map(|info| info.dtype)
                    .unwrap_or(oxionnx_core::DType::F32);
                let typed =
                    oxionnx_core::TypedTensor::from_f32_vec(tensor.data, tensor.shape, dtype)?;
                Ok((name, typed))
            })
            .collect::<Result<_, OnnxError>>()?;

        Ok(typed_outputs)
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
}
