#[cfg(not(target_arch = "wasm32"))]
use crate::graph::OpKind;
#[cfg(not(target_arch = "wasm32"))]
use crate::memory::SizeClassPool;
#[cfg(not(target_arch = "wasm32"))]
use crate::tensor::Tensor;
use crate::OnnxError;
#[cfg(not(target_arch = "wasm32"))]
use oxionnx_core::{OpContext, Operator};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
use super::super::types::NodeProfile;
use super::super::Session;
use super::state::SessionRunState;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

impl Session {
    /// Try to dispatch a single node through the hardware-acceleration hierarchy
    /// (CUDA → DirectML → wgpu GPU).  Returns `Ok(true)` if a backend handled the
    /// node and wrote its outputs into `state`; `Ok(false)` signals that no
    /// accelerator claimed the node and the caller must fall back to CPU.
    ///
    /// This mirrors the dispatch order in `run_sequential_inner` so that the
    /// parallel and sequential paths are consistent.
    #[cfg(not(target_arch = "wasm32"))]
    fn try_accelerated_node(
        &self,
        node: &crate::graph::Node,
        state: &mut SessionRunState,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Result<bool, OnnxError> {
        let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);

        // ── CUDA ────────────────────────────────────────────────────────────
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
                        state.as_map(),
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
                                    state.insert(name.clone(), tensor, pool);
                                }
                            }
                            self.decrement_refs_state(node, state, ref_counts, output_set);
                            return Ok(true);
                        }
                        Ok(None) => {
                            // Op not supported on CUDA — fall through to DirectML/CPU
                        }
                        Err(_e) => {
                            // CUDA dispatch error — fall through gracefully
                            #[cfg(debug_assertions)]
                            tracing::debug!(
                                op = %node.op.as_str(),
                                node = %node.name,
                                err = %_e,
                                "parallel: CUDA dispatch error, falling back",
                            );
                        }
                    }
                }
            }
        }

        // ── DirectML ────────────────────────────────────────────────────────
        #[cfg(feature = "directml")]
        {
            let try_dml = self.dml.is_some()
                && !matches!(
                    self.op_placement,
                    crate::execution_providers::OpPlacement::CpuOnly
                );
            if try_dml {
                if let Some(ctx) = &self.dml {
                    let dml_start = std::time::Instant::now();
                    match oxionnx_directml::try_directml_dispatch(
                        node,
                        &self.weights,
                        state.as_map(),
                        ctx,
                    ) {
                        Ok(Some(results)) => {
                            let dml_elapsed = dml_start.elapsed();
                            if let Some(ref profiling) = self.profiling_data {
                                if let Ok(mut data) = profiling.lock() {
                                    data.push(NodeProfile {
                                        node_name: node.name.clone(),
                                        op_type: node.op.as_str().to_string(),
                                        duration: dml_elapsed,
                                        output_shapes: results
                                            .iter()
                                            .map(|t| t.shape.clone())
                                            .collect(),
                                    });
                                }
                            }
                            for (name, tensor) in node.outputs.iter().zip(results) {
                                if !name.is_empty() {
                                    state.insert(name.clone(), tensor, pool);
                                }
                            }
                            self.decrement_refs_state(node, state, ref_counts, output_set);
                            return Ok(true);
                        }
                        Ok(None) => {
                            // Op not supported by DirectML — fall through to wgpu/CPU
                        }
                        Err(_e) => {
                            #[cfg(debug_assertions)]
                            tracing::debug!(
                                op = %node.op.as_str(),
                                node = %node.name,
                                err = %_e,
                                "parallel: DirectML dispatch error, falling back",
                            );
                        }
                    }
                }
            }
        }

        // ── wgpu GPU ────────────────────────────────────────────────────────
        #[cfg(feature = "gpu")]
        {
            use super::super::gpu_dispatch::{try_gpu_dispatch, GpuExecutionProvider};
            use crate::execution_providers::ProviderKind;

            let output_bytes =
                Self::estimate_output_bytes(node, state.as_map(), &self.weights, resolved);
            let placement = crate::execution_providers::decide_placement(
                &node.op,
                output_bytes,
                &self.op_placement,
            );
            if matches!(placement, ProviderKind::Gpu) {
                if let Some(gpu_ctx) = &self.gpu {
                    if let Some(results) =
                        try_gpu_dispatch(node, &self.weights, state.as_map(), gpu_ctx)?
                    {
                        for (name, tensor) in node.outputs.iter().zip(results) {
                            if !name.is_empty() {
                                state.insert(name.clone(), tensor, pool);
                            }
                        }
                        self.decrement_refs_state(node, state, ref_counts, output_set);
                        return Ok(true);
                    }
                    if GpuExecutionProvider::is_supported(node.op.as_str()) {
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            op = %node.op.as_str(),
                            node = %node.name,
                            "parallel: GPU fallback to CPU",
                        );
                    }
                }
            }
        }

        // Suppress unused-variable warnings when no GPU features are enabled.
        let _ = (node, state, ref_counts, output_set, resolved, pool);
        Ok(false)
    }

    /// Determine whether a node should be routed to hardware acceleration
    /// (CUDA, DirectML, or wgpu GPU) rather than rayon CPU parallelism.
    ///
    /// A node is GPU-eligible when:
    ///   - The `op_placement` is not `CpuOnly`, AND
    ///   - At least one hardware-acceleration backend context is available, AND
    ///   - The op is known to be GPU-capable (wgpu path) or the cuda/dml context
    ///     is present and willing to claim the op at dispatch time.
    ///
    /// GPU nodes within a depth level are serialised deliberately: GPU drivers
    /// queue work on-device and are not thread-safe to call concurrently from
    /// multiple rayon workers.
    #[cfg(not(target_arch = "wasm32"))]
    fn is_gpu_eligible_node(&self, node: &crate::graph::Node) -> bool {
        if matches!(
            self.op_placement,
            crate::execution_providers::OpPlacement::CpuOnly
        ) {
            return false;
        }

        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            return true;
        }

        #[cfg(feature = "directml")]
        if self.dml.is_some() {
            return true;
        }

        #[cfg(feature = "gpu")]
        if self.gpu.is_some() && crate::execution_providers::is_gpu_capable(&node.op) {
            return true;
        }

        // Suppress unused-variable warning when no GPU features are enabled.
        let _ = node;
        false
    }

    /// Parallel execution: group nodes by topological depth and execute each
    /// depth level using a hybrid strategy:
    ///
    /// - GPU-eligible nodes (CUDA / DirectML / wgpu) within a depth level are
    ///   executed **serially** in GPU-dispatch order.  This is intentional: GPU
    ///   driver contexts are not safe to call concurrently from multiple rayon
    ///   workers, and on-device queuing already provides hardware parallelism.
    /// - CPU-only nodes within the same depth level are executed **concurrently**
    ///   via rayon's `par_iter()`.
    ///
    /// Note: inplace and slot-write optimisations are active for single-node levels.
    /// For multi-node levels they are intentionally disabled — those paths require
    /// exclusive mutable access to state during the operator call, which serialises
    /// all workers and defeats the purpose of rayon parallelism.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn run_parallel_inner(
        &self,
        state: &mut SessionRunState,
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

        let resolved = self
            .resolved_shapes
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default();

        for group in &groups {
            if group.is_empty() {
                continue;
            }

            if group.len() == 1 {
                // Single node — try hardware acceleration first, then CPU path.
                let node = &self.sorted_nodes[group[0]];
                if let OpKind::Unknown(_) = &node.op {
                    continue;
                }

                // Try GPU/CUDA/DirectML dispatch; returns true if a backend claimed it.
                if self.try_accelerated_node(node, state, ref_counts, output_set, &resolved)? {
                    continue;
                }

                // No accelerator claimed the node — fall back to CPU dispatch_node
                // (inplace + slot-write optimisations active for single-node levels).
                let op_name = node.op.as_str();
                let operator = self.registry.get(op_name).ok_or_else(|| {
                    OnnxError::UnknownOp(format!("No operator registered for '{}'", op_name))
                })?;

                let elapsed =
                    self.dispatch_node(node, operator, state, ref_counts, output_set, &resolved)?;

                if let Some(ref profiling) = self.profiling_data {
                    if let Ok(mut data) = profiling.lock() {
                        // Collect output shapes from state after dispatch_node wrote them.
                        let output_shapes = node
                            .outputs
                            .iter()
                            .filter(|n| !n.is_empty())
                            .filter_map(|n| state.get(n))
                            .map(|t| t.shape.clone())
                            .collect();
                        data.push(NodeProfile {
                            node_name: node.name.clone(),
                            op_type: node.op.as_str().to_string(),
                            duration: elapsed,
                            output_shapes,
                        });
                    }
                }

                self.decrement_refs_state(node, state, ref_counts, output_set);
            } else {
                // Multiple nodes at this depth — hybrid dispatch:
                //
                //   Phase 1 (serial):   GPU-eligible nodes dispatched one-by-one through the
                //                       hardware acceleration hierarchy.  GPU contexts are not
                //                       thread-safe; serial dispatch is mandatory here.
                //
                //   Phase 2 (parallel): CPU-only nodes dispatched via rayon par_iter.
                //     Read sub-phase:   snapshot inputs (immutable borrow ends before write).
                //     Compute sub-phase: par_iter — no state access, full rayon parallelism.
                //     Write sub-phase:  sequential insert via state.insert (pool-backed).

                let nodes_at_depth: Vec<&crate::graph::Node> =
                    group.iter().map(|&i| &self.sorted_nodes[i]).collect();

                // ── Phase 1: serial GPU dispatch ─────────────────────────────
                let mut cpu_only_nodes: Vec<&crate::graph::Node> =
                    Vec::with_capacity(nodes_at_depth.len());

                for node in &nodes_at_depth {
                    if let OpKind::Unknown(_) = &node.op {
                        continue;
                    }
                    if self.is_gpu_eligible_node(node) {
                        // Try hardware acceleration; on miss fall through to CPU bucket.
                        if self
                            .try_accelerated_node(node, state, ref_counts, output_set, &resolved)?
                        {
                            continue;
                        }
                    }
                    // Either not GPU-eligible or no backend claimed it.
                    cpu_only_nodes.push(node);
                }

                // ── Phase 2: parallel CPU dispatch ───────────────────────────
                // Collect operators and pre-resolve inputs (read-only snapshot).
                let work_items: Vec<(&crate::graph::Node, &dyn Operator, Vec<Option<&Tensor>>)> =
                    cpu_only_nodes
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
                                        state.get(name).or_else(|| self.weights.get(name))
                                    }
                                })
                                .collect();
                            op.map(|o| (*n, o, inputs))
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                // Execute in parallel — each produces (node_name, results, duration).
                type ParResult<'a> = Result<(&'a str, Vec<Tensor>, std::time::Duration), OnnxError>;
                let par_execute = || -> Vec<ParResult<'_>> {
                    work_items
                        .par_iter()
                        .map(|(node, operator, inputs)| {
                            let ctx = OpContext {
                                node,
                                inputs: inputs.clone(),
                                outer_scope: None,
                                weights: Some(&self.weights),
                                registry: Some(&self.registry),
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

                // Write phase: insert all CPU outputs sequentially via state (pool-backed release).
                let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
                for result in par_results {
                    let (node_name, tensors, elapsed) = result?;
                    if let Some(node) = cpu_only_nodes.iter().find(|n| n.name == node_name) {
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
                                state.insert(name.clone(), tensor, pool);
                            }
                        }
                    }
                }

                // Decrement ref counts for all nodes in this group via state.
                // GPU nodes were already decremented inside try_accelerated_node;
                // CPU-only nodes are decremented here after the write phase.
                for node in &cpu_only_nodes {
                    self.decrement_refs_state(node, state, ref_counts, output_set);
                }
            }
        }
        Ok(())
    }

    /// Fallback on wasm32: parallel is not supported, delegate to sequential.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn run_parallel_inner(
        &self,
        state: &mut SessionRunState,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) -> Result<(), OnnxError> {
        self.run_sequential_inner(state, ref_counts, output_set)
    }
}
