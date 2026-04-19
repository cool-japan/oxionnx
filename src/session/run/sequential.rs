use crate::graph::OpKind;
use crate::memory::SizeClassPool;
use crate::OnnxError;
use std::collections::HashMap;
use std::sync::Mutex;

use super::super::types::NodeProfile;
use super::super::Session;
use super::state::SessionRunState;

impl Session {
    // GREP_GUARD: all intermediates writes must go through dispatch_node / SessionRunState::insert

    /// Sequential execution path using `SessionRunState` for buffer-reuse-aware
    /// intermediate storage.
    pub(crate) fn run_sequential_inner(
        &self,
        state: &mut SessionRunState,
        ref_counts: &mut HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
    ) -> Result<(), OnnxError> {
        let resolved = self
            .resolved_shapes
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default();

        for node in &self.sorted_nodes {
            if let OpKind::Unknown(_) = &node.op {
                continue;
            }

            // Determine operator placement based on the configured strategy.
            // output_bytes and placement are used by the GPU dispatch block below.
            // CUDA and DirectML dispatch check op_placement directly (no size threshold).
            #[cfg(feature = "gpu")]
            let output_bytes =
                Self::estimate_output_bytes(node, state.as_map(), &self.weights, &resolved);
            #[cfg(feature = "gpu")]
            let placement = crate::execution_providers::decide_placement(
                &node.op,
                output_bytes,
                &self.op_placement,
            );
            // When no hardware-acceleration feature is active, read op_placement to
            // satisfy the compiler (field is always valid, just unused at runtime).
            #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "directml")))]
            let _ = &self.op_placement;

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
                                let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
                                for (name, tensor) in node.outputs.iter().zip(results) {
                                    if !name.is_empty() {
                                        state.insert(name.clone(), tensor, pool);
                                    }
                                }
                                self.decrement_refs_state(node, state, ref_counts, output_set);
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

            // DirectML dispatch — Windows D3D12 GPU, higher priority than wgpu on Windows
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
                                let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
                                for (name, tensor) in node.outputs.iter().zip(results) {
                                    if !name.is_empty() {
                                        state.insert(name.clone(), tensor, pool);
                                    }
                                }
                                self.decrement_refs_state(node, state, ref_counts, output_set);
                                continue;
                            }
                            Ok(None) => {
                                // Op not supported by DirectML — fall through to wgpu/CPU
                            }
                            Err(_e) => {
                                // DirectML dispatch error — fall back silently
                                #[cfg(debug_assertions)]
                                tracing::debug!(
                                    op = %node.op.as_str(),
                                    node = %node.name,
                                    err = %_e,
                                    "DirectML dispatch error, falling back",
                                );
                            }
                        }
                    }
                }
            }

            // GPU dispatch (only when placement routes to GPU)
            #[cfg(feature = "gpu")]
            {
                use super::super::gpu_dispatch::{try_gpu_dispatch, GpuExecutionProvider};
                use crate::execution_providers::ProviderKind;
                let try_gpu = matches!(placement, ProviderKind::Gpu);
                if try_gpu {
                    if let Some(gpu_ctx) = &self.gpu {
                        if let Some(results) =
                            try_gpu_dispatch(node, &self.weights, state.as_map(), gpu_ctx)?
                        {
                            let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
                            for (name, tensor) in node.outputs.iter().zip(results) {
                                if !name.is_empty() {
                                    state.insert(name.clone(), tensor, pool);
                                }
                            }
                            self.decrement_refs_state(node, state, ref_counts, output_set);
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
            if self.mixed_precision && super::super::mixed_precision::should_use_f16(op_name) {
                let input_refs: Vec<&crate::tensor::Tensor> = node
                    .inputs
                    .iter()
                    .filter_map(|name| {
                        if name.is_empty() {
                            None
                        } else {
                            state.get(name).or_else(|| self.weights.get(name))
                        }
                    })
                    .collect();

                let start = std::time::Instant::now();
                if let Some(f16_result) =
                    super::super::mixed_precision::execute_elementwise_f16(op_name, &input_refs)
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

                    let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
                    for (name, tensor) in node.outputs.iter().zip(results) {
                        if !name.is_empty() {
                            state.insert(name.clone(), tensor, pool);
                        }
                    }
                    self.decrement_refs_state(node, state, ref_counts, output_set);
                    continue;
                }
                // No native f16 path — fall through to normal execution with f16 rounding
            }

            let operator = self.registry.get(op_name).ok_or_else(|| {
                OnnxError::UnknownOp(format!("No operator registered for '{}'", op_name))
            })?;

            let elapsed =
                self.dispatch_node(node, operator, state, ref_counts, output_set, &resolved)?;

            // Mixed precision: round outputs to f16 for f16-safe ops without native f16 path.
            // This simulates f16 storage precision for ops that ran in f32.
            if self.mixed_precision && super::super::mixed_precision::should_use_f16(op_name) {
                let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
                for out_name in &node.outputs {
                    if out_name.is_empty() {
                        continue;
                    }
                    if let Some(t) = state.take(out_name) {
                        let rounded = super::super::mixed_precision::round_to_f16_precision(&t);
                        state.insert(out_name.clone(), rounded, pool);
                    }
                }
            }

            if let Some(ref profiling) = self.profiling_data {
                if let Ok(mut data) = profiling.lock() {
                    // Gather output shapes for profiling
                    let output_shapes: Vec<Vec<usize>> = node
                        .outputs
                        .iter()
                        .filter(|n| !n.is_empty())
                        .filter_map(|n| state.get(n).map(|t| t.shape.clone()))
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
        }
        Ok(())
    }
}
