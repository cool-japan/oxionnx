use crate::graph::OpKind;
use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{OpContext, Operator};
use std::collections::HashMap;
use std::sync::Mutex;

use super::super::types::NodeProfile;
use super::super::Session;
use super::state::SessionRunState;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

impl Session {
    /// Parallel execution: group nodes by topological depth and execute each
    /// depth level concurrently using rayon.
    ///
    /// TODO(v0.1.7): lift GPU/CUDA/DirectML dispatch into this parallel path.
    /// Currently those dispatchers only run in run_sequential_inner (non-parallel path).
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

        for group in &groups {
            if group.is_empty() {
                continue;
            }

            if group.len() == 1 {
                // Single node — execute sequentially via dispatch_node (inplace + slot-write).
                let node = &self.sorted_nodes[group[0]];
                if let OpKind::Unknown(_) = &node.op {
                    continue;
                }
                let op_name = node.op.as_str();
                let operator = self.registry.get(op_name).ok_or_else(|| {
                    OnnxError::UnknownOp(format!("No operator registered for '{}'", op_name))
                })?;

                let resolved = self
                    .resolved_shapes
                    .lock()
                    .map(|s| s.clone())
                    .unwrap_or_default();

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
                // Multiple nodes at this depth — execute in parallel via rayon.
                //
                // Read phase: snapshot inputs from state (immutable borrow ends before write phase).
                // Compute phase: par_iter — zero state access, full rayon parallelism.
                // Write phase: sequential insert via state.insert (pool-backed buffer release).
                let nodes_at_depth: Vec<&crate::graph::Node> =
                    group.iter().map(|&i| &self.sorted_nodes[i]).collect();

                // Collect operators and pre-resolve inputs (read-only snapshot).
                let work_items: Vec<(&crate::graph::Node, &dyn Operator, Vec<Option<&Tensor>>)> =
                    nodes_at_depth
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

                // Write phase: insert all outputs sequentially via state (pool-backed release).
                let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
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
                                state.insert(name.clone(), tensor, pool);
                            }
                        }
                    }
                }

                // Decrement ref counts for all nodes in this group via state.
                for node in &nodes_at_depth {
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
