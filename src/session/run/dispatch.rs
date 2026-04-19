use crate::graph::{Node, OpKind};
use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{OpContext, Operator};
use std::collections::HashMap;
use std::sync::Mutex;

use super::super::Session;
use super::state::SessionRunState;

impl Session {
    /// CPU-path dispatch for a single node: implements the operator dispatch
    /// precedence (inplace → slot-write → execute) and writes results into
    /// `SessionRunState`.
    ///
    /// Returns the execution duration for profiling.
    pub(crate) fn dispatch_node(
        &self,
        node: &Node,
        operator: &dyn Operator,
        state: &mut SessionRunState,
        ref_counts: &HashMap<String, usize>,
        output_set: &std::collections::HashSet<&str>,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<std::time::Duration, OnnxError> {
        let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);

        // 1. Inplace path: first input has refcount 1, op supports inplace, not a model output.
        let can_inplace = operator.supports_inplace()
            && !node.inputs.is_empty()
            && !node.inputs[0].is_empty()
            && !self.weights.contains_key(&node.inputs[0])
            && !output_set.contains(node.inputs[0].as_str())
            && ref_counts.get(&node.inputs[0]).copied().unwrap_or(0) == 1;

        // 2. Slot-write path: op supports output slots and all output shapes are known.
        let can_slot = !can_inplace && operator.supports_output_slots();

        if can_slot {
            let maybe_slots: Option<Vec<Tensor>> = {
                let mut slots = Vec::with_capacity(node.outputs.len());
                let mut all_known = true;
                for out_name in &node.outputs {
                    if out_name.is_empty() {
                        slots.push(Tensor::new(vec![], vec![]));
                        continue;
                    }
                    if let Some(shape) = resolved_shapes.get(out_name) {
                        let size: usize = if shape.is_empty() {
                            1
                        } else {
                            shape.iter().product()
                        };
                        let data = if let Some(pool_mutex) = pool {
                            if let Ok(mut guard) = pool_mutex.lock() {
                                guard.acquire(size)
                            } else {
                                vec![0.0f32; size]
                            }
                        } else {
                            vec![0.0f32; size]
                        };
                        slots.push(Tensor::new(data, shape.clone()));
                    } else {
                        all_known = false;
                        break;
                    }
                }
                if all_known {
                    Some(slots)
                } else {
                    // Release already-acquired slot buffers back to the pool
                    if let Some(pool_mutex) = pool {
                        if let Ok(mut guard) = pool_mutex.lock() {
                            for slot in slots {
                                if !slot.data.is_empty() {
                                    guard.release(slot.data);
                                }
                            }
                        }
                    }
                    None
                }
            };

            if let Some(mut slots) = maybe_slots {
                let resolved_inputs: Vec<Option<&Tensor>> = node
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
                let ctx = OpContext {
                    node,
                    inputs: resolved_inputs,
                    outer_scope: None,
                    registry: None,
                };
                let start = std::time::Instant::now();
                operator.execute_into_slots(&ctx, &mut slots)?;
                let elapsed = start.elapsed();
                for (out_name, tensor) in node.outputs.iter().zip(slots) {
                    if !out_name.is_empty() {
                        state.insert(out_name.clone(), tensor, pool);
                    }
                }
                return Ok(elapsed);
            }
            // Fall through to normal path if not all shapes known
        }

        let start = std::time::Instant::now();

        let results = if can_inplace {
            // Take ownership of the first input for in-place mutation
            let owned_input = state.take(&node.inputs[0]);
            let resolved_inputs: Vec<Option<&Tensor>> = node
                .inputs
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    if name.is_empty() || i == 0 {
                        None
                    } else {
                        state.get(name).or_else(|| self.weights.get(name))
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
            // 3. Default path: standard execute.
            let resolved_inputs: Vec<Option<&Tensor>> = node
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
            let ctx = OpContext {
                node,
                inputs: resolved_inputs,
                outer_scope: None,
                registry: None,
            };
            operator.execute(&ctx)?
        };

        let elapsed = start.elapsed();
        for (out_name, tensor) in node.outputs.iter().zip(results) {
            if !out_name.is_empty() {
                state.insert(out_name.clone(), tensor, pool);
            }
        }
        Ok(elapsed)
    }

    /// Estimate the output tensor size in bytes for a node, using resolved
    /// shapes when available or falling back to input tensor sizes.
    #[cfg(feature = "gpu")]
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
}

// Silence unused import warning for OpKind when no GPU features active
#[allow(unused_imports)]
use self::OpKind as _;
