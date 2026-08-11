//! Asynchronous sequential execution path — the browser's run loop.
//!
//! # Why a second loop exists
//!
//! `run_sequential_inner` (see `super::sequential`) is synchronous top to
//! bottom, and it has to stay that way: it is the native engine's hot path, it
//! is shared with the rayon-parallel path, and none of the native execution
//! providers (CUDA, DirectML, CoreML) has an asynchronous interface to begin
//! with. WebGPU has *only* an asynchronous one — a browser thread may not block
//! on a GPU fence, so `Device::poll(Wait)` is a no-op there and every read-back
//! must be awaited.
//!
//! Rather than make the whole engine `async` to serve one backend on one
//! target, this module adds a second, deliberately smaller loop that runs the
//! same nodes in the same order and differs in exactly one place: it *awaits*
//! the wgpu dispatcher instead of blocking on it. Everything else — shape
//! resolution, reference counting, the in-place / slot-write fast paths, the
//! mixed-precision policy, the CPU operator dispatch — is the same code the
//! synchronous loop calls, reached through the same `pub(crate)` helpers.
//!
//! # What it deliberately does not do
//!
//! * **No parallel variant.** `wasm32-unknown-unknown` has no threads on any
//!   path this crate compiles, and the ordering contract below forbids
//!   overlapping GPU nodes anyway.
//! * **No CUDA / DirectML / CoreML arms.** None of them exists in a browser,
//!   and on native the synchronous loop already covers them.
//!
//! # Ordering contract
//!
//! Exactly one GPU node is in flight at a time. This is not a simplification:
//! wgpu error scopes are a per-thread LIFO stack whose native backend *panics*
//! if scopes are popped out of order, and `oxionnx-gpu` pushes one per dispatch.
//! A future "run two nodes concurrently" optimization has to give each
//! concurrent stream its own device, not just its own `await`.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::execution_providers::{
    decide_placement, provider_supports_op, OpPlacement, ProviderKind,
};
use crate::graph::{Node, OpKind};
use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;

use super::super::gpu_activations::RunActivations;
use super::super::gpu_dispatch::{op_accepts_resident_slot, DispatchOutcome};
use super::super::types::NodeProfile;
use super::super::Session;
use super::sequential::mixed_precision_claims_node;
use super::state::SessionRunState;
use super::{OutputSet, RefCounts};

/// May the wgpu backend be offered this node on the legacy heuristic path?
///
/// The asynchronous counterpart of `super::sequential`'s `accelerator_gate`,
/// narrowed to its only live candidate here. It defers to the same single
/// source of truth — [`decide_placement`], which applies `OpPlacement::CpuOnly`,
/// `Auto`'s `gpu_threshold_bytes`, the `Manual` pin, the
/// `MIN_GPU_DISPATCH_BYTES` floor and each backend's op-support predicate — so
/// a session cannot get a different placement decision merely by calling
/// [`Session::run_gpu_async`] instead of [`Session::run`].
///
/// The `Auto` fall-through is preserved for the same reason the synchronous
/// gate preserves it: `decide_placement` names the *highest-priority* provider
/// that supports the op, and wgpu ranks last, so when a higher-priority
/// accelerator is named but is not present in this loop, wgpu is still a legal
/// target under `Auto`. Under `Manual` it is not: the user pinned that op to
/// that provider, and quietly rerouting it would be the `{Conv: Gpu}` bug the
/// synchronous gate documents.
fn gpu_accelerator_gate(op: &OpKind, output_bytes: usize, placement: &OpPlacement) -> bool {
    let chosen = decide_placement(op, output_bytes, placement);
    if matches!(chosen, ProviderKind::Cpu) {
        return false;
    }
    if chosen == ProviderKind::Gpu {
        return true;
    }
    matches!(placement, OpPlacement::Auto { .. }) && provider_supports_op(ProviderKind::Gpu, op)
}

/// Does the session's explicit provider list reach wgpu for this node?
///
/// Walks `Session::providers` in order and stops at the `Cpu` sentinel exactly
/// as `try_provider_list_dispatch` does, so anything listed after `Cpu` stays
/// unreachable by construction. Providers that cannot run in this loop (CUDA,
/// DirectML) are skipped rather than treated as terminal — they would have
/// declined the node and fallen through anyway.
fn provider_list_reaches_gpu(providers: &[ProviderKind]) -> bool {
    // The first entry that *decides* the walk: `Cpu` ends it, `Gpu` wins it.
    // Anything else is a provider this loop cannot run, which is indistinguish-
    // able from one that declined the node, so the walk passes over it.
    providers
        .iter()
        .find(|provider| matches!(provider, ProviderKind::Cpu | ProviderKind::Gpu))
        .is_some_and(|provider| matches!(provider, ProviderKind::Gpu))
}

impl Session {
    /// Run inference, awaiting GPU work instead of blocking on it.
    ///
    /// The asynchronous twin of [`Session::run`]. Same inputs, same outputs,
    /// same numerics — the difference is that a node the wgpu backend accepts
    /// is `await`ed, so the calling task yields to the browser's event loop
    /// while the GPU works instead of deadlocking the page on a fence it can
    /// never observe.
    ///
    /// # Browser use
    ///
    /// Two steps, in this order:
    ///
    /// ```no_run
    /// # async fn demo(session: &mut oxionnx::Session) -> Result<(), oxionnx::OnnxError> {
    /// # let inputs = std::collections::HashMap::new();
    /// session.enable_gpu_async().await;      // acquire a WebGPU device
    /// let outputs = session.run_gpu_async(&inputs).await?;
    /// # let _ = outputs;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Skipping the first step is not an error: the session simply has no
    /// device and every node runs on the CPU.
    ///
    /// # Native use
    ///
    /// Supported, and it produces identical results to [`Session::run`] for any
    /// session without `with_parallel_execution(true)`. Note that on native the
    /// awaited GPU work is *itself* blocking (see `oxionnx_gpu`'s crate docs),
    /// so this must not be spawned onto a shared async runtime's worker there.
    /// It exists on native so the async path can be tested against a real
    /// adapter, which no `wasm32` unit test can do.
    ///
    /// # Errors
    ///
    /// Propagates input-validation, shape-resolution and operator errors
    /// exactly as [`Session::run`] does. A GPU node that declines is not an
    /// error — it runs on the CPU.
    pub async fn run_gpu_async(
        &self,
        inputs: &HashMap<&str, Tensor>,
    ) -> Result<HashMap<String, Tensor>, OnnxError> {
        let input_refs: HashMap<&str, &Tensor> = inputs.iter().map(|(k, v)| (*k, v)).collect();
        self.run_gpu_async_internal(&input_refs).await
    }

    /// Borrowed-input core of [`Session::run_gpu_async`].
    ///
    /// Mirrors `run::entry`'s `run_internal` preamble and epilogue exactly —
    /// input validation, this run's resolved shapes, a fresh reference-count
    /// map borrowed from the static run plan, the seeded run state, and the
    /// validated output take. Only the node loop differs.
    async fn run_gpu_async_internal(
        &self,
        inputs: &HashMap<&str, &Tensor>,
    ) -> Result<HashMap<String, Tensor>, OnnxError> {
        if !self.input_infos.is_empty() {
            Self::validate_input_shapes(&self.input_infos, inputs)?;
        }
        let resolved_shapes = self.resolve_run_shapes(inputs)?;

        let output_set: OutputSet<'_> = self.output_names.iter().map(|s| s.as_str()).collect();
        let base = &self.run_plan.base_ref_counts;
        let mut ref_counts: RefCounts<'_> = RefCounts::with_capacity(base.len());
        ref_counts.extend(base.iter().map(|(name, count)| (name.as_str(), *count)));

        let mut state = SessionRunState::with_capacity(self.sorted_nodes.len());
        for (name, tensor) in inputs {
            state.insert(
                name.to_string(),
                (*tensor).clone(),
                self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>),
            );
        }

        self.run_sequential_async_inner(&mut state, &mut ref_counts, &output_set, &resolved_shapes)
            .await?;

        let pool_ref = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
        state.take_outputs(&self.output_names, &self.weights, pool_ref)
    }

    /// The node loop itself.
    ///
    /// Precedence is the same as the synchronous loop's, minus the providers
    /// that do not exist here:
    ///
    /// 1. mixed precision claims the node outright (no provider is offered it),
    /// 2. the explicit provider list, when the session has one,
    /// 3. the legacy `decide_placement` heuristic,
    /// 4. the CPU — always the terminal fallback.
    async fn run_sequential_async_inner(
        &self,
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Result<(), OnnxError> {
        self.bind_registry_opset();
        // [r3a] Statistics for this run start clean. See
        // `session::gpu_residency` for why these live in a thread-local
        // rather than on `Session` (this loop is single-threaded and
        // one-node-at-a-time by the ordering contract above, so "this thread"
        // and "this run" are the same scope).
        crate::session::gpu_residency::reset_run_stats();

        // Which of this graph's values may live in a device buffer between
        // nodes, and which node is the last one that will read each of them.
        // Both are properties of the node order, which is fixed, so both are
        // decided once here rather than guessed at per node. A session with no
        // device — or one whose context has residency switched off — gets the
        // empty plan, and every path below then behaves exactly as it did
        // before activations could stay resident.
        let residency_enabled = self
            .gpu
            .as_ref()
            .is_some_and(|ctx| ctx.activation_residency_enabled());
        let mut activations = RunActivations::new(
            residency_enabled,
            &self.sorted_nodes,
            &self.output_names,
            |node, slot| op_accepts_resident_slot(&node.op, slot),
        );

        for (index, node) in self.sorted_nodes.iter().enumerate() {
            let op_name = node.op.as_str();
            let mixed_precision_node = mixed_precision_claims_node(self.mixed_precision, op_name);
            let provider_list_in_use = !self.providers.is_empty();

            // Only compute the payload size when a rule could actually consult
            // it — it walks the node's outputs and the resolved shape map.
            let gpu_eligible = self.gpu.is_some() && !mixed_precision_node;
            let offer_to_gpu = gpu_eligible && {
                let output_bytes = self.async_output_bytes(node, state, &activations, resolved);
                if provider_list_in_use {
                    provider_list_reaches_gpu(&self.providers)
                        && Self::provider_list_clears_dispatch_floor(output_bytes)
                } else {
                    gpu_accelerator_gate(&node.op, output_bytes, &self.op_placement)
                }
            };

            if offer_to_gpu {
                self.promote_operands_async(node, state, &mut activations)
                    .await;
                if self
                    .dispatch_to_wgpu_async(node, state, &mut activations, resolved)
                    .await?
                {
                    self.decrement_refs_state(node, state, ref_counts, output_set);
                    activations.release_after(index);
                    continue;
                }
            }

            // Everything below this line runs on the host, so anything this
            // node reads has to be there. This is the *single* convergence
            // point for that: the mixed-precision arm, the CPU operator and the
            // unsupported-op error all pass through it, so a resident operand
            // cannot reach any of them. One read-back per tensor per run — the
            // host copy is memoized into the run state, and later GPU consumers
            // still bind the device copy in place.
            self.materialize_resident_inputs(node, state, &activations)
                .await?;

            if mixed_precision_node
                && self.try_native_f16_node(node, state, ref_counts, output_set, resolved)?
            {
                activations.release_after(index);
                continue;
            }

            let operator = self
                .registry
                .get(op_name)
                .ok_or_else(|| super::unsupported_op_error(node))?;

            let elapsed =
                self.dispatch_node(node, operator, state, ref_counts, output_set, resolved)?;
            crate::session::gpu_residency::note_cpu_node(op_name, elapsed);

            if mixed_precision_node {
                self.round_node_outputs_to_f16(node, state);
            }

            if let Some(ref profiling) = self.profiling_data {
                if let Ok(mut data) = profiling.lock() {
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
            activations.release_after(index);
        }
        crate::session::gpu_residency::note_activation_peak(activations.peak_bytes());
        // Nothing should be left: every name in the plan has a last consumer,
        // and every node released after itself. Dropping the map destroys
        // whatever a future edit does leave behind rather than carrying it into
        // the next frame, which is what makes the live-byte assertion in
        // `tests::gpu_activation_residency` a statement about the mechanism and
        // not about this loop remembering to clean up.
        drop(activations);
        Ok(())
    }

    /// The payload size the placement gates consult, with device-resident
    /// operands visible to it.
    ///
    /// `estimate_output_bytes` prefers the resolved output shape and falls back
    /// to the first input tensor it can find in the run state. A node whose only
    /// input stayed on the device is absent from that map, so without this the
    /// fallback would answer `0` — below `MIN_GPU_DISPATCH_BYTES` — and close
    /// the gate on precisely the nodes residency exists to keep open. The
    /// substitution is applied only when the estimate is `0`, so a graph with no
    /// resident values gets the identical number it always did.
    fn async_output_bytes(
        &self,
        node: &Node,
        state: &SessionRunState,
        activations: &RunActivations,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> usize {
        let estimate = Self::estimate_output_bytes(node, state.as_map(), &self.weights, resolved);
        if estimate != 0 {
            return estimate;
        }
        node.inputs
            .iter()
            .filter_map(|name| activations.get(name))
            .map(|tensor| tensor.len().saturating_mul(std::mem::size_of::<f32>()))
            .max()
            .unwrap_or(0)
    }

    /// Upload the small host operands of a node whose other operands are
    /// already on the device.
    ///
    /// # Why a node with one host operand is worth an upload
    ///
    /// `node_residency_tier` answers `Resident` only when *every* operand is on
    /// the device, and that strictness is load-bearing — a node with one
    /// transferring activation pays the transferring cost model whatever else it
    /// has. But InSwapper's AdaIN pairs are exactly `Mul([1,C,H,W], [1,C,1,1])`
    /// with the small operand produced by a `Gemm` head that declines to the
    /// CPU, so the *large* operand is resident and the *small* one is not, and
    /// the node declines on `MEMORY_BOUND_TRANSFER_FLOOR` — leaving the big
    /// activation to be read back. The round trip this wave exists to delete,
    /// still there.
    ///
    /// Uploading the small operand makes the tier claim true rather than
    /// redefining it. The trade is arithmetic rather than a tuned constant. With
    /// `n` resident elements and `m` host ones:
    ///
    /// * dispatching costs `4m` bytes up, and the result stays resident;
    /// * declining costs `4n` bytes down, plus `4·out >= 4n` back up when the
    ///   next GPU node consumes the result — at least `8n`.
    ///
    /// So the upload wins whenever `m < 2n`, and the rule below is the
    /// conservative half of that: upload when the host operands together are no
    /// larger than the resident ones. Those bytes really do cross the bus and
    /// are counted as `GpuRunStats::activation_upload_bytes`, not hidden.
    ///
    /// Only ops that decline while transferring and dispatch while resident are
    /// candidates — for anything else the upload would buy nothing.
    async fn promote_operands_async(
        &self,
        node: &Node,
        state: &SessionRunState,
        activations: &mut RunActivations,
    ) {
        use crate::session::gpu_residency::{
            gpu_min_transfer_elements, ResidencyTier, MEMORY_BOUND_TRANSFER_FLOOR,
        };
        if !activations.is_enabled() {
            return;
        }
        let Some(gpu_ctx) = &self.gpu else {
            return;
        };
        // A degraded context declines every dispatch, so an upload here would
        // buy nothing and still cross the bus: the node it was meant to make
        // dispatchable runs on the CPU regardless, and the promoted operand
        // would only have to be read back again.
        if gpu_ctx.is_degraded() {
            return;
        }
        let blocked_while_transferring =
            gpu_min_transfer_elements(&node.op, ResidencyTier::Transferred)
                == MEMORY_BOUND_TRANSFER_FLOOR;
        if !blocked_while_transferring
            || gpu_min_transfer_elements(&node.op, ResidencyTier::Resident).is_none()
        {
            return;
        }

        let mut resident_elements = 0usize;
        let mut candidates: Vec<(&str, usize)> = Vec::new();
        for (slot, name) in node.inputs.iter().enumerate() {
            if name.is_empty() || !op_accepts_resident_slot(&node.op, slot) {
                continue;
            }
            if let Some(tensor) = activations.get(name) {
                resident_elements = resident_elements.saturating_add(tensor.len());
                continue;
            }
            let Some(tensor) = state.get(name).or_else(|| self.weights.get(name)) else {
                continue;
            };
            candidates.push((name.as_str(), tensor.data.len()));
        }
        if resident_elements == 0 || candidates.is_empty() {
            return;
        }
        let host_elements = candidates
            .iter()
            .fold(0usize, |acc, (_, len)| acc.saturating_add(*len));
        if host_elements > resident_elements {
            return;
        }

        let names: Vec<String> = candidates
            .into_iter()
            .map(|(name, _)| name.to_string())
            .collect();
        for name in names {
            let Some(tensor) = state.get(&name).or_else(|| self.weights.get(&name)) else {
                continue;
            };
            // A decline here is not an error: the node simply stays in the
            // transferring tier and takes whatever path it would have taken.
            if let Some(device) =
                gpu_ctx.upload_device_tensor("promoted_operand", &tensor.data, &tensor.shape)
            {
                crate::session::gpu_residency::note_activation_upload(tensor.data.len());
                activations.insert_promoted(&name, device);
            }
        }
    }

    /// Read back every operand of `node` that exists only on the device, once.
    ///
    /// Called immediately before any host-side execution of the node. The host
    /// tensor is memoized into the run state, so a second consumer of the same
    /// value finds it there and no second read-back happens; the device copy is
    /// deliberately kept, so a *later* GPU consumer still binds it in place.
    ///
    /// # Errors
    ///
    /// [`OnnxError::Internal`] when the read-back declines. That is a device
    /// error or an exhausted byte budget, and unlike every other decline in this
    /// engine it has no fallback: the only copy of the value is in a buffer that
    /// cannot be read, so the run cannot produce a correct result and must say
    /// so rather than continue with a missing tensor.
    async fn materialize_resident_inputs(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        activations: &RunActivations,
    ) -> Result<(), OnnxError> {
        if !activations.is_enabled() {
            return Ok(());
        }
        let Some(gpu_ctx) = &self.gpu else {
            return Ok(());
        };
        for name in &node.inputs {
            if name.is_empty() || state.get(name).is_some() {
                continue;
            }
            let Some(device) = activations.get(name) else {
                continue;
            };
            // "Exists only on the device" is the condition, and a *promoted*
            // operand never satisfies it: it was uploaded from the weight map,
            // which still holds those bytes, so reading it back would move
            // bytes the host already has. Only a node output needs this — and
            // the distinction has to be `holds_node_output` rather than "is it
            // in `weights`", because a model may legally name a node output
            // after an initializer, and then the initializer's bytes are
            // precisely the wrong ones to hand the operator.
            if !activations.holds_node_output(name) {
                continue;
            }
            let tensor = oxionnx_gpu::read_device_tensor_async(gpu_ctx, device)
                .await
                .ok_or_else(|| {
                    OnnxError::Internal(format!(
                        "reading device-resident tensor '{}' back for node '{}' ({}) failed; \
                         the value exists only on the device and the node cannot \
                         run without it",
                        name,
                        node.name,
                        node.op.as_str(),
                    ))
                })?;
            crate::session::gpu_residency::note_activation_readback(tensor.data.len());
            state.insert(
                name.clone(),
                tensor,
                self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>),
            );
        }
        Ok(())
    }

    /// Offer `node` to the wgpu backend, awaiting its read-back.
    ///
    /// Returns `Ok(true)` when the backend produced results and they are
    /// committed to `state`; `Ok(false)` when it declined and the caller must
    /// run the CPU operator. An `Err` is propagated rather than swallowed into
    /// a CPU fallback, matching the synchronous `dispatch_to_wgpu`'s
    /// deliberately stricter contract.
    async fn dispatch_to_wgpu_async(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        activations: &mut RunActivations,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<bool, OnnxError> {
        let Some(gpu_ctx) = &self.gpu else {
            return Ok(false);
        };

        let started = crate::time_compat::Instant::now();
        let dispatched = super::super::gpu_dispatch::try_gpu_dispatch_async(
            node,
            &self.weights,
            state.as_map(),
            activations,
            gpu_ctx,
        )
        .await?;

        let Some(outcome) = dispatched else {
            tracing::debug!(
                provider = "wgpu",
                op = %node.op.as_str(),
                node = %node.name,
                "execution provider declined the node; falling back",
            );
            return Ok(false);
        };

        let elapsed = started.elapsed();
        crate::session::gpu_residency::note_gpu_node(node.op.as_str(), elapsed);

        match outcome {
            DispatchOutcome::Host(results) => {
                // A node that read its result back moved those bytes; counting
                // it here rather than inside each kernel keeps the count honest
                // about what the *session* observes.
                for tensor in &results {
                    crate::session::gpu_residency::note_readback(tensor.data.len());
                }
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
                self.write_node_outputs(node, "wgpu", results, state, resolved_shapes)
                    .map(|()| true)
            }
            DispatchOutcome::Device(tensor) => {
                let name = node.outputs.first().ok_or_else(|| {
                    OnnxError::Internal(format!(
                        "wgpu provider kept the result of node '{}' ({}) on the \
                         device, but the node declares no output to store it under",
                        node.name,
                        node.op.as_str(),
                    ))
                })?;
                // The shape check `write_node_outputs` performs for host
                // results, kept for device ones: a kernel that computed the
                // wrong extent would otherwise poison every later node with no
                // diagnostic, and the buffer is even harder to inspect than a
                // host tensor.
                if let Some(expected) = resolved_shapes.get(name) {
                    if tensor.shape() != expected.as_slice() {
                        return Err(OnnxError::ShapeMismatch(format!(
                            "wgpu provider returned device-resident output '{}' of \
                             node '{}' ({}) with shape {:?}, but shape inference \
                             resolved {:?}",
                            name,
                            node.name,
                            node.op.as_str(),
                            tensor.shape(),
                            expected,
                        )));
                    }
                }
                if let Some(ref profiling) = self.profiling_data {
                    if let Ok(mut data) = profiling.lock() {
                        data.push(NodeProfile {
                            node_name: node.name.clone(),
                            op_type: node.op.as_str().to_string(),
                            duration: elapsed,
                            output_shapes: vec![tensor.shape().to_vec()],
                        });
                    }
                }
                activations.insert_output(name, tensor);
                Ok(true)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Graph};

    /// A graph big enough that the wgpu backend is actually *offered* every
    /// node — `Relu` and `Add` clear `EW_GPU_THRESHOLD` at 128×1024 elements —
    /// so the async loop's dispatch arm is exercised rather than skipped.
    ///
    /// Whether the adapter then accepts or declines is not the point: both
    /// outcomes must land on identical values, and on a machine with no adapter
    /// at all this still checks that the async loop's CPU path matches the
    /// synchronous one node for node.
    fn relu_add_graph() -> (Graph, HashMap<String, Tensor>) {
        let relu = Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["h".to_string()],
            attrs: Attributes::default(),
        };
        let add = Node {
            op: OpKind::Add,
            name: "add".to_string(),
            inputs: vec!["h".to_string(), "bias".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![relu, add],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            ..Default::default()
        };
        let mut weights = HashMap::new();
        weights.insert(
            "bias".to_string(),
            Tensor::new(vec![0.25f32; 128 * 1024], vec![128, 1024]),
        );
        (graph, weights)
    }

    /// The whole contract of this module in one assertion: `run_gpu_async` is
    /// [`Session::run`] with awaits, not a second implementation with its own
    /// numerics. Runs natively so a real adapter participates when one exists.
    #[test]
    fn the_async_loop_agrees_with_the_synchronous_one() {
        let (graph, weights) = relu_add_graph();
        let mut session = Session::from_graph(graph, weights).expect("from_graph should succeed");
        // Attach a device when the machine has one; a `false` here just means
        // the comparison runs CPU-vs-CPU, which is still a real check.
        let _has_gpu = pollster::block_on(session.enable_gpu_async());
        // The default placement is `CpuOnly`, which would make both loops take
        // the CPU arm and prove nothing. `Auto` at the documented 64 KiB
        // threshold is what actually offers both nodes to wgpu.
        session.op_placement = OpPlacement::Auto {
            gpu_threshold_bytes: 65_536,
        };

        let x: Vec<f32> = (0..128 * 1024)
            .map(|i| (i as f32).mul_add(0.001, -60.0))
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert("x", Tensor::new(x, vec![128, 1024]));

        let sync_out = session.run(&inputs).expect("sync run");
        let async_out = pollster::block_on(session.run_gpu_async(&inputs)).expect("async run");

        let sync_y = sync_out.get("y").expect("sync output");
        let async_y = async_out.get("y").expect("async output");
        assert_eq!(sync_y.shape, async_y.shape);
        assert_eq!(
            sync_y.data.len(),
            async_y.data.len(),
            "both loops must produce the same element count"
        );
        for (i, (s, a)) in sync_y.data.iter().zip(async_y.data.iter()).enumerate() {
            assert!(
                (s - a).abs() <= 1e-5,
                "async loop diverged at {i}: sync={s} async={a}"
            );
        }

        // Awaiting the error-scope pop must not have found anything: a browser
        // build has no other way to learn that a dispatch failed validation, so
        // a silently degraded context here would be the bug that path exists to
        // catch.
        if let Some(ctx) = &session.gpu {
            assert!(
                !ctx.is_degraded(),
                "device degraded during the async run: {:?}",
                ctx.last_error()
            );
        }
    }

    /// A session with no device must still run to completion — the async entry
    /// point is not allowed to require a GPU.
    #[test]
    fn the_async_loop_runs_without_any_device() {
        let (graph, weights) = relu_add_graph();
        let mut session = Session::from_graph(graph, weights).expect("from_graph should succeed");
        // Drop whatever device the build attached: this is the browser's
        // starting state (construction cannot block on `requestAdapter`) and
        // the state of any machine without an adapter.
        session.gpu = None;

        let mut inputs = HashMap::new();
        inputs.insert("x", Tensor::new(vec![-1.0f32; 128 * 1024], vec![128, 1024]));
        let out = pollster::block_on(session.run_gpu_async(&inputs)).expect("async run");
        let y = out.get("y").expect("output");
        assert!(y.data.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    /// A graph covering the four remaining `try_gpu_dispatch_async` arms this
    /// wave is required to deliver — `Softmax`, `LayerNorm`, `Transpose` and
    /// `ReduceMean` — at shapes above each kernel's own threshold, so every arm
    /// is really dispatched rather than declined on size.
    ///
    /// These arms were produced by a *mechanical* rewrite of the synchronous
    /// dispatcher (rename the kernel, add `.await`). The compiler catches a
    /// misplaced `.await`; it does not catch an arm that ends up bound to the
    /// wrong pipeline or that drops an attribute along the way. Only running
    /// them against a real adapter and diffing against the CPU does.
    fn four_arm_graph() -> (Graph, HashMap<String, Tensor>) {
        let cols = 1024usize;

        let mut softmax_attrs = Attributes::default();
        softmax_attrs.ints.insert("axis".to_string(), -1);
        let mut ln_attrs = Attributes::default();
        ln_attrs.ints.insert("axis".to_string(), -1);
        ln_attrs.floats.insert("epsilon".to_string(), 1e-5);
        let mut transpose_attrs = Attributes::default();
        transpose_attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 0]);
        let mut mean_attrs = Attributes::default();
        mean_attrs.int_lists.insert("axes".to_string(), vec![2]);
        mean_attrs.ints.insert("keepdims".to_string(), 1);

        let nodes = vec![
            Node {
                op: OpKind::Softmax,
                name: "softmax".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["s".to_string()],
                attrs: softmax_attrs,
            },
            Node {
                op: OpKind::LayerNorm,
                name: "layer_norm".to_string(),
                inputs: vec!["s".to_string(), "scale".to_string(), "beta".to_string()],
                outputs: vec!["ln".to_string()],
                attrs: ln_attrs,
            },
            Node {
                op: OpKind::Transpose,
                name: "transpose".to_string(),
                inputs: vec!["ln".to_string()],
                outputs: vec!["t".to_string()],
                attrs: transpose_attrs,
            },
            Node {
                op: OpKind::ReduceMean,
                name: "reduce_mean".to_string(),
                inputs: vec!["r".to_string()],
                outputs: vec!["m".to_string()],
                attrs: mean_attrs,
            },
        ];
        let graph = Graph {
            nodes,
            input_names: vec!["x".to_string()],
            output_names: vec!["t".to_string(), "m".to_string()],
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "scale".to_string(),
            Tensor::new(vec![1.0f32; cols], vec![cols]),
        );
        weights.insert(
            "beta".to_string(),
            Tensor::new(vec![0.0f32; cols], vec![cols]),
        );
        // `ReduceMean` reduces the last axis of [256, 256, 4], leaving 65 536
        // outputs — above `REDUCE_GPU_THRESHOLD` (50 000), which is stated in
        // *output* elements.
        weights.insert(
            "r".to_string(),
            Tensor::new(
                (0..256 * 256 * 4)
                    .map(|i| ((i % 23) as f32) * 0.125 - 1.5)
                    .collect(),
                vec![256, 256, 4],
            ),
        );
        (graph, weights)
    }

    /// Softmax / LayerNorm / Transpose / ReduceMean must agree between the two
    /// loops, node for node.
    #[test]
    fn the_four_remaining_gpu_arms_agree_with_the_cpu() {
        let (graph, weights) = four_arm_graph();
        let mut session =
            Session::from_graph(graph.clone(), weights.clone()).expect("from_graph should succeed");
        let _has_gpu = pollster::block_on(session.enable_gpu_async());
        session.op_placement = OpPlacement::Auto {
            gpu_threshold_bytes: 65_536,
        };
        // Reference: the very same graph with no device at all.
        let mut cpu_session =
            Session::from_graph(graph, weights).expect("from_graph should succeed");
        cpu_session.gpu = None;

        let x: Vec<f32> = (0..256 * 1024)
            .map(|i| ((i % 97) as f32).mul_add(0.03, -1.4))
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert("x", Tensor::new(x, vec![256, 1024]));

        let gpu_out = pollster::block_on(session.run_gpu_async(&inputs)).expect("async run");
        let cpu_out = cpu_session.run(&inputs).expect("cpu run");

        for name in ["t", "m"] {
            let g = gpu_out.get(name).expect("gpu output");
            let c = cpu_out.get(name).expect("cpu output");
            assert_eq!(g.shape, c.shape, "{name}: shape must match");
            assert_eq!(g.data.len(), c.data.len(), "{name}: length must match");
            let worst = g
                .data
                .iter()
                .zip(c.data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= 1e-4,
                "{name}: async GPU arm diverged from the CPU by {worst}"
            );
        }

        if let Some(ctx) = &session.gpu {
            assert!(
                !ctx.is_degraded(),
                "device degraded during the four-arm run: {:?}",
                ctx.last_error()
            );
        }
    }

    /// The `Cpu` sentinel is terminal: nothing listed after it is reachable,
    /// which is the same rule `try_provider_list_dispatch` enforces.
    #[test]
    fn the_cpu_sentinel_stops_the_provider_walk() {
        assert!(provider_list_reaches_gpu(&[ProviderKind::Gpu]));
        assert!(!provider_list_reaches_gpu(&[
            ProviderKind::Cpu,
            ProviderKind::Gpu
        ]));
        assert!(!provider_list_reaches_gpu(&[]));
    }

    /// `CpuOnly` must close the gate for every op and every size — the async
    /// loop cannot become a way to route work to a GPU the session disabled.
    #[test]
    fn cpu_only_placement_closes_the_gate() {
        let huge = 1 << 24;
        assert!(!gpu_accelerator_gate(
            &OpKind::MatMul,
            huge,
            &OpPlacement::CpuOnly
        ));
        assert!(!gpu_accelerator_gate(
            &OpKind::Conv,
            huge,
            &OpPlacement::CpuOnly
        ));
    }

    /// `Auto`'s size threshold binds this path exactly as it binds the
    /// synchronous one: a node below it stays on the CPU.
    #[test]
    fn auto_threshold_binds_the_async_gate() {
        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 65_536,
        };
        assert!(!gpu_accelerator_gate(&OpKind::MatMul, 65_535, &placement));
        assert!(gpu_accelerator_gate(&OpKind::MatMul, 65_536, &placement));
        // An op with no GPU kernel never clears the gate, however large.
        assert!(!gpu_accelerator_gate(&OpKind::Gather, 1 << 24, &placement));
    }

    /// A `Manual` pin naming a *different* provider must not fall through to
    /// wgpu — that is the `{Conv: Gpu}` reroute bug the synchronous gate
    /// documents, and this loop must not reintroduce it.
    #[test]
    fn a_manual_pin_does_not_fall_through_to_wgpu() {
        let mut pins = HashMap::new();
        pins.insert(OpKind::Conv, ProviderKind::Gpu);
        let placement = OpPlacement::Manual(pins);
        assert!(gpu_accelerator_gate(&OpKind::Conv, 1 << 20, &placement));
        // MatMul was not pinned, so `decide_placement` returns Cpu for it.
        assert!(!gpu_accelerator_gate(&OpKind::MatMul, 1 << 20, &placement));
    }
}
