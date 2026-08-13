//! Parallel (rayon) graph execution.
//!
//! # Routing
//!
//! Every node is first handed to a **routing plan**: the ordered list of
//! execution providers that could claim it ([`plan_from_provider_list`] when the
//! session was built with [`crate::SessionBuilder::with_provider_kinds`], else
//! [`plan_from_placement`]).  An **empty** plan means the node belongs on the
//! rayon CPU path; a non-empty plan means it is dispatched serially through the
//! accelerator chain first.
//!
//! The plan is what decides whether a node is pulled out of `par_iter`, so it
//! must be *exact*: a plan that over-claims (the old
//! "`self.cuda.is_some()` ⇒ eligible" gate) drags every node in the graph into
//! the serial phase and destroys the parallelism this module exists to provide.
//! It is therefore built from each backend's **own** op-support predicate
//! (`oxionnx_cuda::is_supported_op`, `oxionnx_directml::is_supported_op`,
//! [`crate::execution_providers::is_gpu_capable`]) — which is documented as a
//! *hard guarantee* in the negative direction — together with the session's
//! placement policy, the node's output size, and the presence of a live device
//! context.
//!
//! Both plan builders take the node's output size, estimated by the one canonical
//! `Session::estimate_output_bytes` in `run/dispatch.rs` (this module used to carry
//! a shim around it, because it was `#[cfg(feature = "gpu")]` and so missing from
//! the `cuda`-only build; its `#[cfg]` is now `any(gpu, cuda, directml)` and the
//! shim is gone).  `plan_from_placement` hands that size to `decide_placement`;
//! `plan_from_provider_list` checks it against the hard
//! `Session::provider_list_clears_dispatch_floor` — the *same* predicate the
//! sequential path uses, so an explicit provider list means exactly the same thing
//! under `with_parallel_execution(true)` as without it.
//!
//! # Write-back
//!
//! No provider result — CPU included — is written into [`SessionRunState`] with a
//! hand-rolled `zip`.  Everything routes through
//! `Session::write_node_outputs`, which rejects arity mismatches, internally
//! inconsistent tensors and shapes that disagree with shape inference *before*
//! it touches `state`.

#[cfg(not(target_arch = "wasm32"))]
use crate::execution_providers::{
    decide_placement, provider_supports_op, OpPlacement, ProviderKind,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::graph::{Node, OpKind};
#[cfg(not(target_arch = "wasm32"))]
use crate::tensor::Tensor;
use crate::OnnxError;
#[cfg(not(target_arch = "wasm32"))]
use oxionnx_core::{OpContext, Operator};
use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
use crate::memory::SizeClassPool;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
use super::super::types::NodeProfile;
use super::super::Session;
// The mixed-precision precedence is defined once, on the sequential path, and
// applied identically here — see `run/sequential.rs`.
#[cfg(not(target_arch = "wasm32"))]
use super::sequential::mixed_precision_claims_node;
use super::state::SessionRunState;
use super::{OutputSet, RefCounts};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// ── Live device contexts ────────────────────────────────────────────────────

/// Which accelerator device contexts are actually alive on a [`Session`].
///
/// Factored out of `Session` on purpose: it makes the routing decision — the part
/// that carries all the interesting logic — a **pure function** that can be
/// unit-tested with any combination of contexts declared "present", on a build
/// machine that has no GPU at all.  Without this, the CUDA and DirectML routing
/// rules would be untestable in CI (`CudaContext::try_new()` returns `None` with
/// no device; `DirectMLContext::try_new()` returns `None` off Windows), which is
/// precisely how the "`self.cuda.is_some()` ⇒ eligible" bug survived.
///
/// The struct has one field per *compiled-in* accelerator, so with no accelerator
/// feature enabled it is a genuine zero-sized type and every routing rule that
/// consults it folds away.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LiveContexts {
    /// A CUDA context was successfully created.
    #[cfg(feature = "cuda")]
    pub(crate) cuda: bool,
    /// A DirectML context was successfully created.
    #[cfg(feature = "directml")]
    pub(crate) dml: bool,
    /// A wgpu context was successfully created.
    #[cfg(feature = "gpu")]
    pub(crate) gpu: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl LiveContexts {
    /// Is `provider`'s device context live?
    ///
    /// [`ProviderKind::Cpu`] is always live — it is the terminal fallback and
    /// needs no device.
    fn has(self, provider: ProviderKind) -> bool {
        match provider {
            ProviderKind::Cpu => true,
            #[cfg(feature = "cuda")]
            ProviderKind::Cuda => self.cuda,
            #[cfg(feature = "directml")]
            ProviderKind::DirectMl => self.dml,
            #[cfg(feature = "gpu")]
            ProviderKind::Gpu => self.gpu,
        }
    }
}

// ── Pure routing core ───────────────────────────────────────────────────────

/// The accelerator priority chain — `Cuda > DirectMl > Gpu` — restricted to the
/// providers compiled into this build.
///
/// This must stay in lockstep with
/// [`crate::execution_providers::select_accelerator`], which is the crate's single
/// source of truth for priority.  `priority_chain_agrees_with_select_accelerator`
/// in this module's tests fails the moment the two disagree.
///
/// Written as three `Option`s rather than `Vec::push` calls under `#[cfg]` so
/// that the no-accelerator build does not end up with an `unused_mut` on the
/// accumulator.
#[cfg(not(target_arch = "wasm32"))]
fn accelerator_priority_chain() -> Vec<ProviderKind> {
    #[cfg(feature = "cuda")]
    let cuda: Option<ProviderKind> = Some(ProviderKind::Cuda);
    #[cfg(not(feature = "cuda"))]
    let cuda: Option<ProviderKind> = None;

    #[cfg(feature = "directml")]
    let dml: Option<ProviderKind> = Some(ProviderKind::DirectMl);
    #[cfg(not(feature = "directml"))]
    let dml: Option<ProviderKind> = None;

    #[cfg(feature = "gpu")]
    let gpu: Option<ProviderKind> = Some(ProviderKind::Gpu);
    #[cfg(not(feature = "gpu"))]
    let gpu: Option<ProviderKind> = None;

    [cuda, dml, gpu].into_iter().flatten().collect()
}

/// Build the routing plan for `op` from an **explicit** provider list
/// (`SessionBuilder::with_provider_kinds`).
///
/// An explicit list is a direct instruction from the caller, so it overrides the
/// [`OpPlacement`] heuristic entirely — exactly as
/// `Session::try_provider_list_dispatch` does on the sequential path.  The two
/// paths *must* agree: a model that runs on DirectML sequentially and silently
/// falls back to the CPU under `with_parallel_execution(true)` is the worst kind
/// of bug, because it is invisible in the output.
///
/// # Semantics (mirroring the sequential path exactly)
///
/// - [`ProviderKind::Cpu`] is a **terminal sentinel**: it means "stop here, run on
///   the CPU".  Providers listed *after* it are never reached.  `[Cpu]` therefore
///   pins the whole graph to the CPU, and `[Cuda, Cpu]` means "CUDA, else CPU".
/// - A provider with no live context is skipped (the caller falls through).
/// - A provider whose backend has no kernel for `op` is skipped.  This is not a
///   heuristic: `is_supported_op(op) == false` is documented by both backend
///   crates as a *hard guarantee* that dispatch would return `Ok(None)`.  Skipping
///   it here is therefore free of behavioural change and saves the node from being
///   serialised out of `par_iter` for a probe that could only ever decline.
/// - The session's [`OpPlacement`] — and therefore its `gpu_threshold_bytes` — is
///   **not** consulted: an explicit list overrides the heuristic outright, so a
///   listed provider claims every node it has a kernel for even under the default
///   `CpuOnly` placement.
/// - The one size rule that *does* bind is the hard `MIN_GPU_DISPATCH_BYTES` floor,
///   via [`Session::provider_list_clears_dispatch_floor`] — the very same predicate
///   `Session::try_provider_list_dispatch` applies on the sequential path, so the
///   two cannot drift.  A sub-page tensor is never worth a PCIe round trip on any
///   backend, and on *this* path it would additionally forfeit the node's place in
///   `par_iter`.  The full argument lives on that function.
///
/// An empty result means "this node belongs on the rayon CPU path".
#[cfg(not(target_arch = "wasm32"))]
fn plan_from_provider_list(
    op: &OpKind,
    output_bytes: usize,
    provider_list: &[ProviderKind],
    live: LiveContexts,
) -> Vec<ProviderKind> {
    // The floor binds every listed accelerator identically, so a sub-page node has
    // no plan at all — it stays on the rayon CPU path rather than being serialised
    // out of it for a round trip that cannot pay for itself.
    if !Session::provider_list_clears_dispatch_floor(output_bytes) {
        return Vec::new();
    }

    let mut plan = Vec::with_capacity(provider_list.len());
    for &provider in provider_list {
        // CPU terminates the list: nothing after it is ever consulted.
        if matches!(provider, ProviderKind::Cpu) {
            break;
        }
        if provider_supports_op(provider, op) && live.has(provider) {
            plan.push(provider);
        }
    }
    plan
}

/// Build the routing plan for `op` from the session's [`OpPlacement`] heuristic
/// (used when no explicit provider list was supplied).
///
/// [`decide_placement`] is the crate's single source of truth for *whether* a node
/// may leave the CPU: it applies `CpuOnly`, the `Auto` size threshold, and the
/// `Manual` pin plus its `MIN_GPU_DISPATCH_BYTES` floor.  This function asks it
/// first and returns an empty plan the moment it answers [`ProviderKind::Cpu`].
///
/// # Why `decide_placement` and not a bare op-support check
///
/// Op support alone is not enough.  `Add` has a kernel in all three backends, but
/// a `[1, 4]` f32 bias-add is 16 bytes: uploading it, launching a kernel, fencing
/// and reading it back costs ~20 µs of fixed round-trip against ~4 ns of actual
/// arithmetic.  Gating on op support alone would ship it across PCIe anyway *and*
/// pull it out of the rayon path to do so — losing twice.  `decide_placement` owns
/// that cost model (see `MIN_GPU_DISPATCH_BYTES`), so the gate defers to it.
///
/// # Why the chain, and not just `decide_placement`'s answer
///
/// `decide_placement` returns a *preference*, computed with no knowledge of which
/// device contexts actually exist — its own documentation says so, and instructs
/// callers to keep falling through when the preferred backend is absent.  Under
/// `Auto` it returns `Cuda` for `Add` whenever the `cuda` feature is compiled in,
/// even on a machine with no NVIDIA card and a perfectly good wgpu context.  So we
/// take its verdict as the *authorisation* to accelerate, then walk the full
/// priority chain, keeping the providers that both implement the op and have a
/// live context.
///
/// Under `Manual` the chain is **not** walked: pinning `Add -> Cuda` is a specific
/// request for CUDA, not a request for "any accelerator", so the plan holds that
/// one provider (and is empty if CUDA cannot claim the op or has no context).
#[cfg(not(target_arch = "wasm32"))]
fn plan_from_placement(
    op: &OpKind,
    output_bytes: usize,
    placement: &OpPlacement,
    live: LiveContexts,
) -> Vec<ProviderKind> {
    // CpuOnly, below-threshold, unpinned, or no backend implements the op.
    let preferred = decide_placement(op, output_bytes, placement);
    if matches!(preferred, ProviderKind::Cpu) {
        return Vec::new();
    }

    let ordered: Vec<ProviderKind> = match placement {
        // An explicit pin names one provider; honour exactly that one.
        OpPlacement::Manual(_) => vec![preferred],
        // `Auto` (the only other way to reach here) authorised acceleration in
        // general — fall through the priority chain to find a backend that is
        // actually present.
        _ => accelerator_priority_chain(),
    };

    ordered
        .into_iter()
        .filter(|&provider| !matches!(provider, ProviderKind::Cpu))
        .filter(|&provider| provider_supports_op(provider, op))
        .filter(|&provider| live.has(provider))
        .collect()
}

// ── CPU work items ──────────────────────────────────────────────────────────

/// One CPU node of a depth level, after the fast-path decision has been made but
/// before its input references have been resolved.
///
/// Split out from [`CpuWorkItem`] because deciding the fast path needs `&mut`
/// access to the run state (an in-place claim *removes* its input from it) while
/// resolving input references needs `&` access to the settled state.  See
/// [`Session::claim_cpu_fast_paths`].
#[cfg(not(target_arch = "wasm32"))]
struct CpuClaim<'a> {
    /// Index into `Session::sorted_nodes` — never the node *name*, which ONNX
    /// permits to be empty or duplicated.
    idx: usize,
    node: &'a Node,
    operator: &'a dyn Operator,
    /// The first input, taken out of the run state, when the in-place path
    /// claimed this node.
    owned_input: Option<Tensor>,
    /// Pool-backed output buffers, when the slot-write path claimed it.
    slots: Option<Vec<Tensor>>,
    /// Has mixed precision claimed this node?
    claimed: bool,
    /// Was the in-place path claimed?  (Distinct from `owned_input.is_some()`:
    /// the tensor may have been absent from the state, in which case the node
    /// falls back to `execute` with an empty first input — exactly what
    /// `dispatch_node` does.)
    inplace: bool,
}

/// One CPU node of a depth level, ready to execute inside `into_par_iter`.
///
/// Everything a worker needs is owned or borrowed-immutably here, so the compute
/// phase touches no shared mutable state: the input references and `outer_scope`
/// are immutable borrows of the run state, while `slots` and `owned_input` are
/// owned outright by this item.
#[cfg(not(target_arch = "wasm32"))]
struct CpuWorkItem<'a> {
    idx: usize,
    node: &'a Node,
    operator: &'a dyn Operator,
    inputs: Vec<Option<&'a Tensor>>,
    owned_input: Option<Tensor>,
    slots: Option<Vec<Tensor>>,
    claimed: bool,
}

/// Human-readable backend name, as `Session::write_node_outputs` expects.
#[cfg(not(target_arch = "wasm32"))]
fn provider_label(provider: ProviderKind) -> &'static str {
    match provider {
        ProviderKind::Cpu => "CPU",
        #[cfg(feature = "cuda")]
        ProviderKind::Cuda => "CUDA",
        #[cfg(feature = "directml")]
        ProviderKind::DirectMl => "DirectML",
        #[cfg(feature = "gpu")]
        ProviderKind::Gpu => "wgpu",
    }
}

impl Session {
    /// Snapshot which accelerator contexts this session actually has.
    #[cfg(not(target_arch = "wasm32"))]
    fn live_contexts(&self) -> LiveContexts {
        LiveContexts {
            #[cfg(feature = "cuda")]
            cuda: self.cuda.is_some(),
            #[cfg(feature = "directml")]
            dml: self.dml.is_some(),
            #[cfg(feature = "gpu")]
            gpu: self.gpu.is_some(),
        }
    }

    /// The ordered providers that may claim `node`, or an empty plan when the node
    /// belongs on the rayon CPU path.
    ///
    /// This is the gate that decides which nodes are pulled out of `par_iter` and
    /// dispatched serially, so its cost is paid per node, per run.  The early
    /// return keeps that cost at *zero* for the overwhelmingly common
    /// configuration — no provider list, default `CpuOnly` placement — by skipping
    /// the output-size estimate entirely.
    #[cfg(not(target_arch = "wasm32"))]
    fn plan_node_providers(
        &self,
        node: &Node,
        state: &SessionRunState,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Vec<ProviderKind> {
        // No explicit list, and the default placement accelerates nothing: there is
        // no plan to build and no estimate to pay for.  (A non-empty provider list
        // overrides `op_placement` outright, `CpuOnly` included — so it must be
        // checked first, not folded into this.)
        if self.providers.is_empty() && matches!(self.op_placement, OpPlacement::CpuOnly) {
            return Vec::new();
        }

        // The payload every size rule below is stated in.  One canonical estimator,
        // shared with the sequential path — see `Session::estimate_output_bytes`.
        //
        // `#[cfg]`: that function is compiled only when an accelerator is.  With
        // none compiled in, `ProviderKind::Cpu` is the enum's sole variant, so every
        // plan below is empty regardless and the size is unobservable; `0` also
        // keeps the provider-list floor closed, which is the same answer by a
        // shorter route.
        #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
        let output_bytes =
            Session::estimate_output_bytes(node, state.as_map(), &self.weights, resolved);
        #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "directml")))]
        let output_bytes = {
            let _ = (state, resolved);
            0
        };

        // An explicit provider list overrides the placement heuristic outright.
        if !self.providers.is_empty() {
            return plan_from_provider_list(
                &node.op,
                output_bytes,
                &self.providers,
                self.live_contexts(),
            );
        }

        plan_from_placement(
            &node.op,
            output_bytes,
            &self.op_placement,
            self.live_contexts(),
        )
    }

    /// Dispatch `node` through its routing `plan`, in order.
    ///
    /// Returns `Ok(true)` when a provider produced results and they were written
    /// into `state`; `Ok(false)` when every provider in the plan declined, and the
    /// caller must run the node on the CPU.
    ///
    /// A provider that returns `Ok(None)` (op declined for this node's specific
    /// configuration) or errors is skipped and the next provider is tried — that is
    /// the documented `Ok(None)` contract of both backend crates.  A *write-back*
    /// failure, by contrast, is propagated: a provider that returns the wrong number
    /// of tensors, or a tensor whose shape contradicts shape inference, has produced
    /// garbage, and silently falling back to the CPU would hide it.
    #[cfg(not(target_arch = "wasm32"))]
    fn try_accelerated_node(
        &self,
        node: &Node,
        plan: &[ProviderKind],
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Result<bool, OnnxError> {
        for &provider in plan {
            let start = crate::time_compat::Instant::now();

            let dispatched: Option<Vec<Tensor>> = match provider {
                // Never planned (it is the terminal sentinel, not an accelerator);
                // handled defensively so the match stays total.
                ProviderKind::Cpu => None,

                #[cfg(feature = "cuda")]
                ProviderKind::Cuda => {
                    let Some(cuda_ctx) = &self.cuda else {
                        continue;
                    };
                    match oxionnx_cuda::try_cuda_dispatch(
                        node,
                        &self.weights,
                        state.as_map(),
                        cuda_ctx,
                    ) {
                        Ok(results) => results,
                        // PROVED WRONG under `OXIONNX_CUDA_STRICT=1` — the same
                        // rule the sequential path applies, and for the same
                        // reason: strict mode is a promise that a demonstrated
                        // GPU fault ends the run, and it must not depend on
                        // whether the session happens to be running in parallel.
                        // See `Session::dispatch_to_cuda` in `run/sequential.rs`.
                        Err(e)
                            if super::sequential::classify_cuda_failure(&e)
                                == super::sequential::CudaFailureAction::FailTheRun =>
                        {
                            tracing::error!(
                                op = %node.op.as_str(),
                                node = %node.name,
                                err = %e,
                                "parallel: CUDA was PROVED WRONG by shadow verification and \
                                 OXIONNX_CUDA_STRICT is set; failing the run",
                            );
                            return Err(e);
                        }
                        Err(_e) => {
                            #[cfg(debug_assertions)]
                            tracing::debug!(
                                op = %node.op.as_str(),
                                node = %node.name,
                                err = %_e,
                                "parallel: CUDA dispatch error, trying next provider",
                            );
                            continue;
                        }
                    }
                }

                #[cfg(feature = "directml")]
                ProviderKind::DirectMl => {
                    let Some(dml_ctx) = &self.dml else {
                        continue;
                    };
                    match oxionnx_directml::try_directml_dispatch(
                        node,
                        &self.weights,
                        state.as_map(),
                        dml_ctx,
                    ) {
                        Ok(results) => results,
                        Err(_e) => {
                            #[cfg(debug_assertions)]
                            tracing::debug!(
                                op = %node.op.as_str(),
                                node = %node.name,
                                err = %_e,
                                "parallel: DirectML dispatch error, trying next provider",
                            );
                            continue;
                        }
                    }
                }

                #[cfg(feature = "gpu")]
                ProviderKind::Gpu => {
                    let Some(gpu_ctx) = &self.gpu else {
                        continue;
                    };
                    super::super::gpu_dispatch::try_gpu_dispatch(
                        node,
                        &self.weights,
                        state.as_map(),
                        gpu_ctx,
                    )?
                }
            };

            // The provider declined this particular node — fall through the chain.
            let Some(results) = dispatched else {
                continue;
            };
            let elapsed = start.elapsed();

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

            // Validated, all-or-nothing write-back: never a bare `zip`.
            self.write_node_outputs(node, provider_label(provider), results, state, resolved)?;
            self.decrement_refs_state(node, state, ref_counts, output_set);
            return Ok(true);
        }

        Ok(false)
    }

    /// Execute one node on the CPU **serially**, through the full `dispatch_node`
    /// path (in-place and slot-write optimisations included) and with the live run
    /// state available as the operator's outer scope.
    ///
    /// Three kinds of node come through here:
    ///
    /// * the sole node of a single-node depth level — there is no concurrency to
    ///   lose, so it takes the better-optimised path;
    /// * every **control-flow** node (`If`, `Loop`, `Scan`, or any node carrying a
    ///   subgraph attribute), at any level.  ONNX subgraphs capture outer-scope
    ///   tensors implicitly by name and `IfOp`/`LoopOp`/`ScanOp` resolve them
    ///   entirely out of `ctx.outer_scope`, so a control-flow node *must* see the
    ///   live state map.  In the `par_iter` phase it cannot: the state is borrowed
    ///   immutably across the whole compute phase and each worker only holds a
    ///   pre-resolved input vector.  Running these serially is what makes
    ///   `with_parallel_execution(true)` produce the same answer as the sequential
    ///   path for a graph containing control flow, instead of failing with
    ///   `TensorNotFound` or silently taking the wrong branch;
    /// * any node **mixed precision has claimed** that has no native f16 kernel,
    ///   when it is alone at its level.
    ///
    /// `mixed_precision_node` must be [`mixed_precision_claims_node`]'s answer for
    /// this node; it selects the same native-f16-then-round policy the sequential
    /// path applies.
    #[cfg(not(target_arch = "wasm32"))]
    fn dispatch_serially(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved: &HashMap<String, Vec<usize>>,
        mixed_precision_node: bool,
    ) -> Result<(), OnnxError> {
        if mixed_precision_node
            && self.try_native_f16_node(node, state, ref_counts, output_set, resolved)?
        {
            return Ok(());
        }

        let op_name = node.op.as_str();
        let operator = self
            .registry
            .get(op_name)
            .ok_or_else(|| super::unsupported_op_error(node))?;

        let elapsed =
            self.dispatch_node(node, operator, state, ref_counts, output_set, resolved)?;

        if mixed_precision_node {
            self.round_node_outputs_to_f16(node, state);
        }

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
                    op_type: op_name.to_string(),
                    duration: elapsed,
                    output_shapes,
                });
            }
        }

        self.decrement_refs_state(node, state, ref_counts, output_set);
        Ok(())
    }

    /// Claim the in-place and output-slot fast paths for one depth level's CPU
    /// nodes, taking ownership of the buffers each claim needs.
    ///
    /// # Why this can be done at all under `par_iter`
    ///
    /// Both fast paths need *ownership* of a buffer, which is why they used to be
    /// sequential-only.  But ownership is established **here**, in the serial
    /// build phase that already holds `&mut state`: the slot buffers are taken out
    /// of the pool, and an in-place input is taken out of the run state, before a
    /// single worker starts.  Each work item then owns its buffers outright, so
    /// the compute phase touches no shared mutable state whatsoever — exactly as
    /// it did when every node allocated a fresh output.
    ///
    /// # Why the in-place claim is sound
    ///
    /// [`Session::node_can_execute_inplace`] — the *same* predicate the sequential
    /// path uses — requires the input's reference count to be exactly 1, i.e. the
    /// claiming node is its only consumer in the entire graph, and requires it not
    /// to be a declared graph output or an initializer.  Two consequences make the
    /// concurrency safe:
    ///
    /// * **No sibling can be reading it.**  Two nodes at this level consuming the
    ///   same tensor would give it a count of ≥ 2, so a count of 1 proves no other
    ///   work item resolved a reference to it.
    /// * **The count cannot change under us.**  `decrement_refs_state` for these
    ///   nodes runs *after* the write phase, so nothing decrements between the
    ///   decision here and the execution in the worker.
    ///
    /// Removing the tensor from `state` also removes it from the `outer_scope` map
    /// the workers share — which is precisely what `dispatch_node` does on the
    /// sequential path, so the two paths present operators with the same scope.
    /// Control-flow nodes (the only operators that read `outer_scope`) never reach
    /// this phase; phase 1 dispatches them serially.
    ///
    /// # Mixed precision
    ///
    /// A node mixed precision has claimed keeps the plain `execute` path: it may
    /// be answered by the native f16 kernel inside the worker, and the f16
    /// rounding in the write phase would otherwise have to un-do a slot write.
    /// The numbers are identical either way; only the allocation is not saved.
    ///
    /// # Errors
    ///
    /// [`OnnxError::UnsupportedOp`] for a node whose operator is not in the
    /// registry — the same gate every other execution path applies.
    ///
    /// Deliberately **not** fused with [`Session::resolve_work_item_inputs`]: this
    /// half needs `&mut state` and that half needs `&state`, and keeping them
    /// separate is what lets the input references (and the shared `outer_scope`
    /// map) be taken *after* every in-place claim has already removed its tensor.
    #[cfg(not(target_arch = "wasm32"))]
    fn claim_cpu_fast_paths<'a>(
        &'a self,
        cpu_nodes: &[usize],
        state: &mut SessionRunState,
        ref_counts: &RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Result<Vec<CpuClaim<'a>>, OnnxError> {
        let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);

        let mut claims: Vec<CpuClaim<'a>> = Vec::with_capacity(cpu_nodes.len());
        for &idx in cpu_nodes {
            let node = &self.sorted_nodes[idx];
            let operator = self
                .registry
                .get(node.op.as_str())
                .ok_or_else(|| super::unsupported_op_error(node))?;
            let claimed = mixed_precision_claims_node(self.mixed_precision, node.op.as_str());

            let inplace =
                !claimed && self.node_can_execute_inplace(node, operator, ref_counts, output_set);
            let owned_input = if inplace {
                state.take(&node.inputs[0])
            } else {
                None
            };
            let slots = if !claimed && !inplace && operator.supports_output_slots() {
                Self::acquire_output_slots(node, resolved, pool)
            } else {
                None
            };

            claims.push(CpuClaim {
                idx,
                node,
                operator,
                owned_input,
                slots,
                claimed,
                inplace,
            });
        }

        Ok(claims)
    }

    /// Resolve each claim's input references against the settled run state,
    /// producing the `par_iter`-ready work items.
    ///
    /// An in-place node's first input slot is `None`, exactly as `dispatch_node`
    /// builds it: the operator receives that tensor by value instead.
    #[cfg(not(target_arch = "wasm32"))]
    fn resolve_work_item_inputs<'a>(
        &'a self,
        claims: Vec<CpuClaim<'a>>,
        state: &'a SessionRunState,
    ) -> Vec<CpuWorkItem<'a>> {
        claims
            .into_iter()
            .map(|claim| {
                let inputs: Vec<Option<&'a Tensor>> = claim
                    .node
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        if name.is_empty() || (claim.inplace && i == 0) {
                            None
                        } else {
                            state.get(name).or_else(|| self.weights.get(name))
                        }
                    })
                    .collect();
                CpuWorkItem {
                    idx: claim.idx,
                    node: claim.node,
                    operator: claim.operator,
                    inputs,
                    owned_input: claim.owned_input,
                    slots: claim.slots,
                    claimed: claim.claimed,
                }
            })
            .collect()
    }

    /// Parallel execution: group nodes by topological depth and execute each
    /// depth level using a hybrid strategy:
    ///
    /// - Nodes with a non-empty routing plan (see [`Session::plan_node_providers`])
    ///   are executed **serially** through their accelerator chain.  This is
    ///   intentional: GPU driver contexts are not safe to call concurrently from
    ///   multiple rayon workers, and on-device queuing already provides hardware
    ///   parallelism.  A node whose plan is empty — which is *every* node in a
    ///   CPU-only session, and every node no accelerator implements — never enters
    ///   this phase.
    /// - The remaining nodes at the same depth are executed **concurrently** via
    ///   rayon's `par_iter()`.
    ///
    /// The in-place and output-slot fast paths are active on **both** kinds of
    /// level.  They used to be sequential-only, on the reasoning that they need
    /// exclusive mutable access to the run state during the operator call — which
    /// is true of the *state*, but not of the *buffers*: the decision and the
    /// buffer acquisition happen in the serial build phase (which already holds
    /// `&mut state`), and each worker then owns its slot vector or its taken input
    /// outright.  See [`Session::claim_cpu_fast_paths`] for the safety argument.
    ///
    /// `resolved` is **this run's** shape map (see `Session::resolve_run_shapes`),
    /// owned by the caller rather than re-read from the session-wide mutex, so two
    /// concurrent runs with different batch sizes cannot swap shape maps.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn run_parallel_inner(
        &self,
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Result<(), OnnxError> {
        // Version-sensitive operators read this off their `OpContext`; it must be
        // bound before the first node executes.  The store is sequenced-before
        // every `par_iter` below, so the workers observe it.
        self.bind_registry_opset();

        // The schedule — depths, depth grouping, critical-path costs and the
        // per-level heaviest-first sort — is a pure function of `sorted_nodes`,
        // `weights` and `shape_cache`, none of which can change after the
        // session is built.  All four passes used to run on *every* inference;
        // they now run once, in `StaticRunPlan::build`.  Levels are still
        // ordered heaviest-first, which is what keeps a level's tail short.
        let groups = &self.run_plan.depth_groups;

        for group in groups {
            if group.is_empty() {
                continue;
            }

            if group.len() == 1 {
                // Single node — try its routing plan first, then the CPU path.
                // No `OpKind::Unknown => continue`: the registry lookup inside
                // `dispatch_serially` is the gate.  See `super::unsupported_op_error`.
                let node = &self.sorted_nodes[group[0]];

                // Mixed precision claims f16-safe ops ahead of every execution
                // provider — the precedence `run/sequential.rs` documents on
                // `mixed_precision_claims_node`.  A claimed node is never offered a
                // plan, exactly as on the sequential path.
                let claimed = mixed_precision_claims_node(self.mixed_precision, node.op.as_str());

                if !claimed {
                    let plan = self.plan_node_providers(node, state, resolved);
                    if !plan.is_empty()
                        && self.try_accelerated_node(
                            node, &plan, state, ref_counts, output_set, resolved,
                        )?
                    {
                        continue;
                    }
                }

                // No provider claimed the node — fall back to CPU dispatch_node
                // (inplace + slot-write optimisations active for single-node levels).
                self.dispatch_serially(node, state, ref_counts, output_set, resolved, claimed)?;
            } else {
                // Multiple nodes at this depth — hybrid dispatch:
                //
                //   Phase 1 (serial):   nodes with a non-empty routing plan are
                //                       dispatched one-by-one through their provider
                //                       chain.  GPU contexts are not thread-safe;
                //                       serial dispatch is mandatory here.
                //
                //   Phase 2 (parallel): everything else, via rayon par_iter.
                //     Read sub-phase:   snapshot inputs (immutable borrow ends before write).
                //     Compute sub-phase: par_iter — no state access, full rayon parallelism.
                //     Write sub-phase:  sequential, validated write-back.

                // ── Phase 1: serial dispatch ─────────────────────────────────
                //
                // Two kinds of node leave the rayon path here: those an execution
                // provider claims (GPU driver contexts are not safe to call from
                // several workers), and those carrying a **subgraph attribute**.
                // The latter — `If`, `Loop`, `Scan` — resolve their bodies' free
                // names out of `ctx.outer_scope`, which only the serial path can
                // supply as the live run state.
                //
                // `cpu_nodes` holds **indices into `self.sorted_nodes`**, never node
                // names.  `NodeProto.name` is optional in the ONNX spec and plenty of
                // exporters emit `""` or duplicates; matching results back to nodes by
                // name silently attributed node B's tensors to node A's output names
                // whenever two nodes at one depth shared a name.  Indices are unique by
                // construction and cannot collide.
                let mut cpu_nodes: Vec<usize> = Vec::with_capacity(group.len());

                for &idx in group {
                    let node = &self.sorted_nodes[idx];
                    // No `OpKind::Unknown => continue`: the registry lookup in the
                    // work-item build below is the gate.

                    // Mixed precision claims f16-safe ops ahead of every execution
                    // provider — see `mixed_precision_claims_node`.
                    let claimed =
                        mixed_precision_claims_node(self.mixed_precision, node.op.as_str());

                    if !claimed {
                        let plan = self.plan_node_providers(node, state, resolved);
                        if !plan.is_empty()
                            && self.try_accelerated_node(
                                node, &plan, state, ref_counts, output_set, resolved,
                            )?
                        {
                            continue;
                        }
                    }

                    // Control flow needs the live outer scope; `par_iter` cannot
                    // give it one.
                    if !node.attrs.graphs.is_empty() {
                        self.dispatch_serially(
                            node, state, ref_counts, output_set, resolved, claimed,
                        )?;
                        continue;
                    }

                    // Empty plan, or every provider in it declined.
                    cpu_nodes.push(idx);
                }

                // ── Phase 2: parallel CPU dispatch ───────────────────────────
                //
                //   Claim sub-phase:   resolve operators and claim the in-place and
                //                      output-slot fast paths.  Needs `&mut state`
                //                      (an in-place claim takes its input out of
                //                      it), so it runs — and ends — before the
                //                      immutable borrow below begins.
                //   Read sub-phase:    resolve every node's input references and
                //                      the shared outer scope, against the state as
                //                      the claims left it.
                //   Compute sub-phase: `into_par_iter` — no state access at all,
                //                      each worker owns its slots / taken input.
                //   Write sub-phase:   sequential, validated write-back.
                let claims =
                    self.claim_cpu_fast_paths(&cpu_nodes, state, ref_counts, output_set, resolved)?;

                // `outer_scope` is the live state map, borrowed immutably for the
                // whole compute phase exactly as the work items' input references
                // are.  It used to be `None`, which silently emptied the enclosing
                // scope for every operator that reads one.
                let outer_scope: &HashMap<String, Tensor> = state.as_map();
                let work_items = self.resolve_work_item_inputs(claims, state);

                // Execute in parallel.  Each result carries the node's **index**, so
                // the write phase can never mis-attribute tensors to a same-named node,
                // and a flag saying whether the native f16 kernel produced it (in which
                // case the outputs are already at f16 precision and must not be rounded
                // a second time).
                type ParResult = Result<(usize, Vec<Tensor>, std::time::Duration, bool), OnnxError>;
                let par_execute = || -> Vec<ParResult> {
                    work_items
                        .into_par_iter()
                        .map(|item| {
                            let CpuWorkItem {
                                idx,
                                node,
                                operator,
                                inputs,
                                owned_input,
                                slots,
                                claimed,
                            } = item;
                            let start = crate::time_compat::Instant::now();

                            // Native f16 elementwise kernel, when mixed precision
                            // claimed the node and one exists for this op.
                            if claimed {
                                let refs: Vec<&Tensor> = inputs.iter().flatten().copied().collect();
                                if let Some(f16_result) =
                                    super::super::mixed_precision::execute_elementwise_f16(
                                        node.op.as_str(),
                                        &refs,
                                    )
                                {
                                    return Ok((idx, f16_result?, start.elapsed(), true));
                                }
                            }

                            let ctx = OpContext {
                                node,
                                inputs,
                                outer_scope: Some(outer_scope),
                                weights: Some(&self.weights),
                                registry: Some(&self.registry),
                            };

                            // Same three-way precedence as `dispatch_node`, and
                            // the same gates decided it (see
                            // `claim_cpu_fast_paths`): slot-write, else in-place,
                            // else a fresh allocation.
                            let res = match (slots, owned_input) {
                                (Some(mut slots), _) => {
                                    operator.execute_into_slots(&ctx, &mut slots)?;
                                    slots
                                }
                                (None, Some(owned)) => operator.execute_inplace(owned, &ctx)?,
                                (None, None) => operator.execute(&ctx)?,
                            };
                            let elapsed = start.elapsed();
                            Ok((idx, res, elapsed, false))
                        })
                        .collect()
                };
                let par_results: Vec<ParResult> = if let Some(ref pool) = self.thread_pool {
                    pool.install(par_execute)
                } else {
                    par_execute()
                };

                // Write phase: validated, all-or-nothing write-back per node.
                //
                // Both immutable borrows of `state` — `outer_scope` and the work
                // items' input references — end with `par_execute`, which consumed
                // the work items and was itself consumed by the call above.
                // `par_results` owns its tensors outright, which is what lets the
                // write-back take `&mut state` again.  Using `outer_scope` (or the
                // work items) past this point re-borrows `state` immutably and the
                // resulting error will point at `write_node_outputs`, not here.
                for result in par_results {
                    let (idx, mut tensors, elapsed, native_f16) = result?;
                    let node = &self.sorted_nodes[idx];
                    let op_name = node.op.as_str();

                    // Mixed precision without a native f16 kernel: the op ran in
                    // f32, so its outputs are squeezed through f16 here — the same
                    // rounding `Session::round_node_outputs_to_f16` applies on the
                    // sequential path.  Elided-output placeholders are skipped:
                    // their contents are meaningless by convention and
                    // `Tensor::new` would assert on the `data: [] / shape: []` one.
                    if !native_f16 && mixed_precision_claims_node(self.mixed_precision, op_name) {
                        for (slot, tensor) in tensors.iter_mut().enumerate() {
                            // `Option::is_none_or` is stable only since 1.82; this
                            // crate's MSRV is 1.75.
                            #[allow(clippy::unnecessary_map_or)]
                            let elided = node.outputs.get(slot).map_or(true, |n| n.is_empty());
                            if elided {
                                continue;
                            }
                            *tensor = super::super::mixed_precision::round_to_f16_precision(tensor);
                        }
                    }

                    if let Some(ref profiling) = self.profiling_data {
                        if let Ok(mut data) = profiling.lock() {
                            data.push(NodeProfile {
                                node_name: node.name.clone(),
                                op_type: if native_f16 {
                                    format!("{op_name}(f16)")
                                } else {
                                    op_name.to_string()
                                },
                                duration: elapsed,
                                output_shapes: tensors.iter().map(|t| t.shape.clone()).collect(),
                            });
                        }
                    }

                    let provider = if native_f16 { "CPU(f16)" } else { "CPU" };
                    self.write_node_outputs(node, provider, tensors, state, resolved)?;
                }

                // Decrement ref counts for the CPU nodes in this group.  Accelerated
                // and serially-dispatched nodes were already decremented.
                for &idx in &cpu_nodes {
                    self.decrement_refs_state(
                        &self.sorted_nodes[idx],
                        state,
                        ref_counts,
                        output_set,
                    );
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
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Result<(), OnnxError> {
        self.run_sequential_inner(state, ref_counts, output_set, resolved)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::execution_providers::select_accelerator;
    use crate::graph::{Attributes, Graph};
    use crate::{OptLevel, SessionBuilder};

    // ── helpers ─────────────────────────────────────────────────────────────

    /// Every compiled-in accelerator has a live device context.
    ///
    /// This is the whole point of [`LiveContexts`]: it lets the routing rules be
    /// tested as if a CUDA card and a DirectML device were present, on CI hardware
    /// that has neither.
    fn all_live() -> LiveContexts {
        LiveContexts {
            #[cfg(feature = "cuda")]
            cuda: true,
            #[cfg(feature = "directml")]
            dml: true,
            #[cfg(feature = "gpu")]
            gpu: true,
        }
    }

    /// Only the DirectML context is live.
    ///
    /// Spelled field-by-field rather than with `..Default::default()` because in a
    /// `directml`-only build `dml` is the struct's *sole* field, and a struct-update
    /// over a fully-specified literal is a clippy error.
    #[cfg(feature = "directml")]
    fn only_dml_live() -> LiveContexts {
        LiveContexts {
            #[cfg(feature = "cuda")]
            cuda: false,
            dml: true,
            #[cfg(feature = "gpu")]
            gpu: false,
        }
    }

    fn node(name: &str, op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs: Attributes::default(),
        }
    }

    /// `Auto` with a zero threshold: the most permissive placement there is.  Any
    /// node that stays on the CPU under *this* stays on the CPU because no backend
    /// implements it — not because it was too small.
    fn auto_everything() -> OpPlacement {
        OpPlacement::Auto {
            gpu_threshold_bytes: 0,
        }
    }

    /// A `Reshape` shape-initializer, as ONNX stores it (integers in an f32 buffer).
    fn shape_initializer(dims: &[usize]) -> Tensor {
        Tensor::new(dims.iter().map(|&d| d as f32).collect(), vec![dims.len()])
    }

    // ── Bug 1: the accelerator gate claimed every node ──────────────────────

    /// **Regression: `is_gpu_eligible_node` returned `true` for every node the
    /// moment a CUDA or DirectML context existed.**
    ///
    /// The old gate was literally `if self.cuda.is_some() { return true; }` — no op
    /// check at all — while `OpKind` has ~166 variants and CUDA implements 40 of
    /// them (DirectML, 15).  So merely *owning* a CUDA context dragged every node in
    /// the graph out of `par_iter` and into the serial phase, to be probed one at a
    /// time on the main thread only to be declined and handed back to the CPU.
    ///
    /// An op **no** backend has a kernel for must run *fully concurrently* even
    /// with every accelerator context live — which is exactly what `all_live()`
    /// asserts here.
    ///
    /// The exemplars are derived rather than named; see
    /// [`crate::execution_providers::ops_no_backend_implements`]. `Reshape` was
    /// this test's original subject (hence its old name) and was the sharpest
    /// case while it was absent from all three dispatch tables — but
    /// `oxionnx-cuda` has since grown a shape-op arm covering
    /// `Reshape`/`Squeeze`/`Unsqueeze`/`Flatten`, so it is now legitimately
    /// planned onto CUDA and can no longer stand in for "nothing claims this".
    /// The regression itself is unchanged: the gate must consult the op, not
    /// merely the presence of a context.
    #[test]
    fn a_cpu_only_op_is_never_planned_onto_an_accelerator_even_with_every_context_live() {
        for op in crate::execution_providers::ops_no_backend_implements() {
            let plan = plan_from_placement(&op, 1 << 24, &auto_everything(), all_live());
            assert!(
                plan.is_empty(),
                "no backend implements {op:?}; it must stay on the rayon CPU path, got {plan:?}",
            );
        }
    }

    /// The end-to-end shape of the same regression: several independent `Reshape`
    /// nodes at one topological depth, executed with the parallel runner and the
    /// most permissive placement policy.
    ///
    /// All four land in the same depth group and must come back with correct,
    /// non-crossed outputs. (When this was written no backend claimed `Reshape`,
    /// so the four also provably shared one `par_iter` pass; `oxionnx-cuda`'s
    /// shape-op arm now claims it, so what survives here is the concurrency-
    /// correctness half. The routing-plan half is pinned by
    /// `a_cpu_only_op_is_never_planned_onto_an_accelerator_even_with_every_context_live`
    /// above, on an op that really is unclaimed.)
    #[test]
    fn several_independent_reshape_nodes_execute_concurrently_and_correctly() {
        let mut weights = HashMap::new();
        weights.insert("shape_2x3".to_string(), shape_initializer(&[2, 3]));
        weights.insert("shape_3x2".to_string(), shape_initializer(&[3, 2]));
        weights.insert("shape_6x1".to_string(), shape_initializer(&[6, 1]));
        weights.insert("shape_1x6".to_string(), shape_initializer(&[1, 6]));

        let graph = Graph {
            nodes: vec![
                node("r0", OpKind::Reshape, &["x", "shape_2x3"], &["o0"]),
                node("r1", OpKind::Reshape, &["x", "shape_3x2"], &["o1"]),
                node("r2", OpKind::Reshape, &["x", "shape_6x1"], &["o2"]),
                node("r3", OpKind::Reshape, &["x", "shape_1x6"], &["o3"]),
            ],
            input_names: vec!["x".to_string()],
            output_names: ["o0", "o1", "o2", "o3"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            ..Default::default()
        };

        let session = SessionBuilder::new()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(true)
            .with_op_placement(auto_everything())
            .build_from_graph(graph, weights)
            .expect("build reshape session");

        // All four are at depth 0 and share the single graph input.
        let depths = Session::compute_node_depths(&session.sorted_nodes, &session.weights);
        let groups = Session::group_by_depth(&depths);
        assert_eq!(
            groups[0].len(),
            4,
            "all four Reshape nodes must sit at one depth, or this test proves nothing",
        );

        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let outputs = session
            .run_one("x", Tensor::new(data.clone(), vec![6]))
            .expect("parallel Reshape run");

        for (name, shape) in [
            ("o0", vec![2, 3]),
            ("o1", vec![3, 2]),
            ("o2", vec![6, 1]),
            ("o3", vec![1, 6]),
        ] {
            let t = outputs
                .get(name)
                .unwrap_or_else(|| panic!("{name} must be present"));
            assert_eq!(t.shape, shape, "{name} shape");
            assert_eq!(t.data, data, "{name} data");
        }
    }

    /// A 16-byte bias-add must not make a round trip to a discrete GPU — and,
    /// just as importantly, must not be pulled out of `par_iter` in order to be
    /// offered one.
    ///
    /// This is why the gate defers to `decide_placement` rather than checking op
    /// support alone: `Add` *is* implemented by every backend, so a support-only
    /// gate would happily serialise it.
    #[test]
    fn a_tiny_add_is_not_planned_onto_an_accelerator() {
        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 1 << 30,
        };
        assert!(
            plan_from_placement(&OpKind::Add, 16, &placement, all_live()).is_empty(),
            "a 16-byte tensor must not be shipped across PCIe, nor serialised to be offered",
        );
        // MatMul is the most GPU-friendly op there is; the threshold still binds it.
        assert!(plan_from_placement(&OpKind::MatMul, 65_536, &placement, all_live()).is_empty(),);
    }

    /// The default placement (`CpuOnly`) accelerates nothing, whatever is live.
    #[test]
    fn cpu_only_placement_plans_nothing() {
        for op in [OpKind::MatMul, OpKind::Add, OpKind::Relu, OpKind::Conv] {
            assert!(
                plan_from_placement(&op, usize::MAX, &OpPlacement::CpuOnly, all_live()).is_empty(),
                "CpuOnly must never plan an accelerator for {op:?}",
            );
        }
    }

    /// A provider whose context is absent is not planned, even when it implements
    /// the op — otherwise the node would be serialised for a dispatch that cannot
    /// happen.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn a_provider_with_no_live_context_is_not_planned() {
        let plan = plan_from_placement(
            &OpKind::Add,
            1 << 20,
            &auto_everything(),
            LiveContexts::default(), // nothing is live
        );
        assert!(
            plan.is_empty(),
            "with no device context, Add belongs on the rayon CPU path, got {plan:?}",
        );
    }

    /// With every context live, an op every backend implements is planned onto the
    /// full priority chain, highest first.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn a_universally_supported_op_is_planned_onto_the_whole_chain() {
        let plan = plan_from_placement(&OpKind::Add, 1 << 20, &auto_everything(), all_live());
        assert_eq!(
            plan,
            accelerator_priority_chain(),
            "Add is implemented by CUDA, DirectML and wgpu alike",
        );
        assert_eq!(
            plan.first().copied(),
            select_accelerator(&OpKind::Add),
            "the head of the plan is the crate-wide preferred accelerator",
        );
    }

    /// The chain in this module must agree with the crate's single source of truth
    /// for provider priority.
    #[test]
    fn priority_chain_agrees_with_select_accelerator() {
        let chain = accelerator_priority_chain();
        // `Add` is implemented by every backend, so `select_accelerator` returns the
        // highest-priority compiled-in provider — which must head the chain.
        assert_eq!(chain.first().copied(), select_accelerator(&OpKind::Add));

        #[cfg(feature = "cuda")]
        assert_eq!(chain.first().copied(), Some(ProviderKind::Cuda));
        #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "directml")))]
        assert!(chain.is_empty(), "no accelerator is compiled in");
    }

    /// CUDA has a real `Conv` kernel (`conv::cuda_conv` dispatches directly to
    /// `oxicuda-dnn`'s `Conv1x1` / `DepthwiseConv` / `ImplicitGemmConv`), so a
    /// `Conv` must be planned onto it, at the head of the chain.  All three
    /// accelerators implement `Conv`, so under `Auto` the plan is the full
    /// priority chain, unfiltered: CUDA, then DirectML, then wgpu.
    ///
    /// Planning several providers is the point of the chain — a node CUDA
    /// *declines* at dispatch time (asymmetric `pads`, say) falls through to the
    /// next entry rather than straight to the CPU.
    ///
    /// This test asserted the exact opposite until CUDA gained a working
    /// convolution; it is inverted rather than deleted because it is the parallel
    /// path's copy of the placement decision that matters most for the
    /// convolution-dominated models this engine runs.
    #[cfg(feature = "cuda")]
    #[test]
    fn conv_is_planned_onto_cuda_first() {
        let plan = plan_from_placement(&OpKind::Conv, 1 << 20, &auto_everything(), all_live());
        assert_eq!(
            plan.first().copied(),
            Some(ProviderKind::Cuda),
            "CUDA implements Conv and heads the priority chain; it must be tried first, got \
             {plan:?}",
        );
        // Every accelerator supports Conv → nothing is filtered out of the chain.
        #[cfg(all(feature = "gpu", feature = "directml"))]
        assert_eq!(
            plan,
            vec![
                ProviderKind::Cuda,
                ProviderKind::DirectMl,
                ProviderKind::Gpu
            ]
        );
        #[cfg(all(feature = "gpu", not(feature = "directml")))]
        assert_eq!(plan, vec![ProviderKind::Cuda, ProviderKind::Gpu]);
        #[cfg(all(feature = "directml", not(feature = "gpu")))]
        assert_eq!(plan, vec![ProviderKind::Cuda, ProviderKind::DirectMl]);
        #[cfg(all(not(feature = "gpu"), not(feature = "directml")))]
        assert_eq!(plan, vec![ProviderKind::Cuda]);
    }

    // ── Bug 3: `self.providers` was ignored on the parallel path ────────────

    /// **Regression: `.with_provider_kinds([DirectMl, Cpu]).with_parallel_execution(true)`
    /// silently never ran DirectML.**
    ///
    /// `self.providers` appeared nowhere in `parallel.rs`; the parallel runner fell
    /// back to `op_placement`, whose default is `CpuOnly`, so the explicitly
    /// requested provider was never even offered the node.  The same session run
    /// sequentially *did* use DirectML — a silent, output-invisible divergence
    /// between the two execution paths.
    ///
    /// Note the placement here is `CpuOnly` (the default): an explicit provider list
    /// must override it, exactly as it does in `try_provider_list_dispatch`.
    #[cfg(feature = "directml")]
    #[test]
    fn an_explicit_directml_provider_list_is_honoured_on_the_parallel_path() {
        let live = only_dml_live();
        let plan = plan_from_provider_list(
            &OpKind::MatMul,
            1 << 20,
            &[ProviderKind::DirectMl, ProviderKind::Cpu],
            live,
        );
        assert_eq!(
            plan,
            vec![ProviderKind::DirectMl],
            "an explicitly requested provider must be planned even under CpuOnly placement",
        );

        // ...and the op-support filter still applies: DirectML has no Reshape kernel.
        assert!(plan_from_provider_list(
            &OpKind::Reshape,
            1 << 20,
            &[ProviderKind::DirectMl, ProviderKind::Cpu],
            live,
        )
        .is_empty());
    }

    /// **Regression: `.with_provider_kinds([Cpu]).with_parallel_execution(true)` ran
    /// an accelerator anyway.**
    ///
    /// The parallel path had never heard of the provider list, so a user who pinned
    /// CPU still had nodes dispatched to whatever context happened to exist.
    /// `ProviderKind::Cpu` is a **terminal sentinel**: it ends the list, and nothing
    /// after it is ever consulted.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn a_cpu_provider_list_pins_every_node_to_the_cpu() {
        for op in [OpKind::MatMul, OpKind::Add, OpKind::Relu] {
            assert!(
                plan_from_provider_list(&op, 1 << 20, &[ProviderKind::Cpu], all_live()).is_empty(),
                "an explicit CPU pin must keep {op:?} off every accelerator",
            );
        }

        // Cpu is terminal, not a fallback appended at the end: providers listed
        // after it are unreachable.
        let chain = accelerator_priority_chain();
        let accelerator = chain
            .first()
            .copied()
            .expect("at least one accelerator is compiled in");
        let mut list = vec![ProviderKind::Cpu];
        list.push(accelerator);
        assert!(
            plan_from_provider_list(&OpKind::MatMul, 1 << 20, &list, all_live()).is_empty(),
            "[Cpu, {accelerator:?}] must never reach {accelerator:?}",
        );

        // Reversed, the accelerator comes first and is planned.
        assert_eq!(
            plan_from_provider_list(
                &OpKind::MatMul,
                1 << 20,
                &[accelerator, ProviderKind::Cpu],
                all_live()
            ),
            vec![accelerator],
        );
    }

    /// An empty provider list is the legacy default and must not disturb the
    /// placement heuristic.
    #[test]
    fn an_empty_provider_list_plans_nothing() {
        assert!(plan_from_provider_list(&OpKind::MatMul, 1 << 20, &[], all_live()).is_empty());
    }

    // ── the explicit-provider-list dispatch floor ───────────────────────────

    /// **The decision, pinned on the parallel path.**
    ///
    /// An explicit provider list overrides `op_placement` — its
    /// `gpu_threshold_bytes` is never consulted — but it does **not** override the
    /// hard `MIN_GPU_DISPATCH_BYTES` floor.  `.with_provider_kinds([DirectMl, Cpu])`
    /// used to ship a 16-byte tensor across PCIe while `OpPlacement::Auto` correctly
    /// kept it on the CPU; it no longer does.
    ///
    /// On *this* path the old behaviour lost twice: the node was pulled out of
    /// `par_iter` (GPU contexts are not thread-safe, so a planned node is dispatched
    /// serially) *and* paid a ~20 µs round trip to replace ~4 ns of arithmetic.
    ///
    /// The predicate is `Session::provider_list_clears_dispatch_floor`, which
    /// `Session::try_provider_list_dispatch` on the sequential path calls too — the
    /// two paths agree by construction, not by coincidence.  See that function for
    /// the full argument.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn an_explicit_provider_list_still_honours_the_hard_dispatch_floor() {
        use crate::execution_providers::MIN_GPU_DISPATCH_BYTES;

        let chain = accelerator_priority_chain();
        let accelerator = chain
            .first()
            .copied()
            .expect("at least one accelerator is compiled in");
        let list = [accelerator, ProviderKind::Cpu];

        // A [1, 4] f32 bias-add: 16 bytes.  Not planned, at any explicit pin.
        assert!(
            plan_from_provider_list(&OpKind::Add, 16, &list, all_live()).is_empty(),
            "a 16-byte tensor must not cross PCIe just because {accelerator:?} was listed",
        );
        assert!(
            plan_from_provider_list(&OpKind::Add, MIN_GPU_DISPATCH_BYTES - 1, &list, all_live())
                .is_empty(),
            "the floor is exclusive below MIN_GPU_DISPATCH_BYTES",
        );

        // From one page upwards the pin is honoured...
        assert_eq!(
            plan_from_provider_list(&OpKind::Add, MIN_GPU_DISPATCH_BYTES, &list, all_live()),
            vec![accelerator],
            "the floor is inclusive at MIN_GPU_DISPATCH_BYTES",
        );

        // ...and honoured *far* below `Auto`'s 64 KiB default threshold, which is
        // exactly what an explicit list is supposed to buy over the heuristic.  A
        // 4 KiB `Add` under `Auto`'s default would stay on the CPU; pinned, it does
        // not.  (That the floor stays below 64 KiB is itself a compile-time
        // invariant, in `execution_providers.rs`.)
        assert!(plan_from_placement(
            &OpKind::Add,
            MIN_GPU_DISPATCH_BYTES,
            &OpPlacement::Auto {
                gpu_threshold_bytes: 65_536,
            },
            all_live(),
        )
        .is_empty());
    }

    /// The sequential and parallel provider-list paths must apply *the same* floor.
    /// They call one function, so this is a tautology by construction — which is the
    /// point: it fails loudly the day someone reintroduces a second copy.
    #[test]
    fn both_execution_paths_share_one_provider_list_floor_predicate() {
        for bytes in [0usize, 16, 4095, 4096, 65_536, usize::MAX] {
            let planned =
                !plan_from_provider_list(&OpKind::Add, bytes, &[ProviderKind::Cpu], all_live())
                    .is_empty();
            // `[Cpu]` plans nothing regardless, so the interesting equality is the
            // predicate itself — the exact call both paths make.
            assert!(!planned);
            assert_eq!(
                Session::provider_list_clears_dispatch_floor(bytes),
                bytes >= crate::execution_providers::MIN_GPU_DISPATCH_BYTES,
            );
        }
    }

    /// A CPU-pinned session must still produce correct results through the parallel
    /// runner — the pin changes routing, never numerics.
    #[test]
    fn a_cpu_pinned_parallel_session_still_computes_correctly() {
        let graph = Graph {
            nodes: vec![
                node("a", OpKind::Relu, &["x"], &["out_a"]),
                node("b", OpKind::Relu, &["y"], &["out_b"]),
            ],
            input_names: vec!["x".to_string(), "y".to_string()],
            output_names: vec!["out_a".to_string(), "out_b".to_string()],
            ..Default::default()
        };
        let session = SessionBuilder::new()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(true)
            .with_provider_kinds([ProviderKind::Cpu])
            .build_from_graph(graph, HashMap::new())
            .expect("build cpu-pinned session");
        assert_eq!(session.providers, vec![ProviderKind::Cpu]);

        let mut inputs = HashMap::new();
        inputs.insert("x", Tensor::new(vec![-1.0, 2.0], vec![2]));
        inputs.insert("y", Tensor::new(vec![3.0, -4.0], vec![2]));
        let outputs = session.run(&inputs).expect("run");

        assert_eq!(outputs.get("out_a").expect("out_a").data, vec![0.0, 2.0]);
        assert_eq!(outputs.get("out_b").expect("out_b").data, vec![3.0, 0.0]);
    }

    // ── Bug 2: results were matched back to nodes by name ───────────────────

    /// **Regression: silent output corruption when two nodes at one depth share a
    /// name.**
    ///
    /// The write phase used to recover the node with
    /// `cpu_only_nodes.iter().find(|n| n.name == node_name)`.  `NodeProto.name` is
    /// *optional* in the ONNX spec — many exporters emit `""` — and nothing anywhere
    /// requires it to be unique.  With two same-named nodes at one topological
    /// depth, `find` returned the **first** for both results: node B's tensors were
    /// written under node A's output names, and node B's outputs were never written
    /// at all.  No error, no warning; just wrong numbers, or a `TensorNotFound`
    /// pointing at an innocent downstream node.
    ///
    /// The two nodes here compute *different* values from *different* inputs, so a
    /// cross-write is unmissable: under the bug `out_b` is absent entirely and
    /// `out_a` holds `out_b`'s data.
    #[test]
    fn two_identically_named_nodes_at_one_depth_do_not_cross_write_their_outputs() {
        for shared_name in ["", "duplicate", "Relu"] {
            let graph = Graph {
                nodes: vec![
                    node(shared_name, OpKind::Relu, &["x"], &["out_a"]),
                    node(shared_name, OpKind::Relu, &["y"], &["out_b"]),
                ],
                input_names: vec!["x".to_string(), "y".to_string()],
                output_names: vec!["out_a".to_string(), "out_b".to_string()],
                ..Default::default()
            };

            let session = SessionBuilder::new()
                .with_optimization_level(OptLevel::None)
                .with_parallel_execution(true)
                .build_from_graph(graph, HashMap::new())
                .expect("build duplicate-name session");

            let depths = Session::compute_node_depths(&session.sorted_nodes, &session.weights);
            let groups = Session::group_by_depth(&depths);
            assert_eq!(
                groups[0].len(),
                2,
                "both nodes must share a depth, or the name-collision path is not exercised",
            );

            let mut inputs = HashMap::new();
            // Deliberately different: a cross-write cannot hide behind equal values.
            inputs.insert("x", Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]));
            inputs.insert("y", Tensor::new(vec![10.0, -20.0, 30.0, -40.0], vec![4]));
            let outputs = session
                .run(&inputs)
                .unwrap_or_else(|e| panic!("run with node name {shared_name:?} failed: {e}"));

            let out_a = outputs
                .get("out_a")
                .unwrap_or_else(|| panic!("out_a missing for node name {shared_name:?}"));
            let out_b = outputs
                .get("out_b")
                .unwrap_or_else(|| panic!("out_b missing for node name {shared_name:?}"));

            assert_eq!(
                out_a.data,
                vec![0.0, 2.0, 0.0, 4.0],
                "out_a was written from the wrong node (name {shared_name:?})",
            );
            assert_eq!(
                out_b.data,
                vec![10.0, 0.0, 30.0, 0.0],
                "out_b was written from the wrong node (name {shared_name:?})",
            );
        }
    }

    /// The same collision, widened: five same-named nodes at one depth, each with a
    /// distinct input and a distinct expected output.  With name-matching, four of
    /// the five outputs never appear.
    #[test]
    fn five_empty_named_nodes_at_one_depth_each_write_their_own_output() {
        const N: usize = 5;
        let nodes: Vec<Node> = (0..N)
            .map(|i| {
                node(
                    "",
                    OpKind::Relu,
                    &[&format!("in{i}")],
                    &[&format!("out{i}")],
                )
            })
            .collect();
        let graph = Graph {
            nodes,
            input_names: (0..N).map(|i| format!("in{i}")).collect(),
            output_names: (0..N).map(|i| format!("out{i}")).collect(),
            ..Default::default()
        };

        let session = SessionBuilder::new()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(true)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let owned: Vec<(String, Tensor)> = (0..N)
            .map(|i| {
                let v = i as f32 + 1.0;
                (format!("in{i}"), Tensor::new(vec![-v, v], vec![2]))
            })
            .collect();
        let inputs: HashMap<&str, Tensor> =
            owned.iter().map(|(k, t)| (k.as_str(), t.clone())).collect();

        let outputs = session.run(&inputs).expect("run");
        assert_eq!(outputs.len(), N, "every node must have written its output");
        for i in 0..N {
            let v = i as f32 + 1.0;
            let name = format!("out{i}");
            let t = outputs
                .get(&name)
                .unwrap_or_else(|| panic!("{name} missing"));
            assert_eq!(t.data, vec![0.0, v], "{name} came from the wrong node");
        }
    }

    // ── Bug 4: `zip` write-back ─────────────────────────────────────────────

    /// The parallel CPU write-back now routes through `write_node_outputs`, so a
    /// node whose outputs disagree with shape inference is rejected loudly instead
    /// of corrupting the rest of the graph.  Nothing here should *trigger* that on a
    /// healthy graph: this pins the happy path, i.e. that the extra validation does
    /// not reject legitimate multi-output nodes at a shared depth.
    #[test]
    fn the_validated_write_back_accepts_a_healthy_multi_node_depth() {
        let graph = Graph {
            nodes: vec![
                node("add", OpKind::Add, &["x", "y"], &["s"]),
                node("mul", OpKind::Mul, &["x", "y"], &["p"]),
                node("sub", OpKind::Sub, &["x", "y"], &["d"]),
            ],
            input_names: vec!["x".to_string(), "y".to_string()],
            output_names: vec!["s".to_string(), "p".to_string(), "d".to_string()],
            ..Default::default()
        };
        let session = SessionBuilder::new()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(true)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let mut inputs = HashMap::new();
        inputs.insert("x", Tensor::new(vec![3.0, 4.0], vec![2]));
        inputs.insert("y", Tensor::new(vec![1.0, 2.0], vec![2]));
        let outputs = session.run(&inputs).expect("run");

        assert_eq!(outputs.get("s").expect("s").data, vec![4.0, 6.0]);
        assert_eq!(outputs.get("p").expect("p").data, vec![3.0, 8.0]);
        assert_eq!(outputs.get("d").expect("d").data, vec![2.0, 2.0]);
    }

    // ── output-size estimate ────────────────────────────────────────────────
    //
    // This module used to carry an `estimate_output_bytes` shim — delegating to
    // `Session::estimate_output_bytes` under `cfg(gpu)` and reimplementing it
    // otherwise, because that function was `#[cfg(feature = "gpu")]` and so absent
    // from the `cuda`-only build.  Its `#[cfg]` is now `any(gpu, cuda, directml)`,
    // the shim is deleted, and the three tests that pinned it
    // (`estimate_output_bytes_prefers_the_resolved_shape`,
    // `estimate_output_bytes_falls_back_to_an_input_tensor`,
    // `estimate_output_bytes_is_zero_when_nothing_is_known`) moved verbatim to
    // `run/dispatch.rs`, next to the one canonical implementation.

    // ── provider labels ─────────────────────────────────────────────────────

    /// The labels must be exactly the ones `write_node_outputs` documents, since
    /// they are what appears in a provider's error messages.
    #[test]
    fn provider_labels_match_the_write_back_contract() {
        assert_eq!(provider_label(ProviderKind::Cpu), "CPU");
        #[cfg(feature = "cuda")]
        assert_eq!(provider_label(ProviderKind::Cuda), "CUDA");
        #[cfg(feature = "directml")]
        assert_eq!(provider_label(ProviderKind::DirectMl), "DirectML");
        #[cfg(feature = "gpu")]
        assert_eq!(provider_label(ProviderKind::Gpu), "wgpu");
    }
}
