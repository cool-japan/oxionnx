//! Sequential (single-threaded) execution path, and the execution-provider
//! dispatch policy that governs it.
//!
//! # Dispatch precedence for one node
//!
//! ```text
//! 1. mixed precision   — `mixed_precision && should_use_f16(op)` claims the node
//!                        outright and no execution provider is offered it at all.
//!                        See `mixed_precision_claims_node`.
//! 2. provider list     — when `Session::providers` is non-empty, the user's
//!                        explicit ordered list decides, terminating at the CPU
//!                        sentinel.  `op_placement` is not consulted at all; the
//!                        one size rule that still binds is the hard
//!                        `MIN_GPU_DISPATCH_BYTES` floor.  See
//!                        `Session::try_provider_list_dispatch` and
//!                        `Session::provider_list_clears_dispatch_floor`.
//! 3. legacy heuristic  — otherwise `decide_placement` decides, and the
//!                        CUDA → DirectML → wgpu → CPU chain is walked from
//!                        whichever provider it names.  See `accelerator_gate`.
//! 4. CPU               — the terminal fallback, always.
//! ```
//!
//! The size every rule in (2) and (3) is stated in comes from the one canonical
//! `Session::estimate_output_bytes` in `run/dispatch.rs`.  This file used to carry
//! a private copy of it, because that function was `#[cfg(feature = "gpu")]` and so
//! did not exist in the `cuda`-only build that needed it most; the `#[cfg]` has
//! been widened to `any(gpu, cuda, directml)` and the copy is gone.

use crate::execution_providers::ProviderKind;
use crate::graph::Node;
// `OpKind` is only named by the accelerator gates (and by this file's tests, which
// import it themselves).  The run loop no longer matches on `OpKind::Unknown` — the
// registry is the gate for unsupported operators, see `super::unsupported_op_error`.
#[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
use crate::graph::OpKind;
use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;
use std::sync::Mutex;

use super::super::types::NodeProfile;
use super::super::Session;
use super::state::SessionRunState;
use super::{OutputSet, RefCounts};

// ── Mixed precision ⟂ execution providers ────────────────────────────────────

/// Does the mixed-precision path claim this node **to the exclusion of every
/// execution provider**?
///
/// # The collision
///
/// DirectML's kernel set is `MatMul | Add | Mul | Relu | Sigmoid`; CUDA's is far
/// larger but likewise covers `Add | Sub | Mul | Div | Relu | Sigmoid | Tanh |
/// Gelu | …`.  `should_use_f16` covers the f16-safe elementwise ops — `Add | Sub
/// | Mul | Div | Relu | Sigmoid | Tanh | …`.  The two sets very nearly coincide:
/// they collide on precisely the ops mixed precision exists to accelerate.
///
/// Because the accelerator gates are evaluated *before* the mixed-precision
/// block, `with_mixed_precision(true)` plus a live accelerator used to mean the
/// node ran on the GPU **in f32** and neither the native-f16 kernel nor the
/// f16 rounding of the outputs ever happened.  Mixed precision was silently a
/// no-op for exactly the ops it was configured for.
///
/// # The precedence, stated explicitly: **`mixed_precision` wins**
///
/// When this returns `true` the node is *never* offered to CUDA, DirectML, wgpu
/// or the explicit provider list — not even to the provider list, which is just
/// as explicit a user request as `with_mixed_precision(true)` is.  Reasons, in
/// order of weight:
///
/// 1. **There is no f16 GPU kernel anywhere in this workspace.**  Every backend
///    (`oxionnx-cuda`, `oxionnx-directml`, `oxionnx-gpu`) uploads, computes and
///    reads back `f32`.  So "accelerator wins" does not trade precision for
///    speed — it silently *discards the precision request altogether* and gives
///    back f32 results.  A flag that is silently ignored is worse than a flag
///    that is slow.
/// 2. **Numerics must not depend on which machine you run on.**  Under
///    "accelerator wins", the same model with the same flag produces f16-rounded
///    values on a CPU-only box and full-f32 values on a box with a GPU.  That is
///    a non-reproducibility bug, and it is invisible: nothing warns.
/// 3. **The cost is bounded and the win is not.**  These are elementwise ops:
///    memory-bound, arithmetic intensity ≈ 1.  They are the *least* profitable
///    ops to ship across PCIe, and giving them up to the CPU costs a
///    bandwidth-bound pass, not an algorithmic blow-up.  `MatMul`, `Gemm` and
///    `Conv` — the ops with the arithmetic intensity that actually justifies a
///    round trip — are **not** in `should_use_f16`, so they keep their
///    accelerator.  The precedence therefore only ever redirects the cheap ops.
///
/// This precedence is pinned by `mixed_precision_beats_every_accelerator_gate`
/// and `matmul_keeps_its_accelerator_under_mixed_precision` in this file's test
/// module.
///
/// # When to revisit
///
/// The moment any backend grows a genuine f16 kernel, this rule should be
/// narrowed to "mixed precision wins *unless* the chosen provider has an f16
/// kernel for this op" — i.e. the predicate gains a provider argument.  It is
/// deliberately a single function so that change has exactly one call site.
pub(super) fn mixed_precision_claims_node(mixed_precision: bool, op_name: &str) -> bool {
    mixed_precision && super::super::mixed_precision::should_use_f16(op_name)
}

// ── Accelerator gating ───────────────────────────────────────────────────────

/// Are the legacy heuristic accelerator gates live for this node *at all*?
///
/// Two things switch the whole legacy chain off before `decide_placement` is
/// even consulted:
///
/// * `provider_list_in_use` — the session was built with `with_provider_kinds()`,
///   so the explicit ordered list has already had its say (and either handled the
///   node or fell through to the CPU sentinel).  Running the legacy gates as well
///   would offer the node to a provider the user did not list.
/// * mixed precision claims the node — see [`mixed_precision_claims_node`].
#[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
fn accelerators_eligible(provider_list_in_use: bool, mixed_precision: bool, op_name: &str) -> bool {
    !provider_list_in_use && !mixed_precision_claims_node(mixed_precision, op_name)
}

/// Position of `provider` in the crate-wide accelerator priority order.
///
/// `Cuda (0) > DirectMl (1) > Gpu (2)`, mirroring
/// `crate::execution_providers::select_accelerator`, which is the single source
/// of truth for that order.  The CPU is not an accelerator and ranks last.
#[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
fn accelerator_rank(provider: ProviderKind) -> u8 {
    match provider {
        ProviderKind::Cpu => u8::MAX,
        #[cfg(feature = "cuda")]
        ProviderKind::Cuda => 0,
        #[cfg(feature = "directml")]
        ProviderKind::DirectMl => 1,
        #[cfg(feature = "gpu")]
        ProviderKind::Gpu => 2,
    }
}

/// Should `candidate` be offered this node on the legacy heuristic path?
///
/// [`crate::execution_providers::decide_placement`] is the **single source of
/// truth**: it — and nothing here — applies `OpPlacement::CpuOnly`, the
/// `Auto { gpu_threshold_bytes }` size threshold, the `Manual` pin, the
/// `MIN_GPU_DISPATCH_BYTES` floor, and each backend's *own* op-support
/// predicate.  This function only decides how the node walks the priority chain
/// *after* `decide_placement` has named its entry point.
///
/// # What this fixes
///
/// The old gates were `self.cuda.is_some() && !matches!(op_placement, CpuOnly)`
/// and `self.dml.is_some() && !matches!(op_placement, CpuOnly)`.  They consulted
/// neither `decide_placement`, nor a size threshold, nor an op filter, with two
/// concrete consequences:
///
/// * `Auto { gpu_threshold_bytes: 1 << 30 }` — "only put enormous tensors on the
///   GPU" — still shipped a `[1, 4]` f32 bias-add (16 bytes) across PCIe:
///   upload → dispatch → fence-wait → readback, a ~20 µs fixed floor, to replace
///   ~4 ns of f32 addition.  On a bias-heavy graph that is a catastrophic
///   slowdown, and the user had explicitly asked for it not to happen.
/// * `Manual(map)` pinning only `{Conv: Gpu}` still rerouted `MatMul`, `Add`,
///   `Mul`, `Relu` and `Sigmoid` to DirectML, because the DirectML gate never
///   looked at the map at all.
///
/// Both are now impossible: an unpinned or below-threshold node makes
/// `decide_placement` return `ProviderKind::Cpu`, and every gate below is then
/// `false`.
///
/// # The fall-through chain
///
/// `decide_placement` names *one* provider.  But a provider that has a kernel for
/// an op *kind* may still decline an individual node whose configuration is out
/// of range (`Ok(None)`) — `provider_supports_op` is explicitly documented as
/// "necessary, not sufficient", and `select_accelerator` tells callers to "still
/// fall through the chain" in that case.  So a declined node must be able to
/// reach the next-best accelerator, which is what the legacy
/// CUDA → DirectML → wgpu → CPU cascade did.  That is preserved here:
///
/// * **`Auto`** — the session asked for "the best accelerator that can take this
///   op", so a decline falls through to every *lower-priority* accelerator that
///   also has a kernel for the op.  (`decide_placement` returned the
///   highest-priority supporter, so "lower priority" and "not the chosen one"
///   coincide.)  The size threshold has already been cleared by
///   `decide_placement`, so it binds the whole chain, not just its head.
/// * **`Manual`** — the user pinned *this* op to *that* provider.  Quietly
///   rerouting it to a different accelerator when the pinned one declines would
///   re-introduce exactly the `{Conv: Gpu}` bug above.  Only the pinned provider
///   is offered the node; the CPU is the sole fallback.
/// * **`CpuOnly`** — no accelerator, ever.
///
/// The chain itself is realised by the *ordering of the dispatch blocks* in
/// `run_sequential_inner`: they run CUDA, then DirectML, then wgpu, and a
/// provider that handles the node `continue`s the loop, so no lower-priority
/// gate is ever reached.  This function is a pure predicate over one candidate.
#[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
fn accelerator_gate(
    candidate: ProviderKind,
    op: &OpKind,
    output_bytes: usize,
    placement_cfg: &crate::execution_providers::OpPlacement,
) -> bool {
    use crate::execution_providers::{decide_placement, provider_supports_op, OpPlacement};

    // The CPU is the terminal fallback, never an accelerator gate.
    if matches!(candidate, ProviderKind::Cpu) {
        return false;
    }

    let chosen = decide_placement(op, output_bytes, placement_cfg);

    // `Cpu` means: CpuOnly, or below the threshold / floor, or no compiled-in
    // backend has a kernel for this op.  Whichever it is, no accelerator runs.
    if matches!(chosen, ProviderKind::Cpu) {
        return false;
    }

    if chosen == candidate {
        return true;
    }

    // Not the chosen provider — the only way it still gets a look is as a
    // fall-through target after a higher-priority provider declined the node,
    // and only `Auto` permits that (see the doc comment).
    if !matches!(placement_cfg, OpPlacement::Auto { .. }) {
        return false;
    }

    accelerator_rank(candidate) > accelerator_rank(chosen) && provider_supports_op(candidate, op)
}

impl Session {
    /// The model's `ai.onnx` (default-domain) opset version.
    ///
    /// ONNX declares opsets per domain; the default domain is spelled either `""`
    /// (what every real exporter emits) or `"ai.onnx"` (legal, and what a
    /// hand-written `ModelProto` may carry), so both are accepted and the highest
    /// of them wins.  Versions ≤ 0 are ignored: they are malformed, and treating
    /// one as "opset 0" would silently select the legacy contract for every
    /// version-sensitive operator.
    ///
    /// Falls back to [`oxionnx_core::operator::DEFAULT_OPSET`] for a model that
    /// declares nothing — including every `Session::from_graph` graph, which has
    /// no `ModelProto` to declare it in.
    pub(crate) fn model_opset(&self) -> i64 {
        self.metadata
            .opset_imports
            .iter()
            .filter(|(domain, version)| {
                *version > 0 && (domain.is_empty() || domain.as_str() == "ai.onnx")
            })
            .map(|(_, version)| *version)
            .max()
            .unwrap_or(oxionnx_core::operator::DEFAULT_OPSET)
    }

    /// Bind the session's operator registry to this model's opset, so that every
    /// `OpContext` built during the run reports it through `OpContext::opset()`.
    ///
    /// Called at the top of each execution path rather than once at load time
    /// because `dispatch_node` — not this file — is what constructs the contexts,
    /// and it can only reach model-level state through the registry it already
    /// passes down.  The store is idempotent and costs one relaxed atomic write
    /// per run.
    pub(crate) fn bind_registry_opset(&self) {
        self.registry.set_model_opset(self.model_opset());
    }

    // GREP_GUARD: all intermediates writes must go through dispatch_node /
    // Session::write_node_outputs / SessionRunState::insert.  In particular, no
    // execution-provider write-back may use the raw
    // `node.outputs.iter().zip(results)` idiom: `zip` silently truncates when a
    // provider returns fewer tensors than the node has outputs, and nothing
    // validates the returned shapes.  `write_node_outputs` closes both holes and
    // is all-or-nothing.

    /// Sequential execution path using `SessionRunState` for buffer-reuse-aware
    /// intermediate storage.
    ///
    /// `resolved` is **this run's** shape map, computed by
    /// `Session::resolve_run_shapes` and owned by the caller.  It used to be read
    /// out of the session-wide mutex here, in a lock acquisition separate from the
    /// one that wrote it, which let two concurrent runs with different batch sizes
    /// execute against each other's shapes.
    pub(crate) fn run_sequential_inner(
        &self,
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Result<(), OnnxError> {
        // Version-sensitive operators read this off their `OpContext`; it must be
        // bound before the first node executes.
        self.bind_registry_opset();

        for node in &self.sorted_nodes {
            // NOTE: there is deliberately no `OpKind::Unknown => continue` here.
            // The registry lookup below is the single gate for "this engine
            // cannot run that operator"; see `super::unsupported_op_error`.
            let op_name = node.op.as_str();

            // ── Mixed precision ⟂ execution providers ─────────────────────────
            // EXPLICIT PRECEDENCE: mixed precision WINS over every execution
            // provider — the legacy CUDA/DirectML/wgpu gates below *and* the
            // explicit provider list.  No backend in this workspace has an f16
            // kernel, so dispatching an f16-safe op to one does not trade
            // precision for speed: it silently throws the precision request away
            // and returns f32.  Full rationale on `mixed_precision_claims_node`.
            let mixed_precision_node = mixed_precision_claims_node(self.mixed_precision, op_name);

            let provider_list_in_use = !self.providers.is_empty();

            // Are the legacy heuristic gates live?  (They are not when an explicit
            // provider list has already had its say, nor when mixed precision has
            // claimed the node.)
            #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
            let accel_eligible =
                accelerators_eligible(provider_list_in_use, self.mixed_precision, op_name);

            // The payload every size rule is stated in — `Auto`'s
            // `gpu_threshold_bytes`, the `Manual` floor (both inside
            // `decide_placement`), and the explicit-provider-list floor
            // (`Session::provider_list_clears_dispatch_floor`).  Both dispatch
            // paths below need it, so it is computed once, and only when one of
            // them could actually take the node.
            //
            // `#[cfg]`: `Session::estimate_output_bytes` is compiled only when an
            // accelerator is.  With none compiled in, `ProviderKind::Cpu` is the
            // enum's sole variant, so nothing can leave the CPU and the size is
            // unobservable — `0` keeps the provider-list floor closed, which is the
            // same answer by a shorter route.
            #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
            let output_bytes = if accel_eligible || (provider_list_in_use && !mixed_precision_node)
            {
                Self::estimate_output_bytes(node, state.as_map(), &self.weights, resolved)
            } else {
                0
            };
            #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "directml")))]
            let output_bytes: usize = 0;

            // ── Provider-list dispatch path ───────────────────────────────────
            // When the session was built with `with_provider_kinds()`, iterate
            // the ordered provider list.  The first provider that returns
            // `Some(results)` wins; CPU is always the implicit terminal fallback
            // (handled by the normal dispatch path below).
            if provider_list_in_use && !mixed_precision_node {
                if let Some(dispatched) = self.try_provider_list_dispatch(
                    node,
                    output_bytes,
                    state,
                    ref_counts,
                    output_set,
                    resolved,
                )? {
                    if dispatched {
                        continue;
                    }
                    // dispatched == false means all explicit providers returned None
                    // (or the node is below the dispatch floor); fall through to the
                    // CPU dispatch path below.
                }
            }

            // ── Legacy heuristic dispatch path ────────────────────────────────
            // Used when `self.providers` is empty (backward-compatible default).
            //
            // Every gate below is `decide_placement`'s decision, full stop: it
            // applies CpuOnly, the `Auto` size threshold, the `Manual` pin, the
            // `MIN_GPU_DISPATCH_BYTES` floor and each backend's own op-support
            // predicate.  See `accelerator_gate`.
            //
            // When no hardware-acceleration feature is active, read op_placement to
            // satisfy the compiler (field is always valid, just unused at runtime).
            #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "directml")))]
            let _ = &self.op_placement;

            // CUDA — highest priority accelerator.
            #[cfg(feature = "cuda")]
            if accel_eligible
                && accelerator_gate(
                    ProviderKind::Cuda,
                    &node.op,
                    output_bytes,
                    &self.op_placement,
                )
                && self.dispatch_to_cuda(node, state, resolved)?
            {
                self.decrement_refs_state(node, state, ref_counts, output_set);
                continue;
            }

            // DirectML — Windows D3D12 GPU, ranked above wgpu on Windows.
            // Reached either because `decide_placement` chose it outright, or
            // because CUDA was chosen and declined this particular node.
            #[cfg(feature = "directml")]
            if accel_eligible
                && accelerator_gate(
                    ProviderKind::DirectMl,
                    &node.op,
                    output_bytes,
                    &self.op_placement,
                )
                && self.dispatch_to_directml(node, state, resolved)?
            {
                self.decrement_refs_state(node, state, ref_counts, output_set);
                continue;
            }

            // wgpu — lowest-priority accelerator, last stop before the CPU.
            #[cfg(feature = "gpu")]
            if accel_eligible
                && accelerator_gate(
                    ProviderKind::Gpu,
                    &node.op,
                    output_bytes,
                    &self.op_placement,
                )
                && self.dispatch_to_wgpu(node, state, resolved)?
            {
                self.decrement_refs_state(node, state, ref_counts, output_set);
                continue;
            }

            // ── Mixed precision: native f16 element-wise execution ────────────
            // No native f16 kernel for this op → fall through to normal execution
            // with f16 rounding of the outputs below.
            if mixed_precision_node
                && self.try_native_f16_node(node, state, ref_counts, output_set, resolved)?
            {
                continue;
            }

            let operator = self
                .registry
                .get(op_name)
                .ok_or_else(|| super::unsupported_op_error(node))?;

            let elapsed =
                self.dispatch_node(node, operator, state, ref_counts, output_set, resolved)?;

            // Mixed precision: round outputs to f16 for f16-safe ops without native f16 path.
            // This simulates f16 storage precision for ops that ran in f32.
            if mixed_precision_node {
                self.round_node_outputs_to_f16(node, state);
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

    // ── Mixed precision, shared by both execution paths ─────────────────────
    //
    // `run/parallel.rs` used to read `self.mixed_precision` nowhere at all: a
    // session built with `.with_mixed_precision(true).with_parallel_execution(true)`
    // silently produced full-f32 results, differing from the very same session run
    // sequentially.  Both halves of the policy — the native-f16 kernel and the f16
    // rounding of the outputs — now live here, in one place, and the parallel path
    // calls exactly these, so the two cannot drift apart.

    /// Run `node` through the native f16 elementwise kernel, if one exists.
    ///
    /// Returns `Ok(true)` when the kernel ran and its outputs are committed to
    /// `state` (profiling recorded, references decremented — the caller is done
    /// with the node), `Ok(false)` when there is no native f16 kernel for this op
    /// and the caller must execute it normally and then round the outputs with
    /// [`Session::round_node_outputs_to_f16`].
    ///
    /// Only call this when [`mixed_precision_claims_node`] returned `true`.
    pub(super) fn try_native_f16_node(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved: &HashMap<String, Vec<usize>>,
    ) -> Result<bool, OnnxError> {
        let op_name = node.op.as_str();
        let input_refs: Vec<&Tensor> = node
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
        let Some(f16_result) =
            super::super::mixed_precision::execute_elementwise_f16(op_name, &input_refs)
        else {
            return Ok(false);
        };
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

        // Same validated, all-or-nothing write-back every execution provider
        // uses: a native-f16 kernel is no more trustworthy than a GPU one, and
        // `zip` truncates just as silently here.
        self.write_node_outputs(node, "CPU(f16)", results, state, resolved)?;
        self.decrement_refs_state(node, state, ref_counts, output_set);
        Ok(true)
    }

    /// Round every output `node` wrote into `state` to f16 storage precision.
    ///
    /// This is what makes `with_mixed_precision(true)` observable for an f16-safe
    /// op that has no native f16 kernel: the op ran in f32, and its result is
    /// then squeezed through `half::f16` so the numerics match what an f16
    /// activation buffer would have held.
    pub(super) fn round_node_outputs_to_f16(&self, node: &Node, state: &mut SessionRunState) {
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

    // ── Per-provider offers ─────────────────────────────────────────────────
    //
    // One helper per backend, shared by the legacy heuristic path and the
    // explicit provider-list path so the two can never drift apart.
    //
    // Each returns:
    //   Ok(true)  — the provider executed the node and its outputs are committed
    //               to `state` (validated, all-or-nothing).
    //   Ok(false) — the node was NOT executed; walk on down the chain.  Two very
    //               different things currently collapse into this:
    //                 * DECLINED — no live context, no kernel for this op, or the
    //                   node's configuration is out of the kernel's range
    //                   (`Ok(None)`).  A normal, expected fall-back.
    //                 * FAILED — the backend errored (`Err`).  Abnormal.
    //   Err(_)    — unrecoverable.  Reserved for a provider that returned results
    //               violating the write-back contract (wrong arity, internally
    //               inconsistent tensor, shape disagreeing with shape inference).
    //               Corrupt results must never be laundered into a quiet CPU
    //               fallback: the graph is already poisoned at that point, and a
    //               wrong answer is worse than a slow one.
    //
    // The DECLINED/FAILED collapse is pre-existing and a later wave will separate
    // them.  It is not made worse here, and the failure information is no longer
    // thrown away: it used to be logged behind `#[cfg(debug_assertions)]
    // tracing::debug!`, i.e. invisible in every release build.  It is now a
    // release-visible `tracing::warn!` carrying the provider, node, op kind and
    // the error itself — the exact record that wave will need.

    /// Offer `node` to the CUDA backend.  See the block comment above.
    #[cfg(feature = "cuda")]
    fn dispatch_to_cuda(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<bool, OnnxError> {
        let Some(cuda_ctx) = &self.cuda else {
            // No CUDA context on this machine — declined.
            return Ok(false);
        };

        let started = std::time::Instant::now();
        match oxionnx_cuda::try_cuda_dispatch(node, &self.weights, state.as_map(), cuda_ctx) {
            Ok(Some(results)) => {
                let elapsed = started.elapsed();
                self.commit_provider_results(node, "CUDA", results, elapsed, state, resolved_shapes)
                    .map(|()| true)
            }
            // DECLINED: no kernel for this op, or this node's configuration is out
            // of the kernel's range.  Fall through to the next provider.
            Ok(None) => Ok(false),
            // FAILED — not the same thing as declined, however similarly it is
            // currently treated.  Kept visible in release builds.
            Err(err) => {
                tracing::warn!(
                    provider = "CUDA",
                    op = %node.op.as_str(),
                    node = %node.name,
                    error = %err,
                    "execution provider FAILED (not declined); falling back to the next provider",
                );
                Ok(false)
            }
        }
    }

    /// Offer `node` to the DirectML backend.  See the block comment above.
    #[cfg(feature = "directml")]
    fn dispatch_to_directml(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<bool, OnnxError> {
        let Some(dml_ctx) = &self.dml else {
            // No DirectML context (e.g. any non-Windows target) — declined.
            return Ok(false);
        };

        let started = std::time::Instant::now();
        match oxionnx_directml::try_directml_dispatch(node, &self.weights, state.as_map(), dml_ctx)
        {
            Ok(Some(results)) => {
                let elapsed = started.elapsed();
                self.commit_provider_results(
                    node,
                    "DirectML",
                    results,
                    elapsed,
                    state,
                    resolved_shapes,
                )
                .map(|()| true)
            }
            // DECLINED.
            Ok(None) => Ok(false),
            // FAILED.
            Err(err) => {
                tracing::warn!(
                    provider = "DirectML",
                    op = %node.op.as_str(),
                    node = %node.name,
                    error = %err,
                    "execution provider FAILED (not declined); falling back to the next provider",
                );
                Ok(false)
            }
        }
    }

    /// Offer `node` to the wgpu backend.  See the block comment above.
    ///
    /// Note the asymmetry with CUDA/DirectML, which is deliberate and
    /// pre-existing: `try_gpu_dispatch`'s `Err` has always been *propagated*
    /// rather than swallowed into a CPU fallback, so this helper never reports a
    /// wgpu failure as `Ok(false)`.  That is the stricter contract, and the one
    /// the CUDA and DirectML paths should eventually converge on.
    #[cfg(feature = "gpu")]
    fn dispatch_to_wgpu(
        &self,
        node: &Node,
        state: &mut SessionRunState,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<bool, OnnxError> {
        let Some(gpu_ctx) = &self.gpu else {
            return Ok(false);
        };

        let started = std::time::Instant::now();
        let dispatched = super::super::gpu_dispatch::try_gpu_dispatch(
            node,
            &self.weights,
            state.as_map(),
            gpu_ctx,
        )?;

        let Some(results) = dispatched else {
            tracing::debug!(
                provider = "wgpu",
                op = %node.op.as_str(),
                node = %node.name,
                "execution provider declined the node; falling back",
            );
            return Ok(false);
        };

        let elapsed = started.elapsed();
        self.commit_provider_results(node, "wgpu", results, elapsed, state, resolved_shapes)
            .map(|()| true)
    }

    /// Record a provider's node profile and commit its results through the
    /// validated, all-or-nothing write-back.
    ///
    /// Every execution-provider write-back in this file funnels through here, so
    /// the `zip`-truncation and unvalidated-shape holes described on
    /// [`Session::write_node_outputs`] are closed for all of them at once — and
    /// stay closed, because there is exactly one place left that could reopen
    /// them.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    fn commit_provider_results(
        &self,
        node: &Node,
        provider: &'static str,
        results: Vec<Tensor>,
        elapsed: std::time::Duration,
        state: &mut SessionRunState,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<(), OnnxError> {
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

        self.write_node_outputs(node, provider, results, state, resolved_shapes)
    }

    /// Attempt to dispatch `node` through the ordered provider list stored in
    /// `self.providers`.
    ///
    /// Returns:
    /// - `Ok(Some(true))` — a provider handled the op; caller should `continue`.
    /// - `Ok(Some(false))` — no provider took the node; caller falls through to CPU.
    /// - `Ok(None)` — providers list is empty (should not be called in that case).
    /// - `Err(_)` — an unrecoverable error from a provider.
    ///
    /// CPU is the implicit terminal fallback and is never invoked here; the
    /// caller handles the CPU path after this method returns `Some(false)`.
    /// `ProviderKind::Cpu` appearing *in* the list is a terminal sentinel: it
    /// stops the walk immediately, so anything listed after it is unreachable by
    /// construction.
    ///
    /// # The size floor
    ///
    /// `output_bytes` is checked against
    /// [`Session::provider_list_clears_dispatch_floor`] — the *same* predicate
    /// `run/parallel.rs`'s `plan_from_provider_list` uses, so the two execution
    /// paths cannot disagree — and a node below the floor goes straight to the CPU
    /// without being offered to any listed accelerator.
    ///
    /// An explicit list still *overrides* `self.op_placement` entirely: its
    /// `gpu_threshold_bytes` is never consulted here, so a listed provider claims
    /// every node it has a kernel for from one page upwards, even under the default
    /// `OpPlacement::CpuOnly`.  Only the hard floor binds it, exactly as the same
    /// floor already binds an `OpPlacement::Manual` pin inside `decide_placement`.
    /// The full argument is on [`Session::provider_list_clears_dispatch_floor`].
    #[allow(unused_variables)]
    // Every parameter is consumed by the cfg'd provider
    // arms below.  With no accelerator feature enabled,
    // `ProviderKind::Cpu` is the enum's sole variant and
    // those arms vanish.
    #[allow(clippy::never_loop)] // The loop CAN iterate when GPU/CUDA/DirectML features are enabled;
                                 // without those features, only ProviderKind::Cpu exists and the first
                                 // iteration always returns — this is correct conditional-compilation behaviour.
    fn try_provider_list_dispatch(
        &self,
        node: &Node,
        output_bytes: usize,
        state: &mut SessionRunState,
        ref_counts: &mut RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<Option<bool>, OnnxError> {
        // The hard floor binds every listed accelerator identically, so a node
        // below it resolves to the CPU fallback without walking the list at all —
        // which is exactly what walking it would produce, one skipped provider at a
        // time.
        if !Self::provider_list_clears_dispatch_floor(output_bytes) {
            return Ok(Some(false));
        }

        for provider in &self.providers {
            match provider {
                // CPU is an explicit terminal fallback — signal caller to use CPU path.
                ProviderKind::Cpu => return Ok(Some(false)),

                #[cfg(feature = "cuda")]
                ProviderKind::Cuda => {
                    if self.dispatch_to_cuda(node, state, resolved_shapes)? {
                        self.decrement_refs_state(node, state, ref_counts, output_set);
                        return Ok(Some(true));
                    }
                    // Declined or failed — try the next provider in the list.
                }

                #[cfg(feature = "directml")]
                ProviderKind::DirectMl => {
                    if self.dispatch_to_directml(node, state, resolved_shapes)? {
                        self.decrement_refs_state(node, state, ref_counts, output_set);
                        return Ok(Some(true));
                    }
                    // Declined or failed — try the next provider in the list.
                }

                #[cfg(feature = "gpu")]
                ProviderKind::Gpu => {
                    if self.dispatch_to_wgpu(node, state, resolved_shapes)? {
                        self.decrement_refs_state(node, state, ref_counts, output_set);
                        return Ok(Some(true));
                    }
                    // Declined — try the next provider in the list.  (A wgpu
                    // *failure* propagated out of `dispatch_to_wgpu` as `Err`.)
                }
            }
        }
        // All explicit providers returned None — signal caller to use CPU path.
        Ok(Some(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Graph};
    // Explicit (not via `use super::*`): the top-level `OpKind` import is gated on
    // an accelerator feature, and these tests need it in every build.
    use crate::graph::OpKind;
    use crate::{OptLevel, SessionBuilder};

    // ── helpers ─────────────────────────────────────────────────────────────

    /// The f16-safe elementwise ops that DirectML *also* claims a kernel for.
    /// This is the collision that makes the mixed-precision precedence a real
    /// decision rather than a theoretical one: DirectML's op set is
    /// `MatMul | Add | Mul | Relu | Sigmoid`, and four of those five are here.
    const DIRECTML_F16_COLLISION: [&str; 4] = ["Add", "Mul", "Relu", "Sigmoid"];

    /// Ops with the arithmetic intensity that actually justifies a GPU round
    /// trip.  None of them is f16-safe, so mixed precision never takes them.
    const ACCELERATOR_WORTHY: [&str; 3] = ["MatMul", "Gemm", "Conv"];

    fn single_node_session(op: OpKind, mixed_precision: bool, threshold: usize) -> Session {
        let graph = Graph {
            nodes: vec![Node {
                name: "n0".to_string(),
                op,
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
                attrs: Attributes::default(),
            }],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            ..Default::default()
        };
        SessionBuilder::new()
            .with_optimization_level(OptLevel::None)
            // Pin the sequential path — this file is what is under test.
            .with_parallel_execution(false)
            .with_mixed_precision(mixed_precision)
            .with_op_placement(crate::execution_providers::OpPlacement::Auto {
                gpu_threshold_bytes: threshold,
            })
            .build_from_graph(graph, HashMap::new())
            .expect("build test session")
    }

    // ── mixed-precision precedence (task 3) ─────────────────────────────────
    //
    // PINNED PRECEDENCE: `mixed_precision` WINS over every execution provider.
    // See `mixed_precision_claims_node` for the full rationale.  These tests
    // exist so that flipping the precedence is a deliberate act with a red test
    // suite attached, not an accident of block ordering — which is exactly how
    // the *previous* precedence came about.

    /// The whole reason a precedence has to be chosen at all.
    #[test]
    fn mixed_precision_and_the_accelerators_fight_over_the_same_ops() {
        for op in DIRECTML_F16_COLLISION {
            assert!(
                super::super::super::mixed_precision::should_use_f16(op),
                "{op} must be f16-safe for the collision to exist",
            );
        }
    }

    #[test]
    fn mixed_precision_claims_the_contested_ops() {
        for op in DIRECTML_F16_COLLISION {
            assert!(
                mixed_precision_claims_node(true, op),
                "with mixed precision on, {op} must NOT be offered to any provider",
            );
        }
    }

    /// With the flag off, nothing is claimed and every provider gate is live —
    /// mixed precision must not have a cost when it is not asked for.
    #[test]
    fn mixed_precision_off_claims_nothing() {
        for op in DIRECTML_F16_COLLISION
            .iter()
            .chain(ACCELERATOR_WORTHY.iter())
        {
            assert!(
                !mixed_precision_claims_node(false, op),
                "with mixed precision off, {op} must stay available to the providers",
            );
        }
    }

    /// The precedence is deliberately *narrow*: it only ever redirects cheap
    /// elementwise ops.  The ops whose arithmetic intensity justifies a PCIe
    /// round trip keep their accelerator even under mixed precision.
    #[test]
    fn matmul_keeps_its_accelerator_under_mixed_precision() {
        for op in ACCELERATOR_WORTHY {
            assert!(
                !mixed_precision_claims_node(true, op),
                "{op} is not f16-safe; mixed precision must not take it from the accelerator",
            );
        }
    }

    /// The run loop's composite gate: mixed precision switches the entire legacy
    /// accelerator chain off for the ops it claims.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn mixed_precision_beats_every_accelerator_gate() {
        use crate::execution_providers::{provider_supports_op, OpPlacement};

        // Threshold 0 + a large tensor: every provider that has a kernel would
        // otherwise take these nodes.
        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 0,
        };
        let contested = [OpKind::Add, OpKind::Mul, OpKind::Relu, OpKind::Sigmoid];

        for op in &contested {
            let op_name = op.as_str();

            // Sanity: at least one compiled-in backend really would have claimed
            // it, so this test is not vacuous.
            let mut any_backend_wanted_it = false;
            #[cfg(feature = "cuda")]
            {
                any_backend_wanted_it |= provider_supports_op(ProviderKind::Cuda, op);
            }
            #[cfg(feature = "directml")]
            {
                any_backend_wanted_it |= provider_supports_op(ProviderKind::DirectMl, op);
            }
            #[cfg(feature = "gpu")]
            {
                any_backend_wanted_it |= provider_supports_op(ProviderKind::Gpu, op);
            }
            assert!(
                any_backend_wanted_it,
                "{op_name}: no compiled backend claims this op — the test proves nothing",
            );
            assert!(
                accelerator_gate(highest_priority_accelerator(), op, 1 << 20, &placement,),
                "{op_name}: without mixed precision the accelerator gate must be open",
            );

            // ...and mixed precision shuts the whole chain off anyway.
            assert!(
                !accelerators_eligible(false, true, op_name),
                "{op_name}: mixed precision must close every accelerator gate",
            );
            assert!(
                accelerators_eligible(false, false, op_name),
                "{op_name}: with mixed precision off the gates must reopen",
            );
        }
    }

    /// An explicit provider list also silences the legacy gates — they would
    /// otherwise offer the node to a backend the user did not list.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn an_explicit_provider_list_silences_the_legacy_gates() {
        assert!(!accelerators_eligible(true, false, "Add"));
        assert!(accelerators_eligible(false, false, "Add"));
    }

    /// End-to-end: with mixed precision on, an f16-safe op's output really is
    /// f16-rounded — i.e. the node went down the mixed-precision path and not
    /// down a provider path that would have returned untouched f32.
    #[test]
    fn mixed_precision_output_is_f16_rounded_end_to_end() {
        // 0.1 is not representable in f16; f32 holds it far more precisely.
        let exact = 0.1f32;
        let f16_rounded = half::f16::from_f32(exact).to_f32();
        assert_ne!(exact, f16_rounded, "0.1 must lose precision in f16");

        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        inputs.insert("x", Tensor::new(vec![exact; 8], vec![8]));

        // Threshold 0 under `Auto`: every accelerator gate is as open as it can
        // possibly be, and mixed precision still wins.
        let session = single_node_session(OpKind::Relu, true, 0);
        let out = session.run(&inputs).expect("run with mixed precision");
        let y = out.get("y").expect("output y");
        for &v in &y.data {
            assert_eq!(
                v, f16_rounded,
                "mixed precision must own this node: expected f16-rounded output",
            );
        }

        // Contrast: with the flag off the value survives at full f32 precision.
        let session = single_node_session(OpKind::Relu, false, 0);
        let out = session.run(&inputs).expect("run without mixed precision");
        let y = out.get("y").expect("output y");
        for &v in &y.data {
            assert_eq!(v, exact, "without mixed precision the output must stay f32");
        }
    }

    // ── accelerator gating (task 1) ─────────────────────────────────────────

    /// The accelerator the priority order `Cuda > DirectMl > Gpu` selects for an
    /// op every compiled backend implements (e.g. `Add`).
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    fn highest_priority_accelerator() -> ProviderKind {
        #[cfg(feature = "cuda")]
        {
            ProviderKind::Cuda
        }
        #[cfg(all(not(feature = "cuda"), feature = "directml"))]
        {
            ProviderKind::DirectMl
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "directml"), feature = "gpu"))]
        {
            ProviderKind::Gpu
        }
    }

    /// Every compiled-in accelerator, in priority order.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    fn all_accelerators() -> Vec<ProviderKind> {
        #[cfg(feature = "cuda")]
        let cuda: Option<ProviderKind> = Some(ProviderKind::Cuda);
        #[cfg(not(feature = "cuda"))]
        let cuda: Option<ProviderKind> = None;

        #[cfg(feature = "directml")]
        let directml: Option<ProviderKind> = Some(ProviderKind::DirectMl);
        #[cfg(not(feature = "directml"))]
        let directml: Option<ProviderKind> = None;

        #[cfg(feature = "gpu")]
        let wgpu: Option<ProviderKind> = Some(ProviderKind::Gpu);
        #[cfg(not(feature = "gpu"))]
        let wgpu: Option<ProviderKind> = None;

        // Priority order: Cuda > DirectMl > Gpu.
        [cuda, directml, wgpu].into_iter().flatten().collect()
    }

    /// THE regression test for the bug this rework exists to kill.
    ///
    /// `Auto { gpu_threshold_bytes: 1 << 30 }` says "only put enormous tensors on
    /// the GPU".  The old CUDA and DirectML gates — `ctx.is_some() &&
    /// !matches!(op_placement, CpuOnly)` — consulted no threshold at all, so a
    /// `[1, 4]` f32 bias-add (16 bytes) was still shipped across PCIe: a ~20 µs
    /// round-trip floor replacing ~4 ns of f32 addition.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn a_huge_auto_threshold_keeps_a_bias_add_off_every_accelerator() {
        use crate::execution_providers::OpPlacement;

        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 1 << 30,
        };

        for accel in all_accelerators() {
            // A [1, 4] f32 bias-add: 16 bytes of output.
            assert!(
                !accelerator_gate(accel, &OpKind::Add, 16, &placement),
                "{accel:?}: a 16-byte bias-add must never leave the CPU under a 1 GiB threshold",
            );
            // Right up to the edge of the threshold.
            assert!(
                !accelerator_gate(accel, &OpKind::Add, (1 << 30) - 1, &placement),
                "{accel:?}: the threshold is exclusive below gpu_threshold_bytes",
            );
            // MatMul is the most GPU-friendly op there is; the threshold binds it too.
            assert!(
                !accelerator_gate(accel, &OpKind::MatMul, 65_536, &placement),
                "{accel:?}: gpu_threshold_bytes must bind every provider, not just wgpu",
            );
        }
    }

    /// `CpuOnly` closes every gate, at any size, for any op.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn cpu_only_closes_every_accelerator_gate() {
        use crate::execution_providers::OpPlacement;

        for accel in all_accelerators() {
            for op in [OpKind::MatMul, OpKind::Add, OpKind::Conv, OpKind::Relu] {
                assert!(
                    !accelerator_gate(accel, &op, usize::MAX, &OpPlacement::CpuOnly),
                    "{accel:?}: CpuOnly must close the gate for {op:?}",
                );
            }
        }
    }

    /// The CPU is never an accelerator gate — it is the terminal fallback, and
    /// the dispatch blocks never call it as one.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn the_cpu_is_never_an_accelerator_gate() {
        use crate::execution_providers::OpPlacement;

        for placement in [
            OpPlacement::CpuOnly,
            OpPlacement::Auto {
                gpu_threshold_bytes: 0,
            },
        ] {
            assert!(!accelerator_gate(
                ProviderKind::Cpu,
                &OpKind::Add,
                1 << 20,
                &placement
            ));
        }
    }

    /// Above the threshold, `Auto` opens the highest-priority accelerator that
    /// has a kernel for the op — and, so the CUDA → DirectML → wgpu → CPU cascade
    /// survives a decline, every lower-priority accelerator that also has one.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn auto_opens_the_priority_chain_above_the_threshold() {
        use crate::execution_providers::{provider_supports_op, OpPlacement};

        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 4096,
        };

        // `Add` has a kernel in CUDA, DirectML and wgpu alike.
        assert!(
            accelerator_gate(
                highest_priority_accelerator(),
                &OpKind::Add,
                4096,
                &placement
            ),
            "the threshold is inclusive at gpu_threshold_bytes",
        );

        for accel in all_accelerators() {
            assert_eq!(
                accelerator_gate(accel, &OpKind::Add, 1 << 20, &placement),
                provider_supports_op(accel, &OpKind::Add),
                "{accel:?}: under Auto, every backend with a kernel stays in the chain so a \
                 declining higher-priority backend can fall through to it",
            );
        }
    }

    /// `Auto` consults each backend's *own* op-support predicate.  CUDA has no
    /// `Conv` kernel (`oxionnx_cuda::is_supported_op(Conv) == false`) even though
    /// wgpu does, so a convolution must never be gated to CUDA — the round trip
    /// would be pure waste.
    #[cfg(feature = "cuda")]
    #[test]
    fn auto_never_gates_conv_to_cuda() {
        use crate::execution_providers::OpPlacement;

        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 0,
        };
        assert!(
            !accelerator_gate(ProviderKind::Cuda, &OpKind::Conv, 1 << 20, &placement),
            "CUDA has no Conv kernel; gating Conv to it guarantees a wasted round trip",
        );

        // wgpu does have one, so with `gpu` also compiled in it takes the node.
        #[cfg(feature = "gpu")]
        assert!(accelerator_gate(
            ProviderKind::Gpu,
            &OpKind::Conv,
            1 << 20,
            &placement
        ));
    }

    /// An op no compiled backend implements stays on the CPU however large it is.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn auto_leaves_non_accelerable_ops_on_the_cpu() {
        use crate::execution_providers::OpPlacement;

        let placement = OpPlacement::Auto {
            gpu_threshold_bytes: 0,
        };
        for accel in all_accelerators() {
            for op in [OpKind::Reshape, OpKind::Shape, OpKind::Gather] {
                assert!(
                    !accelerator_gate(accel, &op, 1 << 24, &placement),
                    "{accel:?}: {op:?} has no kernel there",
                );
            }
        }
    }

    /// The second half of the reported bug: `Manual` pinning *one* op to *one*
    /// provider must not reroute every *other* op to a different accelerator.
    ///
    /// The old DirectML gate never looked at the map, so `Manual({Conv: Gpu})`
    /// silently sent `MatMul`, `Add`, `Mul`, `Relu` and `Sigmoid` — DirectML's
    /// entire op set — to DirectML.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn a_manual_pin_does_not_reroute_the_ops_it_did_not_pin() {
        use crate::execution_providers::{OpPlacement, MIN_GPU_DISPATCH_BYTES};

        let pinned = highest_priority_accelerator();
        let mut map = HashMap::new();
        map.insert(OpKind::Conv, pinned);
        let placement = OpPlacement::Manual(map);

        // Every op that was NOT pinned goes to the CPU, on every gate.
        for accel in all_accelerators() {
            for op in [
                OpKind::MatMul,
                OpKind::Add,
                OpKind::Mul,
                OpKind::Relu,
                OpKind::Sigmoid,
            ] {
                assert!(
                    !accelerator_gate(accel, &op, 1 << 20, &placement),
                    "{accel:?}: {op:?} was not pinned; a pin on Conv must not capture it",
                );
            }
        }

        // The op that WAS pinned goes to exactly the provider it was pinned to —
        // and to no other, because a `Manual` pin has no fall-through chain.
        for accel in all_accelerators() {
            assert_eq!(
                accelerator_gate(accel, &OpKind::Conv, 1 << 20, &placement),
                accel == pinned,
                "{accel:?}: only the pinned provider may be offered a pinned op",
            );
        }

        // And the hard floor overrides even an explicit pin.
        assert!(
            !accelerator_gate(
                pinned,
                &OpKind::Conv,
                MIN_GPU_DISPATCH_BYTES - 1,
                &placement
            ),
            "MIN_GPU_DISPATCH_BYTES must override a Manual pin",
        );
        assert!(
            accelerator_gate(pinned, &OpKind::Conv, MIN_GPU_DISPATCH_BYTES, &placement),
            "the floor is inclusive at MIN_GPU_DISPATCH_BYTES",
        );
    }

    // ── output-size estimation ──────────────────────────────────────────────
    //
    // This file used to carry a private `estimate_node_output_bytes` and the three
    // tests that pinned it (`output_bytes_prefers_the_resolved_shape`,
    // `output_bytes_falls_back_to_the_first_input_then_to_zero`,
    // `output_bytes_saturates_instead_of_overflowing`).  The copy existed only
    // because `Session::estimate_output_bytes` was `#[cfg(feature = "gpu")]` and so
    // was missing from the `cuda`-only build.  That `#[cfg]` is now
    // `any(gpu, cuda, directml)`, the copy is deleted, and all three tests moved
    // verbatim to `run/dispatch.rs` alongside the one canonical implementation.

    // ── the explicit-provider-list dispatch floor ───────────────────────────

    /// `with_provider_kinds` is an instruction, not a hint — but not a licence to
    /// ship a 16-byte tensor across PCIe.
    ///
    /// This pins the sequential call site.  `run/parallel.rs` pins its own, and the
    /// predicate both of them call is `Session::provider_list_clears_dispatch_floor`
    /// — one function, so the two paths cannot diverge.  (Divergence here would be
    /// the worst kind of bug in this crate: the same model would route differently
    /// under `with_parallel_execution(true)`, invisibly, with identical output.)
    #[test]
    fn the_provider_list_path_honours_the_hard_dispatch_floor() {
        use crate::execution_providers::MIN_GPU_DISPATCH_BYTES;

        // A [1, 4] f32 bias-add: 16 bytes.  No listed accelerator may be offered it.
        assert!(!Session::provider_list_clears_dispatch_floor(16));
        assert!(!Session::provider_list_clears_dispatch_floor(
            MIN_GPU_DISPATCH_BYTES - 1
        ));

        // From one page upwards the pin is honoured — and, crucially, honoured
        // *regardless of `op_placement`*, which the provider-list path never reads.
        // That is what an explicit list buys over `Auto`, whose own default
        // threshold is 16× higher.
        assert!(Session::provider_list_clears_dispatch_floor(
            MIN_GPU_DISPATCH_BYTES
        ));
        assert!(Session::provider_list_clears_dispatch_floor(1 << 20));
    }

    // ── priority order ──────────────────────────────────────────────────────

    /// The rank order must mirror `select_accelerator`'s documented priority,
    /// which is the crate-wide single source of truth for it.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn accelerator_rank_encodes_cuda_then_directml_then_wgpu() {
        let ranked = all_accelerators();
        for pair in ranked.windows(2) {
            assert!(
                accelerator_rank(pair[0]) < accelerator_rank(pair[1]),
                "{:?} must outrank {:?}",
                pair[0],
                pair[1],
            );
        }
        for accel in ranked {
            assert!(
                accelerator_rank(accel) < accelerator_rank(ProviderKind::Cpu),
                "{accel:?} must outrank the CPU",
            );
        }
    }
}
