use crate::graph::Node;
use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{OpContext, Operator};
use std::collections::HashMap;
use std::sync::Mutex;

use super::super::Session;
use super::state::SessionRunState;
use super::{OutputSet, RefCounts};

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
        ref_counts: &RefCounts<'_>,
        output_set: &OutputSet<'_>,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<std::time::Duration, OnnxError> {
        let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);

        // 1. Inplace path: first input has refcount 1, op supports inplace, not a model output.
        let can_inplace = self.node_can_execute_inplace(node, operator, ref_counts, output_set);

        // 2. Slot-write path: op supports output slots and all output shapes are known.
        let can_slot = !can_inplace && operator.supports_output_slots();

        if can_slot {
            if let Some(mut slots) = Self::acquire_output_slots(node, resolved_shapes, pool) {
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
                    outer_scope: Some(state.as_map()),
                    weights: Some(&self.weights),
                    registry: Some(&self.registry),
                };
                let start = crate::time_compat::Instant::now();
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

        let start = crate::time_compat::Instant::now();

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
                outer_scope: Some(state.as_map()),
                weights: Some(&self.weights),
                registry: Some(&self.registry),
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
                outer_scope: Some(state.as_map()),
                weights: Some(&self.weights),
                registry: Some(&self.registry),
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

    // ── The two CPU fast-path gates, shared by both execution paths ─────────
    //
    // These were inline in `dispatch_node`, which is why `run/parallel.rs`'s
    // rayon phase had neither: it could not reach them without duplicating them,
    // so `with_parallel_execution(true)` silently turned the memory planner and
    // the buffer pool off and allocated every node's outputs afresh.  They are
    // factored out here so that "the parallel path does exactly what the
    // sequential path does" is a property of *one* implementation rather than an
    // eyeball comparison of two.

    /// May `operator` be handed ownership of `node`'s first input and mutate it
    /// in place?
    ///
    /// All four conditions are load-bearing:
    ///
    /// * the operator must opt in (`supports_inplace`);
    /// * the input must be a real, non-elided name that is **not an
    ///   initializer** — a weight is shared by every run of the session and
    ///   mutating it corrupts the model;
    /// * it must not be a **declared graph output** — the caller receives that
    ///   tensor, and it must hold the value its producer wrote, not the value a
    ///   later consumer left behind;
    /// * its reference count must be exactly **1**, i.e. this node is its sole
    ///   remaining consumer.  Nothing decrements a count between the moment this
    ///   is evaluated and the moment the node executes (`decrement_refs_state`
    ///   runs *after* the node's outputs are committed, on both paths), so a
    ///   count of 1 here means "no other reader can observe the mutation" for the
    ///   whole of the execution — which is exactly what makes the parallel
    ///   in-place path sound as well: no other work item at the level can hold a
    ///   reference to a tensor with one consumer.
    pub(super) fn node_can_execute_inplace(
        &self,
        node: &Node,
        operator: &dyn Operator,
        ref_counts: &RefCounts<'_>,
        output_set: &OutputSet<'_>,
    ) -> bool {
        operator.supports_inplace()
            && !node.inputs.is_empty()
            && !node.inputs[0].is_empty()
            && !self.weights.contains_key(&node.inputs[0])
            && !output_set.contains(node.inputs[0].as_str())
            && ref_counts
                .get(node.inputs[0].as_str())
                .copied()
                .unwrap_or(0)
                == 1
    }

    /// One pool-backed output buffer per declared output of `node`, pre-sized
    /// from shape inference, ready for [`Operator::execute_into_slots`].
    ///
    /// Returns `None` — after releasing every buffer it had already taken — when
    /// shape inference has no shape for some output, because a slot of the wrong
    /// size is worse than no slot at all: the caller must then fall back to the
    /// allocating `execute` path.
    ///
    /// The pool is locked **once for the whole node** rather than once per
    /// output, which is what the previous inline version did.
    ///
    /// # Why the zeroing `acquire` and not `acquire_for_overwrite`
    ///
    /// [`SizeClassPool::acquire`](crate::SizeClassPool::acquire) zeroes every
    /// element of a recycled buffer, and for an operator that writes its whole
    /// output that is a full pass over the tensor immediately before the kernel
    /// overwrites it.  [`SizeClassPool::acquire_for_overwrite`](crate::SizeClassPool::acquire_for_overwrite)
    /// exists to skip it — but it cannot be used **here**, generically, because
    /// nothing tells this function whether *this* operator writes all of its
    /// slot.  The `Operator` trait's default `execute_into_slots` does
    /// (`copy_from_slice`, else replace), but ~86 operators override it, and one
    /// that writes only a sub-region — a scatter, an unpool — would silently
    /// inherit the previous tensor's values in the elements it skips.  That is a
    /// wrong-numbers bug, not a crash.
    ///
    /// Routing this call site therefore waits on a `fully_writes_slots()`
    /// predicate on `oxionnx_core::Operator` (defaulting to `false`, so the
    /// zeroing path stays the default for every existing operator), which is a
    /// change to the trait rather than to the engine.  Until it exists, the safe
    /// answer is the one the pool has always given.
    pub(super) fn acquire_output_slots(
        node: &Node,
        resolved_shapes: &HashMap<String, Vec<usize>>,
        pool: Option<&Mutex<SizeClassPool>>,
    ) -> Option<Vec<Tensor>> {
        let mut guard = pool.and_then(|pool_mutex| pool_mutex.lock().ok());
        let mut slots = Vec::with_capacity(node.outputs.len());
        let mut all_known = true;

        for out_name in &node.outputs {
            if out_name.is_empty() {
                // Positional placeholder for an elided (optional) ONNX output.
                // Its contents are never read and it is dropped after
                // `execute_into_slots`; only its *presence* matters, to keep the
                // remaining slots aligned with `node.outputs`.
                //
                // NOTE: this MUST be a struct literal, not
                // `Tensor::new(vec![], vec![])`.  `Tensor::new` asserts
                // `data.len() == shape.iter().product()`, and the product of an
                // *empty* shape is 1, not 0 — so `Tensor::new` with an empty data
                // buffer and an empty shape trips its own `debug_assert` and
                // panics in every debug-assertions build (which includes
                // `cargo test` and this crate's `[profile.dev]`).  That made any
                // slot-capable operator with an elided output a guaranteed
                // debug-build panic.
                slots.push(Tensor {
                    data: vec![],
                    shape: vec![],
                });
                continue;
            }
            let Some(shape) = resolved_shapes.get(out_name) else {
                all_known = false;
                break;
            };
            let size: usize = if shape.is_empty() {
                1
            } else {
                shape.iter().product()
            };
            let data = match guard.as_mut() {
                Some(pool_guard) => pool_guard.acquire(size),
                None => vec![0.0f32; size],
            };
            slots.push(Tensor::new(data, shape.clone()));
        }

        if all_known {
            return Some(slots);
        }

        // Release the buffers already acquired: an abandoned slot set must not
        // leak them out of the pool.
        if let Some(pool_guard) = guard.as_mut() {
            for slot in slots {
                if !slot.data.is_empty() {
                    pool_guard.release(slot.data);
                }
            }
        }
        None
    }

    /// Validate a provider's results against `node` and write them into `state`.
    ///
    /// **Every** execution-provider write-back must route through this method.
    /// The hand-rolled idiom it replaces —
    ///
    /// ```ignore
    /// for (name, tensor) in node.outputs.iter().zip(results) {
    ///     if !name.is_empty() {
    ///         state.insert(name.clone(), tensor, pool);
    ///     }
    /// }
    /// ```
    ///
    /// — is unsafe in three distinct ways, all of which this method closes:
    ///
    /// 1. **`zip` silently truncates.** If a provider returns fewer tensors than
    ///    the node has outputs, the trailing outputs are never written. A
    ///    downstream node then reads a *stale* tensor from a previous run, or
    ///    fails with a confusing `TensorNotFound` pointing at the wrong node.
    ///    Nothing anywhere reports the provider that actually dropped them.
    /// 2. **Nothing checks the tensors are self-consistent.** A tensor whose
    ///    `data.len()` disagrees with its `shape` corrupts every consumer, and
    ///    `Tensor::new`'s `debug_assert` does not fire on a struct-literal
    ///    tensor built inside a provider crate.
    /// 3. **Nothing checks the shape is the one the graph expects.** A buggy GPU
    ///    kernel that returns a plausibly-shaped-but-wrong tensor propagates
    ///    silently through the entire graph. This is the exact channel through
    ///    which a bad kernel corrupts a whole model's output.
    ///
    /// # Positional convention
    ///
    /// `results` must be **positionally aligned** with `node.outputs`: exactly
    /// `node.outputs.len()` tensors, in order. Where `node.outputs[i]` is the
    /// empty string (an ONNX *optional / elided* output), the provider must still
    /// occupy slot `i` — conventionally with `Tensor::new(vec![], vec![])`, which
    /// is what the slot-write path above produces. Its contents are ignored and
    /// its buffer is recycled; only its *presence* matters, because that is what
    /// keeps the remaining outputs positionally aligned.
    ///
    /// # Errors
    ///
    /// Returns `Err` — naming both the node and the provider — when:
    ///
    /// - `results.len() != node.outputs.len()` ([`OnnxError::Internal`]);
    /// - a tensor is internally inconsistent, i.e. `data.len()` differs from the
    ///   product of its `shape`, or that product overflows
    ///   ([`OnnxError::ShapeMismatch`]);
    /// - a tensor's shape disagrees with `resolved_shapes` for that output name,
    ///   where a resolved shape exists ([`OnnxError::ShapeMismatch`]).
    ///
    /// Validation is performed **in full before any insert**, so a rejected
    /// write-back leaves `state` byte-for-byte untouched. A provider failure can
    /// therefore never half-write a node.
    ///
    /// # Parameters
    ///
    /// * `node` — the node whose outputs are being written.
    /// * `provider` — human-readable backend name for error messages, e.g.
    ///   `"CUDA"`, `"DirectML"`, `"wgpu"`, `"CPU"`.
    /// * `results` — the tensors the provider produced, positionally aligned
    ///   with `node.outputs`.
    /// * `state` — the run state to write into.
    /// * `resolved_shapes` — shape-inference results for this run; outputs absent
    ///   from the map are only checked for internal consistency.
    pub(crate) fn write_node_outputs(
        &self,
        node: &Node,
        provider: &'static str,
        results: Vec<Tensor>,
        state: &mut SessionRunState,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<(), OnnxError> {
        // ── 1. Arity ────────────────────────────────────────────────────────
        //
        // The check `zip` never performed. Both directions are errors: too few
        // tensors silently drops outputs, too many means the provider is
        // misaligned with the node and every index after the first extra tensor
        // is suspect.
        if results.len() != node.outputs.len() {
            return Err(OnnxError::Internal(format!(
                "{provider} provider returned {} output tensor(s) for node '{}' ({}), \
                 but the node declares {} output(s) {:?}; \
                 results must be positionally aligned with node.outputs \
                 (elided outputs occupy their slot with an empty placeholder tensor)",
                results.len(),
                node.name,
                node.op.as_str(),
                node.outputs.len(),
                node.outputs,
            )));
        }

        // ── 2. Validate every tensor before touching `state` ────────────────
        for (out_name, tensor) in node.outputs.iter().zip(results.iter()) {
            // Elided output: the placeholder's contents are meaningless by
            // convention, so there is nothing to validate. Its *presence* was
            // already enforced by the arity check above.
            if out_name.is_empty() {
                continue;
            }

            // 2a. Internal consistency: data.len() == product(shape).
            //     `checked_mul` throughout: a corrupted shape coming back from a
            //     GPU kernel can trivially overflow `usize` on multiply, which
            //     would panic in debug and silently wrap in release.
            let expected_elems = tensor
                .shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .ok_or_else(|| {
                    OnnxError::ShapeMismatch(format!(
                        "{provider} provider returned output '{}' of node '{}' ({}) with shape \
                         {:?}, whose element count overflows usize",
                        out_name,
                        node.name,
                        node.op.as_str(),
                        tensor.shape,
                    ))
                })?;

            if tensor.data.len() != expected_elems {
                return Err(OnnxError::ShapeMismatch(format!(
                    "{provider} provider returned an internally inconsistent tensor for output \
                     '{}' of node '{}' ({}): shape {:?} implies {} element(s) but the buffer \
                     holds {}",
                    out_name,
                    node.name,
                    node.op.as_str(),
                    tensor.shape,
                    expected_elems,
                    tensor.data.len(),
                )));
            }

            // 2b. Agreement with shape inference, where a resolved shape exists.
            //     This is the check that stops a buggy kernel from quietly
            //     poisoning the rest of the graph.
            if let Some(expected_shape) = resolved_shapes.get(out_name) {
                if tensor.shape != *expected_shape {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "{provider} provider returned output '{}' of node '{}' ({}) with shape \
                         {:?}, but shape inference resolved it to {:?}",
                        out_name,
                        node.name,
                        node.op.as_str(),
                        tensor.shape,
                        expected_shape,
                    )));
                }
            }
        }

        // ── 3. Commit ───────────────────────────────────────────────────────
        //
        // Everything validated: no error can occur past this point, so the write
        // is all-or-nothing.
        let pool = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
        for (out_name, tensor) in node.outputs.iter().zip(results) {
            if out_name.is_empty() {
                // Recycle the elided-output placeholder rather than dropping it.
                // Normally it is empty and this is a no-op, but a provider that
                // returned a real buffer here gets it returned to the pool.
                super::state::release_to_pool(tensor, pool);
                continue;
            }
            state.insert(out_name.clone(), tensor, pool);
        }
        Ok(())
    }

    /// Estimate a node's output size in bytes — the payload an accelerator would
    /// have to upload, launch on, fence, and read back, and therefore the quantity
    /// **every** size rule in [`crate::execution_providers`] is stated in:
    /// `OpPlacement::Auto`'s `gpu_threshold_bytes`, the `OpPlacement::Manual`
    /// floor, and the explicit-provider-list floor
    /// ([`Session::provider_list_clears_dispatch_floor`]).
    ///
    /// Uses the resolved shape of the node's first output when shape inference has
    /// one; otherwise the size of the first available input as a proxy; otherwise
    /// `0` — which keeps the node on the CPU under any non-zero threshold, the
    /// right default when nothing at all is known about the node.
    ///
    /// This is the **single** implementation. `run/sequential.rs` and
    /// `run/parallel.rs` both call it; neither keeps a copy.
    ///
    /// # `#[cfg]`: `any(gpu, cuda, directml)` — *not* `gpu`
    ///
    /// This was `#[cfg(feature = "gpu")]`, which was exactly backwards. A
    /// `cuda`-only or `directml`-only build is *precisely* the build whose provider
    /// routing needs a size to threshold on — a 16-byte bias-add must not cross
    /// PCIe to a *CUDA* device either — and the helper did not exist there. The
    /// CUDA and DirectML gates consequently had no size to threshold on at all, and
    /// two independent local copies of this function grew to plug the hole. It is
    /// now compiled whenever any accelerator is, and nowhere else, so it can
    /// neither be missing where it is needed nor rot unused where it is not.
    ///
    /// # Saturating, never panicking
    ///
    /// The element count is folded with `checked_mul` and saturates at
    /// `usize::MAX`. The `shape.iter().product()` this replaces **panics in a
    /// debug build** on an overflowing shape — and `[profile.dev]` in this
    /// workspace has `debug_assertions` on, so that includes `cargo test`. A
    /// resolved shape is not inherently trustworthy (shape inference over a
    /// malformed model can produce one), and a *size estimate* must never be able
    /// to bring the process down.
    ///
    /// Saturating is also the correct answer and not merely the safe one: a shape
    /// whose element count overflows `usize` describes a tensor larger than the
    /// address space, so `usize::MAX` bytes clears every threshold — which is
    /// exactly what an unrepresentably huge tensor should do.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    pub(crate) fn estimate_output_bytes(
        node: &Node,
        intermediates: &HashMap<String, Tensor>,
        weights: &HashMap<String, Tensor>,
        resolved_shapes: &HashMap<String, Vec<usize>>,
    ) -> usize {
        const F32_BYTES: usize = std::mem::size_of::<f32>();

        // Preferred: the shape inference resolved for the first output.
        if let Some(first_out) = node.outputs.first() {
            if let Some(shape) = resolved_shapes.get(first_out) {
                let elems = shape
                    .iter()
                    .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                    .unwrap_or(usize::MAX);
                return elems.saturating_mul(F32_BYTES);
            }
        }

        // Fallback: the first available input tensor is the best proxy we have.
        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            if let Some(t) = intermediates.get(inp).or_else(|| weights.get(inp)) {
                return t.data.len().saturating_mul(F32_BYTES);
            }
        }

        0
    }

    /// May a node whose output is `output_bytes` be dispatched to an accelerator
    /// the user named **explicitly**, through
    /// [`SessionBuilder::with_provider_kinds`](crate::SessionBuilder::with_provider_kinds)?
    ///
    /// This is the *single* predicate behind the explicit-provider-list path on
    /// **both** execution paths — `Session::try_provider_list_dispatch` in
    /// `run/sequential.rs` and `plan_from_provider_list` in `run/parallel.rs` call
    /// exactly this function — so the two cannot drift apart. A model that runs on
    /// DirectML sequentially and silently on the CPU under
    /// `with_parallel_execution(true)` is the worst class of bug in this crate,
    /// because it is invisible in the output.
    ///
    /// # The decision: the hard floor binds an explicit list; the `Auto` threshold does not
    ///
    /// `.with_provider_kinds([DirectMl, Cpu])` is a user saying *"use DirectML"*.
    /// That is a stronger and more specific signal than `OpPlacement::Auto`, and it
    /// is honoured as such: the session's `op_placement` — and therefore its
    /// `gpu_threshold_bytes` — is **not** consulted here at all. A listed provider
    /// claims every node it has a kernel for, at any size from one page upwards,
    /// even in a session whose placement policy is the default
    /// `OpPlacement::CpuOnly`.
    ///
    /// What the list does *not* buy is the right to ship a sub-page tensor across
    /// PCIe: `crate::execution_providers::MIN_GPU_DISPATCH_BYTES` (4 KiB = 1024
    /// `f32`) overrides it — exactly as that constant already overrides an
    /// `OpPlacement::Manual` pin inside `decide_placement`.
    ///
    /// Three reasons, in order of weight:
    ///
    /// 1. **The crate has already made this exact decision once.**
    ///    `OpPlacement::Manual` — "run `Add` on CUDA" — is precisely as explicit an
    ///    instruction as a provider list is, and `decide_placement` applies the
    ///    floor to it, documenting that "an explicit user preference is not a
    ///    reason to ship a 16-byte tensor across PCIe". Two explicit-pin mechanisms
    ///    with *different* sub-page semantics would be indefensible, and the
    ///    tie-break is which of the two already has a written cost model behind it.
    ///    That is the floor.
    /// 2. **Below the floor there is nothing left to honour.** The floor rests on a
    ///    *fixed*-cost argument, not a bandwidth one: ~20 µs of round trip (DMA
    ///    setup, kernel launch, fence, readback) against ~1 µs of CPU work for the
    ///    whole 4 KiB. No operator's arithmetic intensity amortises 20 µs over
    ///    fewer than 1024 `f32`s, on any backend. Honouring the pin down there
    ///    would grant the user nothing but the right to be an order of magnitude
    ///    slower — and `MIN_GPU_DISPATCH_BYTES` exists for exactly this case.
    /// 3. **There is no device residency to protect.** The one principled
    ///    counter-argument — "keep a chain of tiny ops on-device instead of
    ///    bouncing each one through host memory" — does not apply to this engine:
    ///    every provider entry point (`try_cuda_dispatch`, `try_directml_dispatch`,
    ///    `try_gpu_dispatch`) takes host [`Tensor`]s and returns host [`Tensor`]s,
    ///    so *every* dispatch is a full round trip regardless of what ran before
    ///    it. Keeping a 16-byte node on the CPU therefore cannot break a residency
    ///    chain, because no such chain exists to break. (If tensors ever become
    ///    device-resident, this is the function to revisit — and the only one.)
    ///
    /// The floor is deliberately the **only** size rule applied here. Reusing
    /// `Auto`'s `gpu_threshold_bytes` (64 KiB by default) instead would quietly
    /// demote the provider list from an instruction to a hint, and reintroduce the
    /// very failure the list exists to prevent: a user who asked for DirectML and
    /// silently got the CPU.
    ///
    /// # Why this matters twice on the parallel path
    ///
    /// There, a node with a non-empty routing plan is pulled *out* of `par_iter`
    /// and dispatched serially, because GPU driver contexts are not safe to call
    /// concurrently from several rayon workers. A sub-page node offered to a pinned
    /// provider therefore lost twice over — it paid the PCIe round trip *and*
    /// forfeited its concurrency. The floor hands both back.
    ///
    /// # Uniformity
    ///
    /// Deliberately not `#[cfg]`-gated on any accelerator feature: it is called
    /// from code that compiles in every feature combination, and with no
    /// accelerator compiled in `ProviderKind::Cpu` is the enum's only variant, so
    /// the answer is unobservable rather than wrong.
    pub(crate) fn provider_list_clears_dispatch_floor(output_bytes: usize) -> bool {
        output_bytes >= crate::execution_providers::MIN_GPU_DISPATCH_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Graph, OpKind};
    use crate::{OptLevel, SessionBuilder};

    /// A real `Session`, so `write_node_outputs` is exercised through `&self`
    /// exactly as the provider dispatch paths call it.
    fn test_session() -> Session {
        let graph = Graph {
            nodes: vec![Node {
                name: "relu0".to_string(),
                op: OpKind::Relu,
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
            .build_from_graph(graph, HashMap::new())
            .expect("build test session")
    }

    /// A node with the given output names. Empty strings model ONNX's elided
    /// (optional) outputs.
    fn node_with_outputs(outputs: &[&str]) -> Node {
        Node {
            name: "provider_node".to_string(),
            op: OpKind::Add,
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs: Attributes::default(),
        }
    }

    fn resolved(entries: &[(&str, Vec<usize>)]) -> HashMap<String, Vec<usize>> {
        entries
            .iter()
            .map(|(n, s)| ((*n).to_string(), s.clone()))
            .collect()
    }

    // ── happy paths ─────────────────────────────────────────────────────────

    #[test]
    fn writes_a_single_output_agreeing_with_shape_inference() {
        let session = test_session();
        let node = node_with_outputs(&["out"]);
        let mut state = SessionRunState::with_capacity(4);
        let shapes = resolved(&[("out", vec![2, 2])]);

        let results = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
        session
            .write_node_outputs(&node, "CUDA", results, &mut state, &shapes)
            .expect("valid write-back must succeed");

        let written = state.get("out").expect("output must be in state");
        assert_eq!(written.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(written.shape, vec![2, 2]);
    }

    #[test]
    fn writes_all_outputs_of_a_multi_output_node() {
        let session = test_session();
        let node = node_with_outputs(&["lo", "hi"]);
        let mut state = SessionRunState::with_capacity(4);
        let shapes = resolved(&[("lo", vec![2]), ("hi", vec![2])]);

        let results = vec![
            Tensor::new(vec![1.0, 2.0], vec![2]),
            Tensor::new(vec![3.0, 4.0], vec![2]),
        ];
        session
            .write_node_outputs(&node, "DirectML", results, &mut state, &shapes)
            .expect("valid multi-output write-back must succeed");

        assert_eq!(state.get("lo").expect("lo").data, vec![1.0, 2.0]);
        assert_eq!(state.get("hi").expect("hi").data, vec![3.0, 4.0]);
    }

    /// Outputs with no entry in `resolved_shapes` are still written — shape
    /// inference simply had nothing to say about them. Internal consistency is
    /// still enforced.
    #[test]
    fn accepts_an_output_with_no_resolved_shape() {
        let session = test_session();
        let node = node_with_outputs(&["out"]);
        let mut state = SessionRunState::with_capacity(4);
        let shapes = HashMap::new();

        let results = vec![Tensor::new(vec![7.0, 8.0, 9.0], vec![3])];
        session
            .write_node_outputs(&node, "wgpu", results, &mut state, &shapes)
            .expect("an unresolved output shape is not an error");

        assert_eq!(state.get("out").expect("out").shape, vec![3]);
    }

    /// A rank-0 tensor holds exactly one element: `[].iter().product() == 1`.
    /// This matches `Tensor::new`'s own debug assertion and the slot-write path.
    #[test]
    fn accepts_a_rank_zero_scalar_output() {
        let session = test_session();
        let node = node_with_outputs(&["s"]);
        let mut state = SessionRunState::with_capacity(4);

        let results = vec![Tensor::new(vec![42.0], vec![])];
        session
            .write_node_outputs(&node, "CUDA", results, &mut state, &HashMap::new())
            .expect("a rank-0 scalar is internally consistent");

        assert_eq!(state.get("s").expect("s").data, vec![42.0]);
    }

    /// A genuinely empty tensor is `shape == [0]`, product 0, data empty.
    #[test]
    fn accepts_a_genuinely_empty_output() {
        let session = test_session();
        let node = node_with_outputs(&["e"]);
        let mut state = SessionRunState::with_capacity(4);

        let results = vec![Tensor::new(vec![], vec![0])];
        session
            .write_node_outputs(&node, "CUDA", results, &mut state, &HashMap::new())
            .expect("shape [0] with no data is consistent");

        assert!(state.get("e").expect("e").data.is_empty());
    }

    // ── elided-output positional convention ─────────────────────────────────

    /// An elided output (`""`) must still occupy its slot. The placeholder is
    /// not inserted into `state`, but the outputs *after* it must land under the
    /// right names — which only works if the provider kept them aligned.
    #[test]
    fn honours_the_elided_output_placeholder_convention() {
        let session = test_session();
        let node = node_with_outputs(&["kept", "", "also_kept"]);
        let mut state = SessionRunState::with_capacity(4);

        let results = vec![
            Tensor::new(vec![1.0], vec![1]),
            // Byte-for-byte the placeholder the slot-write path in
            // `dispatch_node` pushes.  Note it is a struct literal: an empty
            // shape has product 1, so `Tensor::new(vec![], vec![])` would trip
            // `Tensor::new`'s own debug assertion.
            Tensor {
                data: vec![],
                shape: vec![],
            },
            Tensor::new(vec![2.0], vec![1]),
        ];
        session
            .write_node_outputs(&node, "DirectML", results, &mut state, &HashMap::new())
            .expect("a correctly-aligned elided output must be accepted");

        assert_eq!(state.get("kept").expect("kept").data, vec![1.0]);
        // The trailing output landed under its own name, not the elided slot's.
        assert_eq!(state.get("also_kept").expect("also_kept").data, vec![2.0]);
        assert!(
            state.get("").is_none(),
            "the placeholder must not be stored"
        );
    }

    /// The placeholder's contents are meaningless by convention, so it is exempt
    /// from validation.
    ///
    /// This exemption is load-bearing: the canonical placeholder
    /// (`data: []`, `shape: []`) is *internally inconsistent* by the very rule
    /// `write_node_outputs` enforces, because the product of an empty shape is 1
    /// while the buffer holds 0 elements.  Validating elided slots would reject
    /// the convention the codebase itself uses.
    #[test]
    fn does_not_validate_the_elided_placeholder_tensor() {
        let session = test_session();
        let node = node_with_outputs(&[""]);

        // Whatever a provider parks in an elided slot is accepted: the canonical
        // inconsistent placeholder, a genuinely empty tensor, or even a real one.
        let placeholders = vec![
            Tensor {
                data: vec![],
                shape: vec![],
            },
            Tensor::new(vec![], vec![0]),
            Tensor::new(vec![1.0, 2.0], vec![2]),
        ];

        for placeholder in placeholders {
            let mut state = SessionRunState::with_capacity(4);
            let shape = placeholder.shape.clone();
            session
                .write_node_outputs(
                    &node,
                    "CUDA",
                    vec![placeholder],
                    &mut state,
                    &HashMap::new(),
                )
                .unwrap_or_else(|e| {
                    panic!("an elided slot holding shape {shape:?} must never be validated: {e}")
                });
            assert!(
                state.get("").is_none(),
                "nothing is stored for an elided output"
            );
        }
    }

    /// Regression guard for a live debug-build panic.
    ///
    /// `dispatch_node`'s slot path used to build its elided-output placeholder
    /// with `Tensor::new(vec![], vec![])`.  `Tensor::new` asserts
    /// `data.len() == shape.iter().product()`, and the product of an **empty**
    /// shape is **1**, not 0 — so that call tripped its own `debug_assert` and
    /// panicked in every debug-assertions build (`cargo test` included).  Any
    /// slot-capable operator with an elided output was a guaranteed panic.
    #[test]
    fn empty_shape_has_product_one_so_the_placeholder_must_be_a_struct_literal() {
        let empty_shape: Vec<usize> = vec![];
        assert_eq!(
            empty_shape.iter().product::<usize>(),
            1,
            "the product of an empty shape is 1 — this is why Tensor::new(vec![], vec![]) panics",
        );

        // The struct literal the slot path now uses: constructible, no assertion.
        let placeholder = Tensor {
            data: vec![],
            shape: vec![],
        };
        assert!(placeholder.data.is_empty());
        assert!(placeholder.shape.is_empty());
    }

    // ── the silent-truncation bug ───────────────────────────────────────────

    /// The bug this helper exists to kill: `zip` silently truncates, so a
    /// provider returning too few tensors left the trailing outputs unwritten
    /// and a downstream node then read a missing tensor.
    #[test]
    fn rejects_a_provider_that_returns_too_few_tensors() {
        let session = test_session();
        let node = node_with_outputs(&["lo", "hi"]);
        let mut state = SessionRunState::with_capacity(4);

        let results = vec![Tensor::new(vec![1.0, 2.0], vec![2])];
        let err = session
            .write_node_outputs(&node, "CUDA", results, &mut state, &HashMap::new())
            .expect_err("arity mismatch must be rejected, not silently truncated");

        assert!(matches!(err, OnnxError::Internal(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(msg.contains("provider_node"), "must name the node: {msg}");
        assert!(msg.contains("CUDA"), "must name the provider: {msg}");

        // And nothing was written.
        assert!(state.get("lo").is_none());
        assert!(state.get("hi").is_none());
    }

    /// Dropping the placeholder for an elided output is the same truncation bug
    /// wearing a disguise: it shifts every later output by one slot.
    #[test]
    fn rejects_a_provider_that_omits_the_elided_placeholder() {
        let session = test_session();
        let node = node_with_outputs(&["kept", "", "also_kept"]);
        let mut state = SessionRunState::with_capacity(4);

        // Two tensors for three slots: without the arity check, `also_kept`
        // would silently receive nothing at all.
        let results = vec![
            Tensor::new(vec![1.0], vec![1]),
            Tensor::new(vec![2.0], vec![1]),
        ];
        let err = session
            .write_node_outputs(&node, "DirectML", results, &mut state, &HashMap::new())
            .expect_err("a missing elided placeholder breaks positional alignment");

        assert!(matches!(err, OnnxError::Internal(_)), "got {err:?}");
        assert!(state.get("kept").is_none(), "state must be untouched");
        assert!(state.get("also_kept").is_none());
    }

    #[test]
    fn rejects_a_provider_that_returns_too_many_tensors() {
        let session = test_session();
        let node = node_with_outputs(&["only"]);
        let mut state = SessionRunState::with_capacity(4);

        let results = vec![
            Tensor::new(vec![1.0], vec![1]),
            Tensor::new(vec![2.0], vec![1]),
        ];
        let err = session
            .write_node_outputs(&node, "wgpu", results, &mut state, &HashMap::new())
            .expect_err("surplus tensors mean the provider is misaligned");

        assert!(matches!(err, OnnxError::Internal(_)), "got {err:?}");
        assert!(state.get("only").is_none());
    }

    // ── internally inconsistent tensors ─────────────────────────────────────

    #[test]
    fn rejects_a_tensor_whose_data_length_contradicts_its_shape() {
        let session = test_session();
        let node = node_with_outputs(&["out"]);
        let mut state = SessionRunState::with_capacity(4);

        // Built by struct literal, bypassing `Tensor::new`'s debug assertion —
        // exactly how a provider crate would produce a corrupt tensor.
        let bad = Tensor {
            data: vec![1.0, 2.0, 3.0],
            shape: vec![2, 2],
        };
        let err = session
            .write_node_outputs(&node, "CUDA", vec![bad], &mut state, &HashMap::new())
            .expect_err("3 elements cannot fill a [2, 2] tensor");

        assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(msg.contains("provider_node"), "must name the node: {msg}");
        assert!(msg.contains("CUDA"), "must name the provider: {msg}");
        assert!(state.get("out").is_none());
    }

    /// A corrupted shape coming back from a GPU kernel can overflow `usize` on
    /// multiply. That must be an error, not a debug panic / release wraparound.
    #[test]
    fn rejects_a_shape_whose_element_count_overflows() {
        let session = test_session();
        let node = node_with_outputs(&["out"]);
        let mut state = SessionRunState::with_capacity(4);

        let bad = Tensor {
            data: vec![1.0],
            shape: vec![usize::MAX, 2, 2],
        };
        let err = session
            .write_node_outputs(&node, "DirectML", vec![bad], &mut state, &HashMap::new())
            .expect_err("an overflowing element count must be rejected");

        assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
        assert!(err.to_string().contains("overflow"), "{err}");
        assert!(state.get("out").is_none());
    }

    // ── disagreement with shape inference ───────────────────────────────────

    /// The channel through which a buggy GPU kernel corrupts an entire graph:
    /// a self-consistent tensor of the wrong shape.
    #[test]
    fn rejects_a_tensor_disagreeing_with_the_resolved_shape() {
        let session = test_session();
        let node = node_with_outputs(&["out"]);
        let mut state = SessionRunState::with_capacity(4);
        let shapes = resolved(&[("out", vec![2, 2])]);

        // Internally consistent (4 elements, shape [4]) but the graph expects [2, 2].
        let results = vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4])];
        let err = session
            .write_node_outputs(&node, "CUDA", results, &mut state, &shapes)
            .expect_err("a shape disagreeing with inference must be rejected");

        assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(msg.contains("provider_node"), "must name the node: {msg}");
        assert!(msg.contains("CUDA"), "must name the provider: {msg}");
        assert!(
            msg.contains("[2, 2]"),
            "must report the expected shape: {msg}"
        );
        assert!(state.get("out").is_none());
    }

    // ── all-or-nothing commit ───────────────────────────────────────────────

    /// Validation runs to completion before the first insert, so a provider that
    /// gets its *second* output wrong cannot leave the first one written.
    #[test]
    fn a_rejected_write_back_leaves_state_byte_for_byte_untouched() {
        let session = test_session();
        let node = node_with_outputs(&["lo", "hi"]);
        let mut state = SessionRunState::with_capacity(4);
        let shapes = resolved(&[("lo", vec![2]), ("hi", vec![2])]);

        // Pre-existing tensors that must survive the failed write-back intact.
        state.insert(
            "lo".to_string(),
            Tensor::new(vec![-1.0, -1.0], vec![2]),
            None,
        );
        state.insert(
            "bystander".to_string(),
            Tensor::new(vec![9.0], vec![1]),
            None,
        );

        let results = vec![
            // Perfectly valid...
            Tensor::new(vec![1.0, 2.0], vec![2]),
            // ...but the second output has the wrong shape.
            Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![4]),
        ];
        let err = session
            .write_node_outputs(&node, "CUDA", results, &mut state, &shapes)
            .expect_err("the second output is invalid");
        assert!(matches!(err, OnnxError::ShapeMismatch(_)), "got {err:?}");

        // The valid first tensor must NOT have been committed.
        assert_eq!(
            state
                .get("lo")
                .expect("lo must still hold its old value")
                .data,
            vec![-1.0, -1.0],
            "a rejected write-back must not partially commit",
        );
        assert!(state.get("hi").is_none());
        assert_eq!(state.get("bystander").expect("bystander").data, vec![9.0]);
    }

    /// Overwriting an existing output is fine — that is `state.insert`'s job.
    #[test]
    fn a_successful_write_back_replaces_an_existing_tensor() {
        let session = test_session();
        let node = node_with_outputs(&["out"]);
        let mut state = SessionRunState::with_capacity(4);
        state.insert("out".to_string(), Tensor::new(vec![0.0], vec![1]), None);

        let results = vec![Tensor::new(vec![5.0], vec![1])];
        session
            .write_node_outputs(&node, "wgpu", results, &mut state, &HashMap::new())
            .expect("write-back must overwrite cleanly");

        assert_eq!(state.get("out").expect("out").data, vec![5.0]);
    }

    /// Every provider name a caller may pass shows up in the error text.
    #[test]
    fn error_messages_name_the_offending_provider() {
        let session = test_session();
        let node = node_with_outputs(&["a", "b"]);

        for provider in ["CUDA", "DirectML", "wgpu", "CPU"] {
            let mut state = SessionRunState::with_capacity(2);
            let err = session
                .write_node_outputs(&node, provider, vec![], &mut state, &HashMap::new())
                .expect_err("0 tensors for a 2-output node is an arity error");
            let msg = err.to_string();
            assert!(msg.contains(provider), "'{provider}' missing from: {msg}");
            assert!(
                msg.contains("provider_node"),
                "node name missing from: {msg}"
            );
            assert!(msg.contains("Add"), "op type missing from: {msg}");
        }
    }

    /// A node with no outputs at all: zero results is the only valid answer.
    #[test]
    fn handles_a_node_with_no_outputs() {
        let session = test_session();
        let node = node_with_outputs(&[]);
        let mut state = SessionRunState::with_capacity(2);

        session
            .write_node_outputs(&node, "CUDA", vec![], &mut state, &HashMap::new())
            .expect("zero outputs, zero results");

        let err = session
            .write_node_outputs(
                &node,
                "CUDA",
                vec![Tensor::new(vec![1.0], vec![1])],
                &mut state,
                &HashMap::new(),
            )
            .expect_err("a result for a node with no outputs is an arity error");
        assert!(matches!(err, OnnxError::Internal(_)), "got {err:?}");
    }

    // ── Session::estimate_output_bytes ──────────────────────────────────────
    //
    // These tests were carried over wholesale from the two private copies of this
    // estimator that `run/sequential.rs` (`estimate_node_output_bytes`) and
    // `run/parallel.rs` (a `cfg(gpu)`-delegating shim) had each grown, when both
    // were deleted in favour of the one canonical implementation above.  Nothing
    // was dropped in the move — including the overflow-saturation guard, which is
    // the reason the canonical version now folds with `checked_mul` rather than
    // `product()`.

    /// A node with exactly the given inputs and outputs.  Empty input names model
    /// ONNX's elided (optional) inputs.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    fn estimator_node(inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            name: "n".to_string(),
            op: OpKind::Add,
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs: Attributes::default(),
        }
    }

    /// Moved from `run/sequential.rs`.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn output_bytes_prefers_the_resolved_shape() {
        let node = estimator_node(&["a"], &["y"]);
        let mut resolved = HashMap::new();
        resolved.insert("y".to_string(), vec![2, 8]);

        assert_eq!(
            Session::estimate_output_bytes(&node, &HashMap::new(), &HashMap::new(), &resolved),
            16 * 4,
            "16 f32 elements = 64 bytes",
        );
    }

    /// Moved from `run/sequential.rs`.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn output_bytes_falls_back_to_the_first_input_then_to_zero() {
        let node = estimator_node(&["", "a"], &["y"]);

        // Elided inputs are skipped; the first real one is the proxy.
        let mut intermediates = HashMap::new();
        intermediates.insert("a".to_string(), Tensor::new(vec![0.0; 10], vec![10]));
        assert_eq!(
            Session::estimate_output_bytes(&node, &intermediates, &HashMap::new(), &HashMap::new()),
            40,
        );

        // Weights are searched too.
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), Tensor::new(vec![0.0; 3], vec![3]));
        assert_eq!(
            Session::estimate_output_bytes(&node, &HashMap::new(), &weights, &HashMap::new()),
            12,
        );

        // Nothing known at all → 0, which keeps the node on the CPU under any
        // non-zero threshold.
        assert_eq!(
            Session::estimate_output_bytes(
                &node,
                &HashMap::new(),
                &HashMap::new(),
                &HashMap::new()
            ),
            0,
        );
    }

    /// Moved from `run/sequential.rs` — and the reason the canonical estimator
    /// folds the element count with `checked_mul` instead of `product()`.
    ///
    /// A corrupted resolved shape must not panic in a debug build (which is every
    /// `cargo test` run: `[profile.dev]` has `debug_assertions` on) nor wrap around
    /// in release.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn output_bytes_saturates_instead_of_overflowing() {
        let node = estimator_node(&["a"], &["y"]);
        let mut resolved = HashMap::new();
        resolved.insert("y".to_string(), vec![usize::MAX, 2, 2]);

        assert_eq!(
            Session::estimate_output_bytes(&node, &HashMap::new(), &HashMap::new(), &resolved),
            usize::MAX,
            "an overflowing element count saturates — and thus clears every threshold",
        );
    }

    /// Moved from `run/parallel.rs`.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn estimate_output_bytes_prefers_the_resolved_shape() {
        let node = estimator_node(&["a", "b"], &["c"]);
        let mut resolved = HashMap::new();
        resolved.insert("c".to_string(), vec![16, 16]);

        assert_eq!(
            Session::estimate_output_bytes(&node, &HashMap::new(), &HashMap::new(), &resolved),
            16 * 16 * 4,
        );
    }

    /// Moved from `run/parallel.rs`.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn estimate_output_bytes_falls_back_to_an_input_tensor() {
        let node = estimator_node(&["a"], &["c"]);
        let mut intermediates = HashMap::new();
        intermediates.insert("a".to_string(), Tensor::new(vec![0.0; 32], vec![32]));

        assert_eq!(
            Session::estimate_output_bytes(&node, &intermediates, &HashMap::new(), &HashMap::new()),
            32 * 4,
        );
    }

    /// Moved from `run/parallel.rs`.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "directml"))]
    #[test]
    fn estimate_output_bytes_is_zero_when_nothing_is_known() {
        let node = estimator_node(&["a"], &["c"]);
        assert_eq!(
            Session::estimate_output_bytes(
                &node,
                &HashMap::new(),
                &HashMap::new(),
                &HashMap::new()
            ),
            0,
        );
    }

    // ── the explicit-provider-list dispatch floor ───────────────────────────
    //
    // The predicate itself, pinned here because it is the *shared* one: both
    // `run/sequential.rs` (`try_provider_list_dispatch`) and `run/parallel.rs`
    // (`plan_from_provider_list`) call exactly this function, so the two paths
    // cannot disagree about what an explicit provider list means for a tiny
    // tensor.  Each file additionally pins the behaviour at its own call site.

    #[test]
    fn the_dispatch_floor_is_exclusive_below_and_inclusive_at_min_gpu_dispatch_bytes() {
        use crate::execution_providers::MIN_GPU_DISPATCH_BYTES;

        assert!(
            !Session::provider_list_clears_dispatch_floor(MIN_GPU_DISPATCH_BYTES - 1),
            "the floor is exclusive below MIN_GPU_DISPATCH_BYTES",
        );
        assert!(
            Session::provider_list_clears_dispatch_floor(MIN_GPU_DISPATCH_BYTES),
            "the floor is inclusive at MIN_GPU_DISPATCH_BYTES",
        );
        assert!(Session::provider_list_clears_dispatch_floor(usize::MAX));
    }

    /// The pathological case the floor exists for: a `[1, 4]` f32 bias-add is 16
    /// bytes.  `.with_provider_kinds([DirectMl, Cpu])` used to ship it across PCIe
    /// — a ~20 µs round trip to replace ~4 ns of addition — while
    /// `OpPlacement::Auto` correctly kept it on the CPU.
    #[test]
    fn a_sub_page_tensor_never_clears_the_floor_however_it_was_pinned() {
        for bytes in [0, 4, 16, 64, 1024, 4095] {
            assert!(
                !Session::provider_list_clears_dispatch_floor(bytes),
                "{bytes} bytes must never be shipped to an explicitly listed accelerator",
            );
        }
    }

    /// ...but the floor is *only* a floor.  It is an order of magnitude below
    /// `OpPlacement::Auto`'s 64 KiB default, so an explicit provider list still
    /// beats `Auto` comfortably: everything from one page upwards is honoured, and
    /// the session's `gpu_threshold_bytes` is never consulted on that path.
    #[test]
    fn the_floor_is_a_backstop_not_a_second_auto_threshold() {
        // That `MIN_GPU_DISPATCH_BYTES` stays an order of magnitude below `Auto`'s
        // 64 KiB default is enforced at compile time by the `const _: () = { … }`
        // block in `execution_providers.rs` — asserting it again here would be a
        // `clippy::assertions_on_constants` error.  What *is* worth pinning is the
        // consequence: the gap between the two is precisely what an explicit
        // provider list buys the user over the `Auto` heuristic.

        // A 4 KiB tensor clears the explicit-list floor...
        assert!(Session::provider_list_clears_dispatch_floor(4096));
        // ...while `Auto`'s own default threshold would have kept it on the CPU.
        assert_eq!(
            crate::execution_providers::decide_placement(
                &OpKind::Add,
                4096,
                &crate::execution_providers::OpPlacement::Auto {
                    gpu_threshold_bytes: 65_536,
                },
            ),
            crate::execution_providers::ProviderKind::Cpu,
            "this is precisely the gap an explicit provider list buys the user",
        );
    }
}
