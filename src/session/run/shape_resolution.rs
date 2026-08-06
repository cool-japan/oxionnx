use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{Dim, TensorInfo};
use std::collections::HashMap;
use std::sync::Arc;

use super::super::Session;

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

    /// Resolve the intermediate tensor shapes for **this one run**, and refresh
    /// the session's symbolic dimension bindings.
    ///
    /// # Why the result is returned rather than read back out of the session
    ///
    /// `Session` is `Send + Sync` and callers routinely park one in an `Arc`
    /// behind a web handler, so two `run()` calls with *different* batch sizes
    /// overlap routinely.  The previous shape flow could not survive that: it
    /// wrote the session-wide `resolved_shapes` from one thread and the execution
    /// paths re-read it in a *separate* lock acquisition, so thread B could write
    /// shapes for batch 1, thread A overwrite them for batch 8, and thread B then
    /// execute against A's shapes — a spurious `ShapeMismatch` from
    /// `write_node_outputs` on any provider path, and mis-sized pool acquisitions
    /// on the CPU path.
    ///
    /// The shapes are now a **per-run value**: computed here, owned by the run,
    /// threaded through `run_sequential_inner` / `run_parallel_inner` as an
    /// argument.  No run can observe another run's shapes, whatever the
    /// interleaving.
    ///
    /// # The session map is a memo, keyed by the input shapes themselves
    ///
    /// `infer_shapes` seeds its result with the input shapes it was given, so the
    /// stored map *contains its own cache key*: it is reusable exactly when every
    /// input this run supplies is recorded in it with the same shape (and the
    /// model's other declared inputs are absent from both).  The check and the
    /// clone happen under one lock acquisition, so the value returned is always
    /// internally consistent.
    ///
    /// That key is the concrete shapes, not the symbolic dimension map, which
    /// fixes two further holes:
    ///
    /// * a model with **no** symbolic dimensions used to return early and never
    ///   populate the map at all, leaving the slot-write path in `dispatch_node`
    ///   and the provider shape validation in `write_node_outputs` permanently
    ///   dead for the commonest case — a fully static model;
    /// * a model mixing a named axis with `Dim::Unknown` ones (`[batch, 3, ?, ?]`)
    ///   produced an *unchanged* symbolic map when only H/W changed, so the
    ///   previous run's shapes were reused for a differently-shaped input.
    ///
    /// # The result is an `Arc`, so a repeated run costs no copy at all
    ///
    /// This used to return an owned `HashMap<String, Vec<usize>>`, cloned out of
    /// the memo (or out of the plan cache) on **every** run: one `String` and one
    /// `Vec<usize>` allocation per graph tensor, ~1000 of each for a 500-node
    /// model, thrown away when the run ended.  Nothing mutates the map during a
    /// run — every execution path takes it by shared reference — so the run can
    /// share the cached plan instead of copying it.  [`ShapePlanCache`] already
    /// stores `Arc<ShapeMap>`, so the hot path is now a plan-cache hit plus an
    /// `Arc::clone`, and `&Arc<HashMap<..>>` deref-coerces at every call site.
    ///
    /// The session-wide memo is still refreshed whenever it does *not* already
    /// describe this run's inputs, because [`Session::resolved_shapes`] reports it
    /// to callers; the deep copy that update costs is therefore paid only when the
    /// input shapes actually change, not once per inference.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::resolve_dynamic_shapes`]'s conflicting-symbol error.
    /// A poisoned memo mutex is deliberately **not** an error: one panicking
    /// thread would otherwise break every subsequent run of the session.  The memo
    /// is simply bypassed and the shapes recomputed.
    ///
    /// [`ShapePlanCache`]: super::super::ShapePlanCache
    pub(crate) fn resolve_run_shapes(
        &self,
        inputs: &HashMap<&str, &Tensor>,
    ) -> Result<Arc<HashMap<String, Vec<usize>>>, OnnxError> {
        // Refresh the symbolic bindings the public `dynamic_dims()` accessor
        // reports.  This is informational only — nothing in the run reads it —
        // so a concurrent overwrite by another run cannot affect correctness.
        if !self.input_infos.is_empty() {
            let dims = Self::resolve_dynamic_shapes(&self.input_infos, inputs)?;
            if let Ok(mut current) = self.dynamic_dims.lock() {
                if *current != dims {
                    *current = dims;
                }
            }
        }

        // The concrete input shapes: both the seed for inference and the key the
        // memo is validated against.
        let input_shapes: HashMap<String, Vec<usize>> = inputs
            .iter()
            .map(|(name, tensor)| ((*name).to_string(), tensor.shape.clone()))
            .collect();

        // `shape_plans` keeps the last few plans, keyed by the input shapes that
        // produced them, and hands them out as `Arc`s — so a repeated run pays a
        // pointer bump rather than a full map copy.  It is consulted first for
        // exactly that reason; the single-slot memo below can only ever answer a
        // subset of the same questions, and only by cloning.
        if let Some(plan) = self.shape_plans.lookup(&input_shapes) {
            self.refresh_shape_memo(&plan, &input_shapes);
            return Ok(plan);
        }

        // The plan cache missed but the memo may still hold the answer (it is
        // written on every miss, and survives a plan-cache eviction).
        //
        // LOCK ORDER: the memo guard is released *before* `shape_plans` is
        // touched.  Every other path here takes `shape_plans` first and the memo
        // second (`refresh_shape_memo`), so the two locks are never held at once
        // and no ordering cycle exists.
        if let Ok(cached) = self.resolved_shapes.lock() {
            if self.memo_matches_inputs(&cached, &input_shapes) {
                let plan = Arc::new(cached.clone());
                drop(cached);
                self.shape_plans.store(&input_shapes, &plan);
                return Ok(plan);
            }
        }

        let shapes = Arc::new(crate::optimizer::shape_inference::infer_shapes(
            &self.sorted_nodes,
            &self.weights,
            &input_shapes,
        ));
        self.shape_plans.store(&input_shapes, &shapes);
        if let Ok(mut memo) = self.resolved_shapes.lock() {
            memo.clone_from(&shapes);
        }
        Ok(shapes)
    }

    /// Keep the session-wide memo — what [`Session::resolved_shapes`] reports —
    /// describing the most recent run, without paying a copy per run.
    ///
    /// The memo contains the input shapes it was inferred from, so
    /// [`Session::memo_matches_inputs`] answers "does it already describe these
    /// inputs?" from data it already holds.  A session run repeatedly with one
    /// input shape — the overwhelmingly common case — therefore copies nothing at
    /// all here; only a genuine change of input shapes costs the deep copy.
    fn refresh_shape_memo(
        &self,
        plan: &Arc<HashMap<String, Vec<usize>>>,
        input_shapes: &HashMap<String, Vec<usize>>,
    ) {
        let Ok(mut memo) = self.resolved_shapes.lock() else {
            return;
        };
        if self.memo_matches_inputs(&memo, input_shapes) {
            return;
        }
        memo.clone_from(plan);
    }

    /// Was `memo` produced from exactly the input shapes `input_shapes` describes?
    ///
    /// Every declared model input must agree: supplied with the same shape in
    /// both, or absent from both.  Any input this run supplies that the memo does
    /// not record identically also rejects it, so a partially-supplied input set
    /// can never reuse a fully-supplied run's shapes.
    fn memo_matches_inputs(
        &self,
        memo: &HashMap<String, Vec<usize>>,
        input_shapes: &HashMap<String, Vec<usize>>,
    ) -> bool {
        if memo.is_empty() {
            return false;
        }
        for name in &self.input_names {
            if input_shapes.get(name) != memo.get(name) {
                return false;
            }
        }
        for (name, shape) in input_shapes {
            if memo.get(name) != Some(shape) {
                return false;
            }
        }
        true
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
}
