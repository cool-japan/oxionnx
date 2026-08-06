use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;
use std::sync::Mutex;

use super::super::Session;
use super::state::SessionRunState;
use super::{OutputSet, RefCounts};

impl Session {
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

        // Resolve intermediate shapes for THIS run.  The value is held by the run
        // (as an immutable `Arc`, so a repeated run copies nothing) and threaded
        // through the execution paths as an argument: two concurrent `run()` calls
        // with different batch sizes must not be able to observe each other's
        // shapes.  See `Session::resolve_run_shapes`.
        let resolved_shapes = self.resolve_run_shapes(inputs)?;

        let output_set: OutputSet<'_> = self.output_names.iter().map(|s| s.as_str()).collect();
        // Reference counts drive when an intermediate's buffer may be recycled.
        // A node "consumes" its `inputs` **and** every outer-scope name its
        // subgraph attributes capture: ONNX subgraphs bind those implicitly by
        // name, so they appear nowhere in `node.inputs`, yet an `If`/`Loop`/`Scan`
        // body reads them out of the live run state.  Counting only `node.inputs`
        // freed captured tensors before the subgraph ever ran — and, because the
        // count of 1 also unlocked the in-place path, mutated them first.
        // `Session::decrement_refs_state` releases exactly the same set, so the
        // counts stay symmetric.  See `run::scheduling::subgraph_captures`.
        //
        // The *contents* of this map are the same on every run — they depend only
        // on the node list, the weights and the graph outputs, all fixed at build
        // time — so they are computed once, into `Session::run_plan`.  Rebuilding
        // them here walked every node, every input and every subgraph capture
        // (a recursive free-name walk per control-flow node) on every inference.
        // What is left is one `extend` over a precomputed vector; the run still
        // gets a *fresh* map, because the counts are decremented as it proceeds.
        //
        // The keys **borrow** from `self.run_plan`, which is immutable for the
        // whole run, so the map allocates one table and no strings.  See
        // [`RefCounts`] for why the hasher is `hashbrown`'s rather than SipHash.
        let base = &self.run_plan.base_ref_counts;
        let mut ref_counts: RefCounts<'_> = RefCounts::with_capacity(base.len());
        ref_counts.extend(base.iter().map(|(name, count)| (name.as_str(), *count)));

        let mut state = SessionRunState::with_capacity(self.sorted_nodes.len());
        // Seed state with input tensors (one clone per input, not per op)
        for (name, tensor) in inputs {
            state.insert(
                name.to_string(),
                (*tensor).clone(),
                self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>),
            );
        }

        let use_parallel = self.parallel && cfg!(not(target_arch = "wasm32"));

        if self.mixed_precision {
            tracing::trace!("Running inference with mixed-precision mode");
        }

        if use_parallel {
            self.run_parallel_inner(&mut state, &mut ref_counts, &output_set, &resolved_shapes)?;
        } else {
            self.run_sequential_inner(&mut state, &mut ref_counts, &output_set, &resolved_shapes)?;
        }

        let pool_ref = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
        state.take_outputs(&self.output_names, &self.weights, pool_ref)
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

        // Merge inference outputs back into the binding.
        // For outputs that were pre-allocated via bind_output, copy data in-place
        // if the shape matches, otherwise replace. For new outputs, insert directly.
        for (name, tensor) in outputs {
            match binding.take_output_buffer(&name) {
                Some(mut buf)
                    if buf.data.len() == tensor.data.len() && buf.shape == tensor.shape =>
                {
                    buf.data.copy_from_slice(&tensor.data);
                    binding.put_output_buffer(name, buf);
                }
                Some(_) => {
                    // Shape mismatch: discard the old buffer and use the new tensor
                    binding.put_output_buffer(name, tensor);
                }
                None => {
                    binding.put_output_buffer(name, tensor);
                }
            }
        }
        Ok(())
    }

    /// Convenience wrapper: run with a single input.
    pub fn run_one(&self, name: &str, input: Tensor) -> Result<HashMap<String, Tensor>, OnnxError> {
        let mut inputs = HashMap::new();
        inputs.insert(name, input);
        self.run(&inputs)
    }
}
