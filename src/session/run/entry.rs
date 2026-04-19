use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;
use std::sync::Mutex;

use super::super::Session;
use super::state::SessionRunState;

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

        // Update dynamic dimension bindings and re-resolve intermediate shapes if needed
        self.update_dynamic_dims(inputs)?;

        let output_set: std::collections::HashSet<&str> =
            self.output_names.iter().map(|s| s.as_str()).collect();
        let mut ref_counts: HashMap<String, usize> = HashMap::new();
        for node in &self.sorted_nodes {
            for inp in &node.inputs {
                if !inp.is_empty() && !self.weights.contains_key(inp) {
                    *ref_counts.entry(inp.clone()).or_insert(0) += 1;
                }
            }
        }
        for name in &self.output_names {
            *ref_counts.entry(name.clone()).or_insert(0) += 1;
        }

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
            self.run_parallel_inner(&mut state, &mut ref_counts, &output_set)?;
        } else {
            self.run_sequential_inner(&mut state, &mut ref_counts, &output_set)?;
        }

        let pool_ref = self.pool.as_ref().map(|m| m as &Mutex<SizeClassPool>);
        Ok(state.take_outputs(&self.output_names, pool_ref))
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
