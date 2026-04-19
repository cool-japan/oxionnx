//! Pre-allocated I/O buffer management for zero-copy inference.

use oxionnx_core::Tensor;
use std::collections::HashMap;

/// Pre-allocated I/O buffers for repeated inference with the same shapes.
///
/// `IoBinding` allows you to bind named input tensors and pre-allocate
/// output buffers once, then reuse them across multiple inference calls.
/// This eliminates the per-call `HashMap` allocation and reduces the cost
/// of output buffer management during high-throughput serving.
///
/// # Example
///
/// ```ignore
/// use oxionnx::{Session, IoBinding, Tensor};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let session = Session::from_file("model.onnx".as_ref())?;
///     let mut binding = IoBinding::new();
///
///     for _ in 0..100 {
///         binding.bind_input("x", Tensor::new(vec![1.0; 128], vec![1, 128]));
///         session.run_with_binding(&mut binding)?;
///         let out = binding.get_output("y");
///         // process out...
///         binding.clear_inputs(); // keep output buffers, rebind inputs next round
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Default)]
pub struct IoBinding {
    inputs: HashMap<String, Tensor>,
    outputs: HashMap<String, Tensor>,
}

impl IoBinding {
    /// Create a new empty binding.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a named input tensor.
    ///
    /// Replaces any previously bound tensor with the same name.
    pub fn bind_input(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.inputs.insert(name.into(), tensor);
    }

    /// Pre-allocate a named output buffer.
    ///
    /// If the output buffer already has the correct shape after a `run_with_binding`
    /// call, it is reused via `copy_from_slice` without allocation.
    pub fn bind_output(&mut self, name: impl Into<String>, buffer: Tensor) {
        self.outputs.insert(name.into(), buffer);
    }

    /// Return a reference to an output tensor by name.
    ///
    /// Available after a successful `Session::run_with_binding` call.
    pub fn get_output(&self, name: &str) -> Option<&Tensor> {
        self.outputs.get(name)
    }

    /// Return a mutable reference to an output tensor by name.
    pub fn get_output_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.outputs.get_mut(name)
    }

    /// Remove all bound inputs while keeping pre-allocated output buffers.
    ///
    /// Useful for rebinding inputs between calls without discarding output
    /// buffers that have already been allocated to the right shape.
    pub fn clear_inputs(&mut self) {
        self.inputs.clear();
    }

    /// Remove all bound inputs and outputs.
    pub fn clear(&mut self) {
        self.inputs.clear();
        self.outputs.clear();
    }

    /// Names of currently bound inputs.
    pub fn input_names(&self) -> impl Iterator<Item = &str> {
        self.inputs.keys().map(|s| s.as_str())
    }

    /// Names of currently bound outputs (pre-allocated or filled).
    pub fn output_names(&self) -> impl Iterator<Item = &str> {
        self.outputs.keys().map(|s| s.as_str())
    }

    /// Access the raw input map (for internal use by Session).
    pub(crate) fn inputs(&self) -> &HashMap<String, Tensor> {
        &self.inputs
    }

    /// Extract a pre-bound output tensor by name, removing it from the binding.
    /// Returns None if the name is not bound. Used by run_with_binding to inject
    /// pre-bound buffers as output slots before the run.
    pub(crate) fn take_output_buffer(&mut self, name: &str) -> Option<Tensor> {
        self.outputs.remove(name)
    }

    /// Return an output tensor back into the binding (after the run).
    pub(crate) fn put_output_buffer(&mut self, name: String, tensor: Tensor) {
        self.outputs.insert(name, tensor);
    }
}
