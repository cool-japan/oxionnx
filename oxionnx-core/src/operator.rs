use std::collections::HashMap;

use crate::error::OnnxError;
use crate::graph::{Attributes, Node};
use crate::tensor::Tensor;

/// Runtime context passed to every operator during execution.
pub struct OpContext<'a> {
    /// The node being executed.
    pub node: &'a Node,
    /// Resolved input tensors in order matching node.inputs.
    /// Optional/missing inputs are None.
    pub inputs: Vec<Option<&'a Tensor>>,
    /// Outer scope tensors for subgraph operators (If, Loop, Scan).
    pub outer_scope: Option<&'a HashMap<String, Tensor>>,
    /// Operator registry for subgraph execution (If, Loop, Scan).
    pub registry: Option<&'a OperatorRegistry>,
}

impl<'a> OpContext<'a> {
    /// Get a required input by positional index.
    pub fn input(&self, idx: usize) -> Result<&'a Tensor, OnnxError> {
        self.inputs.get(idx).and_then(|opt| *opt).ok_or_else(|| {
            OnnxError::TensorNotFound(format!(
                "input[{}] not found for node '{}'",
                idx, self.node.name,
            ))
        })
    }

    /// Get an optional input by positional index.
    pub fn optional_input(&self, idx: usize) -> Option<&'a Tensor> {
        self.inputs.get(idx).and_then(|opt| *opt)
    }

    /// Shorthand for &self.node.attrs
    pub fn attrs(&self) -> &Attributes {
        &self.node.attrs
    }

    /// Number of non-empty inputs available
    pub fn num_inputs(&self) -> usize {
        self.inputs.iter().filter(|i| i.is_some()).count()
    }
}

/// Trait for ONNX operator implementations.
/// Operators are stateless -- all runtime state comes through OpContext.
pub trait Operator: Send + Sync {
    /// The canonical ONNX op_type name this operator handles.
    fn op_type(&self) -> &str;

    /// Execute the operator given the resolved context.
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError>;

    /// Whether this operator supports in-place execution on its first input.
    /// When true and the first input tensor has no other consumers, the runtime
    /// can pass an owned tensor to `execute_inplace` to avoid allocation.
    fn supports_inplace(&self) -> bool {
        false
    }

    /// Execute in-place: the first input is passed as an owned `Tensor` whose
    /// data buffer can be mutated directly. The `ctx` still provides access to
    /// the remaining inputs (slot 0 in `ctx.inputs` will be `None`).
    /// Default implementation ignores the owned tensor and falls back to
    /// `execute(ctx)`.
    fn execute_inplace(
        &self,
        _input: Tensor,
        ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        self.execute(ctx)
    }
}

/// Maps ONNX op_type strings to operator implementations.
pub struct OperatorRegistry {
    ops: HashMap<String, Box<dyn Operator>>,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
        }
    }

    /// Register an operator under its op_type() name.
    pub fn register(&mut self, op: Box<dyn Operator>) {
        let name = op.op_type().to_string();
        self.ops.insert(name, op);
    }

    /// Register an operator under an explicit name (for aliases).
    pub fn register_as(&mut self, name: impl Into<String>, op: Box<dyn Operator>) {
        self.ops.insert(name.into(), op);
    }

    /// Look up an operator by ONNX op_type string.
    pub fn get(&self, op_type: &str) -> Option<&dyn Operator> {
        self.ops.get(op_type).map(|b| b.as_ref())
    }

    /// Check if an op_type is registered.
    pub fn contains(&self, op_type: &str) -> bool {
        self.ops.contains_key(op_type)
    }

    /// Number of registered operators.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
