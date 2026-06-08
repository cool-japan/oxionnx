use std::collections::HashMap;

use crate::dtype::{DType, TypedTensor};
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
    /// Model weights, passed by reference so control-flow subgraphs can
    /// resolve initialiser names without cloning the entire weight map.
    pub weights: Option<&'a HashMap<String, Tensor>>,
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

/// Context passed to operators executing via the native typed dispatch path.
pub struct TypedOpContext<'a> {
    /// The node being executed.
    pub node: &'a Node,
    /// Resolved typed input tensors in order matching node.inputs.
    /// Optional/missing inputs are None.
    pub inputs: Vec<Option<&'a TypedTensor>>,
    /// Outer scope typed tensors for subgraph operators (If, Loop, Scan).
    pub outer_scope: Option<&'a HashMap<String, TypedTensor>>,
    /// Operator registry for subgraph execution (If, Loop, Scan).
    pub registry: Option<&'a OperatorRegistry>,
}

impl<'a> TypedOpContext<'a> {
    /// Get a typed input by positional index.
    pub fn input(&self, idx: usize) -> Option<&'a TypedTensor> {
        self.inputs.get(idx).and_then(|v| *v)
    }

    /// Get an optional typed input by positional index.
    pub fn optional_input(&self, idx: usize) -> Option<&'a TypedTensor> {
        self.input(idx)
    }

    /// Number of input slots (including None entries).
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Shorthand for &self.node.attrs
    pub fn attrs(&self) -> &Attributes {
        &self.node.attrs
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

    // ── Phase D: native typed dispatch ─────────────────────────────────────────

    /// Dtypes this operator can execute without an f32 round-trip.
    /// An empty slice (the default) means "f32 only".
    fn native_dtypes(&self) -> &'static [DType] {
        &[]
    }

    /// Execute on typed inputs and return typed outputs.
    /// Default: converts inputs to f32, calls `execute`, returns as F32 TypedTensors.
    fn execute_typed(&self, ctx: &TypedOpContext<'_>) -> Result<Vec<TypedTensor>, OnnxError> {
        use crate::dtype::TensorStorage;
        // Convert each typed input to an f32 Tensor.
        let owned: Vec<Option<Tensor>> = ctx
            .inputs
            .iter()
            .map(|maybe| {
                maybe.map(|tt| {
                    let data = tt.storage.to_f32_vec();
                    Tensor::new(data, tt.shape.clone())
                })
            })
            .collect();
        let refs: Vec<Option<&Tensor>> = owned.iter().map(|opt| opt.as_ref()).collect();
        // Build an f32 OpContext from the typed context's metadata.
        let f32_ctx = OpContext {
            node: ctx.node,
            inputs: refs,
            outer_scope: None,
            weights: None,
            registry: ctx.registry,
        };
        // Execute on f32.
        let f32_results = self.execute(&f32_ctx)?;
        // Wrap each f32 Tensor as an F32 TypedTensor.
        Ok(f32_results
            .into_iter()
            .map(|t| TypedTensor::new(TensorStorage::F32(t.data), t.shape))
            .collect())
    }

    // ── Phase F: output-slot writing ───────────────────────────────────────────

    /// Whether this operator can write directly into pre-allocated output slots.
    fn supports_output_slots(&self) -> bool {
        false
    }

    /// Write outputs into caller-provided slots in place.
    /// Default: calls `execute`, copies results into slots (shape-mismatch falls back to replace).
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let results = self.execute(ctx)?;
        if results.len() != slots.len() {
            return Err(OnnxError::Internal(format!(
                "operator '{}' produced {} outputs but {} slots were provided",
                self.op_type(),
                results.len(),
                slots.len()
            )));
        }
        for (slot, result) in slots.iter_mut().zip(results) {
            if slot.shape == result.shape && slot.data.len() == result.data.len() {
                slot.data.copy_from_slice(&result.data);
            } else {
                *slot = result;
            }
        }
        Ok(())
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
