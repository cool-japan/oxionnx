// `alloc`-backed types/macros, imported unconditionally: `alloc` is always
// linked by the crate root (see lib.rs), and `alloc::vec::Vec` /
// `alloc::string::String` are the exact same items as `std::vec::Vec` /
// `std::string::String`, so this resolves identically whether or not the
// `std` feature is enabled.
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};

// `HashMap` is part of this crate's existing public API (`OpContext`,
// `TypedOpContext` and `OperatorRegistry` are typed/keyed on
// `HashMap<String, _>`), so `std` builds must keep resolving to the exact
// same type downstream crates already construct values of. `no_std` builds
// fall back to `hashbrown` (already a non-optional dependency of this
// crate).
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use core::sync::atomic::{AtomicI64, Ordering};

use crate::dtype::{DType, TypedTensor};
use crate::error::OnnxError;
use crate::graph::{Attributes, Node};
use crate::tensor::Tensor;

/// The `ai.onnx` opset assumed when a model declares none.
///
/// A `Graph` assembled programmatically has no `ModelProto`, hence no
/// `opset_import` list; it executes with the newest semantics this engine
/// implements, which is what a caller building nodes by hand expects.
pub const DEFAULT_OPSET: i64 = 21;

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

    /// The `ai.onnx` (default-domain) opset version the executing model declares.
    ///
    /// This is the hook for **version-sensitive operators**: several ONNX ops
    /// changed contract at an opset boundary and cannot be implemented correctly
    /// from the node alone.  The canonical example is the `Softmax` family, whose
    /// `axis` both changes default (1 → -1) and changes *meaning* (2D coercion
    /// point → reduced axis) at opset 13; `Clip`, `ReduceSum` and `Split` are the
    /// same shape of problem solved through input/attribute presence instead.
    ///
    /// An operator that needs it should branch on a named boundary, never on an
    /// exact version:
    ///
    /// ```ignore
    /// if ctx.opset() < 13 { /* legacy contract */ } else { /* current contract */ }
    /// ```
    ///
    /// The value is carried by the [`OperatorRegistry`] the session binds to the
    /// model ([`OperatorRegistry::set_model_opset`]), so it reaches subgraph
    /// bodies (`If`/`Loop`/`Scan` pass the same registry down) and the typed
    /// dispatch fallback for free.  A context built without a registry — a unit
    /// test, or the constant folder — reports [`DEFAULT_OPSET`].
    pub fn opset(&self) -> i64 {
        self.registry
            .map_or(DEFAULT_OPSET, OperatorRegistry::model_opset)
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

    /// The `ai.onnx` (default-domain) opset version the executing model declares.
    ///
    /// Same contract as [`OpContext::opset`] — see there for how version-sensitive
    /// operators are expected to use it.
    pub fn opset(&self) -> i64 {
        self.registry
            .map_or(DEFAULT_OPSET, OperatorRegistry::model_opset)
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
///
/// A registry is bound one-to-one to the model that executes through it (a
/// `Session` owns its registry by value), so it doubles as the carrier for the
/// one piece of *model-level* execution context every operator may need but
/// cannot read off its own node: the declared opset version.  See
/// [`OperatorRegistry::set_model_opset`] and [`OpContext::opset`].
pub struct OperatorRegistry {
    /// Op-type → implementation.
    ///
    /// # Why `hashbrown` and not the `std` map this file otherwise uses
    ///
    /// This is the **hottest map in the engine**: every execution path resolves
    /// its operator through `OperatorRegistry::get` once per node per `run()`,
    /// the cancellation guard adds a second lookup per executed node, and every
    /// `If`/`Loop`/`Scan` body re-enters it per iteration.  `std`'s default
    /// hasher is SipHash-1-3 — DoS-resistant, and paid on every one of those
    /// lookups for keys (`"Conv"`, `"Add"`) that come from the model's own
    /// op-type strings and are matched against a fixed, engine-defined key set.
    /// A hash flood here is not a threat model: an attacker-chosen op type that
    /// is *not* one of the ~167 registered names simply misses and raises
    /// `UnsupportedOp`.
    ///
    /// The field is **private** and nothing iterates it (only `get`, `contains`,
    /// `len`, `is_empty`), so the change is invisible in this crate's public API
    /// and cannot introduce an iteration-order dependency.  `hashbrown` was
    /// already a non-optional dependency (it backs the `HashMap` alias under
    /// `no_std`), so this adds nothing to the dependency graph.
    ///
    /// Note the explicit path: under `feature = "std"` the bare `HashMap` in
    /// this module resolves to `std::collections::HashMap`, which is exactly
    /// what the *public* API (`OpContext::outer_scope`, `TypedOpContext`) must
    /// keep using — downstream crates construct those values.
    ops: hashbrown::HashMap<String, Box<dyn Operator>>,
    /// `ai.onnx` (default-domain) opset of the model this registry is bound to.
    ///
    /// Atomic rather than a plain field because the session hands the registry
    /// to operators — and to rayon workers — behind `&self` for the whole of a
    /// run; the binding itself happens once, before any node executes.
    model_opset: AtomicI64,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self {
            ops: hashbrown::HashMap::new(),
            model_opset: AtomicI64::new(DEFAULT_OPSET),
        }
    }

    /// The `ai.onnx` (default-domain) opset of the model this registry is bound
    /// to, or [`DEFAULT_OPSET`] when nothing has bound one.
    pub fn model_opset(&self) -> i64 {
        self.model_opset.load(Ordering::Relaxed)
    }

    /// Bind this registry to a model's `ai.onnx` opset version.
    ///
    /// Called by the session once per run, before the first node executes, from
    /// the model's parsed `opset_import` list.  Every `OpContext` built with
    /// `registry: Some(..)` then reports it through [`OpContext::opset`], which
    /// is how version-sensitive operators (the `Softmax` family, today) pick
    /// their contract.
    ///
    /// Takes `&self`: the run paths only ever hold a shared borrow of the
    /// session's registry.  `Relaxed` is sufficient — the store is
    /// sequenced-before the parallel dispatch that publishes it to workers.
    pub fn set_model_opset(&self, version: i64) {
        self.model_opset.store(version, Ordering::Relaxed);
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
