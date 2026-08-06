use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{DType, TensorStorage, TypedOpContext, TypedTensor};
use std::borrow::Cow;
use std::collections::HashMap;

use super::super::Session;
use super::state::TypedSessionRunState;

impl Session {
    /// Run inference with multi-dtype inputs and outputs.
    ///
    /// Dispatches through [`oxionnx_core::Operator::execute_typed`] when all input dtypes
    /// are listed in the operator's [`oxionnx_core::Operator::native_dtypes`] set, preserving
    /// the original dtype without an f32 round-trip. For operators that do not support native
    /// dispatch (or whose inputs span multiple unsupported dtypes), inputs are surgically cast
    /// to f32, the standard `execute` path runs, and the outputs are kept as F32 TypedTensors.
    ///
    /// Output dtypes are reconciled against [`Session::output_info`] after the graph runs:
    /// if an output slot holds F32 data but `output_infos` declares a different dtype, the
    /// data is converted via `TypedTensor::from_f32_vec` to produce the declared dtype.
    ///
    /// # Precision note
    /// The surgical f32 fallback has ~24 bits of significand precision. Integer tensors whose
    /// absolute values exceed 2^24 (~16.7 million) may lose precision on that path. Ops that
    /// declare the relevant integer dtype in `native_dtypes()` bypass f32 entirely.
    pub fn run_typed(
        &self,
        inputs: &HashMap<&str, TypedTensor>,
    ) -> Result<HashMap<String, TypedTensor>, OnnxError> {
        // Convert &str keys to String for run_internal_typed
        let string_inputs: HashMap<String, TypedTensor> = inputs
            .iter()
            .map(|(&name, tt)| (name.to_string(), tt.clone()))
            .collect();
        self.run_internal_typed(&string_inputs)
    }

    /// Inner implementation of typed inference.
    ///
    /// Carries `TypedTensor` intermediates per node and dispatches through
    /// `execute_typed` when the operator natively handles all input dtypes.
    /// Falls back to surgical f32 casting for unsupported ops.
    ///
    /// # Weights are borrowed, never seeded
    ///
    /// This used to open with
    ///
    /// ```ignore
    /// for (name, tensor) in &self.weights {
    ///     state.insert(name.clone(), TypedTensor::new(
    ///         TensorStorage::F32(tensor.data.clone()), tensor.shape.clone()));
    /// }
    /// ```
    ///
    /// — a deep copy of **every model parameter on every call**.  A 500 MB model
    /// allocated and memcpy'd 500 MB before the first node ran, and then copied
    /// each weight *again* at its point of use.  `Session::run`'s own contract is
    /// the opposite ("weights are borrowed (not cloned) to avoid copying hundreds
    /// of MB of model parameters on every inference call"), and the typed path now
    /// honours it: initializers are resolved at lookup time straight out of
    /// `self.weights`, and on the f32 fallback path — the path every operator
    /// without a `native_dtypes()` set takes — they are borrowed with **no copy at
    /// all**.
    pub(crate) fn run_internal_typed(
        &self,
        inputs: &HashMap<String, TypedTensor>,
    ) -> Result<HashMap<String, TypedTensor>, OnnxError> {
        // Version-sensitive operators read this off their `OpContext` /
        // `TypedOpContext`; it must be bound before the first node executes.
        self.bind_registry_opset();

        let mut state = TypedSessionRunState::new();

        // Seed state with user-provided inputs only.
        for (name, tensor) in inputs {
            state.insert(name.clone(), tensor.clone());
        }

        // Topological execution
        for node in &self.sorted_nodes {
            // No `OpKind::Unknown => continue`: the registry lookup below is the
            // gate.  See `super::unsupported_op_error`.
            let op_name = node.op.as_str();
            let operator = self
                .registry
                .get(op_name)
                .ok_or_else(|| super::unsupported_op_error(node))?;

            // Where does each input slot come from?  Deciding this once, without
            // materialising anything, is what lets the f32 path borrow weights.
            let sources: Vec<InputSource> = node
                .inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        InputSource::Absent
                    } else if state.get(name).is_some() {
                        InputSource::Intermediate
                    } else if self.weights.contains_key(name) {
                        InputSource::Weight
                    } else {
                        InputSource::Absent
                    }
                })
                .collect();

            // Are all present inputs in the op's native_dtypes set?  An
            // initializer is f32 storage, so it contributes `DType::F32`.
            let native_dtypes = operator.native_dtypes();
            let all_native = !native_dtypes.is_empty()
                && node.inputs.iter().zip(&sources).all(|(name, source)| {
                    let dtype = match source {
                        InputSource::Absent => return true,
                        InputSource::Weight => DType::F32,
                        InputSource::Intermediate => match state.get(name) {
                            Some(t) => t.dtype(),
                            None => return true,
                        },
                    };
                    native_dtypes.contains(&dtype)
                });

            let results: Vec<TypedTensor> = if all_native {
                // Native typed dispatch — no f32 round-trip.  Intermediates are
                // borrowed out of the run state; only the initializers this node
                // actually reads are materialised as `TypedTensor`.
                let owned: Vec<Option<Cow<'_, TypedTensor>>> = node
                    .inputs
                    .iter()
                    .zip(&sources)
                    .map(|(name, source)| match source {
                        InputSource::Absent => None,
                        InputSource::Intermediate => state.get(name).map(Cow::Borrowed),
                        InputSource::Weight => self.weights.get(name).map(|w| {
                            Cow::Owned(TypedTensor::new(
                                TensorStorage::F32(w.data.clone()),
                                w.shape.clone(),
                            ))
                        }),
                    })
                    .collect();
                let input_refs: Vec<Option<&TypedTensor>> =
                    owned.iter().map(|o| o.as_deref()).collect();
                let typed_ctx = TypedOpContext {
                    node,
                    inputs: input_refs,
                    // The live typed run state *is* the enclosing scope; passing
                    // `None` left `If`/`Loop`/`Scan` bodies unable to resolve any
                    // captured tensor at all.
                    outer_scope: Some(state.slots()),
                    registry: Some(&self.registry),
                };
                operator.execute_typed(&typed_ctx)?
            } else {
                // Surgical f32 cast.  Initializers are already f32 tensors, so
                // they are borrowed verbatim — no copy, whatever their size.
                let f32_inputs: Vec<Option<Cow<'_, Tensor>>> = node
                    .inputs
                    .iter()
                    .zip(&sources)
                    .map(|(name, source)| match source {
                        InputSource::Absent => None,
                        InputSource::Weight => self.weights.get(name).map(Cow::Borrowed),
                        InputSource::Intermediate => state.get(name).map(|tt| {
                            Cow::Owned(Tensor::new(tt.storage.to_f32_vec(), tt.shape.clone()))
                        }),
                    })
                    .collect();
                let f32_refs: Vec<Option<&Tensor>> =
                    f32_inputs.iter().map(|o| o.as_deref()).collect();

                // Subgraph bodies capture outer-scope tensors implicitly by name.
                // Only the captured names are materialised as f32 — projecting the
                // whole run state would cost a full copy of every live
                // intermediate, per control-flow node.
                let captured_scope = self.typed_capture_scope(node, &state);

                let ctx = oxionnx_core::OpContext {
                    node,
                    inputs: f32_refs,
                    outer_scope: captured_scope.as_ref(),
                    weights: Some(&self.weights),
                    registry: Some(&self.registry),
                };
                let f32_results = operator.execute(&ctx)?;
                // Keep outputs as F32 TypedTensors — output_infos reconciliation below
                // converts them to the declared dtype when the graph finishes
                f32_results
                    .into_iter()
                    .map(|t| TypedTensor::new(TensorStorage::F32(t.data), t.shape))
                    .collect()
            };

            // Store outputs
            for (name, result) in node.outputs.iter().zip(results) {
                if !name.is_empty() {
                    state.insert(name.clone(), result);
                }
            }
        }

        // Collect raw outputs
        let mut raw_outputs = state.take_outputs(&self.output_names, &self.weights)?;

        // Reconcile output dtypes against output_infos metadata.
        // When an op fell back to the f32 path, its output will be F32 even if
        // output_infos declares e.g. I64. Convert via from_f32_vec to match.
        for (name, tensor) in raw_outputs.iter_mut() {
            let declared_dtype = self
                .output_info()
                .iter()
                .find(|info| &info.name == name)
                .map(|info| info.dtype);

            if let Some(dtype) = declared_dtype {
                if tensor.dtype() != dtype {
                    // Only attempt conversion when the current storage is F32
                    // (other dtype mismatches are a graph-authoring error, not ours to fix)
                    if let TensorStorage::F32(ref data) = tensor.storage {
                        match TypedTensor::from_f32_vec(data.clone(), tensor.shape.clone(), dtype) {
                            Ok(converted) => *tensor = converted,
                            Err(_) => {
                                // Conversion failed — leave as-is (best-effort)
                            }
                        }
                    }
                }
            }
        }

        Ok(raw_outputs)
    }

    /// The f32 projection of the outer-scope tensors `node`'s subgraphs capture.
    ///
    /// `None` for an ordinary node, so the common case pays one
    /// `HashMap::is_empty`.  Weights are deliberately absent: they reach the
    /// subgraph through `OpContext::weights` without a copy.
    fn typed_capture_scope(
        &self,
        node: &crate::graph::Node,
        state: &TypedSessionRunState,
    ) -> Option<HashMap<String, Tensor>> {
        if node.attrs.graphs.is_empty() {
            return None;
        }
        let mut scope: HashMap<String, Tensor> = HashMap::new();
        for name in super::scheduling::subgraph_captures(node) {
            if let Some(tt) = state.get(name) {
                scope.insert(
                    name.to_string(),
                    Tensor::new(tt.storage.to_f32_vec(), tt.shape.clone()),
                );
            }
        }
        Some(scope)
    }
}

/// Where one of a node's input slots resolves from, decided before anything is
/// materialised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputSource {
    /// An elided input, or a name nothing provides.
    Absent,
    /// An intermediate produced earlier in this run.
    Intermediate,
    /// A model initializer, borrowed from `Session::weights`.
    Weight,
}
