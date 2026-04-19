use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{TensorStorage, TypedOpContext, TypedTensor};
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
    pub(crate) fn run_internal_typed(
        &self,
        inputs: &HashMap<String, TypedTensor>,
    ) -> Result<HashMap<String, TypedTensor>, OnnxError> {
        let mut state = TypedSessionRunState::new();

        // Seed state with user-provided inputs
        for (name, tensor) in inputs {
            state.insert(name.clone(), tensor.clone());
        }

        // Seed state with model weights (converted to TypedTensor::F32)
        for (name, tensor) in &self.weights {
            let typed = TypedTensor::new(
                TensorStorage::F32(tensor.data.clone()),
                tensor.shape.clone(),
            );
            state.insert(name.clone(), typed);
        }

        // Topological execution
        for node in &self.sorted_nodes {
            if let crate::graph::OpKind::Unknown(_) = &node.op {
                continue;
            }
            let op_name = node.op.as_str();
            let operator = self.registry.get(op_name).ok_or_else(|| {
                OnnxError::UnknownOp(format!("No operator registered for '{op_name}'"))
            })?;

            // Resolve typed inputs from state
            let typed_inputs: Vec<Option<TypedTensor>> = node
                .inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        state.get(name).cloned()
                    }
                })
                .collect();

            // Check whether all non-empty inputs are in the op's native_dtypes set
            let native_dtypes = operator.native_dtypes();
            let all_native = !native_dtypes.is_empty()
                && typed_inputs
                    .iter()
                    .filter_map(|o| o.as_ref())
                    .all(|t| native_dtypes.contains(&t.dtype()));

            let results: Vec<TypedTensor> = if all_native {
                // Native typed dispatch — no f32 round-trip
                let input_refs: Vec<Option<&TypedTensor>> =
                    typed_inputs.iter().map(|o| o.as_ref()).collect();
                let typed_ctx = TypedOpContext {
                    node,
                    inputs: input_refs,
                    outer_scope: None,
                    registry: Some(&self.registry),
                };
                operator.execute_typed(&typed_ctx)?
            } else {
                // Surgical f32 cast: convert typed inputs to f32 Tensors, call execute
                let f32_tensors: Vec<Option<Tensor>> = typed_inputs
                    .iter()
                    .map(|opt| {
                        opt.as_ref().map(|tt| {
                            let data = tt.storage.to_f32_vec();
                            Tensor::new(data, tt.shape.clone())
                        })
                    })
                    .collect();
                let f32_refs: Vec<Option<&Tensor>> =
                    f32_tensors.iter().map(|o| o.as_ref()).collect();
                let ctx = oxionnx_core::OpContext {
                    node,
                    inputs: f32_refs,
                    outer_scope: None,
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
        let mut raw_outputs = state.take_outputs(&self.output_names);

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
}
