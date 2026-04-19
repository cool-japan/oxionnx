use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{Dim, TensorInfo};
use std::collections::HashMap;

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

    /// Update the session's dynamic dimension cache and re-resolve intermediate
    /// shapes if the input shapes have changed since the last call.
    pub(crate) fn update_dynamic_dims(
        &self,
        inputs: &HashMap<&str, &Tensor>,
    ) -> Result<(), OnnxError> {
        if self.input_infos.is_empty() {
            return Ok(());
        }

        let new_dims = Self::resolve_dynamic_shapes(&self.input_infos, inputs)?;
        if new_dims.is_empty() {
            return Ok(());
        }

        // Check if dims changed
        let dims_changed = {
            let current = self
                .dynamic_dims
                .lock()
                .map_err(|e| OnnxError::Internal(format!("dynamic_dims lock: {e}")))?;
            *current != new_dims
        };

        if dims_changed {
            // Update dynamic dims
            {
                let mut dd = self
                    .dynamic_dims
                    .lock()
                    .map_err(|e| OnnxError::Internal(format!("dynamic_dims lock: {e}")))?;
                *dd = new_dims;
            }

            // Re-resolve intermediate shapes using actual input shapes
            let input_shapes: HashMap<String, Vec<usize>> = inputs
                .iter()
                .map(|(name, tensor)| (name.to_string(), tensor.shape.clone()))
                .collect();
            let new_shapes = crate::optimizer::shape_inference::infer_shapes(
                &self.sorted_nodes,
                &self.weights,
                &input_shapes,
            );

            let mut rs = self
                .resolved_shapes
                .lock()
                .map_err(|e| OnnxError::Internal(format!("resolved_shapes lock: {e}")))?;
            *rs = new_shapes;
        }

        Ok(())
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
