//! # ScanOp - Trait Implementations
//!
//! This module contains trait implementations for `ScanOp`.
//!
//! ## Implemented Traits
//!
//! - `Operator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxionnx_core::error::OnnxError;
use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::tensor::Tensor;
use std::collections::HashMap;

use super::functions::{execute_subgraph, move_axis0_to, slice_along_axis, stack_tensors_axis0};
use super::types::ScanOp;

impl Operator for ScanOp {
    fn op_type(&self) -> &str {
        "Scan"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let num_scan_inputs = ctx.attrs().i("num_scan_inputs", 0) as usize;
        if num_scan_inputs == 0 {
            return Err(OnnxError::InvalidModel(
                "Scan: num_scan_inputs must be > 0".into(),
            ));
        }
        let total_inputs = ctx.num_inputs();
        if total_inputs < num_scan_inputs {
            return Err(OnnxError::InvalidModel(format!(
                "Scan: expected at least {} inputs (num_scan_inputs), got {}",
                num_scan_inputs, total_inputs
            )));
        }
        let num_state = total_inputs - num_scan_inputs;
        let mut states: Vec<Tensor> = Vec::with_capacity(num_state);
        for i in 0..num_state {
            states.push(ctx.input(i)?.clone());
        }
        let mut scan_inputs: Vec<&Tensor> = Vec::with_capacity(num_scan_inputs);
        for i in num_state..total_inputs {
            scan_inputs.push(ctx.input(i)?);
        }
        // scan_input_axes: ONNX spec allows negative values ("counting
        // dimensions from the back", range [-r, r-1] where r = rank of that
        // *specific* scan input -- ranks may differ across scan inputs, so
        // each entry is normalized against its own tensor's rank rather than
        // cast blindly with `as usize` (which would wrap a negative value to
        // near-usize::MAX and always fail the bounds check downstream).
        let scan_input_axes_attr = ctx.attrs().ints("scan_input_axes");
        let mut scan_input_axes: Vec<usize> = Vec::with_capacity(num_scan_inputs);
        for (si, scan_tensor) in scan_inputs.iter().enumerate() {
            let raw_axis = scan_input_axes_attr.get(si).copied().unwrap_or(0);
            let rank = scan_tensor.shape.len() as i64;
            let normalized = if raw_axis < 0 {
                raw_axis + rank
            } else {
                raw_axis
            };
            if normalized < 0 || normalized >= rank {
                return Err(OnnxError::InvalidModel(format!(
                    "Scan: scan_input_axis {} out of range for scan input {} (rank {})",
                    raw_axis, si, rank
                )));
            }
            scan_input_axes.push(normalized as usize);
        }
        let scan_input_dirs_attr = ctx.attrs().ints("scan_input_directions");
        let scan_input_dirs: Vec<i64> = if scan_input_dirs_attr.is_empty() {
            vec![0; num_scan_inputs]
        } else {
            scan_input_dirs_attr.to_vec()
        };
        let first_scan = scan_inputs
            .first()
            .ok_or_else(|| OnnxError::InvalidModel("Scan: no scan inputs".into()))?;
        // scan_input_axes[0] was already validated above against
        // scan_inputs[0]'s (== first_scan's) own rank.
        let scan_axis = scan_input_axes[0];
        let seq_len = first_scan.shape[scan_axis];
        let body = ctx
            .attrs()
            .graph("body")
            .ok_or_else(|| OnnxError::InvalidModel("Scan: missing 'body' attribute".into()))?;
        let registry = ctx.registry.ok_or_else(|| {
            OnnxError::InvalidModel("Scan: registry not available for subgraph execution".into())
        })?;
        let empty_scope = HashMap::new();
        let outer = ctx.outer_scope.unwrap_or(&empty_scope);
        let empty_weights = HashMap::new();
        let weights = ctx.weights.unwrap_or(&empty_weights);
        let num_body_outputs = body.output_names.len();
        if num_body_outputs < num_state {
            return Err(OnnxError::InvalidModel(format!(
                "Scan: body has {} outputs, expected >= {} state outputs",
                num_body_outputs, num_state
            )));
        }
        let num_scan_outputs = num_body_outputs - num_state;
        let mut scan_accumulators: Vec<Vec<Tensor>> = vec![Vec::new(); num_scan_outputs];
        for step in 0..seq_len {
            let mut subgraph_inputs = HashMap::new();
            for (i, state) in states.iter().enumerate() {
                if let Some(name) = body.input_names.get(i) {
                    if !name.is_empty() {
                        subgraph_inputs.insert(name.clone(), state.clone());
                    }
                }
            }
            for (si, scan_tensor) in scan_inputs.iter().enumerate() {
                let axis = scan_input_axes.get(si).copied().unwrap_or(0);
                let direction = scan_input_dirs.get(si).copied().unwrap_or(0);
                let actual_step = if direction != 0 {
                    seq_len - 1 - step
                } else {
                    step
                };
                let element = slice_along_axis(scan_tensor, axis, actual_step)?;
                if let Some(name) = body.input_names.get(num_state + si) {
                    if !name.is_empty() {
                        subgraph_inputs.insert(name.clone(), element);
                    }
                }
            }
            let outputs = execute_subgraph(body, subgraph_inputs, outer, weights, registry)?;
            states.clear();
            for i in 0..num_state {
                let state = outputs.get(i).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Scan: body missing state output at index {}",
                        i
                    ))
                })?;
                states.push(state.clone());
            }
            for (i, accumulator) in scan_accumulators.iter_mut().enumerate() {
                let scan_out = outputs.get(num_state + i).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Scan: body missing scan output at index {}",
                        num_state + i
                    ))
                })?;
                accumulator.push(scan_out.clone());
            }
        }
        // scan_output_axes: the axis (per output, default 0) at which the new
        // "iteration" axis is inserted into the stacked result. Negative
        // values count from the back of the *output* rank (per-iteration
        // rank + 1), per spec range [-r, r-1].
        //
        // scan_output_directions: 0 = append (forward iteration order), 1 =
        // prepend (each iteration's value is placed before the previous
        // ones, i.e. the accumulated order is reversed relative to
        // iteration order) -- default 0 for every scan output.
        let scan_output_axes_attr = ctx.attrs().ints("scan_output_axes");
        let scan_output_dirs_attr = ctx.attrs().ints("scan_output_directions");

        let mut final_outputs = states;
        for (i, mut accumulator) in scan_accumulators.into_iter().enumerate() {
            if accumulator.is_empty() {
                final_outputs.push(Tensor::new(vec![], vec![0]));
                continue;
            }
            let direction = scan_output_dirs_attr.get(i).copied().unwrap_or(0);
            if direction != 0 {
                accumulator.reverse();
            }
            let stacked = stack_tensors_axis0(&accumulator)?;

            let raw_axis = scan_output_axes_attr.get(i).copied().unwrap_or(0);
            let out_rank = stacked.shape.len() as i64;
            let normalized_axis = if raw_axis < 0 {
                raw_axis + out_rank
            } else {
                raw_axis
            };
            if normalized_axis < 0 || normalized_axis >= out_rank {
                return Err(OnnxError::InvalidModel(format!(
                    "Scan: scan_output_axis {} out of range for scan output {} (rank {})",
                    raw_axis, i, out_rank
                )));
            }
            let placed = if normalized_axis == 0 {
                stacked
            } else {
                move_axis0_to(&stacked, normalized_axis as usize)?
            };
            final_outputs.push(placed);
        }
        Ok(final_outputs)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}
