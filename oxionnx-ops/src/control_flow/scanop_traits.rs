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

use super::functions::{execute_subgraph, slice_along_axis, stack_tensors_axis0};
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
        let scan_input_axes_attr = ctx.attrs().ints("scan_input_axes");
        let scan_input_axes: Vec<usize> = if scan_input_axes_attr.is_empty() {
            vec![0; num_scan_inputs]
        } else {
            scan_input_axes_attr.iter().map(|&x| x as usize).collect()
        };
        let scan_input_dirs_attr = ctx.attrs().ints("scan_input_directions");
        let scan_input_dirs: Vec<i64> = if scan_input_dirs_attr.is_empty() {
            vec![0; num_scan_inputs]
        } else {
            scan_input_dirs_attr.to_vec()
        };
        let first_scan = scan_inputs
            .first()
            .ok_or_else(|| OnnxError::InvalidModel("Scan: no scan inputs".into()))?;
        let scan_axis = scan_input_axes.first().copied().unwrap_or(0);
        if scan_axis >= first_scan.shape.len() {
            return Err(OnnxError::InvalidModel(format!(
                "Scan: scan_input_axis {} >= rank {}",
                scan_axis,
                first_scan.shape.len()
            )));
        }
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
        let weights = HashMap::new();
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
            let outputs = execute_subgraph(body, subgraph_inputs, outer, &weights, registry)?;
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
        let mut final_outputs = states;
        for accumulator in scan_accumulators {
            if accumulator.is_empty() {
                final_outputs.push(Tensor::new(vec![], vec![0]));
            } else {
                let stacked = stack_tensors_axis0(&accumulator)?;
                final_outputs.push(stacked);
            }
        }
        Ok(final_outputs)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}
