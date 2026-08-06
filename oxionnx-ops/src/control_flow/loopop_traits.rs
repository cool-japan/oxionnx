//! # LoopOp - Trait Implementations
//!
//! This module contains trait implementations for `LoopOp`.
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

use super::functions::{execute_subgraph, stack_tensors_axis0};
use super::types::LoopOp;

impl Operator for LoopOp {
    fn op_type(&self) -> &str {
        "Loop"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let max_trip_count: Option<i64> = ctx
            .optional_input(0)
            .and_then(|t| t.data.first().map(|v| *v as i64));
        let initial_cond: bool = ctx
            .optional_input(1)
            .map(|t| t.data.first().copied().unwrap_or(1.0) != 0.0)
            .unwrap_or(true);
        let num_total_inputs = ctx.inputs.len();
        let mut carried_deps: Vec<Tensor> = Vec::new();
        for i in 2..num_total_inputs {
            if let Some(t) = ctx.optional_input(i) {
                carried_deps.push(t.clone());
            }
        }
        let body = ctx
            .attrs()
            .graph("body")
            .ok_or_else(|| OnnxError::InvalidModel("Loop: missing 'body' attribute".into()))?;
        let registry = ctx.registry.ok_or_else(|| {
            OnnxError::InvalidModel("Loop: registry not available for subgraph execution".into())
        })?;
        let empty_scope = HashMap::new();
        let outer = ctx.outer_scope.unwrap_or(&empty_scope);
        let empty_weights = HashMap::new();
        let weights = ctx.weights.unwrap_or(&empty_weights);
        let num_carried = carried_deps.len();
        let num_body_outputs = body.output_names.len();
        if num_body_outputs < 1 + num_carried {
            return Err(OnnxError::InvalidModel(format!(
                "Loop: body has {} outputs but expected at least {} (1 cond + {} carried)",
                num_body_outputs,
                1 + num_carried,
                num_carried
            )));
        }
        let num_scan_outputs = num_body_outputs - 1 - num_carried;
        let mut scan_accumulators: Vec<Vec<Tensor>> = vec![Vec::new(); num_scan_outputs];
        let mut condition = initial_cond;
        let mut iteration: i64 = 0;
        loop {
            if !condition {
                break;
            }
            if let Some(max) = max_trip_count {
                if iteration >= max {
                    break;
                }
            }
            // The Loop body signature declares `iteration_num` and `cond` as rank-0 scalars
            // (`tensor(int64)` / `tensor(bool)` with no dimensions), so feed them as genuine
            // rank-0 tensors rather than the legacy `[1]` of `Tensor::scalar`. Body graphs
            // that read `data[0]` — which is every consumer in this engine — behave
            // identically at either rank; what changes is a body that inspects the value's
            // `Shape`, or broadcasts it against a carried dependency, where a stray leading
            // axis would propagate into the result.
            let mut subgraph_inputs = HashMap::new();
            if let Some(iter_name) = body.input_names.first() {
                if !iter_name.is_empty() {
                    subgraph_inputs.insert(iter_name.clone(), Tensor::rank0(iteration as f32));
                }
            }
            if let Some(cond_name) = body.input_names.get(1) {
                if !cond_name.is_empty() {
                    let cond_val = if condition { 1.0_f32 } else { 0.0_f32 };
                    subgraph_inputs.insert(cond_name.clone(), Tensor::rank0(cond_val));
                }
            }
            for (i, dep) in carried_deps.iter().enumerate() {
                if let Some(name) = body.input_names.get(2 + i) {
                    if !name.is_empty() {
                        subgraph_inputs.insert(name.clone(), dep.clone());
                    }
                }
            }
            let outputs = execute_subgraph(body, subgraph_inputs, outer, weights, registry)?;
            let new_cond = outputs
                .first()
                .ok_or_else(|| OnnxError::InvalidModel("Loop: body produced no outputs".into()))?;
            condition = new_cond.data.first().copied().unwrap_or(0.0) != 0.0;
            carried_deps.clear();
            for i in 0..num_carried {
                let dep = outputs.get(1 + i).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Loop: body missing carried dep output at index {}",
                        1 + i
                    ))
                })?;
                carried_deps.push(dep.clone());
            }
            for (i, accumulator) in scan_accumulators.iter_mut().enumerate() {
                let scan_out = outputs.get(1 + num_carried + i).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Loop: body missing scan output at index {}",
                        1 + num_carried + i
                    ))
                })?;
                accumulator.push(scan_out.clone());
            }
            iteration += 1;
            if iteration > 1_000_000 {
                return Err(OnnxError::InvalidModel(
                    "Loop: exceeded 1,000,000 iterations safety limit".into(),
                ));
            }
        }
        let mut final_outputs = carried_deps;
        for accumulator in scan_accumulators {
            if accumulator.is_empty() {
                final_outputs.push(Tensor::new(vec![], vec![0]));
            } else {
                // ONNX shape inference for Loop (and ONNX Runtime's LoopImpl)
                // build scan outputs as `[num_iterations] + per_iteration_shape`,
                // i.e. a NEW leading axis, not a concatenation along an
                // existing axis 0. Use `stack_tensors_axis0` (already used by
                // ScanOp for the identical accumulation pattern) so a
                // per-iteration shape [K] over N iterations yields [N, K]
                // rather than collapsing to [N*K].
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
