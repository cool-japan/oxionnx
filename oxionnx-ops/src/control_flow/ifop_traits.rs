//! # IfOp - Trait Implementations
//!
//! This module contains trait implementations for `IfOp`.
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

use super::functions::execute_subgraph;
use super::types::IfOp;

impl Operator for IfOp {
    fn op_type(&self) -> &str {
        "If"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let cond = ctx.input(0)?;
        let condition_val = cond
            .data
            .first()
            .ok_or_else(|| OnnxError::InvalidModel("If: condition tensor is empty".into()))?;
        let is_true = *condition_val != 0.0;
        let branch_name = if is_true {
            "then_branch"
        } else {
            "else_branch"
        };
        let graph = ctx.attrs().graph(branch_name).ok_or_else(|| {
            OnnxError::InvalidModel(format!("If: missing '{}' attribute", branch_name))
        })?;
        let registry = ctx.registry.ok_or_else(|| {
            OnnxError::InvalidModel("If: registry not available for subgraph execution".into())
        })?;
        let empty_scope = HashMap::new();
        let outer = ctx.outer_scope.unwrap_or(&empty_scope);
        let subgraph_inputs = HashMap::new();
        let empty_weights = HashMap::new();
        let weights = ctx.weights.unwrap_or(&empty_weights);
        execute_subgraph(graph, subgraph_inputs, outer, weights, registry)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}
