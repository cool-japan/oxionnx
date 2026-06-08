//! Default typed dispatch: converts TypedTensor inputs to f32, runs execute, wraps outputs.

use crate::dtype::{TensorStorage, TypedTensor};
use crate::operator::{OpContext, Operator, TypedOpContext};
use crate::tensor::Tensor;
use crate::OnnxError;

/// Default implementation of [`Operator::execute_typed`].
///
/// Converts all `TypedTensor` inputs to f32 `Tensor`s, calls `op.execute`, and wraps
/// each output as an F32 `TypedTensor`.  Output dtype recovery happens at the session
/// level (`run_typed`) — the default always produces F32 intermediates.
pub fn default_typed_via_f32(
    op: &dyn Operator,
    ctx: &TypedOpContext<'_>,
) -> Result<Vec<TypedTensor>, OnnxError> {
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
    let f32_results = op.execute(&f32_ctx)?;

    // Wrap each f32 Tensor as an F32 TypedTensor.
    Ok(f32_results
        .into_iter()
        .map(|t| TypedTensor::new(TensorStorage::F32(t.data), t.shape))
        .collect())
}
