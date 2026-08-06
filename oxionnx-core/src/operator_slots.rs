//! Default output-slot writing: executes the operator and copies results into pre-allocated slots.

// `alloc::format!`: `alloc` is always linked by the crate root (see
// lib.rs), and this is the exact same macro as `std::format!`, so it
// resolves identically whether or not the `std` feature is enabled.
use alloc::format;

use crate::operator::{OpContext, Operator};
use crate::tensor::Tensor;
use crate::OnnxError;

/// Default implementation of [`Operator::execute_into_slots`].
///
/// Calls `execute`, then copies each result into the corresponding slot.
/// If shape and element count match the slot, the data is copied in place (zero allocation).
/// If they differ, the slot is replaced entirely (correctness over pointer identity).
pub fn default_into_slots(
    op: &dyn Operator,
    ctx: &OpContext<'_>,
    slots: &mut [Tensor],
) -> Result<(), OnnxError> {
    let results = op.execute(ctx)?;
    if results.len() != slots.len() {
        return Err(OnnxError::Internal(format!(
            "operator '{}' produced {} outputs but {} slots were provided",
            op.op_type(),
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
