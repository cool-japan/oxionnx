//! PadOp operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

// ── Pad ─────────────────────────────────────────────────────────────────────

/// Read the `constant_value` (input 2, optional) and opset-18 `axes` (input 3, optional)
/// inputs shared by [`PadOp::execute`] and [`PadOp::execute_into_slots`]. `pads` (input 1) is
/// required and read separately via `ctx.input(1)` so a missing tensor reports
/// `TensorNotFound` instead of silently padding with an empty `pads` list. `mode` is read
/// separately too (`ctx.attrs().s("mode")` already borrows cheaply with no lifetime to thread
/// through a shared helper).
///
/// `constant_value` uses `.data.first()` rather than `.data[0]` because ONNX allows a present
/// tensor to still be 0-element; indexing `[0]` on that would panic instead of falling back to
/// the default, same as a genuinely absent input.
fn read_optional_pad_inputs(ctx: &OpContext<'_>) -> (f32, Option<Vec<i64>>) {
    let constant_value = ctx
        .optional_input(2)
        .and_then(|t| t.data.first().copied())
        .unwrap_or(0.0);
    let axes_vals: Option<Vec<i64>> = ctx
        .optional_input(3)
        .map(|t| t.data.iter().map(|&v| v as i64).collect());
    (constant_value, axes_vals)
}

pub struct PadOp;
impl Operator for PadOp {
    fn op_type(&self) -> &str {
        "Pad"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let pads_tensor = ctx.input(1)?;
        let pads_vals: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
        let (constant_value, axes_vals) = read_optional_pad_inputs(ctx);
        let mode = ctx.attrs().s("mode");
        let mode = if mode.is_empty() { "constant" } else { mode };
        let out = crate::shape::sequence::pad_axes(
            input,
            &pads_vals,
            mode,
            constant_value,
            axes_vals.as_deref(),
        )
        .map_err(OnnxError::ShapeMismatch)?;
        Ok(vec![out])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.len() != 1 {
            return Err(OnnxError::Internal(format!(
                "PadOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let input = ctx.input(0)?;
        let pads_tensor = ctx.input(1)?;
        let pads_vals: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
        let (constant_value, axes_vals) = read_optional_pad_inputs(ctx);
        let mode = ctx.attrs().s("mode");
        let mode = if mode.is_empty() { "constant" } else { mode };

        // Route through the single opset-18-aware implementation (negative pads = crop, `wrap`
        // mode, and the `axes` input) instead of hand-rolling a second copy here that can drift
        // from `execute()`'s behaviour.
        let result = crate::shape::sequence::pad_axes(
            input,
            &pads_vals,
            mode,
            constant_value,
            axes_vals.as_deref(),
        )
        .map_err(OnnxError::ShapeMismatch)?;

        let out = &mut slots[0];
        if out.shape == result.shape && out.data.len() == result.data.len() {
            out.data.copy_from_slice(&result.data);
        } else {
            *out = result;
        }
        Ok(())
    }
}
