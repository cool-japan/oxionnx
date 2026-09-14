//! PadOp operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

// ── Pad ─────────────────────────────────────────────────────────────────────

/// Everything `Pad` needs besides `data` and `mode`, resolved from whichever opset's spelling
/// the model uses.
struct PadOperands {
    /// `[begin_0, .., begin_{k-1}, end_0, .., end_{k-1}]`, over `axes` when that is `Some`.
    pads: Vec<i64>,
    /// Fill value for `mode = "constant"`; `0.0` when the model gives none.
    constant_value: f32,
    /// The opset-18 `axes` input; `None` means "every axis, in input order".
    axes: Option<Vec<i64>>,
}

/// Read the `pads`, `constant_value` and opset-18 `axes` operands shared by
/// [`PadOp::execute`] and [`PadOp::execute_into_slots`]. `mode` is read separately
/// (`ctx.attrs().s("mode")` already borrows cheaply with no lifetime to thread through a shared
/// helper) and is an attribute in every opset.
///
/// `pads` and `constant_value` are **inputs** only since opset 11; before that they were
/// attributes on a node with a single input, so a pre-opset-11 export (`Pad-2`: `pads` + `value`;
/// `Pad-1`: the same attribute spelled `paddings`) carries no input 1 at all and used to fail
/// with `TensorNotFound`. Both forms are accepted, the input winning when present and non-empty
/// — the same precedence `UpsampleOp` applies to its opset-7 `scales` attribute, and a model
/// carrying both is malformed anyway. A *present but empty* `pads` input still yields an empty
/// `pads` list when there is no attribute to fall back to, so a legitimate rank-0 pad (which
/// needs `2 * 0` entries) is not turned into an error.
///
/// `constant_value` uses `.data.first()` rather than `.data[0]` because ONNX allows a present
/// tensor to still be 0-element; indexing `[0]` on that would panic instead of falling back to
/// the default, same as a genuinely absent input.
fn read_pad_operands(ctx: &OpContext<'_>) -> Result<PadOperands, OnnxError> {
    let attrs = ctx.attrs();
    let pads_from_input: Option<Vec<i64>> = ctx
        .optional_input(1)
        .map(|t| t.data.iter().map(|&v| v as i64).collect());
    // Pad-2 spells the attribute `pads`; Pad-1 spells the same list `paddings`.
    let pads_from_attr: Option<&[i64]> = [attrs.ints("pads"), attrs.ints("paddings")]
        .into_iter()
        .find(|list| !list.is_empty());
    let pads = match (pads_from_input, pads_from_attr) {
        (Some(input_pads), _) if !input_pads.is_empty() => input_pads,
        (_, Some(attr_pads)) => attr_pads.to_vec(),
        (Some(input_pads), None) => input_pads,
        (None, None) => {
            return Err(OnnxError::InvalidModel(
                "Pad: no pads given (expected the `pads` input of Pad-11+ or the \
                 `pads`/`paddings` attribute of Pad-1/2)"
                    .into(),
            ))
        }
    };
    let constant_value = ctx
        .optional_input(2)
        .and_then(|t| t.data.first().copied())
        .unwrap_or_else(|| attrs.f("value", 0.0));
    let axes: Option<Vec<i64>> = ctx
        .optional_input(3)
        .map(|t| t.data.iter().map(|&v| v as i64).collect());
    Ok(PadOperands {
        pads,
        constant_value,
        axes,
    })
}

pub struct PadOp;
impl Operator for PadOp {
    fn op_type(&self) -> &str {
        "Pad"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let operands = read_pad_operands(ctx)?;
        let mode = ctx.attrs().s("mode");
        let mode = if mode.is_empty() { "constant" } else { mode };
        let out = crate::shape::sequence::pad_axes(
            input,
            &operands.pads,
            mode,
            operands.constant_value,
            operands.axes.as_deref(),
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
        let operands = read_pad_operands(ctx)?;
        let mode = ctx.attrs().s("mode");
        let mode = if mode.is_empty() { "constant" } else { mode };

        // Route through the single opset-18-aware implementation (negative pads = crop, `wrap`
        // mode, and the `axes` input) instead of hand-rolling a second copy here that can drift
        // from `execute()`'s behaviour.
        let result = crate::shape::sequence::pad_axes(
            input,
            &operands.pads,
            mode,
            operands.constant_value,
            operands.axes.as_deref(),
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
