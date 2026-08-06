//! Operator implementations for reshape/squeeze/unsqueeze/flatten/transpose.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::shape;

// ── Reshape ──────────────────────────────────────────────────────────────────

pub struct ReshapeOp;
impl Operator for ReshapeOp {
    fn op_type(&self) -> &str {
        "Reshape"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let shape_t = ctx.input(1)?;
        let allowzero = ctx.attrs().i("allowzero", 0) != 0;
        let s: Vec<i64> = shape_t.data.iter().map(|&v| v as i64).collect();
        Ok(vec![shape::reshape(x, &s, allowzero)?])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::F64,
            oxionnx_core::DType::I8,
            oxionnx_core::DType::I16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
            oxionnx_core::DType::U8,
            oxionnx_core::DType::U16,
            oxionnx_core::DType::U32,
            oxionnx_core::DType::U64,
            oxionnx_core::DType::Bool,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::OnnxError;
        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Reshape: missing input[0]".into()))?;
        let shape_t = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("Reshape: missing input[1] (shape)".into()))?;

        // Resolve the new shape from the shape tensor (stored as f32 but holds integer values).
        let s: Vec<i64> = shape_t
            .storage
            .to_f32_vec()
            .iter()
            .map(|&v| v as i64)
            .collect();
        let allowzero = ctx.attrs().i("allowzero", 0) != 0;
        let new_shape = shape::resolve_reshape(&input.shape, input.numel(), &s, allowzero)
            .map_err(OnnxError::ShapeMismatch)?;

        // Same typed storage, new shape — zero-copy view.
        let mut out = input.clone();
        out.shape = new_shape;
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
        if slots.is_empty() {
            return Ok(());
        }
        let x = ctx.input(0)?;
        let shape_t = ctx.input(1)?;
        let allowzero = ctx.attrs().i("allowzero", 0) != 0;
        let s: Vec<i64> = shape_t.data.iter().map(|&v| v as i64).collect();
        let result = shape::reshape(x, &s, allowzero)?;
        let out = &mut slots[0];
        if out.shape == result.shape && out.data.len() == result.data.len() {
            out.data.copy_from_slice(&result.data);
        } else {
            *out = result;
        }
        Ok(())
    }
}

// ── Transpose ────────────────────────────────────────────────────────────────

pub struct TransposeOp;
impl Operator for TransposeOp {
    fn op_type(&self) -> &str {
        "Transpose"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let ndim = x.ndim();
        let raw_perm = ctx.attrs().ints("perm");
        // Validate before casting: a raw negative i64 cast straight to `usize` wraps to a huge
        // value, which then indexes `x.shape`/strides out of bounds inside `shape::transpose`.
        let perm: Vec<usize> = raw_perm
            .iter()
            .map(|&v| {
                if v < 0 || v >= ndim as i64 {
                    Err(OnnxError::ShapeMismatch(format!(
                        "Transpose: perm entry {v} out of range for {ndim}D tensor"
                    )))
                } else {
                    Ok(v as usize)
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(vec![shape::transpose(x, &perm)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── Squeeze ──────────────────────────────────────────────────────────────────

pub struct SqueezeOp;
impl Operator for SqueezeOp {
    fn op_type(&self) -> &str {
        "Squeeze"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let axes = if let Some(t) = ctx.optional_input(1) {
            t.data.iter().map(|&v| v as i64).collect()
        } else {
            ctx.attrs().ints("axes").to_vec()
        };
        Ok(vec![shape::squeeze(x, &axes)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Ok(());
        }
        let x = ctx.input(0)?;
        let raw_axes: Vec<i64> = if let Some(t) = ctx.optional_input(1) {
            t.data.iter().map(|&v| v as i64).collect()
        } else {
            ctx.attrs().ints("axes").to_vec()
        };
        // Delegate to the same validated axis-resolution logic `execute()` uses via
        // `shape::squeeze`, rather than re-deriving (and risking re-diverging) it here.
        let new_shape = shape::basic::resolve_squeeze_shape(&x.shape, &raw_axes)?;
        let slot = &mut slots[0];
        if slot.data.len() != x.data.len() {
            slot.data.resize(x.data.len(), 0.0_f32);
        }
        slot.data.copy_from_slice(&x.data);
        slot.shape = new_shape;
        Ok(())
    }
}

// ── Unsqueeze ────────────────────────────────────────────────────────────────

pub struct UnsqueezeOp;
impl Operator for UnsqueezeOp {
    fn op_type(&self) -> &str {
        "Unsqueeze"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let axes = if let Some(t) = ctx.optional_input(1) {
            t.data.iter().map(|&v| v as i64).collect()
        } else {
            ctx.attrs().ints("axes").to_vec()
        };
        Ok(vec![shape::unsqueeze(x, &axes)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Ok(());
        }
        let x = ctx.input(0)?;
        let raw_axes: Vec<i64> = if let Some(t) = ctx.optional_input(1) {
            t.data.iter().map(|&v| v as i64).collect()
        } else {
            ctx.attrs().ints("axes").to_vec()
        };
        // Delegate to the same validated axis-resolution logic `execute()` uses via
        // `shape::unsqueeze` (normalizes negative axes against the OUTPUT rank, bounds-checks,
        // and rejects duplicates), rather than re-deriving it here against the growing shape.
        let new_shape = shape::basic::resolve_unsqueeze_shape(&x.shape, &raw_axes)?;
        let slot = &mut slots[0];
        if slot.data.len() != x.data.len() {
            slot.data.resize(x.data.len(), 0.0_f32);
        }
        slot.data.copy_from_slice(&x.data);
        slot.shape = new_shape;
        Ok(())
    }
}

// ── Flatten ──────────────────────────────────────────────────────────────────

pub struct FlattenOp;
impl Operator for FlattenOp {
    fn op_type(&self) -> &str {
        "Flatten"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let axis = ctx.attrs().i("axis", 1);
        Ok(vec![shape::flatten(x, axis)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Ok(());
        }
        let x = ctx.input(0)?;
        let axis = ctx.attrs().i("axis", 1);
        // Delegate to the same validated (outer, inner) resolution `execute()` uses via
        // `shape::flatten`: correct for both the inclusive `[-r, r]` axis range Flatten allows
        // and for genuinely zero-size dims (no `.max(1)` clamp corrupting the shape/data
        // invariant).
        let (outer, inner) = shape::basic::resolve_flatten_shape(&x.shape, axis)?;
        let new_shape = vec![outer, inner];
        let slot = &mut slots[0];
        if slot.data.len() != x.data.len() {
            slot.data.resize(x.data.len(), 0.0_f32);
        }
        slot.data.copy_from_slice(&x.data);
        slot.shape = new_shape;
        Ok(())
    }
}
