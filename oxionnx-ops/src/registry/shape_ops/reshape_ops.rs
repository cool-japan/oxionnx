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
        let s: Vec<i64> = shape_t.data.iter().map(|&v| v as i64).collect();
        Ok(vec![shape::reshape(x, &s)?])
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
        let numel = input.numel();

        let neg_count = s.iter().filter(|&&d| d == -1).count();
        if neg_count > 1 {
            return Err(OnnxError::ShapeMismatch(
                "Reshape: at most one -1 allowed".into(),
            ));
        }
        let known: usize = s
            .iter()
            .filter(|&&d| d != -1)
            .map(|&d| d as usize)
            .product();
        let new_shape: Vec<usize> = if neg_count == 1 {
            s.iter()
                .map(|&d| if d == -1 { numel / known } else { d as usize })
                .collect()
        } else {
            s.iter().map(|&d| d as usize).collect()
        };

        if new_shape.iter().product::<usize>() != numel {
            return Err(OnnxError::ShapeMismatch(format!(
                "Reshape: element count mismatch ({numel} vs {new_shape:?})"
            )));
        }

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
        let s: Vec<i64> = shape_t.data.iter().map(|&v| v as i64).collect();
        let result = shape::reshape(x, &s)?;
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
        let perm: Vec<usize> = ctx
            .attrs()
            .ints("perm")
            .iter()
            .map(|&v| v as usize)
            .collect();
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
        Ok(vec![shape::squeeze(x, &axes)])
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
        let ndim = x.ndim();
        let raw_axes: Vec<i64> = if let Some(t) = ctx.optional_input(1) {
            t.data.iter().map(|&v| v as i64).collect()
        } else {
            ctx.attrs().ints("axes").to_vec()
        };
        let resolved_axes: Vec<usize> = if raw_axes.is_empty() {
            (0..ndim).filter(|&i| x.shape[i] == 1).collect()
        } else {
            raw_axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (a + ndim as i64) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        };
        let new_shape: Vec<usize> = x
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| {
                if resolved_axes.contains(&i) && d == 1 {
                    None
                } else {
                    Some(d)
                }
            })
            .collect();
        let new_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };
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
        Ok(vec![shape::unsqueeze(x, &axes)])
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
        let mut new_shape = x.shape.clone();
        let mut sorted_axes = raw_axes.clone();
        sorted_axes.sort();
        for &ax in &sorted_axes {
            let ax = if ax < 0 {
                (ax + new_shape.len() as i64 + 1) as usize
            } else {
                ax as usize
            };
            new_shape.insert(ax, 1);
        }
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
        let ndim = x.ndim();
        let ax = if axis < 0 {
            (axis + ndim as i64) as usize
        } else {
            axis as usize
        };
        let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
        let inner: usize = x.shape[ax..].iter().product::<usize>().max(1);
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
