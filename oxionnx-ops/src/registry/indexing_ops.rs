//! Operator trait implementations for indexing and scatter/gather operations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::indexing;

// ── Gather ──────────────────────────────────────────────────────────────────

pub struct GatherOp;
impl Operator for GatherOp {
    fn op_type(&self) -> &str {
        "Gather"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let idx = ctx.input(1)?;
        let axis = ctx.attrs().i("axis", 0);
        Ok(vec![indexing::gather(x, idx, axis)?])
    }
}

// ── GatherElements ──────────────────────────────────────────────────────────

pub struct GatherElementsOp;
impl Operator for GatherElementsOp {
    fn op_type(&self) -> &str {
        "GatherElements"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let idx = ctx.input(1)?;
        let axis = ctx.attrs().i("axis", 0);
        Ok(vec![indexing::gather_elements(x, idx, axis)?])
    }
}

// ── GatherND ────────────────────────────────────────────────────────────────

pub struct GatherNDOp;
impl Operator for GatherNDOp {
    fn op_type(&self) -> &str {
        "GatherND"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let batch_dims = ctx.attrs().i("batch_dims", 0);
        Ok(vec![indexing::gather_nd(
            ctx.input(0)?,
            ctx.input(1)?,
            batch_dims,
        )?])
    }
}

// ── Where ───────────────────────────────────────────────────────────────────

pub struct WhereOp;
impl Operator for WhereOp {
    fn op_type(&self) -> &str {
        "Where"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let cond = ctx.input(0)?;
        let x = ctx.input(1)?;
        let y = ctx.input(2)?;
        Ok(vec![indexing::where_op(cond, x, y)?])
    }
}

// ── ScatterElements ─────────────────────────────────────────────────────────

pub struct ScatterElementsOp;
impl Operator for ScatterElementsOp {
    fn op_type(&self) -> &str {
        "ScatterElements"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let data = ctx.input(0)?;
        let indices = ctx.input(1)?;
        let updates = ctx.input(2)?;
        let axis = ctx.attrs().i("axis", 0);
        Ok(vec![indexing::scatter_elements(
            data, indices, updates, axis,
        )?])
    }
}

// ── ScatterND ───────────────────────────────────────────────────────────────

pub struct ScatterNDOp;
impl Operator for ScatterNDOp {
    fn op_type(&self) -> &str {
        "ScatterND"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let data = ctx.input(0)?;
        let indices = ctx.input(1)?;
        let updates = ctx.input(2)?;
        Ok(vec![indexing::scatter_nd(data, indices, updates)?])
    }
}

// ── QuantizeLinear ──────────────────────────────────────────────────────────

pub struct QuantizeLinearOp;
impl Operator for QuantizeLinearOp {
    fn op_type(&self) -> &str {
        "QuantizeLinear"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let zp = ctx.optional_input(2);
        Ok(vec![indexing::quantize_linear(x, scale, zp)?])
    }
}

// ── DequantizeLinear ────────────────────────────────────────────────────────

pub struct DequantizeLinearOp;
impl Operator for DequantizeLinearOp {
    fn op_type(&self) -> &str {
        "DequantizeLinear"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let zp = ctx.optional_input(2);
        Ok(vec![indexing::dequantize_linear(x, scale, zp)?])
    }
}

// ── OneHot ──────────────────────────────────────────────────────────────────

pub struct OneHotOp;
impl Operator for OneHotOp {
    fn op_type(&self) -> &str {
        "OneHot"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let indices = ctx.input(0)?;
        let depth = ctx.input(1)?.data[0] as usize;
        let values_t = ctx.input(2)?;
        let off_val = values_t.data[0];
        let on_val = if values_t.data.len() > 1 {
            values_t.data[1]
        } else {
            1.0
        };
        let axis = ctx.attrs().i("axis", -1);
        Ok(vec![indexing::one_hot(
            indices,
            depth,
            (off_val, on_val),
            axis,
        )?])
    }
}

// ── Compress ────────────────────────────────────────────────────────────────

pub struct CompressOp;
impl Operator for CompressOp {
    fn op_type(&self) -> &str {
        "Compress"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axis = {
            let v = ctx.attrs().i("axis", i64::MIN);
            if v == i64::MIN {
                None
            } else {
                Some(v)
            }
        };
        Ok(vec![indexing::compress(
            ctx.input(0)?,
            ctx.input(1)?,
            axis,
        )?])
    }
}

// ── Unique ──────────────────────────────────────────────────────────────────

pub struct UniqueOp;
impl Operator for UniqueOp {
    fn op_type(&self) -> &str {
        "Unique"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let axis = {
            let v = attrs.i("axis", i64::MIN);
            if v == i64::MIN {
                None
            } else {
                Some(v)
            }
        };
        let sorted = attrs.i("sorted", 1) != 0;
        let (y, idx, inv, counts) = indexing::unique(ctx.input(0)?, axis, sorted)?;
        Ok(vec![y, idx, inv, counts])
    }
}
