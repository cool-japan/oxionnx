//! Operator trait implementations for shape manipulation operations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::shape;

// ── Reshape ─────────────────────────────────────────────────────────────────

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
}

// ── Transpose ───────────────────────────────────────────────────────────────

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
}

// ── Squeeze ─────────────────────────────────────────────────────────────────

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
}

// ── Unsqueeze ───────────────────────────────────────────────────────────────

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
}

// ── Flatten ─────────────────────────────────────────────────────────────────

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
}

// ── Concat ──────────────────────────────────────────────────────────────────

pub struct ConcatOp;
impl Operator for ConcatOp {
    fn op_type(&self) -> &str {
        "Concat"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axis = ctx.attrs().i("axis", 0);
        let tensors: Vec<&Tensor> = ctx.inputs.iter().filter_map(|opt| *opt).collect();
        Ok(vec![shape::concat(&tensors, axis)?])
    }
}

// ── Slice ───────────────────────────────────────────────────────────────────

pub struct SliceOp;
impl Operator for SliceOp {
    fn op_type(&self) -> &str {
        "Slice"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let starts: Vec<i64> = ctx.input(1)?.data.iter().map(|&v| v as i64).collect();
        let ends: Vec<i64> = ctx.input(2)?.data.iter().map(|&v| v as i64).collect();
        let axes: Option<Vec<i64>> = ctx
            .optional_input(3)
            .map(|t| t.data.iter().map(|&v| v as i64).collect());
        let steps: Option<Vec<i64>> = ctx
            .optional_input(4)
            .map(|t| t.data.iter().map(|&v| v as i64).collect());
        Ok(vec![shape::slice(
            x,
            &starts,
            &ends,
            axes.as_deref(),
            steps.as_deref(),
        )?])
    }
}

// ── Expand ──────────────────────────────────────────────────────────────────

pub struct ExpandOp;
impl Operator for ExpandOp {
    fn op_type(&self) -> &str {
        "Expand"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let new_shape: Vec<usize> = ctx.input(1)?.data.iter().map(|&v| v as usize).collect();
        Ok(vec![crate::indexing::expand(x, &new_shape)?])
    }
}

// ── Split ───────────────────────────────────────────────────────────────────

/// Build equal-sized split chunks for a given axis length and count.
fn equal_split(axis_len: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let chunk = axis_len.div_ceil(n);
    (0..n)
        .map(|i| {
            let start = i * chunk;
            (start + chunk).min(axis_len).saturating_sub(start)
        })
        .filter(|&s| s > 0)
        .collect()
}

pub struct SplitOp;
impl Operator for SplitOp {
    fn op_type(&self) -> &str {
        "Split"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let attrs = ctx.attrs();
        let axis = attrs.i("axis", 0);
        let ndim = x.ndim();
        let ax_u = if axis < 0 {
            (axis + ndim as i64) as usize
        } else {
            axis as usize
        };
        let num_outputs = ctx.node.outputs.len().max(1);
        let split_sizes: Vec<usize> = if let Some(t) = ctx.optional_input(1) {
            t.data.iter().map(|&v| v as usize).collect()
        } else if attrs.i("num_outputs", 0) > 0 {
            equal_split(
                x.shape[ax_u],
                attrs.i("num_outputs", num_outputs as i64) as usize,
            )
        } else {
            equal_split(x.shape[ax_u], num_outputs)
        };
        Ok(shape::split(x, axis, &split_sizes)?)
    }
}

// ── Tile ────────────────────────────────────────────────────────────────────

pub struct TileOp;
impl Operator for TileOp {
    fn op_type(&self) -> &str {
        "Tile"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let repeats: Vec<usize> = ctx.input(1)?.data.iter().map(|&v| v as usize).collect();
        Ok(vec![shape::tile(x, &repeats)?])
    }
}

// ── DepthToSpace ────────────────────────────────────────────────────────────

pub struct DepthToSpaceOp;
impl Operator for DepthToSpaceOp {
    fn op_type(&self) -> &str {
        "DepthToSpace"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let blocksize = attrs.i("blocksize", 1) as usize;
        let mode = attrs.s("mode");
        let mode = if mode.is_empty() { "DCR" } else { mode };
        Ok(vec![shape::depth_to_space(ctx.input(0)?, blocksize, mode)?])
    }
}

// ── SpaceToDepth ────────────────────────────────────────────────────────────

pub struct SpaceToDepthOp;
impl Operator for SpaceToDepthOp {
    fn op_type(&self) -> &str {
        "SpaceToDepth"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let blocksize = ctx.attrs().i("blocksize", 1) as usize;
        Ok(vec![shape::space_to_depth(ctx.input(0)?, blocksize)?])
    }
}

// ── ReverseSequence ─────────────────────────────────────────────────────────

pub struct ReverseSequenceOp;
impl Operator for ReverseSequenceOp {
    fn op_type(&self) -> &str {
        "ReverseSequence"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let batch_axis = attrs.i("batch_axis", 1);
        let time_axis = attrs.i("time_axis", 0);
        Ok(vec![shape::reverse_sequence(
            ctx.input(0)?,
            ctx.input(1)?,
            batch_axis,
            time_axis,
        )?])
    }
}
