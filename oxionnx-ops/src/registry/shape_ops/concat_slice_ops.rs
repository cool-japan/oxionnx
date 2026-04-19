//! Operator implementations for concat, slice, expand, and split.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::shape;

// ── Concat ───────────────────────────────────────────────────────────────────

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
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── Slice ────────────────────────────────────────────────────────────────────

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
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── Expand ───────────────────────────────────────────────────────────────────

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
        let shape_in: Vec<usize> = ctx.input(1)?.data.iter().map(|&v| v as usize).collect();
        let out_shape = Tensor::broadcast_shape(&x.shape, &shape_in)?;
        let n: usize = out_shape.iter().product();

        let ndim = out_shape.len();
        let pad = ndim - x.shape.len();
        let padded: Vec<usize> = (0..pad).map(|_| 1).chain(x.shape.iter().copied()).collect();

        let mut out_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = s;
            s *= out_shape[i];
        }
        let mut in_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            in_strides[i] = if padded[i] == 1 { 0 } else { s };
            s *= padded[i];
        }

        let slot = &mut slots[0];
        if slot.data.len() != n {
            slot.data.resize(n, 0.0_f32);
        }
        for (out_idx, out_val) in slot.data.iter_mut().enumerate() {
            let mut rem = out_idx;
            let mut in_idx = 0usize;
            for d in 0..ndim {
                let coord = rem / out_strides[d];
                rem %= out_strides[d];
                in_idx += coord * in_strides[d];
            }
            *out_val = x.data[in_idx];
        }
        slot.shape = out_shape;
        Ok(())
    }
}

// ── Split ────────────────────────────────────────────────────────────────────

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
        let attrs = ctx.attrs();
        let axis = attrs.i("axis", 0);
        let ndim = x.ndim();
        let ax_u = if axis < 0 {
            (axis + ndim as i64) as usize
        } else {
            axis as usize
        };
        if ax_u >= ndim {
            return Err(OnnxError::ShapeMismatch(format!(
                "Split: axis {ax_u} out of range for {ndim}D tensor"
            )));
        }
        let num_outputs = slots.len().max(1);
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
        if split_sizes.is_empty() {
            return Err(OnnxError::Internal("Split: split_sizes is empty".into()));
        }
        let axis_len = x.shape[ax_u];
        let total: usize = split_sizes.iter().sum();
        if total != axis_len {
            return Err(OnnxError::ShapeMismatch(format!(
                "Split: sizes sum {total} != axis len {axis_len}"
            )));
        }
        if split_sizes.len() != slots.len() {
            return Err(OnnxError::Internal(format!(
                "Split: {} chunks but {} slots",
                split_sizes.len(),
                slots.len()
            )));
        }
        let outer: usize = x.shape[..ax_u].iter().product::<usize>().max(1);
        let inner: usize = x.shape[ax_u + 1..].iter().product::<usize>().max(1);
        let mut start = 0usize;
        for (slot_i, &chunk) in split_sizes.iter().enumerate() {
            let n_out = outer * chunk * inner;
            let slot = &mut slots[slot_i];
            if slot.data.len() != n_out {
                slot.data.resize(n_out, 0.0_f32);
            }
            let mut dst = 0usize;
            for o in 0..outer {
                for j in start..start + chunk {
                    let src = o * axis_len * inner + j * inner;
                    slot.data[dst..dst + inner].copy_from_slice(&x.data[src..src + inner]);
                    dst += inner;
                }
            }
            let mut out_shape = x.shape.clone();
            out_shape[ax_u] = chunk;
            slot.shape = out_shape;
            start += chunk;
        }
        Ok(())
    }
}
