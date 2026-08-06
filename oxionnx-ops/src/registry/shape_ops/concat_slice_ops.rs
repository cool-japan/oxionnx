//! Operator implementations for concat, slice, expand, and split.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::shape;
use crate::shape::basic::normalize_axis;

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

/// Read Expand's `shape` input as non-negative dimension sizes.
///
/// ONNX's `shape` input is a 1-D int64 tensor; unlike Reshape's `-1`/`0` sentinels, Expand
/// defines no meaning for a negative entry. Casting one straight to `usize` (`v as usize`)
/// wraps to a value near `usize::MAX`, which — since the corresponding input dimension is
/// often `1` (an ordinary broadcast), so `Tensor::broadcast_shape` happily accepts the huge
/// target — previously reached `vec![0.0f32; n]` and panicked with "capacity overflow".
fn resolve_expand_shape(shape_tensor: &Tensor) -> Result<Vec<usize>, OnnxError> {
    shape_tensor
        .data
        .iter()
        .map(|&v| {
            let iv = v as i64;
            if iv < 0 {
                Err(OnnxError::ShapeMismatch(format!(
                    "Expand: shape entries must be non-negative, got {iv}"
                )))
            } else {
                Ok(iv as usize)
            }
        })
        .collect()
}

/// Compute the total element count of `shape`, rejecting a `usize` overflow instead of letting
/// it silently wrap (debug builds would instead panic on the wrap) into a nonsensical
/// allocation size.
fn checked_numel(shape: &[usize], op: &str) -> Result<usize, OnnxError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: output element count overflows")))
}

pub struct ExpandOp;
impl Operator for ExpandOp {
    fn op_type(&self) -> &str {
        "Expand"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let new_shape = resolve_expand_shape(ctx.input(1)?)?;
        let out_shape = Tensor::broadcast_shape(&x.shape, &new_shape)?;
        checked_numel(&out_shape, "Expand")?;
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
        let shape_in = resolve_expand_shape(ctx.input(1)?)?;
        let out_shape = Tensor::broadcast_shape(&x.shape, &shape_in)?;
        let n: usize = checked_numel(&out_shape, "Expand")?;

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
///
/// Always returns exactly `n` entries (when `n > 0`), including trailing zero-size chunks: a
/// Split node with `num_outputs` greater than what `axis_len` can fill evenly still declares
/// `num_outputs` graph outputs, and every one of them must be bound to *some* tensor (a
/// zero-size one is a valid, well-defined ONNX tensor) or the graph's later output names go
/// unresolved.
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
        // Bounds-checked up front: previously this indexed `x.shape[ax_u]` below with no range
        // check (unlike `execute_into_slots`), so an out-of-range `axis` attribute panicked
        // here instead of returning a typed error.
        let ax_u = normalize_axis(axis, ndim).map_err(OnnxError::ShapeMismatch)?;
        let num_outputs = ctx.node.outputs.len().max(1);
        let split_attr = attrs.ints("split");
        let split_sizes: Vec<usize> = if let Some(t) = ctx.optional_input(1) {
            t.data.iter().map(|&v| v as usize).collect()
        } else if !split_attr.is_empty() {
            split_attr.iter().map(|&v| v as usize).collect()
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
        let ax_u = normalize_axis(axis, ndim).map_err(OnnxError::ShapeMismatch)?;
        let num_outputs = slots.len().max(1);
        let split_attr = attrs.ints("split");
        let split_sizes: Vec<usize> = if let Some(t) = ctx.optional_input(1) {
            t.data.iter().map(|&v| v as usize).collect()
        } else if !split_attr.is_empty() {
            split_attr.iter().map(|&v| v as usize).collect()
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
        // Empty-slice products are already 1; clamping via `.max(1)` would corrupt a
        // genuinely zero-size leading/trailing dim (see `shape::split`).
        let outer: usize = x.shape[..ax_u].iter().product();
        let inner: usize = x.shape[ax_u + 1..].iter().product();
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
