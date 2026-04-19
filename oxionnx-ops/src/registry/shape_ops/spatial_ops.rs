//! Operator implementations for tile, depth/space reordering, and reverse-sequence.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::shape;

// ── Tile ─────────────────────────────────────────────────────────────────────

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
        let repeats: Vec<usize> = ctx.input(1)?.data.iter().map(|&v| v as usize).collect();
        let ndim = x.ndim();
        if repeats.len() != ndim {
            return Err(OnnxError::ShapeMismatch(format!(
                "Tile: repeats len {} != tensor ndim {ndim}",
                repeats.len()
            )));
        }
        let out_shape: Vec<usize> = x
            .shape
            .iter()
            .zip(repeats.iter())
            .map(|(&d, &r)| d * r)
            .collect();
        let out_n: usize = out_shape.iter().product();

        let mut out_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = s;
            s *= out_shape[i];
        }
        let mut in_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            in_strides[i] = s;
            s *= x.shape[i];
        }

        let slot = &mut slots[0];
        if slot.data.len() != out_n {
            slot.data.resize(out_n, 0.0_f32);
        }
        for (out_idx, out_val) in slot.data.iter_mut().enumerate() {
            let mut rem = out_idx;
            let mut in_idx = 0usize;
            for d in 0..ndim {
                let coord = rem / out_strides[d];
                rem %= out_strides[d];
                in_idx += (coord % x.shape[d]) * in_strides[d];
            }
            *out_val = x.data[in_idx];
        }
        slot.shape = out_shape;
        Ok(())
    }
}

// ── DepthToSpace ─────────────────────────────────────────────────────────────

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
        let blocksize = attrs.i("blocksize", 1) as usize;
        let mode_str = attrs.s("mode");
        let mode = if mode_str.is_empty() { "DCR" } else { mode_str };
        if x.ndim() != 4 {
            return Err(OnnxError::ShapeMismatch(
                "DepthToSpace: input must be 4D [N,C,H,W]".into(),
            ));
        }
        let (n, c_total, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let r = blocksize;
        if c_total % (r * r) != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "DepthToSpace: channels {c_total} not divisible by blocksize^2 {}",
                r * r
            )));
        }
        let c = c_total / (r * r);
        let oh = h * r;
        let ow = w * r;
        let out_n = n * c * oh * ow;
        let slot = &mut slots[0];
        if slot.data.len() != out_n {
            slot.data.resize(out_n, 0.0_f32);
        }
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        for rh in 0..r {
                            for rw in 0..r {
                                let src_c = if mode == "CRD" {
                                    ci * r * r + rh * r + rw
                                } else {
                                    rh * r * c + rw * c + ci
                                };
                                let src_idx = ((ni * c_total + src_c) * h + hi) * w + wi;
                                let dst_idx = ((ni * c + ci) * oh + hi * r + rh) * ow + wi * r + rw;
                                slot.data[dst_idx] = x.data[src_idx];
                            }
                        }
                    }
                }
            }
        }
        slot.shape = vec![n, c, oh, ow];
        Ok(())
    }
}

// ── SpaceToDepth ─────────────────────────────────────────────────────────────

pub struct SpaceToDepthOp;
impl Operator for SpaceToDepthOp {
    fn op_type(&self) -> &str {
        "SpaceToDepth"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let blocksize = ctx.attrs().i("blocksize", 1) as usize;
        Ok(vec![shape::space_to_depth(ctx.input(0)?, blocksize)?])
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
        let blocksize = ctx.attrs().i("blocksize", 1) as usize;
        if x.ndim() != 4 {
            return Err(OnnxError::ShapeMismatch(
                "SpaceToDepth: input must be 4D [N,C,H,W]".into(),
            ));
        }
        let (n, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let r = blocksize;
        if h % r != 0 || w % r != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "SpaceToDepth: spatial dims {h}x{w} not divisible by blocksize {r}"
            )));
        }
        let oh = h / r;
        let ow = w / r;
        let oc = c * r * r;
        let out_n = n * oc * oh * ow;
        let slot = &mut slots[0];
        if slot.data.len() != out_n {
            slot.data.resize(out_n, 0.0_f32);
        }
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..oh {
                    for wi in 0..ow {
                        for rh in 0..r {
                            for rw in 0..r {
                                let src_idx = ((ni * c + ci) * h + hi * r + rh) * w + wi * r + rw;
                                let dst_c = ci * r * r + rh * r + rw;
                                let dst_idx = ((ni * oc + dst_c) * oh + hi) * ow + wi;
                                slot.data[dst_idx] = x.data[src_idx];
                            }
                        }
                    }
                }
            }
        }
        slot.shape = vec![n, oc, oh, ow];
        Ok(())
    }
}

// ── ReverseSequence ──────────────────────────────────────────────────────────

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
        let sequence_lens = ctx.input(1)?;
        let attrs = ctx.attrs();
        let batch_axis = attrs.i("batch_axis", 1);
        let time_axis = attrs.i("time_axis", 0);
        let ndim = x.ndim();
        if ndim < 2 {
            return Err(OnnxError::ShapeMismatch(
                "ReverseSequence: input must be at least 2D".into(),
            ));
        }
        let ba = if batch_axis < 0 {
            (ndim as i64 + batch_axis) as usize
        } else {
            batch_axis as usize
        };
        let ta = if time_axis < 0 {
            (ndim as i64 + time_axis) as usize
        } else {
            time_axis as usize
        };
        if ba >= ndim || ta >= ndim {
            return Err(OnnxError::ShapeMismatch(format!(
                "ReverseSequence: batch_axis {ba} or time_axis {ta} out of range for {ndim}D"
            )));
        }
        if ba == ta {
            return Err(OnnxError::ShapeMismatch(
                "ReverseSequence: batch_axis and time_axis must differ".into(),
            ));
        }
        let batch_size = x.shape[ba];
        if sequence_lens.numel() != batch_size {
            return Err(OnnxError::ShapeMismatch(format!(
                "ReverseSequence: sequence_lens length {} != batch size {batch_size}",
                sequence_lens.numel()
            )));
        }
        let total = x.numel();
        let slot = &mut slots[0];
        if slot.data.len() != total {
            slot.data.resize(total, 0.0_f32);
        }
        slot.data.copy_from_slice(&x.data);

        let mut strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            strides[i] = s;
            s *= x.shape[i];
        }

        for flat_idx in 0..total {
            let mut rem = flat_idx;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = rem / strides[d];
                rem %= strides[d];
            }
            let batch_idx = coords[ba];
            let time_idx = coords[ta];
            let seq_len = sequence_lens.data[batch_idx] as usize;
            if time_idx < seq_len {
                let mut new_coords = coords.clone();
                new_coords[ta] = seq_len - 1 - time_idx;
                let mut src_flat = 0usize;
                for d in 0..ndim {
                    src_flat += new_coords[d] * strides[d];
                }
                slot.data[flat_idx] = x.data[src_flat];
            }
        }
        slot.shape = x.shape.clone();
        Ok(())
    }
}
