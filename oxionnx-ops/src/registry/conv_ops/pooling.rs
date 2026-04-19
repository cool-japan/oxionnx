//! Pooling operator implementations: MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::conv;

// ── MaxPool ─────────────────────────────────────────────────────────────────

pub struct MaxPoolOp;
impl Operator for MaxPoolOp {
    fn op_type(&self) -> &str {
        "MaxPool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let ks_v = attrs.ints("kernel_shape");
        let kernel_shape = [ks_v[0] as usize, ks_v[1] as usize];
        let strides_v = attrs.ints("strides");
        let strides = [
            strides_v.first().copied().unwrap_or(1) as usize,
            strides_v.get(1).copied().unwrap_or(1) as usize,
        ];
        let pads_v = attrs.ints("pads");
        let pads = [
            pads_v.first().copied().unwrap_or(0) as usize,
            pads_v.get(1).copied().unwrap_or(0) as usize,
            pads_v.get(2).copied().unwrap_or(0) as usize,
            pads_v.get(3).copied().unwrap_or(0) as usize,
        ];
        Ok(vec![conv::max_pool2d(input, kernel_shape, strides, pads)])
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
                "MaxPoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let ks_v = attrs.ints("kernel_shape");
        let kh = ks_v
            .first()
            .copied()
            .ok_or_else(|| OnnxError::Internal("MaxPoolOp: kernel_shape missing".into()))?
            as usize;
        let kw =
            ks_v.get(1).copied().ok_or_else(|| {
                OnnxError::Internal("MaxPoolOp: kernel_shape requires 2 dims".into())
            })? as usize;
        let strides_v = attrs.ints("strides");
        let s_h = strides_v.first().copied().unwrap_or(1) as usize;
        let s_w = strides_v.get(1).copied().unwrap_or(1) as usize;
        let pads_v = attrs.ints("pads");
        let p_top = pads_v.first().copied().unwrap_or(0) as usize;
        let p_left = pads_v.get(1).copied().unwrap_or(0) as usize;
        let p_bottom = pads_v.get(2).copied().unwrap_or(0) as usize;
        let p_right = pads_v.get(3).copied().unwrap_or(0) as usize;

        let n = input.shape[0];
        let c = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];
        let oh = (h + p_top + p_bottom - kh) / s_h + 1;
        let ow = (w + p_left + p_right - kw) / s_w + 1;
        let out_shape = vec![n, c, oh, ow];
        let total: usize = n * c * oh * ow;

        if slots[0].data.len() != total {
            slots[0].data.resize(total, f32::NEG_INFINITY);
        }
        slots[0].shape = out_shape;

        for batch in 0..n {
            for ch in 0..c {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut max_val = f32::NEG_INFINITY;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * s_h + ky) as isize - p_top as isize;
                                let ix = (ox * s_w + kx) as isize - p_left as isize;
                                if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                    let idx =
                                        ((batch * c + ch) * h + iy as usize) * w + ix as usize;
                                    if input.data[idx] > max_val {
                                        max_val = input.data[idx];
                                    }
                                }
                            }
                        }
                        slots[0].data[((batch * c + ch) * oh + oy) * ow + ox] = max_val;
                    }
                }
            }
        }
        Ok(())
    }
}

// ── AveragePool ─────────────────────────────────────────────────────────────

pub struct AveragePoolOp;
impl Operator for AveragePoolOp {
    fn op_type(&self) -> &str {
        "AveragePool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let ks_v = attrs.ints("kernel_shape");
        let kernel_shape = [ks_v[0] as usize, ks_v[1] as usize];
        let strides_v = attrs.ints("strides");
        let strides = [
            strides_v.first().copied().unwrap_or(1) as usize,
            strides_v.get(1).copied().unwrap_or(1) as usize,
        ];
        let pads_v = attrs.ints("pads");
        let pads = [
            pads_v.first().copied().unwrap_or(0) as usize,
            pads_v.get(1).copied().unwrap_or(0) as usize,
            pads_v.get(2).copied().unwrap_or(0) as usize,
            pads_v.get(3).copied().unwrap_or(0) as usize,
        ];
        let count_include_pad = attrs.i("count_include_pad", 0) != 0;
        Ok(vec![conv::avg_pool2d(
            input,
            kernel_shape,
            strides,
            pads,
            count_include_pad,
        )])
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
                "AveragePoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let ks_v = attrs.ints("kernel_shape");
        let kh = ks_v
            .first()
            .copied()
            .ok_or_else(|| OnnxError::Internal("AveragePoolOp: kernel_shape missing".into()))?
            as usize;
        let kw = ks_v.get(1).copied().ok_or_else(|| {
            OnnxError::Internal("AveragePoolOp: kernel_shape requires 2 dims".into())
        })? as usize;
        let strides_v = attrs.ints("strides");
        let s_h = strides_v.first().copied().unwrap_or(1) as usize;
        let s_w = strides_v.get(1).copied().unwrap_or(1) as usize;
        let pads_v = attrs.ints("pads");
        let p_top = pads_v.first().copied().unwrap_or(0) as usize;
        let p_left = pads_v.get(1).copied().unwrap_or(0) as usize;
        let p_bottom = pads_v.get(2).copied().unwrap_or(0) as usize;
        let p_right = pads_v.get(3).copied().unwrap_or(0) as usize;
        let count_include_pad = attrs.i("count_include_pad", 0) != 0;

        let n = input.shape[0];
        let c = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];
        let oh = (h + p_top + p_bottom - kh) / s_h + 1;
        let ow = (w + p_left + p_right - kw) / s_w + 1;
        let out_shape = vec![n, c, oh, ow];
        let total: usize = n * c * oh * ow;

        if slots[0].data.len() != total {
            slots[0].data.resize(total, 0.0_f32);
        }
        slots[0].shape = out_shape;

        for batch in 0..n {
            for ch in 0..c {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut sum = 0.0_f32;
                        let mut count = 0_usize;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * s_h + ky) as isize - p_top as isize;
                                let ix = (ox * s_w + kx) as isize - p_left as isize;
                                if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                    let idx =
                                        ((batch * c + ch) * h + iy as usize) * w + ix as usize;
                                    sum += input.data[idx];
                                    count += 1;
                                } else if count_include_pad {
                                    count += 1;
                                }
                            }
                        }
                        let divisor = if count_include_pad { kh * kw } else { count };
                        slots[0].data[((batch * c + ch) * oh + oy) * ow + ox] = if divisor > 0 {
                            sum / divisor as f32
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        Ok(())
    }
}

// ── GlobalAveragePool / GlobalMaxPool ───────────────────────────────────────

pub struct GlobalAveragePoolOp;
impl Operator for GlobalAveragePoolOp {
    fn op_type(&self) -> &str {
        "GlobalAveragePool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![conv::global_avg_pool(ctx.input(0)?)])
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
                "GlobalAveragePoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let x = ctx.input(0)?;
        // Degenerate case: fewer than 3 dims — copy input directly into slot
        if x.ndim() < 3 {
            slots[0].data.resize(x.data.len(), 0.0_f32);
            slots[0].data.copy_from_slice(&x.data);
            slots[0].shape = x.shape.clone();
            return Ok(());
        }
        let n = x.shape[0];
        let c = x.shape[1];
        let spatial: usize = x.shape[2..].iter().product();
        let total = n * c;
        if slots[0].data.len() != total {
            slots[0].data.resize(total, 0.0_f32);
        }
        let mut out_shape = vec![n, c];
        out_shape.extend(vec![1_usize; x.ndim() - 2]);
        slots[0].shape = out_shape;
        for ni in 0..n {
            for ci in 0..c {
                let base = ni * c * spatial + ci * spatial;
                let sum: f32 = x.data[base..base + spatial].iter().sum();
                slots[0].data[ni * c + ci] = sum / spatial as f32;
            }
        }
        Ok(())
    }
}

pub struct GlobalMaxPoolOp;
impl Operator for GlobalMaxPoolOp {
    fn op_type(&self) -> &str {
        "GlobalMaxPool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![conv::global_max_pool(ctx.input(0)?)])
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
                "GlobalMaxPoolOp: expected 1 output slot, got {}",
                slots.len()
            )));
        }
        let x = ctx.input(0)?;
        if x.ndim() < 3 {
            slots[0].data.resize(x.data.len(), 0.0_f32);
            slots[0].data.copy_from_slice(&x.data);
            slots[0].shape = x.shape.clone();
            return Ok(());
        }
        let n = x.shape[0];
        let c = x.shape[1];
        let spatial: usize = x.shape[2..].iter().product();
        let total = n * c;
        if slots[0].data.len() != total {
            slots[0].data.resize(total, f32::NEG_INFINITY);
        }
        let mut out_shape = vec![n, c];
        out_shape.extend(vec![1_usize; x.ndim() - 2]);
        slots[0].shape = out_shape;
        for ni in 0..n {
            for ci in 0..c {
                let base = ni * c * spatial + ci * spatial;
                let max_val = x.data[base..base + spatial]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                slots[0].data[ni * c + ci] = max_val;
            }
        }
        Ok(())
    }
}
