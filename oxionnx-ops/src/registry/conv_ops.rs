//! Operator trait implementations for convolution and pooling operations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::conv;

// ── Conv ────────────────────────────────────────────────────────────────────

pub struct ConvOp;
impl Operator for ConvOp {
    fn op_type(&self) -> &str {
        "Conv"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let weight = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
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
        let dilations_v = attrs.ints("dilations");
        let dilations = [
            dilations_v.first().copied().unwrap_or(1) as usize,
            dilations_v.get(1).copied().unwrap_or(1) as usize,
        ];
        let group = attrs.i("group", 1) as usize;
        let mut result = conv::conv2d(input, weight, bias, strides, pads, dilations, group);

        // Apply fused activation if set by the optimizer
        let activation = ctx.attrs().s("activation");
        if activation == "relu" {
            for v in result.data.iter_mut() {
                *v = v.max(0.0);
            }
        } else if activation == "clip" {
            let min_val = ctx.attrs().f("activation_min", f32::NEG_INFINITY);
            let max_val = ctx.attrs().f("activation_max", f32::INFINITY);
            for v in result.data.iter_mut() {
                *v = v.clamp(min_val, max_val);
            }
        }

        Ok(vec![result])
    }
}

// ── ConvTranspose ───────────────────────────────────────────────────────────

pub struct ConvTransposeOp;
impl Operator for ConvTransposeOp {
    fn op_type(&self) -> &str {
        "ConvTranspose"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let weight = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
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
        let output_padding_v = attrs.ints("output_padding");
        let output_padding = [
            output_padding_v.first().copied().unwrap_or(0) as usize,
            output_padding_v.get(1).copied().unwrap_or(0) as usize,
        ];
        let dilations_v = attrs.ints("dilations");
        let dilations = [
            dilations_v.first().copied().unwrap_or(1) as usize,
            dilations_v.get(1).copied().unwrap_or(1) as usize,
        ];
        let group = attrs.i("group", 1) as usize;
        Ok(vec![conv::conv_transpose2d(
            input,
            weight,
            bias,
            strides,
            pads,
            output_padding,
            dilations,
            group,
        )?])
    }
}

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
}

pub struct GlobalMaxPoolOp;
impl Operator for GlobalMaxPoolOp {
    fn op_type(&self) -> &str {
        "GlobalMaxPool"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![conv::global_max_pool(ctx.input(0)?)])
    }
}

// ── Pad ─────────────────────────────────────────────────────────────────────

pub struct PadOp;
impl Operator for PadOp {
    fn op_type(&self) -> &str {
        "Pad"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let pads_tensor = ctx.input(1)?;
        let pads_vals: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
        let constant_value = ctx.optional_input(2).map(|t| t.data[0]).unwrap_or(0.0);
        let mode = ctx.attrs().s("mode");
        let mode = if mode.is_empty() { "constant" } else { mode };
        Ok(vec![crate::shape::pad(
            input,
            &pads_vals,
            mode,
            constant_value,
        )])
    }
}

// ── Resize ──────────────────────────────────────────────────────────────────

pub struct ResizeOp;
impl Operator for ResizeOp {
    fn op_type(&self) -> &str {
        "Resize"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let scales: Option<Vec<f32>> = ctx.optional_input(2).and_then(|t| {
            if t.numel() > 0 {
                Some(t.data.clone())
            } else {
                None
            }
        });
        let sizes: Option<Vec<usize>> = ctx.optional_input(3).and_then(|t| {
            if t.numel() > 0 {
                Some(t.data.iter().map(|&v| v as usize).collect())
            } else {
                None
            }
        });
        let attrs = ctx.attrs();
        let mode = attrs.s("mode");
        let mode = if mode.is_empty() { "nearest" } else { mode };
        let coord_transform = attrs.s("coordinate_transformation_mode");
        let coord_transform = if coord_transform.is_empty() {
            "half_pixel"
        } else {
            coord_transform
        };
        Ok(vec![crate::resize::resize(
            input,
            scales.as_deref(),
            sizes.as_deref(),
            mode,
            coord_transform,
        )])
    }
}
