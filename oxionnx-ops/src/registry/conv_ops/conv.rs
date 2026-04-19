//! ConvOp and ConvTransposeOp operator implementations.

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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &oxionnx_core::OpContext<'_>,
        slots: &mut [oxionnx_core::Tensor],
    ) -> Result<(), oxionnx_core::OnnxError> {
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

        let out_shape = conv::compute_conv2d_out_shape(
            &input.shape,
            &weight.shape,
            &strides,
            &pads,
            &dilations,
        );
        let out_len: usize = out_shape.iter().product();
        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        slots[0].shape.clone_from(&out_shape);
        conv::conv2d_into(
            input,
            weight,
            bias,
            strides,
            pads,
            dilations,
            group,
            &mut slots[0].data,
            &out_shape,
        );

        // Apply fused activation in-place — mirrors execute() exactly.
        let activation = ctx.attrs().s("activation");
        if activation == "relu" {
            for v in slots[0].data.iter_mut() {
                *v = v.max(0.0);
            }
        } else if activation == "clip" {
            let min_val = ctx.attrs().f("activation_min", f32::NEG_INFINITY);
            let max_val = ctx.attrs().f("activation_max", f32::INFINITY);
            for v in slots[0].data.iter_mut() {
                *v = v.clamp(min_val, max_val);
            }
        }

        Ok(())
    }

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, TypedTensor};

        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("ConvOp: missing input".into()))?;
        let weight = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("ConvOp: missing weight".into()))?;
        let bias = ctx.input(2);

        let strides_v = ctx.attrs().ints("strides");
        let strides = [
            strides_v.first().copied().unwrap_or(1) as usize,
            strides_v.get(1).copied().unwrap_or(1) as usize,
        ];
        let pads_v = ctx.attrs().ints("pads");
        let pads = [
            pads_v.first().copied().unwrap_or(0) as usize,
            pads_v.get(1).copied().unwrap_or(0) as usize,
            pads_v.get(2).copied().unwrap_or(0) as usize,
            pads_v.get(3).copied().unwrap_or(0) as usize,
        ];
        let dilations_v = ctx.attrs().ints("dilations");
        let dilations = [
            dilations_v.first().copied().unwrap_or(1) as usize,
            dilations_v.get(1).copied().unwrap_or(1) as usize,
        ];
        let group = ctx.attrs().i("group", 1) as usize;

        let activation = ctx.attrs().s("activation");
        let activation_min = ctx.attrs().f("activation_min", f32::NEG_INFINITY);
        let activation_max = ctx.attrs().f("activation_max", f32::INFINITY);

        let out_shape = conv::compute_conv2d_out_shape(
            &input.shape,
            &weight.shape,
            &strides,
            &pads,
            &dilations,
        );
        let out_len: usize = out_shape.iter().product();

        let params = crate::conv_typed::Conv2dParams {
            strides,
            pads,
            dilations,
            group,
        };
        let act = crate::conv_typed::FusedActivation {
            activation,
            min: activation_min,
            max: activation_max,
        };

        match (&input.storage, &weight.storage) {
            // ── F32: delegate to existing execute() logic ──
            (TensorStorage::F32(_), TensorStorage::F32(_)) => {
                oxionnx_core::default_typed_via_f32(self, ctx)
            }

            // ── F16 × F16 → F16 ──
            (TensorStorage::F16(ib), TensorStorage::F16(wb)) => {
                let bias_bits = if let Some(b) = bias {
                    match &b.storage {
                        TensorStorage::F16(bb) => Some(bb.as_slice()),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let inputs = crate::conv_typed::Conv2dInputs {
                    input_bits: ib,
                    input_shape: &input.shape,
                    weight_bits: wb,
                    weight_shape: &weight.shape,
                    bias_bits,
                };
                let mut out_bits = vec![0u16; out_len];
                crate::conv_typed::conv2d_f16(&inputs, &params, &act, &mut out_bits, &out_shape);
                Ok(vec![TypedTensor::new(
                    TensorStorage::F16(out_bits),
                    out_shape,
                )])
            }

            // ── BF16 × BF16 → BF16 ──
            (TensorStorage::BF16(ib), TensorStorage::BF16(wb)) => {
                let bias_bits = if let Some(b) = bias {
                    match &b.storage {
                        TensorStorage::BF16(bb) => Some(bb.as_slice()),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let inputs = crate::conv_typed::Conv2dInputs {
                    input_bits: ib,
                    input_shape: &input.shape,
                    weight_bits: wb,
                    weight_shape: &weight.shape,
                    bias_bits,
                };
                let mut out_bits = vec![0u16; out_len];
                crate::conv_typed::conv2d_bf16(&inputs, &params, &act, &mut out_bits, &out_shape);
                Ok(vec![TypedTensor::new(
                    TensorStorage::BF16(out_bits),
                    out_shape,
                )])
            }

            // ── Mixed / unsupported dtypes: fall back to f32 round-trip ──
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
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

        let out_shape = conv::compute_conv_transpose2d_out_shape(
            &input.shape,
            &weight.shape,
            &strides,
            &pads,
            &output_padding,
            &dilations,
            group,
        );
        let out_len: usize = out_shape.iter().product();
        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        slots[0].shape.clone_from(&out_shape);
        conv::conv_transpose2d_into(
            input,
            weight,
            bias,
            &strides,
            &pads,
            &output_padding,
            &dilations,
            group,
            &mut slots[0].data,
            &out_shape,
        )?;

        // ConvTransposeOp has no fused activation.
        Ok(())
    }

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, TypedTensor};

        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("ConvTransposeOp: missing input".into()))?;
        let weight = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("ConvTransposeOp: missing weight".into()))?;
        let bias = ctx.input(2);

        let strides_v = ctx.attrs().ints("strides");
        let strides = [
            strides_v.first().copied().unwrap_or(1) as usize,
            strides_v.get(1).copied().unwrap_or(1) as usize,
        ];
        let pads_v = ctx.attrs().ints("pads");
        let pads = [
            pads_v.first().copied().unwrap_or(0) as usize,
            pads_v.get(1).copied().unwrap_or(0) as usize,
            pads_v.get(2).copied().unwrap_or(0) as usize,
            pads_v.get(3).copied().unwrap_or(0) as usize,
        ];
        let output_padding_v = ctx.attrs().ints("output_padding");
        let output_padding = [
            output_padding_v.first().copied().unwrap_or(0) as usize,
            output_padding_v.get(1).copied().unwrap_or(0) as usize,
        ];
        let dilations_v = ctx.attrs().ints("dilations");
        let dilations = [
            dilations_v.first().copied().unwrap_or(1) as usize,
            dilations_v.get(1).copied().unwrap_or(1) as usize,
        ];
        let group = ctx.attrs().i("group", 1) as usize;

        let out_shape = conv::compute_conv_transpose2d_out_shape(
            &input.shape,
            &weight.shape,
            &strides,
            &pads,
            &output_padding,
            &dilations,
            group,
        );
        let out_len: usize = out_shape.iter().product();

        let params = crate::conv_typed::ConvTranspose2dParams {
            strides,
            pads,
            output_padding,
            dilations,
            group,
        };

        match (&input.storage, &weight.storage) {
            // ── F32: delegate to existing execute() logic ──
            (TensorStorage::F32(_), TensorStorage::F32(_)) => {
                oxionnx_core::default_typed_via_f32(self, ctx)
            }

            // ── F16 × F16 → F16 ──
            (TensorStorage::F16(ib), TensorStorage::F16(wb)) => {
                let bias_bits = if let Some(b) = bias {
                    match &b.storage {
                        TensorStorage::F16(bb) => Some(bb.as_slice()),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let inputs = crate::conv_typed::ConvTranspose2dInputs {
                    input_bits: ib,
                    input_shape: &input.shape,
                    weight_bits: wb,
                    weight_shape: &weight.shape,
                    bias_bits,
                };
                let mut out_bits = vec![0u16; out_len];
                crate::conv_typed::conv_transpose2d_f16(
                    &inputs,
                    &params,
                    &mut out_bits,
                    &out_shape,
                )
                .map_err(OnnxError::ShapeMismatch)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::F16(out_bits),
                    out_shape,
                )])
            }

            // ── BF16 × BF16 → BF16 ──
            (TensorStorage::BF16(ib), TensorStorage::BF16(wb)) => {
                let bias_bits = if let Some(b) = bias {
                    match &b.storage {
                        TensorStorage::BF16(bb) => Some(bb.as_slice()),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let inputs = crate::conv_typed::ConvTranspose2dInputs {
                    input_bits: ib,
                    input_shape: &input.shape,
                    weight_bits: wb,
                    weight_shape: &weight.shape,
                    bias_bits,
                };
                let mut out_bits = vec![0u16; out_len];
                crate::conv_typed::conv_transpose2d_bf16(
                    &inputs,
                    &params,
                    &mut out_bits,
                    &out_shape,
                )
                .map_err(OnnxError::ShapeMismatch)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::BF16(out_bits),
                    out_shape,
                )])
            }

            // ── Mixed / unsupported dtypes: fall back to f32 round-trip ──
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }
}
