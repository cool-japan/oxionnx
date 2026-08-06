//! `QLinearConv` and `ConvInteger` operator implementations.

use oxionnx_core::{
    DType, OnnxError, OpContext, Operator, Tensor, TensorStorage, TypedOpContext, TypedTensor,
};

use super::conv_kernel::{integer_conv, ConvZeroPoints, IntConvGeometry};
use super::{project_to_f32, saturate_i32, scale_input, zero_point_lanes, SatRange};

// ── QLinearConv ─────────────────────────────────────────────────────────────

/// ONNX `QLinearConv` (opset 10+).
///
/// ```text
/// y = saturate(round(((x - x_zp) (*) (w - w_zp) + B) * (x_scale * w_scale / y_scale)) + y_zp)
/// ```
///
/// Inputs, in order: `x, x_scale, x_zero_point, w, w_scale, w_zero_point,
/// y_scale, y_zero_point[, B]`. `w_scale` / `w_zero_point` may be scalars
/// (per-tensor) or 1-D of length `M` (per output channel); `B` is **int32
/// already in the `x_scale * w_scale` domain** and is added straight into the
/// integer accumulator, *not* a float bias.
///
/// See `SatRange::infer` for how the output's uint8-vs-int8 saturation range
/// is determined on this dtype-erased runtime.
pub struct QLinearConvOp;

impl QLinearConvOp {
    /// `range_override`, when `Some`, replaces [`SatRange::infer`]'s
    /// value-based cascade with the exact range [`SatRange::for_dtype`]
    /// resolved from `y_zero_point`'s declared dtype — see
    /// [`Operator::execute_typed`] below, the only caller that can supply
    /// one.
    fn run(
        &self,
        ctx: &OpContext<'_>,
        range_override: Option<SatRange>,
    ) -> Result<Tensor, OnnxError> {
        let op = "QLinearConv";
        let x = ctx.input(0)?;
        let x_scale = scale_input(ctx.input(1)?, "QLinearConv: x_scale")?;
        let x_zp_lanes = zero_point_lanes(ctx.optional_input(2), "QLinearConv: x_zero_point")?;
        let w = ctx.input(3)?;
        let w_scale_t = ctx.input(4)?;
        let w_zp_lanes = zero_point_lanes(ctx.optional_input(5), "QLinearConv: w_zero_point")?;
        let y_scale = scale_input(ctx.input(6)?, "QLinearConv: y_scale")?;
        let y_zp_lanes = zero_point_lanes(ctx.optional_input(7), "QLinearConv: y_zero_point")?;
        let bias = ctx.optional_input(8).filter(|t| !t.data.is_empty());

        if w_scale_t.data.is_empty() {
            return Err(OnnxError::InvalidModel(
                "QLinearConv: w_scale is empty".into(),
            ));
        }
        let x_zp = *x_zp_lanes.first().unwrap_or(&0);
        let y_zp = *y_zp_lanes.first().unwrap_or(&0);

        let geo = IntConvGeometry::from_attrs(ctx.attrs(), &x.shape, &w.shape, op)?;
        let out = integer_conv(
            x,
            w,
            &ConvZeroPoints {
                x: x_zp,
                w: w_zp_lanes,
            },
            bias,
            &geo,
            op,
        )?;

        let c_out = w.shape[0];
        let per_channel = w_scale_t.data.len() > 1;
        if per_channel && w_scale_t.data.len() != c_out {
            return Err(OnnxError::ShapeMismatch(format!(
                "QLinearConv: w_scale has {} entries, expected 1 or {c_out} (per output channel)",
                w_scale_t.data.len()
            )));
        }
        for (idx, &s) in w_scale_t.data.iter().enumerate() {
            if !s.is_finite() {
                return Err(OnnxError::InvalidModel(format!(
                    "QLinearConv: w_scale[{idx}] must be finite, got {s}"
                )));
            }
        }

        let range = range_override.unwrap_or_else(|| SatRange::infer(&y_zp_lanes, &[x_zp]));
        let spatial: usize = geo.out_shape[2..].iter().product();
        let mut data = vec![0.0_f32; out.acc.len()];
        for (flat, &acc) in out.acc.iter().enumerate() {
            let oc = (flat / spatial) % c_out;
            let w_scale = if per_channel {
                w_scale_t.data[oc]
            } else {
                w_scale_t.data[0]
            };
            // The combined scale is formed in f32 and the product taken in f64,
            // exactly as the ONNX reference implementation does.
            let combined = x_scale * w_scale / y_scale;
            data[flat] = range.requantize(acc, combined, y_zp);
        }
        Ok(Tensor::new(data, out.shape))
    }
}

impl Operator for QLinearConvOp {
    fn op_type(&self) -> &str {
        "QLinearConv"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![self.run(ctx, None)?])
    }

    /// `x`/`w`/every zero point may arrive as `I8` or `U8`; `B` (bias) is
    /// `I32`; every scale is `F32`. Declaring all four here is what lets
    /// `execute_typed` below run at all on `Session::run_typed` — and,
    /// crucially, is what makes `y_zero_point`'s *declared* dtype visible to
    /// it in the first place (see `SatRange::for_dtype`).
    ///
    /// Cost note: before this, an empty `native_dtypes()` sent every
    /// `QLinearConv` node down `run_typed`'s f32-fallback branch, which
    /// *borrows* initializer weights straight out of `Session::weights`
    /// (`Cow::Borrowed`, zero copy). A non-empty list here — since `F32` is
    /// in it and every model initializer surfaces as `F32` — instead sends
    /// it down the "native" branch, which *clones* each weight-sourced
    /// input's data into an owned `TypedTensor` once per run (see
    /// `src/session/run/typed.rs`'s `InputSource::Weight` arms in both
    /// branches). For a weight tensor the size of a real conv filter this is
    /// a real, repeated allocation this fix trades for the dtype-aware
    /// saturation range below — accepted because it was the brief's
    /// explicit instruction and `Session::run` (the untyped, and by far the
    /// more common, entry point) is entirely unaffected.
    fn native_dtypes(&self) -> &'static [DType] {
        &[DType::I8, DType::U8, DType::I32, DType::F32]
    }

    /// Same computation as [`Self::execute`] (the kernel operates on f32-lane
    /// [`Tensor`]s regardless — this crate's universal quantized-value
    /// convention, see the module doc comment), except `y_zero_point`'s
    /// *declared* dtype is read from the `TypedTensor` **before** the
    /// f32 projection erases it, and threaded through as an explicit
    /// `SatRange` instead of leaving `Self::run` to fall back to
    /// `SatRange::infer`'s value-based (and, in the ambiguous band,
    /// union-clamped) cascade.
    fn execute_typed(&self, ctx: &TypedOpContext<'_>) -> Result<Vec<TypedTensor>, OnnxError> {
        // Input 7 is `y_zero_point` (see `Self::run`).
        let range_override = ctx.input(7).and_then(|t| SatRange::for_dtype(t.dtype()));

        let owned = project_to_f32(ctx);
        let refs: Vec<Option<&Tensor>> = owned.iter().map(|o| o.as_ref()).collect();
        let f32_ctx = OpContext {
            node: ctx.node,
            inputs: refs,
            outer_scope: None,
            weights: None,
            registry: ctx.registry,
        };
        let out = self.run(&f32_ctx, range_override)?;
        Ok(vec![TypedTensor::new(
            TensorStorage::F32(out.data),
            out.shape,
        )])
    }
}

// ── ConvInteger ─────────────────────────────────────────────────────────────

/// ONNX `ConvInteger` (opset 10+): `y = (x - x_zp) (*) (w - w_zp)` in int32.
///
/// Unlike [`QLinearConvOp`] there are no scales and **no saturation** — the
/// output is a plain int32 accumulator.
pub struct ConvIntegerOp;

impl Operator for ConvIntegerOp {
    fn op_type(&self) -> &str {
        "ConvInteger"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let op = "ConvInteger";
        let x = ctx.input(0)?;
        let w = ctx.input(1)?;
        let x_zp_lanes = zero_point_lanes(ctx.optional_input(2), "ConvInteger: x_zero_point")?;
        let w_zp_lanes = zero_point_lanes(ctx.optional_input(3), "ConvInteger: w_zero_point")?;
        let x_zp = *x_zp_lanes.first().unwrap_or(&0);

        let geo = IntConvGeometry::from_attrs(ctx.attrs(), &x.shape, &w.shape, op)?;
        let out = integer_conv(
            x,
            w,
            &ConvZeroPoints {
                x: x_zp,
                w: w_zp_lanes,
            },
            None,
            &geo,
            op,
        )?;
        let data: Vec<f32> = out.acc.iter().copied().map(saturate_i32).collect();
        Ok(vec![Tensor::new(data, out.shape)])
    }
}
