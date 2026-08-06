//! Typed activation operator implementations: Relu, Sigmoid, Tanh, Gelu, SiLU,
//! HardSwish, Softplus, Softsign, Mish, Dropout, Erf, Abs, Log, Exp.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::nn;

// ── Simple unary activations (return Tensor directly) ───────────────────────

macro_rules! unary_nn_op {
    ($name:ident, $op_type:expr, $func:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                Ok(vec![$func(ctx.input(0)?)])
            }
            fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
                &[
                    oxionnx_core::DType::F32,
                    oxionnx_core::DType::F16,
                    oxionnx_core::DType::BF16,
                    oxionnx_core::DType::I32,
                    oxionnx_core::DType::I64,
                ]
            }
            fn execute_typed(
                &self,
                ctx: &oxionnx_core::TypedOpContext<'_>,
            ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
                oxionnx_core::default_typed_via_f32(self, ctx)
            }
            fn supports_output_slots(&self) -> bool {
                true
            }
            fn execute_into_slots(
                &self,
                ctx: &oxionnx_core::OpContext<'_>,
                slots: &mut [oxionnx_core::Tensor],
            ) -> Result<(), oxionnx_core::OnnxError> {
                use oxionnx_core::OnnxError;
                if slots.len() != 1 {
                    return Err(OnnxError::Internal(format!(
                        "{} expects 1 output slot, got {}",
                        self.op_type(),
                        slots.len()
                    )));
                }
                let results = self.execute(ctx)?;
                let result = results
                    .into_iter()
                    .next()
                    .ok_or_else(|| OnnxError::Internal("no output".into()))?;
                let out = &mut slots[0];
                if out.shape == result.shape && out.data.len() == result.data.len() {
                    out.data.copy_from_slice(&result.data);
                } else {
                    *out = result;
                }
                Ok(())
            }
        }
    };
}

/// Unary NN op with in-place support via a per-element closure.
macro_rules! unary_nn_op_inplace {
    ($name:ident, $op_type:expr, $func:path, $inplace_fn:expr) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                Ok(vec![$func(ctx.input(0)?)])
            }
            fn supports_inplace(&self) -> bool {
                true
            }
            fn execute_inplace(
                &self,
                mut input: Tensor,
                _ctx: &OpContext<'_>,
            ) -> Result<Vec<Tensor>, OnnxError> {
                let f: fn(f32) -> f32 = $inplace_fn;
                for x in input.data.iter_mut() {
                    *x = f(*x);
                }
                Ok(vec![input])
            }
            fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
                &[
                    oxionnx_core::DType::F32,
                    oxionnx_core::DType::F16,
                    oxionnx_core::DType::BF16,
                    oxionnx_core::DType::I32,
                    oxionnx_core::DType::I64,
                ]
            }
            fn execute_typed(
                &self,
                ctx: &oxionnx_core::TypedOpContext<'_>,
            ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
                oxionnx_core::default_typed_via_f32(self, ctx)
            }
            fn supports_output_slots(&self) -> bool {
                true
            }
            fn execute_into_slots(
                &self,
                ctx: &oxionnx_core::OpContext<'_>,
                slots: &mut [oxionnx_core::Tensor],
            ) -> Result<(), oxionnx_core::OnnxError> {
                use oxionnx_core::OnnxError;
                if slots.len() != 1 {
                    return Err(OnnxError::Internal(format!(
                        "{} expects 1 output slot, got {}",
                        self.op_type(),
                        slots.len()
                    )));
                }
                let input = ctx.input(0)?;
                let out = &mut slots[0];
                let f: fn(f32) -> f32 = $inplace_fn;
                if out.shape == input.shape && out.data.len() == input.data.len() {
                    for (dst, &src) in out.data.iter_mut().zip(input.data.iter()) {
                        *dst = f(src);
                    }
                } else {
                    let data: Vec<f32> = input.data.iter().map(|&v| f(v)).collect();
                    *out = oxionnx_core::Tensor::new(data, input.shape.clone());
                }
                Ok(())
            }
        }
    };
}

// ── ReluOp — manual impl to wire typed_relu ─────────────────────────────────

pub struct ReluOp;
impl Operator for ReluOp {
    fn op_type(&self) -> &str {
        "Relu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![nn::relu(ctx.input(0)?)])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = x.max(0.0);
        }
        Ok(vec![input])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::OnnxError;
        let x = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Relu: missing input[0]".into()))?;
        Ok(vec![crate::typed_ops::typed_relu(x)])
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
        let input = ctx.input(0)?;
        let slot = &mut slots[0];
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            slot.data.copy_from_slice(&input.data);
            for v in slot.data.iter_mut() {
                *v = v.max(0.0);
            }
        } else {
            let mut data = input.data.clone();
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
            *slot = Tensor::new(data, input.shape.clone());
        }
        Ok(())
    }
}

// ── SigmoidOp — manual impl to wire typed_sigmoid ───────────────────────────

pub struct SigmoidOp;
impl Operator for SigmoidOp {
    fn op_type(&self) -> &str {
        "Sigmoid"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![nn::sigmoid(ctx.input(0)?)])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
        Ok(vec![input])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::OnnxError;
        let x = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Sigmoid: missing input[0]".into()))?;
        Ok(vec![crate::typed_ops::typed_sigmoid(x)])
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
        let input = ctx.input(0)?;
        let slot = &mut slots[0];
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            for (dst, &src) in slot.data.iter_mut().zip(input.data.iter()) {
                *dst = 1.0 / (1.0 + (-src).exp());
            }
        } else {
            let data: Vec<f32> = input
                .data
                .iter()
                .map(|&v| 1.0 / (1.0 + (-v).exp()))
                .collect();
            *slot = Tensor::new(data, input.shape.clone());
        }
        Ok(())
    }
}

// ── TanhOp — manual impl to wire typed_tanh ─────────────────────────────────

pub struct TanhOp;
impl Operator for TanhOp {
    fn op_type(&self) -> &str {
        "Tanh"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![nn::tanh_op(ctx.input(0)?)])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = x.tanh();
        }
        Ok(vec![input])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::OnnxError;
        let x = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Tanh: missing input[0]".into()))?;
        Ok(vec![crate::typed_ops::typed_tanh(x)])
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
        let input = ctx.input(0)?;
        let slot = &mut slots[0];
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            for (dst, &src) in slot.data.iter_mut().zip(input.data.iter()) {
                *dst = src.tanh();
            }
        } else {
            let data: Vec<f32> = input.data.iter().map(|&v| v.tanh()).collect();
            *slot = Tensor::new(data, input.shape.clone());
        }
        Ok(())
    }
}

// ── GeluOp — manual impl to wire typed_gelu; honors the `approximate` attr ──
//
// [a3-16 / a5-3] ONNX Gelu-20's `approximate` attribute defaults to `"none"`,
// which must compute the *exact* erf-based formula; only `approximate="tanh"`
// should use the tanh approximation. All four execution paths below now read
// the attribute instead of hardcoding tanh unconditionally.

/// True iff the node explicitly requests the tanh approximation. Any other
/// value (including absent, or the spec's own `"none"` default) means exact.
fn gelu_approximate_is_tanh(attrs: &oxionnx_core::Attributes) -> bool {
    attrs.s("approximate") == "tanh"
}

/// Exact GELU (`approximate="none"`, the ONNX Gelu-20 default):
/// `x * 0.5 * (1 + erf(x / sqrt(2)))`. Reuses `erf_approx` below, whose
/// Abramowitz & Stegun approximation is accurate to ~1.5e-7 -- comfortably
/// inside the 1e-5 tolerance ONNX's own `test_gelu_default_*` node tests use.
fn gelu_exact(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub struct GeluOp;
impl Operator for GeluOp {
    fn op_type(&self) -> &str {
        "Gelu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        if gelu_approximate_is_tanh(ctx.attrs()) {
            // `nn::gelu` dispatches to the SIMD tanh-approx kernel when the
            // "simd" feature is enabled; preserve that fast path for "tanh".
            Ok(vec![nn::gelu(x)])
        } else {
            let data: Vec<f32> = x.data.iter().map(|&v| gelu_exact(v)).collect();
            Ok(vec![Tensor::new(data, x.shape.clone())])
        }
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        if gelu_approximate_is_tanh(ctx.attrs()) {
            nn::gelu_slice(&mut input.data);
        } else {
            for x in input.data.iter_mut() {
                *x = gelu_exact(*x);
            }
        }
        Ok(vec![input])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::{OnnxError, TensorStorage, TypedTensor};
        let x = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Gelu: missing input[0]".into()))?;
        if gelu_approximate_is_tanh(ctx.attrs()) {
            Ok(vec![crate::typed_ops::typed_gelu(x)])
        } else {
            // Mirror `typed_gelu`'s own dtype-preserving pattern: compute in
            // f32, then cast back to the input's original dtype so e.g. an
            // F16 Gelu input still yields an F16 output rather than a
            // silently-widened F32 one.
            let f32_data = x.storage.to_f32_vec();
            let result: Vec<f32> = f32_data.iter().map(|&v| gelu_exact(v)).collect();
            let result_tensor = TypedTensor::new(TensorStorage::F32(result), x.shape.clone());
            let out = if x.dtype() == oxionnx_core::DType::F32 {
                result_tensor
            } else {
                result_tensor.cast(x.dtype())
            };
            Ok(vec![out])
        }
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
        let input = ctx.input(0)?;
        let use_tanh = gelu_approximate_is_tanh(ctx.attrs());
        let slot = &mut slots[0];
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            slot.data.copy_from_slice(&input.data);
            if use_tanh {
                nn::gelu_slice(&mut slot.data);
            } else {
                for dst in slot.data.iter_mut() {
                    *dst = gelu_exact(*dst);
                }
            }
        } else {
            let mut data = input.data.clone();
            if use_tanh {
                nn::gelu_slice(&mut data);
            } else {
                for dst in data.iter_mut() {
                    *dst = gelu_exact(*dst);
                }
            }
            *slot = Tensor::new(data, input.shape.clone());
        }
        Ok(())
    }
}

unary_nn_op_inplace!(SiLUOp, "SiLU", nn::silu, |x: f32| {
    x / (1.0 + (-x).exp())
});
unary_nn_op!(HardSwishOp, "HardSwish", nn::hard_swish);
unary_nn_op!(SoftplusOp, "Softplus", nn::softplus);
unary_nn_op!(SoftsignOp, "Softsign", nn::softsign);
unary_nn_op!(MishOp, "Mish", nn::mish);

// ── Dropout — dedicated impl: optional mask output, training_mode guard ─────
//
// [a1-16] `nn::dropout` (= identity) is correct for the always-inference case,
// but a node that declares a second ("mask") output needs that output
// resolved too, or any downstream consumer of it fails with TensorNotFound.
// Per the opset-12+ spec, in inference mode ("training_mode" input false/
// absent) the mask is all-true, so it's an all-ones tensor here (Tensor is
// f32-backed; 1.0/0.0 is this crate's boolean convention, see e.g.
// `comparison::equal`). `training_mode=true` requests genuine stochastic
// dropout, which has no deterministic answer for a static inference engine
// with no RNG-seeding mechanism in the spec, so it is reported as a typed
// error rather than silently producing wrong (or fabricated-random) output.

pub struct DropoutOp;
impl DropoutOp {
    /// `training_mode` is input 2 (optional bool scalar, default false).
    fn check_training_mode(ctx: &OpContext<'_>) -> Result<(), OnnxError> {
        let training_mode = ctx
            .optional_input(2)
            .and_then(|t| t.data.first().copied())
            .unwrap_or(0.0)
            != 0.0;
        if training_mode {
            return Err(OnnxError::Unsupported(
                "Dropout: training_mode=true (stochastic dropout) is not supported by this \
                 inference-only engine; export the model in eval/inference mode"
                    .into(),
            ));
        }
        Ok(())
    }
}
impl Operator for DropoutOp {
    fn op_type(&self) -> &str {
        "Dropout"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Self::check_training_mode(ctx)?;
        let x = ctx.input(0)?;
        let mut outputs = vec![nn::dropout(x)];
        if ctx.node.outputs.len() > 1 {
            outputs.push(Tensor::new(vec![1.0; x.numel()], x.shape.clone()));
        }
        Ok(outputs)
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        oxionnx_core::default_typed_via_f32(self, ctx)
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
        Self::check_training_mode(ctx)?;
        let x = ctx.input(0)?;
        let n = x.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].data.copy_from_slice(&x.data);
        slots[0].shape.clone_from(&x.shape);
        if let Some(mask_slot) = slots.get_mut(1) {
            if mask_slot.data.len() != n {
                mask_slot.data.resize(n, 0.0_f32);
            }
            mask_slot.data.fill(1.0);
            mask_slot.shape.clone_from(&x.shape);
        }
        Ok(())
    }
}

// ── Erf (custom inline implementation) ─────────────────────────────────────

/// Polynomial approximation of error function (Abramowitz & Stegun 7.1.26)
fn erf_approx(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0f32 } else { -1.0f32 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

pub struct ErfOp;
impl Operator for ErfOp {
    fn op_type(&self) -> &str {
        "Erf"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let data: Vec<f32> = x.data.iter().map(|&v| erf_approx(v)).collect();
        Ok(vec![Tensor::new(data, x.shape.clone())])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = erf_approx(*x);
        }
        Ok(vec![input])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        oxionnx_core::default_typed_via_f32(self, ctx)
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
        let input = ctx.input(0)?;
        let slot = &mut slots[0];
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            for (dst, &src) in slot.data.iter_mut().zip(input.data.iter()) {
                *dst = erf_approx(src);
            }
        } else {
            let data: Vec<f32> = input.data.iter().map(|&v| erf_approx(v)).collect();
            *slot = Tensor::new(data, input.shape.clone());
        }
        Ok(())
    }
}

// ── Abs, Log, Exp (inline unary ops) ───────────────────────────────────────

pub struct AbsOp;
impl Operator for AbsOp {
    fn op_type(&self) -> &str {
        "Abs"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        Ok(vec![Tensor::new(
            x.data.iter().map(|v| v.abs()).collect(),
            x.shape.clone(),
        )])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = x.abs();
        }
        Ok(vec![input])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
        ]
    }
    /// Unlike the other activations in this file, `Abs` has a well-defined, exactly
    /// representable result for an integer input (magnitude never needs a fractional bit), so
    /// this computes it with real `i32`/`i64` arithmetic instead of delegating to
    /// `default_typed_via_f32` — an `i64` input above `2^24` would otherwise be silently rounded
    /// through f32 and returned re-tagged as `F32`, the same `native_dtypes()`-overclaims-exactness
    /// bug `unary_op_inplace_exact_int!` in `registry/math_ops/macros.rs` fixed for `Neg`/`Ceil`/
    /// `Floor`/`Round`/`Sign`. `wrapping_abs`, not bare `.abs()`, for the same reason `NegOp` uses
    /// `wrapping_neg`: `i32::MIN`/`i64::MIN` have no positive representation in their own type, so
    /// plain `.abs()` panics on overflow in a debug build; `wrapping_abs` returns the (still
    /// negative) `MIN` value back unchanged, matching two's-complement convention.
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::{OnnxError, TensorStorage, TypedTensor};
        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Abs: missing input[0]".into()))?;
        match &input.storage {
            TensorStorage::I32(data) => {
                let out: Vec<i32> = data.iter().map(|&x| x.wrapping_abs()).collect();
                Ok(vec![TypedTensor::new(
                    TensorStorage::I32(out),
                    input.shape.clone(),
                )])
            }
            TensorStorage::I64(data) => {
                let out: Vec<i64> = data.iter().map(|&x| x.wrapping_abs()).collect();
                Ok(vec![TypedTensor::new(
                    TensorStorage::I64(out),
                    input.shape.clone(),
                )])
            }
            // F32/F16/BF16: real-valued path, f32 round-trip is exact for these (a
            // whole-number FP value round-trips through f32 unchanged, and F16/BF16
            // already have <=24-bit mantissas so promoting to f32 cannot lose anything
            // they had).
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
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
        let input = ctx.input(0)?;
        let slot = &mut slots[0];
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            for (dst, &src) in slot.data.iter_mut().zip(input.data.iter()) {
                *dst = src.abs();
            }
        } else {
            let data: Vec<f32> = input.data.iter().map(|v| v.abs()).collect();
            *slot = Tensor::new(data, input.shape.clone());
        }
        Ok(())
    }
}

pub struct LogOp;
impl Operator for LogOp {
    fn op_type(&self) -> &str {
        "Log"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        Ok(vec![Tensor::new(
            x.data.iter().map(|v| v.ln()).collect(),
            x.shape.clone(),
        )])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = x.ln();
        }
        Ok(vec![input])
    }
    // `Log` is inherently real-valued: `ln` of an arbitrary integer is generally
    // irrational, so there is no exact `I32`/`I64` result to compute even in
    // principle -- unlike `Abs` above. `native_dtypes()`'s contract is "dtypes this
    // operator can execute *without* an f32 round-trip" (see the identical
    // "exact-integer / real-valued boundary" note in `registry/math_ops/macros.rs`,
    // which documents the same distinction for `Sqrt`/the trig family); claiming
    // `I32`/`I64` here while `execute_typed` below silently f32-round-trips them
    // would be exactly that overclaim. Unlike `Sqrt` (which keeps the claim, for
    // compatibility with `tests/typed_io_test.rs::test_native_dtypes_math_pilot_ops`,
    // which pins it), no existing test pins `I32`/`I64` for `Log`, so this omits them
    // rather than documenting a promise nothing here delivers on.
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
        oxionnx_core::default_typed_via_f32(self, ctx)
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
        let input = ctx.input(0)?;
        let slot = &mut slots[0];
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            for (dst, &src) in slot.data.iter_mut().zip(input.data.iter()) {
                *dst = src.ln();
            }
        } else {
            let data: Vec<f32> = input.data.iter().map(|v| v.ln()).collect();
            *slot = Tensor::new(data, input.shape.clone());
        }
        Ok(())
    }
}

pub struct ExpOp;
impl Operator for ExpOp {
    fn op_type(&self) -> &str {
        "Exp"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        Ok(vec![Tensor::new(
            x.data.iter().map(|v| v.exp()).collect(),
            x.shape.clone(),
        )])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = x.exp();
        }
        Ok(vec![input])
    }
    // `Exp` is inherently real-valued, the same reasoning as `Log` above (see its
    // `native_dtypes()` doc comment for the full "exact-integer / real-valued
    // boundary" rationale): no exact `I32`/`I64` result exists even in principle, no
    // existing test pins `I32`/`I64` for `Exp`, so they are omitted here rather than
    // advertised and silently f32-round-tripped by `execute_typed` below.
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
        use oxionnx_core::OnnxError;
        let x = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Exp: missing input[0]".into()))?;
        Ok(vec![crate::typed_ops::typed_exp(x)])
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
        let input = ctx.input(0)?;
        let slot = &mut slots[0];
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            for (dst, &src) in slot.data.iter_mut().zip(input.data.iter()) {
                *dst = src.exp();
            }
        } else {
            let data: Vec<f32> = input.data.iter().map(|v| v.exp()).collect();
            *slot = Tensor::new(data, input.shape.clone());
        }
        Ok(())
    }
}
