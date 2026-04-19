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

// ── GeluOp — manual impl to wire typed_gelu ─────────────────────────────────

pub struct GeluOp;
impl Operator for GeluOp {
    fn op_type(&self) -> &str {
        "Gelu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![nn::gelu(ctx.input(0)?)])
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
            *x = *x * 0.5 * (1.0 + (*x * 0.797_884_6 * (1.0 + 0.044715 * *x * *x)).tanh());
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
            .ok_or_else(|| OnnxError::TensorNotFound("Gelu: missing input[0]".into()))?;
        Ok(vec![crate::typed_ops::typed_gelu(x)])
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
        let gelu_fn =
            |x: f32| x * 0.5 * (1.0 + (x * 0.797_884_6 * (1.0 + 0.044715 * x * x)).tanh());
        if slot.shape == input.shape && slot.data.len() == input.data.len() {
            for (dst, &src) in slot.data.iter_mut().zip(input.data.iter()) {
                *dst = gelu_fn(src);
            }
        } else {
            let data: Vec<f32> = input.data.iter().map(|&v| gelu_fn(v)).collect();
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
unary_nn_op!(DropoutOp, "Dropout", nn::dropout);

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
