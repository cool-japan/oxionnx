//! Shared macro definitions used across math op submodules.

/// Unary op (plain — no Result, no in-place support).
macro_rules! unary_op_plain {
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

/// Unary op with in-place support via a per-element closure.
macro_rules! unary_op_inplace {
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

/// Binary op returning `Result<Tensor, OnnxError>` (no in-place support).
macro_rules! binary_op_result {
    ($name:ident, $op_type:expr, $func:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                Ok(vec![$func(ctx.input(0)?, ctx.input(1)?)?])
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
                let result = $func(ctx.input(0)?, ctx.input(1)?)?;
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

/// Binary op with in-place support (only when shapes match exactly).
macro_rules! binary_op_inplace {
    ($name:ident, $op_type:expr, $func:path, $inplace_fn:expr, $typed_fn:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                Ok(vec![$func(ctx.input(0)?, ctx.input(1)?)?])
            }
            fn supports_inplace(&self) -> bool {
                true
            }
            fn execute_inplace(
                &self,
                mut input: Tensor,
                ctx: &OpContext<'_>,
            ) -> Result<Vec<Tensor>, OnnxError> {
                let other = ctx.input(1)?;
                if input.shape != other.shape {
                    // Shapes differ (broadcasting needed) — fall back to regular path.
                    return Ok(vec![$func(&input, other)?]);
                }
                let f: fn(f32, f32) -> f32 = $inplace_fn;
                for (a, b) in input.data.iter_mut().zip(other.data.iter()) {
                    *a = f(*a, *b);
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
                let a = ctx.input(0).ok_or_else(|| {
                    OnnxError::TensorNotFound(format!("{}: missing input[0]", self.op_type()))
                })?;
                let b = ctx.input(1).ok_or_else(|| {
                    OnnxError::TensorNotFound(format!("{}: missing input[1]", self.op_type()))
                })?;
                Ok(vec![$typed_fn(a, b)?])
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
                let a = ctx.input(0)?;
                let b = ctx.input(1)?;
                let out = &mut slots[0];
                if out.shape == a.shape && a.shape == b.shape && out.data.len() == a.data.len() {
                    let f: fn(f32, f32) -> f32 = $inplace_fn;
                    for ((dst, &sa), &sb) in
                        out.data.iter_mut().zip(a.data.iter()).zip(b.data.iter())
                    {
                        *dst = f(sa, sb);
                    }
                } else {
                    let result = $func(a, b)?;
                    if out.shape == result.shape && out.data.len() == result.data.len() {
                        out.data.copy_from_slice(&result.data);
                    } else {
                        *out = result;
                    }
                }
                Ok(())
            }
        }
    };
}
