//! Operator trait implementations for math operations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::math;

// ── Unary math ops (no Result) ──────────────────────────────────────────────

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
        }
    };
}

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
        }
    };
}

/// Binary op with in-place support (only when shapes match exactly).
macro_rules! binary_op_inplace {
    ($name:ident, $op_type:expr, $func:path, $inplace_fn:expr) => {
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
                    // Reconstruct a full context with the owned tensor for execute().
                    return Ok(vec![$func(&input, other)?]);
                }
                let f: fn(f32, f32) -> f32 = $inplace_fn;
                for (a, b) in input.data.iter_mut().zip(other.data.iter()) {
                    *a = f(*a, *b);
                }
                Ok(vec![input])
            }
        }
    };
}

// Basic binary ops with in-place support (all return Result<Tensor, String>)
binary_op_inplace!(AddOp, "Add", math::add, |a, b| a + b);
binary_op_inplace!(SubOp, "Sub", math::sub, |a, b| a - b);
binary_op_inplace!(MulOp, "Mul", math::mul, |a, b| a * b);
binary_op_inplace!(DivOp, "Div", math::div, |a, b| a / b);
binary_op_result!(PowOp, "Pow", math::pow);

// Basic unary ops with in-place support
unary_op_inplace!(SqrtOp, "Sqrt", math::sqrt, f32::sqrt);
unary_op_plain!(ReciprocalOp, "Reciprocal", math::reciprocal);
unary_op_inplace!(NegOp, "Neg", math::neg, |x| -x);
unary_op_inplace!(CeilOp, "Ceil", math::ceil, f32::ceil);
unary_op_inplace!(FloorOp, "Floor", math::floor_op, f32::floor);
unary_op_inplace!(RoundOp, "Round", math::round_op, f32::round);
unary_op_inplace!(SignOp, "Sign", math::sign, |x: f32| {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
});

// Trig unary ops
unary_op_plain!(SinOp, "Sin", math::sin_op);
unary_op_plain!(CosOp, "Cos", math::cos_op);
unary_op_plain!(TanOp, "Tan", math::tan_op);
unary_op_plain!(AsinOp, "Asin", math::asin_op);
unary_op_plain!(AcosOp, "Acos", math::acos_op);
unary_op_plain!(AtanOp, "Atan", math::atan_op);
unary_op_plain!(SinhOp, "Sinh", math::sinh_op);
unary_op_plain!(CoshOp, "Cosh", math::cosh_op);
unary_op_plain!(AsinhOp, "Asinh", math::asinh_op);
unary_op_plain!(AcoshOp, "Acosh", math::acosh_op);
unary_op_plain!(AtanhOp, "Atanh", math::atanh_op);

// ── MatMul ──────────────────────────────────────────────────────────────────

pub struct MatMulOp;
impl Operator for MatMulOp {
    fn op_type(&self) -> &str {
        "MatMul"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let a = ctx.input(0)?;
        let b = ctx.input(1)?;
        Ok(vec![math::matmul(a, b)?])
    }
}

// ── Gemm ────────────────────────────────────────────────────────────────────

pub struct GemmOp;
impl Operator for GemmOp {
    fn op_type(&self) -> &str {
        "Gemm"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let a = ctx.input(0)?;
        let b = ctx.input(1)?;
        let c = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let alpha = attrs.f("alpha", 1.0);
        let beta = attrs.f("beta", 1.0);
        let trans_a = attrs.i("transA", 0) != 0;
        let trans_b = attrs.i("transB", 0) != 0;
        Ok(vec![math::gemm(a, b, c, alpha, beta, trans_a, trans_b)?])
    }
}

// ── Reduce ops ──────────────────────────────────────────────────────────────

/// Helper: get axes from optional tensor input or attribute.
fn axes_from_ctx(ctx: &OpContext<'_>) -> Vec<i64> {
    if let Some(t) = ctx.optional_input(1) {
        t.data.iter().map(|&v| v as i64).collect()
    } else {
        ctx.attrs().ints("axes").to_vec()
    }
}

macro_rules! reduce_op {
    ($name:ident, $op_type:expr, $func:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                let axes = axes_from_ctx(ctx);
                let keepdims = ctx.attrs().i("keepdims", 1) != 0;
                Ok(vec![$func(ctx.input(0)?, &axes, keepdims)?])
            }
        }
    };
}

reduce_op!(ReduceMeanOp, "ReduceMean", math::reduce_mean);
reduce_op!(ReduceSumOp, "ReduceSum", math::reduce_sum);
reduce_op!(ReduceMaxOp, "ReduceMax", math::reduce_max);
reduce_op!(ReduceMinOp, "ReduceMin", math::reduce_min);
reduce_op!(ReduceProdOp, "ReduceProd", math::reduce_prod);

// ── ArgMax / ArgMin ─────────────────────────────────────────────────────────

pub struct ArgMaxOp;
impl Operator for ArgMaxOp {
    fn op_type(&self) -> &str {
        "ArgMax"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axis = ctx.attrs().i("axis", 0);
        let keepdims = ctx.attrs().i("keepdims", 0) != 0;
        Ok(vec![math::arg_max(ctx.input(0)?, axis, keepdims)?])
    }
}

pub struct ArgMinOp;
impl Operator for ArgMinOp {
    fn op_type(&self) -> &str {
        "ArgMin"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axis = ctx.attrs().i("axis", 0);
        let keepdims = ctx.attrs().i("keepdims", 0) != 0;
        Ok(vec![math::arg_min(ctx.input(0)?, axis, keepdims)?])
    }
}

// ── CumSum ──────────────────────────────────────────────────────────────────

pub struct CumSumOp;
impl Operator for CumSumOp {
    fn op_type(&self) -> &str {
        "CumSum"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let axis = ctx.input(1)?.data[0] as i64;
        let exclusive = ctx.attrs().i("exclusive", 0) != 0;
        let reverse = ctx.attrs().i("reverse", 0) != 0;
        Ok(vec![math::cumsum(x, axis, exclusive, reverse)?])
    }
}

// ── Range ───────────────────────────────────────────────────────────────────

pub struct RangeOp;
impl Operator for RangeOp {
    fn op_type(&self) -> &str {
        "Range"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let start = ctx.input(0)?.data[0];
        let limit = ctx.input(1)?.data[0];
        let delta = ctx.input(2)?.data[0];
        Ok(vec![math::range(start, limit, delta)?])
    }
}

// ── TopK ────────────────────────────────────────────────────────────────────

pub struct TopKOp;
impl Operator for TopKOp {
    fn op_type(&self) -> &str {
        "TopK"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let k = ctx.input(1)?.data[0] as usize;
        let attrs = ctx.attrs();
        let axis = attrs.i("axis", -1);
        let largest = attrs.i("largest", 1) != 0;
        let sorted = attrs.i("sorted", 1) != 0;
        let (values, indices) = math::top_k(x, k, axis, largest, sorted)?;
        Ok(vec![values, indices])
    }
}

// ── Mod ─────────────────────────────────────────────────────────────────────

pub struct ModOp;
impl Operator for ModOp {
    fn op_type(&self) -> &str {
        "Mod"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let fmod = ctx.attrs().i("fmod", 0);
        Ok(vec![math::mod_op(ctx.input(0)?, ctx.input(1)?, fmod)?])
    }
}

// ── BitShift ────────────────────────────────────────────────────────────────

pub struct BitShiftOp;
impl Operator for BitShiftOp {
    fn op_type(&self) -> &str {
        "BitShift"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let direction = ctx.attrs().s("direction");
        Ok(vec![math::bit_shift(
            ctx.input(0)?,
            ctx.input(1)?,
            direction,
        )?])
    }
}

// ── Variadic ops ────────────────────────────────────────────────────────────

macro_rules! variadic_op {
    ($name:ident, $op_type:expr, $func:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                let tensors: Vec<&Tensor> = ctx.inputs.iter().filter_map(|opt| *opt).collect();
                Ok(vec![$func(&tensors)?])
            }
        }
    };
}

variadic_op!(VariadicMinOp, "Min", math::variadic_min);
variadic_op!(VariadicMaxOp, "Max", math::variadic_max);
variadic_op!(VariadicMeanOp, "Mean", math::variadic_mean);
variadic_op!(VariadicSumOp, "Sum", math::variadic_sum);
