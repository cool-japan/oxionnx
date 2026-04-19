//! Reduction operator implementations (ReduceMean, ReduceSum, ArgMax, CumSum,
//! Range, TopK, Mod, BitShift, and variadic min/max/mean/sum).

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::math;

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
            fn supports_output_slots(&self) -> bool {
                true
            }
        }
    };
}

reduce_op!(ReduceMeanOp, "ReduceMean", math::reduce_mean);
reduce_op!(ReduceSumOp, "ReduceSum", math::reduce_sum);
reduce_op!(ReduceMaxOp, "ReduceMax", math::reduce_max);
reduce_op!(ReduceMinOp, "ReduceMin", math::reduce_min);
reduce_op!(ReduceProdOp, "ReduceProd", math::reduce_prod);
reduce_op!(ReduceL1Op, "ReduceL1", math::reduce_l1);
reduce_op!(ReduceL2Op, "ReduceL2", math::reduce_l2);
reduce_op!(ReduceLogSumOp, "ReduceLogSum", math::reduce_log_sum);
reduce_op!(
    ReduceLogSumExpOp,
    "ReduceLogSumExp",
    math::reduce_log_sum_exp
);
reduce_op!(
    ReduceSumSquareOp,
    "ReduceSumSquare",
    math::reduce_sum_square
);

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
    fn supports_output_slots(&self) -> bool {
        true
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
    fn supports_output_slots(&self) -> bool {
        true
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
    fn supports_output_slots(&self) -> bool {
        true
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
    fn supports_output_slots(&self) -> bool {
        true
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
    fn supports_output_slots(&self) -> bool {
        true
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
    fn supports_output_slots(&self) -> bool {
        true
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
    fn supports_output_slots(&self) -> bool {
        true
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
            fn supports_output_slots(&self) -> bool {
                true
            }
        }
    };
}

variadic_op!(VariadicMinOp, "Min", math::variadic_min);
variadic_op!(VariadicMaxOp, "Max", math::variadic_max);
variadic_op!(VariadicMeanOp, "Mean", math::variadic_mean);
variadic_op!(VariadicSumOp, "Sum", math::variadic_sum);
