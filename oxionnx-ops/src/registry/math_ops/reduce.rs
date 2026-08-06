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

/// [a0-15/a11-7] Opset-18 `noop_with_empty_axes`: when set and the resolved
/// axes list is empty (whether because the axes input was omitted entirely
/// or provided as an explicitly empty tensor -- both collapse to the same
/// empty `Vec` from `axes_from_ctx`, and the spec treats them identically),
/// the op is Identity rather than "reduce every dimension".
fn noop_with_empty_axes(ctx: &OpContext<'_>) -> bool {
    ctx.attrs().i("noop_with_empty_axes", 0) != 0
}

macro_rules! reduce_op {
    ($name:ident, $op_type:expr, $func:path, $func_into:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                let axes = axes_from_ctx(ctx);
                let keepdims = ctx.attrs().i("keepdims", 1) != 0;
                let x = ctx.input(0)?;
                if axes.is_empty() && noop_with_empty_axes(ctx) {
                    return Ok(vec![x.clone()]);
                }
                Ok(vec![$func(x, &axes, keepdims)?])
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
                let axes = axes_from_ctx(ctx);
                let keepdims = ctx.attrs().i("keepdims", 1) != 0;
                let x = ctx.input(0)?;
                if axes.is_empty() && noop_with_empty_axes(ctx) {
                    let n = x.numel();
                    if slots[0].data.len() != n {
                        slots[0].data.resize(n, 0.0_f32);
                    }
                    slots[0].data.copy_from_slice(&x.data);
                    slots[0].shape.clone_from(&x.shape);
                    return Ok(());
                }
                // Must run (and propagate its error) before touching `slots[0]`: `reduce_output_shape`
                // validates `axes` before it is used to pre-size the output buffer, so an out-of-range
                // axis is a typed error here instead of sizing the buffer from a silently-wrong shape.
                let (_, out_len) = math::reduce_output_shape(x, &axes, keepdims)?;
                if slots[0].data.len() != out_len {
                    slots[0].data.resize(out_len, 0.0_f32);
                }
                slots[0].shape = $func_into(x, &axes, keepdims, &mut slots[0].data)?;
                Ok(())
            }
        }
    };
}

reduce_op!(
    ReduceMeanOp,
    "ReduceMean",
    math::reduce_mean,
    math::reduce_mean_into
);
reduce_op!(
    ReduceSumOp,
    "ReduceSum",
    math::reduce_sum,
    math::reduce_sum_into
);
reduce_op!(
    ReduceMaxOp,
    "ReduceMax",
    math::reduce_max,
    math::reduce_max_into
);
reduce_op!(
    ReduceMinOp,
    "ReduceMin",
    math::reduce_min,
    math::reduce_min_into
);
reduce_op!(
    ReduceProdOp,
    "ReduceProd",
    math::reduce_prod,
    math::reduce_prod_into
);
reduce_op!(
    ReduceL1Op,
    "ReduceL1",
    math::reduce_l1,
    math::reduce_l1_into
);
reduce_op!(
    ReduceL2Op,
    "ReduceL2",
    math::reduce_l2,
    math::reduce_l2_into
);
reduce_op!(
    ReduceLogSumOp,
    "ReduceLogSum",
    math::reduce_log_sum,
    math::reduce_log_sum_into
);
reduce_op!(
    ReduceLogSumExpOp,
    "ReduceLogSumExp",
    math::reduce_log_sum_exp,
    math::reduce_log_sum_exp_into
);
reduce_op!(
    ReduceSumSquareOp,
    "ReduceSumSquare",
    math::reduce_sum_square,
    math::reduce_sum_square_into
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
        let select_last_index = ctx.attrs().i("select_last_index", 0) != 0;
        Ok(vec![math::arg_max(
            ctx.input(0)?,
            axis,
            keepdims,
            select_last_index,
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
        let axis = ctx.attrs().i("axis", 0);
        let keepdims = ctx.attrs().i("keepdims", 0) != 0;
        let select_last_index = ctx.attrs().i("select_last_index", 0) != 0;
        // Must run (and propagate its error) before touching `slots[0]`:
        // `arg_output_shape` validates `axis` before it indexes/removes it
        // from the shape, so an out-of-range axis is now a typed error here
        // instead of a panic inside that shape computation.
        let (_, out_len) = math::arg_output_shape(x, axis, keepdims)?;
        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        slots[0].shape = math::arg_reduce_into(
            x,
            axis,
            keepdims,
            true,
            select_last_index,
            &mut slots[0].data,
        )?;
        Ok(())
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
        let select_last_index = ctx.attrs().i("select_last_index", 0) != 0;
        Ok(vec![math::arg_min(
            ctx.input(0)?,
            axis,
            keepdims,
            select_last_index,
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
        let axis = ctx.attrs().i("axis", 0);
        let keepdims = ctx.attrs().i("keepdims", 0) != 0;
        let select_last_index = ctx.attrs().i("select_last_index", 0) != 0;
        // See `ArgMaxOp::execute_into_slots` above: `arg_output_shape` must
        // run (and its error propagate) before `slots[0]` is touched.
        let (_, out_len) = math::arg_output_shape(x, axis, keepdims)?;
        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        slots[0].shape = math::arg_reduce_into(
            x,
            axis,
            keepdims,
            false,
            select_last_index,
            &mut slots[0].data,
        )?;
        Ok(())
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
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Ok(());
        }
        let x = ctx.input(0)?;
        let axis = ctx.input(1)?.data[0] as i64;
        let exclusive = ctx.attrs().i("exclusive", 0) != 0;
        let reverse = ctx.attrs().i("reverse", 0) != 0;
        let n = x.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape = math::cumsum_into(x, axis, exclusive, reverse, &mut slots[0].data)?;
        Ok(())
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
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.len() < 2 {
            return Err(OnnxError::Internal(
                "TopKOp: expected 2 output slots".into(),
            ));
        }
        let x = ctx.input(0)?;
        let k = ctx.input(1)?.data[0] as usize;
        let attrs = ctx.attrs();
        let axis = attrs.i("axis", -1);
        let largest = attrs.i("largest", 1) != 0;
        let sorted = attrs.i("sorted", 1) != 0;
        let (_, out_len) = math::top_k_output_shape(x, k, axis);
        // Resize both slots before borrowing mutably as separate slices.
        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        if slots[1].data.len() != out_len {
            slots[1].data.resize(out_len, 0.0_f32);
        }
        // Split into two non-overlapping mutable references.
        let (slot0, rest) = slots.split_at_mut(1);
        let final_shape = math::top_k_into(
            x,
            k,
            axis,
            largest,
            sorted,
            &mut slot0[0].data,
            &mut rest[0].data,
        )?;
        slots[0].shape.clone_from(&final_shape);
        slots[1].shape = final_shape;
        Ok(())
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
            fn execute_into_slots(
                &self,
                ctx: &OpContext<'_>,
                slots: &mut [Tensor],
            ) -> Result<(), OnnxError> {
                if slots.is_empty() {
                    return Ok(());
                }
                let tensors: Vec<&Tensor> = ctx.inputs.iter().filter_map(|opt| *opt).collect();
                let result = $func(&tensors)?;
                let out = &mut slots[0];
                if out.data.len() == result.data.len() && out.shape == result.shape {
                    out.data.copy_from_slice(&result.data);
                } else {
                    *out = result;
                }
                Ok(())
            }
        }
    };
}

variadic_op!(VariadicMinOp, "Min", math::variadic_min);
variadic_op!(VariadicMaxOp, "Max", math::variadic_max);
variadic_op!(VariadicMeanOp, "Mean", math::variadic_mean);
variadic_op!(VariadicSumOp, "Sum", math::variadic_sum);
