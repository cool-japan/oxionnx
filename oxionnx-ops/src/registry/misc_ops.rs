//! Operator trait implementations for comparison, logic, construction,
//! einsum, NMS, and other miscellaneous operations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::comparison;

// ── Comparison binary ops ───────────────────────────────────────────────────

macro_rules! comparison_binary_op {
    ($name:ident, $op_type:expr, $func:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                Ok(vec![$func(ctx.input(0)?, ctx.input(1)?)?])
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
                let result = $func(ctx.input(0)?, ctx.input(1)?)?;
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

comparison_binary_op!(EqualOp, "Equal", comparison::equal);
comparison_binary_op!(GreaterOp, "Greater", comparison::greater);
comparison_binary_op!(
    GreaterOrEqualOp,
    "GreaterOrEqual",
    comparison::greater_or_equal
);
comparison_binary_op!(LessOp, "Less", comparison::less);
comparison_binary_op!(LessOrEqualOp, "LessOrEqual", comparison::less_or_equal);
comparison_binary_op!(AndOp, "And", comparison::and_op);
comparison_binary_op!(OrOp, "Or", comparison::or_op);
comparison_binary_op!(XorOp, "Xor", comparison::xor_op);

// ── Not (unary logic) ──────────────────────────────────────────────────────

pub struct NotOp;
impl Operator for NotOp {
    fn op_type(&self) -> &str {
        "Not"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![comparison::not_op(ctx.input(0)?)])
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
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = if x == 0.0 { 1.0 } else { 0.0 };
        }
        Ok(())
    }
}

// ── IsInf ───────────────────────────────────────────────────────────────────

pub struct IsInfOp;
impl Operator for IsInfOp {
    fn op_type(&self) -> &str {
        "IsInf"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let detect_neg = attrs.i("detect_negative", 1) != 0;
        let detect_pos = attrs.i("detect_positive", 1) != 0;
        Ok(vec![comparison::is_inf(
            ctx.input(0)?,
            detect_neg,
            detect_pos,
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
        if slots.is_empty() {
            return Ok(());
        }
        let input = ctx.input(0)?;
        let attrs = ctx.attrs();
        let detect_neg = attrs.i("detect_negative", 1) != 0;
        let detect_pos = attrs.i("detect_positive", 1) != 0;
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = if (detect_pos && x == f32::INFINITY) || (detect_neg && x == f32::NEG_INFINITY) {
                1.0
            } else {
                0.0
            };
        }
        Ok(())
    }
}

// ── IsNaN ───────────────────────────────────────────────────────────────────

pub struct IsNaNOp;
impl Operator for IsNaNOp {
    fn op_type(&self) -> &str {
        "IsNaN"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![comparison::is_nan(ctx.input(0)?)])
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
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = if x.is_nan() { 1.0 } else { 0.0 };
        }
        Ok(())
    }
}

// ── NonZero ─────────────────────────────────────────────────────────────────

pub struct NonZeroOp;
impl Operator for NonZeroOp {
    fn op_type(&self) -> &str {
        "NonZero"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![comparison::non_zero(ctx.input(0)?)])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── ConstantOfShape ─────────────────────────────────────────────────────────

pub struct ConstantOfShapeOp;
impl Operator for ConstantOfShapeOp {
    fn op_type(&self) -> &str {
        "ConstantOfShape"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let shape_t = ctx.input(0)?;
        let target_shape: Vec<usize> = shape_t.data.iter().map(|&v| v as usize).collect();
        let value = ctx
            .attrs()
            .tensors
            .get("value")
            .and_then(|t| t.data.first().copied())
            .unwrap_or(0.0);
        Ok(vec![comparison::constant_of_shape(&target_shape, value)])
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
        let shape_t = ctx.input(0)?;
        let target_shape: Vec<usize> = shape_t.data.iter().map(|&v| v as usize).collect();
        let fill_value = ctx
            .attrs()
            .tensors
            .get("value")
            .and_then(|t| t.data.first().copied())
            .unwrap_or(0.0_f32);
        // No `.max(1)`: an empty `target_shape` (scalar output) already multiplies to 1
        // (empty-product identity), so the clamp was never needed for that case. What it
        // *did* do is corrupt a genuine zero-size dim -- e.g. shape input `[0, 3]` has
        // product 0, and the correct output is a 0-element tensor of shape `[0, 3]`
        // (matching `comparison::constant_of_shape`, which `execute` above calls and which
        // has no such clamp). `.max(1)` forced `n = 1` there, leaving `slots[0]` with
        // `shape == [0, 3]` but `data.len() == 1` -- a tensor that violates its own
        // `data.len() == shape.product()` invariant instead of the correct empty result.
        let n: usize = target_shape.iter().product::<usize>();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&target_shape);
        for v in slots[0].data.iter_mut() {
            *v = fill_value;
        }
        Ok(())
    }
}

// ── EyeLike ─────────────────────────────────────────────────────────────────

pub struct EyeLikeOp;
impl Operator for EyeLikeOp {
    fn op_type(&self) -> &str {
        "EyeLike"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let k = ctx.attrs().i("k", 0);
        Ok(vec![comparison::eye_like(&x.shape, k)?])
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
        let result = self
            .execute(ctx)?
            .into_iter()
            .next()
            .ok_or_else(|| OnnxError::Internal("EyeLikeOp: no output".into()))?;
        let out = &mut slots[0];
        if out.data.len() == result.data.len() && out.shape == result.shape {
            out.data.copy_from_slice(&result.data);
        } else {
            *out = result;
        }
        Ok(())
    }
}

// ── Trilu ───────────────────────────────────────────────────────────────────

pub struct TriluOp;
impl Operator for TriluOp {
    fn op_type(&self) -> &str {
        "Trilu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let upper = ctx.attrs().i("upper", 1) != 0;
        // `.first()`, not `[0]`: a present-but-empty `k` input (malformed model) must
        // fall back to the default rather than index an empty `data` slice.
        let k = ctx
            .optional_input(1)
            .and_then(|t| t.data.first().copied())
            .map(|v| v as i64)
            .unwrap_or(0);
        Ok(vec![comparison::trilu(x, upper, k)?])
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
        let result = self
            .execute(ctx)?
            .into_iter()
            .next()
            .ok_or_else(|| OnnxError::Internal("TriluOp: no output".into()))?;
        let out = &mut slots[0];
        if out.data.len() == result.data.len() && out.shape == result.shape {
            out.data.copy_from_slice(&result.data);
        } else {
            *out = result;
        }
        Ok(())
    }
}

// ── Identity ────────────────────────────────────────────────────────────────

pub struct IdentityOp;
impl Operator for IdentityOp {
    fn op_type(&self) -> &str {
        "Identity"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![ctx.input(0)?.clone()])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![input])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::F64,
            oxionnx_core::DType::I8,
            oxionnx_core::DType::I16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
            oxionnx_core::DType::U8,
            oxionnx_core::DType::U16,
            oxionnx_core::DType::U32,
            oxionnx_core::DType::U64,
            oxionnx_core::DType::Bool,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::OnnxError;
        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Identity: missing input[0]".into()))?;
        Ok(vec![input.clone()])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        use oxionnx_core::OnnxError;
        if slots.is_empty() {
            return Err(OnnxError::Internal("Identity: no output slots".into()));
        }
        let input = ctx.input(0)?;
        slots[0].data.clear();
        slots[0].data.extend_from_slice(&input.data);
        slots[0].shape = input.shape.clone();
        Ok(())
    }
}

// ── Cast ────────────────────────────────────────────────────────────────────

pub struct CastOp;
impl Operator for CastOp {
    fn op_type(&self) -> &str {
        "Cast"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let to = ctx.attrs().i("to", 1);
        // Reject a `to` that is not a real `TensorProto.DataType` value up
        // front, via the same `DType::from_onnx` mapping `execute_typed`
        // below already uses. Without this, an unrecognized code (a typo, or
        // simply a dtype outside the ONNX enum) fell through every explicit
        // arm below into the `_` catch-all and silently became a no-op cast
        // instead of a typed error.
        if oxionnx_core::DType::from_onnx(to as i32).is_none() {
            return Err(OnnxError::Unsupported(format!(
                "Cast: unrecognized target dtype code {to}"
            )));
        }
        // ONNX Cast to an integer type truncates toward zero (numpy `astype`
        // semantics), not round-to-nearest, and out-of-range values saturate to
        // the destination type's range. Rust's `as` float->int cast is exactly
        // that (truncate + saturate, NaN -> 0) since Rust 1.45, so this mirrors
        // `TypedTensor::cast` (registry/misc_ops.rs execute_typed below) element
        // for element -- the two dispatch paths must agree on the same model.
        //
        // Every dtype the `to` check above accepts but that has no explicit
        // arm here (F32, F16, F64, BF16) is already f32-shaped data in this
        // dtype-erased representation, so `_ => x.data.clone()` is correct
        // for them -- the catch-all now only ever sees recognized dtypes.
        let data: Vec<f32> = match to {
            9 => x
                .data
                .iter()
                .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
                .collect(), // BOOL
            2 => x.data.iter().map(|&v| v as u8 as f32).collect(), // UINT8
            3 => x.data.iter().map(|&v| v as i8 as f32).collect(), // INT8
            4 => x.data.iter().map(|&v| v as u16 as f32).collect(), // UINT16
            5 => x.data.iter().map(|&v| v as i16 as f32).collect(), // INT16
            6 => x.data.iter().map(|&v| v as i32 as f32).collect(), // INT32
            7 => x.data.iter().map(|&v| v as i64 as f32).collect(), // INT64
            12 => x.data.iter().map(|&v| v as u32 as f32).collect(), // UINT32
            13 => x.data.iter().map(|&v| v as u64 as f32).collect(), // UINT64
            _ => x.data.clone(),
        };
        Ok(vec![Tensor::new(data, x.shape.clone())])
    }
    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::F64,
            oxionnx_core::DType::I8,
            oxionnx_core::DType::I16,
            oxionnx_core::DType::I32,
            oxionnx_core::DType::I64,
            oxionnx_core::DType::U8,
            oxionnx_core::DType::U16,
            oxionnx_core::DType::U32,
            oxionnx_core::DType::U64,
            oxionnx_core::DType::Bool,
        ]
    }
    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::OnnxError;
        let to_int = ctx.attrs().i("to", 1);
        // Same validation as the `execute` (f32) path above: an unrecognized
        // `to` must be a typed error, not a silent promotion to F32.
        let target_dtype = oxionnx_core::DType::from_onnx(to_int as i32).ok_or_else(|| {
            OnnxError::Unsupported(format!("Cast: unrecognized target dtype code {to_int}"))
        })?;
        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Cast: missing input[0]".into()))?;
        Ok(vec![input.cast(target_dtype)])
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
        let results = self.execute(ctx)?;
        let result = results
            .into_iter()
            .next()
            .ok_or_else(|| OnnxError::Internal("Cast: no output".into()))?;
        let out = &mut slots[0];
        if out.shape == result.shape && out.data.len() == result.data.len() {
            out.data.copy_from_slice(&result.data);
        } else {
            *out = result;
        }
        Ok(())
    }
}

// ── Shape ───────────────────────────────────────────────────────────────────

/// Resolve Shape's opset-15 `start`/`end` attributes against the input rank,
/// using Python-slice-style negative-index and out-of-range clamping:
/// negative values count from the back (clamped at 0), positive values clamp
/// at `rank`, and `end < start` collapses to an empty range rather than
/// panicking on the subsequent `shape[start..end]` slice.
fn shape_slice_bounds(rank: usize, attrs: &oxionnx_core::Attributes) -> (usize, usize) {
    let rank_i = rank as i64;
    let resolve = |v: i64| -> i64 {
        if v < 0 {
            (v + rank_i).max(0)
        } else {
            v.min(rank_i)
        }
    };
    let start = resolve(attrs.i("start", 0)) as usize;
    let end = resolve(attrs.i("end", rank_i)) as usize;
    (start, end.max(start))
}

pub struct ShapeOp;
impl Operator for ShapeOp {
    fn op_type(&self) -> &str {
        "Shape"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let (start, end) = shape_slice_bounds(x.shape.len(), ctx.attrs());
        let shape_vals: Vec<f32> = x.shape[start..end].iter().map(|&d| d as f32).collect();
        let n = shape_vals.len();
        Ok(vec![Tensor::new(shape_vals, vec![n])])
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
        let (start, end) = shape_slice_bounds(x.shape.len(), ctx.attrs());
        let n = end - start;
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape = vec![n];
        for (d, &dim) in slots[0].data.iter_mut().zip(x.shape[start..end].iter()) {
            *d = dim as f32;
        }
        Ok(())
    }
}

// ── Constant ────────────────────────────────────────────────────────────────

pub struct ConstantOp;
impl Operator for ConstantOp {
    fn op_type(&self) -> &str {
        "Constant"
    }
    /// Opset-21 documents `value_float` / `value_int` as "the value for the sole element for
    /// the scalar ... output tensor", so both produce a rank-0 output (the empty shape), as
    /// does the no-attribute fallback that stands in for them. The `value` tensor attribute is
    /// different: it carries its own shape, which is passed through untouched.
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        if let Some(t) = attrs.tensors.get("value") {
            Ok(vec![t.clone()])
        } else if let Some(&v) = attrs.floats.get("value_float") {
            Ok(vec![Tensor::new(vec![v], Vec::new())])
        } else if let Some(&v) = attrs.ints.get("value_int") {
            Ok(vec![Tensor::new(vec![v as f32], Vec::new())])
        } else {
            Ok(vec![Tensor::new(vec![0.0], Vec::new())])
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
        let attrs = ctx.attrs();
        if let Some(t) = attrs.tensors.get("value") {
            let n = t.data.len();
            if slots[0].data.len() != n {
                slots[0].data.resize(n, 0.0_f32);
            }
            slots[0].shape.clone_from(&t.shape);
            slots[0].data.copy_from_slice(&t.data);
        } else if let Some(&v) = attrs.floats.get("value_float") {
            if slots[0].data.len() != 1 {
                slots[0].data.resize(1, 0.0_f32);
            }
            // Rank 0, matching `execute` above (all three scalar arms).
            slots[0].shape = Vec::new();
            slots[0].data[0] = v;
        } else if let Some(&v) = attrs.ints.get("value_int") {
            if slots[0].data.len() != 1 {
                slots[0].data.resize(1, 0.0_f32);
            }
            slots[0].shape = Vec::new();
            slots[0].data[0] = v as f32;
        } else {
            if slots[0].data.len() != 1 {
                slots[0].data.resize(1, 0.0_f32);
            }
            slots[0].shape = Vec::new();
            slots[0].data[0] = 0.0_f32;
        }
        Ok(())
    }
}

// ── Einsum ──────────────────────────────────────────────────────────────────

pub struct EinsumOp;
impl Operator for EinsumOp {
    fn op_type(&self) -> &str {
        "Einsum"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let equation = ctx.attrs().s("equation");
        let tensors: Vec<&Tensor> = ctx.inputs.iter().filter_map(|opt| *opt).collect();
        Ok(vec![crate::einsum::einsum(equation, &tensors)?])
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
        let result = self
            .execute(ctx)?
            .into_iter()
            .next()
            .ok_or_else(|| OnnxError::Internal("EinsumOp: no output".into()))?;
        let out = &mut slots[0];
        if out.data.len() == result.data.len() && out.shape == result.shape {
            out.data.copy_from_slice(&result.data);
            Ok(())
        } else {
            // Every caller of `execute_into_slots` (`Session::dispatch_node`,
            // `src/session/run/dispatch.rs`, and the parallel scheduler's
            // `claim_cpu_fast_paths`, `src/session/run/parallel.rs`) reaches
            // this only via `acquire_output_slots`, which returns `Some`
            // (and therefore pre-sizes `out` from `resolved_shapes`) only
            // when shape inference already produced a shape for this node's
            // output. For `Einsum` that shape comes from
            // `infer_einsum_shape` (`src/optimizer/shape_inference_ext/advanced.rs`),
            // so landing here means that prediction disagreed with what the
            // executor actually computed -- a shape-inference bug, not a
            // recoverable runtime condition. Reallocating (`*out = result`)
            // used to paper over exactly that: a stale/incorrect inference
            // self-healed silently at the cost of the allocation the slot
            // path exists to avoid, and the underlying bug went undetected
            // until a dedicated end-to-end regression test caught it (see
            // the doc comment on `infer_einsum_shape`). A typed error
            // surfaces that class of bug immediately instead of masking it
            // again.
            Err(OnnxError::Internal(format!(
                "EinsumOp: pre-allocated output slot (shape {:?}, {} elements) disagrees \
                 with the computed result (shape {:?}, {} elements) -- Einsum shape \
                 inference disagrees with the executor",
                out.shape,
                out.data.len(),
                result.shape,
                result.data.len()
            )))
        }
    }
}

// ── Bitwise ──────────────────────────────────────────────────────────────

macro_rules! bitwise_binary_op {
    ($name:ident, $op_type:expr, $func:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                Ok(vec![$func(ctx.input(0)?, ctx.input(1)?)?])
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
                let result = $func(ctx.input(0)?, ctx.input(1)?)?;
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

bitwise_binary_op!(BitwiseAndOp, "BitwiseAnd", crate::bitwise::bitwise_and);
bitwise_binary_op!(BitwiseOrOp, "BitwiseOr", crate::bitwise::bitwise_or);
bitwise_binary_op!(BitwiseXorOp, "BitwiseXor", crate::bitwise::bitwise_xor);

pub struct BitwiseNotOp;
impl Operator for BitwiseNotOp {
    fn op_type(&self) -> &str {
        "BitwiseNot"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![crate::bitwise::bitwise_not(ctx.input(0)?)])
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
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        // Value-preserving i64 round-trip (not u32): a two's-complement NOT of
        // a small value like 0 must come back as -1, which f32 represents
        // exactly, rather than as 4294967295 (u32::MAX), which f32 cannot. See
        // `crate::bitwise::bitwise_not`, which this mirrors.
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = (!(x as i64)) as f32;
        }
        Ok(())
    }
}

// ── Size ─────────────────────────────────────────────────────────────────

pub struct SizeOp;
impl Operator for SizeOp {
    fn op_type(&self) -> &str {
        "Size"
    }
    /// Opset-21 `Size` "outputs an int64 scalar that equals the total number of elements of
    /// the input tensor", so the output is rank 0 (the empty shape) for every input rank —
    /// not the rank-1 `[1]` this used to emit.
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let n = x.numel();
        Ok(vec![Tensor::new(vec![n as f32], Vec::new())])
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
        let n = ctx.input(0)?.numel();
        if slots[0].data.len() != 1 {
            slots[0].data.resize(1, 0.0_f32);
        }
        // Rank 0, matching `execute` above.
        slots[0].shape = Vec::new();
        slots[0].data[0] = n as f32;
        Ok(())
    }
}

// ── NonMaxSuppression ───────────────────────────────────────────────────────

pub struct NonMaxSuppressionOp;
impl Operator for NonMaxSuppressionOp {
    fn op_type(&self) -> &str {
        "NonMaxSuppression"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let boxes = ctx.input(0)?;
        let scores = ctx.input(1)?;
        // `.first()`, not `[0]`, on all three: a present-but-empty scalar input
        // (malformed model) must fall back to the default rather than index an
        // empty `data` slice.
        let max_out = ctx
            .optional_input(2)
            .and_then(|t| t.data.first().copied())
            .map(|v| v as usize)
            .unwrap_or(0);
        let iou_thresh = ctx
            .optional_input(3)
            .and_then(|t| t.data.first().copied())
            .unwrap_or(0.0);
        let score_thresh = ctx
            .optional_input(4)
            .and_then(|t| t.data.first().copied())
            .unwrap_or(f32::NEG_INFINITY);
        let center_point_box = ctx.attrs().i("center_point_box", 0);
        Ok(vec![crate::nms::non_max_suppression(
            boxes,
            scores,
            max_out,
            iou_thresh,
            score_thresh,
            center_point_box,
        )?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}
