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
            .map(|t| t.data[0])
            .unwrap_or(0.0);
        Ok(vec![comparison::constant_of_shape(&target_shape, value)])
    }
    fn supports_output_slots(&self) -> bool {
        true
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
        let k = ctx.optional_input(1).map(|t| t.data[0] as i64).unwrap_or(0);
        Ok(vec![comparison::trilu(x, upper, k)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
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
        let data: Vec<f32> = match to {
            9 => x
                .data
                .iter()
                .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
                .collect(),
            6 | 7 | 12 | 13 => x.data.iter().map(|&v| v.round()).collect(),
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
        let target_dtype =
            oxionnx_core::DType::from_onnx(to_int as i32).unwrap_or(oxionnx_core::DType::F32);
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

pub struct ShapeOp;
impl Operator for ShapeOp {
    fn op_type(&self) -> &str {
        "Shape"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let shape_vals: Vec<f32> = x.shape.iter().map(|&d| d as f32).collect();
        let n = shape_vals.len();
        Ok(vec![Tensor::new(shape_vals, vec![n])])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── Constant ────────────────────────────────────────────────────────────────

pub struct ConstantOp;
impl Operator for ConstantOp {
    fn op_type(&self) -> &str {
        "Constant"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        if let Some(t) = attrs.tensors.get("value") {
            Ok(vec![t.clone()])
        } else if let Some(&v) = attrs.floats.get("value_float") {
            Ok(vec![Tensor::new(vec![v], vec![1])])
        } else if let Some(&v) = attrs.ints.get("value_int") {
            Ok(vec![Tensor::new(vec![v as f32], vec![1])])
        } else {
            Ok(vec![Tensor::new(vec![0.0], vec![1])])
        }
    }
    fn supports_output_slots(&self) -> bool {
        true
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
}

// ── Size ─────────────────────────────────────────────────────────────────

pub struct SizeOp;
impl Operator for SizeOp {
    fn op_type(&self) -> &str {
        "Size"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let n = x.numel();
        Ok(vec![Tensor::new(vec![n as f32], vec![1])])
    }
    fn supports_output_slots(&self) -> bool {
        true
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
        let max_out = ctx
            .optional_input(2)
            .map(|t| t.data[0] as usize)
            .unwrap_or(0);
        let iou_thresh = ctx.optional_input(3).map(|t| t.data[0]).unwrap_or(0.0);
        let score_thresh = ctx
            .optional_input(4)
            .map(|t| t.data[0])
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
