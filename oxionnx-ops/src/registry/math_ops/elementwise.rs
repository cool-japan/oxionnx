//! Elementwise math operator implementations (binary and unary, including trig).

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::math;

// Basic binary ops with in-place support (all return Result<Tensor, String>)
binary_op_inplace!(
    AddOp,
    "Add",
    math::add,
    |a, b| a + b,
    crate::typed_ops::typed_add
);
binary_op_inplace!(
    SubOp,
    "Sub",
    math::sub,
    |a, b| a - b,
    crate::typed_ops::typed_sub
);
binary_op_inplace!(
    MulOp,
    "Mul",
    math::mul,
    |a, b| a * b,
    crate::typed_ops::typed_mul
);
binary_op_inplace!(
    DivOp,
    "Div",
    math::div,
    |a, b| a / b,
    crate::typed_ops::typed_div
);
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
