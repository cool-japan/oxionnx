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

// Basic unary ops with in-place support.
//
// Sqrt/Reciprocal are inherently real-valued (no exact integer result exists
// even in principle) and stay on the plain f32-round-trip macros; Neg/Ceil/
// Floor/Round/Sign are exact on integers and use the `_exact_int` variant so
// an `I32`/`I64` input never takes the lossy f32 round-trip — see
// `macros.rs`'s module-level doc comment for the full rationale.
unary_op_inplace!(SqrtOp, "Sqrt", math::sqrt, f32::sqrt);
unary_op_plain!(ReciprocalOp, "Reciprocal", math::reciprocal);
unary_op_inplace_exact_int!(
    NegOp,
    "Neg",
    math::neg,
    |x| -x,
    // `wrapping_neg`, not bare `-x`: negating `i32::MIN`/`i64::MIN` would
    // otherwise panic on overflow in a debug build.
    |x: i32| x.wrapping_neg(),
    |x: i64| x.wrapping_neg()
);
unary_op_inplace_exact_int!(
    CeilOp,
    "Ceil",
    math::ceil,
    f32::ceil,
    // An already-integer value has no fractional part to round: ceil is the
    // identity on I32/I64.
    |x: i32| x,
    |x: i64| x
);
unary_op_inplace_exact_int!(
    FloorOp,
    "Floor",
    math::floor_op,
    f32::floor,
    |x: i32| x,
    |x: i64| x
);
// [G1-ops-close] The in-place/slot path (`$inplace_fn`, the 4th argument) used to be bare
// `f32::round`, which is round-half-*away*-from-zero and disagrees with the ONNX spec (and with
// `math::round_op`, the `execute()` path via the 3rd argument) on exact `.5` boundaries
// (`2.5 -> 3.0` instead of `2.0`). Fixed by passing `math::round_half_to_even` here too, so
// `execute()`, `execute_inplace()`, and `execute_into_slots()` (the latter two both driven by
// this same `$inplace_fn` argument, see `unary_op_inplace_exact_int!` above) now share the exact
// same banker's-rounding function instead of two independently-written rounding rules that can
// drift apart the way they just did.
unary_op_inplace_exact_int!(
    RoundOp,
    "Round",
    math::round_op,
    math::round_half_to_even,
    |x: i32| x,
    |x: i64| x
);
unary_op_inplace_exact_int!(
    SignOp,
    "Sign",
    math::sign,
    |x: f32| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    },
    // `i32::signum`/`i64::signum` return exactly -1/0/1, matching the f32
    // closure's cases one for one (integers have no `-0.0` to fold into the
    // `0` case the way f32 does, but the observable result is identical:
    // both map "not positive, not negative" to `0`).
    |x: i32| x.signum(),
    |x: i64| x.signum()
);

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
