use oxionnx_core::Tensor;

// ── Unary element-wise: trig & rounding ─────────────────────────────────────

pub fn ceil(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.ceil()).collect(), x.shape.clone())
}

pub fn floor_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.floor()).collect(), x.shape.clone())
}

/// Scalar round-half-to-even (banker's rounding), matching the ONNX `Round` spec: "In case of
/// halfs, the rule is to round them to the nearest even integer." Plain `f32::round` alone is
/// round-half-*away*-from-zero and disagrees with the spec exactly on `.5` boundaries
/// (`2.5 -> 3.0` instead of the spec's `2.0`).
///
/// This is the single source of truth for every `Round` execution path — `execute()` (via
/// [`round_op`] below, which now just maps this per element), `execute_inplace()`, and
/// `execute_into_slots()` (both in `registry/math_ops/elementwise.rs`, which passes this
/// function directly as their per-element closure) — so they cannot independently drift the way
/// `execute_inplace`/`execute_into_slots` once did by using bare `f32::round` while `execute()`
/// used this banker's-rounding logic.
///
/// The halfway test is an *exact* equality on the fractional part, not an epsilon comparison:
/// `0.5` is exactly representable in binary floating point (it is `2^-1`), so a true halfway
/// value's fractional part is bit-for-bit `0.5`, and an epsilon-width comparison here would be
/// wider than the distance between adjacent representable floats near `0.5` — `0.5.next_up()`
/// (`0.500000059...`), which is unambiguously closer to `1.0` than to `0.0` and is not a tie at
/// all, sits inside an `f32::EPSILON`-wide band around `0.5` and would be misclassified as one.
pub fn round_half_to_even(v: f32) -> f32 {
    let rounded = v.round();
    if v.fract().abs() == 0.5 {
        if rounded as i64 % 2 != 0 {
            rounded - v.signum()
        } else {
            rounded
        }
    } else {
        rounded
    }
}

/// Round half to even (banker's rounding), matching ONNX spec. Per-element wrapper over
/// [`round_half_to_even`] — see its doc comment for the full rationale.
pub fn round_op(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data.iter().map(|&v| round_half_to_even(v)).collect(),
        x.shape.clone(),
    )
}

/// Sign function: -1 for negative, 0 for zero, 1 for positive (ONNX convention).
pub fn sign(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|v| {
                if *v > 0.0 {
                    1.0
                } else if *v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect(),
        x.shape.clone(),
    )
}

pub fn sin_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.sin()).collect(), x.shape.clone())
}

pub fn cos_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.cos()).collect(), x.shape.clone())
}

pub fn tan_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.tan()).collect(), x.shape.clone())
}

pub fn asin_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.asin()).collect(), x.shape.clone())
}

pub fn acos_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.acos()).collect(), x.shape.clone())
}

pub fn atan_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.atan()).collect(), x.shape.clone())
}

pub fn sinh_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.sinh()).collect(), x.shape.clone())
}

pub fn cosh_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.cosh()).collect(), x.shape.clone())
}

pub fn asinh_op(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|v| (*v + (v * v + 1.0).sqrt()).ln())
            .collect(),
        x.shape.clone(),
    )
}

pub fn acosh_op(x: &Tensor) -> Tensor {
    Tensor::new(
        x.data
            .iter()
            .map(|v| (*v + (v * v - 1.0).sqrt()).ln())
            .collect(),
        x.shape.clone(),
    )
}

pub fn atanh_op(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|v| v.atanh()).collect(), x.shape.clone())
}
