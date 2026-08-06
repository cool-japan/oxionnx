//! Operator trait implementations for the int8/uint8 quantized operator family.
//!
//! These are the operators emitted by ONNX Runtime's *static* quantizer (the
//! dominant int8 path: `quantize_static` in QOperator format) and by the
//! dynamic-quantization path: `QLinearConv`, `QLinearMatMul`, `MatMulInteger`,
//! `ConvInteger` and `DynamicQuantizeLinear`.
//!
//! # Integer semantics on an f32-backed runtime
//!
//! This engine's [`Tensor`] is a flat `Vec<f32>` with no
//! dtype tag, so a quantized tensor arrives here as integer *values* stored in
//! f32 lanes. Every kernel below therefore
//!
//! 1. converts each lane to `i32` once, up front (`lane_to_i32`),
//! 2. accumulates in `i64` — products reach `255 * 255` and a realistic kernel
//!    volume pushes the sum well past f32's exact-integer range (2^24), so an
//!    f32 accumulator would silently lose low bits, and
//! 3. requantizes exactly the way the ONNX reference implementation does:
//!    the combined scale `x_scale * w_scale / y_scale` is formed in **f32**,
//!    the product with the accumulator is taken in **f64**, and the result is
//!    rounded **ties-to-even** (`np.round` / `np.rint`) before saturation.
//!
//! # Saturation range inference
//!
//! `QLinearConv`/`QLinearMatMul` output `uint8` or `int8` depending on the
//! declared type of their `y_zero_point`, which a dtype-erased `Tensor` cannot
//! report. `SatRange::infer` documents the cascade used instead.

use oxionnx_core::{DType, OnnxError, Tensor, TypedOpContext};

mod conv_kernel;
mod dynamic_quantize;
mod qlinear_conv;
mod qlinear_matmul;

pub use dynamic_quantize::DynamicQuantizeLinearOp;
pub use qlinear_conv::{ConvIntegerOp, QLinearConvOp};
pub use qlinear_matmul::{MatMulIntegerOp, QLinearMatMulOp};

/// Round to the nearest integer, **ties to even** (banker's rounding) — the
/// mode the ONNX quantization operators mandate and the one `numpy.round` /
/// `numpy.rint` implement, so the reference values these kernels are checked
/// against land on the same side of every `.5`.
///
/// Rust's `f64::round` is ties-*away*-from-zero and diverges exactly at that
/// boundary (`2.5 -> 3` instead of `2`). The standard library's
/// `f64::round_ties_even` is stable only since Rust 1.77, above this
/// workspace's declared MSRV (1.75), hence the hand-rolled version — the same
/// reasoning (and shape) as `indexing::quantize::round_ties_even`.
#[inline]
fn round_ties_even_f64(v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    let floor = v.floor();
    let diff = v - floor;
    // Round up when strictly past the midpoint, and — at an exact tie — only
    // when `floor` is odd, so the result lands on the even neighbour.
    let round_up = diff > 0.5 || (diff == 0.5 && floor.rem_euclid(2.0) != 0.0);
    if round_up {
        floor + 1.0
    } else {
        floor
    }
}

/// `f32` view of [`round_ties_even_f64`].
#[inline]
pub(super) fn round_ties_even_f32(v: f32) -> f32 {
    round_ties_even_f64(v as f64) as f32
}

/// Convert one f32 lane holding a quantized integer value into `i32`.
///
/// A non-finite lane is a malformed model (a quantized tensor cannot hold
/// NaN/±inf) and is reported rather than silently becoming `0`. A lane that is
/// not exactly integral is rounded ties-to-even instead of truncated, so a
/// producer that wrote `4.999999` for `5` does not lose a whole step.
fn lane_to_i32(v: f32, what: &str) -> Result<i32, OnnxError> {
    if !v.is_finite() {
        return Err(OnnxError::InvalidModel(format!(
            "{what}: quantized tensor contains a non-finite value ({v})"
        )));
    }
    let r = round_ties_even_f32(v);
    if r < i32::MIN as f32 || r > i32::MAX as f32 {
        return Err(OnnxError::InvalidModel(format!(
            "{what}: quantized value {v} is outside the int32 range"
        )));
    }
    Ok(r as i32)
}

/// Convert a whole tensor of quantized lanes into `i32`.
fn tensor_to_i32(t: &Tensor, what: &str) -> Result<Vec<i32>, OnnxError> {
    t.data.iter().map(|&v| lane_to_i32(v, what)).collect()
}

/// Read a required scalar (or 1-element) f32 input.
fn scalar_f32(t: &Tensor, what: &str) -> Result<f32, OnnxError> {
    match t.data.first() {
        Some(&v) if v.is_finite() => Ok(v),
        Some(&v) => Err(OnnxError::InvalidModel(format!(
            "{what}: scale must be finite, got {v}"
        ))),
        None => Err(OnnxError::InvalidModel(format!("{what}: is empty"))),
    }
}

/// Read a required scalar quantization *scale*, rejecting zero (which would
/// make the requantization divide by zero).
fn scale_input(t: &Tensor, what: &str) -> Result<f32, OnnxError> {
    let v = scalar_f32(t, what)?;
    if v == 0.0 {
        return Err(OnnxError::InvalidModel(format!("{what}: scale is zero")));
    }
    Ok(v)
}

/// Read an optional zero-point input as `i32` lanes, defaulting to a single `0`.
fn zero_point_lanes(zp: Option<&Tensor>, what: &str) -> Result<Vec<i32>, OnnxError> {
    match zp {
        None => Ok(vec![0]),
        Some(t) if t.data.is_empty() => Ok(vec![0]),
        Some(t) => tensor_to_i32(t, what),
    }
}

/// Project every input of a [`TypedOpContext`] down to its f32 value
/// representation — the same conversion
/// [`oxionnx_core::default_typed_via_f32`] performs on its way to building an
/// f32 [`oxionnx_core::OpContext`].
///
/// Split out as its own step (rather than delegating to
/// `default_typed_via_f32` wholesale) because `QLinearConvOp` /
/// `QLinearMatMulOp`'s `execute_typed` need to read `y_zero_point`'s
/// *original* typed dtype — via [`SatRange::for_dtype`] — before this
/// projection erases it down to plain f32 lanes.
fn project_to_f32(ctx: &TypedOpContext<'_>) -> Vec<Option<Tensor>> {
    ctx.inputs
        .iter()
        .map(|maybe| {
            maybe.map(|tt| {
                let data = tt.storage.to_f32_vec();
                Tensor::new(data, tt.shape.clone())
            })
        })
        .collect()
}

/// Saturation range of a quantized 8-bit output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SatRange {
    lo: i64,
    hi: i64,
}

impl SatRange {
    /// `uint8`: `[0, 255]`.
    const U8: Self = Self { lo: 0, hi: 255 };
    /// `int8`: `[-128, 127]`.
    const I8: Self = Self { lo: -128, hi: 127 };
    /// The union of both, used only when neither zero point can disambiguate.
    const UNION: Self = Self { lo: -128, hi: 255 };

    /// Infer the output saturation range of a `QLinear*` operator.
    ///
    /// The spec pins the range to the *declared element type* of
    /// `y_zero_point`, which this runtime's dtype-erased `Tensor` cannot
    /// report. The following deterministic cascade is used instead — the
    /// output zero point first, then the input zero point (every mainstream
    /// static quantizer uses one activation dtype throughout a graph), then a
    /// deliberate refusal to guess:
    ///
    /// 1. any `y_zero_point` lane `> 127`  → `uint8` (only uint8 reaches it),
    /// 2. any `y_zero_point` lane `< 0`    → `int8`  (only int8 reaches it),
    /// 3. any `x_zero_point` lane `> 127`  → `uint8`,
    /// 4. any `x_zero_point` lane `< 0`    → `int8`,
    /// 5. otherwise (every zero point in `0..=127`, which both dtypes can
    ///    produce — symmetric `int8` and post-`ReLU` `uint8` both sit at 0) →
    ///    the **union** `[-128, 255]`.
    ///
    /// Step 5 is chosen over a coin flip because both wrong guesses destroy
    /// data that the union preserves: assuming `int8` clips a legitimate
    /// uint8 `134` down to `127`, and assuming `uint8` clips a legitimate
    /// int8 `-9` up to `0`. Clamping to the union is a no-op for every value
    /// that the true dtype would not have saturated, and a model that
    /// genuinely relies on saturation inside that ambiguous band needs the
    /// typed execution path (where the real dtypes survive).
    fn infer(y_zero_point: &[i32], x_zero_point: &[i32]) -> Self {
        for lanes in [y_zero_point, x_zero_point] {
            if lanes.iter().any(|&v| v > 127) {
                return Self::U8;
            }
            if lanes.iter().any(|&v| v < 0) {
                return Self::I8;
            }
        }
        Self::UNION
    }

    /// The exact range for a *declared* `y_zero_point` dtype, bypassing
    /// [`Self::infer`]'s value-based cascade entirely.
    ///
    /// This is the one avenue this dtype-erased runtime has to resolve
    /// `infer`'s union-range ambiguity exactly: it is reachable only when
    /// `y_zero_point` survives to the caller as a genuinely typed `I8`/`U8`
    /// [`oxionnx_core::TypedTensor`] — a `Session::run_typed` graph input, or
    /// an upstream operator's native typed output — rather than as an
    /// f32-lane [`Tensor`] or an f32-only model initializer, in which case
    /// the dtype tag was already lost before this call and `None` tells the
    /// caller to fall back to `infer`.
    ///
    /// Any other dtype (e.g. `y_zero_point` mistakenly carrying `I32`) also
    /// falls back to `infer` rather than being treated as exact.
    fn for_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::U8 => Some(Self::U8),
            DType::I8 => Some(Self::I8),
            _ => None,
        }
    }

    /// Requantize one integer accumulator into the output lane.
    ///
    /// `combined` is `x_scale * w_scale / y_scale`, already formed in f32 the
    /// way the ONNX reference does; the multiply itself is f64 (numpy promotes
    /// `int32 * float32` to float64), and the rounding is ties-to-even.
    fn requantize(self, acc: i64, combined: f32, zero_point: i32) -> f32 {
        let scaled = acc as f64 * combined as f64 + zero_point as f64;
        let rounded = round_ties_even_f64(scaled);
        rounded.clamp(self.lo as f64, self.hi as f64) as f32
    }
}

/// Saturate an `i64` accumulator into the `int32` output of `MatMulInteger` /
/// `ConvInteger`.
///
/// Those two operators emit `int32` with **no** scale and **no** clamping; an
/// accumulator that overflows `int32` means the model is out of contract, and
/// saturating (rather than wrapping like a C `int32` add) keeps the magnitude
/// monotone instead of flipping sign.
fn saturate_i32(acc: i64) -> f32 {
    acc.clamp(i32::MIN as i64, i32::MAX as i64) as f32
}
