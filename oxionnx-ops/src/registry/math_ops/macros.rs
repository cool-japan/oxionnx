//! Shared macro definitions used across math op submodules.
//!
//! # The exact-integer / real-valued boundary
//!
//! `native_dtypes()`'s contract ([`oxionnx_core::Operator::native_dtypes`]) is
//! "dtypes this operator can execute *without an f32 round-trip*" — and an f32
//! round-trip is lossy above `2^24` (`16_777_216`): every whole-number `i64`
//! beyond that loses its ones digit the moment it is cast to `f32`, since f32's
//! mantissa has only 24 significant bits. Every unary op in this file declares
//! `I32`/`I64` in `native_dtypes()` (kept for compatibility with existing
//! callers that probe it — see `tests/typed_io_test.rs::
//! test_native_dtypes_math_pilot_ops` in the root crate), but only *some* of
//! them can actually honour that "no round-trip" promise, which makes two
//! genuinely different kinds of unary op here:
//!
//! - **Exact on integers** — [`unary_op_inplace_exact_int!`]: `Neg`, `Ceil`,
//!   `Floor`, `Round`, `Sign`. Each has a well-defined, exactly-representable
//!   result for an integer input (negation, or the identity — `ceil`/`floor`/
//!   `round` of an already-integer value never has a fractional part to
//!   round), so `execute_typed()` computes it with real `i32`/`i64` arithmetic
//!   and never touches f32 for those dtypes. This is the fix for the
//!   confirmed a11–a14 finding: before this, `NegOp`/`CeilOp`/`FloorOp`/
//!   `RoundOp`/`SignOp` claimed `I32`/`I64` in `native_dtypes()` (steering
//!   `Session::run_typed` to call `execute_typed` instead of casting) but
//!   `execute_typed()` was `default_typed_via_f32`, which silently rounded any
//!   `i64` above `2^24` through f32 and returned it re-tagged as `F32` —
//!   `native_dtypes()` was promising exactness these ops did not deliver.
//! - **Inherently real-valued** — [`unary_op_plain!`] / [`unary_op_inplace!`]:
//!   `Sqrt`, `Reciprocal`, and the trig family (`Sin`, `Cos`, `Tan`, `Asin`,
//!   …). There is no exact integer result to compute here even in principle —
//!   `sqrt`/`sin` of an arbitrary integer is generally irrational, so there is
//!   nothing an `I64`-native arm could return that both stays exactly `I64`
//!   *and* is the mathematically correct answer. Unlike the exact-integer
//!   family, `execute_typed()` for these **stays** `default_typed_via_f32`
//!   for every dtype including `I32`/`I64` — an `i64` input above `2^24` is
//!   still rounded through f32 here, same as before this fix, because there
//!   is no non-lossy alternative to round to. This half of the boundary is
//!   therefore a documentation fix, not a behavioural one: nothing about
//!   these ops changes.

/// Unary op (plain — no Result, no in-place support).
///
/// For ops whose math is inherently real-valued (trig, `Reciprocal`, …) — see
/// the module-level "exact-integer / real-valued boundary" doc above.
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
///
/// For ops whose math is inherently real-valued (`Sqrt`, …) — see the
/// module-level "exact-integer / real-valued boundary" doc above. Ops that
/// are exact on integers use [`unary_op_inplace_exact_int!`] instead.
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

/// Unary op with in-place f32 support **and** a genuinely exact typed
/// dispatch on `I32`/`I64` — no f32 round-trip for those two dtypes, so an
/// `i64` input above `2^24` survives `execute_typed()` exactly.
///
/// For ops whose math is exact on integers (`Neg`, `Ceil`, `Floor`, `Round`,
/// `Sign`) — see the module-level "exact-integer / real-valued boundary" doc
/// above. `$int32_fn` / `$int64_fn` are the exact `i32 -> i32` / `i64 -> i64`
/// transforms; they must each be overflow-safe (e.g. `i32::wrapping_neg`, not
/// bare `-x`, for `Neg` — negating `i32::MIN`/`i64::MIN` would otherwise
/// panic in a debug build) since a typed error return is not available on
/// this hot per-element path.
macro_rules! unary_op_inplace_exact_int {
    ($name:ident, $op_type:expr, $func:path, $inplace_fn:expr, $int32_fn:expr, $int64_fn:expr) => {
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
                use oxionnx_core::{OnnxError, TensorStorage, TypedTensor};
                let input = ctx.input(0).ok_or_else(|| {
                    OnnxError::TensorNotFound(format!("{}: missing input[0]", $op_type))
                })?;
                match &input.storage {
                    // I32/I64: exact, no f32 round-trip — the whole point of
                    // this macro over `unary_op_inplace!`.
                    TensorStorage::I32(data) => {
                        let f: fn(i32) -> i32 = $int32_fn;
                        let out: Vec<i32> = data.iter().map(|&x| f(x)).collect();
                        Ok(vec![TypedTensor::new(
                            TensorStorage::I32(out),
                            input.shape.clone(),
                        )])
                    }
                    TensorStorage::I64(data) => {
                        let f: fn(i64) -> i64 = $int64_fn;
                        let out: Vec<i64> = data.iter().map(|&x| f(x)).collect();
                        Ok(vec![TypedTensor::new(
                            TensorStorage::I64(out),
                            input.shape.clone(),
                        )])
                    }
                    // F32/F16/BF16: real-valued path, f32 round-trip is exact
                    // for these (a whole-number FP value round-trips through
                    // f32 unchanged, and F16/BF16 already have <=24-bit
                    // mantissas so promoting to f32 cannot lose anything they
                    // had).
                    _ => oxionnx_core::default_typed_via_f32(self, ctx),
                }
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

#[cfg(test)]
mod tests {
    use crate::registry::math_ops::{
        CeilOp, FloorOp, NegOp, ReciprocalOp, RoundOp, SignOp, SinOp, SqrtOp,
    };
    use oxionnx_core::{
        Attributes, DType, Node, OpKind, Operator, TensorStorage, TypedOpContext, TypedTensor,
    };

    /// `2^24 + 1` — the smallest whole number an f32 round-trip cannot
    /// represent exactly (f32's mantissa has 24 significant bits, so every
    /// integer through `2^24` round-trips exactly and every one above it may
    /// not). An op whose `execute_typed()` is genuinely exact on `I64` must
    /// return this value unchanged (modulo a well-defined integer transform,
    /// e.g. negation); an op that silently rounds through f32 turns it into
    /// `16_777_216`, which is exactly the bug this test set pins shut. See
    /// this file's module-level doc comment.
    const ABOVE_F32_EXACT_RANGE: i64 = 16_777_217;

    fn dummy_node(op: OpKind) -> Node {
        Node {
            name: "t".into(),
            op,
            inputs: vec![],
            outputs: vec![],
            attrs: Attributes::default(),
        }
    }

    fn ctx1<'a>(node: &'a Node, x: &'a TypedTensor) -> TypedOpContext<'a> {
        TypedOpContext {
            node,
            inputs: vec![Some(x)],
            outer_scope: None,
            registry: None,
        }
    }

    // ── exact-on-integers ops preserve i64 above 2^24 ───────────────────────
    //
    // NOTE: these tests call `execute_typed()` directly, proving the op-level
    // arithmetic is exact. They do not exercise `Session::run_typed`'s
    // `native_dtypes()` gate (`src/session/run/typed.rs`, not owned by this
    // file) that decides whether `execute_typed` is reached at all instead of
    // the surgical f32-cast branch — an end-to-end regression test for "a
    // model computing `Neg` on an `I64` input above 2^24 through
    // `Session::run_typed` preserves the exact value" belongs in the root
    // crate's session/session-run test suite, not here.

    #[test]
    fn neg_execute_typed_is_exact_on_i64_above_two_pow_24() {
        let node = dummy_node(OpKind::Neg);
        let x = TypedTensor::new(TensorStorage::I64(vec![ABOVE_F32_EXACT_RANGE]), vec![1]);
        let ctx = ctx1(&node, &x);
        let out = NegOp.execute_typed(&ctx).expect("Neg execute_typed");
        match &out[0].storage {
            TensorStorage::I64(d) => assert_eq!(d, &vec![-ABOVE_F32_EXACT_RANGE]),
            other => panic!("expected I64 storage, got {other:?}"),
        }
    }

    #[test]
    fn ceil_floor_round_execute_typed_are_the_identity_on_i64_above_two_pow_24() {
        let x = TypedTensor::new(TensorStorage::I64(vec![ABOVE_F32_EXACT_RANGE]), vec![1]);

        let node = dummy_node(OpKind::Ceil);
        let ctx = ctx1(&node, &x);
        match &CeilOp.execute_typed(&ctx).expect("Ceil execute_typed")[0].storage {
            TensorStorage::I64(d) => assert_eq!(d, &vec![ABOVE_F32_EXACT_RANGE]),
            other => panic!("expected I64 storage, got {other:?}"),
        }

        let node = dummy_node(OpKind::Floor);
        let ctx = ctx1(&node, &x);
        match &FloorOp.execute_typed(&ctx).expect("Floor execute_typed")[0].storage {
            TensorStorage::I64(d) => assert_eq!(d, &vec![ABOVE_F32_EXACT_RANGE]),
            other => panic!("expected I64 storage, got {other:?}"),
        }

        let node = dummy_node(OpKind::Round);
        let ctx = ctx1(&node, &x);
        match &RoundOp.execute_typed(&ctx).expect("Round execute_typed")[0].storage {
            TensorStorage::I64(d) => assert_eq!(d, &vec![ABOVE_F32_EXACT_RANGE]),
            other => panic!("expected I64 storage, got {other:?}"),
        }
    }

    #[test]
    fn sign_execute_typed_is_exact_on_i64_and_i32_including_above_two_pow_24() {
        let node = dummy_node(OpKind::Sign);
        let x = TypedTensor::new(
            TensorStorage::I64(vec![ABOVE_F32_EXACT_RANGE, -ABOVE_F32_EXACT_RANGE, 0]),
            vec![3],
        );
        let ctx = ctx1(&node, &x);
        match &SignOp.execute_typed(&ctx).expect("Sign execute_typed i64")[0].storage {
            TensorStorage::I64(d) => assert_eq!(d, &vec![1, -1, 0]),
            other => panic!("expected I64 storage, got {other:?}"),
        }

        let x32 = TypedTensor::new(TensorStorage::I32(vec![5, -5, 0]), vec![3]);
        let ctx32 = ctx1(&node, &x32);
        match &SignOp
            .execute_typed(&ctx32)
            .expect("Sign execute_typed i32")[0]
            .storage
        {
            TensorStorage::I32(d) => assert_eq!(d, &vec![1, -1, 0]),
            other => panic!("expected I32 storage, got {other:?}"),
        }
    }

    #[test]
    fn neg_execute_typed_wraps_i32_min_and_i64_min_instead_of_panicking() {
        let node = dummy_node(OpKind::Neg);

        let x32 = TypedTensor::new(TensorStorage::I32(vec![i32::MIN]), vec![1]);
        let ctx32 = ctx1(&node, &x32);
        match &NegOp
            .execute_typed(&ctx32)
            .expect("Neg execute_typed i32::MIN")[0]
            .storage
        {
            TensorStorage::I32(d) => assert_eq!(d, &vec![i32::MIN.wrapping_neg()]),
            other => panic!("expected I32 storage, got {other:?}"),
        }

        let x64 = TypedTensor::new(TensorStorage::I64(vec![i64::MIN]), vec![1]);
        let ctx64 = ctx1(&node, &x64);
        match &NegOp
            .execute_typed(&ctx64)
            .expect("Neg execute_typed i64::MIN")[0]
            .storage
        {
            TensorStorage::I64(d) => assert_eq!(d, &vec![i64::MIN.wrapping_neg()]),
            other => panic!("expected I64 storage, got {other:?}"),
        }
    }

    // ── native_dtypes contract: both families claim I32/I64 (see the module-
    //    level doc comment for why); only the exact-int family also *honours*
    //    it without an f32 round-trip. ──────────────────────────────────────

    #[test]
    fn exact_int_ops_claim_i32_and_i64_in_native_dtypes() {
        for dtypes in [
            NegOp.native_dtypes(),
            CeilOp.native_dtypes(),
            FloorOp.native_dtypes(),
            RoundOp.native_dtypes(),
            SignOp.native_dtypes(),
        ] {
            assert!(dtypes.contains(&DType::I32), "{dtypes:?} must claim I32");
            assert!(dtypes.contains(&DType::I64), "{dtypes:?} must claim I64");
        }
    }

    /// Real-valued ops still declare `I32`/`I64` in `native_dtypes()` — for
    /// compatibility with existing callers that probe it (see
    /// `tests/typed_io_test.rs::test_native_dtypes_math_pilot_ops` in the root
    /// crate, which requires `SqrtOp` to claim `I64`) — but, unlike the
    /// exact-int family, `execute_typed()` for these still rounds an I64 input
    /// through f32 and returns it re-tagged as `F32`: there is no exact I64
    /// result for `sqrt` of an arbitrary integer to preserve in the first
    /// place, so nothing regresses by leaving this path exactly as it was.
    #[test]
    fn real_valued_ops_still_claim_i32_i64_but_execute_typed_still_returns_f32_for_them() {
        for dtypes in [
            SqrtOp.native_dtypes(),
            ReciprocalOp.native_dtypes(),
            SinOp.native_dtypes(),
        ] {
            assert!(dtypes.contains(&DType::I32), "{dtypes:?} must claim I32");
            assert!(dtypes.contains(&DType::I64), "{dtypes:?} must claim I64");
        }

        let node = dummy_node(OpKind::Sqrt);
        let x = TypedTensor::new(TensorStorage::I64(vec![16]), vec![1]);
        let ctx = ctx1(&node, &x);
        match &SqrtOp.execute_typed(&ctx).expect("Sqrt execute_typed i64")[0].storage {
            TensorStorage::F32(d) => assert_eq!(d, &vec![4.0]),
            other => panic!(
                "expected F32 storage (real-valued ops round-trip I64 through f32), got {other:?}"
            ),
        }
    }

    #[test]
    fn sqrt_execute_typed_computes_correctly_for_f32_input() {
        let node = dummy_node(OpKind::Sqrt);
        let x = TypedTensor::new(TensorStorage::F32(vec![4.0, 9.0]), vec![2]);
        let ctx = ctx1(&node, &x);
        match &SqrtOp.execute_typed(&ctx).expect("Sqrt execute_typed")[0].storage {
            TensorStorage::F32(d) => assert_eq!(d, &vec![2.0, 3.0]),
            other => panic!("expected F32 storage, got {other:?}"),
        }
    }
}
