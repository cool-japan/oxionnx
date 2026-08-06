//! Bitwise operators: BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot (ONNX opset 18).
//!
//! Tensors are stored as f32, so each element is recovered as an i64 *value*
//! (not a raw bit reinterpretation -- an f32 storing the number `5.0` is
//! treated as the integer `5`, matching how every other op in this crate
//! reads an "integer" tensor that has been carried through as f32) before the
//! bitwise op runs, then the i64 result is cast back to f32 for storage.
//!
//! This round-trips through i64 rather than u32 for two reasons:
//! - Two's-complement NOT/AND/OR/XOR on signed operands must preserve sign:
//!   `BitwiseNot(0)` on a signed int tensor must yield `-1`, and f32 can
//!   represent `-1.0` exactly, whereas the unsigned 32-bit pattern
//!   `4294967295` is *not* exactly representable in f32 (it rounds to
//!   2^32) and a negative operand cast `as u32` first saturates to 0,
//!   silently destroying the sign entirely (e.g. `BitwiseAnd(-1, 5)` would
//!   wrongly become `BitwiseAnd(0, 5) == 0` instead of `5`).
//! - i64 covers the full range ONNX allows for these ops (int8..int64,
//!   uint8..uint64), whereas u32 truncates 64-bit values.
//!
//! Precision limit (inherent to f32-backed storage, not this cast choice):
//! f32 has a 24-bit mantissa, so only integer magnitudes up to 2^24
//! (16,777,216) round-trip through it exactly. A logical int64 value beyond
//! that range is already lossy by the time it reaches this function as an
//! f32 -- the typed dispatch path (`execute_typed` / `native_dtypes`) should
//! be preferred wherever exactness beyond 2^24 matters.

use oxionnx_core::Tensor;

use crate::math::broadcast_to;

fn bitwise_binary(a: &Tensor, b: &Tensor, op: impl Fn(i64, i64) -> i64) -> Result<Tensor, String> {
    let target = Tensor::broadcast_shape(&a.shape, &b.shape)?;
    let ab = broadcast_to(a, &target);
    let bb = broadcast_to(b, &target);
    let data: Vec<f32> = ab
        .data
        .iter()
        .zip(bb.data.iter())
        .map(|(&x, &y)| op(x as i64, y as i64) as f32)
        .collect();
    Ok(Tensor::new(data, target))
}

pub fn bitwise_and(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    bitwise_binary(a, b, |x, y| x & y)
}

pub fn bitwise_or(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    bitwise_binary(a, b, |x, y| x | y)
}

pub fn bitwise_xor(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    bitwise_binary(a, b, |x, y| x ^ y)
}

pub fn bitwise_not(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data.iter().map(|&v| (!(v as i64)) as f32).collect();
    Tensor::new(data, x.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::Tensor;

    #[test]
    fn test_bitwise_and() {
        let a = Tensor::new(vec![5.0, 3.0], vec![2]);
        let b = Tensor::new(vec![3.0, 6.0], vec![2]);
        let out = bitwise_and(&a, &b).unwrap();
        assert_eq!(out.data[0] as u32, 5u32 & 3u32);
        assert_eq!(out.data[1] as u32, 3u32 & 6u32);
    }

    #[test]
    fn test_bitwise_or() {
        let a = Tensor::new(vec![5.0, 3.0], vec![2]);
        let b = Tensor::new(vec![3.0, 6.0], vec![2]);
        let out = bitwise_or(&a, &b).unwrap();
        assert_eq!(out.data[0] as u32, 5u32 | 3u32);
        assert_eq!(out.data[1] as u32, 3u32 | 6u32);
    }

    #[test]
    fn test_bitwise_xor() {
        let a = Tensor::new(vec![5.0], vec![1]);
        let b = Tensor::new(vec![3.0], vec![1]);
        let out = bitwise_xor(&a, &b).unwrap();
        assert_eq!(out.data[0] as u32, 5u32 ^ 3u32);
    }

    #[test]
    fn test_bitwise_not() {
        // [a0-14] BitwiseNot(0) must yield -1 (two's-complement all-ones,
        // interpreted as a signed value) -- not 4294967295 (u32::MAX), which
        // is also not exactly representable in f32. -1.0 IS exact in f32, so
        // this is also a precision fix, not just a sign fix.
        let x = Tensor::new(vec![0.0], vec![1]);
        let out = bitwise_not(&x);
        assert_eq!(out.data[0], -1.0);
        assert_eq!(out.data[0] as i64, !0i64);
    }

    /// [a0-14] Negative (two's-complement) operands must survive the f32
    /// round-trip: BitwiseAnd(-1, 5) == 5 because -1 is all-ones, so it must
    /// NOT saturate to 0 the way a `v as u32` cast would.
    #[test]
    fn test_bitwise_and_negative_operand() {
        let a = Tensor::new(vec![-1.0], vec![1]);
        let b = Tensor::new(vec![5.0], vec![1]);
        let out = bitwise_and(&a, &b).unwrap();
        assert_eq!(out.data[0], 5.0);
    }

    /// [a0-14] BitwiseOr with a negative operand must also preserve sign
    /// through two's-complement, not saturate the negative side to 0.
    #[test]
    fn test_bitwise_or_negative_operand() {
        let a = Tensor::new(vec![-8.0], vec![1]); // ...11111000
        let b = Tensor::new(vec![5.0], vec![1]); //  ...00000101
        let out = bitwise_or(&a, &b).unwrap();
        assert_eq!(out.data[0] as i64, -8i64 | 5i64);
        assert_eq!(out.data[0], -3.0); // -8 | 5 == -3 in two's complement
    }

    /// [a11-24] Documents (rather than hides) the precision limit inherent to
    /// f32-backed storage: only integer magnitudes up to 2^24 round-trip
    /// through f32 exactly, so BitwiseAnd(2^24+1, all-ones) cannot be expected
    /// to reproduce 16777217 exactly -- the literal itself already rounds to
    /// 16777216 (2^24) at f32-parse time, before `bitwise_and` ever runs. This
    /// is a limit of f32 storage, not of the i64 cast this module uses (the
    /// typed dispatch path is the one to use when exactness beyond 2^24
    /// matters).
    #[test]
    fn test_bitwise_and_documents_f32_precision_limit_beyond_2_pow_24() {
        let a = Tensor::new(vec![16_777_217.0], vec![1]); // 2^24 + 1
        let b = Tensor::new(vec![-1.0], vec![1]); // all-ones: AND is a no-op on the value
        assert_eq!(
            a.data[0], 16_777_216.0,
            "the f32 literal itself already lost the +1"
        );
        let out = bitwise_and(&a, &b).unwrap();
        assert_eq!(out.data[0], 16_777_216.0);
        // Below the 2^24 boundary, values remain exact.
        let a_exact = Tensor::new(vec![16_777_216.0], vec![1]); // 2^24, still exact
        let out_exact = bitwise_and(&a_exact, &b).unwrap();
        assert_eq!(out_exact.data[0], 16_777_216.0);
    }

    #[test]
    fn test_bitwise_broadcast() {
        // Two same-shape tensors — shape is preserved
        let a = Tensor::new(vec![7.0, 7.0], vec![2]);
        let b = Tensor::new(vec![3.0, 5.0], vec![2]);
        let out = bitwise_and(&a, &b).unwrap();
        assert_eq!(out.shape, vec![2]);
    }
}
