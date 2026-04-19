//! Per-dtype dispatch for element-wise operations on TypedTensor.
//!
//! Binary operations use two paths:
//!  - **Integer path**: when both operands have the same integer dtype, arithmetic
//!    is performed natively via `i128`/`u128` promotion to avoid f32 precision loss
//!    (f32 can only represent integers exactly up to 2^24).
//!  - **Float path**: all other cases (float/float, mixed int+float, mixed dtypes)
//!    use the existing `to_f32_vec()` path with automatic type promotion.

use oxionnx_core::dtype::{promote, DType, TensorStorage, TypedTensor};
use oxionnx_core::OnnxError;

// ---------------------------------------------------------------------------
// Internal helpers — generic broadcast
// ---------------------------------------------------------------------------

/// Broadcast a flat slice from `src_shape` to `out_shape`, returning a new Vec.
///
/// This is a dtype-generic equivalent of `crate::math::broadcast_to` for use
/// in the native integer path.
fn broadcast_vec<T: Copy>(data: &[T], src_shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    if src_shape == out_shape {
        return data.to_vec();
    }

    let n_out: usize = out_shape.iter().product();
    let n = out_shape.len();
    let pad = n - src_shape.len();

    // Pad src_shape on the left with 1s so both shapes have the same rank.
    let padded: Vec<usize> = (0..pad)
        .map(|_| 1)
        .chain(src_shape.iter().copied())
        .collect();

    // Row-major strides for the padded source shape; broadcast dims get stride 0.
    let mut src_strides = vec![0usize; n];
    let mut stride = 1usize;
    for i in (0..n).rev() {
        if padded[i] == 1 && out_shape[i] != 1 {
            src_strides[i] = 0;
        } else {
            src_strides[i] = stride;
        }
        stride *= padded[i];
    }

    // Row-major strides for the output shape.
    let mut out_strides = vec![0usize; n];
    let mut s = 1usize;
    for i in (0..n).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }

    // Fill output by back-mapping each output index to a source index.
    let mut result = Vec::with_capacity(n_out);
    for out_idx in 0..n_out {
        let mut rem = out_idx;
        let mut src_idx = 0usize;
        for i in 0..n {
            let coord = rem / out_strides[i];
            rem %= out_strides[i];
            src_idx += coord * src_strides[i];
        }
        result.push(data[src_idx]);
    }
    result
}

// ---------------------------------------------------------------------------
// Native integer binary operation
// ---------------------------------------------------------------------------

/// Execute a binary element-wise operation on two tensors that share the same
/// integer dtype, using native i128/u128 arithmetic to preserve precision.
///
/// `int_op` receives two `i128` values (both signed and unsigned integers are
/// promoted — unsigned is widened to i128 as well; the result is narrowed back
/// via `as` truncation which gives two's-complement wrapping at the storage
/// width for all signed types and natural modular wrapping for unsigned).
fn typed_binary_op_int<F>(
    a: &TypedTensor,
    b: &TypedTensor,
    int_op: F,
) -> Result<TypedTensor, OnnxError>
where
    F: Fn(i128, i128) -> Result<i128, OnnxError>,
{
    let out_shape = oxionnx_core::Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(OnnxError::ShapeMismatch)?;

    // Dispatch on the common dtype and apply the operation elementwise.
    macro_rules! int_op_typed {
        ($StorageVariant:ident, $prim:ty, $ResultVariant:ident, $out_ty:ty) => {{
            let av = match &a.storage {
                TensorStorage::$StorageVariant(v) => v.as_slice(),
                _ => {
                    return Err(OnnxError::DTypeMismatch(format!(
                        "expected {} storage",
                        a.dtype()
                    )))
                }
            };
            let bv = match &b.storage {
                TensorStorage::$StorageVariant(v) => v.as_slice(),
                _ => {
                    return Err(OnnxError::DTypeMismatch(format!(
                        "expected {} storage",
                        b.dtype()
                    )))
                }
            };
            let a_bc = broadcast_vec(av, &a.shape, &out_shape);
            let b_bc = broadcast_vec(bv, &b.shape, &out_shape);
            let mut result: Vec<$out_ty> = Vec::with_capacity(a_bc.len());
            for (x, y) in a_bc.iter().zip(b_bc.iter()) {
                let r = int_op(*x as i128, *y as i128)?;
                result.push(r as $out_ty);
            }
            Ok(TypedTensor::new(
                TensorStorage::$ResultVariant(result),
                out_shape,
            ))
        }};
    }

    match a.dtype() {
        DType::I8 => int_op_typed!(I8, i8, I8, i8),
        DType::I16 => int_op_typed!(I16, i16, I16, i16),
        DType::I32 => int_op_typed!(I32, i32, I32, i32),
        DType::I64 => int_op_typed!(I64, i64, I64, i64),
        DType::U8 => int_op_typed!(U8, u8, U8, u8),
        DType::U16 => int_op_typed!(U16, u16, U16, u16),
        DType::U32 => int_op_typed!(U32, u32, U32, u32),
        DType::U64 => int_op_typed!(U64, u64, U64, u64),
        other => Err(OnnxError::DTypeMismatch(format!(
            "typed_binary_op_int called with non-integer dtype {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Combined binary op (integer-native + float fallback)
// ---------------------------------------------------------------------------

/// Perform a binary element-wise operation with two dispatch paths:
///
/// 1. When both operands share the **same integer dtype** (I8/I16/I32/I64/U8/U16/U32/U64),
///    `int_op` is applied natively via i128 arithmetic — no f32 conversion.
/// 2. All other cases (both float, mixed dtypes, Bool) fall through to the
///    `float_op` path via `to_f32_vec()` with automatic type promotion.
fn typed_binary_op<F, G>(
    a: &TypedTensor,
    b: &TypedTensor,
    int_op: F,
    float_op: G,
) -> Result<TypedTensor, OnnxError>
where
    F: Fn(i128, i128) -> Result<i128, OnnxError>,
    G: Fn(f32, f32) -> f32,
{
    // When both tensors have the same integer dtype, use the native path.
    if a.dtype() == b.dtype() && a.dtype().is_integer() {
        return typed_binary_op_int(a, b, int_op);
    }

    // Float path (also covers mixed int+float, different dtypes, Bool).
    let out_shape = oxionnx_core::Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(OnnxError::ShapeMismatch)?;

    let result_dtype = promote(a.dtype(), b.dtype());

    let a_f32 = oxionnx_core::Tensor::new(a.storage.to_f32_vec(), a.shape.clone());
    let b_f32 = oxionnx_core::Tensor::new(b.storage.to_f32_vec(), b.shape.clone());

    let a_bc = crate::math::broadcast_to(&a_f32, &out_shape);
    let b_bc = crate::math::broadcast_to(&b_f32, &out_shape);

    let data: Vec<f32> = a_bc
        .data
        .iter()
        .zip(b_bc.data.iter())
        .map(|(&x, &y)| float_op(x, y))
        .collect();

    let result_f32 = TypedTensor::new(TensorStorage::F32(data), out_shape);
    if result_dtype != DType::F32 {
        Ok(result_f32.cast(result_dtype))
    } else {
        Ok(result_f32)
    }
}

// ---------------------------------------------------------------------------
// Unary op helper (unchanged)
// ---------------------------------------------------------------------------

/// Perform a unary element-wise operation through f32, preserving the input dtype.
fn typed_unary_op(x: &TypedTensor, op: impl Fn(f32) -> f32) -> TypedTensor {
    let f32_data = x.storage.to_f32_vec();
    let result: Vec<f32> = f32_data.iter().map(|&v| op(v)).collect();
    let result_tensor = TypedTensor::new(TensorStorage::F32(result), x.shape.clone());
    if x.dtype() != DType::F32 {
        result_tensor.cast(x.dtype())
    } else {
        result_tensor
    }
}

// ---------------------------------------------------------------------------
// DType helper — is_integer extends DType without modifying core
// ---------------------------------------------------------------------------

trait DTypeExt {
    fn is_integer(&self) -> bool;
}

impl DTypeExt for DType {
    fn is_integer(&self) -> bool {
        matches!(
            self,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
        )
    }
}

// ---------------------------------------------------------------------------
// Binary ops — public API
// ---------------------------------------------------------------------------

/// Element-wise addition with automatic type promotion.
/// Integer operands of the same dtype use native arithmetic (no f32 loss).
pub fn typed_add(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    typed_binary_op(a, b, |x, y| Ok(x.wrapping_add(y)), |x, y| x + y)
}

/// Element-wise subtraction with automatic type promotion.
/// Integer operands of the same dtype use native arithmetic (no f32 loss).
pub fn typed_sub(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    typed_binary_op(a, b, |x, y| Ok(x.wrapping_sub(y)), |x, y| x - y)
}

/// Element-wise multiplication with automatic type promotion.
/// Integer operands of the same dtype use native arithmetic (no f32 loss).
pub fn typed_mul(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    typed_binary_op(a, b, |x, y| Ok(x.wrapping_mul(y)), |x, y| x * y)
}

/// Element-wise division with automatic type promotion.
/// Integer operands of the same dtype use native arithmetic (no f32 loss).
/// Returns `OnnxError::Arithmetic` on integer division by zero.
pub fn typed_div(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    typed_binary_op(
        a,
        b,
        |x, y| {
            if y == 0 {
                Err(OnnxError::Arithmetic(
                    "integer division by zero".to_string(),
                ))
            } else {
                Ok(x / y)
            }
        },
        |x, y| x / y,
    )
}

// ---------------------------------------------------------------------------
// Unary ops
// ---------------------------------------------------------------------------

/// ReLU activation: max(0, x).
pub fn typed_relu(x: &TypedTensor) -> TypedTensor {
    typed_unary_op(x, |v| v.max(0.0))
}

/// Sigmoid activation: 1 / (1 + exp(-x)).
pub fn typed_sigmoid(x: &TypedTensor) -> TypedTensor {
    typed_unary_op(x, |v| 1.0 / (1.0 + (-v).exp()))
}

/// Tanh activation.
pub fn typed_tanh(x: &TypedTensor) -> TypedTensor {
    typed_unary_op(x, |v| v.tanh())
}

/// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
pub fn typed_gelu(x: &TypedTensor) -> TypedTensor {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEF: f32 = 0.044_715;
    typed_unary_op(x, |v| {
        let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

/// Element-wise exponential.
pub fn typed_exp(x: &TypedTensor) -> TypedTensor {
    typed_unary_op(x, |v| v.exp())
}

/// Element-wise square root.
pub fn typed_sqrt(x: &TypedTensor) -> TypedTensor {
    typed_unary_op(x, |v| v.sqrt())
}

/// Element-wise negation.
pub fn typed_neg(x: &TypedTensor) -> TypedTensor {
    typed_unary_op(x, |v| -v)
}

// ---------------------------------------------------------------------------
// MatMul
// ---------------------------------------------------------------------------

/// Matrix multiplication with mixed-precision support.
///
/// Both inputs are converted to f32 for accumulation (numerical stability),
/// and the result dtype follows the promotion rules of the two inputs.
pub fn typed_matmul(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    // Always accumulate in f32 for numerical stability
    let a_f32 = oxionnx_core::Tensor::new(a.storage.to_f32_vec(), a.shape.clone());
    let b_f32 = oxionnx_core::Tensor::new(b.storage.to_f32_vec(), b.shape.clone());
    let result_f32 = crate::math::matmul(&a_f32, &b_f32).map_err(OnnxError::ShapeMismatch)?;

    // Result dtype follows promotion rules, but stays F32 if either input was float
    let result_dtype = promote(a.dtype(), b.dtype());
    let typed = TypedTensor::from_tensor(&result_f32);
    if result_dtype != DType::F32 {
        Ok(typed.cast(result_dtype))
    } else {
        Ok(typed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> TypedTensor {
        TypedTensor::new(TensorStorage::F32(data), shape)
    }

    fn make_i32(data: Vec<i32>, shape: Vec<usize>) -> TypedTensor {
        TypedTensor::new(TensorStorage::I32(data), shape)
    }

    fn make_f16(data: Vec<f32>, shape: Vec<usize>) -> TypedTensor {
        let bits: Vec<u16> = data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        TypedTensor::new(TensorStorage::F16(bits), shape)
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // -----------------------------------------------------------------------
    // Existing tests (unchanged)
    // -----------------------------------------------------------------------

    #[test]
    fn test_typed_add_same_dtype() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![4.0, 5.0, 6.0], vec![3]);
        let c = typed_add(&a, &b).expect("add failed");
        assert_eq!(c.dtype(), DType::F32);
        let vals = c.storage.to_f32_vec();
        assert_eq!(vals, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_typed_add_promotion() {
        // I32 + F32 should promote to F32 (falls through to float path)
        let a = make_i32(vec![1, 2, 3], vec![3]);
        let b = make_f32(vec![0.5, 1.5, 2.5], vec![3]);
        let c = typed_add(&a, &b).expect("add failed");
        assert_eq!(c.dtype(), DType::F32);
        let vals = c.storage.to_f32_vec();
        assert!(approx_eq(vals[0], 1.5, 1e-5));
        assert!(approx_eq(vals[1], 3.5, 1e-5));
        assert!(approx_eq(vals[2], 5.5, 1e-5));
    }

    #[test]
    fn test_typed_add_broadcast() {
        // [3,1] + [1,4] -> [3,4]
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = make_f32(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]);
        let c = typed_add(&a, &b).expect("add failed");
        assert_eq!(c.shape, vec![3, 4]);
        let vals = c.storage.to_f32_vec();
        // Row 0: 1+10, 1+20, 1+30, 1+40
        assert_eq!(vals[0], 11.0);
        assert_eq!(vals[1], 21.0);
        assert_eq!(vals[2], 31.0);
        assert_eq!(vals[3], 41.0);
        // Row 1: 2+10, 2+20, 2+30, 2+40
        assert_eq!(vals[4], 12.0);
        assert_eq!(vals[5], 22.0);
        assert_eq!(vals[6], 32.0);
        assert_eq!(vals[7], 42.0);
        // Row 2: 3+10, 3+20, 3+30, 3+40
        assert_eq!(vals[8], 13.0);
        assert_eq!(vals[9], 23.0);
        assert_eq!(vals[10], 33.0);
        assert_eq!(vals[11], 43.0);
    }

    #[test]
    fn test_typed_mul_f16() {
        let a = make_f16(vec![2.0, 3.0, 4.0], vec![3]);
        let b = make_f16(vec![0.5, 1.0, 2.0], vec![3]);
        let c = typed_mul(&a, &b).expect("mul failed");
        // F16 + F16 promotes to F16 (same size, first wins)
        // But promote_float_float with equal sizes returns the first => F16
        // Actually promote(F16, F16) returns F16 since they are equal
        assert_eq!(c.dtype(), DType::F16);
        let vals = c.storage.to_f32_vec();
        assert!(approx_eq(vals[0], 1.0, 0.01));
        assert!(approx_eq(vals[1], 3.0, 0.01));
        assert!(approx_eq(vals[2], 8.0, 0.01));
    }

    #[test]
    fn test_typed_relu() {
        let x = make_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let y = typed_relu(&x);
        assert_eq!(y.dtype(), DType::F32);
        let vals = y.storage.to_f32_vec();
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_typed_sigmoid() {
        let x = make_f32(vec![-10.0, 0.0, 10.0], vec![3]);
        let y = typed_sigmoid(&x);
        let vals = y.storage.to_f32_vec();
        // sigmoid(-10) ~ 0, sigmoid(0) = 0.5, sigmoid(10) ~ 1
        assert!(vals[0] < 0.001);
        assert!(approx_eq(vals[1], 0.5, 1e-5));
        assert!(vals[2] > 0.999);
    }

    #[test]
    fn test_typed_matmul() {
        // [2,3] x [3,2] = [2,2]
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = typed_matmul(&a, &b).expect("matmul failed");
        assert_eq!(c.shape, vec![2, 2]);
        let vals = c.storage.to_f32_vec();
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!(approx_eq(vals[0], 58.0, 1e-3));
        assert!(approx_eq(vals[1], 64.0, 1e-3));
        assert!(approx_eq(vals[2], 139.0, 1e-3));
        assert!(approx_eq(vals[3], 154.0, 1e-3));
    }

    #[test]
    fn test_typed_matmul_mixed() {
        // F16 x F32 -> accumulates in F32, result is F32 (promote(F16, F32) = F32)
        let a = make_f16(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = make_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = typed_matmul(&a, &b).expect("matmul failed");
        assert_eq!(c.dtype(), DType::F32);
        let vals = c.storage.to_f32_vec();
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert!(approx_eq(vals[0], 19.0, 0.1));
        assert!(approx_eq(vals[1], 22.0, 0.1));
        assert!(approx_eq(vals[2], 43.0, 0.1));
        assert!(approx_eq(vals[3], 50.0, 0.1));
    }

    #[test]
    fn test_typed_neg() {
        let x = make_f32(vec![1.0, -2.0, 0.0, 3.5], vec![4]);
        let y = typed_neg(&x);
        assert_eq!(y.dtype(), DType::F32);
        let vals = y.storage.to_f32_vec();
        assert_eq!(vals, vec![-1.0, 2.0, 0.0, -3.5]);
    }

    #[test]
    fn test_typed_sqrt() {
        let x = make_f32(vec![0.0, 1.0, 4.0, 9.0, 16.0], vec![5]);
        let y = typed_sqrt(&x);
        assert_eq!(y.dtype(), DType::F32);
        let vals = y.storage.to_f32_vec();
        assert!(approx_eq(vals[0], 0.0, 1e-5));
        assert!(approx_eq(vals[1], 1.0, 1e-5));
        assert!(approx_eq(vals[2], 2.0, 1e-5));
        assert!(approx_eq(vals[3], 3.0, 1e-5));
        assert!(approx_eq(vals[4], 4.0, 1e-5));
    }

    // -----------------------------------------------------------------------
    // New tests — native integer arithmetic
    // -----------------------------------------------------------------------

    /// Values above 2^24 cannot be represented exactly in f32; the native I64
    /// path must preserve them.
    #[test]
    fn typed_add_i64_preserves_2pow40() {
        let x: i64 = 1i64 << 40; // 1_099_511_627_776
        let a = TypedTensor::new(TensorStorage::I64(vec![x]), vec![1]);
        let b = TypedTensor::new(TensorStorage::I64(vec![1i64]), vec![1]);
        let c = typed_add(&a, &b).expect("add failed");
        assert_eq!(c.dtype(), DType::I64);
        match &c.storage {
            TensorStorage::I64(v) => assert_eq!(v[0], x + 1),
            other => panic!("expected I64 storage, got {:?}", other.dtype()),
        }
    }

    /// i32::MAX * 2 overflows; the result must wrap in two's-complement.
    #[test]
    fn typed_mul_i32_wraps_on_overflow() {
        let a = TypedTensor::new(TensorStorage::I32(vec![i32::MAX]), vec![1]);
        let b = TypedTensor::new(TensorStorage::I32(vec![2i32]), vec![1]);
        let c = typed_mul(&a, &b).expect("mul failed");
        assert_eq!(c.dtype(), DType::I32);
        match &c.storage {
            TensorStorage::I32(v) => assert_eq!(v[0], i32::MAX.wrapping_mul(2)),
            other => panic!("expected I32 storage, got {:?}", other.dtype()),
        }
    }

    /// Integer division by zero must return an Err.
    #[test]
    fn typed_div_i64_by_zero_returns_err() {
        let a = TypedTensor::new(TensorStorage::I64(vec![10i64]), vec![1]);
        let b = TypedTensor::new(TensorStorage::I64(vec![0i64]), vec![1]);
        let result = typed_div(&a, &b);
        assert!(result.is_err(), "expected Err for div-by-zero, got Ok");
    }

    /// i8 wraps at 127 → -128.
    #[test]
    fn typed_add_i8_wraps_at_127() {
        let a = TypedTensor::new(TensorStorage::I8(vec![127i8]), vec![1]);
        let b = TypedTensor::new(TensorStorage::I8(vec![1i8]), vec![1]);
        let c = typed_add(&a, &b).expect("add failed");
        assert_eq!(c.dtype(), DType::I8);
        match &c.storage {
            TensorStorage::I8(v) => assert_eq!(v[0], -128i8),
            other => panic!("expected I8 storage, got {:?}", other.dtype()),
        }
    }

    /// f32 addition must still work via the float path.
    #[test]
    fn typed_add_f32_exact_roundtrip() {
        let a = TypedTensor::new(TensorStorage::F32(vec![1.0f32, 2.0]), vec![2]);
        let b = TypedTensor::new(TensorStorage::F32(vec![3.0f32, 4.0]), vec![2]);
        let c = typed_add(&a, &b).expect("add failed");
        assert_eq!(c.dtype(), DType::F32);
        match &c.storage {
            TensorStorage::F32(v) => {
                assert!((v[0] - 4.0f32).abs() < 1e-6);
                assert!((v[1] - 6.0f32).abs() < 1e-6);
            }
            other => panic!("expected F32 storage, got {:?}", other.dtype()),
        }
    }

    /// I32 integer arithmetic stays in I32 (no promotion to float).
    #[test]
    fn typed_add_i32_same_dtype_stays_integer() {
        let a = TypedTensor::new(TensorStorage::I32(vec![100i32, 200]), vec![2]);
        let b = TypedTensor::new(TensorStorage::I32(vec![1i32, 2]), vec![2]);
        let c = typed_add(&a, &b).expect("add failed");
        assert_eq!(c.dtype(), DType::I32, "result must stay I32");
        match &c.storage {
            TensorStorage::I32(v) => {
                assert_eq!(v[0], 101);
                assert_eq!(v[1], 202);
            }
            other => panic!("expected I32 storage, got {:?}", other.dtype()),
        }
    }

    /// Broadcast with integer tensors: [3,1] + [1,4] should produce [3,4] I64.
    #[test]
    fn typed_add_i64_broadcast() {
        let a = TypedTensor::new(TensorStorage::I64(vec![1i64, 2, 3]), vec![3, 1]);
        let b = TypedTensor::new(TensorStorage::I64(vec![10i64, 20, 30, 40]), vec![1, 4]);
        let c = typed_add(&a, &b).expect("broadcast add failed");
        assert_eq!(c.shape, vec![3, 4]);
        assert_eq!(c.dtype(), DType::I64);
        match &c.storage {
            TensorStorage::I64(v) => {
                // Row 0 (a=1): 11, 21, 31, 41
                assert_eq!(v[0], 11);
                assert_eq!(v[1], 21);
                assert_eq!(v[2], 31);
                assert_eq!(v[3], 41);
                // Row 1 (a=2): 12, 22, 32, 42
                assert_eq!(v[4], 12);
                assert_eq!(v[5], 22);
                assert_eq!(v[6], 32);
                assert_eq!(v[7], 42);
                // Row 2 (a=3): 13, 23, 33, 43
                assert_eq!(v[8], 13);
                assert_eq!(v[9], 23);
                assert_eq!(v[10], 33);
                assert_eq!(v[11], 43);
            }
            other => panic!("expected I64 storage, got {:?}", other.dtype()),
        }
    }

    /// U8 wrapping: 255u8 + 1 = 0.
    #[test]
    fn typed_add_u8_wraps() {
        let a = TypedTensor::new(TensorStorage::U8(vec![255u8]), vec![1]);
        let b = TypedTensor::new(TensorStorage::U8(vec![1u8]), vec![1]);
        let c = typed_add(&a, &b).expect("add failed");
        assert_eq!(c.dtype(), DType::U8);
        match &c.storage {
            TensorStorage::U8(v) => assert_eq!(v[0], 0u8),
            other => panic!("expected U8 storage, got {:?}", other.dtype()),
        }
    }

    /// I64 subtraction: precision must be maintained above 2^24.
    #[test]
    fn typed_sub_i64_precision() {
        let big: i64 = (1i64 << 40) + 999;
        let a = TypedTensor::new(TensorStorage::I64(vec![big]), vec![1]);
        let b = TypedTensor::new(TensorStorage::I64(vec![999i64]), vec![1]);
        let c = typed_sub(&a, &b).expect("sub failed");
        match &c.storage {
            TensorStorage::I64(v) => assert_eq!(v[0], 1i64 << 40),
            other => panic!("expected I64 storage, got {:?}", other.dtype()),
        }
    }
}
