//! Per-dtype dispatch for element-wise operations on TypedTensor.
//!
//! All binary operations follow automatic type promotion rules from
//! `oxionnx_core::dtype::promote`. Computation is performed in f32
//! (or the promoted type) and the result is cast to the target dtype.

use oxionnx_core::dtype::{promote, DType, TensorStorage, TypedTensor};
use oxionnx_core::OnnxError;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Perform a binary element-wise operation through f32 with automatic promotion.
fn typed_binary_op(
    a: &TypedTensor,
    b: &TypedTensor,
    op: impl Fn(f32, f32) -> f32,
) -> Result<TypedTensor, OnnxError> {
    let out_shape = oxionnx_core::Tensor::broadcast_shape(&a.shape, &b.shape)
        .map_err(OnnxError::ShapeMismatch)?;

    let result_dtype = promote(a.dtype(), b.dtype());

    let a_f32 = oxionnx_core::Tensor::new(a.storage.to_f32_vec(), a.shape.clone());
    let b_f32 = oxionnx_core::Tensor::new(b.storage.to_f32_vec(), b.shape.clone());

    // Broadcast both tensors to the output shape
    let a_bc = crate::math::broadcast_to(&a_f32, &out_shape);
    let b_bc = crate::math::broadcast_to(&b_f32, &out_shape);

    let data: Vec<f32> = a_bc
        .data
        .iter()
        .zip(b_bc.data.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();

    let result_f32 = TypedTensor::new(TensorStorage::F32(data), out_shape);
    if result_dtype != DType::F32 {
        Ok(result_f32.cast(result_dtype))
    } else {
        Ok(result_f32)
    }
}

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
// Binary ops
// ---------------------------------------------------------------------------

/// Element-wise addition with automatic type promotion.
pub fn typed_add(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    typed_binary_op(a, b, |x, y| x + y)
}

/// Element-wise subtraction with automatic type promotion.
pub fn typed_sub(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    typed_binary_op(a, b, |x, y| x - y)
}

/// Element-wise multiplication with automatic type promotion.
pub fn typed_mul(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    typed_binary_op(a, b, |x, y| x * y)
}

/// Element-wise division with automatic type promotion.
pub fn typed_div(a: &TypedTensor, b: &TypedTensor) -> Result<TypedTensor, OnnxError> {
    typed_binary_op(a, b, |x, y| x / y)
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
        // I32 + F32 should promote to F32
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
}
