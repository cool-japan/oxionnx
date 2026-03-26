//! Property-based tests for tensor operations using proptest.

use oxionnx_core::Tensor;
use proptest::prelude::*;

/// Strategy to generate a tensor with random shape and data.
fn arb_tensor(max_dims: usize, max_dim_size: usize) -> impl Strategy<Value = Tensor> {
    prop::collection::vec(1..=max_dim_size, 1..=max_dims).prop_flat_map(|shape| {
        let n: usize = shape.iter().product();
        prop::collection::vec(-100.0f32..100.0f32, n)
            .prop_map(move |data| Tensor::new(data, shape.clone()))
    })
}

/// Strategy for two tensors with the same shape.
fn arb_same_shape_pair(
    max_dims: usize,
    max_size: usize,
) -> impl Strategy<Value = (Tensor, Tensor)> {
    prop::collection::vec(1..=max_size, 1..=max_dims).prop_flat_map(|shape| {
        let n: usize = shape.iter().product();
        let s = shape.clone();
        (
            prop::collection::vec(-100.0f32..100.0f32, n)
                .prop_map(move |data| Tensor::new(data, s.clone())),
            {
                let s2 = shape.clone();
                prop::collection::vec(-100.0f32..100.0f32, n)
                    .prop_map(move |data| Tensor::new(data, s2.clone()))
            },
        )
    })
}

proptest! {
    // Add is commutative: a + b == b + a
    #[test]
    fn test_add_commutative((a, b) in arb_same_shape_pair(3, 10)) {
        let ab = oxionnx_ops::math::add(&a, &b).map_err(TestCaseError::fail)?;
        let ba = oxionnx_ops::math::add(&b, &a).map_err(TestCaseError::fail)?;
        for (x, y) in ab.data.iter().zip(ba.data.iter()) {
            prop_assert!((x - y).abs() < 1e-6);
        }
    }

    // Mul is commutative: a * b == b * a
    #[test]
    fn test_mul_commutative((a, b) in arb_same_shape_pair(3, 10)) {
        let ab = oxionnx_ops::math::mul(&a, &b).map_err(TestCaseError::fail)?;
        let ba = oxionnx_ops::math::mul(&b, &a).map_err(TestCaseError::fail)?;
        for (x, y) in ab.data.iter().zip(ba.data.iter()) {
            prop_assert!((x - y).abs() < 1e-6);
        }
    }

    // Add identity: a + 0 == a
    #[test]
    fn test_add_identity(a in arb_tensor(3, 10)) {
        let zeros = Tensor::zeros(&a.shape);
        let result = oxionnx_ops::math::add(&a, &zeros).map_err(TestCaseError::fail)?;
        for (x, y) in result.data.iter().zip(a.data.iter()) {
            prop_assert!((x - y).abs() < 1e-6);
        }
    }

    // Mul identity: a * 1 == a
    #[test]
    fn test_mul_identity(a in arb_tensor(3, 10)) {
        let ones = Tensor::new(vec![1.0; a.numel()], a.shape.clone());
        let result = oxionnx_ops::math::mul(&a, &ones).map_err(TestCaseError::fail)?;
        for (x, y) in result.data.iter().zip(a.data.iter()) {
            prop_assert!((x - y).abs() < 1e-6);
        }
    }

    // Softmax outputs sum to 1
    #[test]
    fn test_softmax_sum_to_one(data in prop::collection::vec(-10.0f32..10.0f32, 2..50)) {
        let n = data.len();
        let tensor = Tensor::new(data, vec![1, n]);
        let result = oxionnx_ops::nn::softmax(&tensor, -1).map_err(TestCaseError::fail)?;
        let sum: f32 = result.data.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "softmax sum {} != 1.0", sum);
    }

    // Softmax outputs are non-negative
    #[test]
    fn test_softmax_non_negative(data in prop::collection::vec(-100.0f32..100.0f32, 2..50)) {
        let n = data.len();
        let tensor = Tensor::new(data, vec![1, n]);
        let result = oxionnx_ops::nn::softmax(&tensor, -1).map_err(TestCaseError::fail)?;
        for &v in &result.data {
            prop_assert!(v >= 0.0, "softmax output {} < 0", v);
        }
    }

    // Relu is idempotent: relu(relu(x)) == relu(x)
    #[test]
    fn test_relu_idempotent(a in arb_tensor(3, 10)) {
        let r1 = oxionnx_ops::nn::relu(&a);
        let r2 = oxionnx_ops::nn::relu(&r1);
        prop_assert_eq!(&r1.data, &r2.data);
    }

    // Broadcast shape is symmetric: if both succeed, results match; if one fails, both fail
    #[test]
    fn test_broadcast_symmetric(
        a_shape in prop::collection::vec(1..=5usize, 1..=4),
        b_shape in prop::collection::vec(1..=5usize, 1..=4),
    ) {
        let ab = Tensor::broadcast_shape(&a_shape, &b_shape);
        let ba = Tensor::broadcast_shape(&b_shape, &a_shape);
        match (&ab, &ba) {
            (Ok(ab_shape), Ok(ba_shape)) => prop_assert_eq!(ab_shape, ba_shape),
            (Err(_), Err(_)) => {} // both fail => symmetric
            _ => prop_assert!(false, "broadcast symmetry violated: ab={:?}, ba={:?}", ab, ba),
        }
    }

    // NCHW -> NHWC -> NCHW roundtrip
    #[test]
    fn test_layout_roundtrip(
        n in 1..=3usize, c in 1..=4usize,
        h in 1..=8usize, w in 1..=8usize,
    ) {
        let size = n * c * h * w;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::new(data.clone(), vec![n, c, h, w]);
        let nhwc = oxionnx_core::tensor::nchw_to_nhwc(&tensor).map_err(TestCaseError::fail)?;
        let back = oxionnx_core::tensor::nhwc_to_nchw(&nhwc).map_err(TestCaseError::fail)?;
        prop_assert_eq!(&back.data, &data);
        prop_assert_eq!(&back.shape, &vec![n, c, h, w]);
    }
}
