//! Tests for ndarray interoperability (feature = "ndarray").
//!
//! All tests are conditional on the `ndarray` feature being enabled.

#[cfg(feature = "ndarray")]
mod ndarray_tests {
    use ndarray::{array, Array2, ArrayD, IxDyn};
    use oxionnx::Tensor;

    // ── from_ndarray ─────────────────────────────────────────────────────────

    #[test]
    fn test_from_ndarray_2d() {
        let arr: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let tensor = Tensor::from_ndarray(arr);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_from_ndarray_1d() {
        let arr = ndarray::Array1::from_vec(vec![10.0f32, 20.0, 30.0]);
        let tensor = Tensor::from_ndarray(arr);
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.data, vec![10.0, 20.0, 30.0]);
    }

    // ── from_ndarray_view ────────────────────────────────────────────────────

    #[test]
    fn test_from_ndarray_view() {
        let arr: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let view = arr.view();
        let tensor = Tensor::from_ndarray_view(view);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── to_ndarray ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_ndarray_roundtrip() {
        let original = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let arr: ArrayD<f32> = original.to_ndarray();
        assert_eq!(arr.shape(), &[2, 3]);
        // Row-major iteration order should match the flat data.
        let flat: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ── as_ndarray_view ──────────────────────────────────────────────────────

    #[test]
    fn test_as_ndarray_view() {
        let tensor = Tensor::new(vec![7.0, 8.0, 9.0], vec![3]);
        let view = tensor.as_ndarray_view().expect("valid shape");
        assert_eq!(view.shape(), &[3]);
        assert_eq!(view[[0]], 7.0);
        assert_eq!(view[[2]], 9.0);
    }

    // ── try_extract_tensor ───────────────────────────────────────────────────

    #[test]
    fn test_try_extract_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let result = tensor.try_extract_tensor::<f32>();
        assert!(result.is_ok());
        let (shape, data) = result.unwrap();
        assert_eq!(shape, &[1, 3]);
        assert_eq!(data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_try_extract_tensor_phantom_type_ignored() {
        // T is a phantom — calling with different types should give the same result.
        let tensor = Tensor::zeros(&[4]);
        let r1 = tensor.try_extract_tensor::<f32>();
        let r2 = tensor.try_extract_tensor::<u8>();
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert_eq!(r1.unwrap().0, r2.unwrap().0);
    }

    // ── try_extract_array ────────────────────────────────────────────────────

    #[test]
    fn test_try_extract_array() {
        let tensor = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let result = tensor.try_extract_array::<f32>();
        assert!(result.is_ok());
        let view = result.unwrap();
        assert_eq!(view.shape(), &[2, 2]);
        // IxDyn indexing
        assert_eq!(view[IxDyn(&[0, 0])], 10.0);
        assert_eq!(view[IxDyn(&[1, 1])], 40.0);
    }

    // ── roundtrip: Tensor → ndarray → Tensor ────────────────────────────────

    #[test]
    fn test_full_roundtrip() {
        let original = Tensor::new((0..12).map(|x| x as f32).collect(), vec![3, 4]);
        let arr = original.to_ndarray();
        let recovered = Tensor::from_ndarray(arr);
        assert_eq!(original.shape, recovered.shape);
        assert_eq!(original.data, recovered.data);
    }
}
