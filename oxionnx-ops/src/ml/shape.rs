//! Shared input-shape resolution for the ONNX-ML operator family.
//!
//! Every ONNX-ML operator that consumes a feature matrix (`Scaler`,
//! `Normalizer`, `LinearClassifier`, `LinearRegressor`, `TreeEnsemble*`,
//! `SVM*`) accepts either
//!
//! * a 2-D `[N, C]` tensor — `N` samples with `C` features each, or
//! * a 1-D `[C]` tensor — **one** sample with `C` features.
//!
//! The 1-D form must never be read as `C` samples of a single feature: that
//! mis-shaping silently produces `C` independent predictions instead of one.
//! Ranks above 2 are flattened to `[shape[0], prod(shape[1..])]`, matching
//! onnxruntime's `x_shape.SizeFromDimension(1)` stride convention.

use oxionnx_core::{OnnxError, Tensor};

/// Resolve an ONNX-ML operator input into `(n_samples, n_features)`.
///
/// The returned pair is guaranteed to satisfy
/// `n_samples * n_features <= x.data.len()`, so callers may slice rows as
/// `x.data[i * n_features..(i + 1) * n_features]` without further checks.
///
/// # Errors
///
/// Returns [`OnnxError::ShapeMismatch`] when the shape overflows `usize` or
/// when the tensor holds fewer elements than its shape claims.
pub(crate) fn batch_dims(x: &Tensor, op: &str) -> Result<(usize, usize), OnnxError> {
    let (n, features) = match x.shape.len() {
        // A scalar is a single sample with a single feature.
        0 => (1usize, 1usize),
        // `[C]` is ONE sample with C features (ONNX-ML convention).
        1 => (1usize, x.shape[0]),
        _ => {
            let mut features = 1usize;
            for &dim in &x.shape[1..] {
                features = features.checked_mul(dim).ok_or_else(|| {
                    OnnxError::ShapeMismatch(format!("{op}: feature count overflows usize"))
                })?;
            }
            (x.shape[0], features)
        }
    };

    let needed = n.checked_mul(features).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!("{op}: input element count overflows usize"))
    })?;
    if x.data.len() < needed {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: input holds {} elements but shape {:?} requires {needed}",
            x.data.len(),
            x.shape
        )));
    }

    Ok((n, features))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_dim_input_is_a_single_sample() {
        let x = Tensor::new(vec![3.0, 4.0], vec![2]);
        assert_eq!(batch_dims(&x, "T").expect("rank-1"), (1, 2));
    }

    #[test]
    fn two_dim_input_keeps_batch() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        assert_eq!(batch_dims(&x, "T").expect("rank-2"), (3, 2));
    }

    #[test]
    fn higher_rank_flattens_trailing_dims() {
        let x = Tensor::new(vec![0.0; 12], vec![2, 3, 2]);
        assert_eq!(batch_dims(&x, "T").expect("rank-3"), (2, 6));
    }

    #[test]
    fn scalar_is_one_by_one() {
        let x = Tensor::new(vec![7.0], vec![]);
        assert_eq!(batch_dims(&x, "T").expect("rank-0"), (1, 1));
    }

    #[test]
    fn truncated_data_is_rejected() {
        let x = Tensor {
            data: vec![1.0, 2.0],
            shape: vec![4, 3],
        };
        assert!(batch_dims(&x, "T").is_err());
    }
}
