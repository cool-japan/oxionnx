//! `Det` operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

/// ONNX `Det` (opset 11+): batched matrix determinant.
///
/// Input `X` has shape `[..., M, M]` (rank >= 2, square in the last two
/// dims); output `Y` has the batch shape `[...]` — a **true rank-0 scalar**
/// (shape `[]`, not the legacy `[1]`) when `X` is exactly 2-D, matching what
/// `numpy.linalg.det` and the ONNX reference implementation both return for a
/// single matrix (`onnx.reference`'s `Det` node test on a 2-D input yields a
/// 0-d array). See the `oxionnx_core::tensor` module docs for this engine's
/// rank-0 contract.
///
/// # Algorithm
///
/// Gaussian elimination with partial pivoting (i.e. an LU decomposition with
/// row pivoting, `PA = LU`) computed in `f64` for numerical stability and
/// narrowed to `f32` only in the final result — see the `det_lu` free
/// function below. This is the standard textbook approach (Golub & Van Loan,
/// *Matrix Computations*, ch. 3) and is what every mainstream reference
/// implementation (LAPACK's `getrf`, NumPy's `linalg.det`) uses internally,
/// so results agree with `numpy.linalg.det` to float precision rather than
/// merely up to a convention difference.
pub struct DetOp;

impl Operator for DetOp {
    fn op_type(&self) -> &str {
        "Det"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let rank = x.ndim();
        if rank < 2 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Det: input must have rank >= 2 ([..., M, M]), got shape {:?}",
                x.shape
            )));
        }
        let m = x.shape[rank - 1];
        let m2 = x.shape[rank - 2];
        if m != m2 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Det: the last two dimensions must be square, got [{m2}, {m}]"
            )));
        }

        let batch_shape = &x.shape[..rank - 2];
        let batch: usize = batch_shape.iter().product();
        let mat_size = m * m;
        if batch.checked_mul(mat_size) != Some(x.data.len()) {
            return Err(OnnxError::ShapeMismatch(format!(
                "Det: input has {} elements but shape {:?} implies {}",
                x.data.len(),
                x.shape,
                batch.saturating_mul(mat_size)
            )));
        }

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mat = &x.data[b * mat_size..(b + 1) * mat_size];
            out.push(det_lu(mat, m));
        }

        if batch_shape.is_empty() {
            // `batch == 1` here (the empty-product identity): a bare `[M, M]`
            // input is a single matrix, and its determinant is an ONNX
            // scalar -- rank 0, not the legacy `[1]` representation.
            let value = out.first().copied().ok_or_else(|| {
                OnnxError::Internal("Det: expected exactly one determinant".into())
            })?;
            Ok(vec![Tensor::rank0(value)])
        } else {
            Ok(vec![Tensor::new(out, batch_shape.to_vec())])
        }
    }
}

/// Determinant of one `m x m` matrix (row-major `mat`) via Gaussian
/// elimination with partial pivoting.
///
/// `m == 0` (the empty matrix) returns `1.0` by convention -- the empty
/// product, matching `numpy.linalg.det(np.empty((0, 0)))`.
fn det_lu(mat: &[f32], m: usize) -> f32 {
    if m == 0 {
        return 1.0;
    }
    // f64 accumulation: partial pivoting bounds the growth factor, but
    // narrow-precision (f32) elimination on anything beyond a handful of
    // rows still loses digits catastrophically. f64 keeps the result at
    // f32-output precision for any matrix size a real ONNX model would
    // realistically carry.
    let mut a: Vec<f64> = mat.iter().map(|&v| v as f64).collect();
    let mut sign = 1.0f64;

    for col in 0..m {
        // Partial pivoting: swap in the largest-magnitude entry at or below
        // the diagonal in this column, both for numerical stability and to
        // avoid dividing by an exact-zero pivot that a naive (unpivoted)
        // elimination would hit on matrices like `[[0,1],[1,0]]`.
        let mut pivot_row = col;
        let mut pivot_val = a[col * m + col].abs();
        for row in (col + 1)..m {
            let v = a[row * m + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val == 0.0 {
            // The whole column (at and below the diagonal) is zero: the
            // matrix is singular.
            return 0.0;
        }
        if pivot_row != col {
            for k in 0..m {
                a.swap(col * m + k, pivot_row * m + k);
            }
            sign = -sign;
        }

        let pivot = a[col * m + col];
        for row in (col + 1)..m {
            let factor = a[row * m + col] / pivot;
            if factor != 0.0 {
                for k in col..m {
                    a[row * m + k] -= factor * a[col * m + k];
                }
            }
        }
    }

    let mut det = sign;
    for i in 0..m {
        det *= a[i * m + i];
    }
    det as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(node: &'a oxionnx_core::Node, x: &'a Tensor) -> OpContext<'a> {
        OpContext {
            node,
            inputs: vec![Some(x)],
            outer_scope: None,
            weights: None,
            registry: None,
        }
    }

    fn dummy_node() -> oxionnx_core::Node {
        oxionnx_core::Node {
            name: "det".into(),
            op: oxionnx_core::OpKind::Det,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attrs: oxionnx_core::Attributes::default(),
        }
    }

    /// `numpy.linalg.det([[1,2,3],[4,5,6],[7,8,10]]) == -3.0` (`onnx.reference`
    /// agrees, and returns shape `()`).
    #[test]
    fn det_3x3_is_rank0_and_matches_numpy() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            vec![3, 3],
        );
        let node = dummy_node();
        let out = DetOp.execute(&ctx(&node, &x)).expect("execute");
        assert_eq!(
            out[0].shape,
            Vec::<usize>::new(),
            "Det of 2-D input is rank-0"
        );
        assert!((out[0].data[0] - (-3.0)).abs() < 1e-4);
    }

    /// Batched `[2, 2, 2]`: `numpy.linalg.det` gives `[-2.0, 13.0]`, shape
    /// `(2,)`.
    #[test]
    fn det_batched_matches_numpy() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0], vec![2, 2, 2]);
        let node = dummy_node();
        let out = DetOp.execute(&ctx(&node, &x)).expect("execute");
        assert_eq!(out[0].shape, vec![2]);
        assert!((out[0].data[0] - (-2.0)).abs() < 1e-4);
        assert!((out[0].data[1] - 13.0).abs() < 1e-4);
    }

    /// A singular matrix (`row2 == 2*row1`) has determinant exactly 0.
    #[test]
    fn det_singular_matrix_is_zero() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 7.0, 1.0, 5.0],
            vec![3, 3],
        );
        let node = dummy_node();
        let out = DetOp.execute(&ctx(&node, &x)).expect("execute");
        assert_eq!(out[0].data[0], 0.0);
    }

    /// A 1x1 "matrix" degenerates to the identity function.
    #[test]
    fn det_1x1() {
        let x = Tensor::new(vec![7.5], vec![1, 1]);
        let node = dummy_node();
        let out = DetOp.execute(&ctx(&node, &x)).expect("execute");
        assert!((out[0].data[0] - 7.5).abs() < 1e-6);
    }

    /// A matrix with a zero pivot on the natural diagonal forces a row swap;
    /// getting partial pivoting wrong (or omitting it) either divides by
    /// zero or returns the wrong sign. `numpy.linalg.det` gives `-13.0`.
    #[test]
    fn det_requires_partial_pivoting() {
        let x = Tensor::new(
            vec![
                0.0, 2.0, 1.0, 0.0, //
                1.0, 1.0, 0.0, 2.0, //
                2.0, 0.0, 1.0, 1.0, //
                0.0, 1.0, 2.0, 1.0,
            ],
            vec![4, 4],
        );
        let node = dummy_node();
        let out = DetOp.execute(&ctx(&node, &x)).expect("execute");
        assert!((out[0].data[0] - (-13.0)).abs() < 1e-3);
    }

    #[test]
    fn det_rejects_non_square() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let node = dummy_node();
        let err = DetOp
            .execute(&ctx(&node, &x))
            .expect_err("non-square must error");
        assert!(format!("{err}").contains("square"), "got: {err}");
    }

    #[test]
    fn det_rejects_rank_below_2() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let node = dummy_node();
        let err = DetOp
            .execute(&ctx(&node, &x))
            .expect_err("rank 1 must error");
        assert!(format!("{err}").contains("rank >= 2"), "got: {err}");
    }
}
