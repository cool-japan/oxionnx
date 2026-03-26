use crate::math::broadcast_to;
use oxionnx_core::Tensor;

// ── Element-wise comparison ops ─────────────────────────────────────────────

fn comparison_binary(
    a: &Tensor,
    b: &Tensor,
    op: impl Fn(f32, f32) -> bool,
) -> Result<Tensor, String> {
    let target = Tensor::broadcast_shape(&a.shape, &b.shape)?;
    let ab = broadcast_to(a, &target);
    let bb = broadcast_to(b, &target);
    let data: Vec<f32> = ab
        .data
        .iter()
        .zip(bb.data.iter())
        .map(|(&x, &y)| if op(x, y) { 1.0 } else { 0.0 })
        .collect();
    Ok(Tensor::new(data, target))
}

/// Element-wise equal: 1.0 where a == b, 0.0 otherwise.
pub fn equal(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    comparison_binary(a, b, |x, y| (x - y).abs() < f32::EPSILON)
}

/// Element-wise greater: 1.0 where a > b, 0.0 otherwise.
pub fn greater(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    comparison_binary(a, b, |x, y| x > y)
}

/// Element-wise greater or equal: 1.0 where a >= b, 0.0 otherwise.
pub fn greater_or_equal(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    comparison_binary(a, b, |x, y| x >= y)
}

/// Element-wise less: 1.0 where a < b, 0.0 otherwise.
pub fn less(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    comparison_binary(a, b, |x, y| x < y)
}

/// Element-wise less or equal: 1.0 where a <= b, 0.0 otherwise.
pub fn less_or_equal(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    comparison_binary(a, b, |x, y| x <= y)
}

// ── Logical ops ─────────────────────────────────────────────────────────────

fn to_bool(v: f32) -> bool {
    v != 0.0
}

/// Element-wise logical AND. Inputs treated as booleans (!=0.0 is true).
pub fn and_op(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    comparison_binary(a, b, |x, y| to_bool(x) && to_bool(y))
}

/// Element-wise logical OR.
pub fn or_op(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    comparison_binary(a, b, |x, y| to_bool(x) || to_bool(y))
}

/// Element-wise logical XOR.
pub fn xor_op(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    comparison_binary(a, b, |x, y| to_bool(x) ^ to_bool(y))
}

/// Element-wise logical NOT. Unary: 1.0 if input == 0.0, else 0.0.
pub fn not_op(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|v| if *v == 0.0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, x.shape.clone())
}

// ── Special comparison ops ──────────────────────────────────────────────────

/// Detect infinities. Returns 1.0 for detected infinities, 0.0 otherwise.
pub fn is_inf(x: &Tensor, detect_neg: bool, detect_pos: bool) -> Tensor {
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|v| {
            if (detect_pos && *v == f32::INFINITY) || (detect_neg && *v == f32::NEG_INFINITY) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Tensor::new(data, x.shape.clone())
}

/// Detect NaN. Returns 1.0 for NaN values, 0.0 otherwise.
pub fn is_nan(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|v| if v.is_nan() { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, x.shape.clone())
}

// ── Index / construction ops ────────────────────────────────────────────────

/// NonZero: returns a 2D tensor of shape `[ndim, num_nonzero]`.
/// Each column is the multi-dimensional index of a non-zero element.
pub fn non_zero(x: &Tensor) -> Tensor {
    let ndim = x.ndim();
    let mut indices: Vec<Vec<f32>> = vec![Vec::new(); ndim];

    for (flat_idx, val) in x.data.iter().enumerate() {
        if *val != 0.0 {
            let mut remaining = flat_idx;
            for d in (0..ndim).rev() {
                let dim_size = x.shape[d];
                indices[d].push((remaining % dim_size) as f32);
                remaining /= dim_size;
            }
        }
    }

    let num_nonzero = if ndim > 0 { indices[0].len() } else { 0 };
    if num_nonzero == 0 {
        return Tensor::new(vec![], vec![ndim, 0]);
    }
    let data: Vec<f32> = indices.into_iter().flatten().collect();
    Tensor::new(data, vec![ndim, num_nonzero])
}

/// ConstantOfShape: create a tensor of the given shape, filled with `value`.
pub fn constant_of_shape(shape: &[usize], value: f32) -> Tensor {
    let numel: usize = shape.iter().product();
    Tensor::new(vec![value; numel], shape.to_vec())
}

/// EyeLike: identity-like matrix with diagonal offset `k`.
/// Shape must be 2D `[M, N]`.
pub fn eye_like(shape: &[usize], k: i64) -> Result<Tensor, String> {
    if shape.len() != 2 {
        return Err(format!("eye_like: expected 2D shape, got {}D", shape.len()));
    }
    let m = shape[0];
    let n = shape[1];
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        let j = i as i64 + k;
        if j >= 0 && (j as usize) < n {
            data[i * n + j as usize] = 1.0;
        }
    }
    Ok(Tensor::new(data, shape.to_vec()))
}

/// Trilu: extract the upper or lower triangular part of a matrix.
/// Works on the last 2 dimensions; batch dimensions are preserved.
pub fn trilu(x: &Tensor, upper: bool, k: i64) -> Result<Tensor, String> {
    let ndim = x.ndim();
    if ndim < 2 {
        return Err(format!("trilu: expected at least 2D tensor, got {}D", ndim));
    }
    let rows = x.shape[ndim - 2];
    let cols = x.shape[ndim - 1];
    let batch: usize = x.shape[..ndim - 2].iter().product::<usize>().max(1);
    let mat_size = rows * cols;
    let mut data = vec![0.0f32; x.data.len()];

    for b in 0..batch {
        let offset = b * mat_size;
        for i in 0..rows {
            for j in 0..cols {
                let keep = if upper {
                    j as i64 >= i as i64 + k
                } else {
                    j as i64 <= i as i64 + k
                };
                if keep {
                    data[offset + i * cols + j] = x.data[offset + i * cols + j];
                }
            }
        }
    }
    Ok(Tensor::new(data, x.shape.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![1.0, 0.0, 3.0], vec![3]);
        let r = equal(&a, &b).expect("equal failed");
        assert_eq!(r.data, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_greater() {
        let a = Tensor::new(vec![3.0, 1.0, 2.0], vec![3]);
        let b = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]);
        let r = greater(&a, &b).expect("greater failed");
        assert_eq!(r.data, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_greater_or_equal() {
        let a = Tensor::new(vec![3.0, 1.0, 2.0], vec![3]);
        let b = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]);
        let r = greater_or_equal(&a, &b).expect("greater_or_equal failed");
        assert_eq!(r.data, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_less() {
        let a = Tensor::new(vec![3.0, 1.0, 2.0], vec![3]);
        let b = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]);
        let r = less(&a, &b).expect("less failed");
        assert_eq!(r.data, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_less_or_equal() {
        let a = Tensor::new(vec![3.0, 1.0, 2.0], vec![3]);
        let b = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]);
        let r = less_or_equal(&a, &b).expect("less_or_equal failed");
        assert_eq!(r.data, vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_and_op() {
        let a = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
        let b = Tensor::new(vec![1.0, 1.0, 0.0], vec![3]);
        let r = and_op(&a, &b).expect("and_op failed");
        assert_eq!(r.data, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_or_op() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0], vec![3]);
        let b = Tensor::new(vec![0.0, 0.0, 1.0], vec![3]);
        let r = or_op(&a, &b).expect("or_op failed");
        assert_eq!(r.data, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_xor_op() {
        let a = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
        let b = Tensor::new(vec![1.0, 1.0, 0.0, 0.0], vec![4]);
        let r = xor_op(&a, &b).expect("xor_op failed");
        assert_eq!(r.data, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_not_op() {
        let a = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
        let r = not_op(&a);
        assert_eq!(r.data, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_is_inf() {
        let t = Tensor::new(vec![1.0, f32::INFINITY, f32::NEG_INFINITY, 0.0], vec![4]);
        let r = is_inf(&t, true, true);
        assert_eq!(r.data, vec![0.0, 1.0, 1.0, 0.0]);

        // Detect only positive infinity
        let r_pos = is_inf(&t, false, true);
        assert_eq!(r_pos.data, vec![0.0, 1.0, 0.0, 0.0]);

        // Detect only negative infinity
        let r_neg = is_inf(&t, true, false);
        assert_eq!(r_neg.data, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_is_nan() {
        let t = Tensor::new(vec![1.0, f32::NAN, 0.0], vec![3]);
        let r = is_nan(&t);
        assert_eq!(r.data[0], 0.0);
        assert_eq!(r.data[1], 1.0);
        assert_eq!(r.data[2], 0.0);
    }

    #[test]
    fn test_non_zero() {
        let t = Tensor::new(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0], vec![6]);
        let r = non_zero(&t);
        assert_eq!(r.shape, vec![1, 3]);
        assert_eq!(r.data, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_non_zero_2d() {
        let t = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let r = non_zero(&t);
        assert_eq!(r.shape, vec![2, 2]); // 2 dims, 2 nonzero elements
                                         // Element (0,0)=1.0 and (1,1)=2.0
        assert_eq!(r.data, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_non_zero_empty() {
        let t = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]);
        let r = non_zero(&t);
        assert_eq!(r.shape, vec![1, 0]);
        assert!(r.data.is_empty());
    }

    #[test]
    fn test_constant_of_shape() {
        let r = constant_of_shape(&[2, 3], 5.0);
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(r.data, vec![5.0; 6]);
    }

    #[test]
    fn test_eye_like_identity() {
        let r = eye_like(&[3, 3], 0).expect("eye_like failed");
        assert_eq!(r.shape, vec![3, 3]);
        assert_eq!(r.data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_eye_like_offset() {
        let r = eye_like(&[3, 4], 1).expect("eye_like with offset failed");
        assert_eq!(r.shape, vec![3, 4]);
        // k=1 means diagonal shifted right by 1
        assert_eq!(
            r.data,
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_eye_like_negative_offset() {
        let r = eye_like(&[3, 3], -1).expect("eye_like negative offset failed");
        assert_eq!(r.data, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_trilu_upper() {
        let t = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        let r = trilu(&t, true, 0).expect("trilu upper failed");
        assert_eq!(r.data, vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn test_trilu_lower() {
        let t = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        let r = trilu(&t, false, 0).expect("trilu lower failed");
        assert_eq!(r.data, vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_trilu_upper_with_k() {
        let t = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        let r = trilu(&t, true, 1).expect("trilu upper k=1 failed");
        // Upper triangle with k=1: keep j >= i+1
        assert_eq!(r.data, vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_trilu_batched() {
        let t = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                70.0, 80.0, 90.0,
            ],
            vec![2, 3, 3],
        );
        let r = trilu(&t, true, 0).expect("trilu batched failed");
        assert_eq!(r.shape, vec![2, 3, 3]);
        // First batch
        assert_eq!(
            &r.data[0..9],
            &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
        );
        // Second batch
        assert_eq!(
            &r.data[9..18],
            &[10.0, 20.0, 30.0, 0.0, 50.0, 60.0, 0.0, 0.0, 90.0]
        );
    }

    #[test]
    fn test_comparison_broadcast() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![2.0], vec![1]);
        let r = less(&a, &b).expect("less broadcast failed");
        assert_eq!(r.data, vec![1.0, 0.0, 0.0]);
    }
}
