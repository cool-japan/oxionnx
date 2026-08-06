use oxionnx_core::Tensor;

/// Compress: select elements based on boolean condition along an axis.
/// If axis is None, flatten input first.
pub fn compress(input: &Tensor, condition: &Tensor, axis: Option<i64>) -> Result<Tensor, String> {
    if let Some(ax_val) = axis {
        let ndim = input.ndim();
        let ax = if ax_val < 0 {
            (ndim as i64 + ax_val) as usize
        } else {
            ax_val as usize
        };
        if ax >= ndim {
            return Err(format!(
                "compress: axis {ax} out of range for {ndim}D tensor"
            ));
        }

        // Count true values in condition
        let true_count = condition.data.iter().filter(|v| **v != 0.0).count();

        let mut out_shape = input.shape.clone();
        out_shape[ax] = true_count;

        // NOTE: no `.max(1)` here — an empty slice's `.product()` is already 1 (the correct
        // vacuous case), so clamping would instead corrupt a genuine zero-size outer/inner
        // dimension (e.g. shape [0,3,4] axis=1, or [2,3,0] axis=0) into a phantom size-1 dim,
        // producing a wrong output shape/data length instead of the correct empty result.
        let outer: usize = input.shape[..ax].iter().product::<usize>();
        let axis_size = input.shape[ax];
        let inner: usize = input.shape[ax + 1..].iter().product::<usize>();

        let mut data = Vec::with_capacity(out_shape.iter().product());

        for o in 0..outer {
            for a in 0..axis_size {
                if a < condition.data.len() && condition.data[a] != 0.0 {
                    for i in 0..inner {
                        data.push(input.data[(o * axis_size + a) * inner + i]);
                    }
                }
            }
        }

        Ok(Tensor::new(data, out_shape))
    } else {
        // Flatten and select
        let mut data = Vec::new();
        for (i, v) in input.data.iter().enumerate() {
            if i < condition.data.len() && condition.data[i] != 0.0 {
                data.push(*v);
            }
        }
        let len = data.len();
        Ok(Tensor::new(data, vec![len]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_axis0_selects_rows() {
        // numpy: np.compress([1,0,1], [[1,2],[3,4],[5,6]], axis=0) -> [[1,2],[5,6]]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let cond = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
        let y = compress(&x, &cond, Some(0)).expect("compress failed");
        assert_eq!(y.shape, vec![2, 2]);
        assert_eq!(y.data, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn compress_axis_none_flattens() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let cond = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4]);
        let y = compress(&x, &cond, None).expect("compress failed");
        assert_eq!(y.shape, vec![2]);
        assert_eq!(y.data, vec![2.0, 3.0]);
    }

    /// [`.max(1)` zero-dim regression] A leading dim of size 0 (outer product across axis 1
    /// crosses that zero) used to be clamped from 0 up to 1 by a stray `.max(1)`, which then
    /// indexed the (correctly zero-length) `input.data` out of bounds — a panic, not just a
    /// wrong shape. Must instead flow through as a genuinely empty result.
    #[test]
    fn compress_zero_size_outer_dim_does_not_panic() {
        let x = Tensor::new(Vec::new(), vec![0, 3, 4]); // 0 elements
        let cond = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
        let y = compress(&x, &cond, Some(1)).expect("compress failed");
        assert_eq!(y.shape, vec![0, 2, 4]);
        assert!(y.data.is_empty());
    }

    /// Same idiom, this time the zero dim is the *inner* (trailing) product.
    #[test]
    fn compress_zero_size_inner_dim_does_not_panic() {
        let x = Tensor::new(Vec::new(), vec![2, 3, 0]); // 0 elements
        let cond = Tensor::new(vec![1.0, 0.0, 1.0], vec![3]);
        let y = compress(&x, &cond, Some(1)).expect("compress failed");
        assert_eq!(y.shape, vec![2, 2, 0]);
        assert!(y.data.is_empty());
    }
}
