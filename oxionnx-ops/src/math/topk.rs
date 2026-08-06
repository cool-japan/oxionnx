use oxionnx_core::Tensor;

// ── Zero-copy _into variants ─────────────────────────────────────────────────

/// Compute the output shape for top_k (values and indices share this shape).
pub(crate) fn top_k_output_shape(x: &Tensor, k: usize, axis: i64) -> (Vec<usize>, usize) {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    let k = k.min(if ax < ndim { x.shape[ax] } else { 0 });
    let mut s = x.shape.clone();
    if ax < ndim {
        s[ax] = k;
    }
    // No `.max(1)`: an empty slice already multiplies to 1, so this only ever matters when `s`
    // holds a genuine zero-size dim elsewhere, which must produce `len == 0` (the caller sizes
    // its output buffer from this), not a phantom size-1 buffer.
    let len = s.iter().product::<usize>();
    (s, len)
}

/// Like top_k but writes values into `values_out` and indices into `indices_out`.
pub(crate) fn top_k_into(
    x: &Tensor,
    k: usize,
    axis: i64,
    largest: bool,
    sorted: bool,
    values_out: &mut [f32],
    indices_out: &mut [f32],
) -> Result<Vec<usize>, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("top_k: axis {ax} out of range for {ndim}D tensor"));
    }
    let k = k.min(x.shape[ax]);
    // No `.max(1)`: must stay consistent with `top_k_output_shape`'s (also un-clamped) `len`,
    // which sizes `values_out`/`indices_out` — clamping a genuine zero-size outer/inner dim to 1
    // here would walk `x.data`/`values_out`/`indices_out` past their real (zero) length.
    let outer: usize = x.shape[..ax].iter().product::<usize>();
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>();
    let axis_len = x.shape[ax];
    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(f32, usize)> = (0..axis_len)
                .map(|j| (x.data[o * axis_len * inner + j * inner + i], j))
                .collect();
            if largest {
                pairs.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                pairs.sort_unstable_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            if !sorted {
                pairs[..k].sort_unstable_by_key(|p| p.1);
            }
            for (ki, &(v, j)) in pairs[..k].iter().enumerate() {
                let dst = o * k * inner + ki * inner + i;
                values_out[dst] = v;
                indices_out[dst] = j as f32;
            }
        }
    }
    let (shape, _) = top_k_output_shape(x, k, axis);
    Ok(shape)
}

// ── TopK ────────────────────────────────────────────────────────────────────

/// Top-k values and indices along `axis`.
/// Returns `(values, indices)`.
pub fn top_k(
    x: &Tensor,
    k: usize,
    axis: i64,
    largest: bool,
    sorted: bool,
) -> Result<(Tensor, Tensor), String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("top_k: axis {ax} out of range for {ndim}D tensor"));
    }
    let k = k.min(x.shape[ax]);

    let mut out_shape = x.shape.clone();
    out_shape[ax] = k;
    let out_n: usize = out_shape.iter().product();

    let mut values = vec![0.0f32; out_n];
    let mut indices = vec![0.0f32; out_n];

    // No `.max(1)`: `out_n` above is already correctly 0 for a genuine zero-size outer/inner
    // dim (it is a plain, un-clamped `.product()`), so `values`/`indices` are zero-length in
    // that case — clamping `outer`/`inner` to 1 would make the loop below index past them.
    let outer: usize = x.shape[..ax].iter().product::<usize>();
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>();
    let axis_len = x.shape[ax];

    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(f32, usize)> = (0..axis_len)
                .map(|j| (x.data[o * axis_len * inner + j * inner + i], j))
                .collect();

            if largest {
                pairs.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                pairs.sort_unstable_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            // If not sorted by value, sort top-k by original index
            if !sorted {
                pairs[..k].sort_unstable_by_key(|p| p.1);
            }

            for (ki, (v, j)) in pairs[..k].iter().enumerate() {
                let dst = o * k * inner + ki * inner + i;
                values[dst] = *v;
                indices[dst] = *j as f32;
            }
        }
    }

    Ok((
        Tensor::new(values, out_shape.clone()),
        Tensor::new(indices, out_shape),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_into_matches_top_k() {
        let x = Tensor::new(vec![3.0, 1.0, 2.0], vec![3]);
        let (values, indices) = top_k(&x, 2, 0, true, true).expect("top_k failed");
        assert_eq!(values.data, vec![3.0, 2.0]);
        assert_eq!(indices.data, vec![0.0, 2.0]);

        let (_, out_len) = top_k_output_shape(&x, 2, 0);
        let mut values_out = vec![0.0f32; out_len];
        let mut indices_out = vec![0.0f32; out_len];
        let shape = top_k_into(&x, 2, 0, true, true, &mut values_out, &mut indices_out)
            .expect("top_k_into failed");
        assert_eq!(shape, values.shape);
        assert_eq!(values_out, values.data);
        assert_eq!(indices_out, indices.data);
    }

    /// [`.max(1)` zero-dim regression] `x` has shape `[2,0,3]`: axis 0 (size 2) is reduced, but
    /// the middle dim is genuinely 0, so `inner = product(shape[1..]) = product([0,3]) = 0`.
    /// `outer`/`inner` used to be clamped from a genuine 0 up to 1 by a stray `.max(1)` (in both
    /// `top_k` and `top_k_into`), which then indexed `x.data`/`values`/`indices` — all correctly
    /// zero-length for this shape — out of bounds. Must instead produce an empty result.
    #[test]
    fn top_k_zero_size_middle_dim_does_not_panic() {
        let x = Tensor::new(Vec::new(), vec![2, 0, 3]); // 0 elements
        let (values, indices) = top_k(&x, 1, 0, true, true).expect("top_k failed");
        assert_eq!(values.shape, vec![1, 0, 3]);
        assert!(values.data.is_empty());
        assert_eq!(indices.shape, vec![1, 0, 3]);
        assert!(indices.data.is_empty());

        // `top_k_into` / `top_k_output_shape` must agree with `top_k` and not panic either.
        let (shape, out_len) = top_k_output_shape(&x, 1, 0);
        assert_eq!(shape, vec![1, 0, 3]);
        assert_eq!(out_len, 0);
        let mut values_out: Vec<f32> = Vec::new();
        let mut indices_out: Vec<f32> = Vec::new();
        let into_shape = top_k_into(&x, 1, 0, true, true, &mut values_out, &mut indices_out)
            .expect("top_k_into failed");
        assert_eq!(into_shape, vec![1, 0, 3]);
    }
}
