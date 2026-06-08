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
    let len = s.iter().product::<usize>().max(1);
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
    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
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

    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);
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
