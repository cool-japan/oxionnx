//! Basic shape manipulation operations: reshape, flatten, transpose, squeeze, unsqueeze.

use oxionnx_core::Tensor;

/// Reshape `x` to the target `shape`, honouring the ONNX `allowzero` attribute.
///
/// Special entries in `shape`:
/// - `-1` infers a single dimension from the remaining element count (at most one allowed).
/// - `0` copies the corresponding input dimension when `allowzero` is `false` (the ONNX
///   default); when `allowzero` is `true` a `0` is a literal zero-size dimension (NumPy
///   semantics) and is taken verbatim.
///
/// When `allowzero` is `true`, combining a `-1` with an explicit `0` is ambiguous and
/// returns an error.
pub fn reshape(x: &Tensor, shape: &[i64], allowzero: bool) -> Result<Tensor, String> {
    let new_shape = resolve_reshape(&x.shape, x.numel(), shape, allowzero)?;
    Ok(Tensor::new(x.data.clone(), new_shape))
}

/// Resolve a concrete output shape for Reshape from the target `shape` spec.
///
/// `input_dims` and `numel` describe the source tensor; `allowzero` selects whether a `0`
/// copies the input dimension (`false`, ONNX default) or is a literal zero (`true`).
pub fn resolve_reshape(
    input_dims: &[usize],
    numel: usize,
    shape: &[i64],
    allowzero: bool,
) -> Result<Vec<usize>, String> {
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    if neg_count > 1 {
        return Err("reshape: at most one -1 allowed".into());
    }
    let has_explicit_zero = shape.contains(&0);
    if allowzero && neg_count == 1 && has_explicit_zero {
        return Err(
            "Reshape: cannot infer dimension (-1) when an explicit 0 is present under allowzero=1"
                .into(),
        );
    }

    // Resolve every non-(-1) entry first: a 0 either copies the input dim (allowzero=false)
    // or stays a literal zero (allowzero=true). The -1 entry is filled in afterwards.
    let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
    for (i, &d) in shape.iter().enumerate() {
        let dim = if d == -1 {
            usize::MAX // placeholder, overwritten below
        } else if d == 0 && !allowzero {
            *input_dims.get(i).ok_or_else(|| {
                format!(
                    "reshape: 0 at index {i} has no matching input dimension (rank {})",
                    input_dims.len()
                )
            })?
        } else {
            d as usize
        };
        new_shape.push(dim);
    }

    if neg_count == 1 {
        let known: usize = new_shape.iter().filter(|&&d| d != usize::MAX).product();
        if known == 0 {
            return Err(format!(
                "reshape: cannot infer dimension (-1) when remaining dimensions multiply to 0 ({new_shape:?})"
            ));
        }
        let inferred = numel / known;
        for d in new_shape.iter_mut() {
            if *d == usize::MAX {
                *d = inferred;
            }
        }
    }

    if new_shape.iter().product::<usize>() != numel {
        return Err(format!(
            "reshape: element count mismatch ({numel} vs {new_shape:?})"
        ));
    }
    Ok(new_shape)
}

pub fn flatten(x: &Tensor, axis: i64) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax..].iter().product::<usize>().max(1);
    Ok(Tensor::new(x.data.clone(), vec![outer, inner]))
}

/// Transpose according to a permutation. If perm is empty, reverses all dims.
pub fn transpose(x: &Tensor, perm: &[usize]) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let perm: Vec<usize> = if perm.is_empty() {
        (0..ndim).rev().collect()
    } else {
        perm.to_vec()
    };
    if perm.len() != ndim {
        return Err(format!(
            "transpose: perm len {} != ndim {}",
            perm.len(),
            ndim
        ));
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    let out_n = x.numel();
    let mut out = vec![0.0f32; out_n];
    let mut in_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        in_strides[i] = s;
        s *= x.shape[i];
    }
    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    for (out_idx, out_val) in out.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut in_idx = 0usize;
        for i in 0..ndim {
            let coord = rem / out_strides[i];
            rem %= out_strides[i];
            in_idx += coord * in_strides[perm[i]];
        }
        *out_val = x.data[in_idx];
    }
    Ok(Tensor::new(out, out_shape))
}

/// Remove axes of size 1. If axes is empty, remove all size-1 dims.
pub fn squeeze(x: &Tensor, axes: &[i64]) -> Tensor {
    let ndim = x.ndim();
    let axes: Vec<usize> = if axes.is_empty() {
        (0..ndim).filter(|&i| x.shape[i] == 1).collect()
    } else {
        axes.iter()
            .map(|&a| {
                if a < 0 {
                    (a + ndim as i64) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    };
    let new_shape: Vec<usize> = x
        .shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if axes.contains(&i) && d == 1 {
                None
            } else {
                Some(d)
            }
        })
        .collect();
    let new_shape = if new_shape.is_empty() {
        vec![1]
    } else {
        new_shape
    };
    Tensor::new(x.data.clone(), new_shape)
}

/// Insert size-1 axes at given positions.
pub fn unsqueeze(x: &Tensor, axes: &[i64]) -> Tensor {
    let mut new_shape = x.shape.clone();
    let mut sorted_axes: Vec<i64> = axes.to_vec();
    sorted_axes.sort();
    for &ax in &sorted_axes {
        let ax = if ax < 0 {
            (ax + new_shape.len() as i64 + 1) as usize
        } else {
            ax as usize
        };
        new_shape.insert(ax, 1);
    }
    Tensor::new(x.data.clone(), new_shape)
}
