//! Basic shape manipulation operations: reshape, flatten, transpose, squeeze, unsqueeze.

use oxionnx_core::Tensor;

pub fn reshape(x: &Tensor, shape: &[i64]) -> Result<Tensor, String> {
    let numel = x.numel();
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    if neg_count > 1 {
        return Err("reshape: at most one -1 allowed".into());
    }
    let known: usize = shape
        .iter()
        .filter(|&&d| d != -1)
        .map(|&d| d as usize)
        .product();
    let new_shape: Vec<usize> = if neg_count == 1 {
        shape
            .iter()
            .map(|&d| if d == -1 { numel / known } else { d as usize })
            .collect()
    } else {
        shape.iter().map(|&d| d as usize).collect()
    };
    if new_shape.iter().product::<usize>() != numel {
        return Err(format!(
            "reshape: element count mismatch ({numel} vs {:?})",
            new_shape
        ));
    }
    Ok(Tensor::new(x.data.clone(), new_shape))
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
