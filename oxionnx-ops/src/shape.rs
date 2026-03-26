use oxionnx_core::Tensor;

pub fn reshape(x: &Tensor, shape: &[i64]) -> Result<Tensor, String> {
    let numel = x.numel();

    // Resolve -1 dimension
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

    // Compute input strides
    let mut in_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        in_strides[i] = s;
        s *= x.shape[i];
    }

    // Compute output strides
    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }

    for (out_idx, out_val) in out.iter_mut().enumerate() {
        // Decode out_idx to multi-dim out coord
        let mut rem = out_idx;
        let mut in_idx = 0usize;
        for i in 0..ndim {
            let coord = rem / out_strides[i];
            rem %= out_strides[i];
            // This coord corresponds to perm[i]-th dim of input
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
    // Sort axes so we insert in increasing order
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

/// Concatenate tensors along the given axis.
pub fn concat(tensors: &[&Tensor], axis: i64) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("concat: no tensors".into());
    }
    let ndim = tensors[0].ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };

    // Verify all tensors have same shape except on axis
    for t in tensors.iter().skip(1) {
        if t.ndim() != ndim {
            return Err("concat: tensors must have same ndim".into());
        }
        for d in 0..ndim {
            if d != ax && t.shape[d] != tensors[0].shape[d] {
                return Err(format!("concat: shape mismatch at dim {d}"));
            }
        }
    }

    let mut out_shape = tensors[0].shape.clone();
    out_shape[ax] = tensors.iter().map(|t| t.shape[ax]).sum();

    let mut out = Vec::with_capacity(out_shape.iter().product());

    // Compute outer/inner around the concat axis
    let outer: usize = tensors[0].shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = tensors[0].shape[ax + 1..].iter().product::<usize>().max(1);

    for o in 0..outer {
        for t in tensors {
            let seg = t.shape[ax];
            for s in 0..seg {
                let src_start = (o * t.shape[ax] + s) * inner;
                out.extend_from_slice(&t.data[src_start..src_start + inner]);
            }
        }
    }

    Ok(Tensor::new(out, out_shape))
}

/// Slice tensor along given axes with start/end/step.
pub fn slice(
    x: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<Tensor, String> {
    let ndim = x.ndim();
    let default_axes: Vec<i64> = (0..starts.len() as i64).collect();
    let axes = axes.unwrap_or(&default_axes);
    let default_steps: Vec<i64> = vec![1; starts.len()];
    let steps = steps.unwrap_or(&default_steps);

    // Build per-dim (start, end, step)
    let mut dim_slices: Vec<(usize, usize, usize)> = x.shape.iter().map(|&d| (0, d, 1)).collect();

    for (i, &ax) in axes.iter().enumerate() {
        let ax = if ax < 0 {
            (ax + ndim as i64) as usize
        } else {
            ax as usize
        };
        let dim = x.shape[ax];
        let step = steps[i].max(1) as usize;
        let start = starts[i].clamp(0, dim as i64) as usize;
        let end = ends[i].clamp(0, dim as i64) as usize;
        dim_slices[ax] = (start, end, step);
    }

    let out_shape: Vec<usize> = dim_slices
        .iter()
        .map(|&(s, e, step)| e.saturating_sub(s).div_ceil(step))
        .collect();

    let out_n: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(out_n);

    // Recursive iteration via flat index
    let mut strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        strides[i] = s;
        s *= x.shape[i];
    }

    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }

    for out_idx in 0..out_n {
        let mut rem = out_idx;
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            let (start, _, step) = dim_slices[d];
            in_idx += (start + coord * step) * strides[d];
        }
        out.push(x.data[in_idx]);
    }

    Ok(Tensor::new(out, out_shape))
}

/// Pad tensor with constant, reflect, or edge values.
/// pads format: [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
pub fn pad(input: &Tensor, pads: &[i64], mode: &str, constant_value: f32) -> Tensor {
    let ndim = input.ndim();
    assert!(pads.len() == 2 * ndim, "pad: pads length must be 2 * ndim");

    let begin: Vec<usize> = pads[..ndim].iter().map(|&p| p.max(0) as usize).collect();
    let end: Vec<usize> = pads[ndim..].iter().map(|&p| p.max(0) as usize).collect();

    let out_shape: Vec<usize> = (0..ndim)
        .map(|d| input.shape[d] + begin[d] + end[d])
        .collect();
    let out_n: usize = out_shape.iter().product();

    // Compute strides
    let mut in_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        in_strides[i] = s;
        s *= input.shape[i];
    }
    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }

    let mut out = vec![constant_value; out_n];

    match mode {
        "reflect" => {
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                let mut valid = true;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let in_coord_signed = out_coord as isize - begin[d] as isize;
                    let dim = input.shape[d] as isize;
                    // Reflect: bounce off boundaries
                    let mut c = in_coord_signed;
                    if dim <= 1 {
                        c = 0;
                    } else {
                        // Reflect into [0, 2*(dim-1)) range, then fold back
                        let period = 2 * (dim - 1);
                        c = c.rem_euclid(period);
                        if c >= dim {
                            c = period - c;
                        }
                    }
                    if c < 0 || c >= dim {
                        valid = false;
                        break;
                    }
                    in_idx += c as usize * in_strides[d];
                }
                if valid {
                    *out_val = input.data[in_idx];
                }
            }
        }
        "edge" => {
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let in_coord = (out_coord as isize - begin[d] as isize)
                        .max(0)
                        .min(input.shape[d] as isize - 1)
                        as usize;
                    in_idx += in_coord * in_strides[d];
                }
                *out_val = input.data[in_idx];
            }
        }
        _ => {
            // "constant" mode (default): fill with constant_value, copy input into center
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                let mut inside = true;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let in_coord = out_coord as isize - begin[d] as isize;
                    if in_coord < 0 || in_coord >= input.shape[d] as isize {
                        inside = false;
                        break;
                    }
                    in_idx += in_coord as usize * in_strides[d];
                }
                if inside {
                    *out_val = input.data[in_idx];
                }
            }
        }
    }

    Tensor::new(out, out_shape)
}

/// Split tensor along axis into chunks of given sizes.
pub fn split(x: &Tensor, axis: i64, split_sizes: &[usize]) -> Result<Vec<Tensor>, String> {
    if split_sizes.is_empty() {
        return Err("split: split_sizes must not be empty".into());
    }
    let ndim = x.ndim();
    let ax = if axis < 0 {
        (axis + ndim as i64) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(format!("split: axis {ax} out of range for {ndim}D tensor"));
    }

    let axis_len = x.shape[ax];
    let total: usize = split_sizes.iter().sum();
    if total != axis_len {
        return Err(format!("split: sizes sum {total} != axis len {axis_len}"));
    }

    let outer: usize = x.shape[..ax].iter().product::<usize>().max(1);
    let inner: usize = x.shape[ax + 1..].iter().product::<usize>().max(1);

    let mut results = Vec::with_capacity(split_sizes.len());
    let mut start = 0usize;

    for &chunk in split_sizes {
        let n_out = outer * chunk * inner;
        let mut out = Vec::with_capacity(n_out);
        for o in 0..outer {
            for j in start..start + chunk {
                let src = o * axis_len * inner + j * inner;
                out.extend_from_slice(&x.data[src..src + inner]);
            }
        }
        let mut out_shape = x.shape.clone();
        out_shape[ax] = chunk;
        results.push(Tensor::new(out, out_shape));
        start += chunk;
    }

    Ok(results)
}

/// Repeat tensor along each axis N times.
pub fn tile(x: &Tensor, repeats: &[usize]) -> Result<Tensor, String> {
    let ndim = x.ndim();
    if repeats.len() != ndim {
        return Err(format!(
            "tile: repeats len {} != tensor ndim {}",
            repeats.len(),
            ndim
        ));
    }

    let out_shape: Vec<usize> = x.shape.iter().zip(repeats).map(|(&d, &r)| d * r).collect();
    let out_n: usize = out_shape.iter().product();

    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    let mut in_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        in_strides[i] = s;
        s *= x.shape[i];
    }

    let mut out = vec![0.0f32; out_n];
    for (out_idx, out_val) in out.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += (coord % x.shape[d]) * in_strides[d];
        }
        *out_val = x.data[in_idx];
    }

    Ok(Tensor::new(out, out_shape))
}

/// DepthToSpace: rearrange data from depth into spatial dimensions
/// Input: [N, C*blocksize*blocksize, H, W]
/// Output: [N, C, H*blocksize, W*blocksize]
/// mode: "DCR" (default) or "CRD"
pub fn depth_to_space(x: &Tensor, blocksize: usize, mode: &str) -> Result<Tensor, String> {
    if x.ndim() != 4 {
        return Err("depth_to_space: input must be 4D [N,C,H,W]".into());
    }
    let (n, c_total, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let r = blocksize;
    if c_total % (r * r) != 0 {
        return Err(format!(
            "depth_to_space: channels {c_total} not divisible by blocksize^2 {}",
            r * r
        ));
    }
    let c = c_total / (r * r);
    let oh = h * r;
    let ow = w * r;
    let mut data = vec![0.0f32; n * c * oh * ow];

    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    for rh in 0..r {
                        for rw in 0..r {
                            let src_c = if mode == "CRD" {
                                ci * r * r + rh * r + rw
                            } else {
                                // DCR: depth-column-row ordering
                                rh * r * c + rw * c + ci
                            };
                            let src_idx = ((ni * c_total + src_c) * h + hi) * w + wi;
                            let dst_idx = ((ni * c + ci) * oh + hi * r + rh) * ow + wi * r + rw;
                            data[dst_idx] = x.data[src_idx];
                        }
                    }
                }
            }
        }
    }
    Ok(Tensor::new(data, vec![n, c, oh, ow]))
}

/// SpaceToDepth: rearrange spatial data into depth
/// Input: [N, C, H*blocksize, W*blocksize]
/// Output: [N, C*blocksize*blocksize, H, W]
pub fn space_to_depth(x: &Tensor, blocksize: usize) -> Result<Tensor, String> {
    if x.ndim() != 4 {
        return Err("space_to_depth: input must be 4D [N,C,H,W]".into());
    }
    let (n, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let r = blocksize;
    if h % r != 0 || w % r != 0 {
        return Err(format!(
            "space_to_depth: spatial dims {h}x{w} not divisible by blocksize {r}"
        ));
    }
    let oh = h / r;
    let ow = w / r;
    let oc = c * r * r;
    let mut data = vec![0.0f32; n * oc * oh * ow];

    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..oh {
                for wi in 0..ow {
                    for rh in 0..r {
                        for rw in 0..r {
                            let src_idx = ((ni * c + ci) * h + hi * r + rh) * w + wi * r + rw;
                            let dst_c = ci * r * r + rh * r + rw;
                            let dst_idx = ((ni * oc + dst_c) * oh + hi) * ow + wi;
                            data[dst_idx] = x.data[src_idx];
                        }
                    }
                }
            }
        }
    }
    Ok(Tensor::new(data, vec![n, oc, oh, ow]))
}

/// ReverseSequence: reverse parts of sequences along time_axis for each batch.
/// For each batch element i, reverse the first `sequence_lens[i]` elements along time_axis.
pub fn reverse_sequence(
    x: &Tensor,
    sequence_lens: &Tensor,
    batch_axis: i64,
    time_axis: i64,
) -> Result<Tensor, String> {
    let ndim = x.ndim();
    if ndim < 2 {
        return Err("reverse_sequence: input must be at least 2D".into());
    }
    let ba = if batch_axis < 0 {
        (ndim as i64 + batch_axis) as usize
    } else {
        batch_axis as usize
    };
    let ta = if time_axis < 0 {
        (ndim as i64 + time_axis) as usize
    } else {
        time_axis as usize
    };
    if ba >= ndim || ta >= ndim {
        return Err(format!(
            "reverse_sequence: batch_axis {ba} or time_axis {ta} out of range for {ndim}D"
        ));
    }
    if ba == ta {
        return Err("reverse_sequence: batch_axis and time_axis must differ".into());
    }

    let batch_size = x.shape[ba];
    if sequence_lens.numel() != batch_size {
        return Err(format!(
            "reverse_sequence: sequence_lens length {} != batch size {batch_size}",
            sequence_lens.numel()
        ));
    }

    let mut out = x.data.clone();

    // Compute strides
    let mut strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        strides[i] = s;
        s *= x.shape[i];
    }

    let total = x.numel();
    let mut out_strides = vec![0usize; ndim];
    let mut s2 = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s2;
        s2 *= x.shape[i];
    }

    // For each element, determine its batch index and time index,
    // and if time < seq_len[batch], reverse it
    for (flat_idx, out_val) in out.iter_mut().enumerate().take(total) {
        let mut rem = flat_idx;
        let mut coords = vec![0usize; ndim];
        for d in 0..ndim {
            coords[d] = rem / strides[d];
            rem %= strides[d];
        }

        let batch_idx = coords[ba];
        let time_idx = coords[ta];
        let seq_len = sequence_lens.data[batch_idx] as usize;

        if time_idx < seq_len {
            // Reverse: map time_idx -> seq_len - 1 - time_idx
            let mut new_coords = coords.clone();
            new_coords[ta] = seq_len - 1 - time_idx;
            let mut src_flat = 0usize;
            for d in 0..ndim {
                src_flat += new_coords[d] * strides[d];
            }
            *out_val = x.data[src_flat];
        }
    }

    Ok(Tensor::new(out, x.shape.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape() {
        let x = Tensor::new(vec![1.0; 6], vec![2, 3]);
        let y = reshape(&x, &[3, 2]).unwrap();
        assert_eq!(y.shape, vec![3, 2]);
    }

    #[test]
    fn test_reshape_neg1() {
        let x = Tensor::new(vec![1.0; 6], vec![6]);
        let y = reshape(&x, &[2, -1]).unwrap();
        assert_eq!(y.shape, vec![2, 3]);
    }

    #[test]
    fn test_transpose_2d() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y = transpose(&x, &[1, 0]).unwrap();
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data[0], 1.0);
        assert_eq!(y.data[1], 4.0);
        assert_eq!(y.data[2], 2.0);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let sq = squeeze(&x, &[]);
        assert_eq!(sq.shape, vec![3]);
        let un = unsqueeze(&sq, &[0, 2]);
        assert_eq!(un.shape, vec![1, 3, 1]);
    }

    #[test]
    fn test_concat() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![1, 2]);
        let c = concat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5]);
        let y = slice(&x, &[1], &[4], None, None).unwrap();
        assert_eq!(y.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pad_constant() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // pad 1 on all sides: pads = [1, 1, 1, 1]
        let y = pad(&x, &[1, 1, 1, 1], "constant", 0.0);
        assert_eq!(y.shape, vec![4, 4]);
        // center should be [1,2,3,4], rest 0
        assert_eq!(y.data[0], 0.0); // top-left corner
        assert_eq!(y.data[5], 1.0); // (1,1) = first element
        assert_eq!(y.data[6], 2.0); // (1,2)
        assert_eq!(y.data[9], 3.0); // (2,1)
        assert_eq!(y.data[10], 4.0); // (2,2)
    }

    #[test]
    fn test_split_equal() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let chunks = split(&x, 1, &[1, 2]).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape, vec![2, 1]);
        assert_eq!(chunks[1].shape, vec![2, 2]);
        assert_eq!(chunks[0].data, vec![1.0, 4.0]);
        assert_eq!(chunks[1].data, vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tile() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let y = tile(&x, &[2, 1]).unwrap();
        assert_eq!(y.shape, vec![2, 3]);
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tile_2d() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y = tile(&x, &[1, 2]).unwrap();
        assert_eq!(y.shape, vec![2, 4]);
        assert_eq!(y.data, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pad_reflect() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // pad 1 on left and right of dim 1: pads = [0, 1, 0, 1]
        let y = pad(&x, &[0, 1, 0, 1], "reflect", 0.0);
        assert_eq!(y.shape, vec![2, 5]);
        // row 0: reflect [1,2,3] with pad 1 left and 1 right -> [2, 1, 2, 3, 2]
        assert_eq!(y.data[0], 2.0);
        assert_eq!(y.data[1], 1.0);
        assert_eq!(y.data[2], 2.0);
        assert_eq!(y.data[3], 3.0);
        assert_eq!(y.data[4], 2.0);
    }

    #[test]
    fn test_depth_to_space() {
        // [1, 4, 1, 1] with blocksize=2 -> [1, 1, 2, 2]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1, 1]);
        let out = depth_to_space(&x, 2, "DCR").expect("depth_to_space DCR failed");
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_depth_to_space_crd() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1, 1]);
        let out = depth_to_space(&x, 2, "CRD").expect("depth_to_space CRD failed");
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // CRD: [ci*r*r + rh*r + rw] maps directly
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_space_to_depth() {
        // [1, 1, 2, 2] with blocksize=2 -> [1, 4, 1, 1]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = space_to_depth(&x, 2).expect("space_to_depth failed");
        assert_eq!(out.shape, vec![1, 4, 1, 1]);
    }

    #[test]
    fn test_depth_to_space_roundtrip() {
        let x = Tensor::new((0..16).map(|i| i as f32).collect(), vec![1, 4, 2, 2]);
        let d2s = depth_to_space(&x, 2, "CRD").expect("d2s failed");
        let s2d = space_to_depth(&d2s, 2).expect("s2d failed");
        assert_eq!(x.shape, s2d.shape);
        // Values should roundtrip for CRD mode
        for (a, b) in x.data.iter().zip(s2d.data.iter()) {
            assert!((a - b).abs() < 1e-6, "roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_depth_to_space_errors() {
        let x3d = Tensor::new(vec![1.0; 8], vec![2, 2, 2]);
        assert!(depth_to_space(&x3d, 2, "DCR").is_err());

        let x = Tensor::new(vec![1.0; 6], vec![1, 6, 1, 1]);
        assert!(depth_to_space(&x, 2, "DCR").is_err()); // 6 % 4 != 0
    }

    #[test]
    fn test_space_to_depth_errors() {
        let x3d = Tensor::new(vec![1.0; 8], vec![2, 2, 2]);
        assert!(space_to_depth(&x3d, 2).is_err());

        let x = Tensor::new(vec![1.0; 6], vec![1, 1, 3, 2]);
        assert!(space_to_depth(&x, 2).is_err()); // 3 % 2 != 0
    }

    #[test]
    fn test_reverse_sequence() {
        // [2, 4] tensor, batch_axis=0, time_axis=1
        // batch 0: reverse first 3 elements, batch 1: reverse first 2 elements
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
        let seq_lens = Tensor::new(vec![3.0, 2.0], vec![2]);
        let out = reverse_sequence(&x, &seq_lens, 0, 1).expect("reverse_sequence failed");
        assert_eq!(out.shape, vec![2, 4]);
        // batch 0: [3,2,1,4], batch 1: [6,5,7,8]
        assert_eq!(out.data, vec![3.0, 2.0, 1.0, 4.0, 6.0, 5.0, 7.0, 8.0]);
    }

    #[test]
    fn test_reverse_sequence_full() {
        // Reverse entire sequence
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let seq_lens = Tensor::new(vec![4.0], vec![1]);
        let out = reverse_sequence(&x, &seq_lens, 0, 1).expect("reverse_sequence failed");
        assert_eq!(out.data, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_reverse_sequence_errors() {
        let x = Tensor::new(vec![1.0], vec![1]);
        let seq_lens = Tensor::new(vec![1.0], vec![1]);
        assert!(reverse_sequence(&x, &seq_lens, 0, 1).is_err()); // 1D input

        let x2 = Tensor::new(vec![1.0; 4], vec![2, 2]);
        assert!(reverse_sequence(&x2, &seq_lens, 0, 0).is_err()); // same axis
        assert!(reverse_sequence(&x2, &seq_lens, 0, 1).is_err()); // seq_lens len mismatch
    }
}
