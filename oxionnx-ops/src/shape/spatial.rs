//! Spatial rearrangement operations: depth_to_space, space_to_depth, reverse_sequence.

use oxionnx_core::Tensor;

/// DepthToSpace: rearrange data from depth into spatial dimensions.
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

/// SpaceToDepth: rearrange spatial data into depth.
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
    let mut strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        strides[i] = s;
        s *= x.shape[i];
    }
    let total = x.numel();
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
