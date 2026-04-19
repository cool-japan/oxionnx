use oxionnx_core::Tensor;

/// Resize tensor using nearest or linear interpolation.
/// Supports coordinate transform modes: half_pixel, pytorch_half_pixel, asymmetric, align_corners
pub fn resize(
    input: &Tensor,
    scales: Option<&[f32]>,
    sizes: Option<&[usize]>,
    mode: &str,
    coord_transform: &str,
) -> Tensor {
    let ndim = input.ndim();

    // Determine output shape
    let out_shape: Vec<usize> = if let Some(sizes) = sizes {
        sizes.to_vec()
    } else if let Some(scales) = scales {
        input
            .shape
            .iter()
            .zip(scales.iter())
            .map(|(&d, &s)| (d as f32 * s).round() as usize)
            .collect()
    } else {
        // No resize info, return clone
        return input.clone();
    };

    let out_n: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_n];

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

    // Compute per-dim scales
    let dim_scales: Vec<f32> = (0..ndim)
        .map(|d| {
            if out_shape[d] == input.shape[d] {
                1.0
            } else if let Some(scales) = scales {
                scales[d]
            } else {
                out_shape[d] as f32 / input.shape[d] as f32
            }
        })
        .collect();

    match mode {
        "nearest" => {
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let src = transform_coord(
                        out_coord,
                        dim_scales[d],
                        input.shape[d],
                        out_shape[d],
                        coord_transform,
                    );
                    let src_idx = nearest_index(src, input.shape[d]);
                    in_idx += src_idx * in_strides[d];
                }
                *out_val = input.data[in_idx];
            }
        }
        "linear" | "bilinear" => {
            // For N-D linear interpolation, we only interpolate the last 2 spatial dims
            // (or all dims if ndim <= 2). Batch/channel dims use nearest (floor).
            // This matches typical ONNX Resize behavior for 4D tensors.
            let spatial_start = ndim.saturating_sub(2);

            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut out_coords = vec![0usize; ndim];
                for d in 0..ndim {
                    out_coords[d] = rem / out_strides[d];
                    rem %= out_strides[d];
                }

                // For non-spatial dims, just map directly
                let mut base_idx = 0usize;
                for d in 0..spatial_start {
                    let src = transform_coord(
                        out_coords[d],
                        dim_scales[d],
                        input.shape[d],
                        out_shape[d],
                        coord_transform,
                    );
                    let idx = src.round().max(0.0).min((input.shape[d] - 1) as f32) as usize;
                    base_idx += idx * in_strides[d];
                }

                if ndim - spatial_start == 2 {
                    // Bilinear interpolation on last 2 dims
                    let d0 = spatial_start;
                    let d1 = spatial_start + 1;
                    let sy = transform_coord(
                        out_coords[d0],
                        dim_scales[d0],
                        input.shape[d0],
                        out_shape[d0],
                        coord_transform,
                    );
                    let sx = transform_coord(
                        out_coords[d1],
                        dim_scales[d1],
                        input.shape[d1],
                        out_shape[d1],
                        coord_transform,
                    );

                    let y0 = sy.floor().max(0.0) as usize;
                    let x0 = sx.floor().max(0.0) as usize;
                    let y1 = (y0 + 1).min(input.shape[d0] - 1);
                    let x1 = (x0 + 1).min(input.shape[d1] - 1);

                    let fy = sy - sy.floor();
                    let fx = sx - sx.floor();

                    let v00 = input.data[base_idx + y0 * in_strides[d0] + x0 * in_strides[d1]];
                    let v01 = input.data[base_idx + y0 * in_strides[d0] + x1 * in_strides[d1]];
                    let v10 = input.data[base_idx + y1 * in_strides[d0] + x0 * in_strides[d1]];
                    let v11 = input.data[base_idx + y1 * in_strides[d0] + x1 * in_strides[d1]];

                    let val = v00 * (1.0 - fy) * (1.0 - fx)
                        + v01 * (1.0 - fy) * fx
                        + v10 * fy * (1.0 - fx)
                        + v11 * fy * fx;
                    *out_val = val;
                } else if ndim - spatial_start == 1 {
                    // Linear interpolation on last dim
                    let d = spatial_start;
                    let sx = transform_coord(
                        out_coords[d],
                        dim_scales[d],
                        input.shape[d],
                        out_shape[d],
                        coord_transform,
                    );
                    let x0 = sx.floor().max(0.0) as usize;
                    let x1 = (x0 + 1).min(input.shape[d] - 1);
                    let fx = sx - sx.floor();
                    let v0 = input.data[base_idx + x0 * in_strides[d]];
                    let v1 = input.data[base_idx + x1 * in_strides[d]];
                    *out_val = v0 * (1.0 - fx) + v1 * fx;
                } else {
                    // 0 spatial dims, just copy
                    *out_val = input.data[base_idx];
                }
            }
        }
        _ => {
            // Fallback: nearest
            for (out_idx, out_val) in out.iter_mut().enumerate() {
                let mut rem = out_idx;
                let mut in_idx = 0usize;
                for d in 0..ndim {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let src = transform_coord(
                        out_coord,
                        dim_scales[d],
                        input.shape[d],
                        out_shape[d],
                        coord_transform,
                    );
                    let src_idx = nearest_index(src, input.shape[d]);
                    in_idx += src_idx * in_strides[d];
                }
                *out_val = input.data[in_idx];
            }
        }
    }

    Tensor::new(out, out_shape)
}

pub(crate) fn transform_coord(
    dst: usize,
    scale: f32,
    input_size: usize,
    output_size: usize,
    mode: &str,
) -> f32 {
    match mode {
        "half_pixel" => (dst as f32 + 0.5) / scale - 0.5,
        "pytorch_half_pixel" => {
            if output_size > 1 {
                (dst as f32 + 0.5) / scale - 0.5
            } else {
                0.0
            }
        }
        "align_corners" => {
            if output_size <= 1 || input_size <= 1 {
                0.0
            } else {
                dst as f32 * (input_size - 1) as f32 / (output_size - 1) as f32
            }
        }
        // "asymmetric" and default
        _ => dst as f32 / scale,
    }
}

pub(crate) fn nearest_index(src: f32, dim_size: usize) -> usize {
    // ONNX default nearest mode: round to nearest, prefer floor for .5
    // But for asymmetric coordinates, floor is more standard
    let idx = src.floor().max(0.0) as usize;
    idx.min(dim_size - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_nearest_2x() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = resize(&input, None, Some(&[1, 1, 4, 4]), "nearest", "asymmetric");
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
        #[rustfmt::skip]
        assert_eq!(out.data, vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ]);
    }

    #[test]
    fn test_resize_bilinear_2x() {
        // 1x1x2x2 -> 1x1x4x4 with bilinear, align_corners
        let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 1, 2, 2]);
        let out = resize(&input, None, Some(&[1, 1, 4, 4]), "linear", "align_corners");
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
        // corners should be preserved
        assert!((out.data[0] - 0.0).abs() < 1e-5);
        assert!((out.data[3] - 1.0).abs() < 1e-5);
        assert!((out.data[12] - 2.0).abs() < 1e-5);
        assert!((out.data[15] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_resize_pytorch_half_pixel() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = resize(
            &input,
            Some(&[1.0, 1.0, 2.0, 2.0]),
            None,
            "nearest",
            "pytorch_half_pixel",
        );
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
    }
}
