//! Spatial operator kernels: GridSample and RoiAlign.

use oxionnx_core::{OnnxError, Tensor};

// ── GridSample ──────────────────────────────────────────────────────────────

/// Map grid coordinate from [-1, 1] to pixel coordinate.
fn grid_to_pixel(grid_val: f32, size: usize, align_corners: bool) -> f32 {
    if align_corners {
        (grid_val + 1.0) / 2.0 * (size as f32 - 1.0)
    } else {
        ((grid_val + 1.0) * size as f32 - 1.0) / 2.0
    }
}

/// Apply padding mode to a coordinate.
fn apply_padding(coord: f32, size: usize, padding_mode: &str) -> f32 {
    let s = size as f32;
    match padding_mode {
        "border" => coord.clamp(0.0, s - 1.0),
        "reflection" => {
            let max_val = if size <= 1 { 0.0 } else { s - 1.0 };
            if max_val == 0.0 {
                return 0.0;
            }
            // Reflect: map into [0, 2*max_val], then fold back
            let mut c = coord;
            if c < 0.0 {
                c = -c;
            }
            let period = 2.0 * max_val;
            c %= period;
            if c > max_val {
                c = period - c;
            }
            c
        }
        _ => coord, // "zeros" - will be handled by bounds check
    }
}

/// Sample from input using bilinear interpolation.
fn sample_bilinear(
    input: &[f32],
    c_offset: usize,
    h_in: usize,
    w_in: usize,
    y: f32,
    x: f32,
    padding_mode: &str,
) -> f32 {
    let y = apply_padding(y, h_in, padding_mode);
    let x = apply_padding(x, w_in, padding_mode);

    let y0 = y.floor() as i64;
    let x0 = x.floor() as i64;
    let y1 = y0 + 1;
    let x1 = x0 + 1;

    let ly = y - y0 as f32;
    let lx = x - x0 as f32;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;

    let get = |iy: i64, ix: i64| -> f32 {
        if iy < 0 || iy >= h_in as i64 || ix < 0 || ix >= w_in as i64 {
            if padding_mode == "zeros" {
                0.0
            } else {
                // border/reflection already clamped
                let iy_c = iy.clamp(0, h_in as i64 - 1) as usize;
                let ix_c = ix.clamp(0, w_in as i64 - 1) as usize;
                input[c_offset + iy_c * w_in + ix_c]
            }
        } else {
            input[c_offset + iy as usize * w_in + ix as usize]
        }
    };

    hy * hx * get(y0, x0) + hy * lx * get(y0, x1) + ly * hx * get(y1, x0) + ly * lx * get(y1, x1)
}

/// Sample from input using nearest neighbor.
fn sample_nearest(
    input: &[f32],
    c_offset: usize,
    h_in: usize,
    w_in: usize,
    y: f32,
    x: f32,
    padding_mode: &str,
) -> f32 {
    let y = apply_padding(y, h_in, padding_mode);
    let x = apply_padding(x, w_in, padding_mode);

    let iy = y.round() as i64;
    let ix = x.round() as i64;

    if iy < 0 || iy >= h_in as i64 || ix < 0 || ix >= w_in as i64 {
        if padding_mode == "zeros" {
            0.0
        } else {
            let iy_c = iy.clamp(0, h_in as i64 - 1) as usize;
            let ix_c = ix.clamp(0, w_in as i64 - 1) as usize;
            input[c_offset + iy_c * w_in + ix_c]
        }
    } else {
        input[c_offset + iy as usize * w_in + ix as usize]
    }
}

/// Cubic interpolation helper: f(x) using Keys' cubic.
fn cubic_weight(x: f32) -> f32 {
    let a = -0.75f32;
    let ax = x.abs();
    if ax <= 1.0 {
        ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
    } else if ax < 2.0 {
        ((a * ax - 5.0 * a) * ax + 8.0 * a) * ax - 4.0 * a
    } else {
        0.0
    }
}

/// Sample from input using bicubic interpolation.
fn sample_bicubic(
    input: &[f32],
    c_offset: usize,
    h_in: usize,
    w_in: usize,
    y: f32,
    x: f32,
    padding_mode: &str,
) -> f32 {
    let y = apply_padding(y, h_in, padding_mode);
    let x = apply_padding(x, w_in, padding_mode);

    let iy = y.floor() as i64;
    let ix = x.floor() as i64;
    let fy = y - iy as f32;
    let fx = x - ix as f32;

    let get = |dy: i64, dx: i64| -> f32 {
        let py = iy + dy;
        let px = ix + dx;
        if py < 0 || py >= h_in as i64 || px < 0 || px >= w_in as i64 {
            if padding_mode == "zeros" {
                0.0
            } else {
                let py_c = py.clamp(0, h_in as i64 - 1) as usize;
                let px_c = px.clamp(0, w_in as i64 - 1) as usize;
                input[c_offset + py_c * w_in + px_c]
            }
        } else {
            input[c_offset + py as usize * w_in + px as usize]
        }
    };

    let mut val = 0.0f32;
    for dy in -1i64..=2 {
        let wy = cubic_weight(fy - dy as f32);
        for dx in -1i64..=2 {
            let wx = cubic_weight(fx - dx as f32);
            val += wy * wx * get(dy, dx);
        }
    }
    val
}

/// Grid sample operator.
///
/// # Arguments
/// * `input` - `[N, C, H_in, W_in]`
/// * `grid` - `[N, H_out, W_out, 2]` grid coordinates in [-1, 1]
/// * `mode` - Interpolation: "bilinear", "nearest", "bicubic"
/// * `padding_mode` - "zeros", "border", "reflection"
/// * `align_corners` - If true, grid corners map to pixel corners
///
/// # Returns
/// `[N, C, H_out, W_out]`
pub fn grid_sample(
    input: &Tensor,
    grid: &Tensor,
    mode: &str,
    padding_mode: &str,
    align_corners: bool,
) -> Result<Tensor, OnnxError> {
    if input.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "grid_sample: input must be 4D, got {}D",
            input.ndim()
        )));
    }
    if grid.ndim() != 4 || grid.shape[3] != 2 {
        return Err(OnnxError::ShapeMismatch(format!(
            "grid_sample: grid must be [N, H_out, W_out, 2], got {:?}",
            grid.shape
        )));
    }

    let n = input.shape[0];
    let c = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];
    let h_out = grid.shape[1];
    let w_out = grid.shape[2];

    let mut output = vec![0.0f32; n * c * h_out * w_out];

    let input_batch_stride = c * h_in * w_in;
    let input_channel_stride = h_in * w_in;
    let grid_batch_stride = h_out * w_out * 2;
    let out_channel_stride = h_out * w_out;
    let out_batch_stride = c * h_out * w_out;

    for ni in 0..n {
        let grid_base = ni * grid_batch_stride;

        for ho in 0..h_out {
            for wo in 0..w_out {
                let grid_idx = grid_base + (ho * w_out + wo) * 2;
                let gx = grid.data[grid_idx];
                let gy = grid.data[grid_idx + 1];

                let px = grid_to_pixel(gx, w_in, align_corners);
                let py = grid_to_pixel(gy, h_in, align_corners);

                for ci in 0..c {
                    let c_off = ni * input_batch_stride + ci * input_channel_stride;
                    let out_idx = ni * out_batch_stride + ci * out_channel_stride + ho * w_out + wo;

                    output[out_idx] = match mode {
                        "nearest" => {
                            sample_nearest(&input.data, c_off, h_in, w_in, py, px, padding_mode)
                        }
                        "bicubic" => {
                            sample_bicubic(&input.data, c_off, h_in, w_in, py, px, padding_mode)
                        }
                        _ => sample_bilinear(&input.data, c_off, h_in, w_in, py, px, padding_mode),
                    };
                }
            }
        }
    }

    Ok(Tensor::new(output, vec![n, c, h_out, w_out]))
}

// ── RoiAlign ────────────────────────────────────────────────────────────────

/// Bilinear sample a single value from a 2D feature map.
fn bilinear_sample(data: &[f32], h: usize, w: usize, y: f32, x: f32) -> f32 {
    if y < -1.0 || y > h as f32 || x < -1.0 || x > w as f32 {
        return 0.0;
    }

    let y = y.max(0.0);
    let x = x.max(0.0);

    let y0 = y.floor() as usize;
    let x0 = x.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let y0 = y0.min(h - 1);
    let x0 = x0.min(w - 1);

    let ly = y - y0 as f32;
    let lx = x - x0 as f32;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;

    hy * hx * data[y0 * w + x0]
        + hy * lx * data[y0 * w + x1]
        + ly * hx * data[y1 * w + x0]
        + ly * lx * data[y1 * w + x1]
}

/// RoI Align operator.
///
/// # Arguments
/// * `input` - `[N, C, H, W]`
/// * `rois` - `[num_rois, 4]` (x1, y1, x2, y2) in image coordinates
/// * `batch_indices` - `[num_rois]` batch index for each ROI
/// * `output_height` - Pool output height
/// * `output_width` - Pool output width
/// * `sampling_ratio` - Samples per bin (0 = ceil(roi_size / output_size))
/// * `spatial_scale` - Scale factor for ROI coordinates
/// * `mode` - "avg" or "max"
///
/// # Returns
/// `[num_rois, C, output_height, output_width]`
#[allow(clippy::too_many_arguments)]
pub fn roi_align(
    input: &Tensor,
    rois: &Tensor,
    batch_indices: &Tensor,
    output_height: usize,
    output_width: usize,
    sampling_ratio: usize,
    spatial_scale: f32,
    mode: &str,
) -> Result<Tensor, OnnxError> {
    if input.ndim() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "roi_align: input must be 4D, got {}D",
            input.ndim()
        )));
    }

    let c = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let num_rois = rois.shape[0];

    let mut output = vec![0.0f32; num_rois * c * output_height * output_width];

    let channel_stride = h * w;
    let batch_stride = c * channel_stride;

    for roi_idx in 0..num_rois {
        let batch_idx = batch_indices.data[roi_idx] as usize;
        let roi_base = roi_idx * 4;

        let x1 = rois.data[roi_base] * spatial_scale;
        let y1 = rois.data[roi_base + 1] * spatial_scale;
        let x2 = rois.data[roi_base + 2] * spatial_scale;
        let y2 = rois.data[roi_base + 3] * spatial_scale;

        let roi_w = (x2 - x1).max(1e-6);
        let roi_h = (y2 - y1).max(1e-6);

        let bin_h = roi_h / output_height as f32;
        let bin_w = roi_w / output_width as f32;

        let sample_h = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            bin_h.ceil() as usize
        };
        let sample_w = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            bin_w.ceil() as usize
        };

        let num_samples = sample_h * sample_w;
        let inv_samples = 1.0 / num_samples as f32;

        for ci in 0..c {
            let feat_base = batch_idx * batch_stride + ci * channel_stride;
            let feat_map = &input.data[feat_base..feat_base + channel_stride];

            for oh in 0..output_height {
                for ow in 0..output_width {
                    let out_idx = roi_idx * c * output_height * output_width
                        + ci * output_height * output_width
                        + oh * output_width
                        + ow;

                    let mut accum = if mode == "max" {
                        f32::NEG_INFINITY
                    } else {
                        0.0f32
                    };

                    for sy in 0..sample_h {
                        for sx in 0..sample_w {
                            let y = y1
                                + bin_h * oh as f32
                                + bin_h * (sy as f32 + 0.5) / sample_h as f32;
                            let x = x1
                                + bin_w * ow as f32
                                + bin_w * (sx as f32 + 0.5) / sample_w as f32;

                            let val = bilinear_sample(feat_map, h, w, y, x);

                            if mode == "max" {
                                if val > accum {
                                    accum = val;
                                }
                            } else {
                                accum += val;
                            }
                        }
                    }

                    output[out_idx] = if mode == "max" {
                        if accum == f32::NEG_INFINITY {
                            0.0
                        } else {
                            accum
                        }
                    } else {
                        accum * inv_samples
                    };
                }
            }
        }
    }

    Ok(Tensor::new(
        output,
        vec![num_rois, c, output_height, output_width],
    ))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_sample_identity_bilinear() {
        // Input: [1, 1, 2, 2] = [[1, 2], [3, 4]]
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        // Grid that maps to the corners: [-1,-1], [1,-1], [-1,1], [1,1]
        // align_corners=true -> grid corners map to pixel corners
        let grid = Tensor::new(
            vec![-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            vec![1, 2, 2, 2],
        );

        let out = grid_sample(&input, &grid, "bilinear", "zeros", true)
            .expect("grid_sample should not fail");

        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // Should recover the original values at corners
        assert!((out.data[0] - 1.0).abs() < 1e-5); // (-1,-1) -> (0,0) -> 1
        assert!((out.data[1] - 2.0).abs() < 1e-5); // (1,-1) -> (0,1) -> 2
        assert!((out.data[2] - 3.0).abs() < 1e-5); // (-1,1) -> (1,0) -> 3
        assert!((out.data[3] - 4.0).abs() < 1e-5); // (1,1) -> (1,1) -> 4
    }

    #[test]
    fn test_grid_sample_center_bilinear() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        // Grid at center (0,0) with align_corners=true -> pixel (0.5, 0.5)
        let grid = Tensor::new(vec![0.0, 0.0], vec![1, 1, 1, 2]);

        let out =
            grid_sample(&input, &grid, "bilinear", "zeros", true).expect("grid_sample center");

        assert_eq!(out.shape, vec![1, 1, 1, 1]);
        // Bilinear interpolation at center of 2x2 -> average = 2.5
        assert!((out.data[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_grid_sample_nearest() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        // Grid at slightly right of center -> nearest to (0, 1)
        let grid = Tensor::new(vec![0.3, -0.3], vec![1, 1, 1, 2]);

        let out =
            grid_sample(&input, &grid, "nearest", "zeros", true).expect("grid_sample nearest");

        assert_eq!(out.shape, vec![1, 1, 1, 1]);
        // (0.3, -0.3) with align_corners, 2x2: px = (0.3+1)/2*1 = 0.65, py = (-0.3+1)/2*1 = 0.35
        // nearest: round(0.65) = 1, round(0.35) = 0 -> pixel (0, 1) -> value 2.0
        assert!((out.data[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_grid_sample_zeros_padding() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        // Grid outside bounds
        let grid = Tensor::new(vec![3.0, 3.0], vec![1, 1, 1, 2]);

        let out = grid_sample(&input, &grid, "bilinear", "zeros", true)
            .expect("grid_sample zeros padding");

        // Way outside -> should be 0
        assert!((out.data[0]).abs() < 1e-5);
    }

    #[test]
    fn test_roi_align_basic() {
        // Input: [1, 1, 4, 4] filled with sequential values
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input = Tensor::new(data, vec![1, 1, 4, 4]);

        // Single ROI covering the whole feature map
        let rois = Tensor::new(vec![0.0, 0.0, 4.0, 4.0], vec![1, 4]);
        let batch_indices = Tensor::new(vec![0.0], vec![1]);

        let out = roi_align(&input, &rois, &batch_indices, 2, 2, 2, 1.0, "avg")
            .expect("roi_align should not fail");

        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // Each 2x2 bin averages 4 sample points
        // Values should be reasonable averages of the quadrants
        for &v in &out.data {
            assert!(v >= 0.0 && v <= 15.0, "roi_align value out of range: {v}");
        }
    }

    #[test]
    fn test_roi_align_max_mode() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input = Tensor::new(data, vec![1, 1, 4, 4]);

        let rois = Tensor::new(vec![0.0, 0.0, 4.0, 4.0], vec![1, 4]);
        let batch_indices = Tensor::new(vec![0.0], vec![1]);

        let out = roi_align(&input, &rois, &batch_indices, 2, 2, 2, 1.0, "max")
            .expect("roi_align max mode");

        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // Max mode: values should be >= avg mode values
        for &v in &out.data {
            assert!(
                v >= 0.0 && v <= 15.0,
                "roi_align max value out of range: {v}"
            );
        }
        // Bottom-right bin should have the highest value
        assert!(out.data[3] >= out.data[0]);
    }

    #[test]
    fn test_roi_align_spatial_scale() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input = Tensor::new(data, vec![1, 1, 4, 4]);

        // ROI in original image coords, spatial_scale = 0.5
        let rois = Tensor::new(vec![0.0, 0.0, 8.0, 8.0], vec![1, 4]);
        let batch_indices = Tensor::new(vec![0.0], vec![1]);

        let out = roi_align(&input, &rois, &batch_indices, 2, 2, 2, 0.5, "avg")
            .expect("roi_align spatial scale");

        assert_eq!(out.shape, vec![1, 1, 2, 2]);
    }
}
