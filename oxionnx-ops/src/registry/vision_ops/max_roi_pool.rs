//! `MaxRoiPool` operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

/// ONNX `MaxRoiPool` (opset 1+), the Fast R-CNN region-of-interest max pool.
///
/// Inputs: `X` shaped `[N, C, H, W]` and `rois` shaped `[num_rois, 5]` whose
/// rows are `[batch_index, x1, y1, x2, y2]` — **width-first**, the opposite of
/// the `[y1, x1, y2, x2]` layout used by `RoiAlign`'s TF-derived cousins.
///
/// Output: `[num_rois, C, pooled_h, pooled_w]`.
///
/// The algorithm is Caffe's `ROIPoolingLayer`, which the ONNX spec is derived
/// from and ONNX Runtime implements:
///
/// ```text
/// start = round(coord * spatial_scale)                 // per axis
/// roi_extent = max(end - start + 1, 1)
/// bin = roi_extent / pooled_extent                     // real division
/// window = [floor(p * bin) + start, ceil((p+1) * bin) + start)   clamped to [0, extent]
/// ```
///
/// An empty window (reachable when a RoI is degenerate or falls outside the
/// feature map) yields `0.0`, **not** `-inf` — pooled RoIs feed straight into a
/// classifier head and a `-inf` there would poison the whole detection.
///
/// A `batch_index` outside `[0, N)` or a non-finite RoI coordinate is a
/// malformed model and produces a typed error rather than an out-of-bounds read.
pub struct MaxRoiPoolOp;

/// Read the `[pooled_h, pooled_w]` attribute.
fn read_pooled_shape(ctx: &OpContext<'_>) -> Result<[usize; 2], OnnxError> {
    let values = ctx.attrs().ints("pooled_shape");
    if values.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "MaxRoiPool: pooled_shape requires exactly 2 entries, got {}",
            values.len()
        )));
    }
    let mut out = [0_usize; 2];
    for (axis, slot) in out.iter_mut().enumerate() {
        if values[axis] < 1 {
            return Err(OnnxError::InvalidModel(format!(
                "MaxRoiPool: pooled_shape[{axis}] must be >= 1, got {}",
                values[axis]
            )));
        }
        *slot = values[axis] as usize;
    }
    Ok(out)
}

/// Scale one RoI coordinate into feature-map space, rounding half away from
/// zero the way Caffe's `round()` does.
#[inline]
fn scaled(coord: f32, spatial_scale: f32) -> f32 {
    (coord * spatial_scale).round()
}

impl Operator for MaxRoiPoolOp {
    fn op_type(&self) -> &str {
        "MaxRoiPool"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let rois = ctx.input(1)?;
        if x.ndim() != 4 {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxRoiPool: X must be 4D [N, C, H, W], got {:?}",
                x.shape
            )));
        }
        if rois.ndim() != 2 || rois.shape[1] != 5 {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxRoiPool: rois must be 2D [num_rois, 5], got {:?}",
                rois.shape
            )));
        }
        let [pooled_h, pooled_w] = read_pooled_shape(ctx)?;
        let spatial_scale = ctx.attrs().f("spatial_scale", 1.0);
        if !spatial_scale.is_finite() || spatial_scale <= 0.0 {
            return Err(OnnxError::InvalidModel(format!(
                "MaxRoiPool: spatial_scale must be finite and > 0, got {spatial_scale}"
            )));
        }

        let batches = x.shape[0];
        let channels = x.shape[1];
        let height = x.shape[2];
        let width = x.shape[3];
        if x.data.len() != batches * channels * height * width {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxRoiPool: X data length {} does not match shape {:?}",
                x.data.len(),
                x.shape
            )));
        }
        let num_rois = rois.shape[0];
        if rois.data.len() != num_rois * 5 {
            return Err(OnnxError::ShapeMismatch(format!(
                "MaxRoiPool: rois data length {} does not match shape {:?}",
                rois.data.len(),
                rois.shape
            )));
        }

        let mut out = vec![0.0_f32; num_rois * channels * pooled_h * pooled_w];

        for r in 0..num_rois {
            let row = &rois.data[r * 5..r * 5 + 5];
            for (idx, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(OnnxError::InvalidModel(format!(
                        "MaxRoiPool: rois[{r}][{idx}] is not finite ({v})"
                    )));
                }
            }
            let batch = row[0];
            if batch < 0.0 || batch >= batches as f32 {
                return Err(OnnxError::InvalidModel(format!(
                    "MaxRoiPool: rois[{r}] batch index {batch} is outside [0, {batches})"
                )));
            }
            let batch = batch as usize;

            let start_w = scaled(row[1], spatial_scale);
            let start_h = scaled(row[2], spatial_scale);
            let end_w = scaled(row[3], spatial_scale);
            let end_h = scaled(row[4], spatial_scale);
            let roi_h = (end_h - start_h + 1.0).max(1.0);
            let roi_w = (end_w - start_w + 1.0).max(1.0);
            let bin_h = roi_h / pooled_h as f32;
            let bin_w = roi_w / pooled_w as f32;

            for c in 0..channels {
                let plane = (batch * channels + c) * height * width;
                for ph in 0..pooled_h {
                    let h_lo = clamp_axis((ph as f32 * bin_h).floor() + start_h, height);
                    let h_hi = clamp_axis(((ph + 1) as f32 * bin_h).ceil() + start_h, height);
                    for pw in 0..pooled_w {
                        let w_lo = clamp_axis((pw as f32 * bin_w).floor() + start_w, width);
                        let w_hi = clamp_axis(((pw + 1) as f32 * bin_w).ceil() + start_w, width);
                        let dst = ((r * channels + c) * pooled_h + ph) * pooled_w + pw;
                        if h_hi <= h_lo || w_hi <= w_lo {
                            out[dst] = 0.0;
                            continue;
                        }
                        let mut best = f32::NEG_INFINITY;
                        for iy in h_lo..h_hi {
                            let row_base = plane + iy * width;
                            for ix in w_lo..w_hi {
                                let v = x.data[row_base + ix];
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        out[dst] = best;
                    }
                }
            }
        }

        Ok(vec![Tensor::new(
            out,
            vec![num_rois, channels, pooled_h, pooled_w],
        )])
    }
}

/// Clamp a (possibly negative or huge) float bin edge into `[0, extent]`.
#[inline]
fn clamp_axis(edge: f32, extent: usize) -> usize {
    if edge <= 0.0 {
        0
    } else if edge >= extent as f32 {
        extent
    } else {
        edge as usize
    }
}
