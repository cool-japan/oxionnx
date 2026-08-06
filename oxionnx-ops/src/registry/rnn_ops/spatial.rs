//! Spatial operator implementations: RotaryEmbeddingOp, GridSampleOp, RoiAlignOp.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::{attention, spatial};

// ── RotaryEmbedding ─────────────────────────────────────────────────────────

pub struct RotaryEmbeddingOp;
impl Operator for RotaryEmbeddingOp {
    fn op_type(&self) -> &str {
        "RotaryEmbedding"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let position_ids = ctx.input(1)?;
        let cos_cache = ctx.optional_input(2);
        let sin_cache = ctx.optional_input(3);

        let attrs = ctx.attrs();
        let base = attrs.f("base", 10000.0);

        let out = attention::rotary_embedding(input, position_ids, cos_cache, sin_cache, base)?;
        Ok(vec![out])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── GridSample ──────────────────────────────────────────────────────────────

/// Map the ONNX `mode` string onto the kernel's interpolation name.
///
/// `mode` is an ONNX **string** attribute. Opset 16 spelled the values
/// `"bilinear"` / `"nearest"` / `"bicubic"`; opset 20 renamed them to
/// `"linear"` / `"nearest"` / `"cubic"`. Both spellings are accepted, an empty
/// value means "attribute absent" (default `linear`), and anything else is a
/// malformed model.
fn grid_sample_mode(mode: &str) -> Result<&'static str, OnnxError> {
    match mode {
        "" | "linear" | "bilinear" => Ok("bilinear"),
        "nearest" => Ok("nearest"),
        "cubic" | "bicubic" => Ok("bicubic"),
        other => Err(OnnxError::InvalidModel(format!(
            "GridSample: mode must be one of 'linear'/'bilinear', 'nearest', \
             'cubic'/'bicubic', got '{other}'"
        ))),
    }
}

/// Map the ONNX `padding_mode` string onto the kernel's padding name.
fn grid_sample_padding_mode(padding_mode: &str) -> Result<&'static str, OnnxError> {
    match padding_mode {
        "" | "zeros" => Ok("zeros"),
        "border" => Ok("border"),
        "reflection" => Ok("reflection"),
        other => Err(OnnxError::InvalidModel(format!(
            "GridSample: padding_mode must be one of 'zeros', 'border', 'reflection', \
             got '{other}'"
        ))),
    }
}

pub struct GridSampleOp;
impl Operator for GridSampleOp {
    fn op_type(&self) -> &str {
        "GridSample"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let grid = ctx.input(1)?;

        let attrs = ctx.attrs();
        // `mode` and `padding_mode` are STRING attributes; reading them as ints
        // silently pinned every model to the defaults.
        let mode = grid_sample_mode(attrs.s("mode"))?;
        let padding_mode = grid_sample_padding_mode(attrs.s("padding_mode"))?;
        let align_corners = attrs.i("align_corners", 0) != 0;

        // Guard the kernel's unchecked indexing against a truncated model.
        if input.ndim() == 4 && input.data.len() < input.shape.iter().product::<usize>() {
            return Err(OnnxError::ShapeMismatch(format!(
                "GridSample: X holds {} elements but shape {:?} needs {}",
                input.data.len(),
                input.shape,
                input.shape.iter().product::<usize>()
            )));
        }
        if grid.ndim() == 4 && grid.data.len() < grid.shape.iter().product::<usize>() {
            return Err(OnnxError::ShapeMismatch(format!(
                "GridSample: grid holds {} elements but shape {:?} needs {}",
                grid.data.len(),
                grid.shape,
                grid.shape.iter().product::<usize>()
            )));
        }

        let out = spatial::grid_sample(input, grid, mode, padding_mode, align_corners)?;
        Ok(vec![out])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── RoiAlign ────────────────────────────────────────────────────────────────

/// Upper bound on the RoiAlign output element count, so a malformed
/// `output_height` / `output_width` cannot request an absurd allocation.
const MAX_ROI_ALIGN_OUTPUT_ELEMS: usize = 1 << 30;
/// Upper bound on `sampling_ratio`; `ratio²` samples are taken per output bin.
const MAX_ROI_ALIGN_SAMPLING_RATIO: i64 = 4096;

/// Apply `spatial_scale` and `coordinate_transformation_mode` to the raw ROI
/// corners, producing ROIs already expressed in input-feature-map coordinates.
///
/// Matches ONNX Runtime's `RoiAlign`:
///
/// ```text
/// offset      = coordinate_transformation_mode == "half_pixel" ? 0.5 : 0.0
/// start/end   = raw * spatial_scale - offset
/// if !half_pixel:  size = max(end - start, 1.0)   // legacy output_half_pixel
/// ```
///
/// The clamp is folded into the returned `x2`/`y2` so the downstream kernel can
/// keep taking pre-scaled ROIs with `spatial_scale = 1.0`.
fn transform_rois(
    rois: &Tensor,
    num_rois: usize,
    rois_elems: usize,
    spatial_scale: f32,
    half_pixel: bool,
) -> Tensor {
    let offset = if half_pixel { 0.5f32 } else { 0.0f32 };
    let mut data = vec![0.0f32; rois_elems];
    for roi_idx in 0..num_rois {
        let base = roi_idx * 4;
        let x1 = rois.data[base] * spatial_scale - offset;
        let y1 = rois.data[base + 1] * spatial_scale - offset;
        let mut x2 = rois.data[base + 2] * spatial_scale - offset;
        let mut y2 = rois.data[base + 3] * spatial_scale - offset;
        if !half_pixel {
            // Legacy `output_half_pixel` clamps the ROI extent to one pixel.
            x2 = x1 + (x2 - x1).max(1.0);
            y2 = y1 + (y2 - y1).max(1.0);
        }
        data[base] = x1;
        data[base + 1] = y1;
        data[base + 2] = x2;
        data[base + 3] = y2;
    }
    Tensor::new(data, vec![num_rois, 4])
}

pub struct RoiAlignOp;
impl Operator for RoiAlignOp {
    fn op_type(&self) -> &str {
        "RoiAlign"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let rois = ctx.input(1)?;
        let batch_indices = ctx.input(2)?;

        let attrs = ctx.attrs();
        let output_height = attrs.i("output_height", 1);
        let output_width = attrs.i("output_width", 1);
        let sampling_ratio = attrs.i("sampling_ratio", 0);
        let spatial_scale = attrs.f("spatial_scale", 1.0);
        let mode_str = attrs.s("mode");
        let mode = match mode_str {
            "" | "avg" => "avg",
            "max" => "max",
            other => {
                return Err(OnnxError::InvalidModel(format!(
                    "RoiAlign: mode must be 'avg' or 'max', got '{other}'"
                )))
            }
        };
        // ONNX `coordinate_transformation_mode` (string, default "half_pixel").
        let ctm = attrs.s("coordinate_transformation_mode");
        let half_pixel = match ctm {
            "" | "half_pixel" => true,
            "output_half_pixel" => false,
            other => {
                return Err(OnnxError::InvalidModel(format!(
                    "RoiAlign: coordinate_transformation_mode must be 'half_pixel' or \
                     'output_half_pixel', got '{other}'"
                )))
            }
        };

        if output_height <= 0 || output_width <= 0 || sampling_ratio < 0 {
            return Err(OnnxError::InvalidModel(format!(
                "RoiAlign: output_height/output_width must be positive and sampling_ratio \
                 non-negative, got {output_height}/{output_width}/{sampling_ratio}"
            )));
        }
        if input.ndim() != 4 {
            return Err(OnnxError::ShapeMismatch(format!(
                "RoiAlign: X must be 4D [N, C, H, W], got {:?}",
                input.shape
            )));
        }
        if rois.ndim() != 2 || rois.shape[1] != 4 {
            return Err(OnnxError::ShapeMismatch(format!(
                "RoiAlign: rois must be 2D [num_rois, 4], got {:?}",
                rois.shape
            )));
        }
        let num_rois = rois.shape[0];
        let rois_elems = num_rois.checked_mul(4).ok_or_else(|| {
            OnnxError::ShapeMismatch("RoiAlign: rois element count overflows usize".into())
        })?;
        if rois.data.len() < rois_elems {
            return Err(OnnxError::ShapeMismatch(format!(
                "RoiAlign: rois holds {} elements but shape {:?} needs {rois_elems}",
                rois.data.len(),
                rois.shape
            )));
        }
        // Guard the output allocation and the per-bin sample loop against
        // absurd attribute values before any size math is done.
        let out_elems = (output_height as usize)
            .checked_mul(output_width as usize)
            .and_then(|v| v.checked_mul(num_rois))
            .and_then(|v| v.checked_mul(input.shape[1]))
            .ok_or_else(|| {
                OnnxError::InvalidModel(format!(
                    "RoiAlign: output size {output_height}x{output_width} for {num_rois} rois \
                     overflows usize"
                ))
            })?;
        if out_elems > MAX_ROI_ALIGN_OUTPUT_ELEMS {
            return Err(OnnxError::InvalidModel(format!(
                "RoiAlign: output would hold {out_elems} elements, above the \
                 {MAX_ROI_ALIGN_OUTPUT_ELEMS} guard"
            )));
        }
        if sampling_ratio > MAX_ROI_ALIGN_SAMPLING_RATIO {
            return Err(OnnxError::InvalidModel(format!(
                "RoiAlign: sampling_ratio {sampling_ratio} is above the \
                 {MAX_ROI_ALIGN_SAMPLING_RATIO} guard"
            )));
        }
        if batch_indices.data.len() < num_rois {
            return Err(OnnxError::ShapeMismatch(format!(
                "RoiAlign: batch_indices holds {} entries but there are {num_rois} rois",
                batch_indices.data.len()
            )));
        }
        let n = input.shape[0];
        if input.data.len() < input.shape.iter().product::<usize>() {
            return Err(OnnxError::ShapeMismatch(format!(
                "RoiAlign: X holds {} elements but shape {:?} needs {}",
                input.data.len(),
                input.shape,
                input.shape.iter().product::<usize>()
            )));
        }
        for (roi_idx, &b) in batch_indices.data.iter().take(num_rois).enumerate() {
            if !(b >= 0.0 && (b as usize) < n) {
                return Err(OnnxError::InvalidModel(format!(
                    "RoiAlign: batch_indices[{roi_idx}] = {b} is outside [0, {n})"
                )));
            }
        }

        // Fold spatial_scale + coordinate_transformation_mode into the ROIs so the
        // kernel can consume already-transformed coordinates.
        let scaled_rois = transform_rois(rois, num_rois, rois_elems, spatial_scale, half_pixel);

        let out = spatial::roi_align(
            input,
            &scaled_rois,
            batch_indices,
            output_height as usize,
            output_width as usize,
            sampling_ratio as usize,
            1.0,
            mode,
        )?;
        Ok(vec![out])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}
