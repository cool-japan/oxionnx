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

pub struct GridSampleOp;
impl Operator for GridSampleOp {
    fn op_type(&self) -> &str {
        "GridSample"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let grid = ctx.input(1)?;

        let attrs = ctx.attrs();
        let mode_i = attrs.i("mode", 0);
        let mode = match mode_i {
            0 => "bilinear",
            1 => "nearest",
            2 => "bicubic",
            _ => "bilinear",
        };
        let padding_mode_i = attrs.i("padding_mode", 0);
        let padding_mode = match padding_mode_i {
            0 => "zeros",
            1 => "border",
            2 => "reflection",
            _ => "zeros",
        };
        let align_corners = attrs.i("align_corners", 0) != 0;

        let out = spatial::grid_sample(input, grid, mode, padding_mode, align_corners)?;
        Ok(vec![out])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── RoiAlign ────────────────────────────────────────────────────────────────

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
        let output_height = attrs.i("output_height", 1) as usize;
        let output_width = attrs.i("output_width", 1) as usize;
        let sampling_ratio = attrs.i("sampling_ratio", 0) as usize;
        let spatial_scale = attrs.f("spatial_scale", 1.0);
        let mode_str = attrs.s("mode");
        let mode = if mode_str.is_empty() { "avg" } else { mode_str };

        let out = spatial::roi_align(
            input,
            rois,
            batch_indices,
            output_height,
            output_width,
            sampling_ratio,
            spatial_scale,
            mode,
        )?;
        Ok(vec![out])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}
