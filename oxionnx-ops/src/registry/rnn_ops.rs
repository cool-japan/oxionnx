//! Operator trait implementations for RNN, attention, and spatial ops.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::{attention, rnn, spatial};

// ── LSTM ────────────────────────────────────────────────────────────────────

pub struct LSTMOp;
impl Operator for LSTMOp {
    fn op_type(&self) -> &str {
        "LSTM"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let w = ctx.input(1)?;
        let r = ctx.input(2)?;
        let b = ctx.optional_input(3);
        let sequence_lens = ctx.optional_input(4);
        let initial_h = ctx.optional_input(5);
        let initial_c = ctx.optional_input(6);

        let attrs = ctx.attrs();
        let hidden_size = attrs.i("hidden_size", 1) as usize;
        let direction_str = attrs.s("direction");
        let direction = if direction_str.is_empty() {
            "forward"
        } else {
            direction_str
        };

        let (y, y_h, y_c) = rnn::lstm(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            initial_c,
            hidden_size,
            direction,
        )?;

        Ok(vec![y, y_h, y_c])
    }
}

// ── GRU ─────────────────────────────────────────────────────────────────────

pub struct GRUOp;
impl Operator for GRUOp {
    fn op_type(&self) -> &str {
        "GRU"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let w = ctx.input(1)?;
        let r = ctx.input(2)?;
        let b = ctx.optional_input(3);
        let sequence_lens = ctx.optional_input(4);
        let initial_h = ctx.optional_input(5);

        let attrs = ctx.attrs();
        let hidden_size = attrs.i("hidden_size", 1) as usize;
        let direction_str = attrs.s("direction");
        let direction = if direction_str.is_empty() {
            "forward"
        } else {
            direction_str
        };
        let linear_before_reset = attrs.i("linear_before_reset", 0) != 0;

        let (y, y_h) = rnn::gru(
            x,
            w,
            r,
            b,
            sequence_lens,
            initial_h,
            hidden_size,
            direction,
            linear_before_reset,
        )?;

        Ok(vec![y, y_h])
    }
}

// ── Attention (Scaled Dot-Product) ──────────────────────────────────────────

pub struct AttentionOp;
impl Operator for AttentionOp {
    fn op_type(&self) -> &str {
        "Attention"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let q = ctx.input(0)?;
        let k = ctx.input(1)?;
        let v = ctx.input(2)?;
        let mask = ctx.optional_input(3);

        let attrs = ctx.attrs();
        let scale = {
            let s = attrs.f("scale", 0.0);
            if s == 0.0 {
                None
            } else {
                Some(s)
            }
        };

        let out = attention::scaled_dot_product_attention(q, k, v, mask, scale)?;
        Ok(vec![out])
    }
}

// ── MultiHeadAttention ──────────────────────────────────────────────────────

pub struct MultiHeadAttentionOp;
impl Operator for MultiHeadAttentionOp {
    fn op_type(&self) -> &str {
        "MultiHeadAttention"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let query = ctx.input(0)?;
        let key = ctx.input(1)?;
        let value = ctx.input(2)?;
        let qkv_weight = ctx.optional_input(3);
        let qkv_bias = ctx.optional_input(4);
        let out_proj_weight = ctx.optional_input(5);
        let out_proj_bias = ctx.optional_input(6);
        let mask = ctx.optional_input(7);

        let attrs = ctx.attrs();
        let num_heads = attrs.i("num_heads", 1) as usize;

        let out = attention::multi_head_attention(
            query,
            key,
            value,
            qkv_weight,
            qkv_bias,
            out_proj_weight,
            out_proj_bias,
            mask,
            num_heads,
        )?;
        Ok(vec![out])
    }
}

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
}
