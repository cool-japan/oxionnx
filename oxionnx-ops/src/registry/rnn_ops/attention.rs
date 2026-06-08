//! Attention operator implementations: AttentionOp and MultiHeadAttentionOp.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::attention;

// ── Typed helpers ─────────────────────────────────────────────────────────────

/// Extract f32 mask data from a `TypedTensor` option.
///
/// Returns `(mask_f32_data, is_broadcast)` where `is_broadcast` indicates the mask
/// has a batch dimension of 1 and should be reused for all batch elements.
/// `batch_total` is used only to determine the broadcast flag; it is the flat loop
/// count (batch × num_heads for MHA, batch for pure SDPA).
pub(super) fn extract_f32_mask(
    mask_tt: Option<&oxionnx_core::TypedTensor>,
    batch_total: usize,
) -> (Option<Vec<f32>>, bool) {
    match mask_tt {
        None => (None, false),
        Some(mt) => {
            let data = mt.storage.to_f32_vec();
            // Determine whether mask is broadcast: mask batch dim == 1 while batch_total > 1.
            let mask_ndim = mt.shape.len();
            let mask_batch: usize = mt.shape[..mask_ndim.saturating_sub(2)]
                .iter()
                .product::<usize>()
                .max(1);
            let is_broadcast = mask_batch == 1 && batch_total > 1;
            (Some(data), is_broadcast)
        }
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

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, TypedTensor};

        let q = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("AttentionOp: missing input Q".into()))?;
        let k = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("AttentionOp: missing input K".into()))?;
        let v = ctx
            .input(2)
            .ok_or_else(|| OnnxError::TensorNotFound("AttentionOp: missing input V".into()))?;
        let mask_tt = ctx.input(3);

        let scale_raw = ctx.attrs().f("scale", 0.0_f32);

        match (&q.storage, &k.storage, &v.storage) {
            // ── F32: delegate to existing f32 path ──
            (TensorStorage::F32(_), TensorStorage::F32(_), TensorStorage::F32(_)) => {
                oxionnx_core::default_typed_via_f32(self, ctx)
            }

            // ── F16 ──
            (TensorStorage::F16(qb), TensorStorage::F16(kb), TensorStorage::F16(vb)) => {
                // Q shape: [..., seq_q, head_dim]
                let q_ndim = q.shape.len();
                if q_ndim < 2 {
                    return Err(OnnxError::ShapeMismatch(
                        "AttentionOp F16: Q must be at least 2D".into(),
                    ));
                }
                let head_dim = q.shape[q_ndim - 1];
                let seq_q = q.shape[q_ndim - 2];
                let seq_kv = k.shape[k.shape.len() - 2];
                let batch_total: usize = q.shape[..q_ndim - 2].iter().product::<usize>().max(1);

                let effective_scale = if scale_raw == 0.0 {
                    1.0 / (head_dim as f32).sqrt()
                } else {
                    scale_raw
                };

                let (mask_data, mask_is_broadcast) = extract_f32_mask(mask_tt, batch_total);

                let out_len = batch_total * seq_q * head_dim;
                let mut out_bits = vec![0u16; out_len];
                let dims = attention::SdpaDims {
                    batch_total,
                    seq_q,
                    seq_kv,
                    head_dim,
                };
                attention::scaled_dot_product_attention_f16(
                    qb,
                    kb,
                    vb,
                    &dims,
                    mask_data.as_deref().map(|d| (d, mask_is_broadcast)),
                    effective_scale,
                    &mut out_bits,
                );

                let mut out_shape = q.shape[..q_ndim - 2].to_vec();
                out_shape.push(seq_q);
                out_shape.push(head_dim);
                Ok(vec![TypedTensor::new(
                    TensorStorage::F16(out_bits),
                    out_shape,
                )])
            }

            // ── BF16 ──
            (TensorStorage::BF16(qb), TensorStorage::BF16(kb), TensorStorage::BF16(vb)) => {
                let q_ndim = q.shape.len();
                if q_ndim < 2 {
                    return Err(OnnxError::ShapeMismatch(
                        "AttentionOp BF16: Q must be at least 2D".into(),
                    ));
                }
                let head_dim = q.shape[q_ndim - 1];
                let seq_q = q.shape[q_ndim - 2];
                let seq_kv = k.shape[k.shape.len() - 2];
                let batch_total: usize = q.shape[..q_ndim - 2].iter().product::<usize>().max(1);

                let effective_scale = if scale_raw == 0.0 {
                    1.0 / (head_dim as f32).sqrt()
                } else {
                    scale_raw
                };

                let (mask_data, mask_is_broadcast) = extract_f32_mask(mask_tt, batch_total);

                let out_len = batch_total * seq_q * head_dim;
                let mut out_bits = vec![0u16; out_len];
                let dims = attention::SdpaDims {
                    batch_total,
                    seq_q,
                    seq_kv,
                    head_dim,
                };
                attention::scaled_dot_product_attention_bf16(
                    qb,
                    kb,
                    vb,
                    &dims,
                    mask_data.as_deref().map(|d| (d, mask_is_broadcast)),
                    effective_scale,
                    &mut out_bits,
                );

                let mut out_shape = q.shape[..q_ndim - 2].to_vec();
                out_shape.push(seq_q);
                out_shape.push(head_dim);
                Ok(vec![TypedTensor::new(
                    TensorStorage::BF16(out_bits),
                    out_shape,
                )])
            }

            // ── Mixed dtypes: fall back to f32 round-trip ──
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal(
                "AttentionOp: expected at least 1 output slot, got 0".into(),
            ));
        }
        let q = ctx.input(0)?;
        let k = ctx.input(1)?;
        let v = ctx.input(2)?;
        let mask = ctx.optional_input(3);
        let attrs = ctx.attrs();
        let scale = {
            let s = attrs.f("scale", 0.0_f32);
            if s == 0.0 {
                None
            } else {
                Some(s)
            }
        };

        let (out_shape, len) = attention::sdpa_output_shape(q, k, v);
        if slots[0].data.len() != len {
            slots[0].data.resize(len, 0.0_f32);
        }
        slots[0].shape.clone_from(&out_shape);
        attention::sdpa_into(q, k, v, mask, scale, &mut slots[0].data)?;
        Ok(())
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

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, TypedTensor};

        let query = ctx.input(0).ok_or_else(|| {
            OnnxError::TensorNotFound("MultiHeadAttentionOp: missing query".into())
        })?;
        let key = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("MultiHeadAttentionOp: missing key".into()))?;
        let value = ctx.input(2).ok_or_else(|| {
            OnnxError::TensorNotFound("MultiHeadAttentionOp: missing value".into())
        })?;

        // Inputs 3,4 = qkv_weight / qkv_bias (optional projection). When present we
        // fall back to the f32 path because implementing typed QKV projection is out
        // of scope for W2.1.
        let qkv_weight = ctx.input(3);
        let out_proj_weight = ctx.input(5);
        let out_proj_bias = ctx.input(6);
        let mask_tt = ctx.input(7);

        let num_heads = ctx.attrs().i("num_heads", 1) as usize;

        // If qkv_weight projection is required, fall back to default.
        if qkv_weight.is_some() {
            return oxionnx_core::default_typed_via_f32(self, ctx);
        }

        // out_proj_weight is required for the typed F16/BF16 MHA kernel.
        // If absent we run SDPA-only (no output projection) via the F32 fallback.
        let Some(opw) = out_proj_weight else {
            return oxionnx_core::default_typed_via_f32(self, ctx);
        };

        match (&query.storage, &key.storage, &value.storage, &opw.storage) {
            // ── F32: delegate to existing f32 path ──
            (
                TensorStorage::F32(_),
                TensorStorage::F32(_),
                TensorStorage::F32(_),
                TensorStorage::F32(_),
            ) => oxionnx_core::default_typed_via_f32(self, ctx),

            // ── F16 ──
            (
                TensorStorage::F16(qb),
                TensorStorage::F16(kb),
                TensorStorage::F16(vb),
                TensorStorage::F16(wob),
            ) => {
                let batch = query.shape[0];
                let seq_q = query.shape[1];
                let embed_dim = query.shape[2];
                let seq_kv = key.shape[1];

                if embed_dim % num_heads != 0 {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "MultiHeadAttentionOp F16: embed_dim {embed_dim} not divisible by num_heads {num_heads}"
                    )));
                }
                let head_dim = embed_dim / num_heads;

                let effective_scale = 1.0 / (head_dim as f32).sqrt();
                let (mask_data, mask_is_broadcast) = extract_f32_mask(mask_tt, batch * num_heads);

                let out_proj_b_f16: Option<Vec<u16>> =
                    out_proj_bias.and_then(|t| match &t.storage {
                        TensorStorage::F16(bb) => Some(bb.clone()),
                        _ => None,
                    });

                let out_len = batch * seq_q * embed_dim;
                let mut out_bits = vec![0u16; out_len];
                let dims = attention::MhaDims {
                    batch,
                    seq_q,
                    seq_kv,
                    num_heads,
                    head_dim,
                    embed_dim,
                };

                attention::multi_head_attention_f16(
                    qb,
                    kb,
                    vb,
                    wob,
                    out_proj_b_f16.as_deref(),
                    &dims,
                    mask_data.as_deref().map(|d| (d, mask_is_broadcast)),
                    effective_scale,
                    &mut out_bits,
                )
                .map_err(OnnxError::ShapeMismatch)?;

                let out_shape = vec![batch, seq_q, embed_dim];
                Ok(vec![TypedTensor::new(
                    TensorStorage::F16(out_bits),
                    out_shape,
                )])
            }

            // ── BF16 ──
            (
                TensorStorage::BF16(qb),
                TensorStorage::BF16(kb),
                TensorStorage::BF16(vb),
                TensorStorage::BF16(wob),
            ) => {
                let batch = query.shape[0];
                let seq_q = query.shape[1];
                let embed_dim = query.shape[2];
                let seq_kv = key.shape[1];

                if embed_dim % num_heads != 0 {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "MultiHeadAttentionOp BF16: embed_dim {embed_dim} not divisible by num_heads {num_heads}"
                    )));
                }
                let head_dim = embed_dim / num_heads;

                let effective_scale = 1.0 / (head_dim as f32).sqrt();
                let (mask_data, mask_is_broadcast) = extract_f32_mask(mask_tt, batch * num_heads);

                let out_proj_b_bf16: Option<Vec<u16>> =
                    out_proj_bias.and_then(|t| match &t.storage {
                        TensorStorage::BF16(bb) => Some(bb.clone()),
                        _ => None,
                    });

                let out_len = batch * seq_q * embed_dim;
                let mut out_bits = vec![0u16; out_len];
                let dims = attention::MhaDims {
                    batch,
                    seq_q,
                    seq_kv,
                    num_heads,
                    head_dim,
                    embed_dim,
                };

                attention::multi_head_attention_bf16(
                    qb,
                    kb,
                    vb,
                    wob,
                    out_proj_b_bf16.as_deref(),
                    &dims,
                    mask_data.as_deref().map(|d| (d, mask_is_broadcast)),
                    effective_scale,
                    &mut out_bits,
                )
                .map_err(OnnxError::ShapeMismatch)?;

                let out_shape = vec![batch, seq_q, embed_dim];
                Ok(vec![TypedTensor::new(
                    TensorStorage::BF16(out_bits),
                    out_shape,
                )])
            }

            // ── Mixed dtypes: fall back to f32 round-trip ──
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal(
                "MultiHeadAttentionOp: expected at least 1 output slot, got 0".into(),
            ));
        }
        let query = ctx.input(0)?;
        let key = ctx.input(1)?;
        let value = ctx.input(2)?;
        let qkv_weight = ctx.optional_input(3);
        let qkv_bias = ctx.optional_input(4);
        let out_proj_weight = ctx.optional_input(5);
        let out_proj_bias = ctx.optional_input(6);
        let mask = ctx.optional_input(7);
        let num_heads = ctx.attrs().i("num_heads", 1) as usize;

        // Output shape is always [batch, seq_q, embed_dim].
        let batch = query.shape[0];
        let seq_q = query.shape[1];
        let embed_dim = query.shape[2];
        let out_len = batch * seq_q * embed_dim;

        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        slots[0].shape = vec![batch, seq_q, embed_dim];

        attention::multi_head_attention_into(
            query,
            key,
            value,
            qkv_weight,
            qkv_bias,
            out_proj_weight,
            out_proj_bias,
            mask,
            num_heads,
            &mut slots[0].data,
        )?;
        Ok(())
    }
}
