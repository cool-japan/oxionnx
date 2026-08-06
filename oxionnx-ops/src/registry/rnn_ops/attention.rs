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

/// Whether the half-precision kernels can apply this mask correctly.
///
/// They index the mask with a single flat batch stride, so they only handle a
/// mask whose trailing dims are exactly `[seq_q, seq_kv]` and whose leading dims
/// collapse to either `1` (one shared slice) or the full flat batch. Anything
/// else — notably the common `[B, 1, S_q, S_k]` padding mask against a
/// `batch × num_heads` loop — must go through the f32 path, which broadcasts
/// per dimension.
pub(super) fn typed_mask_is_supported(
    mask_tt: Option<&oxionnx_core::TypedTensor>,
    batch_total: usize,
    seq_q: usize,
    seq_kv: usize,
) -> bool {
    let Some(mt) = mask_tt else {
        return true;
    };
    let ndim = mt.shape.len();
    if ndim < 2 {
        return false;
    }
    if mt.shape[ndim - 2] != seq_q || mt.shape[ndim - 1] != seq_kv {
        return false;
    }
    let mask_batch: usize = mt.shape[..ndim - 2].iter().product::<usize>().max(1);
    if mask_batch != 1 && mask_batch != batch_total {
        return false;
    }
    mt.storage.len() >= mask_batch * seq_q * seq_kv
}

/// NumPy-broadcast Q/K/V's leading (batch-like) dims for the half-precision
/// typed SDPA kernels, mirroring [`attention::sdpa_output_shape`] (`attention/
/// core.rs`) — the f32 path's single source of truth for this computation —
/// so the returned shape's leading part is never undersold by using only one
/// operand's own batch.
///
/// Returns `None` when Q/K/V's leading shapes are not literally NumPy-
/// broadcastable (e.g. `[2]` against `[3]`); callers decline to the f32
/// fallback in that case, which has its own (documented) collapse-to-flat-
/// batch behaviour for that fringe case.
fn broadcast_sdpa_lead_dims(
    q_lead: &[usize],
    k_lead: &[usize],
    v_lead: &[usize],
) -> Option<Vec<usize>> {
    Tensor::broadcast_shape(q_lead, k_lead)
        .and_then(|qk| Tensor::broadcast_shape(&qk, v_lead))
        .ok()
}

/// Whether every one of Q/K/V's own leading-dim batch can be handed directly
/// to the native F16/BF16 kernel once the loop count is `batch_total`.
///
/// The kernel (`attention::typed::sdpa_f32_kernel`) indexes each operand by a
/// plain `b * stride` for `b` in `0..batch_total` — no per-operand modulo like
/// the f32 `SdpaJob::run_slice`'s `b % operand_batch` — so an operand is only
/// safe to pass as-is when its own batch already equals `batch_total`. The one
/// exception [`tile_bits_for_batch`] handles below: an operand whose own batch
/// is exactly `1` (the only leading-dim value NumPy broadcasting can widen)
/// can be tiled up to `batch_total` first, which makes the same flat indexing
/// correct without touching the kernel itself.
fn batch_is_tileable(own_batch: usize, batch_total: usize) -> bool {
    own_batch == batch_total || own_batch == 1
}

/// Tile a flat per-batch bit slice (`F16`/`BF16` storage is `Vec<u16>` either
/// way) up to `batch_total` copies when its own batch is `1`, or return it
/// unchanged when it already spans `batch_total`.
///
/// Callers must first confirm [`batch_is_tileable`]`(own_batch, batch_total)`
/// — those are the only two cases this function handles, and it is not meant
/// to validate anything beyond them.
fn tile_bits_for_batch(
    bits: &[u16],
    own_batch: usize,
    batch_total: usize,
) -> std::borrow::Cow<'_, [u16]> {
    if own_batch == batch_total {
        std::borrow::Cow::Borrowed(bits)
    } else {
        std::borrow::Cow::Owned(bits.repeat(batch_total))
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
        let is_causal = attrs.i("is_causal", 0) != 0;

        let out = attention::core::sdpa_causal(q, k, v, mask, scale, is_causal)?;
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

        // The half-precision kernels have no causal flag; the f32 path does.
        if ctx.attrs().i("is_causal", 0) != 0 {
            return oxionnx_core::default_typed_via_f32(self, ctx);
        }

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

                // NumPy-broadcast Q/K/V's leading (batch-like) dims — mirrors
                // the fixed `attention::sdpa_output_shape` (`attention/core.rs`)
                // so out_shape is never undersold by using only Q's own batch.
                // See `batch_is_tileable`/`tile_bits_for_batch` for why an
                // operand whose own batch is `1` is tiled up to the broadcast
                // batch below rather than handed to the kernel as-is, and why
                // anything else that needs real per-element broadcasting (an
                // operand batch that is neither `1` nor already `batch_total`,
                // or leading shapes that are not literally NumPy-broadcastable
                // at all) declines to the f32 fallback instead.
                let q_lead = &q.shape[..q_ndim - 2];
                let k_lead = &k.shape[..k.shape.len().saturating_sub(2)];
                let v_lead = &v.shape[..v.shape.len().saturating_sub(2)];
                let Some(lead_bcast) = broadcast_sdpa_lead_dims(q_lead, k_lead, v_lead) else {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                };
                let batch_total = lead_bcast.iter().product::<usize>().max(1);
                let q_batch = q_lead.iter().product::<usize>().max(1);
                let k_batch = k_lead.iter().product::<usize>().max(1);
                let v_batch = v_lead.iter().product::<usize>().max(1);
                if !batch_is_tileable(q_batch, batch_total)
                    || !batch_is_tileable(k_batch, batch_total)
                    || !batch_is_tileable(v_batch, batch_total)
                {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                }

                let effective_scale = if scale_raw == 0.0 {
                    1.0 / (head_dim as f32).sqrt()
                } else {
                    scale_raw
                };

                if !typed_mask_is_supported(mask_tt, batch_total, seq_q, seq_kv) {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                }
                let (mask_data, mask_is_broadcast) = extract_f32_mask(mask_tt, batch_total);

                let q_tiled = tile_bits_for_batch(qb, q_batch, batch_total);
                let k_tiled = tile_bits_for_batch(kb, k_batch, batch_total);
                let v_tiled = tile_bits_for_batch(vb, v_batch, batch_total);

                let out_len = batch_total * seq_q * head_dim;
                let mut out_bits = vec![0u16; out_len];
                let dims = attention::SdpaDims {
                    batch_total,
                    seq_q,
                    seq_kv,
                    head_dim,
                };
                attention::scaled_dot_product_attention_f16(
                    &q_tiled,
                    &k_tiled,
                    &v_tiled,
                    &dims,
                    mask_data.as_deref().map(|d| (d, mask_is_broadcast)),
                    effective_scale,
                    &mut out_bits,
                );

                let mut out_shape = lead_bcast;
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

                // See the F16 arm above for the full rationale — identical
                // broadcast computation, only the storage variant differs.
                let q_lead = &q.shape[..q_ndim - 2];
                let k_lead = &k.shape[..k.shape.len().saturating_sub(2)];
                let v_lead = &v.shape[..v.shape.len().saturating_sub(2)];
                let Some(lead_bcast) = broadcast_sdpa_lead_dims(q_lead, k_lead, v_lead) else {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                };
                let batch_total = lead_bcast.iter().product::<usize>().max(1);
                let q_batch = q_lead.iter().product::<usize>().max(1);
                let k_batch = k_lead.iter().product::<usize>().max(1);
                let v_batch = v_lead.iter().product::<usize>().max(1);
                if !batch_is_tileable(q_batch, batch_total)
                    || !batch_is_tileable(k_batch, batch_total)
                    || !batch_is_tileable(v_batch, batch_total)
                {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                }

                let effective_scale = if scale_raw == 0.0 {
                    1.0 / (head_dim as f32).sqrt()
                } else {
                    scale_raw
                };

                if !typed_mask_is_supported(mask_tt, batch_total, seq_q, seq_kv) {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                }
                let (mask_data, mask_is_broadcast) = extract_f32_mask(mask_tt, batch_total);

                let q_tiled = tile_bits_for_batch(qb, q_batch, batch_total);
                let k_tiled = tile_bits_for_batch(kb, k_batch, batch_total);
                let v_tiled = tile_bits_for_batch(vb, v_batch, batch_total);

                let out_len = batch_total * seq_q * head_dim;
                let mut out_bits = vec![0u16; out_len];
                let dims = attention::SdpaDims {
                    batch_total,
                    seq_q,
                    seq_kv,
                    head_dim,
                };
                attention::scaled_dot_product_attention_bf16(
                    &q_tiled,
                    &k_tiled,
                    &v_tiled,
                    &dims,
                    mask_data.as_deref().map(|d| (d, mask_is_broadcast)),
                    effective_scale,
                    &mut out_bits,
                );

                let mut out_shape = lead_bcast;
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

        let is_causal = attrs.i("is_causal", 0) != 0;

        let (out_shape, len) = attention::sdpa_output_shape(q, k, v);
        if slots[0].data.len() != len {
            slots[0].data.resize(len, 0.0_f32);
        }
        slots[0].shape.clone_from(&out_shape);
        attention::sdpa_into(q, k, v, mask, scale, is_causal, &mut slots[0].data)?;
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
                let seq_q = query.shape[1];
                let embed_dim = query.shape[2];
                let seq_kv = key.shape[1];

                if embed_dim % num_heads != 0 {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "MultiHeadAttentionOp F16: embed_dim {embed_dim} not divisible by num_heads {num_heads}"
                    )));
                }
                let head_dim = embed_dim / num_heads;

                // NumPy-broadcast Q/K/V's batch dim (index 0), mirroring
                // `AttentionOp`'s fixed leading-dim broadcast above (see
                // `broadcast_sdpa_lead_dims`/`batch_is_tileable`/
                // `tile_bits_for_batch`) so `out_shape` is never undersold by
                // using only Q's own batch (e.g. a batch-of-1 query shared
                // against a larger cached-KV batch). `mha_f32_kernel` indexes
                // every operand by a plain `b * stride` for `b` in
                // `0..batch`, so it is only safe once every operand's own
                // batch already equals that flat `batch` -- a shared
                // (batch-of-1) operand is tiled up to it below instead.
                let q_batch = query.shape[0];
                let k_batch = key.shape[0];
                let v_batch = value.shape.first().copied().unwrap_or(1);
                let Some(batch_bcast) =
                    broadcast_sdpa_lead_dims(&[q_batch], &[k_batch], &[v_batch])
                else {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                };
                let batch = batch_bcast.first().copied().unwrap_or(1);
                if !batch_is_tileable(q_batch, batch)
                    || !batch_is_tileable(k_batch, batch)
                    || !batch_is_tileable(v_batch, batch)
                {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                }

                let effective_scale = 1.0 / (head_dim as f32).sqrt();
                if !typed_mask_is_supported(mask_tt, batch * num_heads, seq_q, seq_kv) {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                }
                let (mask_data, mask_is_broadcast) = extract_f32_mask(mask_tt, batch * num_heads);

                let out_proj_b_f16: Option<Vec<u16>> =
                    out_proj_bias.and_then(|t| match &t.storage {
                        TensorStorage::F16(bb) => Some(bb.clone()),
                        _ => None,
                    });

                let q_tiled = tile_bits_for_batch(qb, q_batch, batch);
                let k_tiled = tile_bits_for_batch(kb, k_batch, batch);
                let v_tiled = tile_bits_for_batch(vb, v_batch, batch);

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
                    &q_tiled,
                    &k_tiled,
                    &v_tiled,
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
                let seq_q = query.shape[1];
                let embed_dim = query.shape[2];
                let seq_kv = key.shape[1];

                if embed_dim % num_heads != 0 {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "MultiHeadAttentionOp BF16: embed_dim {embed_dim} not divisible by num_heads {num_heads}"
                    )));
                }
                let head_dim = embed_dim / num_heads;

                // See the F16 arm above for the full rationale -- identical
                // broadcast computation, only the storage variant differs.
                let q_batch = query.shape[0];
                let k_batch = key.shape[0];
                let v_batch = value.shape.first().copied().unwrap_or(1);
                let Some(batch_bcast) =
                    broadcast_sdpa_lead_dims(&[q_batch], &[k_batch], &[v_batch])
                else {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                };
                let batch = batch_bcast.first().copied().unwrap_or(1);
                if !batch_is_tileable(q_batch, batch)
                    || !batch_is_tileable(k_batch, batch)
                    || !batch_is_tileable(v_batch, batch)
                {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                }

                let effective_scale = 1.0 / (head_dim as f32).sqrt();
                if !typed_mask_is_supported(mask_tt, batch * num_heads, seq_q, seq_kv) {
                    return oxionnx_core::default_typed_via_f32(self, ctx);
                }
                let (mask_data, mask_is_broadcast) = extract_f32_mask(mask_tt, batch * num_heads);

                let out_proj_b_bf16: Option<Vec<u16>> =
                    out_proj_bias.and_then(|t| match &t.storage {
                        TensorStorage::BF16(bb) => Some(bb.clone()),
                        _ => None,
                    });

                let q_tiled = tile_bits_for_batch(qb, q_batch, batch);
                let k_tiled = tile_bits_for_batch(kb, k_batch, batch);
                let v_tiled = tile_bits_for_batch(vb, v_batch, batch);

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
                    &q_tiled,
                    &k_tiled,
                    &v_tiled,
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::{
        dtype::TensorStorage, Attributes, Node, OpKind, TypedOpContext, TypedTensor,
    };

    fn f32_to_f16_bits(vals: &[f32]) -> Vec<u16> {
        vals.iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect()
    }

    fn f32_to_bf16_bits(vals: &[f32]) -> Vec<u16> {
        vals.iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect()
    }

    fn seq_f32(start: f32, step: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| start + i as f32 * step).collect()
    }

    fn attention_node() -> Node {
        Node {
            name: "test_attention".into(),
            op: OpKind::Attention,
            inputs: vec![],
            outputs: vec![],
            attrs: Attributes::default(),
        }
    }

    fn ctx3<'a>(
        node: &'a Node,
        q: &'a TypedTensor,
        k: &'a TypedTensor,
        v: &'a TypedTensor,
    ) -> TypedOpContext<'a> {
        TypedOpContext {
            node,
            inputs: vec![Some(q), Some(k), Some(v)],
            outer_scope: None,
            registry: None,
        }
    }

    /// The central broadcast claim: `execute_typed`'s F16 arm must NumPy-
    /// broadcast Q/K/V's leading dims for `out_shape` (mirroring the fixed
    /// `attention::sdpa_output_shape`) rather than using only Q's own batch —
    /// and it must do so through the **native** F16 kernel (tiling Q's single
    /// slice), not by silently falling back to f32.
    ///
    /// Checked against an independent baseline, not just its own shape claim:
    /// explicitly tiling Q to `[kv_batch, seq_q, head_dim]` and re-running
    /// takes the ordinary (already-equal-batches) native-kernel path that
    /// `attention_native_dtype_test.rs`'s `test_sdpa_f16_parity` already
    /// covers — both calls decode to identical f32 values and execute the
    /// same deterministic kernel math, so the two outputs must be bit-for-bit
    /// identical, not merely close.
    #[test]
    fn f16_execute_typed_broadcasts_q_batch_of_one_against_larger_kv_batch_via_native_kernel() {
        let (seq_q, seq_kv, head_dim, kv_batch) = (3usize, 3usize, 4usize, 4usize);
        let q_f32 = seq_f32(0.01, 0.01, seq_q * head_dim);
        let k_f32 = seq_f32(0.02, 0.01, kv_batch * seq_kv * head_dim);
        let v_f32 = seq_f32(0.05, 0.02, kv_batch * seq_kv * head_dim);

        let node = attention_node();
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![1, seq_q, head_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![kv_batch, seq_kv, head_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![kv_batch, seq_kv, head_dim],
        );
        let ctx = ctx3(&node, &q_tt, &k_tt, &v_tt);

        let out = AttentionOp
            .execute_typed(&ctx)
            .expect("F16 broadcast SDPA typed");
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].shape,
            vec![kv_batch, seq_q, head_dim],
            "out_shape must be the NumPy broadcast of Q/K/V leading dims (kv_batch), not Q's own (1)"
        );
        let out_bits = match &out[0].storage {
            TensorStorage::F16(b) => b.clone(),
            other => {
                panic!("expected native F16 storage for a batch-of-1 broadcast, got {other:?}")
            }
        };

        // Independent baseline: explicitly tile Q and re-run.
        let q_tiled_f32: Vec<f32> = q_f32
            .iter()
            .copied()
            .cycle()
            .take(kv_batch * seq_q * head_dim)
            .collect();
        let q_tiled_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_tiled_f32)),
            vec![kv_batch, seq_q, head_dim],
        );
        let ctx_tiled = ctx3(&node, &q_tiled_tt, &k_tt, &v_tt);
        let out_tiled = AttentionOp
            .execute_typed(&ctx_tiled)
            .expect("F16 explicitly-tiled SDPA typed");
        let out_tiled_bits = match &out_tiled[0].storage {
            TensorStorage::F16(b) => b.clone(),
            other => panic!("expected native F16 storage, got {other:?}"),
        };
        assert_eq!(
            out_bits, out_tiled_bits,
            "broadcasting Q's batch of 1 must match explicitly tiling it beforehand"
        );
    }

    /// The reverse broadcast direction — K/V shared (batch 1), Q batched — a
    /// common "cached K/V reused across a query batch" shape.
    #[test]
    fn f16_execute_typed_broadcasts_shared_kv_against_larger_q_batch_via_native_kernel() {
        let (seq_q, seq_kv, head_dim, q_batch) = (3usize, 3usize, 4usize, 4usize);
        let q_f32 = seq_f32(0.01, 0.01, q_batch * seq_q * head_dim);
        let k_f32 = seq_f32(0.02, 0.01, seq_kv * head_dim);
        let v_f32 = seq_f32(0.05, 0.02, seq_kv * head_dim);

        let node = attention_node();
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![q_batch, seq_q, head_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![1, seq_kv, head_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![1, seq_kv, head_dim],
        );
        let ctx = ctx3(&node, &q_tt, &k_tt, &v_tt);

        let out = AttentionOp
            .execute_typed(&ctx)
            .expect("F16 broadcast SDPA typed (shared KV)");
        assert_eq!(out[0].shape, vec![q_batch, seq_q, head_dim]);
        assert!(
            matches!(out[0].storage, TensorStorage::F16(_)),
            "a batch-of-1 K/V broadcast must still take the native F16 kernel"
        );
    }

    /// BF16 analog of the main broadcast case, same bit-exact-vs-explicitly-
    /// tiled methodology.
    #[test]
    fn bf16_execute_typed_broadcasts_q_batch_of_one_against_larger_kv_batch_via_native_kernel() {
        let (seq_q, seq_kv, head_dim, kv_batch) = (3usize, 3usize, 4usize, 3usize);
        let q_f32 = seq_f32(0.01, 0.01, seq_q * head_dim);
        let k_f32 = seq_f32(0.02, 0.01, kv_batch * seq_kv * head_dim);
        let v_f32 = seq_f32(0.05, 0.02, kv_batch * seq_kv * head_dim);

        let node = attention_node();
        let q_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&q_f32)),
            vec![1, seq_q, head_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&k_f32)),
            vec![kv_batch, seq_kv, head_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&v_f32)),
            vec![kv_batch, seq_kv, head_dim],
        );
        let ctx = ctx3(&node, &q_tt, &k_tt, &v_tt);

        let out = AttentionOp
            .execute_typed(&ctx)
            .expect("BF16 broadcast SDPA typed");
        assert_eq!(out[0].shape, vec![kv_batch, seq_q, head_dim]);
        let out_bits = match &out[0].storage {
            TensorStorage::BF16(b) => b.clone(),
            other => {
                panic!("expected native BF16 storage for a batch-of-1 broadcast, got {other:?}")
            }
        };

        let q_tiled_f32: Vec<f32> = q_f32
            .iter()
            .copied()
            .cycle()
            .take(kv_batch * seq_q * head_dim)
            .collect();
        let q_tiled_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&q_tiled_f32)),
            vec![kv_batch, seq_q, head_dim],
        );
        let ctx_tiled = ctx3(&node, &q_tiled_tt, &k_tt, &v_tt);
        let out_tiled = AttentionOp
            .execute_typed(&ctx_tiled)
            .expect("BF16 explicitly-tiled SDPA typed");
        let out_tiled_bits = match &out_tiled[0].storage {
            TensorStorage::BF16(b) => b.clone(),
            other => panic!("expected native BF16 storage, got {other:?}"),
        };
        assert_eq!(out_bits, out_tiled_bits);
    }

    /// Regression guard: when Q/K/V already share the same batch (the common
    /// case, and everything `attention_native_dtype_test.rs` already
    /// exercises), `execute_typed` must still take the native F16 kernel —
    /// the broadcast guard must not turn into an unconditional f32 fallback.
    #[test]
    fn f16_execute_typed_uses_native_kernel_when_batches_already_match() {
        let (batch, seq, hd) = (2usize, 3usize, 4usize);
        let n = batch * seq * hd;
        let q_f32 = seq_f32(0.01, 0.01, n);
        let k_f32 = seq_f32(0.02, 0.01, n);
        let v_f32 = seq_f32(0.03, 0.01, n);
        let node = attention_node();
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![batch, seq, hd],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![batch, seq, hd],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![batch, seq, hd],
        );
        let ctx = ctx3(&node, &q_tt, &k_tt, &v_tt);

        let out = AttentionOp.execute_typed(&ctx).expect("F16 SDPA typed");
        assert_eq!(out[0].shape, vec![batch, seq, hd]);
        assert!(
            matches!(out[0].storage, TensorStorage::F16(_)),
            "no broadcast is needed here, so the native F16 kernel must run"
        );
    }

    /// A genuinely NumPy-incompatible leading-dim combination (`[2]` against
    /// `[3]`, neither broadcastable to the other) must decline cleanly to the
    /// f32 fallback rather than panic or silently mistile.
    #[test]
    fn f16_execute_typed_declines_to_f32_when_leading_dims_are_incompatible() {
        let (seq_q, seq_kv, head_dim) = (2usize, 2usize, 2usize);
        let q_f32 = seq_f32(0.01, 0.01, 2 * seq_q * head_dim);
        let k_f32 = seq_f32(0.02, 0.01, 3 * seq_kv * head_dim);
        let v_f32 = seq_f32(0.05, 0.02, 3 * seq_kv * head_dim);

        let node = attention_node();
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![2, seq_q, head_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![3, seq_kv, head_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![3, seq_kv, head_dim],
        );
        let ctx = ctx3(&node, &q_tt, &k_tt, &v_tt);

        let out = AttentionOp
            .execute_typed(&ctx)
            .expect("an incompatible broadcast must decline, not error");
        assert!(
            matches!(out[0].storage, TensorStorage::F32(_)),
            "an incompatible Q/K/V leading-dim combination must decline to the f32 fallback"
        );
    }

    /// An operand batch that is neither `1` nor the broadcast result (e.g. Q
    /// batch `2` against a broadcast batch of `4`) is exactly the pattern the
    /// native kernel's flat `b * stride` indexing cannot tile correctly —
    /// this must decline to f32, not attempt a wrong tile.
    #[test]
    fn f16_execute_typed_declines_to_f32_when_an_operand_batch_is_neither_one_nor_the_broadcast() {
        let (seq_q, seq_kv, head_dim) = (2usize, 2usize, 2usize);
        let q_f32 = seq_f32(0.01, 0.01, 2 * seq_q * head_dim);
        let k_f32 = seq_f32(0.02, 0.01, 4 * seq_kv * head_dim);
        let v_f32 = seq_f32(0.05, 0.02, 4 * seq_kv * head_dim);

        let node = attention_node();
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![2, seq_q, head_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![4, seq_kv, head_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![4, seq_kv, head_dim],
        );
        let ctx = ctx3(&node, &q_tt, &k_tt, &v_tt);

        let out = AttentionOp
            .execute_typed(&ctx)
            .expect("Q batch 2 vs broadcast batch 4 must decline, not error");
        assert_eq!(out[0].shape, vec![4, seq_q, head_dim]);
        assert!(
            matches!(out[0].storage, TensorStorage::F32(_)),
            "Q batch 2 is neither 1 nor the broadcast batch 4, so this must decline to f32"
        );
    }

    // ── MultiHeadAttentionOp::execute_typed F16/BF16 batch broadcast ───────
    //
    // Same claim as the `AttentionOp` suite above, specialized to MHA's fixed
    // 3-D `[batch, seq, embed_dim]` Q/K/V shapes: `execute_typed`'s F16/BF16
    // arms must NumPy-broadcast Q/K/V's batch dim (index 0) for `out_shape`
    // rather than using only Q's own (`query.shape[0]`), and must do so
    // through the native kernel (tiling the batch-of-1 operand), not by
    // silently falling back to f32.

    fn mha_node(num_heads: i64) -> Node {
        let mut attrs = Attributes::default();
        attrs.ints.insert("num_heads".into(), num_heads);
        Node {
            name: "test_mha".into(),
            op: OpKind::MultiHeadAttention,
            inputs: vec![],
            outputs: vec![],
            attrs,
        }
    }

    /// `MultiHeadAttentionOp` input order: 0=query, 1=key, 2=value,
    /// 3=qkv_weight, 4=qkv_bias, 5=out_proj_weight, 6=out_proj_bias,
    /// 7=mask. Slots 3/4/6/7 are left absent: no QKV projection (`execute_typed`
    /// falls back to f32 unconditionally when `qkv_weight` is present) and no
    /// output bias/mask, so the native F16/BF16 kernel path is reachable and
    /// isolates the batch-broadcast behaviour under test.
    fn ctx_mha<'a>(
        node: &'a Node,
        q: &'a TypedTensor,
        k: &'a TypedTensor,
        v: &'a TypedTensor,
        out_proj_weight: &'a TypedTensor,
    ) -> TypedOpContext<'a> {
        TypedOpContext {
            node,
            inputs: vec![
                Some(q),
                Some(k),
                Some(v),
                None,
                None,
                Some(out_proj_weight),
                None,
            ],
            outer_scope: None,
            registry: None,
        }
    }

    /// The central broadcast claim, MHA analog of the `AttentionOp` test of
    /// the same shape of name above: a batch-of-1 Q against a larger K/V
    /// batch must broadcast to the K/V batch, not collapse `out_shape` to
    /// Q's own batch of 1 -- checked against an independent baseline
    /// (explicitly tiling Q first) for bit-exact equality, not just a shape
    /// claim.
    #[test]
    fn mha_f16_execute_typed_broadcasts_q_batch_of_one_against_larger_kv_batch() {
        let (seq_q, seq_kv, embed_dim, num_heads, kv_batch) =
            (3usize, 3usize, 4usize, 2i64, 4usize);
        let q_f32 = seq_f32(0.01, 0.01, seq_q * embed_dim);
        let k_f32 = seq_f32(0.02, 0.01, kv_batch * seq_kv * embed_dim);
        let v_f32 = seq_f32(0.05, 0.02, kv_batch * seq_kv * embed_dim);
        let w_f32 = seq_f32(0.001, 0.001, embed_dim * embed_dim);

        let node = mha_node(num_heads);
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![1, seq_q, embed_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![kv_batch, seq_kv, embed_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![kv_batch, seq_kv, embed_dim],
        );
        let w_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&w_f32)),
            vec![embed_dim, embed_dim],
        );
        let ctx = ctx_mha(&node, &q_tt, &k_tt, &v_tt, &w_tt);

        let out = MultiHeadAttentionOp
            .execute_typed(&ctx)
            .expect("F16 broadcast MHA typed");
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].shape,
            vec![kv_batch, seq_q, embed_dim],
            "out_shape must be the NumPy broadcast of Q/K/V's batch dim (kv_batch), not Q's own (1)"
        );
        let out_bits = match &out[0].storage {
            TensorStorage::F16(b) => b.clone(),
            other => {
                panic!("expected native F16 storage for a batch-of-1 broadcast, got {other:?}")
            }
        };

        // Independent baseline: explicitly tile Q and re-run.
        let q_tiled_f32: Vec<f32> = q_f32
            .iter()
            .copied()
            .cycle()
            .take(kv_batch * seq_q * embed_dim)
            .collect();
        let q_tiled_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_tiled_f32)),
            vec![kv_batch, seq_q, embed_dim],
        );
        let ctx_tiled = ctx_mha(&node, &q_tiled_tt, &k_tt, &v_tt, &w_tt);
        let out_tiled = MultiHeadAttentionOp
            .execute_typed(&ctx_tiled)
            .expect("F16 explicitly-tiled MHA typed");
        let out_tiled_bits = match &out_tiled[0].storage {
            TensorStorage::F16(b) => b.clone(),
            other => panic!("expected native F16 storage, got {other:?}"),
        };
        assert_eq!(
            out_bits, out_tiled_bits,
            "broadcasting Q's batch of 1 must match explicitly tiling it beforehand"
        );
    }

    /// The reverse direction: K/V shared (batch 1), Q batched -- a "cached
    /// K/V reused across a query batch" shape.
    #[test]
    fn mha_f16_execute_typed_broadcasts_shared_kv_against_larger_q_batch() {
        let (seq_q, seq_kv, embed_dim, num_heads, q_batch) = (3usize, 3usize, 4usize, 2i64, 4usize);
        let q_f32 = seq_f32(0.01, 0.01, q_batch * seq_q * embed_dim);
        let k_f32 = seq_f32(0.02, 0.01, seq_kv * embed_dim);
        let v_f32 = seq_f32(0.05, 0.02, seq_kv * embed_dim);
        let w_f32 = seq_f32(0.001, 0.001, embed_dim * embed_dim);

        let node = mha_node(num_heads);
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![q_batch, seq_q, embed_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![1, seq_kv, embed_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![1, seq_kv, embed_dim],
        );
        let w_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&w_f32)),
            vec![embed_dim, embed_dim],
        );
        let ctx = ctx_mha(&node, &q_tt, &k_tt, &v_tt, &w_tt);

        let out = MultiHeadAttentionOp
            .execute_typed(&ctx)
            .expect("F16 broadcast MHA typed (shared KV)");
        assert_eq!(out[0].shape, vec![q_batch, seq_q, embed_dim]);
        assert!(
            matches!(out[0].storage, TensorStorage::F16(_)),
            "a batch-of-1 K/V broadcast must still take the native F16 kernel"
        );
    }

    /// BF16 analog of the main broadcast case, same bit-exact-vs-explicitly-
    /// tiled methodology.
    #[test]
    fn mha_bf16_execute_typed_broadcasts_q_batch_of_one_against_larger_kv_batch() {
        let (seq_q, seq_kv, embed_dim, num_heads, kv_batch) =
            (3usize, 3usize, 4usize, 2i64, 3usize);
        let q_f32 = seq_f32(0.01, 0.01, seq_q * embed_dim);
        let k_f32 = seq_f32(0.02, 0.01, kv_batch * seq_kv * embed_dim);
        let v_f32 = seq_f32(0.05, 0.02, kv_batch * seq_kv * embed_dim);
        let w_f32 = seq_f32(0.001, 0.001, embed_dim * embed_dim);

        let node = mha_node(num_heads);
        let q_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&q_f32)),
            vec![1, seq_q, embed_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&k_f32)),
            vec![kv_batch, seq_kv, embed_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&v_f32)),
            vec![kv_batch, seq_kv, embed_dim],
        );
        let w_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&w_f32)),
            vec![embed_dim, embed_dim],
        );
        let ctx = ctx_mha(&node, &q_tt, &k_tt, &v_tt, &w_tt);

        let out = MultiHeadAttentionOp
            .execute_typed(&ctx)
            .expect("BF16 broadcast MHA typed");
        assert_eq!(out[0].shape, vec![kv_batch, seq_q, embed_dim]);
        let out_bits = match &out[0].storage {
            TensorStorage::BF16(b) => b.clone(),
            other => {
                panic!("expected native BF16 storage for a batch-of-1 broadcast, got {other:?}")
            }
        };

        let q_tiled_f32: Vec<f32> = q_f32
            .iter()
            .copied()
            .cycle()
            .take(kv_batch * seq_q * embed_dim)
            .collect();
        let q_tiled_tt = TypedTensor::new(
            TensorStorage::BF16(f32_to_bf16_bits(&q_tiled_f32)),
            vec![kv_batch, seq_q, embed_dim],
        );
        let ctx_tiled = ctx_mha(&node, &q_tiled_tt, &k_tt, &v_tt, &w_tt);
        let out_tiled = MultiHeadAttentionOp
            .execute_typed(&ctx_tiled)
            .expect("BF16 explicitly-tiled MHA typed");
        let out_tiled_bits = match &out_tiled[0].storage {
            TensorStorage::BF16(b) => b.clone(),
            other => panic!("expected native BF16 storage, got {other:?}"),
        };
        assert_eq!(out_bits, out_tiled_bits);
    }

    /// Regression guard: when Q/K/V already share the same batch (the common
    /// case), `execute_typed` must still take the native F16 kernel -- the
    /// broadcast guard must not turn into an unconditional f32 fallback.
    #[test]
    fn mha_f16_execute_typed_uses_native_kernel_when_batches_already_match() {
        let (batch, seq, embed_dim, num_heads) = (2usize, 3usize, 4usize, 2i64);
        let n = batch * seq * embed_dim;
        let q_f32 = seq_f32(0.01, 0.01, n);
        let k_f32 = seq_f32(0.02, 0.01, n);
        let v_f32 = seq_f32(0.03, 0.01, n);
        let w_f32 = seq_f32(0.001, 0.001, embed_dim * embed_dim);
        let node = mha_node(num_heads);
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![batch, seq, embed_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![batch, seq, embed_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![batch, seq, embed_dim],
        );
        let w_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&w_f32)),
            vec![embed_dim, embed_dim],
        );
        let ctx = ctx_mha(&node, &q_tt, &k_tt, &v_tt, &w_tt);

        let out = MultiHeadAttentionOp
            .execute_typed(&ctx)
            .expect("F16 MHA typed");
        assert_eq!(out[0].shape, vec![batch, seq, embed_dim]);
        assert!(
            matches!(out[0].storage, TensorStorage::F16(_)),
            "no broadcast is needed here, so the native F16 kernel must run"
        );
    }

    /// A genuinely NumPy-incompatible batch combination (Q batch `2` against
    /// K/V batch `3`, neither broadcastable to the other) must decline
    /// cleanly to the f32 fallback rather than panic or silently mistile.
    ///
    /// Deliberately Q *smaller* than K/V (not the reverse): the f32 fallback
    /// this declines to (`multi_head_attention_into`, not owned by this
    /// file) derives its own loop count solely from `query.shape[0]` and
    /// indexes K/V by the same count, so a larger Q batch here would run
    /// that fallback past the end of K/V's buffer -- a panic that would look
    /// like this fix's bug, not the fallback's preexisting one.
    #[test]
    fn mha_f16_execute_typed_declines_to_f32_when_batches_are_incompatible() {
        let (seq_q, seq_kv, embed_dim, num_heads) = (2usize, 2usize, 4usize, 2i64);
        let q_f32 = seq_f32(0.01, 0.01, 2 * seq_q * embed_dim);
        let k_f32 = seq_f32(0.02, 0.01, 3 * seq_kv * embed_dim);
        let v_f32 = seq_f32(0.05, 0.02, 3 * seq_kv * embed_dim);
        let w_f32 = seq_f32(0.001, 0.001, embed_dim * embed_dim);

        let node = mha_node(num_heads);
        let q_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&q_f32)),
            vec![2, seq_q, embed_dim],
        );
        let k_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&k_f32)),
            vec![3, seq_kv, embed_dim],
        );
        let v_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&v_f32)),
            vec![3, seq_kv, embed_dim],
        );
        let w_tt = TypedTensor::new(
            TensorStorage::F16(f32_to_f16_bits(&w_f32)),
            vec![embed_dim, embed_dim],
        );
        let ctx = ctx_mha(&node, &q_tt, &k_tt, &v_tt, &w_tt);

        let out = MultiHeadAttentionOp
            .execute_typed(&ctx)
            .expect("an incompatible broadcast must decline, not error");
        assert!(
            matches!(out[0].storage, TensorStorage::F32(_)),
            "an incompatible Q/K/V batch combination must decline to the f32 fallback"
        );
    }
}
