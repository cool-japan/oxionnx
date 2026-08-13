//! CUDA `Concat` dispatch: contiguous device-to-device copies, no kernel.
//!
//! # Status
//!
//! Unlike every other op in this wave, `Concat` needs no PTX kernel at all.
//! Concatenating along axis `d` of `[outer, axis_len, inner]`-shaped operands
//! (`outer = prod(shape[..d])`, `inner = prod(shape[d+1..])`) is, per input
//! tensor and per `outer` index, one *contiguous* run of `segment_len *
//! inner` elements landing at one contiguous destination offset — exactly
//! what `cuMemcpyDtoDAsync_v2` already does. [`cuda_concat_bound`] issues
//! `outer * num_inputs` such copies (all enqueued on the same stream as
//! every other op in this crate, so ordering is stream-order, not a fence
//! per copy) via
//! [`memcpy_device_to_device_async`](oxicuda_driver::memory_info::memcpy_device_to_device_async).
//!
//! For `axis == 0` (or, more generally, whenever every axis before the
//! concat axis has extent 1) this degenerates to exactly one copy per input
//! — the common case, and the one both of `det_10g.onnx`'s `Concat` nodes
//! use (`axis = 0`, assembling a `Resize` node's dynamic `sizes` input from
//! a `Shape`/`Slice` chain and two `Unsqueeze`d scalars).
//!
//! # Scope: any rank, but every operand must agree on axis and shape
//!
//! Unlike this wave's other ops, nothing here is NCHW/rank-4-specific — the
//! `[outer, axis_len, inner]` decomposition works for any rank, so a Concat
//! over rank-1 shape-arithmetic tensors (`det_10g.onnx`'s pattern) and one
//! over rank-4 feature maps are both claimed by the same code path.
//! [`concat_params_from_node`] still declines a node whose operands disagree
//! on rank or on any non-concatenated dimension, or that supplies fewer than
//! two inputs (a one-input "concat" is a `Reshape`-shaped question this
//! module leaves to the CPU rather than adding a special case nothing in
//! this workspace's models exercises).
//!
//! ## Advertised as CUDA-supported
//!
//! [`crate::is_supported_op`] reports `true` for `OpKind::Concat`; a node
//! outside the whitelist above still declines to `Ok(None)`.
//! Shadow-verifiable via [`crate::reference::ref_concat`] through the same
//! `verify_or_fallback` gate every other claimable op uses.

use oxicuda_driver::memory_info::memcpy_device_to_device_async;

use oxionnx_core::Attributes;

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;
use crate::residency::Operand;

/// Residency slot label shared by every `Concat` operand. Safe to share
/// across all of them: every binding here is `id = None` (an activation, not
/// a graph initializer — see [`crate::activation::InputBinding::bind`]), and
/// the weight-cache conflict machinery the label exists for is never
/// consulted for an `id = None` upload.
pub(crate) const INPUT_LABEL: &str = "concat_input";

/// Normalizes a possibly-negative ONNX axis against `rank`. A local
/// duplicate — see [`crate::pad`]'s identically-named helper's doc comment
/// for why.
#[must_use]
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    (0..r).contains(&a).then_some(a as usize)
}

/// Resolved geometry for one `Concat` dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatParams {
    /// Normalized concat axis.
    pub axis: usize,
    /// Full output shape (every input tensor's shape, with `axis` replaced
    /// by the sum of each input's extent there).
    pub out_shape: Vec<usize>,
    /// Each input's extent along `axis`, in node-input order.
    pub segment_lens: Vec<usize>,
}

/// Builds [`ConcatParams`] for an ONNX `Concat` node from its `axis`
/// attribute and its operands' shapes, or declines.
///
/// Pure and allocation-light: unit-testable without a CUDA device.
#[must_use]
pub fn concat_params_from_node(
    attrs: &Attributes,
    input_shapes: &[&[usize]],
) -> Option<ConcatParams> {
    // A one-operand "concat" is the identity, which this module treats as
    // out of scope rather than as a free win -- see the module docs.
    if input_shapes.len() < 2 {
        return None;
    }
    let ndim = input_shapes[0].len();
    if ndim == 0 {
        return None;
    }
    // This engine's `ConcatOp` (mirrored, not called -- see
    // `mod@crate::reference`'s "why this does not depend on `oxionnx-ops`")
    // defaults an absent `axis` to `0` rather than treating it as a required
    // attribute; matched here so a node this engine's CPU kernel accepts is
    // never declined here for a reason the CPU kernel does not itself apply.
    let axis = normalize_axis(attrs.i("axis", 0), ndim)?;

    for shape in &input_shapes[1..] {
        if shape.len() != ndim {
            return None;
        }
        for (d, (&a, &b)) in shape.iter().zip(input_shapes[0]).enumerate() {
            if d != axis && a != b {
                return None;
            }
        }
    }

    let segment_lens: Vec<usize> = input_shapes.iter().map(|s| s[axis]).collect();
    let out_axis_len = segment_lens
        .iter()
        .try_fold(0_usize, |acc, &v| acc.checked_add(v))?;
    let mut out_shape = input_shapes[0].to_vec();
    out_shape[axis] = out_axis_len;

    Some(ConcatParams {
        axis,
        out_shape,
        segment_lens,
    })
}

/// ONNX `Concat` forward on the GPU: `outer * inputs.len()` device-to-device
/// copies, no kernel launch. Operands may already be on the device (leaving
/// the result there too, when the caller asks for it).
///
/// `inputs`/`input_shapes` must be the same length as
/// `params.segment_lens`, in node-input order — the caller
/// (`try_cuda_dispatch_resident`'s `Concat` arm) has already resolved every
/// input via the same residency-aware lookup every other multi-operand op
/// uses.
///
/// # Returns
/// * `Ok(Some(_))` — computed on the GPU.
/// * `Ok(None)` — not accelerated; see the [module docs](self).
/// * `Err(_)` — a real failure after dispatch was already committed to.
///
/// # Errors
/// See "Returns" above.
pub(crate) fn cuda_concat_bound(
    ctx: &CudaContext,
    inputs: &[InputBinding<'_>],
    params: &ConcatParams,
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    if inputs.len() != params.segment_lens.len() || inputs.is_empty() {
        return Ok(None);
    }
    let ndim = params.out_shape.len();
    if params.axis >= ndim {
        return Ok(None);
    }
    let outer: usize = params.out_shape[..params.axis].iter().product();
    let inner: usize = params.out_shape[params.axis + 1..].iter().product();
    let out_axis_len = params.out_shape[params.axis];

    let Some(out_needed) = outer
        .checked_mul(out_axis_len)
        .and_then(|v| v.checked_mul(inner))
    else {
        return Ok(None);
    };

    let stream = ctx.dnn.stream();

    // Bind every operand up front (each may be resident or a fresh pooled
    // upload; see `INPUT_LABEL`'s doc comment for why one shared label is
    // safe here) before issuing any copy, so a bounds failure on a later
    // operand declines cleanly rather than leaving a partially-issued copy
    // sequence queued on the stream.
    let mut bound: Vec<Operand<'_>> = Vec::with_capacity(inputs.len());
    for (input, &seg_len) in inputs.iter().zip(&params.segment_lens) {
        let Some(needed) = outer
            .checked_mul(seg_len)
            .and_then(|v| v.checked_mul(inner))
        else {
            return Ok(None);
        };
        if input.len() < needed {
            return Ok(None);
        }
        let Some(operand) = input.bind(ctx, INPUT_LABEL, needed, stream)? else {
            return Ok(None);
        };
        bound.push(operand);
    }

    let d_output = ctx.scratch(out_needed)?;
    const ELEM_BYTES: usize = std::mem::size_of::<f32>();

    if out_needed > 0 {
        let mut axis_offset = 0_usize;
        for (operand, &seg_len) in bound.iter().zip(&params.segment_lens) {
            let seg_elems = seg_len.saturating_mul(inner);
            if seg_elems > 0 {
                for o in 0..outer {
                    let src_off = (o * seg_len).saturating_mul(inner);
                    let dst_off = (o * out_axis_len + axis_offset).saturating_mul(inner);
                    let src_ptr = operand.device_ptr() + (src_off * ELEM_BYTES) as u64;
                    let dst_ptr = d_output.device_ptr() + (dst_off * ELEM_BYTES) as u64;
                    memcpy_device_to_device_async(dst_ptr, src_ptr, seg_elems * ELEM_BYTES, stream)
                        .map_err(CudaDispatchError::Driver)?;
                }
            }
            axis_offset += seg_len;
        }
    }

    let out = finish_output(
        ctx,
        d_output,
        out_needed,
        &params.out_shape,
        placement,
        stream,
    )?;
    match &out {
        KernelOutput::Host(_) => {
            for operand in &mut bound {
                operand.retire();
            }
        }
        KernelOutput::Device(_) => {
            for operand in &mut bound {
                retire_queued(ctx, operand);
            }
        }
    }
    Ok(Some(out))
}

/// [`cuda_concat_bound`] over plain host slices, always reading the result
/// back. The non-resident entry point this module's own tests use.
///
/// # Errors
/// As [`cuda_concat_bound`].
#[must_use = "the concat result is only computed if this is consumed"]
pub fn cuda_concat(
    ctx: &CudaContext,
    inputs: &[&[f32]],
    params: &ConcatParams,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    let bindings: Vec<InputBinding<'_>> = inputs.iter().map(|d| InputBinding::Host(d)).collect();
    match cuda_concat_bound(ctx, &bindings, params, CudaOutputPlacement::Host)? {
        Some(KernelOutput::Host(data)) => Ok(Some(data)),
        Some(KernelOutput::Device(_)) => Err(CudaDispatchError::Shape {
            op: "Concat",
            msg: "host placement produced a device-resident result".to_string(),
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_axis(axis: i64) -> Attributes {
        let mut a = Attributes::default();
        a.ints.insert("axis".into(), axis);
        a
    }

    // ── concat_params_from_node ─────────────────────────────────────────────

    #[test]
    fn scrfd_style_axis_0_concat_of_shape_vectors_is_claimed() {
        // Real pattern: Concat_122/Concat_142 in det_10g.onnx, axis=0,
        // assembling a Resize `sizes` input from three rank-1 pieces.
        let shapes: [&[usize]; 3] = [&[2], &[1], &[1]];
        let params = concat_params_from_node(&attrs_axis(0), &shapes).expect("must be claimable");
        assert_eq!(params.out_shape, vec![4]);
        assert_eq!(params.segment_lens, vec![2, 1, 1]);
    }

    #[test]
    fn axis_1_concat_of_feature_maps_is_claimed() {
        let shapes: [&[usize]; 2] = [&[1, 3, 8, 8], &[1, 5, 8, 8]];
        let params = concat_params_from_node(&attrs_axis(1), &shapes).expect("must be claimable");
        assert_eq!(params.out_shape, vec![1, 8, 8, 8]);
        assert_eq!(params.segment_lens, vec![3, 5]);
    }

    #[test]
    fn a_negative_axis_normalizes() {
        let shapes: [&[usize]; 2] = [&[1, 3, 8, 8], &[1, 5, 8, 8]];
        let params = concat_params_from_node(&attrs_axis(-3), &shapes).expect("must be claimable");
        assert_eq!(params.axis, 1);
    }

    #[test]
    fn a_single_operand_declines() {
        let shapes: [&[usize]; 1] = [&[1, 3, 8, 8]];
        assert!(concat_params_from_node(&attrs_axis(0), &shapes).is_none());
    }

    #[test]
    fn mismatched_non_concat_dims_decline() {
        let shapes: [&[usize]; 2] = [&[1, 3, 8, 8], &[1, 5, 9, 8]]; // dim 2 disagrees
        assert!(concat_params_from_node(&attrs_axis(1), &shapes).is_none());
    }

    #[test]
    fn mismatched_rank_declines() {
        let shapes: [&[usize]; 2] = [&[1, 3, 8, 8], &[1, 5, 8]];
        assert!(concat_params_from_node(&attrs_axis(1), &shapes).is_none());
    }

    #[test]
    fn out_of_range_axis_declines() {
        let shapes: [&[usize]; 2] = [&[1, 3], &[1, 5]];
        assert!(concat_params_from_node(&attrs_axis(5), &shapes).is_none());
    }
}
