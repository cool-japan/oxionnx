//! CUDA `Resize` dispatch: nearest-neighbour and bilinear interpolation.
//!
//! # Status
//!
//! Both modes dispatch straight to `oxicuda-dnn`'s existing resize kernels —
//! [`resize_nearest`](oxicuda_dnn::resize::resize_nearest) and
//! [`resize_bilinear`](oxicuda_dnn::resize::resize_bilinear) — mirroring
//! [`crate::pool`]'s relationship to `oxicuda-dnn`'s pooling kernels: neither
//! was called from anywhere in this workspace before this module.
//!
//! # Why `Resize` needs a *data* input, not just attributes
//!
//! Every other claimable op in this crate derives its dispatch parameters
//! from a node's attributes and shapes alone. `Resize`'s output shape is not
//! one of those: it is computed from the `sizes` or `scales` **tensor**
//! input, whose concrete values [`resize_params_from_node`] must be handed
//! directly (the `lib.rs` dispatch arm resolves that tensor from `weights`/
//! `intermediates` exactly as it already resolves a `Conv`'s weight/bias).
//!
//! ## Dispatch rule
//!
//! [`resize_params_from_node`] is a whitelist over `mode`,
//! `coordinate_transformation_mode`, and `nearest_mode`, matching the exact
//! coordinate formula each `oxicuda_dnn` kernel computes — the same
//! discipline [`crate::conv::conv_params_from_attrs`] and
//! [`crate::pool::pool_params_from_attrs`] use, for the identical reason.
//!
//! * **Rank** — input must be 4-D NCHW; `N` and `C` must be unchanged by the
//!   resize (only `H`/`W` may scale). This workspace's two Resize-bearing
//!   models (`det_10g.onnx`'s FPN upsamples, `inswapper_128.onnx`'s decoder
//!   upsamples) both resize exactly `[H, W]` and never touch `N`/`C`.
//! * **`axes`** — must be absent. Both real models omit it (the default,
//!   "every axis in order"), and a `sizes`/`scales` tensor is then required
//!   to carry exactly 4 entries.
//! * **`sizes` XOR `scales`** — exactly one of the two must be supplied,
//!   per spec. `sizes` is read as declared dimensions directly (matching
//!   `keep_aspect_ratio_policy = "stretch"`, the ONNX default — anything else
//!   declines); `scales` resolves each output dim as
//!   `floor(input_dim * scale)`, matching `oxionnx-ops::resize`'s own
//!   `resolve_plan`.
//! * **`mode = "nearest"`** — requires `coordinate_transformation_mode =
//!   "asymmetric"` and `nearest_mode = "floor"`. Those two together are
//!   exactly `oxicuda_dnn::resize::resize_nearest`'s formula
//!   (`ih = floor(oh * in_h / out_h)`, computed as an *exact* unsigned
//!   integer division — see [`crate::reference::ref_resize`]'s nearest-mode
//!   arm for why that makes the oracle agreement exact, not merely close).
//!   Both of
//!   `det_10g.onnx`'s Resize nodes use precisely this combination.
//! * **`mode = "linear"` / `"bilinear"`** — requires
//!   `coordinate_transformation_mode` to be `"half_pixel"`,
//!   `"pytorch_half_pixel"`, or `"align_corners"`. `resize_bilinear`'s
//!   `align_corners = false` formula is `half_pixel`'s formula exactly; it is
//!   also `pytorch_half_pixel`'s formula whenever the output extent exceeds
//!   `1` on both axes (the only case where the two coordinate modes agree —
//!   see `pytorch_half_pixel`'s definition), so a `pytorch_half_pixel` node
//!   is accepted only then. Both of `inswapper_128.onnx`'s Resize nodes are
//!   `pytorch_half_pixel` 2x upsamples (`out_h`/`out_w` always well above 1),
//!   so this is not a permanent decline for that model either.
//! * **`antialias`, `exclude_outside`** — must be absent/zero; neither kernel
//!   models them.
//!
//! ## Advertised as CUDA-supported
//!
//! [`crate::is_supported_op`] reports `true` for `OpKind::Resize`; a node
//! outside the whitelist above still declines to `Ok(None)` rather than being
//! silently miscomputed. Shadow-verifiable via [`crate::reference::ref_resize`]
//! (one oracle, dispatched on [`ResizeMode`]) through the same
//! `verify_or_fallback` gate every other claimable op uses.

use oxicuda_dnn::resize::{resize_bilinear, resize_nearest};
use oxicuda_dnn::{DnnError, TensorDesc, TensorDescMut};

use oxionnx_core::Attributes;

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Residency slot label for `Resize`'s data operand (input 0). The
/// `sizes`/`scales` tensor is never bound this way — its host bytes are read
/// directly by [`resize_params_from_node`]'s caller, the same treatment
/// [`crate::conv`] gives a convolution's weight/bias.
pub(crate) const INPUT_LABEL: &str = "resize_input";

/// Which interpolation kernel a [`ResizeParams`] describes, with the one
/// per-mode parameter each `oxicuda_dnn` kernel takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// `resize_nearest` — asymmetric coordinate mapping, floor rounding. No
    /// further parameters: the kernel has none.
    Nearest,
    /// `resize_bilinear` — `align_corners` selects between the `"ac"` and
    /// `"noac"` kernel variants (see [`crate::resize`] module docs for which
    /// ONNX `coordinate_transformation_mode`s map to `false`).
    Bilinear {
        /// Corner-alignment mode.
        align_corners: bool,
    },
}

/// Resolved geometry for one `Resize` dispatch: which kernel to run, and the
/// output spatial extent it must produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeParams {
    /// Interpolation kernel and its parameter.
    pub mode: ResizeMode,
    /// Output height.
    pub out_h: usize,
    /// Output width.
    pub out_w: usize,
}

/// Resolves the target `[N, C, H, W]` shape from a `sizes` tensor (already
/// cast to `usize`, ONNX's `keep_aspect_ratio_policy = "stretch"` semantics —
/// the dims are taken verbatim).
#[must_use]
fn shape_from_sizes(sizes: &[f32]) -> Option<[usize; 4]> {
    if sizes.len() != 4 {
        return None;
    }
    let mut out = [0_usize; 4];
    for (slot, &v) in out.iter_mut().zip(sizes) {
        if !v.is_finite() || v < 0.0 || v > usize::MAX as f32 {
            return None;
        }
        *slot = v as usize;
    }
    Some(out)
}

/// Resolves the target `[N, C, H, W]` shape from a `scales` tensor and the
/// input shape: `out_dim = floor(in_dim * scale)`, matching
/// `oxionnx-ops::resize`'s `resolve_plan` (the product evaluated in `f32`,
/// matching onnxruntime).
#[must_use]
fn shape_from_scales(input_shape: &[usize; 4], scales: &[f32]) -> Option<[usize; 4]> {
    if scales.len() != 4 {
        return None;
    }
    let mut out = [0_usize; 4];
    for i in 0..4 {
        let s = scales[i];
        if !s.is_finite() || s <= 0.0 {
            return None;
        }
        let width = input_shape[i] as f32 * s;
        let dim = width.floor();
        if !(0.0..=usize::MAX as f32).contains(&dim) {
            return None;
        }
        out[i] = dim as usize;
    }
    Some(out)
}

/// Builds [`ResizeParams`] for an ONNX `Resize` node from its attributes, its
/// 4-D NCHW input shape, and its already-resolved `sizes`/`scales` operand —
/// or declines. See the [module docs](self) "Dispatch rule" section for the
/// full whitelist.
///
/// Exactly one of `sizes`/`scales` must be `Some`, per the ONNX spec (the
/// caller — `try_cuda_dispatch_resident`'s `Resize` arm — resolves whichever
/// input the node actually supplied and passes `None` for the other).
///
/// Pure and allocation-light: unit-testable without a CUDA device, mirroring
/// [`crate::conv::conv_params_from_attrs`] / [`crate::pool::pool_params_from_attrs`].
#[must_use]
pub fn resize_params_from_node(
    attrs: &Attributes,
    input_shape: &[usize],
    sizes: Option<&[f32]>,
    scales: Option<&[f32]>,
) -> Option<ResizeParams> {
    if input_shape.len() != 4 {
        return None;
    }
    // No axis-subsetting support: both models this crate targets omit `axes`
    // entirely (the default -- every axis, in order), which is the only
    // configuration where a plain 4-entry `sizes`/`scales` tensor is
    // unambiguous without also carrying an axis list.
    if !attrs.ints("axes").is_empty() {
        return None;
    }
    if attrs.i("antialias", 0) != 0 || attrs.i("exclude_outside", 0) != 0 {
        return None;
    }
    if !matches!(attrs.s("keep_aspect_ratio_policy"), "" | "stretch") {
        return None;
    }

    let in_shape: [usize; 4] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let out_shape = match (sizes, scales) {
        (Some(sizes), None) => shape_from_sizes(sizes)?,
        (None, Some(scales)) => shape_from_scales(&in_shape, scales)?,
        // Exactly one of the two must be supplied -- both or neither is a
        // malformed node (or, for "neither", an opset this crate does not
        // model); the CPU operator raises the correct diagnostic either way.
        _ => return None,
    };

    // Only H/W may change: N and C passing through unchanged is what makes
    // this a 2-D image resize rather than a batch/channel reinterpretation
    // `oxicuda_dnn`'s NCHW-only kernels have no way to express.
    if out_shape[0] != in_shape[0] || out_shape[1] != in_shape[1] {
        return None;
    }
    let (in_h, in_w) = (in_shape[2], in_shape[3]);
    let (out_h, out_w) = (out_shape[2], out_shape[3]);
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return None;
    }

    let raw_mode = attrs.s("mode");
    let mode_name = if raw_mode.is_empty() {
        "nearest"
    } else {
        raw_mode
    };
    let raw_coord = attrs.s("coordinate_transformation_mode");
    let coord = if raw_coord.is_empty() {
        "half_pixel"
    } else {
        raw_coord
    };

    let mode = match mode_name {
        "nearest" => {
            let raw_nearest = attrs.s("nearest_mode");
            let nearest_mode = if raw_nearest.is_empty() {
                "round_prefer_floor"
            } else {
                raw_nearest
            };
            if coord != "asymmetric" || nearest_mode != "floor" {
                return None;
            }
            ResizeMode::Nearest
        }
        "linear" | "bilinear" => {
            let align_corners = match coord {
                "half_pixel" => false,
                // Agrees with `half_pixel`'s formula only when both spatial
                // outputs exceed 1 -- see the module docs.
                "pytorch_half_pixel" if out_h > 1 && out_w > 1 => false,
                "align_corners" => true,
                _ => return None,
            };
            ResizeMode::Bilinear { align_corners }
        }
        // "cubic"/anything else: no kernel.
        _ => return None,
    };

    Some(ResizeParams { mode, out_h, out_w })
}

/// Maps an `oxicuda_dnn` failure into this crate's dispatch error type.
fn dnn_err(e: DnnError) -> CudaDispatchError {
    CudaDispatchError::Dnn(e.to_string())
}

/// ONNX `Resize` forward on the GPU, over an operand that may already be on
/// the device, leaving the result there when the caller asks for it.
///
/// Mirrors [`crate::pool::cuda_pool_bound`]'s shape: a single operand, no
/// epilogue, so the requested `placement` is always honoured once the node is
/// claimed.
///
/// # Returns
/// * `Ok(Some(_))` — computed on the GPU.
/// * `Ok(None)` — not accelerated; see the [module docs](self).
/// * `Err(_)` — a real failure after dispatch was already committed to.
///
/// # Errors
/// See "Returns" above.
pub(crate) fn cuda_resize_bound(
    ctx: &CudaContext,
    input: InputBinding<'_>,
    input_shape: &[usize],
    params: &ResizeParams,
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    if input_shape.len() != 4 {
        return Ok(None);
    }
    let n = input_shape[0];
    let c = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];
    if n == 0 || c == 0 || in_h == 0 || in_w == 0 {
        return Ok(None);
    }

    let (Some(in_needed), Some(out_needed)) = (
        n.checked_mul(c)
            .and_then(|v| v.checked_mul(in_h))
            .and_then(|v| v.checked_mul(in_w)),
        n.checked_mul(c)
            .and_then(|v| v.checked_mul(params.out_h))
            .and_then(|v| v.checked_mul(params.out_w)),
    ) else {
        return Ok(None);
    };
    if input.len() < in_needed {
        return Ok(None);
    }

    let (Ok(n_u32), Ok(c_u32), Ok(in_h_u32), Ok(in_w_u32), Ok(out_h_u32), Ok(out_w_u32)) = (
        u32::try_from(n),
        u32::try_from(c),
        u32::try_from(in_h),
        u32::try_from(in_w),
        u32::try_from(params.out_h),
        u32::try_from(params.out_w),
    ) else {
        return Ok(None);
    };

    let stream = ctx.dnn.stream();
    let Some(mut d_input) = input.bind(ctx, INPUT_LABEL, in_needed, stream)? else {
        return Ok(None);
    };
    let mut d_output = ctx.scratch(out_needed)?;
    // No zero-fill: every one of the `out_needed` output elements is written
    // by exactly one thread (see `cuda_pool_bound`'s identical reasoning).

    let in_desc = TensorDesc::<f32>::nchw(d_input.buffer(), n_u32, c_u32, in_h_u32, in_w_u32)
        .map_err(dnn_err)?;
    let mut out_desc =
        TensorDescMut::<f32>::nchw(d_output.buffer_mut(), n_u32, c_u32, out_h_u32, out_w_u32)
            .map_err(dnn_err)?;

    match params.mode {
        ResizeMode::Nearest => {
            resize_nearest::<f32>(&ctx.dnn, &in_desc, &mut out_desc).map_err(dnn_err)?;
        }
        ResizeMode::Bilinear { align_corners } => {
            resize_bilinear::<f32>(&ctx.dnn, &in_desc, &mut out_desc, align_corners)
                .map_err(dnn_err)?;
        }
    }

    let out_shape = vec![n, c, params.out_h, params.out_w];
    let out = finish_output(ctx, d_output, out_needed, &out_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => d_input.retire(),
        KernelOutput::Device(_) => retire_queued(ctx, &mut d_input),
    }
    Ok(Some(out))
}

/// [`cuda_resize_bound`] over plain host slices, always reading the result
/// back. The non-resident entry point this module's own tests use.
///
/// # Errors
/// As [`cuda_resize_bound`].
#[must_use = "the resize result is only computed if this is consumed"]
pub fn cuda_resize(
    ctx: &CudaContext,
    input: &[f32],
    input_shape: &[usize],
    params: &ResizeParams,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    match cuda_resize_bound(
        ctx,
        InputBinding::Host(input),
        input_shape,
        params,
        CudaOutputPlacement::Host,
    )? {
        Some(KernelOutput::Host(data)) => Ok(Some(data)),
        Some(KernelOutput::Device(_)) => Err(CudaDispatchError::Shape {
            op: "Resize",
            msg: "host placement produced a device-resident result".to_string(),
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_str(pairs: &[(&str, &str)]) -> Attributes {
        let mut a = Attributes::default();
        for (k, v) in pairs {
            a.strings.insert((*k).to_string(), (*v).to_string());
        }
        a
    }

    // ── shape resolution ────────────────────────────────────────────────────

    #[test]
    fn sizes_are_taken_verbatim() {
        let sizes = [1.0_f32, 56.0, 320.0, 320.0];
        assert_eq!(shape_from_sizes(&sizes), Some([1, 56, 320, 320]));
    }

    #[test]
    fn scales_floor_the_scaled_dimension() {
        let scales = [1.0_f32, 1.0, 2.0, 2.0];
        assert_eq!(
            shape_from_scales(&[1, 3, 64, 64], &scales),
            Some([1, 3, 128, 128])
        );
    }

    #[test]
    fn a_negative_scale_declines() {
        let scales = [1.0_f32, 1.0, -2.0, 2.0];
        assert!(shape_from_scales(&[1, 3, 64, 64], &scales).is_none());
    }

    // ── resize_params_from_node: nearest (SCRFD FPN) ────────────────────────

    #[test]
    fn scrfd_style_nearest_asymmetric_floor_upsample_is_claimed() {
        let attrs = attrs_str(&[
            ("mode", "nearest"),
            ("coordinate_transformation_mode", "asymmetric"),
            ("nearest_mode", "floor"),
        ]);
        let sizes = [1.0_f32, 56.0, 40.0, 40.0];
        let params = resize_params_from_node(&attrs, &[1, 56, 20, 20], Some(&sizes), None)
            .expect("asymmetric+floor nearest must be claimable");
        assert_eq!(params.mode, ResizeMode::Nearest);
        assert_eq!(params.out_h, 40);
        assert_eq!(params.out_w, 40);
    }

    #[test]
    fn nearest_with_the_default_round_prefer_floor_declines() {
        // Default `nearest_mode` (attribute absent) does not match the
        // kernel's floor-only formula.
        let attrs = attrs_str(&[
            ("mode", "nearest"),
            ("coordinate_transformation_mode", "asymmetric"),
        ]);
        let sizes = [1.0_f32, 1.0, 40.0, 40.0];
        assert!(resize_params_from_node(&attrs, &[1, 1, 20, 20], Some(&sizes), None).is_none());
    }

    #[test]
    fn nearest_half_pixel_declines() {
        let attrs = attrs_str(&[("mode", "nearest"), ("nearest_mode", "floor")]);
        let sizes = [1.0_f32, 1.0, 40.0, 40.0];
        assert!(resize_params_from_node(&attrs, &[1, 1, 20, 20], Some(&sizes), None).is_none());
    }

    // ── resize_params_from_node: bilinear (InSwapper decoder) ───────────────

    #[test]
    fn inswapper_style_pytorch_half_pixel_2x_upsample_is_claimed() {
        let attrs = attrs_str(&[
            ("mode", "linear"),
            ("coordinate_transformation_mode", "pytorch_half_pixel"),
        ]);
        let scales = [1.0_f32, 1.0, 2.0, 2.0];
        let params = resize_params_from_node(&attrs, &[1, 128, 32, 32], None, Some(&scales))
            .expect("pytorch_half_pixel bilinear 2x upsample must be claimable");
        assert_eq!(
            params.mode,
            ResizeMode::Bilinear {
                align_corners: false
            }
        );
        assert_eq!(params.out_h, 64);
        assert_eq!(params.out_w, 64);
    }

    #[test]
    fn half_pixel_bilinear_is_claimed_as_not_align_corners() {
        let attrs = attrs_str(&[("mode", "bilinear")]); // default coord = half_pixel
        let sizes = [1.0_f32, 3.0, 128.0, 128.0];
        let params = resize_params_from_node(&attrs, &[1, 3, 64, 64], Some(&sizes), None)
            .expect("half_pixel is the ONNX default coordinate mode");
        assert_eq!(
            params.mode,
            ResizeMode::Bilinear {
                align_corners: false
            }
        );
    }

    #[test]
    fn align_corners_bilinear_is_claimed() {
        let attrs = attrs_str(&[
            ("mode", "linear"),
            ("coordinate_transformation_mode", "align_corners"),
        ]);
        let sizes = [1.0_f32, 3.0, 128.0, 128.0];
        let params = resize_params_from_node(&attrs, &[1, 3, 64, 64], Some(&sizes), None)
            .expect("align_corners must be claimable");
        assert_eq!(
            params.mode,
            ResizeMode::Bilinear {
                align_corners: true
            }
        );
    }

    #[test]
    fn pytorch_half_pixel_declines_when_an_output_axis_is_1() {
        let attrs = attrs_str(&[
            ("mode", "linear"),
            ("coordinate_transformation_mode", "pytorch_half_pixel"),
        ]);
        let sizes = [1.0_f32, 3.0, 1.0, 128.0];
        assert!(resize_params_from_node(&attrs, &[1, 3, 64, 64], Some(&sizes), None).is_none());
    }

    #[test]
    fn cubic_mode_declines() {
        let attrs = attrs_str(&[("mode", "cubic")]);
        let sizes = [1.0_f32, 3.0, 128.0, 128.0];
        assert!(resize_params_from_node(&attrs, &[1, 3, 64, 64], Some(&sizes), None).is_none());
    }

    // ── general shape/axis rules ─────────────────────────────────────────────

    #[test]
    fn changing_n_or_c_declines() {
        let attrs = attrs_str(&[
            ("mode", "nearest"),
            ("coordinate_transformation_mode", "asymmetric"),
            ("nearest_mode", "floor"),
        ]);
        let sizes = [1.0_f32, 2.0, 40.0, 40.0]; // C: 1 -> 2
        assert!(resize_params_from_node(&attrs, &[1, 1, 20, 20], Some(&sizes), None).is_none());
    }

    #[test]
    fn both_sizes_and_scales_present_declines() {
        let attrs = attrs_str(&[
            ("mode", "nearest"),
            ("coordinate_transformation_mode", "asymmetric"),
            ("nearest_mode", "floor"),
        ]);
        let sizes = [1.0_f32, 1.0, 40.0, 40.0];
        let scales = [1.0_f32, 1.0, 2.0, 2.0];
        assert!(
            resize_params_from_node(&attrs, &[1, 1, 20, 20], Some(&sizes), Some(&scales)).is_none()
        );
    }

    #[test]
    fn neither_sizes_nor_scales_declines() {
        let attrs = attrs_str(&[
            ("mode", "nearest"),
            ("coordinate_transformation_mode", "asymmetric"),
            ("nearest_mode", "floor"),
        ]);
        assert!(resize_params_from_node(&attrs, &[1, 1, 20, 20], None, None).is_none());
    }

    #[test]
    fn non_default_axes_declines() {
        let mut attrs = attrs_str(&[
            ("mode", "nearest"),
            ("coordinate_transformation_mode", "asymmetric"),
            ("nearest_mode", "floor"),
        ]);
        attrs.int_lists.insert("axes".into(), vec![2, 3]);
        let sizes = [40.0_f32, 40.0];
        assert!(resize_params_from_node(&attrs, &[1, 1, 20, 20], Some(&sizes), None).is_none());
    }

    #[test]
    fn non_4d_input_declines() {
        let attrs = attrs_str(&[
            ("mode", "nearest"),
            ("coordinate_transformation_mode", "asymmetric"),
            ("nearest_mode", "floor"),
        ]);
        let sizes = [40.0_f32, 40.0];
        assert!(resize_params_from_node(&attrs, &[20, 20], Some(&sizes), None).is_none());
    }
}
