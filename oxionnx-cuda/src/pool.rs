//! CUDA `MaxPool` / `AveragePool` dispatch.
//!
//! # Status
//!
//! Both ops dispatch straight to `oxicuda-dnn`'s existing forward-pooling
//! kernels — [`max_pool2d`](oxicuda_dnn::pool::max_pool2d) and
//! [`avg_pool2d`](oxicuda_dnn::pool::avg_pool2d) — which sat unused in this
//! workspace before this module: nothing in `oxionnx-cuda` called them.
//! [`cuda_pool_bound`] is the shared dispatch body both `OpKind::MaxPool` and
//! `OpKind::AveragePool` arms in `lib.rs` route through, mirroring the
//! `TensorDesc`/`TensorDescMut`/`DnnHandle` convention [`crate::conv`]
//! already established for `Conv`.
//!
//! ## Dispatch rule
//!
//! [`pool_params_from_attrs`] is a whitelist, the same discipline
//! [`crate::conv::conv_params_from_attrs`] uses and for the identical reason
//! (see that module's docs for what silently ignoring an attribute cost
//! there): every attribute that changes what a `MaxPool`/`AveragePool` node
//! computes is either modelled here or causes a decline.
//!
//! * `kernel_shape` — required, exactly 2 entries (this dispatch is 2-D
//!   only, matching `oxicuda_dnn::pool`'s kernels).
//! * `strides` — defaults to `[1, 1]`.
//! * `dilations` — must be absent or exactly `[1, 1]`. Neither
//!   `max_pool2d`/`avg_pool2d`'s PTX kernel bodies model dilation at all
//!   (unlike [`crate::conv`]'s `ImplicitGemmConv`), so a dilated pool
//!   declines rather than silently computing the un-dilated window.
//! * `pads` — `[top, left, bottom, right]`; declines when asymmetric
//!   (`pads[0] != pads[2] || pads[1] != pads[3]`), the same rule
//!   [`crate::conv::problem_from_params`] applies, because
//!   `oxicuda_dnn::types::TensorDesc` (via the pool kernels' `(u32, u32)`
//!   padding parameter) has no representation for asymmetric padding either.
//! * `auto_pad` — must be absent or `"NOTSET"`. Neither pooling op in this
//!   workspace's two target models (`det_10g.onnx`'s four MaxPool/AveragePool
//!   nodes) uses `auto_pad`, so — unlike [`crate::conv`], which resolves
//!   `SAME_UPPER`/`SAME_LOWER` explicitly — this module simply declines them
//!   rather than adding an unexercised code path.
//! * `ceil_mode` — modelled *without* a `ceil_mode`-aware kernel: `oxicuda_dnn`'s
//!   pooling kernels are floor-mode only (see
//!   [`oxicuda_dnn::pool_output_size`]'s own doc comment). [`problem_from_params`]
//!   computes **both** the floor-mode extent the kernel will actually produce
//!   and the true `ceil_mode`-aware extent (via a from-scratch reimplementation
//!   of the ONNX `pool_out_dim` formula, including its "drop a trailing window
//!   that starts in the right padding" correction — mirrored from
//!   `oxionnx-ops::conv::spatial::pool_out_dim`, not called from it: see
//!   [`mod@crate::reference`]'s "why this does not depend on `oxionnx-ops`") and
//!   declines whenever they disagree. They agree whenever the padded input
//!   divides evenly by the stride — true for every pooling node in this
//!   workspace's models, where `kernel == stride` and the input is always a
//!   power-of-two-derived feature-map extent — so this is a real acceleration
//!   for the models that matter here, not a permanent decline in disguise.
//! * `count_include_pad` (`AveragePool` only) — passed straight through to
//!   `avg_pool2d`, which already supports both variants.
//! * `storage_order` and a requested `Indices` second output (`MaxPool` only)
//!   are declined by the `lib.rs` dispatch arm before this module is ever
//!   reached, because `max_pool2d`'s index encoding
//!   (`hw_off` — the flat offset within one channel's `H*W` plane) does not
//!   match `oxionnx-ops`' CPU encoding (a full flattened `N*C*H*W` index, with
//!   an alternate column-major form for `storage_order=1`), and no model in
//!   this workspace requests it.
//!
//! ## Advertised as CUDA-supported
//!
//! [`crate::is_supported_op`] reports `true` for `OpKind::MaxPool` and
//! `OpKind::AveragePool`; a node whose configuration is not modelled above
//! still declines to `Ok(None)` rather than being silently miscomputed — see
//! [`crate::is_supported_op`]'s "Necessary, not sufficient". Both arms are
//! shadow-verifiable: [`crate::reference::ref_pool`] (one oracle, dispatched
//! on [`PoolKind`] exactly like this module's own `kind` parameter) backs the
//! same `verify_or_fallback` gate every other claimable op in this crate uses.

use oxicuda_dnn::pool::{avg_pool2d, max_pool2d};
use oxicuda_dnn::{DnnError, TensorDesc, TensorDescMut};

use oxionnx_core::Attributes;

use crate::activation::{
    finish_output, retire_queued, CudaOutputPlacement, InputBinding, KernelOutput,
};
use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Residency slot label for a pooling node's activation.
///
/// Pooling has no invariant operand at all (no weight, no bias) — the single
/// input is always this frame's data — but the binding API takes a label
/// uniformly, and a name distinct from every other op's keeps it structurally
/// impossible to collide with an unrelated kernel slot.
pub(crate) const INPUT_LABEL: &str = "pool_input";

/// Which pooling reduction a dispatch computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    /// `MaxPool`: the maximum of each window.
    Max,
    /// `AveragePool`: the mean of each window.
    Avg,
}

/// Resolved 2-D pooling geometry, extracted from an ONNX `MaxPool`/`AveragePool`
/// node's attributes by [`pool_params_from_attrs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolParams {
    /// Pooling window `[kh, kw]`.
    pub kernel: [usize; 2],
    /// Stride for `[H, W]`.
    pub strides: [usize; 2],
    /// Padding `[pad_top, pad_left, pad_bottom, pad_right]`.
    pub pads: [usize; 4],
    /// ONNX `ceil_mode` attribute — consulted only by [`problem_from_params`]'s
    /// consistency check; the kernel itself is always floor-mode. See the
    /// [module docs](self) "Dispatch rule" section.
    pub ceil_mode: bool,
    /// `AveragePool`'s `count_include_pad` attribute. Read for every node
    /// (cheap) but meaningless for `MaxPool`, which ignores it.
    pub count_include_pad: bool,
}

/// Reads a 2-entry spatial attribute (`strides` / `dilations`).
///
/// Independent of [`crate::conv`]'s identically-named private helper —
/// duplicated rather than shared across sibling dispatch modules, the same
/// choice [`mod@crate::reference`] makes about not depending on
/// `oxionnx-ops`: a few lines of pure, unit-tested-in-place logic is cheaper
/// to keep correct than a cross-module `pub(crate)` surface two independently
/// evolving dispatch arms would both have to agree not to break.
#[must_use]
fn read_spatial_pair(raw: &[i64], default: usize) -> Option<[usize; 2]> {
    match raw.len() {
        0 => Some([default, default]),
        2 => {
            let h = usize::try_from(raw[0]).ok()?;
            let w = usize::try_from(raw[1]).ok()?;
            (h >= 1 && w >= 1).then_some([h, w])
        }
        _ => None,
    }
}

/// Reads the `pads` attribute as `[top, left, bottom, right]`. See
/// [`read_spatial_pair`] on why this is a local duplicate of `conv`'s helper.
#[must_use]
fn read_pads_quad(raw: &[i64]) -> Option<[usize; 4]> {
    match raw.len() {
        0 => Some([0; 4]),
        4 => {
            let mut out = [0_usize; 4];
            for (slot, &v) in out.iter_mut().zip(raw) {
                *slot = usize::try_from(v).ok()?;
            }
            Some(out)
        }
        _ => None,
    }
}

/// Builds [`PoolParams`] for an ONNX `MaxPool`/`AveragePool` node, or declines
/// it. See the [module docs](self) "Dispatch rule" section for the exact
/// whitelist.
///
/// Pure and allocation-free: unit-testable without a CUDA device, mirroring
/// [`crate::conv::conv_params_from_attrs`].
#[must_use]
pub fn pool_params_from_attrs(attrs: &Attributes) -> Option<PoolParams> {
    let kernel_raw = attrs.ints("kernel_shape");
    if kernel_raw.len() != 2 {
        return None;
    }
    let kh = usize::try_from(kernel_raw[0]).ok().filter(|&v| v >= 1)?;
    let kw = usize::try_from(kernel_raw[1]).ok().filter(|&v| v >= 1)?;

    let strides = read_spatial_pair(attrs.ints("strides"), 1)?;
    let dilations = read_spatial_pair(attrs.ints("dilations"), 1)?;
    if dilations != [1, 1] {
        // Neither pooling kernel in `oxicuda_dnn` has a dilation parameter.
        return None;
    }
    let pads = read_pads_quad(attrs.ints("pads"))?;
    if !matches!(attrs.s("auto_pad"), "" | "NOTSET") {
        return None;
    }
    let ceil_mode = attrs.i("ceil_mode", 0) != 0;
    let count_include_pad = attrs.i("count_include_pad", 0) != 0;

    Some(PoolParams {
        kernel: [kh, kw],
        strides,
        pads,
        ceil_mode,
        count_include_pad,
    })
}

/// The pooling output extent for one axis: `oxicuda_dnn::pool_output_size`'s
/// floor formula, generalised with an optional `ceil_mode` correction.
///
/// A from-scratch reimplementation of the ONNX pooling-extent formula
/// (mirroring, not calling, `oxionnx-ops::conv::spatial::pool_out_dim` — see
/// [`mod@crate::reference`]'s "why this does not depend on `oxionnx-ops`"),
/// including the ceil-mode correction that drops a trailing window whose
/// start falls inside the right-hand padding. `dilation` is always `1` for
/// every caller in this module (a dilated node already declined in
/// [`pool_params_from_attrs`]), so unlike its `oxionnx-ops` cousin this
/// version has no `dilation` parameter at all.
///
/// `None` for a degenerate window (zero stride, or a padded extent smaller
/// than the kernel).
#[must_use]
fn pool_out_dim(
    in_dim: usize,
    pad_begin: usize,
    pad_end: usize,
    kernel: usize,
    stride: usize,
    ceil_mode: bool,
) -> Option<usize> {
    if stride == 0 || kernel == 0 {
        return None;
    }
    let padded = in_dim.checked_add(pad_begin)?.checked_add(pad_end)?;
    if padded < kernel {
        return None;
    }
    let span = padded - kernel;
    let mut out = if ceil_mode {
        span.div_ceil(stride) + 1
    } else {
        span / stride + 1
    };
    if ceil_mode && out > 1 && (out - 1).saturating_mul(stride) >= in_dim + pad_begin {
        out -= 1;
    }
    Some(out)
}

/// Validated shape geometry for one pooling dispatch.
#[derive(Debug, Clone, Copy)]
struct PoolProblem {
    n: usize,
    c: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
}

/// The `[N, C, out_H, out_W]` shape a claimed `MaxPool`/`AveragePool` node
/// produces, or `None` if [`pool_params_from_attrs`]'s output would decline
/// the node — the same question [`cuda_pool_bound`] answers internally via
/// [`problem_from_params`], exposed so a caller that needs the shape *before*
/// (or independent of) dispatching — `lib.rs`'s `MaxPool`/`AveragePool` arm,
/// which must attach a shape to a `Host`-placement result — does not have to
/// re-derive the padding/stride/`ceil_mode` arithmetic itself. Mirrors how
/// [`crate::resize::ResizeParams`]/[`crate::slice::SliceParams`] carry their
/// own output shape as a field; [`PoolParams`] cannot do the same because its
/// output shape also depends on the input shape, which is not part of it.
#[must_use]
pub fn pool_output_shape(input_shape: &[usize], params: &PoolParams) -> Option<[usize; 4]> {
    let problem = problem_from_params(input_shape, params)?;
    Some([problem.n, problem.c, problem.out_h, problem.out_w])
}

/// Resolves [`PoolProblem`] from the ONNX input shape and [`PoolParams`], or
/// declines — see the [module docs](self) "Dispatch rule" section for the
/// full list of what this rejects.
///
/// Pure: unit-testable without a CUDA device, same rationale as
/// [`crate::conv::problem_from_params`].
#[must_use]
fn problem_from_params(input_shape: &[usize], params: &PoolParams) -> Option<PoolProblem> {
    // ONNX `MaxPool`/`AveragePool` are always NCHW for the 2-D case this
    // module claims.
    if input_shape.len() != 4 {
        return None;
    }
    let [pad_top, pad_left, pad_bottom, pad_right] = params.pads;
    // `TensorDesc`'s padding parameter (via the pool kernels' `(u32, u32)`
    // argument) is symmetric-only; decline rather than use one side's value.
    if pad_top != pad_bottom || pad_left != pad_right {
        return None;
    }

    let n = input_shape[0];
    let c = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];
    if n == 0 || c == 0 || in_h == 0 || in_w == 0 {
        return None;
    }

    let [kh, kw] = params.kernel;
    let [sh, sw] = params.strides;

    // What `max_pool2d`/`avg_pool2d` will actually compute: always floor-mode.
    let out_h = pool_out_dim(in_h, pad_top, pad_bottom, kh, sh, false)?;
    let out_w = pool_out_dim(in_w, pad_left, pad_right, kw, sw, false)?;

    if params.ceil_mode {
        // What the ONNX node's declared shape actually requires. Decline
        // unless the two formulas agree -- see the module docs for when
        // (and why) they do for this workspace's real models.
        let true_out_h = pool_out_dim(in_h, pad_top, pad_bottom, kh, sh, true)?;
        let true_out_w = pool_out_dim(in_w, pad_left, pad_right, kw, sw, true)?;
        if true_out_h != out_h || true_out_w != out_w {
            return None;
        }
    }
    if out_h == 0 || out_w == 0 {
        return None;
    }

    Some(PoolProblem {
        n,
        c,
        in_h,
        in_w,
        out_h,
        out_w,
    })
}

/// Maps an `oxicuda_dnn` failure into this crate's dispatch error type.
fn dnn_err(e: DnnError) -> CudaDispatchError {
    CudaDispatchError::Dnn(e.to_string())
}

/// ONNX `MaxPool`/`AveragePool` forward on the GPU, over an operand that may
/// already be on the device, leaving the result there when the caller asks
/// for it.
///
/// Mirrors [`crate::elementwise::cuda_elementwise_bound`]'s shape (a single
/// operand, no bias/activation epilogue to complicate placement) rather than
/// [`crate::conv::cuda_conv_bound`]'s (which has to reconcile three engines
/// and a fused activation): pooling has neither, so the requested `placement`
/// is always honoured when the node itself is claimed.
///
/// # Returns
///
/// * `Ok(Some(_))` — computed on the GPU.
/// * `Ok(None)` — this configuration is not accelerated; see the
///   [module docs](self). The caller falls back to the CPU.
/// * `Err(_)` — a real failure after dispatch was already committed to.
///
/// # Errors
/// See "Returns" above.
pub(crate) fn cuda_pool_bound(
    ctx: &CudaContext,
    input: InputBinding<'_>,
    input_shape: &[usize],
    kind: PoolKind,
    params: &PoolParams,
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
    let Some(problem) = problem_from_params(input_shape, params) else {
        return Ok(None);
    };
    let PoolProblem {
        n,
        c,
        in_h,
        in_w,
        out_h,
        out_w,
    } = problem;

    let (Some(in_needed), Some(out_needed)) = (
        n.checked_mul(c)
            .and_then(|v| v.checked_mul(in_h))
            .and_then(|v| v.checked_mul(in_w)),
        n.checked_mul(c)
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w)),
    ) else {
        return Ok(None);
    };
    if input.len() < in_needed {
        return Ok(None);
    }

    let (
        Ok(n_u32),
        Ok(c_u32),
        Ok(in_h_u32),
        Ok(in_w_u32),
        Ok(out_h_u32),
        Ok(out_w_u32),
        Ok(kh_u32),
        Ok(kw_u32),
        Ok(sh_u32),
        Ok(sw_u32),
        Ok(ph_u32),
        Ok(pw_u32),
    ) = (
        u32::try_from(n),
        u32::try_from(c),
        u32::try_from(in_h),
        u32::try_from(in_w),
        u32::try_from(out_h),
        u32::try_from(out_w),
        u32::try_from(params.kernel[0]),
        u32::try_from(params.kernel[1]),
        u32::try_from(params.strides[0]),
        u32::try_from(params.strides[1]),
        u32::try_from(params.pads[0]),
        u32::try_from(params.pads[1]),
    )
    else {
        return Ok(None);
    };

    // Everything below rides `ctx.dnn.stream()`; see `cuda_conv_bound`'s
    // identical comment for why stream order alone is enough to sequence
    // upload -> kernel -> readback with no fence between them.
    let stream = ctx.dnn.stream();

    let Some(mut d_input) = input.bind(ctx, INPUT_LABEL, in_needed, stream)? else {
        return Ok(None);
    };
    let mut d_output = ctx.scratch(out_needed)?;
    // No zero-fill: the launch geometry covers exactly `[0, out_needed)` and
    // every one of the two pooling kernels writes every element of that
    // range (one thread per output element, unconditionally) -- see
    // `cuda_elementwise_bound`'s identical reasoning.

    let in_desc = TensorDesc::<f32>::nchw(d_input.buffer(), n_u32, c_u32, in_h_u32, in_w_u32)
        .map_err(dnn_err)?;
    let mut out_desc =
        TensorDescMut::<f32>::nchw(d_output.buffer_mut(), n_u32, c_u32, out_h_u32, out_w_u32)
            .map_err(dnn_err)?;

    match kind {
        PoolKind::Max => {
            max_pool2d::<f32>(
                &ctx.dnn,
                &in_desc,
                &mut out_desc,
                None,
                (kh_u32, kw_u32),
                (sh_u32, sw_u32),
                (ph_u32, pw_u32),
            )
            .map_err(dnn_err)?;
        }
        PoolKind::Avg => {
            avg_pool2d::<f32>(
                &ctx.dnn,
                &in_desc,
                &mut out_desc,
                (kh_u32, kw_u32),
                (sh_u32, sw_u32),
                (ph_u32, pw_u32),
                params.count_include_pad,
            )
            .map_err(dnn_err)?;
        }
    }

    let out_shape = vec![n, c, out_h, out_w];
    let out = finish_output(ctx, d_output, out_needed, &out_shape, placement, stream)?;
    match &out {
        KernelOutput::Host(_) => d_input.retire(),
        KernelOutput::Device(_) => retire_queued(ctx, &mut d_input),
    }
    Ok(Some(out))
}

/// [`cuda_pool_bound`] over plain host slices, always reading the result
/// back. The non-resident entry point this module's own tests use.
///
/// # Errors
/// As [`cuda_pool_bound`].
#[must_use = "the pooling result is only computed if this is consumed"]
pub fn cuda_pool(
    ctx: &CudaContext,
    input: &[f32],
    input_shape: &[usize],
    kind: PoolKind,
    params: &PoolParams,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    match cuda_pool_bound(
        ctx,
        InputBinding::Host(input),
        input_shape,
        kind,
        params,
        CudaOutputPlacement::Host,
    )? {
        Some(KernelOutput::Host(data)) => Ok(Some(data)),
        Some(KernelOutput::Device(_)) => Err(CudaDispatchError::Shape {
            op: "MaxPool/AveragePool",
            msg: "host placement produced a device-resident result".to_string(),
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── pool_params_from_attrs ──────────────────────────────────────────────

    #[test]
    fn requires_a_2d_kernel_shape() {
        let attrs = Attributes::default();
        assert!(pool_params_from_attrs(&attrs).is_none());
    }

    #[test]
    fn plain_2x2_stride_2_matches_scrfd_maxpool_9() {
        // Real node from det_10g.onnx: MaxPool_9, kernel=[2,2], strides=[2,2],
        // pads=[0,0,0,0], ceil_mode=0.
        let mut attrs = Attributes::default();
        attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
        attrs.int_lists.insert("strides".into(), vec![2, 2]);
        attrs.int_lists.insert("pads".into(), vec![0, 0, 0, 0]);
        let params = pool_params_from_attrs(&attrs).expect("must accept");
        assert_eq!(
            params,
            PoolParams {
                kernel: [2, 2],
                strides: [2, 2],
                pads: [0, 0, 0, 0],
                ceil_mode: false,
                count_include_pad: false,
            }
        );
    }

    #[test]
    fn ceil_mode_average_pool_matches_scrfd_averagepool_36() {
        // Real node: AveragePool_36, kernel=[2,2], strides=[2,2],
        // pads=[0,0,0,0], ceil_mode=1.
        let mut attrs = Attributes::default();
        attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
        attrs.int_lists.insert("strides".into(), vec![2, 2]);
        attrs.int_lists.insert("pads".into(), vec![0, 0, 0, 0]);
        attrs.ints.insert("ceil_mode".into(), 1);
        let params = pool_params_from_attrs(&attrs).expect("must accept");
        assert!(params.ceil_mode);
    }

    #[test]
    fn dilated_pooling_declines() {
        let mut attrs = Attributes::default();
        attrs.int_lists.insert("kernel_shape".into(), vec![3, 3]);
        attrs.int_lists.insert("dilations".into(), vec![2, 2]);
        assert!(pool_params_from_attrs(&attrs).is_none());
    }

    #[test]
    fn same_upper_auto_pad_declines() {
        let mut attrs = Attributes::default();
        attrs.int_lists.insert("kernel_shape".into(), vec![2, 2]);
        attrs
            .strings
            .insert("auto_pad".into(), "SAME_UPPER".to_string());
        assert!(pool_params_from_attrs(&attrs).is_none());
    }

    // ── pool_out_dim ─────────────────────────────────────────────────────────

    #[test]
    fn floor_mode_matches_oxicuda_dnn_pool_output_size() {
        for (in_dim, pad, k, s) in [(4u32, 0u32, 2u32, 2u32), (5, 1, 3, 1), (7, 0, 2, 2)] {
            let expected = oxicuda_dnn::pool_output_size(in_dim, k, s, pad);
            let got = pool_out_dim(
                in_dim as usize,
                pad as usize,
                pad as usize,
                k as usize,
                s as usize,
                false,
            );
            assert_eq!(
                got,
                expected.map(|v| v as usize),
                "in={in_dim} k={k} s={s} pad={pad}"
            );
        }
    }

    #[test]
    fn ceil_mode_on_an_even_input_matches_floor_mode() {
        // kernel == stride, pad == 0, input divisible by stride: the case
        // every real pooling node in this workspace's models hits.
        for in_dim in [20usize, 40, 80, 160] {
            let floor_out = pool_out_dim(in_dim, 0, 0, 2, 2, false).unwrap();
            let ceil_out = pool_out_dim(in_dim, 0, 0, 2, 2, true).unwrap();
            assert_eq!(floor_out, ceil_out, "in_dim={in_dim}");
        }
    }

    #[test]
    fn ceil_mode_on_an_odd_input_disagrees_with_floor_mode() {
        // 5 -> floor gives (5-2)/2+1 = 2, ceil gives ceil(3/2)+1 = 2+1 = 3,
        // and the "last window starts inside input+pad" check keeps the
        // extra window (start = 2*2 = 4 < 5). The two disagree, so
        // `problem_from_params` must decline this configuration.
        let floor_out = pool_out_dim(5, 0, 0, 2, 2, false).unwrap();
        let ceil_out = pool_out_dim(5, 0, 0, 2, 2, true).unwrap();
        assert_ne!(floor_out, ceil_out);
    }

    #[test]
    fn zero_stride_or_kernel_is_declined() {
        assert!(pool_out_dim(10, 0, 0, 2, 0, false).is_none());
        assert!(pool_out_dim(10, 0, 0, 0, 2, false).is_none());
    }

    // ── problem_from_params ─────────────────────────────────────────────────

    fn plain_params() -> PoolParams {
        PoolParams {
            kernel: [2, 2],
            strides: [2, 2],
            pads: [0, 0, 0, 0],
            ceil_mode: false,
            count_include_pad: false,
        }
    }

    #[test]
    fn accepts_a_well_formed_nchw_input() {
        let problem = problem_from_params(&[1, 56, 160, 160], &plain_params()).expect("accept");
        assert_eq!(problem.n, 1);
        assert_eq!(problem.c, 56);
        assert_eq!(problem.out_h, 80);
        assert_eq!(problem.out_w, 80);
    }

    #[test]
    fn pool_output_shape_matches_problem_from_params() {
        assert_eq!(
            pool_output_shape(&[1, 56, 160, 160], &plain_params()),
            Some([1, 56, 80, 80])
        );
        assert_eq!(pool_output_shape(&[1, 56, 160], &plain_params()), None);
    }

    #[test]
    fn declines_a_non_4d_shape() {
        assert!(problem_from_params(&[1, 56, 160], &plain_params()).is_none());
    }

    #[test]
    fn declines_asymmetric_pads() {
        let mut params = plain_params();
        params.pads = [1, 0, 0, 0];
        assert!(problem_from_params(&[1, 56, 160, 160], &params).is_none());
    }

    #[test]
    fn declines_a_zero_sized_dimension() {
        assert!(problem_from_params(&[1, 56, 0, 160], &plain_params()).is_none());
    }

    #[test]
    fn ceil_mode_accepts_when_it_agrees_with_floor_mode() {
        let mut params = plain_params();
        params.ceil_mode = true;
        // 160 is evenly divisible by stride 2 -- floor and ceil agree.
        let problem = problem_from_params(&[1, 56, 160, 160], &params).expect("accept");
        assert_eq!(problem.out_h, 80);
    }

    #[test]
    fn ceil_mode_declines_when_it_disagrees_with_floor_mode() {
        let mut params = plain_params();
        params.ceil_mode = true;
        // 5 is not evenly divisible by stride 2 (kernel 2): floor gives 2,
        // ceil gives 3 -- must decline rather than silently pick one.
        assert!(problem_from_params(&[1, 1, 5, 5], &params).is_none());
    }
}
