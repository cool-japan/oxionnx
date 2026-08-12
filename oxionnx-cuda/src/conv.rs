//! CUDA 2-D convolution dispatch.
//!
//! # Status
//!
//! [`cuda_conv`] dispatches ONNX `Conv` nodes directly to three of
//! `oxicuda-dnn`'s forward-convolution engines —
//! [`Conv1x1`](oxicuda_dnn::conv::fprop::direct::Conv1x1),
//! [`DepthwiseConv`](oxicuda_dnn::conv::fprop::direct::DepthwiseConv), and
//! [`ImplicitGemmConv`](oxicuda_dnn::conv::fprop::implicit_gemm::ImplicitGemmConv)
//! — each independently validated on real hardware (`cargo test -p
//! oxicuda-dnn --features gpu-tests conv`: 396 passed / 0 failed on this
//! machine's RTX A4000, sm_86, including the numeric-CPU-oracle tests for
//! all three engines: `conv1x1_f32_nchw_matches_cpu`, the depthwise
//! strided/dilated f32/f64 cases, and the implicit-GEMM 3x3
//! general/strided/dilated/grouped/bias NCHW+NHWC f32/f64 cases). For a
//! shape it recognises (see "Dispatch rule" below) it uploads, launches,
//! downloads, and returns a real answer; for everything else it returns
//! `Ok(None)` so the caller falls back to the CPU (or wgpu).
//!
//! ## Why not `oxicuda_dnn::conv::api::conv_forward`
//!
//! The natural-looking implementation would hand a
//! [`ConvProblem`](oxicuda_dnn::conv::descriptor::ConvProblem) to
//! `conv_forward` and let its `algo_select::select_algorithm` auto-selector
//! pick an engine. That must not be used here: for some shapes the
//! auto-selector routes into the Winograd fprop path, whose safety for
//! every eligible shape is tracked and gated by a separate, sibling fix in
//! this same investigation. Rather than make this dispatch's correctness
//! depend on whether that gate has landed, [`cuda_conv`] bypasses
//! `conv_forward`/`select_algorithm` entirely and calls the three engines
//! above directly, by hand, from its own small dispatch rule — see
//! [`pick_engine`]. None of the three engines this function calls ever
//! touch the Winograd path, so this dispatch is correct regardless of the
//! sibling fix's state.
//!
//! ## Dispatch rule
//!
//! Given the [`ConvProblem`](oxicuda_dnn::conv::descriptor::ConvProblem)
//! built from [`ConvParams`] and the ONNX input/filter shapes:
//!
//! 1. `filter == 1x1 && stride == [1,1] && dilation == [1,1] && padding ==
//!    [0,0]` → [`Conv1x1`](oxicuda_dnn::conv::fprop::direct::Conv1x1).
//! 2. `groups == in_channels && groups == out_channels` (true depthwise,
//!    checked *after* rule 1 — a filter that is simultaneously 1x1 *and*
//!    depthwise-shaped still takes the
//!    [`Conv1x1`](oxicuda_dnn::conv::fprop::direct::Conv1x1) path) →
//!    [`DepthwiseConv`](oxicuda_dnn::conv::fprop::direct::DepthwiseConv).
//! 3. Everything else →
//!    [`ImplicitGemmConv`](oxicuda_dnn::conv::fprop::implicit_gemm::ImplicitGemmConv),
//!    the general-purpose engine (arbitrary padding, stride, dilation and
//!    grouping).
//!
//! See [`pick_engine`] for the exact, unit-tested-without-a-GPU
//! implementation.
//!
//! Neither [`Conv1x1::execute`](oxicuda_dnn::conv::fprop::direct::Conv1x1)
//! nor [`DepthwiseConv::execute`](oxicuda_dnn::conv::fprop::direct::DepthwiseConv)
//! accepts a bias argument at all — their `execute` methods hard-code a
//! null bias pointer internally, with no parameter to override it. When the
//! ONNX node supplies a bias and dispatch takes either of those two paths,
//! [`cuda_conv`] adds it on the host after the download instead (see
//! [`add_bias_nchw`]). `ImplicitGemmConv` does support a native bias
//! epilogue in its kernel and is given the bias descriptor directly.
//!
//! ## What still declines
//!
//! [`ConvProblem`]'s padding is symmetric-only (one value per spatial
//! dimension); ONNX's `pads` attribute is `[pad_top, pad_left, pad_bottom,
//! pad_right]` and can be asymmetric. Rather than silently use only one
//! side's padding value, [`cuda_conv`] declines (`Ok(None)`) whenever
//! `pads[0] != pads[2] || pads[1] != pads[3]`. It also declines: a non-4-D
//! input/filter shape; a `group` that does not divide `in_channels` or
//! `out_channels` evenly; a filter whose declared per-group input-channel
//! count disagrees with the input tensor; a zero-sized dimension; a filter
//! that does not fit the (padded) input; and a `Tensor` whose `data` is
//! shorter than its declared `shape` promises (release builds do not
//! validate that invariant at construction — see
//! [`oxionnx_core::Tensor::new`]'s own doc comment). Every one of these
//! means "this configuration is not accelerated", not "the model is
//! broken" — the CPU operator one frame up computes it correctly (or
//! raises the right diagnostic) either way. See [`problem_from_params`].
//!
//! ## Advertised as CUDA-supported
//!
//! [`crate::is_supported_op`] reports `true` for `OpKind::Conv`, so
//! `oxionnx::execution_providers::decide_placement` routes convolutions here
//! and `oxionnx`'s session runners reach this function on the hot path. It is
//! production dispatch, not a direct-call-only escape hatch.
//!
//! Being advertised is a claim about the op *kind*, not a promise about every
//! node: the configurations listed under "What still declines" above still
//! come back `Ok(None)` and are computed by the CPU operator one frame up —
//! exactly as an over-wide `Softmax` row or a broadcasting `Add` already do.
//! Callers must handle `Ok(None)` from a `Conv` node the same way they
//! already handle it from every other advertised op (see
//! [`crate::is_supported_op`]'s "Necessary, not sufficient").
//!
//! The advertisement rests on the same footing as every other claimable op in
//! this crate, not a weaker one: this arm's output is shadow-verifiable
//! against a [`crate::reference`] CPU oracle.
//! [`crate::reference::ref_conv`] is that oracle, and
//! [`crate::try_cuda_dispatch`]'s `Conv` arm hands it to the same
//! `verify_or_fallback` gate every other arm uses — so `OXIONNX_CUDA_VERIFY=1`
//! shadow-verifies a `Conv` dispatch element by element, discarding a
//! mismatching GPU result in favour of the CPU under the default
//! [`FailurePolicy::Fallback`](crate::context::FailurePolicy) and raising a
//! hard error under `OXIONNX_CUDA_STRICT=1`.
//!
//! ## Validation
//!
//! Numerically checked on this machine's RTX A4000 three different ways.
//! First, directly against each of the three dispatched engines
//! (`Conv1x1`/`DepthwiseConv`/`ImplicitGemmConv`) via an independent,
//! `f64`-accumulated, from-scratch NCHW cross-correlation reference written
//! in this module's own `tests` submodule (deliberately *not*
//! `crate::reference::ref_conv` — two independently-written oracles that
//! agree are stronger evidence than one oracle checked against itself): a
//! 1x1 conv with and without bias (exercising both `Conv1x1` and the
//! host-side bias add), a dilated depthwise conv with bias (exercising
//! `DepthwiseConv` and the same host-side bias add), and 3x3 stride-1 and
//! stride-2 convolutions with bias (exercising `ImplicitGemmConv`'s native
//! bias epilogue). Run with `cargo test -p oxionnx-cuda --features
//! gpu-tests conv` on a CUDA-capable host. Second, end-to-end through
//! [`crate::try_cuda_dispatch`] itself with `OXIONNX_CUDA_VERIFY=1` live in
//! the process environment, against [`crate::reference::ref_conv`] — see
//! `conv_verify_path_agrees_live_on_real_hardware` in
//! `oxionnx-cuda/tests/verify_path_gpu.rs`. Third, along the full production
//! sequence this advertisement switches on — [`crate::is_supported_op`]
//! pre-filter, then [`crate::try_cuda_dispatch`], then an
//! [`crate::reference::ref_conv`] comparison — plus the complementary case
//! that an advertised-but-undispatchable configuration (asymmetric `pads`)
//! is still declined rather than silently computed with the wrong padding:
//! `conv_claimed_by_the_pre_filter_is_actually_dispatched` and
//! `advertised_conv_still_declines_the_configurations_it_cannot_compute` in
//! `lib.rs`'s `gpu-tests` module.

use oxicuda_dnn::conv::descriptor::ConvProblem;
use oxicuda_dnn::conv::fprop::direct::{Conv1x1, DepthwiseConv};
use oxicuda_dnn::conv::fprop::implicit_gemm::ImplicitGemmConv;
use oxicuda_dnn::{DnnError, TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_ptx::ir::PtxType;

use oxionnx_core::Tensor;

use crate::context::CudaContext;
use crate::error::CudaDispatchError;
use crate::residency::{Operand, WeightId};

/// Residency slot label for a convolution's filter.
///
/// Part of a cached operand's identity, so a name cached as a convolution's
/// weight can never be served to a kernel asking for a GEMM's `B`. See
/// [`crate::residency::WeightId`].
pub(crate) const WEIGHT_LABEL: &str = "conv_weight";

/// Residency slot label for a convolution's bias.
pub(crate) const BIAS_LABEL: &str = "conv_bias";

/// The identities of a convolution's invariant operands, as the dispatch layer
/// knows them.
///
/// `None` in either slot means "these bytes are not a graph initializer;
/// upload them for this dispatch only" — which is what a caller with no
/// residency information passes, and what [`cuda_conv`]'s pre-residency
/// behaviour was for every operand.
///
/// The *input activation* has no slot here on purpose: it is this frame's
/// data, so caching it would serve the previous frame's numbers.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ConvWeightIds<'a> {
    /// Identity of the filter tensor.
    pub(crate) weight: Option<WeightId<'a>>,
    /// Identity of the bias tensor.
    pub(crate) bias: Option<WeightId<'a>>,
}

/// Grouped convolution parameters extracted from ONNX node attributes.
///
/// Constructed by [`crate::try_cuda_dispatch`] from the node's `strides`,
/// `pads`, `dilations`, and `group` attributes and handed to [`cuda_conv`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvParams {
    /// Stride for `[H, W]`.
    pub strides: [usize; 2],
    /// Padding for `[pad_top, pad_left, pad_bottom, pad_right]`.
    pub pads: [usize; 4],
    /// Dilation for `[H, W]`.
    pub dilations: [usize; 2],
    /// Convolution groups.
    pub group: usize,
}

/// Which validated `oxicuda-dnn` forward-convolution engine a given
/// [`ConvProblem`] should be dispatched to. See the [module docs](self)
/// "Dispatch rule" section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConvEngine {
    /// [`Conv1x1`] — unpadded, unit-stride, unit-dilation 1x1 filter.
    Conv1x1,
    /// [`DepthwiseConv`] — `groups == in_channels == out_channels`.
    Depthwise,
    /// [`ImplicitGemmConv`] — the general-purpose fallback.
    ImplicitGemm,
}

/// Chooses which of the three validated engines a [`ConvProblem`] must be
/// dispatched to.
///
/// Pure and allocation-free, so this — the actual decision this module
/// exists to make — is unit-testable on any host, including one with no
/// CUDA device. Order matters: rule 1 is checked before rule 2, so a
/// problem that is simultaneously 1x1-unpadded-unit-stride *and*
/// depthwise-shaped takes the [`ConvEngine::Conv1x1`] path, never
/// [`ConvEngine::Depthwise`].
#[must_use]
fn pick_engine(problem: &ConvProblem) -> ConvEngine {
    let unpadded_unit_1x1 = problem.is_1x1() && problem.padding.iter().all(|&p| p == 0);
    if unpadded_unit_1x1 {
        ConvEngine::Conv1x1
    } else if problem.is_depthwise() {
        ConvEngine::Depthwise
    } else {
        ConvEngine::ImplicitGemm
    }
}

/// Attempts to build a validated [`ConvProblem`] from this crate's
/// [`ConvParams`] and the ONNX-supplied input/filter shapes.
///
/// Returns `None` for anything [`cuda_conv`] must decline rather than
/// compute — see the [module docs](self) "What still declines" section for
/// the full list. None of these are hard errors: they mean "this
/// configuration isn't one we accelerate", and the caller falls back to
/// the CPU operator, which raises the correct diagnostic for a genuinely
/// malformed model.
///
/// Pure and allocation-light (two small `Vec<u32>`s for the returned
/// problem's dims): unit-testable without a CUDA device, same rationale as
/// [`pick_engine`].
#[must_use]
fn problem_from_params(
    input_shape: &[usize],
    weight_shape: &[usize],
    params: &ConvParams,
) -> Option<ConvProblem> {
    // ONNX `Conv` is always NCHW / `[K, C/g, R, S]`: both tensors are rank 4.
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return None;
    }

    // `ConvProblem::padding` is symmetric-only (one value per spatial
    // dim); ONNX's [top, left, bottom, right] can disagree. Decline rather
    // than silently use only one side's value.
    let [pad_top, pad_left, pad_bottom, pad_right] = params.pads;
    if pad_top != pad_bottom || pad_left != pad_right {
        return None;
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];
    let out_channels = weight_shape[0];
    let filter_h = weight_shape[2];
    let filter_w = weight_shape[3];

    // A zero-sized dimension is degenerate (an empty tensor); decline
    // rather than risk a zero-sized/zero-block GPU launch. `ConvProblem`'s
    // own `validate()` (called below) does not check this on its own.
    if batch == 0 || in_channels == 0 || in_h == 0 || in_w == 0 {
        return None;
    }
    if out_channels == 0 || filter_h == 0 || filter_w == 0 {
        return None;
    }

    let group = params.group;
    if group == 0 || in_channels % group != 0 || out_channels % group != 0 {
        return None;
    }
    // The filter's declared per-group input-channel dim must agree with
    // what the input tensor's channel count and `group` imply. Nothing
    // else checks this — `ConvProblem` does not retain the filter's own
    // channel-per-group dim at all, it is implicit in `in_channels/groups`.
    if weight_shape[1] != in_channels / group {
        return None;
    }

    // Checked `usize -> u32` narrowing: a model-supplied shape this large
    // is already nonsensical for a GPU launch (`oxicuda_dnn`'s descriptors
    // are `u32`-dimensioned throughout) — decline rather than truncate.
    let batch_u32 = u32::try_from(batch).ok()?;
    let in_channels_u32 = u32::try_from(in_channels).ok()?;
    let in_h_u32 = u32::try_from(in_h).ok()?;
    let in_w_u32 = u32::try_from(in_w).ok()?;
    let out_channels_u32 = u32::try_from(out_channels).ok()?;
    let filter_h_u32 = u32::try_from(filter_h).ok()?;
    let filter_w_u32 = u32::try_from(filter_w).ok()?;
    let pad_h_u32 = u32::try_from(pad_top).ok()?;
    let pad_w_u32 = u32::try_from(pad_left).ok()?;
    let stride_h_u32 = u32::try_from(params.strides[0]).ok()?;
    let stride_w_u32 = u32::try_from(params.strides[1]).ok()?;
    let dilation_h_u32 = u32::try_from(params.dilations[0]).ok()?;
    let dilation_w_u32 = u32::try_from(params.dilations[1]).ok()?;
    let group_u32 = u32::try_from(group).ok()?;

    let problem = ConvProblem {
        batch: batch_u32,
        in_channels: in_channels_u32,
        in_dims: vec![in_h_u32, in_w_u32],
        out_channels: out_channels_u32,
        filter_dims: vec![filter_h_u32, filter_w_u32],
        padding: vec![pad_h_u32, pad_w_u32],
        stride: vec![stride_h_u32, stride_w_u32],
        dilation: vec![dilation_h_u32, dilation_w_u32],
        groups: group_u32,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    };

    // Authoritative second opinion: zero stride/dilation, and (crucially)
    // whether the filter actually fits inside the padded input at all —
    // `ConvProblem::validate` calls `output_dims()` internally as its last
    // check. If this is `Ok`, `output_dims()` is guaranteed to succeed
    // when `cuda_conv` calls it again immediately afterwards.
    problem.validate().ok()?;

    Some(problem)
}

/// Maps an `oxicuda_dnn` failure into this crate's dispatch error type.
fn dnn_err(e: DnnError) -> CudaDispatchError {
    CudaDispatchError::Dnn(e.to_string())
}

/// Adds a per-output-channel bias to a convolution output already
/// downloaded to the host, laid out NCHW (`n` batches of `channels`
/// channel-blocks of `spatial` elements each).
///
/// Only used for the [`ConvEngine::Conv1x1`] / [`ConvEngine::Depthwise`]
/// dispatch paths — see the [module docs](self) for why neither engine's
/// `execute` accepts a bias argument at all.
fn add_bias_nchw(data: &mut [f32], bias: &[f32], n: usize, channels: usize, spatial: usize) {
    debug_assert_eq!(data.len(), n * channels * spatial);
    debug_assert_eq!(bias.len(), channels);
    for ni in 0..n {
        for c in 0..channels {
            let base = (ni * channels + c) * spatial;
            let Some(slice) = data.get_mut(base..base + spatial) else {
                continue; // Unreachable given the `debug_assert`s above; defensive only.
            };
            let Some(&b) = bias.get(c) else {
                continue; // Same: unreachable given the `debug_assert` above.
            };
            for v in slice {
                *v += b;
            }
        }
    }
}

/// ONNX `Conv` forward on the GPU.
///
/// * `ctx`    — live CUDA context (device + DNN handle). Re-activated by
///   [`crate::try_cuda_dispatch`] on the calling thread before this runs
///   (see that function's `activate_context` doc comment); a caller that
///   invokes this directly — as this module's own tests do — is
///   responsible for using a `ctx` that is current on the calling thread.
/// * `input`  — ONNX input tensor, shape `[N, C_in, H, W]` (NCHW; ONNX
///   `Conv` has no other layout).
/// * `weight` — ONNX filter tensor, shape `[C_out, C_in/group, kH, kW]`.
/// * `bias`   — optional bias tensor, shape `[C_out]`.
/// * `params` — strides, pads, dilations and group from the ONNX node attrs.
///
/// # Returns
///
/// * `Ok(Some(tensor))` — the convolution was computed on the GPU. `tensor`
///   has the ONNX-standard output shape `[N, C_out, P, Q]`.
/// * `Ok(None)` — this configuration is not (yet) accelerated; see the
///   [module docs](self) "What still declines" section. The caller falls
///   back to the CPU (or wgpu) operator.
/// * `Err(_)` — a real CUDA/DNN failure *after* dispatch was already
///   committed to (allocation, upload, kernel launch, or download). Unlike
///   a decline, this is not "the CPU can just do it instead" territory:
///   the node was already claimed.
///
/// # Errors
///
/// See "Returns" above.
pub fn cuda_conv(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &ConvParams,
) -> Result<Option<Tensor>, CudaDispatchError> {
    cuda_conv_cached(ctx, input, weight, bias, params, ConvWeightIds::default())
}

/// [`cuda_conv`] with residency identities for the invariant operands.
///
/// The dispatch layer knows which of a node's inputs are graph initializers
/// and this module does not, so the identities arrive as a parameter rather
/// than being derived here. Passing [`ConvWeightIds::default`] — which
/// [`cuda_conv`] does — reproduces the pre-residency behaviour exactly: every
/// operand uploads for this dispatch alone.
///
/// # Why the filter is the operand worth caching
///
/// A convolution's filter and bias are the *only* megabyte-scale bytes in this
/// crate's hot path that are provably identical on every frame. InSwapper-128
/// alone re-uploaded ~503 MB of invariant convolution weights per forward pass
/// before residency existed. The input activation, by contrast, is this
/// frame's data and must never be cached — see [`ConvWeightIds`].
///
/// # Errors
///
/// As [`cuda_conv`].
pub(crate) fn cuda_conv_cached(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &ConvParams,
    ids: ConvWeightIds<'_>,
) -> Result<Option<Tensor>, CudaDispatchError> {
    let Some(problem) = problem_from_params(&input.shape, &weight.shape, params) else {
        return Ok(None);
    };

    // Everything downstream reads out of `problem` (validated, `u32`)
    // rather than re-deriving from the caller's `usize` shapes, so the two
    // can never silently disagree.
    let n = problem.batch as usize;
    let in_channels = problem.in_channels as usize;
    let in_h = problem.in_dims[0] as usize;
    let in_w = problem.in_dims[1] as usize;
    let out_channels = problem.out_channels as usize;
    let filter_h = problem.filter_dims[0] as usize;
    let filter_w = problem.filter_dims[1] as usize;
    let group = problem.groups as usize;
    let in_ch_per_group = in_channels / group;

    // Guaranteed to succeed: `problem_from_params` already confirmed
    // `problem.validate()` (which itself calls `output_dims()`) passed.
    // Handled as a decline rather than trusted blindly regardless — this
    // crate never unwraps/expects on a data-dependent path.
    let Ok(out_dims) = problem.output_dims() else {
        return Ok(None);
    };
    let out_h = out_dims[0] as usize;
    let out_w = out_dims[1] as usize;

    // Bounds-check the *data* length against the *declared* shape up front
    // rather than trusting `Tensor::new`'s debug-only invariant (a release
    // build can violate it for a malformed model) — the same discipline
    // `try_cuda_dispatch`'s MatMul arm applies in `lib.rs`.
    let (Some(in_needed), Some(fil_needed), Some(out_needed)) = (
        n.checked_mul(in_channels)
            .and_then(|v| v.checked_mul(in_h))
            .and_then(|v| v.checked_mul(in_w)),
        out_channels
            .checked_mul(in_ch_per_group)
            .and_then(|v| v.checked_mul(filter_h))
            .and_then(|v| v.checked_mul(filter_w)),
        n.checked_mul(out_channels)
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w)),
    ) else {
        return Ok(None);
    };
    let Some(in_slice) = input.data.get(..in_needed) else {
        return Ok(None);
    };
    let Some(fil_slice) = weight.data.get(..fil_needed) else {
        return Ok(None);
    };

    let bias_data: Option<&[f32]> = match bias {
        Some(b) => {
            if b.shape.len() != 1 || b.shape[0] != out_channels {
                return Ok(None);
            }
            let Some(slice) = b.data.get(..out_channels) else {
                return Ok(None);
            };
            Some(slice)
        }
        None => None,
    };

    // -- Upload input/filter, allocate output ------------------------------
    //
    // Everything below rides `ctx.dnn.stream()`, which is the stream all three
    // engines launch on (they are handed `&ctx.dnn` and use its stream). Being
    // on one stream is what orders the uploads before the kernel and the kernel
    // before the readback, without a fence between them: the single
    // synchronise at the end is the only host/device rendezvous in the
    // dispatch. It also replaces one context-wide `cuCtxSynchronize` *per
    // operand*, which is what `DeviceBuffer::copy_from_host` costs.
    let stream = ctx.dnn.stream();

    // The activation is this frame's data: pooled, never cached.
    let mut d_input = ctx.scratch(in_needed)?;
    d_input.upload(in_slice, stream)?;
    // The filter is a graph initializer when the caller says so, in which case
    // it crosses the bus once per session rather than once per frame.
    let mut d_filter = ctx.operand(ids.weight, WEIGHT_LABEL, fil_slice, stream)?;
    let mut d_output = ctx.scratch(out_needed)?;
    // Zero-filled exactly as the `DeviceBuffer::zeroed` this replaces was: the
    // output is a recycled allocation now, and no engine is required to write
    // every element of a buffer larger than its problem. Stream-ordered, so it
    // costs a queued memset rather than a context-wide fence.
    d_output.zero_fill(stream)?;

    let in_desc = TensorDesc::<f32>::nchw(
        d_input.buffer(),
        problem.batch,
        problem.in_channels,
        problem.in_dims[0],
        problem.in_dims[1],
    )
    .map_err(dnn_err)?;
    let filter_in_channels = problem.in_channels / problem.groups;
    let fil_desc = TensorDesc::<f32>::nchw(
        d_filter.buffer(),
        problem.out_channels,
        filter_in_channels,
        problem.filter_dims[0],
        problem.filter_dims[1],
    )
    .map_err(dnn_err)?;
    let mut out_desc = TensorDescMut::<f32>::nchw(
        d_output.buffer_mut(),
        problem.batch,
        problem.out_channels,
        out_dims[0],
        out_dims[1],
    )
    .map_err(dnn_err)?;

    let sm = ctx.dnn.sm_version();
    let engine = pick_engine(&problem);

    // Declared here rather than inside the `ImplicitGemm` arm that fills it,
    // because a buffer must outlive the *fence*, not just the launch that
    // reads it: `execute` only queues the kernel. A `d_bias` scoped to the arm
    // would drop -- and, being still in flight, be freed rather than pooled --
    // while the convolution that reads it was still queued. See
    // `residency::PooledBuffer`'s "a borrow is only recycled once its stream
    // work is known to be done".
    let mut d_bias: Option<Operand<'_>> = None;

    match engine {
        ConvEngine::Conv1x1 => {
            Conv1x1::new(problem, sm)
                .map_err(dnn_err)?
                .execute(&ctx.dnn, &in_desc, &fil_desc, &mut out_desc)
                .map_err(dnn_err)?;
        }
        ConvEngine::Depthwise => {
            DepthwiseConv::new(problem, sm)
                .map_err(dnn_err)?
                .execute(&ctx.dnn, &in_desc, &fil_desc, &mut out_desc)
                .map_err(dnn_err)?;
        }
        ConvEngine::ImplicitGemm => {
            // `ImplicitGemmConv` natively supports a bias epilogue: get the
            // bias onto the device and pass its descriptor directly, rather
            // than the host-side post-add the other two engines need below.
            // Like the filter, a bias the caller identified as an initializer
            // is uploaded once per session.
            d_bias = match bias_data {
                Some(b) => Some(ctx.operand(ids.bias, BIAS_LABEL, b, stream)?),
                None => None,
            };
            let bias_desc = match &d_bias {
                Some(buf) => Some(
                    TensorDesc::<f32>::from_raw(
                        buf.device_ptr(),
                        vec![problem.out_channels],
                        vec![1],
                        TensorLayout::Nchw,
                    )
                    .map_err(dnn_err)?,
                ),
                None => None,
            };
            ImplicitGemmConv::new(problem, sm)
                .execute(
                    &ctx.dnn,
                    &in_desc,
                    &fil_desc,
                    bias_desc.as_ref(),
                    &mut out_desc,
                )
                .map_err(dnn_err)?;
        }
    }

    let mut out_data = vec![0.0_f32; out_needed];
    d_output.download(&mut out_data, stream)?;
    // The kernel launches above are asynchronous, and so is the readback just
    // queued behind them; this is where the host waits for all of it.
    stream.synchronize().map_err(CudaDispatchError::Driver)?;
    // ...and only now may these allocations go back to the pool. See
    // `PooledBuffer`'s "a borrow is only recycled once its stream work is
    // known to be done".
    d_input.retire();
    d_filter.retire();
    d_output.retire();
    if let Some(bias_buffer) = &mut d_bias {
        bias_buffer.retire();
    }

    if matches!(engine, ConvEngine::Conv1x1 | ConvEngine::Depthwise) {
        if let Some(bias_slice) = bias_data {
            add_bias_nchw(&mut out_data, bias_slice, n, out_channels, out_h * out_w);
        }
    }

    Ok(Some(Tensor::new(
        out_data,
        vec![n, out_channels, out_h, out_w],
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_params_round_trip() {
        let params = ConvParams {
            strides: [2, 2],
            pads: [1, 1, 1, 1],
            dilations: [1, 1],
            group: 1,
        };
        assert_eq!(params.strides, [2, 2]);
        assert_eq!(params.pads, [1, 1, 1, 1]);
        assert_eq!(params.dilations, [1, 1]);
        assert_eq!(params.group, 1);
    }

    /// `Conv` must be advertised as a CUDA-supported op: [`cuda_conv`]
    /// dispatches to a real `oxicuda-dnn` engine for every configuration
    /// [`problem_from_params`] accepts, so placement logic must be allowed to
    /// route convolutions here.
    ///
    /// The mirror image of this module's original `Conv`-is-unsupported
    /// assertion, kept in the same place on purpose: this file is where a
    /// future change would most plausibly re-break the link between what
    /// [`cuda_conv`] can do and what [`crate::is_supported_op`] says it can
    /// do, so the check lives next to the implementation as well as next to
    /// the predicate (`crate::tests::conv_is_advertised_as_supported`).
    #[test]
    fn conv_is_advertised_as_supported() {
        use oxionnx_core::graph::OpKind;
        assert!(
            crate::is_supported_op(&OpKind::Conv),
            "cuda_conv() computes real convolutions on the GPU (Conv1x1 / DepthwiseConv / \
             ImplicitGemmConv); is_supported_op must report Conv as supported so \
             decide_placement routes Conv nodes to CUDA",
        );
    }

    // ── problem_from_params: pure, GPU-free ─────────────────────────────────

    fn base_params() -> ConvParams {
        ConvParams {
            strides: [1, 1],
            pads: [0, 0, 0, 0],
            dilations: [1, 1],
            group: 1,
        }
    }

    #[test]
    fn problem_from_params_rejects_non_4d_input() {
        assert!(problem_from_params(&[1, 3, 5], &[4, 3, 3, 3], &base_params()).is_none());
    }

    #[test]
    fn problem_from_params_rejects_non_4d_weight() {
        assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3], &base_params()).is_none());
    }

    #[test]
    fn problem_from_params_rejects_asymmetric_pad_top_bottom() {
        let mut params = base_params();
        params.pads = [1, 1, 0, 1];
        assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3, 3], &params).is_none());
    }

    #[test]
    fn problem_from_params_rejects_asymmetric_pad_left_right() {
        let mut params = base_params();
        params.pads = [1, 1, 1, 0];
        assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3, 3], &params).is_none());
    }

    #[test]
    fn problem_from_params_accepts_symmetric_pad_and_derives_every_field() {
        let mut params = base_params();
        params.pads = [1, 1, 1, 1];
        params.strides = [2, 3];
        params.dilations = [1, 2];
        let p = problem_from_params(&[1, 3, 5, 7], &[4, 3, 3, 3], &params)
            .expect("well-formed symmetric-pad convolution must be accepted");
        assert_eq!(p.batch, 1);
        assert_eq!(p.in_channels, 3);
        assert_eq!(p.in_dims, vec![5, 7]);
        assert_eq!(p.out_channels, 4);
        assert_eq!(p.filter_dims, vec![3, 3]);
        assert_eq!(p.padding, vec![1, 1]);
        assert_eq!(p.stride, vec![2, 3]);
        assert_eq!(p.dilation, vec![1, 2]);
        assert_eq!(p.groups, 1);
    }

    #[test]
    fn problem_from_params_rejects_zero_group() {
        let mut params = base_params();
        params.group = 0;
        assert!(problem_from_params(&[1, 4, 5, 5], &[4, 4, 3, 3], &params).is_none());
    }

    #[test]
    fn problem_from_params_rejects_in_channels_not_divisible_by_group() {
        let mut params = base_params();
        params.group = 3;
        // in_channels=4 is not divisible by group=3.
        assert!(problem_from_params(&[1, 4, 5, 5], &[6, 2, 3, 3], &params).is_none());
    }

    #[test]
    fn problem_from_params_rejects_out_channels_not_divisible_by_group() {
        let mut params = base_params();
        params.group = 2;
        // in_channels=4 divides group=2 fine, but out_channels=5 does not.
        assert!(problem_from_params(&[1, 4, 5, 5], &[5, 2, 3, 3], &params).is_none());
    }

    #[test]
    fn problem_from_params_rejects_filter_in_channel_mismatch() {
        // in_channels=4, group=1 -> filter dim[1] must be 4; it claims 3.
        assert!(problem_from_params(&[1, 4, 5, 5], &[4, 3, 3, 3], &base_params()).is_none());
    }

    #[test]
    fn problem_from_params_accepts_grouped_conv_with_correct_filter_channels() {
        let mut params = base_params();
        params.group = 2;
        // in_channels=4, group=2 -> filter dim[1] must be 2.
        let p = problem_from_params(&[1, 4, 5, 5], &[6, 2, 3, 3], &params)
            .expect("a correctly-shaped grouped conv must be accepted");
        assert_eq!(p.groups, 2);
    }

    #[test]
    fn problem_from_params_rejects_zero_batch() {
        assert!(problem_from_params(&[0, 3, 5, 5], &[4, 3, 3, 3], &base_params()).is_none());
    }

    #[test]
    fn problem_from_params_rejects_zero_spatial_dim() {
        assert!(problem_from_params(&[1, 3, 0, 5], &[4, 3, 3, 3], &base_params()).is_none());
    }

    #[test]
    fn problem_from_params_rejects_zero_filter_spatial_dim() {
        assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 0, 3], &base_params()).is_none());
    }

    #[test]
    fn problem_from_params_rejects_filter_larger_than_padded_input() {
        // 3x3 input, 5x5 filter, no padding: the filter does not fit.
        assert!(problem_from_params(&[1, 3, 3, 3], &[4, 3, 5, 5], &base_params()).is_none());
    }

    #[test]
    fn problem_from_params_rejects_zero_stride() {
        let mut params = base_params();
        params.strides = [0, 1];
        assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3, 3], &params).is_none());
    }

    #[test]
    fn problem_from_params_rejects_zero_dilation() {
        let mut params = base_params();
        params.dilations = [1, 0];
        assert!(problem_from_params(&[1, 3, 5, 5], &[4, 3, 3, 3], &params).is_none());
    }

    // ── pick_engine: pure, GPU-free ──────────────────────────────────────────

    fn tiny_problem(
        filter: [u32; 2],
        stride: [u32; 2],
        dilation: [u32; 2],
        pad: [u32; 2],
        groups: u32,
        in_channels: u32,
        out_channels: u32,
    ) -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels,
            in_dims: vec![16, 16],
            out_channels,
            filter_dims: filter.to_vec(),
            padding: pad.to_vec(),
            stride: stride.to_vec(),
            dilation: dilation.to_vec(),
            groups,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    #[test]
    fn pick_engine_selects_conv1x1_for_unpadded_unit_stride_1x1() {
        let p = tiny_problem([1, 1], [1, 1], [1, 1], [0, 0], 1, 8, 8);
        assert_eq!(pick_engine(&p), ConvEngine::Conv1x1);
    }

    #[test]
    fn pick_engine_does_not_select_conv1x1_when_padded() {
        let p = tiny_problem([1, 1], [1, 1], [1, 1], [1, 1], 1, 8, 8);
        assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
    }

    #[test]
    fn pick_engine_does_not_select_conv1x1_when_strided() {
        let p = tiny_problem([1, 1], [2, 2], [1, 1], [0, 0], 1, 8, 8);
        assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
    }

    #[test]
    fn pick_engine_does_not_select_conv1x1_when_dilated() {
        // Dilation is mathematically moot for a 1x1 tap, but rule 1 checks
        // it literally, per spec -- pin that down explicitly.
        let p = tiny_problem([1, 1], [1, 1], [2, 2], [0, 0], 1, 8, 8);
        assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
    }

    #[test]
    fn pick_engine_selects_depthwise_for_true_depthwise_3x3() {
        let p = tiny_problem([3, 3], [1, 1], [1, 1], [1, 1], 8, 8, 8);
        assert_eq!(pick_engine(&p), ConvEngine::Depthwise);
    }

    #[test]
    fn pick_engine_selects_implicit_gemm_for_grouped_non_depthwise() {
        // groups=2, in=out=8: depthwise would require groups==8.
        let p = tiny_problem([3, 3], [1, 1], [1, 1], [1, 1], 2, 8, 8);
        assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
    }

    #[test]
    fn pick_engine_selects_implicit_gemm_for_plain_3x3() {
        let p = tiny_problem([3, 3], [1, 1], [1, 1], [1, 1], 1, 16, 32);
        assert_eq!(pick_engine(&p), ConvEngine::ImplicitGemm);
    }

    #[test]
    fn pick_engine_prioritises_conv1x1_over_depthwise_when_both_apply() {
        // filter=1x1, groups==in==out==8: matches BOTH the Conv1x1 and the
        // Depthwise conditions -- rule 1 (Conv1x1) must win, per the
        // ordered 3-way dispatch rule in the module docs.
        let p = tiny_problem([1, 1], [1, 1], [1, 1], [0, 0], 8, 8, 8);
        assert_eq!(pick_engine(&p), ConvEngine::Conv1x1);
    }

    // ── add_bias_nchw: pure, GPU-free ────────────────────────────────────────

    #[test]
    fn add_bias_nchw_broadcasts_per_channel_across_batch_and_spatial() {
        // n=2, channels=2, spatial=3 -> 12 elements, layout
        // [n0c0(3), n0c1(3), n1c0(3), n1c1(3)].
        let mut data = vec![0.0_f32; 12];
        let bias = [10.0_f32, 100.0];
        add_bias_nchw(&mut data, &bias, 2, 2, 3);
        assert_eq!(
            data,
            vec![10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0]
        );
    }

    // ── On-device numeric validation ────────────────────────────────────────
    //
    // `cargo test -p oxionnx-cuda --features gpu-tests conv` on a
    // CUDA-capable host. `problem_from_params`/`pick_engine` above are
    // exercised for free by every case here (`cuda_conv` calls both), so
    // these tests additionally confirm the dispatched engine actually
    // computes the right numbers, not merely that it runs.
    //
    // The CPU oracle below (`naive_conv2d_f64`) predates `crate::reference`
    // having a `Conv` formula and is kept as its own from-scratch reference
    // rather than switched over to `crate::reference::ref_conv` now that the
    // latter exists: this module's tests check each of the three dispatched
    // engines (`Conv1x1`/`DepthwiseConv`/`ImplicitGemmConv`) directly, and
    // two independently-written oracles agreeing is stronger evidence than
    // one oracle re-used to check itself. `reference::ref_conv` gets its own
    // from-scratch validation, both as unit tests in `reference.rs` and
    // end-to-end (through `try_cuda_dispatch`'s `OXIONNX_CUDA_VERIFY=1`
    // wiring) in `oxionnx-cuda/tests/verify_path_gpu.rs`. It deliberately
    // mirrors the *cross-correlation* convention (no 180-degree kernel
    // flip) `oxicuda_dnn`'s shared `emit_standard_conv_body` /
    // `emit_depthwise_pixel` kernel bodies use, `f64`-accumulated, O(naive)
    // nested loops -- correct over fast.
    #[cfg(feature = "gpu-tests")]
    #[allow(clippy::needless_range_loop)]
    mod gpu_numeric {
        use super::*;
        use crate::context::Activation;

        /// A small deterministic LCG (Knuth/MMIX multiplier) — avoids a
        /// `rand` dependency for throwaway, reproducible test data.
        struct Lcg(u64);

        impl Lcg {
            fn new(seed: u64) -> Self {
                Self(seed)
            }
            fn next_u32(&mut self) -> u32 {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (self.0 >> 32) as u32
            }
            /// Uniform `f32` in `[lo, hi)`.
            fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
                let unit = f64::from(self.next_u32()) / 4_294_967_296.0;
                (f64::from(lo) + (f64::from(hi) - f64::from(lo)) * unit) as f32
            }
        }

        /// Scalar convolution geometry (mirrors the fields a `ConvProblem`
        /// needs, plus the raw ONNX-style asymmetric-capable `pads` this
        /// crate accepts -- kept symmetric here since that is what
        /// `cuda_conv` can claim).
        #[derive(Clone, Copy)]
        struct ConvCase {
            n: usize,
            c: usize,
            h: usize,
            w: usize,
            k: usize,
            r: usize,
            s: usize,
            pad_h: usize,
            pad_w: usize,
            stride_h: usize,
            stride_w: usize,
            dil_h: usize,
            dil_w: usize,
            group: usize,
        }

        impl ConvCase {
            fn out_hw(self) -> (usize, usize) {
                let eff_r = self.dil_h * (self.r - 1) + 1;
                let eff_s = self.dil_w * (self.s - 1) + 1;
                let out_h = (self.h + 2 * self.pad_h - eff_r) / self.stride_h + 1;
                let out_w = (self.w + 2 * self.pad_w - eff_s) / self.stride_w + 1;
                (out_h, out_w)
            }

            fn params(self) -> ConvParams {
                ConvParams {
                    strides: [self.stride_h, self.stride_w],
                    pads: [self.pad_h, self.pad_w, self.pad_h, self.pad_w],
                    dilations: [self.dil_h, self.dil_w],
                    group: self.group,
                }
            }

            fn in_ch_per_group(self) -> usize {
                self.c / self.group
            }
        }

        /// Independent `f64`-accumulated NCHW cross-correlation reference.
        /// Indexing follows the same `[K, C/g, R, S]` filter layout ONNX
        /// (and this crate's `cuda_conv`) uses.
        fn naive_conv2d_f64(
            case: ConvCase,
            input: &[f32],
            weight: &[f32],
            bias: Option<&[f32]>,
        ) -> Vec<f32> {
            let (out_h, out_w) = case.out_hw();
            let in_ch_per_group = case.in_ch_per_group();
            let out_ch_per_group = case.k / case.group;

            let mut out = vec![0.0_f32; case.n * case.k * out_h * out_w];
            for ni in 0..case.n {
                for ki in 0..case.k {
                    let g = ki / out_ch_per_group;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut acc = 0.0_f64;
                            for cg in 0..in_ch_per_group {
                                let ci = g * in_ch_per_group + cg;
                                for ri in 0..case.r {
                                    let ih = oh as isize * case.stride_h as isize
                                        - case.pad_h as isize
                                        + ri as isize * case.dil_h as isize;
                                    if ih < 0 || ih as usize >= case.h {
                                        continue;
                                    }
                                    let ih = ih as usize;
                                    for si in 0..case.s {
                                        let iw = ow as isize * case.stride_w as isize
                                            - case.pad_w as isize
                                            + si as isize * case.dil_w as isize;
                                        if iw < 0 || iw as usize >= case.w {
                                            continue;
                                        }
                                        let iw = iw as usize;
                                        let in_idx =
                                            ((ni * case.c + ci) * case.h + ih) * case.w + iw;
                                        let f_idx = ((ki * in_ch_per_group + cg) * case.r + ri)
                                            * case.s
                                            + si;
                                        acc += f64::from(input[in_idx]) * f64::from(weight[f_idx]);
                                    }
                                }
                            }
                            if let Some(bv) = bias {
                                acc += f64::from(bv[ki]);
                            }
                            let o_idx = ((ni * case.k + ki) * out_h + oh) * out_w + ow;
                            out[o_idx] = acc as f32;
                        }
                    }
                }
            }
            out
        }

        /// Real `CudaContext`, bypassing the `OXIONNX_CUDA` env-var opt-in
        /// gate (unit-tested separately in `context::tests`) -- still
        /// requires a real, working CUDA driver and device 0. The
        /// `expect` is deliberate: a `gpu-tests`-gated run with no GPU
        /// present is a misconfigured invocation, not a case to degrade
        /// quietly out of.
        fn gpu_ctx() -> CudaContext {
            CudaContext::try_new_with(Activation::Enabled)
                .expect("gpu-tests requires a real CUDA device -- run on a CUDA-capable host")
        }

        /// Resolves which engine `cuda_conv` will pick for `case`, so each
        /// test can assert it is actually exercising the branch it claims
        /// to (rather than silently drifting onto a different one).
        fn engine_for(case: ConvCase) -> ConvEngine {
            let input_shape = [case.n, case.c, case.h, case.w];
            let weight_shape = [case.k, case.in_ch_per_group(), case.r, case.s];
            let problem = problem_from_params(&input_shape, &weight_shape, &case.params())
                .expect("ConvCase must describe a valid, non-declined convolution");
            pick_engine(&problem)
        }

        /// Runs `case` through the real `cuda_conv` dispatch on-device and
        /// checks the result against `naive_conv2d_f64`, using this
        /// crate's own `reference::compare` tolerance (tight: `ATOL=1e-4`,
        /// `RTOL=1e-3`).
        fn run_case_and_compare(case: ConvCase, with_bias: bool, seed: u64, tag: &str) {
            let ctx = gpu_ctx();
            let (out_h, out_w) = case.out_hw();
            let in_ch_per_group = case.in_ch_per_group();

            let mut lcg = Lcg::new(seed);
            let in_data: Vec<f32> = (0..case.n * case.c * case.h * case.w)
                .map(|_| lcg.range_f32(-1.0, 1.0))
                .collect();
            let fil_data: Vec<f32> = (0..case.k * in_ch_per_group * case.r * case.s)
                .map(|_| lcg.range_f32(-1.0, 1.0))
                .collect();
            let bias_data: Option<Vec<f32>> =
                with_bias.then(|| (0..case.k).map(|_| lcg.range_f32(-0.5, 0.5)).collect());

            let input = Tensor::new(in_data.clone(), vec![case.n, case.c, case.h, case.w]);
            let weight = Tensor::new(
                fil_data.clone(),
                vec![case.k, in_ch_per_group, case.r, case.s],
            );
            let bias_tensor = bias_data
                .as_ref()
                .map(|b| Tensor::new(b.clone(), vec![case.k]));

            let params = case.params();
            let result = cuda_conv(&ctx, &input, &weight, bias_tensor.as_ref(), &params)
                .unwrap_or_else(|e| panic!("{tag}: cuda_conv hard-errored: {e}"))
                .unwrap_or_else(|| panic!("{tag}: cuda_conv declined a shape it must claim"));

            assert_eq!(
                result.shape,
                vec![case.n, case.k, out_h, out_w],
                "{tag}: output shape mismatch"
            );

            let expected = naive_conv2d_f64(case, &in_data, &fil_data, bias_data.as_deref());
            if let Err(e) = crate::reference::compare(&result.data, &expected) {
                panic!("{tag}: GPU output disagrees with the naive CPU oracle: {e}");
            }
        }

        #[test]
        fn conv1x1_no_bias_matches_naive_cpu_reference() {
            let case = ConvCase {
                n: 1,
                c: 24,
                h: 18,
                w: 26,
                k: 40,
                r: 1,
                s: 1,
                pad_h: 0,
                pad_w: 0,
                stride_h: 1,
                stride_w: 1,
                dil_h: 1,
                dil_w: 1,
                group: 1,
            };
            assert_eq!(engine_for(case), ConvEngine::Conv1x1);
            run_case_and_compare(case, false, 0x51ED_0000_0000_0001, "conv1x1_no_bias");
        }

        #[test]
        fn conv1x1_with_bias_matches_naive_cpu_reference() {
            // Exercises the host-side `add_bias_nchw` path: `Conv1x1::execute`
            // itself has no bias parameter at all.
            let case = ConvCase {
                n: 1,
                c: 32,
                h: 22,
                w: 15,
                k: 17,
                r: 1,
                s: 1,
                pad_h: 0,
                pad_w: 0,
                stride_h: 1,
                stride_w: 1,
                dil_h: 1,
                dil_w: 1,
                group: 1,
            };
            assert_eq!(engine_for(case), ConvEngine::Conv1x1);
            run_case_and_compare(case, true, 0x51ED_0000_0000_0002, "conv1x1_with_bias");
        }

        #[test]
        fn depthwise_dilated_with_bias_matches_naive_cpu_reference() {
            // Exercises DepthwiseConv AND the host-side `add_bias_nchw`
            // path (same reason as conv1x1_with_bias: no bias parameter on
            // `DepthwiseConv::execute`). `pad=2, dilation=2` is a
            // "same"-style dilated depthwise conv (output spatial size ==
            // input spatial size).
            let case = ConvCase {
                n: 1,
                c: 28,
                h: 21,
                w: 25,
                k: 28,
                r: 3,
                s: 3,
                pad_h: 2,
                pad_w: 2,
                stride_h: 1,
                stride_w: 1,
                dil_h: 2,
                dil_w: 2,
                group: 28,
            };
            assert_eq!(engine_for(case), ConvEngine::Depthwise);
            run_case_and_compare(
                case,
                true,
                0x51ED_0000_0000_0003,
                "depthwise_dilated_with_bias",
            );
        }

        #[test]
        fn implicit_gemm_3x3_stride1_with_bias_matches_naive_cpu_reference() {
            let case = ConvCase {
                n: 2,
                c: 19,
                h: 20,
                w: 28,
                k: 37,
                r: 3,
                s: 3,
                pad_h: 1,
                pad_w: 1,
                stride_h: 1,
                stride_w: 1,
                dil_h: 1,
                dil_w: 1,
                group: 1,
            };
            assert_eq!(engine_for(case), ConvEngine::ImplicitGemm);
            run_case_and_compare(
                case,
                true,
                0x51ED_0000_0000_0004,
                "implicit_gemm_3x3_stride1_with_bias",
            );
        }

        #[test]
        fn implicit_gemm_3x3_stride2_with_bias_matches_naive_cpu_reference() {
            let case = ConvCase {
                n: 1,
                c: 33,
                h: 27,
                w: 19,
                k: 21,
                r: 3,
                s: 3,
                pad_h: 1,
                pad_w: 1,
                stride_h: 2,
                stride_w: 2,
                dil_h: 1,
                dil_w: 1,
                group: 1,
            };
            assert_eq!(engine_for(case), ConvEngine::ImplicitGemm);
            // Hand-verified output size: floor((27+2-2-1)/2)+1=14, floor((19+2-2-1)/2)+1=10.
            assert_eq!(case.out_hw(), (14, 10));
            run_case_and_compare(
                case,
                true,
                0x51ED_0000_0000_0005,
                "implicit_gemm_3x3_stride2_with_bias",
            );
        }
    }
}
