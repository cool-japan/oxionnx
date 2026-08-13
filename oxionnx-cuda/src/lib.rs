//! # oxionnx-cuda
//!
//! CUDA-accelerated dispatch for ONNX ops via the OxiCUDA GPU stack.
//!
//! This crate provides:
//!
//! - [`CudaContext`] — a wrapper around a CUDA device context + DNN handle,
//!   constructed lazily via [`CudaContext::try_new`].
//! - [`CudaError`] — error type returned by the CUDA dispatch layer.
//! - [`try_cuda_dispatch`] — the top-level dispatch function called from
//!   `oxionnx::session::run_sequential_inner` when the `cuda` feature is enabled.
//! - [`is_supported_op`] — a cheap, pure predicate reporting exactly which
//!   [`OpKind`]s [`try_cuda_dispatch`] is able to claim.  Placement logic in
//!   `oxionnx` consults this *before* paying for an upload/dispatch/readback.
//!
//! ## Dispatch flow
//!
//! ```text
//! CUDA (highest priority)
//!   └─ try_cuda_dispatch → Ok(Some(results))   ← GPU handled it
//!      └─ Ok(None)                              ← fall back to wgpu / CPU
//! wgpu GPU dispatch
//! CPU dispatch
//! ```
//!
//! ## Graceful degradation
//!
//! On any CUDA error during dispatch, the function returns `Err(...)` which
//! the caller maps to `OnnxError::Internal`.  If CUDA is not available at
//! session build time, `CudaContext::try_new()` returns `None` and no CUDA
//! dispatch is attempted.
//!
//! ## Activation, shadow verification, and strict mode
//!
//! CUDA acquisition is opt-in (`OXIONNX_CUDA=1`), every claimed op can be
//! shadow-compared against a CPU oracle (`OXIONNX_CUDA_VERIFY=1`), and a
//! shadow-verification mismatch can be turned into a hard error instead of a
//! silent CPU fallback (`OXIONNX_CUDA_STRICT=1`). See the [`context`] and
//! [`mod@reference`] module docs for the full rationale.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_safety_doc)]

pub mod activation;
pub mod broadcast;
pub mod concat;
pub mod context;
pub mod conv;
/// Small free-standing helpers [`try_cuda_dispatch_resident`]'s match arms
/// share, split out purely for `lib.rs`'s size — see that file's own header.
#[path = "dispatch_helpers.rs"]
mod dispatch_helpers;
pub mod elementwise;
pub mod error;
pub mod graph_cache;
pub mod matmul;
pub mod norm;
pub mod pad;
pub mod pool;
pub mod prelu;
pub mod reduce;
pub mod reference;
pub mod reshape;
pub mod residency;
pub mod resize;
pub mod slice;
pub mod softmax;

pub use activation::{
    CudaDeviceTensor, CudaDispatchOutcome, CudaOutputPlacement, NoActivations, ResidentActivations,
};
pub use context::CudaContext;
/// Re-exported at the crate root because the *session runner* — not a CUDA
/// caller — is what has to ask this question, on the error it just got back
/// from [`try_cuda_dispatch`]. See [`error::is_verify_mismatch`].
pub use error::is_verify_mismatch;
pub use error::CudaDispatchError as CudaError;

use std::collections::HashMap;

use oxionnx_core::graph::{Node, OpKind};
use oxionnx_core::{OnnxError, Tensor};

use activation::InputBinding;
use dispatch_helpers::{
    apply_gemm_bias, initializer_id, operand_form, transpose_2d_batched, verify_or_fallback,
};

/// Report whether [`try_cuda_dispatch`] is *capable* of claiming `op`.
///
/// Cheap, pure, allocation-free: a single `matches!` over the op kind.  Safe to
/// call on every node of every graph, including when no CUDA device exists.
///
/// # Derivation
///
/// The list below is derived directly from the `match &node.op` arms of
/// [`try_cuda_dispatch`], minus any arm whose kernel is a *permanent decline*:
///
/// | dispatch arm                              | claimable | backing kernel                             |
/// |-------------------------------------------|-----------|--------------------------------------------|
/// | `MatMul`, `Gemm`                          | **yes**   | [`matmul::cuda_matmul`]                    |
/// | `Conv`                                    | **yes**   | [`conv::cuda_conv`] — dispatches *directly* to `oxicuda-dnn`'s `Conv1x1` / `DepthwiseConv` / `ImplicitGemmConv` (never `conv_forward`'s auto-selector) |
/// | 16 unary activations (`Relu` … `LeakyRelu`) | **yes** | [`elementwise::cuda_elementwise`]          |
/// | `Add`, `Sub`, `Mul`, `Div` (exact-shape)   | **yes**   | [`elementwise::cuda_binary_elementwise`]   |
/// | `Add`, `Sub`, `Mul`, `Div` ([1,C,1,1]/scalar broadcast) | **yes** | [`broadcast::cuda_broadcast_bound`] |
/// | `PRelu`                                   | **yes**   | [`prelu::cuda_prelu_bound`]                |
/// | `BatchNormalization` (inference only)     | **yes**   | [`norm::cuda_batch_norm_bound`]            |
/// | `OxiInstanceNorm`                          | **yes**   | [`norm::cuda_oxi_instance_norm_bound`]     |
/// | `ReduceSum`, `ReduceMax`                  | **yes**   | [`reduce::cuda_reduce`]                    |
/// | `ReduceMean` (contiguous axes)             | **yes**   | [`reduce::cuda_reduce_mean_bound`]         |
/// | `Softmax`                                 | **yes**   | [`softmax::cuda_softmax`]                  |
/// | `MaxPool`, `AveragePool`                  | **yes**   | [`pool::cuda_pool_bound`] — dispatches to `oxicuda_dnn::pool::{max_pool2d, avg_pool2d}` |
/// | `Resize` (nearest/bilinear)                | **yes**   | [`resize::cuda_resize_bound`] — dispatches to `oxicuda_dnn::resize::{resize_nearest, resize_bilinear}` |
/// | `Pad` (reflect/constant)                   | **yes**   | [`pad::cuda_pad_bound`] — this crate's own PTX, no `oxicuda-dnn` kernel exists |
/// | `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten` | **yes** | [`activation::CudaDeviceTensor::alias`] — zero-cost, no kernel; resident input only |
/// | `Slice`                                    | **yes**   | [`slice::cuda_slice_bound`] — this crate's own PTX |
/// | `Concat`                                   | **yes**   | [`concat::cuda_concat_bound`] — device-to-device copies, no kernel |
///
/// ## `Conv`
///
/// [`conv::cuda_conv`] hands the node to exactly one of three
/// individually-validated `oxicuda-dnn` forward-convolution engines —
/// [`Conv1x1`](oxicuda_dnn::conv::fprop::direct::Conv1x1),
/// [`DepthwiseConv`](oxicuda_dnn::conv::fprop::direct::DepthwiseConv), or
/// [`ImplicitGemmConv`](oxicuda_dnn::conv::fprop::implicit_gemm::ImplicitGemmConv)
/// — picked by its own small, GPU-free, unit-tested rule. It deliberately
/// does **not** call `oxicuda_dnn::conv::api::conv_forward`, whose
/// `select_algorithm` auto-selector can route into the Winograd fprop path;
/// see the [`conv`] module docs' "Why not `conv_forward`" for the full
/// reasoning. Like every other claimable op, the `Conv` arm's output is
/// shadow-verifiable: [`mod@reference`]'s `ref_conv` CPU oracle backs its
/// `verify_or_fallback` call, so `OXIONNX_CUDA_VERIFY=1` covers a `Conv`
/// dispatch exactly as it covers MatMul/elementwise/reduce/Softmax.
///
/// ## Channel/scalar broadcast, `PRelu`, `BatchNormalization`, `OxiInstanceNorm`, `ReduceMean`
///
/// [`broadcast::classify`] recognises exactly two operand-shape patterns for
/// `Add`/`Sub`/`Mul`/`Div` beyond exact-shape equality — `[1,C,1,1]`-vs-
/// `[1,C,H,W]` and scalar-vs-tensor — which is 100% of what the op-coverage
/// audit found declining in the three real face-pipeline models; any other
/// broadcast shape still declines. [`prelu::cuda_prelu_bound`] and
/// [`norm::cuda_batch_norm_bound`]/[`norm::cuda_oxi_instance_norm_bound`]
/// reuse that same per-channel addressing (`PRelu`) or
/// [`oxicuda_ptx::templates::batch_norm::BatchNormTemplate`] unmodified
/// (`BatchNormalization`/`OxiInstanceNorm` — the latter by launching the
/// template's *training*-mode kernel once per sample with an identity
/// `gamma=1, beta=0` affine, since it has none of its own; see [`mod@norm`]'s
/// module docs). [`reduce::cuda_reduce_mean_bound`] is `ReduceSum`'s
/// `reduce_axis` machinery plus an in-place device-side divide, generalised
/// from one axis to a *contiguous* axis range so it can claim the
/// `axes=[2,3]` shape InSwapper's un-fused `InstanceNorm` decomposition
/// emits (see [`reduce::resolve_contiguous_axes`]). Every one of these five
/// arms is shadow-verified exactly like `Conv`/`MatMul` above — `ref_prelu`,
/// `ref_batch_norm`, `ref_oxi_instance_norm`, `ref_binary_broadcast`, and
/// `ref_reduce`'s new `ReduceMean` case in [`mod@reference`].
///
/// ## Data movement: `MaxPool`/`AveragePool`/`Resize`/`Pad`/`Slice`/`Concat`,
/// and the zero-cost `Reshape` family
///
/// [`pool::cuda_pool_bound`] and [`resize::cuda_resize_bound`] dispatch
/// straight to `oxicuda-dnn` forward kernels that existed in this workspace
/// but were never called from anywhere in it; [`pad::cuda_pad_bound`] and
/// [`slice::cuda_slice_bound`] generate their own PTX (no `oxicuda-dnn`
/// kernel exists for either); [`concat::cuda_concat_bound`] needs no kernel
/// at all — concatenation decomposes into `outer * num_inputs` device-to-
/// device copies. `Reshape`/`Squeeze`/`Unsqueeze`/`Flatten` need no kernel
/// either, for a different reason: they change only a tensor's declared
/// shape, so [`activation::CudaDeviceTensor::alias`] (a second handle to the
/// same allocation under a new shape) *is* the whole implementation, and only
/// when the input is already device-resident — a host-resident input
/// declines, since a CPU reshape is already `O(1)`. See each module's own
/// docs for its exact whitelist (every one of these six kernels narrows the
/// ONNX spec to precisely what its backing kernel computes, the same
/// discipline `Conv`'s fused-activation whitelist established) and
/// [`mod@reshape`]'s "why this module carries no oracle" for why the
/// `Reshape` family is the one arm here with no shadow-verification story —
/// every numeric arm (`MaxPool`, `AveragePool`, `Resize`, `Pad`, `Slice`,
/// `Concat`) is shadow-verified via `ref_pool`, `ref_resize`, `ref_pad`,
/// `ref_slice`, and `ref_concat` in [`mod@reference`], exactly like every
/// other claimable op above.
///
/// # Necessary, not sufficient
///
/// - `is_supported_op(op) == false` is a **hard guarantee**: [`try_cuda_dispatch`]
///   returns `Ok(None)` for every such node.  Callers may skip CUDA entirely.
/// - `is_supported_op(op) == true` means a kernel exists *for that op kind*.
///   [`try_cuda_dispatch`] may still decline an individual node whose
///   *configuration* is out of range — e.g. `Softmax` with a row wider than 1024,
///   a reduction over a non-flat axis, an `Add` whose two operand shapes are
///   neither equal nor the narrow broadcast pattern above, a `ReduceMean`
///   over non-contiguous axes, or a `Conv` with asymmetric `pads` (see the
///   [`conv`] module docs' "What still declines").  Callers must still
///   handle `Ok(None)`.
///
/// # Example
///
/// ```
/// use oxionnx_core::graph::OpKind;
///
/// assert!(oxionnx_cuda::is_supported_op(&OpKind::MatMul));
/// assert!(oxionnx_cuda::is_supported_op(&OpKind::Relu));
/// // `cuda_conv` dispatches straight to oxicuda-dnn's Conv1x1 /
/// // DepthwiseConv / ImplicitGemmConv engines, and is shadow-verified
/// // against `reference::ref_conv` like every other claimable op. An
/// // individual node can still be declined at dispatch time (asymmetric
/// // padding, non-4-D shapes, ...) -- see "Necessary, not sufficient".
/// assert!(oxionnx_cuda::is_supported_op(&OpKind::Conv));
/// // `Reshape`/`Squeeze`/`Unsqueeze`/`Flatten` are zero-cost residency
/// // aliases (no kernel at all) when their input is already device-resident
/// // -- see the `reshape` module docs -- and decline otherwise.
/// assert!(oxionnx_cuda::is_supported_op(&OpKind::Reshape));
/// // No dispatch arm at all.
/// assert!(!oxionnx_cuda::is_supported_op(&OpKind::Transpose));
/// ```
pub fn is_supported_op(op: &OpKind) -> bool {
    matches!(
        op,
        // ── matmul.rs: `OpKind::MatMul | OpKind::Gemm` arm ───────────────────
        OpKind::MatMul
            | OpKind::Gemm
            // ── conv.rs: `OpKind::Conv` arm ──────────────────────────────────
            | OpKind::Conv
            // ── elementwise.rs: unary activation arm (16 ops) ────────────────
            | OpKind::Relu
            | OpKind::Sigmoid
            | OpKind::Gelu
            | OpKind::Tanh
            | OpKind::Exp
            | OpKind::Sqrt
            | OpKind::Abs
            | OpKind::Neg
            | OpKind::Log
            | OpKind::Ceil
            | OpKind::Floor
            | OpKind::HardSigmoid
            | OpKind::HardSwish
            | OpKind::SiLU
            | OpKind::Softplus
            | OpKind::LeakyRelu
            // ── elementwise.rs (exact shape) / broadcast.rs ([1,C,1,1] & scalar) ─
            | OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            // ── prelu.rs ──────────────────────────────────────────────────────
            | OpKind::PRelu
            // ── norm.rs: BatchNormalization (inference) / OxiInstanceNorm ────
            | OpKind::BatchNorm
            | OpKind::OxiInstanceNorm
            // ── reduce.rs: reduction arm ─────────────────────────────────────
            | OpKind::ReduceSum
            | OpKind::ReduceMax
            | OpKind::ReduceMean
            // ── softmax.rs ───────────────────────────────────────────────────
            | OpKind::Softmax
            // ── pool.rs: MaxPool / AveragePool ───────────────────────────────
            | OpKind::MaxPool
            | OpKind::AveragePool
            // ── resize.rs: nearest / bilinear ────────────────────────────────
            | OpKind::Resize
            // ── pad.rs: reflect / constant ───────────────────────────────────
            | OpKind::Pad
            // ── reshape.rs: zero-cost residency aliases, no kernel ───────────
            | OpKind::Unsqueeze
            | OpKind::Squeeze
            | OpKind::Reshape
            | OpKind::Flatten
            // ── slice.rs ──────────────────────────────────────────────────────
            | OpKind::Slice
            // ── concat.rs: device-to-device copies, no kernel ────────────────
            | OpKind::Concat
    )
}

/// Make `ctx`'s CUDA context current **on the calling OS thread**,
/// unconditionally, before any driver/BLAS/DNN call reachable from
/// [`try_cuda_dispatch`].
///
/// # Why this must run on every dispatch, not just once at construction
///
/// A CUDA context's "current-ness" is a property of the **OS thread** —
/// set by `cuCtxSetCurrent` — not a property of the [`CudaContext`] value
/// itself. [`CudaContext::try_new_with`] activates the context exactly
/// once, on whichever thread happens to be *building* it, but nothing
/// requires that to be the thread that later calls `try_cuda_dispatch`.
///
/// A real caller hits this today: `oxiface-convert`'s
/// `Converter::load_models_concurrently` loads the SCRFD detector and
/// ArcFace embedder on `std::thread::scope`-spawned worker threads (while
/// InSwapper loads on the calling thread), and each model's `oxionnx::Session`
/// builds its own `CudaContext` as part of that load. `thread::scope` joins
/// the workers before returning, so by the time inference actually runs —
/// on whatever thread calls into the `Converter` — the two threads that
/// built SCRFD's and ArcFace's contexts are gone. Their contexts are now
/// current on *no* thread: `ctx.dnn`'s stream, BLAS handle, and PTX-kernel
/// launches all resolve against "whatever context is current on this
/// thread" rather than a context reference they carry themselves, so the
/// memory-allocation and kernel-launch calls `cuda_matmul` et al. make
/// (`cuMemAlloc`/`cuLaunchKernel`-family, via `oxicuda-memory`'s
/// `DeviceBuffer` and `oxicuda-blas`'s GEMM dispatch) fail — permanently,
/// since nothing downstream of construction ever re-activates the context.
/// Confirmed on real hardware while validating this fix: the calling
/// thread's exact driver error depends on what (if anything) else is
/// current there — a thread that never activated *any* context observes
/// `CUDA_ERROR_INVALID_CONTEXT` ("CUDA: invalid context"); a thread with a
/// *different* context current can instead observe
/// `CUDA_ERROR_INVALID_HANDLE` ("CUDA: invalid handle") on a resource that
/// belongs to this one. Both are the same root cause. Notably,
/// `Stream::synchronize`'s `cuStreamSynchronize` call (also reachable from
/// `ctx.dnn`) does *not* fail this way on this driver even with no context
/// current on the calling thread — synchronization is more permissive than
/// allocation/launch — so it is not by itself evidence that a context is
/// usable. InSwapper's own context, built on and dispatched from the same
/// thread, is unaffected by any of this — which is what makes the bug easy
/// to miss in ad hoc testing that only exercises a single model.
///
/// `cuCtxSetCurrent` is a thread-local pointer write: no allocation, no
/// device round-trip. Paying it unconditionally on every dispatch —
/// including the common case where `ctx` is already current on the calling
/// thread — is negligible next to the alternative of a context that
/// silently, permanently stops working the moment its builder thread exits.
///
/// # Errors
/// Propagates the underlying driver error (e.g. a corrupted or already-
/// destroyed context) as an [`OnnxError::Internal`].
fn activate_context(ctx: &CudaContext) -> Result<(), OnnxError> {
    ctx.driver_context()
        .set_current()
        .map_err(|e| OnnxError::from(CudaError::from(e)))
}

/// Attempt to dispatch a single ONNX node to the CUDA backend, returning host
/// tensors.
///
/// The pre-residency entry point, and still the one every non-`oxionnx` caller
/// wants: it passes [`NoActivations`] and asks for
/// [`CudaOutputPlacement::Host`], so its behaviour is byte-for-byte what it was
/// before activations could stay on the device — one upload per operand, one
/// read-back, one fence.
///
/// Returns `Ok(Some(results))` if the op was handled by CUDA,
/// `Ok(None)` if the op is unsupported or the configuration is not
/// acceleratable (caller should try GPU/CPU fallback), or
/// `Err(OnnxError::Internal(...))` on a hard CUDA failure.
///
/// [`is_supported_op`] is the cheap pre-filter for this function: when it
/// returns `false`, this function is guaranteed to return `Ok(None)`.
///
/// # Errors
///
/// As [`try_cuda_dispatch_resident`].
pub fn try_cuda_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &CudaContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    match try_cuda_dispatch_resident(
        node,
        weights,
        intermediates,
        &NoActivations,
        CudaOutputPlacement::Host,
        ctx,
    )? {
        Some(CudaDispatchOutcome::Host(results)) => Ok(Some(results)),
        // Unreachable: `Host` placement was requested and every arm honours a
        // host request. Reported rather than unwrapped, because "unreachable"
        // is a claim about code that can be edited.
        Some(CudaDispatchOutcome::Device(_)) => Err(OnnxError::Internal(format!(
            "CUDA provider kept the result of node '{}' ({}) on the device although host \
             placement was requested",
            node.name,
            node.op.as_str(),
        ))),
        None => Ok(None),
    }
}

/// Whether the CUDA arm for `op` can bind a **device-resident** value in slot
/// `index`, rather than needing its bytes on the host.
///
/// The session consults this once per graph, before the first node runs, to
/// decide which values may stay on the device at all: a name is keepable only
/// when *every* consumer can bind it in place, because one consumer that
/// cannot would read it back a node later — the same round trip, moved.
///
/// Keeping this in step with the match arms in
/// [`try_cuda_dispatch_resident`] is the one maintenance obligation residency
/// adds. Getting it wrong is not a correctness bug — an unlisted slot only
/// means the value is read back one node earlier, and a wrongly-listed one
/// means the arm declines and the caller reads it back — but both give up the
/// traffic residency exists to remove.
///
/// Mirrors `oxionnx::session::gpu_dispatch::op_accepts_resident_slot`, which
/// answers the same question for the wgpu backend.
///
/// # Example
///
/// ```
/// use oxionnx_core::graph::OpKind;
///
/// // A convolution's activation may be resident; its filter and bias are
/// // graph initializers, read on the host and cached device-side by name.
/// assert!(oxionnx_cuda::accepts_resident_slot(&OpKind::Conv, 0));
/// assert!(!oxionnx_cuda::accepts_resident_slot(&OpKind::Conv, 1));
/// // Both operands of an element-wise binary are real bindings.
/// assert!(oxionnx_cuda::accepts_resident_slot(&OpKind::Add, 1));
/// // `Reshape`'s activation (slot 0) may be resident; its `shape` operand
/// // (slot 1) is always read on the host, like a convolution's weight.
/// assert!(oxionnx_cuda::accepts_resident_slot(&OpKind::Reshape, 0));
/// assert!(!oxionnx_cuda::accepts_resident_slot(&OpKind::Reshape, 1));
/// // No CUDA arm at all.
/// assert!(!oxionnx_cuda::accepts_resident_slot(&OpKind::Transpose, 0));
/// ```
#[must_use]
pub fn accepts_resident_slot(op: &OpKind, index: usize) -> bool {
    match op {
        // Slot 0 is the activation. A convolution's filter/bias and a Gemm's
        // `B`/`C` are graph initializers whose *host* bytes the arm reads —
        // to key them into the weight cache, and (for `transB`) to transpose
        // them — so they are never bound from a run-scoped activation.
        OpKind::Conv
        | OpKind::MatMul
        | OpKind::Gemm
        | OpKind::Relu
        | OpKind::Sigmoid
        | OpKind::Gelu
        | OpKind::Tanh
        | OpKind::Exp
        | OpKind::Sqrt
        | OpKind::Abs
        | OpKind::Neg
        | OpKind::Log
        | OpKind::Ceil
        | OpKind::Floor
        | OpKind::HardSigmoid
        | OpKind::HardSwish
        | OpKind::SiLU
        | OpKind::Softplus
        | OpKind::LeakyRelu
        | OpKind::ReduceSum
        | OpKind::ReduceMax
        | OpKind::ReduceMean
        | OpKind::Softmax => index == 0,
        // `PRelu`'s slope and `BatchNormalization`'s scale/bias/mean/var are,
        // like a convolution's filter/bias, host-read to key them into the
        // weight cache — never bound from a run-scoped activation. So is
        // `OxiInstanceNorm`, which simply has no slot beyond 0.
        OpKind::PRelu | OpKind::BatchNorm | OpKind::OxiInstanceNorm => index == 0,
        // Both operands are real bindings.
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => index < 2,
        // Slot 0 is the activation for every one of these too; their other
        // slots (`sizes`/`scales`, `pads`/`axes`, `starts`/`ends`/`axes`/
        // `steps`, a `Reshape` shape or a `Squeeze`/`Unsqueeze` axes list)
        // are read on the host to decide the dispatch's *geometry* (an
        // output shape, a coordinate-remap formula), the same treatment
        // `Conv`'s filter/bias get — never bound from a run-scoped
        // activation. See `pool`/`resize`/`pad`/`slice`/`reshape`'s own
        // module docs.
        OpKind::MaxPool
        | OpKind::AveragePool
        | OpKind::Resize
        | OpKind::Pad
        | OpKind::Slice
        | OpKind::Unsqueeze
        | OpKind::Squeeze
        | OpKind::Reshape
        | OpKind::Flatten => index == 0,
        // `Concat` has no fixed arity and no non-data slot at all -- every
        // one of its inputs is a real binding, so every index is eligible
        // (an index beyond the node's actual input count is harmless: the
        // caller's own `node.inputs.get(slot)` bounds it before this is ever
        // consulted for a slot that does not exist).
        OpKind::Concat => true,
        _ => false,
    }
}

/// One of a node's operands, wherever it currently lives.
///
/// The shape has to come from the *operand*, not from a host `Tensor`, because
/// a resident operand has no host tensor at all — that is the whole point.
struct NodeOperand<'a> {
    /// The ONNX shape.
    shape: &'a [usize],
    /// Elements the operand holds.
    len: usize,
    /// Where the kernel will read it from.
    binding: InputBinding<'a>,
    /// The host tensor, when the operand has one. `None` for a resident value,
    /// which is what stops it being keyed into the weight cache or handed to
    /// the shadow-verification oracle.
    host: Option<&'a Tensor>,
}

/// Attempt to dispatch a single ONNX node to the CUDA backend, binding
/// device-resident operands in place and optionally leaving the result on the
/// device.
///
/// This is the dispatcher proper; [`try_cuda_dispatch`] is a thin wrapper that
/// supplies an empty activation map and a host placement request.
///
/// Returns `Ok(Some(outcome))` if CUDA handled the node, `Ok(None)` if the op
/// is unsupported or this node's configuration is not acceleratable (the
/// caller falls back), or `Err` on a hard CUDA failure.
///
/// # Thread affinity
///
/// Unlike a raw driver context, callers do **not** need to dispatch from
/// the same OS thread that built `ctx`: this function re-activates `ctx`
/// on the calling thread itself (see the private `activate_context` helper
/// above) before doing anything else, defensively, on every call. See its doc
/// comment for the concrete scenario (concurrent model loading) that makes
/// this necessary.
///
/// # `weights` is treated as invariant, and that is a contract
///
/// `ctx` caches device copies of the tensors it finds in `weights`, keyed by
/// name, for its whole lifetime — that is what stops a graph initializer from
/// crossing the bus once per node per frame (see [`mod@residency`]). The
/// contract that makes it sound is the one `oxionnx::Session` already
/// satisfies by construction: **a name in `weights` denotes the same bytes for
/// as long as `ctx` lives.** A session builds its initializer map once at load
/// time and never mutates it, so this holds for every caller that goes through
/// `Session::run`.
///
/// A direct caller that violates it — passing one context two different weight
/// maps that share a name — is *usually* caught rather than silently served
/// stale numbers: an entry also records the address and length of the host
/// allocation it was uploaded from, and a mismatch demotes that operand to a
/// per-dispatch upload without disturbing the cache. That is a backstop, not a
/// guarantee (an allocator may hand a freed address straight back). A caller
/// that needs to swap weights should build a new context, or call
/// [`CudaContext::release_device_caches`] between the two.
///
/// `intermediates` is never cached: those bytes are this run's activations.
/// Neither is anything `activations` holds — the session owns those buffers
/// and releases them on its own last-use schedule.
///
/// # Interaction with `OXIONNX_CUDA_VERIFY`
///
/// Shadow verification needs the exact host bytes the kernel read *and* the
/// exact host bytes it wrote, and a resident operand has neither. The session
/// therefore switches activation residency off wholesale when
/// [`reference::verify_enabled`] is true, so a verifying run is the fully
/// materialising run it always was — slower, and comparable node for node with
/// the oracle. The guard here is the belt to that braces: an arm whose result
/// stayed on the device is not verified (there is nothing to compare), and an
/// oracle that cannot be built from host bytes reports "no formula" rather
/// than comparing against a substitute.
///
/// # Errors
///
/// [`OnnxError::Internal`] wrapping a [`CudaError`] on a driver, PTX or
/// allocation failure, or [`error::CudaDispatchError::Verify`] when
/// `OXIONNX_CUDA_VERIFY=1` proved a kernel wrong and `OXIONNX_CUDA_STRICT=1`
/// says that ends the run.
pub fn try_cuda_dispatch_resident(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    activations: &dyn ResidentActivations,
    placement: CudaOutputPlacement,
    ctx: &CudaContext,
) -> Result<Option<CudaDispatchOutcome>, OnnxError> {
    activate_context(ctx)?;

    let resolve = |name: &str| -> Option<&Tensor> {
        if name.is_empty() {
            None
        } else {
            intermediates.get(name).or_else(|| weights.get(name))
        }
    };

    // The residency-aware operand lookup: a value this run left on the device
    // is bound in place, everything else resolves to its host tensor exactly as
    // before. Arms that read an operand's *contents* on the host (an `axes`
    // list, a bias to fold, a weight to transpose) keep using `resolve` —
    // which is why `accepts_resident_slot` names slots and not just ops.
    let operand = |slot: usize| -> Option<NodeOperand<'_>> {
        let name = node.inputs.get(slot)?;
        if name.is_empty() {
            return None;
        }
        if accepts_resident_slot(&node.op, slot) {
            if let Some(tensor) = activations.resident(name) {
                return Some(NodeOperand {
                    shape: tensor.shape(),
                    len: tensor.len(),
                    binding: InputBinding::Device(tensor),
                    host: None,
                });
            }
        }
        let tensor = resolve(name)?;
        Some(NodeOperand {
            shape: &tensor.shape,
            len: tensor.data.len(),
            binding: InputBinding::Host(&tensor.data),
            host: Some(tensor),
        })
    };

    // A node with more than one declared output cannot hand back a single
    // device buffer, and no op below produces more than one result — so this
    // only ever demotes a malformed request, never a legitimate one.
    let placement = if node.outputs.len() == 1 {
        placement
    } else {
        CudaOutputPlacement::Host
    };

    match &node.op {
        // ------------------------------------------------------------------ //
        // MatMul / Gemm                                                        //
        // ------------------------------------------------------------------ //
        OpKind::MatMul | OpKind::Gemm => {
            let (Some(a), Some(b)) = (operand(0), operand(1)) else {
                return Ok(None);
            };
            // Extract Gemm attributes (MatMul uses defaults).
            let is_gemm = matches!(node.op, OpKind::Gemm);
            let alpha = if is_gemm {
                node.attrs.f("alpha", 1.0)
            } else {
                1.0
            };
            let beta = if is_gemm {
                node.attrs.f("beta", 1.0)
            } else {
                0.0
            };
            let trans_a = is_gemm && node.attrs.i("transA", 0) != 0;
            let trans_b = is_gemm && node.attrs.i("transB", 0) != 0;

            // A transposed operand is consumed as a *different byte sequence*
            // from the one on the device, and this crate has no transpose
            // kernel — the transpose is a host copy. So a resident operand the
            // node wants transposed cannot be bound in place. Declining sends
            // the node to the CPU, which materialises it; the alternative
            // (transposing on the host) would need the bytes read back anyway.
            // Not observed in any model in this workspace: `transA` is for a
            // weight-shaped operand, and weights are not run-scoped
            // activations.
            if (trans_a && a.host.is_none()) || (trans_b && b.host.is_none()) {
                return Ok(None);
            }

            let an = a.shape.len();
            let bn = b.shape.len();
            if an < 2 || bn < 2 {
                return Ok(None);
            }
            // Determine M, K, N accounting for transposes.
            let m = if trans_a {
                a.shape[an - 1]
            } else {
                a.shape[an - 2]
            };
            let k = if trans_a {
                a.shape[an - 2]
            } else {
                a.shape[an - 1]
            };
            let k2 = if trans_b {
                b.shape[bn - 1]
            } else {
                b.shape[bn - 2]
            };
            let n = if trans_b {
                b.shape[bn - 2]
            } else {
                b.shape[bn - 1]
            };

            // A malformed model (K mismatch) or a degenerate zero-sized
            // operand is declined rather than risking a divide-by-zero in the
            // bias epilogue or a nonsensical GPU launch; the CPU kernel raises
            // the proper diagnostic for the former.
            if k != k2 || m == 0 || k == 0 || n == 0 {
                return Ok(None);
            }

            // Batch-broadcast the leading dims with the same numpy rule the
            // CPU sibling (`oxionnx-ops::math::matmul::matmul`) uses for its
            // output shape. Unlike that CPU path, which unconditionally
            // modulo-indexes each operand (`b_idx % operand_batches`), we
            // additionally require each operand's own batch count to be
            // exactly `1` (broadcasts) or exactly `batch` (no broadcast on
            // that operand) before dispatching to the GPU: unconditional
            // modulo indexing silently computes the *wrong* slice whenever
            // both operands broadcast on different sub-axes of a
            // multi-dimensional batch — e.g. A's batch `[2, 1]` against B's
            // `[1, 3]` needs the A-slice sequence `0,0,0,1,1,1`, but `i % 2`
            // yields `0,1,0,1,0,1`. Declining that narrow case is a missed
            // acceleration, not a wrong answer: the CPU path runs it either
            // way.
            let a_batch_dims = &a.shape[..an - 2];
            let b_batch_dims = &b.shape[..bn - 2];
            let Ok(out_batch) = Tensor::broadcast_shape(a_batch_dims, b_batch_dims) else {
                return Ok(None);
            };
            let batch: usize = out_batch.iter().product::<usize>().max(1);
            let a_batches: usize = a_batch_dims.iter().product::<usize>().max(1);
            let b_batches: usize = b_batch_dims.iter().product::<usize>().max(1);
            if !(a_batches == 1 || a_batches == batch) || !(b_batches == 1 || b_batches == batch) {
                return Ok(None);
            }

            // Checked size math: these are all derived from model-supplied
            // shape dims, so a corrupted/adversarial shape must overflow into
            // a decline, never a panic or a silently-wrapped (wrong) buffer
            // size.
            let (Some(slice_a), Some(slice_b), Some(slice_c)) =
                (m.checked_mul(k), k.checked_mul(n), m.checked_mul(n))
            else {
                return Ok(None);
            };
            let (Some(a_needed), Some(b_needed), Some(out_total)) = (
                a_batches.checked_mul(slice_a),
                b_batches.checked_mul(slice_b),
                batch.checked_mul(slice_c),
            ) else {
                return Ok(None);
            };
            // Bounds-check up front rather than trusting `Tensor::new`'s
            // debug-only length invariant (a malformed model can violate it in
            // a release build).
            if a.len < a_needed || b.len < b_needed {
                return Ok(None);
            }

            // ── Operand residency ─────────────────────────────────────────
            //
            // A `MatMul`/`Gemm` operand that resolved out of `weights` rather
            // than `intermediates` is a graph initializer: the same bytes on
            // frame 1 and frame 10 000. `initializer_id` says so, keyed
            // additionally by the *form* the GEMM needs, because a
            // `transA`/`transB` node consumes the transpose of those bytes
            // rather than the bytes themselves and the two must never share a
            // cache slot. An operand bound from a run-scoped activation has no
            // identity at all — `a.host` is `None`, and `initializer_id` is
            // never reached for it.
            let a_id = a.host.and_then(|tensor| {
                initializer_id(
                    node.inputs.first().map(String::as_str),
                    weights,
                    intermediates,
                    activations,
                    tensor,
                    operand_form(trans_a),
                )
            });
            let b_id = b.host.and_then(|tensor| {
                initializer_id(
                    node.inputs.get(1).map(String::as_str),
                    weights,
                    intermediates,
                    activations,
                    tensor,
                    operand_form(trans_b),
                )
            });

            // Ask the device before building anything. A resident transposed
            // weight means the `transpose_2d_batched` call below -- an O(k*n)
            // host copy, ~12 MiB of memcpy for ArcFace's head -- does not
            // happen at all this frame.
            let a_resident =
                a_id.and_then(|id| ctx.resident_operand(id, matmul::A_LABEL, a_needed));
            let b_resident =
                b_id.and_then(|id| ctx.resident_operand(id, matmul::B_LABEL, b_needed));

            // Materialise only what still has to cross the bus, and only when
            // it genuinely differs from the tensor's own bytes. Transposition
            // is done per the operand's *own* batch count, not the (possibly
            // larger) broadcast `batch`: the host data holds only
            // `a_batches`/`b_batches` slices, and transposing past that would
            // read out of bounds.
            //
            // The untransposed case borrows the tensor's data in place. It
            // used to `.clone()` it -- a full host copy of every operand on
            // every dispatch, which for a 49 MiB weight is a memcpy the size of
            // the upload it was feeding.
            //
            // `|| verify_on` is not an optimisation gap, it closes one: the
            // oracle below has to see the same bytes the GPU multiplied, and
            // for a *resident* transposed weight those bytes exist only on the
            // device. Under `OXIONNX_CUDA_VERIFY=1` the transpose is therefore
            // rebuilt anyway -- a diagnostic mode that already recomputes every
            // op on the CPU is the right place to pay it, and skipping it would
            // silently compare against the untransposed operand and report a
            // mismatch on a correct dispatch.
            let verify_on = reference::verify_enabled();
            let a_transposed = (trans_a && (a_resident.is_none() || verify_on))
                .then(|| {
                    a.host.map(|tensor| {
                        transpose_2d_batched(
                            &tensor.data,
                            a_batches,
                            a.shape[an - 2],
                            a.shape[an - 1],
                        )
                    })
                })
                .flatten();
            let b_transposed = (trans_b && (b_resident.is_none() || verify_on))
                .then(|| {
                    b.host.map(|tensor| {
                        transpose_2d_batched(
                            &tensor.data,
                            b_batches,
                            b.shape[bn - 2],
                            b.shape[bn - 1],
                        )
                    })
                })
                .flatten();

            let a_operand = match (a_resident, a.binding) {
                (Some(resident), _) => matmul::GemmOperand::from_resident(resident),
                // A run-scoped activation is already a device buffer: bind it
                // exactly as a resident weight is bound. This is the edge the
                // whole wave turns on -- for SCRFD's `Conv -> Relu -> Add`
                // chains it is every operand of every node after the first.
                (None, InputBinding::Device(tensor)) => {
                    ctx.caches.note_resident_bind();
                    matmul::GemmOperand::from_resident(residency::Operand::Resident(tensor.share()))
                }
                (None, InputBinding::Host(data)) => {
                    matmul::GemmOperand::from_host(a_transposed.as_deref().unwrap_or(data), a_id)
                }
            };
            let b_operand = match (b_resident, b.binding) {
                (Some(resident), _) => matmul::GemmOperand::from_resident(resident),
                (None, InputBinding::Device(tensor)) => {
                    ctx.caches.note_resident_bind();
                    matmul::GemmOperand::from_resident(residency::Operand::Resident(tensor.share()))
                }
                (None, InputBinding::Host(data)) => {
                    matmul::GemmOperand::from_host(b_transposed.as_deref().unwrap_or(data), b_id)
                }
            };

            // ── Can the result stay on the device? ────────────────────────
            //
            // Only when nothing is left to do to it on the host. Two Gemm
            // epilogues are computed host-side by design: `alpha` (kept off the
            // kernel so the oracle grades an unscaled product -- see the
            // `matmul` module docs) and the `beta * C` bias. Either one demotes
            // the request to a read-back. Both are absent from a plain
            // `MatMul`, and `alpha = 1, beta = 0` covers the fused-bias Gemms
            // this pipeline actually runs... but not ArcFace's embedding head,
            // whose bias is folded here and which therefore still reads back.
            let host_epilogue = (alpha - 1.0).abs() > f32::EPSILON
                || (is_gemm
                    && beta.abs() > f32::EPSILON
                    && node.inputs.get(2).and_then(|name| resolve(name)).is_some());
            let effective_placement = if host_epilogue {
                CudaOutputPlacement::Host
            } else {
                placement
            };

            let mut out_shape = out_batch;
            out_shape.push(m);
            out_shape.push(n);

            // One upload / launch / readback for the whole batch, replacing
            // what used to be `batch` complete round trips. The broadcast rule
            // the loop expressed as `(i % operand_batches) * slice` is carried
            // through unchanged as a batch stride -- see `matmul::plan_gemm`.
            let request = matmul::BatchedGemm {
                a: a_operand,
                b: b_operand,
                m,
                k,
                n,
                batch,
                a_batches,
                b_batches,
            };
            let Some(kernel_out) =
                matmul::cuda_gemm_batched_placed(ctx, request, &out_shape, effective_placement)
                    .map_err(OnnxError::from)?
            else {
                return Ok(None);
            };

            let mut out = match kernel_out {
                activation::KernelOutput::Device(tensor) => {
                    return Ok(Some(CudaDispatchOutcome::Device(tensor)));
                }
                activation::KernelOutput::Host(data) => data,
            };

            // Shadow-verify the whole batch at once against the same per-slice
            // oracle the loop used, slice by slice, with the same
            // `(i % operand_batches)` indexing. Built only when verification is
            // actually on: `verify_or_fallback` takes the oracle as a closure
            // precisely so this costs nothing in production.
            if !verify_or_fallback("MatMul/Gemm", &out, || {
                let a_data = a_transposed.as_deref().or_else(|| a.binding.host())?;
                let b_data = b_transposed.as_deref().or_else(|| b.binding.host())?;
                let mut expected = Vec::with_capacity(out_total);
                for i in 0..batch {
                    let a_start = (i % a_batches) * slice_a;
                    let b_start = (i % b_batches) * slice_b;
                    // `.get()` rather than direct indexing: never index a
                    // model-derived slice range without a bounds check, even
                    // though `a_needed`/`b_needed` above already guarantee this
                    // succeeds.
                    let (Some(a_slice), Some(b_slice)) = (
                        a_data.get(a_start..a_start + slice_a),
                        b_data.get(b_start..b_start + slice_b),
                    ) else {
                        // The oracle cannot be built, so there is nothing to
                        // compare against; `shadow_verify` treats `None` as "no
                        // formula" and says so loudly rather than passing
                        // silently.
                        return None;
                    };
                    expected.extend(reference::ref_matmul(a_slice, b_slice, m, k, n));
                }
                Some(expected)
            })? {
                return Ok(None);
            }

            // Apply alpha scaling — after verification, so the oracle stays the
            // plain unscaled product. See the `matmul` module docs' "Alpha
            // stays on the host".
            if (alpha - 1.0).abs() > f32::EPSILON {
                for v in &mut out {
                    *v *= alpha;
                }
            }

            // Gemm: C = alpha * A @ B + beta * bias
            if is_gemm && beta.abs() > f32::EPSILON {
                if let Some(bias) = node.inputs.get(2).and_then(|n| resolve(n)) {
                    if !apply_gemm_bias(&mut out, &bias.data, &bias.shape, m, n, beta) {
                        // `bias`'s shape isn't unidirectionally broadcastable
                        // to [m, n] — a malformed model. Decline rather than
                        // silently dropping the bias; the CPU kernel raises a
                        // proper diagnostic.
                        return Ok(None);
                    }
                }
            }

            Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                out, out_shape,
            )])))
        }

        // ------------------------------------------------------------------ //
        // Conv                                                                  //
        //                                                                      //
        // `conv::cuda_conv_bound` computes a real answer for three validated   //
        // shape classes — 1x1, true-depthwise, and the general                 //
        // symmetric-padding case, dispatched straight to `oxicuda-dnn`'s       //
        // `Conv1x1` / `DepthwiseConv` / `ImplicitGemmConv` engines             //
        // respectively (see the `conv` module docs' "Dispatch rule"; note it   //
        // never goes through `conv_forward`'s auto-selector) — and declines    //
        // (`Ok(None)`) everything else, so this arm can yield either           //
        // `Ok(Some(_))` or `Ok(None)` depending on the node's configuration.   //
        // Its output is shadow-verified against `reference::ref_conv` through  //
        // the same `verify_or_fallback` gate every other claimable op below    //
        // uses, so `OXIONNX_CUDA_VERIFY=1` covers a Conv dispatch exactly like //
        // it covers MatMul/elementwise/reduce/Softmax. `is_supported_op`       //
        // reports `true` for `Conv`, so `decide_placement` routes production   //
        // convolutions here — this arm is on the hot path, not test-only.      //
        // ------------------------------------------------------------------ //
        OpKind::Conv => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let Some(weight) = node.inputs.get(1).and_then(|n| resolve(n)) else {
                return Ok(None);
            };
            let bias = node.inputs.get(2).and_then(|n| resolve(n));

            // Every attribute that changes what a `Conv` node computes is
            // resolved in exactly one place, and anything this backend does not
            // model declines the node rather than being ignored — see
            // [`conv::conv_params_from_attrs`].
            //
            // This arm used to read `strides`/`pads`/`dilations`/`group`
            // inline and nothing else. That silently dropped the optimizer's
            // *fused activation* — `oxionnx` rewrites every `Conv -> Relu` pair
            // into one `Conv_*_fused_activation` node carrying
            // `activation="relu"`, so 26 of SCRFD det_10g's 58 convolutions
            // returned their raw, un-rectified output. The failure was
            // invisible to `OXIONNX_CUDA_VERIFY=1` because the oracle read the
            // same `ConvParams` and therefore skipped the same activation; see
            // `reference::ref_conv`.
            let Some(conv_params) =
                conv::conv_params_from_attrs(&node.attrs, input.shape, &weight.shape)
            else {
                return Ok(None);
            };

            // A convolution's filter and bias are the megabyte-scale invariant
            // bytes in this workload -- InSwapper-128 alone re-uploaded ~503 MB
            // of them per forward pass before residency existed. The input
            // activation deliberately has no identity: it is this frame's data.
            let conv_ids = conv::ConvWeightIds {
                weight: initializer_id(
                    node.inputs.get(1).map(String::as_str),
                    weights,
                    intermediates,
                    activations,
                    weight,
                    residency::OperandForm::Raw,
                ),
                bias: bias.and_then(|bias_tensor| {
                    initializer_id(
                        node.inputs.get(2).map(String::as_str),
                        weights,
                        intermediates,
                        activations,
                        bias_tensor,
                        residency::OperandForm::Raw,
                    )
                }),
            };

            match conv::cuda_conv_bound(
                ctx,
                input.binding,
                input.shape,
                weight,
                bias,
                &conv_params,
                conv_ids,
                placement,
            )
            .map_err(OnnxError::from)?
            {
                Some(conv::ConvOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(conv::ConvOutput::Host(tensor)) => {
                    let bias_data = bias.map(|b| b.data.as_slice());
                    let input_shape = input.shape.to_vec();
                    if !verify_or_fallback("Conv", &tensor.data, || {
                        Some(reference::ref_conv(
                            input.binding.host()?,
                            &weight.data,
                            bias_data,
                            &input_shape,
                            &weight.shape,
                            &conv_params,
                        ))
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![tensor])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // Unary elementwise activations                                        //
        // ------------------------------------------------------------------ //
        OpKind::Relu
        | OpKind::Sigmoid
        | OpKind::Gelu
        | OpKind::Tanh
        | OpKind::Exp
        | OpKind::Sqrt
        | OpKind::Abs
        | OpKind::Neg
        | OpKind::Log
        | OpKind::Ceil
        | OpKind::Floor
        | OpKind::HardSigmoid
        | OpKind::HardSwish
        | OpKind::SiLU
        | OpKind::Softplus
        | OpKind::LeakyRelu => {
            // The PTX kernels for these three ops hard-code one specific
            // constant configuration with no launch-time override to
            // change it (see `oxicuda_ptx`'s `generate_leaky_relu` /
            // `generate_hard_sigmoid` / `generate_gelu`):
            //   - LeakyRelu:   alpha = 0.01 (the ONNX default)
            //   - HardSigmoid: alpha = 0.2, beta = 0.5 (the ONNX defaults)
            //   - Gelu:        the `tanh`-approximation formula — NOT the
            //                  ONNX-20 *default* `approximate="none"`
            //                  exact/erf formula.
            // A node whose attributes don't match the constant the kernel
            // actually computes is declined so the attribute-aware CPU
            // kernel handles it.
            const LEAKY_RELU_DEFAULT_ALPHA: f32 = 0.01;
            const HARD_SIGMOID_DEFAULT_ALPHA: f32 = 0.2;
            const HARD_SIGMOID_DEFAULT_BETA: f32 = 0.5;
            match node.op {
                OpKind::LeakyRelu
                    if (node.attrs.f("alpha", LEAKY_RELU_DEFAULT_ALPHA)
                        - LEAKY_RELU_DEFAULT_ALPHA)
                        .abs()
                        > f32::EPSILON =>
                {
                    return Ok(None);
                }
                OpKind::HardSigmoid
                    if (node.attrs.f("alpha", HARD_SIGMOID_DEFAULT_ALPHA)
                        - HARD_SIGMOID_DEFAULT_ALPHA)
                        .abs()
                        > f32::EPSILON
                        || (node.attrs.f("beta", HARD_SIGMOID_DEFAULT_BETA)
                            - HARD_SIGMOID_DEFAULT_BETA)
                            .abs()
                            > f32::EPSILON =>
                {
                    return Ok(None);
                }
                // Opposite polarity from the two ops above: the kernel
                // computes the *tanh* approximation, which is only correct
                // when the node explicitly asks for it. The ONNX-default
                // (attribute absent, or `"none"`) exact erf-based formula is
                // not what this kernel computes.
                OpKind::Gelu if node.attrs.s("approximate") != "tanh" => {
                    return Ok(None);
                }
                _ => {}
            }

            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let op_name = node.op.as_str();
            let out_shape = input.shape.to_vec();
            let out = elementwise::cuda_elementwise_bound(
                ctx,
                input.binding,
                &out_shape,
                op_name,
                placement,
            )
            .map_err(OnnxError::from)?;
            match out {
                activation::KernelOutput::Device(tensor) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                activation::KernelOutput::Host(data) => {
                    if !verify_or_fallback(op_name, &data, || {
                        reference::ref_unary_vec(&node.op, input.binding.host()?)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, out_shape,
                    )])))
                }
            }
        }

        // ------------------------------------------------------------------ //
        // Binary elementwise (Add, Sub, Mul, Div)                              //
        // ------------------------------------------------------------------ //
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            let (Some(a), Some(b)) = (operand(0), operand(1)) else {
                return Ok(None);
            };
            let op_name = node.op.as_str();

            // Fast, exact path: identical shapes, no broadcasting to reason
            // about at all.
            if a.shape == b.shape {
                let out_shape = a.shape.to_vec();
                let out = elementwise::cuda_binary_elementwise_bound(
                    ctx, a.binding, b.binding, &out_shape, op_name, placement,
                )
                .map_err(OnnxError::from)?;
                return match out {
                    activation::KernelOutput::Device(tensor) => {
                        Ok(Some(CudaDispatchOutcome::Device(tensor)))
                    }
                    activation::KernelOutput::Host(data) => {
                        if !verify_or_fallback(op_name, &data, || {
                            reference::ref_binary_vec(
                                &node.op,
                                a.binding.host()?,
                                b.binding.host()?,
                            )
                        })? {
                            return Ok(None);
                        }
                        Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                            data, out_shape,
                        )])))
                    }
                };
            }

            // Shapes disagree: try the narrow `[1,C,1,1]`-vs-`[1,C,H,W]` /
            // scalar-vs-tensor broadcast pattern the op-coverage audit found
            // behind every one of the three real face-pipeline models'
            // declined Add/Sub/Mul/Div nodes -- see `broadcast`'s module
            // docs. Any other broadcast shape still declines (`Ok(None)`),
            // exactly as before this arm existed.
            let Some(plan) = broadcast::classify(a.shape, b.shape) else {
                return Ok(None);
            };
            let (full, small) = if plan.lhs_is_small {
                (&b, &a)
            } else {
                (&a, &b)
            };
            let out_shape = full.shape.to_vec();
            let out = broadcast::cuda_broadcast_bound(
                ctx,
                full.binding,
                small.binding,
                plan,
                &node.op,
                &out_shape,
                placement,
            )
            .map_err(OnnxError::from)?;
            match out {
                activation::KernelOutput::Device(tensor) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                activation::KernelOutput::Host(data) => {
                    if !verify_or_fallback(op_name, &data, || {
                        reference::ref_binary_broadcast(
                            &node.op,
                            full.binding.host()?,
                            small.binding.host()?,
                            plan.channels,
                            plan.spatial,
                            broadcast::reverse_for(&node.op, plan),
                        )
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, out_shape,
                    )])))
                }
            }
        }

        // ------------------------------------------------------------------ //
        // PRelu                                                                //
        // ------------------------------------------------------------------ //
        OpKind::PRelu => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let Some(slope) = node.inputs.get(1).and_then(|n| resolve(n)) else {
                return Ok(None);
            };
            let slope_id = initializer_id(
                node.inputs.get(1).map(String::as_str),
                weights,
                intermediates,
                activations,
                slope,
                residency::OperandForm::Raw,
            );
            let in_shape = input.shape.to_vec();
            match prelu::cuda_prelu_bound(
                ctx,
                input.binding,
                &in_shape,
                &slope.data,
                slope_id,
                placement,
            )
            .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("PRelu", &data, || {
                        let (channels, spatial, _) =
                            prelu::prelu_plan(&in_shape, slope.data.len())?;
                        reference::ref_prelu(input.binding.host()?, &slope.data, channels, spatial)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, in_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // BatchNormalization (inference)                                       //
        // ------------------------------------------------------------------ //
        OpKind::BatchNorm => {
            // Training mode has three outputs (running-stat updates this
            // dispatcher never computes) and this node declares one -- decline
            // rather than silently returning only the normalised activation.
            if node.attrs.i("training_mode", 0) != 0 {
                return Ok(None);
            }
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let (Some(scale), Some(bias), Some(mean), Some(var)) = (
                node.inputs.get(1).and_then(|n| resolve(n)),
                node.inputs.get(2).and_then(|n| resolve(n)),
                node.inputs.get(3).and_then(|n| resolve(n)),
                node.inputs.get(4).and_then(|n| resolve(n)),
            ) else {
                return Ok(None);
            };
            let epsilon = node.attrs.f("epsilon", 1e-5);
            let ids = norm::BatchNormWeightIds {
                scale: initializer_id(
                    node.inputs.get(1).map(String::as_str),
                    weights,
                    intermediates,
                    activations,
                    scale,
                    residency::OperandForm::Raw,
                ),
                bias: initializer_id(
                    node.inputs.get(2).map(String::as_str),
                    weights,
                    intermediates,
                    activations,
                    bias,
                    residency::OperandForm::Raw,
                ),
                mean: initializer_id(
                    node.inputs.get(3).map(String::as_str),
                    weights,
                    intermediates,
                    activations,
                    mean,
                    residency::OperandForm::Raw,
                ),
                var: initializer_id(
                    node.inputs.get(4).map(String::as_str),
                    weights,
                    intermediates,
                    activations,
                    var,
                    residency::OperandForm::Raw,
                ),
            };
            let in_shape = input.shape.to_vec();
            match norm::cuda_batch_norm_bound(
                ctx,
                input.binding,
                &in_shape,
                &scale.data,
                &bias.data,
                &mean.data,
                &var.data,
                epsilon,
                ids,
                placement,
            )
            .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("BatchNormalization", &data, || {
                        let (_, channels, spatial) = norm::batch_norm_plan(&in_shape)?;
                        reference::ref_batch_norm(
                            input.binding.host()?,
                            &scale.data,
                            &bias.data,
                            &mean.data,
                            &var.data,
                            channels,
                            spatial,
                            epsilon,
                        )
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, in_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // OxiInstanceNorm                                                      //
        // ------------------------------------------------------------------ //
        OpKind::OxiInstanceNorm => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let epsilon = node.attrs.f("epsilon", 1e-5);
            let in_shape = input.shape.to_vec();
            match norm::cuda_oxi_instance_norm_bound(
                ctx,
                input.binding,
                &in_shape,
                epsilon,
                placement,
            )
            .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("OxiInstanceNorm", &data, || {
                        reference::ref_oxi_instance_norm(input.binding.host()?, &in_shape, epsilon)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, in_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // ReduceMean (one or more contiguous axes)                             //
        // ------------------------------------------------------------------ //
        OpKind::ReduceMean => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let raw_axes = node.attrs.ints("axes");
            let rank = input.shape.len();
            let Some((start_axis, end_axis)) = reduce::resolve_contiguous_axes(rank, raw_axes)
            else {
                return Ok(None);
            };
            let keepdims = node.attrs.i("keepdims", 1) != 0;
            let mut out_shape = input.shape.to_vec();
            if keepdims {
                for ax in &mut out_shape[start_axis..=end_axis] {
                    *ax = 1;
                }
            } else {
                out_shape.drain(start_axis..=end_axis);
            }
            let in_shape = input.shape.to_vec();
            match reduce::cuda_reduce_mean_bound(
                ctx,
                input.binding,
                &in_shape,
                start_axis,
                end_axis,
                &out_shape,
                placement,
            )
            .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("ReduceMean", &data, || {
                        let host = input.binding.host()?;
                        let outer: usize = in_shape[..start_axis].iter().product();
                        let axis_len: usize = in_shape[start_axis..=end_axis].iter().product();
                        let inner: usize = in_shape[end_axis + 1..].iter().product();
                        reference::ref_reduce(
                            &OpKind::ReduceMean,
                            host,
                            &[outer, axis_len, inner],
                            1,
                        )
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, out_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // Reductions                                                           //
        // ------------------------------------------------------------------ //
        OpKind::ReduceSum | OpKind::ReduceMax => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let axes = node.attrs.ints("axes");
            if axes.len() != 1 {
                return Ok(None);
            }
            let rank = input.shape.len();
            let raw_axis = axes[0];
            // ONNX permits a negative axis, resolved against rank.
            let resolved_axis = if raw_axis < 0 {
                raw_axis + rank as i64
            } else {
                raw_axis
            };
            if resolved_axis < 0 || resolved_axis as usize >= rank {
                // Out of range for `input`'s rank — a malformed model; decline
                // and let the CPU kernel raise the proper diagnostic.
                return Ok(None);
            }
            let axis = resolved_axis as usize;
            // ONNX ReduceSum/ReduceMax default `keepdims` to 1 (the reduced
            // axis stays, size 1); `keepdims=0` removes it from the output
            // shape entirely. `cuda_reduce`'s data layout (`[outer, inner]`)
            // is identical either way — only the declared shape differs.
            let keepdims = node.attrs.i("keepdims", 1) != 0;
            let mut out_shape = input.shape.to_vec();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            let op_name = node.op.as_str();
            let in_shape = input.shape.to_vec();
            match reduce::cuda_reduce_bound(
                ctx,
                input.binding,
                &in_shape,
                axis,
                op_name,
                &out_shape,
                placement,
            )
            .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback(op_name, &data, || {
                        reference::ref_reduce(&node.op, input.binding.host()?, &in_shape, axis)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, out_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // Softmax                                                              //
        // ------------------------------------------------------------------ //
        OpKind::Softmax => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let rank = input.shape.len();
            if rank == 0 {
                return Ok(None);
            }
            let raw_axis = node.attrs.i("axis", -1);
            let resolved_axis = if raw_axis < 0 {
                raw_axis + rank as i64
            } else {
                raw_axis
            };
            // Out of `[-rank, rank)` is a malformed model — decline and let the
            // CPU kernel raise the proper diagnostic. Anything other than the
            // last dimension is not malformed, but `cuda_softmax`'s row/batch
            // decomposition cannot express it (it always normalizes
            // `shape[rank-1]`), so it also declines, cleanly, to the axis-aware
            // CPU kernel.
            if resolved_axis < 0
                || resolved_axis as usize >= rank
                || resolved_axis as usize != rank - 1
            {
                return Ok(None);
            }
            let out_shape = input.shape.to_vec();
            match softmax::cuda_softmax_bound(ctx, input.binding, &out_shape, placement)
                .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("Softmax", &data, || {
                        reference::ref_softmax(input.binding.host()?, &out_shape)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, out_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // MaxPool / AveragePool                                                //
        // ------------------------------------------------------------------ //
        OpKind::MaxPool | OpKind::AveragePool => {
            // `MaxPool`'s optional second output (`Indices`) has no matching
            // kernel here -- `oxicuda_dnn::pool::max_pool2d`'s index encoding
            // (a per-plane `H*W` offset) does not match `oxionnx-ops`' CPU
            // encoding, nor its `storage_order=1` column-major form. A node
            // that actually wants it declines to the CPU rather than being
            // silently served the wrong numbers under the right shape.
            if node.outputs.get(1).is_some_and(|name| !name.is_empty()) {
                return Ok(None);
            }
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let Some(params) = pool::pool_params_from_attrs(&node.attrs) else {
                return Ok(None);
            };
            let kind = if matches!(node.op, OpKind::MaxPool) {
                pool::PoolKind::Max
            } else {
                pool::PoolKind::Avg
            };
            let Some(out_shape) = pool::pool_output_shape(input.shape, &params) else {
                return Ok(None);
            };
            let op_name = node.op.as_str();
            let in_shape = input.shape.to_vec();
            match pool::cuda_pool_bound(ctx, input.binding, &in_shape, kind, &params, placement)
                .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback(op_name, &data, || {
                        reference::ref_pool(input.binding.host()?, &in_shape, kind, &params)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data,
                        out_shape.to_vec(),
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // Resize (nearest / bilinear)                                          //
        // ------------------------------------------------------------------ //
        OpKind::Resize => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            // Resize(11+): `(X, roi, scales, sizes)`. `roi` is never read --
            // it is only meaningful for `coordinate_transformation_mode =
            // "tf_crop_and_resize"`, which `resize_params_from_node` already
            // declines. Opset-10's 2-input `(X, scales)` layout is not
            // supported (falls back to the CPU): both models this crate
            // targets use the opset-11+ layout. Read on the host regardless
            // of residency, like a convolution's weight -- the *values*, not
            // just the shape, decide this dispatch's output geometry.
            let sizes = node
                .inputs
                .get(3)
                .and_then(|n| resolve(n))
                .filter(|t| !t.data.is_empty());
            let scales = node
                .inputs
                .get(2)
                .and_then(|n| resolve(n))
                .filter(|t| !t.data.is_empty());
            let Some(params) = resize::resize_params_from_node(
                &node.attrs,
                input.shape,
                sizes.map(|t| t.data.as_slice()),
                scales.map(|t| t.data.as_slice()),
            ) else {
                return Ok(None);
            };
            let out_shape = vec![input.shape[0], input.shape[1], params.out_h, params.out_w];
            let in_shape = input.shape.to_vec();
            match resize::cuda_resize_bound(ctx, input.binding, &in_shape, &params, placement)
                .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("Resize", &data, || {
                        reference::ref_resize(input.binding.host()?, &in_shape, &params)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, out_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // Pad (reflect / constant)                                             //
        // ------------------------------------------------------------------ //
        OpKind::Pad => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let Some(pads_tensor) = node.inputs.get(1).and_then(|n| resolve(n)) else {
                return Ok(None);
            };
            let pads: Vec<i64> = pads_tensor.data.iter().map(|&v| v as i64).collect();
            let constant_value = node
                .inputs
                .get(2)
                .and_then(|n| resolve(n))
                .and_then(|t| t.data.first().copied())
                .unwrap_or(0.0);
            let axes: Option<Vec<i64>> = node
                .inputs
                .get(3)
                .and_then(|n| resolve(n))
                .map(|t| t.data.iter().map(|&v| v as i64).collect());
            let Some(params) = pad::pad_params_from_node(
                &node.attrs,
                input.shape,
                &pads,
                axes.as_deref(),
                constant_value,
            ) else {
                return Ok(None);
            };
            let Some(out_shape) = pad::pad_output_shape(input.shape, &params) else {
                return Ok(None);
            };
            let in_shape = input.shape.to_vec();
            match pad::cuda_pad_bound(ctx, input.binding, &in_shape, &params, placement)
                .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("Pad", &data, || {
                        reference::ref_pad(input.binding.host()?, &in_shape, &params)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data,
                        out_shape.to_vec(),
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // Reshape / Squeeze / Unsqueeze / Flatten: zero-cost residency        //
        // aliases -- see `reshape`'s module docs. No kernel, no oracle: only  //
        // claimed when the input is already device-resident, in which case   //
        // the "dispatch" is `CudaDeviceTensor::alias` (an `Arc::clone`) and   //
        // nothing else. A host-resident input declines outright -- the CPU's //
        // own reshape is already `O(1)`, so there is nothing to accelerate.  //
        // ------------------------------------------------------------------ //
        OpKind::Unsqueeze | OpKind::Squeeze | OpKind::Reshape | OpKind::Flatten => {
            let Some(name) = node.inputs.first().filter(|n| !n.is_empty()) else {
                return Ok(None);
            };
            let Some(tensor) = activations.resident(name) else {
                return Ok(None);
            };
            let in_shape = tensor.shape();

            let new_shape: Option<Vec<usize>> = match &node.op {
                OpKind::Reshape => node.inputs.get(1).and_then(|n| resolve(n)).and_then(|t| {
                    let shape_spec: Vec<i64> = t.data.iter().map(|&v| v as i64).collect();
                    let allowzero = node.attrs.i("allowzero", 0) != 0;
                    reshape::resolve_reshape_shape(in_shape, tensor.len(), &shape_spec, allowzero)
                }),
                OpKind::Squeeze => {
                    let axes: Vec<i64> = node
                        .inputs
                        .get(1)
                        .and_then(|n| resolve(n))
                        .map(|t| t.data.iter().map(|&v| v as i64).collect())
                        .unwrap_or_else(|| node.attrs.ints("axes").to_vec());
                    reshape::resolve_squeeze_shape(in_shape, &axes)
                }
                OpKind::Unsqueeze => {
                    let axes: Vec<i64> = node
                        .inputs
                        .get(1)
                        .and_then(|n| resolve(n))
                        .map(|t| t.data.iter().map(|&v| v as i64).collect())
                        .unwrap_or_else(|| node.attrs.ints("axes").to_vec());
                    reshape::resolve_unsqueeze_shape(in_shape, &axes)
                }
                OpKind::Flatten => {
                    reshape::resolve_flatten_shape(in_shape, node.attrs.i("axis", 1))
                }
                // Unreachable: this whole arm is already guarded by the same
                // four-way `OpKind` match above.
                _ => None,
            };
            let Some(new_shape) = new_shape else {
                return Ok(None);
            };
            let Some(aliased) = tensor.alias(new_shape) else {
                return Ok(None);
            };
            if placement == CudaOutputPlacement::Device {
                Ok(Some(CudaDispatchOutcome::Device(aliased)))
            } else {
                let host = aliased.read_back(ctx).map_err(OnnxError::from)?;
                Ok(Some(CudaDispatchOutcome::Host(vec![host])))
            }
        }

        // ------------------------------------------------------------------ //
        // Slice                                                                //
        // ------------------------------------------------------------------ //
        OpKind::Slice => {
            let Some(input) = operand(0) else {
                return Ok(None);
            };
            let (Some(starts_t), Some(ends_t)) = (
                node.inputs.get(1).and_then(|n| resolve(n)),
                node.inputs.get(2).and_then(|n| resolve(n)),
            ) else {
                return Ok(None);
            };
            let starts: Vec<i64> = starts_t.data.iter().map(|&v| v as i64).collect();
            let ends: Vec<i64> = ends_t.data.iter().map(|&v| v as i64).collect();
            let axes: Option<Vec<i64>> = node
                .inputs
                .get(3)
                .and_then(|n| resolve(n))
                .map(|t| t.data.iter().map(|&v| v as i64).collect());
            let steps: Option<Vec<i64>> = node
                .inputs
                .get(4)
                .and_then(|n| resolve(n))
                .map(|t| t.data.iter().map(|&v| v as i64).collect());
            let Some(params) = slice::slice_params_from_node(
                input.shape,
                &starts,
                &ends,
                axes.as_deref(),
                steps.as_deref(),
            ) else {
                return Ok(None);
            };
            let out_shape = params.out_shape.to_vec();
            let in_shape = input.shape.to_vec();
            match slice::cuda_slice_bound(ctx, input.binding, &in_shape, &params, placement)
                .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("Slice", &data, || {
                        reference::ref_slice(input.binding.host()?, &in_shape, &params)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, out_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        // ------------------------------------------------------------------ //
        // Concat -- variable arity, so this arm gathers every operand itself  //
        // rather than using the fixed `operand(0)`/`operand(1)` pattern above.//
        // ------------------------------------------------------------------ //
        OpKind::Concat => {
            if node.inputs.is_empty() {
                return Ok(None);
            }
            let mut operands: Vec<NodeOperand<'_>> = Vec::with_capacity(node.inputs.len());
            for slot in 0..node.inputs.len() {
                let Some(op) = operand(slot) else {
                    return Ok(None);
                };
                operands.push(op);
            }
            let shapes: Vec<&[usize]> = operands.iter().map(|o| o.shape).collect();
            let Some(params) = concat::concat_params_from_node(&node.attrs, &shapes) else {
                return Ok(None);
            };
            let bindings: Vec<InputBinding<'_>> = operands.iter().map(|o| o.binding).collect();
            let out_shape = params.out_shape.clone();
            match concat::cuda_concat_bound(ctx, &bindings, &params, placement)
                .map_err(OnnxError::from)?
            {
                Some(activation::KernelOutput::Device(tensor)) => {
                    Ok(Some(CudaDispatchOutcome::Device(tensor)))
                }
                Some(activation::KernelOutput::Host(data)) => {
                    if !verify_or_fallback("Concat", &data, || {
                        let host_slices: Option<Vec<&[f32]>> =
                            operands.iter().map(|o| o.binding.host()).collect();
                        reference::ref_concat(&host_slices?, &params)
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(CudaDispatchOutcome::Host(vec![Tensor::new(
                        data, out_shape,
                    )])))
                }
                None => Ok(None),
            }
        }

        _ => Ok(None),
    }
}

/// Unit tests for this module, in `dispatch_tests.rs` — see that file's header
/// for why they live beside `lib.rs` rather than inside it.
#[cfg(test)]
mod dispatch_tests;
