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

pub mod context;
pub mod conv;
pub mod elementwise;
pub mod error;
pub mod graph_cache;
pub mod matmul;
pub mod reduce;
pub mod reference;
pub mod residency;
pub mod softmax;

pub use context::CudaContext;
pub use error::CudaDispatchError as CudaError;

use std::collections::HashMap;

use oxionnx_core::graph::{Node, OpKind};
use oxionnx_core::{OnnxError, Tensor};

use context::FailurePolicy;

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
/// | `Add`, `Sub`, `Mul`, `Div`                | **yes**   | [`elementwise::cuda_binary_elementwise`]   |
/// | `ReduceSum`, `ReduceMax`                  | **yes**   | [`reduce::cuda_reduce`]                    |
/// | `Softmax`                                 | **yes**   | [`softmax::cuda_softmax`]                  |
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
/// # Necessary, not sufficient
///
/// - `is_supported_op(op) == false` is a **hard guarantee**: [`try_cuda_dispatch`]
///   returns `Ok(None)` for every such node.  Callers may skip CUDA entirely.
/// - `is_supported_op(op) == true` means a kernel exists *for that op kind*.
///   [`try_cuda_dispatch`] may still decline an individual node whose
///   *configuration* is out of range — e.g. `Softmax` with a row wider than 1024,
///   a reduction over a non-flat axis, a broadcasting `Add` where the two
///   operand shapes differ, or a `Conv` with asymmetric `pads` (see the [`conv`]
///   module docs' "What still declines").  Callers must still handle `Ok(None)`.
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
/// // No dispatch arm at all.
/// assert!(!oxionnx_cuda::is_supported_op(&OpKind::Reshape));
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
            // ── elementwise.rs: binary arm ───────────────────────────────────
            | OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            // ── reduce.rs: reduction arm ─────────────────────────────────────
            | OpKind::ReduceSum
            | OpKind::ReduceMax
            // ── softmax.rs ───────────────────────────────────────────────────
            | OpKind::Softmax
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

/// Attempt to dispatch a single ONNX node to the CUDA backend.
///
/// Returns `Ok(Some(results))` if the op was handled by CUDA,
/// `Ok(None)` if the op is unsupported or the configuration is not
/// acceleratable (caller should try GPU/CPU fallback), or
/// `Err(OnnxError::Internal(...))` on a hard CUDA failure.
///
/// [`is_supported_op`] is the cheap pre-filter for this function: when it
/// returns `false`, this function is guaranteed to return `Ok(None)`.
///
/// # Thread affinity
///
/// Unlike a raw driver context, callers do **not** need to dispatch from
/// the same OS thread that built `ctx`: this function re-activates `ctx`
/// on the calling thread itself (see the private `activate_context` helper
/// just above this function) before doing anything else, defensively, on
/// every call. See its doc comment for the concrete scenario (concurrent
/// model loading) that makes this necessary.
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
pub fn try_cuda_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &CudaContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    activate_context(ctx)?;

    let resolve = |name: &str| -> Option<&Tensor> {
        if name.is_empty() {
            None
        } else {
            intermediates.get(name).or_else(|| weights.get(name))
        }
    };

    match &node.op {
        // ------------------------------------------------------------------ //
        // MatMul / Gemm                                                        //
        // ------------------------------------------------------------------ //
        OpKind::MatMul | OpKind::Gemm => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
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

                let an = a.ndim();
                let bn = b.ndim();
                if an >= 2 && bn >= 2 {
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

                    // A malformed model (K mismatch) or a degenerate
                    // zero-sized operand is declined rather than risking a
                    // divide-by-zero in the bias epilogue or a nonsensical
                    // GPU launch; the CPU kernel raises the proper
                    // diagnostic for the former.
                    if k != k2 || m == 0 || k == 0 || n == 0 {
                        return Ok(None);
                    }

                    // Batch-broadcast the leading dims with the same numpy
                    // rule the CPU sibling (`oxionnx-ops::math::matmul::matmul`)
                    // uses for its output shape. Unlike that CPU path, which
                    // unconditionally modulo-indexes each operand
                    // (`b_idx % operand_batches`), we additionally require
                    // each operand's own batch count to be exactly `1`
                    // (broadcasts) or exactly `batch` (no broadcast on that
                    // operand) before dispatching to the GPU: unconditional
                    // modulo indexing silently computes the *wrong* slice
                    // whenever both operands broadcast on different
                    // sub-axes of a multi-dimensional batch — e.g. A's
                    // batch `[2, 1]` against B's `[1, 3]` needs the A-slice
                    // sequence `0,0,0,1,1,1`, but `i % 2` yields
                    // `0,1,0,1,0,1`. Declining that narrow case is a missed
                    // acceleration, not a wrong answer: the CPU path runs
                    // it either way.
                    let a_batch_dims = &a.shape[..an - 2];
                    let b_batch_dims = &b.shape[..bn - 2];
                    let Ok(out_batch) = Tensor::broadcast_shape(a_batch_dims, b_batch_dims) else {
                        return Ok(None);
                    };
                    let batch: usize = out_batch.iter().product::<usize>().max(1);
                    let a_batches: usize = a_batch_dims.iter().product::<usize>().max(1);
                    let b_batches: usize = b_batch_dims.iter().product::<usize>().max(1);
                    if !(a_batches == 1 || a_batches == batch)
                        || !(b_batches == 1 || b_batches == batch)
                    {
                        return Ok(None);
                    }

                    // Checked size math: these are all derived from
                    // model-supplied shape dims, so a corrupted/adversarial
                    // shape must overflow into a decline, never a panic or a
                    // silently-wrapped (wrong) buffer size.
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
                    // Bounds-check up front rather than trusting
                    // `Tensor::new`'s debug-only length invariant (a
                    // malformed model can violate it in a release build).
                    if a.data.len() < a_needed || b.data.len() < b_needed {
                        return Ok(None);
                    }

                    // ── Operand residency ─────────────────────────────────
                    //
                    // A `MatMul`/`Gemm` operand that resolved out of `weights`
                    // rather than `intermediates` is a graph initializer: the
                    // same bytes on frame 1 and frame 10 000. `initializer_id`
                    // says so, keyed additionally by the *form* the GEMM needs,
                    // because a `transA`/`transB` node consumes the transpose
                    // of those bytes rather than the bytes themselves and the
                    // two must never share a cache slot.
                    let a_id = initializer_id(
                        node.inputs.first().map(String::as_str),
                        weights,
                        intermediates,
                        a,
                        operand_form(trans_a),
                    );
                    let b_id = initializer_id(
                        node.inputs.get(1).map(String::as_str),
                        weights,
                        intermediates,
                        b,
                        operand_form(trans_b),
                    );

                    // Ask the device before building anything. A resident
                    // transposed weight means the `transpose_2d_batched` call
                    // below -- an O(k*n) host copy, ~12 MiB of memcpy for
                    // ArcFace's head -- does not happen at all this frame.
                    let a_resident =
                        a_id.and_then(|id| ctx.resident_operand(id, matmul::A_LABEL, a_needed));
                    let b_resident =
                        b_id.and_then(|id| ctx.resident_operand(id, matmul::B_LABEL, b_needed));

                    // Materialise only what still has to cross the bus, and
                    // only when it genuinely differs from the tensor's own
                    // bytes. Transposition is done per the operand's *own*
                    // batch count, not the (possibly larger) broadcast
                    // `batch`: `a.data`/`b.data` hold only
                    // `a_batches`/`b_batches` slices, and transposing past
                    // that would read out of bounds.
                    //
                    // The untransposed case borrows `a.data` in place. It used
                    // to `.clone()` it -- a full host copy of every operand on
                    // every dispatch, which for a 49 MiB weight is a memcpy the
                    // size of the upload it was feeding.
                    //
                    // `|| verify_on` is not an optimisation gap, it closes one:
                    // the oracle below has to see the same bytes the GPU
                    // multiplied, and for a *resident* transposed weight those
                    // bytes exist only on the device. Under `OXIONNX_CUDA_VERIFY=1`
                    // the transpose is therefore rebuilt anyway -- a diagnostic
                    // mode that already recomputes every op on the CPU is the
                    // right place to pay it, and skipping it would silently
                    // compare against the untransposed operand and report a
                    // mismatch on a correct dispatch.
                    let verify_on = reference::verify_enabled();
                    let a_transposed =
                        (trans_a && (a_resident.is_none() || verify_on)).then(|| {
                            transpose_2d_batched(
                                &a.data,
                                a_batches,
                                a.shape[an - 2],
                                a.shape[an - 1],
                            )
                        });
                    let b_transposed =
                        (trans_b && (b_resident.is_none() || verify_on)).then(|| {
                            transpose_2d_batched(
                                &b.data,
                                b_batches,
                                b.shape[bn - 2],
                                b.shape[bn - 1],
                            )
                        });

                    let a_operand = match a_resident {
                        Some(resident) => matmul::GemmOperand::from_resident(resident),
                        None => matmul::GemmOperand::from_host(
                            a_transposed.as_deref().unwrap_or(&a.data),
                            a_id,
                        ),
                    };
                    let b_operand = match b_resident {
                        Some(resident) => matmul::GemmOperand::from_resident(resident),
                        None => matmul::GemmOperand::from_host(
                            b_transposed.as_deref().unwrap_or(&b.data),
                            b_id,
                        ),
                    };

                    // One upload / launch / readback for the whole batch,
                    // replacing what used to be `batch` complete round trips.
                    // The broadcast rule the loop expressed as
                    // `(i % operand_batches) * slice` is carried through
                    // unchanged as a batch stride -- see `matmul::plan_gemm`.
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
                    let Some(mut out) =
                        matmul::cuda_gemm_batched(ctx, request).map_err(OnnxError::from)?
                    else {
                        return Ok(None);
                    };

                    // Shadow-verify the whole batch at once against the same
                    // per-slice oracle the loop used, slice by slice, with the
                    // same `(i % operand_batches)` indexing. Built only when
                    // verification is actually on: `verify_or_fallback` takes
                    // the oracle as a closure precisely so this costs nothing
                    // in production.
                    if !verify_or_fallback("MatMul/Gemm", &out, || {
                        let a_data = a_transposed.as_deref().unwrap_or(&a.data);
                        let b_data = b_transposed.as_deref().unwrap_or(&b.data);
                        let mut expected = Vec::with_capacity(out_total);
                        for i in 0..batch {
                            let a_start = (i % a_batches) * slice_a;
                            let b_start = (i % b_batches) * slice_b;
                            // `.get()` rather than direct indexing: never
                            // index a model-derived slice range without a
                            // bounds check, even though `a_needed`/`b_needed`
                            // above already guarantee this succeeds.
                            let (Some(a_slice), Some(b_slice)) = (
                                a_data.get(a_start..a_start + slice_a),
                                b_data.get(b_start..b_start + slice_b),
                            ) else {
                                // The oracle cannot be built, so there is
                                // nothing to compare against; `shadow_verify`
                                // treats `None` as "no formula" and says so
                                // loudly rather than passing silently.
                                return None;
                            };
                            expected.extend(reference::ref_matmul(a_slice, b_slice, m, k, n));
                        }
                        Some(expected)
                    })? {
                        return Ok(None);
                    }

                    // Apply alpha scaling — after verification, so the oracle
                    // stays the plain unscaled product. See the `matmul`
                    // module docs' "Alpha stays on the host".
                    if (alpha - 1.0).abs() > f32::EPSILON {
                        for v in &mut out {
                            *v *= alpha;
                        }
                    }

                    // Gemm: C = alpha * A @ B + beta * bias
                    if is_gemm && beta.abs() > f32::EPSILON {
                        if let Some(bias) = node.inputs.get(2).and_then(|n| resolve(n)) {
                            if !apply_gemm_bias(&mut out, &bias.data, &bias.shape, m, n, beta) {
                                // `bias`'s shape isn't unidirectionally
                                // broadcastable to [m, n] — a malformed
                                // model. Decline rather than silently
                                // dropping the bias; the CPU kernel raises
                                // a proper diagnostic.
                                return Ok(None);
                            }
                        }
                    }

                    let mut out_shape = out_batch;
                    out_shape.push(m);
                    out_shape.push(n);
                    return Ok(Some(vec![Tensor::new(out, out_shape)]));
                }
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Conv                                                                  //
        //                                                                      //
        // `conv::cuda_conv` computes a real answer for three validated shape   //
        // classes — 1x1, true-depthwise, and the general symmetric-padding     //
        // case, dispatched straight to `oxicuda-dnn`'s `Conv1x1` /             //
        // `DepthwiseConv` / `ImplicitGemmConv` engines respectively (see the   //
        // `conv` module docs' "Dispatch rule"; note it never goes through      //
        // `conv_forward`'s auto-selector) — and declines (`Ok(None)`)          //
        // everything else, so this arm can yield either `Ok(Some(_))` or       //
        // `Ok(None)` depending on the node's configuration. Its output is      //
        // shadow-verified against `reference::ref_conv` through the same       //
        // `verify_or_fallback` gate every other claimable op below uses, so    //
        // `OXIONNX_CUDA_VERIFY=1` covers a Conv dispatch exactly like it       //
        // covers MatMul/elementwise/reduce/Softmax. `is_supported_op` reports  //
        // `true` for `Conv`, so `decide_placement` routes production           //
        // convolutions here — this arm is on the hot path, not test-only.      //
        // ------------------------------------------------------------------ //
        OpKind::Conv => {
            let input = resolve(&node.inputs[0]);
            let weight = resolve(&node.inputs[1]);
            let bias = node.inputs.get(2).and_then(|n| resolve(n));
            if let (Some(input), Some(weight)) = (input, weight) {
                let attrs = &node.attrs;
                let strides_v = attrs.ints("strides");
                let strides = [
                    strides_v.first().copied().unwrap_or(1) as usize,
                    strides_v.get(1).copied().unwrap_or(1) as usize,
                ];
                let pads_v = attrs.ints("pads");
                let pads = [
                    pads_v.first().copied().unwrap_or(0) as usize,
                    pads_v.get(1).copied().unwrap_or(0) as usize,
                    pads_v.get(2).copied().unwrap_or(0) as usize,
                    pads_v.get(3).copied().unwrap_or(0) as usize,
                ];
                let dilations_v = attrs.ints("dilations");
                let dilations = [
                    dilations_v.first().copied().unwrap_or(1) as usize,
                    dilations_v.get(1).copied().unwrap_or(1) as usize,
                ];
                let group = attrs.i("group", 1) as usize;

                let conv_params = conv::ConvParams {
                    strides,
                    pads,
                    dilations,
                    group,
                };

                // A convolution's filter and bias are the megabyte-scale
                // invariant bytes in this workload -- InSwapper-128 alone
                // re-uploaded ~503 MB of them per forward pass before
                // residency existed. The input activation deliberately has no
                // identity: it is this frame's data.
                let conv_ids = conv::ConvWeightIds {
                    weight: initializer_id(
                        node.inputs.get(1).map(String::as_str),
                        weights,
                        intermediates,
                        weight,
                        residency::OperandForm::Raw,
                    ),
                    bias: bias.and_then(|bias_tensor| {
                        initializer_id(
                            node.inputs.get(2).map(String::as_str),
                            weights,
                            intermediates,
                            bias_tensor,
                            residency::OperandForm::Raw,
                        )
                    }),
                };

                match conv::cuda_conv_cached(ctx, input, weight, bias, &conv_params, conv_ids)
                    .map_err(OnnxError::from)?
                {
                    Some(tensor) => {
                        let bias_data = bias.map(|b| b.data.as_slice());
                        if !verify_or_fallback("Conv", &tensor.data, || {
                            Some(reference::ref_conv(
                                &input.data,
                                &weight.data,
                                bias_data,
                                &input.shape,
                                &weight.shape,
                                &conv_params,
                            ))
                        })? {
                            return Ok(None);
                        }
                        return Ok(Some(vec![tensor]));
                    }
                    None => return Ok(None),
                }
            }
            Ok(None)
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

            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let op_name = node.op.as_str();
                let out = elementwise::cuda_elementwise(ctx, &input.data, op_name)
                    .map_err(OnnxError::from)?;
                if !verify_or_fallback(op_name, &out, || {
                    reference::ref_unary_vec(&node.op, &input.data)
                })? {
                    return Ok(None);
                }
                return Ok(Some(vec![Tensor::new(out, input.shape.clone())]));
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Binary elementwise (Add, Sub, Mul, Div)                              //
        // ------------------------------------------------------------------ //
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            let a = resolve(&node.inputs[0]);
            let b = resolve(&node.inputs[1]);
            if let (Some(a), Some(b)) = (a, b) {
                // Only dispatch when shapes match exactly (no broadcasting).
                if a.shape == b.shape {
                    let op_name = node.op.as_str();
                    let out = elementwise::cuda_binary_elementwise(ctx, &a.data, &b.data, op_name)
                        .map_err(OnnxError::from)?;
                    if !verify_or_fallback(op_name, &out, || {
                        reference::ref_binary_vec(&node.op, &a.data, &b.data)
                    })? {
                        return Ok(None);
                    }
                    return Ok(Some(vec![Tensor::new(out, a.shape.clone())]));
                }
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Reductions                                                           //
        // ------------------------------------------------------------------ //
        OpKind::ReduceSum | OpKind::ReduceMax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
                let axes = node.attrs.ints("axes");
                if axes.len() == 1 {
                    let rank = input.shape.len();
                    let raw_axis = axes[0];
                    // ONNX permits a negative axis, resolved against rank.
                    let resolved_axis = if raw_axis < 0 {
                        raw_axis + rank as i64
                    } else {
                        raw_axis
                    };
                    if resolved_axis < 0 || resolved_axis as usize >= rank {
                        // Out of range for `input`'s rank — a malformed
                        // model; decline and let the CPU kernel raise the
                        // proper diagnostic.
                        return Ok(None);
                    }
                    let axis = resolved_axis as usize;
                    // ONNX ReduceSum/ReduceMax default `keepdims` to 1 (the
                    // reduced axis stays, size 1); `keepdims=0` removes it
                    // from the output shape entirely. `cuda_reduce`'s data
                    // layout (`[outer, inner]`) is identical either way —
                    // only the declared shape differs.
                    let keepdims = node.attrs.i("keepdims", 1) != 0;
                    let op_name = node.op.as_str();
                    match reduce::cuda_reduce(ctx, &input.data, &input.shape, axis, op_name)
                        .map_err(OnnxError::from)?
                    {
                        Some(out) => {
                            if !verify_or_fallback(op_name, &out, || {
                                reference::ref_reduce(&node.op, &input.data, &input.shape, axis)
                            })? {
                                return Ok(None);
                            }
                            let mut out_shape = input.shape.clone();
                            if keepdims {
                                out_shape[axis] = 1;
                            } else {
                                out_shape.remove(axis);
                            }
                            return Ok(Some(vec![Tensor::new(out, out_shape)]));
                        }
                        None => return Ok(None),
                    }
                }
            }
            Ok(None)
        }

        // ------------------------------------------------------------------ //
        // Softmax                                                              //
        // ------------------------------------------------------------------ //
        OpKind::Softmax => {
            let input = resolve(&node.inputs[0]);
            if let Some(input) = input {
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
                // Out of `[-rank, rank)` is a malformed model — decline and
                // let the CPU kernel raise the proper diagnostic. Anything
                // other than the last dimension is not malformed, but
                // `cuda_softmax`'s row/batch decomposition cannot express
                // it (it always normalizes `shape[rank-1]`), so it also
                // declines, cleanly, to the axis-aware CPU kernel.
                if resolved_axis < 0
                    || resolved_axis as usize >= rank
                    || resolved_axis as usize != rank - 1
                {
                    return Ok(None);
                }
                match softmax::cuda_softmax(ctx, &input.data, &input.shape)
                    .map_err(OnnxError::from)?
                {
                    Some(out) => {
                        if !verify_or_fallback("Softmax", &out, || {
                            reference::ref_softmax(&input.data, &input.shape)
                        })? {
                            return Ok(None);
                        }
                        return Ok(Some(vec![Tensor::new(out, input.shape.clone())]));
                    }
                    None => return Ok(None),
                }
            }
            Ok(None)
        }

        _ => Ok(None),
    }
}

/// The residency identity for `name`, or `None` when these bytes must not be
/// cached.
///
/// This is the one place that decides whether an operand is a *graph
/// initializer* — bytes that are invariant for the session — as opposed to an
/// activation this run just produced. Getting it wrong in the permissive
/// direction would be a correctness bug of the worst kind: caching an
/// activation means every later frame is computed against the first frame's
/// numbers, silently.
///
/// Two conditions, both necessary:
///
/// * **Not an intermediate.** `resolve` prefers `intermediates` over
///   `weights`, so a name a node has already produced this run is *not* an
///   initializer here whatever the weight map also holds under it. Keying such
///   a name would cache one tensor's bytes and then serve them for another's.
/// * **Present in `weights`.** That is what "initializer" means at this layer.
///
/// Mirrors `oxionnx::session::gpu_dispatch`'s `initializer_key`, which makes
/// the identical decision for the wgpu backend, for the identical reasons.
fn initializer_id<'a>(
    name: Option<&'a str>,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    tensor: &Tensor,
    form: residency::OperandForm,
) -> Option<residency::WeightId<'a>> {
    let name = name?;
    if name.is_empty() || intermediates.contains_key(name) {
        return None;
    }
    weights
        .contains_key(name)
        .then(|| residency::WeightId::new(name, &tensor.data, form))
}

/// Which derived form of an operand's bytes a GEMM will read.
///
/// A `transA`/`transB` node consumes the *transpose* of its operand, which is
/// a different byte sequence from the operand itself — and one graph can
/// legitimately consume the same initializer both ways from two different
/// nodes. Keeping them in separate cache slots is what stops one node being
/// served the other's bytes.
fn operand_form(transposed: bool) -> residency::OperandForm {
    if transposed {
        residency::OperandForm::Transposed
    } else {
        residency::OperandForm::Raw
    }
}

/// Read the live `OXIONNX_CUDA_VERIFY` / `OXIONNX_CUDA_STRICT` state and
/// delegate to [`reference::shadow_verify`], converting a strict-mode
/// mismatch into an [`OnnxError`].
///
/// Every `try_cuda_dispatch` call site that has just gotten a result back
/// from a GPU kernel routes through this before trusting it — see the
/// [`mod@reference`] module docs and finding a8-5 for why: a wrong GPU kernel
/// returns `Ok(Some(wrong_data))`, a successful-looking answer, and nothing
/// else in this crate's error handling can tell the difference.
///
/// # Returns
/// * `Ok(true)` — the caller should proceed with `gpu`'s numbers (this is
///   the fast path taken unconditionally whenever `OXIONNX_CUDA_VERIFY` is
///   unset, which is every production dispatch by default).
/// * `Ok(false)` — the caller must discard `gpu` and `return Ok(None)`
///   instead, so the real CPU operator recomputes the node. Already logged
///   at `error!`.
///
/// # Errors
/// [`OnnxError::Internal`] (wrapping [`error::CudaDispatchError::Verify`])
/// only when `OXIONNX_CUDA_STRICT=1` and the comparison disagreed.
fn verify_or_fallback(
    op: &str,
    gpu: &[f32],
    oracle: impl FnOnce() -> Option<Vec<f32>>,
) -> Result<bool, OnnxError> {
    reference::shadow_verify(
        op,
        gpu,
        reference::verify_enabled(),
        FailurePolicy::current(),
        oracle,
    )
    .map_err(OnnxError::from)
}

/// Transpose the last two dims of batched 2-D data in-place.
///
/// Input layout: `batch` blocks of `rows * cols` elements (row-major).
/// Output layout: `batch` blocks of `cols * rows` elements (row-major).
fn transpose_2d_batched(data: &[f32], batch: usize, rows: usize, cols: usize) -> Vec<f32> {
    let slice = rows * cols;
    let mut out = vec![0.0_f32; data.len()];
    for b in 0..batch {
        let base_in = b * slice;
        let base_out = b * slice;
        for r in 0..rows {
            for c in 0..cols {
                out[base_out + c * rows + r] = data[base_in + r * cols + c];
            }
        }
    }
    out
}

/// Apply Gemm bias: `out += beta * bias`, unidirectionally broadcasting
/// `bias` (of `bias_shape`, rank 0/1/2 per the ONNX Gemm spec for `C`)
/// across the `[rows, n]` output, where `rows` is a multiple of `m` (`out`
/// may stack several batch slices, though a spec-conformant `Gemm` node
/// always has `rows == m`).
///
/// Returns `false` — leaving `out` unmodified — when `bias_shape` is not
/// unidirectionally broadcastable to `[m, n]`, or `bias`'s data is shorter
/// than `bias_shape` declares (a malformed model in either case); the
/// caller declines the whole node rather than silently omitting the bias,
/// so the CPU kernel produces the correct result or a proper diagnostic.
fn apply_gemm_bias(
    out: &mut [f32],
    bias: &[f32],
    bias_shape: &[usize],
    m: usize,
    n: usize,
    beta: f32,
) -> bool {
    if m == 0 || n == 0 {
        return false;
    }

    // Right-align `bias_shape` against `[m, n]` (numpy/ONNX unidirectional
    // broadcast): a trailing dim of `1` broadcasts, a trailing dim equal to
    // the target matches, anything else is an illegal (malformed-model) `C`
    // shape. Any dim further left must also be `1` — Gemm's `C` is spec'd
    // as at most 2-D, but this stays defensive rather than assuming it.
    let rank = bias_shape.len();
    let bias_n = if rank >= 1 { bias_shape[rank - 1] } else { 1 };
    let bias_m = if rank >= 2 { bias_shape[rank - 2] } else { 1 };
    let leading_ok = bias_shape[..rank.saturating_sub(2)].iter().all(|&d| d == 1);
    if !leading_ok || !(bias_m == 1 || bias_m == m) || !(bias_n == 1 || bias_n == n) {
        return false;
    }
    let Some(bias_needed) = bias_m.checked_mul(bias_n) else {
        return false;
    };
    if bias.len() < bias_needed {
        return false;
    }

    let total_rows = out.len() / n;
    for row in 0..total_rows {
        let bias_row = if bias_m == 1 { 0 } else { row % m };
        let base = row * n;
        let bias_base = bias_row * bias_n;
        for col in 0..n {
            let bias_col = if bias_n == 1 { 0 } else { col };
            out[base + col] += beta * bias[bias_base + bias_col];
        }
    }
    true
}

/// Unit tests for this module, in `dispatch_tests.rs` — see that file's header
/// for why they live beside `lib.rs` rather than inside it.
#[cfg(test)]
mod dispatch_tests;
