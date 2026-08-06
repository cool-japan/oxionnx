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
pub mod matmul;
pub mod reduce;
pub mod reference;
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
/// | `Conv`                                    | **no**    | [`conv::cuda_conv`] — always `Ok(None)`    |
/// | 16 unary activations (`Relu` … `LeakyRelu`) | **yes** | [`elementwise::cuda_elementwise`]          |
/// | `Add`, `Sub`, `Mul`, `Div`                | **yes**   | [`elementwise::cuda_binary_elementwise`]   |
/// | `ReduceSum`, `ReduceMax`                  | **yes**   | [`reduce::cuda_reduce`]                    |
/// | `Softmax`                                 | **yes**   | [`softmax::cuda_softmax`]                  |
///
/// `Conv` is deliberately excluded: [`try_cuda_dispatch`] does have a `Conv`
/// arm, but it delegates to [`conv::cuda_conv`], which unconditionally returns
/// `Ok(None)` because no CUDA convolution kernel exists (see the [`conv`] module
/// docs).  Reporting `Conv` as supported would send every convolution on a
/// pointless GPU round-trip that always falls back to the CPU anyway.
///
/// # Necessary, not sufficient
///
/// - `is_supported_op(op) == false` is a **hard guarantee**: [`try_cuda_dispatch`]
///   returns `Ok(None)` for every such node.  Callers may skip CUDA entirely.
/// - `is_supported_op(op) == true` means a kernel exists *for that op kind*.
///   [`try_cuda_dispatch`] may still decline an individual node whose
///   *configuration* is out of range — e.g. `Softmax` with a row wider than 1024,
///   a reduction over a non-flat axis, or a broadcasting `Add` where the two
///   operand shapes differ.  Callers must still handle `Ok(None)`.
///
/// # Example
///
/// ```
/// use oxionnx_core::graph::OpKind;
///
/// assert!(oxionnx_cuda::is_supported_op(&OpKind::MatMul));
/// assert!(oxionnx_cuda::is_supported_op(&OpKind::Relu));
/// // No CUDA convolution kernel exists — Conv always runs on the CPU.
/// assert!(!oxionnx_cuda::is_supported_op(&OpKind::Conv));
/// // No dispatch arm at all.
/// assert!(!oxionnx_cuda::is_supported_op(&OpKind::Reshape));
/// ```
pub fn is_supported_op(op: &OpKind) -> bool {
    matches!(
        op,
        // ── matmul.rs: `OpKind::MatMul | OpKind::Gemm` arm ───────────────────
        OpKind::MatMul
            | OpKind::Gemm
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
    // NOTE: `OpKind::Conv` is intentionally absent — see the doc comment above.
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
pub fn try_cuda_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &CudaContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
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

                    // Prepare (possibly transposed) data — transposed per
                    // the operand's *own* batch count, not the (possibly
                    // larger) broadcast `batch`: `a.data`/`b.data` hold only
                    // `a_batches`/`b_batches` slices, and transposing past
                    // that would read out of bounds.
                    let a_data = if trans_a {
                        transpose_2d_batched(&a.data, a_batches, a.shape[an - 2], a.shape[an - 1])
                    } else {
                        a.data.clone()
                    };
                    let b_data = if trans_b {
                        transpose_2d_batched(&b.data, b_batches, b.shape[bn - 2], b.shape[bn - 1])
                    } else {
                        b.data.clone()
                    };

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
                    if a_data.len() < a_needed || b_data.len() < b_needed {
                        return Ok(None);
                    }

                    let mut out = Vec::with_capacity(out_total);
                    for i in 0..batch {
                        let a_start = (i % a_batches) * slice_a;
                        let b_start = (i % b_batches) * slice_b;
                        // `.get()` rather than direct indexing: never index
                        // a model-derived slice range without a bounds
                        // check, even though `a_needed`/`b_needed` above
                        // already guarantee this succeeds.
                        let (Some(a_slice), Some(b_slice)) = (
                            a_data.get(a_start..a_start + slice_a),
                            b_data.get(b_start..b_start + slice_b),
                        ) else {
                            return Ok(None);
                        };
                        let mut c = matmul::cuda_matmul(ctx, a_slice, b_slice, m, k, n)
                            .map_err(OnnxError::from)?;
                        if !verify_or_fallback("MatMul/Gemm", &c, || {
                            Some(reference::ref_matmul(a_slice, b_slice, m, k, n))
                        })? {
                            return Ok(None);
                        }

                        // Apply alpha scaling.
                        if (alpha - 1.0).abs() > f32::EPSILON {
                            for v in &mut c {
                                *v *= alpha;
                            }
                        }
                        out.append(&mut c);
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
        // Conv — ALWAYS DECLINES.                                              //
        //                                                                      //
        // `conv::cuda_conv` unconditionally returns `Ok(None)` (there is no    //
        // CUDA convolution kernel), so this arm can only ever yield `Ok(None)`.//
        // It is retained so the ONNX attrs → `ConvParams` mapping stays live   //
        // for the eventual real kernel.  `is_supported_op` reports `false` for //
        // `Conv` so placement never routes a convolution here in the first     //
        // place.                                                               //
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

                match conv::cuda_conv(ctx, input, weight, bias, &conv_params)
                    .map_err(OnnxError::from)?
                {
                    Some(tensor) => return Ok(Some(vec![tensor])),
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            op,
            name: "test_node".to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: Attributes::default(),
        }
    }

    // ── is_supported_op ⇄ try_cuda_dispatch agreement ───────────────────────

    /// The ops `try_cuda_dispatch` can actually claim, transcribed from its
    /// `match &node.op` arms *with the permanently-declining `Conv` arm removed*.
    ///
    /// Read the dispatch match top-to-bottom and this list must fall out of it.
    fn claimable_ops() -> Vec<OpKind> {
        vec![
            // `OpKind::MatMul | OpKind::Gemm` arm → matmul::cuda_matmul
            OpKind::MatMul,
            OpKind::Gemm,
            // unary activation arm → elementwise::cuda_elementwise
            OpKind::Relu,
            OpKind::Sigmoid,
            OpKind::Gelu,
            OpKind::Tanh,
            OpKind::Exp,
            OpKind::Sqrt,
            OpKind::Abs,
            OpKind::Neg,
            OpKind::Log,
            OpKind::Ceil,
            OpKind::Floor,
            OpKind::HardSigmoid,
            OpKind::HardSwish,
            OpKind::SiLU,
            OpKind::Softplus,
            OpKind::LeakyRelu,
            // binary arm → elementwise::cuda_binary_elementwise
            OpKind::Add,
            OpKind::Sub,
            OpKind::Mul,
            OpKind::Div,
            // reduction arm → reduce::cuda_reduce
            OpKind::ReduceSum,
            OpKind::ReduceMax,
            // softmax arm → softmax::cuda_softmax
            OpKind::Softmax,
            // NOTE: `OpKind::Conv` has a dispatch arm but conv::cuda_conv always
            // returns Ok(None), so it is NOT claimable and must not appear here.
        ]
    }

    /// Every unit variant of `OpKind` (i.e. excluding `OpKind::Unknown(_)`).
    ///
    /// Enumerated exhaustively so that `is_supported_op` can be pinned to
    /// *exactly* the claimable set — an op accidentally added to the predicate
    /// without a matching dispatch arm makes this test fail.
    fn all_op_kinds() -> Vec<OpKind> {
        vec![
            OpKind::MatMul,
            OpKind::Gemm,
            OpKind::Add,
            OpKind::Sub,
            OpKind::Mul,
            OpKind::Div,
            OpKind::Pow,
            OpKind::Sqrt,
            OpKind::Reciprocal,
            OpKind::Neg,
            OpKind::ReduceMean,
            OpKind::ReduceSum,
            OpKind::ReduceMax,
            OpKind::ReduceMin,
            OpKind::ReduceProd,
            OpKind::ArgMax,
            OpKind::ArgMin,
            OpKind::CumSum,
            OpKind::Range,
            OpKind::TopK,
            OpKind::Softmax,
            OpKind::LayerNorm,
            OpKind::GroupNorm,
            OpKind::BatchNorm,
            OpKind::Gelu,
            OpKind::Relu,
            OpKind::Sigmoid,
            OpKind::Tanh,
            OpKind::Erf,
            OpKind::SiLU,
            OpKind::HardSigmoid,
            OpKind::HardSwish,
            OpKind::RMSNorm,
            OpKind::Reshape,
            OpKind::Transpose,
            OpKind::Squeeze,
            OpKind::Unsqueeze,
            OpKind::Flatten,
            OpKind::Concat,
            OpKind::Slice,
            OpKind::Expand,
            OpKind::Split,
            OpKind::Tile,
            OpKind::Gather,
            OpKind::GatherElements,
            OpKind::Where,
            OpKind::ScatterElements,
            OpKind::ScatterND,
            OpKind::Conv,
            OpKind::MaxPool,
            OpKind::AveragePool,
            OpKind::Pad,
            OpKind::LeakyRelu,
            OpKind::PRelu,
            OpKind::Resize,
            OpKind::GlobalAveragePool,
            OpKind::GlobalMaxPool,
            OpKind::QuantizeLinear,
            OpKind::DequantizeLinear,
            OpKind::Identity,
            OpKind::Cast,
            OpKind::Shape,
            OpKind::Constant,
            OpKind::Clip,
            OpKind::Abs,
            OpKind::Log,
            OpKind::Exp,
            OpKind::Ceil,
            OpKind::Floor,
            OpKind::Round,
            OpKind::Sign,
            OpKind::Mod,
            OpKind::BitShift,
            OpKind::Sin,
            OpKind::Cos,
            OpKind::Tan,
            OpKind::Asin,
            OpKind::Acos,
            OpKind::Atan,
            OpKind::Sinh,
            OpKind::Cosh,
            OpKind::Asinh,
            OpKind::Acosh,
            OpKind::Atanh,
            OpKind::VariadicMin,
            OpKind::VariadicMax,
            OpKind::VariadicMean,
            OpKind::VariadicSum,
            OpKind::Equal,
            OpKind::Greater,
            OpKind::GreaterOrEqual,
            OpKind::Less,
            OpKind::LessOrEqual,
            OpKind::And,
            OpKind::Or,
            OpKind::Xor,
            OpKind::Not,
            OpKind::IsInf,
            OpKind::IsNaN,
            OpKind::NonZero,
            OpKind::ConstantOfShape,
            OpKind::EyeLike,
            OpKind::Trilu,
            OpKind::LogSoftmax,
            OpKind::Softplus,
            OpKind::Softsign,
            OpKind::Mish,
            OpKind::Celu,
            OpKind::Elu,
            OpKind::Selu,
            OpKind::ThresholdedRelu,
            OpKind::InstanceNorm,
            OpKind::LpNorm,
            OpKind::MeanVarianceNormalization,
            OpKind::Dropout,
            OpKind::DepthToSpace,
            OpKind::SpaceToDepth,
            OpKind::ReverseSequence,
            OpKind::GatherND,
            OpKind::OneHot,
            OpKind::Compress,
            OpKind::Unique,
            OpKind::Einsum,
            OpKind::ConvTranspose,
            OpKind::NonMaxSuppression,
            OpKind::LSTM,
            OpKind::GRU,
            OpKind::Attention,
            OpKind::MultiHeadAttention,
            OpKind::RotaryEmbedding,
            OpKind::GridSample,
            OpKind::RoiAlign,
            OpKind::If,
            OpKind::Loop,
            OpKind::Scan,
            OpKind::LinearClassifier,
            OpKind::LinearRegressor,
            OpKind::Normalizer,
            OpKind::Scaler,
            OpKind::LabelEncoder,
            OpKind::TreeEnsembleClassifier,
            OpKind::TreeEnsembleRegressor,
            OpKind::SVMClassifier,
            OpKind::SVMRegressor,
            OpKind::TfIdfVectorizer,
            OpKind::StringNormalizer,
            OpKind::DFT,
            OpKind::STFT,
            OpKind::BlackmanWindow,
            OpKind::HannWindow,
            OpKind::HammingWindow,
            OpKind::MelWeightMatrix,
            OpKind::Bernoulli,
            OpKind::ReduceL1,
            OpKind::ReduceL2,
            OpKind::ReduceLogSum,
            OpKind::ReduceLogSumExp,
            OpKind::ReduceSumSquare,
            OpKind::BitwiseAnd,
            OpKind::BitwiseOr,
            OpKind::BitwiseXor,
            OpKind::BitwiseNot,
            OpKind::Size,
            OpKind::Hardmax,
            OpKind::Shrink,
            OpKind::ConvAddRelu,
        ]
    }

    /// `is_supported_op` must return `true` for **exactly** the ops that
    /// `try_cuda_dispatch` can claim — no more, no fewer.
    ///
    /// This is the contract `oxionnx::execution_providers::decide_placement`
    /// relies on to avoid an upload → dispatch → fence → readback round-trip
    /// for an op CUDA was never going to handle.
    #[test]
    fn is_supported_op_matches_dispatch_arms() {
        let claimable = claimable_ops();

        // 1. Every claimable op is reported supported.
        for op in &claimable {
            assert!(
                is_supported_op(op),
                "{op:?} has a live try_cuda_dispatch arm but is_supported_op says false",
            );
        }

        // 2. Nothing outside the claimable set is reported supported.
        //    Sweeping every OpKind unit variant makes this an "exactly" check.
        let all = all_op_kinds();
        for op in &all {
            let expected = claimable.contains(op);
            assert_eq!(
                is_supported_op(op),
                expected,
                "is_supported_op({op:?}) disagrees with the try_cuda_dispatch match arms",
            );
        }

        // 3. Guard the enumeration itself: if `OpKind` grows a variant and
        //    `all_op_kinds` is not updated, the arity check below trips.
        assert_eq!(
            all.len(),
            166,
            "OpKind gained/lost a unit variant — update all_op_kinds() and re-audit \
             is_supported_op against the try_cuda_dispatch match arms",
        );
        assert_eq!(
            claimable.len(),
            25,
            "claimable_ops() changed — re-audit against the try_cuda_dispatch match arms",
        );
    }

    /// Every op `try_cuda_dispatch` can claim through the unary/binary/reduce arms must
    /// have a live [`reference`] oracle formula.
    ///
    /// Without this, an op added to a dispatch arm with no matching `reference::ref_*`
    /// case doesn't fail loudly: `verify_or_fallback`'s oracle closure returns `None`,
    /// [`reference::shadow_verify`] treats that as "the oracle has no formula, skip the
    /// check" and logs a `warn!` — which only a human staring at a real CUDA machine's
    /// logs under `OXIONNX_CUDA_VERIFY=1` would ever see. This test makes that gap fail
    /// on every host, including this one with no GPU, by driving the same
    /// `claimable_ops()` list `is_supported_op_matches_dispatch_arms` already pins so the
    /// two enumerations cannot silently drift apart.
    ///
    /// `MatMul`/`Gemm` (verified via [`reference::ref_matmul`], which takes no `OpKind` —
    /// it is unconditionally applicable) and `Softmax` (verified via
    /// [`reference::ref_softmax`], likewise `OpKind`-free) are excluded from the
    /// per-op-formula check for that reason, and asserted present in `claimable_ops()`
    /// instead so removing one of them from the enum list still trips the arity check.
    #[test]
    fn oracle_covers_every_op_the_unary_binary_and_reduce_dispatch_arms_claim() {
        const UNARY_OPS: &[OpKind] = &[
            OpKind::Relu,
            OpKind::Sigmoid,
            OpKind::Gelu,
            OpKind::Tanh,
            OpKind::Exp,
            OpKind::Sqrt,
            OpKind::Abs,
            OpKind::Neg,
            OpKind::Log,
            OpKind::Ceil,
            OpKind::Floor,
            OpKind::HardSigmoid,
            OpKind::HardSwish,
            OpKind::SiLU,
            OpKind::Softplus,
            OpKind::LeakyRelu,
        ];
        const BINARY_OPS: &[OpKind] = &[OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div];
        const REDUCE_OPS: &[OpKind] = &[OpKind::ReduceSum, OpKind::ReduceMax];
        const NO_OPKIND_NEEDED: &[OpKind] = &[OpKind::MatMul, OpKind::Gemm, OpKind::Softmax];

        let claimable = claimable_ops();
        for op in &claimable {
            if UNARY_OPS.contains(op) {
                assert!(
                    reference::ref_unary(op, 0.5).is_some(),
                    "{op:?} is claimable by the unary elementwise dispatch arm but \
                     reference::ref_unary has no formula for it",
                );
            } else if BINARY_OPS.contains(op) {
                assert!(
                    reference::ref_binary(op, 1.0, 2.0).is_some(),
                    "{op:?} is claimable by the binary elementwise dispatch arm but \
                     reference::ref_binary has no formula for it",
                );
            } else if REDUCE_OPS.contains(op) {
                assert!(
                    reference::ref_reduce(op, &[1.0, 2.0, 3.0, 4.0], &[4], 0).is_some(),
                    "{op:?} is claimable by the reduce dispatch arm but reference::ref_reduce \
                     has no formula for it",
                );
            } else {
                assert!(
                    NO_OPKIND_NEEDED.contains(op),
                    "{op:?} is claimable but not classified into any op-family list in this \
                     test — add it to one of the lists above so oracle coverage stays pinned",
                );
            }
        }

        // The four lists above partition `claimable_ops()` exactly; this catches an op
        // quietly removed from one list (rather than from `claimable_ops()` itself, which
        // `is_supported_op_matches_dispatch_arms` already pins at 25).
        assert_eq!(
            UNARY_OPS.len() + BINARY_OPS.len() + REDUCE_OPS.len() + NO_OPKIND_NEEDED.len(),
            claimable.len(),
            "the op-family lists in this test no longer partition claimable_ops() exactly",
        );
    }

    /// `Conv` has a dispatch arm, but `conv::cuda_conv` always returns `Ok(None)`.
    /// Advertising it would route every convolution to the GPU only to fall back.
    #[test]
    fn conv_has_an_arm_but_is_not_claimable() {
        assert!(
            !is_supported_op(&OpKind::Conv),
            "Conv must not be advertised: conv::cuda_conv unconditionally declines",
        );
    }

    /// `Unknown` ops can never be claimed.
    #[test]
    fn unknown_op_is_not_supported() {
        assert!(!is_supported_op(&OpKind::Unknown("Frobnicate".to_string())));
    }

    /// The predicate must be pure and side-effect free — callable without a device.
    #[test]
    fn is_supported_op_needs_no_cuda_device() {
        // No CudaContext is constructed anywhere in this test.
        for op in all_op_kinds() {
            let _ = is_supported_op(&op);
        }
    }

    /// Validates that try_cuda_dispatch returns Ok(None) for unsupported ops
    /// when no CUDA context is available (unit test only touches the match arm).
    #[test]
    fn dispatch_unknown_op_returns_none() {
        // Without a real CUDA device we can only test the None-returning path.
        // We verify the dispatch fn returns None for an op that has no CUDA kernel.
        let node = make_node(OpKind::Identity, &["x"], &["y"]);
        let weights: HashMap<String, Tensor> = HashMap::new();
        let mut intermediates: HashMap<String, Tensor> = HashMap::new();
        let t = Tensor::new(vec![1.0f32], vec![1]);
        intermediates.insert("x".to_string(), t);

        // We cannot construct a real CudaContext in CI, so we skip the actual
        // dispatch and just verify the type signature compiles.
        let _ = &node;
        let _ = &weights;
        let _ = &intermediates;
    }

    #[test]
    fn cuda_context_try_new_no_panic() {
        // try_new must never panic — it should return None if no GPU present.
        let _ctx = CudaContext::try_new();
    }

    #[test]
    fn cuda_error_displays_correctly() {
        let e = CudaError::Ptx("bad ptx".to_string());
        let s = format!("{e}");
        assert!(
            s.contains("bad ptx"),
            "Expected error message to contain 'bad ptx', got: {s}"
        );
    }

    #[test]
    fn cuda_error_maps_to_onnx_internal() {
        let e = CudaError::Shape {
            op: "Conv",
            msg: "wrong shape".to_string(),
        };
        let onnx_err: OnnxError = e.into();
        match onnx_err {
            OnnxError::Internal(msg) => {
                assert!(
                    msg.contains("wrong shape"),
                    "Expected 'wrong shape' in: {msg}"
                );
            }
            other => panic!("Expected OnnxError::Internal, got: {other:?}"),
        }
    }

    // ── apply_gemm_bias (finding a8-4): every spec-legal broadcastable `C` ──
    //
    // ONNX Gemm's `C` may be unidirectionally broadcastable to `[M, N]` as a true
    // scalar (`[]` or `[1]`), `[N]`, `[M, 1]`, or `[M, N]`. The pre-fix code only
    // handled `[N]` and `[M, N]`; `[M, 1]` (M != N) and a genuine scalar (N != 1)
    // silently added nothing. Each case below is hand-verified.

    #[test]
    fn gemm_bias_row_broadcast_n_shape() {
        // bias = [N] = [10, 20, 30], M=2, N=3, beta=1.0 — broadcasts across every row.
        let mut out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(apply_gemm_bias(
            &mut out,
            &[10.0, 20.0, 30.0],
            &[3],
            2,
            3,
            1.0
        ));
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn gemm_bias_one_by_n_shape_matches_plain_n_shape() {
        // bias = [1, N] = [[10, 20, 30]] — same broadcast as the plain [N] case above,
        // exercised through the rank-2-with-leading-1 code path instead of rank-1.
        let mut out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(apply_gemm_bias(
            &mut out,
            &[10.0, 20.0, 30.0],
            &[1, 3],
            2,
            3,
            1.0
        ));
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn gemm_bias_full_m_by_n_matrix() {
        // bias = [M, N] = [[100,200,300],[400,500,600]], M=2, N=3, beta=1.0.
        let mut out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
        assert!(apply_gemm_bias(&mut out, &bias, &[2, 3], 2, 3, 1.0));
        assert_eq!(out, vec![101.0, 202.0, 303.0, 404.0, 505.0, 606.0]);
    }

    #[test]
    fn gemm_bias_m_by_one_column_broadcast_with_m_ne_n() {
        // The exact a8-4 regression case: bias = [M, 1] = [[7],[70]], M=2, N=3 (M != N),
        // beta=1.0. Before the fix, neither `bias.len() == n` (2 != 3) nor
        // `bias.len() == m*n` (2 != 6) matched, so the bias was silently dropped.
        let mut out = vec![0.0; 6];
        assert!(apply_gemm_bias(&mut out, &[7.0, 70.0], &[2, 1], 2, 3, 1.0));
        assert_eq!(out, vec![7.0, 7.0, 7.0, 70.0, 70.0, 70.0]);
    }

    #[test]
    fn gemm_bias_true_scalar_broadcasts_to_every_element_with_n_ne_one() {
        // The other a8-4 regression case: bias = [1] (a true scalar), M=2, N=3 (N != 1),
        // beta=2.0. Before the fix, `bias.len() == n` (1 != 3) and `bias.len() == m*n`
        // (1 != 6) both failed, so the bias was silently dropped.
        let mut out = vec![0.0; 6];
        assert!(apply_gemm_bias(&mut out, &[5.0], &[1], 2, 3, 2.0));
        assert_eq!(out, vec![10.0; 6]);
    }

    #[test]
    fn gemm_bias_rank_zero_scalar_broadcasts_too() {
        // A genuine ONNX scalar tensor has shape `[]` (rank 0), not `[1]`.
        let mut out = vec![0.0; 4];
        assert!(apply_gemm_bias(&mut out, &[9.0], &[], 2, 2, 1.0));
        assert_eq!(out, vec![9.0; 4]);
    }

    #[test]
    fn gemm_bias_declines_an_incompatible_shape_leaving_out_untouched() {
        // bias = [5] against N=3: neither equal nor 1 — not unidirectionally broadcastable.
        let mut out = vec![1.0, 2.0, 3.0];
        let untouched = out.clone();
        assert!(!apply_gemm_bias(
            &mut out,
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[5],
            1,
            3,
            1.0
        ));
        assert_eq!(
            out, untouched,
            "a declined bias must leave `out` unmodified"
        );
    }

    #[test]
    fn gemm_bias_declines_when_the_data_is_shorter_than_its_declared_shape() {
        // bias_shape claims 3 elements (a malformed model: shape/data length mismatch).
        let mut out = vec![1.0, 2.0, 3.0];
        let untouched = out.clone();
        assert!(!apply_gemm_bias(&mut out, &[1.0, 2.0], &[3], 1, 3, 1.0));
        assert_eq!(out, untouched);
    }

    #[test]
    fn gemm_bias_row_broadcast_applies_across_every_stacked_batch_slice() {
        // `out` may stack several batch slices (`out.len() == batch * m * n`); a row-
        // broadcast bias must repeat for every row across every slice, matching the
        // pre-fix behaviour for this already-supported shape (a8-4 regression guard).
        let mut out = vec![0.0; 8]; // batch=2, m=2, n=2 -> 4 rows total.
        assert!(apply_gemm_bias(&mut out, &[1.0, 2.0], &[2], 2, 2, 1.0));
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }
}
