//! CUDA-accelerated MatMul / Gemm dispatch.
//!
//! One entry point, `cuda_gemm_batched`, computes an entire batched matrix
//! multiplication — `batch` independent `[m, k] x [k, n]` products, with numpy
//! broadcasting on either operand — in **one** upload / launch / readback
//! round trip. [`cuda_matmul`] is the 2-D special case of it.
//!
//! # What this replaces, and why the round trip was the whole cost
//!
//! Until this rewrite, a batched MatMul ran `for i in 0..batch {
//! cuda_matmul(...) }`, and each of those iterations was a *complete* GPU
//! transaction: three `cuMemAlloc`s, two host uploads (each ending in a
//! context-wide `cuCtxSynchronize` — see `DeviceBuffer::copy_from_host`), one
//! GEMM, a two-stream fence, a blocking readback, and three `cuMemFree`s. For
//! a batch of 16 that is 48 allocations and 32 fences to compute 16 small
//! matrix products whose *kernels* take microseconds.
//!
//! The batch is now uploaded once, computed once, and read back once, out of
//! buffers the context keeps between calls (see [`mod@crate::residency`]).
//!
//! # Two dispatch shapes, and why the first one exists
//!
//! `plan_gemm` chooses between them from the operands' batch counts alone —
//! pure, unit-tested without a device, and the only place the choice is made:
//!
//! | plan | when | launch |
//! |---|---|---|
//! | `GemmPlan::Collapsed` | `b_batches == 1`: B is shared by every slice, which includes every unbatched 2-D MatMul | **one** tuned GEMM of `[batch*m, k] x [k, n]` |
//! | `GemmPlan::StridedBatch` | B varies across the batch | `gemm_strided_batched` over the whole batch, with a stride of `0` for a broadcast operand |
//!
//! The collapsed plan is not a shortcut, it is an identity: a row-major
//! `[batch, m, k]` tensor *is* a `[batch*m, k]` matrix, and multiplying it by
//! one shared `[k, n]` matrix produces exactly the `[batch, m, n]` stack the
//! per-slice loop produced, element for element. Taking it matters because it
//! is both the common case (an ONNX MatMul or Gemm against a graph initializer
//! is always this shape) and the fast case: one launch of the tuned
//! `GemmDispatcher` kernel with `batch*m` rows of parallelism, rather than
//! `batch` launches of `m` rows each.
//!
//! `gemm_strided_batched` handles what cannot be collapsed. Worth knowing
//! what it is, because the name promises more than the implementation
//! delivers: it is a host-side loop of one kernel launch per batch element,
//! and the kernel it launches is `GemmTemplate`'s naive triple loop rather
//! than the tuned `GemmDispatcher` kernel a plain `gemm` call gets. What it
//! buys here is therefore *not* a better kernel — it is that all `batch`
//! launches read operands already on the device and write into one output
//! allocation, with no host in the loop. That is the entire win, and it is
//! large because the host round trip was the cost; see this crate's
//! `examples/dispatch_bench.rs`.
//!
//! # Alpha stays on the host
//!
//! `Gemm`'s `alpha` could be handed to the kernel (both dispatch paths take
//! one) and deliberately is not. The shadow-verification oracle
//! ([`crate::reference::ref_matmul`]) computes an *unscaled* product, so
//! scaling inside the kernel would mean comparing a scaled GPU result against
//! an unscaled oracle — either silently weakening the check or teaching the
//! oracle about `alpha`. The caller applies it after verification, on a buffer
//! it is already walking. Identical arithmetic (`alpha * x` is commutative and
//! rounded once either way), strictly stronger verification.

use oxicuda_blas::batched::gemm_strided_batched;
use oxicuda_blas::{level3::gemm_api::gemm, Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_driver::Stream;

use crate::context::CudaContext;
use crate::error::CudaDispatchError;
use crate::residency::{Operand, WeightId};

/// Residency slot label for the left-hand GEMM operand.
///
/// A label is part of a cached operand's identity, so a name cached as a
/// GEMM's `A` can never be served to a kernel asking for its `B` (or for a
/// convolution's weight). See [`crate::residency::WeightId`].
pub(crate) const A_LABEL: &str = "matmul_a";

/// Residency slot label for the right-hand GEMM operand.
pub(crate) const B_LABEL: &str = "matmul_b";

/// One GEMM operand, already resolved against the context's residency cache.
///
/// The two data fields are mutually exclusive by construction, and the *caller*
/// is what makes that worth expressing: when [`Self::resident`] is `Some`, the
/// caller never had to build the host bytes at all — which for a `transB=1`
/// weight is the difference between one full host transpose per frame and
/// none.
pub(crate) struct GemmOperand<'a, 'c> {
    /// A device copy the context already holds for this identity.
    pub(crate) resident: Option<Operand<'c>>,
    /// Host bytes to upload, in the exact layout the GEMM reads (already
    /// transposed if the node asked for that). `None` exactly when
    /// [`Self::resident`] is `Some`.
    pub(crate) bytes: Option<&'a [f32]>,
    /// The identity to remember an upload under — `Some` only for a graph
    /// initializer, whose bytes are invariant for the session. `None` for an
    /// activation, whose bytes change every frame and must never be cached.
    pub(crate) id: Option<WeightId<'a>>,
}

impl<'a, 'c> GemmOperand<'a, 'c> {
    /// An operand to upload from `bytes`, remembered under `id` when `id` is
    /// `Some`.
    pub(crate) fn from_host(bytes: &'a [f32], id: Option<WeightId<'a>>) -> Self {
        Self {
            resident: None,
            bytes: Some(bytes),
            id,
        }
    }

    /// An operand the context already holds on the device.
    pub(crate) fn from_resident(resident: Operand<'c>) -> Self {
        Self {
            resident: Some(resident),
            bytes: None,
            id: None,
        }
    }

    /// Get this operand onto the device, uploading onto `stream` if it is not
    /// there already.
    ///
    /// `needed` is the element count the GEMM will actually read; a host slice
    /// longer than that (a `Tensor` whose data outruns its declared shape) is
    /// truncated rather than trusted, and a shorter one is an error rather
    /// than an out-of-bounds read.
    fn resolve(
        self,
        ctx: &'c CudaContext,
        label: &'static str,
        needed: usize,
        stream: &Stream,
    ) -> Result<Operand<'c>, CudaDispatchError> {
        match (self.resident, self.bytes) {
            (Some(resident), _) => Ok(resident),
            (None, Some(bytes)) => {
                let Some(slice) = bytes.get(..needed) else {
                    return Err(CudaDispatchError::Shape {
                        op: "MatMul",
                        msg: format!(
                            "operand holds {} elements but the declared shape needs {needed}",
                            bytes.len(),
                        ),
                    });
                };
                ctx.operand(self.id, label, slice, stream)
            }
            (None, None) => Err(CudaDispatchError::Shape {
                op: "MatMul",
                msg: "operand carries neither a resident device copy nor host bytes".to_string(),
            }),
        }
    }
}

/// A whole batched GEMM, as the dispatch layer describes it.
///
/// Every field is already validated by the caller (`try_cuda_dispatch`'s
/// MatMul arm): dimensions are non-zero, the inner dimensions agree, each
/// operand's batch count is exactly `1` or exactly `batch`, and each operand
/// holds at least `batches * rows * cols` elements.
pub(crate) struct BatchedGemm<'a, 'c> {
    /// Left operand: `a_batches` slices of `[m, k]`, row-major.
    pub(crate) a: GemmOperand<'a, 'c>,
    /// Right operand: `b_batches` slices of `[k, n]`, row-major.
    pub(crate) b: GemmOperand<'a, 'c>,
    /// Rows of each output slice.
    pub(crate) m: usize,
    /// Shared inner dimension.
    pub(crate) k: usize,
    /// Columns of each output slice.
    pub(crate) n: usize,
    /// Output batch slices — the broadcast of the two operands' batch shapes.
    pub(crate) batch: usize,
    /// Batch slices actually stored in `a`: `1` (broadcast) or `batch`.
    pub(crate) a_batches: usize,
    /// Batch slices actually stored in `b`: `1` (broadcast) or `batch`.
    pub(crate) b_batches: usize,
}

/// How a batched GEMM reaches the device.
///
/// Chosen by [`plan_gemm`] from the batch counts alone — no device, no
/// heuristics, no measurement-dependent thresholds. See the [module
/// docs](self) for what each one costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmPlan {
    /// One tuned GEMM over `[batch*m, k] x [k, n]`.
    ///
    /// Valid exactly when B is shared by every batch slice: the slices of A
    /// are then contiguous rows of one row-major matrix, so stacking them is a
    /// reinterpretation of the same bytes rather than a copy.
    Collapsed,
    /// [`gemm_strided_batched`] over the whole batch.
    StridedBatch {
        /// Elements between consecutive A slices; `0` when A is broadcast.
        stride_a: i64,
        /// Elements between consecutive B slices.
        stride_b: i64,
    },
}

/// Decide how to launch, from the broadcast bookkeeping the caller computed.
///
/// Pure and total, so the one decision that could silently pair the wrong
/// batch slices is testable on a host with no GPU.
///
/// # The stride-0 rule *is* the broadcast contract
///
/// The per-slice loop this replaces indexed operand `i` as `(i %
/// operand_batches) * slice`, having already established upstream that
/// `operand_batches` is either `1` or `batch`. Those two cases are exactly a
/// stride of `0` and a stride of one slice — which is what a strided-batch
/// launch expresses natively, so the translation is an identity rather than a
/// reinterpretation. A caller that skipped the upstream check would get a
/// stride-0 read (the same slice repeatedly) rather than a buffer overrun,
/// because this treats "not `batch`" as "broadcast".
#[must_use]
pub(crate) fn plan_gemm(
    m: usize,
    k: usize,
    n: usize,
    a_batches: usize,
    b_batches: usize,
) -> GemmPlan {
    if b_batches == 1 {
        // Includes every unbatched 2-D MatMul, where batch == 1.
        return GemmPlan::Collapsed;
    }
    GemmPlan::StridedBatch {
        stride_a: if a_batches == 1 {
            0
        } else {
            (m.saturating_mul(k)) as i64
        },
        stride_b: (k.saturating_mul(n)) as i64,
    }
}

/// Run a whole batched `A @ B` on the GPU and return the stacked row-major
/// result (`batch * m * n` elements).
///
/// Returns `Ok(None)` for a configuration this path declines — currently only
/// a dimension too large for the kernels' `u32` launch parameters, which the
/// CPU operator has no equivalent limit on.
///
/// # Errors
///
/// A real CUDA failure after dispatch was committed to: allocation, upload,
/// launch, or readback.
pub(crate) fn cuda_gemm_batched(
    ctx: &CudaContext,
    request: BatchedGemm<'_, '_>,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    let BatchedGemm {
        a,
        b,
        m,
        k,
        n,
        batch,
        a_batches,
        b_batches,
    } = request;

    // Every dimension the kernels take is a `u32`, and every length below is
    // model-derived. An out-of-range one is a decline (the CPU computes the
    // node) rather than a silent truncation into a wrong-but-plausible launch
    // geometry.
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
    let (Ok(m_u32), Ok(k_u32), Ok(n_u32), Ok(batch_u32), Ok(slice_c_i64)) = (
        u32::try_from(m),
        u32::try_from(k),
        u32::try_from(n),
        u32::try_from(batch),
        i64::try_from(slice_c),
    ) else {
        return Ok(None);
    };

    // Everything below — the uploads, the zero-fill, the GEMM launches and the
    // readback — is issued on the BLAS handle's *own* stream, which is not
    // `ctx.dnn.stream()` (see `DnnHandle::build`). Being on one stream is what
    // orders them against each other without a single fence: the GEMM cannot
    // start before its operands have landed, and the readback cannot start
    // before the GEMM has finished, by stream semantics alone. One synchronise
    // at the end is then the only host/device rendezvous in the dispatch.
    let stream = ctx.dnn.blas().stream();

    let mut d_a = a.resolve(ctx, A_LABEL, a_needed, stream)?;
    let mut d_b = b.resolve(ctx, B_LABEL, b_needed, stream)?;

    let mut d_c = ctx.scratch(out_total)?;
    // A GEMM with beta = 0 still evaluates `beta * C`, and `0.0 * NaN` is
    // `NaN`. A recycled or freshly allocated buffer may hold either, so the
    // output is zeroed exactly as the pre-pool code's `DeviceBuffer::zeroed`
    // did — but stream-ordered, without that call's context-wide fence.
    d_c.zero_fill(stream)?;

    match plan_gemm(m, k, n, a_batches, b_batches) {
        GemmPlan::Collapsed => {
            // `[batch*m, k] x [k, n]`. The row count is the only quantity that
            // changes, and the only one that can overflow a `u32` when the
            // per-slice `m` could not.
            let Some(rows) = batch.checked_mul(m) else {
                return Ok(None);
            };
            let Ok(rows_u32) = u32::try_from(rows) else {
                return Ok(None);
            };
            let desc_a =
                MatrixDesc::<f32>::from_buffer(d_a.buffer(), rows_u32, k_u32, Layout::RowMajor)
                    .map_err(blas_err)?;
            let desc_b =
                MatrixDesc::<f32>::from_buffer(d_b.buffer(), k_u32, n_u32, Layout::RowMajor)
                    .map_err(blas_err)?;
            let mut desc_c = MatrixDescMut::<f32>::from_buffer(
                d_c.buffer_mut(),
                rows_u32,
                n_u32,
                Layout::RowMajor,
            )
            .map_err(blas_err)?;

            gemm(
                ctx.dnn.blas(),
                Transpose::NoTrans,
                Transpose::NoTrans,
                1.0_f32,
                &desc_a,
                &desc_b,
                0.0_f32,
                &mut desc_c,
            )
            .map_err(blas_err)?;
        }
        GemmPlan::StridedBatch { stride_a, stride_b } => {
            let c_ptr = d_c.device_ptr();
            gemm_strided_batched::<f32>(
                ctx.dnn.blas(),
                Transpose::NoTrans,
                Transpose::NoTrans,
                m_u32,
                n_u32,
                k_u32,
                1.0_f32,
                d_a.device_ptr(),
                k_u32,
                stride_a,
                d_b.device_ptr(),
                n_u32,
                stride_b,
                0.0_f32,
                // C and D are the same buffer: the kernel computes
                // `D = alpha*A*B + beta*C` in place, which over the zero-filled
                // output above with `beta = 0` is exactly `alpha*A*B`. Passing
                // one buffer for both also skips the device-to-device C -> D
                // snapshot `gemm_strided_batched` would otherwise make per
                // batch element.
                c_ptr,
                n_u32,
                slice_c_i64,
                c_ptr,
                n_u32,
                slice_c_i64,
                batch_u32,
            )
            .map_err(blas_err)?;
        }
    }

    let mut out = vec![0.0_f32; out_total];
    d_c.download(&mut out, stream)?;
    // The one fence in the whole dispatch. Everything above was enqueued on
    // `stream` in order; this is where the host waits for all of it, and after
    // it `out` holds the finished result.
    stream.synchronize().map_err(CudaDispatchError::Driver)?;
    // ...and only now may these allocations go back to the pool. See
    // `PooledBuffer`'s "a borrow is only recycled once its stream work is
    // known to be done".
    d_a.retire();
    d_b.retire();
    d_c.retire();
    Ok(Some(out))
}

/// Run a 2-D `A @ B` on the GPU.
///
/// The `batch = 1` special case of this module's crate-private
/// `cuda_gemm_batched`, kept as the documented single-matrix entry point.
///
/// * `a_data` — flattened row-major f32 data for A, shape `[m, k]`.
/// * `b_data` — flattened row-major f32 data for B, shape `[k, n]`.
/// * `m`, `k`, `n` — matrix dimensions.
///
/// Neither operand is cached across calls: this entry point takes bare slices
/// and so has no identity to key them under. `try_cuda_dispatch` calls
/// `cuda_gemm_batched` directly, where an operand that *is* a graph
/// initializer carries one.
///
/// # Errors
///
/// A CUDA failure (allocation, upload, launch, readback), or
/// [`CudaDispatchError::Shape`] for dimensions too large for a `u32` kernel
/// launch — which, unlike the batched entry point's `Ok(None)`, has to be an
/// error here because this signature has no way to decline.
///
/// [`CudaDispatchError::Shape`]: crate::error::CudaDispatchError::Shape
pub fn cuda_matmul(
    ctx: &CudaContext,
    a_data: &[f32],
    b_data: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, CudaDispatchError> {
    let request = BatchedGemm {
        a: GemmOperand::from_host(a_data, None),
        b: GemmOperand::from_host(b_data, None),
        m,
        k,
        n,
        batch: 1,
        a_batches: 1,
        b_batches: 1,
    };
    match cuda_gemm_batched(ctx, request)? {
        Some(out) => Ok(out),
        None => Err(CudaDispatchError::Shape {
            op: "MatMul",
            msg: format!("dimensions {m}x{k}x{n} exceed a u32 kernel launch"),
        }),
    }
}

/// Wrap an `oxicuda-blas` error in this crate's dispatch error.
fn blas_err(e: oxicuda_blas::error::BlasError) -> CudaDispatchError {
    CudaDispatchError::Blas(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `plan_gemm` is the one decision in this module that can silently pair
    // the wrong batch slices, and it is pure -- so it is tested here, on every
    // host, rather than only on the machine that has a GPU. The numeric
    // consequences of each plan are pinned on-device by
    // `tests/batched_matmul_gpu.rs`.

    #[test]
    fn a_plain_2d_matmul_collapses() {
        assert_eq!(plan_gemm(4, 5, 6, 1, 1), GemmPlan::Collapsed);
    }

    #[test]
    fn a_shared_right_operand_collapses_whatever_the_batch() {
        // The ONNX shape that matters most: `[batch, m, k] @ [k, n]`, a batch
        // of activations against one weight matrix.
        for batch in [2usize, 8, 64] {
            assert_eq!(
                plan_gemm(7, 9, 3, batch, 1),
                GemmPlan::Collapsed,
                "batch={batch}: a shared B must collapse into one GEMM",
            );
        }
    }

    #[test]
    fn both_operands_batched_take_the_strided_path_with_real_strides() {
        let (m, k, n) = (4usize, 5usize, 6usize);
        assert_eq!(
            plan_gemm(m, k, n, 3, 3),
            GemmPlan::StridedBatch {
                stride_a: (m * k) as i64,
                stride_b: (k * n) as i64,
            },
        );
    }

    #[test]
    fn a_shared_left_operand_takes_the_strided_path_with_a_zero_a_stride() {
        // `[m, k] @ [batch, k, n]` cannot collapse -- B varies per slice -- so
        // A is read at stride 0, reusing its single slice for every launch.
        let (m, k, n) = (6usize, 4usize, 5usize);
        assert_eq!(
            plan_gemm(m, k, n, 1, 3),
            GemmPlan::StridedBatch {
                stride_a: 0,
                stride_b: (k * n) as i64,
            },
        );
    }

    #[test]
    fn the_strided_plan_never_emits_two_zero_strides() {
        // `gemm_strided_batched` rejects an all-zero stride set with
        // `batch_count > 1`, and rightly so: it would compute the same product
        // `batch` times. The only way to reach the strided plan is
        // `b_batches > 1`, which forces a non-zero B stride.
        for a_batches in [1usize, 5] {
            match plan_gemm(3, 3, 3, a_batches, 5) {
                GemmPlan::StridedBatch { stride_b, .. } => {
                    assert_ne!(stride_b, 0, "a batched B must advance");
                }
                GemmPlan::Collapsed => panic!("b_batches > 1 must not collapse"),
            }
        }
    }

    #[test]
    fn the_collapsed_plan_covers_exactly_the_shared_b_case() {
        // Stated as a biconditional so a future edit cannot widen `Collapsed`
        // to a case where B varies -- which would multiply every slice of A by
        // B's *first* slice and hand back a correctly-shaped wrong answer.
        for a_batches in [1usize, 2, 7] {
            for b_batches in [1usize, 2, 7] {
                let collapsed = matches!(
                    plan_gemm(2, 3, 4, a_batches, b_batches),
                    GemmPlan::Collapsed
                );
                assert_eq!(
                    collapsed,
                    b_batches == 1,
                    "a_batches={a_batches}, b_batches={b_batches}",
                );
            }
        }
    }

    #[test]
    fn the_operand_labels_are_distinct() {
        // Two operands sharing a label would let a name cached as A be served
        // to a kernel asking for B.
        assert_ne!(A_LABEL, B_LABEL);
    }
}
