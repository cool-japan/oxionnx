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
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::Stream;

use crate::activation::{finish_output, retire_queued, CudaOutputPlacement, KernelOutput};
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
    match cuda_gemm_batched_placed(ctx, request, &[], CudaOutputPlacement::Host)? {
        Some(KernelOutput::Host(out)) => Ok(Some(out)),
        Some(KernelOutput::Device(_)) => Err(CudaDispatchError::Shape {
            op: "MatMul",
            msg: "host placement produced a device-resident result".to_string(),
        }),
        None => Ok(None),
    }
}

/// [`cuda_gemm_batched`], with the result left on the device when the caller
/// asks for it.
///
/// `out_shape` is the ONNX output shape (the broadcast batch prefix plus
/// `[m, n]`); it is only consulted on the device path, where it becomes the
/// resident tensor's shape.
///
/// # The graph-replay path is host-only, deliberately
///
/// [`crate::graph_cache`] replays a recorded launch **and its read-back** into
/// a host `Vec`, so it cannot serve a device-placement request. Rather than
/// teach it two output modes, a device request skips it. Nothing is lost:
/// graph replay is off by default, and its whole value — amortising launch
/// overhead — is dwarfed on the resident path by the fence and the round trip
/// that residency has already removed. Residency also *improves* the graph
/// path where it does run: an operand bound from a resident activation is an
/// externally-owned pointer in the key, exactly like a resident weight, so a
/// steady-state frame that reuses the same pooled allocations hits the same
/// recording rather than re-recording.
///
/// # Errors
///
/// As [`cuda_gemm_batched`].
pub(crate) fn cuda_gemm_batched_placed(
    ctx: &CudaContext,
    request: BatchedGemm<'_, '_>,
    out_shape: &[usize],
    placement: CudaOutputPlacement,
) -> Result<Option<KernelOutput>, CudaDispatchError> {
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

    // ── Graph replay, when it applies ─────────────────────────────────────
    //
    // Tried before anything is uploaded, because the graph path uploads into
    // its *own* buffers rather than the pool's (see `mod@crate::graph_cache`
    // on pointer stability) and doing it after would pay for both. Declines —
    // graphs off, an operand not yet resident, a poisoned key — cost one
    // boolean and fall through to the ordinary path below with `a`/`b`
    // untouched.
    if ctx.graphs.enabled() && placement == CudaOutputPlacement::Host {
        let plan = plan_gemm(m, k, n, a_batches, b_batches);
        if let Some(out) = try_graph_gemm(
            ctx,
            &a,
            &b,
            GemmDims {
                m_u32,
                k_u32,
                n_u32,
                batch_u32,
                slice_c_i64,
            },
            (a_needed, b_needed, out_total),
            plan,
            stream,
        )? {
            return Ok(Some(KernelOutput::Host(out)));
        }
    }

    let mut d_a = a.resolve(ctx, A_LABEL, a_needed, stream)?;
    let mut d_b = b.resolve(ctx, B_LABEL, b_needed, stream)?;

    let d_c = ctx.scratch(out_total)?;

    issue_gemm(
        ctx,
        plan_gemm(m, k, n, a_batches, b_batches),
        GemmDims {
            m_u32,
            k_u32,
            n_u32,
            batch_u32,
            slice_c_i64,
        },
        GemmPointers {
            a: d_a.device_ptr(),
            b: d_b.device_ptr(),
            c: d_c.device_ptr(),
        },
        GemmCapacities {
            a: a_needed,
            b: b_needed,
            c: out_total,
        },
        stream,
    )?;

    // The one fence in the whole dispatch, on the host path: everything above
    // was enqueued on `stream` in order, and this is where the host waits for
    // all of it. On the device path there is no fence and no read-back at all.
    let out = finish_output(ctx, d_c, out_total, out_shape, placement, stream)?;
    // ...and only now may these allocations go back to the pool. See
    // `PooledBuffer`'s "a borrow is only recycled once its stream work is
    // known to be done", and `mod@crate::activation` for why the device path
    // may recycle without a fence.
    match &out {
        KernelOutput::Host(_) => {
            d_a.retire();
            d_b.retire();
        }
        KernelOutput::Device(_) => {
            retire_queued(ctx, &mut d_a);
            retire_queued(ctx, &mut d_b);
        }
    }
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

// ── The launch, shared by the ordinary and the graph-recorded path ─────────

/// The `u32`/`i64` launch dimensions a GEMM needs, already range-checked by
/// [`cuda_gemm_batched`].
#[derive(Debug, Clone, Copy)]
struct GemmDims {
    /// Rows of one output slice.
    m_u32: u32,
    /// Shared inner dimension.
    k_u32: u32,
    /// Columns of one output slice.
    n_u32: u32,
    /// Output batch slices.
    batch_u32: u32,
    /// Elements between consecutive output slices.
    slice_c_i64: i64,
}

/// Where the three operands live on the device for one launch.
#[derive(Debug, Clone, Copy)]
struct GemmPointers {
    a: CUdeviceptr,
    b: CUdeviceptr,
    c: CUdeviceptr,
}

/// How many `f32` elements each allocation behind [`GemmPointers`] actually
/// holds.
///
/// Carried explicitly because the launch below builds its matrix descriptors
/// with `MatrixDesc::from_raw`, which — unlike `from_buffer` — performs no
/// bounds check of its own. Dropping the check rather than moving it would
/// turn a malformed model into an out-of-bounds device read.
#[derive(Debug, Clone, Copy)]
struct GemmCapacities {
    a: usize,
    b: usize,
    c: usize,
}

/// Zero the output and issue the GEMM launches for `plan`.
///
/// **The single definition of what a GEMM dispatch launches.** Both the
/// ordinary path and [`crate::graph_cache`]'s recording call exactly this, so
/// a recorded graph replays the same work the ordinary path would have done —
/// which is what makes the two comparable at all, and what stops the two from
/// drifting apart under later edits.
///
/// # Errors
///
/// [`CudaDispatchError::Shape`] when an allocation is too small for the
/// descriptor it would back, or the BLAS error from the launch.
fn issue_gemm(
    ctx: &CudaContext,
    plan: GemmPlan,
    dims: GemmDims,
    ptrs: GemmPointers,
    capacities: GemmCapacities,
    stream: &Stream,
) -> Result<(), CudaDispatchError> {
    // A GEMM with beta = 0 still evaluates `beta * C`, and `0.0 * NaN` is
    // `NaN`. A recycled or freshly allocated buffer may hold either, so the
    // output is zeroed exactly as the pre-pool code's `DeviceBuffer::zeroed`
    // did — but stream-ordered, without that call's context-wide fence.
    // `cuMemsetD32Async` is a stream-ordered operation, so it records into a
    // graph as a memset node like any launch.
    oxicuda_driver::memory_info::memset_d32_async(ptrs.c, 0, capacities.c, stream)
        .map_err(CudaDispatchError::Driver)?;

    match plan {
        GemmPlan::Collapsed => {
            // `[batch*m, k] x [k, n]`. The row count is the only quantity that
            // changes, and the only one that can overflow a `u32` when the
            // per-slice `m` could not.
            let Some(rows_u32) = dims.batch_u32.checked_mul(dims.m_u32) else {
                return Err(CudaDispatchError::Shape {
                    op: "MatMul",
                    msg: format!(
                        "collapsed row count {} x {} exceeds a u32 kernel launch",
                        dims.batch_u32, dims.m_u32,
                    ),
                });
            };
            let desc_a = raw_desc(ptrs.a, rows_u32, dims.k_u32, capacities.a, "A")?;
            let desc_b = raw_desc(ptrs.b, dims.k_u32, dims.n_u32, capacities.b, "B")?;
            check_capacity(rows_u32, dims.n_u32, capacities.c, "C")?;
            let mut desc_c = MatrixDescMut::<f32>::from_raw(
                ptrs.c,
                rows_u32,
                dims.n_u32,
                dims.n_u32,
                Layout::RowMajor,
            );

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
            gemm_strided_batched::<f32>(
                ctx.dnn.blas(),
                Transpose::NoTrans,
                Transpose::NoTrans,
                dims.m_u32,
                dims.n_u32,
                dims.k_u32,
                1.0_f32,
                ptrs.a,
                dims.k_u32,
                stride_a,
                ptrs.b,
                dims.n_u32,
                stride_b,
                0.0_f32,
                // C and D are the same buffer: the kernel computes
                // `D = alpha*A*B + beta*C` in place, which over the zero-filled
                // output above with `beta = 0` is exactly `alpha*A*B`. Passing
                // one buffer for both also skips the device-to-device C -> D
                // snapshot `gemm_strided_batched` would otherwise make per
                // batch element.
                ptrs.c,
                dims.n_u32,
                dims.slice_c_i64,
                ptrs.c,
                dims.n_u32,
                dims.slice_c_i64,
                dims.batch_u32,
            )
            .map_err(blas_err)?;
        }
    }
    Ok(())
}

/// Refuse a descriptor whose `rows x cols` outruns the `held` elements behind
/// its pointer.
///
/// This is `MatrixDesc::from_buffer`'s length check, relocated: the launch
/// path takes raw pointers so that a graph recording and an ordinary dispatch
/// can share it, and a raw pointer carries no length.
fn check_capacity(
    rows: u32,
    cols: u32,
    held: usize,
    operand: &str,
) -> Result<(), CudaDispatchError> {
    let required = rows as usize * cols as usize;
    if held < required {
        return Err(CudaDispatchError::Shape {
            op: "MatMul",
            msg: format!(
                "operand {operand} holds {held} elements but a {rows}x{cols} descriptor needs \
                 {required}"
            ),
        });
    }
    Ok(())
}

/// A row-major read-only descriptor over `ptr`, bounds-checked against `held`.
fn raw_desc(
    ptr: CUdeviceptr,
    rows: u32,
    cols: u32,
    held: usize,
    operand: &str,
) -> Result<MatrixDesc<f32>, CudaDispatchError> {
    check_capacity(rows, cols, held, operand)?;
    Ok(MatrixDesc::<f32>::from_raw(
        ptr,
        rows,
        cols,
        cols,
        Layout::RowMajor,
    ))
}

// ── The graph-recorded path ───────────────────────────────────────────────

/// Attempt this GEMM through [`crate::graph_cache`], returning the finished
/// result when the graph path handled it.
///
/// Returns `Ok(None)` — meaning "use the ordinary path" — for every case the
/// graph path declines, and those declines are the interesting part:
///
/// * **An operand carries an identity but is not resident yet.** That is the
///   *first* dispatch of a graph initializer. Recording it now would bake in
///   the address of a pooled scratch buffer, which the pool recycles; letting
///   the ordinary path run instead makes the weight resident, and the second
///   dispatch — the one that repeats forever — records against its stable
///   address. Costing one un-recorded frame to get a sound recording is the
///   right trade in a workload measured in thousands of frames.
/// * **The cache is full, or this key already failed to record.** See
///   [`crate::graph_cache::GraphCache::run`].
///
/// # Errors
///
/// Only a genuine device failure during the replay itself (upload, launch,
/// readback, synchronise). A failed *recording* is a decline, not an error.
#[allow(clippy::too_many_arguments)]
fn try_graph_gemm(
    ctx: &CudaContext,
    a: &GemmOperand<'_, '_>,
    b: &GemmOperand<'_, '_>,
    dims: GemmDims,
    needed: (usize, usize, usize),
    plan: GemmPlan,
    stream: &Stream,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    let (a_needed, b_needed, out_total) = needed;
    let (Some(a_source), Some(b_source)) = (
        GraphOperand::classify(a, a_needed),
        GraphOperand::classify(b, b_needed),
    ) else {
        return Ok(None);
    };

    // Owned-buffer layout: every uploaded operand first, in A-then-B order,
    // then the output. `GraphOperand::Upload` records its own slot index so a
    // caller never has to recompute this ordering. Stack arrays throughout:
    // this runs on every dispatch of every eligible node, and the shapes this
    // cache exists for are the ones that repeat thousands of times.
    let mut owned_lens = [0usize; 3];
    let mut owned_len = 0usize;
    let mut external_ptrs = [0 as CUdeviceptr; 2];
    let mut external_len = 0usize;
    let a_slot = a_source.reserve(
        &mut owned_lens,
        &mut owned_len,
        &mut external_ptrs,
        &mut external_len,
    );
    let b_slot = b_source.reserve(
        &mut owned_lens,
        &mut owned_len,
        &mut external_ptrs,
        &mut external_len,
    );
    let c_slot = owned_len;
    owned_lens[c_slot] = out_total;
    owned_len += 1;
    let owned_lens = &owned_lens[..owned_len];

    // Everything that changes what gets launched goes into the key. The plan
    // discriminant and its strides are in there explicitly rather than being
    // trusted to follow from the dimensions: `plan_gemm` is where that mapping
    // lives, and a future change to it must not silently alias two different
    // recordings.
    let (plan_tag, stride_a, stride_b) = match plan {
        GemmPlan::Collapsed => (0u64, 0i64, 0i64),
        GemmPlan::StridedBatch { stride_a, stride_b } => (1u64, stride_a, stride_b),
    };
    let Some(key) = crate::graph_cache::GraphKey::new(
        "gemm",
        &[
            u64::from(dims.m_u32),
            u64::from(dims.k_u32),
            u64::from(dims.n_u32),
            u64::from(dims.batch_u32),
            plan_tag,
            stride_a as u64,
            stride_b as u64,
            a_slot.map_or(u64::MAX, |slot| slot as u64),
            b_slot.map_or(u64::MAX, |slot| slot as u64),
        ],
        &external_ptrs[..external_len],
    ) else {
        return Ok(None);
    };

    let mut out = vec![0.0_f32; out_total];
    let ran = ctx.graphs.run(
        key,
        owned_lens,
        stream,
        // pre: this frame's activations cross the bus into the entry's own
        // buffers. A resident operand uploads nothing.
        |ptrs| {
            a_source.upload(a_slot, ptrs, stream)?;
            b_source.upload(b_slot, ptrs, stream)
        },
        // record: exactly the launches the ordinary path issues.
        |ptrs| {
            let Some(&c_ptr) = ptrs.get(c_slot) else {
                return Err(CudaDispatchError::Shape {
                    op: "MatMul",
                    msg: "graph cache did not provide the output buffer".to_string(),
                });
            };
            issue_gemm(
                ctx,
                plan,
                dims,
                GemmPointers {
                    a: a_source.device_ptr(a_slot, ptrs)?,
                    b: b_source.device_ptr(b_slot, ptrs)?,
                    c: c_ptr,
                },
                GemmCapacities {
                    a: a_needed,
                    b: b_needed,
                    c: out_total,
                },
                stream,
            )
        },
        // post: read the result back out of the entry's own output buffer.
        |ptrs| {
            let Some(&c_ptr) = ptrs.get(c_slot) else {
                return Err(CudaDispatchError::Shape {
                    op: "MatMul",
                    msg: "graph cache did not provide the output buffer".to_string(),
                });
            };
            // SAFETY: a non-owning view of exactly `out.len()` elements over a
            // buffer the cache allocated with `out_total >= out.len()`
            // elements; its drop frees nothing.
            let view = unsafe { oxicuda_memory::DeviceBuffer::<f32>::from_raw(c_ptr, out.len()) };
            view.copy_to_host_async(&mut out, stream)
                .map_err(CudaDispatchError::Driver)
        },
    )?;

    Ok(ran.then_some(out))
}

/// How one GEMM operand reaches a recorded graph.
///
/// The distinction is exactly the pointer-stability one: a resident weight
/// already has an address that outlives the recording, while an activation
/// needs the cache to own a buffer on its behalf.
enum GraphOperand<'a> {
    /// Already on the device at a stable address, which the recording bakes in
    /// and the key therefore carries.
    Resident(CUdeviceptr),
    /// Uploaded from host bytes into a buffer the cache owns, every call.
    Upload(&'a [f32]),
}

impl<'a> GraphOperand<'a> {
    /// Classify `operand`, or `None` if it cannot take part in a recording.
    ///
    /// The `None` case is an operand that has a residency identity but no
    /// device copy yet — see [`try_graph_gemm`]'s decline list.
    fn classify(operand: &'a GemmOperand<'_, '_>, needed: usize) -> Option<Self> {
        if let Some(resident) = &operand.resident {
            return Some(Self::Resident(resident.device_ptr()));
        }
        if operand.id.is_some() {
            return None;
        }
        // An activation: its bytes change every frame, so the graph reads them
        // out of a cache-owned buffer this dispatch refills.
        operand.bytes?.get(..needed).map(Self::Upload)
    }

    /// Claim this operand's place in the entry's owned-buffer layout.
    ///
    /// Returns the slot index for an uploaded operand, or `None` for a
    /// resident one (whose address goes into `external_ptrs` instead, becoming
    /// part of the key).
    ///
    /// Writes into caller-provided fixed arrays rather than `Vec`s so a
    /// dispatch that takes this path allocates nothing; both arrays are sized
    /// by the caller for a GEMM's two operands plus its output.
    fn reserve(
        &self,
        owned_lens: &mut [usize; 3],
        owned_len: &mut usize,
        external_ptrs: &mut [CUdeviceptr; 2],
        external_len: &mut usize,
    ) -> Option<usize> {
        match self {
            Self::Resident(ptr) => {
                external_ptrs[*external_len] = *ptr;
                *external_len += 1;
                None
            }
            Self::Upload(bytes) => {
                let slot = *owned_len;
                owned_lens[slot] = bytes.len();
                *owned_len += 1;
                Some(slot)
            }
        }
    }

    /// The device address a recorded launch should read this operand from.
    fn device_ptr(
        &self,
        slot: Option<usize>,
        ptrs: &[CUdeviceptr],
    ) -> Result<CUdeviceptr, CudaDispatchError> {
        match self {
            Self::Resident(ptr) => Ok(*ptr),
            Self::Upload(_) => slot
                .and_then(|slot| ptrs.get(slot).copied())
                .ok_or_else(|| CudaDispatchError::Shape {
                    op: "MatMul",
                    msg: "graph cache did not provide an operand buffer".to_string(),
                }),
        }
    }

    /// Put this frame's bytes on the device, for an uploaded operand.
    fn upload(
        &self,
        slot: Option<usize>,
        ptrs: &[CUdeviceptr],
        stream: &Stream,
    ) -> Result<(), CudaDispatchError> {
        let Self::Upload(bytes) = self else {
            return Ok(());
        };
        let ptr = self.device_ptr(slot, ptrs)?;
        // SAFETY: a non-owning view of exactly `bytes.len()` elements over a
        // buffer the cache allocated with that many elements (see `reserve`);
        // its drop frees nothing.
        let mut view = unsafe { oxicuda_memory::DeviceBuffer::<f32>::from_raw(ptr, bytes.len()) };
        view.copy_from_host_async(bytes, stream)
            .map_err(CudaDispatchError::Driver)
    }
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
