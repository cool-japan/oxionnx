//! The D3D12 compute-shader engine — the fallback used when `DMLCreateDevice` fails.
//!
//! Owner **B4**.  Its semantics are *defined* by [`crate::reference`]: the shaders in
//! [`crate::hlsl`] are a literal transcription of those functions, and
//! `DirectMLContext::self_check` diffs the two on real hardware.
//!
//! # The canonical sequence — written exactly once, in [`HlslEngine::execute`]
//!
//! ```text
//! core.begin()
//!   → for each input:  device.barrier_to(COPY_DEST)
//!                      device.record_copy_from(upload)        (UPLOAD → DEFAULT)
//!   → for each input:  device.barrier_to(NON_PIXEL_SHADER_RESOURCE)
//!   → output.barrier_to(UNORDERED_ACCESS)
//!   → SetPipelineState / SetComputeRootSignature
//!   → SetComputeRootShaderResourceView(t0) / (t1) / SetComputeRootUnorderedAccessView(u0)
//!   → per dispatch:    SetComputeRoot32BitConstants(b0, 8)
//!                      Dispatch(grid.x, grid.y, grid.z)       ← grid comes from `plan`
//!   → output.record_uav_barrier()                             ← THE ONE EVERYONE OMITS
//!   → output.barrier_to(COPY_SOURCE)
//!   → readback.record_copy_from(output)                       (DEFAULT → READBACK)
//! core.submit_and_wait()
//!   → readback.read_f32()
//! ```
//!
//! `matmul`, `binary` and `unary` differ only in *what they stage* and *what program
//! they hand to [`HlslEngine::execute`]*.  None of them records a barrier itself.  That
//! is deliberate: a barrier sequence that exists in three copies is a barrier sequence
//! that is wrong in at least one of them, and the failure mode — a missing UAV barrier —
//! is *correct on NVIDIA and garbage on AMD*.  One copy, or none.
//!
//! # Why the recorded region cannot fail
//!
//! Every fallible step — PSO compilation, buffer creation, upload, grid math, root
//! constants — happens **before** `core.begin()`.  Everything between `begin()` and
//! `submit_and_wait()` is infallible by construction: `barrier_to`,
//! `record_copy_from`, `record_uav_barrier` and the `ID3D12GraphicsCommandList::Set*` /
//! `Dispatch` methods all return `()`.
//!
//! This is not tidiness.  `D3d12Core::begin()` does `allocator.Reset()` +
//! `list.Reset()`, and an early `return Err(…)` in the middle of recording would leave
//! the command list *open*, so the next node's `begin()` would reset a list that is
//! still in the recording state.  Making the region infallible removes the failure mode
//! rather than handling it.
//!
//! # Why the GPU is never left holding a freed buffer
//!
//! Every [`GpuBuffer`] in a dispatch is a local of `matmul` / `binary` / `unary`, so it
//! is dropped when that function returns — which is *after* `submit_and_wait()` has
//! fence-waited for the GPU to finish. There is no path that releases a resource the
//! GPU still owns.
//!
//! # What this module never does
//!
//! * It **never** computes a dispatch grid.  [`MatMulPlan::hlsl_grid`] and
//!   [`ElementwisePlan::hlsl_grid`] do, and they are unit-tested on Linux against the
//!   orientation the shader actually reads (`row = tid.y`, `col = tid.x`, hence
//!   `x = ceil(N/16)`, `y = ceil(M/16)` — the transposition of which silently leaves
//!   part of the output matrix as whatever the allocator left behind).
//! * It **never** writes a bare `as u32` on a shape-derived value.  [`crate::plan`]
//!   range-checked all of them once, up front.
//! * It **never** creates a CBV, so the 256-byte constant-buffer alignment rule cannot
//!   be violated: the eight root constants go straight into `b0` via
//!   `SetComputeRoot32BitConstants`.
//! * It **never** barriers an UPLOAD or a READBACK buffer.  Those heaps are pinned to
//!   `GENERIC_READ` and `COPY_DEST` respectively, for their whole lifetime, by D3D12.

use windows::Win32::Graphics::Direct3D12::{
    ID3D12GraphicsCommandList, ID3D12PipelineState, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
};

use super::buffer::GpuBuffer;
use super::device::D3d12Core;
use super::pso::{
    PsoCache, ROOT_PARAM_CONSTANTS, ROOT_PARAM_SRV_A, ROOT_PARAM_SRV_B, ROOT_PARAM_UAV_C,
};
use super::shader::ShaderKind;
use crate::error::{DirectMLError, Result};
use crate::plan::{
    apply_gemm_epilogue, broadcast_expand, numel, transpose_2d, BinaryOp, DispatchGrid,
    ElementwisePlan, MatMulPlan, ReducePlan, SoftmaxPlan, UnaryOp, ELEM_SIZE, ROOT_CONSTANT_COUNT,
};

/// [`ROOT_CONSTANT_COUNT`] as the `u32` that `SetComputeRoot32BitConstants` wants.
///
/// This is *not* a shape-derived value — it is the fixed width of the one root
/// signature shared by every entry point, and it is `8`.  The `as` cast is therefore
/// exact, and const-evaluated.
const ROOT_CONSTANTS_U32: u32 = ROOT_CONSTANT_COUNT as u32;

/// The root constants for one `Dispatch`, paired with the grid to dispatch them over.
///
/// A `matmul` produces one of these per batch slice; the elementwise kernels produce
/// exactly one.  Both are built entirely from [`crate::plan`] — see
/// [`matmul_program`] and [`elementwise_program`], which are pure, GPU-free, and the
/// only two functions in this module that a test can exercise without a device.
type DispatchStep = ([u32; ROOT_CONSTANT_COUNT], DispatchGrid);

/// One shader operand, staged as the UPLOAD → DEFAULT pair the copy needs.
///
/// The DEFAULT buffer is what the shader reads through `t0` / `t1`; the UPLOAD buffer
/// is the CPU-writable staging copy it is filled from.  They are held together because
/// the upload must outlive the recorded `CopyBufferRegion` — i.e. it must survive until
/// `submit_and_wait()` returns, which owning it here guarantees.
struct StagedInput {
    /// UPLOAD heap.  `GENERIC_READ` for its entire life; **never** barriered.
    upload: GpuBuffer,
    /// DEFAULT heap.  Moves `COMMON` → `COPY_DEST` → `NON_PIXEL_SHADER_RESOURCE`.
    device: GpuBuffer,
    /// Bytes to copy from `upload` into `device`.
    bytes: u64,
}

impl StagedInput {
    /// Allocate both halves and memcpy `data` into the UPLOAD half.
    ///
    /// `bytes` comes from the plan (`MatMulPlan::a_bytes`, `ElementwisePlan::buffer_bytes`,
    /// …), and is cross-checked against `data` here.  A plan whose byte count disagrees
    /// with the buffer it is about to describe would size the DEFAULT allocation from one
    /// number and copy the other — a partially-initialised operand, and therefore a
    /// plausible-looking wrong answer rather than a crash.  It is a one-comparison guard
    /// against the whole class.
    ///
    /// # Errors
    /// [`DirectMLError::TransferError`] when `data` is not exactly `bytes` long.
    /// [`DirectMLError::Win32`] when either allocation or the `Map` fails.
    fn new(core: &D3d12Core, data: &[f32], bytes: u64) -> Result<Self> {
        let data_bytes = data
            .len()
            .checked_mul(ELEM_SIZE)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| {
                DirectMLError::TransferError(format!(
                    "operand of {} elements overflows a byte count",
                    data.len()
                ))
            })?;
        if data_bytes != bytes {
            return Err(DirectMLError::TransferError(format!(
                "operand of {} elements is {data_bytes} bytes, but the plan sized its \
                 buffer at {bytes} bytes",
                data.len()
            )));
        }
        Ok(Self {
            upload: GpuBuffer::upload_from_f32(core, data)?,
            device: GpuBuffer::new_default(core, bytes)?,
            bytes,
        })
    }
}

/// Everything one GPU submission needs, all of it already allocated and validated.
///
/// Assembled by `matmul` / `binary` / `unary`; consumed by [`HlslEngine::execute`],
/// which is the only function in this crate that records an HLSL barrier.
struct Workload<'a> {
    /// The shader's operands, in `t0`, `t1` order.
    ///
    /// **One entry for a unary op, two for everything else.**  With one entry, `t1` is
    /// bound to the *same* GPU address as `t0`: the D3D12 debug layer errors on an unset
    /// root parameter even when the bound shader never declares the register, and the
    /// unary sources declare no `t1`.  Binding the same buffer twice is the cheapest
    /// legal way to satisfy the root signature, and the shader never reads it.
    ///
    /// It is a slice of `StagedInput`, not two `&GpuBuffer`s pointing at one buffer,
    /// because [`GpuBuffer`] tracks its own resource state in a `Cell`: two handles to
    /// one resource would be two independent state trackers, and the second barrier
    /// would be emitted with a `StateBefore` that is no longer true.  That is silent
    /// corruption.  One resource, one tracker, bound twice.
    inputs: &'a [StagedInput],
    /// DEFAULT heap.  Moves `COMMON` → `UNORDERED_ACCESS` → `COPY_SOURCE`.
    output: &'a GpuBuffer,
    /// READBACK heap.  `COPY_DEST` for its entire life; **never** barriered.
    readback: &'a GpuBuffer,
    /// Bytes to copy from `output` into `readback`.
    output_bytes: u64,
    /// f32s to read back out of `readback` once the fence has been waited on.
    output_elems: usize,
    /// The compiled compute PSO for this kernel, from the cache — never compiled per node.
    pso: &'a ID3D12PipelineState,
    /// One entry per `Dispatch`, in order.  Never empty.
    dispatches: &'a [DispatchStep],
}

/// The D3D12 compute-shader engine.
pub(crate) struct HlslEngine {
    /// The shared root signature, plus the lazily-compiled PSO per [`ShaderKind`].
    ///
    /// Shaders are compiled **once, on first use**, and the compiled `ID3D12PipelineState`
    /// is reused for every subsequent node of that kind — `D3DCompile` is far too
    /// expensive to run per dispatch, and a transformer runs the same eight kernels
    /// thousands of times.
    psos: PsoCache,
}

impl HlslEngine {
    /// Build the shared root signature.  Shaders are compiled lazily, on first use.
    ///
    /// # Errors
    /// [`DirectMLError::Win32`] when the root signature cannot be serialised or created.
    pub(crate) fn new(core: &D3d12Core) -> Result<Self> {
        Ok(Self {
            psos: PsoCache::new(core)?,
        })
    }

    /// Batched MatMul / Gemm.
    ///
    /// 1. Validate the operand buffers against the plan.
    /// 2. CPU-transpose `A` and/or `B` when `Gemm` asked for it — the shader computes a
    ///    plain row-major product and has no transpose flags.  (The DirectML backend
    ///    folds this into `DML_GEMM_OPERATOR_DESC::TransA`/`TransB` and copies nothing.)
    /// 3. Build the dispatch program with [`matmul_program`] — one step per batch slice,
    ///    all sharing one grid, differing only in `AOff` / `BOff` / `COff`.  Batch
    ///    broadcasting needs **no** CPU work: it falls out of a zero batch stride, which
    ///    [`MatMulPlan::constants_for_slice`] turns into a zero offset.
    /// 4. Stage the buffers and run the canonical sequence ([`Self::execute`]).
    ///    **No UAV barrier is needed between slices** — each writes a disjoint `COff`
    ///    region of the output.  The one *before the readback copy* is emitted by
    ///    `execute`, and it is not optional.
    /// 5. Apply `alpha` and `beta·C` on the CPU with [`apply_gemm_epilogue`].
    ///
    /// # Errors
    /// [`DirectMLError::ShapeMismatch`] when `a` / `b` / `c` do not match the plan —
    /// i.e. the model is malformed.
    /// [`DirectMLError::Declined`] when the plan's grid exceeds D3D12's per-dimension
    /// limit; the router turns that into a correct CPU fallback.
    /// [`DirectMLError::ShaderCompile`] / [`DirectMLError::Win32`] /
    /// [`DirectMLError::TransferError`] on a genuine GPU failure — **never** silently
    /// swallowed here, because "the GPU broke" and "we declined" are different facts.
    pub(crate) fn matmul(
        &self,
        core: &D3d12Core,
        plan: &MatMulPlan,
        a: &[f32],
        b: &[f32],
        c: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        check_operand_len(a, &plan.a_stored_shape, "MatMul A")?;
        check_operand_len(b, &plan.b_stored_shape, "MatMul B")?;

        // `transpose_2d` is a no-op allocation we skip entirely when the flag is clear —
        // which is every `MatMul` and most `Gemm`s.  The `Cow`-shaped `Option` dance
        // keeps the owned buffer alive for exactly as long as the borrow needs it.
        let transposed_a = maybe_transpose(a, &plan.a_stored_shape, plan.trans_a, "A")?;
        let transposed_b = maybe_transpose(b, &plan.b_stored_shape, plan.trans_b, "B")?;
        let a_data: &[f32] = transposed_a.as_deref().unwrap_or(a);
        let b_data: &[f32] = transposed_b.as_deref().unwrap_or(b);

        let a_bytes = byte_count(plan.a_bytes()?, "MatMul A")?;
        let b_bytes = byte_count(plan.b_bytes()?, "MatMul B")?;
        let output_bytes = byte_count(plan.output_bytes()?, "MatMul output")?;
        let output_elems = plan.output_elems()?;

        // Everything fallible, before `begin()`.
        let dispatches = matmul_program(plan)?;
        let pso = self.psos.get(core, ShaderKind::MatMul)?;
        let inputs = [
            StagedInput::new(core, a_data, a_bytes)?,
            StagedInput::new(core, b_data, b_bytes)?,
        ];
        let output = GpuBuffer::new_default(core, output_bytes)?;
        let readback = GpuBuffer::new_readback(core, output_bytes)?;

        let mut out = self.execute(
            core,
            &Workload {
                inputs: &inputs,
                output: &output,
                readback: &readback,
                output_bytes,
                output_elems,
                pso: &pso,
                dispatches: &dispatches,
            },
        )?;

        // The shader computes the bare product; `alpha` and `beta·C` are the CPU
        // epilogue.  `MatMul` has `alpha == 1`, `beta == 0`, so this is a no-op for it.
        apply_gemm_epilogue(plan, &mut out, c)?;
        Ok(out)
    }

    /// Binary elementwise (`Add`, `Sub`, `Mul`, `Div`).
    ///
    /// The shaders are index-parallel — `C[i] = A[i] ⊕ B[i]`, with no notion of a shape
    /// at all — so the operands must already be dense and the same length.
    /// [`ElementwisePlan::binary`] guarantees exactly that today by declining every
    /// non-identical shape pair.  We nevertheless route both operands through
    /// [`broadcast_expand`], which
    ///
    /// * returns `Cow::Borrowed` — i.e. costs nothing, not even a memcpy — whenever the
    ///   shapes already match, which is *always*, today; and
    /// * validates each buffer's length against its shape, which is the check that stands
    ///   between a future relaxation of `ElementwisePlan::binary` and a kernel that reads
    ///   `B[0..24]` out of a 4-element buffer and returns a right-shaped tensor full of
    ///   whatever the allocator left there.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn binary(
        &self,
        core: &D3d12Core,
        plan: &ElementwisePlan,
        op: BinaryOp,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        let b_shape = plan.b_shape.as_ref().ok_or_else(|| {
            DirectMLError::ShapeMismatch(format!("{}: binary plan carries no B shape", op.as_str()))
        })?;
        let a_data = broadcast_expand(a, &plan.a_shape, &plan.output_shape)?;
        let b_data = broadcast_expand(b, b_shape, &plan.output_shape)?;

        let bytes = byte_count(plan.buffer_bytes()?, op.as_str())?;
        let output_elems = elem_count_usize(plan);

        let dispatches = elementwise_program(plan)?;
        let pso = self.psos.get(core, ShaderKind::for_binary(op))?;
        let inputs = [
            StagedInput::new(core, &a_data, bytes)?,
            StagedInput::new(core, &b_data, bytes)?,
        ];
        let output = GpuBuffer::new_default(core, bytes)?;
        let readback = GpuBuffer::new_readback(core, bytes)?;

        self.execute(
            core,
            &Workload {
                inputs: &inputs,
                output: &output,
                readback: &readback,
                output_bytes: bytes,
                output_elems,
                pso: &pso,
                dispatches: &dispatches,
            },
        )
    }

    /// Unary elementwise (`Relu`, `Sigmoid`, `Tanh`).
    ///
    /// Stages a **single** operand, which [`Self::execute`] binds to both `t0` and `t1`.
    /// See [`Workload::inputs`] for why that is a shared binding of one resource rather
    /// than two handles to it.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn unary(
        &self,
        core: &D3d12Core,
        plan: &ElementwisePlan,
        op: UnaryOp,
        a: &[f32],
    ) -> Result<Vec<f32>> {
        let a_data = broadcast_expand(a, &plan.a_shape, &plan.output_shape)?;

        let bytes = byte_count(plan.buffer_bytes()?, op.as_str())?;
        let output_elems = elem_count_usize(plan);

        let dispatches = elementwise_program(plan)?;
        let pso = self.psos.get(core, ShaderKind::for_unary(op))?;
        let inputs = [StagedInput::new(core, &a_data, bytes)?];
        let output = GpuBuffer::new_default(core, bytes)?;
        let readback = GpuBuffer::new_readback(core, bytes)?;

        self.execute(
            core,
            &Workload {
                inputs: &inputs,
                output: &output,
                readback: &readback,
                output_bytes: bytes,
                output_elems,
                pso: &pso,
                dispatches: &dispatches,
            },
        )
    }

    /// Numerically-stable single-axis Softmax.
    ///
    /// Stages a **single** operand, which [`Self::execute`] binds to both `t0` and
    /// `t1` — exactly as [`Self::unary`] does.  [`crate::hlsl::SOFTMAX_HLSL`] declares
    /// no `t1`, and the D3D12 debug layer errors on an unset root parameter; see
    /// [`Workload::inputs`] for why that is a shared binding of *one* resource rather
    /// than two independent handles to it.
    ///
    /// The shader subtracts the row max before every `exp`, and the oracle
    /// ([`crate::reference::ref_softmax`]) uses the same max-subtracted form, so a
    /// shadow-compare lands within the transcendental tolerance rather than diverging
    /// on any row that holds a large positive value.  Softmax is shape-preserving, so
    /// the input and output buffers are the same size.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn softmax(
        &self,
        core: &D3d12Core,
        plan: &SoftmaxPlan,
        a: &[f32],
    ) -> Result<Vec<f32>> {
        check_operand_len(a, plan.output_shape(), "Softmax")?;

        let bytes = byte_count(plan.buffer_bytes()?, "Softmax")?;
        let output_elems = plan.output_elems()?;

        // Everything fallible, before `begin()` — as in every other kernel here.
        let dispatches = softmax_program(plan)?;
        let pso = self.psos.get(core, ShaderKind::Softmax)?;
        let inputs = [StagedInput::new(core, a, bytes)?];
        let output = GpuBuffer::new_default(core, bytes)?;
        let readback = GpuBuffer::new_readback(core, bytes)?;

        self.execute(
            core,
            &Workload {
                inputs: &inputs,
                output: &output,
                readback: &readback,
                output_bytes: bytes,
                output_elems,
                pso: &pso,
                dispatches: &dispatches,
            },
        )
    }

    /// Single-axis Reduce (`Sum`, `Mean`, `Max`, `Min`).
    ///
    /// Like [`Self::softmax`] and [`Self::unary`], stages **one** operand bound to both
    /// `t0` and `t1`; [`crate::hlsl::REDUCE_HLSL`] declares no `t1`.  The entry point is
    /// selected from `plan.kind` with [`ShaderKind::for_reduce`] — the *only* per-op
    /// difference between the four reductions.
    ///
    /// This is the one kernel in this module whose **output buffer is smaller than its
    /// input**: the reduced axis collapses to a single element per `(outer, inner)`
    /// position, so the `output` and `readback` buffers are sized from
    /// [`ReducePlan::output_bytes`], never from the input.  `Sum` / `Mean` accumulate
    /// in the `k`-order the oracle ([`crate::reference::ref_reduce`]) walks; `Max` /
    /// `Min` merely select, and so reproduce it bit for bit.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn reduce(
        &self,
        core: &D3d12Core,
        plan: &ReducePlan,
        a: &[f32],
    ) -> Result<Vec<f32>> {
        check_operand_len(a, &plan.input_shape, plan.kind.as_str())?;

        let input_bytes = byte_count(plan.input_bytes()?, plan.kind.as_str())?;
        let output_bytes = byte_count(plan.output_bytes()?, plan.kind.as_str())?;
        let output_elems = plan.output_elems()?;

        let dispatches = reduce_program(plan)?;
        let pso = self.psos.get(core, ShaderKind::for_reduce(plan.kind))?;
        let inputs = [StagedInput::new(core, a, input_bytes)?];
        let output = GpuBuffer::new_default(core, output_bytes)?;
        let readback = GpuBuffer::new_readback(core, output_bytes)?;

        self.execute(
            core,
            &Workload {
                inputs: &inputs,
                output: &output,
                readback: &readback,
                output_bytes,
                output_elems,
                pso: &pso,
                dispatches: &dispatches,
            },
        )
    }

    /// **The canonical sequence.**  The only place in the HLSL path that records a
    /// barrier, a dispatch or a copy.
    ///
    /// See this module's header for the sequence, and for why the region between
    /// `begin()` and `submit_and_wait()` contains no `?`.
    ///
    /// # Errors
    /// [`DirectMLError::DispatchFailed`] when the workload is malformed (no operands, no
    /// dispatches) — a bug in this module, not in the caller's model, so it is *not* a
    /// `Declined`.
    /// [`DirectMLError::Win32`] when the submission, the fence wait or the readback `Map`
    /// fails.  A GPU that broke is reported as a failure, never as a decline.
    fn execute(&self, core: &D3d12Core, work: &Workload<'_>) -> Result<Vec<f32>> {
        // ── Preconditions.  Checked *before* `begin()`, so that the recorded region
        //    below stays infallible.
        let Some(first_input) = work.inputs.first() else {
            return Err(DirectMLError::DispatchFailed(
                "HLSL dispatch with no input operands".into(),
            ));
        };
        if work.inputs.len() > 2 {
            return Err(DirectMLError::DispatchFailed(format!(
                "HLSL root signature exposes t0 and t1 only, but {} operands were staged",
                work.inputs.len()
            )));
        }
        if work.dispatches.is_empty() {
            return Err(DirectMLError::DispatchFailed(
                "HLSL dispatch program is empty — the output buffer would be read back \
                 without ever being written"
                    .into(),
            ));
        }

        // `t0` and `t1` are root descriptors: raw GPU virtual addresses, with no size and
        // therefore no bounds checking.  Both buffers are committed resources bound at
        // offset 0, so their addresses satisfy every root-descriptor alignment rule
        // trivially.  A unary kernel stages one operand and binds it to both registers.
        let srv_a = first_input.device.gpu_address();
        let srv_b = work
            .inputs
            .get(1)
            .map_or(srv_a, |second| second.device.gpu_address());
        let uav_c = work.output.gpu_address();

        core.begin()?;
        let list: &ID3D12GraphicsCommandList = &core.list;

        // ── 1. Host → device.  DEFAULT buffers are born in COMMON; COMMON → COPY_DEST is
        //       a legal promotion, and `barrier_to` tracks the state so `StateBefore` is
        //       never a guess.  The UPLOAD buffers are pinned to GENERIC_READ by D3D12
        //       and are deliberately *not* barriered.
        for input in work.inputs {
            input
                .device
                .barrier_to(list, D3D12_RESOURCE_STATE_COPY_DEST);
            input
                .device
                .record_copy_from(list, &input.upload, input.bytes);
        }

        // ── 2. Device → shader.  NON_PIXEL_SHADER_RESOURCE is the SRV state a compute
        //       shader reads from; PIXEL_SHADER_RESOURCE would be wrong here.
        for input in work.inputs {
            input
                .device
                .barrier_to(list, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

        // ── 3. Output → writable.  COMMON → UNORDERED_ACCESS.
        work.output
            .barrier_to(list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // ── 4. Bind and dispatch.
        //
        // SAFETY: every call below is an `ID3D12GraphicsCommandList` recording method on
        // `core.list`, which `D3d12Core::begin()` has just `Reset` into the recording
        // state, and which is not being executed by the GPU (the previous
        // `submit_and_wait()` fence-waited before returning, and this `Backend` is
        // single-threaded behind `DirectMLContext`'s mutex).  They are `unsafe` only
        // because they are COM FFI; none of them can fail or return an `HRESULT`.
        //
        // * `SetComputeRootSignature` is issued *before* every root argument, because it
        //   invalidates all previously-set root arguments.
        // * `pso` and the root signature outlive this block: `pso` is owned by the caller
        //   (a refcounted clone out of the PSO cache) and the root signature is owned by
        //   `self.psos`.
        // * `psrcdata` points at `constants`, a `[u32; 8]` borrowed out of
        //   `work.dispatches`, which outlives this whole block — the array is not a
        //   temporary and cannot be dropped mid-call.  `SetComputeRoot32BitConstants`
        //   copies the eight words into the command list immediately, so the pointer need
        //   only be valid for the duration of the call; it is valid for far longer.
        // * `ROOT_CONSTANTS_U32 == 8` is exactly the `Num32BitValues` declared for root
        //   parameter 0 and exactly the width of the array, so the copy cannot overrun
        //   either side.
        // * The grid comes from `crate::plan`, which range-checked both dimensions against
        //   D3D12's 65 535-groups-per-dimension limit; `Dispatch` cannot be passed an
        //   out-of-range value from here.
        unsafe {
            list.SetPipelineState(work.pso);
            list.SetComputeRootSignature(self.psos.root().raw());
            list.SetComputeRootShaderResourceView(ROOT_PARAM_SRV_A, srv_a);
            list.SetComputeRootShaderResourceView(ROOT_PARAM_SRV_B, srv_b);
            list.SetComputeRootUnorderedAccessView(ROOT_PARAM_UAV_C, uav_c);

            for (constants, grid) in work.dispatches {
                list.SetComputeRoot32BitConstants(
                    ROOT_PARAM_CONSTANTS,
                    ROOT_CONSTANTS_U32,
                    constants.as_ptr().cast::<core::ffi::c_void>(),
                    0,
                );
                list.Dispatch(grid.x, grid.y, grid.z);
            }
        }

        // ── 5. THE UAV BARRIER.
        //
        // Without it, the `CopyBufferRegion` below is not ordered against the UAV writes
        // of the dispatch(es) above.  On most NVIDIA parts it happens to work anyway; on
        // AMD it reads a partially-written buffer and returns *plausible numbers that are
        // wrong*.  Nothing in this repository can catch that — there is no Windows host
        // and no D3D12 GPU here — so it is written once, here, and never conditionally.
        //
        // Between the per-slice dispatches themselves no UAV barrier is needed: each
        // slice writes a disjoint `COff` region of the output.
        work.output.record_uav_barrier(list);

        // ── 6. Device → host.  The READBACK buffer is pinned to COPY_DEST by D3D12 and is
        //       deliberately *not* barriered.
        work.output
            .barrier_to(list, D3D12_RESOURCE_STATE_COPY_SOURCE);
        work.readback
            .record_copy_from(list, work.output, work.output_bytes);

        // Closes the list, executes it, signals the fence and **waits**.  Reading the
        // readback buffer before this returns would race the copy.
        core.submit_and_wait()?;

        work.readback.read_f32(work.output_elems)
    }
}

// ─── pure, GPU-free planning helpers ─────────────────────────────────────────────
//
// Everything below is a pure function of `crate::plan`'s types.  No device, no COM, no
// `unsafe`.  They exist as separate functions precisely so that the dispatch *program* —
// the grid, and the root constants for every batch slice — can be asserted against
// without a GPU, which is the only part of this module that any test anywhere can reach.

/// The dispatch program for a MatMul / Gemm: one step per batch slice, all sharing the
/// grid from [`MatMulPlan::hlsl_grid`].
///
/// The grid is taken from the plan and **never** recomputed.  `plan.rs` pins its
/// orientation (`x = ceil(N/16)` counts columns, `y = ceil(M/16)` counts rows) with a
/// test named `hlsl_grid_is_not_transposed`, because the scaffold this crate grew out of
/// documented it backwards, and following that comment leaves part of the output matrix
/// unwritten on any non-square shape.
///
/// # Errors
/// [`DirectMLError::Declined`] when the grid exceeds D3D12's per-dimension limit, or
/// when a slice offset overflows `u32`.
fn matmul_program(plan: &MatMulPlan) -> Result<Vec<DispatchStep>> {
    let grid = plan.hlsl_grid()?;
    let mut steps = Vec::with_capacity(plan.batch as usize);
    for slice in 0..plan.batch {
        steps.push((plan.constants_for_slice(slice)?.to_root_constants(), grid));
    }
    if steps.is_empty() {
        return Err(DirectMLError::Declined(format!(
            "MatMul plan has batch = {}, so it would dispatch nothing",
            plan.batch
        )));
    }
    Ok(steps)
}

/// The dispatch program for an elementwise kernel: exactly one step.
///
/// `constants.groups_x` **must** equal the `x` actually passed to `Dispatch`, or the
/// shader's `i = (gid.y * GroupsX + gid.x) * 256 + lid.x` addresses the wrong elements —
/// silently.  Both numbers come from [`ElementwisePlan`], which derives them from a
/// single [`DispatchGrid::linear`] call; `elementwise_program_cannot_desync_groups_x`
/// below pins that they agree.
///
/// # Errors
/// [`DirectMLError::Declined`] when the element count needs more than `65535 * 65535`
/// thread groups.
fn elementwise_program(plan: &ElementwisePlan) -> Result<Vec<DispatchStep>> {
    let grid = plan.hlsl_grid()?;
    let constants = plan.constants()?;
    if constants.groups_x != grid.x {
        return Err(DirectMLError::DispatchFailed(format!(
            "GroupsX desync: root constant says {}, Dispatch says {}",
            constants.groups_x, grid.x
        )));
    }
    Ok(vec![(constants.to_root_constants(), grid)])
}

/// The dispatch program for a Softmax: exactly one step, one thread per softmax row.
///
/// Carries the same `GroupsX` self-check as [`elementwise_program`]: the root constant
/// `groups_x` **must** equal the `x` actually dispatched, or the shader's
/// `row = (gid.y · GroupsX + gid.x) · 256 + lid.x` addresses the wrong row for every
/// group past the first — silently.  Both numbers come from [`SoftmaxPlan`], which
/// derives them from a single [`DispatchGrid::linear`] call.
///
/// # Errors
/// [`DirectMLError::Declined`] when the row count exceeds D3D12's per-dimension limit.
fn softmax_program(plan: &SoftmaxPlan) -> Result<Vec<DispatchStep>> {
    let grid = plan.hlsl_grid()?;
    let constants = plan.constants()?;
    if constants.groups_x != grid.x {
        return Err(DirectMLError::DispatchFailed(format!(
            "Softmax GroupsX desync: root constant says {}, Dispatch says {}",
            constants.groups_x, grid.x
        )));
    }
    Ok(vec![(constants.to_root_constants(), grid)])
}

/// The dispatch program for a Reduce: exactly one step, one thread per output element.
///
/// Same `GroupsX` self-check as [`softmax_program`]; the guarded count here is
/// `out_count` — the *output* length — because one thread writes one output element,
/// not one input element.
///
/// # Errors
/// [`DirectMLError::Declined`] when the output count exceeds D3D12's per-dimension
/// limit.
fn reduce_program(plan: &ReducePlan) -> Result<Vec<DispatchStep>> {
    let grid = plan.hlsl_grid()?;
    let constants = plan.constants()?;
    if constants.groups_x != grid.x {
        return Err(DirectMLError::DispatchFailed(format!(
            "Reduce GroupsX desync: root constant says {}, Dispatch says {}",
            constants.groups_x, grid.x
        )));
    }
    Ok(vec![(constants.to_root_constants(), grid)])
}

/// Transpose `data` when `flag` is set, so the shader — which has no transpose flags —
/// sees a plain row-major operand.
///
/// Returns `None` when no transpose is needed, so the caller keeps borrowing the
/// original slice and nothing is allocated.  Only `Gemm` ever sets the flag, and its
/// transposed operand is almost always a constant weight.
///
/// # Errors
/// [`DirectMLError::Declined`] when the operand is not 2-D — [`MatMulPlan`] guarantees
/// it is, so this is a defensive decline rather than a reachable path.
/// [`DirectMLError::ShapeMismatch`] when `data` does not match `shape`.
fn maybe_transpose(
    data: &[f32],
    shape: &[usize],
    flag: bool,
    what: &str,
) -> Result<Option<Vec<f32>>> {
    if !flag {
        return Ok(None);
    }
    let [rows, cols] = shape else {
        return Err(DirectMLError::Declined(format!(
            "Gemm: cannot transpose a non-2-D {what} operand of shape {shape:?}"
        )));
    };
    transpose_2d(data, *rows, *cols).map(Some)
}

/// A buffer's length must match the shape the plan says it has.
///
/// The shader indexes with root-constant offsets and has no bounds check of its own; a
/// root SRV carries no size, so an operand shorter than its shape reads past the end of
/// its allocation and returns a right-shaped tensor of wrong numbers.  This is the only
/// thing standing between that and a `Tensor` whose `data` and `shape` were never
/// checked against each other.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when they disagree.
fn check_operand_len(data: &[f32], shape: &[usize], what: &str) -> Result<()> {
    let expected = numel(shape)?;
    if data.len() != expected {
        return Err(DirectMLError::ShapeMismatch(format!(
            "{what}: buffer of {} elements does not match shape {shape:?} \
             ({expected} elements)",
            data.len()
        )));
    }
    Ok(())
}

/// A plan's `usize` byte count as the `u64` D3D12 wants.
///
/// # Errors
/// [`DirectMLError::Declined`] when it does not fit — unreachable on any 64-bit target,
/// checked rather than assumed.
fn byte_count(bytes: usize, what: &str) -> Result<u64> {
    u64::try_from(bytes)
        .map_err(|_| DirectMLError::Declined(format!("{what}: {bytes} bytes overflows u64")))
}

/// An [`ElementwisePlan`]'s `elem_count` as the `usize` [`GpuBuffer::read_f32`] wants.
///
/// A *widening* conversion of a value [`crate::plan`] has already range-checked — not a
/// narrowing `as u32` on a shape, which this crate forbids.  It is exact on every target
/// this crate supports.
fn elem_count_usize(plan: &ElementwisePlan) -> usize {
    plan.elem_count as usize
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::plan::{
        ReduceKind, ELEMENTWISE_THREADS_PER_GROUP, MATMUL_TILE, REDUCTION_THREADS_PER_GROUP,
    };

    // ── What these tests can and cannot do ──────────────────────────────────────
    //
    // This module is `#[cfg(target_os = "windows")]`, so these tests do **not** run in
    // this repository's Linux CI — there, they are only *type-checked*, via
    // `cargo clippy --target x86_64-pc-windows-gnu --all-targets`.
    //
    // They do, however, run for real on any Windows box, **including one with no GPU at
    // all**, because every one of them exercises only the pure planning helpers above and
    // never touches a D3D12 device.  That is deliberate: the dispatch *program* — the
    // grid orientation, the per-slice offsets, the GroupsX agreement — is the part of
    // this module whose failure mode is a silently wrong answer, and it is the part that
    // needs no hardware to check.
    //
    // The barrier sequence, the root-descriptor bindings and the shader itself remain
    // unverifiable without a real D3D12 GPU.  `DirectMLContext::self_check` is the only
    // thing that can validate those.

    // ── the transposition trap ──────────────────────────────────────────────────

    #[test]
    fn matmul_program_takes_the_grid_from_the_plan_and_does_not_transpose_it() {
        // 32 x 48 output: 3 groups across the 48 columns (X), 2 down the 32 rows (Y).
        // Swapping them dispatches 2 groups over 48 columns — 32 of them never written.
        let plan = MatMulPlan::matmul(&[32, 16], &[16, 48]).expect("valid 2-D matmul");
        let program = matmul_program(&plan).expect("within the D3D12 limits");

        assert_eq!(program.len(), 1, "batch is 1 in this wave");
        let (_, grid) = program[0];
        assert_eq!(
            grid,
            DispatchGrid { x: 3, y: 2, z: 1 },
            "X must count COLUMNS (ceil(48/16) = 3) and Y must count ROWS (ceil(32/16) = 2)"
        );
        assert_eq!(grid, plan.hlsl_grid().unwrap(), "must come from the plan");
    }

    #[test]
    fn matmul_grid_covers_every_output_element() {
        // Deliberately routed through `matmul_program`, not through `plan.hlsl_grid()`:
        // the grid the GPU is actually dispatched over is the one *this module* emits,
        // and asserting on the plan's would leave a transposition introduced here
        // invisible.  The asymmetric shapes are the point — a transposed grid still
        // covers a square matrix.
        for (m, n) in [(1_usize, 1_usize), (17, 3), (3, 17), (256, 64), (1, 1000)] {
            let plan = MatMulPlan::matmul(&[m, 8], &[8, n]).expect("valid");
            let (_, grid) = matmul_program(&plan).expect("within limits")[0];
            let covered_cols = u64::from(grid.x) * u64::from(MATMUL_TILE);
            let covered_rows = u64::from(grid.y) * u64::from(MATMUL_TILE);
            assert!(
                covered_rows >= m as u64,
                "{m}x{n}: Y covers {covered_rows} rows, need {m} — the grid is transposed"
            );
            assert!(
                covered_cols >= n as u64,
                "{m}x{n}: X covers {covered_cols} cols, need {n} — the grid is transposed"
            );
        }
    }

    // ── root constants ──────────────────────────────────────────────────────────

    #[test]
    fn matmul_program_root_constants_are_in_cbuffer_order() {
        let plan = MatMulPlan::matmul(&[4, 3], &[3, 5]).expect("valid");
        let program = matmul_program(&plan).expect("valid");
        let (constants, _) = program[0];
        // MATMUL_HLSL's cbuffer is `M, K, N, AOff, BOff, COff, _p0, _p1`.
        assert_eq!(constants[0], 4, "M");
        assert_eq!(constants[1], 3, "K");
        assert_eq!(constants[2], 5, "N");
        assert_eq!(constants[3], 0, "AOff — slice 0");
        assert_eq!(constants[4], 0, "BOff — slice 0");
        assert_eq!(constants[5], 0, "COff — slice 0");
        assert_eq!(
            constants.len(),
            ROOT_CONSTANT_COUNT,
            "the root signature declares exactly this many"
        );
        assert_eq!(ROOT_CONSTANTS_U32 as usize, ROOT_CONSTANT_COUNT);
    }

    #[test]
    fn matmul_program_emits_one_dispatch_per_batch_slice_with_disjoint_offsets() {
        // `MatMulPlan::matmul` is 2-D-only today, so `batch == 1`.  Build the batched
        // case by hand: the *program builder* is batch-general, and must stay that way,
        // or lifting the 2-D restriction silently computes slice 0 `batch` times.
        let mut plan = MatMulPlan::matmul(&[4, 3], &[3, 5]).expect("valid");
        plan.batch = 3;
        plan.batch_shape = vec![3];
        plan.output_shape = vec![3, 4, 5];
        plan.a_batch_stride = 12; // 4 x 3
        plan.b_batch_stride = 15; // 3 x 5

        let program = matmul_program(&plan).expect("valid");
        assert_eq!(program.len(), 3, "one dispatch per batch slice");

        let mut seen_c_offsets = Vec::new();
        for (slice, (constants, grid)) in program.iter().enumerate() {
            let s = u32::try_from(slice).unwrap();
            assert_eq!(constants[3], s * 12, "AOff");
            assert_eq!(constants[4], s * 15, "BOff");
            assert_eq!(constants[5], s * 20, "COff = slice * M * N");
            assert_eq!(
                *grid,
                plan.hlsl_grid().unwrap(),
                "every slice shares one grid"
            );
            seen_c_offsets.push(constants[5]);
        }
        // Disjoint output regions — which is exactly why no UAV barrier is needed
        // *between* the slices.
        assert_eq!(seen_c_offsets, vec![0, 20, 40]);
    }

    #[test]
    fn a_zero_batch_stride_is_the_whole_batch_broadcast_implementation() {
        let mut plan = MatMulPlan::matmul(&[4, 3], &[3, 5]).expect("valid");
        plan.batch = 4;
        plan.a_batch_stride = 0; // A is broadcast across the batch
        plan.b_batch_stride = 15;

        let program = matmul_program(&plan).expect("valid");
        for (constants, _) in &program {
            assert_eq!(constants[3], 0, "a zero stride must keep AOff at 0");
        }
        assert_eq!(program[3].0[4], 45, "B still advances");
    }

    // ── hazard 10: GroupsX desync ───────────────────────────────────────────────

    #[test]
    fn elementwise_program_cannot_desync_groups_x() {
        // The shader recovers `i = (gid.y * GroupsX + gid.x) * 256 + lid.x`.  If the root
        // constant and the dispatched `x` ever disagree, every thread past the first row
        // of groups reads the wrong element — with no crash and no bounds violation.
        for elems in [1_usize, 255, 256, 257, 65_535, 16_776_960, 16_777_216] {
            let plan = ElementwisePlan::unary(&[elems]).expect("non-empty");
            let program = elementwise_program(&plan).expect("within limits");
            assert_eq!(program.len(), 1, "elementwise is a single dispatch");
            let (constants, grid) = program[0];
            assert_eq!(constants[0], plan.elem_count, "N");
            assert_eq!(
                constants[1], grid.x,
                "GroupsX ({}) must equal the dispatched x ({}) for {elems} elements",
                constants[1], grid.x
            );
        }
    }

    #[test]
    fn elementwise_grid_covers_every_element() {
        for elems in [1_usize, 255, 256, 257, 100_000, 16_777_216] {
            let plan = ElementwisePlan::unary(&[elems]).expect("non-empty");
            let (_, grid) = elementwise_program(&plan).expect("within limits")[0];
            let threads = grid.total_groups() * u64::from(ELEMENTWISE_THREADS_PER_GROUP);
            assert!(
                threads >= elems as u64,
                "{elems} elements need at least that many threads, got {threads}"
            );
        }
    }

    #[test]
    fn elementwise_grid_folds_past_the_one_dimensional_cliff() {
        // 65_535 groups x 256 threads = 16_776_960 elements is the last 1-D grid.  One
        // more element must fold onto Y rather than exceed the per-dimension limit.
        let plan = ElementwisePlan::unary(&[16_776_961]).expect("non-empty");
        let (constants, grid) = elementwise_program(&plan).expect("within limits")[0];
        assert_eq!(grid.x, DispatchGrid::MAX_GROUPS_PER_DIM);
        assert!(grid.y > 1, "must fold onto Y");
        assert_eq!(constants[1], grid.x, "GroupsX still tracks x");
    }

    // ── transpose ───────────────────────────────────────────────────────────────

    #[test]
    fn maybe_transpose_is_a_no_op_when_the_flag_is_clear() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        assert!(maybe_transpose(&a, &[2, 2], false, "A")
            .expect("no-op")
            .is_none());
    }

    #[test]
    fn maybe_transpose_produces_the_row_major_operand_the_shader_reads() {
        // Gemm with transB: B is stored [n, k] = [3, 2] and the shader wants [k, n] = [2, 3].
        let b_stored = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2],[3,4],[5,6]]
        let out = maybe_transpose(&b_stored, &[3, 2], true, "B")
            .expect("2-D")
            .expect("transposed");
        assert_eq!(out, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]); // [[1,3,5],[2,4,6]]

        // And that is exactly the shape `MatMulPlan::gemm` says the logical operand has.
        let plan = MatMulPlan::gemm(&[4, 2], &[3, 2], None, 1.0, 0.0, false, true).expect("valid");
        assert_eq!((plan.m, plan.k, plan.n), (4, 2, 3));
    }

    #[test]
    fn maybe_transpose_declines_a_non_2d_operand_rather_than_indexing_past_the_end() {
        let a = [1.0_f32, 2.0];
        let e = maybe_transpose(&a, &[2], true, "A").expect_err("1-D cannot be transposed");
        assert!(matches!(e, DirectMLError::Declined(_)), "got {e:?}");
    }

    // ── operand validation ──────────────────────────────────────────────────────

    #[test]
    fn a_short_operand_is_a_shape_mismatch_not_a_silent_out_of_bounds_read() {
        let a = [1.0_f32, 2.0, 3.0]; // claims to be 2x3
        let e = check_operand_len(&a, &[2, 3], "MatMul A").expect_err("3 != 6");
        assert!(matches!(e, DirectMLError::ShapeMismatch(_)), "got {e:?}");

        check_operand_len(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], "MatMul A")
            .expect("6 elements is 2x3");
    }

    #[test]
    fn byte_count_round_trips_a_plan_size() {
        let plan = MatMulPlan::matmul(&[4, 3], &[3, 5]).expect("valid");
        assert_eq!(byte_count(plan.a_bytes().unwrap(), "A").unwrap(), 48);
        assert_eq!(byte_count(plan.b_bytes().unwrap(), "B").unwrap(), 60);
        assert_eq!(byte_count(plan.output_bytes().unwrap(), "out").unwrap(), 80);
    }

    #[test]
    fn output_shape_elems_matches_the_plan() {
        let plan = ElementwisePlan::binary(&[2, 3, 4], &[2, 3, 4]).expect("identical shapes");
        assert_eq!(elem_count_usize(&plan), 24);
        assert_eq!(plan.buffer_bytes().unwrap(), 96);
    }

    // ── softmax / reduce dispatch programs ──────────────────────────────────────
    //
    // Same class of pure, GPU-free assertions as the elementwise family above: the
    // failure mode of a transposed constant or a desynced `GroupsX` is a *silently*
    // wrong answer, and it is checkable with no device.

    #[test]
    fn softmax_program_is_one_dispatch_with_constants_in_cbuffer_order() {
        // SOFTMAX_HLSL's cbuffer is `Rows, GroupsX, AxisLen, Inner, _p0 … _p3`; a
        // transposition here dispatches the right grid over the wrong row geometry.
        for (shape, axis) in [
            (&[2_usize, 3, 4][..], 2_i64),
            (&[5, 7][..], -1),
            (&[8, 9][..], 0),
            (&[6, 5, 4][..], 1),
        ] {
            let plan = SoftmaxPlan::softmax(shape, axis).expect("valid softmax");
            let program = softmax_program(&plan).expect("within the D3D12 limits");
            assert_eq!(program.len(), 1, "softmax is a single dispatch");
            let (constants, grid) = program[0];
            assert_eq!(constants[0], plan.rows, "Rows");
            assert_eq!(
                constants[1], grid.x,
                "GroupsX ({}) must equal the dispatched x ({}) for {shape:?}@{axis}",
                constants[1], grid.x
            );
            assert_eq!(constants[2], plan.axis_len, "AxisLen");
            assert_eq!(constants[3], plan.inner, "Inner");
        }
    }

    #[test]
    fn softmax_grid_covers_every_row() {
        // One thread owns one softmax row; a short grid leaves the tail rows unwritten
        // — a right-shaped output tensor full of allocator garbage on those rows.
        for (shape, axis) in [
            (&[2_usize, 3, 4][..], 1_i64),
            (&[1000, 4][..], 0),
            (&[7, 7, 7][..], 2),
            (&[65_537, 2][..], 1),
        ] {
            let plan = SoftmaxPlan::softmax(shape, axis).expect("valid");
            let (_, grid) = softmax_program(&plan).expect("within limits")[0];
            let threads = grid.total_groups() * u64::from(REDUCTION_THREADS_PER_GROUP);
            assert!(
                threads >= u64::from(plan.rows),
                "{shape:?}@{axis}: {} rows need at least that many threads, got {threads}",
                plan.rows
            );
        }
    }

    #[test]
    fn reduce_program_is_one_dispatch_per_kind_with_constants_in_cbuffer_order() {
        // REDUCE_HLSL's cbuffer is `N, GroupsX, AxisLen, Inner, _p0 … _p3`, and all four
        // kinds share that layout — only the entry point differs, which this program
        // does not encode (the caller selects it via `ShaderKind::for_reduce`).
        for kind in [
            ReduceKind::Sum,
            ReduceKind::Mean,
            ReduceKind::Max,
            ReduceKind::Min,
        ] {
            let plan = ReducePlan::reduce(kind, &[2, 3, 4], &[1], false).expect("valid reduce");
            let program = reduce_program(&plan).expect("within limits");
            assert_eq!(program.len(), 1, "reduce is a single dispatch");
            let (constants, grid) = program[0];
            assert_eq!(constants[0], plan.out_count, "N == out_count");
            assert_eq!(
                constants[1],
                grid.x,
                "GroupsX ({}) must equal the dispatched x ({}) for {}",
                constants[1],
                grid.x,
                kind.as_str()
            );
            assert_eq!(constants[2], plan.axis_len, "AxisLen");
            assert_eq!(constants[3], plan.inner, "Inner");
        }
    }

    #[test]
    fn reduce_sizes_its_output_buffer_from_the_output_not_the_input() {
        // The only kernel in this module whose output is smaller than its input.  Sizing
        // `output` / `readback` from the *input* would read back 24 elements out of an
        // 8-element buffer — a right-shaped tensor of whatever the allocator left behind.
        let plan = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[1], false).expect("valid");
        assert_eq!(plan.input_bytes().unwrap(), 24 * 4, "2*3*4 f32 in");
        assert_eq!(
            plan.output_bytes().unwrap(),
            8 * 4,
            "axis 1 collapses: 2*4 f32 out"
        );
        assert!(
            plan.output_bytes().unwrap() < plan.input_bytes().unwrap(),
            "a reduce output must be smaller than its input"
        );
        assert_eq!(plan.output_elems().unwrap(), 8);
    }

    #[test]
    fn reduce_grid_covers_every_output_element() {
        for (shape, axis) in [(&[300_usize, 128][..], 1_i64), (&[4, 65_540][..], 0)] {
            let plan = ReducePlan::reduce(ReduceKind::Sum, shape, &[axis], false).expect("valid");
            let (_, grid) = reduce_program(&plan).expect("within limits")[0];
            let threads = grid.total_groups() * u64::from(REDUCTION_THREADS_PER_GROUP);
            assert!(
                threads >= u64::from(plan.out_count),
                "{shape:?}@{axis}: {} outputs need at least that many threads, got {threads}",
                plan.out_count
            );
        }
    }
}
