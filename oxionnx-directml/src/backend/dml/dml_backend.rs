//! The genuine DirectML engine: compiled-operator cache + the two-submission init dance.
//!
//! # The two-submission rule — this is *not* an optimisation oversight
//!
//! On a cache miss, an operator must be **compiled**, then **initialised**, then
//! **executed**, and the initialisation must be *submitted and waited on* before the
//! execution is even recorded:
//!
//! 1. `compile_*` → `IDMLCompiledOperator`.
//! 2. `CreateOperatorInitializer(Some(&[Some(compiled.clone())]))`.
//! 3. `init_props = initializer.GetBindingProperties()`;
//!    `exec_props = compiled.GetBindingProperties()`.
//!    Descriptor heap capacity = **max** of the two `RequiredDescriptorCount`s.
//!    Temporary bytes at init = `init_props.TemporaryResourceSize`; at execute =
//!    `exec_props.TemporaryResourceSize`.  **They differ**, and sizing a temporary
//!    buffer from the wrong one overruns it.
//! 4. Persistent buffer of `exec_props.PersistentResourceSize`, allocated **once**, kept
//!    in the cache entry, and never reallocated — DirectML stores per-operator compiled
//!    state in it.
//! 5. `BindingTable` over the *initializer*; bind the initializer's temporary, and bind
//!    the persistent buffer as the initializer's **output**; `RecordDispatch`;
//!    **`core.submit_and_wait()`**.
//!
//! That `submit_and_wait` is **mandatory**.  `IDMLBindingTable::Reset` rewrites the
//! descriptors in the shader-visible heap from the **CPU, immediately**.  Recording the
//! initializer dispatch and the operator dispatch into one command list and calling
//! `Reset` between them overwrites the very descriptors the *initializer's* dispatch will
//! read when the GPU eventually reaches it.  The initializer then writes nothing useful
//! into the persistent resource, and the operator reads uninitialised memory out of it.
//!
//! **The failure mode is a plausible-looking wrong answer, not a crash, not an
//! `HRESULT`.**  Nothing in this repository can catch it.  Microsoft's own
//! `HelloDirectML` sample waits here for exactly this reason.
//!
//! [`OpCacheKey`] amortises the second submission away for every repeat shape — which is
//! every node in a transformer.  The *first* dispatch of a given shape costs two
//! submissions; every subsequent one costs a single submission.
//!
//! # The execute half — and the barriers, in order
//!
//! ```text
//! (CPU) allocate execute-time temporary; build DML_BUFFER_BINDINGs
//! (CPU) table.reset(heap, compiled)  ← rewrites heap descriptors IMMEDIATELY
//! (CPU) bind_inputs / bind_outputs / bind_temporary / bind_persistent
//!
//! core.begin()                                       ← allocator + list Reset
//!   for each input:
//!     input.device  COMMON        → COPY_DEST         (transition)
//!     CopyBufferRegion(input.device ← input.staging)
//!     input.device  COPY_DEST     → UNORDERED_ACCESS  (transition)
//!   output.device   COMMON        → UNORDERED_ACCESS  (transition)
//!   temporary       COMMON        → UNORDERED_ACCESS  (transition)
//!   persistent      (already UAV) → UNORDERED_ACCESS  (records nothing; see below)
//!
//!   list.SetDescriptorHeaps(&[heap])                 ← MUST precede RecordDispatch
//!   recorder.RecordDispatch(list, compiled, table)
//!
//!   output.device   UAV barrier                       ← THE ONE EVERYONE OMITS
//!   output.device   UNORDERED_ACCESS → COPY_SOURCE    (transition)
//!   CopyBufferRegion(readback ← output.device)
//! core.submit_and_wait()
//!   readback.read_f32()
//! ```
//!
//! Four things in that sequence are load-bearing and easy to get wrong:
//!
//! * **DirectML binds its *inputs* as UAVs too**, unlike the HLSL path, which binds them
//!   as SRVs.  The post-copy state of an input is therefore `UNORDERED_ACCESS`, **not**
//!   `NON_PIXEL_SHADER_RESOURCE`.  A transition barrier whose `StateBefore` disagrees
//!   with reality is corruption, so this must match what DirectML actually does.
//! * **UPLOAD buffers stay in `GENERIC_READ` forever and READBACK buffers stay in
//!   `COPY_DEST` forever.**  They are never barriered, anywhere in this file.  Only
//!   DEFAULT-heap buffers move.
//! * **The UAV barrier before the readback copy** is the one everyone omits.  A
//!   `UNORDERED_ACCESS → COPY_SOURCE` transition is a *state* change; it is not, on every
//!   IHV, a guarantee that the dispatch's UAV writes have landed.  Omitting the UAV
//!   barrier is correct-by-luck on NVIDIA and garbage on AMD — precisely the bug that
//!   cannot be caught from a machine with no GPU.
//! * **`SetDescriptorHeaps` must be called on the command list before `RecordDispatch`.**
//!   Omitting it is a device removal, not a wrong answer — which, for once, is the
//!   friendly failure mode.
//!
//! # Why the persistent buffer needs no barrier after the first submission
//!
//! [`GpuBuffer`] tracks its own state in a `Cell`, and `compiled_for` already transitioned
//! the persistent buffer `COMMON → UNORDERED_ACCESS`, so `barrier_to(UNORDERED_ACCESS)` in
//! `execute` records nothing.  That is still correct, and deliberately so: D3D12 **decays**
//! a buffer resource back to `COMMON` when the `ExecuteCommandLists` that last touched it
//! retires, and **implicitly promotes** a buffer in `COMMON` to `UNORDERED_ACCESS` on its
//! first UAV access in the next list.  So the real state at the next dispatch is
//! `COMMON → (implicit) UNORDERED_ACCESS`, which is exactly what DirectML needs.  The call
//! is kept — rather than dropped — because it is also correct in the hypothetical where
//! the tracker says `COMMON`: it would then emit an explicit, legal `COMMON → UAV`
//! transition.  Either way DirectML sees a UAV; neither way emits an *illegal* barrier.

use core::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
};
use windows::Win32::AI::MachineLearning::DirectML::{
    IDMLCompiledOperator, IDMLDevice, IDMLOperatorInitializer, DML_BINDING_DESC,
    DML_BINDING_TYPE_NONE,
};

use crate::backend::d3d12::buffer::GpuBuffer;
use crate::backend::d3d12::device::{D3d12Core, DescriptorHeap};
use crate::backend::dml::binding::{BindingTable, BufferBindings};
use crate::backend::dml::device::DmlDevice;
use crate::backend::dml::op::{
    compile_binary, compile_conv, compile_gemm, compile_reduce, compile_softmax, compile_unary,
};
use crate::error::{DirectMLError, HrExt, Result};
use crate::layout::{
    dml_conv_bias_layout, dml_reduce_layouts, DmlElementwiseLayout, DmlGemmLayout, DmlTensorLayout,
    OpCacheKey,
};
use crate::plan::{
    align_up, BinaryOp, ConvPlan, ElementwisePlan, MatMulPlan, ReducePlan, SoftmaxPlan, UnaryOp,
    DML_BUFFER_ALIGNMENT, ELEM_SIZE,
};

/// `DML_OPERATOR_GEMM`'s input-tensor count: `ATensor`, `BTensor`, `CTensor`.
///
/// # A bias-free GEMM binds THREE inputs, the third being `DML_BINDING_TYPE_NONE`
///
/// `CTensor` is optional, and the temptation is to bind only `[A, B]` when there is no
/// bias.  This binds `[A, B, NONE]` instead, deliberately.  The reasoning, in full,
/// because `IDMLBindingTable::BindInputs` **returns `void`** — a wrong binding count
/// raises no `HRESULT` and cannot fail loudly, so this cannot be settled by trying it:
///
/// * DirectML documents `bindingCount` as "the number of input tensors of the operator",
///   and `DML_BINDING_TYPE_NONE` exists for exactly one purpose — "binding optional
///   tensors that you don't want to supply".  That purpose is meaningless if an
///   unsupplied optional input could simply be *omitted* from the array.
/// * ONNX Runtime's DirectML EP binds one desc per operator input slot, using a NONE desc
///   for every absent optional input.  It is the largest DirectML consumer in existence.
/// * The two candidate models of `BindInputs` are (a) "the table has one slot per operator
///   input tensor, every slot defaults to NONE, and `BindInputs` overwrites the first
///   `bindingCount` of them", and (b) "`bindingCount` must equal the input-tensor count".
///   Model (a) is what `HelloDirectML` relies on when it never calls `BindInputs` on the
///   operator initializer at all.  `[A, B, NONE]` is **correct under both models**;
///   `[A, B]` is correct only under (a).  Three strictly dominates two.
///
/// If a hardware run ever shows DirectML rejecting a 3-binding bias-free GEMM, the fix is
/// to make [`DmlEngine::matmul`] pass a 2-element array when `layout.c` is `None` — but do
/// not make that change on a hunch, because both variants "work" on a machine with the
/// debug layer off, right up until they don't.
///
/// (Note for reviewers: `binding.rs`'s `bind_inputs` doc describes a bias-free GEMM as
/// `[A, B]`.  That is a comment, not a constraint — `bind_inputs` takes a slice of any
/// length — and this is the deliberate, argued departure from it.)
const GEMM_INPUT_SLOTS: usize = 3;

/// `DML_OPERATOR_ELEMENT_WISE_{ADD,SUBTRACT,MULTIPLY,DIVIDE}`: `ATensor`, `BTensor`.
const BINARY_INPUT_SLOTS: usize = 2;

/// `DML_OPERATOR_ACTIVATION_{RELU,SIGMOID,TANH}`: `InputTensor`.
const UNARY_INPUT_SLOTS: usize = 1;

/// `DML_OPERATOR_ACTIVATION_SOFTMAX`: `InputTensor`.
const SOFTMAX_INPUT_SLOTS: usize = 1;

/// `DML_OPERATOR_REDUCE`: `InputTensor`.
const REDUCE_INPUT_SLOTS: usize = 1;

/// `DML_OPERATOR_CONVOLUTION`: `InputTensor`, `FilterTensor`, and the **optional**
/// `BiasTensor`.
///
/// Like a bias-free GEMM, a bias-free conv still presents **three** input bindings — the
/// third a `DML_BINDING_TYPE_NONE`, not a shorter array — for the reasons argued on
/// [`GEMM_INPUT_SLOTS`].
const CONV_INPUT_SLOTS: usize = 3;

/// The genuine DirectML engine.
///
/// Every field is `!Send`/`!Sync` COM or `RefCell` interior mutability.  That is sound
/// only because the whole `Backend` sits behind `DirectMLContext`'s `Mutex`, and every
/// method here holds that lock across the entire record → submit → fence-wait sequence.
pub(crate) struct DmlEngine {
    /// `IDMLDevice` + the `IDMLCommandRecorder` reused across every dispatch.
    dml: DmlDevice,
    /// Compiled **and already-initialised** operators, keyed by everything that affects
    /// the compiled binary.  `RefCell`, not `Mutex`: see the struct docs.
    cache: RefCell<HashMap<OpCacheKey, Rc<CachedOp>>>,
}

/// A compiled **and already-initialised** DirectML operator, plus the descriptor heap and
/// binding table it dispatches through.
///
/// # Why the heap and the table live here
///
/// The design sketch created both per dispatch.  They are cached per *operator* instead,
/// because `CreateDescriptorHeap` is not cheap and both objects are a pure function of the
/// operator: the heap's capacity and the table's `SizeInDescriptors` are fixed by
/// [`Self::descriptor_count`], which is fixed at compile time.  Reuse is safe because
/// every dispatch in this file ends in `submit_and_wait`, so no previous dispatch can
/// still be reading the descriptors that `BindingTable::reset` is about to rewrite.
struct CachedOp {
    /// The compiled operator.  Dispatched via `IDMLCommandRecorder::RecordDispatch`.
    compiled: IDMLCompiledOperator,
    /// The persistent resource DirectML asked for at compile time, **already written by
    /// the operator initializer** in [`DmlEngine::compiled_for`].  `None` when
    /// `PersistentResourceSize == 0`.
    ///
    /// # Lifetime
    ///
    /// Allocated once, before the initializer dispatch that fills it; bound as that
    /// dispatch's output; bound as `BindPersistentResource` on **every** subsequent
    /// execute; dropped only when this `CachedOp` is dropped, which happens only when the
    /// whole [`DmlEngine`] is dropped (nothing evicts the cache).  Every GPU submission
    /// that references it is fully awaited before the function that created the reference
    /// returns, so the buffer strictly outlives every GPU access to it.  It is the **same
    /// buffer** in the initialize pass and in every execute pass — which is the entire
    /// point: a *different* buffer at execute time reads as uninitialised weights.
    persistent: Option<GpuBuffer>,
    /// `DML_BUFFER_BINDING::SizeInBytes` for [`Self::persistent`]; `0` when there is none.
    persistent_bytes: u64,
    /// `TemporaryResourceSize` needed at **execute** time.  This is *not* the
    /// initializer's temporary size; the two differ, and the initializer's is used once,
    /// in `compiled_for`, and then thrown away.
    temporary_bytes: u64,
    /// `max(initializer, compiled)` `RequiredDescriptorCount`, floored at 1.
    ///
    /// Used for **both** the heap's capacity and the binding table's `SizeInDescriptors`.
    /// `SizeInDescriptors` must be at least the dispatchable's `RequiredDescriptorCount`
    /// and at most the heap's capacity; taking the max of the two counts and using the
    /// same number for both satisfies that for the initializer *and* the operator, and
    /// cannot be too small for either.
    descriptor_count: u32,
    /// The shader-visible CBV/SRV/UAV heap the binding table writes descriptors into.
    /// Must be bound with `SetDescriptorHeaps` before every `RecordDispatch`.
    heap: DescriptorHeap,
    /// Created over the *initializer*, then `reset` onto [`Self::compiled`] on every
    /// execute.
    table: BindingTable,
}

/// One input operand, staged for one DirectML dispatch.
struct StagedInput {
    /// UPLOAD-heap copy of the host data.  Lives in `GENERIC_READ` for its whole life and
    /// is **never** barriered.
    staging: GpuBuffer,
    /// DEFAULT-heap buffer DirectML actually binds — as a **UAV**, inputs included.
    device: GpuBuffer,
    /// Bytes to `CopyBufferRegion` from `staging` into `device`.  Equal to the tensor's
    /// `TotalTensorSizeInBytes`, which [`stage_input`] has proved equals the host slice's
    /// length in bytes.
    copy_bytes: u64,
    /// `DML_BUFFER_BINDING::SizeInBytes`: `copy_bytes` rounded up to
    /// [`DML_BUFFER_ALIGNMENT`].  Never smaller than the tensor footprint DirectML was
    /// told about, and never larger than the DEFAULT buffer, which is rounded up to 256.
    binding_bytes: u64,
}

/// The output operand, staged for one DirectML dispatch.
struct StagedOutput {
    /// DEFAULT-heap buffer DirectML writes through its output UAV.
    device: GpuBuffer,
    /// READBACK-heap buffer.  Lives in `COPY_DEST` for its whole life and is **never**
    /// barriered.
    readback: GpuBuffer,
    /// Bytes to `CopyBufferRegion` from `device` into `readback`.
    copy_bytes: u64,
    /// `DML_BUFFER_BINDING::SizeInBytes` for the output binding.
    binding_bytes: u64,
    /// Number of `f32`s to read back.
    elems: usize,
}

impl DmlEngine {
    /// `None` when `DMLCreateDevice` is unavailable — the caller then falls back to the
    /// HLSL engine.  **Never panics.**
    pub(crate) fn new(core: &D3d12Core) -> Option<Self> {
        let dml = DmlDevice::try_new(core)?;
        Some(Self {
            dml,
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// `DML_OPERATOR_GEMM`, with alpha / beta / transposes folded into the operator.
    ///
    /// The DirectML path needs **no** CPU transpose and **no** CPU epilogue: `TransA`,
    /// `TransB`, `Alpha` and `Beta` all live in `DML_GEMM_OPERATOR_DESC`, and a broadcast
    /// bias is expressed as 0-strides in its `CTensor` descriptor, so nothing is copied.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when the plan cannot be expressed as a rank-4 GEMM.
    /// [`DirectMLError::DispatchFailed`] when the plan says there is a bias but the caller
    /// supplied no `C` data — that is a caller bug, and silently dropping the bias would
    /// produce a plausible, wrong answer.
    /// [`DirectMLError::Win32`] on a genuine GPU failure.
    pub(crate) fn matmul(
        &self,
        core: &D3d12Core,
        plan: &MatMulPlan,
        a: &[f32],
        b: &[f32],
        c: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let layout = DmlGemmLayout::from_plan(plan)?;
        let key = OpCacheKey::gemm(plan, &layout);
        let cached = self.compiled_for(core, key, |dml| compile_gemm(dml, plan, &layout))?;

        let a_input = stage_input(core, a, &layout.a, "Gemm A")?;
        let b_input = stage_input(core, b, &layout.b, "Gemm B")?;
        let c_input = match (layout.c.as_ref(), c) {
            (Some(c_layout), Some(c_data)) => Some(stage_input(core, c_data, c_layout, "Gemm C")?),
            (Some(_), None) => {
                return Err(DirectMLError::DispatchFailed(
                    "Gemm: the plan carries a C operand (beta != 0) but no C data was \
                     supplied; dispatching without it would silently drop the bias"
                        .into(),
                ));
            }
            // `beta == 0`, so `plan.rs` already dropped the `C` shape and the compiled
            // operator has a null `CTensor`.  ONNX says `beta * C`; `0 * C` is nothing.
            // Any `c` the caller passed contributes exactly nothing, so ignoring it here
            // is the correct answer, not a shortcut.
            (None, _) => None,
        };

        let output = stage_output(core, &layout.output, plan.output_elems()?)?;

        // Three slots — `A`, `B`, and the *optional* `C`.  An absent `C` is a
        // `DML_BINDING_TYPE_NONE` binding in slot 2, not a shorter array.
        let inputs: [Option<StagedInput>; GEMM_INPUT_SLOTS] =
            [Some(a_input), Some(b_input), c_input];
        self.execute(core, &cached, &inputs, &output)
    }

    /// `DML_OPERATOR_ELEMENT_WISE_{ADD,SUBTRACT,MULTIPLY,DIVIDE}`.
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
        let layout = DmlElementwiseLayout::from_plan(plan)?;
        let key = OpCacheKey::binary(op, &layout)?;
        let cached = self.compiled_for(core, key, |dml| compile_binary(dml, op, &layout))?;

        let b_layout = layout.b.ok_or_else(|| {
            DirectMLError::Declined(format!(
                "{}: binary elementwise op has no B operand layout",
                op.as_str()
            ))
        })?;

        let a_input = stage_input(core, a, &layout.a, "elementwise A")?;
        let b_input = stage_input(core, b, &b_layout, "elementwise B")?;
        let output = stage_output(core, &layout.output, elems_usize(plan.elem_count)?)?;

        let inputs: [Option<StagedInput>; BINARY_INPUT_SLOTS] = [Some(a_input), Some(b_input)];
        self.execute(core, &cached, &inputs, &output)
    }

    /// `DML_OPERATOR_ACTIVATION_{RELU,SIGMOID,TANH}`.
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
        let layout = DmlElementwiseLayout::from_plan(plan)?;
        let key = OpCacheKey::unary(op, &layout);
        let cached = self.compiled_for(core, key, |dml| compile_unary(dml, op, &layout))?;

        let a_input = stage_input(core, a, &layout.a, "activation input")?;
        let output = stage_output(core, &layout.output, elems_usize(plan.elem_count)?)?;

        let inputs: [Option<StagedInput>; UNARY_INPUT_SLOTS] = [Some(a_input)];
        self.execute(core, &cached, &inputs, &output)
    }

    /// `DML_OPERATOR_ACTIVATION_SOFTMAX` — the axis-less operator, which normalises the
    /// innermost dimension.
    ///
    /// # The non-innermost-axis decline lives here
    ///
    /// windows-0.62.2 has no `SOFTMAX1` operator with an explicit axis, so the axis-less
    /// `DML_ACTIVATION_SOFTMAX_OPERATOR_DESC` can only normalise the tensor's *last*
    /// dimension.  ONNX softmax over a non-terminal axis is therefore **declined** here — a
    /// `DirectMLError::Declined`, not an `Err` — so the router falls back to the CPU/HLSL
    /// path, whose `inner`-strided kernel handles any axis.  This is the exact boundary
    /// [`crate::plan::SoftmaxPlan::reduces_last_axis`] draws.  Declining an *unsupported
    /// configuration* is distinct from failing a *broken GPU op*: this is the former.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when the softmax axis is not innermost, or the shape is
    /// not rank-4-describable.  [`DirectMLError::Win32`] on a genuine GPU failure.
    pub(crate) fn softmax(
        &self,
        core: &D3d12Core,
        plan: &SoftmaxPlan,
        a: &[f32],
    ) -> Result<Vec<f32>> {
        if !plan.reduces_last_axis() {
            return Err(DirectMLError::Declined(format!(
                "Softmax: axis {} is not the innermost dimension (inner = {}); the axis-less \
                 DML_ACTIVATION_SOFTMAX normalises the last dimension only.  Declining to the \
                 CPU/HLSL path, which handles any axis.",
                plan.axis, plan.inner
            )));
        }

        let layout = DmlTensorLayout::packed(&plan.shape)?;
        let key = OpCacheKey::softmax(plan)?;
        let cached = self.compiled_for(core, key, |dml| compile_softmax(dml, plan))?;

        // Softmax is shape-preserving: input and output share the one packed layout.
        let a_input = stage_input(core, a, &layout, "Softmax input")?;
        let output = stage_output(core, &layout, plan.output_elems()?)?;

        let inputs: [Option<StagedInput>; SOFTMAX_INPUT_SLOTS] = [Some(a_input)];
        self.execute(core, &cached, &inputs, &output)
    }

    /// `DML_OPERATOR_REDUCE` over a single axis (`ReduceSum` / `Mean` / `Max` / `Min`).
    ///
    /// The output tensor DirectML writes is the input sizes with the reduced axis collapsed
    /// to 1; its element count equals [`crate::plan::ReducePlan::output_elems`] whatever
    /// ONNX `keepdims` says, so the readback length is the same either way.
    ///
    /// # Errors
    /// [`DirectMLError::Declined`] when the input is not rank-4-describable (rank > 4 — the
    /// CPU kernel handles it).  [`DirectMLError::Win32`] on a genuine GPU failure.
    pub(crate) fn reduce(
        &self,
        core: &D3d12Core,
        plan: &ReducePlan,
        a: &[f32],
    ) -> Result<Vec<f32>> {
        // The same derivation `compile_reduce` and `OpCacheKey::reduce` use; the axis is
        // folded into the compiled operator, so it is not needed again for staging.
        let (input, _axis, output) = dml_reduce_layouts(plan)?;
        let key = OpCacheKey::reduce(plan)?;
        let cached = self.compiled_for(core, key, |dml| compile_reduce(dml, plan))?;

        let a_input = stage_input(core, a, &input, plan.kind.as_str())?;
        let out = stage_output(core, &output, plan.output_elems()?)?;

        let inputs: [Option<StagedInput>; REDUCE_INPUT_SLOTS] = [Some(a_input)];
        self.execute(core, &cached, &inputs, &out)
    }

    /// `DML_OPERATOR_CONVOLUTION` — a genuine, forward, cross-correlation 2-D conv mapping
    /// straight onto Microsoft's validated metacommand.
    ///
    /// This is the whole reason `Conv` is DirectML-only: there is no HLSL conv kernel, so
    /// the HLSL engine declines `Conv` and only this path executes it.
    ///
    /// # Errors
    /// [`DirectMLError::DispatchFailed`] when the plan carries a bias but the caller
    /// supplied none — a caller bug, caught rather than silently dropping the bias, exactly
    /// as [`Self::matmul`] does for a missing `C`.
    /// [`DirectMLError::Declined`] on the size limits of the layout constructors.
    /// [`DirectMLError::Win32`] on a genuine GPU failure.
    pub(crate) fn conv(
        &self,
        core: &D3d12Core,
        plan: &ConvPlan,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let input_layout = DmlTensorLayout::packed(&plan.input_shape)?;
        let filter_layout = DmlTensorLayout::packed(&plan.weight_shape)?;
        let output_layout = DmlTensorLayout::packed(&plan.output_shape)?;
        let key = OpCacheKey::conv(plan)?;
        let cached = self.compiled_for(core, key, |dml| compile_conv(dml, plan))?;

        let input_staged = stage_input(core, input, &input_layout, "Conv input")?;
        let weight_staged = stage_input(core, weight, &filter_layout, "Conv weight")?;
        let bias_staged = match (plan.has_bias, bias) {
            (true, Some(bias_data)) => {
                let bias_layout = dml_conv_bias_layout(plan)?;
                Some(stage_input(core, bias_data, &bias_layout, "Conv bias")?)
            }
            (true, None) => {
                return Err(DirectMLError::DispatchFailed(
                    "Conv: the plan carries a bias but no bias data was supplied; dispatching \
                     without it would silently drop the bias"
                        .into(),
                ));
            }
            // The plan was compiled with a null `BiasTensor`, so there is nothing to bind
            // any bias data to; ignoring it matches how `matmul` ignores a `C` when
            // `beta == 0`.
            (false, _) => None,
        };

        let output = stage_output(core, &output_layout, plan.output_elems()?)?;

        // Three slots — input, filter, and the *optional* bias.  An absent bias is a
        // `DML_BINDING_TYPE_NONE` binding in slot 2, not a shorter array.
        let inputs: [Option<StagedInput>; CONV_INPUT_SLOTS] =
            [Some(input_staged), Some(weight_staged), bias_staged];
        self.execute(core, &cached, &inputs, &output)
    }

    /// Get the cached, initialised operator for `key`, or compile **and initialise** one.
    ///
    /// # The initialize submission
    ///
    /// On a miss this performs the **first** of the two submissions described in this
    /// module's header, and blocks on it.  See that header for why the wait cannot be
    /// merged into the execute submission: `IDMLBindingTable::Reset` rewrites the heap's
    /// descriptors from the CPU immediately, and would clobber the descriptors the
    /// not-yet-executed initializer dispatch is going to read.
    ///
    /// # Errors
    /// Whatever `build` returns, plus [`DirectMLError::Win32`] from
    /// `CreateOperatorInitializer`, the resource creations, or the submission.
    fn compiled_for(
        &self,
        core: &D3d12Core,
        key: OpCacheKey,
        build: impl FnOnce(&IDMLDevice) -> Result<IDMLCompiledOperator>,
    ) -> Result<Rc<CachedOp>> {
        // Take the `Ref` and drop it inside this statement: holding a `RefCell` borrow
        // across the compile-and-initialise below would panic the moment anything
        // re-entered the cache, and `#![deny(clippy::panic)]` does not catch that.
        let hit = self.cache.borrow().get(&key).map(Rc::clone);
        if let Some(cached) = hit {
            return Ok(cached);
        }

        let compiled = build(&self.dml.device)?;

        // SAFETY: `compiled` is a live `IDMLCompiledOperator` just returned by
        // `CompileOperator`.  `GetBindingProperties` (on the `IDMLDispatchable` it derefs
        // to) only reads immutable operator metadata and writes a POD out-parameter that
        // `windows` allocates on our stack.  It cannot fail and returns no `HRESULT`.
        let exec_props = unsafe { compiled.GetBindingProperties() };

        // SAFETY: `self.dml.device` is a live `IDMLDevice`.  The slice we pass is a stack
        // temporary that lives across the whole call, which is all DirectML requires — it
        // AddRefs any operator it keeps.  The `compiled.clone()` inside it is an explicit
        // AddRef balanced by the `Option<IDMLCompiledOperator>`'s own `Drop` when the
        // temporary array dies at the end of this statement, so the reference count is
        // net-unchanged by us.
        let initializer: IDMLOperatorInitializer = unsafe {
            self.dml
                .device
                .CreateOperatorInitializer(Some(&[Some(compiled.clone())]))
        }
        .ctx("IDMLDevice::CreateOperatorInitializer")?;

        // SAFETY: as for `exec_props`, on the freshly-created initializer.
        let init_props = unsafe { initializer.GetBindingProperties() };

        // The heap must be big enough for whichever dispatchable is bound to the table at
        // the time, and the same table is used for both.  Floored at 1 because D3D12
        // rejects a zero-descriptor heap.
        let descriptor_count = init_props
            .RequiredDescriptorCount
            .max(exec_props.RequiredDescriptorCount)
            .max(1);
        let heap = DescriptorHeap::new(core, descriptor_count)?;

        // The persistent resource.  Allocated exactly once, here, and never again.
        let persistent_bytes = binding_size(exec_props.PersistentResourceSize)?;
        let persistent = if persistent_bytes > 0 {
            Some(default_buffer_for_binding(
                core,
                persistent_bytes,
                "DirectML persistent resource",
            )?)
        } else {
            None
        };

        // The *initializer's* temporary — sized from `init_props`, never from
        // `exec_props`.  It is transient: it dies at the end of this function, after the
        // submission that uses it has been waited on.
        let init_temp_bytes = binding_size(init_props.TemporaryResourceSize)?;
        let init_temp = if init_temp_bytes > 0 {
            Some(default_buffer_for_binding(
                core,
                init_temp_bytes,
                "DirectML initializer temporary",
            )?)
        } else {
            None
        };

        let table = BindingTable::new(&self.dml.device, &heap, &initializer, descriptor_count)?;

        // The initializer takes **no inputs**: every tensor descriptor this crate builds
        // carries `DML_TENSOR_FLAG_NONE`, never `DML_TENSOR_FLAG_OWNED_BY_DML`, so there
        // is no weight for DirectML to pre-process and no input to route through the
        // initializer.  A freshly-created binding table has every binding set to
        // `DML_BINDING_TYPE_NONE` already, so *not* calling `BindInputs` is exactly
        // equivalent to binding NONE — which is what we want, and what `HelloDirectML`
        // does.
        //
        // The initializer's single **output** is the persistent resource.
        let persistent_bindings = persistent
            .as_ref()
            .map(|buf| BufferBindings::new(&[(buf, persistent_bytes)]));
        if let Some(bindings) = persistent_bindings.as_ref() {
            table.bind_outputs(&bindings.descs());
        }

        let init_temp_bindings = init_temp
            .as_ref()
            .map(|buf| BufferBindings::new(&[(buf, init_temp_bytes)]));
        let init_temp_desc = init_temp_bindings.as_ref().and_then(|b| b.desc_at(0));
        if init_temp_desc.is_some() {
            table.bind_temporary(init_temp_desc.as_ref());
        }

        core.begin()?;

        // DirectML requires every buffer bound to a binding table to be in
        // `UNORDERED_ACCESS` when the dispatch runs.  Both of these were created in
        // `COMMON`, so these are legal explicit transitions.
        if let Some(buf) = persistent.as_ref() {
            buf.barrier_to(&core.list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if let Some(buf) = init_temp.as_ref() {
            buf.barrier_to(&core.list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        // SAFETY: `core.list` is open — `core.begin()` just `Reset` it, and nothing has
        // closed it.  `heap` is a live shader-visible CBV/SRV/UAV heap that outlives this
        // submission (it is moved into the `CachedOp` below, which the cache owns).  D3D12
        // permits exactly one CBV/SRV/UAV heap to be bound at a time; we bind exactly one.
        // The `clone()` AddRefs the heap and the temporary array Releases it again at the
        // end of the statement, so the count is net-unchanged.
        unsafe {
            core.list.SetDescriptorHeaps(&[Some(heap.raw().clone())]);
        }

        // SAFETY: `core.list` is open and has `heap` bound.  `initializer` is a live
        // `IDMLDispatchable`.  `table` was created over that same initializer and over
        // `heap`, and the `DML_BUFFER_BINDING`s it references are owned by
        // `persistent_bindings` / `init_temp_bindings`, which are alive until the end of
        // this function — i.e. across the submission below.  `RecordDispatch` only records
        // into the command list; it does not execute anything.
        unsafe {
            self.dml
                .recorder
                .RecordDispatch(&core.list, &initializer, table.raw());
        }

        // **The mandatory wait.**  See this module's header.  Everything below this line
        // — in particular `BindingTable::reset` in `execute` — rewrites `heap`'s
        // descriptors from the CPU, and must not do so until the GPU has finished reading
        // the ones the initializer's dispatch was recorded against.
        core.submit_and_wait()?;

        // `initializer`, `init_temp`, `init_temp_bindings` and `persistent_bindings` all
        // die here.  The GPU is provably finished with every one of them: `submit_and_wait`
        // returned, which means the fence for this submission was signalled.
        let cached = Rc::new(CachedOp {
            compiled,
            persistent,
            persistent_bytes,
            temporary_bytes: binding_size(exec_props.TemporaryResourceSize)?,
            descriptor_count,
            heap,
            table,
        });
        self.cache.borrow_mut().insert(key, Rc::clone(&cached));
        Ok(cached)
    }

    /// The execute half, shared by all three entry points.
    ///
    /// `inputs` is indexed by the operator's **input slot**; a `None` slot is an absent
    /// optional input and becomes a `DML_BINDING_TYPE_NONE` binding, because DirectML
    /// requires the binding count to equal the operator's input-tensor count.
    ///
    /// # Errors
    /// [`DirectMLError::Win32`] on any failing `HRESULT` — resource creation, the binding
    /// table reset, or the submission.  A GPU failure is never swallowed and never
    /// silently downgraded to a CPU fallback here; the router upstream decides that.
    fn execute(
        &self,
        core: &D3d12Core,
        op: &CachedOp,
        inputs: &[Option<StagedInput>],
        output: &StagedOutput,
    ) -> Result<Vec<f32>> {
        // ── 1. The execute-time temporary.  Sized from the *compiled operator's*
        //       properties, never the initializer's.  The two differ, and DirectML will
        //       happily write past the end of a buffer sized from the wrong one.
        let temporary = if op.temporary_bytes > 0 {
            Some(default_buffer_for_binding(
                core,
                op.temporary_bytes,
                "DirectML execute temporary",
            )?)
        } else {
            None
        };

        // ── 2. Bindings.  `BufferBindings` owns the AddRef'd `ID3D12Resource` handles
        //       inside the `DML_BUFFER_BINDING`s and Releases each exactly once on drop.
        //       The `DML_BINDING_DESC`s below point *into* those `Vec`s, so every
        //       `BufferBindings` must outlive the `RecordDispatch` — each of them lives to
        //       the end of this function, i.e. past `submit_and_wait`.
        let present: Vec<(&GpuBuffer, u64)> = inputs
            .iter()
            .flatten()
            .map(|staged| (&staged.device, staged.binding_bytes))
            .collect();
        let input_bindings = BufferBindings::new(&present);
        let slots_present: Vec<bool> = inputs.iter().map(Option::is_some).collect();
        let input_descs = interleave_input_descs(&slots_present, &input_bindings.descs())?;

        let output_bindings = BufferBindings::new(&[(&output.device, output.binding_bytes)]);
        let temporary_bindings = temporary
            .as_ref()
            .map(|buf| BufferBindings::new(&[(buf, op.temporary_bytes)]));
        let persistent_bindings = op
            .persistent
            .as_ref()
            .map(|buf| BufferBindings::new(&[(buf, op.persistent_bytes)]));

        // ── 3. Point the cached binding table at the compiled operator.  This rewrites
        //       `op.heap`'s descriptors from the **CPU, immediately** — which is exactly
        //       why `compiled_for` had to submit *and wait for* the initializer before
        //       anything could get here, and why the previous `execute` (if any) also
        //       waited before returning.
        op.table
            .reset(&op.heap, &op.compiled, op.descriptor_count)?;
        op.table.bind_inputs(&input_descs);
        op.table.bind_outputs(&output_bindings.descs());

        let temporary_desc = temporary_bindings.as_ref().and_then(|b| b.desc_at(0));
        if temporary_desc.is_some() {
            op.table.bind_temporary(temporary_desc.as_ref());
        }
        // The **same** persistent buffer the initializer wrote in `compiled_for`.  A
        // freshly-allocated one here would read as uninitialised operator state — a
        // plausible, wrong answer.
        let persistent_desc = persistent_bindings.as_ref().and_then(|b| b.desc_at(0));
        if persistent_desc.is_some() {
            op.table.bind_persistent(persistent_desc.as_ref());
        }

        // ── 4. Record.  `begin()` is legal only because the previous `submit_and_wait`
        //       returned; resetting an allocator the GPU still owns is undefined behaviour
        //       that nothing catches.
        core.begin()?;
        let list = &core.list;

        for staged in inputs.iter().flatten() {
            // `COMMON → COPY_DEST → (copy) → UNORDERED_ACCESS`.
            //
            // DirectML binds its **inputs as UAVs**, unlike the HLSL path, which binds
            // them as SRVs — so the post-copy state is `UNORDERED_ACCESS`, not
            // `NON_PIXEL_SHADER_RESOURCE`.  `staged.staging` is an UPLOAD buffer: it lives
            // in `GENERIC_READ` forever and is never barriered.
            staged
                .device
                .barrier_to(list, D3D12_RESOURCE_STATE_COPY_DEST);
            staged
                .device
                .record_copy_from(list, &staged.staging, staged.copy_bytes);
            staged
                .device
                .barrier_to(list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        output
            .device
            .barrier_to(list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if let Some(buf) = temporary.as_ref() {
            buf.barrier_to(list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if let Some(buf) = op.persistent.as_ref() {
            // Records nothing: the tracker already says `UNORDERED_ACCESS` from
            // `compiled_for`.  Correct anyway — see this module's header, "Why the
            // persistent buffer needs no barrier after the first submission".
            buf.barrier_to(list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        // SAFETY: `list` is open (`core.begin()` just `Reset` it).  `op.heap` is the
        // shader-visible heap that `op.table`'s descriptors live in, and it outlives this
        // submission — the caller holds an `Rc<CachedOp>` for the whole call.  Exactly one
        // CBV/SRV/UAV heap is bound, as D3D12 requires.  Omitting this call is a device
        // removal.
        unsafe {
            list.SetDescriptorHeaps(&[Some(op.heap.raw().clone())]);
        }

        // SAFETY: `list` is open and has the right heap bound; `op.compiled` is a live
        // `IDMLDispatchable` that `op.table` was just `Reset` onto; every `GpuBuffer` the
        // table references — inputs, output, temporary, persistent — is alive until after
        // `submit_and_wait` below, and every one of them has been transitioned to
        // `UNORDERED_ACCESS` (explicitly, or by D3D12's implicit promotion of a buffer out
        // of `COMMON`).  `RecordDispatch` only records into the command list.
        unsafe {
            self.dml
                .recorder
                .RecordDispatch(list, &op.compiled, op.table.raw());
        }

        // ── 5. Readback.
        //
        // The UAV barrier is the one everyone omits.  A `UNORDERED_ACCESS → COPY_SOURCE`
        // transition is a *state* change; it is not, on every IHV, a promise that the
        // dispatch's UAV writes are visible to the copy engine.  Without this barrier the
        // readback is correct-by-luck on NVIDIA and garbage on AMD.
        output.device.record_uav_barrier(list);
        output
            .device
            .barrier_to(list, D3D12_RESOURCE_STATE_COPY_SOURCE);
        // `output.readback` is a READBACK buffer: created in `COPY_DEST`, stays there
        // forever, never barriered.
        output
            .readback
            .record_copy_from(list, &output.device, output.copy_bytes);

        core.submit_and_wait()?;
        output.readback.read_f32(output.elems)
    }
}

/// Rebuild the operator's full input-binding array, filling every absent optional slot
/// with a `DML_BINDING_TYPE_NONE` binding.
///
/// `slots_present[i]` says whether operator input slot `i` was supplied; `present` holds
/// the `DML_BINDING_DESC`s of the supplied operands, in slot order, with the absent ones
/// missing.  DirectML requires `BindInputs` to receive **exactly** one binding per input
/// tensor of the operator, so a bias-free `DML_OPERATOR_GEMM` gets three bindings, the
/// third being NONE — not two bindings.
///
/// A `DML_BINDING_DESC` is a plain `{ tag, *const c_void }` pair and holds **no** COM
/// reference, so constructing the NONE variant here does not go anywhere near
/// [`crate::backend::dml::binding`]'s `ManuallyDrop` hazard: that rule is about
/// `DML_BUFFER_BINDING` and `DML_BINDING_TABLE_DESC`, both of which own an interface
/// pointer.  The descs copied out of `present` carry raw pointers into the caller's
/// [`BufferBindings`], which must outlive the resulting `Vec`.
///
/// # Errors
/// [`DirectMLError::DispatchFailed`] when `present` holds fewer descs than
/// `slots_present` has `true`s — a caller bug, caught here rather than handed to DirectML
/// as a dangling read.
fn interleave_input_descs(
    slots_present: &[bool],
    present: &[DML_BINDING_DESC],
) -> Result<Vec<DML_BINDING_DESC>> {
    let mut descs = Vec::with_capacity(slots_present.len());
    let mut next = 0usize;
    for &supplied in slots_present {
        if supplied {
            let desc = *present.get(next).ok_or_else(|| {
                DirectMLError::DispatchFailed(format!(
                    "DirectML input bindings: slot {next} has no descriptor (got {} for {} \
                     supplied slots)",
                    present.len(),
                    slots_present.iter().filter(|s| **s).count()
                ))
            })?;
            descs.push(desc);
            next += 1;
        } else {
            descs.push(DML_BINDING_DESC {
                Type: DML_BINDING_TYPE_NONE,
                Desc: core::ptr::null(),
            });
        }
    }
    Ok(descs)
}

/// `DML_BUFFER_BINDING::SizeInBytes` for a region of `bytes` bytes.
///
/// Rounded up to [`DML_BUFFER_ALIGNMENT`], so the binding is never smaller than the
/// tensor footprint DirectML was told about and always satisfies DirectML's buffer
/// alignment.  Every DEFAULT buffer this file allocates is rounded up to 256 bytes by
/// [`GpuBuffer::new_default`], and 256 is a multiple of 16, so the binding can never
/// exceed its resource.
///
/// # Errors
/// [`DirectMLError::Declined`] when the rounding overflows.
fn binding_size(bytes: u64) -> Result<u64> {
    let unaligned = usize::try_from(bytes).map_err(|_| {
        DirectMLError::Declined(format!("DirectML binding size {bytes} exceeds usize"))
    })?;
    let aligned = align_up(unaligned, DML_BUFFER_ALIGNMENT).ok_or_else(|| {
        DirectMLError::Declined(format!(
            "DirectML binding size {bytes} overflows when aligned to {DML_BUFFER_ALIGNMENT}"
        ))
    })?;
    u64::try_from(aligned).map_err(|_| {
        DirectMLError::Declined(format!(
            "aligned DirectML binding size {aligned} exceeds u64"
        ))
    })
}

/// Allocate a DEFAULT-heap buffer, and *prove* it is at least as large as the
/// `DML_BUFFER_BINDING::SizeInBytes` we are about to declare over it.
///
/// [`GpuBuffer::new_default`] rounds its request up to 256 bytes, and 256 is a multiple of
/// [`DML_BUFFER_ALIGNMENT`], so this can only fail if that contract changes underneath us.
/// It is checked rather than assumed because the failure mode is DirectML writing past the
/// end of the resource — silent heap corruption, no `HRESULT`, no crash at the call site.
///
/// # Errors
/// [`DirectMLError::Win32`] on a creation failure; [`DirectMLError::DispatchFailed`] when
/// the resource came back smaller than the binding we intend to declare over it.
fn default_buffer_for_binding(
    core: &D3d12Core,
    binding_bytes: u64,
    what: &str,
) -> Result<GpuBuffer> {
    let buffer = GpuBuffer::new_default(core, binding_bytes)?;
    let actual = buffer.size_bytes();
    if actual < binding_bytes {
        return Err(DirectMLError::DispatchFailed(format!(
            "{what}: the DEFAULT buffer is {actual} bytes but the DirectML binding declares \
             {binding_bytes}; DirectML would write past the end of the resource"
        )));
    }
    Ok(buffer)
}

/// A `u32` element count as a `usize`.
///
/// Honestly fallible: `usize` is not guaranteed to be at least 32 bits, and this crate's
/// rule is that no shape-derived value is ever converted with a bare `as`.
///
/// # Errors
/// [`DirectMLError::Declined`] when `n` does not fit a `usize`.
fn elems_usize(n: u32) -> Result<usize> {
    usize::try_from(n)
        .map_err(|_| DirectMLError::Declined(format!("element count {n} exceeds usize")))
}

/// The byte length of a host operand, as a `u64`.
///
/// # Errors
/// [`DirectMLError::Declined`] on overflow.
fn host_bytes(data: &[f32]) -> Result<u64> {
    let bytes = data.len().checked_mul(ELEM_SIZE).ok_or_else(|| {
        DirectMLError::Declined(format!(
            "host operand of {} f32 overflows usize",
            data.len()
        ))
    })?;
    u64::try_from(bytes)
        .map_err(|_| DirectMLError::Declined(format!("host operand of {bytes} bytes exceeds u64")))
}

/// Stage one input tensor: an UPLOAD copy of the host data plus the DEFAULT buffer
/// DirectML will bind.
///
/// # The footprint cross-check
///
/// `layout.total_bytes` is `DML_BUFFER_TENSOR_DESC::TotalTensorSizeInBytes` — the *true*
/// memory footprint given the strides, which for a 0-stride broadcast operand is the
/// **source's** packed size, not `product(sizes) * 4`.  It must equal the host slice's
/// byte length, because that host slice is precisely what gets uploaded.  If the two ever
/// disagree, DirectML has been handed a descriptor that reads past (or short of) the
/// buffer actually bound: it returns a tensor of the right *shape* full of the wrong
/// *values*, and no `HRESULT` is raised.  This check turns that into a loud error.
///
/// # Errors
/// [`DirectMLError::DispatchFailed`] when the layout and the host operand disagree.
/// [`DirectMLError::Win32`] on a resource-creation failure.
fn stage_input(
    core: &D3d12Core,
    data: &[f32],
    layout: &DmlTensorLayout,
    what: &str,
) -> Result<StagedInput> {
    let bytes = host_bytes(data)?;
    if layout.total_bytes != bytes {
        return Err(DirectMLError::DispatchFailed(format!(
            "{what}: DirectML tensor footprint is {} bytes (sizes {:?}, strides {:?}) but the \
             host operand holds {bytes} bytes ({} f32).  Binding these together would make \
             DirectML read outside the buffer and return plausible, wrong numbers.",
            layout.total_bytes,
            layout.sizes,
            layout.strides,
            data.len()
        )));
    }

    let binding_bytes = binding_size(bytes)?;
    let staging = GpuBuffer::upload_from_f32(core, data)?;
    let device = default_buffer_for_binding(core, binding_bytes, what)?;
    Ok(StagedInput {
        staging,
        device,
        copy_bytes: bytes,
        binding_bytes,
    })
}

/// Stage the output tensor: the DEFAULT buffer DirectML writes, plus the READBACK buffer
/// the result is copied into.
///
/// # Errors
/// [`DirectMLError::DispatchFailed`] when the layout and `elems` disagree — same hazard,
/// and same reasoning, as [`stage_input`].
/// [`DirectMLError::Win32`] on a resource-creation failure.
fn stage_output(core: &D3d12Core, layout: &DmlTensorLayout, elems: usize) -> Result<StagedOutput> {
    let bytes = elems
        .checked_mul(ELEM_SIZE)
        .ok_or_else(|| DirectMLError::Declined(format!("output of {elems} f32 overflows usize")))?;
    let bytes = u64::try_from(bytes)
        .map_err(|_| DirectMLError::Declined(format!("output of {bytes} bytes exceeds u64")))?;

    if layout.total_bytes != bytes {
        return Err(DirectMLError::DispatchFailed(format!(
            "output: DirectML tensor footprint is {} bytes (sizes {:?}) but the plan says \
             {elems} f32 = {bytes} bytes",
            layout.total_bytes, layout.sizes
        )));
    }

    let binding_bytes = binding_size(bytes)?;
    let device = default_buffer_for_binding(core, binding_bytes, "output")?;
    let readback = GpuBuffer::new_readback(core, bytes)?;
    Ok(StagedOutput {
        device,
        readback,
        copy_bytes: bytes,
        binding_bytes,
        elems,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{binding_size, elems_usize, interleave_input_descs, GEMM_INPUT_SLOTS};
    use crate::error::DirectMLError;
    use windows::Win32::AI::MachineLearning::DirectML::{
        DML_BINDING_DESC, DML_BINDING_TYPE_BUFFER, DML_BINDING_TYPE_NONE,
    };

    /// Distinct, non-null, permanently-valid tags, so each fake binding gets a
    /// distinguishable `Desc` pointer.  A `static`, so the pointers are trivially live.
    static DESC_TAGS: [u32; 4] = [11, 22, 33, 44];

    /// A `DML_BINDING_DESC` that is distinguishable from the NONE variant without owning
    /// any COM reference.  The `Desc` pointer is never dereferenced by these tests, and it
    /// points into a `static`, so it is valid regardless.
    fn fake_buffer_desc(slot: usize) -> DML_BINDING_DESC {
        DML_BINDING_DESC {
            Type: DML_BINDING_TYPE_BUFFER,
            Desc: core::ptr::addr_of!(DESC_TAGS[slot]).cast(),
        }
    }

    #[test]
    fn binding_size_rounds_up_to_the_dml_alignment() {
        assert_eq!(binding_size(0).unwrap(), 0);
        assert_eq!(binding_size(4).unwrap(), 16);
        assert_eq!(binding_size(16).unwrap(), 16);
        assert_eq!(binding_size(17).unwrap(), 32);
        // 12 f32 = 48 bytes, already 16-aligned.
        assert_eq!(binding_size(48).unwrap(), 48);
    }

    #[test]
    fn elems_usize_round_trips() {
        assert_eq!(elems_usize(0).unwrap(), 0);
        assert_eq!(
            elems_usize(u32::MAX).unwrap(),
            usize::try_from(u32::MAX).unwrap()
        );
    }

    /// A bias-free `Gemm` must still present **three** input bindings, the third being
    /// `DML_BINDING_TYPE_NONE`.  Presenting two is an `E_INVALIDARG` from DirectML.
    #[test]
    fn absent_optional_gemm_bias_becomes_a_none_binding_in_slot_two() {
        let present = [fake_buffer_desc(0), fake_buffer_desc(1)];
        let descs = interleave_input_descs(&[true, true, false], &present).unwrap();

        assert_eq!(descs.len(), GEMM_INPUT_SLOTS);
        assert_eq!(descs[0].Type, DML_BINDING_TYPE_BUFFER);
        assert_eq!(descs[1].Type, DML_BINDING_TYPE_BUFFER);
        assert_eq!(descs[2].Type, DML_BINDING_TYPE_NONE);
        assert!(descs[2].Desc.is_null());
        // The supplied descs must land in their own slots, in order.
        assert_eq!(descs[0].Desc, present[0].Desc);
        assert_eq!(descs[1].Desc, present[1].Desc);
    }

    #[test]
    fn a_present_gemm_bias_fills_all_three_slots() {
        let present = [
            fake_buffer_desc(0),
            fake_buffer_desc(1),
            fake_buffer_desc(2),
        ];
        let descs = interleave_input_descs(&[true, true, true], &present).unwrap();
        assert_eq!(descs.len(), GEMM_INPUT_SLOTS);
        assert!(descs.iter().all(|d| d.Type == DML_BINDING_TYPE_BUFFER));
    }

    #[test]
    fn elementwise_slots_pass_through_unchanged() {
        let present = [fake_buffer_desc(2), fake_buffer_desc(3)];
        let descs = interleave_input_descs(&[true, true], &present).unwrap();
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].Desc, present[0].Desc);
        assert_eq!(descs[1].Desc, present[1].Desc);
    }

    /// Fewer descs than supplied slots is a caller bug.  It must be an honest `Err`, never
    /// a panic and never a silently short binding array — DirectML would read a dangling
    /// descriptor.
    #[test]
    fn too_few_descs_is_an_error_not_a_panic() {
        let present = [fake_buffer_desc(0)];
        let err = interleave_input_descs(&[true, true, false], &present).unwrap_err();
        assert!(
            matches!(err, DirectMLError::DispatchFailed(_)),
            "got {err:?}"
        );
    }
}
