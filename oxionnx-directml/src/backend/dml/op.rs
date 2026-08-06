//! `IDMLDevice::CreateOperator` + `CompileOperator` for the ops this crate claims.
//!
//! # The four-level raw-pointer chain, and the rule that keeps it alive
//!
//! `DML_OPERATOR_DESC` is a **tagged union**: `Type` names the operator and `Desc` is a
//! bare `*const c_void` that must point at the matching concrete `DML_*_OPERATOR_DESC`.
//! That concrete descriptor's tensor fields are `*const DML_TENSOR_DESC`, which is
//! *itself* a tagged union whose `Desc` points at a `DML_BUFFER_TENSOR_DESC`, whose
//! `Sizes` / `Strides` are `*const u32` into arrays owned by
//! [`DmlTensorStorage`](super::tensor::DmlTensorStorage):
//!
//! ```text
//! DML_OPERATOR_DESC.Desc ──► DML_GEMM_OPERATOR_DESC          (level 1)
//!                              .ATensor ──► DML_TENSOR_DESC   (level 2)
//!                                             .Desc ──► DML_BUFFER_TENSOR_DESC  (level 3)
//!                                                         .Sizes   ──► [u32; 4]  (level 4)
//!                                                         .Strides ──► [u32; 4]
//! ```
//!
//! **Not one of those four levels carries a Rust lifetime**, so the borrow checker sees
//! nothing.  A dangling link anywhere in the chain is a use-after-free that *usually
//! appears to work*, because the freed stack slot is very often still intact when
//! DirectML reads it microseconds later.
//!
//! ## The lifetime argument
//!
//! DirectML **deep-copies** the entire description during `IDMLDevice::CreateOperator`;
//! the returned `IDMLOperator` holds no reference to the caller's memory.  So every
//! level only has to be live *for the duration of that single call*.  This module
//! guarantees that by shape:
//!
//! > Every level of the chain is a **named local** (never a temporary, never a field of
//! > a returned value) declared in the same function body that calls `CreateOperator`,
//! > in strictly bottom-up order.  Rust drops locals at the end of the enclosing block,
//! > **not** at the end of their last use, so all four levels are still alive when
//! > `create_and_compile` runs, and all four die together when the function returns.
//!
//! Two corollaries, both load-bearing:
//!
//! * **No function in this module may return a descriptor**, at any level.  A
//!   `DML_BUFFER_TENSOR_DESC` returned by value would carry `Sizes` pointers into the
//!   dead frame of the function that built the storage.
//! * **No level may be built as a temporary inside the expression that consumes it.**
//!   Writing `ATensor: BoundTensorDesc::new(&a_buffer).as_ptr()` compiles, and dangles:
//!   the temporary `BoundTensorDesc` is dropped at the end of that `let` statement,
//!   long before `CreateOperator`.  Hence the deliberately verbose
//!   `let a_tensor = …; … ATensor: a_tensor.as_ptr()`.
//!
//! # The cache-key invariant
//!
//! [`crate::backend::dml::dml_backend`] compiles each operator **once**, keyed by
//! [`crate::layout::OpCacheKey`].  That is sound only if the key captures *every* input
//! these functions read.  It does, exactly:
//!
//! | function | reads | `OpCacheKey` variant captures |
//! |---|---|---|
//! | [`compile_gemm`] | `layout.{a,b,c,output}`, `plan.{trans_a,trans_b,alpha,beta}` | `Gemm { a, b, c, out, trans_a, trans_b, alpha_bits, beta_bits }` |
//! | [`compile_binary`] | `op`, `layout.{a,b,output}` | `Binary { op, a, b, out }` |
//! | [`compile_unary`] | `op`, `layout.{a,output}` | `Unary { op, a, out }` |
//! | [`compile_softmax`] | `packed(plan.shape)` | `Softmax { tensor }` |
//! | [`compile_reduce`] | `plan.kind`, `dml_reduce_layouts(plan)` | `Reduce { kind, input, out, axis }` |
//! | [`compile_conv`] | `packed(plan.{input,weight,output}_shape)`, `dml_conv_bias_layout`, `plan.{stride,dilation,pad,group}*` | `Conv { input, filter, bias, out, stride_*, dilation_*, pad_*, group }` |
//!
//! The Softmax / Reduce / Conv rows read the layout derivations
//! [`crate::layout::dml_reduce_layouts`] and [`crate::layout::dml_conv_bias_layout`], which
//! `OpCacheKey::{reduce,conv}` call too — so key and descriptor share one source of truth
//! and cannot drift.
//!
//! **If you teach a `compile_*` function to read a new field, add it to `OpCacheKey` in
//! the same commit**, or the cache will hand back an operator compiled for a different
//! parameter — a wrong answer, silently, only on the second node with that shape.
//!
//! # Operator IDs
//!
//! Verified against `windows-0.62.2`: `DML_OPERATOR_GEMM` (54),
//! `DML_OPERATOR_ELEMENT_WISE_ADD` (4) / `_SUBTRACT` (30) / `_MULTIPLY` (24) /
//! `_DIVIDE` (10), `DML_OPERATOR_ACTIVATION_RELU` (44) / `_SIGMOID` (47) / `_TANH` (51) /
//! `DML_OPERATOR_ACTIVATION_SOFTMAX` (48), `DML_OPERATOR_REDUCE` (55),
//! `DML_OPERATOR_CONVOLUTION` (53).

use core::ffi::c_void;
use core::ptr;

use windows::Win32::AI::MachineLearning::DirectML::{
    IDMLCompiledOperator, IDMLDevice, IDMLOperator, DML_ACTIVATION_RELU_OPERATOR_DESC,
    DML_ACTIVATION_SIGMOID_OPERATOR_DESC, DML_ACTIVATION_SOFTMAX_OPERATOR_DESC,
    DML_ACTIVATION_TANH_OPERATOR_DESC, DML_CONVOLUTION_DIRECTION_FORWARD,
    DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_OPERATOR_DESC,
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC, DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC,
    DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC, DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC,
    DML_EXECUTION_FLAG_NONE, DML_GEMM_OPERATOR_DESC, DML_MATRIX_TRANSFORM,
    DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE, DML_OPERATOR_ACTIVATION_RELU,
    DML_OPERATOR_ACTIVATION_SIGMOID, DML_OPERATOR_ACTIVATION_SOFTMAX, DML_OPERATOR_ACTIVATION_TANH,
    DML_OPERATOR_CONVOLUTION, DML_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_ADD,
    DML_OPERATOR_ELEMENT_WISE_DIVIDE, DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
    DML_OPERATOR_ELEMENT_WISE_SUBTRACT, DML_OPERATOR_GEMM, DML_OPERATOR_REDUCE, DML_OPERATOR_TYPE,
    DML_REDUCE_FUNCTION, DML_REDUCE_FUNCTION_AVERAGE, DML_REDUCE_FUNCTION_MAX,
    DML_REDUCE_FUNCTION_MIN, DML_REDUCE_FUNCTION_SUM, DML_REDUCE_OPERATOR_DESC,
};

use super::tensor::{BoundTensorDesc, DmlTensorStorage};
use crate::error::{DirectMLError, HrExt, Result};
use crate::layout::{
    dml_conv_bias_layout, dml_reduce_layouts, DmlElementwiseLayout, DmlGemmLayout, DmlTensorLayout,
};
use crate::plan::{BinaryOp, ConvPlan, MatMulPlan, ReduceKind, ReducePlan, SoftmaxPlan, UnaryOp};

// ─── call sites ──────────────────────────────────────────────────────────────

/// The two `&'static str` call-site names one operator needs.
///
/// [`HrExt::ctx`] takes a `&'static str`, so a failing `HRESULT` cannot carry a
/// `format!`-ed operator name.  Naming each site as a constant keeps
/// `DirectMLError::Win32 { context, .. }` precise — `"IDMLDevice::CompileOperator
/// (DML_OPERATOR_GEMM)"` rather than a bare `"CompileOperator"` shared by eight
/// different operators.
#[derive(Clone, Copy)]
struct CallSite {
    /// The DirectML operator's name, for the "succeeded but returned null" messages.
    name: &'static str,
    /// `HrExt` context for this operator's `CreateOperator` call.
    create: &'static str,
    /// `HrExt` context for this operator's `CompileOperator` call.
    compile: &'static str,
}

const SITE_GEMM: CallSite = CallSite {
    name: "DML_OPERATOR_GEMM",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_GEMM)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_GEMM)",
};

const SITE_ADD: CallSite = CallSite {
    name: "DML_OPERATOR_ELEMENT_WISE_ADD",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_ELEMENT_WISE_ADD)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_ELEMENT_WISE_ADD)",
};

const SITE_SUBTRACT: CallSite = CallSite {
    name: "DML_OPERATOR_ELEMENT_WISE_SUBTRACT",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_ELEMENT_WISE_SUBTRACT)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_ELEMENT_WISE_SUBTRACT)",
};

const SITE_MULTIPLY: CallSite = CallSite {
    name: "DML_OPERATOR_ELEMENT_WISE_MULTIPLY",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_ELEMENT_WISE_MULTIPLY)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_ELEMENT_WISE_MULTIPLY)",
};

const SITE_DIVIDE: CallSite = CallSite {
    name: "DML_OPERATOR_ELEMENT_WISE_DIVIDE",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_ELEMENT_WISE_DIVIDE)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_ELEMENT_WISE_DIVIDE)",
};

const SITE_RELU: CallSite = CallSite {
    name: "DML_OPERATOR_ACTIVATION_RELU",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_ACTIVATION_RELU)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_ACTIVATION_RELU)",
};

const SITE_SIGMOID: CallSite = CallSite {
    name: "DML_OPERATOR_ACTIVATION_SIGMOID",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_ACTIVATION_SIGMOID)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_ACTIVATION_SIGMOID)",
};

const SITE_TANH: CallSite = CallSite {
    name: "DML_OPERATOR_ACTIVATION_TANH",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_ACTIVATION_TANH)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_ACTIVATION_TANH)",
};

const SITE_SOFTMAX: CallSite = CallSite {
    name: "DML_OPERATOR_ACTIVATION_SOFTMAX",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_ACTIVATION_SOFTMAX)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_ACTIVATION_SOFTMAX)",
};

const SITE_REDUCE: CallSite = CallSite {
    name: "DML_OPERATOR_REDUCE",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_REDUCE)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_REDUCE)",
};

const SITE_CONV: CallSite = CallSite {
    name: "DML_OPERATOR_CONVOLUTION",
    create: "IDMLDevice::CreateOperator(DML_OPERATOR_CONVOLUTION)",
    compile: "IDMLDevice::CompileOperator(DML_OPERATOR_CONVOLUTION)",
};

// ─── the one FFI funnel ──────────────────────────────────────────────────────

/// Wrap `concrete_desc` in the `DML_OPERATOR_DESC` tagged union, hand it to
/// `IDMLDevice::CreateOperator`, and compile the result.
///
/// Every operator in this module funnels through here, so the union is constructed in
/// exactly **one** place and the `Type` ↔ `Desc` correspondence is checkable by reading
/// eight call sites rather than eight copies of this sequence.
///
/// `DML_EXECUTION_FLAG_NONE` — deliberately **not**
/// `DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION`.  This crate is f32, and
/// [`crate::DirectMLContext::self_check`] diffs the GPU against an f32 CPU oracle at a
/// tight tolerance; silently letting the driver compute in fp16 would make that check
/// fail for a reason that has nothing to do with a bug.
///
/// # Safety
///
/// * `concrete_desc` must point at a live, initialised `DML_*_OPERATOR_DESC` of exactly
///   the type that `op_type` names — this is the union's tag/payload correspondence, and
///   nothing in the type system checks it.
/// * That descriptor, and *everything it transitively points at* (the `DML_TENSOR_DESC`s,
///   their `DML_BUFFER_TENSOR_DESC`s, and those descs' `Sizes` / `Strides` arrays), must
///   remain live and unmoved for the whole of this call.  It need not outlive it:
///   `CreateOperator` deep-copies the description.
///
/// # Errors
/// [`DirectMLError::Win32`] when `CreateOperator` or `CompileOperator` returns a failing
/// `HRESULT`; [`DirectMLError::DispatchFailed`] when either returns success but a null
/// interface pointer (which DirectML must not do, but `Option<T>` makes representable).
unsafe fn create_and_compile(
    dml: &IDMLDevice,
    op_type: DML_OPERATOR_TYPE,
    concrete_desc: *const c_void,
    site: CallSite,
) -> Result<IDMLCompiledOperator> {
    // The tagged union itself: a named local, so it outlives the call below.
    let operator_desc = DML_OPERATOR_DESC {
        Type: op_type,
        Desc: concrete_desc,
    };

    let mut op: Option<IDMLOperator> = None;
    // SAFETY: `operator_desc` is a live, fully-initialised local for the whole call, and
    // its `Desc` payload matches its `Type` tag by this function's safety contract, which
    // every caller discharges by constructing both in the same expression.  `&mut op` is
    // a valid, aligned, exclusively-borrowed `*mut Option<IDMLOperator>` out-parameter;
    // `CreateOperator` either leaves it `None` and returns a failing `HRESULT`, or writes
    // an owned interface pointer into it.
    unsafe { dml.CreateOperator(ptr::addr_of!(operator_desc), &mut op) }.ctx(site.create)?;

    let op = op.ok_or_else(|| {
        DirectMLError::DispatchFailed(format!(
            "{}: CreateOperator returned success but a null IDMLOperator",
            site.name
        ))
    })?;

    let mut compiled: Option<IDMLCompiledOperator> = None;
    // SAFETY: `op` is a live, non-null `IDMLOperator` produced by `CreateOperator` on
    // this same `dml`, so it is a valid `Param<IDMLOperator>`.  `&mut compiled` is a
    // valid out-parameter, as above.  `CompileOperator` borrows `op` for the duration of
    // the call only; the compiled operator it returns is independent of it.
    unsafe { dml.CompileOperator(&op, DML_EXECUTION_FLAG_NONE, &mut compiled) }
        .ctx(site.compile)?;

    compiled.ok_or_else(|| {
        DirectMLError::DispatchFailed(format!(
            "{}: CompileOperator returned success but a null IDMLCompiledOperator",
            site.name
        ))
    })
}

/// `DML_GEMM_OPERATOR_DESC::TransA` / `TransB` from a plan's boolean.
const fn matrix_transform(transposed: bool) -> DML_MATRIX_TRANSFORM {
    if transposed {
        DML_MATRIX_TRANSFORM_TRANSPOSE
    } else {
        DML_MATRIX_TRANSFORM_NONE
    }
}

// ─── GEMM ────────────────────────────────────────────────────────────────────

/// `DML_OPERATOR_GEMM`, with `TransA` / `TransB` / `Alpha` / `Beta` / `CTensor` all
/// folded into the operator — so the DirectML path needs **no** CPU transpose and **no**
/// CPU epilogue, unlike the HLSL path.
///
/// DirectML computes `Output = Alpha · TransA(A) × TransB(B) + Beta · C`, which is ONNX
/// `Gemm` exactly.  The transposes are an *interpretation* of the stored buffer, not a
/// permutation of it: [`DmlGemmLayout::from_plan`] therefore describes `A` and `B` with
/// their **stored** shapes (`plan.a_stored_shape` / `plan.b_stored_shape`), and this
/// function only forwards the two flags.  Deriving the sizes from `plan.{m,k,n}` instead
/// would hand DirectML a transposed *description* of an untransposed *buffer* — a wrong
/// answer rather than an error.
///
/// ## `Beta` when there is no `C`
///
/// [`DmlGemmLayout::from_plan`] populates `layout.c` iff [`MatMulPlan::has_bias`] — i.e.
/// a `C` operand was supplied **and** `beta != 0`.  When it is `None` we pass
/// `CTensor: null` and force `Beta: 0.0`, rather than forwarding a `beta` that names a
/// tensor that is not bound.  DirectML documents `Beta` as ignored when `CTensor` is
/// null, but a validator that disagrees would reject the operator, and a `beta` that is
/// live in the descriptor but dead in the binding is exactly the sort of inconsistency
/// that turns into a wrong number on one driver and not another.
///
/// `FusedActivation` is always null: fusing would change the numbers the
/// [`crate::reference`] oracle is diffed against, and the router dispatches activations
/// as their own nodes anyway.
///
/// # Errors
/// [`DirectMLError::Win32`] when `CreateOperator` or `CompileOperator` fails;
/// [`DirectMLError::DispatchFailed`] on a null-but-successful return.
pub(crate) fn compile_gemm(
    dml: &IDMLDevice,
    plan: &MatMulPlan,
    layout: &DmlGemmLayout,
) -> Result<IDMLCompiledOperator> {
    // Level 4 — the `Sizes[]` / `Strides[]` arrays.  Named locals: they outlive the
    // `CreateOperator` call at the bottom of this function, which is all DirectML needs.
    let a_storage = DmlTensorStorage::new(&layout.a);
    let b_storage = DmlTensorStorage::new(&layout.b);
    let c_storage = layout.c.as_ref().map(DmlTensorStorage::new);
    let out_storage = DmlTensorStorage::new(&layout.output);

    // Level 3 — `DML_BUFFER_TENSOR_DESC`s, whose `Sizes`/`Strides` point into level 4.
    let a_buffer = a_storage.buffer_desc();
    let b_buffer = b_storage.buffer_desc();
    let c_buffer = c_storage.as_ref().map(DmlTensorStorage::buffer_desc);
    let out_buffer = out_storage.buffer_desc();

    // Level 2 — `DML_TENSOR_DESC` unions, whose `Desc` points into level 3.
    // `BoundTensorDesc` brands this level with the lifetime of the level below it, so
    // *this* link — and only this one — is checked by the borrow checker.
    let a_tensor = BoundTensorDesc::new(&a_buffer);
    let b_tensor = BoundTensorDesc::new(&b_buffer);
    let c_tensor = c_buffer.as_ref().map(BoundTensorDesc::new);
    let out_tensor = BoundTensorDesc::new(&out_buffer);

    // Level 1 — the concrete operator descriptor, pointing into level 2.
    let gemm = DML_GEMM_OPERATOR_DESC {
        ATensor: a_tensor.as_ptr(),
        BTensor: b_tensor.as_ptr(),
        CTensor: c_tensor
            .as_ref()
            .map_or(ptr::null(), BoundTensorDesc::as_ptr),
        OutputTensor: out_tensor.as_ptr(),
        TransA: matrix_transform(plan.trans_a),
        TransB: matrix_transform(plan.trans_b),
        Alpha: plan.alpha,
        Beta: if layout.c.is_some() { plan.beta } else { 0.0 },
        FusedActivation: ptr::null(),
    };

    // SAFETY: `gemm` is a live, fully-initialised `DML_GEMM_OPERATOR_DESC`, which is
    // exactly what `DML_OPERATOR_GEMM` tags.  Every pointer inside it — the four tensor
    // descs, their buffer descs, and those descs' size/stride arrays — is a named local
    // of *this* function, so the entire chain is alive and unmoved across the call and
    // dies only when this function returns.  `CTensor` and `FusedActivation` are null,
    // which DirectML defines as "absent".
    unsafe {
        create_and_compile(
            dml,
            DML_OPERATOR_GEMM,
            ptr::addr_of!(gemm).cast::<c_void>(),
            SITE_GEMM,
        )
    }
}

// ─── elementwise binary ──────────────────────────────────────────────────────

/// `DML_OPERATOR_ELEMENT_WISE_{ADD,SUBTRACT,MULTIPLY,DIVIDE}`.
///
/// A broadcast operand is expressed as **0-strides** in `layout` (see
/// [`crate::layout::DmlTensorLayout::broadcast_to`]), so DirectML reads the original,
/// un-expanded buffer and this path copies nothing — unlike the HLSL path, which has to
/// materialise the expansion on the CPU first.
///
/// The four descriptors happen to share an identical `#[repr(C)]` layout (three
/// `*const DML_TENSOR_DESC`).  They are nevertheless constructed **individually**, by
/// name, in four match arms: transmuting one into another would compile, would work
/// today, and would silently corrupt the union the day Microsoft adds a `ScaleBias`
/// field to one of them — as they already have on `DML_ELEMENT_WISE_TANH_OPERATOR_DESC`,
/// which is why this crate uses `DML_ACTIVATION_TANH_OPERATOR_DESC` instead.
///
/// # Errors
/// [`DirectMLError::Declined`] when `layout.b` is absent — a binary op with no second
/// operand is a caller bug, not a shape problem, and is reported the same way
/// [`crate::layout::OpCacheKey::binary`] reports it, so the cache and the compiler
/// cannot disagree about whether this plan is compilable.
/// [`DirectMLError::Win32`] when `CreateOperator` or `CompileOperator` fails.
pub(crate) fn compile_binary(
    dml: &IDMLDevice,
    op: BinaryOp,
    layout: &DmlElementwiseLayout,
) -> Result<IDMLCompiledOperator> {
    let b_layout = layout.b.ok_or_else(|| {
        DirectMLError::Declined(format!(
            "{}: binary op has no B operand layout",
            op.as_str()
        ))
    })?;

    // Levels 4 → 2, exactly as in `compile_gemm`; see this module's documentation.
    let a_storage = DmlTensorStorage::new(&layout.a);
    let b_storage = DmlTensorStorage::new(&b_layout);
    let out_storage = DmlTensorStorage::new(&layout.output);

    let a_buffer = a_storage.buffer_desc();
    let b_buffer = b_storage.buffer_desc();
    let out_buffer = out_storage.buffer_desc();

    let a_tensor = BoundTensorDesc::new(&a_buffer);
    let b_tensor = BoundTensorDesc::new(&b_buffer);
    let out_tensor = BoundTensorDesc::new(&out_buffer);

    let a = a_tensor.as_ptr();
    let b = b_tensor.as_ptr();
    let out = out_tensor.as_ptr();

    // Level 1 lives inside each arm.  The arm's block does not end until
    // `create_and_compile` has returned, so the descriptor is alive across the call.
    match op {
        BinaryOp::Add => {
            let desc = DML_ELEMENT_WISE_ADD_OPERATOR_DESC {
                ATensor: a,
                BTensor: b,
                OutputTensor: out,
            };
            // SAFETY: `desc` is a live `DML_ELEMENT_WISE_ADD_OPERATOR_DESC`, which is what
            // `DML_OPERATOR_ELEMENT_WISE_ADD` tags.  `a` / `b` / `out` point at the
            // `BoundTensorDesc` locals above, whose own chains reach the storage locals
            // above those; all of them outlive this call.
            unsafe {
                create_and_compile(
                    dml,
                    DML_OPERATOR_ELEMENT_WISE_ADD,
                    ptr::addr_of!(desc).cast::<c_void>(),
                    SITE_ADD,
                )
            }
        }
        BinaryOp::Sub => {
            let desc = DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC {
                ATensor: a,
                BTensor: b,
                OutputTensor: out,
            };
            // SAFETY: as the `Add` arm, for `DML_OPERATOR_ELEMENT_WISE_SUBTRACT`.
            unsafe {
                create_and_compile(
                    dml,
                    DML_OPERATOR_ELEMENT_WISE_SUBTRACT,
                    ptr::addr_of!(desc).cast::<c_void>(),
                    SITE_SUBTRACT,
                )
            }
        }
        BinaryOp::Mul => {
            let desc = DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC {
                ATensor: a,
                BTensor: b,
                OutputTensor: out,
            };
            // SAFETY: as the `Add` arm, for `DML_OPERATOR_ELEMENT_WISE_MULTIPLY`.
            unsafe {
                create_and_compile(
                    dml,
                    DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                    ptr::addr_of!(desc).cast::<c_void>(),
                    SITE_MULTIPLY,
                )
            }
        }
        BinaryOp::Div => {
            let desc = DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC {
                ATensor: a,
                BTensor: b,
                OutputTensor: out,
            };
            // SAFETY: as the `Add` arm, for `DML_OPERATOR_ELEMENT_WISE_DIVIDE`.
            unsafe {
                create_and_compile(
                    dml,
                    DML_OPERATOR_ELEMENT_WISE_DIVIDE,
                    ptr::addr_of!(desc).cast::<c_void>(),
                    SITE_DIVIDE,
                )
            }
        }
    }
}

// ─── elementwise unary ───────────────────────────────────────────────────────

/// `DML_OPERATOR_ACTIVATION_{RELU,SIGMOID,TANH}`.
///
/// The **`ACTIVATION_`** family, not the `ELEMENT_WISE_` one.  Both exist for `Tanh`, and
/// they are not interchangeable: `DML_ELEMENT_WISE_TANH_OPERATOR_DESC` carries a third
/// `ScaleBias: *const DML_SCALE_BIAS` field, so building one from a two-field literal is
/// not even possible, and reaching for it here would mean carrying a scale/bias this
/// crate has no use for.  `DML_ACTIVATION_TANH_OPERATOR_DESC` is `{ InputTensor,
/// OutputTensor }`, which is precisely ONNX `Tanh`.
///
/// `layout.b`, if present, is ignored — a unary plan never populates it, and
/// [`crate::layout::OpCacheKey::unary`] ignores it too, so the cache key and the
/// compiled operator read exactly the same inputs.
///
/// # Errors
/// [`DirectMLError::Win32`] when `CreateOperator` or `CompileOperator` fails;
/// [`DirectMLError::DispatchFailed`] on a null-but-successful return.
pub(crate) fn compile_unary(
    dml: &IDMLDevice,
    op: UnaryOp,
    layout: &DmlElementwiseLayout,
) -> Result<IDMLCompiledOperator> {
    // Levels 4 → 2, exactly as in `compile_gemm`; see this module's documentation.
    let a_storage = DmlTensorStorage::new(&layout.a);
    let out_storage = DmlTensorStorage::new(&layout.output);

    let a_buffer = a_storage.buffer_desc();
    let out_buffer = out_storage.buffer_desc();

    let a_tensor = BoundTensorDesc::new(&a_buffer);
    let out_tensor = BoundTensorDesc::new(&out_buffer);

    let input = a_tensor.as_ptr();
    let out = out_tensor.as_ptr();

    match op {
        UnaryOp::Relu => {
            let desc = DML_ACTIVATION_RELU_OPERATOR_DESC {
                InputTensor: input,
                OutputTensor: out,
            };
            // SAFETY: `desc` is a live `DML_ACTIVATION_RELU_OPERATOR_DESC`, which is what
            // `DML_OPERATOR_ACTIVATION_RELU` tags.  `input` / `out` point at the
            // `BoundTensorDesc` locals above, whose chains reach the storage locals above
            // those; all of them outlive this call.
            unsafe {
                create_and_compile(
                    dml,
                    DML_OPERATOR_ACTIVATION_RELU,
                    ptr::addr_of!(desc).cast::<c_void>(),
                    SITE_RELU,
                )
            }
        }
        UnaryOp::Sigmoid => {
            let desc = DML_ACTIVATION_SIGMOID_OPERATOR_DESC {
                InputTensor: input,
                OutputTensor: out,
            };
            // SAFETY: as the `Relu` arm, for `DML_OPERATOR_ACTIVATION_SIGMOID`.
            unsafe {
                create_and_compile(
                    dml,
                    DML_OPERATOR_ACTIVATION_SIGMOID,
                    ptr::addr_of!(desc).cast::<c_void>(),
                    SITE_SIGMOID,
                )
            }
        }
        UnaryOp::Tanh => {
            let desc = DML_ACTIVATION_TANH_OPERATOR_DESC {
                InputTensor: input,
                OutputTensor: out,
            };
            // SAFETY: as the `Relu` arm, for `DML_OPERATOR_ACTIVATION_TANH`.
            unsafe {
                create_and_compile(
                    dml,
                    DML_OPERATOR_ACTIVATION_TANH,
                    ptr::addr_of!(desc).cast::<c_void>(),
                    SITE_TANH,
                )
            }
        }
    }
}

// ─── softmax ─────────────────────────────────────────────────────────────────

/// `DML_OPERATOR_ACTIVATION_SOFTMAX` — the **axis-less** softmax, which normalises the
/// tensor's innermost dimension.
///
/// windows-0.62.2 exposes only this form; there is no `DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC`
/// with an explicit `AxisCount` / `Axes`, so a softmax over a *non-innermost* axis cannot be
/// expressed here.  [`super::dml_backend::DmlEngine::softmax`] declines that case to the
/// CPU/HLSL path **before** it reaches this function, using
/// [`crate::plan::SoftmaxPlan::reduces_last_axis`]; every plan that gets here has its
/// softmax axis innermost, and [`crate::layout::DmlTensorLayout::packed`]'s left-pad keeps
/// it the innermost rank-4 dimension, so the axis-less operator normalises exactly the axis
/// ONNX asked for.
///
/// Softmax is shape-preserving, so the input and output share one [`DmlTensorLayout`].
///
/// # Errors
/// [`DirectMLError::Declined`] when the shape is not rank-4-describable (rank > 5 after
/// padding — declined by `packed`); [`DirectMLError::Win32`] when `CreateOperator` or
/// `CompileOperator` fails; [`DirectMLError::DispatchFailed`] on a null-but-successful
/// return.
pub(crate) fn compile_softmax(
    dml: &IDMLDevice,
    plan: &SoftmaxPlan,
) -> Result<IDMLCompiledOperator> {
    // Levels 4 → 2, exactly as in `compile_gemm`; see this module's documentation.  Input
    // and output are the same shape, so the same layout backs both descriptors.
    let layout = DmlTensorLayout::packed(&plan.shape)?;
    let in_storage = DmlTensorStorage::new(&layout);
    let out_storage = DmlTensorStorage::new(&layout);

    let in_buffer = in_storage.buffer_desc();
    let out_buffer = out_storage.buffer_desc();

    let in_tensor = BoundTensorDesc::new(&in_buffer);
    let out_tensor = BoundTensorDesc::new(&out_buffer);

    let desc = DML_ACTIVATION_SOFTMAX_OPERATOR_DESC {
        InputTensor: in_tensor.as_ptr(),
        OutputTensor: out_tensor.as_ptr(),
    };

    // SAFETY: `desc` is a live `DML_ACTIVATION_SOFTMAX_OPERATOR_DESC`, which is what
    // `DML_OPERATOR_ACTIVATION_SOFTMAX` tags.  `InputTensor` / `OutputTensor` point at the
    // `BoundTensorDesc` locals above, whose chains reach the storage locals above those;
    // all of them outlive this call.
    unsafe {
        create_and_compile(
            dml,
            DML_OPERATOR_ACTIVATION_SOFTMAX,
            ptr::addr_of!(desc).cast::<c_void>(),
            SITE_SOFTMAX,
        )
    }
}

// ─── reduce ──────────────────────────────────────────────────────────────────

/// `DML_REDUCE_OPERATOR_DESC::Function` for a [`ReduceKind`].
///
/// `Mean` maps to `DML_REDUCE_FUNCTION_AVERAGE`, DirectML's name for the same thing.  The
/// `L1` / `L2` / `LOG_SUM` / `MULTIPLY` / `ARGMAX` functions have no [`ReduceKind`], so they
/// cannot be produced here.
const fn reduce_function(kind: ReduceKind) -> DML_REDUCE_FUNCTION {
    match kind {
        ReduceKind::Sum => DML_REDUCE_FUNCTION_SUM,
        ReduceKind::Mean => DML_REDUCE_FUNCTION_AVERAGE,
        ReduceKind::Max => DML_REDUCE_FUNCTION_MAX,
        ReduceKind::Min => DML_REDUCE_FUNCTION_MIN,
    }
}

/// `DML_OPERATOR_REDUCE` over a **single** axis.
///
/// The output tensor is the input sizes with the reduced axis collapsed to 1 — DirectML's
/// documented `OutputTensor` rule — and `Axes` is the single rank-4 axis.  Both come from
/// [`crate::layout::dml_reduce_layouts`], the same derivation [`crate::layout::OpCacheKey::reduce`]
/// keys on, so the compiled operator and its cache key can never describe different
/// reductions.  ONNX `keepdims` never reaches here: it changes the logical output rank, not
/// the buffer, so it is a router concern only.
///
/// # Errors
/// [`DirectMLError::Declined`] when the input is not rank-4-describable (rank > 4 — the CPU
/// kernel handles it); [`DirectMLError::Win32`] when `CreateOperator` or `CompileOperator`
/// fails; [`DirectMLError::DispatchFailed`] on a null-but-successful return.
pub(crate) fn compile_reduce(dml: &IDMLDevice, plan: &ReducePlan) -> Result<IDMLCompiledOperator> {
    let (input, rank4_axis, output) = dml_reduce_layouts(plan)?;

    // Levels 4 → 2, exactly as in `compile_gemm`; see this module's documentation.
    let in_storage = DmlTensorStorage::new(&input);
    let out_storage = DmlTensorStorage::new(&output);

    let in_buffer = in_storage.buffer_desc();
    let out_buffer = out_storage.buffer_desc();

    let in_tensor = BoundTensorDesc::new(&in_buffer);
    let out_tensor = BoundTensorDesc::new(&out_buffer);

    // The `Axes[]` array — a named local, so it outlives the `CreateOperator` call, exactly
    // like a tensor's `Sizes[]` does.  DirectML deep-copies it, but only during the call.
    let axes = [rank4_axis];

    let desc = DML_REDUCE_OPERATOR_DESC {
        Function: reduce_function(plan.kind),
        InputTensor: in_tensor.as_ptr(),
        OutputTensor: out_tensor.as_ptr(),
        AxisCount: 1,
        Axes: axes.as_ptr(),
    };

    // SAFETY: `desc` is a live `DML_REDUCE_OPERATOR_DESC`, which is what `DML_OPERATOR_REDUCE`
    // tags.  Its two tensor pointers reach the `BoundTensorDesc` / buffer / storage locals
    // above, and `Axes` points at the `axes` local; every one of them is a named local of
    // this function and so is alive and unmoved across the call.  `AxisCount == 1` matches
    // the one-element `axes` array.
    unsafe {
        create_and_compile(
            dml,
            DML_OPERATOR_REDUCE,
            ptr::addr_of!(desc).cast::<c_void>(),
            SITE_REDUCE,
        )
    }
}

// ─── convolution ─────────────────────────────────────────────────────────────

/// `DML_CONVOLUTION_OPERATOR_DESC::DimensionCount` — the number of **spatial** dimensions.
///
/// This crate plans 2-D convolutions only, so the strides / dilations / start- and
/// end-padding arrays each carry exactly two entries `[h, w]`, while the tensors stay rank
/// 4 (`2 + DimensionCount`).
const CONV_SPATIAL_DIMS: u32 = 2;

/// `DML_OPERATOR_CONVOLUTION`, forward, cross-correlation — ONNX `Conv` exactly.
///
/// # `Mode` is cross-correlation, not mathematical convolution
///
/// ONNX `Conv` (like PyTorch and TensorFlow) applies the kernel **without flipping** it —
/// it is a cross-correlation.  [`crate::reference::ref_conv`], the CPU oracle this operator
/// is diffed against, indexes the weight in the same direction as the input and so is also
/// a cross-correlation.  DirectML's `DML_CONVOLUTION_MODE_CONVOLUTION` flips the kernel and
/// would disagree with the oracle on every non-symmetric filter; `DML_CONVOLUTION_MODE_CROSS_CORRELATION`
/// is the one that matches, and the one ONNX Runtime's DirectML EP uses for `Conv`.  This
/// is a deliberate departure from a mode name; using the other would be a silent
/// wrong-answer.
///
/// `Direction` is forward and `OutputPadding` is all-zero — output padding is meaningful
/// only for the backward (transposed) direction.  `FusedActivation` is null, for the same
/// reason [`compile_gemm`] never fuses: fusion would change the numbers the oracle checks.
///
/// The tensor layouts come from [`crate::layout::DmlTensorLayout::packed`] (input, filter,
/// output) and [`crate::layout::dml_conv_bias_layout`] (the `[1, C_out, 1, 1]` bias), the
/// same derivations [`crate::layout::OpCacheKey::conv`] keys on.  When the plan has no bias,
/// `BiasTensor` is null.
///
/// # Errors
/// [`DirectMLError::Declined`] on the size limits of the layout constructors;
/// [`DirectMLError::Win32`] when `CreateOperator` or `CompileOperator` fails;
/// [`DirectMLError::DispatchFailed`] on a null-but-successful return.
pub(crate) fn compile_conv(dml: &IDMLDevice, plan: &ConvPlan) -> Result<IDMLCompiledOperator> {
    let input = DmlTensorLayout::packed(&plan.input_shape)?;
    let filter = DmlTensorLayout::packed(&plan.weight_shape)?;
    let output = DmlTensorLayout::packed(&plan.output_shape)?;
    let bias = if plan.has_bias {
        Some(dml_conv_bias_layout(plan)?)
    } else {
        None
    };

    // Levels 4 → 2, exactly as in `compile_gemm`; the optional bias mirrors GEMM's `C`.
    let in_storage = DmlTensorStorage::new(&input);
    let filter_storage = DmlTensorStorage::new(&filter);
    let bias_storage = bias.as_ref().map(DmlTensorStorage::new);
    let out_storage = DmlTensorStorage::new(&output);

    let in_buffer = in_storage.buffer_desc();
    let filter_buffer = filter_storage.buffer_desc();
    let bias_buffer = bias_storage.as_ref().map(DmlTensorStorage::buffer_desc);
    let out_buffer = out_storage.buffer_desc();

    let in_tensor = BoundTensorDesc::new(&in_buffer);
    let filter_tensor = BoundTensorDesc::new(&filter_buffer);
    let bias_tensor = bias_buffer.as_ref().map(BoundTensorDesc::new);
    let out_tensor = BoundTensorDesc::new(&out_buffer);

    // The four spatial attribute arrays plus the zero output-padding — named locals, so they
    // outlive `CreateOperator`, exactly like a tensor's `Sizes[]`.  Every entry is an
    // already-`u32` plan field: nothing here is a shape-derived `as` cast.
    let strides = [plan.stride_h, plan.stride_w];
    let dilations = [plan.dilation_h, plan.dilation_w];
    let start_padding = [plan.pad_top, plan.pad_left];
    let end_padding = [plan.pad_bottom, plan.pad_right];
    let output_padding = [0u32, 0u32];

    let desc = DML_CONVOLUTION_OPERATOR_DESC {
        InputTensor: in_tensor.as_ptr(),
        FilterTensor: filter_tensor.as_ptr(),
        BiasTensor: bias_tensor
            .as_ref()
            .map_or(ptr::null(), BoundTensorDesc::as_ptr),
        OutputTensor: out_tensor.as_ptr(),
        Mode: DML_CONVOLUTION_MODE_CROSS_CORRELATION,
        Direction: DML_CONVOLUTION_DIRECTION_FORWARD,
        DimensionCount: CONV_SPATIAL_DIMS,
        Strides: strides.as_ptr(),
        Dilations: dilations.as_ptr(),
        StartPadding: start_padding.as_ptr(),
        EndPadding: end_padding.as_ptr(),
        OutputPadding: output_padding.as_ptr(),
        GroupCount: plan.group,
        FusedActivation: ptr::null(),
    };

    // SAFETY: `desc` is a live `DML_CONVOLUTION_OPERATOR_DESC`, which is what
    // `DML_OPERATOR_CONVOLUTION` tags.  Its four tensor pointers reach the `BoundTensorDesc`
    // / buffer / storage locals above (the bias null iff the plan has none), and its five
    // spatial arrays point at the `[u32; 2]` locals above; every one of them is a named
    // local of this function, alive and unmoved across the call.  `DimensionCount == 2`
    // matches the two-element arrays.  `BiasTensor` and `FusedActivation` are null when
    // absent, which DirectML defines as "not supplied".
    unsafe {
        create_and_compile(
            dml,
            DML_OPERATOR_CONVOLUTION,
            ptr::addr_of!(desc).cast::<c_void>(),
            SITE_CONV,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{matrix_transform, reduce_function, CallSite};
    use crate::plan::ReduceKind;
    use windows::Win32::AI::MachineLearning::DirectML::{
        DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE, DML_REDUCE_FUNCTION_AVERAGE,
        DML_REDUCE_FUNCTION_MAX, DML_REDUCE_FUNCTION_MIN, DML_REDUCE_FUNCTION_SUM,
    };

    /// These tests are compiled (and type-checked by
    /// `cargo clippy --target x86_64-pc-windows-gnu --all-targets`) on Windows only, and
    /// executed only on a Windows host.  They cover the pure logic in this module; the
    /// FFI itself is unreachable without a D3D12 device, and is covered by
    /// `DirectMLContext::self_check` on real hardware.
    #[test]
    fn matrix_transform_maps_the_flag() {
        assert_eq!(matrix_transform(false), DML_MATRIX_TRANSFORM_NONE);
        assert_eq!(matrix_transform(true), DML_MATRIX_TRANSFORM_TRANSPOSE);
    }

    /// Every call site must name the operator it belongs to, so that a
    /// `DirectMLError::Win32` identifies *which* of the eleven operators failed.
    #[test]
    fn every_call_site_names_its_operator() {
        for site in [
            super::SITE_GEMM,
            super::SITE_ADD,
            super::SITE_SUBTRACT,
            super::SITE_MULTIPLY,
            super::SITE_DIVIDE,
            super::SITE_RELU,
            super::SITE_SIGMOID,
            super::SITE_TANH,
            super::SITE_SOFTMAX,
            super::SITE_REDUCE,
            super::SITE_CONV,
        ] {
            let CallSite {
                name,
                create,
                compile,
            } = site;
            assert!(name.starts_with("DML_OPERATOR_"), "{name}");
            assert!(create.contains(name), "{create} must name {name}");
            assert!(compile.contains(name), "{compile} must name {name}");
            assert!(create.contains("CreateOperator"), "{create}");
            assert!(compile.contains("CompileOperator"), "{compile}");
        }
    }

    /// The `ReduceKind` → `DML_REDUCE_FUNCTION` map is 1:1, and `Mean` is DirectML's
    /// `AVERAGE`.  A mis-map would compute the wrong reduction with no `HRESULT`.
    #[test]
    fn reduce_function_maps_each_kind() {
        assert_eq!(reduce_function(ReduceKind::Sum), DML_REDUCE_FUNCTION_SUM);
        assert_eq!(
            reduce_function(ReduceKind::Mean),
            DML_REDUCE_FUNCTION_AVERAGE
        );
        assert_eq!(reduce_function(ReduceKind::Max), DML_REDUCE_FUNCTION_MAX);
        assert_eq!(reduce_function(ReduceKind::Min), DML_REDUCE_FUNCTION_MIN);
    }
}
