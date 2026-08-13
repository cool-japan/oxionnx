//! CUDA-accelerated elementwise operator dispatch.
//!
//! Covers unary ONNX ops (`Relu`, `Sigmoid`, `Gelu`, `Tanh`, `Exp`, `Sqrt`,
//! `Abs`, `Neg`, `Log`, `Ceil`, `Floor`, `HardSigmoid`, `HardSwish`, `SiLU`,
//! `Softplus`, `LeakyRelu`) and binary ops (`Add`, `Sub`, `Mul`, `Div`).
//!
//! Each op has a PTX kernel generated from [`ElementwiseTemplate`], compiled
//! **once per context** (see `CudaContext`'s module cache) and launched with
//! the input/output device pointers as arguments.
//!
//! # Per-call cost, and what is left of it
//!
//! These are the cheapest kernels in the crate and used to be among its most
//! expensive dispatches, because everything around the kernel dominated it.
//! Two of those things are now gone:
//!
//! * **The JIT.** `template.generate()` + `Module::from_ptx` ran on *every*
//!   dispatch — a PTX string rebuilt and handed to the driver's compiler per
//!   node per frame, for the 57 elementwise nodes an InSwapper frame runs.
//!   The context's module cache makes that a once-per-context cost.
//! * **The allocations and the fences.** Two or three `cuMemAlloc`/`cuMemFree`
//!   pairs plus a context-wide synchronise per operand, replaced by pooled
//!   buffers and stream-ordered copies (see [`mod@crate::residency`]).
//!
//! What is left is the unavoidable part: the operand crosses the bus, the
//! kernel runs, the result crosses back. That round trip is why `oxionnx`'s
//! placement layer gates memory-bound ops on size in the first place — see
//! `oxionnx::session::gpu_residency`'s measured table, where seven of ten op
//! types lose to their CPU kernels *while transferring*. Removing the transfer
//! itself needs session-level activation residency, which is a layer above
//! this crate.
//!
//! Operands here are deliberately **not** weight-cached. An elementwise
//! operand is essentially always an activation, and the rare initializer one
//! (a per-channel constant in an `Add`) is a few hundred bytes — the residency
//! bookkeeping would cost more than the upload it saves. `MatMul`/`Gemm` and
//! `Conv`, whose initializers are megabytes each, do cache theirs.

use oxicuda_launch::{grid_size_for, Dim3, Kernel, LaunchParams};
use oxicuda_ptx::{
    ir::PtxType,
    templates::elementwise::{ElementwiseOp, ElementwiseTemplate},
};

use crate::context::CudaContext;
use crate::error::CudaDispatchError;

const BLOCK_SIZE: u32 = 256;

/// Map an ONNX unary op name to the corresponding [`ElementwiseOp`].
fn unary_op_for(op_name: &str) -> Result<ElementwiseOp, CudaDispatchError> {
    match op_name {
        "Relu" => Ok(ElementwiseOp::Relu),
        "Sigmoid" => Ok(ElementwiseOp::Sigmoid),
        "Gelu" => Ok(ElementwiseOp::Gelu),
        "Tanh" => Ok(ElementwiseOp::Tanh),
        "Exp" => Ok(ElementwiseOp::Exp),
        "Sqrt" => Ok(ElementwiseOp::Sqrt),
        "Abs" => Ok(ElementwiseOp::Abs),
        "Neg" => Ok(ElementwiseOp::Neg),
        "Log" => Ok(ElementwiseOp::Log),
        "Ceil" => Ok(ElementwiseOp::Ceil),
        "Floor" => Ok(ElementwiseOp::Floor),
        "HardSigmoid" => Ok(ElementwiseOp::HardSigmoid),
        "HardSwish" => Ok(ElementwiseOp::HardSwish),
        "Silu" | "SiLU" => Ok(ElementwiseOp::Silu),
        "Softplus" => Ok(ElementwiseOp::Softplus),
        "LeakyRelu" => Ok(ElementwiseOp::LeakyRelu),
        other => Err(CudaDispatchError::Unsupported {
            op: "elementwise",
            reason: format!("no CUDA kernel for ONNX op '{other}'"),
        }),
    }
}

/// Map an ONNX binary op name to the corresponding [`ElementwiseOp`].
fn binary_op_for(op_name: &str) -> Result<ElementwiseOp, CudaDispatchError> {
    match op_name {
        "Add" => Ok(ElementwiseOp::Add),
        "Sub" => Ok(ElementwiseOp::Sub),
        "Mul" => Ok(ElementwiseOp::Mul),
        "Div" => Ok(ElementwiseOp::Div),
        other => Err(CudaDispatchError::Unsupported {
            op: "binary_elementwise",
            reason: format!("no CUDA binary kernel for ONNX op '{other}'"),
        }),
    }
}

/// Fetch — compiling on first use — the kernel for one elementwise op.
///
/// The template is cheap to construct and its `kernel_name` is the cache key,
/// so a hit costs a hash lookup and an `Arc` clone; only a miss pays for PTX
/// generation and the driver's JIT.
pub(crate) fn kernel_for(
    ctx: &CudaContext,
    op: ElementwiseOp,
) -> Result<Kernel, CudaDispatchError> {
    let template = ElementwiseTemplate::new(op, PtxType::F32, ctx.dnn.sm_version());
    let kernel_name = template.kernel_name();
    let module = ctx.module(&kernel_name, || {
        template
            .generate()
            .map_err(|e| CudaDispatchError::Ptx(e.to_string()))
    })?;
    Kernel::from_module(module, &kernel_name).map_err(CudaDispatchError::Driver)
}

/// Launch a unary elementwise kernel on the CUDA device.
///
/// `op_name` is the ONNX op type string.
///
/// Returns the output data vector on success.
///
/// # Errors
///
/// [`CudaDispatchError::Unsupported`] for an op with no kernel,
/// [`CudaDispatchError::Shape`] for a tensor too large for a `u32` launch, or
/// a driver error from PTX compilation, allocation, upload, launch or
/// readback.
pub fn cuda_elementwise(
    ctx: &CudaContext,
    data: &[f32],
    op_name: &str,
) -> Result<Vec<f32>, CudaDispatchError> {
    let kernel = kernel_for(ctx, unary_op_for(op_name)?)?;

    let n = data.len();
    let Ok(n_u32) = u32::try_from(n) else {
        return Err(CudaDispatchError::Shape {
            op: "elementwise",
            msg: format!("{n} elements exceed a u32 kernel launch"),
        });
    };

    // Upload, launch and readback all ride `ctx.dnn.stream()` — the stream the
    // kernel is launched on. Stream order alone sequences them, so the only
    // host/device rendezvous is the single synchronise at the end.
    let stream = ctx.dnn.stream();
    let mut d_input = ctx.scratch(n)?;
    d_input.upload(data, stream)?;
    let mut d_output = ctx.scratch(n)?;

    // No zero-fill: the grid covers `[0, n)` and the kernel writes every
    // element of that range, so nothing a previous borrower left is ever read.
    // (The pooled allocation's tail beyond `n` stays stale and is never
    // touched — `download` copies exactly `n`.)
    let grid = grid_size_for(n_u32, BLOCK_SIZE);
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(BLOCK_SIZE));
    let args = (d_input.device_ptr(), d_output.device_ptr(), n_u32);
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    let mut out = vec![0.0_f32; n];
    d_output.download(&mut out, stream)?;
    stream.synchronize().map_err(CudaDispatchError::Driver)?;
    // ...and only now may these allocations go back to the pool. See
    // `PooledBuffer`'s "a borrow is only recycled once its stream work is
    // known to be done".
    d_input.retire();
    d_output.retire();
    Ok(out)
}

/// Launch a unary elementwise kernel **in place** over a buffer that is
/// already on the device, and do *not* synchronise.
///
/// This is the epilogue form of [`cuda_elementwise`]: no upload, no readback,
/// no fence. Everything is queued on `ctx.dnn.stream()`, so stream order alone
/// sequences it behind whatever produced `device_ptr` and ahead of whatever
/// reads it — which is what lets [`crate::conv::cuda_conv_cached`] fold a
/// fused activation into a convolution it has already launched, without adding
/// a host/device rendezvous to the dispatch — and what would keep that sequence
/// legal inside a CUDA graph capture.
///
/// In-place aliasing is safe for every op in [`unary_op_for`]'s set: the
/// generated kernel gives thread `i` exactly one read of `in[i]` and one write
/// of `out[i]`, so passing the same pointer for both makes each element
/// thread-private. Do **not** reuse this for an op whose kernel reads a
/// neighbour.
///
/// # Errors
///
/// [`CudaDispatchError::Shape`] when `n` exceeds a `u32` launch, or a driver
/// error from PTX compilation or the launch itself.
pub(crate) fn launch_unary_in_place(
    ctx: &CudaContext,
    op: ElementwiseOp,
    device_ptr: oxicuda_driver::ffi::CUdeviceptr,
    n: usize,
) -> Result<(), CudaDispatchError> {
    if n == 0 {
        return Ok(());
    }
    let Ok(n_u32) = u32::try_from(n) else {
        return Err(CudaDispatchError::Shape {
            op: "elementwise_in_place",
            msg: format!("{n} elements exceed a u32 kernel launch"),
        });
    };
    let kernel = kernel_for(ctx, op)?;
    let grid = grid_size_for(n_u32, BLOCK_SIZE);
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(BLOCK_SIZE));
    let args = (device_ptr, device_ptr, n_u32);
    kernel
        .launch(&params, ctx.dnn.stream(), &args)
        .map_err(CudaDispatchError::Driver)
}

/// Launch a binary elementwise kernel (Add, Sub, Mul, Div) on the CUDA device.
///
/// Both `a` and `b` must have the same length.  Returns the output data vector.
///
/// # Errors
///
/// [`CudaDispatchError::Shape`] if the operands differ in length or are too
/// large for a `u32` launch, [`CudaDispatchError::Unsupported`] for an op with
/// no kernel, or a driver error from PTX compilation, allocation, upload,
/// launch or readback.
pub fn cuda_binary_elementwise(
    ctx: &CudaContext,
    a: &[f32],
    b: &[f32],
    op_name: &str,
) -> Result<Vec<f32>, CudaDispatchError> {
    if a.len() != b.len() {
        return Err(CudaDispatchError::Shape {
            op: "binary_elementwise",
            msg: format!(
                "binary elementwise requires equal-length inputs, got {} vs {}",
                a.len(),
                b.len()
            ),
        });
    }

    let kernel = kernel_for(ctx, binary_op_for(op_name)?)?;

    let n = a.len();
    let Ok(n_u32) = u32::try_from(n) else {
        return Err(CudaDispatchError::Shape {
            op: "binary_elementwise",
            msg: format!("{n} elements exceed a u32 kernel launch"),
        });
    };

    let stream = ctx.dnn.stream();
    let mut d_a = ctx.scratch(n)?;
    d_a.upload(a, stream)?;
    let mut d_b = ctx.scratch(n)?;
    d_b.upload(b, stream)?;
    let mut d_output = ctx.scratch(n)?;

    let grid = grid_size_for(n_u32, BLOCK_SIZE);
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(BLOCK_SIZE));
    let args = (
        d_a.device_ptr(),
        d_b.device_ptr(),
        d_output.device_ptr(),
        n_u32,
    );
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    let mut out = vec![0.0_f32; n];
    d_output.download(&mut out, stream)?;
    stream.synchronize().map_err(CudaDispatchError::Driver)?;
    d_a.retire();
    d_b.retire();
    d_output.retire();
    Ok(out)
}
