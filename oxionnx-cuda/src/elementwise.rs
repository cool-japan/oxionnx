//! CUDA-accelerated elementwise operator dispatch.
//!
//! Covers unary ONNX ops (`Relu`, `Sigmoid`, `Gelu`, `Tanh`, `Exp`, `Sqrt`,
//! `Abs`, `Neg`, `Log`, `Ceil`, `Floor`, `HardSigmoid`, `HardSwish`, `SiLU`,
//! `Softplus`, `LeakyRelu`) and binary ops (`Add`, `Sub`, `Mul`, `Div`).
//!
//! Each op generates a PTX kernel via [`ElementwiseTemplate`], compiles it
//! into a module, and launches it with the input/output device pointers as
//! arguments.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{grid_size_for, Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
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

/// Launch a unary elementwise kernel on the CUDA device.
///
/// `op_name` is the ONNX op type string.  The PTX template is generated at
/// call time (in practice the driver caches compiled modules).
///
/// Returns the output data vector on success.
pub fn cuda_elementwise(
    ctx: &CudaContext,
    data: &[f32],
    op_name: &str,
) -> Result<Vec<f32>, CudaDispatchError> {
    let ew_op = unary_op_for(op_name)?;

    let sm = ctx.dnn.sm_version();
    let template = ElementwiseTemplate::new(ew_op, PtxType::F32, sm);
    let kernel_name = template.kernel_name();

    let ptx = template
        .generate()
        .map_err(|e| CudaDispatchError::Ptx(e.to_string()))?;

    let module = Arc::new(Module::from_ptx(&ptx).map_err(CudaDispatchError::Driver)?);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(CudaDispatchError::Driver)?;

    let n = data.len();
    let mut d_input: DeviceBuffer<f32> = DeviceBuffer::alloc(n)?;
    d_input.copy_from_host(data)?;
    let d_output: DeviceBuffer<f32> = DeviceBuffer::alloc(n)?;

    let grid = grid_size_for(n as u32, BLOCK_SIZE);
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(BLOCK_SIZE));

    let stream = ctx.dnn.stream();
    let args = (d_input.as_device_ptr(), d_output.as_device_ptr(), n as u32);
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    stream.synchronize().map_err(CudaDispatchError::Driver)?;

    let mut out = vec![0.0_f32; n];
    d_output.copy_to_host(&mut out)?;
    Ok(out)
}

/// Launch a binary elementwise kernel (Add, Sub, Mul, Div) on the CUDA device.
///
/// Both `a` and `b` must have the same length.  Returns the output data vector.
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

    let ew_op = binary_op_for(op_name)?;

    let sm = ctx.dnn.sm_version();
    let template = ElementwiseTemplate::new(ew_op, PtxType::F32, sm);
    let kernel_name = template.kernel_name();

    let ptx = template
        .generate()
        .map_err(|e| CudaDispatchError::Ptx(e.to_string()))?;

    let module = Arc::new(Module::from_ptx(&ptx).map_err(CudaDispatchError::Driver)?);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(CudaDispatchError::Driver)?;

    let n = a.len();
    let mut d_a: DeviceBuffer<f32> = DeviceBuffer::alloc(n)?;
    d_a.copy_from_host(a)?;
    let mut d_b: DeviceBuffer<f32> = DeviceBuffer::alloc(n)?;
    d_b.copy_from_host(b)?;
    let d_output: DeviceBuffer<f32> = DeviceBuffer::alloc(n)?;

    let grid = grid_size_for(n as u32, BLOCK_SIZE);
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(BLOCK_SIZE));

    let stream = ctx.dnn.stream();
    let args = (
        d_a.as_device_ptr(),
        d_b.as_device_ptr(),
        d_output.as_device_ptr(),
        n as u32,
    );
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    stream.synchronize().map_err(CudaDispatchError::Driver)?;

    let mut out = vec![0.0_f32; n];
    d_output.copy_to_host(&mut out)?;
    Ok(out)
}
