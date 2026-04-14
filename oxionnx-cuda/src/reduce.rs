//! CUDA-accelerated ReduceSum / ReduceMax dispatch.
//!
//! The underlying PTX template performs a flat N-to-1 block reduction (3 kernel
//! parameters: `input_ptr`, `output_ptr`, `n`).  We therefore only accelerate
//! the case where `axis` is the *only* axis and the slice to reduce covers the
//! entire tensor (i.e. `outer == 1 && inner == 1`).  All other configurations
//! return `Ok(None)` and fall back to the CPU path.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::{
    ir::PtxType,
    templates::reduction::{ReductionOp, ReductionTemplate},
};

use crate::context::CudaContext;
use crate::error::CudaDispatchError;

const REDUCE_BLOCK_SIZE: u32 = 256;

/// GPU reduction for a single axis.
///
/// Returns `Ok(None)` when the configuration is not handled by the flat
/// reduction template (non-trivial outer or inner dimensions).  The caller
/// should fall back to CPU in that case.
pub fn cuda_reduce(
    ctx: &CudaContext,
    data: &[f32],
    shape: &[usize],
    axis: usize,
    op_name: &str,
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    if axis >= shape.len() {
        return Ok(None);
    }

    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let axis_len = shape[axis];

    // Only handle the flat full-tensor case.
    if outer != 1 || inner != 1 {
        return Ok(None);
    }

    let reduce_op = match op_name {
        "ReduceSum" => ReductionOp::Sum,
        "ReduceMax" => ReductionOp::Max,
        other => {
            return Err(CudaDispatchError::Unsupported {
                op: "reduce",
                reason: format!("no CUDA reduction kernel for ONNX op '{other}'"),
            });
        }
    };

    let sm = ctx.dnn.sm_version();
    let template = ReductionTemplate {
        op: reduce_op,
        precision: PtxType::F32,
        target: sm,
        block_size: REDUCE_BLOCK_SIZE,
    };
    let kernel_name = template.kernel_name();

    let ptx = template
        .generate()
        .map_err(|e| CudaDispatchError::Ptx(e.to_string()))?;

    let module = Arc::new(Module::from_ptx(&ptx).map_err(CudaDispatchError::Driver)?);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(CudaDispatchError::Driver)?;

    let n = axis_len;
    let mut d_input: DeviceBuffer<f32> = DeviceBuffer::alloc(n)?;
    d_input.copy_from_host(data)?;

    // Output is a single scalar.
    let d_output: DeviceBuffer<f32> = DeviceBuffer::zeroed(1)?;

    // Launch a single block large enough to cover `n` elements.
    let block = REDUCE_BLOCK_SIZE;
    let grid: u32 = 1;
    let params = LaunchParams::new(Dim3::from(grid), Dim3::from(block));

    let stream = ctx.dnn.stream();
    let args = (d_input.as_device_ptr(), d_output.as_device_ptr(), n as u32);
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    stream.synchronize().map_err(CudaDispatchError::Driver)?;

    let mut result = vec![0.0_f32; 1];
    d_output.copy_to_host(&mut result)?;
    Ok(Some(result))
}
