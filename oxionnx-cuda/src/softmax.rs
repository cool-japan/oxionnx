//! CUDA-accelerated Softmax dispatch.
//!
//! Uses [`SoftmaxTemplate`] to generate a PTX kernel where each CUDA block
//! processes one row of the input matrix.  The kernel handles up to 1024
//! elements per row; larger rows fall back to `Ok(None)`.
//!
//! The ONNX `axis` attribute determines what constitutes a "row": all
//! dimensions from `axis` to the end form the row, and all leading dimensions
//! form the batch.  We support 2-D tensors only (last axis == the row axis).

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::{ir::PtxType, templates::softmax::SoftmaxTemplate};

use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// GPU softmax over the last axis.
///
/// `shape` must have at least one dimension.  The kernel treats
/// `shape[..shape.len()-1]` as the batch dimension and `shape[shape.len()-1]`
/// as the row width.
///
/// Returns `Ok(None)` when `row_size > 1024` (template limit).
pub fn cuda_softmax(
    ctx: &CudaContext,
    data: &[f32],
    shape: &[usize],
) -> Result<Option<Vec<f32>>, CudaDispatchError> {
    if shape.is_empty() {
        return Ok(None);
    }

    let row_size = match shape.last() {
        Some(&s) => s as u32,
        None => return Ok(None),
    };
    if row_size > 1024 {
        return Ok(None);
    }

    let batch_size: u32 = shape[..shape.len() - 1].iter().product::<usize>().max(1) as u32;

    let sm = ctx.dnn.sm_version();
    let template = SoftmaxTemplate {
        precision: PtxType::F32,
        target: sm,
        row_size,
    };
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

    // The warp-shuffle kernel (row_size <= 32) maps one warp (32 threads) per
    // row, so we must launch 32 threads/block regardless of row_size.  The
    // shared-memory kernel (row_size > 32) uses row_size.next_power_of_two().min(256).
    let block_threads = if row_size <= 32 {
        32u32
    } else {
        row_size.next_power_of_two().min(256)
    };
    let params = LaunchParams::new(Dim3::from(batch_size), Dim3::from(block_threads));

    let stream = ctx.dnn.stream();
    let args = (
        d_input.as_device_ptr(),
        d_output.as_device_ptr(),
        batch_size,
    );
    kernel
        .launch(&params, stream, &args)
        .map_err(CudaDispatchError::Driver)?;

    stream.synchronize().map_err(CudaDispatchError::Driver)?;

    let mut out = vec![0.0_f32; n];
    d_output.copy_to_host(&mut out)?;
    Ok(Some(out))
}
