//! CUDA-accelerated Softmax dispatch.
//!
//! Uses [`SoftmaxTemplate`] to generate a PTX kernel where each CUDA block
//! processes one row of the input matrix.  The kernel handles up to 1024
//! elements per row; larger rows fall back to `Ok(None)`.
//!
//! The ONNX `axis` attribute determines what constitutes a "row": all
//! dimensions from `axis` to the end form the row, and all leading dimensions
//! form the batch.  We support 2-D tensors only (last axis == the row axis).
//!
//! # Kernel caching is per row width, and that is deliberate
//!
//! `SoftmaxTemplate` bakes `row_size` into the kernel (it decides between the
//! warp-shuffle and shared-memory variants, and sizes the reduction), so the
//! generated `kernel_name` differs per width — which makes it exactly the
//! right cache key. A graph with three distinct softmax widths compiles three
//! modules once each, instead of one module per dispatch as this used to.

use oxicuda_launch::{Dim3, Kernel, LaunchParams};
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
///
/// # Errors
///
/// A driver error from PTX compilation, allocation, upload, launch or
/// readback.
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

    let template = SoftmaxTemplate {
        precision: PtxType::F32,
        target: ctx.dnn.sm_version(),
        row_size,
    };
    let kernel_name = template.kernel_name();
    let module = ctx.module(&kernel_name, || {
        template
            .generate()
            .map_err(|e| CudaDispatchError::Ptx(e.to_string()))
    })?;
    let kernel = Kernel::from_module(module, &kernel_name).map_err(CudaDispatchError::Driver)?;

    // Upload, launch and readback all ride the stream the kernel runs on, so
    // stream order sequences them and the single synchronise at the end is the
    // only host/device rendezvous.
    let stream = ctx.dnn.stream();
    let n = data.len();
    let mut d_input = ctx.scratch(n)?;
    d_input.upload(data, stream)?;
    // No zero-fill: the kernel writes one full row per block for all
    // `batch_size` blocks, covering every element the readback reads.
    let mut d_output = ctx.scratch(n)?;

    // The warp-shuffle kernel (row_size <= 32) maps one warp (32 threads) per
    // row, so we must launch 32 threads/block regardless of row_size.  The
    // shared-memory kernel (row_size > 32) uses row_size.next_power_of_two().min(256).
    let block_threads = if row_size <= 32 {
        32u32
    } else {
        row_size.next_power_of_two().min(256)
    };
    let params = LaunchParams::new(Dim3::from(batch_size), Dim3::from(block_threads));

    let args = (d_input.device_ptr(), d_output.device_ptr(), batch_size);
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
    Ok(Some(out))
}
