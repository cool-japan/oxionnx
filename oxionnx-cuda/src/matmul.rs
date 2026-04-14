//! CUDA-accelerated MatMul / Gemm dispatch.
//!
//! Supports 2-D matrix multiplication (`[M, K] x [K, N] -> [M, N]`) via
//! `oxicuda_blas`.  Batched matmul (batch_size > 1) is handled by the caller
//! via per-slice dispatch.

use oxicuda_blas::{level3::gemm_api::gemm, Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_memory::DeviceBuffer;

use crate::context::CudaContext;
use crate::error::CudaDispatchError;

/// Run `A @ B` on the GPU.
///
/// * `a_data` — flattened row-major f32 data for A, shape `[m, k]`.
/// * `b_data` — flattened row-major f32 data for B, shape `[k, n]`.
/// * `m`, `k`, `n` — matrix dimensions.
///
/// Returns the output vector (length `m * n`, row-major) on success, or
/// `Err` if the CUDA operation fails (caller will map to `OnnxError::Internal`).
pub fn cuda_matmul(
    ctx: &CudaContext,
    a_data: &[f32],
    b_data: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, CudaDispatchError> {
    // Upload A and B to device.
    let mut d_a: DeviceBuffer<f32> = DeviceBuffer::alloc(m * k)?;
    d_a.copy_from_host(a_data)?;

    let mut d_b: DeviceBuffer<f32> = DeviceBuffer::alloc(k * n)?;
    d_b.copy_from_host(b_data)?;

    let mut d_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(m * n)?;

    // Build matrix descriptors — RowMajor layout, straightforward A @ B.
    let desc_a = MatrixDesc::<f32>::from_buffer(&d_a, m as u32, k as u32, Layout::RowMajor)
        .map_err(|e| CudaDispatchError::Blas(e.to_string()))?;
    let desc_b = MatrixDesc::<f32>::from_buffer(&d_b, k as u32, n as u32, Layout::RowMajor)
        .map_err(|e| CudaDispatchError::Blas(e.to_string()))?;
    let mut desc_c =
        MatrixDescMut::<f32>::from_buffer(&mut d_c, m as u32, n as u32, Layout::RowMajor)
            .map_err(|e| CudaDispatchError::Blas(e.to_string()))?;

    let blas_handle = ctx.dnn.blas();
    gemm(
        blas_handle,
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0_f32,
        &desc_a,
        &desc_b,
        0.0_f32,
        &mut desc_c,
    )
    .map_err(|e| CudaDispatchError::Blas(e.to_string()))?;

    // Synchronize the stream to ensure the BLAS operation completes before
    // reading results back to the host.
    ctx.dnn
        .stream()
        .synchronize()
        .map_err(CudaDispatchError::Driver)?;

    let mut out = vec![0.0_f32; m * n];
    d_c.copy_to_host(&mut out)?;
    Ok(out)
}
