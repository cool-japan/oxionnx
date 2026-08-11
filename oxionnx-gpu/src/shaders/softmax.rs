//! GPU-accelerated softmax dispatch.

use crate::context::GpuContext;

use super::common::{
    block_on_gpu, checked_storage_bytes, plan_dispatch, read_back_and_recycle_async, ErrorScope,
    SoftmaxParams, SOFTMAX_DIM_THRESHOLD,
};

// The kernel's own workgroup size (256, matching its `array<f32, 256>` shared
// buffer) lives entirely in `SOFTMAX_SHADER`: the host no longer needs it,
// because the dispatch is one *workgroup* per row rather than one thread, so
// the grid is planned with a workgroup size of 1.

// ========================================================================
// Softmax
// ========================================================================

/// GPU-accelerated softmax over the last dimension.
///
/// [a7-18] One 256-thread workgroup per row, with shared-memory tree
/// reductions for the row max and the row sum, fused into a single dispatch
/// (see `SOFTMAX_SHADER` for the layout and the rationale).
///
/// # Numerics
///
/// The tree reduction changes the *order* in which a row's exponentials are
/// summed relative to the old serial scan, so results can differ in the last
/// ulp or two; pairwise summation is if anything more accurate than a left
/// fold over 1000+ terms. The guarded tolerance is 1e-6 absolute against an
/// f64 CPU reference (`tests/w2_gpu_perf.rs`), which the outputs — all in
/// `[0, 1]` — clear comfortably. The max-subtraction that makes the
/// exponentials safe is unchanged.
///
/// # Accepted shapes
///
/// Row counts that need a 2-D workgroup grid are now handled instead of
/// declined, which widens the accepted range from `65_535 * 256` ≈ 16.8M rows
/// to `65_535²` ≈ 4.29G rows (in practice bounded first by the `u32` element
/// index in `checked_storage_bytes`).
///
/// Returns `None` if the last dimension is below the threshold (caller should use CPU).
pub async fn gpu_softmax_async(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
) -> Option<Vec<f32>> {
    if ctx.is_degraded() {
        return None;
    }
    let last_dim = *shape.last()?;
    if last_dim < SOFTMAX_DIM_THRESHOLD {
        return None;
    }
    let num_rows: usize = shape
        .iter()
        .rev()
        .skip(1)
        .try_fold(1usize, |a, &d| a.checked_mul(d))?;
    if num_rows == 0 {
        return None;
    }
    let total = num_rows.checked_mul(last_dim)?;
    if data.len() < total {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let out_size = checked_storage_bytes(&ctx.limits, total)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    // One workgroup per row. `plan_dispatch` with a workgroup size of 1 gives
    // one grid slot per row and splits into a 2-D grid when the row count
    // exceeds the device's per-dimension limit; the kernel rebuilds the row as
    // `wid.y * wg_per_row + wid.x`. It still declines when even a 2-D grid
    // cannot cover the rows, rather than silently skipping any.
    let grid = plan_dispatch(&ctx.limits, u64::try_from(num_rows).ok()?, 1)?;
    // Input, output and read-back staging, all the same length here.
    if !ctx.budget_admits(&[out_size, out_size, out_size]) {
        return None;
    }

    let scope = ErrorScope::begin(ctx);

    let input_buf = ctx.upload_buffer(
        "softmax_in",
        bytemuck::cast_slice(&data[..total]),
        wgpu::BufferUsages::STORAGE,
    )?;

    let output_buf = ctx.pooled_buffer(
        out_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;
    // The pool may hand back a buffer up to 2x `out_size` -- see
    // `GpuBufferPool::get_buffer` -- and `as_entire_binding()` would then bind
    // that larger size, which can exceed `max_storage_buffer_binding_size`
    // even though `out_size` itself was validated. Bind the exact range
    // instead, as `conv2d::gpu_conv2d_implicit_resident_async`'s `output_binding` does.
    let output_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &output_buf,
        offset: 0,
        size: wgpu::BufferSize::new(out_size),
    });

    let params = SoftmaxParams {
        num_rows: u32::try_from(num_rows).ok()?,
        row_len: u32::try_from(last_dim).ok()?,
        wg_per_row: grid.x,
        _pad: 0,
    };
    let params_buf = ctx.upload_buffer(
        "softmax_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = ctx.staging_buffer("softmax_staging", out_size)?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("softmax_bg"),
        layout: &ctx.softmax_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_binding,
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("softmax_enc"),
    });

    // Single fused dispatch: max reduction, exp, sum reduction, normalize.
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.softmax_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish_async(ctx).await {
        return None;
    }

    read_back_and_recycle_async(ctx, &staging_buf, total, output_buf).await
}

/// Blocking form of [`gpu_softmax_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_softmax(ctx: &GpuContext, data: &[f32], shape: &[usize]) -> Option<Vec<f32>> {
    block_on_gpu(gpu_softmax_async(ctx, data, shape))
}
