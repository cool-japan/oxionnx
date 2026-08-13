//! GPU-accelerated reduction operations (sum, max, min, mean).

use crate::context::GpuContext;

use super::common::{
    block_on_gpu, checked_storage_bytes, plan_dispatch, read_back_and_recycle_async, DispatchGrid,
    ErrorScope, ReduceParams, WG_SIZE,
};

// ========================================================================
// Reduction ops
// ========================================================================

/// Internal helper to compute (outer_size, axis_len, inner_size) from shape + axis.
///
/// Returns `None` for a shape with a zero-length dimension: the reduced axis
/// would make the kernels read out of range (`reduce_max`/`reduce_min` seed the
/// accumulator from `input[in_base]`) and divide by zero (`reduce_mean`). The
/// CPU operators implement the ONNX identity rules for empty reductions, so
/// declining routes those cases to the correct implementation.
pub(super) fn reduction_dims(shape: &[usize], axis: usize) -> Option<(usize, usize, usize)> {
    if axis >= shape.len() || shape.contains(&0) {
        return None;
    }
    // Products of empty slices are 1, so no zero-coercion is needed once every
    // dimension is known to be non-zero.
    let outer: usize = shape[..axis]
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))?;
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..]
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))?;
    Some((outer, axis_len, inner))
}

/// Shared front-half of both reduction dispatchers: validate the shape, check
/// the device limits and plan the grid.
fn reduce_plan(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<(usize, usize, usize, usize, u64, DispatchGrid)> {
    if ctx.is_degraded() {
        return None;
    }
    let (outer, axis_len, inner) = reduction_dims(shape, axis)?;
    let out_count = outer.checked_mul(inner)?;
    if out_count < ctx.tuning().reduce_min_output_elements {
        return None;
    }
    let total_in = out_count.checked_mul(axis_len)?;
    if data.len() < total_in {
        return None;
    }
    // Both the input and the output are bound as storage buffers.
    let out_size = checked_storage_bytes(&ctx.limits, out_count)?;
    let in_size = checked_storage_bytes(&ctx.limits, total_in)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    // Input, output and read-back staging. Checked here rather than in each
    // dispatcher because both of them allocate exactly these three.
    if !ctx.budget_admits(&[in_size, out_size, out_size]) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, out_count as u64, WG_SIZE)?;
    Some((outer, axis_len, inner, out_count, out_size, grid))
}

/// Internal helper for reduction GPU dispatch.
pub(super) async fn gpu_reduce_dispatch(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
    pipeline: &wgpu::ComputePipeline,
) -> Option<Vec<f32>> {
    let (outer, axis_len, inner, out_count, out_size, grid) = reduce_plan(ctx, data, axis, shape)?;
    let total_in = out_count * axis_len;

    let device = &ctx.device;
    let queue = &ctx.queue;

    let scope = ErrorScope::begin(ctx);

    let input_buf = ctx.upload_buffer(
        "reduce_in",
        bytemuck::cast_slice(&data[..total_in]),
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

    let params = ReduceParams {
        outer_size: outer as u32,
        axis_len: axis_len as u32,
        inner_size: inner as u32,
        row_threads: grid.threads_per_row,
    };
    let params_buf = ctx.upload_buffer(
        "reduce_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = ctx.staging_buffer("reduce_staging", out_size)?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reduce_bg"),
        layout: &ctx.reduce_bind_group_layout,
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
        label: Some("reduce_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reduce_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish_async(ctx).await {
        return None;
    }

    read_back_and_recycle_async(ctx, &staging_buf, out_count, output_buf).await
}

/// GPU-accelerated parallel reduction (sum) along an axis.
///
/// Returns `None` if the output is too small for GPU benefit.
pub async fn gpu_reduce_sum_async(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_reduce_dispatch(ctx, data, axis, shape, &ctx.reduce_sum_pipeline).await
}

/// Blocking form of [`gpu_reduce_sum_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_reduce_sum(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_reduce_sum_async(ctx, data, axis, shape))
}

/// GPU-accelerated parallel reduction (max) along an axis.
///
/// Returns `None` if the output is too small for GPU benefit.
pub async fn gpu_reduce_max_async(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_reduce_dispatch(ctx, data, axis, shape, &ctx.reduce_max_pipeline).await
}

/// Blocking form of [`gpu_reduce_max_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_reduce_max(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_reduce_max_async(ctx, data, axis, shape))
}

/// GPU-accelerated parallel reduction (min) along an axis.
///
/// Returns `None` if the output is too small for GPU benefit.
pub async fn gpu_reduce_min_async(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_reduce_dispatch(ctx, data, axis, shape, &ctx.reduce_min_pipeline).await
}

/// Blocking form of [`gpu_reduce_min_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_reduce_min(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_reduce_min_async(ctx, data, axis, shape))
}

// ========================================================================
// ReduceMean
// ========================================================================

/// GPU-accelerated ReduceMean along specified axes.
///
/// Reduces the input along each axis in `axes` sequentially.
/// For a single axis, uses the GPU reduce_mean kernel directly.
/// Returns `None` if below threshold or on error.
pub async fn gpu_reduce_mean_async(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    axes: &[usize],
    _keepdims: bool,
) -> Option<Vec<f32>> {
    if axes.is_empty() || shape.is_empty() {
        return None;
    }
    // For single-axis reduction, dispatch directly
    if axes.len() == 1 {
        let axis = axes[0];
        return gpu_reduce_mean_single(ctx, data, axis, shape).await;
    }
    // For multi-axis: reduce axes one at a time (largest first to keep indices valid)
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable();
    // A repeated axis is invalid per the ONNX spec; reducing it twice would
    // silently hit a *different* (shifted) axis on the second pass, so decline
    // and let the CPU operator report the malformed attribute.
    if sorted_axes.windows(2).any(|w| w[0] == w[1]) {
        return None;
    }
    sorted_axes.reverse();

    let mut current_data = data.to_vec();
    let mut current_shape = shape.to_vec();

    for &axis in &sorted_axes {
        if axis >= current_shape.len() {
            return None;
        }
        let result = gpu_reduce_mean_single(ctx, &current_data, axis, &current_shape).await?;
        // Update shape: remove the reduced axis
        current_shape.remove(axis);
        if current_shape.is_empty() {
            current_shape.push(1);
        }
        current_data = result;
    }

    Some(current_data)
}

/// Blocking form of [`gpu_reduce_mean_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_reduce_mean(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    axes: &[usize],
    keepdims: bool,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_reduce_mean_async(ctx, data, shape, axes, keepdims))
}

/// GPU-accelerated ReduceMean along a single axis.
async fn gpu_reduce_mean_single(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    let (outer, axis_len, inner, out_count, out_size, grid) = reduce_plan(ctx, data, axis, shape)?;
    let total_in = out_count * axis_len;

    let device = &ctx.device;
    let queue = &ctx.queue;

    let scope = ErrorScope::begin(ctx);

    let input_buf = ctx.upload_buffer(
        "reduce_mean_in",
        bytemuck::cast_slice(&data[..total_in]),
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

    let params = ReduceParams {
        outer_size: outer as u32,
        axis_len: axis_len as u32,
        inner_size: inner as u32,
        row_threads: grid.threads_per_row,
    };
    let params_buf = ctx.upload_buffer(
        "reduce_mean_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = ctx.staging_buffer("reduce_mean_staging", out_size)?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reduce_mean_bg"),
        layout: &ctx.reduce_bind_group_layout,
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
        label: Some("reduce_mean_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reduce_mean_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.reduce_mean_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish_async(ctx).await {
        return None;
    }

    read_back_and_recycle_async(ctx, &staging_buf, out_count, output_buf).await
}
