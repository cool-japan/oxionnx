//! GPU-accelerated reduction operations (sum, max, min, mean).

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

use super::common::{read_back, ReduceParams, REDUCE_GPU_THRESHOLD};

// ========================================================================
// Reduction ops
// ========================================================================

/// Internal helper to compute (outer_size, axis_len, inner_size) from shape + axis.
pub(super) fn reduction_dims(shape: &[usize], axis: usize) -> Option<(usize, usize, usize)> {
    if axis >= shape.len() {
        return None;
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer = if outer == 0 { 1 } else { outer };
    let inner = if inner == 0 { 1 } else { inner };
    Some((outer, axis_len, inner))
}

/// Internal helper for reduction GPU dispatch.
pub(super) fn gpu_reduce_dispatch(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
    pipeline: &wgpu::ComputePipeline,
) -> Option<Vec<f32>> {
    let (outer, axis_len, inner) = reduction_dims(shape, axis)?;
    let out_count = outer * inner;
    if out_count < REDUCE_GPU_THRESHOLD {
        return None;
    }
    let total_in = outer * axis_len * inner;
    if data.len() < total_in {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reduce_in"),
        contents: bytemuck::cast_slice(&data[..total_in]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = (out_count * std::mem::size_of::<f32>()) as u64;
    let output_buf = {
        let mut pool = ctx.pool.lock().ok()?;
        pool.get_buffer(
            device,
            out_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    };

    let params = ReduceParams {
        outer_size: outer as u32,
        axis_len: axis_len as u32,
        inner_size: inner as u32,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reduce_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reduce_staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (out_count as u32).div_ceil(256);

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
        cpass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_back(device, &staging_buf, out_count);

    let mut pool = ctx.pool.lock().ok()?;
    pool.return_buffer(output_buf, out_size);

    result
}

/// GPU-accelerated parallel reduction (sum) along an axis.
///
/// Returns `None` if the output is too small for GPU benefit.
pub fn gpu_reduce_sum(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_reduce_dispatch(ctx, data, axis, shape, &ctx.reduce_sum_pipeline)
}

/// GPU-accelerated parallel reduction (max) along an axis.
///
/// Returns `None` if the output is too small for GPU benefit.
pub fn gpu_reduce_max(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_reduce_dispatch(ctx, data, axis, shape, &ctx.reduce_max_pipeline)
}

/// GPU-accelerated parallel reduction (min) along an axis.
///
/// Returns `None` if the output is too small for GPU benefit.
pub fn gpu_reduce_min(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_reduce_dispatch(ctx, data, axis, shape, &ctx.reduce_min_pipeline)
}

// ========================================================================
// ReduceMean
// ========================================================================

/// GPU-accelerated ReduceMean along specified axes.
///
/// Reduces the input along each axis in `axes` sequentially.
/// For a single axis, uses the GPU reduce_mean kernel directly.
/// Returns `None` if below threshold or on error.
pub fn gpu_reduce_mean(
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
        return gpu_reduce_mean_single(ctx, data, axis, shape);
    }
    // For multi-axis: reduce axes one at a time (largest first to keep indices valid)
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable();
    sorted_axes.reverse();

    let mut current_data = data.to_vec();
    let mut current_shape = shape.to_vec();

    for &axis in &sorted_axes {
        if axis >= current_shape.len() {
            return None;
        }
        let result = gpu_reduce_mean_single(ctx, &current_data, axis, &current_shape)?;
        // Update shape: remove the reduced axis
        current_shape.remove(axis);
        if current_shape.is_empty() {
            current_shape.push(1);
        }
        current_data = result;
    }

    Some(current_data)
}

/// GPU-accelerated ReduceMean along a single axis.
fn gpu_reduce_mean_single(
    ctx: &GpuContext,
    data: &[f32],
    axis: usize,
    shape: &[usize],
) -> Option<Vec<f32>> {
    let (outer, axis_len, inner) = reduction_dims(shape, axis)?;
    let out_count = outer * inner;
    if out_count < REDUCE_GPU_THRESHOLD {
        return None;
    }
    let total_in = outer * axis_len * inner;
    if data.len() < total_in {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reduce_mean_in"),
        contents: bytemuck::cast_slice(&data[..total_in]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = (out_count * std::mem::size_of::<f32>()) as u64;
    let output_buf = {
        let mut pool = ctx.pool.lock().ok()?;
        pool.get_buffer(
            device,
            out_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    };

    let params = ReduceParams {
        outer_size: outer as u32,
        axis_len: axis_len as u32,
        inner_size: inner as u32,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reduce_mean_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reduce_mean_staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (out_count as u32).div_ceil(256);

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
        cpass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_back(device, &staging_buf, out_count);

    let mut pool = ctx.pool.lock().ok()?;
    pool.return_buffer(output_buf, out_size);

    result
}
