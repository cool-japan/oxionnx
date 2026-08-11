//! GPU-accelerated normalization operations (LayerNorm and BatchNorm).

use crate::context::GpuContext;

use super::common::{
    block_on_gpu, checked_storage_bytes, plan_dispatch, read_back_and_recycle_async,
    BatchNormParams, ErrorScope, LayerNormParams, BATCH_NORM_GPU_THRESHOLD,
    LAYER_NORM_GPU_THRESHOLD, WG_SIZE,
};

// ========================================================================
// LayerNorm
// ========================================================================

/// GPU-accelerated LayerNormalization over the last axis (ONNX `axis = -1`).
///
/// `scale` and `bias` must have exactly `shape.last()` elements — that is the
/// shape ONNX requires for `axis = -1`, and a mismatch means the node uses some
/// other axis, in which case this declines so the CPU operator (which honours
/// `axis`) handles it. Use [`gpu_layer_norm_axis`] to normalize over an
/// explicit axis.
///
/// Returns `None` if the problem is below the GPU threshold or if GPU is unavailable.
pub async fn gpu_layer_norm_async(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    scale: &[f32],
    bias: &[f32],
    eps: f32,
) -> Option<Vec<f32>> {
    gpu_layer_norm_axis_async(ctx, input, shape, scale, bias, eps, -1).await
}

/// Blocking form of [`gpu_layer_norm_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_layer_norm(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    scale: &[f32],
    bias: &[f32],
    eps: f32,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_layer_norm_async(ctx, input, shape, scale, bias, eps))
}

/// GPU-accelerated LayerNormalization over `product(shape[axis..])`.
///
/// `axis` follows ONNX conventions: negative values count from the end, and the
/// normalized region is the suffix of the shape starting at `axis` (matching the
/// CPU `LayerNormOp`). `scale` and `bias` must have exactly that many elements.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_layer_norm_axis_async(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    scale: &[f32],
    bias: &[f32],
    eps: f32,
    axis: i64,
) -> Option<Vec<f32>> {
    if shape.is_empty() || ctx.is_degraded() {
        return None;
    }
    let rank = i64::try_from(shape.len()).ok()?;
    let axis = if axis < 0 {
        axis.checked_add(rank)?
    } else {
        axis
    };
    if axis < 0 || axis >= rank {
        return None;
    }
    let axis = usize::try_from(axis).ok()?;

    // Normalized region = product(shape[axis..]); instances = product(shape[..axis]).
    let n_elements: usize = shape[axis..]
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))?;
    let batch_count: usize = shape[..axis]
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))?;
    if n_elements == 0 || batch_count == 0 {
        return None;
    }
    let total = batch_count.checked_mul(n_elements)?;
    if input.len() != total {
        return None;
    }
    // ONNX requires scale/bias to have exactly the normalized shape. Accepting
    // a longer slice would silently normalize the wrong region.
    if scale.len() != n_elements || bias.len() != n_elements {
        return None;
    }
    if total < LAYER_NORM_GPU_THRESHOLD {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let out_size = checked_storage_bytes(&ctx.limits, total)?;
    let vec_size = checked_storage_bytes(&ctx.limits, n_elements)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    // One workgroup per normalization instance, split across a 2-D grid when
    // there are more instances than the device allows along one dimension.
    let grid = plan_dispatch(&ctx.limits, batch_count as u64, 1)?;
    // Input, scale, bias, output and read-back staging.
    if !ctx.budget_admits(&[out_size, vec_size, vec_size, out_size, out_size]) {
        return None;
    }

    let scope = ErrorScope::begin(ctx);

    let input_buf = ctx.upload_buffer(
        "ln_input",
        bytemuck::cast_slice(&input[..total]),
        wgpu::BufferUsages::STORAGE,
    )?;
    let scale_buf = ctx.upload_buffer(
        "ln_scale",
        bytemuck::cast_slice(&scale[..n_elements]),
        wgpu::BufferUsages::STORAGE,
    )?;
    let bias_buf = ctx.upload_buffer(
        "ln_bias",
        bytemuck::cast_slice(&bias[..n_elements]),
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

    let params = LayerNormParams {
        n_elements: u32::try_from(n_elements).ok()?,
        batch_count: u32::try_from(batch_count).ok()?,
        eps,
        wg_per_row: grid.x,
    };
    let params_buf = ctx.upload_buffer(
        "ln_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = ctx.staging_buffer("ln_staging", out_size)?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ln_bg"),
        layout: &ctx.layer_norm_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scale_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_binding,
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ln_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ln_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.layer_norm_pipeline);
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

// ========================================================================
// BatchNorm (inference)
// ========================================================================

/// GPU-accelerated BatchNormalization (inference mode).
///
/// Input shape is [N, C, H, W] (or [N, C, D1, D2, ...]). Per-channel parameters.
/// Returns `None` if the problem is below the GPU threshold.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_batch_norm_async(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> Option<Vec<f32>> {
    if shape.len() < 2 || ctx.is_degraded() {
        return None;
    }
    let channels = shape[1];
    if channels == 0 || shape.contains(&0) {
        return None;
    }
    let spatial_size: usize = shape[2..]
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))?;
    let total: usize = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d))?;
    if total == 0 || input.len() < total {
        return None;
    }
    if scale.len() < channels
        || bias.len() < channels
        || mean.len() < channels
        || var.len() < channels
    {
        return None;
    }
    if total < BATCH_NORM_GPU_THRESHOLD {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let out_size = checked_storage_bytes(&ctx.limits, total)?;
    let vec_size = checked_storage_bytes(&ctx.limits, channels)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, total as u64, WG_SIZE)?;
    // Input, four per-channel vectors, output and read-back staging.
    if !ctx.budget_admits(&[
        out_size, vec_size, vec_size, vec_size, vec_size, out_size, out_size,
    ]) {
        return None;
    }

    let scope = ErrorScope::begin(ctx);

    let input_buf = ctx.upload_buffer(
        "bn_input",
        bytemuck::cast_slice(&input[..total]),
        wgpu::BufferUsages::STORAGE,
    )?;
    let scale_buf = ctx.upload_buffer(
        "bn_scale",
        bytemuck::cast_slice(&scale[..channels]),
        wgpu::BufferUsages::STORAGE,
    )?;
    let bias_buf = ctx.upload_buffer(
        "bn_bias",
        bytemuck::cast_slice(&bias[..channels]),
        wgpu::BufferUsages::STORAGE,
    )?;
    let mean_buf = ctx.upload_buffer(
        "bn_mean",
        bytemuck::cast_slice(&mean[..channels]),
        wgpu::BufferUsages::STORAGE,
    )?;
    let var_buf = ctx.upload_buffer(
        "bn_var",
        bytemuck::cast_slice(&var[..channels]),
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

    let params = BatchNormParams {
        total_elements: u32::try_from(total).ok()?,
        channels: u32::try_from(channels).ok()?,
        spatial_size: u32::try_from(spatial_size).ok()?,
        eps,
        row_threads: grid.threads_per_row,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = ctx.upload_buffer(
        "bn_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = ctx.staging_buffer("bn_staging", out_size)?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bn_bg"),
        layout: &ctx.batch_norm_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scale_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: mean_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: var_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_binding,
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bn_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bn_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.batch_norm_pipeline);
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

/// Blocking form of [`gpu_layer_norm_axis_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_layer_norm_axis(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    scale: &[f32],
    bias: &[f32],
    eps: f32,
    axis: i64,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_layer_norm_axis_async(
        ctx, input, shape, scale, bias, eps, axis,
    ))
}

/// Blocking form of [`gpu_batch_norm_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_batch_norm(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_batch_norm_async(
        ctx, input, shape, scale, bias, mean, var, eps,
    ))
}
