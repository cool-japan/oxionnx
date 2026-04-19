//! GPU-accelerated normalization operations (LayerNorm and BatchNorm).

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

use super::common::{
    read_back, BatchNormParams, LayerNormParams, BATCH_NORM_GPU_THRESHOLD, LAYER_NORM_GPU_THRESHOLD,
};

// ========================================================================
// LayerNorm
// ========================================================================

/// GPU-accelerated LayerNormalization.
///
/// Normalizes the last `norm_size` elements of each instance, applying scale and bias.
/// `shape` must have at least 2 dimensions; the last dimension is the normalized axis.
/// Returns `None` if the problem is below the GPU threshold or if GPU is unavailable.
pub fn gpu_layer_norm(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    scale: &[f32],
    bias: &[f32],
    eps: f32,
) -> Option<Vec<f32>> {
    if shape.is_empty() {
        return None;
    }
    let n_elements = *shape.last()?;
    if n_elements == 0 {
        return None;
    }
    let batch_count = input.len() / n_elements;
    if batch_count == 0 || input.len() != batch_count * n_elements {
        return None;
    }
    if scale.len() < n_elements || bias.len() < n_elements {
        return None;
    }
    if input.len() < LAYER_NORM_GPU_THRESHOLD {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;
    let total = batch_count * n_elements;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ln_input"),
        contents: bytemuck::cast_slice(&input[..total]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scale_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ln_scale"),
        contents: bytemuck::cast_slice(&scale[..n_elements]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ln_bias"),
        contents: bytemuck::cast_slice(&bias[..n_elements]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = (total * std::mem::size_of::<f32>()) as u64;
    let output_buf = {
        let mut pool = ctx.pool.lock().ok()?;
        pool.get_buffer(
            device,
            out_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    };

    let params = LayerNormParams {
        n_elements: n_elements as u32,
        batch_count: batch_count as u32,
        eps,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ln_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ln_staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // One workgroup per normalization instance
    let wg = batch_count as u32;

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
        cpass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_back(device, &staging_buf, total);

    let mut pool = ctx.pool.lock().ok()?;
    pool.return_buffer(output_buf, out_size);

    result
}

// ========================================================================
// BatchNorm (inference)
// ========================================================================

/// GPU-accelerated BatchNormalization (inference mode).
///
/// Input shape is [N, C, H, W] (or [N, C, D1, D2, ...]). Per-channel parameters.
/// Returns `None` if the problem is below the GPU threshold.
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
    if shape.len() < 2 {
        return None;
    }
    let channels = shape[1];
    if channels == 0 {
        return None;
    }
    let spatial_size: usize = shape[2..].iter().product();
    let spatial_size = if spatial_size == 0 { 1 } else { spatial_size };
    let total: usize = shape.iter().product();
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

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_input"),
        contents: bytemuck::cast_slice(&input[..total]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scale_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_scale"),
        contents: bytemuck::cast_slice(&scale[..channels]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_bias"),
        contents: bytemuck::cast_slice(&bias[..channels]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mean_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_mean"),
        contents: bytemuck::cast_slice(&mean[..channels]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let var_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_var"),
        contents: bytemuck::cast_slice(&var[..channels]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = (total * std::mem::size_of::<f32>()) as u64;
    let output_buf = {
        let mut pool = ctx.pool.lock().ok()?;
        pool.get_buffer(
            device,
            out_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    };

    let params = BatchNormParams {
        total_elements: total as u32,
        channels: channels as u32,
        spatial_size: spatial_size as u32,
        eps,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bn_staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (total as u32).div_ceil(256);

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
        cpass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_back(device, &staging_buf, total);

    let mut pool = ctx.pool.lock().ok()?;
    pool.return_buffer(output_buf, out_size);

    result
}
