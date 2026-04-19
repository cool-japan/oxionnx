//! GPU-accelerated element-wise (unary and binary) operations.

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

use super::common::{read_back, EwParams, BINARY_EW_GPU_THRESHOLD, EW_GPU_THRESHOLD};

// ========================================================================
// Unary element-wise ops
// ========================================================================

/// Internal helper for element-wise GPU dispatch.
pub(super) fn gpu_elementwise_dispatch(
    ctx: &GpuContext,
    data: &[f32],
    pipeline: &wgpu::ComputePipeline,
) -> Option<Vec<f32>> {
    if data.len() < EW_GPU_THRESHOLD {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;
    let len = data.len();

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ew_in"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = std::mem::size_of_val(data) as u64;
    let output_buf = {
        let mut pool = ctx.pool.lock().ok()?;
        pool.get_buffer(
            device,
            out_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    };

    let params = EwParams {
        len: len as u32,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ew_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ew_staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ew_bg"),
        layout: &ctx.elementwise_bind_group_layout,
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

    let wg = (len as u32).div_ceil(256);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ew_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ew_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_back(device, &staging_buf, len);

    let mut pool = ctx.pool.lock().ok()?;
    pool.return_buffer(output_buf, out_size);

    result
}

/// GPU-accelerated ReLU: max(x, 0).
///
/// Returns `None` for tensors below the threshold.
pub fn gpu_relu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.relu_pipeline)
}

/// GPU-accelerated Sigmoid: 1 / (1 + exp(-x)).
///
/// Returns `None` for tensors below the threshold.
pub fn gpu_sigmoid(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.sigmoid_pipeline)
}

/// GPU-accelerated GELU approximation.
///
/// Returns `None` for tensors below the threshold.
pub fn gpu_gelu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.gelu_pipeline)
}

/// GPU-accelerated Tanh: tanh(x).
pub fn gpu_tanh(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.tanh_pipeline)
}

/// GPU-accelerated Exp: exp(x).
pub fn gpu_exp(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.exp_pipeline)
}

/// GPU-accelerated Sqrt: sqrt(x).
pub fn gpu_sqrt(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.sqrt_pipeline)
}

/// GPU-accelerated Abs: abs(x).
pub fn gpu_abs(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.abs_pipeline)
}

/// GPU-accelerated Neg: -x.
pub fn gpu_neg(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.neg_pipeline)
}

/// GPU-accelerated Log: log(x) (natural logarithm).
pub fn gpu_log(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.log_pipeline)
}

/// GPU-accelerated SiLU: x / (1 + exp(-x)).
pub fn gpu_silu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.silu_pipeline)
}

/// GPU-accelerated LeakyRelu: select(0.01 * x, x, x >= 0).
pub fn gpu_leaky_relu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, &ctx.leaky_relu_pipeline)
}

// ========================================================================
// Binary element-wise ops
// ========================================================================

/// Internal helper for binary element-wise GPU dispatch.
pub(super) fn gpu_binary_elementwise_dispatch(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    pipeline: &wgpu::ComputePipeline,
) -> Option<Vec<f32>> {
    let len = a.len();
    if len != b.len() || len < BINARY_EW_GPU_THRESHOLD {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("binary_a"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("binary_b"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = std::mem::size_of_val(a) as u64;
    let output_buf = {
        let mut pool = ctx.pool.lock().ok()?;
        pool.get_buffer(
            device,
            out_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    };

    let params = EwParams {
        len: len as u32,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("binary_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("binary_staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("binary_bg"),
        layout: &ctx.binary_elementwise_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (len as u32).div_ceil(256);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("binary_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("binary_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_back(device, &staging_buf, len);

    let mut pool = ctx.pool.lock().ok()?;
    pool.return_buffer(output_buf, out_size);

    result
}

/// GPU-accelerated element-wise addition: a + b.
///
/// Both inputs must have the same length (no broadcasting).
/// Returns `None` for tensors below the threshold or mismatched lengths.
pub fn gpu_add(ctx: &GpuContext, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    gpu_binary_elementwise_dispatch(ctx, a, b, &ctx.add_pipeline)
}

/// GPU-accelerated element-wise multiplication: a * b.
///
/// Both inputs must have the same length (no broadcasting).
/// Returns `None` for tensors below the threshold or mismatched lengths.
pub fn gpu_mul(ctx: &GpuContext, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    gpu_binary_elementwise_dispatch(ctx, a, b, &ctx.mul_pipeline)
}
