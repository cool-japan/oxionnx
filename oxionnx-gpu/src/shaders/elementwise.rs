//! GPU-accelerated element-wise (unary and binary) operations.

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

use super::common::{
    checked_storage_bytes, plan_dispatch, read_back_and_recycle, ErrorScope, EwParams,
    BINARY_EW_GPU_THRESHOLD, EW_GPU_THRESHOLD, WG_SIZE,
};

/// LeakyRelu slope used when the caller does not supply one (ONNX default).
pub const DEFAULT_LEAKY_RELU_ALPHA: f32 = 0.01;

// ========================================================================
// Unary element-wise ops
// ========================================================================

/// Internal helper for element-wise GPU dispatch.
pub(super) fn gpu_elementwise_dispatch(
    ctx: &GpuContext,
    data: &[f32],
    pipeline: &wgpu::ComputePipeline,
    alpha: f32,
) -> Option<Vec<f32>> {
    if data.len() < EW_GPU_THRESHOLD || ctx.is_degraded() {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;
    let len = data.len();

    // Decline before allocating anything the device cannot hold, and before
    // planning a dispatch it cannot express.
    let out_size = checked_storage_bytes(&ctx.limits, len)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, len as u64, WG_SIZE)?;

    let scope = ErrorScope::begin(ctx);

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ew_in"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });

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
        alpha,
        row_threads: grid.threads_per_row,
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
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    // Surface any validation/OOM error as a decline before touching the result.
    if !scope.finish(ctx) {
        return None;
    }

    read_back_and_recycle(ctx, &staging_buf, len, output_buf)
}

/// Dispatch a unary kernel that takes no scalar parameter.
#[inline]
fn dispatch_unary(
    ctx: &GpuContext,
    data: &[f32],
    pipeline: &wgpu::ComputePipeline,
) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, pipeline, 0.0)
}

/// GPU-accelerated ReLU: max(x, 0).
///
/// Returns `None` for tensors below the threshold.
pub fn gpu_relu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.relu_pipeline)
}

/// GPU-accelerated Sigmoid: 1 / (1 + exp(-x)).
///
/// Returns `None` for tensors below the threshold.
pub fn gpu_sigmoid(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.sigmoid_pipeline)
}

/// GPU-accelerated GELU approximation.
///
/// Returns `None` for tensors below the threshold.
pub fn gpu_gelu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.gelu_pipeline)
}

/// GPU-accelerated Tanh: tanh(x).
pub fn gpu_tanh(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.tanh_pipeline)
}

/// GPU-accelerated Exp: exp(x).
pub fn gpu_exp(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.exp_pipeline)
}

/// GPU-accelerated Sqrt: sqrt(x).
pub fn gpu_sqrt(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.sqrt_pipeline)
}

/// GPU-accelerated Abs: abs(x).
pub fn gpu_abs(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.abs_pipeline)
}

/// GPU-accelerated Neg: -x.
pub fn gpu_neg(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.neg_pipeline)
}

/// GPU-accelerated Log: log(x) (natural logarithm).
pub fn gpu_log(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.log_pipeline)
}

/// GPU-accelerated SiLU: x / (1 + exp(-x)).
pub fn gpu_silu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    dispatch_unary(ctx, data, &ctx.silu_pipeline)
}

/// GPU-accelerated LeakyRelu with the ONNX default slope (`alpha = 0.01`).
///
/// Callers that have a node with an explicit `alpha` attribute must use
/// [`gpu_leaky_relu_alpha`]; this entry point is only correct for the default.
pub fn gpu_leaky_relu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_leaky_relu_alpha(ctx, data, DEFAULT_LEAKY_RELU_ALPHA)
}

/// GPU-accelerated LeakyRelu: `x >= 0 ? x : alpha * x`.
///
/// `alpha` is the node's `alpha` attribute (ONNX default `0.01`); it is uploaded
/// as a uniform rather than baked into the kernel, so models such as YOLOv3
/// (`alpha = 0.1`) compute the same values on GPU and CPU.
///
/// Non-finite `alpha` values are rejected (the node falls back to the CPU
/// operator, which reports the malformed attribute).
pub fn gpu_leaky_relu_alpha(ctx: &GpuContext, data: &[f32], alpha: f32) -> Option<Vec<f32>> {
    if !alpha.is_finite() {
        return None;
    }
    gpu_elementwise_dispatch(ctx, data, &ctx.leaky_relu_pipeline, alpha)
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
    if len != b.len() || len < BINARY_EW_GPU_THRESHOLD || ctx.is_degraded() {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    // a, b and the output all get bound as storage buffers.
    let out_size = checked_storage_bytes(&ctx.limits, len)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, len as u64, WG_SIZE)?;

    let scope = ErrorScope::begin(ctx);

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
        alpha: 0.0,
        row_threads: grid.threads_per_row,
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
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish(ctx) {
        return None;
    }

    read_back_and_recycle(ctx, &staging_buf, len, output_buf)
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
