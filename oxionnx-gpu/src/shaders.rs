//! GPU compute shader dispatch functions for softmax, element-wise ops, and reductions.

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

/// Minimum tensor elements before GPU dispatch is worthwhile for element-wise ops.
const EW_GPU_THRESHOLD: usize = 100_000;

/// Minimum last-dimension size before GPU softmax is worthwhile.
const SOFTMAX_DIM_THRESHOLD: usize = 1000;

/// Minimum output elements before GPU reduction is worthwhile.
const REDUCE_GPU_THRESHOLD: usize = 50_000;

// --- Uniform param structs ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    num_rows: u32,
    row_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EwParams {
    len: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    outer_size: u32,
    axis_len: u32,
    inner_size: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    n_elements: u32,
    batch_count: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchNormParams {
    total_elements: u32,
    channels: u32,
    spatial_size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeParams {
    total_elements: u32,
    ndim: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Minimum elements before GPU LayerNorm is worthwhile.
const LAYER_NORM_GPU_THRESHOLD: usize = 50_000;

/// Minimum elements before GPU BatchNorm is worthwhile.
const BATCH_NORM_GPU_THRESHOLD: usize = 50_000;

/// Minimum elements before GPU Transpose is worthwhile.
const TRANSPOSE_GPU_THRESHOLD: usize = 50_000;

// ========================================================================
// Helper: read back a staging buffer into Vec<f32>
// ========================================================================

/// Read back GPU staging buffer contents into a `Vec<f32>`.
///
/// On wasm32, blocking device poll is not supported, so this returns `None`.
fn read_back(_device: &wgpu::Device, _staging: &wgpu::Buffer, _count: usize) -> Option<Vec<f32>> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let slice = _staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        _device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if receiver.recv().ok()?.is_err() {
            return None;
        }
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data)[.._count].to_vec();
        drop(data);
        _staging.unmap();
        Some(result)
    }
}

// ========================================================================
// Softmax
// ========================================================================

/// GPU-accelerated softmax over the last dimension.
///
/// Returns `None` if the last dimension is below the threshold (caller should use CPU).
pub fn gpu_softmax(ctx: &GpuContext, data: &[f32], shape: &[usize]) -> Option<Vec<f32>> {
    let last_dim = *shape.last()?;
    if last_dim < SOFTMAX_DIM_THRESHOLD {
        return None;
    }
    let num_rows: usize = shape.iter().rev().skip(1).product();
    if num_rows == 0 {
        return None;
    }
    let total = num_rows * last_dim;
    if data.len() < total {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("softmax_in"),
        contents: bytemuck::cast_slice(&data[..total]),
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

    let params = SoftmaxParams {
        num_rows: num_rows as u32,
        row_len: last_dim as u32,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("softmax_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("softmax_staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (num_rows as u32).div_ceil(64);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("softmax_enc"),
    });

    // Pass 1: exp(x - max)
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax_p1"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.softmax_pass1_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
    }
    // Pass 2: normalize
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax_p2"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.softmax_pass2_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_back(device, &staging_buf, total);

    // Return output buffer to pool.
    let mut pool = ctx.pool.lock().ok()?;
    pool.return_buffer(output_buf, out_size);

    result
}

// ========================================================================
// Element-wise ops
// ========================================================================

/// Internal helper for element-wise GPU dispatch.
fn gpu_elementwise_dispatch(
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

// ========================================================================
// Reduction ops
// ========================================================================

/// Internal helper to compute (outer_size, axis_len, inner_size) from shape + axis.
fn reduction_dims(shape: &[usize], axis: usize) -> Option<(usize, usize, usize)> {
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
fn gpu_reduce_dispatch(
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
// New unary element-wise ops
// ========================================================================

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

/// Minimum tensor elements before GPU dispatch is worthwhile for binary element-wise ops.
const BINARY_EW_GPU_THRESHOLD: usize = 100_000;

/// Internal helper for binary element-wise GPU dispatch.
fn gpu_binary_elementwise_dispatch(
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

// ========================================================================
// Transpose
// ========================================================================

/// GPU-accelerated general Transpose with arbitrary permutation.
///
/// `shape` is the input shape, `perm` is the permutation of dimensions.
/// Returns `None` if below threshold or on error.
pub fn gpu_transpose(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    perm: &[usize],
) -> Option<Vec<f32>> {
    let ndim = shape.len();
    if ndim == 0 || perm.len() != ndim {
        return None;
    }
    let total: usize = shape.iter().product();
    if total == 0 || input.len() < total {
        return None;
    }
    if total < TRANSPOSE_GPU_THRESHOLD {
        return None;
    }

    // Compute input strides (row-major)
    let mut input_strides = vec![1u32; ndim];
    for i in (0..ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * (shape[i + 1] as u32);
    }

    // Compute output shape and strides
    let mut out_shape = vec![0usize; ndim];
    for d in 0..ndim {
        out_shape[d] = shape[perm[d]];
    }
    let mut output_strides = vec![1u32; ndim];
    for i in (0..ndim - 1).rev() {
        output_strides[i] = output_strides[i + 1] * (out_shape[i + 1] as u32);
    }

    // Build perm_data buffer: [input_strides..., output_strides..., perm...]
    let mut meta_data = Vec::with_capacity(3 * ndim);
    meta_data.extend_from_slice(&input_strides);
    meta_data.extend_from_slice(&output_strides);
    for &p in perm {
        meta_data.push(p as u32);
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tr_input"),
        contents: bytemuck::cast_slice(&input[..total]),
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

    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tr_meta"),
        contents: bytemuck::cast_slice(&meta_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params = TransposeParams {
        total_elements: total as u32,
        ndim: ndim as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tr_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tr_staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tr_bg"),
        layout: &ctx.transpose_bind_group_layout,
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
                resource: meta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (total as u32).div_ceil(256);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tr_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tr_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.transpose_pipeline);
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

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuBufferPool;

    #[test]
    fn test_gpu_buffer_pool_basic() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return, // skip if no GPU
        };

        let mut pool = GpuBufferPool::new(16);
        assert_eq!(pool.available_count(), 0);

        // Get a buffer (creates new since pool is empty).
        let buf = pool.get_buffer(
            &ctx.device,
            1024,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        assert_eq!(pool.available_count(), 0);

        // Return it.
        pool.return_buffer(buf, 1024);
        assert_eq!(pool.available_count(), 1);

        // Clear.
        pool.clear();
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_gpu_buffer_pool_reuse() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let mut pool = GpuBufferPool::new(16);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        // Get and return a 1024-byte buffer.
        let buf = pool.get_buffer(&ctx.device, 1024, usage);
        pool.return_buffer(buf, 1024);
        assert_eq!(pool.available_count(), 1);

        // Request 1024 again — should reuse (count stays 0 after get).
        let _buf2 = pool.get_buffer(&ctx.device, 1024, usage);
        assert_eq!(pool.available_count(), 0);

        // Request something much larger — pool won't have it, creates new.
        let _buf3 = pool.get_buffer(&ctx.device, 1_000_000, usage);
        assert_eq!(pool.available_count(), 0);

        // Return multiple buffers and verify they accumulate.
        let b1 = pool.get_buffer(&ctx.device, 512, usage);
        let b2 = pool.get_buffer(&ctx.device, 2048, usage);
        let b3 = pool.get_buffer(&ctx.device, 4096, usage);
        pool.return_buffer(b1, 512);
        pool.return_buffer(b2, 2048);
        pool.return_buffer(b3, 4096);
        assert_eq!(pool.available_count(), 3);
    }

    #[test]
    fn test_gpu_softmax() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        // Shape: [2, 2000] — last dim > 1000 so GPU should accept.
        let rows = 2usize;
        let cols = 2000usize;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let shape = vec![rows, cols];

        let result = gpu_softmax(&ctx, &data, &shape);
        let result = match result {
            Some(r) => r,
            None => return, // GPU declined
        };

        assert_eq!(result.len(), rows * cols);

        // Verify each row sums to ~1.0.
        for row in 0..rows {
            let row_sum: f32 = result[row * cols..(row + 1) * cols].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 0.01,
                "softmax row {row} sum = {row_sum}, expected ~1.0"
            );
        }

        // Verify all values are non-negative.
        for (i, &v) in result.iter().enumerate() {
            assert!(v >= 0.0, "softmax output[{i}] = {v} is negative");
        }
    }

    #[test]
    fn test_gpu_relu() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let len = 200_000;
        let data: Vec<f32> = (0..len)
            .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
            .collect();

        let result = gpu_relu(&ctx, &data);
        let result = match result {
            Some(r) => r,
            None => return,
        };

        assert_eq!(result.len(), len);
        for (i, (&out, &inp)) in result.iter().zip(data.iter()).enumerate() {
            let expected = inp.max(0.0);
            assert!(
                (out - expected).abs() < 1e-5,
                "relu mismatch at {i}: got {out}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gpu_sigmoid() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let len = 200_000;
        let data: Vec<f32> = (0..len).map(|i| (i as f32 - 100_000.0) * 0.0001).collect();

        let result = gpu_sigmoid(&ctx, &data);
        let result = match result {
            Some(r) => r,
            None => return,
        };

        assert_eq!(result.len(), len);
        for (i, (&out, &inp)) in result.iter().zip(data.iter()).enumerate() {
            let expected = 1.0 / (1.0 + (-inp).exp());
            assert!(
                (out - expected).abs() < 1e-4,
                "sigmoid mismatch at {i}: got {out}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gpu_gelu() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let len = 200_000;
        let data: Vec<f32> = (0..len).map(|i| (i as f32 - 100_000.0) * 0.00005).collect();

        let result = gpu_gelu(&ctx, &data);
        let result = match result {
            Some(r) => r,
            None => return,
        };

        assert_eq!(result.len(), len);
        // Spot-check a few values.
        for &idx in &[0usize, 1000, 50000, 100000, 150000, 199999] {
            if idx >= len {
                continue;
            }
            let x = data[idx];
            let c = 0.797_884_6_f32;
            let inner = c * (x + 0.044715 * x * x * x);
            let expected = 0.5 * x * (1.0 + inner.tanh());
            assert!(
                (result[idx] - expected).abs() < 1e-3,
                "gelu mismatch at {idx}: got {}, expected {expected}",
                result[idx]
            );
        }
    }

    #[test]
    fn test_gpu_layer_norm() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        // [batch=250, n_elements=256] — 64000 > threshold
        let batch = 250usize;
        let n = 256usize;
        let total = batch * n;
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let scale: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002 - 0.1).collect();
        let shape = vec![batch, n];
        let eps = 1e-5_f32;

        let result = match gpu_layer_norm(&ctx, &data, &shape, &scale, &bias, eps) {
            Some(r) => r,
            None => return,
        };

        assert_eq!(result.len(), total);

        // CPU reference for each instance
        for b in 0..batch {
            let row = &data[b * n..(b + 1) * n];
            let mean: f32 = row.iter().sum::<f32>() / n as f32;
            let var: f32 = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..n {
                let expected = (row[i] - mean) * inv_std * scale[i] + bias[i];
                let got = result[b * n + i];
                assert!(
                    (got - expected).abs() < 1e-3,
                    "layer_norm mismatch at batch={b}, i={i}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_gpu_batch_norm() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        // [N=10, C=2, H=50, W=50] → 50000 elements
        let (nn, c, h, w) = (10, 2, 50, 50);
        let total = nn * c * h * w;
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let shape = vec![nn, c, h, w];
        let bn_scale = vec![1.5_f32, 0.8];
        let bn_bias = vec![0.1_f32, -0.2];
        let bn_mean = vec![0.5_f32, -0.3];
        let bn_var = vec![1.0_f32, 2.0];
        let eps = 1e-5_f32;

        let result = match gpu_batch_norm(
            &ctx, &data, &shape, &bn_scale, &bn_bias, &bn_mean, &bn_var, eps,
        ) {
            Some(r) => r,
            None => return,
        };

        assert_eq!(result.len(), total);

        let spatial = h * w;
        // CPU reference
        for idx in 0..total {
            let ch = (idx / spatial) % c;
            let x = data[idx];
            let expected =
                bn_scale[ch] * (x - bn_mean[ch]) / (bn_var[ch] + eps).sqrt() + bn_bias[ch];
            assert!(
                (result[idx] - expected).abs() < 1e-3,
                "batch_norm mismatch at {idx}: got {}, expected {expected}",
                result[idx]
            );
        }
    }

    #[test]
    fn test_gpu_transpose() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        // [200, 256] → [256, 200] — 51200 > threshold
        let (rows, cols) = (200usize, 256usize);
        let total = rows * cols;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let shape = vec![rows, cols];
        let perm = vec![1, 0];

        let result = match gpu_transpose(&ctx, &data, &shape, &perm) {
            Some(r) => r,
            None => return,
        };

        assert_eq!(result.len(), total);

        // CPU reference: output[j*rows+i] = input[i*cols+j]
        for i in 0..rows {
            for j in 0..cols {
                let expected = data[i * cols + j];
                let got = result[j * rows + i];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "transpose mismatch at ({i},{j}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_gpu_reduce_mean() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        // [200, 300, 4] along axis 1 → output [200, 4] = 800 elements
        // total input = 200*300*4 = 240000 > threshold, but output = 800 < 50000
        // Use bigger dims: [200, 300] along axis 1 → output [200]
        // output = 200 < 50000... need bigger.
        // [1000, 100] along axis 1 → output [1000] < 50000
        // Use [256, 256] along axis 1 → output [256] < 50000
        // Need output >= 50000: [500, 200, 2] along axis 1 → output = 500*2 = 1000
        // Actually the threshold check for reduce is on output elements, so we need
        // output >= 50000. Let's use a large enough shape.
        // [1000, 100] axis=0 → output [100] (too small)
        // [100000, 3] axis=1 → output [100000] (output >= 50000)
        let (d0, d1) = (100_000usize, 3usize);
        let total = d0 * d1;
        let data: Vec<f32> = (0..total).map(|i| (i % 100) as f32 * 0.1).collect();
        let shape = vec![d0, d1];

        let result = match gpu_reduce_mean(&ctx, &data, &shape, &[1], false) {
            Some(r) => r,
            None => return,
        };

        assert_eq!(result.len(), d0);

        // CPU reference
        for i in 0..d0 {
            let start = i * d1;
            let sum: f32 = data[start..start + d1].iter().sum();
            let expected = sum / d1 as f32;
            assert!(
                (result[i] - expected).abs() < 1e-4,
                "reduce_mean mismatch at {i}: got {}, expected {expected}",
                result[i]
            );
        }
    }
}
