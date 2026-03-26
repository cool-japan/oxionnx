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
            let c = 0.7978845608_f32;
            let inner = c * (x + 0.044715 * x * x * x);
            let expected = 0.5 * x * (1.0 + inner.tanh());
            assert!(
                (result[idx] - expected).abs() < 1e-3,
                "gelu mismatch at {idx}: got {}, expected {expected}",
                result[idx]
            );
        }
    }
}
