//! GPU-accelerated softmax dispatch.

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

use super::common::{read_back, SoftmaxParams, SOFTMAX_DIM_THRESHOLD};

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
