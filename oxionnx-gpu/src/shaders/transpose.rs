//! GPU-accelerated general Transpose with arbitrary permutation.

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

use super::common::{read_back, TransposeParams, TRANSPOSE_GPU_THRESHOLD};

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
