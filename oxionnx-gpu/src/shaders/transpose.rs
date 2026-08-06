//! GPU-accelerated general Transpose with arbitrary permutation.

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

use super::common::{
    checked_storage_bytes, plan_dispatch, read_back_and_recycle, ErrorScope, TransposeParams,
    TRANSPOSE_GPU_THRESHOLD, WG_SIZE,
};

// ========================================================================
// Transpose
// ========================================================================

/// True when `perm` is a genuine permutation of `0..ndim`.
///
/// `perm` comes straight out of the model file, so an out-of-range or repeated
/// entry must decline rather than index out of bounds.
fn is_valid_perm(perm: &[usize], ndim: usize) -> bool {
    if perm.len() != ndim {
        return false;
    }
    let mut seen = vec![false; ndim];
    for &p in perm {
        match seen.get_mut(p) {
            Some(slot) if !*slot => *slot = true,
            // Out of range, or already used: not a permutation.
            _ => return false,
        }
    }
    true
}

/// GPU-accelerated general Transpose with arbitrary permutation.
///
/// `shape` is the input shape, `perm` is the permutation of dimensions.
/// Returns `None` if below threshold, if `perm` is not a permutation of
/// `0..shape.len()`, or on error — the CPU operator then reports the malformed
/// attribute as a typed error.
pub fn gpu_transpose(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    perm: &[usize],
) -> Option<Vec<f32>> {
    let ndim = shape.len();
    if ndim == 0 || ctx.is_degraded() || !is_valid_perm(perm, ndim) {
        return None;
    }
    // A zero dimension makes the output strides zero, which the kernel would
    // divide by; the total-element guard below rejects those shapes.
    let total: usize = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d))?;
    if total == 0 || input.len() < total {
        return None;
    }
    if total < TRANSPOSE_GPU_THRESHOLD {
        return None;
    }
    // Strides and the flat index are `u32` in the kernel.
    let total_u32 = u32::try_from(total).ok()?;

    // Compute input strides (row-major)
    let mut input_strides = vec![1u32; ndim];
    for i in (0..ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1].checked_mul(u32::try_from(shape[i + 1]).ok()?)?;
    }

    // Compute output shape and strides
    let mut out_shape = vec![0usize; ndim];
    for (d, slot) in out_shape.iter_mut().enumerate() {
        // `is_valid_perm` guarantees `perm[d] < ndim`.
        *slot = *shape.get(*perm.get(d)?)?;
    }
    let mut output_strides = vec![1u32; ndim];
    for i in (0..ndim - 1).rev() {
        output_strides[i] =
            output_strides[i + 1].checked_mul(u32::try_from(out_shape[i + 1]).ok()?)?;
    }

    // Build perm_data buffer: [input_strides..., output_strides..., perm...]
    let mut meta_data = Vec::with_capacity(3 * ndim);
    meta_data.extend_from_slice(&input_strides);
    meta_data.extend_from_slice(&output_strides);
    for &p in perm {
        meta_data.push(u32::try_from(p).ok()?);
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let out_size = checked_storage_bytes(&ctx.limits, total)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, total as u64, WG_SIZE)?;

    let scope = ErrorScope::begin(ctx);

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tr_input"),
        contents: bytemuck::cast_slice(&input[..total]),
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

    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tr_meta"),
        contents: bytemuck::cast_slice(&meta_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params = TransposeParams {
        total_elements: total_u32,
        ndim: u32::try_from(ndim).ok()?,
        row_threads: grid.threads_per_row,
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
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish(ctx) {
        return None;
    }

    read_back_and_recycle(ctx, &staging_buf, total, output_buf)
}
