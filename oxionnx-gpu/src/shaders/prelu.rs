//! GPU-accelerated PRelu: `f(x) = x if x >= 0, slope[c] * x if x < 0`.
//!
//! A one-line variant of `elementwise.rs`'s `leaky_relu` kernel (see its
//! `select(params.alpha * x, x, x >= 0.0)` body) with the single scalar
//! `alpha` uniform replaced by a per-channel slope storage buffer, matching
//! `oxionnx-ops::nn::activations::prelu`'s per-channel branch (`x` is
//! `[N, C, ...]`, `slope` is `[C]`). A slope of length 1 (a scalar PRelu, or
//! the degenerate `C == 1` case) is also supported: every element then uses
//! `slope[0]`.
//!
//! See [`kernel_support`](super::kernel_support) for why this kernel's
//! pipeline is rebuilt on every call and why there is no minimum-size gate.

use crate::context::activation::{GpuOutput, OutputPlacement, TensorSource};
use crate::context::GpuContext;
use crate::device_guard::{
    block_on_gpu, checked_storage_bytes, finish_output_async, plan_dispatch, ErrorScope,
};

use super::kernel_support::{bgl_ro, bgl_rw, bgl_uniform, build_pipeline, WG_SIZE};

/// Uniform block for the PRelu kernel.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PreluParams {
    total_len: u32,
    channels: u32,
    /// Product of every dimension after the channel axis (`H * W * ...`).
    spatial: u32,
    slope_len: u32,
    row_threads: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const PRELU_SHADER: &str = r#"
struct Params {
    total_len: u32,
    channels: u32,
    spatial: u32,
    slope_len: u32,
    row_threads: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> slope: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.y * params.row_threads + gid.x;
}

@compute @workgroup_size(256)
fn prelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    // NCHW-style layout: channel = (idx / spatial) % channels.
    let channel = (idx / params.spatial) % params.channels;
    // Select the *index* first (not `select(slope[channel], slope[0], ...)`,
    // which would evaluate both operands and read `slope[channel]`
    // out-of-bounds whenever `slope_len == 1` and `channel > 0`).
    let alpha_idx = select(channel, 0u, params.slope_len == 1u);
    let alpha = slope[alpha_idx];
    let x = input[idx];
    output[idx] = select(alpha * x, x, x >= 0.0);
}
"#;

pub(crate) fn build_prelu_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_pipeline(
        device,
        "prelu",
        PRELU_SHADER,
        "prelu",
        &[bgl_ro(0), bgl_ro(1), bgl_rw(2), bgl_uniform(3)],
    )
}

/// GPU-accelerated PRelu.
///
/// `shape` is `[N, C, ...]` (rank >= 2; any number of trailing spatial
/// dimensions). `slope` must have length `shape[1]` (per-channel) or `1`
/// (scalar, broadcast to every element). Declines (`None`) on any other
/// slope length, a shape of rank < 2, a `data` length that does not match
/// `shape`, or a degraded/unavailable context -- never silently
/// mis-broadcasts.
pub async fn gpu_prelu_async(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    slope: &[f32],
) -> Option<Vec<f32>> {
    gpu_prelu_placed_async(
        ctx,
        TensorSource::host(data, shape),
        TensorSource::host(slope, &[slope.len()]),
        OutputPlacement::Host,
    )
    .await?
    .into_vec()
}

/// [`gpu_prelu_async`] over operands that may already be on the device, with a
/// result that may stay there.
pub async fn gpu_prelu_placed_async(
    ctx: &GpuContext,
    data: TensorSource<'_>,
    slope: TensorSource<'_>,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    if ctx.is_degraded() {
        return None;
    }
    let shape = data.shape();
    if shape.len() < 2 {
        return None;
    }
    let channels = shape[1];
    let spatial: usize = shape[2..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))?;
    let total_len = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))?;
    if total_len == 0 || data.len() != total_len {
        return None;
    }
    if slope.len() != channels && slope.len() != 1 {
        return None;
    }

    let in_size = checked_storage_bytes(&ctx.limits, total_len)?;
    let slope_size = checked_storage_bytes(&ctx.limits, slope.len())?;
    if !ctx.limits.buffer_fits(in_size) || !ctx.limits.buffer_fits(slope_size) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, total_len as u64, WG_SIZE)?;
    // Input, slope, output and read-back staging — minus the ones this
    // dispatch will not allocate.
    if !ctx.budget_admits(&[
        ctx.source_admission_bytes(data, in_size),
        ctx.source_admission_bytes(slope, slope_size),
        in_size,
        placement.staging_bytes(in_size),
    ]) {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let scope = ErrorScope::begin(ctx);
    let (pipeline, bgl) = build_prelu_pipeline(device);

    let input_buf = ctx.operand_source("prelu_in", data, wgpu::BufferUsages::STORAGE)?;
    let slope_buf = ctx.operand_source("prelu_slope", slope, wgpu::BufferUsages::STORAGE)?;
    let output_buf = ctx.pooled_buffer(
        in_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;
    // The pool may hand back a buffer up to 2x `in_size` (PRelu's output is
    // the same length as its input -- see `GpuBufferPool::get_buffer`), and
    // `as_entire_binding()` would then bind that larger size, which can
    // exceed `max_storage_buffer_binding_size` even though `in_size` itself
    // was validated. Bind the exact range instead, as
    // `conv2d::gpu_conv2d_implicit_resident_async`'s `output_binding` does.
    let output_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &output_buf,
        offset: 0,
        size: wgpu::BufferSize::new(in_size),
    });

    let params = PreluParams {
        total_len: total_len as u32,
        channels: channels as u32,
        spatial: spatial as u32,
        slope_len: slope.len() as u32,
        row_threads: grid.threads_per_row,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = ctx.upload_buffer(
        "prelu_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("prelu_staging", in_size)?),
        OutputPlacement::Device => None,
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("prelu_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: slope_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_binding,
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("prelu_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prelu_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }
    if let Some(staging) = &staging_buf {
        encoder.copy_buffer_to_buffer(&output_buf, 0, staging, 0, in_size);
    }
    queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish_async(ctx).await {
        return None;
    }
    finish_output_async(
        ctx,
        placement,
        staging_buf,
        output_buf,
        total_len,
        in_size,
        shape.to_vec(),
    )
    .await
}

/// Blocking form of [`gpu_prelu_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_prelu(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    slope: &[f32],
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_prelu_async(ctx, data, shape, slope))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_prelu_declines_rank_below_2() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        assert!(gpu_prelu(&ctx, &[1.0, -1.0], &[2], &[0.5]).is_none());
    }

    #[test]
    fn gpu_prelu_declines_mismatched_slope_length() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        // shape [1,3,2,2] (C=3) but slope has 4 entries: neither 1 nor C.
        let data = vec![1.0f32; 12];
        let slope = vec![0.1f32; 4];
        assert!(gpu_prelu(&ctx, &data, &[1, 3, 2, 2], &slope).is_none());
    }

    #[test]
    fn gpu_prelu_declines_data_shape_mismatch() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let data = vec![1.0f32; 10]; // shape below claims 12
        assert!(gpu_prelu(&ctx, &data, &[1, 3, 2, 2], &[0.1, 0.2, 0.3]).is_none());
    }

    /// `gpu_prelu` (the `block_on_gpu` wrapper) and `gpu_prelu_async` (the
    /// real implementation) must dispatch the same kernel on the same input
    /// and produce identical output -- `gpu_prelu` is nothing more than
    /// `block_on_gpu(gpu_prelu_async(..))`. `.expect` on both sides (not a
    /// bare `assert_eq!` of the `Option`s) so a future decline-path change
    /// that makes both sides silently return `None` fails this test instead
    /// of passing vacuously.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_prelu_async_matches_sync() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let data: Vec<f32> = (0..48).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
        let shape = [1usize, 3, 4, 4];
        let slope = [0.1f32, 0.2, 0.3];

        let sync_result = gpu_prelu(&ctx, &data, &shape, &slope).expect("gpu_prelu must dispatch");
        let async_result = pollster::block_on(gpu_prelu_async(&ctx, &data, &shape, &slope))
            .expect("gpu_prelu_async must dispatch on the same input");
        assert_eq!(
            sync_result, async_result,
            "sync and async entry points must produce identical output"
        );
    }
}
