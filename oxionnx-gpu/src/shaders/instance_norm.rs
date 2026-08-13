//! GPU-accelerated `OxiInstanceNorm`: spatial mean/variance normalization
//! without an affine term.
//!
//! `out = (x - mean) / sqrt(var + eps)`, with `mean` and `var` taken over
//! `shape[2..]` independently for each `(n, c)` pair. This is the kernel for
//! the fused node the `fuse_instance_norm` optimizer pass emits for AdaIN
//! generators, whose per-channel scale and shift are runtime tensors and
//! therefore stay outside the fused op (see `oxionnx`'s
//! `optimizer::fusion::fuse_instance_norm`).
//!
//! # Shape
//!
//! One workgroup of 256 threads owns one `(n, c)` plane and reduces it in
//! shared memory — the same structure as `LAYER_NORM_SHADER` in
//! `context/functions.rs`, but reducing over the *spatial* suffix rather than
//! the last axis, and with no `scale`/`bias` buffers. Planes are dispatched
//! across a 2-D grid when there are more of them than the device allows along
//! one dimension, exactly as the LayerNorm path does.
//!
//! # Numerics
//!
//! Two passes (mean, then variance about that mean), matching
//! `oxionnx-ops`'s CPU kernel and the arithmetic of the graph both replace.
//! The one-pass `mean(x²) − mean(x)²` form would be a single reduction but
//! loses catastrophically on large-magnitude activations, and would no longer
//! agree with the CPU path element for element.
//!
//! # Async-first
//!
//! [`gpu_instance_norm_async`] is the implementation;
//! [`gpu_instance_norm`] is a one-line `block_on_gpu` wrapper that declines
//! outright on wasm32 (where the future would be dropped unpolled). Anything
//! that must work in a browser has to await the async form.
//!
//! See [`kernel_support`](super::kernel_support) for why this kernel builds its
//! pipeline per call and carries no minimum-size gate.

use crate::context::activation::{GpuOutput, OutputPlacement, TensorSource};
use crate::context::GpuContext;
use crate::device_guard::{
    block_on_gpu, checked_storage_bytes, finish_output_async, plan_dispatch, ErrorScope,
};

use super::kernel_support::{bgl_ro, bgl_rw, bgl_uniform, build_pipeline};

/// Uniform block for the spatial-normalization kernel.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceNormParams {
    /// Elements in one `(n, c)` plane (`product(shape[2..])`).
    spatial: u32,
    /// Number of planes (`shape[0] * shape[1]`).
    plane_count: u32,
    eps: f32,
    /// Workgroups along X, so the shader can rebuild the plane index from a
    /// 2-D grid.
    wg_per_row: u32,
}

/// One workgroup of 256 threads reduces one `(n, c)` plane.
///
/// The width appears three times inside the WGSL — the `WG_SIZE` constant the
/// strided accumulation loops step by, the `@workgroup_size` attribute, and the
/// `array<f32, 256>` workgroup buffer the reduction tree halves through — and
/// all three must agree or the tree reads past the partials it wrote. Nothing
/// on the Rust side needs the number (the dispatch is one *workgroup* per
/// plane, so `plan_dispatch` is called with a workgroup size of 1), so it is
/// not mirrored as a constant here; `reduction_width_matches_the_shader` pins
/// the three occurrences against each other instead.
const INSTANCE_NORM_SHADER: &str = r#"
struct Params {
    spatial: u32,
    plane_count: u32,
    eps: f32,
    wg_per_row: u32,
}

const WG_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(256)
fn instance_norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // One workgroup per (n, c) plane. `wid` is uniform across the workgroup,
    // so this early exit cannot desynchronise the barriers below.
    let plane = wid.y * params.wg_per_row + wid.x;
    if (plane >= params.plane_count) { return; }
    let tid = lid.x;
    let n = params.spatial;
    let base = plane * n;

    // Phase 1: sum -> mean.
    var local_sum: f32 = 0.0;
    for (var step: u32 = 0u; step < n; step = step + WG_SIZE) {
        let i = tid + step;
        if (i < n) {
            local_sum = local_sum + input[base + i];
        }
    }
    partials[tid] = local_sum;
    workgroupBarrier();

    for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            partials[tid] = partials[tid] + partials[tid + s];
        }
        workgroupBarrier();
    }

    let mean_val = partials[0] / f32(n);
    workgroupBarrier();

    // Phase 2: sum of squared deviations -> variance.
    var local_var: f32 = 0.0;
    for (var step: u32 = 0u; step < n; step = step + WG_SIZE) {
        let i = tid + step;
        if (i < n) {
            let diff = input[base + i] - mean_val;
            local_var = local_var + diff * diff;
        }
    }
    partials[tid] = local_var;
    workgroupBarrier();

    for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            partials[tid] = partials[tid] + partials[tid + s];
        }
        workgroupBarrier();
    }

    let variance = partials[0] / f32(n);
    let inv_std = 1.0 / sqrt(variance + params.eps);
    workgroupBarrier();

    // Phase 3: normalize. No scale/bias: the affine term is a separate node.
    for (var step: u32 = 0u; step < n; step = step + WG_SIZE) {
        let i = tid + step;
        if (i < n) {
            output[base + i] = (input[base + i] - mean_val) * inv_std;
        }
    }
}
"#;

fn build_instance_norm_pipeline(
    ctx: &GpuContext,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_pipeline(
        ctx,
        "instance_norm",
        INSTANCE_NORM_SHADER,
        "instance_norm",
        &[bgl_ro(0), bgl_rw(1), bgl_uniform(2)],
    )
}

/// GPU-accelerated spatial (InstanceNorm-style) normalization, no affine term.
///
/// `shape` is `[N, C, d1, …]` with rank >= 3; the reduction runs over
/// `d1, …` for each `(n, c)` pair. Declines (`None`) — never guesses — on a
/// rank below 3, an `input` length that disagrees with `shape`, a zero
/// dimension, a buffer the device cannot bind, a dispatch wider than the
/// device allows, or a degraded context.
pub async fn gpu_instance_norm_async(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    eps: f32,
) -> Option<Vec<f32>> {
    gpu_instance_norm_placed_async(
        ctx,
        TensorSource::host(input, shape),
        eps,
        OutputPlacement::Host,
    )
    .await?
    .into_vec()
}

/// [`gpu_instance_norm_async`] over an operand that may already be on the
/// device, with a result that may stay there.
pub async fn gpu_instance_norm_placed_async(
    ctx: &GpuContext,
    input: TensorSource<'_>,
    eps: f32,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    let shape = input.shape();
    if ctx.is_degraded() || shape.len() < 3 {
        return None;
    }
    let plane_count = shape[..2]
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))?;
    let spatial = shape[2..]
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))?;
    if plane_count == 0 || spatial == 0 {
        return None;
    }
    let total = plane_count.checked_mul(spatial)?;
    if input.len() != total {
        return None;
    }

    let out_size = checked_storage_bytes(&ctx.limits, total)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    // One workgroup per plane: pass a workgroup size of 1 so `plan_dispatch`
    // lays out `plane_count` *workgroups*, not `plane_count` threads.
    let grid = plan_dispatch(&ctx.limits, plane_count as u64, 1)?;

    let params = InstanceNormParams {
        spatial: u32::try_from(spatial).ok()?,
        plane_count: u32::try_from(plane_count).ok()?,
        eps,
        wg_per_row: grid.x,
    };

    // Input, output and read-back staging, all the same length here — minus
    // the ones this dispatch will not allocate.
    if !ctx.budget_admits(&[
        ctx.source_admission_bytes(input, out_size),
        out_size,
        placement.staging_bytes(out_size),
    ]) {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let scope = ErrorScope::begin(ctx);
    let (pipeline, bgl) = build_instance_norm_pipeline(ctx);

    let input_buf = ctx.operand_source("in_input", input, wgpu::BufferUsages::STORAGE)?;
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
    let params_buf = ctx.upload_buffer(
        "in_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;
    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("in_staging", out_size)?),
        OutputPlacement::Device => None,
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("in_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_binding,
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("in_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("in_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }
    if let Some(staging) = &staging_buf {
        encoder.copy_buffer_to_buffer(&output_buf, 0, staging, 0, out_size);
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
        total,
        out_size,
        shape.to_vec(),
    )
    .await
}

/// Blocking form of [`gpu_instance_norm_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`): the future would be
/// dropped unpolled there, so a browser caller must await the async form.
pub fn gpu_instance_norm(
    ctx: &GpuContext,
    input: &[f32],
    shape: &[usize],
    eps: f32,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_instance_norm_async(ctx, input, shape, eps))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU reference: the same two-pass arithmetic `oxionnx-ops`'
    /// `oxi_instance_norm` performs. Duplicated here rather than imported
    /// because `oxionnx-gpu` does not depend on `oxionnx-ops` — which is also
    /// what makes it an independent check.
    fn cpu_reference(input: &[f32], shape: &[usize], eps: f32) -> Vec<f32> {
        let spatial: usize = shape[2..].iter().product();
        let mut out = vec![0.0f32; input.len()];
        for (plane_in, plane_out) in input.chunks(spatial).zip(out.chunks_mut(spatial)) {
            let count = spatial as f32;
            let mean = plane_in.iter().sum::<f32>() / count;
            let var = plane_in
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>()
                / count;
            let inv_std = 1.0 / (var + eps).sqrt();
            for (dst, &src) in plane_out.iter_mut().zip(plane_in.iter()) {
                *dst = (src - mean) * inv_std;
            }
        }
        out
    }

    fn ramp(shape: &[usize], scale: f32, offset: f32) -> Vec<f32> {
        let n: usize = shape.iter().product();
        (0..n)
            .map(|i| offset + scale * ((i % 23) as f32 - 11.0) + 0.5 * (i as f32).cos())
            .collect()
    }

    /// `atol + rtol * |expected|`: the normalised output has zero mean per
    /// plane, so elements sit near zero and a pure relative bound would be
    /// meaningless there.
    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= atol + rtol * e.abs(),
                "element {i}: {a} vs {e}"
            );
        }
    }

    /// Guards every other test in this module against a false green.
    ///
    /// The parity tests below follow this crate's house idiom and return early
    /// when `GpuContext::try_new()` yields `None`, which makes "no adapter" and
    /// "the kernel matched the CPU" look identical from the outside. This test
    /// closes the *other* half of that hole: given an adapter that is present
    /// and not degraded, a valid shape must actually dispatch. A `None` here
    /// means the parity assertions are running on nothing.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn kernel_dispatches_when_an_adapter_is_present() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        if ctx.is_degraded() {
            return;
        }
        let shape = [1usize, 2, 4, 4];
        let input = ramp(&shape, 1.0, 0.0);
        assert!(
            gpu_instance_norm(&ctx, &input, &shape, 1e-8).is_some(),
            "an adapter is present and the shape is valid, so the kernel must \
             dispatch — a None here would make every parity test in this module \
             pass vacuously"
        );
    }

    #[test]
    fn matches_cpu_reference_nchw() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let shape = [2usize, 3, 8, 8];
        let input = ramp(&shape, 1.5, 4.0);
        let eps = 1e-8;
        let out = match gpu_instance_norm(&ctx, &input, &shape, eps) {
            Some(out) => out,
            // wasm32 (or a device that declined): nothing to compare.
            None => return,
        };
        assert_close(&out, &cpu_reference(&input, &shape, eps), 1e-5, 1e-5);
    }

    /// A plane longer than one workgroup exercises the strided accumulation
    /// loop; a plane shorter than one exercises the `i < n` guard.
    #[test]
    fn matches_cpu_reference_across_workgroup_boundary() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        for shape in [
            [1usize, 2, 1, 3],   // 3 elements per plane, far below WG_SIZE
            [1usize, 1, 16, 16], // exactly WG_SIZE
            [1usize, 2, 20, 37], // 740: not a multiple of WG_SIZE
            [3usize, 4, 32, 32], // 1024 per plane, 12 planes
        ] {
            let input = ramp(&shape, 0.8, -3.0);
            let eps = 1e-6;
            let out = match gpu_instance_norm(&ctx, &input, &shape, eps) {
                Some(out) => out,
                None => return,
            };
            assert_close(&out, &cpu_reference(&input, &shape, eps), 1e-4, 1e-4);
        }
    }

    /// Rank 3 (`[N, C, L]`): the spatial suffix is a single axis.
    #[test]
    fn matches_cpu_reference_rank_3() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let shape = [2usize, 4, 50];
        let input = ramp(&shape, 2.0, 1.0);
        let eps = 1e-5;
        let out = match gpu_instance_norm(&ctx, &input, &shape, eps) {
            Some(out) => out,
            None => return,
        };
        assert_close(&out, &cpu_reference(&input, &shape, eps), 1e-5, 1e-5);
    }

    /// `epsilon` must reach the shader: a large value visibly shrinks the
    /// result, and a kernel that hard-coded a stabiliser would not follow.
    #[test]
    fn epsilon_reaches_the_shader() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let shape = [1usize, 1, 8, 8];
        let input = ramp(&shape, 1.0, 0.0);
        let loose = match gpu_instance_norm(&ctx, &input, &shape, 100.0) {
            Some(out) => out,
            None => return,
        };
        assert_close(&loose, &cpu_reference(&input, &shape, 100.0), 1e-5, 1e-5);
        let tight_energy: f32 = cpu_reference(&input, &shape, 1e-8)
            .iter()
            .map(|v| v * v)
            .sum();
        let loose_energy: f32 = loose.iter().map(|v| v * v).sum();
        assert!(loose_energy < tight_energy * 0.5, "epsilon had no effect");
    }

    /// A constant plane has zero variance; `epsilon` is the only thing keeping
    /// the division finite.
    #[test]
    fn constant_plane_stays_finite() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let shape = [1usize, 2, 4, 4];
        let input = vec![7.0f32; 32];
        let out = match gpu_instance_norm(&ctx, &input, &shape, 1e-8) {
            Some(out) => out,
            None => return,
        };
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        assert!(out.iter().all(|v| v.abs() < 1e-3), "{out:?}");
    }

    /// The async form is the implementation; the sync wrapper must agree with
    /// it wherever the wrapper produces anything at all.
    #[test]
    fn async_and_sync_forms_agree() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let shape = [1usize, 3, 6, 6];
        let input = ramp(&shape, 1.1, 2.0);
        let sync = gpu_instance_norm(&ctx, &input, &shape, 1e-7);
        let asynced = pollster::block_on(gpu_instance_norm_async(&ctx, &input, &shape, 1e-7));
        match (sync, asynced) {
            (Some(a), Some(b)) => assert_eq!(a, b),
            (None, _) => {}
            (Some(_), None) => panic!("sync produced a result the async form did not"),
        }
    }

    #[test]
    fn declines_rank_below_3() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        assert!(gpu_instance_norm(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2], 1e-5).is_none());
    }

    #[test]
    fn declines_on_data_shape_mismatch() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        // shape claims 1*2*3*3 = 18 elements, the slice has 10.
        let input = vec![1.0f32; 10];
        assert!(gpu_instance_norm(&ctx, &input, &[1, 2, 3, 3], 1e-5).is_none());
    }

    #[test]
    fn declines_on_a_zero_dimension() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        assert!(gpu_instance_norm(&ctx, &[], &[1, 2, 0, 3], 1e-5).is_none());
    }

    /// The three places the reduction width appears in the WGSL must agree:
    /// the loop stride constant, the `@workgroup_size` attribute and the
    /// workgroup buffer the tree reduction halves through. A mismatch reads
    /// past the partials the workgroup actually wrote.
    #[test]
    fn reduction_width_matches_the_shader() {
        assert!(INSTANCE_NORM_SHADER.contains("const WG_SIZE: u32 = 256u;"));
        assert!(INSTANCE_NORM_SHADER.contains("@compute @workgroup_size(256)"));
        assert!(INSTANCE_NORM_SHADER.contains("array<f32, 256>"));
    }
}
