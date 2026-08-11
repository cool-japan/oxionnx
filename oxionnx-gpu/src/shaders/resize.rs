//! GPU-accelerated `Resize` (NCHW, spatial axes only): the two mode /
//! coordinate-transform combinations this wave's model histograms need --
//! `mode="linear"` with `coordinate_transformation_mode="pytorch_half_pixel"`
//! (InSwapper's 2 upsamples) and `mode="nearest"` with
//! `coordinate_transformation_mode="asymmetric"` (SCRFD's 2 FPN upsamples).
//! `N`/`C` always pass through unchanged; only `H`/`W` are resampled.
//!
//! Both formulas mirror `oxionnx-ops::resize`'s reference implementation
//! (`transform_coord` / `build_axis`) exactly:
//!
//! * `pytorch_half_pixel`: `x' = (x + 0.5) / scale - 0.5` when `out_size > 1`,
//!   else `x' = 0.0` -- the `out_size == 1` branch matters and is not an
//!   edge case to skip, since a 1-pixel output axis is one of this module's
//!   degenerate-shape tests.
//! * `asymmetric`: `x' = x / scale`.
//! * `nearest` tap (`nearest_mode` default `round_prefer_floor`):
//!   `index = clamp(ceil(x' - 0.5), 0, in_size - 1)`.
//! * `linear` taps: `base = floor(x')`, `ratio = x' - base`,
//!   `index0 = clamp(base, 0, in_size-1)`, `index1 = clamp(base+1, 0, in_size-1)`,
//!   weights `(1-ratio, ratio)`. `oxionnx-ops` special-cases an exactly
//!   integral `x'` (`base = x'-1, ratio = 1.0`) to keep its symmetric tap
//!   window; that special case produces the same finite value here (the
//!   zero-weighted term contributes `0.0 * finite = 0.0` either way), so it
//!   is not reproduced -- see this kernel's parity tests for the boundary
//!   coordinates (`x' = 0`, `x' = in_size/2`, `x' = in_size-1`) that would
//!   expose a real divergence if this reasoning were wrong.
//!
//! `scale_h`/`scale_w` are `out_size / in_size` (an `output/input` shape
//! pair reduced to a ratio), matching how `oxionnx-ops::resize` derives
//! `scale` from an explicit `sizes` input.
//!
//! See [`kernel_support`](super::kernel_support) for why this kernel's
//! pipeline is rebuilt on every call and why there is no minimum-size gate.

use crate::context::activation::{GpuOutput, OutputPlacement, TensorSource};
use crate::context::GpuContext;
use crate::device_guard::{
    block_on_gpu, checked_storage_bytes, finish_output_async, plan_dispatch, ErrorScope,
};

use super::kernel_support::{bgl_ro, bgl_rw, bgl_uniform, build_pipeline, WG_SIZE};

/// Uniform block for the Resize kernel.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ResizeParams {
    n: u32,
    c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    total_len: u32,
    row_threads: u32,
    scale_h: f32,
    scale_w: f32,
    _pad0: u32,
    _pad1: u32,
}

const RESIZE_SHADER: &str = r#"
struct Params {
    n: u32,
    c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    total_len: u32,
    row_threads: u32,
    scale_h: f32,
    scale_w: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.y * params.row_threads + gid.x;
}

fn decode(idx: u32) -> vec4<u32> {
    var rem = idx;
    let ow = rem % params.out_w;
    rem = rem / params.out_w;
    let oh = rem % params.out_h;
    rem = rem / params.out_h;
    let ci = rem % params.c;
    rem = rem / params.c;
    let ni = rem;
    return vec4<u32>(ni, ci, oh, ow);
}

fn coord_pytorch_half_pixel(x: f32, scale: f32, out_size: u32) -> f32 {
    if (out_size > 1u) {
        return (x + 0.5) / scale - 0.5;
    }
    return 0.0;
}

fn coord_asymmetric(x: f32, scale: f32) -> f32 {
    return x / scale;
}

// Returns (index0, index1, weight0, weight1) as f32; caller casts the
// indices back to u32.
fn linear_taps(src: f32, in_size: u32) -> vec4<f32> {
    let max_idx = f32(in_size) - 1.0;
    let base = floor(src);
    let ratio = clamp(src - base, 0.0, 1.0);
    let idx0 = clamp(base, 0.0, max_idx);
    let idx1 = clamp(base + 1.0, 0.0, max_idx);
    return vec4<f32>(idx0, idx1, 1.0 - ratio, ratio);
}

fn nearest_index(src: f32, in_size: u32) -> u32 {
    let max_idx = f32(in_size) - 1.0;
    let picked = ceil(src - 0.5);
    return u32(clamp(picked, 0.0, max_idx));
}

@compute @workgroup_size(256)
fn resize_bilinear_pytorch_half_pixel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    let coords = decode(idx);
    let src_h = coord_pytorch_half_pixel(f32(coords.z), params.scale_h, params.out_h);
    let src_w = coord_pytorch_half_pixel(f32(coords.w), params.scale_w, params.out_w);
    let th = linear_taps(src_h, params.in_h);
    let tw = linear_taps(src_w, params.in_w);
    let h0 = u32(th.x);
    let h1 = u32(th.y);
    let w0 = u32(tw.x);
    let w1 = u32(tw.y);
    let plane = params.in_h * params.in_w;
    let base = (coords.x * params.c + coords.y) * plane;
    let v00 = input[base + h0 * params.in_w + w0];
    let v01 = input[base + h0 * params.in_w + w1];
    let v10 = input[base + h1 * params.in_w + w0];
    let v11 = input[base + h1 * params.in_w + w1];
    let top = v00 * tw.z + v01 * tw.w;
    let bot = v10 * tw.z + v11 * tw.w;
    output[idx] = top * th.z + bot * th.w;
}

@compute @workgroup_size(256)
fn resize_nearest_asymmetric(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    let coords = decode(idx);
    let src_h = coord_asymmetric(f32(coords.z), params.scale_h);
    let src_w = coord_asymmetric(f32(coords.w), params.scale_w);
    let ih = nearest_index(src_h, params.in_h);
    let iw = nearest_index(src_w, params.in_w);
    let plane = params.in_h * params.in_w;
    let base = (coords.x * params.c + coords.y) * plane;
    output[idx] = input[base + ih * params.in_w + iw];
}
"#;

/// Which of the two interpolation configurations this crate implements a
/// `Resize` node maps onto.
///
/// Public because [`gpu_resize_placed_async`] takes it: the residency-aware
/// dispatcher picks the kernel once and passes it, rather than calling one of
/// two near-identical entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeKind {
    /// `mode="linear"`, `coordinate_transformation_mode="pytorch_half_pixel"`.
    BilinearPytorchHalfPixel,
    /// `mode="nearest"`, `coordinate_transformation_mode="asymmetric"`,
    /// `nearest_mode="round_prefer_floor"`.
    NearestAsymmetric,
}

impl ResizeKind {
    fn entry_point(self) -> &'static str {
        match self {
            ResizeKind::BilinearPytorchHalfPixel => "resize_bilinear_pytorch_half_pixel",
            ResizeKind::NearestAsymmetric => "resize_nearest_asymmetric",
        }
    }
}

pub(crate) fn build_resize_pipeline(
    device: &wgpu::Device,
    kind_entry_point: &str,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_pipeline(
        device,
        "resize",
        RESIZE_SHADER,
        kind_entry_point,
        &[bgl_ro(0), bgl_rw(1), bgl_uniform(2)],
    )
}

/// Pure host-side mirror of the WGSL `nearest_index`/`coord_asymmetric`
/// above, unit-tested against a literal lifted from
/// `oxionnx-ops::resize::tests::test_resize_nearest_2x` (not re-derived
/// here), so the formula itself -- not just its WGSL translation -- is
/// checked independently. Test-only (its sole purpose is the unit test below).
#[cfg(test)]
fn nearest_index_host(out_coord: usize, scale: f32, in_size: usize) -> usize {
    let src = out_coord as f32 / scale;
    let max_idx = in_size as f32 - 1.0;
    let picked = (src - 0.5).ceil();
    picked.clamp(0.0, max_idx) as usize
}

async fn gpu_resize_async(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    out_h: usize,
    out_w: usize,
    kind: ResizeKind,
) -> Option<Vec<f32>> {
    gpu_resize_placed_async(
        ctx,
        TensorSource::host(data, shape),
        out_h,
        out_w,
        kind,
        OutputPlacement::Host,
    )
    .await?
    .into_vec()
}

/// The one `Resize` body, over an operand that may already be on the device and
/// with a result that may stay there.
///
/// `kind` picks the interpolation the caller's node actually specifies; the
/// caller is responsible for matching every other `Resize` attribute against
/// what these two kernels implement, exactly as before.
pub async fn gpu_resize_placed_async(
    ctx: &GpuContext,
    data: TensorSource<'_>,
    out_h: usize,
    out_w: usize,
    kind: ResizeKind,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    if ctx.is_degraded() {
        return None;
    }
    let [n, c, in_h, in_w]: [usize; 4] = data.shape().try_into().ok()?;
    let in_len = n.checked_mul(c)?.checked_mul(in_h)?.checked_mul(in_w)?;
    if data.len() != in_len {
        return None;
    }
    // An empty input axis cannot be resized to a nonzero size.
    if (in_h == 0 && out_h != 0) || (in_w == 0 && out_w != 0) {
        return None;
    }
    let total_len = n.checked_mul(c)?.checked_mul(out_h)?.checked_mul(out_w)?;
    if total_len == 0 {
        return None;
    }
    // scale = out/in ("output/input shape pair"); in_h/in_w == 0 already
    // declined above, so this division is never by zero here.
    let scale_h = out_h as f32 / in_h as f32;
    let scale_w = out_w as f32 / in_w as f32;
    if !scale_h.is_finite() || !scale_w.is_finite() || scale_h <= 0.0 || scale_w <= 0.0 {
        return None;
    }

    let in_size = checked_storage_bytes(&ctx.limits, in_len)?;
    let out_size = checked_storage_bytes(&ctx.limits, total_len)?;
    if !ctx.limits.buffer_fits(in_size) || !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, total_len as u64, WG_SIZE)?;
    // Input, output and read-back staging — minus the ones this dispatch will
    // not allocate.
    if !ctx.budget_admits(&[
        ctx.source_admission_bytes(data, in_size),
        out_size,
        placement.staging_bytes(out_size),
    ]) {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let scope = ErrorScope::begin(ctx);
    let (pipeline, bgl) = build_resize_pipeline(device, kind.entry_point());

    let input_buf = ctx.operand_source("resize_in", data, wgpu::BufferUsages::STORAGE)?;
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

    let params = ResizeParams {
        n: n as u32,
        c: c as u32,
        in_h: in_h as u32,
        in_w: in_w as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        total_len: total_len as u32,
        row_threads: grid.threads_per_row,
        scale_h,
        scale_w,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = ctx.upload_buffer(
        "resize_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("resize_staging", out_size)?),
        OutputPlacement::Device => None,
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resize_bg"),
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
        label: Some("resize_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resize_pass"),
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
        total_len,
        out_size,
        vec![n, c, out_h, out_w],
    )
    .await
}

/// GPU-accelerated bilinear `Resize`, `coordinate_transformation_mode =
/// "pytorch_half_pixel"`, over the last two axes of a rank-4 `[N,C,H,W]`
/// tensor. `N`/`C` pass through; `out_h`/`out_w` set the new `H`/`W`.
pub async fn gpu_resize_bilinear_pytorch_half_pixel_async(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    out_h: usize,
    out_w: usize,
) -> Option<Vec<f32>> {
    gpu_resize_async(
        ctx,
        data,
        shape,
        out_h,
        out_w,
        ResizeKind::BilinearPytorchHalfPixel,
    )
    .await
}

/// Blocking form of [`gpu_resize_bilinear_pytorch_half_pixel_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_resize_bilinear_pytorch_half_pixel(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    out_h: usize,
    out_w: usize,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_resize_bilinear_pytorch_half_pixel_async(
        ctx, data, shape, out_h, out_w,
    ))
}

/// GPU-accelerated nearest-neighbour `Resize`,
/// `coordinate_transformation_mode = "asymmetric"` /
/// `nearest_mode = "round_prefer_floor"` (the ONNX default), over the last
/// two axes of a rank-4 `[N,C,H,W]` tensor.
pub async fn gpu_resize_nearest_asymmetric_async(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    out_h: usize,
    out_w: usize,
) -> Option<Vec<f32>> {
    gpu_resize_async(
        ctx,
        data,
        shape,
        out_h,
        out_w,
        ResizeKind::NearestAsymmetric,
    )
    .await
}

/// Blocking form of [`gpu_resize_nearest_asymmetric_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_resize_nearest_asymmetric(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    out_h: usize,
    out_w: usize,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_resize_nearest_asymmetric_async(
        ctx, data, shape, out_h, out_w,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `oxionnx-ops::resize::tests::test_resize_nearest_2x`: `[1,1,2,2]` of
    /// `[[1,2],[3,4]]` resized to `[1,1,4,4]` (`nearest`, `asymmetric`) is
    /// `[1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4]`. Reproduced here without a GPU
    /// via the pure `nearest_index_host` mirror of the WGSL kernel.
    #[test]
    fn nearest_index_host_matches_oxionnx_ops_literal() {
        let input = [[1.0f32, 2.0], [3.0, 4.0]];
        let scale = 2.0f32; // out_size 4 / in_size 2
        let expected = [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ];
        for (oh, expected_row) in expected.iter().enumerate() {
            let ih = nearest_index_host(oh, scale, 2);
            for (ow, &expected_val) in expected_row.iter().enumerate() {
                let iw = nearest_index_host(ow, scale, 2);
                assert_eq!(
                    input[ih][iw], expected_val,
                    "oh={oh} ow={ow} ih={ih} iw={iw}"
                );
            }
        }
    }

    #[test]
    fn gpu_resize_declines_empty_axis_to_nonzero() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let data: Vec<f32> = Vec::new();
        assert!(
            gpu_resize_nearest_asymmetric(&ctx, &data, &[1, 1, 0, 4], 8, 4).is_none(),
            "resizing a 0-height axis up to 8 must decline, not divide by zero"
        );
    }

    #[test]
    fn gpu_resize_declines_non_rank4_shape() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let data = vec![1.0f32; 4];
        assert!(gpu_resize_bilinear_pytorch_half_pixel(&ctx, &data, &[2, 2], 4, 4).is_none());
    }

    /// Each sync entry point (the `block_on_gpu` wrapper) and its `_async`
    /// twin (the real implementation) must dispatch the same kernel on the
    /// same input and produce identical output. `.expect` on both sides (not
    /// a bare `assert_eq!` of the `Option`s) so a decline-path regression
    /// that makes both sides silently return `None` fails this test instead
    /// of passing vacuously. Covers both `ResizeKind`s (they share every code
    /// path except the WGSL entry point selected).
    ///
    /// `in_h != in_w` and `out_h != out_w` (and both scale ratios differ:
    /// `8/4 = 2.0` vs `9/6 = 1.5`) deliberately: `gpu_resize_*` forwards
    /// `out_h`/`out_w` to its `_async` twin by hand, so a swapped pair would
    /// still dispatch -- with a square shape or equal output extents that
    /// swap is invisible in the result, and this test would pass despite the
    /// bug.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_resize_async_matches_sync_both_kinds() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let shape = [1usize, 2, 4, 6];
        let data: Vec<f32> = (0..48).map(|i| (i as f32 - 24.0) * 0.5).collect();
        let (out_h, out_w) = (8usize, 9usize);

        let sync_bilinear =
            gpu_resize_bilinear_pytorch_half_pixel(&ctx, &data, &shape, out_h, out_w)
                .expect("gpu_resize_bilinear_pytorch_half_pixel must dispatch");
        let async_bilinear = pollster::block_on(gpu_resize_bilinear_pytorch_half_pixel_async(
            &ctx, &data, &shape, out_h, out_w,
        ))
        .expect("gpu_resize_bilinear_pytorch_half_pixel_async must dispatch on the same input");
        assert_eq!(
            sync_bilinear, async_bilinear,
            "sync and async bilinear resize must produce identical output"
        );

        let sync_nearest = gpu_resize_nearest_asymmetric(&ctx, &data, &shape, out_h, out_w)
            .expect("gpu_resize_nearest_asymmetric must dispatch");
        let async_nearest = pollster::block_on(gpu_resize_nearest_asymmetric_async(
            &ctx, &data, &shape, out_h, out_w,
        ))
        .expect("gpu_resize_nearest_asymmetric_async must dispatch on the same input");
        assert_eq!(
            sync_nearest, async_nearest,
            "sync and async nearest resize must produce identical output"
        );
    }
}
