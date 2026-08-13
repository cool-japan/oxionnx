//! GPU-accelerated `Pad` (NCHW, spatial axes only): `reflect` and `constant`
//! modes.
//!
//! Only the last two axes (`H`, `W`) of a rank-4 `[N, C, H, W]` tensor are
//! padded -- `N`/`C` always pass through unpadded, matching how InSwapper's
//! 14 reflect-pad nodes are actually used (padding the spatial extent ahead
//! of a convolution). `pad_top`/`pad_bottom`/`pad_left`/`pad_right` may be
//! negative (crop), per ONNX `Pad` since opset 11; the output shape is
//! computed from all four independently of mode.
//!
//! `mode = "constant"` is implemented alongside `"reflect"` because it is
//! nearly free once the coordinate-decode/dispatch scaffolding exists (per
//! the task's framing) -- it shares this module's shader source, bind group
//! layout and Rust wrapper, differing only in the WGSL entry point.
//!
//! The reflect formula mirrors `oxionnx-ops::shape::sequence::pad_axes`'s
//! `"reflect"` arm bit-for-bit:
//! `c = (out_coord - begin).rem_euclid(2*(dim-1)); if c >= dim { c = 2*(dim-1) - c }`
//! (with `dim <= 1` forced to `c = 0`, since there is nothing to reflect
//! across).
//!
//! See [`kernel_support`](super::kernel_support) for why this kernel's
//! pipeline is rebuilt on every call and why there is no minimum-size gate.

use crate::context::activation::{GpuOutput, OutputPlacement, TensorSource};
use crate::context::GpuContext;
use crate::device_guard::{
    block_on_gpu, checked_storage_bytes, finish_output_async, plan_dispatch, ErrorScope,
};

use super::kernel_support::{bgl_ro, bgl_rw, bgl_uniform, build_pipeline, WG_SIZE};

/// Pad mode. Only the two modes this wave's model histograms actually need
/// are implemented; `edge`/`wrap` (which `oxionnx-ops::shape::sequence::pad_axes`
/// also supports on CPU) are not -- there is no kernel to select if asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    Reflect,
    Constant,
}

impl PadMode {
    fn entry_point(self) -> &'static str {
        match self {
            PadMode::Reflect => "pad_reflect",
            PadMode::Constant => "pad_constant",
        }
    }
}

/// Uniform block for the Pad kernel. `pad_top`/`pad_left` are signed
/// (`i32`) so a negative pad (crop) is expressed the same way ONNX does.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PadParams {
    n: u32,
    c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    pad_top: i32,
    pad_left: i32,
    total_len: u32,
    row_threads: u32,
    constant_value: f32,
    _pad: u32,
}

const PAD_SHADER: &str = r#"
struct Params {
    n: u32,
    c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    pad_top: i32,
    pad_left: i32,
    total_len: u32,
    row_threads: u32,
    constant_value: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.y * params.row_threads + gid.x;
}

// Decode a flat index over [N, C, out_h, out_w] into (n, c, oh, ow).
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

// ONNX/numpy "reflect" padding along one axis: reflects without repeating
// the edge value, with period 2*(dim-1). `dim <= 1` has nothing to reflect
// across, so every output position reads the sole (or no) source element.
//
// The reduction mod `period` is done entirely in u32, on a value that is
// non-negative by construction, because `%` on a *negative* i32 dividend is
// not dependable here: on Vulkan/NVIDIA (naga 29, driver 550) `-1 % 6`
// evaluates to 3 and `-2 % 6` to 2 -- the unsigned results for the two's
// complement bit patterns (0xFFFFFFFF % 6, 0xFFFFFFFE % 6), not the
// sign-of-dividend remainder WGSL specifies. That made the obvious
// `var c = (out_coord - pad_before) % period; if (c < 0) { c = c + period; }`
// silently wrong in the leading pad region: the correction branch never ran
// because the modulo never returned a negative, so `out_coord` 1 with
// `pad_before` 2 over dim 4 read source index 3 instead of 1 -- wrap-around
// instead of reflection. `-2` landing on the correct index 2 by coincidence
// hid it further. See `reflect_coord_host` for the plain-Rust statement of
// the same formula and `k2_pad.rs`'s numpy-literal tests for the pin.
fn reflect_coord(out_coord: i32, pad_before: i32, dim: i32) -> i32 {
    if (dim <= 1) { return 0; }
    let period = u32(2 * (dim - 1));
    let raw = out_coord - pad_before;
    // `c = raw mod period`, always in [0, period), never dividing a negative.
    var c: u32;
    if (raw >= 0) {
        c = u32(raw) % period;
    } else {
        // -raw is positive (|raw| is bounded by the tensor extent, so this
        // cannot be the i32::MIN negation), so this modulo is unsigned-safe.
        let m = u32(-raw) % period;
        c = select(period - m, 0u, m == 0u);
    }
    var folded = i32(c);
    if (folded >= dim) { folded = i32(period) - folded; }
    return folded;
}

@compute @workgroup_size(256)
fn pad_reflect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    let coords = decode(idx);
    let ih = reflect_coord(i32(coords.z), params.pad_top, i32(params.in_h));
    let iw = reflect_coord(i32(coords.w), params.pad_left, i32(params.in_w));
    let in_idx = ((coords.x * params.c + coords.y) * params.in_h + u32(ih)) * params.in_w + u32(iw);
    output[idx] = input[in_idx];
}

@compute @workgroup_size(256)
fn pad_constant(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    let coords = decode(idx);
    let ih = i32(coords.z) - params.pad_top;
    let iw = i32(coords.w) - params.pad_left;
    if (ih < 0 || ih >= i32(params.in_h) || iw < 0 || iw >= i32(params.in_w)) {
        output[idx] = params.constant_value;
    } else {
        let in_idx = ((coords.x * params.c + coords.y) * params.in_h + u32(ih)) * params.in_w + u32(iw);
        output[idx] = input[in_idx];
    }
}
"#;

fn build_pad_pipeline(
    ctx: &GpuContext,
    mode: PadMode,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_pipeline(
        ctx,
        "pad",
        PAD_SHADER,
        mode.entry_point(),
        &[bgl_ro(0), bgl_rw(1), bgl_uniform(2)],
    )
}

/// Pure host-side mirror of the WGSL `reflect_coord` above -- unit-tested
/// directly (no GPU) against a literal numpy `reflect` example, so a bug
/// in the *formula* (as opposed to the WGSL translation of it) is caught
/// without a device. Test-only (its sole purpose is the unit test below).
#[cfg(test)]
fn reflect_coord_host(out_coord: i64, pad_before: i64, dim: i64) -> i64 {
    if dim <= 1 {
        return 0;
    }
    let period = 2 * (dim - 1);
    let mut c = (out_coord - pad_before).rem_euclid(period);
    if c >= dim {
        c = period - c;
    }
    c
}

/// GPU-accelerated `Pad` over the last two axes of a rank-4 `[N, C, H, W]`
/// tensor.
///
/// `pad_top`/`pad_bottom`/`pad_left`/`pad_right` may be negative (crop).
/// Declines when: the context is degraded, `shape` is not rank 4, `data`
/// does not match `shape`, the resulting `H`/`W` would be negative, or
/// (`reflect` only) a padded axis is empty (nothing to reflect from --
/// `oxionnx-ops` rejects this too).
#[allow(clippy::too_many_arguments)]
pub async fn gpu_pad_async(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    pad_top: i64,
    pad_bottom: i64,
    pad_left: i64,
    pad_right: i64,
    mode: PadMode,
    constant_value: f32,
) -> Option<Vec<f32>> {
    gpu_pad_placed_async(
        ctx,
        TensorSource::host(data, shape),
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        mode,
        constant_value,
        OutputPlacement::Host,
    )
    .await?
    .into_vec()
}

/// [`gpu_pad_async`] over an operand that may already be on the device, with a
/// result that may stay there.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_pad_placed_async(
    ctx: &GpuContext,
    data: TensorSource<'_>,
    pad_top: i64,
    pad_bottom: i64,
    pad_left: i64,
    pad_right: i64,
    mode: PadMode,
    constant_value: f32,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    if ctx.is_degraded() {
        return None;
    }
    let [n, c, h, w]: [usize; 4] = data.shape().try_into().ok()?;
    let in_len = n.checked_mul(c)?.checked_mul(h)?.checked_mul(w)?;
    if data.len() != in_len {
        return None;
    }

    let out_h = (h as i64).checked_add(pad_top)?.checked_add(pad_bottom)?;
    let out_w = (w as i64).checked_add(pad_left)?.checked_add(pad_right)?;
    if out_h < 0 || out_w < 0 {
        return None;
    }
    let (out_h, out_w) = (out_h as usize, out_w as usize);

    if mode == PadMode::Reflect && out_h > 0 && out_w > 0 && (h == 0 || w == 0) {
        // Nothing to reflect from -- mirrors `pad_axes`'s own rejection of
        // reflect-padding a 0-element axis.
        return None;
    }

    // WGSL coordinate math is i32; every dimension and pad value used there
    // must fit.
    let pad_top_i32 = i32::try_from(pad_top).ok()?;
    let pad_left_i32 = i32::try_from(pad_left).ok()?;
    i32::try_from(h).ok()?;
    i32::try_from(w).ok()?;
    i32::try_from(out_h).ok()?;
    i32::try_from(out_w).ok()?;

    let total_len = n.checked_mul(c)?.checked_mul(out_h)?.checked_mul(out_w)?;
    if total_len == 0 {
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
    let (pipeline, bgl) = build_pad_pipeline(ctx, mode);

    let input_buf = ctx.operand_source("pad_in", data, wgpu::BufferUsages::STORAGE)?;
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

    let params = PadParams {
        n: n as u32,
        c: c as u32,
        in_h: h as u32,
        in_w: w as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        pad_top: pad_top_i32,
        pad_left: pad_left_i32,
        total_len: total_len as u32,
        row_threads: grid.threads_per_row,
        constant_value,
        _pad: 0,
    };
    let params_buf = ctx.upload_buffer(
        "pad_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("pad_staging", out_size)?),
        OutputPlacement::Device => None,
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pad_bg"),
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
        label: Some("pad_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pad_pass"),
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

/// Blocking form of [`gpu_pad_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_pad(
    ctx: &GpuContext,
    data: &[f32],
    shape: &[usize],
    pad_top: i64,
    pad_bottom: i64,
    pad_left: i64,
    pad_right: i64,
    mode: PadMode,
    constant_value: f32,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_pad_async(
        ctx,
        data,
        shape,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        mode,
        constant_value,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy: `np.pad([1,2,3,4], 2, mode='reflect')` == `[3,2,1,2,3,4,3,2]`.
    /// Pinned against a literal this crate did not derive, so the *formula*
    /// (not just its WGSL translation) is checked independently.
    #[test]
    fn reflect_coord_host_matches_numpy_literal() {
        let dim = 4i64;
        let pad_before = 2i64;
        let expected_src = [2, 1, 0, 1, 2, 3, 2, 1]; // indices into [1,2,3,4]
        let want = [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0];
        let data = [1.0f32, 2.0, 3.0, 4.0];
        for (out_coord, (&exp_idx, &exp_val)) in expected_src.iter().zip(want.iter()).enumerate() {
            let idx = reflect_coord_host(out_coord as i64, pad_before, dim);
            assert_eq!(idx, exp_idx, "out_coord={out_coord}");
            assert_eq!(data[idx as usize], exp_val, "out_coord={out_coord}");
        }
    }

    #[test]
    fn reflect_coord_host_degenerate_dim_is_always_zero() {
        assert_eq!(reflect_coord_host(5, 2, 1), 0);
        assert_eq!(reflect_coord_host(0, 0, 0), 0);
    }

    #[test]
    fn gpu_pad_declines_non_rank4_shape() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let data = vec![1.0f32; 4];
        assert!(gpu_pad(&ctx, &data, &[2, 2], 1, 1, 1, 1, PadMode::Reflect, 0.0).is_none());
    }

    #[test]
    fn gpu_pad_declines_reflect_from_empty_axis() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let data: Vec<f32> = Vec::new();
        assert!(gpu_pad(
            &ctx,
            &data,
            &[1, 1, 0, 4],
            1,
            1,
            0,
            0,
            PadMode::Reflect,
            0.0
        )
        .is_none());
    }

    #[test]
    fn gpu_pad_declines_negative_output_extent() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let data = vec![1.0f32; 16];
        // Cropping 5 off each side of a width-4 axis goes negative.
        assert!(gpu_pad(
            &ctx,
            &data,
            &[1, 1, 4, 4],
            0,
            0,
            -5,
            -5,
            PadMode::Constant,
            0.0
        )
        .is_none());
    }

    /// `gpu_pad` (the `block_on_gpu` wrapper) and `gpu_pad_async` (the real
    /// implementation) must dispatch the same kernel on the same input and
    /// produce identical output. `.expect` on both sides (not a bare
    /// `assert_eq!` of the `Option`s) so a decline-path regression that makes
    /// both sides silently return `None` fails this test instead of passing
    /// vacuously. Covers both `PadMode` entry points in one test, since they
    /// share every code path except the WGSL entry point selected.
    ///
    /// The four pad amounts (`1, 2, 3, 0`) are deliberately all different, and
    /// the input is not square: `gpu_pad` forwards nine positional arguments
    /// to `gpu_pad_async` by hand, so a transposed pair (e.g. `pad_top` and
    /// `pad_left` swapped) would still produce *a* result -- with symmetric
    /// pads or a square input that result is bit-identical to the correct
    /// one, and this test would pass despite the bug.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_pad_async_matches_sync_both_modes() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let data: Vec<f32> = (0..48).map(|i| (i as f32 - 24.0) * 0.25).collect();
        let shape = [1usize, 1, 6, 8];

        for mode in [PadMode::Reflect, PadMode::Constant] {
            let sync_result = gpu_pad(&ctx, &data, &shape, 1, 2, 3, 0, mode, -1.0)
                .expect("gpu_pad must dispatch");
            let async_result =
                pollster::block_on(gpu_pad_async(&ctx, &data, &shape, 1, 2, 3, 0, mode, -1.0))
                    .expect("gpu_pad_async must dispatch on the same input");
            assert_eq!(
                sync_result, async_result,
                "sync and async entry points must produce identical output ({mode:?})"
            );
        }
    }
}
