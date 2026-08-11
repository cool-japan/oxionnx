//! GPU-accelerated binary element-wise ops with general NumPy-style
//! broadcasting, up to rank 4 (`[N, C, H, W]`).
//!
//! [`elementwise.rs`](super::elementwise)'s `gpu_add`/`gpu_mul` require the
//! two operands to have *exactly* equal length — every shape mismatch
//! declines, including the extremely common `[1,C,H,W] op [1,C,1,1]`
//! per-channel broadcast (InSwapper's AdaIN residual path: 49 of its 67
//! Add/Mul nodes broadcast this way). The four entry points here cover that
//! case, its `[1,C,1,1] op [1,C,H,W]` mirror, and scalar operands (rank-0 or
//! a single element), by walking the output's flat index against a
//! per-operand stride vector computed on the host — stride `0` on a
//! broadcast axis reads the same source element for every step along it,
//! exactly mirroring `oxionnx-ops::math::broadcast::broadcast_strides`
//! (`elementwise_binary`'s general path).
//!
//! See [`kernel_support`](super::kernel_support) for why this kernel's
//! pipeline is rebuilt on every call and why there is no minimum-size gate.

use crate::context::activation::{GpuOutput, OutputPlacement, TensorSource};
use crate::context::GpuContext;
use crate::device_guard::{
    block_on_gpu, checked_storage_bytes, finish_output_async, plan_dispatch, ErrorScope,
};

use super::kernel_support::{bgl_ro, bgl_rw, bgl_uniform, build_pipeline, WG_SIZE};

/// Binary op selected at dispatch time (all four entry points share one
/// shader module and bind group layout; only the entry point differs).
///
/// `pub(crate)` (not private) purely so [`build_broadcast_pipeline`] can
/// stay `pub(crate)` too, matching its sibling `build_*_pipeline` functions
/// in this kernel batch (see `kernel_support`'s module docs on why that
/// matters for a future cached-pipeline hoist).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BroadcastOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BroadcastOp {
    fn entry_point(self) -> &'static str {
        match self {
            BroadcastOp::Add => "bcast_add",
            BroadcastOp::Sub => "bcast_sub",
            BroadcastOp::Mul => "bcast_mul",
            BroadcastOp::Div => "bcast_div",
        }
    }
}

/// Uniform block for the broadcast-binary kernel.
///
/// `out_strides`/`a_strides`/`b_strides` are row-major strides over the
/// common (rank-4, left-padded) output shape; a `0` entry marks a broadcast
/// axis for that operand. `out_strides` alone is enough to decode a flat
/// output index back into per-axis coordinates (successive divide/modulo,
/// the same technique `TRANSPOSE_SHADER` uses), so the shape itself never
/// needs to be uploaded.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BroadcastParams {
    out_strides: [u32; 4],
    a_strides: [u32; 4],
    b_strides: [u32; 4],
    total_len: u32,
    row_threads: u32,
    _pad0: u32,
    _pad1: u32,
}

const BROADCAST_SHADER: &str = r#"
struct Params {
    out_strides: vec4<u32>,
    a_strides: vec4<u32>,
    b_strides: vec4<u32>,
    total_len: u32,
    row_threads: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.y * params.row_threads + gid.x;
}

// Decode a flat output index into (a_offset, b_offset) via the output's own
// strides (component d of the result is `idx`'s coordinate along axis d,
// found by successive divide/modulo -- no explicit shape needed), then
// re-project that coordinate through each operand's own stride vector
// (0 on a broadcast axis, so the coordinate contributes nothing there).
fn operand_offsets(idx: u32) -> vec2<u32> {
    var rem = idx;
    var a_off: u32 = 0u;
    var b_off: u32 = 0u;
    for (var d: u32 = 0u; d < 4u; d = d + 1u) {
        let s = params.out_strides[d];
        let coord = rem / s;
        rem = rem % s;
        a_off = a_off + coord * params.a_strides[d];
        b_off = b_off + coord * params.b_strides[d];
    }
    return vec2<u32>(a_off, b_off);
}

@compute @workgroup_size(256)
fn bcast_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    let off = operand_offsets(idx);
    output[idx] = a[off.x] + b[off.y];
}

@compute @workgroup_size(256)
fn bcast_sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    let off = operand_offsets(idx);
    output[idx] = a[off.x] - b[off.y];
}

@compute @workgroup_size(256)
fn bcast_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    let off = operand_offsets(idx);
    output[idx] = a[off.x] * b[off.y];
}

@compute @workgroup_size(256)
fn bcast_div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.total_len) { return; }
    let off = operand_offsets(idx);
    output[idx] = a[off.x] / b[off.y];
}
"#;

pub(crate) fn build_broadcast_pipeline(
    device: &wgpu::Device,
    op: BroadcastOp,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_pipeline(
        device,
        "broadcast_binary",
        BROADCAST_SHADER,
        op.entry_point(),
        &[bgl_ro(0), bgl_ro(1), bgl_rw(2), bgl_uniform(3)],
    )
}

// ========================================================================
// Pure shape/stride math (unit-tested without a GPU below)
// ========================================================================

/// Left-pad `shape` with leading 1s to rank 4 (NumPy broadcast convention).
/// `None` when `shape` already has more than 4 dimensions -- this kernel's
/// stated scope is "up to 4D".
fn pad_to_rank4(shape: &[usize]) -> Option<[usize; 4]> {
    if shape.len() > 4 {
        return None;
    }
    let mut out = [1usize; 4];
    let offset = 4 - shape.len();
    out[offset..].copy_from_slice(shape);
    Some(out)
}

/// Per-axis NumPy broadcast: equal, or exactly one side is `1`. `None` on an
/// incompatible pair (e.g. `[m,1]` against `[n]` with `m != n`).
fn broadcast_out_shape4(a4: [usize; 4], b4: [usize; 4]) -> Option<[usize; 4]> {
    let mut out = [0usize; 4];
    for i in 0..4 {
        out[i] = match (a4[i], b4[i]) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

/// Row-major strides of `shape4` itself, `0` on any axis where `shape4`
/// broadcasts against `out4` (`shape4[i] == 1 && out4[i] != 1`). Mirrors
/// `oxionnx-ops::math::broadcast::broadcast_strides` exactly: the running
/// stride accumulates the *operand's own* dimension at every step
/// (including broadcast axes, where it is always `1` and so leaves the
/// running product unchanged), matching a real row-major walk of the
/// operand's own (padded) data.
fn broadcast_strides4(shape4: [usize; 4], out4: [usize; 4]) -> Option<[u32; 4]> {
    let mut strides = [0u32; 4];
    let mut stride: u64 = 1;
    for i in (0..4).rev() {
        if shape4[i] == 1 && out4[i] != 1 {
            strides[i] = 0;
        } else {
            strides[i] = u32::try_from(stride).ok()?;
        }
        stride = stride.checked_mul(shape4[i] as u64)?;
    }
    Some(strides)
}

/// Plain row-major strides of `shape4` (no broadcasting), plus the total
/// element count (the final accumulated stride). Used for the output shape,
/// which is never itself broadcast.
fn row_major_strides4(shape4: [usize; 4]) -> Option<([u32; 4], u64)> {
    let mut strides = [0u32; 4];
    let mut stride: u64 = 1;
    for i in (0..4).rev() {
        strides[i] = u32::try_from(stride).ok()?;
        stride = stride.checked_mul(shape4[i] as u64)?;
    }
    Some((strides, stride))
}

/// `(out_strides, a_strides, b_strides, total_output_len)`.
type ResolvedBroadcast = ([u32; 4], [u32; 4], [u32; 4], u64);

/// Resolve two operand shapes into (output strides, a strides, b strides,
/// total output length), or `None` on any of: rank > 4, incompatible
/// shapes, a length that overflows the `u32` flat index every kernel in
/// this crate uses, or a data length that does not match its claimed shape.
fn resolve_broadcast(
    a_len: usize,
    a_shape: &[usize],
    b_len: usize,
    b_shape: &[usize],
) -> Option<ResolvedBroadcast> {
    let a4 = pad_to_rank4(a_shape)?;
    let b4 = pad_to_rank4(b_shape)?;
    let out4 = broadcast_out_shape4(a4, b4)?;

    let a4_numel: u64 = a4
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))?;
    let b4_numel: u64 = b4
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))?;
    if a_len as u64 != a4_numel || b_len as u64 != b4_numel {
        return None;
    }

    let (out_strides, total_len) = row_major_strides4(out4)?;
    let a_strides = broadcast_strides4(a4, out4)?;
    let b_strides = broadcast_strides4(b4, out4)?;
    if total_len == 0 {
        return None;
    }
    Some((out_strides, a_strides, b_strides, total_len))
}

// ========================================================================
// Dispatch
// ========================================================================

/// The one broadcasting-binary body, in both regimes.
///
/// `out_shape` is the caller's ONNX-level result shape. `None` means the caller
/// asked for [`OutputPlacement::Host`] and will discard whatever shape comes
/// back (every pre-residency entry point does — they return `Vec<f32>`), so a
/// flat one is used. `Some` is validated against the element count the stride
/// resolution derived, because a device tensor carries its shape onward and a
/// disagreement there would mis-shape everything downstream.
async fn gpu_broadcast_binary_placed(
    ctx: &GpuContext,
    a: TensorSource<'_>,
    b: TensorSource<'_>,
    out_shape: Option<&[usize]>,
    op: BroadcastOp,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    if ctx.is_degraded() {
        return None;
    }
    let (out_strides, a_strides, b_strides, total_len) =
        resolve_broadcast(a.len(), a.shape(), b.len(), b.shape())?;
    let result_shape = match out_shape {
        Some(shape) => {
            let elems = shape
                .iter()
                .try_fold(1u64, |acc, &dim| acc.checked_mul(dim as u64))?;
            if elems != total_len {
                return None;
            }
            shape.to_vec()
        }
        None => vec![total_len as usize],
    };

    let a_bytes = checked_storage_bytes(&ctx.limits, a.len())?;
    let b_bytes = checked_storage_bytes(&ctx.limits, b.len())?;
    let out_size = checked_storage_bytes(&ctx.limits, total_len as usize)?;
    if !ctx.limits.buffer_fits(a_bytes)
        || !ctx.limits.buffer_fits(b_bytes)
        || !ctx.limits.buffer_fits(out_size)
    {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, total_len, WG_SIZE)?;
    // a, b, output and read-back staging — minus the ones this dispatch will
    // not allocate.
    if !ctx.budget_admits(&[
        ctx.source_admission_bytes(a, a_bytes),
        ctx.source_admission_bytes(b, b_bytes),
        out_size,
        placement.staging_bytes(out_size),
    ]) {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    // Pipeline construction must sit *inside* the error scope: unlike
    // `elementwise.rs`, the pipeline here is not pre-built, so a WGSL
    // problem would otherwise only surface via the context-wide degraded
    // flag instead of this dispatch's own decline.
    let scope = ErrorScope::begin(ctx);
    let (pipeline, bgl) = build_broadcast_pipeline(device, op);

    let a_buf = ctx.operand_source("bcast_a", a, wgpu::BufferUsages::STORAGE)?;
    let b_buf = ctx.operand_source("bcast_b", b, wgpu::BufferUsages::STORAGE)?;
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

    let params = BroadcastParams {
        out_strides,
        a_strides,
        b_strides,
        total_len: total_len as u32,
        row_threads: grid.threads_per_row,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = ctx.upload_buffer(
        "bcast_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("bcast_staging", out_size)?),
        OutputPlacement::Device => None,
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bcast_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.binding(),
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
        label: Some("bcast_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bcast_pass"),
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
        total_len as usize,
        out_size,
        result_shape,
    )
    .await
}

/// The pre-residency form: host operands in, host values out.
async fn gpu_broadcast_binary_async(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    op: BroadcastOp,
) -> Option<Vec<f32>> {
    gpu_broadcast_binary_placed(
        ctx,
        TensorSource::host(a, a_shape),
        TensorSource::host(b, b_shape),
        None,
        op,
        OutputPlacement::Host,
    )
    .await?
    .into_vec()
}

/// Which broadcasting binary op a placed dispatch computes.
///
/// The public spelling of the private `BroadcastOp`, so a caller outside this
/// crate can pick an op without the four near-identical entry points the
/// pre-residency surface has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BroadcastKind {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b`.
    Div,
}

impl BroadcastKind {
    fn op(self) -> BroadcastOp {
        match self {
            Self::Add => BroadcastOp::Add,
            Self::Sub => BroadcastOp::Sub,
            Self::Mul => BroadcastOp::Mul,
            Self::Div => BroadcastOp::Div,
        }
    }
}

/// Broadcasting binary op over operands that may already be on the device.
///
/// `out_shape` is the ONNX broadcast result shape the caller resolved; it is
/// checked against the element count this kernel's own stride resolution
/// derived, so the two cannot disagree.
pub async fn gpu_broadcast_placed_async(
    ctx: &GpuContext,
    a: TensorSource<'_>,
    b: TensorSource<'_>,
    out_shape: &[usize],
    kind: BroadcastKind,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    gpu_broadcast_binary_placed(ctx, a, b, Some(out_shape), kind.op(), placement).await
}

/// GPU-accelerated broadcasting addition: `a + b`, NumPy rules, up to rank 4.
pub async fn gpu_broadcast_add_async(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_broadcast_binary_async(ctx, a, a_shape, b, b_shape, BroadcastOp::Add).await
}

/// Blocking form of [`gpu_broadcast_add_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_broadcast_add(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_broadcast_add_async(ctx, a, a_shape, b, b_shape))
}

/// GPU-accelerated broadcasting subtraction: `a - b`, NumPy rules, up to rank 4.
pub async fn gpu_broadcast_sub_async(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_broadcast_binary_async(ctx, a, a_shape, b, b_shape, BroadcastOp::Sub).await
}

/// Blocking form of [`gpu_broadcast_sub_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_broadcast_sub(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_broadcast_sub_async(ctx, a, a_shape, b, b_shape))
}

/// GPU-accelerated broadcasting multiplication: `a * b`, NumPy rules, up to rank 4.
pub async fn gpu_broadcast_mul_async(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_broadcast_binary_async(ctx, a, a_shape, b, b_shape, BroadcastOp::Mul).await
}

/// Blocking form of [`gpu_broadcast_mul_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_broadcast_mul(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_broadcast_mul_async(ctx, a, a_shape, b, b_shape))
}

/// GPU-accelerated broadcasting division: `a / b`, NumPy rules, up to rank 4.
pub async fn gpu_broadcast_div_async(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<Vec<f32>> {
    gpu_broadcast_binary_async(ctx, a, a_shape, b, b_shape, BroadcastOp::Div).await
}

/// Blocking form of [`gpu_broadcast_div_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_broadcast_div(
    ctx: &GpuContext,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_broadcast_div_async(ctx, a, a_shape, b, b_shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_to_rank4_left_pads_with_ones() {
        assert_eq!(pad_to_rank4(&[]), Some([1, 1, 1, 1]));
        assert_eq!(pad_to_rank4(&[5]), Some([1, 1, 1, 5]));
        assert_eq!(pad_to_rank4(&[2, 3]), Some([1, 1, 2, 3]));
        assert_eq!(pad_to_rank4(&[1, 2, 3, 4]), Some([1, 2, 3, 4]));
        assert_eq!(pad_to_rank4(&[1, 2, 3, 4, 5]), None);
    }

    #[test]
    fn broadcast_out_shape4_matches_numpy_rules() {
        // [1,C,H,W] op [1,C,1,1]
        assert_eq!(
            broadcast_out_shape4([1, 8, 4, 4], [1, 8, 1, 1]),
            Some([1, 8, 4, 4])
        );
        // scalar op tensor
        assert_eq!(
            broadcast_out_shape4([1, 1, 1, 1], [1, 8, 4, 4]),
            Some([1, 8, 4, 4])
        );
        // incompatible
        assert_eq!(broadcast_out_shape4([1, 3, 1, 1], [1, 5, 1, 1]), None);
    }

    #[test]
    fn broadcast_strides4_zeroes_broadcast_axes() {
        let a4 = [1usize, 8, 1, 1];
        // N=2 here (not 1) so axis 0 is a *genuine* broadcast axis --
        // distinct from every real InSwapper shape below, where N=1 on both
        // operands makes axis 0 degenerate (its output coordinate is always
        // 0 regardless of the stride value there, so the two cases would be
        // indistinguishable by this assertion; see `resolve_broadcast_channel_case`).
        let out4 = [2usize, 8, 4, 4];
        let strides = broadcast_strides4(a4, out4).expect("no overflow");
        // N (broadcast, a4[0]=1 < out4[0]=2), channel real, H/W broadcast.
        assert_eq!(strides, [0, 1, 0, 0]);
    }

    #[test]
    fn row_major_strides4_matches_hand_computed_example() {
        let (strides, total) = row_major_strides4([1, 2, 3, 4]).expect("no overflow");
        assert_eq!(strides, [24, 12, 4, 1]);
        assert_eq!(total, 24);
    }

    #[test]
    fn resolve_broadcast_channel_case() {
        // [1,2,3,4] op [1,2,1,1] -- the real InSwapper `[1,C,H,W] op [1,C,1,1]`
        // pattern, N=1 on both sides.
        let (out_strides, a_strides, b_strides, total) =
            resolve_broadcast(24, &[1, 2, 3, 4], 2, &[1, 2, 1, 1]).expect("compatible shapes");
        assert_eq!(total, 24);
        assert_eq!(out_strides, [24, 12, 4, 1]);
        assert_eq!(a_strides, [24, 12, 4, 1]);
        // b_strides[0] is 2, not 0: with out4[0] == 1 too (N=1 on both
        // operands), axis 0 is not a genuine broadcast axis by the
        // `shape4[i]==1 && out4[i]!=1` test (mirrored exactly from
        // `oxionnx-ops::math::broadcast::broadcast_strides`), so it falls to
        // the running-stride `else` branch instead of being forced to 0.
        // This is provably inert rather than a bug: with out_strides[0] ==
        // total_len (since out4[0]==1), every flat index < total_len divides
        // to coordinate 0 on that axis, so `coord * b_strides[0]` is always
        // `0 * 2 == 0` regardless of what value b_strides[0] holds. See
        // `broadcast_strides4_zeroes_broadcast_axes` for a shape where axis 0
        // genuinely broadcasts (out4[0] > 1) and is actually forced to 0.
        assert_eq!(b_strides, [2, 1, 0, 0]);
    }

    #[test]
    fn resolve_broadcast_declines_shape_data_mismatch() {
        // Claims shape [1,2,1,1] (2 elements) but hands over 3 -- must decline,
        // not silently read out of bounds or panic.
        assert!(resolve_broadcast(24, &[1, 2, 3, 4], 3, &[1, 2, 1, 1]).is_none());
    }

    #[test]
    fn resolve_broadcast_declines_rank_above_4() {
        assert!(resolve_broadcast(1, &[1, 1, 1, 1, 1], 1, &[1]).is_none());
    }

    #[test]
    fn resolve_broadcast_declines_incompatible_shapes() {
        assert!(resolve_broadcast(3, &[3], 5, &[5]).is_none());
    }

    /// Each sync entry point (the `block_on_gpu` wrapper) and its `_async`
    /// twin (the real implementation) must dispatch the same kernel on the
    /// same input and produce identical output. `.expect` on both sides (not
    /// a bare `assert_eq!` of the `Option`s) so a decline-path regression
    /// that makes both sides silently return `None` fails this test instead
    /// of passing vacuously. Covers all four ops at a real channel-broadcast
    /// shape (`[1,C,H,W] op [1,C,1,1]`, InSwapper's AdaIN residual pattern).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_broadcast_async_matches_sync_all_ops() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let a_shape = [1usize, 4, 3, 3];
        let b_shape = [1usize, 4, 1, 1];
        let a: Vec<f32> = (0..36).map(|i| (i as f32 - 18.0) * 0.25).collect();
        // Bounded away from 0 so it is also safe as a Div denominator.
        let b: Vec<f32> = (0..4).map(|i| (i as f32) * 0.5 + 1.0).collect();

        let sync_add = gpu_broadcast_add(&ctx, &a, &a_shape, &b, &b_shape)
            .expect("gpu_broadcast_add must dispatch");
        let async_add =
            pollster::block_on(gpu_broadcast_add_async(&ctx, &a, &a_shape, &b, &b_shape))
                .expect("gpu_broadcast_add_async must dispatch on the same input");
        assert_eq!(sync_add, async_add, "add: sync and async must agree");

        let sync_sub = gpu_broadcast_sub(&ctx, &a, &a_shape, &b, &b_shape)
            .expect("gpu_broadcast_sub must dispatch");
        let async_sub =
            pollster::block_on(gpu_broadcast_sub_async(&ctx, &a, &a_shape, &b, &b_shape))
                .expect("gpu_broadcast_sub_async must dispatch on the same input");
        assert_eq!(sync_sub, async_sub, "sub: sync and async must agree");

        let sync_mul = gpu_broadcast_mul(&ctx, &a, &a_shape, &b, &b_shape)
            .expect("gpu_broadcast_mul must dispatch");
        let async_mul =
            pollster::block_on(gpu_broadcast_mul_async(&ctx, &a, &a_shape, &b, &b_shape))
                .expect("gpu_broadcast_mul_async must dispatch on the same input");
        assert_eq!(sync_mul, async_mul, "mul: sync and async must agree");

        let sync_div = gpu_broadcast_div(&ctx, &a, &a_shape, &b, &b_shape)
            .expect("gpu_broadcast_div must dispatch");
        let async_div =
            pollster::block_on(gpu_broadcast_div_async(&ctx, &a, &a_shape, &b, &b_shape))
                .expect("gpu_broadcast_div_async must dispatch on the same input");
        assert_eq!(sync_div, async_div, "div: sync and async must agree");
    }
}
