use crate::context::activation::{GpuOutput, OutputPlacement, TensorSource};
use crate::context::GpuContext;
use crate::device_guard::{
    block_on_gpu, checked_storage_bytes, dispatch_2d_fits, read_back_async, ErrorScope,
};
use oxionnx_core::Tensor;
use wgpu;

// The minimum problem size a GEMM must reach before this crate will dispatch
// it — once a flat `GPU_THRESHOLD: u64 = 10_000_000` here — now lives on the
// context as `GpuTuning::gemm_min_mac`, alongside the *shape* rule that a FLOP
// count alone cannot express. See `crate::context::tuning` for the measured
// table behind both, and why a skinny `[1, 25088] × [25088, 512]` — 12.8 M
// multiply-accumulates, comfortably past any flat threshold — is 1.54x slower
// on this GPU than the CPU kernel it displaces.
//
// `GpuTuning::gemm_mac` is the widened (`u64`, not `usize`) product this file
// used to compute inline; the widening is load-bearing on wasm32 and is
// documented there.
use crate::context::tuning::GemmWeightTraffic;

/// Minimum dimension size for tiled matmul (shared-memory tiles are 16x16).
const TILED_MIN_DIM: usize = 32;

/// Soft byte budget for the buffers of one batched Conv2D submission.
///
/// [a7-21] `gpu_conv2d` groups its `(batch, group)` iterations into chunks that
/// fit this budget and submits each chunk as a single command buffer with a
/// single read-back. Batching trades fence latency for a larger *live* working
/// set: every iteration in a chunk holds its own im2col upload and result
/// buffer at the same time, where the old one-at-a-time loop reused the same
/// few megabytes. Measured on Metal with the audit's 32-iteration conv, an
/// unbounded chunk was slower than the old loop under memory pressure while a
/// bounded one was consistently faster, so both bounds below are real.
const CONV_CHUNK_BYTE_BUDGET: u64 = 32 << 20;

/// Hard cap on iterations per submission, independent of their size.
///
/// Fence latency is amortized almost completely after a handful of dispatches,
/// so batching past this point buys little and costs residency.
const CONV_MAX_CHUNK_ITERS: usize = 16;

/// Uniform buffer params — must match the WGSL `Params` struct layout.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32, // align to 16 bytes
}

// ========================================================================
// Helper: GEMM shape / resource validation
// ========================================================================

/// Byte sizes of the three GEMM operands, or `None` when any of them overflows
/// or exceeds what this device can allocate and bind.
///
/// `a` is [M, K], `b` is [K, N], `c` is [M, N]. Declining here is what keeps an
/// `lm_head`-sized projection (`b = [4096, 32000]`, 524 MB) from turning into a
/// wgpu validation error — which, by default, is a process-wide panic.
fn gemm_buffer_sizes(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<u64> {
    if m == 0 || k == 0 || n == 0 {
        return None;
    }
    let a_len = m.checked_mul(k)?;
    let b_len = k.checked_mul(n)?;
    let c_len = m.checked_mul(n)?;
    if a.len() < a_len || b.len() < b_len {
        return None;
    }
    let a_size = checked_storage_bytes(&ctx.limits, a_len)?;
    let b_size = checked_storage_bytes(&ctx.limits, b_len)?;
    let c_size = checked_storage_bytes(&ctx.limits, c_len)?;
    // The staging copy is not bound, but still has to be allocatable.
    if !ctx.limits.buffer_fits(c_size) {
        return None;
    }
    // A, B, C and the read-back staging copy — the four buffers both matmul
    // kernels allocate.
    if !ctx.budget_admits(&[a_size, b_size, c_size, c_size]) {
        return None;
    }
    Some(c_size)
}

/// Run matrix multiplication on GPU: C = A * B
/// A: [M, K], B: [K, N] -> C: [M, N]
///
/// Automatically selects tiled (shared memory) kernel for large matrices
/// and falls back to basic kernel for smaller ones.
///
/// Returns `None` if the problem is too small **or the wrong shape** for the
/// GPU (caller should use CPU).
///
/// # Both operands upload on every call
///
/// This entry point takes two host slices and has nowhere to cache either, so
/// its gate is [`GemmWeightTraffic::PerDispatch`] — the strict one, including
/// the arithmetic-intensity rule that declines skinny problems whatever their
/// total size. That is not a limitation of the gate but a description of this
/// function: a `[1, 25088] × [25088, 512]` call moves a 51.4 MB `B` across the
/// bus to perform 25.7 MFLOP, and measured 1.54x slower than
/// `oxionnx_ops::math::matmul` on an RTX A4000 for exactly that reason.
///
/// The same shape through [`crate::gpu_gemm_nt_resident_async`], whose `B` has
/// a cache identity, measured **0.43x** — the fix for a skinny GEMM is
/// residency, not a lower threshold. See [`crate::context::tuning`].
pub async fn gpu_matmul_async(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    if !ctx
        .tuning()
        .gemm_admits(m, k, n, GemmWeightTraffic::PerDispatch)
    {
        return None;
    }

    // Use tiled kernel for large dimensions, basic for small.
    if m >= TILED_MIN_DIM && n >= TILED_MIN_DIM && k >= TILED_MIN_DIM {
        gpu_matmul_tiled_inner(ctx, a, b, m, k, n).await
    } else {
        gpu_matmul_basic(ctx, a, b, m, k, n).await
    }
}

/// Blocking form of [`gpu_matmul_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_matmul(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_matmul_async(ctx, a, b, m, k, n))
}

/// Tiled matrix multiply using shared memory for improved cache locality.
/// Uses TILE_SIZE x TILE_SIZE tiles loaded into workgroup shared memory.
/// Falls back to the basic kernel for small matrices.
///
/// Returns `None` if the problem is too small for GPU (caller should use CPU).
pub async fn gpu_matmul_tiled_async(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    // Same gate as `gpu_matmul_async`, for the same reason: both operands are
    // host slices, so both upload on every call.
    if !ctx
        .tuning()
        .gemm_admits(m, k, n, GemmWeightTraffic::PerDispatch)
    {
        return None;
    }

    // Use tiled kernel for large dimensions, basic for small
    if m >= TILED_MIN_DIM && n >= TILED_MIN_DIM && k >= TILED_MIN_DIM {
        gpu_matmul_tiled_inner(ctx, a, b, m, k, n).await
    } else {
        gpu_matmul_basic(ctx, a, b, m, k, n).await
    }
}

/// Blocking form of [`gpu_matmul_tiled_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
pub fn gpu_matmul_tiled(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_matmul_tiled_async(ctx, a, b, m, k, n))
}

/// Inner implementation of tiled matmul using 16x16 shared-memory tiles.
async fn gpu_matmul_tiled_inner(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    if ctx.is_degraded() {
        return None;
    }
    let c_size = gemm_buffer_sizes(ctx, a, b, m, k, n)?;
    // Tiled kernel uses workgroup_size(16, 16): dispatch enough workgroups so
    // global_invocation covers all (col, row) pairs.
    let wg_x = u32::try_from(n).ok()?.div_ceil(16);
    let wg_y = u32::try_from(m).ok()?.div_ceil(16);
    if !dispatch_2d_fits(&ctx.limits, wg_x, wg_y) {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let scope = ErrorScope::begin(ctx);

    let a_buf = ctx.upload_buffer(
        "tiled_A",
        bytemuck::cast_slice(&a[..m * k]),
        wgpu::BufferUsages::STORAGE,
    )?;

    let b_buf = ctx.upload_buffer(
        "tiled_B",
        bytemuck::cast_slice(&b[..k * n]),
        wgpu::BufferUsages::STORAGE,
    )?;

    let c_buf = ctx.alloc_buffer(
        "tiled_C",
        c_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;

    let params = GemmParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        _pad: 0,
    };
    let params_buf = ctx.upload_buffer(
        "tiled_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = ctx.staging_buffer("tiled_staging", c_size)?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tiled_matmul_bg"),
        layout: &ctx.tiled_matmul_bind_group_layout,
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
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tiled_matmul_enc"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tiled_matmul_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.tiled_matmul_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging_buf, 0, c_size);
    queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish_async(ctx).await {
        return None;
    }

    read_back_async(ctx, &staging_buf, m * n).await
}

/// Basic (non-tiled) GPU matmul — used as fallback for matrices with small dimensions.
async fn gpu_matmul_basic(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    if ctx.is_degraded() {
        return None;
    }
    let c_size = gemm_buffer_sizes(ctx, a, b, m, k, n)?;
    // Basic kernel uses workgroup_size(8, 8) with `gid.x` = row, `gid.y` = col.
    let wg_x = u32::try_from(m).ok()?.div_ceil(8);
    let wg_y = u32::try_from(n).ok()?.div_ceil(8);
    if !dispatch_2d_fits(&ctx.limits, wg_x, wg_y) {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let scope = ErrorScope::begin(ctx);

    let a_buf = ctx.upload_buffer(
        "A",
        bytemuck::cast_slice(&a[..m * k]),
        wgpu::BufferUsages::STORAGE,
    )?;

    let b_buf = ctx.upload_buffer(
        "B",
        bytemuck::cast_slice(&b[..k * n]),
        wgpu::BufferUsages::STORAGE,
    )?;

    let c_buf = ctx.alloc_buffer(
        "C",
        c_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;

    let params = GemmParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        _pad: 0,
    };
    let params_buf = ctx.upload_buffer(
        "params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = ctx.staging_buffer("staging", c_size)?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bg"),
        layout: &ctx.matmul_bind_group_layout,
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
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul_enc"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.matmul_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging_buf, 0, c_size);
    queue.submit(std::iter::once(encoder.finish()));

    if !scope.finish_async(ctx).await {
        return None;
    }

    read_back_async(ctx, &staging_buf, m * n).await
}

/// How many `(batch, group)` iterations of a conv may share one submission.
///
/// Split out from [`gpu_conv2d`] so the bounds are testable without a device.
/// `per_iter_bytes` is what one iteration holds live (its im2col upload, its
/// result buffer and its slice of the shared staging buffer); `c_size` is one
/// result, which the chunk's staging buffer holds `chunk_len` of.
///
/// A zero divisor means "this term imposes no bound" — an iteration that costs
/// no bytes cannot exhaust a byte budget. Returns 0 only for an empty range.
fn conv_chunk_len(
    total_iters: usize,
    per_iter_bytes: u64,
    c_size: u64,
    max_buffer_size: u64,
) -> usize {
    if total_iters == 0 {
        return 0;
    }
    let by_budget = CONV_CHUNK_BYTE_BUDGET
        .checked_div(per_iter_bytes)
        .map_or(total_iters, |v| usize::try_from(v).unwrap_or(usize::MAX));
    let by_staging = max_buffer_size
        .checked_div(c_size)
        .map_or(total_iters, |v| usize::try_from(v).unwrap_or(usize::MAX));
    by_budget
        .min(by_staging)
        .min(CONV_MAX_CHUNK_ITERS)
        .clamp(1, total_iters)
}

/// Which GEMM kernel a conv's inner multiply uses, plus its dispatch grid.
///
/// Every `(batch, group)` iteration of a conv runs the *same* `[m, k] x [k, n]`
/// shape, so the kernel choice and the grid are computed once for the whole
/// call instead of being re-derived per iteration inside `gpu_matmul`.
struct ConvGemmPlan<'a> {
    pipeline: &'a wgpu::ComputePipeline,
    layout: &'a wgpu::BindGroupLayout,
    wg_x: u32,
    wg_y: u32,
}

/// Pick the tiled or basic GEMM kernel for `[m, k] x [k, n]` and validate its
/// dispatch grid against the device limits. Mirrors `gpu_matmul`'s selection
/// exactly, so a conv runs the same kernel it always did.
fn plan_conv_gemm(ctx: &GpuContext, m: usize, k: usize, n: usize) -> Option<ConvGemmPlan<'_>> {
    if m >= TILED_MIN_DIM && n >= TILED_MIN_DIM && k >= TILED_MIN_DIM {
        // Tiled kernel: workgroup_size(16, 16), gid.x = col, gid.y = row.
        let wg_x = u32::try_from(n).ok()?.div_ceil(16);
        let wg_y = u32::try_from(m).ok()?.div_ceil(16);
        if !dispatch_2d_fits(&ctx.limits, wg_x, wg_y) {
            return None;
        }
        Some(ConvGemmPlan {
            pipeline: &ctx.tiled_matmul_pipeline,
            layout: &ctx.tiled_matmul_bind_group_layout,
            wg_x,
            wg_y,
        })
    } else {
        // Basic kernel: workgroup_size(8, 8), gid.x = row, gid.y = col.
        let wg_x = u32::try_from(m).ok()?.div_ceil(8);
        let wg_y = u32::try_from(n).ok()?.div_ceil(8);
        if !dispatch_2d_fits(&ctx.limits, wg_x, wg_y) {
            return None;
        }
        Some(ConvGemmPlan {
            pipeline: &ctx.matmul_pipeline,
            layout: &ctx.matmul_bind_group_layout,
            wg_x,
            wg_y,
        })
    }
}

/// The GEMM shape `[m, k] x [k, n]` a Conv2D reduces to, or `None` when the
/// convolution is malformed or degenerate.
///
/// [c3] Split out so the size gate below can be applied *before* choosing
/// between the direct kernel and the hybrid im2col path — both must decline at
/// exactly the same size, or wiring in the direct kernel would silently start
/// dispatching convolutions the crate has always sent to the CPU.
fn conv_gemm_shape(
    input_shape: &[usize],
    weight: &Tensor,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Option<(usize, usize, usize)> {
    if input_shape.len() != 4 || weight.shape.len() != 4 {
        return None;
    }
    let (h, w) = (input_shape[2], input_shape[3]);
    let (c_out, c_per_group, kh, kw) = (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    );
    if group == 0 || strides[0] == 0 || strides[1] == 0 || kh == 0 || kw == 0 {
        return None;
    }
    if c_out % group != 0 {
        return None;
    }
    let padded_h = h.checked_add(pads[0])?.checked_add(pads[2])?;
    let padded_w = w.checked_add(pads[1])?.checked_add(pads[3])?;
    let span_h = dilations[0].checked_mul(kh - 1)?.checked_add(1)?;
    let span_w = dilations[1].checked_mul(kw - 1)?.checked_add(1)?;
    let oh = padded_h.checked_sub(span_h)? / strides[0] + 1;
    let ow = padded_w.checked_sub(span_w)? / strides[1] + 1;
    Some((
        c_out / group,
        c_per_group.checked_mul(kh)?.checked_mul(kw)?,
        oh.checked_mul(ow)?,
    ))
}

/// GPU-accelerated Conv2D with a fused bias and activation.
///
/// \[c3\] The entry point the whole conv path now goes through. It tries the
/// direct implicit-GEMM kernel first
/// ([`gpu_conv2d_implicit_async`](crate::shaders::gpu_conv2d_implicit_async):
/// no host im2col, bias and activation folded into the epilogue) and falls
/// back to the hybrid im2col path below for anything that kernel declines —
/// today that means `group > 1`, plus any shape the device cannot bind or
/// dispatch. The hybrid path has no fused activation, so `act` is applied on
/// the host there; the *result* of this function carries the activation either
/// way.
///
/// Both paths are gated on the same `GPU_THRESHOLD`, applied here once, so
/// the set of convolutions this crate accepts is exactly what it always was.
///
/// See [`gpu_conv2d_async`] for the un-fused form and why it exists separately.
///
/// Uploads the weight on every call; [`gpu_conv2d_fused_resident_async`] is the
/// form that keeps it on the device.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_conv2d_fused_async(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    act: crate::shaders::ConvActivation,
) -> Option<Tensor> {
    gpu_conv2d_fused_resident_async(
        ctx,
        input,
        weight,
        bias,
        crate::context::WeightKeys::default(),
        strides,
        pads,
        dilations,
        group,
        act,
    )
    .await
}

/// [`gpu_conv2d_fused_async`] with the weight and bias kept on the device.
///
/// `keys` names the invariant operands; see
/// [`gpu_conv2d_implicit_resident_async`](crate::shaders::gpu_conv2d_implicit_resident_async),
/// which is where they are honoured.
///
/// # The hybrid fallback deliberately does not take them
///
/// The im2col path below runs for what the direct kernel declines — grouped
/// convolution, and shapes the device cannot bind — and it does not upload the
/// weight tensor at all: it uploads one buffer per `group`, each a *slice* of
/// the weight, so one caller identity does not name one buffer there. Its
/// dominant traffic is the column matrix anyway, which is derived from the
/// input and changes every frame. Making that path resident is a different
/// change (per-group identities) with a much smaller prize, and inventing
/// composite keys here would put ONNX-shaped naming into this crate.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_conv2d_fused_resident_async(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    keys: crate::context::WeightKeys<'_>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    act: crate::shaders::ConvActivation,
) -> Option<Tensor> {
    gpu_conv2d_fused_placed_async(
        ctx,
        TensorSource::tensor(input),
        weight,
        bias,
        keys,
        strides,
        pads,
        dilations,
        group,
        act,
        OutputPlacement::Host,
    )
    .await?
    .into_tensor()
}

/// [`gpu_conv2d_fused_resident_async`] with the activation free to arrive on,
/// and stay on, the device.
///
/// The conv entry point the residency-aware dispatcher calls. Same gate
/// (`GPU_THRESHOLD` on the implied GEMM's FLOPs), same kernel, same numerics —
/// the only differences are where the input comes from and where the result is
/// left.
///
/// # The hybrid fallback is host-only, and that is a decline rather than a bug
///
/// When the implicit kernel declines (a grouped convolution, a shape the device
/// cannot bind), the host path falls back to `gpu_conv2d_hybrid_async`, which
/// im2cols on the CPU and therefore needs the input's bytes. A device-resident
/// input has none to give, so this returns `None` and the caller — which is the
/// one holding the activation — reads it back and runs the CPU operator. That
/// is exactly the contract every other decline in this crate has.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_conv2d_fused_placed_async(
    ctx: &GpuContext,
    input: TensorSource<'_>,
    weight: &Tensor,
    bias: Option<&Tensor>,
    keys: crate::context::WeightKeys<'_>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    act: crate::shaders::ConvActivation,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    let (m, k, n) = conv_gemm_shape(input.shape(), weight, strides, pads, dilations, group)?;
    // `conv_min_mac`, not `gemm_min_mac`: `Conv` is the one op every
    // measurement has found a clear GPU winner (0.44x the CPU kernel across
    // InSwapper's 20 convolutions), its implicit-GEMM cost model is not the flat
    // GEMM one, and it keeps the threshold it was measured with. See
    // `crate::context::tuning::GpuTuning::conv_min_mac`.
    if !ctx.tuning().conv_admits(m, k, n) {
        return None;
    }
    if ctx.is_degraded() {
        return None;
    }
    if let Some(out) = crate::shaders::gpu_conv2d_implicit_placed_async(
        ctx, input, weight, bias, keys, strides, pads, dilations, group, act, placement,
    )
    .await
    {
        return Some(out);
    }
    let mut out =
        gpu_conv2d_hybrid_async(ctx, input, weight, bias, strides, pads, dilations, group).await?;
    act.apply_host(&mut out.data);
    Some(GpuOutput::Host(out))
}

/// Blocking form of [`gpu_conv2d_fused_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d_fused(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    act: crate::shaders::ConvActivation,
) -> Option<Tensor> {
    block_on_gpu(gpu_conv2d_fused_async(
        ctx, input, weight, bias, strides, pads, dilations, group, act,
    ))
}

/// GPU-accelerated Conv2D, no fused activation.
///
/// \[c3\] This is deliberately **not** an activation-fusing entry point, and the
/// distinction is load-bearing rather than stylistic. `src/session/gpu_dispatch.rs`
/// reads the `activation` attribute the optimizer's Conv+Relu / Conv+Clip
/// fusion folded into the node, calls *this* function, and then applies that
/// activation itself (`apply_conv_activation`). Fusing an activation in here
/// too would apply it twice. ReLU and Clip are idempotent so the bug would
/// hide, but `leaky_relu(leaky_relu(x)) = alpha^2 * x` for `x < 0` — the slope
/// would be silently squared. A caller that wants the fused form must ask for
/// it explicitly, via [`gpu_conv2d_fused_async`].
///
/// The convolution itself still runs on the direct implicit-GEMM kernel; only
/// the epilogue's activation is `None`.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_conv2d_async(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Option<Tensor> {
    gpu_conv2d_fused_async(
        ctx,
        input,
        weight,
        bias,
        strides,
        pads,
        dilations,
        group,
        crate::shaders::ConvActivation::None,
    )
    .await
}

/// Hybrid Conv2D: im2col on CPU, GEMM on GPU, bias on CPU.
///
/// \[c3\] Kept as the fallback for everything the direct kernel in
/// `shaders/conv2d.rs` declines (grouped convolution, and any shape whose
/// operands the direct kernel cannot bind). It is *not* the fast path any
/// more: for InSwapper's dominant layer it uploads a 37.7 MB column matrix per
/// call where the direct kernel uploads a 4 MB input, and it pays a CPU im2col
/// ahead of every dispatch.
///
/// [a7-21] The `(batch, group)` iterations are batched. Previously this loop
/// called `gpu_matmul` once per iteration, and each call allocated fresh
/// A/B/C/params/staging buffers, built a bind group, submitted, and then
/// blocked in `read_back` on `poll(Wait)` before the next iteration could even
/// start — for a batch of 8 with `group = 4`, 32 serialized submit-and-fence
/// round trips, each paying the full pipeline-drain latency, plus 160 buffer
/// allocations that never touched `ctx.pool`.
///
/// Now the per-group weight buffers and the GEMM params buffer are created
/// once for the whole call (their contents do not vary with `batch`), the
/// iterations are grouped into chunks bounded by `CONV_CHUNK_BYTE_BUDGET`,
/// and each chunk is encoded into **one** command buffer — a single compute
/// pass holding every dispatch, then one copy per result into a shared staging
/// buffer — and submitted and read back **once**. The C buffers are routed
/// through `ctx.pool` like every other kernel in this crate.
///
/// # Numerics
///
/// Results are bit-identical to the previous implementation: the same kernel
/// runs on the same inputs in the same order, only the submission grouping
/// changed. Each dispatch in a chunk owns distinct col and C buffers, so no
/// dispatch can observe another's writes.
///
/// Falls back to `None` if the GEMM is too small for GPU benefit.
///
/// \[c3\] Public so the direct kernel's benchmark
/// (`examples/c3_conv2d_gflops.rs`) can time the path it replaced against it
/// on the same shapes. Production callers should use [`gpu_conv2d_async`] or
/// [`gpu_conv2d_fused_async`], which pick between the two paths.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_conv2d_hybrid_async(
    ctx: &GpuContext,
    input: TensorSource<'_>,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Option<Tensor> {
    // im2col runs on the host, so this path needs the input's bytes. A
    // device-resident activation has none to give and declines here — its
    // holder reads it back and runs the CPU operator, the same contract every
    // other decline in this crate has.
    let input_data = input.host_data()?;
    let input_shape = input.shape();
    // Every dimension here comes from the model file: validate the ranks and
    // the divisors before doing any arithmetic, so a malformed Conv declines
    // (and the CPU operator reports a typed error) instead of panicking.
    if input_shape.len() != 4 || weight.shape.len() != 4 {
        return None;
    }
    let [n, c_in, h, w] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];
    let [c_out, c_per_group, kh, kw] = [
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    ];
    if group == 0 || strides[0] == 0 || strides[1] == 0 || kh == 0 || kw == 0 {
        return None;
    }
    if c_out % group != 0 || c_in != c_per_group.checked_mul(group)? {
        return None;
    }
    if input_data.len() < n.checked_mul(c_in)?.checked_mul(h)?.checked_mul(w)?
        || weight.data.len()
            < c_out
                .checked_mul(c_per_group)?
                .checked_mul(kh)?
                .checked_mul(kw)?
    {
        return None;
    }

    // Output extents, using checked arithmetic: a padded/dilated window larger
    // than the padded input would underflow `usize`.
    let padded_h = h.checked_add(pads[0])?.checked_add(pads[2])?;
    let padded_w = w.checked_add(pads[1])?.checked_add(pads[3])?;
    let span_h = dilations[0].checked_mul(kh - 1)?.checked_add(1)?;
    let span_w = dilations[1].checked_mul(kw - 1)?.checked_add(1)?;
    let oh = padded_h.checked_sub(span_h)? / strides[0] + 1;
    let ow = padded_w.checked_sub(span_w)? / strides[1] + 1;

    let c_out_per_group = c_out / group;
    let col_rows = c_per_group.checked_mul(kh)?.checked_mul(kw)?;
    let col_cols = oh.checked_mul(ow)?;
    if col_rows == 0 || col_cols == 0 || c_out_per_group == 0 {
        return None;
    }
    if let Some(b) = bias {
        if b.data.len() < c_out {
            return None;
        }
    }

    // Check if the GEMM is large enough for GPU. Same `u64` widening as
    // `GpuTuning::gemm_mac` documents: a conv whose inner GEMM is at or above
    // 2^32 multiply-accumulates used to decline on wasm32 instead of
    // dispatching. `conv_min_mac`, not `gemm_min_mac` — see
    // `gpu_conv2d_fused_placed_async`.
    if !ctx
        .tuning()
        .conv_admits(c_out_per_group, col_rows, col_cols)
    {
        return None;
    }

    if ctx.is_degraded() {
        return None;
    }

    let out_len = n.checked_mul(c_out)?.checked_mul(oh)?.checked_mul(ow)?;

    // Every iteration runs the same GEMM shape, so validate and plan it once.
    let (gemm_m, gemm_k, gemm_n) = (c_out_per_group, col_rows, col_cols);
    let a_len = gemm_m.checked_mul(gemm_k)?;
    let col_len = gemm_k.checked_mul(gemm_n)?;
    let c_len = gemm_m.checked_mul(gemm_n)?;
    // All three operands must be allocatable and bindable on this device. The
    // weight and col buffers are sized by their contents, so for those only the
    // check matters; `c_size` is also the copy/staging stride below.
    checked_storage_bytes(&ctx.limits, a_len)?;
    let col_size = checked_storage_bytes(&ctx.limits, col_len)?;
    let c_size = checked_storage_bytes(&ctx.limits, c_len)?;
    let plan = plan_conv_gemm(ctx, gemm_m, gemm_k, gemm_n)?;

    // Flat list of the (batch, group) iterations, in the original order.
    let total_iters = n.checked_mul(group)?;

    let device = &ctx.device;
    let mut out = vec![0.0f32; out_len];

    // A zero-length batch has nothing to convolve. The old loop simply never
    // ran; returning here keeps that behaviour and, more importantly, keeps the
    // chunk clamp below from being handed an empty range.
    if total_iters == 0 {
        return Some(Tensor::new(out, vec![n, c_out, oh, ow]));
    }

    // How many iterations may share one submission. Each one holds a col
    // buffer, a C buffer and its slice of the shared staging buffer; the
    // per-group weight buffers are hoisted out and counted once.
    let per_iter_bytes = col_size.checked_add(c_size)?.checked_add(c_size)?;
    let chunk_len = conv_chunk_len(
        total_iters,
        per_iter_bytes,
        c_size,
        ctx.limits.max_buffer_size,
    );

    // Hoisted: one buffer per group (weights do not vary with `batch`) and one
    // params buffer (m/k/n are identical for every iteration).
    let mut weight_bufs = Vec::with_capacity(group);
    for g in 0..group {
        let w_off = g.checked_mul(c_out_per_group)?.checked_mul(col_rows)?;
        let w_end = w_off.checked_add(a_len)?;
        let weight_slice = weight.data.get(w_off..w_end)?;
        weight_bufs.push(ctx.upload_buffer(
            "conv_weight",
            bytemuck::cast_slice(weight_slice),
            wgpu::BufferUsages::STORAGE,
        )?);
    }
    let params = GemmParams {
        m: u32::try_from(gemm_m).ok()?,
        k: u32::try_from(gemm_k).ok()?,
        n: u32::try_from(gemm_n).ok()?,
        _pad: 0,
    };
    let params_buf = ctx.upload_buffer(
        "conv_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let mut col = vec![0.0f32; col_len];
    let mut iter_start = 0usize;
    while iter_start < total_iters {
        let this_chunk = chunk_len.min(total_iters - iter_start);
        let staging_size = c_size.checked_mul(u64::try_from(this_chunk).ok()?)?;
        if !ctx.limits.buffer_fits(staging_size) {
            return None;
        }
        // `per_iter_bytes` already covers this chunk's col, C and staging
        // slice; the weight buffers above are live and counted already.
        if !ctx.budget_admits(&[per_iter_bytes.checked_mul(u64::try_from(this_chunk).ok()?)?]) {
            return None;
        }

        let scope = ErrorScope::begin(ctx);

        // Build every buffer and bind group of the chunk before encoding, so
        // each dispatch owns distinct col and C buffers — no dispatch in the
        // pass can observe another's writes, whatever the backend does about
        // same-usage barriers.
        let mut col_bufs = Vec::with_capacity(this_chunk);
        let mut c_bufs = Vec::with_capacity(this_chunk);
        let mut bind_groups = Vec::with_capacity(this_chunk);
        for slot in 0..this_chunk {
            let flat = iter_start + slot;
            let batch = flat / group;
            let g = flat % group;

            // im2col on CPU into the reused scratch buffer.
            im2col(
                input_data,
                c_in,
                h,
                w,
                g * c_per_group,
                c_per_group,
                kh,
                kw,
                strides,
                pads,
                dilations,
                oh,
                ow,
                batch,
                &mut col,
            );
            col_bufs.push(ctx.upload_buffer(
                "conv_col",
                bytemuck::cast_slice(&col),
                wgpu::BufferUsages::STORAGE,
            )?);
            let c_buf = ctx.pooled_buffer(
                c_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )?;
            // The pooled buffer may be *larger* than `c_size` (the pool hands
            // back anything within 2x -- see `GpuBufferPool::get_buffer`), and
            // `as_entire_binding()` would then bind that larger size, which can
            // exceed `max_storage_buffer_binding_size` even though `c_size`
            // itself was validated. Bind the exact range instead, as
            // `conv2d::gpu_conv2d_implicit_resident_async`'s `output_binding` does for the same
            // reason.
            let c_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &c_buf,
                offset: 0,
                size: wgpu::BufferSize::new(c_size),
            });
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("conv_gemm_bg"),
                layout: plan.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weight_bufs.get(g)?.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: col_bufs.get(slot)?.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_binding,
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            }));
            c_bufs.push(c_buf);
        }

        let staging_buf = ctx.staging_buffer("conv_staging", staging_size)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv_enc"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_gemm_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(plan.pipeline);
            for bind_group in &bind_groups {
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.dispatch_workgroups(plan.wg_x, plan.wg_y, 1);
            }
        }
        // One copy per result into the shared staging buffer. `c_size` is a
        // multiple of 4 (it is a count of f32s), so every destination offset
        // satisfies `COPY_BUFFER_ALIGNMENT`.
        for (slot, c_buf) in c_bufs.iter().enumerate() {
            let dst = c_size.checked_mul(u64::try_from(slot).ok()?)?;
            encoder.copy_buffer_to_buffer(c_buf, 0, &staging_buf, dst, c_size);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        if !scope.finish_async(ctx).await {
            return None;
        }

        let chunk_data = read_back_async(ctx, &staging_buf, c_len.checked_mul(this_chunk)?).await?;

        // The read-back completed, so the submission has retired and the C
        // buffers are safe to recycle.
        if let Ok(mut pool) = ctx.pool.lock() {
            for c_buf in c_bufs {
                pool.return_buffer(c_buf);
            }
        }

        // Scatter each result into its place in the output, adding bias.
        for slot in 0..this_chunk {
            let flat = iter_start + slot;
            let batch = flat / group;
            let g = flat % group;
            let gemm_result = chunk_data.get(slot * c_len..(slot + 1) * c_len)?;

            let o_off = batch
                .checked_mul(c_out)?
                .checked_add(g.checked_mul(c_out_per_group)?)?
                .checked_mul(col_cols)?;
            let o_end = o_off.checked_add(c_out_per_group.checked_mul(col_cols)?)?;
            let out_slice = out.get_mut(o_off..o_end)?;
            if out_slice.len() != gemm_result.len() {
                return None;
            }
            out_slice.copy_from_slice(gemm_result);

            if let Some(b) = bias {
                for (oc, chunk) in out_slice.chunks_exact_mut(col_cols).enumerate() {
                    let bv = *b.data.get(g * c_out_per_group + oc)?;
                    for value in chunk.iter_mut() {
                        *value += bv;
                    }
                }
            }
        }

        iter_start += this_chunk;
    }

    Some(Tensor::new(out, vec![n, c_out, oh, ow]))
}

/// Blocking form of [`gpu_conv2d_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Option<Tensor> {
    block_on_gpu(gpu_conv2d_async(
        ctx, input, weight, bias, strides, pads, dilations, group,
    ))
}

/// Build the im2col column matrix for one (batch, group) slice.
/// Identical logic to the CPU version in `ops/conv.rs`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    in_c_start: usize,
    c_per_group: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    oh: usize,
    ow: usize,
    batch: usize,
    col: &mut [f32],
) {
    let col_cols = oh * ow;
    let mut row = 0;
    for ic in 0..c_per_group {
        let in_c = in_c_start + ic;
        let in_plane = &input[(batch * c_in + in_c) * h * w..][..h * w];
        for ky in 0..kh {
            for kx in 0..kw {
                for oy in 0..oh {
                    let iy = (oy * strides[0] + ky * dilations[0]) as isize - pads[0] as isize;
                    if iy < 0 || iy >= h as isize {
                        let base = row * col_cols + oy * ow;
                        for ox in 0..ow {
                            col[base + ox] = 0.0;
                        }
                        continue;
                    }
                    let iy = iy as usize;
                    let base = row * col_cols + oy * ow;
                    for ox in 0..ow {
                        let ix = (ox * strides[1] + kx * dilations[1]) as isize - pads[1] as isize;
                        col[base + ox] = if ix >= 0 && ix < w as isize {
                            in_plane[iy * w + ix as usize]
                        } else {
                            0.0
                        };
                    }
                }
                row += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::tuning::GpuTuning;

    /// The FLOP count must be computed in `u64`, not `usize`.
    ///
    /// On wasm32 `usize` is 32 bits, so a 2048³ GEMM (8.59 GFLOP) overflowed
    /// the old `usize` product and made `gpu_matmul` decline the *largest*
    /// multiplies while accepting small ones. This asserts the widened
    /// arithmetic directly, so the regression cannot come back on a 64-bit host
    /// where the bug is invisible.
    #[test]
    fn gemm_flops_does_not_overflow_a_32_bit_usize() {
        let tuning = GpuTuning::for_class(crate::context::GpuPerfClass::Discrete);
        let mac = GpuTuning::gemm_mac;
        assert_eq!(mac(2048, 2048, 2048), Some(8_589_934_592));
        assert!(mac(2048, 2048, 2048).is_some_and(|f| f > u64::from(u32::MAX)));
        assert!(mac(2048, 2048, 2048).is_some_and(|f| f >= tuning.gemm_min_mac));
        // Small shapes are unchanged, and still below the threshold.
        assert_eq!(mac(32, 32, 32), Some(32_768));
        assert!(mac(32, 32, 32).is_some_and(|f| f < tuning.gemm_min_mac));
        // Only a product that overflows `u64` declines.
        assert_eq!(mac(usize::MAX, usize::MAX, 2), None);
        assert_eq!(mac(0, 1 << 20, 1 << 20), Some(0));
    }

    /// CPU reference matmul for verification.
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    #[test]
    fn test_gpu_matmul_small_falls_back() {
        // Below threshold: should return None.
        let ctx = GpuContext::try_new();
        if let Some(ref ctx) = ctx {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![5.0, 6.0, 7.0, 8.0];
            let result = gpu_matmul(ctx, &a, &b, 2, 2, 2);
            assert!(result.is_none(), "small matmul should fall back to CPU");
        }
    }

    #[test]
    fn test_gpu_matmul_large() {
        let ctx = GpuContext::try_new();
        if let Some(ref ctx) = ctx {
            let m = 64;
            let k = 64;
            let n = 64;
            let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();

            let gpu_result = gpu_matmul(ctx, &a, &b, m, k, n);
            let gpu_out = match gpu_result {
                Some(out) => out,
                None => return,
            };
            let cpu_out = cpu_matmul(&a, &b, m, k, n);

            for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
                assert!((g - c).abs() < 1.0, "mismatch at {i}: gpu={g} cpu={c}");
            }
        }
    }

    #[test]
    fn test_tiled_matmul_basic() {
        // 32x32 tiled multiply, verify against CPU
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return, // skip if no GPU
        };

        let m = 32;
        let k = 32;
        let n = 32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.3 - 1.5).collect();

        // gpu_matmul_tiled requires flops >= threshold, so use large enough matrices
        // that pass the threshold: 32*32*32 = 32768 < 10M, so it returns None.
        // Use the inner function directly for testing.
        let gpu_out = match block_on_gpu(gpu_matmul_tiled_inner(&ctx, &a, &b, m, k, n)) {
            Some(out) => out,
            None => return,
        };
        let cpu_out = cpu_matmul(&a, &b, m, k, n);

        assert_eq!(gpu_out.len(), cpu_out.len());
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1.0,
                "tiled basic mismatch at {i}: gpu={g} cpu={c}"
            );
        }
    }

    #[test]
    fn test_tiled_matmul_non_square() {
        // 64x48 * 48x32 non-square tiled multiply
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let m = 64;
        let k = 48;
        let n = 32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32) * 0.2 - 0.8).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.4 - 1.2).collect();

        let gpu_out = match block_on_gpu(gpu_matmul_tiled_inner(&ctx, &a, &b, m, k, n)) {
            Some(out) => out,
            None => return,
        };
        let cpu_out = cpu_matmul(&a, &b, m, k, n);

        assert_eq!(gpu_out.len(), m * n);
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1.0,
                "tiled non-square mismatch at {i}: gpu={g} cpu={c}"
            );
        }
    }

    #[test]
    fn test_tiled_matmul_large() {
        // 256x256 large tiled multiply
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };

        let m = 256;
        let k = 256;
        let n = 256;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.15 - 0.9).collect();

        // 256^3 = 16M > threshold, so gpu_matmul_tiled should work
        let gpu_out = match gpu_matmul_tiled(&ctx, &a, &b, m, k, n) {
            Some(out) => out,
            None => return,
        };
        let cpu_out = cpu_matmul(&a, &b, m, k, n);

        assert_eq!(gpu_out.len(), m * n);
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - c).abs() < 2.0,
                "tiled large mismatch at {i}: gpu={g} cpu={c}"
            );
        }
    }

    // ------------------------------------------------------------------
    // a7-21 — conv chunk planning (pure arithmetic, no device needed)
    // ------------------------------------------------------------------

    const HUGE_BUFFER: u64 = 8 << 30;

    #[test]
    fn conv_chunk_len_batches_small_iterations_and_caps_at_the_limit() {
        // The audit's conv: col = 72 * 8836 f32, C = 16 * 8836 f32.
        let col_size = 72 * 8836 * 4;
        let c_size = 16 * 8836 * 4;
        let per_iter = col_size + 2 * c_size;
        let chunk = conv_chunk_len(32, per_iter, c_size, HUGE_BUFFER);
        assert!(
            (2..=CONV_MAX_CHUNK_ITERS).contains(&chunk),
            "expected a real batch bounded by the cap, got {chunk}"
        );
        // Never more than the hard cap, however cheap the iterations are.
        assert_eq!(
            conv_chunk_len(1000, 1, 1, HUGE_BUFFER),
            CONV_MAX_CHUNK_ITERS
        );
        // Never more iterations than exist.
        assert_eq!(conv_chunk_len(3, 1, 1, HUGE_BUFFER), 3);
    }

    #[test]
    fn conv_chunk_len_degenerates_to_one_for_huge_iterations() {
        // An im2col matrix larger than the whole budget: one per submission,
        // i.e. exactly the pre-a7-21 behaviour, so a big conv cannot regress.
        let per_iter = CONV_CHUNK_BYTE_BUDGET * 2;
        assert_eq!(conv_chunk_len(32, per_iter, 1 << 20, HUGE_BUFFER), 1);
    }

    #[test]
    fn conv_chunk_len_respects_the_staging_buffer_limit() {
        // A device that can only allocate 4 results' worth must not be asked
        // for a staging buffer holding more than that.
        let c_size = 1 << 20;
        assert_eq!(conv_chunk_len(32, 1, c_size, c_size * 4), 4);
    }

    #[test]
    fn conv_chunk_len_handles_degenerate_inputs_without_panicking() {
        // An empty iteration range must not reach `clamp(1, 0)`.
        assert_eq!(conv_chunk_len(0, 1024, 1024, HUGE_BUFFER), 0);
        // Zero-byte divisors mean "no bound from this term", never a divide by
        // zero; the cap still applies.
        assert_eq!(conv_chunk_len(4, 0, 0, HUGE_BUFFER), 4);
        assert_eq!(conv_chunk_len(100, 0, 0, 0), CONV_MAX_CHUNK_ITERS);
        // The result is always a usable chunk size for a non-empty range.
        assert!(conv_chunk_len(7, u64::MAX, u64::MAX, 1) >= 1);
    }

    #[test]
    fn test_gpu_conv2d_zero_batch_returns_an_empty_tensor() {
        let Some(ctx) = GpuContext::try_new() else {
            return;
        };
        // N = 0 used to run the loop zero times; the batched path must reach
        // the same answer rather than dividing by, or clamping to, an empty
        // range. Shapes come from model files, so this must never panic.
        let input = Tensor::new(Vec::new(), vec![0, 32, 96, 96]);
        let weight = Tensor::new(vec![0.5; 32 * 32 * 9], vec![32, 32, 3, 3]);
        let result = gpu_conv2d(&ctx, &input, &weight, None, [1, 1], [0, 0, 0, 0], [1, 1], 1);
        if let Some(out) = result {
            assert_eq!(out.shape, vec![0, 32, 94, 94]);
            assert!(out.data.is_empty());
        }
    }

    #[test]
    fn test_gpu_conv2d_small_falls_back() {
        let ctx = GpuContext::try_new();
        if let Some(ref ctx) = ctx {
            let input = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
            let weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
            let result = gpu_conv2d(ctx, &input, &weight, None, [1, 1], [0, 0, 0, 0], [1, 1], 1);
            assert!(result.is_none(), "small conv should fall back to CPU");
        }
    }
}
