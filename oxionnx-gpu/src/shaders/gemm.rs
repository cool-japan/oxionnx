//! GPU-accelerated `Gemm` with `transB = 1`: `out = alpha * A @ B^T + beta * C`.
//!
//! Named `gpu_gemm_nt` after `oxionnx-ops::attention::gemm`'s own
//! `matmul_nn` / `matmul_nt` convention (`nt` = "A normal, B transposed"),
//! since that is the only access pattern this kernel implements: every one
//! of InSwapper's 12 Gemm nodes (the AdaIN heads, `[1..64,512] x
//! [2048,512]^T -> [1..64,2048]`) has `alpha=1, beta=1, transA=0, transB=1`.
//! `B` is `[N, K]` row-major (a PyTorch `nn.Linear` weight layout) and is
//! read with the transposed access pattern directly in WGSL --
//! `b[col*K + i]` instead of `b[i*N + col]` -- rather than physically
//! transposed first, exactly as asked.
//!
//! `alpha`/`beta` are real uniform values (not hardcoded to `1.0`), matching
//! `elementwise.rs::gpu_leaky_relu_alpha`'s precedent of uploading a scalar
//! attribute instead of baking it into the kernel; the task's stated
//! configuration (`alpha=1, beta=1`) is this kernel's primary tested case,
//! not its only supported one. `C` may be absent (beta term contributes
//! nothing), a length-`N` row vector broadcast over every output row (the
//! bias case the task describes), or a full `[M, N]` matrix; any other `C`
//! length is declined rather than silently mis-indexed.
//!
//! A simple (non-shared-memory) kernel is used, per the task's framing --
//! the shapes involved (`M` up to 64, `K = 512`, `N = 2048`) are a few tens
//! of millions of FLOPs, trivial either way. Adjacent threads (`gid.x`,
//! spanning `N`) read `B` rows `K` floats apart, so `B` reads do not
//! coalesce across a warp; a shared-memory tile (as in
//! `TILED_MATMUL_SHADER`) would fix this and is the natural next
//! optimization once this kernel is integrated into session dispatch.
//!
//! See [`kernel_support`](super::kernel_support) for why this kernel builds its
//! pipeline at its entry point rather than eagerly on the context, and why
//! there is no minimum-size gate.

use crate::context::activation::{GpuOutput, OutputPlacement, TensorSource};
use crate::context::pipeline_cache::PipelineLookup;
use crate::context::{GpuContext, WeightFormat, WeightKeys};
use crate::device_guard::{
    block_on_gpu, checked_storage_bytes, dispatch_2d_fits, finish_output_async, ErrorScope,
};

use super::kernel_support::{bgl_ro, bgl_rw, bgl_uniform, build_pipeline};

/// Tile width/height for the 2-D dispatch grid (`@workgroup_size(16, 16)`).
const GEMM_WG: u32 = 16;

/// How the (optional) `C` bias is read. Encoded as a uniform `u32` because
/// WGSL bindings cannot be conditionally omitted -- when the caller passes
/// `c: None`, a 1-element dummy buffer is still bound, but `c_mode == 0`
/// means the shader's `if` never touches it (WGSL `if` is real control flow,
/// unlike `select`, so the dummy binding is never actually read out of
/// bounds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CMode {
    None = 0,
    RowBroadcast = 1,
    Full = 2,
}

/// Uniform block for the Gemm kernel.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmNtParams {
    m: u32,
    k: u32,
    n: u32,
    c_mode: u32,
    alpha: f32,
    beta: f32,
    _pad0: u32,
    _pad1: u32,
}

/// The kernel, in `f32`.
///
/// [w2-f16] Unchanged by the half-precision work — the `f16` variant is derived
/// from this text by [`super::f16_variant::gemm_nt_f16`], so a toggle-off
/// dispatch compiles exactly the shader it always did. `pub(super)` only so
/// that module can read it.
pub(super) const GEMM_NT_SHADER: &str = r#"
struct Params {
    m: u32,
    k: u32,
    n: u32,
    c_mode: u32,
    alpha: f32,
    beta: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;      // [M, K] row-major
@group(0) @binding(1) var<storage, read> b: array<f32>;      // [N, K] row-major (read as B^T)
@group(0) @binding(2) var<storage, read> c_buf: array<f32>;  // see CMode
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn c_value(row: u32, col: u32) -> f32 {
    if (params.c_mode == 0u) {
        return 0.0;
    }
    if (params.c_mode == 1u) {
        return c_buf[col];
    }
    return c_buf[row * params.n + col];
}

@compute @workgroup_size(16, 16)
fn gemm_nt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.m || col >= params.n) { return; }

    var acc: f32 = 0.0;
    let a_base = row * params.k;
    let b_base = col * params.k;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        acc = acc + a[a_base + i] * b[b_base + i];
    }
    output[row * params.n + col] = params.alpha * acc + params.beta * c_value(row, col);
}
"#;

/// The bind group layout both precision variants of this kernel use:
/// `A`, `B`, `C`, output, params.
fn gemm_bgl_entries() -> [wgpu::BindGroupLayoutEntry; 5] {
    [bgl_ro(0), bgl_ro(1), bgl_ro(2), bgl_rw(3), bgl_uniform(4)]
}

/// This kernel's shader module label, which is also its `@compute` entry point.
const GEMM_LABEL: &str = "gemm_nt";

/// [w2-f16] The half-precision variant's label — a different label for the same
/// entry point, so the two can never share a slot in the pipeline cache.
const GEMM_F16_LABEL: &str = "gemm_nt_f16";

/// This kernel's `f32` pipeline, compiled once per context.
fn gemm_nt_pipeline(ctx: &GpuContext) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_pipeline(
        ctx,
        GEMM_LABEL,
        GEMM_NT_SHADER,
        GEMM_LABEL,
        &gemm_bgl_entries(),
    )
}

/// The compiled `f16` `gemm_nt` pipeline for this context, or `None` when half
/// precision is unavailable here.
///
/// Pipeline-cache keying needs no extra work in this kernel: entries are keyed
/// on `(label, entry_point, src)`, and the two variants differ in both `label`
/// and `src`. So a context that flips the toggle cannot be served the other
/// variant's pipeline by construction.
///
/// Like `conv2d`'s counterpart, the first compile on a context runs inside its
/// own error scope so a driver rejecting the extension declines this kernel
/// rather than degrading the whole context, and the refusal is remembered as a
/// `Rejected` slot in that context's cache so it is never retried. A hit — of
/// either kind — costs one lookup and never pushes an error scope at all.
async fn gemm_nt_pipeline_f16_async(
    ctx: &GpuContext,
) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    if !ctx.f16_compute_enabled() {
        return None;
    }
    let src = super::f16_variant::gemm_nt_f16(GEMM_NT_SHADER)?;
    match ctx.pipelines().lookup(GEMM_F16_LABEL, GEMM_LABEL, src) {
        PipelineLookup::Ready(pipeline, layout) => return Some((pipeline, layout)),
        PipelineLookup::Rejected => return None,
        PipelineLookup::Absent => {}
    }

    let device = &ctx.device;
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let built = crate::context::pipeline_cache::compile(
        device,
        GEMM_F16_LABEL,
        src,
        GEMM_LABEL,
        &gemm_bgl_entries(),
    );
    if guard.pop().await.is_some() {
        ctx.pipelines()
            .insert_rejected(GEMM_F16_LABEL, GEMM_LABEL, src);
        return None;
    }
    ctx.pipelines()
        .insert_ready(GEMM_F16_LABEL, GEMM_LABEL, src, &built);
    Some(built)
}

/// GPU-accelerated `out = alpha * A @ B^T + beta * C`.
///
/// `a` is `[m, k]` row-major, `b` is `[n, k]` row-major (`B^T` is applied by
/// the kernel's access pattern, never materialised). `c` is `None` (no bias
/// term), `Some` of length `n` (broadcast over every output row) or `Some`
/// of length `m * n` (a full matrix, no broadcast); any other length
/// declines. Declines on a degraded context, a shape/data-length mismatch,
/// non-finite `alpha`/`beta`, or a dispatch/buffer the device cannot cover.
///
/// Uploads `b` and `c` on every call. A caller whose `B` is a graph
/// initializer — ArcFace's `[512, 25088]` embedding weight, 51.4 MB, identical
/// on every frame — should use [`gpu_gemm_nt_resident_async`] and pass their
/// identities.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_gemm_nt_async(
    ctx: &GpuContext,
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    c: Option<&[f32]>,
    alpha: f32,
    beta: f32,
) -> Option<Vec<f32>> {
    gpu_gemm_nt_resident_async(ctx, a, m, k, b, n, c, alpha, beta, WeightKeys::default()).await
}

/// [`gpu_gemm_nt_async`] with `B` and `C` kept on the device.
///
/// `keys.weight` names `B`, `keys.bias` names `C` — the same matrix/vector
/// distinction the convolution makes. `A` is the activation and is never
/// cached. A named operand uploads on its first sight and binds from the
/// residency cache after that; numerics are unaffected, because the same bytes
/// reach the same binding either way.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_gemm_nt_resident_async(
    ctx: &GpuContext,
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    c: Option<&[f32]>,
    alpha: f32,
    beta: f32,
    keys: WeightKeys<'_>,
) -> Option<Vec<f32>> {
    let a_shape = [m, k];
    gpu_gemm_nt_placed_async(
        ctx,
        TensorSource::host(a, &a_shape),
        m,
        k,
        b,
        n,
        c,
        alpha,
        beta,
        keys,
        OutputPlacement::Host,
    )
    .await?
    .into_vec()
}

/// [`gpu_gemm_nt_resident_async`] with `A` free to arrive on, and the result
/// free to stay on, the device.
///
/// `B` and `C` keep going through the session-lifetime weight cache (`keys`);
/// they are the invariant operands. `A` is the activation, so it is the one
/// that gains from run-scoped residency.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_gemm_nt_placed_async(
    ctx: &GpuContext,
    a: TensorSource<'_>,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    c: Option<&[f32]>,
    alpha: f32,
    beta: f32,
    keys: WeightKeys<'_>,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    if ctx.is_degraded() || !alpha.is_finite() || !beta.is_finite() {
        return None;
    }
    if m == 0 || k == 0 || n == 0 {
        return None;
    }
    let a_len = m.checked_mul(k)?;
    let b_len = n.checked_mul(k)?;
    let out_len = m.checked_mul(n)?;
    if a.len() != a_len || b.len() != b_len {
        return None;
    }
    let (c_mode, c_data): (CMode, &[f32]) = match c {
        None => (CMode::None, &[]),
        Some(cd) if cd.len() == n => (CMode::RowBroadcast, cd),
        Some(cd) if cd.len() == out_len => (CMode::Full, cd),
        Some(_) => return None,
    };
    // WGSL always binds a `c_buf`; a 1-element placeholder stands in when
    // there is nothing to bind (never read -- see `CMode` docs above). That
    // placeholder is not the caller's tensor, so it never takes the caller's
    // identity in the residency cache.
    let c_upload: &[f32] = if c_data.is_empty() { &[0.0] } else { c_data };
    let c_key = if c_data.is_empty() { None } else { keys.bias };

    // [w2-f16] Resolve the half-precision variant before the dispatch's error
    // scope opens, and let its answer decide `B`'s on-device format. `A` (the
    // activation) and `C` (the bias) stay f32 in every case.
    let f16_pipeline = gemm_nt_pipeline_f16_async(ctx).await;
    let b_format = if f16_pipeline.is_some() {
        WeightFormat::F16
    } else {
        WeightFormat::F32
    };

    let a_bytes = checked_storage_bytes(&ctx.limits, a_len)?;
    // [w2-f16] `B` is validated against the device at its **f32** size in both
    // modes, and only the *budget* figure follows the on-device format.
    //
    // That asymmetry is deliberate: it keeps the set of shapes this kernel
    // declines independent of the toggle. Validating the f16 size would let a
    // `B` that is too large to bind as f32 dispatch to the GPU with half
    // precision on and fall to the CPU with it off — so flipping a numerics
    // switch would silently change *which* nodes run where, and a toggle-off
    // A/B would no longer be comparing the same placement. `conv2d` validates
    // its weight the same way, for the same reason.
    let b_bytes_f32 = checked_storage_bytes(&ctx.limits, b_len)?;
    let b_bytes = b_format.byte_len(b_len);
    let c_bytes = checked_storage_bytes(&ctx.limits, c_upload.len())?;
    let out_size = checked_storage_bytes(&ctx.limits, out_len)?;
    if !ctx.limits.buffer_fits(a_bytes)
        || !ctx.limits.buffer_fits(b_bytes_f32)
        || !ctx.limits.buffer_fits(c_bytes)
        || !ctx.limits.buffer_fits(out_size)
    {
        return None;
    }
    let wg_x = u32::try_from(n).ok()?.div_ceil(GEMM_WG);
    let wg_y = u32::try_from(m).ok()?.div_ceil(GEMM_WG);
    if !dispatch_2d_fits(&ctx.limits, wg_x, wg_y) {
        return None;
    }
    // a, b, c, output and read-back staging — minus whatever is already
    // resident, whose bytes the budget is counting already.
    if !ctx.budget_admits(&[
        ctx.source_admission_bytes(a, a_bytes),
        ctx.operand_admission_bytes_for(keys.weight, b_format, b_bytes),
        ctx.operand_admission_bytes(c_key, c_bytes),
        out_size,
        placement.staging_bytes(out_size),
    ]) {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let scope = ErrorScope::begin(ctx);
    let (pipeline, bgl) = match f16_pipeline {
        Some(pair) => pair,
        None => gemm_nt_pipeline(ctx),
    };

    let a_buf = ctx.operand_source("gemm_a", a, wgpu::BufferUsages::STORAGE)?;
    let b_buf = ctx.operand_buffer_typed(
        keys.weight,
        "gemm_b",
        b,
        b_format,
        wgpu::BufferUsages::STORAGE,
    )?;
    let c_buf = ctx.operand_buffer(
        c_key,
        "gemm_c",
        bytemuck::cast_slice(c_upload),
        wgpu::BufferUsages::STORAGE,
    )?;
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

    let params = GemmNtParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        c_mode: c_mode as u32,
        alpha,
        beta,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = ctx.upload_buffer(
        "gemm_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("gemm_staging", out_size)?),
        OutputPlacement::Device => None,
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm_bg"),
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
                resource: c_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_binding,
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
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
        out_len,
        out_size,
        vec![m, n],
    )
    .await
}

/// Blocking form of [`gpu_gemm_nt_async`].
///
/// Declines outright on wasm32 (see `block_on_gpu`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_gemm_nt(
    ctx: &GpuContext,
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    c: Option<&[f32]>,
    alpha: f32,
    beta: f32,
) -> Option<Vec<f32>> {
    block_on_gpu(gpu_gemm_nt_async(ctx, a, m, k, b, n, c, alpha, beta))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure, hand-rolled `A @ B^T` reference (plain triple loop, no
    /// `matrixmultiply`), checked against the numpy-verified literal in
    /// `oxionnx-ops::attention::gemm::tests::nt_matches_numpy_small`'s doc
    /// comment -- lifted, not re-derived, so the *formula* is pinned
    /// independently of both this crate's WGSL and `oxionnx-ops`'s own sgemm
    /// path.
    fn gemm_nt_host_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[j * k + p];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    #[test]
    fn gemm_nt_host_reference_matches_numpy_literal() {
        let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let got = gemm_nt_host_reference(&a, &b, 2, 3, 4);
        let want = [5.0, 14.0, 23.0, 32.0, 14.0, 50.0, 86.0, 122.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?} want {want:?}");
        }
    }

    #[test]
    fn gpu_gemm_nt_declines_zero_dims() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        assert!(gpu_gemm_nt(&ctx, &[], 0, 1, &[1.0], 1, None, 1.0, 1.0).is_none());
    }

    #[test]
    fn gpu_gemm_nt_declines_bad_c_length() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let a = vec![1.0f32; 2 * 3];
        let b = vec![1.0f32; 4 * 3];
        // C of length 2 is neither N(=4) nor M*N(=8) nor absent.
        let c = vec![0.0f32; 2];
        assert!(gpu_gemm_nt(&ctx, &a, 2, 3, &b, 4, Some(&c), 1.0, 1.0).is_none());
    }

    #[test]
    fn gpu_gemm_nt_declines_non_finite_alpha() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let a = vec![1.0f32; 2 * 3];
        let b = vec![1.0f32; 4 * 3];
        assert!(gpu_gemm_nt(&ctx, &a, 2, 3, &b, 4, None, f32::NAN, 1.0).is_none());
    }

    /// `gpu_gemm_nt` (the `block_on_gpu` wrapper) and `gpu_gemm_nt_async`
    /// (the real implementation) must dispatch the same kernel on the same
    /// input and produce identical output. `.expect` on both sides (not a
    /// bare `assert_eq!` of the `Option`s) so a decline-path regression that
    /// makes both sides silently return `None` fails this test instead of
    /// passing vacuously.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_gemm_nt_async_matches_sync() {
        let ctx = match GpuContext::try_new() {
            Some(ctx) => ctx,
            None => return,
        };
        let (m, k, n) = (4usize, 6usize, 5usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32 - 15.0) * 0.1).collect();
        let c: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();

        let sync_result = gpu_gemm_nt(&ctx, &a, m, k, &b, n, Some(&c), 1.0, 1.0)
            .expect("gpu_gemm_nt must dispatch");
        let async_result =
            pollster::block_on(gpu_gemm_nt_async(&ctx, &a, m, k, &b, n, Some(&c), 1.0, 1.0))
                .expect("gpu_gemm_nt_async must dispatch on the same input");
        assert_eq!(
            sync_result, async_result,
            "sync and async entry points must produce identical output"
        );
    }
}
