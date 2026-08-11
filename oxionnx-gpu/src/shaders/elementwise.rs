//! GPU-accelerated element-wise (unary and binary) operations.
//!
//! Every operation comes in two forms: `gpu_x_async` is the implementation, and
//! `gpu_x` is a `block_on_gpu` wrapper around it — see the crate docs for why
//! the async one is the real function and what the sync one does in a browser.

use crate::context::activation::{skips_size_threshold, GpuOutput, OutputPlacement, TensorSource};
use crate::context::GpuContext;

use super::common::{
    block_on_gpu, checked_storage_bytes, finish_output_async, plan_dispatch, ErrorScope, EwParams,
    BINARY_EW_GPU_THRESHOLD, EW_GPU_THRESHOLD, WG_SIZE,
};

/// LeakyRelu slope used when the caller does not supply one (ONNX default).
pub const DEFAULT_LEAKY_RELU_ALPHA: f32 = 0.01;

// ========================================================================
// Unary element-wise ops
// ========================================================================

/// Internal helper for element-wise GPU dispatch.
///
/// The one body behind every unary entry point, in both regimes: `input` is
/// either host bytes this dispatch uploads or an activation already on the
/// device, and `placement` decides whether the result is read back or handed
/// back as a device tensor. Neither choice reaches the shader — the same
/// pipeline runs over the same bytes with the same uniform block — so the two
/// regimes are bit-identical by construction.
pub(super) async fn gpu_elementwise_placed(
    ctx: &GpuContext,
    input: TensorSource<'_>,
    pipeline: &wgpu::ComputePipeline,
    alpha: f32,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    let len = input.len();
    // The size gate answers "is a round trip cheaper than the CPU kernel?",
    // which is not the question being asked once an operand is on the device.
    // See `context::activation::skips_size_threshold`.
    if (len < EW_GPU_THRESHOLD && !skips_size_threshold(&[input])) || ctx.is_degraded() {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    // Decline before allocating anything the device cannot hold, and before
    // planning a dispatch it cannot express.
    let out_size = checked_storage_bytes(&ctx.limits, len)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, len as u64, WG_SIZE)?;
    // Input, output and read-back staging, in that order — minus whatever this
    // dispatch will not actually allocate.
    if !ctx.budget_admits(&[
        ctx.source_admission_bytes(input, out_size),
        out_size,
        placement.staging_bytes(out_size),
    ]) {
        return None;
    }

    let scope = ErrorScope::begin(ctx);

    let input_buf = ctx.operand_source("ew_in", input, wgpu::BufferUsages::STORAGE)?;

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

    let params = EwParams {
        len: len as u32,
        alpha,
        row_threads: grid.threads_per_row,
        _pad: 0,
    };
    let params_buf = ctx.upload_buffer(
        "ew_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("ew_staging", out_size)?),
        OutputPlacement::Device => None,
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ew_bg"),
        layout: &ctx.elementwise_bind_group_layout,
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
        label: Some("ew_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ew_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(grid.x, grid.y, 1);
    }
    if let Some(staging) = &staging_buf {
        encoder.copy_buffer_to_buffer(&output_buf, 0, staging, 0, out_size);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // Surface any validation/OOM error as a decline before touching the result.
    if !scope.finish_async(ctx).await {
        return None;
    }

    finish_output_async(
        ctx,
        placement,
        staging_buf,
        output_buf,
        len,
        out_size,
        input.shape().to_vec(),
    )
    .await
}

/// Internal helper for element-wise GPU dispatch, host in and host out.
pub(super) async fn gpu_elementwise_dispatch(
    ctx: &GpuContext,
    data: &[f32],
    pipeline: &wgpu::ComputePipeline,
    alpha: f32,
) -> Option<Vec<f32>> {
    let shape = [data.len()];
    gpu_elementwise_placed(
        ctx,
        TensorSource::host(data, &shape),
        pipeline,
        alpha,
        OutputPlacement::Host,
    )
    .await?
    .into_vec()
}

/// Dispatch a unary kernel that takes no scalar parameter.
#[inline]
async fn dispatch_unary(
    ctx: &GpuContext,
    data: &[f32],
    pipeline: &wgpu::ComputePipeline,
) -> Option<Vec<f32>> {
    gpu_elementwise_dispatch(ctx, data, pipeline, 0.0).await
}

/// Declare the `gpu_x` / `gpu_x_async` pair for a parameterless unary kernel.
///
/// Every one of these is the *same* three lines around a different cached
/// pipeline; writing them out by hand was 11 opportunities to bind the wrong
/// one. The doc comment attached to the invocation lands on the async entry
/// point (the implementation), and the blocking wrapper gets a generated one
/// pointing at it.
macro_rules! unary_op {
    ($(#[$doc:meta])* $sync_name:ident, $async_name:ident, $placed_name:ident, $pipeline:ident) => {
        $(#[$doc])*
        ///
        /// Returns `None` for tensors below the element-count threshold, and on
        /// any device limit or validation failure — the caller runs the CPU
        /// operator instead.
        pub async fn $async_name(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
            dispatch_unary(ctx, data, &ctx.$pipeline).await
        }

        #[doc = concat!("[`", stringify!($async_name), "`] over an operand that may already be on the device.")]
        ///
        /// `input` is either host bytes or a run-scoped activation; `placement`
        /// says whether the result comes back to the host or stays in its
        /// device buffer. The un-placed form above is this one with
        /// `TensorSource::Host` and `OutputPlacement::Host`, so there is one
        /// body and one set of numerics.
        pub async fn $placed_name(
            ctx: &GpuContext,
            input: TensorSource<'_>,
            placement: OutputPlacement,
        ) -> Option<GpuOutput> {
            gpu_elementwise_placed(ctx, input, &ctx.$pipeline, 0.0, placement).await
        }

        #[doc = concat!("Blocking form of [`", stringify!($async_name), "`].")]
        ///
        /// Declines outright on wasm32 (see `block_on_gpu`).
        pub fn $sync_name(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
            block_on_gpu($async_name(ctx, data))
        }
    };
}

unary_op! {
    /// GPU-accelerated ReLU: `max(x, 0)`.
    gpu_relu, gpu_relu_async, gpu_relu_placed_async, relu_pipeline
}
unary_op! {
    /// GPU-accelerated Sigmoid: `1 / (1 + exp(-x))`.
    gpu_sigmoid, gpu_sigmoid_async, gpu_sigmoid_placed_async, sigmoid_pipeline
}
unary_op! {
    /// GPU-accelerated GELU approximation.
    gpu_gelu, gpu_gelu_async, gpu_gelu_placed_async, gelu_pipeline
}
unary_op! {
    /// GPU-accelerated Tanh: `tanh(x)`.
    gpu_tanh, gpu_tanh_async, gpu_tanh_placed_async, tanh_pipeline
}
unary_op! {
    /// GPU-accelerated Exp: `exp(x)`.
    gpu_exp, gpu_exp_async, gpu_exp_placed_async, exp_pipeline
}
unary_op! {
    /// GPU-accelerated Sqrt: `sqrt(x)`.
    gpu_sqrt, gpu_sqrt_async, gpu_sqrt_placed_async, sqrt_pipeline
}
unary_op! {
    /// GPU-accelerated Abs: `abs(x)`.
    gpu_abs, gpu_abs_async, gpu_abs_placed_async, abs_pipeline
}
unary_op! {
    /// GPU-accelerated Neg: `-x`.
    gpu_neg, gpu_neg_async, gpu_neg_placed_async, neg_pipeline
}
unary_op! {
    /// GPU-accelerated Log: `log(x)` (natural logarithm).
    gpu_log, gpu_log_async, gpu_log_placed_async, log_pipeline
}
unary_op! {
    /// GPU-accelerated SiLU: `x / (1 + exp(-x))`.
    gpu_silu, gpu_silu_async, gpu_silu_placed_async, silu_pipeline
}

/// GPU-accelerated LeakyRelu with the ONNX default slope (`alpha = 0.01`).
///
/// Callers that have a node with an explicit `alpha` attribute must use
/// [`gpu_leaky_relu_alpha`]; this entry point is only correct for the default.
pub fn gpu_leaky_relu(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_leaky_relu_alpha(ctx, data, DEFAULT_LEAKY_RELU_ALPHA)
}

/// Async form of [`gpu_leaky_relu`].
pub async fn gpu_leaky_relu_async(ctx: &GpuContext, data: &[f32]) -> Option<Vec<f32>> {
    gpu_leaky_relu_alpha_async(ctx, data, DEFAULT_LEAKY_RELU_ALPHA).await
}

/// GPU-accelerated LeakyRelu: `x >= 0 ? x : alpha * x`.
///
/// `alpha` is the node's `alpha` attribute (ONNX default `0.01`); it is uploaded
/// as a uniform rather than baked into the kernel, so models such as YOLOv3
/// (`alpha = 0.1`) compute the same values on GPU and CPU.
///
/// Non-finite `alpha` values are rejected (the node falls back to the CPU
/// operator, which reports the malformed attribute).
pub async fn gpu_leaky_relu_alpha_async(
    ctx: &GpuContext,
    data: &[f32],
    alpha: f32,
) -> Option<Vec<f32>> {
    if !alpha.is_finite() {
        return None;
    }
    gpu_elementwise_dispatch(ctx, data, &ctx.leaky_relu_pipeline, alpha).await
}

/// Blocking form of [`gpu_leaky_relu_alpha_async`].
pub fn gpu_leaky_relu_alpha(ctx: &GpuContext, data: &[f32], alpha: f32) -> Option<Vec<f32>> {
    block_on_gpu(gpu_leaky_relu_alpha_async(ctx, data, alpha))
}

/// [`gpu_leaky_relu_alpha_async`] over an operand that may already be on the
/// device.
pub async fn gpu_leaky_relu_placed_async(
    ctx: &GpuContext,
    input: TensorSource<'_>,
    alpha: f32,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    if !alpha.is_finite() {
        return None;
    }
    gpu_elementwise_placed(ctx, input, &ctx.leaky_relu_pipeline, alpha, placement).await
}

// ========================================================================
// Binary element-wise ops
// ========================================================================

/// Internal helper for binary element-wise GPU dispatch.
///
/// Residency-aware in the same way [`gpu_elementwise_placed`] is: either
/// operand may already be on the device, and the result may stay there.
pub(super) async fn gpu_binary_elementwise_placed(
    ctx: &GpuContext,
    a: TensorSource<'_>,
    b: TensorSource<'_>,
    pipeline: &wgpu::ComputePipeline,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    let len = a.len();
    if len != b.len() || ctx.is_degraded() {
        return None;
    }
    if len < BINARY_EW_GPU_THRESHOLD && !skips_size_threshold(&[a, b]) {
        return None;
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    // a, b and the output all get bound as storage buffers.
    let out_size = checked_storage_bytes(&ctx.limits, len)?;
    if !ctx.limits.buffer_fits(out_size) {
        return None;
    }
    let grid = plan_dispatch(&ctx.limits, len as u64, WG_SIZE)?;
    // a, b, output and read-back staging all have the same length here —
    // minus the ones this dispatch will not allocate.
    if !ctx.budget_admits(&[
        ctx.source_admission_bytes(a, out_size),
        ctx.source_admission_bytes(b, out_size),
        out_size,
        placement.staging_bytes(out_size),
    ]) {
        return None;
    }

    let scope = ErrorScope::begin(ctx);

    let a_buf = ctx.operand_source("binary_a", a, wgpu::BufferUsages::STORAGE)?;

    let b_buf = ctx.operand_source("binary_b", b, wgpu::BufferUsages::STORAGE)?;

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

    let params = EwParams {
        len: len as u32,
        alpha: 0.0,
        row_threads: grid.threads_per_row,
        _pad: 0,
    };
    let params_buf = ctx.upload_buffer(
        "binary_params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let staging_buf = match placement {
        OutputPlacement::Host => Some(ctx.staging_buffer("binary_staging", out_size)?),
        OutputPlacement::Device => None,
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("binary_bg"),
        layout: &ctx.binary_elementwise_bind_group_layout,
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
        label: Some("binary_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("binary_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
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
        len,
        out_size,
        a.shape().to_vec(),
    )
    .await
}

/// Internal helper for binary element-wise GPU dispatch, host in and host out.
pub(super) async fn gpu_binary_elementwise_dispatch(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    pipeline: &wgpu::ComputePipeline,
) -> Option<Vec<f32>> {
    let shape = [a.len()];
    gpu_binary_elementwise_placed(
        ctx,
        TensorSource::host(a, &shape),
        TensorSource::host(b, &shape),
        pipeline,
        OutputPlacement::Host,
    )
    .await?
    .into_vec()
}

/// GPU-accelerated element-wise addition: `a + b`.
///
/// Both inputs must have the same length (no broadcasting).
/// Returns `None` for tensors below the threshold or mismatched lengths.
pub async fn gpu_add_async(ctx: &GpuContext, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    gpu_binary_elementwise_dispatch(ctx, a, b, &ctx.add_pipeline).await
}

/// [`gpu_add_async`] over operands that may already be on the device.
pub async fn gpu_add_placed_async(
    ctx: &GpuContext,
    a: TensorSource<'_>,
    b: TensorSource<'_>,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    gpu_binary_elementwise_placed(ctx, a, b, &ctx.add_pipeline, placement).await
}

/// Blocking form of [`gpu_add_async`].
pub fn gpu_add(ctx: &GpuContext, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    block_on_gpu(gpu_add_async(ctx, a, b))
}

/// GPU-accelerated element-wise multiplication: `a * b`.
///
/// Both inputs must have the same length (no broadcasting).
/// Returns `None` for tensors below the threshold or mismatched lengths.
pub async fn gpu_mul_async(ctx: &GpuContext, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    gpu_binary_elementwise_dispatch(ctx, a, b, &ctx.mul_pipeline).await
}

/// [`gpu_mul_async`] over operands that may already be on the device.
pub async fn gpu_mul_placed_async(
    ctx: &GpuContext,
    a: TensorSource<'_>,
    b: TensorSource<'_>,
    placement: OutputPlacement,
) -> Option<GpuOutput> {
    gpu_binary_elementwise_placed(ctx, a, b, &ctx.mul_pipeline, placement).await
}

/// Blocking form of [`gpu_mul_async`].
pub fn gpu_mul(ctx: &GpuContext, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    block_on_gpu(gpu_mul_async(ctx, a, b))
}
