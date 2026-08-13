//! Integration tests for GPU shader dispatch functions.

use crate::context::GpuBufferPool;
use crate::context::GpuContext;
use crate::shaders::{
    gpu_batch_norm, gpu_broadcast_add, gpu_conv2d_implicit, gpu_gelu, gpu_gemm_nt, gpu_layer_norm,
    gpu_reduce_mean, gpu_relu, gpu_sigmoid, gpu_softmax, gpu_transpose, ConvActivation,
};

/// A context whose *placement* floors are lifted, for the kernel tests below.
///
/// Every guard that protects correctness — device limits, the memory budget,
/// dispatch planning, the degraded flag — stays exactly as it is; only the
/// "is this dispatch worth making" floors are zeroed
/// (`crate::context::tuning::GpuTuning::PARITY`).
///
/// Without this, these tests would pass by *skipping*. Their shapes are chosen
/// small enough to check by hand, and the real floors are measured: on a native
/// discrete GPU the memory-bound kernels decline at every transferring size
/// (they lose to their CPU counterparts by 1.8x-45x), and the reduction and
/// transpose floors are in the millions of elements. Each test's
/// `None => return` would then fire on every run, on every machine, and report
/// green — the exact false-green shape `w3_gpu_kernel_parity.rs` was written to
/// eliminate. The floors themselves are covered there and in
/// `tests/p1_dispatch_gating.rs`.
fn kernel_ctx() -> Option<GpuContext> {
    let mut ctx = GpuContext::try_new()?;
    ctx.set_tuning(crate::context::tuning::GpuTuning::PARITY);
    Some(ctx)
}

/// After a dispatch, the only device memory this crate still holds is what the
/// pool deliberately retains — every operand, params and staging buffer has
/// been destroyed and its bytes released.
///
/// This is the regression test for the browser stall: the old kernels dropped
/// those buffers, which on the WebGPU backend released nothing, so live bytes
/// grew by the whole working set of every node, every frame, until
/// `createBuffer` started failing. Native drops do free, so the assertion this
/// makes here is about the *accounting* — but the accounting is what the
/// budget declines on, and it is the same code on both targets.
#[test]
fn a_dispatch_leaves_only_pooled_bytes_live() {
    let Some(ctx) = kernel_ctx() else { return };
    assert_eq!(ctx.live_gpu_bytes(), 0, "a fresh context owns nothing");

    // Comfortably over `EW_GPU_THRESHOLD` so the kernel does not decline.
    let data = vec![-1.0f32; 200_000];
    for _ in 0..4 {
        let out = match gpu_relu(&ctx, &data) {
            Some(out) => out,
            None => return, // The device declined; nothing to assert.
        };
        assert_eq!(out.len(), data.len());

        let pooled = match ctx.pool.lock() {
            Ok(pool) => pool.pooled_bytes(),
            Err(_) => return,
        };
        assert_eq!(
            ctx.live_gpu_bytes(),
            pooled,
            "input, params and staging buffers must all be released after a dispatch",
        );
    }

    if let Ok(mut pool) = ctx.pool.lock() {
        pool.clear();
    }
    assert_eq!(
        ctx.live_gpu_bytes(),
        0,
        "clearing the pool must release every remaining byte",
    );
}

/// An exhausted budget is a decline, not an error: the node falls back to the
/// CPU and the context stays perfectly usable afterwards.
#[test]
fn an_exhausted_budget_declines_instead_of_allocating() {
    let Some(ctx) = kernel_ctx() else { return };
    let data = vec![0.5f32; 200_000];
    if gpu_relu(&ctx, &data).is_none() {
        return; // No dispatch on this device at all; nothing to compare against.
    }

    ctx.set_gpu_byte_budget(0);
    assert!(
        gpu_relu(&ctx, &data).is_none(),
        "a node that cannot fit the budget must decline",
    );
    assert_eq!(ctx.live_gpu_bytes(), 0, "a decline must allocate nothing");
    assert!(
        !ctx.is_degraded(),
        "a budget decline is transient and must not degrade the context",
    );

    ctx.set_gpu_byte_budget(crate::context::DEFAULT_LIVE_BYTE_BUDGET);
    assert!(
        gpu_relu(&ctx, &data).is_some(),
        "restoring the budget must restore dispatch",
    );
}

#[test]
fn test_gpu_buffer_pool_basic() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return, // skip if no GPU
    };

    let mut pool = GpuBufferPool::new(16);
    assert_eq!(pool.available_count(), 0);

    // Get a buffer (creates new since pool is empty).
    let buf = pool
        .get_buffer(
            &ctx.device,
            1024,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
        .expect("an empty pool with an unlimited budget must allocate");
    assert_eq!(pool.available_count(), 0);

    // Return it.
    pool.return_buffer(buf);
    assert_eq!(pool.available_count(), 1);

    // Clear.
    pool.clear();
    assert_eq!(pool.available_count(), 0);
}

#[test]
fn test_gpu_buffer_pool_reuse() {
    let Some(ctx) = kernel_ctx() else { return };

    let mut pool = GpuBufferPool::new(16);
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

    // Get and return a 1024-byte buffer.
    let buf = pool
        .get_buffer(&ctx.device, 1024, usage)
        .expect("allocation must succeed");
    pool.return_buffer(buf);
    assert_eq!(pool.available_count(), 1);

    // Request 1024 again — should reuse (count stays 0 after get).
    let _buf2 = pool.get_buffer(&ctx.device, 1024, usage);
    assert_eq!(pool.available_count(), 0);

    // Request something much larger — pool won't have it, creates new.
    let _buf3 = pool.get_buffer(&ctx.device, 1_000_000, usage);
    assert_eq!(pool.available_count(), 0);

    // Return multiple buffers and verify they accumulate.
    let b1 = pool
        .get_buffer(&ctx.device, 512, usage)
        .expect("allocation must succeed");
    let b2 = pool
        .get_buffer(&ctx.device, 2048, usage)
        .expect("allocation must succeed");
    let b3 = pool
        .get_buffer(&ctx.device, 4096, usage)
        .expect("allocation must succeed");
    pool.return_buffer(b1);
    pool.return_buffer(b2);
    pool.return_buffer(b3);
    assert_eq!(pool.available_count(), 3);
}

/// [F5] `GpuBufferPool::get_buffer` may hand back an idle entry up to 2x the
/// requested size (its own doc comment above, and `test_gpu_buffer_pool_reuse`
/// above already exercises the mechanism). Binding that reused buffer's
/// *actual* capacity via `as_entire_binding()` -- instead of the caller's
/// requested size -- can then exceed `max_storage_buffer_binding_size` even
/// though the request itself was validated. Every `output_buf` / `c_buf` site
/// that binds a pooled buffer in this crate now binds an explicit
/// `wgpu::BufferBinding { size: Some(requested), .. }` instead, the same
/// pattern `conv2d.rs`'s `output_binding` already used; this test is the
/// regression guard for that fix.
///
/// Reproduced against a purpose-built device with a deliberately small
/// storage-binding limit rather than the real adapter's: `GpuContext::try_new`
/// requests the adapter's own limits (see `acquire_device`), which on a
/// modern GPU can be several GiB -- reaching that boundary for real would mean
/// allocating gigabytes just to run this test. Requesting a device whose
/// `required_limits.max_storage_buffer_binding_size` is tiny (with a roomier
/// `max_buffer_size`, since only the latter gates buffer *creation*, not
/// binding) reproduces the identical wgpu validation path with two tiny
/// allocations, deterministically, on any adapter.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn a_reused_oversized_pooled_buffer_binds_at_its_requested_size() {
    let Some((device, queue)) = pollster::block_on(tiny_limited_device()) else {
        return; // No adapter on this machine.
    };

    // The device's storage-binding limit, and an entry sized at exactly 2x
    // it -- the inclusive edge of `GpuBufferPool::get_buffer`'s `<= 2 *
    // min_size` reuse window, so asking for `LIMIT` after returning a `BIG`
    // buffer reuses it deterministically rather than probabilistically.
    const LIMIT: u64 = 4096;
    const BIG: u64 = 2 * LIMIT;

    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let mut pool = GpuBufferPool::new(4);

    let big = pool
        .get_buffer(&device, BIG, usage)
        .expect("creating a buffer within max_buffer_size must succeed");
    assert_eq!(big.reserved_bytes(), BIG);
    pool.return_buffer(big);

    let reused = pool
        .get_buffer(&device, LIMIT, usage)
        .expect("the oversized idle entry must be reused for the smaller request");
    assert_eq!(
        reused.reserved_bytes(),
        BIG,
        "test is meaningless unless the pool actually handed back the bigger buffer"
    );

    // A minimal read-write storage pipeline -- correctness is not the point,
    // only whether the device's validator accepts the binding.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("f5_probe_shader"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
             @compute @workgroup_size(1)\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
             \x20   if (gid.x < arrayLength(&data)) { data[gid.x] = data[gid.x]; }\n\
             }"
            .into(),
        ),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("f5_probe_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("f5_probe_pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("f5_probe_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // The fix under test: bind exactly `LIMIT` bytes of the oversized
    // buffer -- never its full `reserved_bytes()` -- and run a real
    // dispatch through it end to end.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f5_bg_exact"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &reused,
                offset: 0,
                size: wgpu::BufferSize::new(LIMIT),
            }),
        }],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("f5_enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("f5_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    let error = pollster::block_on(scope.pop());
    assert!(
        error.is_none(),
        "binding the pool's reused buffer at its requested size must validate: {error:?}"
    );

    // Sanity check: binding the *same* reused buffer at its full (2x,
    // oversized) capacity -- what `as_entire_binding()` would do -- must be
    // exactly what this device's `max_storage_buffer_binding_size` rejects,
    // so the assertion above is not vacuously true on this adapter.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let oversized_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f5_bg_oversized"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: reused.as_entire_binding(),
        }],
    });
    drop(oversized_bind_group);
    let oversized_error = pollster::block_on(scope.pop());
    assert!(
        oversized_error.is_some(),
        "binding the reused buffer at its full 2x capacity was expected to \
         exceed this device's max_storage_buffer_binding_size; if it did not, \
         LIMIT/BIG below need adjusting"
    );
}

/// A device whose storage-binding limit is deliberately much smaller than its
/// buffer-size limit, so [`a_reused_oversized_pooled_buffer_binds_at_its_requested_size`]
/// can reach the validation boundary with byte-scale buffers instead of the
/// real adapter's (often gigabyte-scale) one. `None` when no adapter exists on
/// this machine, matching every other device-backed test in this crate.
#[cfg(not(target_arch = "wasm32"))]
async fn tiny_limited_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok()?;
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: 4096,
        max_buffer_size: 1 << 20,
        ..wgpu::Limits::default()
    };
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxionnx_f5_probe"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .ok()
}

#[test]
fn test_gpu_softmax() {
    let Some(ctx) = kernel_ctx() else { return };

    // Shape: [2, 2000] — last dim > 1000 so GPU should accept.
    let rows = 2usize;
    let cols = 2000usize;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
    let shape = vec![rows, cols];

    let result = gpu_softmax(&ctx, &data, &shape);
    let result = match result {
        Some(r) => r,
        None => return, // GPU declined
    };

    assert_eq!(result.len(), rows * cols);

    // Verify each row sums to ~1.0.
    for row in 0..rows {
        let row_sum: f32 = result[row * cols..(row + 1) * cols].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 0.01,
            "softmax row {row} sum = {row_sum}, expected ~1.0"
        );
    }

    // Verify all values are non-negative.
    for (i, &v) in result.iter().enumerate() {
        assert!(v >= 0.0, "softmax output[{i}] = {v} is negative");
    }
}

#[test]
fn test_gpu_relu() {
    let Some(ctx) = kernel_ctx() else { return };

    let len = 200_000;
    let data: Vec<f32> = (0..len)
        .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
        .collect();

    let result = gpu_relu(&ctx, &data);
    let result = match result {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), len);
    for (i, (&out, &inp)) in result.iter().zip(data.iter()).enumerate() {
        let expected = inp.max(0.0);
        assert!(
            (out - expected).abs() < 1e-5,
            "relu mismatch at {i}: got {out}, expected {expected}"
        );
    }
}

#[test]
fn test_gpu_sigmoid() {
    let Some(ctx) = kernel_ctx() else { return };

    let len = 200_000;
    let data: Vec<f32> = (0..len).map(|i| (i as f32 - 100_000.0) * 0.0001).collect();

    let result = gpu_sigmoid(&ctx, &data);
    let result = match result {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), len);
    for (i, (&out, &inp)) in result.iter().zip(data.iter()).enumerate() {
        let expected = 1.0 / (1.0 + (-inp).exp());
        assert!(
            (out - expected).abs() < 1e-4,
            "sigmoid mismatch at {i}: got {out}, expected {expected}"
        );
    }
}

#[test]
fn test_gpu_gelu() {
    let Some(ctx) = kernel_ctx() else { return };

    let len = 200_000;
    let data: Vec<f32> = (0..len).map(|i| (i as f32 - 100_000.0) * 0.00005).collect();

    let result = gpu_gelu(&ctx, &data);
    let result = match result {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), len);
    // Spot-check a few values.
    for &idx in &[0usize, 1000, 50000, 100000, 150000, 199999] {
        if idx >= len {
            continue;
        }
        let x = data[idx];
        let c = 0.797_884_6_f32;
        let inner = c * (x + 0.044715 * x * x * x);
        let expected = 0.5 * x * (1.0 + inner.tanh());
        assert!(
            (result[idx] - expected).abs() < 1e-3,
            "gelu mismatch at {idx}: got {}, expected {expected}",
            result[idx]
        );
    }
}

#[test]
fn test_gpu_layer_norm() {
    let Some(ctx) = kernel_ctx() else { return };

    // [batch=250, n_elements=256] — 64000 > threshold
    let batch = 250usize;
    let n = 256usize;
    let total = batch * n;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let scale: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002 - 0.1).collect();
    let shape = vec![batch, n];
    let eps = 1e-5_f32;

    let result = match gpu_layer_norm(&ctx, &data, &shape, &scale, &bias, eps) {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), total);

    // CPU reference for each instance
    for b in 0..batch {
        let row = &data[b * n..(b + 1) * n];
        let mean: f32 = row.iter().sum::<f32>() / n as f32;
        let var: f32 = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..n {
            let expected = (row[i] - mean) * inv_std * scale[i] + bias[i];
            let got = result[b * n + i];
            assert!(
                (got - expected).abs() < 1e-3,
                "layer_norm mismatch at batch={b}, i={i}: got {got}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_gpu_batch_norm() {
    let Some(ctx) = kernel_ctx() else { return };

    // [N=10, C=2, H=50, W=50] → 50000 elements
    let (nn, c, h, w) = (10, 2, 50, 50);
    let total = nn * c * h * w;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let shape = vec![nn, c, h, w];
    let bn_scale = vec![1.5_f32, 0.8];
    let bn_bias = vec![0.1_f32, -0.2];
    let bn_mean = vec![0.5_f32, -0.3];
    let bn_var = vec![1.0_f32, 2.0];
    let eps = 1e-5_f32;

    let result = match gpu_batch_norm(
        &ctx, &data, &shape, &bn_scale, &bn_bias, &bn_mean, &bn_var, eps,
    ) {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), total);

    let spatial = h * w;
    // CPU reference
    for idx in 0..total {
        let ch = (idx / spatial) % c;
        let x = data[idx];
        let expected = bn_scale[ch] * (x - bn_mean[ch]) / (bn_var[ch] + eps).sqrt() + bn_bias[ch];
        assert!(
            (result[idx] - expected).abs() < 1e-3,
            "batch_norm mismatch at {idx}: got {}, expected {expected}",
            result[idx]
        );
    }
}

#[test]
fn test_gpu_transpose() {
    let Some(ctx) = kernel_ctx() else { return };

    // [200, 256] → [256, 200] — 51200 > threshold
    let (rows, cols) = (200usize, 256usize);
    let total = rows * cols;
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let shape = vec![rows, cols];
    let perm = vec![1, 0];

    let result = match gpu_transpose(&ctx, &data, &shape, &perm) {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), total);

    // CPU reference: output[j*rows+i] = input[i*cols+j]
    for i in 0..rows {
        for j in 0..cols {
            let expected = data[i * cols + j];
            let got = result[j * rows + i];
            assert!(
                (got - expected).abs() < 1e-6,
                "transpose mismatch at ({i},{j}): got {got}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_gpu_reduce_mean() {
    let Some(ctx) = kernel_ctx() else { return };

    // [100000, 3] axis=1 → output [100000] (output >= 50000)
    let (d0, d1) = (100_000usize, 3usize);
    let total = d0 * d1;
    let data: Vec<f32> = (0..total).map(|i| (i % 100) as f32 * 0.1).collect();
    let shape = vec![d0, d1];

    let result = match gpu_reduce_mean(&ctx, &data, &shape, &[1], false) {
        Some(r) => r,
        None => return,
    };

    assert_eq!(result.len(), d0);

    // CPU reference
    for (i, &result_val) in result.iter().enumerate() {
        let start = i * d1;
        let sum: f32 = data[start..start + d1].iter().sum();
        let expected = sum / d1 as f32;
        assert!(
            (result_val - expected).abs() < 1e-4,
            "reduce_mean mismatch at {i}: got {result_val}, expected {expected}",
        );
    }
}

/// \[w5\] A context's compiled pipelines belong to that context, and a second
/// context compiles its own.
///
/// # What this replaces, and why the assertion changed shape
///
/// This used to assert that dropping a context purged *this thread's* entries
/// from five device-keyed thread-local caches. Those caches were the
/// second-session crash (`crate::context::pipeline_cache`): they identified a
/// device by `&cached.device == device`, which on wgpu 29 is a per-`Instance`
/// id comparison, and this crate builds one `Instance` per context — so two
/// live contexts were indistinguishable and the second was served the first's
/// `BindGroupLayout`. Purging on drop could not fix that, and the count it
/// asserted on could not even be attributed to one context.
///
/// The cache is now a field of `GpuContext`, so "dropping a context releases its
/// pipelines" is a structural fact rather than something to test. What is worth
/// pinning is the property the old arrangement got wrong: two contexts alive at
/// once each compile into their own table, and neither is empty because one of
/// them ran first.
#[test]
fn each_context_compiles_into_its_own_pipeline_cache() {
    let Some(first) = kernel_ctx() else { return };
    assert_eq!(
        first.pipelines().len(),
        0,
        "a fresh context must start with no lazily compiled pipelines",
    );

    // `gpu_broadcast_add` compiles through `kernel_support`'s helper; a
    // convolution compiles `conv2d`'s own. Both land in this context's table.
    let a = vec![1.0f32; 64 * 64];
    let b = vec![2.0f32; 64];
    let _ = gpu_broadcast_add(&first, &a, &[64, 64], &b, &[1, 64]);
    let input = oxionnx_core::Tensor::new(vec![0.25f32; 32 * 32 * 32], vec![1, 32, 32, 32]);
    let weight = oxionnx_core::Tensor::new(vec![0.05f32; 32 * 32 * 3 * 3], vec![32, 32, 3, 3]);
    let _ = gpu_conv2d_implicit(
        &first,
        &input,
        &weight,
        None,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
    );

    // [w2-f16] On an adapter with `shader-f16` this also exercises the variant
    // path, whose verdict — compiled, or refused by the driver — now lands as a
    // slot in this same per-context table instead of in a device-keyed
    // thread-local list of "devices where it failed".
    if first.set_f16_compute(true) {
        let (m, k, n) = (32usize, 512usize, 512usize);
        let ga: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
        let gb: Vec<f32> = (0..n * k).map(|i| (i % 13) as f32 * 0.02).collect();
        let _ = gpu_gemm_nt(&first, &ga, m, k, &gb, n, None, 1.0, 0.0);
        first.set_f16_compute(false);
    }

    let populated = first.pipelines().len();
    if populated == 0 {
        eprintln!("skip: the adapter declined every dispatch, so no pipeline was compiled");
        return;
    }
    // Printed rather than asserted at an exact value: how many pipelines the
    // dispatches above compile depends on what this adapter accepted (the `f16`
    // variants only exist where `SHADER_F16` does), and pinning the number would
    // make the test a statement about one machine.
    eprintln!("  {populated} pipelines compiled by the first context");

    // A second, independent context starts empty however much the first has
    // compiled — the old thread-local caches would have reported the first
    // context's entries here, and then handed them out.
    let Some(second) = kernel_ctx() else { return };
    assert_eq!(
        second.pipelines().len(),
        0,
        "a second context must not inherit the first's {populated} compiled pipelines",
    );

    // And using the second context does not disturb the first's table.
    let _ = gpu_broadcast_add(&second, &a, &[64, 64], &b, &[1, 64]);
    assert_ne!(
        second.pipelines().len(),
        0,
        "the second context compiled nothing of its own",
    );
    assert_eq!(
        first.pipelines().len(),
        populated,
        "the second context's dispatch changed the first context's cache",
    );
}
