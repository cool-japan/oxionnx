//! Wave-2 performance harness + GPU-vs-CPU parity for the kernels touched by
//! domain W2-gpu-perf.
//!
//! These tests never assert on wall-clock time — a timing assertion on shared
//! CI hardware is a flake generator. They print the measured durations so the
//! before/after numbers in the wave report are reproducible with
//! `cargo nextest run -p oxionnx-gpu -E 'test(w2_)' --no-capture`, and they
//! *do* assert on numerical parity against a CPU reference, which is the part
//! that must never regress.
//!
//! Covered findings:
//! * a7-11 — the device is created with the adapter's real limits
//! * a7-14 — the buffer pool is bounded by bytes with LRU eviction
//! * a7-18 — softmax uses a workgroup-parallel row reduction
//! * a7-21 — `gpu_conv2d` reuses buffers and reads back once per chunk

use std::time::Instant;

use oxionnx_core::Tensor;
use oxionnx_gpu::{gpu_conv2d, gpu_softmax, GpuBufferPool, GpuContext, TrackedBuffer};

/// A context whose *placement* floors are lifted (`GpuTuning::PARITY`).
///
/// Every correctness guard stays: device limits, the live-byte budget, dispatch
/// planning, the degraded flag. Only the "is this dispatch worth making at all"
/// size floors are zeroed.
///
/// This is load-bearing rather than cosmetic. Those floors are now measured and
/// adapter-derived (`oxionnx_gpu::context::tuning`), and on a native discrete
/// GPU the memory-bound kernels decline at *every* transferring size while the
/// reduction and transpose floors sit in the millions of elements. The shapes in
/// this file are deliberately small — they exist to pin index arithmetic,
/// dispatch-grid splitting and attribute validation, not throughput — so on a
/// real context every one of them would take its `else {{ return }}` and report
/// green while verifying nothing. The floors themselves are covered by
/// `p1_dispatch_gating.rs` and `w3_gpu_kernel_parity.rs`.
fn context() -> Option<GpuContext> {
    let mut ctx = GpuContext::try_new()?;
    ctx.set_tuning(oxionnx_gpu::GpuTuning::PARITY);
    Some(ctx)
}

/// Deterministic, exactly-representable f32 pattern so GPU and CPU agree
/// bit-for-bit on the inputs.
fn pattern(i: usize) -> f32 {
    ((i % 17) as f32 - 8.0) * 0.5
}

/// CPU reference softmax over the last dimension, accumulated in f64 so the
/// reference itself is not the source of the error we measure against.
fn cpu_softmax_rows(data: &[f32], num_rows: usize, row_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_rows * row_len];
    for r in 0..num_rows {
        let row = &data[r * row_len..(r + 1) * row_len];
        let mut max_val = row[0];
        for &v in row.iter() {
            if v > max_val {
                max_val = v;
            }
        }
        let exps: Vec<f64> = row
            .iter()
            .map(|&v| (f64::from(v) - f64::from(max_val)).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        for (i, e) in exps.iter().enumerate() {
            out[r * row_len + i] = (e / sum) as f32;
        }
    }
    out
}

// ========================================================================
// a7-18 — softmax row reduction
// ========================================================================

/// The audit's attention shape: `[1, 32, 1024, 1024]` — 32_768 rows of 1024.
///
/// Before the rewrite this ran one *thread* per row, so 32_768 invocations each
/// walked ~4096 dependent, uncoalesced global loads across two dispatches.
#[test]
fn w2_softmax_attention_shape_timing_and_parity() {
    let Some(ctx) = context() else { return };

    let shape = [1usize, 32, 1024, 1024];
    let num_rows = 32 * 1024;
    let row_len = 1024;
    let total = num_rows * row_len;
    let data: Vec<f32> = (0..total).map(pattern).collect();

    // Warm up: the first dispatch pays shader/pipeline warm-up on some drivers.
    let _ = gpu_softmax(&ctx, &data, &shape);

    let start = Instant::now();
    let Some(result) = gpu_softmax(&ctx, &data, &shape) else {
        return;
    };
    let elapsed = start.elapsed();
    println!("[w2/a7-18] softmax {shape:?} ({num_rows} rows x {row_len}): {elapsed:?}");

    assert_eq!(result.len(), total);

    // Parity: spot-check whole rows against the f64 CPU reference. Checking a
    // sample of rows rather than all 32_768 keeps the test quick while still
    // covering the first row, the last row, and a stride through the middle.
    let sample_rows: Vec<usize> = {
        let mut v = vec![0usize, 1, num_rows - 1];
        v.extend((0..num_rows).step_by(4099));
        v
    };
    for r in sample_rows {
        let expected = cpu_softmax_rows(&data[r * row_len..(r + 1) * row_len], 1, row_len);
        let got = &result[r * row_len..(r + 1) * row_len];
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6,
                "softmax row {r} element {i}: got {g}, expected {e}"
            );
        }
        let sum: f64 = got.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax row {r} sums to {sum}, expected 1.0"
        );
    }
}

/// Boundary row lengths around the 256-thread workgroup: the reduction must be
/// exact whether or not `row_len` is a multiple of the workgroup size.
#[test]
fn w2_softmax_row_length_boundaries_match_the_cpu() {
    let Some(ctx) = context() else { return };

    // 1000 is the smallest accepted row (SOFTMAX_DIM_THRESHOLD) and is not a
    // multiple of 256; 1023 / 1025 straddle the 4-step boundary; 1024 and 2048
    // are exact multiples.
    for &row_len in &[1000usize, 1023, 1024, 1025, 2048, 4096] {
        for &num_rows in &[1usize, 2, 37] {
            let total = num_rows * row_len;
            let data: Vec<f32> = (0..total)
                .map(|i| pattern(i) + (i % 3) as f32 * 0.125)
                .collect();
            let Some(result) = gpu_softmax(&ctx, &data, &[num_rows, row_len]) else {
                continue;
            };
            assert_eq!(result.len(), total);
            let expected = cpu_softmax_rows(&data, num_rows, row_len);
            for (i, (&g, &e)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-6,
                    "softmax [{num_rows}, {row_len}] element {i}: got {g}, expected {e}"
                );
            }
        }
    }
}

/// A row whose values span a wide dynamic range: the max-subtraction has to
/// survive the tree reduction or the exponentials overflow to `inf`/`0`.
#[test]
fn w2_softmax_is_numerically_stable_across_the_reduction() {
    let Some(ctx) = context() else { return };

    let row_len = 2048usize;
    let num_rows = 4usize;
    // Values up to +90: `exp(90)` overflows f32 (max ~3.4e38, exp(89) ~ 4.5e38),
    // so a kernel that failed to subtract the row max would produce inf/NaN.
    let data: Vec<f32> = (0..num_rows * row_len)
        .map(|i| ((i % 181) as f32) * 0.5)
        .collect();

    let Some(result) = gpu_softmax(&ctx, &data, &[num_rows, row_len]) else {
        return;
    };
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "softmax produced {v} at {i}");
        assert!((0.0..=1.0).contains(&v), "softmax produced {v} at {i}");
    }
    let expected = cpu_softmax_rows(&data, num_rows, row_len);
    for (i, (&g, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-6,
            "stable softmax element {i}: got {g}, expected {e}"
        );
    }
}

// ========================================================================
// a7-21 — conv2d buffer reuse / batched submission
// ========================================================================

/// CPU reference Conv2D (direct convolution, f64 accumulation).
#[allow(clippy::too_many_arguments)]
fn cpu_conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&[f32]>,
    stride: usize,
    group: usize,
) -> Tensor {
    let [n, c_in, h, w] = [
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    ];
    let [c_out, c_per_group, kh, kw] = [
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    ];
    let oh = (h - kh) / stride + 1;
    let ow = (w - kw) / stride + 1;
    let c_out_per_group = c_out / group;
    let mut out = vec![0.0f32; n * c_out * oh * ow];
    for b in 0..n {
        for oc in 0..c_out {
            let g = oc / c_out_per_group;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0f64;
                    for ic in 0..c_per_group {
                        let in_c = g * c_per_group + ic;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy * stride + ky;
                                let ix = ox * stride + kx;
                                let iv = input.data[((b * c_in + in_c) * h + iy) * w + ix];
                                let wv = weight.data[((oc * c_per_group + ic) * kh + ky) * kw + kx];
                                acc += f64::from(iv) * f64::from(wv);
                            }
                        }
                    }
                    if let Some(bias) = bias {
                        acc += f64::from(bias[oc]);
                    }
                    out[((b * c_out + oc) * oh + oy) * ow + ox] = acc as f32;
                }
            }
        }
    }
    Tensor::new(out, vec![n, c_out, oh, ow])
}

/// The audit's conv case: batch 8, group 4 — 32 `(batch, group)` iterations,
/// each of which used to allocate five fresh buffers, submit, and block on a
/// full pipeline drain before the next one could start.
#[test]
fn w2_conv2d_batch_group_timing_and_parity() {
    let Some(ctx) = context() else { return };

    // [8, 32, 96, 96] * [64, 8, 3, 3], group = 4:
    //   c_per_group = 8, c_out_per_group = 16, col_rows = 72, col_cols = 94*94
    //   GEMM = 16 * 72 * 8836 = 10.18M >= GPU_THRESHOLD (10M), so the GPU accepts.
    let (n, c_in, h, w) = (8usize, 32usize, 96usize, 96usize);
    let (c_out, kh, kw, group) = (64usize, 3usize, 3usize, 4usize);
    let c_per_group = c_in / group;

    let input = Tensor::new(
        (0..n * c_in * h * w).map(|i| pattern(i) * 0.125).collect(),
        vec![n, c_in, h, w],
    );
    let weight = Tensor::new(
        (0..c_out * c_per_group * kh * kw)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.0625)
            .collect(),
        vec![c_out, c_per_group, kh, kw],
    );
    let bias_values: Vec<f32> = (0..c_out).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect();
    let bias = Tensor::new(bias_values.clone(), vec![c_out]);

    // Warm up.
    let _ = gpu_conv2d(
        &ctx,
        &input,
        &weight,
        Some(&bias),
        [1, 1],
        [0, 0, 0, 0],
        [1, 1],
        group,
    );

    let start = Instant::now();
    let Some(result) = gpu_conv2d(
        &ctx,
        &input,
        &weight,
        Some(&bias),
        [1, 1],
        [0, 0, 0, 0],
        [1, 1],
        group,
    ) else {
        return;
    };
    let elapsed = start.elapsed();
    println!(
        "[w2/a7-21] conv2d [{n},{c_in},{h},{w}] * [{c_out},{c_per_group},{kh},{kw}] group={group} \
         ({} batch/group iterations): {elapsed:?}",
        n * group
    );

    let expected = cpu_conv2d(&input, &weight, Some(&bias_values), 1, group);
    assert_eq!(result.shape, expected.shape, "conv2d output shape");
    assert_eq!(result.data.len(), expected.data.len());
    for (i, (&g, &e)) in result.data.iter().zip(expected.data.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-3,
            "conv2d element {i}: got {g}, expected {e}"
        );
    }
}

/// A single-iteration conv (batch 1, group 1) must produce the same values as
/// the multi-iteration path — this is the case that must not regress when the
/// chunking logic degenerates to one iteration per submission.
#[test]
fn w2_conv2d_single_iteration_matches_the_cpu() {
    let Some(ctx) = context() else { return };

    let (n, c_in, h, w) = (1usize, 32usize, 96usize, 96usize);
    let (c_out, kh, kw, group) = (32usize, 3usize, 3usize, 1usize);
    let input = Tensor::new(
        (0..n * c_in * h * w).map(|i| pattern(i) * 0.125).collect(),
        vec![n, c_in, h, w],
    );
    let weight = Tensor::new(
        (0..c_out * c_in * kh * kw)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03125)
            .collect(),
        vec![c_out, c_in, kh, kw],
    );

    let Some(result) = gpu_conv2d(
        &ctx,
        &input,
        &weight,
        None,
        [1, 1],
        [0, 0, 0, 0],
        [1, 1],
        group,
    ) else {
        return;
    };
    let expected = cpu_conv2d(&input, &weight, None, 1, group);
    assert_eq!(result.shape, expected.shape);
    for (i, (&g, &e)) in result.data.iter().zip(expected.data.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-3,
            "conv2d(1 iteration) element {i}: got {g}, expected {e}"
        );
    }
}

// ========================================================================
// a7-14 — buffer pool byte budget
// ========================================================================

/// The pool must not be able to pin unbounded VRAM: a segmentation network
/// whose activations run to tens of megabytes used to leave "the 64 largest
/// buffers ever produced" resident forever, with no byte cap at all.
///
/// Note the acquire-all-then-return-all shape. A get/return loop would never
/// exercise eviction: `get_buffer` reclaims the buffer just returned, so the
/// pool merely oscillates between 0 and 1 entries and any byte assertion
/// passes trivially — including on the old count-bound pool.
#[test]
fn w2_buffer_pool_is_bounded_by_bytes() {
    let Some(ctx) = context() else { return };

    let budget: u64 = 1 << 20; // 1 MiB
    let one: u64 = 256 * 1024; // four buffers' worth fits the budget exactly
    let mut pool = GpuBufferPool::with_byte_budget(64, budget);
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

    // Hold all 32 at once (8 MiB), then hand them all back.
    let buffers: Vec<TrackedBuffer> = (0..32)
        .map(|_| {
            pool.get_buffer(&ctx.device, one, usage)
                .expect("allocation must succeed")
        })
        .collect();
    assert_eq!(pool.available_count(), 0);
    for buf in buffers {
        pool.return_buffer(buf);
    }

    assert!(
        pool.pooled_bytes() <= budget,
        "pool holds {} bytes, over the {budget}-byte budget",
        pool.pooled_bytes()
    );
    assert_eq!(
        pool.available_count(),
        4,
        "8 MiB of returns against a 1 MiB budget must evict down to 4 buffers"
    );
    println!(
        "[w2/a7-14] pool after 32 x 256 KiB returns: {} buffers / {} bytes (budget {budget})",
        pool.available_count(),
        pool.pooled_bytes()
    );
}

/// Eviction must be least-recently-used, not "keep the largest".
///
/// This is the assertion that actually pins a7-14's behaviour change: the old
/// pool evicted the *smallest* entry to make room and refused to admit a
/// buffer smaller than everything it already held, so its steady state was the
/// largest buffers ever seen. Returning in descending size order makes the two
/// policies disagree completely.
#[test]
fn w2_buffer_pool_evicts_least_recently_used_not_smallest() {
    let Some(ctx) = context() else { return };

    // Budget holds exactly three of the four buffers below (3072+2048+1024).
    let mut pool = GpuBufferPool::with_byte_budget(64, 6144);
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

    let sizes = [4096u64, 3072, 2048, 1024];
    let buffers: Vec<TrackedBuffer> = sizes
        .iter()
        .map(|&s| {
            pool.get_buffer(&ctx.device, s, usage)
                .expect("allocation must succeed")
        })
        .collect();
    for buf in buffers {
        pool.return_buffer(buf);
    }

    // LRU keeps the three most recently returned: 3072, 2048, 1024 = 6144.
    // The old policy would have kept 4096, 3072, 2048.
    assert_eq!(pool.available_count(), 3);
    assert_eq!(pool.pooled_bytes(), 6144);
    // Prove 4096 is gone: asking for it must allocate rather than reuse, so the
    // pool still holds the same three afterwards.
    let fresh = pool
        .get_buffer(&ctx.device, 4096, usage)
        .expect("allocation must succeed");
    assert_eq!(fresh.size(), 4096);
    assert_eq!(
        pool.available_count(),
        3,
        "the 4096-byte buffer should have been evicted as least-recently-used"
    );
    // ... while the most recently returned one is still served from the pool.
    let reused = pool
        .get_buffer(&ctx.device, 1024, usage)
        .expect("the pooled buffer must be reused");
    assert_eq!(reused.size(), 1024);
    assert_eq!(pool.available_count(), 2);
}

/// A buffer bigger than the entire budget is dropped, not pooled: admitting it
/// would evict everything else and still leave the pool over budget.
#[test]
fn w2_buffer_pool_refuses_a_buffer_larger_than_the_budget() {
    let Some(ctx) = context() else { return };

    let mut pool = GpuBufferPool::with_byte_budget(64, 4096);
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

    let small = pool
        .get_buffer(&ctx.device, 2048, usage)
        .expect("allocation must succeed");
    pool.return_buffer(small);
    assert_eq!(pool.available_count(), 1);

    let oversized = pool
        .get_buffer(&ctx.device, 8192, usage)
        .expect("allocation must succeed");
    pool.return_buffer(oversized);
    assert_eq!(
        pool.available_count(),
        1,
        "a buffer larger than the whole budget must be dropped, not pooled"
    );
    assert_eq!(
        pool.pooled_bytes(),
        2048,
        "and must not evict the small one"
    );
}
