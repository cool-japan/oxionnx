//! Wave-1 regression tests for the wgpu backend (domain N-gpu-backend).
//!
//! Every test degrades to a no-op when no wgpu adapter is available, so the
//! suite is portable; on a machine with a Metal/Vulkan/DX adapter they exercise
//! the real kernels and compare against a CPU reference.
//!
//! Covered findings:
//! * a7-2  / a7-12 — dispatches wider than `max_compute_workgroups_per_dimension`
//! * a7-3          — buffer sizes checked against the device limits
//! * a7-4          — wgpu errors degrade to CPU instead of panicking
//! * a7-6          — LayerNorm honours `axis`
//! * a7-8          — LeakyRelu honours `alpha`
//! * a7-16         — `gpu_transpose` declines a malformed `perm`
//! * a7-17         — the buffer pool matches `BufferUsages` when reusing
//! * a7-20         — reductions decline a zero-length axis

use oxionnx_gpu::{
    gpu_add, gpu_batch_norm, gpu_layer_norm, gpu_layer_norm_axis, gpu_leaky_relu,
    gpu_leaky_relu_alpha, gpu_reduce_max, gpu_reduce_mean, gpu_reduce_min, gpu_reduce_sum,
    gpu_relu, gpu_softmax, gpu_transpose, GpuBufferPool, GpuContext, GpuLimits,
};

/// Number of elements whose 256-thread dispatch needs 65_536 workgroups, i.e.
/// one more than `max_compute_workgroups_per_dimension` (65_535) allows along a
/// single dimension. Anything at or above this must use a 2-D grid.
const OVER_ONE_DIM_LEN: usize = 65_536 * 256; // 16_777_216

/// Largest element count that still fits in a one-dimensional dispatch.
const MAX_ONE_DIM_LEN: usize = 65_535 * 256; // 16_776_960

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

/// A context whose per-dimension workgroup limit has been lowered so the
/// dispatch planner is *forced* to emit a 2-D grid for modest tensors.
///
/// Lowering the advertised limit only makes the planner split earlier — the
/// emitted dispatch stays far inside what the device can really do — so this is
/// safe on every adapter and, unlike the 16.7-million-element cases below, it
/// exercises the shaders' `gid.y * row_threads + gid.x` index reconstruction
/// deterministically and cheaply even on a backend that has no 65535 cap.
fn context_with_dimension_limit(max_workgroups_per_dimension: u32) -> Option<GpuContext> {
    let mut ctx = context()?;
    ctx.limits.max_workgroups_per_dimension = max_workgroups_per_dimension;
    Some(ctx)
}

/// Deterministic, exactly-representable f32 pattern (multiples of 0.5 in
/// [-4.0, 4.0]) so GPU and CPU agree bit-for-bit on the inputs.
fn pattern(i: usize) -> f32 {
    ((i % 17) as f32 - 8.0) * 0.5
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{what}: mismatch at {i}: got {got}, expected {want}"
        );
    }
}

// ========================================================================
// a7-8 — LeakyRelu alpha
// ========================================================================

#[test]
fn leaky_relu_honours_the_alpha_attribute() {
    let Some(ctx) = context() else { return };

    let len = 200_000;
    let data: Vec<f32> = (0..len).map(pattern).collect();
    let alpha = 0.1_f32;

    let Some(result) = gpu_leaky_relu_alpha(&ctx, &data, alpha) else {
        return; // GPU declined (e.g. below threshold on this device)
    };

    // Reference values computed with numpy float32 for the 17 distinct inputs
    // (x = -4.0, -3.5, ... 4.0) at alpha = 0.1: x < 0 -> alpha * x, else x.
    // numpy reported -0.4000000059604645, -0.3499999940395355,
    // -0.30000001192092896, -0.25, -0.20000000298023224, -0.15000000596046448,
    // -0.10000000149011612, -0.05000000074505806 for the negative half; each is
    // bit-identical to the shorter f32 literal below.
    let reference: [f32; 17] = [
        -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
        4.0,
    ];
    for (i, &got) in result.iter().enumerate() {
        let want = reference[i % 17];
        assert!(
            (got - want).abs() <= 1e-7,
            "leaky_relu(alpha=0.1) mismatch at {i}: got {got}, expected {want}"
        );
    }

    // The old kernel baked in 0.01: a negative input would come back 10x too
    // small. Prove the slope really came from the attribute.
    assert!(
        (result[0] - (-0.4)).abs() < 1e-6,
        "alpha attribute was ignored: got {} for x = -4.0",
        result[0]
    );
}

#[test]
fn leaky_relu_default_alpha_matches_the_onnx_default() {
    let Some(ctx) = context() else { return };

    let len = 200_000;
    let data: Vec<f32> = (0..len).map(pattern).collect();

    let Some(defaulted) = gpu_leaky_relu(&ctx, &data) else {
        return;
    };
    let explicit = gpu_leaky_relu_alpha(&ctx, &data, 0.01).expect("explicit alpha must also run");
    assert_close(
        &defaulted,
        &explicit,
        0.0,
        "leaky_relu default vs alpha=0.01",
    );

    let expected: Vec<f32> = data
        .iter()
        .map(|&x| if x >= 0.0 { x } else { 0.01 * x })
        .collect();
    assert_close(&defaulted, &expected, 1e-7, "leaky_relu(alpha=0.01)");
}

#[test]
fn leaky_relu_declines_a_non_finite_alpha() {
    let Some(ctx) = context() else { return };
    let data: Vec<f32> = (0..200_000).map(pattern).collect();
    assert!(gpu_leaky_relu_alpha(&ctx, &data, f32::NAN).is_none());
    assert!(gpu_leaky_relu_alpha(&ctx, &data, f32::INFINITY).is_none());
}

// ========================================================================
// a7-6 — LayerNorm axis
// ========================================================================

/// CPU reference LayerNorm over `n` trailing elements per instance.
fn cpu_layer_norm(data: &[f32], n: usize, scale: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for (instance, chunk) in data.chunks_exact(n).enumerate() {
        let mean = chunk.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let var = chunk
            .iter()
            .map(|&v| (f64::from(v) - mean) * (f64::from(v) - mean))
            .sum::<f64>()
            / n as f64;
        let inv_std = 1.0 / (var + f64::from(eps)).sqrt();
        for i in 0..n {
            out[instance * n + i] = ((f64::from(chunk[i]) - mean) * inv_std * f64::from(scale[i])
                + f64::from(bias[i])) as f32;
        }
    }
    out
}

#[test]
fn layer_norm_axis_1_normalizes_the_trailing_region() {
    let Some(ctx) = context() else { return };

    // The exact configuration from the audit: [64, 32, 64] with axis = 1, so
    // the normalized region is 32 * 64 = 2048 elements over 64 instances.
    let (d0, d1, d2) = (64usize, 32usize, 64usize);
    let shape = vec![d0, d1, d2];
    let n = d1 * d2;
    let total = d0 * n;
    let data: Vec<f32> = (0..total)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.25)
        .collect();
    let scale: Vec<f32> = (0..n).map(|i| 1.0 + (i % 13) as f32 * 0.01).collect();
    let bias: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    let eps = 1e-5_f32;

    let Some(result) = gpu_layer_norm_axis(&ctx, &data, &shape, &scale, &bias, eps, 1) else {
        return;
    };
    assert_eq!(result.len(), total);

    let expected = cpu_layer_norm(&data, n, &scale, &bias, eps);
    assert_close(&result, &expected, 2e-3, "layer_norm(axis=1)");

    // Reference values computed with numpy (float64 accumulation) for the first
    // instance: mean = -0.0577392578125, var = 49.35753653943539.
    for (idx, want) in [
        (0usize, -1.849_846_7_f32),
        (1, -1.780_904_7),
        (100, -1.786_471),
        (1000, -0.558_182_66),
        (2047, -1.424_639_7),
    ] {
        assert!(
            (result[idx] - want).abs() < 2e-3,
            "layer_norm(axis=1) numpy reference mismatch at {idx}: got {}, expected {want}",
            result[idx]
        );
    }

    // A negative axis addresses the same region.
    let negative = gpu_layer_norm_axis(&ctx, &data, &shape, &scale, &bias, eps, -2)
        .expect("axis = -2 must resolve to axis 1");
    assert_close(&negative, &result, 0.0, "layer_norm(axis=-2)");
}

#[test]
fn layer_norm_declines_when_scale_does_not_match_the_last_axis() {
    let Some(ctx) = context() else { return };

    // Same tensor, but through the last-axis entry point: scale/bias are sized
    // for axis = 1, so the GPU must decline instead of normalizing over 64
    // elements with the first 64 scale values.
    let shape = vec![64usize, 32, 64];
    let n = 32 * 64;
    let total = 64 * n;
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let scale = vec![1.0f32; n];
    let bias = vec![0.0f32; n];

    assert!(
        gpu_layer_norm(&ctx, &data, &shape, &scale, &bias, 1e-5).is_none(),
        "last-axis LayerNorm must decline a scale sized for another axis"
    );
    // Out-of-range axes decline too.
    assert!(gpu_layer_norm_axis(&ctx, &data, &shape, &scale, &bias, 1e-5, 3).is_none());
    assert!(gpu_layer_norm_axis(&ctx, &data, &shape, &scale, &bias, 1e-5, -4).is_none());
}

// ========================================================================
// a7-12 — one workgroup per LayerNorm instance, past the 65535 limit
// ========================================================================

#[test]
fn layer_norm_dispatches_more_instances_than_one_dimension_allows() {
    let Some(ctx) = context() else { return };
    if ctx.limits.max_workgroups_per_dimension > 1 << 20 {
        return; // a device that wide cannot be pushed past one dimension cheaply
    }

    // 65_536 instances of 8 elements: exactly one past the per-dimension limit
    // on every mainstream backend (65_535).
    let batch = ctx.limits.max_workgroups_per_dimension as usize + 1;
    let n = 8usize;
    let shape = vec![batch, n];
    let total = batch * n;
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let scale: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.125).collect();
    let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 1.0).collect();
    let eps = 1e-5_f32;

    let Some(result) = gpu_layer_norm(&ctx, &data, &shape, &scale, &bias, eps) else {
        return;
    };
    let expected = cpu_layer_norm(&data, n, &scale, &bias, eps);
    assert_close(&result, &expected, 1e-3, "layer_norm 2-D instance grid");
}

// ========================================================================
// a7-2 — element-wise / reduction / transpose dispatches past 65535 workgroups
// ========================================================================

/// The real-device case from the audit: a `[1, 64, 512, 512]` Relu activation
/// needs 65_536 workgroups, one past `max_compute_workgroups_per_dimension`.
///
/// The old code issued that as `dispatch_workgroups(65536, 1, 1)`, an invalid
/// dispatch and therefore a process-wide panic. Assertions concentrate on the
/// indices that the 2-D reconstruction can actually get wrong — the two sides of
/// the row boundary at `threads_per_row` — plus a sampled sweep; asserting all
/// 16.7 million elements adds no signal and a lot of wall time.
#[test]
fn relu_dispatch_past_one_workgroup_dimension_matches_the_cpu() {
    let Some(ctx) = context() else { return };
    if ctx.limits.max_workgroups_per_dimension > 65_535 {
        // This adapter expresses the dispatch in one dimension; the split path
        // is covered deterministically by the forced-limit tests below.
        return;
    }
    let len = OVER_ONE_DIM_LEN;
    assert!(len > MAX_ONE_DIM_LEN);

    let data: Vec<f32> = (0..len).map(pattern).collect();
    let Some(result) = gpu_relu(&ctx, &data) else {
        return;
    };
    assert_eq!(result.len(), len);

    // `threads_per_row` is 65_535 * 256; the element at that index is the first
    // one served by `gid.y == 1`, so it is the exact seam of the split.
    let seam = MAX_ONE_DIM_LEN;
    let mut checked: Vec<usize> = vec![0, 1, seam - 1, seam, seam + 1, len - 2, len - 1];
    checked.extend((0..len).step_by(7919));
    for i in checked {
        let want = data[i].max(0.0);
        assert!(
            (result[i] - want).abs() < 1e-6,
            "relu 2-D grid mismatch at {i}: got {}, expected {want}",
            result[i]
        );
    }

    // Guard against a false pass: relu maps every negative input to 0.0, which
    // is also what an untouched output buffer would hold, so the split is only
    // really proven by indices past the seam whose expected value is non-zero.
    let positive_past_seam: Vec<usize> = (seam..seam + 64).filter(|&i| data[i] > 0.0).collect();
    assert!(
        !positive_past_seam.is_empty(),
        "test data must contain positive values past the dispatch seam"
    );
    for i in positive_past_seam {
        assert_eq!(
            result[i], data[i],
            "relu did not write element {i}, which is served by gid.y == 1"
        );
    }
}

#[test]
fn one_dimensional_boundary_dispatch_is_still_correct() {
    let Some(ctx) = context() else { return };
    if ctx.limits.max_workgroups_per_dimension != 65_535 {
        return;
    }
    // Exactly the largest 1-D dispatch: the split must not kick in and the tail
    // guard must still cover every element.
    let len = MAX_ONE_DIM_LEN;
    let data: Vec<f32> = (0..len).map(pattern).collect();
    let Some(result) = gpu_relu(&ctx, &data) else {
        return;
    };
    assert_eq!(result.len(), len);
    assert_eq!(result[len - 1], data[len - 1].max(0.0));
    for (i, (&got, &x)) in result.iter().zip(data.iter()).enumerate().step_by(7919) {
        let want = x.max(0.0);
        assert!(
            (got - want).abs() < 1e-6,
            "relu 1-D boundary mismatch at {i}: got {got}, expected {want}"
        );
    }
}

// ------------------------------------------------------------------------
// a7-2 / a7-12 — the same split, forced on any adapter with a lowered limit.
//
// These run in milliseconds and, unlike the 16.7M-element cases above, they
// still exercise the WGSL index reconstruction on a backend (Metal, WebGPU)
// that advertises no 65535 cap.
// ------------------------------------------------------------------------

#[test]
fn forced_split_unary_elementwise_covers_every_element() {
    // 300_000 elements = 1172 workgroups; a limit of 64 forces 19 rows of 64.
    let Some(ctx) = context_with_dimension_limit(64) else {
        return;
    };
    let len = 300_000usize;
    let data: Vec<f32> = (0..len).map(pattern).collect();
    let Some(result) = gpu_relu(&ctx, &data) else {
        return;
    };
    let expected: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
    assert_close(&result, &expected, 0.0, "relu forced 2-D grid");
}

#[test]
fn forced_split_binary_elementwise_covers_every_element() {
    let Some(ctx) = context_with_dimension_limit(64) else {
        return;
    };
    let len = 300_000usize;
    let a: Vec<f32> = (0..len).map(pattern).collect();
    let b: Vec<f32> = (0..len).map(|i| pattern(i + 5)).collect();
    let Some(result) = gpu_add(&ctx, &a, &b) else {
        return;
    };
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
    assert_close(&result, &expected, 0.0, "add forced 2-D grid");
}

#[test]
fn forced_split_transpose_covers_every_element() {
    let Some(ctx) = context_with_dimension_limit(64) else {
        return;
    };
    // [200, 256] = 51_200 elements, over TRANSPOSE_GPU_THRESHOLD; 200 workgroups
    // against a limit of 64 gives a 64 x 4 grid.
    let (rows, cols) = (200usize, 256usize);
    let data: Vec<f32> = (0..rows * cols).map(pattern).collect();
    let result = gpu_transpose(&ctx, &data, &[rows, cols], &[1, 0])
        .expect("PARITY tuning lifts the size floor; the 2-D grid split must be exercised");
    assert_eq!(result.len(), rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(
                result[j * rows + i],
                data[i * cols + j],
                "transpose forced 2-D grid mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn forced_split_reduction_covers_every_output() {
    let Some(ctx) = context_with_dimension_limit(64) else {
        return;
    };
    // out_count = 60_000 (over REDUCE_GPU_THRESHOLD) needs 235 workgroups.
    let rows = 60_000usize;
    let data: Vec<f32> = (0..rows * 3).map(pattern).collect();
    let result = gpu_reduce_sum(&ctx, &data, 1, &[rows, 3])
        .expect("PARITY tuning lifts the size floor; the 2-D grid split must be exercised");
    assert_eq!(result.len(), rows);
    let expected: Vec<f32> = data.chunks_exact(3).map(|c| c[0] + c[1] + c[2]).collect();
    assert_close(&result, &expected, 1e-6, "reduce_sum forced 2-D grid");

    let mean = gpu_reduce_mean(&ctx, &data, &[rows, 3], &[1], false)
        .expect("PARITY tuning lifts the size floor; the 2-D grid split must be exercised");
    let expected_mean: Vec<f32> = expected.iter().map(|&s| s / 3.0).collect();
    assert_close(&mean, &expected_mean, 1e-6, "reduce_mean forced 2-D grid");
}

#[test]
fn forced_split_batch_norm_covers_every_element() {
    let Some(ctx) = context_with_dimension_limit(64) else {
        return;
    };
    // [10, 2, 50, 50] = 50_000 elements, over BATCH_NORM_GPU_THRESHOLD.
    let shape = [10usize, 2, 50, 50];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let scale = [1.5f32, 0.8];
    let bias = [0.1f32, -0.25];
    let mean = [0.5f32, -0.25];
    let var = [1.0f32, 2.0];
    let eps = 1e-5f32;
    let Some(result) = gpu_batch_norm(&ctx, &data, &shape, &scale, &bias, &mean, &var, eps) else {
        panic!("PARITY tuning lifts the size floor; the 2-D grid split must be exercised");
    };
    let spatial = 50 * 50;
    let expected: Vec<f32> = (0..total)
        .map(|idx| {
            let ch = (idx / spatial) % 2;
            scale[ch] * (data[idx] - mean[ch]) / (var[ch] + eps).sqrt() + bias[ch]
        })
        .collect();
    assert_close(&result, &expected, 1e-4, "batch_norm forced 2-D grid");
}

#[test]
fn forced_split_layer_norm_rebuilds_the_instance_index() {
    // LayerNorm dispatches one workgroup per instance, so a limit of 100 with
    // 250 instances forces `instance = wid.y * 100 + wid.x` to be exercised.
    let Some(ctx) = context_with_dimension_limit(100) else {
        return;
    };
    let (batch, n) = (250usize, 256usize);
    let total = batch * n;
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let scale: Vec<f32> = (0..n).map(|i| 1.0 + (i % 9) as f32 * 0.125).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
    let eps = 1e-5f32;

    let Some(result) = gpu_layer_norm(&ctx, &data, &[batch, n], &scale, &bias, eps) else {
        return;
    };
    let expected = cpu_layer_norm(&data, n, &scale, &bias, eps);
    assert_close(
        &result,
        &expected,
        1e-3,
        "layer_norm forced 2-D instance grid",
    );
}

#[test]
fn softmax_declines_a_row_count_no_grid_can_cover() {
    // [a7-18] Softmax now dispatches one workgroup per row and rebuilds the row
    // index from a 2-D grid, so a second dimension is handled rather than
    // declined (see `forced_split_softmax_rebuilds_the_row_index`). What must
    // still decline is a limit so small that even a 2-D grid cannot cover the
    // rows: 200 rows against a per-dimension limit of 2 needs 2 x 100, and
    // against a limit of 1 needs 1 x 200 — both exceed the limit on Y.
    let (rows, cols) = (200usize, 1024usize);
    let data: Vec<f32> = (0..rows * cols).map(pattern).collect();
    let Some(ctx) = context_with_dimension_limit(2) else {
        return;
    };
    assert!(
        gpu_softmax(&ctx, &data, &[rows, cols]).is_none(),
        "softmax must decline when the row grid does not fit one dimension"
    );
    // And a limit so small that even a 2-D grid cannot cover the rows.
    let Some(ctx) = context_with_dimension_limit(1) else {
        return;
    };
    assert!(gpu_softmax(&ctx, &data, &[rows, cols]).is_none());

    // With the real limit it runs and every row sums to 1.
    let Some(ctx) = context() else { return };
    let Some(result) = gpu_softmax(&ctx, &data, &[rows, cols]) else {
        return;
    };
    assert_eq!(result.len(), rows * cols);
    for row in 0..rows {
        let sum: f32 = result[row * cols..(row + 1) * cols].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "softmax row {row} sums to {sum}, expected 1.0"
        );
    }
}

#[test]
fn forced_split_softmax_rebuilds_the_row_index() {
    // [a7-18] One workgroup per row: 200 rows against a per-dimension limit of
    // 16 forces a 16 x 13 grid, so `row = wid.y * wg_per_row + wid.x` has to be
    // reconstructed correctly or rows past the first 16 come back untouched.
    let Some(ctx) = context_with_dimension_limit(16) else {
        return;
    };
    let (rows, cols) = (200usize, 1024usize);
    let data: Vec<f32> = (0..rows * cols).map(pattern).collect();
    let Some(result) = gpu_softmax(&ctx, &data, &[rows, cols]) else {
        return;
    };
    assert_eq!(result.len(), rows * cols);

    // Every row — including the ones served by wid.y > 0 — must be a real
    // softmax, not an untouched buffer.
    for row in 0..rows {
        let slice = &result[row * cols..(row + 1) * cols];
        let sum: f64 = slice.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax 2-D grid row {row} sums to {sum}, expected 1.0"
        );
        for (i, &v) in slice.iter().enumerate() {
            assert!(
                v > 0.0 && v.is_finite(),
                "softmax 2-D grid row {row} element {i} = {v}"
            );
        }
    }

    // The unlimited context must agree element-for-element with the split one.
    let Some(ctx_full) = context() else { return };
    let Some(reference) = gpu_softmax(&ctx_full, &data, &[rows, cols]) else {
        return;
    };
    assert_close(&result, &reference, 1e-7, "softmax 2-D grid vs 1-D grid");
}

#[test]
fn dispatch_declines_when_even_a_two_dimensional_grid_is_too_small() {
    // A limit of 1 leaves room for 256 threads total, far short of the 300_000
    // this tensor needs — the planner must decline, not truncate the work.
    let Some(ctx) = context_with_dimension_limit(1) else {
        return;
    };
    let data: Vec<f32> = (0..300_000).map(pattern).collect();
    assert!(gpu_relu(&ctx, &data).is_none());
    assert!(gpu_add(&ctx, &data, &data).is_none());
    assert!(gpu_transpose(&ctx, &data, &[600, 500], &[1, 0]).is_none());
    assert!(gpu_reduce_sum(&ctx, &data, 1, &[150_000, 2]).is_none());
}

// ========================================================================
// a7-16 — malformed Transpose perm
// ========================================================================

#[test]
fn transpose_declines_a_malformed_perm_instead_of_panicking() {
    // PARITY, so every `is_none()` below is the perm check answering rather
    // than the size floor short-circuiting ahead of it.
    let Some(ctx) = context() else {
        return;
    };

    // [250, 256] = 64_000 elements, comfortably over TRANSPOSE_GPU_THRESHOLD.
    let (rows, cols) = (250usize, 256usize);
    let shape = vec![rows, cols];
    let data: Vec<f32> = (0..rows * cols).map(pattern).collect();

    // The audit case: Transpose(perm=[0, 5]) on a rank-2 tensor.
    assert!(gpu_transpose(&ctx, &data, &shape, &[0, 5]).is_none());
    // A negative i64 attribute reaches us as a huge usize.
    assert!(gpu_transpose(&ctx, &data, &shape, &[0, usize::MAX]).is_none());
    // Repeated entries are not a permutation either.
    assert!(gpu_transpose(&ctx, &data, &shape, &[1, 1]).is_none());
    assert!(gpu_transpose(&ctx, &data, &shape, &[0, 0]).is_none());
    // Wrong length.
    assert!(gpu_transpose(&ctx, &data, &shape, &[0]).is_none());
    assert!(gpu_transpose(&ctx, &data, &shape, &[0, 1, 2]).is_none());
    // A zero dimension would make an output stride zero (division by zero in
    // the kernel), so it declines as well.
    assert!(gpu_transpose(&ctx, &data, &[rows, 0], &[1, 0]).is_none());

    // The valid permutation still works.
    let result = gpu_transpose(&ctx, &data, &shape, &[1, 0]);
    if let Some(result) = result {
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(result[j * rows + i], data[i * cols + j]);
            }
        }
    }
}

// ========================================================================
// a7-20 — zero-length reduced axis
// ========================================================================

#[test]
fn reductions_decline_a_zero_length_axis() {
    let Some(ctx) = context() else { return };

    // shape [100_000, 0], axis = 1: out_count = 100_000 clears the reduction
    // threshold while the reduced axis is empty. The kernels would read out of
    // range and divide by zero; the CPU path implements the ONNX identity rules.
    let shape = [100_000usize, 0];
    let data: Vec<f32> = Vec::new();
    assert!(gpu_reduce_sum(&ctx, &data, 1, &shape).is_none());
    assert!(gpu_reduce_max(&ctx, &data, 1, &shape).is_none());
    assert!(gpu_reduce_min(&ctx, &data, 1, &shape).is_none());
    assert!(gpu_reduce_mean(&ctx, &data, &shape, &[1], false).is_none());

    // A zero-length non-reduced dimension is rejected as well: `outer`/`inner`
    // used to be coerced to 1, which silently reinterpreted the buffer.
    let shape = [0usize, 100_000];
    assert!(gpu_reduce_sum(&ctx, &data, 1, &shape).is_none());
    assert!(gpu_reduce_mean(&ctx, &data, &shape, &[1], false).is_none());

    // Out-of-range axis still declines.
    assert!(gpu_reduce_sum(&ctx, &data, 5, &[100_000, 3]).is_none());

    // Sanity: the ordinary case still runs.
    let rows = 60_000usize;
    let data: Vec<f32> = (0..rows * 3).map(pattern).collect();
    if let Some(result) = gpu_reduce_sum(&ctx, &data, 1, &[rows, 3]) {
        assert_eq!(result.len(), rows);
        for (i, &got) in result.iter().enumerate().take(64) {
            let want = data[i * 3] + data[i * 3 + 1] + data[i * 3 + 2];
            assert!(
                (got - want).abs() < 1e-6,
                "reduce_sum at {i}: {got} vs {want}"
            );
        }
    }
}

// ========================================================================
// a7-17 — buffer pool usage matching
// ========================================================================

#[test]
fn buffer_pool_only_reuses_buffers_with_compatible_usage() {
    let Some(ctx) = context() else { return };

    let mut pool = GpuBufferPool::new(16);
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

    let buf = pool
        .get_buffer(&ctx.device, 4096, storage)
        .expect("allocation must succeed");
    assert!(buf.usage().contains(storage));
    pool.return_buffer(buf);
    assert_eq!(pool.available_count(), 1);

    // A request that needs COPY_DST must NOT get the STORAGE|COPY_SRC buffer.
    let needs_copy_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let fresh = pool
        .get_buffer(&ctx.device, 4096, needs_copy_dst)
        .expect("allocation must succeed");
    assert!(
        fresh.usage().contains(needs_copy_dst),
        "pool handed back a buffer missing the requested usage flags"
    );
    assert_eq!(
        pool.available_count(),
        1,
        "the incompatible buffer must stay in the pool"
    );

    // The original request is still served from the pool.
    let reused = pool
        .get_buffer(&ctx.device, 4096, storage)
        .expect("the pooled buffer must be reused");
    assert!(reused.usage().contains(storage));
    assert_eq!(pool.available_count(), 0);

    // A superset request is compatible with a superset buffer.
    pool.return_buffer(fresh);
    let subset = pool
        .get_buffer(&ctx.device, 4096, wgpu::BufferUsages::STORAGE)
        .expect("the pooled buffer must be reused");
    assert!(subset.usage().contains(wgpu::BufferUsages::STORAGE));
    assert_eq!(pool.available_count(), 0);
}

// ========================================================================
// a7-3 / a7-4 — limits and error handling
// ========================================================================

#[test]
fn gpu_limits_reflect_the_real_device() {
    let Some(ctx) = context() else { return };
    let raw = ctx.device.limits();
    let cached = GpuLimits {
        max_storage_buffer_binding_size: raw.max_storage_buffer_binding_size,
        max_buffer_size: raw.max_buffer_size,
        max_workgroups_per_dimension: raw.max_compute_workgroups_per_dimension,
    };
    assert_eq!(ctx.limits, cached);
    // Requesting the adapter's limits must never give us less than the
    // conservative defaults the crate used to hard-code.
    let defaults = wgpu::Limits::default();
    assert!(ctx.limits.max_storage_buffer_binding_size >= defaults.max_storage_buffer_binding_size);
    assert!(ctx.limits.max_buffer_size >= defaults.max_buffer_size);
    // Nothing that fits the limits may overflow the checked helpers.
    if let Some(over) = ctx.limits.max_storage_buffer_binding_size.checked_add(1) {
        assert!(!ctx.limits.storage_fits(over));
    }
    if let Some(over) = ctx.limits.max_buffer_size.checked_add(1) {
        assert!(!ctx.limits.buffer_fits(over));
    }
}

#[test]
fn a_device_error_degrades_to_cpu_instead_of_panicking() {
    let Some(ctx) = context() else { return };
    assert!(!ctx.is_degraded());
    assert!(ctx.last_error().is_none());

    // Ask for a buffer far beyond `max_buffer_size` outside any error scope.
    // wgpu's default behaviour for an unhandled device error is to panic; the
    // installed handler must turn it into a recorded, recoverable failure.
    let oversized = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("w1_oversized"),
        size: ctx.limits.max_buffer_size.saturating_add(4096),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    drop(oversized);

    assert!(
        ctx.is_degraded(),
        "an unhandled wgpu error must mark the context degraded"
    );
    assert!(ctx.last_error().is_some(), "the error must be recorded");

    // Every entry point now declines so the session keeps running on the CPU.
    let data: Vec<f32> = (0..200_000).map(pattern).collect();
    assert!(gpu_relu(&ctx, &data).is_none());
    assert!(gpu_leaky_relu_alpha(&ctx, &data, 0.1).is_none());
    assert!(gpu_add(&ctx, &data, &data).is_none());
    assert!(gpu_reduce_sum(&ctx, &data, 1, &[100_000, 2]).is_none());
    assert!(gpu_transpose(&ctx, &data, &[400, 500], &[1, 0]).is_none());
    assert!(gpu_layer_norm(
        &ctx,
        &data,
        &[1000, 200],
        &vec![1.0; 200],
        &vec![0.0; 200],
        1e-5
    )
    .is_none());
}
