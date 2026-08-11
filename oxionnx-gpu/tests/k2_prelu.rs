//! Parity tests for `oxionnx_gpu::shaders::gpu_prelu` (agent K2's standalone
//! WGSL kernel batch).
//!
//! Skips silently when no wgpu adapter is reachable, matching
//! `oxionnx-gpu/src/shaders/tests.rs`'s convention (see
//! `k2_broadcast_binary.rs`'s module docs for why that convention, not
//! `w3_gpu_kernel_parity.rs`'s loud-panic one, applies to this wave's
//! not-yet-integrated kernels).
//!
//! `cpu_prelu` below is a direct `(n, c, spatial)` nested loop -- no
//! division or modulo -- independent of the WGSL kernel's
//! `(idx / spatial) % channels` flat-index decode and of
//! `prelu.rs`'s own inline pure-math tests.

use oxionnx_gpu::shaders::gpu_prelu;
use oxionnx_gpu::GpuContext;

fn pattern(i: usize) -> f32 {
    ((i % 23) as f32 - 11.0) * 0.25 // range [-2.75, 2.75], crosses zero
}

fn assert_allclose(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: length mismatch (gpu={}, cpu={})",
        actual.len(),
        expected.len()
    );
    let mut worst = (0usize, 0.0f32);
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        // NaN would otherwise pass every comparison below silently.
        assert!(
            a.is_finite(),
            "{what}: non-finite GPU output at {i}: got {a}, expected {e}"
        );
        let diff = (a - e).abs();
        if diff > worst.1 {
            worst = (i, diff);
        }
    }
    assert!(
        worst.1 <= tol,
        "{what}: max abs diff {} at index {} (gpu={}, cpu={}) exceeds tolerance {tol}",
        worst.1,
        worst.0,
        actual[worst.0],
        expected[worst.0],
    );
}

/// Direct nested-loop PRelu reference over `[N, C, spatial...]`, structurally
/// independent of the kernel's flat-index decode (see module docs).
fn cpu_prelu(data: &[f32], shape: &[usize], slope: &[f32]) -> Vec<f32> {
    let n = shape[0];
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let mut out = vec![0.0f32; data.len()];
    for ni in 0..n {
        for ci in 0..channels {
            let alpha = if slope.len() == 1 {
                slope[0]
            } else {
                slope[ci]
            };
            for si in 0..spatial {
                let idx = (ni * channels + ci) * spatial + si;
                let x = data[idx];
                out[idx] = if x >= 0.0 { x } else { alpha * x };
            }
        }
    }
    out
}

// ── Representative ArcFace-r50-scale activation: [1,64,56,56], per-channel slope ──

#[test]
fn prelu_matches_cpu_at_arcface_scale_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 64, 56, 56];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let slope: Vec<f32> = (0..64).map(|c| 0.01 + (c as f32) * 0.003).collect();
    let expected = cpu_prelu(&data, &shape, &slope);
    let got = gpu_prelu(&ctx, &data, &shape, &slope)
        .expect("gpu_prelu must dispatch at the ArcFace-scale shape");
    assert_allclose(
        &got,
        &expected,
        1e-6,
        "gpu_prelu (ArcFace-scale, per-channel)",
    );
}

// ── Ragged/odd shape, rank-2 (no spatial dims) ──────────────────────────────

#[test]
fn prelu_matches_cpu_rank2_no_spatial_dims() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [5usize, 7];
    let data: Vec<f32> = (0..35).map(pattern).collect();
    let slope: Vec<f32> = (0..7).map(|c| 0.05 * (c as f32 + 1.0)).collect();
    let expected = cpu_prelu(&data, &shape, &slope);
    let got = gpu_prelu(&ctx, &data, &shape, &slope)
        .expect("gpu_prelu must dispatch on a rank-2 [N,C] shape");
    assert_allclose(&got, &expected, 1e-6, "gpu_prelu (rank-2)");
}

// ── Odd, non-power-of-two spatial shape ─────────────────────────────────────

#[test]
fn prelu_matches_cpu_ragged_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [2usize, 5, 7, 3];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let slope: Vec<f32> = (0..5).map(|c| 0.02 * (c as f32 + 1.0)).collect();
    let expected = cpu_prelu(&data, &shape, &slope);
    let got = gpu_prelu(&ctx, &data, &shape, &slope)
        .expect("gpu_prelu must dispatch at a ragged (odd) shape");
    assert_allclose(&got, &expected, 1e-6, "gpu_prelu (ragged)");
}

// ── Scalar slope (length 1) broadcast across every channel ─────────────────

#[test]
fn prelu_matches_cpu_scalar_slope() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 4, 3, 3];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let slope = [0.2f32];
    let expected = cpu_prelu(&data, &shape, &slope);
    let got = gpu_prelu(&ctx, &data, &shape, &slope)
        .expect("gpu_prelu must dispatch with a scalar (length-1) slope");
    assert_allclose(&got, &expected, 1e-6, "gpu_prelu (scalar slope)");
}

// ── Degenerate: single element ───────────────────────────────────────────

#[test]
fn prelu_matches_cpu_degenerate_single_element() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 1, 1, 1];
    for &(x, alpha, want) in &[(3.0f32, 0.3f32, 3.0f32), (-3.0f32, 0.3f32, -0.9f32)] {
        let data = [x];
        let slope = [alpha];
        let got = gpu_prelu(&ctx, &data, &shape, &slope)
            .expect("gpu_prelu must dispatch at a 1-element shape (no minimum-size gate)");
        assert_allclose(&got, &[want], 1e-6, "gpu_prelu (1-element)");
    }
}

// ── Forced 2-D dispatch grid ─────────────────────────────────────────────
// See `k2_broadcast_binary.rs`'s identically-named test for why: every shape
// above is small enough that the real device never forces `gid.y > 0` in the
// `gid.y * row_threads + gid.x` reconstruction. Lowering the *advertised*
// `max_workgroups_per_dimension` (the `w1_gpu_backend.rs` pattern) forces it
// safely on any real adapter.

#[test]
fn prelu_matches_cpu_with_a_forced_2d_dispatch_grid() {
    let mut ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    // 200704 elements / 256 = 784 workgroups needed.
    // max_workgroups_per_dimension=32 forces a 2-D grid: x=32, y=25 (both <= 32).
    ctx.limits.max_workgroups_per_dimension = 32;
    let shape = [1usize, 64, 56, 56];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let slope: Vec<f32> = (0..64).map(|c| 0.01 + (c as f32) * 0.003).collect();
    let expected = cpu_prelu(&data, &shape, &slope);
    let got = gpu_prelu(&ctx, &data, &shape, &slope)
        .expect("gpu_prelu must dispatch on a forced 2-D grid, not silently decline");
    assert!(
        !ctx.is_degraded(),
        "the lowered limit must produce a valid 2-D dispatch, not a device error"
    );
    assert_allclose(&got, &expected, 1e-6, "gpu_prelu (forced 2-D grid)");
}

// ── Decline paths ────────────────────────────────────────────────────────

#[test]
fn gpu_prelu_declines_mismatched_slope_length() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let data = vec![1.0f32; 12];
    let slope = vec![0.1f32; 4]; // shape has C=3, slope has 4: neither 1 nor C
    assert!(gpu_prelu(&ctx, &data, &[1, 3, 2, 2], &slope).is_none());
}
