//! Parity tests for `oxionnx_gpu::shaders::gpu_resize_{bilinear_pytorch_half_pixel,
//! nearest_asymmetric}` (agent K2's standalone WGSL kernel batch).
//!
//! Skips silently when no wgpu adapter is reachable (see
//! `k2_broadcast_binary.rs`'s module docs for why).
//!
//! `cpu_resize_bilinear` below runs two separate 1-D passes (H, then W) --
//! mirroring how `oxionnx-ops::resize`'s *real* reference implementation is
//! separable -- rather than the WGSL kernel's single fused 4-neighbour
//! gather. The multiply grouping differs (a fused 2x2 blend vs two
//! sequential 1-D lerps), so this is both an independent structural check
//! and the reason parity here uses a relative, not bit-exact, tolerance.
//! `nearest_index_matches_oxionnx_ops_literal_end_to_end` drives a real GPU
//! dispatch against `oxionnx-ops::resize::tests::test_resize_nearest_2x`'s
//! literal output.
//!
//! No exact InSwapper/SCRFD resize shape was given in this wave's task
//! description (unlike Pad's explicit 1.18M-element figure), so the shapes
//! below are representative NCHW upsamples of a plausible scale, clearly
//! called out as such.

use oxionnx_gpu::shaders::{gpu_resize_bilinear_pytorch_half_pixel, gpu_resize_nearest_asymmetric};
use oxionnx_gpu::GpuContext;

fn pattern(i: usize) -> f32 {
    ((i % 23) as f32 - 11.0) * 0.25
}

/// `numpy.allclose`-style check: `|actual - expected| <= atol + rtol * |expected|`.
/// Scans the whole slice, rejects a non-finite GPU output immediately
/// (`diff`/tolerance comparisons against NaN are `false` either way, which
/// would otherwise let an all-NaN output pass silently), and on a genuine
/// tolerance violation reports both that entry and the true worst absolute
/// diff seen anywhere in the slice.
fn assert_allclose(actual: &[f32], expected: &[f32], rtol: f32, atol: f32, what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: length mismatch (gpu={}, cpu={})",
        actual.len(),
        expected.len()
    );
    let mut max_abs_diff = 0.0f32;
    let mut violation: Option<(usize, f32, f32)> = None;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite(),
            "{what}: non-finite GPU output at {i}: got {a}, expected {e}"
        );
        let tol = atol + rtol * e.abs();
        let diff = (a - e).abs();
        max_abs_diff = max_abs_diff.max(diff);
        if diff > tol && violation.is_none() {
            violation = Some((i, diff, tol));
        }
    }
    if let Some((i, diff, tol)) = violation {
        panic!(
            "{what}: mismatch at {i}: got {}, expected {} (diff {diff} > tol {tol}; \
             max abs diff over the whole slice was {max_abs_diff})",
            actual[i], expected[i]
        );
    }
}

fn coord_pytorch_half_pixel(x: usize, scale: f32, out_size: usize) -> f32 {
    if out_size > 1 {
        (x as f32 + 0.5) / scale - 0.5
    } else {
        0.0
    }
}

/// One 1-D lerp pass over axis `in_len` (with `outer` rows above it and
/// `inner` elements below it in the flattened layout), producing `out_len`.
fn lerp_axis_pass(
    data: &[f32],
    outer: usize,
    in_len: usize,
    inner: usize,
    out_len: usize,
    scale: f32,
) -> Vec<f32> {
    let max_idx = in_len as f32 - 1.0;
    let mut out = vec![0.0f32; outer * out_len * inner];
    for o in 0..outer {
        for j in 0..out_len {
            let src = coord_pytorch_half_pixel(j, scale, out_len);
            let base = src.floor();
            let ratio = (src - base).clamp(0.0, 1.0);
            let idx0 = base.clamp(0.0, max_idx) as usize;
            let idx1 = (base + 1.0).clamp(0.0, max_idx) as usize;
            for i in 0..inner {
                let v0 = data[(o * in_len + idx0) * inner + i];
                let v1 = data[(o * in_len + idx1) * inner + i];
                out[(o * out_len + j) * inner + i] = v0 * (1.0 - ratio) + v1 * ratio;
            }
        }
    }
    out
}

/// Separable bilinear resize (H pass, then W pass) -- see module docs for
/// why this is structurally independent of the WGSL kernel's fused gather.
fn cpu_resize_bilinear(data: &[f32], shape: [usize; 4], out_h: usize, out_w: usize) -> Vec<f32> {
    let [n, c, in_h, in_w] = shape;
    let scale_h = out_h as f32 / in_h as f32;
    let scale_w = out_w as f32 / in_w as f32;
    let after_h = lerp_axis_pass(data, n * c, in_h, in_w, out_h, scale_h);
    lerp_axis_pass(&after_h, n * c * out_h, in_w, 1, out_w, scale_w)
}

fn nearest_index(out_coord: usize, scale: f32, in_len: usize) -> usize {
    let src = out_coord as f32 / scale;
    let max_idx = in_len as f32 - 1.0;
    (src - 0.5).ceil().clamp(0.0, max_idx) as usize
}

fn cpu_resize_nearest(data: &[f32], shape: [usize; 4], out_h: usize, out_w: usize) -> Vec<f32> {
    let [n, c, in_h, in_w] = shape;
    let scale_h = out_h as f32 / in_h as f32;
    let scale_w = out_w as f32 / in_w as f32;
    let h_idx: Vec<usize> = (0..out_h)
        .map(|j| nearest_index(j, scale_h, in_h))
        .collect();
    let w_idx: Vec<usize> = (0..out_w)
        .map(|j| nearest_index(j, scale_w, in_w))
        .collect();
    let mut out = Vec::with_capacity(n * c * out_h * out_w);
    for ni in 0..n {
        for ci in 0..c {
            for &ih in &h_idx {
                for &iw in &w_idx {
                    out.push(data[((ni * c + ci) * in_h + ih) * in_w + iw]);
                }
            }
        }
    }
    out
}

// ── Representative InSwapper-scale bilinear upsample: 2x, [1,64,32,32] -> [1,64,64,64] ──

#[test]
fn bilinear_matches_cpu_at_representative_upsample_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 64, 32, 32];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let expected = cpu_resize_bilinear(&data, shape, 64, 64);
    let got = gpu_resize_bilinear_pytorch_half_pixel(&ctx, &data, &shape, 64, 64)
        .expect("gpu_resize_bilinear_pytorch_half_pixel must dispatch at a representative shape");
    assert_allclose(
        &got,
        &expected,
        1e-5,
        1e-5,
        "gpu_resize_bilinear (2x upsample)",
    );
}

// ── Forced 2-D dispatch grid ─────────────────────────────────────────────
// See `k2_broadcast_binary.rs`'s identically-named test for why: every shape
// above is small enough that the real device never forces `gid.y > 0` in the
// `gid.y * row_threads + gid.x` reconstruction. Lowering the *advertised*
// `max_workgroups_per_dimension` (the `w1_gpu_backend.rs` pattern) forces it
// safely on any real adapter.

#[test]
fn bilinear_matches_cpu_with_a_forced_2d_dispatch_grid() {
    let mut ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    // Same shape as `bilinear_matches_cpu_at_representative_upsample_shape`:
    // output is [1,64,64,64] = 262144 elements / 256 = 1024 workgroups needed.
    // max_workgroups_per_dimension=33 forces a 2-D grid: x=33, y=32 (both <= 33).
    ctx.limits.max_workgroups_per_dimension = 33;
    let shape = [1usize, 64, 32, 32];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let expected = cpu_resize_bilinear(&data, shape, 64, 64);
    let got = gpu_resize_bilinear_pytorch_half_pixel(&ctx, &data, &shape, 64, 64)
        .expect("gpu_resize_bilinear_pytorch_half_pixel must dispatch on a forced 2-D grid");
    assert!(
        !ctx.is_degraded(),
        "the lowered limit must produce a valid 2-D dispatch, not a device error"
    );
    assert_allclose(
        &got,
        &expected,
        1e-5,
        1e-5,
        "gpu_resize_bilinear (forced 2-D grid)",
    );
}

// ── Representative SCRFD-scale nearest FPN upsample: 2x, [1,32,20,20] -> [1,32,40,40] ──

#[test]
fn nearest_matches_cpu_at_representative_fpn_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 32, 20, 20];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let expected = cpu_resize_nearest(&data, shape, 40, 40);
    let got = gpu_resize_nearest_asymmetric(&ctx, &data, &shape, 40, 40)
        .expect("gpu_resize_nearest_asymmetric must dispatch at a representative shape");
    assert_allclose(
        &got,
        &expected,
        1e-6,
        1e-6,
        "gpu_resize_nearest (2x FPN upsample)",
    );
}

// ── Pinned against a literal from oxionnx-ops's own test suite ─────────────

#[test]
fn nearest_index_matches_oxionnx_ops_literal_end_to_end() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    // oxionnx-ops::resize::tests::test_resize_nearest_2x
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let shape = [1usize, 1, 2, 2];
    let got = gpu_resize_nearest_asymmetric(&ctx, &data, &shape, 4, 4)
        .expect("gpu_resize_nearest_asymmetric must dispatch on the literal 2x2 case");
    #[rustfmt::skip]
    let expected = vec![
        1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
        3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(got, expected);
}

// ── Degenerate: bilinear resize down to a single output pixel ──────────────
// Exercises `coord_pytorch_half_pixel`'s `out_size == 1` branch (source
// coordinate forced to 0.0, not `(0.5)/scale - 0.5`) on both axes.

#[test]
fn bilinear_matches_cpu_degenerate_output_of_one_pixel() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 2, 4, 4];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let expected = cpu_resize_bilinear(&data, shape, 1, 1);
    let got = gpu_resize_bilinear_pytorch_half_pixel(&ctx, &data, &shape, 1, 1)
        .expect("gpu_resize_bilinear_pytorch_half_pixel must dispatch down to a 1x1 output");
    assert_allclose(
        &got,
        &expected,
        1e-6,
        1e-6,
        "gpu_resize_bilinear (1x1 output)",
    );
    // The pytorch_half_pixel out_size==1 branch forces src=0.0 on both axes,
    // so a 1x1 output must equal exactly the top-left input pixel per
    // channel -- pin that explicitly, not just against the (same-formula)
    // CPU reference.
    for c in 0..2 {
        assert!((got[c] - data[c * 16]).abs() < 1e-6);
    }
}

// ── Degenerate: 1x1 spatial input ─────────────────────────────────────────

#[test]
fn nearest_matches_cpu_degenerate_1x1_input() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 3, 1, 1];
    let data = [1.0f32, 2.0, 3.0];
    let expected = cpu_resize_nearest(&data, shape, 5, 5);
    let got = gpu_resize_nearest_asymmetric(&ctx, &data, &shape, 5, 5)
        .expect("gpu_resize_nearest_asymmetric must dispatch from a 1x1 spatial input");
    assert_allclose(
        &got,
        &expected,
        1e-6,
        1e-6,
        "gpu_resize_nearest (1x1 input)",
    );
    assert!(got.iter().take(25).all(|&v| (v - 1.0).abs() < 1e-6));
}

// ── Ragged/odd shape ─────────────────────────────────────────────────────

#[test]
fn bilinear_matches_cpu_ragged_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 3, 7, 5];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let expected = cpu_resize_bilinear(&data, shape, 13, 9);
    let got = gpu_resize_bilinear_pytorch_half_pixel(&ctx, &data, &shape, 13, 9)
        .expect("gpu_resize_bilinear_pytorch_half_pixel must dispatch at a ragged shape");
    assert_allclose(&got, &expected, 1e-5, 1e-5, "gpu_resize_bilinear (ragged)");
}

// ── Decline paths ────────────────────────────────────────────────────────

#[test]
fn gpu_resize_declines_empty_axis_upsized() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let data: Vec<f32> = Vec::new();
    assert!(gpu_resize_bilinear_pytorch_half_pixel(&ctx, &data, &[1, 1, 0, 4], 8, 4).is_none());
}
