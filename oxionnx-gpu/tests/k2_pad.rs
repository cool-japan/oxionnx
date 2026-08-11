//! Parity tests for `oxionnx_gpu::shaders::gpu_pad` (agent K2's standalone
//! WGSL kernel batch).
//!
//! Skips silently when no wgpu adapter is reachable (see
//! `k2_broadcast_binary.rs`'s module docs for why).
//!
//! `cpu_pad_reflect` below builds an explicit "bounce" lookup table per axis
//! (list `0,1,..,dim-1,dim-2,..,1` and index into it) rather than evaluating
//! `pad.rs`'s closed-form `rem_euclid` formula per output element -- a
//! structurally different derivation of the same `reflect` semantics, so a
//! sign/off-by-one bug in that formula would not automatically reappear
//! here. `reflect_matches_numpy_literal_end_to_end` additionally drives a
//! real GPU dispatch against literal numbers lifted from a numpy
//! `np.pad(..., mode='reflect')` example, independent of both.

use oxionnx_gpu::shaders::{gpu_pad, PadMode};
use oxionnx_gpu::GpuContext;

fn pattern(i: usize) -> f32 {
    ((i % 23) as f32 - 11.0) * 0.25
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

/// Per-axis reflect index map, built by listing the explicit bounce cycle
/// (`0,1,...,dim-1,dim-2,...,1`) and indexing into it with `rem_euclid` --
/// structurally different from `pad.rs`'s closed-form fold-back formula
/// (see module docs).
fn reflect_indices(dim: usize, pad_before: i64, out_len: usize) -> Vec<usize> {
    if dim <= 1 {
        return vec![0; out_len];
    }
    let period = 2 * (dim - 1);
    let mut cycle = Vec::with_capacity(period);
    cycle.extend(0..dim);
    cycle.extend((1..dim - 1).rev());
    debug_assert_eq!(cycle.len(), period);
    (0..out_len)
        .map(|out_coord| {
            let virtual_idx = out_coord as i64 - pad_before;
            let m = virtual_idx.rem_euclid(period as i64) as usize;
            cycle[m]
        })
        .collect()
}

fn cpu_pad_reflect(
    data: &[f32],
    shape: [usize; 4],
    pad_top: i64,
    pad_bottom: i64,
    pad_left: i64,
    pad_right: i64,
) -> (Vec<f32>, [usize; 4]) {
    let [n, c, h, w] = shape;
    let out_h = (h as i64 + pad_top + pad_bottom) as usize;
    let out_w = (w as i64 + pad_left + pad_right) as usize;
    let h_idx = reflect_indices(h, pad_top, out_h);
    let w_idx = reflect_indices(w, pad_left, out_w);
    let mut out = Vec::with_capacity(n * c * out_h * out_w);
    for ni in 0..n {
        for ci in 0..c {
            for &ih in &h_idx {
                for &iw in &w_idx {
                    out.push(data[((ni * c + ci) * h + ih) * w + iw]);
                }
            }
        }
    }
    (out, [n, c, out_h, out_w])
}

fn cpu_pad_constant(
    data: &[f32],
    shape: [usize; 4],
    pad_top: i64,
    pad_bottom: i64,
    pad_left: i64,
    pad_right: i64,
    cval: f32,
) -> (Vec<f32>, [usize; 4]) {
    let [n, c, h, w] = shape;
    let out_h = (h as i64 + pad_top + pad_bottom) as usize;
    let out_w = (w as i64 + pad_left + pad_right) as usize;
    let mut out = vec![cval; n * c * out_h * out_w];
    let mut o = 0usize;
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih = oh as i64 - pad_top;
                    let iw = ow as i64 - pad_left;
                    if ih >= 0 && ih < h as i64 && iw >= 0 && iw < w as i64 {
                        out[o] = data[((ni * c + ci) * h + ih as usize) * w + iw as usize];
                    }
                    o += 1;
                }
            }
        }
    }
    (out, [n, c, out_h, out_w])
}

// ── Exact InSwapper shape: [1,72,126,126] reflect-pad (1,1,1,1) -> [1,72,128,128]
// (72*128*128 = 1_179_648, the exact element count the task's histogram
// calls out for these 14 reflect-Pad nodes) ─────────────────────────────────

#[test]
fn reflect_matches_cpu_at_exact_inswapper_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 72, 126, 126];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let (expected, expected_shape) = cpu_pad_reflect(&data, shape, 1, 1, 1, 1);
    assert_eq!(expected_shape, [1, 72, 128, 128]);
    assert_eq!(expected.len(), 1_179_648);
    let got = gpu_pad(&ctx, &data, &shape, 1, 1, 1, 1, PadMode::Reflect, 0.0)
        .expect("gpu_pad(reflect) must dispatch at the exact InSwapper shape");
    assert_allclose(&got, &expected, 1e-6, "gpu_pad reflect (InSwapper shape)");
}

// ── Pinned against a literal numpy example, end-to-end through the GPU ─────
// `np.pad([1,2,3,4], 2, mode='reflect') == [3,2,1,2,3,4,3,2]` -- verified by
// hand against `oxionnx-ops::shape::sequence::pad_axes`'s reflect formula in
// this agent's design notes, reproduced here as a real dispatch rather than
// only a pure-Rust check (`pad.rs::tests::reflect_coord_host_matches_numpy_literal`
// already covers the pure formula).

#[test]
fn reflect_matches_numpy_literal_end_to_end_w_axis() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let shape = [1usize, 1, 1, 4]; // pad only W
    let got = gpu_pad(&ctx, &data, &shape, 0, 0, 2, 2, PadMode::Reflect, 0.0)
        .expect("gpu_pad(reflect) must dispatch on the literal 1x4 case");
    assert_eq!(got, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
}

#[test]
fn reflect_matches_numpy_literal_end_to_end_h_axis() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let shape = [1usize, 1, 4, 1]; // pad only H
    let got = gpu_pad(&ctx, &data, &shape, 2, 2, 0, 0, PadMode::Reflect, 0.0)
        .expect("gpu_pad(reflect) must dispatch on the literal 4x1 case");
    assert_eq!(got, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
}

// ── Constant mode ────────────────────────────────────────────────────────

#[test]
fn constant_matches_cpu_asymmetric_pads() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 3, 5, 7];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let (expected, _) = cpu_pad_constant(&data, shape, 1, 3, 2, 0, -1.0);
    let got = gpu_pad(&ctx, &data, &shape, 1, 3, 2, 0, PadMode::Constant, -1.0)
        .expect("gpu_pad(constant) must dispatch with asymmetric pads");
    assert_allclose(&got, &expected, 1e-6, "gpu_pad constant (asymmetric)");
}

#[test]
fn constant_matches_cpu_negative_pad_crop() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 2, 6, 6];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let (expected, expected_shape) = cpu_pad_constant(&data, shape, -1, -1, -2, -2, 0.0);
    assert_eq!(expected_shape, [1, 2, 4, 2]);
    let got = gpu_pad(&ctx, &data, &shape, -1, -1, -2, -2, PadMode::Constant, 0.0)
        .expect("gpu_pad(constant) must dispatch with negative (cropping) pads");
    assert_allclose(&got, &expected, 1e-6, "gpu_pad constant (crop)");
}

// ── Ragged/odd shape ─────────────────────────────────────────────────────

#[test]
fn reflect_matches_cpu_ragged_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [2usize, 3, 13, 9];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let (expected, _) = cpu_pad_reflect(&data, shape, 3, 1, 0, 5);
    let got = gpu_pad(&ctx, &data, &shape, 3, 1, 0, 5, PadMode::Reflect, 0.0)
        .expect("gpu_pad(reflect) must dispatch at a ragged shape with asymmetric pads");
    assert_allclose(&got, &expected, 1e-6, "gpu_pad reflect (ragged)");
}

// ── Forced 2-D dispatch grid ─────────────────────────────────────────────
// See `k2_broadcast_binary.rs`'s identically-named test for why: every shape
// above is small enough that the real device never forces `gid.y > 0` in the
// `gid.y * row_threads + gid.x` reconstruction. Lowering the *advertised*
// `max_workgroups_per_dimension` (the `w1_gpu_backend.rs` pattern) forces it
// safely on any real adapter.

#[test]
fn reflect_matches_cpu_with_a_forced_2d_dispatch_grid() {
    let mut ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    // Same ragged shape as `reflect_matches_cpu_ragged_shape`: output is
    // [2,3,17,14] = 1428 elements / 256 = 6 workgroups needed.
    // max_workgroups_per_dimension=3 forces a 2-D grid: x=3, y=2 (both <= 3).
    ctx.limits.max_workgroups_per_dimension = 3;
    let shape = [2usize, 3, 13, 9];
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(pattern).collect();
    let (expected, _) = cpu_pad_reflect(&data, shape, 3, 1, 0, 5);
    let got = gpu_pad(&ctx, &data, &shape, 3, 1, 0, 5, PadMode::Reflect, 0.0)
        .expect("gpu_pad(reflect) must dispatch on a forced 2-D grid, not silently decline");
    assert!(
        !ctx.is_degraded(),
        "the lowered limit must produce a valid 2-D dispatch, not a device error"
    );
    assert_allclose(&got, &expected, 1e-6, "gpu_pad reflect (forced 2-D grid)");
}

// ── Degenerate: 1x1 spatial input, single element ───────────────────────────

#[test]
fn reflect_matches_cpu_degenerate_1x1_spatial() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    // H=W=1: reflect_coord's `dim <= 1` branch (always index 0) is the only
    // code path exercised.
    let shape = [1usize, 1, 1, 1];
    let data = [7.0f32];
    let (expected, expected_shape) = cpu_pad_reflect(&data, shape, 2, 2, 2, 2);
    assert_eq!(expected_shape, [1, 1, 5, 5]);
    let got = gpu_pad(&ctx, &data, &shape, 2, 2, 2, 2, PadMode::Reflect, 0.0)
        .expect("gpu_pad(reflect) must dispatch on a 1x1 spatial input (no minimum-size gate)");
    assert_allclose(&got, &expected, 1e-6, "gpu_pad reflect (1x1 spatial)");
    assert!(got.iter().all(|&v| v == 7.0));
}

// ── Decline paths ────────────────────────────────────────────────────────

#[test]
fn gpu_pad_declines_reflect_on_empty_axis() {
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
