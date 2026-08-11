//! Parity tests for `oxionnx_gpu::shaders::gpu_broadcast_{add,sub,mul,div}`
//! (agent K2's standalone WGSL kernel batch).
//!
//! Every test degrades silently when no wgpu adapter is reachable
//! (`let Some(ctx) = GpuContext::try_new() else { return };`), matching the
//! convention `oxionnx-gpu/src/shaders/tests.rs` already uses (as opposed to
//! `tests/w3_gpu_kernel_parity.rs`'s loud-panic variant) -- these kernels are
//! not yet wired into session dispatch, so a missing adapter here is not a
//! regression to fail loudly about.
//!
//! The CPU reference below (`cpu_broadcast`) is *not* the same code as the
//! kernel's stride-based decode (`broadcast_binary.rs::resolve_broadcast` /
//! the WGSL `operand_offsets`): it walks explicit 4-D coordinates and zeroes
//! a coordinate per-axis whenever that operand's own dimension is `1`,
//! rather than computing a stride vector once and reprojecting a flat index
//! through it. A broadcast-math bug shared between the kernel's Rust and
//! WGSL halves (the same formula, translated twice) would not automatically
//! reproduce in this differently-structured reference.

use oxionnx_gpu::shaders::{
    gpu_broadcast_add, gpu_broadcast_div, gpu_broadcast_mul, gpu_broadcast_sub,
};
use oxionnx_gpu::GpuContext;

fn pattern(i: usize) -> f32 {
    ((i % 23) as f32 - 11.0) * 0.25 // range [-2.75, 2.75]
}

/// Strictly positive (and bounded away from 0), safe as a Div denominator.
fn positive_pattern(i: usize) -> f32 {
    ((i % 23) as f32) * 0.25 + 0.5 // range [0.5, 6.0]
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

fn pad4(shape: &[usize]) -> [usize; 4] {
    let mut out = [1usize; 4];
    let offset = 4 - shape.len();
    out[offset..].copy_from_slice(shape);
    out
}

/// Independent CPU broadcast reference -- see module docs for why this is
/// structured differently from the kernel's own stride walk.
fn cpu_broadcast(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    op: fn(f32, f32) -> f32,
) -> (Vec<f32>, [usize; 4]) {
    let a4 = pad4(a_shape);
    let b4 = pad4(b_shape);
    let out4 = [
        a4[0].max(b4[0]),
        a4[1].max(b4[1]),
        a4[2].max(b4[2]),
        a4[3].max(b4[3]),
    ];
    let idx = |shape4: [usize; 4], n: usize, c: usize, h: usize, w: usize| -> usize {
        let n = if shape4[0] == 1 { 0 } else { n };
        let c = if shape4[1] == 1 { 0 } else { c };
        let h = if shape4[2] == 1 { 0 } else { h };
        let w = if shape4[3] == 1 { 0 } else { w };
        ((n * shape4[1] + c) * shape4[2] + h) * shape4[3] + w
    };
    let mut out = Vec::with_capacity(out4[0] * out4[1] * out4[2] * out4[3]);
    for n in 0..out4[0] {
        for c in 0..out4[1] {
            for h in 0..out4[2] {
                for w in 0..out4[3] {
                    out.push(op(a[idx(a4, n, c, h, w)], b[idx(b4, n, c, h, w)]));
                }
            }
        }
    }
    (out, out4)
}

// ── Exact InSwapper shape: [1,72,128,128] op [1,72,1,1] (72*128*128 = 1_179_648,
// the same element count as the reflect-Pad outputs this wave's histogram
// calls out) ─────────────────────────────────────────────────────────────

#[test]
fn add_matches_cpu_at_inswapper_channel_broadcast_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let a_shape = [1usize, 72, 128, 128];
    let b_shape = [1usize, 72, 1, 1];
    let a: Vec<f32> = (0..a_shape.iter().product()).map(pattern).collect();
    let b: Vec<f32> = (0..72).map(|i| pattern(i + 3)).collect();
    let (expected, _) = cpu_broadcast(&a, &a_shape, &b, &b_shape, |x, y| x + y);
    let got = gpu_broadcast_add(&ctx, &a, &a_shape, &b, &b_shape)
        .expect("gpu_broadcast_add must dispatch at the InSwapper channel-broadcast shape");
    assert_allclose(
        &got,
        &expected,
        1e-5,
        1e-5,
        "gpu_broadcast_add (InSwapper shape)",
    );
}

#[test]
fn mul_matches_cpu_at_inswapper_channel_broadcast_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let a_shape = [1usize, 72, 128, 128];
    let b_shape = [1usize, 72, 1, 1];
    let a: Vec<f32> = (0..a_shape.iter().product()).map(pattern).collect();
    let b: Vec<f32> = (0..72).map(|i| pattern(i + 5)).collect();
    let (expected, _) = cpu_broadcast(&a, &a_shape, &b, &b_shape, |x, y| x * y);
    let got = gpu_broadcast_mul(&ctx, &a, &a_shape, &b, &b_shape)
        .expect("gpu_broadcast_mul must dispatch at the InSwapper channel-broadcast shape");
    assert_allclose(
        &got,
        &expected,
        1e-5,
        1e-5,
        "gpu_broadcast_mul (InSwapper shape)",
    );
}

// ── Non-commutative ops: both operand orders ────────────────────────────────
// Add/Mul are commutative, so a swapped `a_strides`/`b_strides` bug in the
// kernel would be invisible to them. Sub/Div are not: testing both
// `big op small` and `small op big` is this suite's best single check for
// that bug class.

#[test]
fn sub_matches_cpu_both_operand_orders() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let big_shape = [1usize, 8, 4, 4];
    let small_shape = [1usize, 8, 1, 1];
    let big: Vec<f32> = (0..128).map(pattern).collect();
    let small: Vec<f32> = (0..8).map(|i| pattern(i + 11)).collect();

    let (expected_fwd, _) = cpu_broadcast(&big, &big_shape, &small, &small_shape, |x, y| x - y);
    let got_fwd = gpu_broadcast_sub(&ctx, &big, &big_shape, &small, &small_shape)
        .expect("gpu_broadcast_sub must dispatch (big - small)");
    assert_allclose(
        &got_fwd,
        &expected_fwd,
        1e-5,
        1e-5,
        "gpu_broadcast_sub (big - small)",
    );

    let (expected_rev, _) = cpu_broadcast(&small, &small_shape, &big, &big_shape, |x, y| x - y);
    let got_rev = gpu_broadcast_sub(&ctx, &small, &small_shape, &big, &big_shape)
        .expect("gpu_broadcast_sub must dispatch (small - big)");
    assert_allclose(
        &got_rev,
        &expected_rev,
        1e-5,
        1e-5,
        "gpu_broadcast_sub (small - big)",
    );

    // The two orders must actually differ somewhere -- otherwise this test
    // could not have caught a swapped-operand bug.
    assert_ne!(
        got_fwd, got_rev,
        "sub is not commutative; both orders must differ"
    );
}

#[test]
fn div_matches_cpu_both_operand_orders() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let big_shape = [1usize, 8, 4, 4];
    let small_shape = [1usize, 8, 1, 1];
    let big: Vec<f32> = (0..128).map(positive_pattern).collect();
    let small: Vec<f32> = (0..8).map(|i| positive_pattern(i + 13)).collect();

    let (expected_fwd, _) = cpu_broadcast(&big, &big_shape, &small, &small_shape, |x, y| x / y);
    let got_fwd = gpu_broadcast_div(&ctx, &big, &big_shape, &small, &small_shape)
        .expect("gpu_broadcast_div must dispatch (big / small)");
    assert_allclose(
        &got_fwd,
        &expected_fwd,
        1e-5,
        1e-5,
        "gpu_broadcast_div (big / small)",
    );

    let (expected_rev, _) = cpu_broadcast(&small, &small_shape, &big, &big_shape, |x, y| x / y);
    let got_rev = gpu_broadcast_div(&ctx, &small, &small_shape, &big, &big_shape)
        .expect("gpu_broadcast_div must dispatch (small / big)");
    assert_allclose(
        &got_rev,
        &expected_rev,
        1e-5,
        1e-5,
        "gpu_broadcast_div (small / big)",
    );

    assert_ne!(
        got_fwd, got_rev,
        "div is not commutative; both orders must differ"
    );
}

// ── Scalar operand ───────────────────────────────────────────────────────

#[test]
fn add_matches_cpu_with_a_true_scalar_operand() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let a_shape = [1usize, 3, 4, 4];
    let a: Vec<f32> = (0..48).map(pattern).collect();
    let b_shape: [usize; 0] = [];
    let b = [2.5f32];
    let (expected, _) = cpu_broadcast(&a, &a_shape, &b, &b_shape, |x, y| x + y);
    let got = gpu_broadcast_add(&ctx, &a, &a_shape, &b, &b_shape)
        .expect("gpu_broadcast_add must dispatch with a rank-0 scalar operand");
    assert_allclose(
        &got,
        &expected,
        1e-5,
        1e-5,
        "gpu_broadcast_add (scalar operand)",
    );
}

// ── Ragged broadcast: only some axes broadcast, and not the leading ones ───

#[test]
fn mul_matches_cpu_ragged_partial_broadcast_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let a_shape = [1usize, 5, 7, 11];
    let b_shape = [1usize, 1, 7, 1]; // only H survives; N, C, W all broadcast
    let a: Vec<f32> = (0..5 * 7 * 11).map(pattern).collect();
    let b: Vec<f32> = (0..7).map(|i| pattern(i + 2)).collect();
    let (expected, _) = cpu_broadcast(&a, &a_shape, &b, &b_shape, |x, y| x * y);
    let got = gpu_broadcast_mul(&ctx, &a, &a_shape, &b, &b_shape)
        .expect("gpu_broadcast_mul must dispatch at a ragged partial-broadcast shape");
    assert_allclose(
        &got,
        &expected,
        1e-5,
        1e-5,
        "gpu_broadcast_mul (ragged shape)",
    );
}

// ── Degenerate: both operands a single element ──────────────────────────────

#[test]
fn add_matches_cpu_degenerate_single_element_shapes() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let shape = [1usize, 1, 1, 1];
    let a = [3.5f32];
    let b = [1.25f32];
    let got = gpu_broadcast_add(&ctx, &a, &shape, &b, &shape)
        .expect("gpu_broadcast_add must dispatch at a 1-element shape (no minimum-size gate)");
    assert_allclose(&got, &[4.75], 1e-6, 1e-6, "gpu_broadcast_add (1-element)");
}

// ── Forced 2-D dispatch grid ─────────────────────────────────────────────
// Every shape above is small enough (or, for the InSwapper case, a multiple
// of 256 that stays under 65535 workgroups) that the real device's
// `max_workgroups_per_dimension` never forces the `gid.y * row_threads +
// gid.x` reconstruction to exercise `gid.y > 0` -- `w1_gpu_backend.rs`'s
// `context_with_dimension_limit` pattern (lowering the *advertised* limit,
// which only makes the planner split earlier and is safe on every real
// adapter) is reused here to actually drive a 2-D grid.

#[test]
fn add_matches_cpu_with_a_forced_2d_dispatch_grid() {
    let mut ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    // 16384 elements / 256 threads-per-workgroup = 64 workgroups needed.
    // max_workgroups_per_dimension=9 forces a 2-D grid: x=9, y=8 (both <= 9).
    ctx.limits.max_workgroups_per_dimension = 9;
    let a_shape = [1usize, 16, 32, 32];
    let b_shape = [1usize, 16, 1, 1];
    let a: Vec<f32> = (0..16384).map(pattern).collect();
    let b: Vec<f32> = (0..16).map(|i| pattern(i + 3)).collect();
    let (expected, _) = cpu_broadcast(&a, &a_shape, &b, &b_shape, |x, y| x + y);
    let got = gpu_broadcast_add(&ctx, &a, &a_shape, &b, &b_shape)
        .expect("gpu_broadcast_add must dispatch on a forced 2-D grid, not silently decline");
    assert!(
        !ctx.is_degraded(),
        "the lowered limit must produce a valid 2-D dispatch, not a device error"
    );
    assert_allclose(
        &got,
        &expected,
        1e-5,
        1e-5,
        "gpu_broadcast_add (forced 2-D grid)",
    );
}

// ── Decline paths ────────────────────────────────────────────────────────

#[test]
fn gpu_broadcast_add_declines_incompatible_shapes() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let a = vec![1.0f32; 3];
    let b = vec![1.0f32; 5];
    assert!(
        gpu_broadcast_add(&ctx, &a, &[3], &b, &[5]).is_none(),
        "3 and 5 are not NumPy-broadcast compatible"
    );
}

#[test]
fn gpu_broadcast_add_declines_rank_above_4() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let a = vec![1.0f32; 32];
    assert!(
        gpu_broadcast_add(&ctx, &a, &[1, 1, 1, 1, 32], &a, &[1, 1, 1, 1, 32]).is_none(),
        "this kernel's stated scope is up to rank 4"
    );
}

/// [r3a] Division by zero must agree between the GPU kernel and the CPU
/// operator it stands in for.
///
/// This gap was worth closing because `Div` only became GPU-dispatchable in
/// the r3a wave (`try_gpu_dispatch_async` grew `Sub`/`Div` arms), and it is
/// the one op in that batch where the two implementations can disagree on a
/// *defined* value rather than on arithmetic: IEEE-754 gives `x/0 = ±inf` and
/// `0/0 = NaN`, whereas WGSL leaves division by zero implementation-defined.
/// A disagreement here would not be a rounding difference — it would be a
/// finite number on one path and an infinity on the other, propagating
/// through everything downstream.
///
/// Compared against `oxionnx_ops::math::broadcast::div`, the exact kernel the
/// dispatcher falls back to, rather than against a hand-written expectation.
#[test]
fn gpu_broadcast_div_agrees_with_the_cpu_kernel_on_zero_denominators() {
    let Some(ctx) = GpuContext::try_new() else {
        return;
    };
    use oxionnx_core::Tensor;

    let a = vec![1.0f32, -1.0, 0.0, 6.0];
    let b = vec![0.0f32, 0.0, 0.0, 3.0];
    let shape = [1usize, 1, 1, 4];

    let Some(gpu) = gpu_broadcast_div(&ctx, &a, &shape, &b, &shape) else {
        panic!("rank-4 div of 4 elements must not decline");
    };
    let cpu = oxionnx_ops::math::div(
        &Tensor::new(a, shape.to_vec()),
        &Tensor::new(b, shape.to_vec()),
    )
    .expect("cpu div runs");

    for (i, (g, c)) in gpu.iter().zip(cpu.data.iter()).enumerate() {
        assert_eq!(
            g.is_nan(),
            c.is_nan(),
            "element {i}: NaN-ness differs (gpu={g}, cpu={c})",
        );
        if !c.is_nan() {
            assert_eq!(
                g, c,
                "element {i}: gpu={g} cpu={c} (infinities must match in sign too)",
            );
        }
    }
}
