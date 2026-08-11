//! Parity tests for `oxionnx_gpu::shaders::gpu_gemm_nt` (agent K2's
//! standalone WGSL kernel batch).
//!
//! Skips silently when no wgpu adapter is reachable (see
//! `k2_broadcast_binary.rs`'s module docs for why).
//!
//! `cpu_gemm_nt` below accumulates in `f64` with an `i-p-j` loop order
//! (`p`, the reduction axis, in the middle) -- different from both the WGSL
//! kernel's per-output-element `i-j` dispatch with an inner `p` loop, and
//! from `gemm.rs`'s own inline `gemm_nt_host_reference` (`i-j-p`, `f32`) --
//! so a bug shared by those two would not automatically reappear here too.

use oxionnx_gpu::shaders::gpu_gemm_nt;
use oxionnx_gpu::GpuContext;

fn pattern(i: usize) -> f32 {
    ((i % 23) as f32 - 11.0) * 0.1 // range [-1.1, 1.1], keeps K=512 accumulations bounded
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

/// `out = alpha * A @ B^T + beta * C`, `f64` accumulation, `i-p-j` loop
/// order -- see module docs for why this is structurally independent of
/// both the WGSL kernel and `gemm.rs`'s own inline reference.
#[allow(clippy::too_many_arguments)]
fn cpu_gemm_nt(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    c: Option<&[f32]>,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut acc = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = f64::from(a[i * k + p]);
            if av == 0.0 {
                continue;
            }
            for j in 0..n {
                acc[i * n + j] += av * f64::from(b[j * k + p]);
            }
        }
    }
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let cval: f64 = match c {
                None => 0.0,
                Some(cd) if cd.len() == n => f64::from(cd[j]),
                Some(cd) => f64::from(cd[i * n + j]),
            };
            out[i * n + j] = (f64::from(alpha) * acc[i * n + j] + f64::from(beta) * cval) as f32;
        }
    }
    out
}

// ── Pinned against a numpy-verified literal (no C) ──────────────────────────
// oxionnx-ops::attention::gemm::tests::nt_matches_numpy_small's doc comment:
// A=[2,3] (0..6), B=[4,3] (0..12) -> A @ B^T = [5,14,23,32,14,50,86,122].

#[test]
fn gemm_nt_matches_numpy_literal() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let got = gpu_gemm_nt(&ctx, &a, 2, 3, &b, 4, None, 1.0, 1.0)
        .expect("gpu_gemm_nt must dispatch on the literal 2x3x4 case");
    let expected = [5.0f32, 14.0, 23.0, 32.0, 14.0, 50.0, 86.0, 122.0];
    assert_allclose(&got, &expected, 1e-5, 1e-4, "gpu_gemm_nt (numpy literal)");
}

// ── Exact task-scope shapes: alpha=1, beta=1, transB=1, K=512, N=2048,
// M across the full stated 1..=64 range, C a length-N row-broadcast bias
// (InSwapper's AdaIN heads) ─────────────────────────────────────────────────

#[test]
fn gemm_nt_matches_cpu_at_m1_k512_n2048_adain_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let (m, k, n) = (1usize, 512usize, 2048usize);
    let a: Vec<f32> = (0..m * k).map(pattern).collect();
    let b: Vec<f32> = (0..n * k).map(|i| pattern(i + 3)).collect();
    let c: Vec<f32> = (0..n).map(|i| pattern(i + 7) * 0.1).collect();
    let expected = cpu_gemm_nt(&a, &b, m, k, n, Some(&c), 1.0, 1.0);
    let got = gpu_gemm_nt(&ctx, &a, m, k, &b, n, Some(&c), 1.0, 1.0)
        .expect("gpu_gemm_nt must dispatch at the M=1 AdaIN shape");
    assert_allclose(
        &got,
        &expected,
        1e-4,
        1e-3,
        "gpu_gemm_nt (M=1, K=512, N=2048)",
    );
}

#[test]
fn gemm_nt_matches_cpu_at_m64_k512_n2048_adain_shape() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let (m, k, n) = (64usize, 512usize, 2048usize);
    let a: Vec<f32> = (0..m * k).map(pattern).collect();
    let b: Vec<f32> = (0..n * k).map(|i| pattern(i + 3)).collect();
    let c: Vec<f32> = (0..n).map(|i| pattern(i + 7) * 0.1).collect();
    let expected = cpu_gemm_nt(&a, &b, m, k, n, Some(&c), 1.0, 1.0);
    let got = gpu_gemm_nt(&ctx, &a, m, k, &b, n, Some(&c), 1.0, 1.0)
        .expect("gpu_gemm_nt must dispatch at the M=64 (top of stated range) AdaIN shape");
    assert_allclose(
        &got,
        &expected,
        1e-4,
        1e-3,
        "gpu_gemm_nt (M=64, K=512, N=2048)",
    );
}

// ── Non-default alpha/beta, full [M,N] C (no broadcast) -- proves alpha/beta
// are real uniforms, not hardcoded to the task's stated alpha=1/beta=1 ──────

#[test]
fn gemm_nt_matches_cpu_with_non_default_alpha_beta_and_full_c() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let (m, k, n) = (4usize, 6usize, 5usize);
    let a: Vec<f32> = (0..m * k).map(pattern).collect();
    let b: Vec<f32> = (0..n * k).map(|i| pattern(i + 3)).collect();
    let c: Vec<f32> = (0..m * n).map(|i| pattern(i + 9)).collect();
    let (alpha, beta) = (0.5f32, 2.0f32);
    let expected = cpu_gemm_nt(&a, &b, m, k, n, Some(&c), alpha, beta);
    let got = gpu_gemm_nt(&ctx, &a, m, k, &b, n, Some(&c), alpha, beta)
        .expect("gpu_gemm_nt must dispatch with non-default alpha/beta and a full C");
    assert_allclose(
        &got,
        &expected,
        1e-5,
        1e-5,
        "gpu_gemm_nt (alpha=0.5, beta=2.0, full C)",
    );
}

// ── Ragged shape, no C ───────────────────────────────────────────────────

#[test]
fn gemm_nt_matches_cpu_ragged_shape_no_c() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let (m, k, n) = (3usize, 7usize, 5usize);
    let a: Vec<f32> = (0..m * k).map(pattern).collect();
    let b: Vec<f32> = (0..n * k).map(|i| pattern(i + 3)).collect();
    let expected = cpu_gemm_nt(&a, &b, m, k, n, None, 1.0, 1.0);
    let got = gpu_gemm_nt(&ctx, &a, m, k, &b, n, None, 1.0, 1.0)
        .expect("gpu_gemm_nt must dispatch at a ragged shape");
    assert_allclose(&got, &expected, 1e-5, 1e-5, "gpu_gemm_nt (ragged, no C)");
}

// ── Degenerate: 1x1x1 ────────────────────────────────────────────────────

#[test]
fn gemm_nt_matches_cpu_degenerate_1x1x1() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    let a = [3.0f32];
    let b = [4.0f32];
    let c = [1.5f32];
    let got = gpu_gemm_nt(&ctx, &a, 1, 1, &b, 1, Some(&c), 1.0, 1.0)
        .expect("gpu_gemm_nt must dispatch at a 1x1x1 shape (no minimum-size gate)");
    // 3*4 + 1*1.5 = 13.5
    assert_allclose(&got, &[13.5], 1e-6, 1e-6, "gpu_gemm_nt (1x1x1)");
}
