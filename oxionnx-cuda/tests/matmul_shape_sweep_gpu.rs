//! On-device shape-sweep regression test for [`matmul::cuda_matmul`] against
//! the [`reference::ref_matmul`] CPU oracle.
//!
//! # Background
//!
//! This is the crate whose bug report started this investigation: on real
//! Ampere hardware, `cuda_matmul` (which dispatches through
//! `oxicuda_blas::level3::gemm_api::gemm` via `ctx.dnn.blas()`) was returning
//! numerically wrong results for the exact shapes below — reproduced here in
//! this test's own history before the fix:
//!
//! ```text
//! [64x64x64_ones]  M=64  K=64    N=64  : 3456/4096 wrong, first=(640, 0.0, 64.0)
//! [m1_arcface]      M=1  K=25088 N=512 :  439/512  wrong, first=(0,   0.0, -0.0016)
//! [m8_inswapper]    M=8  K=25088 N=512 : 3220/4096 wrong, first=(0,   0.0, -0.0016)
//! ```
//!
//! The root cause was **not** the GEMM grid/block computation in
//! `oxicuda-blas` (that launch, traced independently in
//! `oxicuda-blas/tests/gemm_shape_sweep_gpu.rs`, covers every output element
//! correctly for every shape below) — it was that `oxicuda_dnn::DnnHandle`
//! gives its internal `BlasHandle` its *own* CUDA stream (deliberately, to
//! let BLAS and DNN launches overlap), and `cuda_matmul` was synchronizing
//! `ctx.dnn.stream()` — the *other* stream — before reading the result back.
//! Every `oxicuda-driver` stream is `CU_STREAM_NON_BLOCKING`, so the two
//! streams do not implicitly order against each other: the host could read
//! `C` back before the kernel queued on the BLAS stream had actually
//! finished, observing whatever was in that freshly-zeroed buffer already —
//! plausible-looking zeros, not a crash. The `0.0`-vs-nonzero pattern above,
//! and the *inconsistent* cutoff index between the two `K=25088` shapes
//! (both dispatch identically but land at different points in a race), is
//! the signature of exactly that: a coverage bug would cut off at a
//! consistent, shape-derived boundary; a race cuts off wherever the reader
//! happened to catch up to the writer, which came out at element `0` for the
//! `K=25088` shapes in the run above (the slow, poorly-parallelised
//! *pre-occupancy-fix* single-pass kernel lost the race almost immediately)
//! and element `640` for `64x64x64` (fast enough for the race to usually —
//! but not always — resolve in the reader's favour).
//!
//! Fixed by `oxicuda_dnn::handle::DnnHandle::synchronize_all` (waits for
//! *both* streams) plus routing `matmul.rs`/`reduce.rs` through it instead of
//! `ctx.dnn.stream()` alone. This file is the regression test that would
//! have caught it: every shape below is checked *element-for-element*
//! through the exact production call (`matmul::cuda_matmul`), not a
//! hand-synchronized variant.
//!
//! # Gating
//!
//! `required-features = ["gpu-tests"]` in `Cargo.toml` keeps this file's
//! tests out of a plain `cargo test -p oxionnx-cuda` entirely (see that
//! entry's own comment for why that matters — on a CUDA-capable host it is
//! not a cosmetic distinction). With the feature on, [`gpu_ctx`] returns
//! `None` and each test skips when no device is present, so that a
//! `--all-features` run stays green on a CPU-only host (a Mac, this
//! workspace's own CI). That is the OxiCUDA convention this crate follows —
//! see `oxicuda-blas`'s `src/gpu_tests.rs` ("Every device test returns early
//! (skips) when no CUDA device is present, so the suite stays green on
//! CPU-only machines") and its `tests/gemm_shape_sweep_gpu.rs`. For the
//! tests to actually *run*, use `cargo test -p oxionnx-cuda --features
//! gpu-tests --test matmul_shape_sweep_gpu` on a CUDA-capable machine.

use oxionnx_cuda::context::{Activation, CudaContext};
use oxionnx_cuda::{matmul, reference};

// ---------------------------------------------------------------------------
// Fixture & helpers
// ---------------------------------------------------------------------------

/// Acquires a real GPU context, bypassing the `OXIONNX_CUDA` env-gate
/// (`Activation::Enabled`) so this test doesn't depend on the invoking
/// shell's environment beyond the `gpu-tests` feature that already gates
/// this whole file, or `None` when no CUDA driver / device is present —
/// each test then skips. See the module docs' "Gating" section.
fn gpu_ctx() -> Option<CudaContext> {
    CudaContext::try_new_with(Activation::Enabled)
}

/// A small deterministic LCG (same algorithm as `oxicuda-blas`'s
/// `src/gpu_tests.rs::Lcg`, duplicated here as this is a separate crate).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    fn range_f32(&mut self, lo: f64, hi: f64) -> f32 {
        let unit = f64::from(self.next_u32()) / 4_294_967_296.0;
        (lo + (hi - lo) * unit) as f32
    }
}

fn make_matrix(rng: &mut Lcg, len: usize, lo: f64, hi: f64) -> Vec<f32> {
    (0..len).map(|_| rng.range_f32(lo, hi)).collect()
}

/// Checks every element of `got` (from `cuda_matmul`) against
/// `reference::ref_matmul`'s `f64`-accumulated oracle, reporting every
/// disagreement (not just the first).
fn assert_matches_oracle(
    got: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    tag: &str,
) {
    let expect = reference::ref_matmul(a, b, m, k, n);
    assert_eq!(got.len(), expect.len(), "{tag}: length mismatch");
    let mut mismatches = Vec::new();
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        let tol = 1e-4 + 2e-3 * f64::from(e.abs());
        if f64::from((g - e).abs()) > tol {
            mismatches.push((i, g, e));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{tag}: {} / {} elements mismatched (showing up to 10): {:?}",
        mismatches.len(),
        got.len(),
        &mismatches[..mismatches.len().min(10)],
    );
}

// ---------------------------------------------------------------------------
// The exact reproduction cases named in the bug report
// ---------------------------------------------------------------------------

/// The exact `64x64x64` all-ones reproduction: every element must be `64.0`.
/// Pre-fix this failed with 3456/4096 elements silently reading back `0.0`.
#[test]
fn square_64_all_ones_matches_64_everywhere() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!("no CUDA device present, skipping square_64_all_ones_matches_64_everywhere");
        return;
    };
    let (m, k, n) = (64usize, 64usize, 64usize);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let got = matmul::cuda_matmul(&ctx, &a, &b, m, k, n).expect("cuda_matmul");
    for (i, &v) in got.iter().enumerate() {
        assert!(
            (v - 64.0).abs() < 1e-3,
            "element {i}: got {v}, expected 64.0 (64x64x64 all-ones reproduction)"
        );
    }
}

/// ArcFace's `1x512` embedding projection (`M=1, K=25088, N=512`) and
/// InSwapper's emap projection batched to `M=8`, the two shapes the bug
/// report names as "100% wrong (every element)" pre-fix.
#[test]
fn arcface_and_inswapper_dominant_shapes_match_cpu_oracle() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!("no CUDA device present, skipping arcface_and_inswapper_dominant_shapes_match_cpu_oracle");
        return;
    };
    let mut rng = Lcg::new(0xA4CE_FACE_0000_0002);
    for &m in &[1usize, 8] {
        let (k, n) = (25088usize, 512usize);
        let a = make_matrix(&mut rng, m * k, -1.0, 1.0);
        let b = make_matrix(&mut rng, k * n, -1.0, 1.0);
        let got = matmul::cuda_matmul(&ctx, &a, &b, m, k, n).expect("cuda_matmul");
        assert_matches_oracle(&got, &a, &b, m, k, n, &format!("M={m} K={k} N={n}"));
    }
}

// ---------------------------------------------------------------------------
// Broader M sweep through the same production entry point
// ---------------------------------------------------------------------------

/// `M` in `{1, 2, 7, 8, 16, 17, 32, 64, 100, 128, 1024}` crossed with two
/// `(K, N)` pairs, through `cuda_matmul` end-to-end (upload, dispatch,
/// `synchronize_all`, readback) — every element checked, not sampled.
/// `2048` is intentionally left to the dedicated `oxicuda-blas` sweep: this
/// file's purpose is proving the *dispatch layer* (stream sync, buffer
/// wiring) is correct across shapes, which the smaller sweep already
/// establishes without paying `2048`'s extra upload/readback cost twice.
#[test]
fn m_sweep_matches_cpu_oracle_exactly() {
    let Some(ctx) = gpu_ctx() else {
        eprintln!("no CUDA device present, skipping m_sweep_matches_cpu_oracle_exactly");
        return;
    };
    let m_values = [1usize, 2, 7, 8, 16, 17, 32, 64, 100, 128, 1024];
    let kn_pairs = [(64usize, 64usize), (257, 129)];
    let mut rng = Lcg::new(0x5EED_F00D_9999_0001);

    for &m in &m_values {
        for &(k, n) in &kn_pairs {
            let a = make_matrix(&mut rng, m * k, -1.0, 1.0);
            let b = make_matrix(&mut rng, k * n, -1.0, 1.0);
            let got = matmul::cuda_matmul(&ctx, &a, &b, m, k, n).expect("cuda_matmul");
            assert_matches_oracle(&got, &a, &b, m, k, n, &format!("M={m} K={k} N={n}"));
        }
    }
}
