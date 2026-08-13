//! Wave-3 (T7-tests-engine): parity coverage for the 13 WGSL kernels that had
//! zero tests before this file — `gpu_tanh`, `gpu_exp`, `gpu_sqrt`, `gpu_abs`,
//! `gpu_neg`, `gpu_log`, `gpu_silu`, `gpu_leaky_relu`, `gpu_add`, `gpu_mul`
//! (elementwise.rs) and `gpu_reduce_sum`, `gpu_reduce_max`, `gpu_reduce_min`
//! (reduction.rs) — plus the "no adapter" failure mode.
//!
//! ## Honest skipping
//!
//! Every other GPU test file in this crate degrades *silently* on a missing
//! adapter: `let Some(ctx) = GpuContext::try_new() else { return };`. That
//! makes "ran N tests, all green because every kernel matched the CPU" and
//! "ran N tests, all green because every one of them no-opped" look identical
//! from the outside — the exact complaint this file exists to fix. Every test
//! below instead calls [`ctx_or_panic`], which panics with a named message if
//! no adapter is reachable, so a CI machine that lost its Metal/Vulkan/DX12
//! adapter reports FAILED for this file, not a quiet 0-signal pass.
//!
//! That loud failure is only worth the noise if the adapter really is meant to
//! be there, so the requirement is stated rather than assumed: `GpuContext`
//! asks for `Backends::VULKAN` *only* on Linux (`requested_backends` in
//! `src/context/types.rs`) — no GL fallback — so a Linux host needs both an ICD
//! (e.g. `/usr/share/vulkan/icd.d/nvidia_icd.json`, shipped with the driver)
//! **and** the Vulkan loader `libvulkan.so.1` (Debian/Ubuntu: `libvulkan1`),
//! which is a separate package the driver does not pull in. Installed driver +
//! missing loader enumerates zero adapters and fails every test here, which
//! looks like 17 broken kernels and is not; [`ctx_or_panic`]'s message names
//! that case. Verified reachable on the current host — Vulkan / NVIDIA RTX
//! A4000 (discrete), driver 550.144.03 — see
//! `adapter_is_reachable_in_this_environment`.
//!
//! ## Boundary shapes, and why the parity tests override them
//!
//! `oxionnx-gpu` declines (`None`) below a size floor, so a tiny tensor is not
//! worth a device round trip. Those floors used to be flat constants in
//! `src/shaders/common.rs`; they are now
//! [`oxionnx_gpu::GpuTuning`] on the context, derived from the adapter, and for
//! the memory-bound kernels on a native discrete GPU they are `usize::MAX` —
//! measured, because `gpu_relu`/`gpu_add`/`gpu_layer_norm`/`gpu_batch_norm` lose
//! to their CPU kernels at *every* size while their operands transfer.
//!
//! That is a **placement policy**, and it is not the question this file asks.
//! Whether a kernel computes the right values and whether a workload should
//! dispatch it are separate properties, so every parity test below runs on a
//! context with [`oxionnx_gpu::GpuTuning::PARITY`] installed
//! ([`parity_ctx`]) — floors of zero, so the shader actually runs at the shape
//! chosen for it — while the `*_declines_*` tests keep the context's **real**
//! tuning and pin the policy instead. Each of those first asserts that its
//! chosen length really is below this context's own floor, so it cannot quietly
//! become vacuous if a future tuning moves.
//!
//! [`policy_and_not_the_kernel_is_what_declines_a_small_tensor`] closes the
//! loop: the same tensor a real context declines must dispatch and compute
//! correctly under `PARITY`, which is what makes "declined" a placement
//! decision rather than an unproven claim about the kernel.
//!
//! ## Tolerance
//!
//! Comparisons use an absolute tolerance, documented per test: `1e-6` for
//! exact/near-exact operations (abs, neg, leaky_relu, add, reduce_max,
//! reduce_min), `1e-5` for transcendental kernels (tanh, exp, sqrt, log, silu,
//! mul) where the GPU and the Rust `f32` libm implementations are not
//! guaranteed bit-identical, and `1e-4` for `reduce_sum` where the GPU's
//! reduction order and the CPU reference's sequential order can differ (with
//! only 4 terms per group here, the actual observed drift is far smaller; the
//! looser bound is headroom, not a tuned-to-pass value).

use oxionnx_gpu::{
    gpu_abs, gpu_add, gpu_exp, gpu_leaky_relu, gpu_log, gpu_mul, gpu_neg, gpu_reduce_max,
    gpu_reduce_min, gpu_reduce_sum, gpu_silu, gpu_sqrt, gpu_tanh, GpuContext, GpuTuning,
};

/// Element count every element-wise parity test below runs at.
///
/// Historically this mirrored the crate's private `EW_GPU_THRESHOLD` because
/// the tests had to sit exactly on it. They no longer do — they install
/// [`GpuTuning::PARITY`] — but the value is kept because it is a genuinely good
/// parity shape: 100_000 is not a multiple of the 256-wide workgroup, so the
/// ragged tail of the dispatch is exercised on every one of these kernels
/// rather than only on a size that happens to divide evenly.
const EW_THRESHOLD: usize = 100_000;

/// Output element count every reduction parity test below runs at. Same story
/// as [`EW_THRESHOLD`], and 50_000 is likewise not a multiple of 256.
const REDUCE_OUT_THRESHOLD: usize = 50_000;

/// Get a GPU context or fail the test loudly.
///
/// See the module docs' "Honest skipping" section: this is the one function
/// every test in this file routes through instead of the silent
/// `let Some(ctx) = ... else { return }` pattern used elsewhere in this crate.
fn ctx_or_panic() -> GpuContext {
    // `try_new_diagnosed` rather than `try_new`: the most common way a Linux
    // host with a perfectly good GPU lands here is a missing Vulkan *loader*
    // (`libvulkan.so.1`, Debian/Ubuntu package `libvulkan1`) that the driver
    // package does not pull in, and the diagnostic detects that case by
    // re-enumerating every other backend and reporting the GPU it finds there.
    // A bare `None` cannot tell an operator any of that.
    match GpuContext::try_new_diagnosed() {
        Ok(ctx) => ctx,
        Err(err) => panic!(
            "no wgpu adapter available -- this test cannot verify anything without \
             one and is failing loudly rather than silently passing.\n  {err}\n\
             See oxionnx-gpu/tests/w3_gpu_kernel_parity.rs module docs."
        ),
    }
}

/// A context whose placement policy declines nothing, for the parity tests.
///
/// See the module docs: a real context's floors are a statement about whether a
/// dispatch is *worth making*, and on a native discrete GPU the measured answer
/// for the memory-bound kernels is "never, while transferring". Testing shader
/// numerics through that gate would mean testing nothing at all, so these tests
/// install [`GpuTuning::PARITY`] and let the shader run.
fn parity_ctx() -> GpuContext {
    let mut ctx = ctx_or_panic();
    ctx.set_tuning(GpuTuning::PARITY);
    ctx
}

/// A dedicated, fast, first-to-triage test: if only this one fails, the
/// problem is the environment (no adapter), not a kernel.
///
/// `eprintln!` alone would not verify anything under `cargo nextest` without
/// `--no-capture` (output is swallowed on success) -- the exact false-green
/// shape this file exists to eliminate, in the one test named for the
/// property. `is_degraded()` is asserted instead: a degraded context makes
/// every kernel below decline (`None`), which would surface as a misleading
/// `.expect()` panic in an unrelated-looking test rather than here.
#[test]
fn adapter_is_reachable_in_this_environment() {
    let ctx = ctx_or_panic();
    eprintln!("w3_gpu_kernel_parity: adapter reachable, is_degraded=false");
    assert!(
        !ctx.is_degraded(),
        "adapter was reachable but reports degraded -- every kernel test below \
         will decline (None) and fail with a misleading .expect() message instead \
         of this clearer one"
    );
}

/// Deterministic f32 pattern, magnitude bounded so every op below (tanh, exp,
/// silu, ...) stays comfortably away from overflow/underflow.
fn pattern(i: usize) -> f32 {
    ((i % 23) as f32 - 11.0) * 0.25 // range [-2.75, 2.75]
}

/// Same pattern shifted strictly positive, for sqrt/log's domain.
fn positive_pattern(i: usize) -> f32 {
    ((i % 23) as f32) * 0.25 + 0.01 // range [0.01, 5.51]
}

fn assert_allclose(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: length mismatch (gpu={}, cpu={})",
        actual.len(),
        expected.len()
    );
    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let d = (a - e).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{what}: max abs diff {max_diff} at index {max_at} (gpu={}, cpu={}) exceeds tolerance {tol}",
        actual[max_at],
        expected[max_at]
    );
}

// ── unary elementwise: parity at the accept-side boundary ──────────────────

#[test]
fn tanh_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let data: Vec<f32> = (0..EW_THRESHOLD).map(pattern).collect();
    let expected: Vec<f32> = data.iter().map(|x| x.tanh()).collect();
    let got = gpu_tanh(&ctx, &data).expect("gpu_tanh must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-5, "gpu_tanh");
}

#[test]
fn exp_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let data: Vec<f32> = (0..EW_THRESHOLD).map(|i| pattern(i) * 0.1).collect();
    let expected: Vec<f32> = data.iter().map(|x| x.exp()).collect();
    let got = gpu_exp(&ctx, &data).expect("gpu_exp must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-5, "gpu_exp");
}

#[test]
fn sqrt_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let data: Vec<f32> = (0..EW_THRESHOLD).map(positive_pattern).collect();
    let expected: Vec<f32> = data.iter().map(|x| x.sqrt()).collect();
    let got = gpu_sqrt(&ctx, &data).expect("gpu_sqrt must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-5, "gpu_sqrt");
}

#[test]
fn abs_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let data: Vec<f32> = (0..EW_THRESHOLD).map(pattern).collect();
    let expected: Vec<f32> = data.iter().map(|x| x.abs()).collect();
    let got = gpu_abs(&ctx, &data).expect("gpu_abs must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-6, "gpu_abs");
}

#[test]
fn neg_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let data: Vec<f32> = (0..EW_THRESHOLD).map(pattern).collect();
    let expected: Vec<f32> = data.iter().map(|x| -x).collect();
    let got = gpu_neg(&ctx, &data).expect("gpu_neg must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-6, "gpu_neg");
}

#[test]
fn log_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let data: Vec<f32> = (0..EW_THRESHOLD).map(positive_pattern).collect();
    let expected: Vec<f32> = data.iter().map(|x| x.ln()).collect();
    let got = gpu_log(&ctx, &data).expect("gpu_log must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-5, "gpu_log");
}

#[test]
fn silu_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let data: Vec<f32> = (0..EW_THRESHOLD).map(pattern).collect();
    let expected: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    let got = gpu_silu(&ctx, &data).expect("gpu_silu must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-5, "gpu_silu");
}

#[test]
fn leaky_relu_default_alpha_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let data: Vec<f32> = (0..EW_THRESHOLD).map(pattern).collect();
    // gpu_leaky_relu (no explicit alpha) applies the ONNX default, 0.01.
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| if x >= 0.0 { x } else { 0.01 * x })
        .collect();
    let got =
        gpu_leaky_relu(&ctx, &data).expect("gpu_leaky_relu must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-6, "gpu_leaky_relu");
}

/// The policy side of the same boundary, on a context with its **real**
/// tuning: a tensor below this adapter's own element-wise floor must be handed
/// back (`None`), not dispatched, not panicked on and not silently truncated.
/// `gpu_tanh` stands in for the shared `gpu_elementwise_placed` gate that all
/// eight unary kernels above go through identically.
///
/// The precondition is asserted rather than assumed: if a future tuning lowered
/// the floor below this length, the test would otherwise start passing for the
/// wrong reason (dispatching and returning `Some`) or, worse, keep passing
/// while verifying nothing.
#[test]
fn tanh_declines_one_below_threshold() {
    let ctx = ctx_or_panic();
    let len = EW_THRESHOLD - 1;
    assert!(
        len < ctx.tuning().elementwise_min_elements,
        "this test's length must sit below the context's own floor to mean anything \
         (len={len}, floor={})",
        ctx.tuning().elementwise_min_elements
    );
    let data: Vec<f32> = (0..len).map(pattern).collect();
    assert!(
        gpu_tanh(&ctx, &data).is_none(),
        "gpu_tanh must decline below this context's elementwise floor instead of dispatching"
    );
}

/// What makes the decline above a *placement* decision rather than an untested
/// claim: the identical tensor, on a context whose only difference is
/// [`GpuTuning::PARITY`], dispatches and computes the right values.
///
/// Without this pairing, "the kernel declined" and "the kernel cannot do it"
/// are indistinguishable from the outside — which is the same false-green shape
/// this file's `ctx_or_panic` was written to eliminate, one level up.
#[test]
fn policy_and_not_the_kernel_is_what_declines_a_small_tensor() {
    let len = EW_THRESHOLD - 1;
    let data: Vec<f32> = (0..len).map(pattern).collect();
    assert!(
        gpu_tanh(&ctx_or_panic(), &data).is_none(),
        "precondition: the real tuning must decline this length"
    );
    let got = gpu_tanh(&parity_ctx(), &data)
        .expect("the same tensor must dispatch once the size gate is lifted");
    let expected: Vec<f32> = data.iter().map(|v| v.tanh()).collect();
    assert_allclose(&got, &expected, 1e-5, "gpu_tanh under PARITY tuning");
}

// ── binary elementwise: parity at the accept-side boundary ─────────────────

#[test]
fn add_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let a: Vec<f32> = (0..EW_THRESHOLD).map(pattern).collect();
    let b: Vec<f32> = (0..EW_THRESHOLD).map(|i| pattern(i + 7)).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let got = gpu_add(&ctx, &a, &b).expect("gpu_add must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-6, "gpu_add");
}

#[test]
fn mul_matches_cpu_at_threshold() {
    let ctx = parity_ctx();
    let a: Vec<f32> = (0..EW_THRESHOLD).map(pattern).collect();
    let b: Vec<f32> = (0..EW_THRESHOLD).map(|i| pattern(i + 7)).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    let got = gpu_mul(&ctx, &a, &b).expect("gpu_mul must dispatch at EW_THRESHOLD elements");
    assert_allclose(&got, &expected, 1e-5, "gpu_mul");
}

/// A length mismatch must decline even though both lengths individually clear
/// the threshold -- the guard is `len != b.len()`, checked before the length
/// comparison, not implied by it.
#[test]
fn add_declines_on_mismatched_lengths_even_above_threshold() {
    let ctx = parity_ctx();
    let a: Vec<f32> = (0..EW_THRESHOLD).map(pattern).collect();
    let b: Vec<f32> = (0..EW_THRESHOLD - 1).map(pattern).collect();
    assert!(
        gpu_add(&ctx, &a, &b).is_none(),
        "gpu_add must decline (not panic) when a and b have different lengths"
    );
}

// ── reductions: parity at the accept-side output-size boundary ─────────────

/// CPU reference for reduction along `axis` of a row-major `shape` tensor.
/// `op`/`seed` select sum (`+`, `0.0`), max (`f32::max`, `-inf`) or min
/// (`f32::min`, `+inf`).
fn cpu_reduce(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    op: fn(f32, f32) -> f32,
    seed: f32,
) -> Vec<f32> {
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut out = vec![seed; outer * inner];
    for o in 0..outer {
        for a in 0..axis_len {
            for i in 0..inner {
                let in_idx = (o * axis_len + a) * inner + i;
                let out_idx = o * inner + i;
                out[out_idx] = op(out[out_idx], data[in_idx]);
            }
        }
    }
    out
}

#[test]
fn reduce_sum_matches_cpu_at_output_threshold() {
    let ctx = parity_ctx();
    // shape [REDUCE_OUT_THRESHOLD, 4], axis=1 reduced away: output size is
    // REDUCE_OUT_THRESHOLD * 1, exactly on the boundary.
    let outer = REDUCE_OUT_THRESHOLD;
    let axis_len = 4usize;
    let shape = vec![outer, axis_len];
    let data: Vec<f32> = (0..outer * axis_len).map(pattern).collect();
    let expected = cpu_reduce(&data, &shape, 1, |a, b| a + b, 0.0);
    let got = gpu_reduce_sum(&ctx, &data, 1, &shape)
        .expect("gpu_reduce_sum must dispatch at REDUCE_OUT_THRESHOLD output elements");
    assert_allclose(&got, &expected, 1e-4, "gpu_reduce_sum");
}

#[test]
fn reduce_max_matches_cpu_at_output_threshold() {
    let ctx = parity_ctx();
    let outer = REDUCE_OUT_THRESHOLD;
    let axis_len = 4usize;
    let shape = vec![outer, axis_len];
    let data: Vec<f32> = (0..outer * axis_len).map(pattern).collect();
    let expected = cpu_reduce(&data, &shape, 1, f32::max, f32::NEG_INFINITY);
    let got = gpu_reduce_max(&ctx, &data, 1, &shape)
        .expect("gpu_reduce_max must dispatch at REDUCE_OUT_THRESHOLD output elements");
    assert_allclose(&got, &expected, 1e-6, "gpu_reduce_max");
}

#[test]
fn reduce_min_matches_cpu_at_output_threshold() {
    let ctx = parity_ctx();
    let outer = REDUCE_OUT_THRESHOLD;
    let axis_len = 4usize;
    let shape = vec![outer, axis_len];
    let data: Vec<f32> = (0..outer * axis_len).map(pattern).collect();
    let expected = cpu_reduce(&data, &shape, 1, f32::min, f32::INFINITY);
    let got = gpu_reduce_min(&ctx, &data, 1, &shape)
        .expect("gpu_reduce_min must dispatch at REDUCE_OUT_THRESHOLD output elements");
    assert_allclose(&got, &expected, 1e-6, "gpu_reduce_min");
}

/// The output-size boundary's decline side: one element short, must return
/// `None`, not a partially-computed or panicking result.
#[test]
fn reduce_sum_declines_one_below_output_threshold() {
    let ctx = ctx_or_panic();
    let outer = REDUCE_OUT_THRESHOLD - 1;
    assert!(
        outer < ctx.tuning().reduce_min_output_elements,
        "this test's output count must sit below the context's own floor to mean \
         anything (out={outer}, floor={})",
        ctx.tuning().reduce_min_output_elements
    );
    let axis_len = 4usize;
    let shape = vec![outer, axis_len];
    let data: Vec<f32> = (0..outer * axis_len).map(pattern).collect();
    assert!(
        gpu_reduce_sum(&ctx, &data, 1, &shape).is_none(),
        "gpu_reduce_sum must decline below this context's reduction floor"
    );
}
