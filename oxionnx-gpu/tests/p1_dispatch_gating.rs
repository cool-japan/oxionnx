//! On-device coverage for the dispatch **gate** — the decision to hand a node
//! to the GPU at all — as opposed to the kernels, which every other test file
//! here covers.
//!
//! Four properties, each of which was a real defect or a real unverified claim:
//!
//! 1. [`gemm_over_the_binding_limit_declines_instead_of_panicking`] — the
//!    buffer-size guard. wgpu turns a binding larger than
//!    `max_storage_buffer_binding_size` into a validation error, and its default
//!    handler for that is a **process abort**. A `Gemm` whose `B` exceeded the
//!    limit therefore used to take the host down instead of falling back to the
//!    CPU. The guard that fixed it (`gemm_buffer_sizes` → `checked_storage_bytes`
//!    → `GpuLimits::storage_fits`) had no test that actually exercised it against
//!    a live device.
//! 2. [`a_skinny_gemm_declines_even_though_it_clears_the_flop_floor`] — the
//!    shape gate. A FLOP count alone cannot express "this dispatch moves as many
//!    elements as it does arithmetic".
//! 3. [`residency_changes_the_answer_for_the_same_skinny_shape`] — that the
//!    shape gate is residency-aware, because the same shape is a 2.3x *win* when
//!    `B` does not cross the bus.
//! 4. [`the_adapter_is_classified_and_the_thresholds_follow_from_it`] and
//!    [`a_software_adapter_would_decline_every_size`] — that the thresholds are
//!    derived from the adapter rather than compiled in.
//!
//! Every test states its own skip condition rather than silently returning on a
//! missing adapter; see `w3_gpu_kernel_parity.rs` for why that matters.

use oxionnx_core::Tensor;
use oxionnx_gpu::{
    gpu_matmul, GemmWeightTraffic, GpuContext, GpuLimits, GpuPerfClass, GpuTuning, WeightKeys,
};

/// The WebGPU baseline limits — `wgpu::Limits::default()`.
///
/// This crate deliberately asks for the *adapter's* limits instead (see
/// `GpuContext::acquire_device`), which is why a 132 MiB binding is ordinary on
/// the reference box. Pinning a live context back to the baseline is how the
/// test below reproduces the historical failure on hardware that would
/// otherwise never reach it — and the baseline is not a hypothetical: it is
/// exactly what a browser device reports.
const WEBGPU_BASELINE_STORAGE_BINDING: u64 = 128 << 20;
const WEBGPU_BASELINE_BUFFER: u64 = 256 << 20;

fn ctx_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::try_new_diagnosed() {
        Ok(ctx) => Some(ctx),
        Err(err) => {
            eprintln!("[skip] {what}: {err}");
            None
        }
    }
}

fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            ((x % 23) as f32) * 0.037 - 0.4
        })
        .collect()
}

fn assert_allclose(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut at = 0usize;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let d = (a - e).abs();
        if d > max_diff {
            max_diff = d;
            at = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{what}: max abs diff {max_diff} at {at} (gpu={}, cpu={}) exceeds {tol}",
        actual[at],
        expected[at]
    );
}

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let at = Tensor::new(a.to_vec(), vec![m, k]);
    let bt = Tensor::new(b.to_vec(), vec![k, n]);
    oxionnx_ops::math::matmul(&at, &bt)
        .expect("cpu reference matmul")
        .data
}

// ========================================================================
// 1. The buffer-size guard — the historical "GEMM over the limit panics"
// ========================================================================

/// A GEMM whose `B` exceeds the device's storage-binding limit must **decline**,
/// leaving the context healthy, rather than reaching wgpu's validation error.
///
/// # The bug this pins
///
/// `wgpu` reports a binding larger than `max_storage_buffer_binding_size` as a
/// validation error, and an uncaptured validation error is a `panic!` by
/// default — in a library whose entire contract is "return `None` and let the
/// CPU handle it", that is a process abort triggered by an ordinary model. The
/// canonical shape is a transformer's output projection, `B = [4096, 32000]` =
/// 524 MB, against the WebGPU baseline's 128 MiB binding cap; anything past
/// 256 MB also fails the `max_buffer_size` cap, which is where the
/// "GEMM > 256 MB panics" description comes from.
///
/// # Why it pins the limits rather than allocating 524 MB
///
/// The reference box's adapter reports a **2 GiB** binding limit, and this
/// crate deliberately requests the adapter's real limits rather than the
/// baseline — so the historical shape simply succeeds here and proves nothing.
/// Pinning `ctx.limits` (a `pub` field) to the WebGPU baseline puts a live,
/// real-hardware context into exactly the configuration a browser device is in
/// permanently, at a `B` of 132 MiB rather than 524 MB.
///
/// The test then **restores** the real limits and runs the same call again. That
/// second half is what makes the first half mean something: without it, `None`
/// could equally be an unrelated decline (a shape the kernel dislikes, a
/// degraded device), and the test would pass while verifying nothing.
#[test]
fn gemm_over_the_binding_limit_declines_instead_of_panicking() {
    let Some(mut ctx) = ctx_or_skip("gemm_over_the_binding_limit") else {
        return;
    };
    if ctx.perf_class() == GpuPerfClass::Software {
        eprintln!("[skip] software adapter declines every size by design");
        return;
    }

    // 8192 x 4224 f32 = 132 MiB — over the baseline's 128 MiB binding cap and
    // under its 256 MiB allocation cap, so the *binding* guard is what fires.
    let (m, k, n) = (64usize, 8192usize, 4224usize);
    let b_bytes = (k * n * 4) as u64;
    assert!(
        b_bytes > WEBGPU_BASELINE_STORAGE_BINDING && b_bytes < WEBGPU_BASELINE_BUFFER,
        "fixture must sit between the two baseline caps, got {b_bytes} bytes"
    );
    // ...and it must clear the *placement* gate, or the decline below would be
    // the size heuristic rather than the safety guard.
    assert!(
        ctx.tuning()
            .gemm_admits(m, k, n, GemmWeightTraffic::PerDispatch),
        "fixture must be a shape the tuning would otherwise dispatch"
    );

    let a = fill(m * k, 2_654_435_761);
    let b = fill(k * n, 40_503);

    let real_limits = ctx.limits;
    ctx.limits = GpuLimits {
        max_storage_buffer_binding_size: WEBGPU_BASELINE_STORAGE_BINDING,
        max_buffer_size: WEBGPU_BASELINE_BUFFER,
        ..real_limits
    };
    assert!(
        !ctx.limits.storage_fits(b_bytes),
        "precondition: B must not fit the pinned baseline binding limit"
    );

    // The whole point: this returns rather than aborting the test process.
    let declined = gpu_matmul(&ctx, &a, &b, m, k, n);
    assert!(
        declined.is_none(),
        "a {b_bytes}-byte B over a {WEBGPU_BASELINE_STORAGE_BINDING}-byte binding limit \
         must decline, not dispatch"
    );
    // A decline is not an error: nothing was submitted, so nothing can have
    // degraded the device. If this fires, the guard ran *after* the buffer was
    // created rather than before it.
    assert!(
        !ctx.is_degraded(),
        "declining must not degrade the context; last_error = {:?}",
        ctx.last_error()
    );
    assert_eq!(ctx.last_error(), None);

    // Restore the adapter's real limits: the identical call must now dispatch
    // and be correct, which is what proves the decline above was the size guard.
    ctx.limits = real_limits;
    if !real_limits.storage_fits(b_bytes) {
        eprintln!(
            "[skip half] this adapter's own binding limit ({}) is below the fixture; \
             the decline was verified, the dispatch cannot be",
            real_limits.max_storage_buffer_binding_size
        );
        return;
    }
    let dispatched = gpu_matmul(&ctx, &a, &b, m, k, n)
        .expect("the same call must dispatch once the real limits are restored");
    assert_allclose(
        &dispatched,
        &cpu_matmul(&a, &b, m, k, n),
        2e-2,
        "132 MiB-B GEMM under the adapter's real limits",
    );
}

/// The same guard, stated as arithmetic on the historical shape, so the
/// 524 MB `lm_head` projection is pinned by name without allocating it.
#[test]
fn the_lm_head_projection_does_not_fit_the_webgpu_baseline() {
    let baseline = GpuLimits {
        max_storage_buffer_binding_size: WEBGPU_BASELINE_STORAGE_BINDING,
        max_buffer_size: WEBGPU_BASELINE_BUFFER,
        max_workgroups_per_dimension: 65_535,
    };
    // B = [4096, 32000] f32.
    let lm_head_bytes = 4096u64 * 32_000 * 4;
    assert_eq!(lm_head_bytes, 524_288_000);
    assert!(!baseline.storage_fits(lm_head_bytes));
    assert!(!baseline.buffer_fits(lm_head_bytes));
    // And the 256 MB boundary the historical report named, from both sides.
    assert!(baseline.buffer_fits(WEBGPU_BASELINE_BUFFER));
    assert!(!baseline.buffer_fits(WEBGPU_BASELINE_BUFFER + 1));
}

// ========================================================================
// 2 & 3. Shape-aware, residency-aware gating
// ========================================================================

/// ArcFace's embedding head shape, `[1, 25088] × [25088, 512]`: 12.8 M
/// multiply-accumulates, which clears every FLOP-only threshold this crate has
/// ever had, and 1.54x **slower** than the CPU kernel when measured on an RTX
/// A4000 — because it moves a 51.4 MB `B` across the bus to perform 25.7 MFLOP.
///
/// The gate must decline it on shape, not admit it on size.
#[test]
fn a_skinny_gemm_declines_even_though_it_clears_the_flop_floor() {
    let Some(ctx) = ctx_or_skip("skinny_gemm_declines") else {
        return;
    };
    let (m, k, n) = (1usize, 25_088usize, 512usize);
    // Both historical FLOP-only gates admitted this shape: the session's
    // `GEMM_GPU_MIN_FLOPS` (10 MFLOP = 5 M multiply-accumulates, preserved as
    // `LEGACY_FLAT.gemm_min_mac_cached`) and `compute.rs`'s flat 10 M `m*k*n`.
    let mac = GpuTuning::gemm_mac(m, k, n).expect("no overflow");
    assert!(mac >= GpuTuning::LEGACY_FLAT.gemm_min_mac_cached);
    assert!(mac >= GpuTuning::LEGACY_FLAT.gemm_min_mac);
    let a = fill(m * k, 2_654_435_761);
    let b = fill(k * n, 40_503);
    assert!(
        gpu_matmul(&ctx, &a, &b, m, k, n).is_none(),
        "a shape that moves 12.8 M elements to do 12.8 M multiply-accumulates must \
         decline however large its FLOP count is"
    );
}

/// The other side of the same rule: a shape with the *same* operand sizes but a
/// batch dimension wide enough to amortize them must still dispatch, and be
/// correct. Without this, "declines skinny GEMMs" and "declines everything"
/// look identical.
#[test]
fn a_well_shaped_gemm_still_dispatches_and_is_correct() {
    let Some(ctx) = ctx_or_skip("well_shaped_gemm_dispatches") else {
        return;
    };
    if ctx.perf_class() == GpuPerfClass::Software {
        eprintln!("[skip] software adapter declines every size by design");
        return;
    }
    let (m, k, n) = (64usize, 1024usize, 1024usize);
    let a = fill(m * k, 2_654_435_761);
    let b = fill(k * n, 40_503);
    let got = gpu_matmul(&ctx, &a, &b, m, k, n)
        .expect("64x1024x1024 measured 0.54x the CPU kernel and must dispatch");
    assert_allclose(&got, &cpu_matmul(&a, &b, m, k, n), 1e-2, "64x1024x1024");
}

/// The skinny shape above is a 2.3x **win** when `B` has a residency identity,
/// so the gate must not decline it there — the fix for a skinny GEMM is
/// residency, not a bigger threshold.
///
/// Measured on the reference box: `[1, 25088] × [512, 25088]ᵀ` ran at 0.43x the
/// CPU kernel with a warm weight cache and 1.54x with an uploaded `B`. A gate
/// that used one rule for both would have to be wrong for one of them.
#[test]
fn residency_changes_the_answer_for_the_same_skinny_shape() {
    let Some(ctx) = ctx_or_skip("residency_changes_the_answer") else {
        return;
    };
    if ctx.perf_class() == GpuPerfClass::Software {
        eprintln!("[skip] software adapter declines every size by design");
        return;
    }
    let (m, k, n) = (1usize, 25_088usize, 512usize);
    // The policy, first: same shape, two answers.
    assert!(!ctx
        .tuning()
        .gemm_admits(m, k, n, GemmWeightTraffic::PerDispatch));
    assert!(ctx.tuning().gemm_admits(m, k, n, GemmWeightTraffic::Cached));

    // Then the kernel, to show the admitted side actually computes.
    let a = fill(m * k, 2_654_435_761);
    // `gpu_gemm_nt` reads B as [N, K] — the PyTorch `nn.Linear` layout.
    let b_nt = fill(n * k, 40_503);
    let keys = WeightKeys::new(Some("p1_arcface_head"), None);
    let got = pollster::block_on(oxionnx_gpu::gpu_gemm_nt_resident_async(
        &ctx, &a, m, k, &b_nt, n, None, 1.0, 1.0, keys,
    ))
    .expect("a cached-weight Gemm at this shape measured 0.43x the CPU kernel");

    let at = Tensor::new(a.clone(), vec![m, k]);
    let bt = Tensor::new(b_nt.clone(), vec![n, k]);
    let expected = oxionnx_ops::math::gemm(&at, &bt, None, 1.0, 1.0, false, true)
        .expect("cpu reference gemm")
        .data;
    assert_allclose(&got, &expected, 1e-2, "cached-weight skinny Gemm");

    // ...and the second dispatch must be served from the cache, which is the
    // property the `Cached` arm of the gate is asserting. (Item: the persistent
    // weight buffer pool is present and working.)
    let before = ctx.resident_counters();
    let again = pollster::block_on(oxionnx_gpu::gpu_gemm_nt_resident_async(
        &ctx, &a, m, k, &b_nt, n, None, 1.0, 1.0, keys,
    ))
    .expect("second dispatch");
    let delta = ctx.resident_counters().since(before);
    assert_eq!(
        delta.uploaded_bytes, 0,
        "the 51.4 MB weight must not cross the bus a second time"
    );
    assert!(delta.hits >= 1, "the second dispatch must be a cache hit");
    assert_allclose(
        &again,
        &expected,
        1e-2,
        "cached-weight skinny Gemm, 2nd call",
    );
}

// ========================================================================
// 4. Device-awareness
// ========================================================================

/// The thresholds must come from the adapter, not from a constant.
#[test]
fn the_adapter_is_classified_and_the_thresholds_follow_from_it() {
    let Some(ctx) = ctx_or_skip("adapter_is_classified") else {
        return;
    };
    let class = ctx.perf_class();
    eprintln!(
        "p1_dispatch_gating: adapter classified as {}",
        class.as_str()
    );
    assert_eq!(
        *ctx.tuning(),
        GpuTuning::for_class(class),
        "a context's tuning must be exactly the table for its class"
    );
}

/// A software rasterizer — Mesa `lavapipe`, SwiftShader, Direct3D WARP — is
/// this same CPU running one invocation per shader thread, without
/// `matrixmultiply`'s packing and without rayon. It cannot beat the CPU kernel
/// at any size, and a headless container that installs `mesa-vulkan-drivers`
/// gets one and a perfectly valid `wgpu::Adapter` with no hardware behind it.
///
/// Asserted on the table rather than on a device, since the reference box has
/// real hardware: what matters is that the classification exists and that its
/// answer is "never", so such a machine runs its model on the CPU operators
/// instead of through a shader interpreter.
#[test]
fn a_software_adapter_would_decline_every_size() {
    let t = GpuTuning::for_class(GpuPerfClass::Software);
    assert!(!t.gemm_admits(4096, 4096, 4096, GemmWeightTraffic::PerDispatch));
    assert!(!t.gemm_admits(4096, 4096, 4096, GemmWeightTraffic::Cached));
    assert!(!t.conv_admits(4096, 4096, 4096));
    assert_eq!(t.elementwise_min_elements, usize::MAX);
    assert_eq!(t.reduce_min_output_elements, usize::MAX);
    assert_eq!(t.softmax_min_row_len, usize::MAX);
}

/// The initialization diagnostic must succeed here, and must report the class.
///
/// On a Linux host the usual reason it would *not* is a missing Vulkan loader
/// (`libvulkan.so.1`, Debian/Ubuntu package `libvulkan1`) that the GPU driver
/// package does not pull in — reproduced on this crate's reference box, where
/// `nvidia-smi` worked, `/usr/share/vulkan/icd.d/nvidia_icd.json` was present,
/// and `GpuContext::try_new()` still returned a bare `None`. The error type is
/// what turns that into an actionable message; see `context::init_error`.
#[test]
fn initialization_reports_its_outcome() {
    match GpuContext::try_new_diagnosed() {
        Ok(ctx) => {
            assert!(!ctx.is_degraded());
            eprintln!(
                "p1_dispatch_gating: context acquired, class={}",
                ctx.perf_class().as_str()
            );
        }
        Err(err) => {
            let text = err.to_string();
            eprintln!("[skip] no adapter: {text}");
            // Whatever the reason, it must be *stated*. A silent `None` is the
            // failure mode this type exists to remove.
            assert!(!text.is_empty());
        }
    }
}
