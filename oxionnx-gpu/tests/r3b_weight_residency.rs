//! [r3b] Device-backed proof that invariant operands are uploaded once per
//! context and bound from the device on every dispatch after that.
//!
//! The claim these tests exist to settle is quantitative, so they assert on a
//! counter rather than on a duration: `GpuContext::resident_counters()` is
//! cumulative and monotonic, and the difference of two snapshots around a
//! dispatch is exactly "what did this dispatch upload". Run 1 must upload the
//! weight; run 2 must upload **zero** bytes of it.
//!
//! Timing is printed, never asserted — a wall-clock assertion on shared
//! hardware is a flake generator. Reproduce the numbers with
//! `cargo nextest run -p oxionnx-gpu -E 'test(r3b_)' --no-capture`.
//!
//! Every test degrades to a no-op when no adapter is available, the convention
//! the rest of this suite uses.

use std::time::Instant;

use oxionnx_core::Tensor;
use oxionnx_gpu::{
    gpu_conv2d_implicit_resident_async, gpu_gemm_nt_resident_async, ConvActivation, GpuContext,
    WeightKeys,
};

fn context() -> Option<GpuContext> {
    GpuContext::try_new()
}

/// Deterministic, signed, non-monotonic fill: a plain ramp hides index bugs
/// because many wrong indices carry a plausible value.
fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            ((x % 23) as f32) * 0.037 - 0.4
        })
        .collect()
}

/// A convolution whose weight dominates its input, which is the shape the whole
/// optimization is about: `[1,512,16,16] * [512,512,3,3]` is a 9.44 MB weight
/// against a 0.52 MB activation, and 1.21 GFLOP — comfortably above the 10
/// MFLOP dispatch gate.
const C: usize = 512;
const HW: usize = 16;
const WEIGHT_BYTES: u64 = (C * C * 3 * 3 * 4) as u64;
const BIAS_BYTES: u64 = (C * 4) as u64;

fn conv_operands() -> (Tensor, Tensor, Tensor) {
    (
        Tensor::new(fill(C * HW * HW, 7), vec![1, C, HW, HW]),
        Tensor::new(fill(C * C * 3 * 3, 13), vec![C, C, 3, 3]),
        Tensor::new(fill(C, 29), vec![C]),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_conv(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    keys: WeightKeys<'_>,
) -> Option<Tensor> {
    pollster::block_on(gpu_conv2d_implicit_resident_async(
        ctx,
        input,
        weight,
        Some(bias),
        keys,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
    ))
}

/// **The measurement.** Two identical convolutions through one context: the
/// first uploads the weight and the bias, the second uploads neither.
#[test]
fn r3b_a_second_conv_uploads_no_weight_bytes_at_all() {
    let Some(ctx) = context() else {
        eprintln!("skip: no GPU adapter available");
        return;
    };
    let (input, weight, bias) = conv_operands();
    let keys = WeightKeys::new(Some("conv.weight"), Some("conv.bias"));

    // Warm-up, deliberately un-keyed: this kernel rebuilds its pipeline per
    // call (see `shaders::kernel_support`), and the first WGSL compilation on a
    // cold driver costs tens of milliseconds. Paying it here keeps that cost
    // out of the two timings below, which are supposed to differ by an upload
    // and nothing else. Un-keyed so it leaves the residency counters idle —
    // and so its result is the **pre-residency** answer to compare against.
    let Some(unkeyed) = run_conv(&ctx, &input, &weight, &bias, WeightKeys::default()) else {
        eprintln!("skip: adapter declined a 1.21 GFLOP conv");
        return;
    };

    let before_first = ctx.resident_counters();
    let started = Instant::now();
    let Some(first) = run_conv(&ctx, &input, &weight, &bias, keys) else {
        eprintln!("skip: adapter declined a 1.21 GFLOP conv");
        return;
    };
    let first_elapsed = started.elapsed();
    let first_delta = ctx.resident_counters().since(before_first);

    assert_eq!(
        first_delta.uploaded_bytes,
        WEIGHT_BYTES + BIAS_BYTES,
        "the first dispatch must upload the weight and the bias exactly once",
    );
    assert_eq!(first_delta.misses, 2, "two operands, neither seen before");
    assert_eq!(first_delta.hits, 0);

    let before_second = ctx.resident_counters();
    let started = Instant::now();
    let second = run_conv(&ctx, &input, &weight, &bias, keys).expect("the same conv must dispatch");
    let second_elapsed = started.elapsed();
    let second_delta = ctx.resident_counters().since(before_second);

    assert_eq!(
        second_delta.uploaded_bytes, 0,
        "an invariant weight must not cross the bus twice",
    );
    assert_eq!(second_delta.misses, 0);
    assert_eq!(second_delta.hits, 2, "both operands served from the device");

    // Numerics, and the assertion that actually carries the "no kernel math
    // changed" claim: the keyed dispatch must agree with the **un-keyed** one,
    // which is the code path as it was before residency existed. Exact
    // equality, not a tolerance — the same bytes reach the same binding, the
    // shader is the same, and the accumulation order does not depend on where a
    // buffer came from, so anything but equality is a bug rather than drift.
    assert_eq!(unkeyed.shape, first.shape);
    assert_eq!(
        unkeyed.data, first.data,
        "a resident weight must compute exactly what uploading it computed",
    );
    assert_eq!(first.shape, second.shape);
    assert_eq!(
        first.data, second.data,
        "and must keep computing it on every later dispatch",
    );

    // The cache holds exactly what it was asked to hold.
    assert_eq!(ctx.resident_len(), 2);
    assert_eq!(ctx.resident_bytes(), WEIGHT_BYTES + BIAS_BYTES);
    assert!(
        ctx.live_gpu_bytes() >= ctx.resident_bytes(),
        "resident buffers are part of the live-byte total, not beside it",
    );

    // Informational only, never asserted. This is the honest A/B for the
    // change: the same convolution, the same binary, the same second, keyed
    // against un-keyed — and un-keyed *is* the pre-residency behaviour, since
    // that path is unchanged code. Several samples each, because one is noise.
    let sample = |keys: WeightKeys<'_>| -> Vec<f64> {
        (0..4)
            .map(|_| {
                let started = Instant::now();
                let _ = run_conv(&ctx, &input, &weight, &bias, keys);
                started.elapsed().as_secs_f64() * 1e3
            })
            .collect()
    };
    let uploading = sample(WeightKeys::default());
    let resident = sample(keys);
    eprintln!(
        "r3b conv [1,{C},{HW},{HW}] x [{C},{C},3,3] (pipeline warm, ms): \
         first-upload {:.3}, first-resident {:.3}, \
         uploading {uploading:.3?}, resident {resident:.3?}",
        first_elapsed.as_secs_f64() * 1e3,
        second_elapsed.as_secs_f64() * 1e3,
    );
}

/// The un-keyed entry point must keep behaving exactly as it did: nothing
/// becomes resident by accident, and the totals still move.
#[test]
fn r3b_an_unkeyed_conv_still_uploads_everything_every_time() {
    let Some(ctx) = context() else {
        eprintln!("skip: no GPU adapter available");
        return;
    };
    let (input, weight, bias) = conv_operands();

    let before = ctx.uploaded_bytes();
    let Some(_) = run_conv(&ctx, &input, &weight, &bias, WeightKeys::default()) else {
        eprintln!("skip: adapter declined a 1.21 GFLOP conv");
        return;
    };
    let first_total = ctx.uploaded_bytes() - before;
    let _ = run_conv(&ctx, &input, &weight, &bias, WeightKeys::default());
    let second_total = ctx.uploaded_bytes() - before - first_total;

    assert!(first_total >= WEIGHT_BYTES);
    assert_eq!(
        first_total, second_total,
        "without keys every dispatch uploads the same bytes over again",
    );
    assert_eq!(ctx.resident_len(), 0, "no key, no residency");
    assert!(ctx.resident_counters().is_idle());
}

/// Requirement: a resident weight is not reclaimable. The buffer pool evicts
/// under budget pressure; a weight must survive that, because nothing in the
/// pool's ownership model can reach it.
#[test]
fn r3b_budget_pressure_evicts_pooled_buffers_and_never_a_weight() {
    let Some(ctx) = context() else {
        eprintln!("skip: no GPU adapter available");
        return;
    };
    let (input, weight, bias) = conv_operands();
    let keys = WeightKeys::new(Some("conv.weight"), Some("conv.bias"));

    let Some(first) = run_conv(&ctx, &input, &weight, &bias, keys) else {
        eprintln!("skip: adapter declined a 1.21 GFLOP conv");
        return;
    };
    let resident_bytes = ctx.resident_bytes();
    assert_eq!(resident_bytes, WEIGHT_BYTES + BIAS_BYTES);

    // Squeeze the budget to exactly what is live: the next dispatch cannot fit
    // its input, output and staging buffers, so it must decline — after
    // reclaiming every idle pooled buffer it can find.
    let original_budget = ctx.gpu_byte_budget();
    ctx.set_gpu_byte_budget(ctx.live_gpu_bytes());
    let declined = run_conv(&ctx, &input, &weight, &bias, keys);
    assert!(
        declined.is_none(),
        "a dispatch with no room for its activations must decline to the CPU",
    );
    assert_eq!(
        ctx.resident_bytes(),
        resident_bytes,
        "reclaiming for a squeezed budget must not touch a resident weight",
    );
    assert_eq!(ctx.resident_len(), 2);

    // And the weight is still usable, unchanged, once there is room again.
    ctx.set_gpu_byte_budget(original_budget);
    let before = ctx.resident_counters();
    let after_squeeze =
        run_conv(&ctx, &input, &weight, &bias, keys).expect("room restored, dispatch must succeed");
    assert_eq!(
        ctx.resident_counters().since(before).uploaded_bytes,
        0,
        "the weight survived the squeeze, so nothing needed re-uploading",
    );
    assert_eq!(
        first.data, after_squeeze.data,
        "a weight that survived eviction pressure must still compute the same values",
    );

    // The one thing that *does* release a weight: asking. A caller that has
    // switched models needs the memory back, and nothing else in the crate
    // will take it.
    let live_before_clear = ctx.live_gpu_bytes();
    ctx.clear_resident_buffers();
    assert_eq!(ctx.resident_len(), 0);
    assert_eq!(ctx.resident_bytes(), 0);
    assert!(
        ctx.live_gpu_bytes() <= live_before_clear.saturating_sub(resident_bytes),
        "clearing must return the weight's bytes to the budget, not just forget them",
    );

    let before = ctx.resident_counters();
    let after_clear =
        run_conv(&ctx, &input, &weight, &bias, keys).expect("a cleared cache re-uploads and runs");
    assert_eq!(
        ctx.resident_counters().since(before).uploaded_bytes,
        resident_bytes,
        "a cleared weight is uploaded again on next use",
    );
    assert_eq!(first.data, after_clear.data);
}

/// `Gemm`'s `B` is the other initializer-backed operand the engine dispatches
/// — ArcFace's embedding head. Same contract, different kernel.
#[test]
fn r3b_a_second_gemm_uploads_no_b_matrix_bytes() {
    let Some(ctx) = context() else {
        eprintln!("skip: no GPU adapter available");
        return;
    };
    // `[1, 4096] x [512, 4096]^T` — a 8.4 MB `B` against a 16 KB `A`, the
    // shape of an embedding head.
    const M: usize = 1;
    const K: usize = 4096;
    const N: usize = 512;
    let a = fill(M * K, 3);
    let b = fill(N * K, 11);
    let c = fill(N, 17);
    let keys = WeightKeys::new(Some("fc.weight"), Some("fc.bias"));

    // The pre-residency answer, from the un-keyed path, to compare against.
    let Some(unkeyed) = pollster::block_on(gpu_gemm_nt_resident_async(
        &ctx,
        &a,
        M,
        K,
        &b,
        N,
        Some(&c),
        1.0,
        1.0,
        WeightKeys::default(),
    )) else {
        eprintln!("skip: adapter declined the gemm");
        return;
    };

    let before_first = ctx.resident_counters();
    let first = pollster::block_on(gpu_gemm_nt_resident_async(
        &ctx,
        &a,
        M,
        K,
        &b,
        N,
        Some(&c),
        1.0,
        1.0,
        keys,
    ))
    .expect("the same gemm must dispatch");
    assert_eq!(
        unkeyed, first,
        "a resident B must compute exactly what uploading it computed",
    );
    let first_delta = ctx.resident_counters().since(before_first);
    assert_eq!(
        first_delta.uploaded_bytes,
        ((N * K + N) * 4) as u64,
        "the first gemm uploads B and C",
    );

    let before_second = ctx.resident_counters();
    let second = pollster::block_on(gpu_gemm_nt_resident_async(
        &ctx,
        &a,
        M,
        K,
        &b,
        N,
        Some(&c),
        1.0,
        1.0,
        keys,
    ))
    .expect("the same gemm must dispatch");
    let second_delta = ctx.resident_counters().since(before_second);

    assert_eq!(second_delta.uploaded_bytes, 0);
    assert_eq!(second_delta.hits, 2);
    assert_eq!(
        first, second,
        "a resident B must give bit-identical results"
    );
}

/// One identity, two different byte lengths: the cache must refuse to serve the
/// first entry for the second request, and must not overwrite it either. A
/// caller can only get here by breaking its own promise, and the safe answer is
/// a per-dispatch upload — visible as upload bytes that keep growing.
#[test]
fn r3b_a_reused_identity_with_different_bytes_falls_back_to_uploading() {
    let Some(ctx) = context() else {
        eprintln!("skip: no GPU adapter available");
        return;
    };
    let (input, weight, bias) = conv_operands();
    // Both convolutions claim the identity "w", but the second one's weight is
    // a different tensor with a different length.
    let keys = WeightKeys::new(Some("w"), None);
    let Some(_) = run_conv(&ctx, &input, &weight, &bias, keys) else {
        eprintln!("skip: adapter declined a 1.21 GFLOP conv");
        return;
    };
    assert_eq!(ctx.resident_bytes(), WEIGHT_BYTES);

    let small_input = Tensor::new(fill(64 * 16 * 16, 5), vec![1, 64, 16, 16]);
    let small_weight = Tensor::new(fill(64 * 64 * 3 * 3, 19), vec![64, 64, 3, 3]);
    let small_bias = Tensor::new(fill(64, 23), vec![64]);
    let before = ctx.resident_counters();
    let conflicted = run_conv(&ctx, &small_input, &small_weight, &small_bias, keys);
    if conflicted.is_none() {
        eprintln!("skip: adapter declined the smaller conv");
        return;
    }
    let delta = ctx.resident_counters().since(before);
    assert_eq!(
        delta.uploaded_bytes,
        (64 * 64 * 3 * 3 * 4) as u64,
        "a conflicting identity uploads for that dispatch alone",
    );
    assert_eq!(delta.hits, 0);
    assert_eq!(
        ctx.resident_bytes(),
        WEIGHT_BYTES,
        "the original entry must survive a conflicting request untouched",
    );
}
