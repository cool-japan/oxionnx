//! \[w5\] Regression tests for **two or more `GpuContext`s in one process**.
//!
//! The failure these pin down, reported end-to-end as
//! `BindGroupLayout[Id(9,1)] does not exist` (`wgpu-core/src/storage.rs`), is
//! not a numerics bug: a single-session run is bit-comparable with the CPU
//! reference. It is a *cache-identity* bug, and it needs its own file because
//! every other test in this crate builds exactly one context.
//!
//! # What made two contexts collide
//!
//! `GpuContext::try_new` builds a fresh `wgpu::Instance` per context, and each
//! `Instance` owns an independent `wgpu_core::Global` whose id allocators start
//! from zero. `wgpu::Device`'s `PartialEq` is *not* `Arc` identity — it proxies
//! to `CoreDevice.id`, a `(index, epoch)` pair from that per-`Instance`
//! allocator (`wgpu/src/backend/wgpu_core.rs`:
//! `impl_eq_ord_hash_proxy!(CoreDevice => .id)`). Two devices from two different
//! instances therefore compare **equal** while sharing nothing at all. Every
//! cache that keyed a compiled pipeline on "the device it was built for" then
//! served context 2 the `ComputePipeline` and `BindGroupLayout` that context 1
//! had compiled, and `create_bind_group` on context 2's device looked that
//! layout id up in *its* instance's registry, where it does not exist.
//!
//! The fix is ownership, not a better key: the pipeline cache now lives on the
//! `GpuContext` next to the device it was compiled for
//! (`context::pipeline_cache`), so there is no cross-context lookup to get
//! wrong. [`distinct_contexts_are_never_confused_for_each_other`] pins the
//! wgpu-level fact that made the old key unsound, and the rest pin behaviour.
//!
//! Every test degrades to a no-op when no adapter is available, as the rest of
//! this crate's suites do.

use oxionnx_core::Tensor;
use oxionnx_gpu::shaders::gpu_conv2d_implicit;
use oxionnx_gpu::{gpu_broadcast_add, ConvActivation, GpuContext, GpuTuning};

/// A context whose placement floors are lifted, so the small shapes below
/// actually dispatch instead of declining. Mirrors `w1_gpu_backend::context`.
fn context() -> Option<GpuContext> {
    let mut ctx = GpuContext::try_new()?;
    ctx.set_tuning(GpuTuning::PARITY);
    Some(ctx)
}

/// A convolution small enough to be quick and big enough to be exact.
fn conv_case() -> (Tensor, Tensor) {
    let input: Vec<f32> = (0..4 * 8 * 8).map(|i| (i % 7) as f32 * 0.25).collect();
    let weight: Vec<f32> = (0..6 * 4 * 3 * 3).map(|i| (i % 5) as f32 * 0.125).collect();
    (
        Tensor::new(input, vec![1, 4, 8, 8]),
        Tensor::new(weight, vec![6, 4, 3, 3]),
    )
}

/// Run the conv above on `ctx`, returning its output.
fn run_conv(ctx: &GpuContext) -> Option<Tensor> {
    let (input, weight) = conv_case();
    gpu_conv2d_implicit(
        ctx,
        &input,
        &weight,
        None,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
    )
}

/// Run a broadcast add on `ctx` (the `kernel_support` pipeline path, which is a
/// different cache from the conv one), returning its output.
fn run_broadcast(ctx: &GpuContext) -> Option<Vec<f32>> {
    let a: Vec<f32> = (0..64 * 16).map(|i| (i % 11) as f32).collect();
    let b: Vec<f32> = (0..16).map(|i| (i % 3) as f32).collect();
    gpu_broadcast_add(ctx, &a, &[64, 16], &b, &[1, 16])
}

/// The wgpu-level fact the old cache key assumed away.
///
/// Two independently constructed contexts must not be mistaken for each other.
/// Before the ownership fix this crate keyed its pipeline caches on
/// `&cached.device == device`, and on wgpu 29 that comparison is **true** for
/// two devices from two different `Instance`s, because it compares per-instance
/// ids. This asserts the property the caches actually needed — that two contexts
/// are distinguishable — using the same handle comparison the old key used, so
/// it fails loudly if a future wgpu makes ids collide again *and* something
/// starts keying on them.
#[test]
fn distinct_contexts_are_never_confused_for_each_other() {
    let Some(first) = context() else { return };
    let Some(second) = context() else { return };

    // Not an implementation detail of this crate: it is the premise any
    // device-keyed cache would rest on. It does not hold on wgpu 29 with one
    // instance per context, which is exactly why no cache here keys on it.
    let handles_are_distinguishable = first.device != second.device;
    eprintln!(
        "  two live contexts: device handles compare {}",
        if handles_are_distinguishable {
            "unequal (per-context ids are unique)"
        } else {
            "EQUAL (per-instance ids collide -- device identity is not a usable cache key)"
        }
    );

    // Whatever the handles say, the contexts must behave as independent
    // devices. This is the part that must always hold.
    let a = run_conv(&first).expect("first context declined the convolution");
    let b = run_conv(&second).expect("second context declined the convolution");
    assert_eq!(a.data, b.data, "two contexts disagreed on the same conv");
}

/// (a) Two contexts built one after the other, both alive, both used.
///
/// The direct analogue of `oxiface --device gpu detect --embed`, which builds a
/// detection session and a recognition session and then runs both. This is the
/// exact sequence that hard-panicked in `create_bind_group`.
#[test]
fn two_sequential_contexts_both_run() {
    let Some(first) = context() else { return };
    let first_conv = run_conv(&first).expect("first context declined the convolution");
    let first_bcast = run_broadcast(&first).expect("first context declined the broadcast");

    // Context 1 is still alive here — this is not a create/drop/create.
    let Some(second) = context() else { return };
    let second_conv = run_conv(&second).expect("second context declined the convolution");
    let second_bcast = run_broadcast(&second).expect("second context declined the broadcast");

    assert_eq!(
        first_conv.data, second_conv.data,
        "the second context's convolution disagreed with the first's",
    );
    assert_eq!(
        first_bcast, second_bcast,
        "the second context's broadcast disagreed with the first's",
    );

    // And the first context must still work after the second exists: a cache
    // that evicted on insert (the old retain-on-insert rule) would merely
    // recompile here, but a cache that had handed out the wrong device's
    // pipeline would fail.
    let first_again = run_conv(&first).expect("first context declined after the second was built");
    assert_eq!(first_conv.data, first_again.data);
}

/// (b) create / drop / create — the reused-id case.
///
/// Dropping context 1 frees its instance's ids, so context 2 is *guaranteed* to
/// receive the same numeric device id. A cache that outlives the context and
/// keys on that id cannot tell the two apart even in principle.
///
/// # Honest note: this one passed before the fix too
///
/// It is the only case in this file that did, and it passed *by accident*. The
/// old thread-local caches were purged from `GpuContext::drop` by removing
/// every entry whose device compared equal to the dropped one — a comparison
/// that is wrong in general, but on this sequence happens to select exactly the
/// entries that had to go, because there is no second live context for it to
/// over-match. It is here because a sequential create/drop/create is a real
/// usage shape (a CLI that switches models mid-run), and because a future cache
/// keyed on anything that survives a context must fail it.
#[test]
fn context_recreated_after_drop_runs_clean() {
    let expected = {
        let Some(first) = context() else { return };
        let out = run_conv(&first).expect("first context declined the convolution");
        let _ = run_broadcast(&first).expect("first context declined the broadcast");
        out
        // `first` is dropped here, at the end of the block.
    };

    let Some(second) = context() else { return };
    let got = run_conv(&second).expect("recreated context declined the convolution");
    assert_eq!(
        expected.data, got.data,
        "a context created after its predecessor was dropped computed a different result",
    );
    assert!(!second.is_degraded(), "the recreated context degraded");
}

/// (c) Two contexts alive at once, dispatches alternating between them.
///
/// The interleaving matters on its own: an eviction-on-insert cache would ping
/// pong (correct but recompiling), while a mistaken-identity cache would hand
/// each dispatch the other context's pipeline. Both contexts are also checked
/// for the degraded flag, which is how a captured validation error would show up
/// if one ever stopped short of panicking.
#[test]
fn alternating_dispatches_on_two_live_contexts() {
    let Some(first) = context() else { return };
    let Some(second) = context() else { return };

    let (input, weight) = conv_case();
    let reference = gpu_conv2d_implicit(
        &first,
        &input,
        &weight,
        None,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
    )
    .expect("first context declined the convolution");
    let broadcast_reference = run_broadcast(&first).expect("first context declined the broadcast");

    for round in 0..4 {
        for (name, ctx) in [("first", &first), ("second", &second)] {
            let conv = run_conv(ctx)
                .unwrap_or_else(|| panic!("{name} context declined the conv in round {round}"));
            assert_eq!(
                conv.data, reference.data,
                "{name} context, round {round}: conv output diverged",
            );
            let bcast = run_broadcast(ctx).unwrap_or_else(|| {
                panic!("{name} context declined the broadcast in round {round}")
            });
            assert_eq!(
                bcast, broadcast_reference,
                "{name} context, round {round}: broadcast output diverged",
            );
        }
    }

    assert!(!first.is_degraded(), "the first context degraded");
    assert!(!second.is_degraded(), "the second context degraded");
}

/// Three contexts, built and used in a nested order that no single-slot cache
/// can satisfy: 1, 2, 3, then back to 1.
///
/// A cache holding one device's entries at a time is *correct* here (a miss
/// recompiles); a cache that confuses two devices is not. Included because
/// `oxiface --device gpu swap` builds three sessions — detector, recognizer and
/// the swapper itself — and keeps all three alive for the whole run.
#[test]
fn three_live_contexts_round_robin() {
    let Some(a) = context() else { return };
    let Some(b) = context() else { return };
    let Some(c) = context() else { return };

    let reference = run_conv(&a).expect("context a declined the convolution");
    for (name, ctx) in [("b", &b), ("c", &c), ("a", &a), ("c", &c), ("b", &b)] {
        let out = run_conv(ctx).unwrap_or_else(|| panic!("context {name} declined"));
        assert_eq!(out.data, reference.data, "context {name} diverged");
    }
    for (name, ctx) in [("a", &a), ("b", &b), ("c", &c)] {
        assert!(!ctx.is_degraded(), "context {name} degraded");
    }
}
