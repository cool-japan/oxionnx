//! Regression tests for the *runtime AVX2 dispatch mechanism itself*.
//!
//! Every other test that touches `simd_ops` (here and in
//! `tests/simd_expanded_test.rs`, `tests/conv_simd_test.rs`,
//! `tests/simd_softmax_layernorm_test.rs`, `attention/simd_tests.rs`, ...)
//! checks *numeric* correctness: SIMD output vs. a scalar reference. For
//! pure elementwise ops (add/mul/sub/div/relu/abs/neg/sqrt/...) that
//! comparison can never actually distinguish "the AVX2 branch ran" from
//! "the scalar fallback silently ran instead" -- the two are bit-identical
//! for those ops, since there is no reduction/reassociation for a numeric
//! diff to catch. A dispatcher bug that always falls through to scalar
//! (a typo'd feature string passed to `is_x86_feature_detected!`, an
//! inverted condition, a stray `cfg` that guards the whole AVX2 arm away,
//! ...) would pass every existing numeric test while silently discarding
//! the AVX2 speedup on every AVX2-capable machine forever. This is a
//! materially different, more serious failure than "the LLVM auto-vectorizer
//! baseline is a bit conservative" (the `-C target-cpu` codegen gap these
//! tests were added alongside) -- it is the hand-written fast path itself
//! going dark silently.
//!
//! These tests close that gap by instrumenting the two dispatch chokepoints
//! in `functions.rs` (`AVX2_DISPATCH_HITS`, `#[cfg(test)]`-only, see there)
//! and asserting the AVX2 arm was *actually entered* -- independent of
//! whatever `-C target-cpu`/`-C target-feature` baseline the ambient build
//! flags happen to set, since `is_x86_feature_detected!` is a *runtime*
//! CPUID check performed once per call, not a compile-time decision.
use super::functions::AVX2_DISPATCH_HITS;

fn avx2_hits() -> usize {
    AVX2_DISPATCH_HITS.with(|c| c.get())
}

/// `dispatch_binary`'s AVX2 arm backs `simd_add`/`simd_mul`/`simd_sub`/`simd_div`.
/// It must actually execute on a CPU that supports AVX2.
#[test]
fn avx2_binary_dispatch_is_actually_taken_when_supported() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!(
            "skipping: this CPU has no AVX2, so there is nothing for the \
             AVX2 dispatch arm to exercise here"
        );
        return;
    }
    let before = avx2_hits();
    // Length 9 = one full 8-lane AVX2 chunk plus a 1-element scalar
    // remainder, so a regression here also exercises each kernel's tail loop.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = [9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut out = [0.0f32; 9];
    super::simd_add(&a, &b, &mut out);
    assert_eq!(out, [10.0; 9], "simd_add produced a wrong result");

    let after = avx2_hits();
    assert!(
        after > before,
        "simd_add() did not take the AVX2 branch on an AVX2-capable CPU \
         (hit count before={before}, after={after}) -- the runtime dispatch \
         mechanism itself is broken (e.g. `is_x86_feature_detected!` guard \
         regressed), independent of any codegen/build-flag baseline"
    );
}

/// Same guard, for `dispatch_unary` (backs `simd_relu`/`simd_sigmoid`/`simd_tanh`/
/// `simd_gelu`/`simd_silu`/`simd_exp`/`simd_neg`/`simd_abs`/`simd_sqrt`/`simd_log`).
#[test]
fn avx2_unary_dispatch_is_actually_taken_when_supported() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("skipping: this CPU has no AVX2");
        return;
    }
    let before = avx2_hits();
    let mut data = [-3.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    super::simd_relu(&mut data);
    assert_eq!(
        data,
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "simd_relu produced a wrong result"
    );

    let after = avx2_hits();
    assert!(
        after > before,
        "simd_relu() did not take the AVX2 branch on an AVX2-capable CPU \
         (hit count before={before}, after={after})"
    );
}

/// Cross-check from a different angle: call the low-level AVX2 kernel
/// directly (bypassing the dispatcher entirely) and confirm it agrees with
/// an independently-written scalar reference, on a length that is not a
/// multiple of the 8-lane width so both the vectorised chunk loop and the
/// kernel's own internal scalar remainder tail run in the same call.
///
/// This is a lower-level guard than the two tests above: even if
/// `dispatch_binary`'s `is_x86_feature_detected!` guard were somehow
/// bypassed entirely, this proves the AVX2 kernel underneath it still
/// compiles, links, and computes correctly on real AVX2 hardware.
#[test]
fn avx2_kernel_matches_independent_scalar_reference() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("skipping: this CPU has no AVX2");
        return;
    }
    let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..19).map(|i| (18 - i) as f32 * 0.25 + 1.0).collect();
    let mut avx2_out = vec![0.0f32; 19];
    // SAFETY: AVX2 support confirmed by the `is_x86_feature_detected!` check above,
    // matching the exact safety contract documented on `avx2_impl::add`.
    unsafe {
        super::avx2::avx2_impl::add(&a, &b, &mut avx2_out);
    }
    // Independent reference, computed here rather than by calling back into
    // any of this crate's own scalar helpers, so this is a genuine
    // cross-check rather than a comparison against a shared implementation.
    let scalar_out: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(
        avx2_out, scalar_out,
        "AVX2 add kernel disagrees with an independently computed scalar sum"
    );
}
