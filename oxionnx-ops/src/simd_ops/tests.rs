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
//! The reduction-shaped ops (`simd_reduce_sum`/`_max`/`_min`,
//! `simd_dot_product`, `simd_softmax_inplace`, `simd_layer_norm`) have a
//! related but distinct blindness: their scalar fallbacks use compensated
//! (Kahan) summation while the AVX2 kernels reduce lane-parallel, so a small
//! numeric gap between the two is *expected* and already tolerance-checked
//! by the numeric tests referenced above. That makes a silent always-scalar
//! dispatch bug in one of *these* six functions even easier to miss than in
//! the bit-identical elementwise case: the "difference" a broken dispatcher
//! produces is exactly zero, which looks like an unusually clean pass rather
//! than a red flag.
//!
//! These tests close both gaps by instrumenting all eight of this module's
//! `is_x86_feature_detected!("avx2")` call sites in `functions.rs` --
//! the two shared chokepoints (`dispatch_binary`, `dispatch_unary`) and the
//! six standalone reduction/normalization functions above -- with a shared
//! counter (`AVX2_DISPATCH_HITS`, `#[cfg(test)]`-only, see there) and
//! asserting the AVX2 arm was *actually entered* -- independent of whatever
//! `-C target-cpu`/`-C target-feature` baseline the ambient build flags
//! happen to set, since `is_x86_feature_detected!` is a *runtime* CPUID
//! check performed once per call, not a compile-time decision.
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

// ---------------------------------------------------------------------
// The six standalone reduction/normalization dispatch sites below are not
// routed through `dispatch_binary`/`dispatch_unary` above -- each has its
// own inline `is_x86_feature_detected!` guard in `functions.rs` -- so they
// need their own dispatch-was-actually-taken coverage rather than being
// implied by the two tests above. See the `AVX2_DISPATCH_HITS` doc comment
// in `functions.rs` for why numeric-only comparison is an even weaker
// guarantee here than for the pure elementwise ops.
// ---------------------------------------------------------------------

/// `simd_reduce_sum`'s AVX2 arm is guarded by AVX2 *and* FMA (unlike the
/// plain elementwise ops above), so it must actually execute on a CPU that
/// supports both.
#[test]
fn avx2_reduce_sum_dispatch_is_actually_taken_when_supported() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!(
            "skipping: this CPU lacks AVX2+FMA, so there is nothing for \
             simd_reduce_sum's AVX2 arm to exercise here"
        );
        return;
    }
    let before = avx2_hits();
    // Length 19 = two full 8-lane AVX2 chunks plus a 3-element scalar
    // remainder, so a regression here also exercises the kernel's tail loop.
    let data: Vec<f32> = (0..19).map(|i| i as f32 * 0.5 - 2.0).collect();
    let expected: f32 = data.iter().sum();
    let got = super::simd_reduce_sum(&data);
    assert!(
        (got - expected).abs() < 1e-3,
        "simd_reduce_sum produced a wrong result: got {got}, expected ~{expected}"
    );

    let after = avx2_hits();
    assert!(
        after > before,
        "simd_reduce_sum() did not take the AVX2 branch on an AVX2+FMA-capable \
         CPU (hit count before={before}, after={after}) -- the runtime dispatch \
         mechanism itself is broken, independent of any codegen/build-flag baseline"
    );
}

/// `simd_reduce_max`'s AVX2 arm is guarded by AVX2 alone (no FMA
/// requirement, unlike `simd_reduce_sum`), so it must actually execute on
/// any AVX2-capable CPU.
#[test]
fn avx2_reduce_max_dispatch_is_actually_taken_when_supported() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("skipping: this CPU has no AVX2");
        return;
    }
    let before = avx2_hits();
    let mut data: Vec<f32> = (0..19).map(|i| (i as f32 - 9.0) * 0.7).collect();
    data[13] = 42.0; // an unambiguous max planted away from either end
    let got = super::simd_reduce_max(&data);
    assert_eq!(got, 42.0, "simd_reduce_max produced a wrong result");

    let after = avx2_hits();
    assert!(
        after > before,
        "simd_reduce_max() did not take the AVX2 branch on an AVX2-capable CPU \
         (hit count before={before}, after={after})"
    );
}

/// Same guard as `simd_reduce_max`, for `simd_reduce_min`.
#[test]
fn avx2_reduce_min_dispatch_is_actually_taken_when_supported() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("skipping: this CPU has no AVX2");
        return;
    }
    let before = avx2_hits();
    let mut data: Vec<f32> = (0..19).map(|i| (i as f32 - 9.0) * 0.7).collect();
    data[5] = -42.0; // an unambiguous min planted away from either end
    let got = super::simd_reduce_min(&data);
    assert_eq!(got, -42.0, "simd_reduce_min produced a wrong result");

    let after = avx2_hits();
    assert!(
        after > before,
        "simd_reduce_min() did not take the AVX2 branch on an AVX2-capable CPU \
         (hit count before={before}, after={after})"
    );
}

/// `simd_dot_product`'s AVX2 arm is guarded by AVX2+FMA, like
/// `simd_reduce_sum`.
#[test]
fn avx2_dot_product_dispatch_is_actually_taken_when_supported() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("skipping: this CPU lacks AVX2+FMA");
        return;
    }
    let before = avx2_hits();
    let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.3 - 1.0).collect();
    let b: Vec<f32> = (0..19).map(|i| (18 - i) as f32 * 0.2 + 0.5).collect();
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let got = super::simd_dot_product(&a, &b);
    assert!(
        (got - expected).abs() < 1e-2,
        "simd_dot_product produced a wrong result: got {got}, expected ~{expected}"
    );

    let after = avx2_hits();
    assert!(
        after > before,
        "simd_dot_product() did not take the AVX2 branch on an AVX2+FMA-capable \
         CPU (hit count before={before}, after={after})"
    );
}

/// `simd_softmax_inplace`'s AVX2 arm is guarded by AVX2+FMA.
#[test]
fn avx2_softmax_dispatch_is_actually_taken_when_supported() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("skipping: this CPU lacks AVX2+FMA");
        return;
    }
    let before = avx2_hits();
    let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    super::simd_softmax_inplace(&mut data);
    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "simd_softmax_inplace output does not sum to 1 (sum={sum})"
    );
    assert!(
        data.windows(2).all(|w| w[1] > w[0]),
        "simd_softmax_inplace should preserve strictly increasing input order"
    );

    let after = avx2_hits();
    assert!(
        after > before,
        "simd_softmax_inplace() did not take the AVX2 branch on an AVX2+FMA-capable \
         CPU (hit count before={before}, after={after})"
    );
}

/// `simd_layer_norm`'s AVX2 arm is guarded by AVX2+FMA.
#[test]
fn avx2_layer_norm_dispatch_is_actually_taken_when_supported() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("skipping: this CPU lacks AVX2+FMA");
        return;
    }
    let before = avx2_hits();
    let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let scale = [1.0f32; 9];
    super::simd_layer_norm(&mut data, &scale, None, 1e-5);
    let n = data.len() as f32;
    let mean: f32 = data.iter().sum::<f32>() / n;
    let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    assert!(
        mean.abs() < 1e-3,
        "simd_layer_norm output mean should be ~0, got {mean}"
    );
    assert!(
        (var - 1.0).abs() < 1e-2,
        "simd_layer_norm output variance should be ~1, got {var}"
    );

    let after = avx2_hits();
    assert!(
        after > before,
        "simd_layer_norm() did not take the AVX2 branch on an AVX2+FMA-capable \
         CPU (hit count before={before}, after={after})"
    );
}
