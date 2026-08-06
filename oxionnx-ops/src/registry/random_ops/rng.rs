//! Shared pure-Rust PRNG for the `Random*` / `Multinomial` operator family.
//!
//! # Non-goals: bitwise ORT compatibility
//!
//! This is a splitmix64-seeded xorshift128+ generator -- small, dependency-free,
//! and passes the standard statistical batteries (xorshift128+ is the engine
//! behind, among others, the V8 and JDK `ThreadLocalRandom` generators). It
//! does **not**, and cannot, reproduce onnxruntime's bit-for-bit output:
//! ORT draws from the C++ `<random>` Mersenne Twister / `normal_distribution`,
//! whose exact bit sequence is implementation-defined even across compilers
//! and standard-library versions -- there is no "the" reference sequence to
//! match. The ONNX spec for `RandomNormal`/`RandomUniform` requires
//! *distributional* correctness (independent draws from the named
//! distribution with the given parameters), not a specific bit sequence, so
//! this is spec-conformant.
//!
//! What this implementation **does** guarantee: given the same `seed`
//! attribute (and the same node name, shape and sample count), the output is
//! reproducible run-to-run and platform-to-platform, since every step is
//! ordinary wrapping integer arithmetic with no OS or hardware source of
//! entropy involved.
//!
//! # Seed resolution
//!
//! ONNX marks `seed` optional on every operator in this family ("if not
//! specified we will auto generate one" -- ai.onnx `RandomNormal` op doc),
//! i.e. non-determinism is spec-legal when the attribute is absent. When
//! present, [`resolve_seed`] is used so this engine is deterministic for that
//! node.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use oxionnx_core::Attributes;

/// SplitMix64: used only to expand a `u64` seed into two well-distributed
/// `u64` lanes for xorshift128+'s state (feeding xorshift128+ small/sequential
/// raw seeds such as `0, 1, 2, ..` directly would otherwise start highly
/// correlated streams across nodes whose only difference is a `+1` seed).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xorshift128+ generator: two `u64` state words, four shifts and an add per
/// output word. See Vigna, "Further scramblings of Marsaglia's xorshift
/// generators" (2017).
pub(crate) struct Rng {
    s0: u64,
    s1: u64,
}

impl Rng {
    /// Seed the generator. `stream` differentiates generators that must not
    /// draw correlated sequences despite sharing a `seed` -- e.g. the two
    /// independent uniforms one Box-Muller normal draw consumes call
    /// [`Self::next_f32`] on the *same* `Rng`, so `stream` is not needed
    /// there; it exists for the fallback "no seed attribute" wall-clock path
    /// in [`resolve_seed`], where two sibling nodes executed within the same
    /// nanosecond would otherwise collide.
    pub(crate) fn new(seed: u64, stream: u64) -> Self {
        let mut sm = seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let s0 = splitmix64(&mut sm);
        let mut s1 = splitmix64(&mut sm);
        if s0 == 0 && s1 == 0 {
            // All-zero is xorshift's one forbidden fixed point (it would
            // generate an all-zero stream forever); splitmix64 output can in
            // principle (astronomically unlikely, but let's not rely on
            // "unlikely") collide to zero for both lanes.
            s1 = 1;
        }
        Self { s0, s1 }
    }

    fn next_u64(&mut self) -> u64 {
        let x0 = self.s0;
        let y = self.s1;
        self.s0 = y;
        let mut x = x0;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        self.s1 = x;
        x.wrapping_add(y)
    }

    /// Uniform `f32` in `[0, 1)`, using the top 24 bits (a full `f32`
    /// mantissa's worth of entropy).
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform `f32` in `[lo, hi)`.
    pub(crate) fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Standard-normal (mean 0, std 1) sample via the Box-Muller transform.
    pub(crate) fn next_standard_normal(&mut self) -> f32 {
        // `next_f32() == 0.0` would make `ln(u1) == -inf`; clamp to the
        // smallest representable positive gap instead of resampling (keeps
        // this a pure function of a fixed number of `next_u64` calls, so a
        // given seed always consumes exactly the same amount of state
        // regardless of which values happen to land on the boundary).
        let u1 = self.next_f32().max(f32::EPSILON);
        let u2 = self.next_f32();
        let r = (-2.0_f64 * (u1 as f64).ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2 as f64;
        (r * theta.cos()) as f32
    }
}

/// Hash a string to a `u64` (used to disambiguate the wall-clock fallback
/// seed in [`resolve_seed`] across distinct unseeded nodes).
fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Resolve the effective PRNG seed for a `Random*`/`Multinomial` node.
///
/// If the node's `seed` (float) attribute is present, its bit pattern is used
/// directly -- deterministic, including across an explicit `seed = 0.0`
/// (distinguished from "absent" by checking the attribute map directly rather
/// than a fill-in-a-default accessor). If absent, the spec permits
/// non-determinism; the fallback combines wall-clock time with a hash of the
/// node's name so that two unseeded nodes executed within the same
/// nanosecond (realistic on a fast test run) still diverge.
pub(crate) fn resolve_seed(attrs: &Attributes, node_name: &str) -> u64 {
    if let Some(&seed) = attrs.floats.get("seed") {
        return seed.to_bits() as u64;
    }
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    t ^ hash_str(node_name).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream_is_deterministic() {
        let mut a = Rng::new(42, 0);
        let mut b = Rng::new(42, 0);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1, 0);
        let mut b = Rng::new(2, 0);
        let seq_a: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn different_streams_same_seed_diverge() {
        let mut a = Rng::new(7, 0);
        let mut b = Rng::new(7, 1);
        let seq_a: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn next_f32_stays_in_unit_interval() {
        let mut rng = Rng::new(123, 0);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "next_f32 out of [0,1): {v}");
        }
    }

    #[test]
    fn zero_seed_and_zero_stream_does_not_lock_to_the_zero_fixed_point() {
        // seed=0, stream=0 drives splitmix64 from state 0 -- exercise it
        // explicitly since it is the one input `Rng::new` special-cases.
        let mut rng = Rng::new(0, 0);
        let mut saw_nonzero = false;
        for _ in 0..8 {
            if rng.next_u64() != 0 {
                saw_nonzero = true;
            }
        }
        assert!(saw_nonzero, "generator locked to the all-zero state");
    }

    #[test]
    fn resolve_seed_is_stable_for_an_explicit_seed_including_zero() {
        let mut attrs = Attributes::default();
        attrs.floats.insert("seed".into(), 0.0);
        let a = resolve_seed(&attrs, "node_a");
        let b = resolve_seed(&attrs, "node_a");
        assert_eq!(a, b, "an explicit seed (even 0.0) must be deterministic");
    }

    #[test]
    fn resolve_seed_differs_by_attribute_value() {
        let mut attrs_a = Attributes::default();
        attrs_a.floats.insert("seed".into(), 1.0);
        let mut attrs_b = Attributes::default();
        attrs_b.floats.insert("seed".into(), 2.0);
        assert_ne!(
            resolve_seed(&attrs_a, "n"),
            resolve_seed(&attrs_b, "n"),
            "different seed attributes must resolve to different seeds"
        );
    }
}
