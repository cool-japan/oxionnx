//! [w2-f16] Deriving each half-precision kernel from the `f32` kernel it must
//! stay in step with.
//!
//! # Why the f16 shaders are derived rather than written
//!
//! `conv2d.rs`'s implicit-GEMM body is ~250 lines of index arithmetic that took
//! measurement to get right (the `(ky, kx, ic)` loop nest, the `vec4` block
//! tiling, the ragged-tile zero padding). A hand-written half-precision copy
//! would be a second place for all of it to be wrong, and — worse — a second
//! place for it to *drift*: a fix to one would silently not reach the other,
//! and the `f16` path would quietly stop matching the `f32` path it is checked
//! against.
//!
//! So there is exactly one copy of every kernel body in this crate, in `f32`,
//! and the `f16` variant is that source with a short, enumerated list of
//! substitutions applied. Two properties fall out:
//!
//! * **The `f32` source is untouched.** Not "regenerated identically" — the
//!   same `const &str` the tree has always had, byte for byte, which `git diff`
//!   shows directly. A context with the toggle off compiles and dispatches the
//!   same shader it did before this module existed.
//! * **A broken anchor is a decline, not a miscompile.** Every rule below
//!   declares how many times it must match. If an edit to the `f32` source
//!   moves an anchor, [`apply`] returns `None`, the kernel reports no `f16`
//!   variant, and every dispatch takes the `f32` path — visibly slower, never
//!   wrong. A silent no-op substitution (which is what a bare `str::replace`
//!   would give) is the one failure mode this design refuses to have.
//!
//! # The rounding points, in the substitutions themselves
//!
//! Each rule below is annotated with whether it changes numerics. In summary,
//! and identically for both kernels:
//!
//! * weights are `f16` **in storage** — narrowed host-side once per session,
//!   never per dispatch (`context::weight_format`);
//! * activations stay `f32` in storage and are narrowed to `f16` as they are
//!   read (conv: while staging the input tile; gemm: at the multiply);
//! * the **product** is evaluated in `f16`;
//! * every product is widened with `f32(...)` **before** it reaches the
//!   accumulator, which stays `f32` — so a `K = 9216` reduction still sums at
//!   single precision;
//! * bias, the fused activation epilogue, and `alpha`/`beta` are untouched and
//!   remain `f32`.

use std::sync::OnceLock;

/// One textual substitution, with the number of matches that proves it landed.
struct Rule {
    /// Text to find in the `f32` source.
    from: String,
    /// Text to put in its place.
    to: String,
    /// Exactly how many times `from` must occur. A mismatch fails the whole
    /// derivation — see the module docs.
    expect: usize,
}

impl Rule {
    fn new(from: impl Into<String>, to: impl Into<String>, expect: usize) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            expect,
        }
    }
}

/// Apply `rules` to `src`, or `None` if any rule did not match exactly as many
/// times as it declared.
fn apply(src: &str, rules: &[Rule]) -> Option<String> {
    let mut out = String::from("enable f16;\n");
    let mut body = src.to_string();
    for rule in rules {
        if body.matches(rule.from.as_str()).count() != rule.expect {
            return None;
        }
        body = body.replace(rule.from.as_str(), rule.to.as_str());
    }
    out.push_str(&body);
    Some(out)
}

/// The `f16` form of the conv2d implicit-GEMM kernel, or `None` if the `f32`
/// source has drifted away from the anchors below.
fn conv2d_rules() -> Vec<Rule> {
    let mut rules = vec![
        // ROUNDING POINT 1 — the weight operand is f16 *in storage*. The host
        // narrows it once per session at upload (`WeightFormat::convert`).
        Rule::new(
            "var<storage, read> wgt: array<f32>",
            "var<storage, read> wgt: array<f16>",
            1,
        ),
        // Both staged tiles become half-width. Beyond the obvious bandwidth
        // halving this cuts workgroup memory from 4 KiB to 2 KiB per tile,
        // which is what lets more workgroups be resident per core.
        Rule::new(
            "var<workgroup> tile_w: array<array<vec4<f32>, 16>, 16>;",
            "var<workgroup> tile_w: array<array<vec4<f16>, 16>, 16>;",
            1,
        ),
        Rule::new(
            "var<workgroup> tile_x: array<array<vec4<f32>, 16>, 16>;",
            "var<workgroup> tile_x: array<array<vec4<f16>, 16>, 16>;",
            1,
        ),
        // The weight loader now returns what the storage buffer holds; only its
        // zero-padding literal needs a type. No rounding here — the value is
        // already f16 on the device.
        Rule::new(
            "fn load_weight(oc: u32, a_col: u32, channel_ok: bool) -> f32 {",
            "fn load_weight(oc: u32, a_col: u32, channel_ok: bool) -> f16 {",
            1,
        ),
        Rule::new(
            "        return wgt[oc * params.k_total + a_col];\n    }\n    return 0.0;",
            "        return wgt[oc * params.k_total + a_col];\n    }\n    return f16(0.0);",
            1,
        ),
        // ROUNDING POINT 2 — the activation stays f32 in its buffer (residency
        // semantics are untouched) and is narrowed here, as it is staged.
        Rule::new(
            "fn load_input(plane: u32, spatial: i32, ok: bool) -> f32 {",
            "fn load_input(plane: u32, spatial: i32, ok: bool) -> f16 {",
            1,
        ),
        Rule::new(
            "        return inp[plane + u32(spatial)];\n    }\n    return 0.0;",
            "        return f16(inp[plane + u32(spatial)]);\n    }\n    return f16(0.0);",
            1,
        ),
        Rule::new(
            "tile_w[ty][tx] = vec4<f32>(",
            "tile_w[ty][tx] = vec4<f16>(",
            1,
        ),
        Rule::new(
            "tile_x[ty][tx] = vec4<f32>(",
            "tile_x[ty][tx] = vec4<f16>(",
            1,
        ),
    ];

    // ROUNDING POINT 3 — the 4x4 register tile's multiplies. `a4` and `b4` are
    // now f16, so each product is evaluated at half precision and then widened
    // by the explicit `f32(...)` before it reaches the accumulator.
    //
    // The accumulators themselves are deliberately NOT rewritten: `var accIJ:
    // f32 = 0.0;` stays exactly as the f32 kernel declares it, which is what
    // keeps a K = 9216 reduction summing at single precision.
    const LANE: [&str; 4] = ["x", "y", "z", "w"];
    for (i, a) in LANE.iter().enumerate() {
        for (j, b) in LANE.iter().enumerate() {
            rules.push(Rule::new(
                format!("acc{i}{j} = acc{i}{j} + a4.{a} * b4.{b};"),
                format!("acc{i}{j} = acc{i}{j} + f32(a4.{a} * b4.{b});"),
                1,
            ));
        }
    }
    rules
}

/// The `f16` form of the `gemm_nt` kernel.
fn gemm_nt_rules() -> Vec<Rule> {
    vec![
        // ROUNDING POINT 1 — `B` (the [N, K] weight matrix) is f16 in storage.
        // `A` is the activation and stays f32, as does `c_buf` (the bias) and
        // every f32 term in the epilogue.
        Rule::new(
            "var<storage, read> b: array<f32>;",
            "var<storage, read> b: array<f16>;",
            1,
        ),
        // ROUNDING POINTS 2 and 3, in one expression: the f32 activation is
        // narrowed at the read (`f16(a[...])`), the product is evaluated in
        // f16, and `f32(...)` widens it before the f32 `acc` takes it. `var
        // acc: f32 = 0.0;` is untouched.
        Rule::new(
            "        acc = acc + a[a_base + i] * b[b_base + i];",
            "        acc = acc + f32(f16(a[a_base + i]) * b[b_base + i]);",
            1,
        ),
    ]
}

/// Memoize one derived shader for the process's lifetime.
///
/// The `&'static str` is what `kernel_support::build_pipeline` and `conv2d`'s
/// own pipeline cache key on, so the derivation happens at most once per
/// kernel per process — not per dispatch, and not per device.
fn derived(
    slot: &'static OnceLock<Option<String>>,
    src: &str,
    rules: &[Rule],
) -> Option<&'static str> {
    slot.get_or_init(|| apply(src, rules)).as_deref()
}

/// The half-precision conv2d source, derived from `conv2d.rs`'s `f32` source.
pub(super) fn conv2d_f16(f32_src: &str) -> Option<&'static str> {
    static SLOT: OnceLock<Option<String>> = OnceLock::new();
    derived(&SLOT, f32_src, &conv2d_rules())
}

/// The half-precision `gemm_nt` source, derived from `gemm.rs`'s `f32` source.
pub(super) fn gemm_nt_f16(f32_src: &str) -> Option<&'static str> {
    static SLOT: OnceLock<Option<String>> = OnceLock::new();
    derived(&SLOT, f32_src, &gemm_nt_rules())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard the whole design rests on: a rule that no longer matches must
    /// fail loudly rather than silently doing nothing.
    #[test]
    fn a_rule_that_does_not_match_fails_the_derivation() {
        let src = "var<storage, read> wgt: array<f32>;\n";
        assert!(apply(src, &[Rule::new("array<f32>", "array<f16>", 1)]).is_some());
        assert!(
            apply(src, &[Rule::new("array<f64>", "array<f16>", 1)]).is_none(),
            "an anchor that vanished must decline"
        );
        assert!(
            apply(src, &[Rule::new("array<f32>", "array<f16>", 2)]).is_none(),
            "a count that disagrees must decline"
        );
    }

    #[test]
    fn every_derived_source_enables_f16_first() {
        let out = apply("fn main() {}", &[]).expect("no rules always applies");
        assert!(
            out.starts_with("enable f16;\n"),
            "the extension directive must precede all declarations"
        );
    }

    /// Both real kernels must derive on the sources actually in the tree. This
    /// is what turns "the anchors are right" from a claim into a check that
    /// runs on every build, with no GPU involved.
    #[test]
    fn both_kernels_derive_from_the_sources_in_this_tree() {
        let conv = conv2d_f16(super::super::conv2d::CONV2D_SHADER)
            .expect("the conv2d f32 source must still carry every f16 anchor");
        let gemm = gemm_nt_f16(super::super::gemm::GEMM_NT_SHADER)
            .expect("the gemm_nt f32 source must still carry every f16 anchor");

        for (name, src) in [("conv2d", conv), ("gemm_nt", gemm)] {
            assert!(src.starts_with("enable f16;\n"), "{name}: no enable");
            assert!(
                !src.contains("vec4<f32>"),
                "{name}: a staged tile is still f32"
            );
        }

        // The accumulators — the one thing that must NOT have been narrowed.
        assert!(
            conv.contains("var acc00: f32 = 0.0;") && conv.contains("var acc33: f32 = 0.0;"),
            "conv2d accumulators must stay f32"
        );
        assert_eq!(
            conv.matches("f32(a4.").count(),
            16,
            "all sixteen products must be widened before accumulation"
        );
        assert!(
            gemm.contains("var acc: f32 = 0.0;"),
            "gemm accumulator must stay f32"
        );
        assert!(
            gemm.contains("f32(f16(a[a_base + i]) * b[b_base + i])"),
            "gemm must narrow A, multiply in f16, and widen before accumulating"
        );
        // The epilogues stay in f32 throughout.
        assert!(
            conv.contains("outp[row_base + c0] = activate(acc00 + bv);"),
            "conv2d bias + activation epilogue must be untouched"
        );
        assert!(
            gemm.contains("params.alpha * acc + params.beta * c_value(row, col)"),
            "gemm alpha/beta epilogue must be untouched"
        );
    }

    /// Deriving twice must hand back the very same memoized string, because
    /// `build_pipeline` keys its cache on the pointer.
    #[test]
    fn derivation_is_memoized_by_pointer() {
        let a = conv2d_f16(super::super::conv2d::CONV2D_SHADER).expect("derives");
        let b = conv2d_f16(super::super::conv2d::CONV2D_SHADER).expect("derives");
        assert!(std::ptr::eq(a, b), "the derived source must be memoized");
    }
}
