//! Post-transform helpers for ONNX-ML operators.
//!
//! Provides the `PostTransform` enum and `apply_post_transform` for applying
//! softmax, logistic, probit, and related transforms to score buffers.

use oxionnx_core::OnnxError;

/// Apply softmax row-wise to a \[N, C\] buffer stored in row-major order.
pub(super) fn softmax_rows(data: &mut [f32], n: usize, c: usize) {
    for row in 0..n {
        let offset = row * c;
        let row_slice = &mut data[offset..offset + c];
        let max_val = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row_slice.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row_slice.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Apply logistic (sigmoid) element-wise.
pub(super) fn logistic_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Apply probit transform element-wise (approximate inverse of the standard normal CDF).
/// Uses the Abramowitz & Stegun 26.2.23 rational approximation.
pub(super) fn probit_inplace(data: &mut [f32]) {
    // Distance from the nearest domain edge (0 or 1) that the clamp is
    // willing to tolerate.
    const EPS: f32 = 1e-7;

    for v in data.iter_mut() {
        let raw = *v;
        let upper = raw >= 0.5;

        // Work in the tail probability `q = min(p, 1-p)`, derived from the
        // *original* value rather than from an already-clamped `p`. The
        // nearest f32 to the literal `1.0 - 1e-7` is one ULP below 1.0
        // (`f32::EPSILON`, ~1.1921e-7 away from 1.0, not `1e-7` away); a
        // clamp-then-subtract sequence (`p = v.clamp(EPS, 1.0 - EPS)` then
        // `1.0 - p`) recomputes `1.0 - p` from that already-rounded `p` and
        // recovers `f32::EPSILON` instead of the intended `EPS` -- a ~19%
        // relative error from catastrophic cancellation, well outside this
        // approximation's own error budget. Subtracting from the raw value
        // first keeps that subtraction exact (Sterbenz's lemma covers
        // `1.0 - raw` whenever `raw` is itself a float, which it always is
        // here) and clamping the *result* symmetrically means any
        // out-of-domain input collapses onto the same `EPS` regardless of
        // which side of the domain it overshot, so the two tails are exactly
        // mirror images of one another.
        let q = if upper { 1.0 - raw } else { raw };
        let q = q.clamp(EPS, 0.5);

        // Rational approximation of the inverse normal CDF's tail (A&S
        // 26.2.23), applied to `q` and mirrored by sign for the upper tail.
        let t = (-2.0 * q.ln()).sqrt();
        let c0 = 2.515_517_f32;
        let c1 = 0.802_853_f32;
        let c2 = 0.010_328_f32;
        let d1 = 1.432_788_f32;
        let d2 = 0.189_269_f32;
        let d3 = 0.001_308_f32;
        let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
        *v = if upper { result } else { -result };
    }
}

/// Apply softmax-zero: like softmax but zero entries remain zero.
pub(super) fn softmax_zero_rows(data: &mut [f32], n: usize, c: usize) {
    for row in 0..n {
        let offset = row * c;
        let row_slice = &mut data[offset..offset + c];
        let max_val = row_slice
            .iter()
            .copied()
            .filter(|&v| v != 0.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row_slice.iter_mut() {
            if *v != 0.0 {
                *v = (*v - max_val).exp();
                sum += *v;
            }
        }
        if sum > 0.0 {
            for v in row_slice.iter_mut() {
                if *v != 0.0 {
                    *v /= sum;
                }
            }
        }
    }
}

/// Post-transform enumeration used by ML operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostTransform {
    None,
    Softmax,
    SoftmaxZero,
    Logistic,
    Probit,
}

impl PostTransform {
    /// Parse a `post_transform` attribute string into the enum variant.
    ///
    /// An absent attribute (`attrs.s` returns `""`) resolves to the ONNX
    /// default of `NONE`, same as an explicit `"NONE"`. Any other,
    /// unrecognized value is a malformed model, not a silent fallback to
    /// `NONE` -- the same "bad enum falls through to a default variant"
    /// pitfall guarded against elsewhere in this crate (`KernelType::parse`
    /// in `ml_svm.rs`, `NodeMode::parse`/`Aggregate::parse` in `ml_tree.rs`).
    ///
    /// `op` names the calling operator for the error message only (e.g.
    /// `"SVMClassifier"`, `"TreeEnsembleRegressor"`).
    pub fn parse(s: &str, op: &str) -> Result<Self, OnnxError> {
        match s {
            "" | "NONE" => Ok(Self::None),
            "SOFTMAX" => Ok(Self::Softmax),
            "SOFTMAX_ZERO" => Ok(Self::SoftmaxZero),
            "LOGISTIC" => Ok(Self::Logistic),
            "PROBIT" => Ok(Self::Probit),
            other => Err(OnnxError::InvalidModel(format!(
                "{op}: unrecognized post_transform '{other}' \
                 (expected NONE, SOFTMAX, SOFTMAX_ZERO, LOGISTIC or PROBIT)"
            ))),
        }
    }
}

/// Apply a post-transform to a row-major \[N, C\] score buffer.
pub fn apply_post_transform(data: &mut [f32], n: usize, c: usize, transform: PostTransform) {
    match transform {
        PostTransform::Softmax => softmax_rows(data, n, c),
        PostTransform::SoftmaxZero => softmax_zero_rows(data, n, c),
        PostTransform::Logistic => logistic_inplace(data),
        PostTransform::Probit => probit_inplace(data),
        PostTransform::None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_none_same_as_absent_attribute() {
        assert_eq!(
            PostTransform::parse("", "T").expect("empty"),
            PostTransform::None
        );
    }

    #[test]
    fn recognized_variants_parse() {
        assert_eq!(
            PostTransform::parse("NONE", "T").expect("NONE"),
            PostTransform::None
        );
        assert_eq!(
            PostTransform::parse("SOFTMAX", "T").expect("SOFTMAX"),
            PostTransform::Softmax
        );
        assert_eq!(
            PostTransform::parse("SOFTMAX_ZERO", "T").expect("SOFTMAX_ZERO"),
            PostTransform::SoftmaxZero
        );
        assert_eq!(
            PostTransform::parse("LOGISTIC", "T").expect("LOGISTIC"),
            PostTransform::Logistic
        );
        assert_eq!(
            PostTransform::parse("PROBIT", "T").expect("PROBIT"),
            PostTransform::Probit
        );
    }

    /// The core fix: an unrecognized string must be a typed error, not a
    /// silent fallback to `None` -- mirroring `KernelType::parse` (`ml_svm.rs`)
    /// and `NodeMode::parse`/`Aggregate::parse` (`ml_tree.rs`).
    #[test]
    fn unrecognized_string_is_a_typed_error_not_a_silent_none_fallback() {
        let err = PostTransform::parse("BOGUS", "SVMClassifier")
            .expect_err("an unrecognized post_transform must be rejected");
        match err {
            OnnxError::InvalidModel(msg) => {
                assert!(msg.contains("SVMClassifier"), "{msg}");
                assert!(msg.contains("BOGUS"), "{msg}");
            }
            other => panic!("expected OnnxError::InvalidModel, got {other:?}"),
        }
    }

    /// Case sensitivity is part of the contract: ONNX attribute values are
    /// case-sensitive constants, so a lowercase variant is unrecognized, not
    /// silently normalized.
    #[test]
    fn lowercase_variant_is_unrecognized() {
        assert!(PostTransform::parse("softmax", "T").is_err());
    }
}
