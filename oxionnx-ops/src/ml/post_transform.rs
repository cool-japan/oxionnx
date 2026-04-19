//! Post-transform helpers for ONNX-ML operators.
//!
//! Provides the `PostTransform` enum and `apply_post_transform` for applying
//! softmax, logistic, probit, and related transforms to score buffers.

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
/// Uses the Abramowitz & Stegun rational approximation.
pub(super) fn probit_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        // Clamp to (0, 1) to avoid infinities
        let p = v.clamp(1e-7, 1.0 - 1e-7);
        // Approximate probit via the rational approximation of the inverse normal CDF
        // Using the Beasley-Springer-Moro algorithm (simplified)
        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };
        // Rational approximation constants
        let c0 = 2.515_517_f32;
        let c1 = 0.802_853_f32;
        let c2 = 0.010_328_f32;
        let d1 = 1.432_788_f32;
        let d2 = 0.189_269_f32;
        let d3 = 0.001_308_f32;
        let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
        *v = if p < 0.5 { -result } else { result };
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
    /// Parse a post-transform string into the enum variant.
    pub fn parse(s: &str) -> Self {
        match s {
            "SOFTMAX" => Self::Softmax,
            "SOFTMAX_ZERO" => Self::SoftmaxZero,
            "LOGISTIC" => Self::Logistic,
            "PROBIT" => Self::Probit,
            _ => Self::None,
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
