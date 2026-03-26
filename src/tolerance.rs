//! Numerical tolerance validation for inference results.

use oxionnx_core::Tensor;

/// Result of comparing two tensors with tolerance.
#[derive(Debug, Clone)]
pub struct ToleranceReport {
    /// Maximum absolute error across all elements.
    pub max_abs_error: f32,
    /// Mean absolute error.
    pub mean_abs_error: f32,
    /// Maximum relative error (excluding near-zero values).
    pub max_rel_error: f32,
    /// Mean relative error.
    pub mean_rel_error: f32,
    /// Number of elements compared.
    pub num_elements: usize,
    /// Number of elements exceeding absolute tolerance.
    pub num_abs_violations: usize,
    /// Number of elements exceeding relative tolerance.
    pub num_rel_violations: usize,
    /// Whether the comparison passed (within tolerance).
    pub passed: bool,
}

/// Compare two tensors with specified tolerances.
///
/// `abs_tol`: maximum allowed absolute error per element.
/// `rel_tol`: maximum allowed relative error per element.
/// An element passes if `|a - b| <= abs_tol + rel_tol * |b|`.
pub fn compare_tensors(
    actual: &Tensor,
    expected: &Tensor,
    abs_tol: f32,
    rel_tol: f32,
) -> Result<ToleranceReport, String> {
    if actual.shape != expected.shape {
        return Err(format!(
            "Shape mismatch: {:?} vs {:?}",
            actual.shape, expected.shape
        ));
    }
    if actual.data.len() != expected.data.len() {
        return Err(format!(
            "Length mismatch: {} vs {}",
            actual.data.len(),
            expected.data.len()
        ));
    }

    let n = actual.data.len();
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    let mut sum_rel = 0.0f64;
    let mut abs_violations = 0usize;
    let mut rel_violations = 0usize;

    for i in 0..n {
        let a = actual.data[i];
        let e = expected.data[i];
        let abs_err = (a - e).abs();

        if abs_err > max_abs {
            max_abs = abs_err;
        }
        sum_abs += abs_err as f64;

        let threshold = abs_tol + rel_tol * e.abs();
        if abs_err > threshold {
            abs_violations += 1;
        }

        // Relative error (avoid division by near-zero)
        if e.abs() > 1e-8 {
            let rel_err = abs_err / e.abs();
            if rel_err > max_rel {
                max_rel = rel_err;
            }
            sum_rel += rel_err as f64;
            if rel_err > rel_tol {
                rel_violations += 1;
            }
        }
    }

    let mean_abs = if n > 0 {
        (sum_abs / n as f64) as f32
    } else {
        0.0
    };
    let mean_rel = if n > 0 {
        (sum_rel / n as f64) as f32
    } else {
        0.0
    };

    Ok(ToleranceReport {
        max_abs_error: max_abs,
        mean_abs_error: mean_abs,
        max_rel_error: max_rel,
        mean_rel_error: mean_rel,
        num_elements: n,
        num_abs_violations: abs_violations,
        num_rel_violations: rel_violations,
        passed: abs_violations == 0,
    })
}

/// Quick check: are two tensors approximately equal?
pub fn tensors_close(actual: &Tensor, expected: &Tensor, atol: f32, rtol: f32) -> bool {
    compare_tensors(actual, expected, atol, rtol)
        .map(|r| r.passed)
        .ok()
        .unwrap_or(false)
}

impl std::fmt::Display for ToleranceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tolerance Report: {} elements, max_abs={:.6}, mean_abs={:.6}, \
             max_rel={:.6}, abs_violations={}, passed={}",
            self.num_elements,
            self.max_abs_error,
            self.mean_abs_error,
            self.max_rel_error,
            self.num_abs_violations,
            self.passed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_tensors_exact() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let report = compare_tensors(&a, &b, 1e-6, 1e-6).expect("should succeed");
        assert!(report.passed);
        assert_eq!(report.num_abs_violations, 0);
        assert_eq!(report.num_rel_violations, 0);
        assert!(report.max_abs_error < 1e-10);
    }

    #[test]
    fn test_compare_tensors_within_tol() {
        let a = Tensor::new(vec![1.001, 2.001, 3.001], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        // abs_tol=0.01 should pass since max diff is 0.001
        let report = compare_tensors(&a, &b, 0.01, 0.01).expect("should succeed");
        assert!(report.passed);
        assert_eq!(report.num_abs_violations, 0);
    }

    #[test]
    fn test_compare_tensors_violation() {
        let a = Tensor::new(vec![1.0, 2.0, 10.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        // abs_tol=0.1, rel_tol=0.1: element at index 2 differs by 7.0
        let report = compare_tensors(&a, &b, 0.1, 0.1).expect("should succeed");
        assert!(!report.passed);
        assert!(report.num_abs_violations > 0);
        assert!((report.max_abs_error - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_compare_tensors_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let result = compare_tensors(&a, &b, 1e-6, 1e-6);
        assert!(result.is_err());
        let err = result.expect_err("should be error");
        assert!(err.contains("Shape mismatch"));
    }

    #[test]
    fn test_tensors_close() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(tensors_close(&a, &b, 1e-6, 1e-6));

        let c = Tensor::new(vec![1.0, 2.0, 100.0], vec![3]);
        assert!(!tensors_close(&a, &c, 1e-6, 1e-6));

        // Shape mismatch returns false
        let d = Tensor::new(vec![1.0, 2.0], vec![2]);
        assert!(!tensors_close(&a, &d, 1e-6, 1e-6));
    }

    #[test]
    fn test_tolerance_report_display() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let report = compare_tensors(&a, &b, 1e-6, 1e-6).expect("should succeed");
        let display = format!("{report}");
        assert!(display.contains("Tolerance Report"));
        assert!(display.contains("3 elements"));
        assert!(display.contains("passed=true"));
    }
}
