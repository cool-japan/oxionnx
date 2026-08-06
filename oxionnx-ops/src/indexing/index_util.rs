//! Shared axis/index normalization helpers for the indexing operator family.
//!
//! ONNX allows both `axis` attributes and index tensor values to be negative,
//! counting backward from the end of the corresponding dimension. Every
//! indexing op needs the same normalize-then-bounds-check logic; centralizing
//! it here means every call site gets the identical (correct) behavior
//! instead of each op hand-rolling (and occasionally omitting) its own copy.

/// Normalize a possibly-negative ONNX `axis` value against a tensor rank.
///
/// Returns `Err` if the normalized axis falls outside `[0, ndim)`.
pub(crate) fn normalize_axis(axis: i64, ndim: usize, op: &str) -> Result<usize, String> {
    let normalized = if axis < 0 { axis + ndim as i64 } else { axis };
    if normalized < 0 || normalized as usize >= ndim {
        return Err(format!("{op}: axis {axis} out of range for {ndim}D tensor"));
    }
    Ok(normalized as usize)
}

/// Normalize a possibly-negative ONNX index value against a dimension size.
///
/// The float index is truncated toward zero (ONNX index tensors always hold
/// integer values) then, if negative, counted from the end of the dimension
/// per the ONNX convention. Returns `Err` if the normalized index still
/// falls outside `[0, dim_size)` rather than silently clamping or wrapping.
pub(crate) fn normalize_index(idx: f32, dim_size: usize, op: &str) -> Result<usize, String> {
    let raw = idx as i64;
    let normalized = if raw < 0 { raw + dim_size as i64 } else { raw };
    if normalized < 0 || normalized as usize >= dim_size {
        return Err(format!(
            "{op}: index {raw} out of bounds for dim size {dim_size}"
        ));
    }
    Ok(normalized as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_axis_negative_counts_from_end() {
        assert_eq!(normalize_axis(-1, 3, "test").expect("in range"), 2);
        assert_eq!(normalize_axis(-3, 3, "test").expect("in range"), 0);
        assert_eq!(normalize_axis(0, 3, "test").expect("in range"), 0);
    }

    #[test]
    fn normalize_axis_out_of_range_errors() {
        assert!(normalize_axis(3, 3, "test").is_err());
        assert!(normalize_axis(-4, 3, "test").is_err());
    }

    #[test]
    fn normalize_index_negative_counts_from_end() {
        assert_eq!(normalize_index(-1.0, 4, "test").expect("in range"), 3);
        assert_eq!(normalize_index(-4.0, 4, "test").expect("in range"), 0);
        assert_eq!(normalize_index(0.0, 4, "test").expect("in range"), 0);
    }

    #[test]
    fn normalize_index_out_of_range_errors_not_clamps() {
        assert!(normalize_index(4.0, 4, "test").is_err());
        assert!(normalize_index(-5.0, 4, "test").is_err());
        assert!(normalize_index(999.0, 4, "test").is_err());
        assert!(normalize_index(-100.0, 10, "test").is_err());
    }
}
