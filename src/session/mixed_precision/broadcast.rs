//! Broadcasting helpers for f16 binary operations.
//!
//! Provides utilities for multi-dimensional index computation when applying
//! binary operations with NumPy-style broadcasting semantics.

/// Execute a binary op in f16 with broadcasting.
pub(super) fn broadcast_binary_f16(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
    out_size: usize,
    op: impl Fn(half::f16, half::f16) -> half::f16,
) -> Vec<f32> {
    let a_strides = broadcast_strides(a_shape, out_shape);
    let b_strides = broadcast_strides(b_shape, out_shape);
    let out_strides = compute_row_major_strides(out_shape);
    let mut result = Vec::with_capacity(out_size);
    for i in 0..out_size {
        let a_idx = broadcast_flat_index(i, out_shape, &out_strides, &a_strides);
        let b_idx = broadcast_flat_index(i, out_shape, &out_strides, &b_strides);
        let ha = half::f16::from_f32(a_data[a_idx]);
        let hb = half::f16::from_f32(b_data[b_idx]);
        result.push(op(ha, hb).to_f32());
    }
    result
}

/// Compute C-order (row-major) strides from shape.
pub(super) fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute effective strides for a tensor shape broadcast to `out_shape`.
/// Dimensions of size 1 get stride 0 (broadcast dimension).
pub(super) fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let ndim = out_shape.len();
    let offset = ndim.saturating_sub(shape.len());
    let mut strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        if shape[i] == out_shape[i + offset] {
            strides[i + offset] = stride;
            stride = stride.saturating_mul(shape[i]);
        }
        // else: size 1 => stride stays 0 (broadcast)
    }
    strides
}

/// Convert a flat index in the output to a flat index in a broadcast-strided source tensor.
pub(super) fn broadcast_flat_index(
    flat_idx: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    src_strides: &[usize],
) -> usize {
    let ndim = out_shape.len();
    let mut idx = 0usize;
    let mut remaining = flat_idx;
    for d in 0..ndim {
        let out_stride = out_strides[d];
        let coord = remaining.checked_div(out_stride).unwrap_or(0);
        remaining = if out_stride > 0 {
            remaining % out_stride
        } else {
            remaining
        };
        idx += coord * src_strides[d];
    }
    idx
}
