//! Shape inference for MatMul and Gemm operators.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

use super::helpers::get_input_shape;

/// MatMul shape: [..., M, K] x [..., K, N] -> [..., M, N]
/// with batch dimension broadcasting.
pub(super) fn infer_matmul_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let a = get_input_shape(node, 0, known)?;
    let b = get_input_shape(node, 1, known)?;

    if a.is_empty() || b.is_empty() {
        return None;
    }

    // Handle 1-D cases per ONNX MatMul spec
    let (a_shape, a_was_1d) = if a.len() == 1 {
        (vec![1, a[0]], true)
    } else {
        (a.clone(), false)
    };

    let (b_shape, b_was_1d) = if b.len() == 1 {
        (vec![b[0], 1], true)
    } else {
        (b.clone(), false)
    };

    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    let m = a_shape[a_rank - 2];
    let n = b_shape[b_rank - 1];

    // Broadcast batch dimensions
    let a_batch = &a_shape[..a_rank - 2];
    let b_batch = &b_shape[..b_rank - 2];

    let batch = if a_batch.is_empty() && b_batch.is_empty() {
        vec![]
    } else if a_batch.is_empty() {
        b_batch.to_vec()
    } else if b_batch.is_empty() {
        a_batch.to_vec()
    } else {
        Tensor::broadcast_shape(a_batch, b_batch).ok()?
    };

    let mut out = batch;
    if !a_was_1d {
        out.push(m);
    }
    if !b_was_1d {
        out.push(n);
    }
    // If both were 1-D, result is scalar (empty shape) per spec,
    // but we represent as [1] for simplicity
    if a_was_1d && b_was_1d {
        out.push(1);
    }

    Some(vec![out])
}

/// Gemm: Y = alpha * A' * B' + beta * C
/// Output shape is [M, N] considering transA/transB.
pub(super) fn infer_gemm_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let a = get_input_shape(node, 0, known)?;
    let b = get_input_shape(node, 1, known)?;

    if a.len() != 2 || b.len() != 2 {
        return None;
    }

    let trans_a = node.attrs.i("transA", 0) != 0;
    let trans_b = node.attrs.i("transB", 0) != 0;

    let m = if trans_a { a[1] } else { a[0] };
    let n = if trans_b { b[0] } else { b[1] };

    Some(vec![vec![m, n]])
}
