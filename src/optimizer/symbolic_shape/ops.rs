//! Op-specific symbolic shape inference helpers.
//!
//! Each function handles the symbolic output shape computation for a particular
//! ONNX operator. These are called from the main dispatcher in [`super::inference`].

use oxionnx_core::graph::Attributes;

use super::types::{SymDim, SymbolicShape};
use super::utils::{broadcast_symbolic, symbolic_numel};

/// MatMul: `[..., M, K] x [..., K, N] -> [..., M, N]`
pub(super) fn infer_matmul_symbolic(
    inputs: &[Option<&SymbolicShape>],
) -> Option<Vec<SymbolicShape>> {
    let a = inputs.first().and_then(|o| *o)?;
    let b = inputs.get(1).and_then(|o| *o)?;

    if a.is_empty() || b.is_empty() {
        return None;
    }

    // 1-D cases
    if a.len() == 1 && b.len() == 1 {
        // (K) x (K) -> scalar ()
        return Some(vec![vec![]]);
    }
    if a.len() == 1 {
        // (K) x [..., K, N] -> [..., N]
        let mut out: SymbolicShape = b[..b.len() - 2].to_vec();
        if let Some(last) = b.last() {
            out.push(last.clone());
        }
        return Some(vec![out]);
    }
    if b.len() == 1 {
        // [..., M, K] x (K) -> [..., M]
        let out: SymbolicShape = a[..a.len() - 1].to_vec();
        return Some(vec![out]);
    }

    // General: [..., M, K] x [..., K, N] -> [..., M, N]
    let a_batch = &a[..a.len() - 2];
    let b_batch = &b[..b.len() - 2];
    let batch = if !a_batch.is_empty() && !b_batch.is_empty() {
        broadcast_symbolic(a_batch, b_batch)?
    } else if !a_batch.is_empty() {
        a_batch.to_vec()
    } else {
        b_batch.to_vec()
    };

    let m = a[a.len() - 2].clone();
    let n = b[b.len() - 1].clone();

    let mut out = batch;
    out.push(m);
    out.push(n);
    Some(vec![out])
}

/// Gemm: `alpha * A' @ B' + beta * C` where `A'` is optionally transposed.
pub(super) fn infer_gemm_symbolic(
    inputs: &[Option<&SymbolicShape>],
    attrs: &Attributes,
) -> Option<Vec<SymbolicShape>> {
    let a = inputs.first().and_then(|o| *o)?;
    let b = inputs.get(1).and_then(|o| *o)?;
    if a.len() != 2 || b.len() != 2 {
        return None;
    }
    let trans_a = attrs.i("transA", 0) != 0;
    let trans_b = attrs.i("transB", 0) != 0;
    let m = if trans_a { a[1].clone() } else { a[0].clone() };
    let n = if trans_b { b[0].clone() } else { b[1].clone() };
    Some(vec![vec![m, n]])
}

/// Reshape: try to resolve shape from the second input if it is all-concrete.
pub(super) fn infer_reshape_symbolic(
    inputs: &[Option<&SymbolicShape>],
) -> Option<Vec<SymbolicShape>> {
    let data_shape = inputs.first().and_then(|o| *o)?;
    let target_shape = inputs.get(1).and_then(|o| *o)?;

    // The shape tensor must be 1-D and all-known to be useful here.
    // Each element of the shape tensor is a SymDim that represents a
    // target dimension value. We can propagate directly.
    // Special value: Known(0) means "copy from input", and we handle -1
    // (represented as a very large usize from signed reinterpretation).

    let total_input = symbolic_numel(data_shape);
    let mut result = Vec::with_capacity(target_shape.len());
    let mut neg_one_idx: Option<usize> = None;

    for (i, d) in target_shape.iter().enumerate() {
        match d {
            SymDim::Known(v) => {
                let v = *v;
                // In ONNX, shape values are i64. A -1 stored as usize wraps to usize::MAX.
                if v == usize::MAX {
                    if neg_one_idx.is_some() {
                        return None; // only one -1 allowed
                    }
                    neg_one_idx = Some(i);
                    result.push(SymDim::Known(0)); // placeholder
                } else if v == 0 {
                    // Copy from input dimension at same position
                    if i < data_shape.len() {
                        result.push(data_shape[i].clone());
                    } else {
                        return None;
                    }
                } else {
                    result.push(SymDim::Known(v));
                }
            }
            SymDim::Symbol(s) => {
                result.push(SymDim::Symbol(s.clone()));
            }
        }
    }

    // Resolve -1 if possible
    if let Some(idx) = neg_one_idx {
        if let Some(input_total) = total_input {
            let known_product: Option<usize> = result
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .try_fold(1usize, |acc, (_, d)| {
                    if let SymDim::Known(v) = d {
                        acc.checked_mul(*v)
                    } else {
                        None
                    }
                });
            if let Some(kp) = known_product {
                if let Some(inferred) = input_total.checked_div(kp) {
                    result[idx] = SymDim::Known(inferred);
                }
            }
        }
        // If we cannot resolve -1, leave it as symbol
        if result[idx] == SymDim::Known(0) {
            result[idx] = SymDim::Symbol("_reshape_inferred".to_string());
        }
    }

    Some(vec![result])
}

/// Transpose: permute dimensions according to `perm` attribute.
pub(super) fn infer_transpose_symbolic(
    inputs: &[Option<&SymbolicShape>],
    attrs: &Attributes,
) -> Option<Vec<SymbolicShape>> {
    let shape = inputs.first().and_then(|o| *o)?;
    let perm = attrs.ints("perm");
    if perm.is_empty() {
        // Default: reverse
        let mut out = shape.clone();
        out.reverse();
        return Some(vec![out]);
    }
    let out: SymbolicShape = perm
        .iter()
        .filter_map(|&p| {
            let idx = if p < 0 {
                (shape.len() as i64 + p) as usize
            } else {
                p as usize
            };
            shape.get(idx).cloned()
        })
        .collect();
    if out.len() != shape.len() {
        return None;
    }
    Some(vec![out])
}

/// Concat: sum the axis dimension, all other dimensions must match.
pub(super) fn infer_concat_symbolic(
    inputs: &[Option<&SymbolicShape>],
    attrs: &Attributes,
) -> Option<Vec<SymbolicShape>> {
    let first = inputs.first().and_then(|o| *o)?;
    let rank = first.len();
    let raw_axis = attrs.i("axis", 0);
    let axis = if raw_axis < 0 {
        (rank as i64 + raw_axis) as usize
    } else {
        raw_axis as usize
    };
    if axis >= rank {
        return None;
    }

    let mut out = first.clone();

    for inp in inputs.iter().skip(1) {
        let s = inp.as_ref()?;
        if s.len() != rank {
            return None;
        }
        // Sum along axis
        match (&out[axis], &s[axis]) {
            (SymDim::Known(a), SymDim::Known(b)) => {
                out[axis] = SymDim::Known(a + b);
            }
            _ => {
                // Cannot resolve symbolic concat dimension — use a generated symbol
                out[axis] = SymDim::Symbol("_concat_dim".to_string());
            }
        }
    }

    Some(vec![out])
}

/// Squeeze: remove dimension(s) of size 1.
pub(super) fn infer_squeeze_symbolic(
    inputs: &[Option<&SymbolicShape>],
    attrs: &Attributes,
) -> Option<Vec<SymbolicShape>> {
    let shape = inputs.first().and_then(|o| *o)?;
    let axes_attr = attrs.ints("axes");

    // If axes given via attribute
    let axes: Vec<usize> = if !axes_attr.is_empty() {
        axes_attr
            .iter()
            .map(|&a| {
                if a < 0 {
                    (shape.len() as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else {
        // Squeeze all dims of size 1
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                if d == &SymDim::Known(1) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    };

    let out: SymbolicShape = shape
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            if axes.contains(&i) {
                None
            } else {
                Some(d.clone())
            }
        })
        .collect();
    Some(vec![out])
}

/// Unsqueeze: insert dimensions of size 1.
pub(super) fn infer_unsqueeze_symbolic(
    inputs: &[Option<&SymbolicShape>],
    attrs: &Attributes,
) -> Option<Vec<SymbolicShape>> {
    let shape = inputs.first().and_then(|o| *o)?;
    let axes_attr = attrs.ints("axes");

    if axes_attr.is_empty() {
        // ONNX opset >= 13 takes axes as second input; we cannot resolve
        // dynamic axes here, so return None.
        return None;
    }

    let out_rank = shape.len() + axes_attr.len();
    let mut axes: Vec<usize> = axes_attr
        .iter()
        .map(|&a| {
            if a < 0 {
                (out_rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    axes.sort_unstable();

    let mut out = Vec::with_capacity(out_rank);
    let mut src_idx = 0usize;
    for i in 0..out_rank {
        if axes.contains(&i) {
            out.push(SymDim::Known(1));
        } else {
            if src_idx < shape.len() {
                out.push(shape[src_idx].clone());
            }
            src_idx += 1;
        }
    }

    Some(vec![out])
}

/// Flatten: collapse dims `[0..axis)` and `[axis..)` into two dimensions.
pub(super) fn infer_flatten_symbolic(
    inputs: &[Option<&SymbolicShape>],
    attrs: &Attributes,
) -> Option<Vec<SymbolicShape>> {
    let shape = inputs.first().and_then(|o| *o)?;
    let raw_axis = attrs.i("axis", 1);
    let axis = if raw_axis < 0 {
        (shape.len() as i64 + raw_axis) as usize
    } else {
        raw_axis as usize
    };

    let left = &shape[..axis];
    let right = &shape[axis..];

    let left_dim = fold_product(left);
    let right_dim = fold_product(right);

    Some(vec![vec![left_dim, right_dim]])
}

/// Multiply a slice of SymDim together; returns Known if all are known,
/// otherwise returns a placeholder Symbol.
pub(super) fn fold_product(dims: &[SymDim]) -> SymDim {
    if dims.is_empty() {
        return SymDim::Known(1);
    }
    let mut product = 1usize;
    for d in dims {
        match d {
            SymDim::Known(v) => {
                product = product.saturating_mul(*v);
            }
            SymDim::Symbol(_) => {
                return SymDim::Symbol("_product".to_string());
            }
        }
    }
    SymDim::Known(product)
}
