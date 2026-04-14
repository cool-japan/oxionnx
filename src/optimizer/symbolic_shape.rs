//! Symbolic shape propagation for dynamic dimensions.
//!
//! Extends the concrete shape inference in [`super::shape_inference`] to handle
//! dimensions that remain unresolved until runtime (e.g. batch size, sequence
//! length). Each dimension is represented as [`SymDim`] — either a known
//! `usize` or a named symbol.

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::Tensor;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A dimension that may be either a concrete value or a symbolic name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymDim {
    /// A known concrete dimension.
    Known(usize),
    /// A symbolic/dynamic dimension (e.g., `"batch_size"`, `"seq_len"`).
    Symbol(String),
}

impl SymDim {
    /// Return the concrete value if known.
    pub fn as_known(&self) -> Option<usize> {
        match self {
            Self::Known(v) => Some(*v),
            Self::Symbol(_) => None,
        }
    }

    /// Return the symbol name if symbolic.
    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            Self::Known(_) => None,
            Self::Symbol(s) => Some(s),
        }
    }

    /// Check if the dimension is known (concrete).
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }
}

impl std::fmt::Display for SymDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Known(v) => write!(f, "{v}"),
            Self::Symbol(s) => write!(f, "{s}"),
        }
    }
}

/// A shape where each dimension may be concrete or symbolic.
pub type SymbolicShape = Vec<SymDim>;

/// Environment mapping symbol names to concrete values.
pub type SymbolEnv = HashMap<String, usize>;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Resolve a symbolic shape to concrete dimensions using the given environment.
/// Returns `None` if any symbol cannot be resolved.
pub fn resolve_shape(shape: &[SymDim], env: &SymbolEnv) -> Option<Vec<usize>> {
    shape
        .iter()
        .map(|d| match d {
            SymDim::Known(v) => Some(*v),
            SymDim::Symbol(s) => env.get(s).copied(),
        })
        .collect()
}

/// Convert a concrete shape to a symbolic shape (all dimensions known).
pub fn from_concrete(shape: &[usize]) -> SymbolicShape {
    shape.iter().map(|&d| SymDim::Known(d)).collect()
}

/// Compute the total number of elements if all dimensions are known.
/// Returns `None` when any dimension is symbolic.
pub fn symbolic_numel(shape: &[SymDim]) -> Option<usize> {
    let mut total = 1usize;
    for d in shape {
        match d {
            SymDim::Known(v) => {
                total = total.checked_mul(*v)?;
            }
            SymDim::Symbol(_) => return None,
        }
    }
    Some(total)
}

/// Try to broadcast two symbolic shapes following NumPy broadcasting rules.
///
/// Returns `None` when the shapes are provably incompatible or when two
/// different symbolic dimensions appear in the same position without one of
/// them being the literal 1.
pub fn broadcast_symbolic(a: &[SymDim], b: &[SymDim]) -> Option<SymbolicShape> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    let a_pad = n - a.len();
    let b_pad = n - b.len();
    let one = SymDim::Known(1);
    for i in 0..n {
        let ai = if i < a_pad { &one } else { &a[i - a_pad] };
        let bi = if i < b_pad { &one } else { &b[i - b_pad] };
        match (ai, bi) {
            (SymDim::Known(1), other) | (other, SymDim::Known(1)) => out.push(other.clone()),
            (SymDim::Known(a_val), SymDim::Known(b_val)) => {
                if a_val != b_val {
                    return None;
                }
                out.push(SymDim::Known(*a_val));
            }
            (SymDim::Symbol(s1), SymDim::Symbol(s2)) if s1 == s2 => {
                out.push(SymDim::Symbol(s1.clone()));
            }
            // A symbol paired with itself-as-known(1) was already handled above.
            // Two different symbols or symbol vs concrete > 1 cannot be resolved.
            _ => return None,
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Per-node symbolic inference
// ---------------------------------------------------------------------------

/// Infer symbolic output shapes for a single node.
///
/// Returns `None` when a required input shape is unavailable or the op is
/// not yet supported by the symbolic engine.
fn infer_node_symbolic(
    op: &OpKind,
    inputs: &[Option<&SymbolicShape>],
    attrs: &Attributes,
) -> Option<Vec<SymbolicShape>> {
    match op {
        // --- unary element-wise ---
        OpKind::Identity
        | OpKind::Cast
        | OpKind::Relu
        | OpKind::Sigmoid
        | OpKind::Tanh
        | OpKind::Gelu
        | OpKind::SiLU
        | OpKind::Erf
        | OpKind::Abs
        | OpKind::Log
        | OpKind::Exp
        | OpKind::Neg
        | OpKind::Sqrt
        | OpKind::Ceil
        | OpKind::Floor
        | OpKind::Round
        | OpKind::Sign
        | OpKind::Reciprocal
        | OpKind::Sin
        | OpKind::Cos
        | OpKind::Tan
        | OpKind::Asin
        | OpKind::Acos
        | OpKind::Atan
        | OpKind::Sinh
        | OpKind::Cosh
        | OpKind::Asinh
        | OpKind::Acosh
        | OpKind::Atanh
        | OpKind::HardSigmoid
        | OpKind::HardSwish
        | OpKind::Not
        | OpKind::LeakyRelu
        | OpKind::LogSoftmax
        | OpKind::Softplus
        | OpKind::Softsign
        | OpKind::Mish
        | OpKind::Celu
        | OpKind::Elu
        | OpKind::Selu
        | OpKind::ThresholdedRelu
        | OpKind::Clip
        | OpKind::BitwiseNot
        | OpKind::Hardmax
        | OpKind::Shrink => {
            let s = inputs.first().and_then(|o| *o)?;
            Some(vec![s.clone()])
        }

        // --- normalization (output shape == input[0] shape) ---
        OpKind::Softmax
        | OpKind::LayerNorm
        | OpKind::BatchNorm
        | OpKind::GroupNorm
        | OpKind::RMSNorm
        | OpKind::InstanceNorm
        | OpKind::LpNorm
        | OpKind::MeanVarianceNormalization => {
            let s = inputs.first().and_then(|o| *o)?;
            Some(vec![s.clone()])
        }

        // --- dropout: first output same as input, optional mask ---
        OpKind::Dropout => {
            let s = inputs.first().and_then(|o| *o)?;
            // output 0 = same shape, output 1 = mask (same shape)
            Some(vec![s.clone(), s.clone()])
        }

        // --- binary element-wise with broadcasting ---
        OpKind::Add
        | OpKind::Sub
        | OpKind::Mul
        | OpKind::Div
        | OpKind::Pow
        | OpKind::Mod
        | OpKind::BitwiseAnd
        | OpKind::BitwiseOr
        | OpKind::BitwiseXor => {
            let a = inputs.first().and_then(|o| *o)?;
            let b = inputs.get(1).and_then(|o| *o)?;
            let out = broadcast_symbolic(a, b)?;
            Some(vec![out])
        }

        // --- comparison / logic (binary broadcast -> same shape) ---
        OpKind::Equal
        | OpKind::Greater
        | OpKind::GreaterOrEqual
        | OpKind::Less
        | OpKind::LessOrEqual
        | OpKind::And
        | OpKind::Or
        | OpKind::Xor => {
            let a = inputs.first().and_then(|o| *o)?;
            let b = inputs.get(1).and_then(|o| *o)?;
            let out = broadcast_symbolic(a, b)?;
            Some(vec![out])
        }

        // --- MatMul ---
        OpKind::MatMul => infer_matmul_symbolic(inputs),

        // --- Gemm ---
        OpKind::Gemm => infer_gemm_symbolic(inputs, attrs),

        // --- Reshape ---
        OpKind::Reshape => infer_reshape_symbolic(inputs),

        // --- Transpose ---
        OpKind::Transpose => infer_transpose_symbolic(inputs, attrs),

        // --- Concat ---
        OpKind::Concat => infer_concat_symbolic(inputs, attrs),

        // --- Squeeze ---
        OpKind::Squeeze => infer_squeeze_symbolic(inputs, attrs),

        // --- Unsqueeze ---
        OpKind::Unsqueeze => infer_unsqueeze_symbolic(inputs, attrs),

        // --- Flatten ---
        OpKind::Flatten => infer_flatten_symbolic(inputs, attrs),

        // --- Expand ---
        OpKind::Expand => {
            // Expand follows broadcast rules between input and shape tensor.
            // If shape tensor is known we use broadcast, otherwise pass through.
            let a = inputs.first().and_then(|o| *o)?;
            let b = inputs.get(1).and_then(|o| *o)?;
            let out = broadcast_symbolic(a, b)?;
            Some(vec![out])
        }

        // --- Where (ternary broadcast) ---
        OpKind::Where => {
            let cond = inputs.first().and_then(|o| *o)?;
            let x = inputs.get(1).and_then(|o| *o)?;
            let y = inputs.get(2).and_then(|o| *o)?;
            let tmp = broadcast_symbolic(cond, x)?;
            let out = broadcast_symbolic(&tmp, y)?;
            Some(vec![out])
        }

        // --- Shape op: output is always [rank] ---
        OpKind::Shape => {
            let s = inputs.first().and_then(|o| *o)?;
            Some(vec![vec![SymDim::Known(s.len())]])
        }

        // Unsupported ops return None so the caller can skip gracefully.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Op-specific helpers
// ---------------------------------------------------------------------------

/// MatMul: `[..., M, K] x [..., K, N] -> [..., M, N]`
fn infer_matmul_symbolic(inputs: &[Option<&SymbolicShape>]) -> Option<Vec<SymbolicShape>> {
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
fn infer_gemm_symbolic(
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
fn infer_reshape_symbolic(inputs: &[Option<&SymbolicShape>]) -> Option<Vec<SymbolicShape>> {
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
fn infer_transpose_symbolic(
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
fn infer_concat_symbolic(
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
fn infer_squeeze_symbolic(
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
fn infer_unsqueeze_symbolic(
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
fn infer_flatten_symbolic(
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
fn fold_product(dims: &[SymDim]) -> SymDim {
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

// ---------------------------------------------------------------------------
// Graph-level symbolic inference
// ---------------------------------------------------------------------------

/// Infer symbolic shapes for all tensors in the graph.
///
/// Similar to [`super::shape_inference::infer_shapes`] but propagates symbolic
/// dimensions. `input_shapes` maps tensor name to symbolic shape for graph
/// inputs. Returns a map from tensor name to symbolic shape.
pub fn infer_symbolic_shapes(
    nodes: &[Node],
    weights: &HashMap<String, Tensor>,
    input_shapes: &HashMap<String, SymbolicShape>,
) -> HashMap<String, SymbolicShape> {
    let mut shapes: HashMap<String, SymbolicShape> = HashMap::new();

    // Seed with weight shapes (all concrete).
    for (name, tensor) in weights {
        shapes.insert(name.clone(), from_concrete(&tensor.shape));
    }

    // Seed with user-provided input shapes.
    for (name, shape) in input_shapes {
        shapes.insert(name.clone(), shape.clone());
    }

    // Propagate through nodes (assumed topologically sorted).
    for node in nodes {
        let op = &node.op;
        let input_syms: Vec<Option<&SymbolicShape>> = node
            .inputs
            .iter()
            .map(|name| {
                if name.is_empty() {
                    None
                } else {
                    shapes.get(name)
                }
            })
            .collect();

        if let Some(out_shapes) = infer_node_symbolic(op, &input_syms, &node.attrs) {
            for (out_name, out_shape) in node.outputs.iter().zip(out_shapes) {
                if !out_name.is_empty() {
                    shapes.insert(out_name.clone(), out_shape);
                }
            }
        }
    }

    shapes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_node(op: OpKind, inputs: Vec<&str>, outputs: Vec<&str>) -> Node {
        Node {
            op,
            name: String::new(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
            attrs: Attributes::default(),
        }
    }

    fn make_node_with_attrs(
        op: OpKind,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        attrs: Attributes,
    ) -> Node {
        Node {
            op,
            name: String::new(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
            attrs,
        }
    }

    // 1. test_sym_dim_known_and_symbol
    #[test]
    fn test_sym_dim_known_and_symbol() {
        let k = SymDim::Known(42);
        assert_eq!(k.as_known(), Some(42));
        assert_eq!(k.as_symbol(), None);
        assert!(k.is_known());
        assert_eq!(format!("{k}"), "42");

        let s = SymDim::Symbol("batch".to_string());
        assert_eq!(s.as_known(), None);
        assert_eq!(s.as_symbol(), Some("batch"));
        assert!(!s.is_known());
        assert_eq!(format!("{s}"), "batch");
    }

    // 2. test_resolve_shape
    #[test]
    fn test_resolve_shape() {
        let shape = vec![
            SymDim::Symbol("N".to_string()),
            SymDim::Known(64),
            SymDim::Symbol("S".to_string()),
        ];
        let mut env = SymbolEnv::new();
        env.insert("N".to_string(), 4);
        env.insert("S".to_string(), 128);

        let resolved = resolve_shape(&shape, &env);
        assert_eq!(resolved, Some(vec![4, 64, 128]));

        // Missing symbol
        let mut partial_env = SymbolEnv::new();
        partial_env.insert("N".to_string(), 4);
        assert_eq!(resolve_shape(&shape, &partial_env), None);

        // All-concrete always resolves
        let concrete = vec![SymDim::Known(2), SymDim::Known(3)];
        assert_eq!(
            resolve_shape(&concrete, &SymbolEnv::new()),
            Some(vec![2, 3])
        );
    }

    // 3. test_broadcast_symbolic
    #[test]
    fn test_broadcast_symbolic() {
        // Concrete broadcast
        let a = vec![SymDim::Known(3), SymDim::Known(1)];
        let b = vec![SymDim::Known(1), SymDim::Known(4)];
        assert_eq!(
            broadcast_symbolic(&a, &b),
            Some(vec![SymDim::Known(3), SymDim::Known(4)])
        );

        // Symbol with 1
        let a = vec![SymDim::Symbol("N".to_string()), SymDim::Known(1)];
        let b = vec![SymDim::Known(1), SymDim::Known(64)];
        assert_eq!(
            broadcast_symbolic(&a, &b),
            Some(vec![SymDim::Symbol("N".to_string()), SymDim::Known(64)])
        );

        // Same symbol
        let a = vec![SymDim::Symbol("B".to_string()), SymDim::Known(3)];
        let b = vec![SymDim::Symbol("B".to_string()), SymDim::Known(3)];
        assert_eq!(
            broadcast_symbolic(&a, &b),
            Some(vec![SymDim::Symbol("B".to_string()), SymDim::Known(3)])
        );

        // Different symbols => None
        let a = vec![SymDim::Symbol("A".to_string())];
        let b = vec![SymDim::Symbol("B".to_string())];
        assert_eq!(broadcast_symbolic(&a, &b), None);

        // Incompatible concrete => None
        let a = vec![SymDim::Known(3)];
        let b = vec![SymDim::Known(4)];
        assert_eq!(broadcast_symbolic(&a, &b), None);

        // Different ranks
        let a = vec![SymDim::Known(5)];
        let b = vec![SymDim::Known(3), SymDim::Known(5)];
        assert_eq!(
            broadcast_symbolic(&a, &b),
            Some(vec![SymDim::Known(3), SymDim::Known(5)])
        );
    }

    // 4. test_from_concrete
    #[test]
    fn test_from_concrete() {
        let shape = from_concrete(&[2, 3, 4]);
        assert_eq!(
            shape,
            vec![SymDim::Known(2), SymDim::Known(3), SymDim::Known(4)]
        );
        assert_eq!(from_concrete(&[]), Vec::<SymDim>::new());
    }

    // 5. test_symbolic_numel
    #[test]
    fn test_symbolic_numel() {
        assert_eq!(
            symbolic_numel(&[SymDim::Known(2), SymDim::Known(3), SymDim::Known(4)]),
            Some(24)
        );
        assert_eq!(symbolic_numel(&[]), Some(1));
        assert_eq!(
            symbolic_numel(&[SymDim::Known(2), SymDim::Symbol("N".to_string())]),
            None
        );
    }

    // 6. test_infer_symbolic_identity
    #[test]
    fn test_infer_symbolic_identity() {
        let node = make_node(OpKind::Identity, vec!["x"], vec!["y"]);
        let mut input_shapes = HashMap::new();
        input_shapes.insert(
            "x".to_string(),
            vec![SymDim::Symbol("B".to_string()), SymDim::Known(64)],
        );

        let result = infer_symbolic_shapes(&[node], &HashMap::new(), &input_shapes);
        assert_eq!(
            result.get("y"),
            Some(&vec![SymDim::Symbol("B".to_string()), SymDim::Known(64)])
        );
    }

    // 7. test_infer_symbolic_matmul
    #[test]
    fn test_infer_symbolic_matmul() {
        // [B, M, K] x [B, K, N] -> [B, M, N]
        let node = make_node(OpKind::MatMul, vec!["a", "b"], vec!["c"]);
        let mut input_shapes = HashMap::new();
        input_shapes.insert(
            "a".to_string(),
            vec![
                SymDim::Symbol("B".to_string()),
                SymDim::Known(32),
                SymDim::Known(64),
            ],
        );
        input_shapes.insert(
            "b".to_string(),
            vec![
                SymDim::Symbol("B".to_string()),
                SymDim::Known(64),
                SymDim::Known(128),
            ],
        );

        let result = infer_symbolic_shapes(&[node], &HashMap::new(), &input_shapes);
        assert_eq!(
            result.get("c"),
            Some(&vec![
                SymDim::Symbol("B".to_string()),
                SymDim::Known(32),
                SymDim::Known(128),
            ])
        );

        // 1-D x 2-D: (K) x (K, N) -> (N)
        let node2 = make_node(OpKind::MatMul, vec!["v", "m"], vec!["r"]);
        let mut input_shapes2 = HashMap::new();
        input_shapes2.insert("v".to_string(), vec![SymDim::Known(64)]);
        input_shapes2.insert("m".to_string(), vec![SymDim::Known(64), SymDim::Known(128)]);
        let result2 = infer_symbolic_shapes(&[node2], &HashMap::new(), &input_shapes2);
        assert_eq!(result2.get("r"), Some(&vec![SymDim::Known(128)]));
    }

    // 8. test_infer_symbolic_elementwise_broadcast
    #[test]
    fn test_infer_symbolic_elementwise_broadcast() {
        let node = make_node(OpKind::Add, vec!["a", "b"], vec!["c"]);
        let mut input_shapes = HashMap::new();
        input_shapes.insert(
            "a".to_string(),
            vec![
                SymDim::Symbol("B".to_string()),
                SymDim::Known(3),
                SymDim::Known(1),
            ],
        );
        input_shapes.insert("b".to_string(), vec![SymDim::Known(1), SymDim::Known(4)]);

        let result = infer_symbolic_shapes(&[node], &HashMap::new(), &input_shapes);
        assert_eq!(
            result.get("c"),
            Some(&vec![
                SymDim::Symbol("B".to_string()),
                SymDim::Known(3),
                SymDim::Known(4),
            ])
        );
    }

    // 9. test_infer_symbolic_transpose
    #[test]
    fn test_infer_symbolic_transpose() {
        let mut attrs = Attributes::default();
        attrs.int_lists.insert("perm".to_string(), vec![0, 2, 1]);
        let node = make_node_with_attrs(OpKind::Transpose, vec!["x"], vec!["y"], attrs);
        let mut input_shapes = HashMap::new();
        input_shapes.insert(
            "x".to_string(),
            vec![
                SymDim::Symbol("B".to_string()),
                SymDim::Symbol("S".to_string()),
                SymDim::Known(64),
            ],
        );

        let result = infer_symbolic_shapes(&[node], &HashMap::new(), &input_shapes);
        assert_eq!(
            result.get("y"),
            Some(&vec![
                SymDim::Symbol("B".to_string()),
                SymDim::Known(64),
                SymDim::Symbol("S".to_string()),
            ])
        );

        // Default perm (reverse)
        let node2 = make_node(OpKind::Transpose, vec!["x"], vec!["z"]);
        let result2 = infer_symbolic_shapes(&[node2], &HashMap::new(), &input_shapes);
        assert_eq!(
            result2.get("z"),
            Some(&vec![
                SymDim::Known(64),
                SymDim::Symbol("S".to_string()),
                SymDim::Symbol("B".to_string()),
            ])
        );
    }

    // Additional tests for completeness

    #[test]
    fn test_infer_symbolic_concat() {
        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".to_string(), 1);
        let node = make_node_with_attrs(OpKind::Concat, vec!["a", "b"], vec!["c"], attrs);
        let mut input_shapes = HashMap::new();
        input_shapes.insert(
            "a".to_string(),
            vec![SymDim::Symbol("B".to_string()), SymDim::Known(10)],
        );
        input_shapes.insert(
            "b".to_string(),
            vec![SymDim::Symbol("B".to_string()), SymDim::Known(20)],
        );

        let result = infer_symbolic_shapes(&[node], &HashMap::new(), &input_shapes);
        assert_eq!(
            result.get("c"),
            Some(&vec![SymDim::Symbol("B".to_string()), SymDim::Known(30)])
        );
    }

    #[test]
    fn test_infer_symbolic_squeeze_unsqueeze() {
        // Squeeze axis 1 (size 1)
        let mut sq_attrs = Attributes::default();
        sq_attrs.int_lists.insert("axes".to_string(), vec![1]);
        let sq_node = make_node_with_attrs(OpKind::Squeeze, vec!["x"], vec!["y"], sq_attrs);
        let mut input_shapes = HashMap::new();
        input_shapes.insert(
            "x".to_string(),
            vec![
                SymDim::Symbol("B".to_string()),
                SymDim::Known(1),
                SymDim::Known(64),
            ],
        );

        let result = infer_symbolic_shapes(&[sq_node], &HashMap::new(), &input_shapes);
        assert_eq!(
            result.get("y"),
            Some(&vec![SymDim::Symbol("B".to_string()), SymDim::Known(64)])
        );

        // Unsqueeze axis 0
        let mut usq_attrs = Attributes::default();
        usq_attrs.int_lists.insert("axes".to_string(), vec![0]);
        let usq_node = make_node_with_attrs(OpKind::Unsqueeze, vec!["a"], vec!["b"], usq_attrs);
        let mut input_shapes2 = HashMap::new();
        input_shapes2.insert("a".to_string(), vec![SymDim::Known(3), SymDim::Known(4)]);
        let result2 = infer_symbolic_shapes(&[usq_node], &HashMap::new(), &input_shapes2);
        assert_eq!(
            result2.get("b"),
            Some(&vec![SymDim::Known(1), SymDim::Known(3), SymDim::Known(4)])
        );
    }

    #[test]
    fn test_infer_symbolic_flatten() {
        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".to_string(), 2);
        let node = make_node_with_attrs(OpKind::Flatten, vec!["x"], vec!["y"], attrs);
        let mut input_shapes = HashMap::new();
        input_shapes.insert(
            "x".to_string(),
            vec![
                SymDim::Known(2),
                SymDim::Known(3),
                SymDim::Known(4),
                SymDim::Known(5),
            ],
        );
        let result = infer_symbolic_shapes(&[node], &HashMap::new(), &input_shapes);
        assert_eq!(
            result.get("y"),
            Some(&vec![SymDim::Known(6), SymDim::Known(20)])
        );
    }

    #[test]
    fn test_infer_symbolic_multi_node_chain() {
        // x -> Relu -> y -> Add(y, bias) -> z
        let relu = make_node(OpKind::Relu, vec!["x"], vec!["y"]);
        let add = make_node(OpKind::Add, vec!["y", "bias"], vec!["z"]);

        let mut input_shapes = HashMap::new();
        input_shapes.insert(
            "x".to_string(),
            vec![SymDim::Symbol("B".to_string()), SymDim::Known(64)],
        );

        let mut weights = HashMap::new();
        weights.insert("bias".to_string(), Tensor::new(vec![0.0; 64], vec![64]));

        let result = infer_symbolic_shapes(&[relu, add], &weights, &input_shapes);
        assert_eq!(
            result.get("z"),
            Some(&vec![SymDim::Symbol("B".to_string()), SymDim::Known(64)])
        );
    }
}
