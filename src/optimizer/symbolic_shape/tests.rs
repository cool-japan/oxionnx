//! Tests for symbolic shape propagation.

use std::collections::HashMap;

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::Tensor;

use super::inference::infer_symbolic_shapes;
use super::types::{SymDim, SymbolEnv};
use super::utils::{broadcast_symbolic, from_concrete, resolve_shape, symbolic_numel};

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
