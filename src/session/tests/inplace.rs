use super::super::types::OptLevel;
use super::super::Session;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Verify ReLU produces correct output (in-place path used when ref_count==1).
#[test]
fn test_inplace_relu() {
    let node = Node {
        op: OpKind::Relu,
        name: "relu".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build");

    let input = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);
    let outputs = session.run_one("x", input).expect("run");
    let y = outputs.get("y").expect("y");
    assert_eq!(y.data, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
}

/// Verify element-wise Add works in-place when shapes match.
#[test]
fn test_inplace_add_same_shape() {
    // x -> Add(x, w) -> y   where x and w have same shape
    let node = Node {
        op: OpKind::Add,
        name: "add".to_string(),
        inputs: vec!["x".to_string(), "w".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    weights.insert(
        "w".to_string(),
        Tensor::new(vec![10.0, 20.0, 30.0], vec![3]),
    );

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build");

    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let outputs = session.run_one("x", input).expect("run");
    let y = outputs.get("y").expect("y");
    assert_eq!(y.data, vec![11.0, 22.0, 33.0]);
}

/// Verify broadcast Add falls back to regular path (shapes differ).
#[test]
fn test_inplace_fallback_broadcast() {
    // x [2,3] + w [3] -> y [2,3]   (broadcasting needed, inplace should fallback)
    let node = Node {
        op: OpKind::Add,
        name: "add".to_string(),
        inputs: vec!["x".to_string(), "w".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    weights.insert(
        "w".to_string(),
        Tensor::new(vec![10.0, 20.0, 30.0], vec![3]),
    );

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build");

    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let outputs = session.run_one("x", input).expect("run");
    let y = outputs.get("y").expect("y");
    assert_eq!(y.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    assert_eq!(y.shape, vec![2, 3]);
}

/// A tensor consumed by 2 nodes should NOT be modified in-place.
#[test]
fn test_inplace_respects_refcount() {
    // input -> relu_a -> out_a
    // input -> relu_b -> out_b
    // "input" has refcount 2, so neither relu should modify it in-place.
    let node_a = Node {
        op: OpKind::Relu,
        name: "relu_a".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["out_a".to_string()],
        attrs: Attributes::default(),
    };
    let node_b = Node {
        op: OpKind::Relu,
        name: "relu_b".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["out_b".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node_a, node_b],
        input_names: vec!["input".to_string()],
        output_names: vec!["out_a".to_string(), "out_b".to_string()],
        ..Default::default()
    };

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build");

    let input = Tensor::new(vec![-2.0, 3.0, -1.0, 5.0], vec![2, 2]);
    let outputs = session.run_one("input", input).expect("run");

    let expected = vec![0.0, 3.0, 0.0, 5.0];
    let out_a = outputs.get("out_a").expect("out_a");
    let out_b = outputs.get("out_b").expect("out_b");
    assert_eq!(out_a.data, expected);
    assert_eq!(out_b.data, expected);
}

/// Test depth computation helper directly.
#[test]
fn test_compute_node_depths() {
    // A linear chain: input -> relu1 -> relu2 -> output
    let node1 = Node {
        op: OpKind::Relu,
        name: "relu1".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["mid".to_string()],
        attrs: Attributes::default(),
    };
    let node2 = Node {
        op: OpKind::Relu,
        name: "relu2".to_string(),
        inputs: vec!["mid".to_string()],
        outputs: vec!["output".to_string()],
        attrs: Attributes::default(),
    };
    let nodes = vec![node1, node2];
    let weights = HashMap::new();
    let depths = Session::compute_node_depths(&nodes, &weights);
    assert_eq!(depths, vec![0, 1]);
}

/// Test depth computation with independent branches.
#[test]
fn test_compute_node_depths_parallel_branches() {
    // input -> relu_a -> out_a  (depth 0)
    // input -> relu_b -> out_b  (depth 0)
    let node_a = Node {
        op: OpKind::Relu,
        name: "relu_a".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["out_a".to_string()],
        attrs: Attributes::default(),
    };
    let node_b = Node {
        op: OpKind::Relu,
        name: "relu_b".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["out_b".to_string()],
        attrs: Attributes::default(),
    };
    let nodes = vec![node_a, node_b];
    let weights = HashMap::new();
    let depths = Session::compute_node_depths(&nodes, &weights);
    assert_eq!(depths, vec![0, 0]);
}

#[test]
fn test_group_by_depth() {
    let depths = vec![0, 0, 1, 2, 1];
    let groups = Session::group_by_depth(&depths);
    assert_eq!(groups.len(), 3);
    assert_eq!(groups[0], vec![0, 1]);
    assert_eq!(groups[1], vec![2, 4]);
    assert_eq!(groups[2], vec![3]);
}
