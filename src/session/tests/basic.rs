use super::super::types::OptLevel;
use super::super::Session;
use super::super::SessionBuilder;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

#[test]
fn test_session_from_empty_bytes() {
    let session = Session::from_bytes(&[]).expect("should load empty model");
    let inputs = HashMap::new();
    let outputs = session.run(&inputs).expect("should run empty model");
    assert!(outputs.is_empty());
}

#[test]
fn test_equal_split_helper() {
    use super::equal_split;
    assert_eq!(equal_split(6, 3), vec![2, 2, 2]);
    assert_eq!(equal_split(7, 3), vec![3, 3, 1]);
    assert_eq!(equal_split(4, 1), vec![4]);
}

#[test]
fn test_from_graph_identity() {
    // Build a simple Identity graph: input -> Identity -> output
    let node = Node {
        op: OpKind::Identity,
        name: "id_node".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["output".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };
    let weights = HashMap::new();

    let session = Session::from_graph(graph, weights).expect("from_graph should succeed");
    assert_eq!(session.input_names(), &["input".to_string()]);
    assert_eq!(session.output_names(), &["output".to_string()]);

    let input_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let outputs = session
        .run_one("input", input_tensor.clone())
        .expect("run should succeed");
    let out = outputs.get("output").expect("output should exist");
    assert_eq!(out.data, input_tensor.data);
    assert_eq!(out.shape, input_tensor.shape);
}

#[test]
fn test_builder_load_from_empty_bytes() {
    let result = Session::builder()
        .with_optimization_level(OptLevel::None)
        .load_from_bytes(&[]);
    assert!(result.is_ok());
    let session = result.expect("builder should load empty model");
    assert!(session.input_names().is_empty());
}

#[test]
fn test_model_info() {
    let node1 = Node {
        op: OpKind::Relu,
        name: "relu1".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["r1".to_string()],
        attrs: Attributes::default(),
    };
    let node2 = Node {
        op: OpKind::Relu,
        name: "relu2".to_string(),
        inputs: vec!["r1".to_string()],
        outputs: vec!["out".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node1, node2],
        input_names: vec!["x".to_string()],
        output_names: vec!["out".to_string()],
        ..Default::default()
    };
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), Tensor::new(vec![1.0; 12], vec![3, 4]));

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build_from_graph");
    let info = session.model_info();
    assert_eq!(info.node_count, 2);
    assert_eq!(info.parameter_count, 12);
    assert_eq!(info.weight_bytes, 48); // 12 * 4
    assert_eq!(info.op_histogram.get("Relu").copied().unwrap_or(0), 2);
}

#[test]
fn test_export_dot() {
    let node = Node {
        op: OpKind::Relu,
        name: "relu1".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["x".to_string()],
        output_names: vec!["out".to_string()],
        ..Default::default()
    };
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), Tensor::new(vec![1.0], vec![1]));

    let session = Session::from_graph(graph, weights).expect("from_graph");
    let dot = session.export_dot();

    assert!(dot.starts_with("digraph model {"));
    assert!(dot.ends_with("}\n"));
    assert!(dot.contains("relu1"));
    assert!(dot.contains("Relu"));
    // Weight node should appear as ellipse
    assert!(dot.contains("\"w\""));
    assert!(dot.contains("ellipse"));
    // Edges
    assert!(dot.contains("\"x\" -> \"relu1\""));
    assert!(dot.contains("\"relu1\" -> \"out\""));
}

#[test]
fn test_profiling() {
    let node = Node {
        op: OpKind::Identity,
        name: "id_node".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["output".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };
    let weights = HashMap::new();

    let session = Session::builder()
        .with_profiling()
        .build_from_graph(graph, weights)
        .expect("build should succeed");

    // Before running, profiling data should be empty
    let initial = session.profiling_results().expect("profiling enabled");
    assert!(initial.is_empty());

    let input_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let _outputs = session
        .run_one("input", input_tensor)
        .expect("run should succeed");

    let profiles = session.profiling_results().expect("profiling enabled");
    assert!(!profiles.is_empty());
    assert_eq!(profiles[0].node_name, "id_node");
    assert_eq!(profiles[0].op_type, "Identity");
    assert_eq!(profiles[0].output_shapes, vec![vec![1, 3]]);
}

#[test]
fn test_profiling_disabled_returns_none() {
    let session = Session::from_bytes(&[]).expect("load empty model");
    assert!(session.profiling_results().is_none());
}

#[test]
fn test_builder_default() {
    let builder = SessionBuilder::default();
    assert_eq!(builder.opt_level, OptLevel::All);
    assert!(!builder.enable_profiling);
    assert!(builder.enable_memory_pool);
    assert!(builder.registry.is_none());
}

#[test]
fn test_opt_level_variants() {
    assert_ne!(OptLevel::None, OptLevel::Basic);
    assert_ne!(OptLevel::Basic, OptLevel::Extended);
    assert_ne!(OptLevel::Extended, OptLevel::All);
    // Clone + Copy
    let level = OptLevel::All;
    let cloned = level;
    assert_eq!(level, cloned);
}
