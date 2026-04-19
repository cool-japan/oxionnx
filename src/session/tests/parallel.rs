use super::super::types::OptLevel;
use super::super::Session;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Build a graph with two independent branches at the same depth:
///   input -> Relu(branch_a) -> output_a
///   input -> Relu(branch_b) -> output_b
/// Both Relu nodes are at depth 0 and should run in parallel.
#[test]
fn test_parallel_execution_basic() {
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
        .with_parallel_execution(true)
        .build_from_graph(graph, HashMap::new())
        .expect("build");

    let input = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
    let outputs = session.run_one("input", input).expect("run");

    let expected = vec![0.0, 2.0, 0.0, 4.0];
    let out_a = outputs.get("out_a").expect("out_a");
    let out_b = outputs.get("out_b").expect("out_b");
    assert_eq!(out_a.data, expected);
    assert_eq!(out_b.data, expected);
}

/// All nodes sequential (linear chain). Parallel mode should not break anything.
#[test]
fn test_parallel_single_node_levels() {
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
    let graph = Graph {
        nodes: vec![node1, node2],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_parallel_execution(true)
        .build_from_graph(graph, HashMap::new())
        .expect("build");

    let input = Tensor::new(vec![-1.0, 5.0, -2.0], vec![1, 3]);
    let outputs = session.run_one("input", input).expect("run");
    let out = outputs.get("output").expect("output");
    assert_eq!(out.data, vec![0.0, 5.0, 0.0]);
}

/// Run the same two-branch parallel graph 100 times and verify output values
/// match a single sequential run. Exercises pool buffer reuse across repeated
/// parallel invocations (`SessionRunState` path via `run_parallel_inner`).
#[test]
fn test_parallel_pool_reuse() {
    // Graph: input -> Relu(branch_a) -> out_a
    //                Relu(branch_b) -> out_b
    // Both Relu nodes share the same input and lie at the same topological depth,
    // so they are dispatched via the multi-node rayon path on every call.
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

    // Build sequential session for reference.
    let seq_session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_parallel_execution(false)
        .build_from_graph(graph.clone(), HashMap::new())
        .expect("build seq");

    let reference_input = Tensor::new(vec![-2.0, 0.5, 1.0, -0.5], vec![2, 2]);
    let reference_outputs = seq_session
        .run_one("input", reference_input)
        .expect("seq run");
    let ref_a = reference_outputs
        .get("out_a")
        .expect("ref out_a")
        .data
        .clone();
    let ref_b = reference_outputs
        .get("out_b")
        .expect("ref out_b")
        .data
        .clone();

    // Build parallel session with memory pool enabled (exercises buffer reuse).
    let par_session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_parallel_execution(true)
        .with_memory_pool(true)
        .build_from_graph(graph, HashMap::new())
        .expect("build par");

    // Run 100 times — buffers released on each iteration should be reused.
    for i in 0..100_usize {
        let input = Tensor::new(vec![-2.0, 0.5, 1.0, -0.5], vec![2, 2]);
        let outputs = par_session
            .run_one("input", input)
            .unwrap_or_else(|e| panic!("parallel run failed at iteration {i}: {e}"));
        let out_a = outputs.get("out_a").expect("out_a");
        let out_b = outputs.get("out_b").expect("out_b");
        assert_eq!(out_a.data, ref_a, "out_a mismatch at iteration {i}");
        assert_eq!(out_b.data, ref_b, "out_b mismatch at iteration {i}");
    }
}
