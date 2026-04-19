use super::super::Session;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

#[test]
fn test_with_intra_threads_loads_empty_model() {
    let session = Session::builder()
        .with_intra_threads(2)
        .load_from_bytes(&[]);
    assert!(session.is_ok());
    let session = session.expect("should build with intra_threads");
    assert!(session.parallel);
    #[cfg(not(target_arch = "wasm32"))]
    assert!(session.thread_pool.is_some());
}

#[test]
fn test_with_inter_threads_accepted() {
    let session = Session::builder()
        .with_inter_threads(3)
        .load_from_bytes(&[]);
    assert!(session.is_ok());
    let session = session.expect("should build with inter_threads");
    assert!(session.parallel);
    #[cfg(not(target_arch = "wasm32"))]
    assert!(session.thread_pool.is_some());
}

#[test]
fn test_thread_pool_single_thread_same_results() {
    // Build a simple Relu graph and verify 1-thread pool produces same results
    let node = Node {
        op: OpKind::Relu,
        name: "relu".to_string(),
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

    // Default session (no thread pool)
    let session_default =
        Session::from_graph(graph.clone(), HashMap::new()).expect("default build");
    let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
    let out_default = session_default
        .run_one("input", input.clone())
        .expect("default run");

    // Session with 1 thread
    let session_1t = Session::builder()
        .with_intra_threads(1)
        .build_from_graph(graph, HashMap::new())
        .expect("1-thread build");
    let out_1t = session_1t.run_one("input", input).expect("1-thread run");

    let d = out_default.get("output").expect("default output");
    let t = out_1t.get("output").expect("1-thread output");
    assert_eq!(d.data, t.data);
    assert_eq!(d.shape, t.shape);
}

#[test]
fn test_thread_pool_four_threads_same_results() {
    // Build a graph with two independent branches to exercise parallel path.
    // Use different inputs so CSE doesn't merge the two Relu nodes.
    let relu_a = Node {
        op: OpKind::Relu,
        name: "relu_a".to_string(),
        inputs: vec!["input_a".to_string()],
        outputs: vec!["out_a".to_string()],
        attrs: Attributes::default(),
    };
    let relu_b = Node {
        op: OpKind::Relu,
        name: "relu_b".to_string(),
        inputs: vec!["input_b".to_string()],
        outputs: vec!["out_b".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![relu_a, relu_b],
        input_names: vec!["input_a".to_string(), "input_b".to_string()],
        output_names: vec!["out_a".to_string(), "out_b".to_string()],
        ..Default::default()
    };

    let input_a = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let input_b = Tensor::new(vec![3.0, -4.0, 5.0, -6.0, 7.0], vec![5]);

    // Default session
    let session_default =
        Session::from_graph(graph.clone(), HashMap::new()).expect("default build");
    let mut inputs_default = HashMap::new();
    inputs_default.insert("input_a", input_a.clone());
    inputs_default.insert("input_b", input_b.clone());
    let out_default = session_default.run(&inputs_default).expect("default run");

    // Session with 4 threads
    let session_4t = Session::builder()
        .with_intra_threads(4)
        .build_from_graph(graph, HashMap::new())
        .expect("4-thread build");
    let mut inputs_4t = HashMap::new();
    inputs_4t.insert("input_a", input_a);
    inputs_4t.insert("input_b", input_b);
    let out_4t = session_4t.run(&inputs_4t).expect("4-thread run");

    let da = out_default.get("out_a").expect("default out_a");
    let db = out_default.get("out_b").expect("default out_b");
    let ta = out_4t.get("out_a").expect("4t out_a");
    let tb = out_4t.get("out_b").expect("4t out_b");
    assert_eq!(da.data, ta.data);
    assert_eq!(db.data, tb.data);
}

#[test]
fn test_no_thread_pool_by_default() {
    let session = Session::from_bytes(&[]).expect("load empty model");
    #[cfg(not(target_arch = "wasm32"))]
    assert!(session.thread_pool.is_none());
    assert!(!session.parallel);
}
