//! Shared helpers for operator integration tests.
#![allow(dead_code)]

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

pub fn make_node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

pub fn make_node_with_attrs(
    op: OpKind,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Attributes,
) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs,
    }
}

/// Run a single-op graph and return outputs.
pub fn run_single_op(
    op: OpKind,
    inputs: Vec<(&str, Tensor)>,
    weights: Vec<(&str, Tensor)>,
    input_names: Vec<&str>,
    node_inputs: Vec<&str>,
    node_output: &str,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let node = make_node_with_attrs(op, "op0", &node_inputs, &[node_output], attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: input_names.iter().map(|s| s.to_string()).collect(),
        output_names: vec![node_output.to_string()],
        ..Default::default()
    };
    let mut w: HashMap<String, Tensor> = HashMap::new();
    for (name, tensor) in weights {
        w.insert(name.to_string(), tensor);
    }
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, w)
        .expect("build session");
    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    for (name, tensor) in inputs {
        feed.insert(name, tensor);
    }
    session.run(&feed).expect("run")
}

/// Run a single-op graph with multiple outputs.
pub fn run_single_op_multi_output(
    op: OpKind,
    inputs: Vec<(&str, Tensor)>,
    weights: Vec<(&str, Tensor)>,
    input_names: Vec<&str>,
    node_inputs: Vec<&str>,
    node_outputs: Vec<&str>,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let node = make_node_with_attrs(op, "op0", &node_inputs, &node_outputs, attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: input_names.iter().map(|s| s.to_string()).collect(),
        output_names: node_outputs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let mut w: HashMap<String, Tensor> = HashMap::new();
    for (name, tensor) in weights {
        w.insert(name.to_string(), tensor);
    }
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, w)
        .expect("build session");
    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    for (name, tensor) in inputs {
        feed.insert(name, tensor);
    }
    session.run(&feed).expect("run")
}

/// Run a single-op graph (conformance-style: separate graph_inputs and node_outputs).
pub fn run_op(
    op: OpKind,
    node_inputs: Vec<&str>,
    node_outputs: Vec<&str>,
    graph_inputs: Vec<&str>,
    input_tensors: Vec<(&str, Tensor)>,
    weights: Vec<(&str, Tensor)>,
    attrs: Attributes,
) -> HashMap<String, Tensor> {
    let node = make_node_with_attrs(op, "op0", &node_inputs, &node_outputs, attrs);
    let graph = Graph {
        nodes: vec![node],
        input_names: graph_inputs.iter().map(|s| s.to_string()).collect(),
        output_names: node_outputs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let mut w: HashMap<String, Tensor> = HashMap::new();
    for (name, tensor) in weights {
        w.insert(name.to_string(), tensor);
    }
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, w)
        .expect("build session");
    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    for (name, tensor) in input_tensors {
        feed.insert(name, tensor);
    }
    session.run(&feed).expect("run")
}

pub fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch (got {} expected {})",
        msg,
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{}: idx {} got {} expected {} (tol {})",
            msg,
            i,
            a,
            e,
            tol
        );
    }
}

pub fn assert_shape(tensor: &Tensor, expected: &[usize], msg: &str) {
    assert_eq!(tensor.shape, expected, "{}: shape mismatch", msg);
}

pub fn assert_tensor_approx(actual: &Tensor, expected: &[f32], tol: f32) {
    assert_eq!(
        actual.data.len(),
        expected.len(),
        "length mismatch: got {} expected {}",
        actual.data.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.data.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {}: {} vs {} (tol={})",
            i,
            a,
            e,
            tol
        );
    }
}
