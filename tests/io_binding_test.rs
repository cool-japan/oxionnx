//! Tests for IoBinding and Session::run_with_binding.

use oxionnx::{IoBinding, Session, Tensor};
use oxionnx_core::{Attributes, Graph, Node, OpKind};
use std::collections::HashMap;

fn make_identity_session() -> Session {
    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Identity,
            name: "id".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };
    Session::from_graph(graph, HashMap::new()).expect("session creation should succeed")
}

#[test]
fn test_io_binding_basic() {
    let session = make_identity_session();
    let mut binding = IoBinding::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    binding.bind_input("x", Tensor::new(data.clone(), shape.clone()));

    session
        .run_with_binding(&mut binding)
        .expect("run_with_binding should succeed");

    let out = binding
        .get_output("y")
        .expect("output 'y' should be present");
    assert_eq!(out.shape, shape);
    assert_eq!(out.data, data);
}

#[test]
fn test_io_binding_output_reuse() {
    let session = make_identity_session();
    let mut binding = IoBinding::new();

    let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];

    // First run
    binding.bind_input("x", Tensor::new(data1.clone(), shape.clone()));
    session
        .run_with_binding(&mut binding)
        .expect("first run_with_binding should succeed");

    let out1 = binding.get_output("y").expect("output 'y' after first run");
    assert_eq!(out1.data, data1);
    assert_eq!(out1.shape, shape);

    // Second run with same shapes — output buffer should be reused
    let data2 = vec![5.0f32, 6.0, 7.0, 8.0];
    binding.clear_inputs();
    binding.bind_input("x", Tensor::new(data2.clone(), shape.clone()));
    session
        .run_with_binding(&mut binding)
        .expect("second run_with_binding should succeed");

    let out2 = binding
        .get_output("y")
        .expect("output 'y' after second run");
    assert_eq!(out2.data, data2, "output should reflect new input values");
    assert_eq!(out2.shape, shape);
}

#[test]
fn test_io_binding_clear_inputs() {
    let session = make_identity_session();
    let mut binding = IoBinding::new();

    let data1 = vec![10.0f32, 20.0, 30.0];
    let shape = vec![3];

    // First run
    binding.bind_input("x", Tensor::new(data1.clone(), shape.clone()));
    session
        .run_with_binding(&mut binding)
        .expect("first run_with_binding should succeed");

    let out1 = binding.get_output("y").expect("output 'y' after first run");
    assert_eq!(out1.data, data1);

    // clear_inputs then rebind with different values
    binding.clear_inputs();

    // After clear_inputs, input_names should be empty
    assert_eq!(binding.input_names().count(), 0);

    let data2 = vec![100.0f32, 200.0, 300.0];
    binding.bind_input("x", Tensor::new(data2.clone(), shape.clone()));
    session
        .run_with_binding(&mut binding)
        .expect("second run_with_binding should succeed");

    let out2 = binding
        .get_output("y")
        .expect("output 'y' after second run");
    assert_eq!(
        out2.data, data2,
        "output should have changed after rebinding input"
    );
    assert_ne!(out2.data, data1, "output should not be the old values");
}
