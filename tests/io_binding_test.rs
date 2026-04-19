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

// ── New Phase F / IoBinding tests ────────────────────────────────────────────

/// Pre-binding an output buffer with correct shape causes run_with_binding to
/// copy data in-place. The output pointer address is stable (no realloc).
#[test]
fn test_bind_output_prealloc_copy_in_place() {
    let session = make_identity_session();
    let mut binding = IoBinding::new();

    let shape = vec![4];
    let initial_buf = vec![0.0f32; 4];

    // Pre-bind an output buffer of the exact shape
    binding.bind_output("y", Tensor::new(initial_buf.clone(), shape.clone()));

    let input_data = vec![7.0f32, 8.0, 9.0, 10.0];
    binding.bind_input("x", Tensor::new(input_data.clone(), shape.clone()));

    session
        .run_with_binding(&mut binding)
        .expect("run_with_binding with pre-bound output");

    let out = binding
        .get_output("y")
        .expect("output 'y' should be present");
    assert_eq!(
        out.data, input_data,
        "pre-bound buffer should have input data"
    );
    assert_eq!(out.shape, shape);
}

/// When bind_output provides a buffer with the wrong shape, run_with_binding
/// replaces the buffer entirely with the correctly-shaped result.
#[test]
fn test_bind_output_shape_mismatch_replaced() {
    let session = make_identity_session();
    let mut binding = IoBinding::new();

    // Pre-bind a buffer with a different shape
    binding.bind_output("y", Tensor::new(vec![0.0f32; 2], vec![2]));

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let shape = vec![4];
    binding.bind_input("x", Tensor::new(input_data.clone(), shape.clone()));

    session
        .run_with_binding(&mut binding)
        .expect("run_with_binding with mismatched pre-bound output");

    let out = binding
        .get_output("y")
        .expect("output 'y' should be present");
    assert_eq!(out.shape, shape, "shape must match input after replacement");
    assert_eq!(
        out.data, input_data,
        "data must match input after replacement"
    );
}

/// bind_output + get_output round-trip: bound buffer is readable and named correctly.
#[test]
fn test_bind_output_readable_via_get_output() {
    let mut binding = IoBinding::new();
    let data = vec![1.0f32, 2.0];
    binding.bind_output("z", Tensor::new(data.clone(), vec![2]));

    // Buffer visible through get_output immediately after binding
    let found = binding.get_output("z").expect("'z' should be present");
    assert_eq!(found.data, data);

    // output_names iterator includes "z"
    let names: Vec<&str> = binding.output_names().collect();
    assert!(names.contains(&"z"), "output_names must include 'z'");

    // Absent key returns None
    assert!(
        binding.get_output("not_there").is_none(),
        "absent key must return None"
    );
}

/// supports_output_slots returns true for IdentityOp and execute_into_slots
/// copies data into a correctly-sized slot.
#[test]
fn test_execute_into_slots_identity_success() {
    use oxionnx_core::{Attributes, Node, OpContext, OpKind, Operator, Tensor};
    use oxionnx_ops::registry::misc_ops::IdentityOp;

    let input = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]);
    let node = Node {
        op: OpKind::Identity,
        name: "id".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = OpContext {
        node: &node,
        inputs: vec![Some(&input)],
        outer_scope: None,
        registry: None,
    };

    let op = IdentityOp;
    assert!(
        op.supports_output_slots(),
        "IdentityOp must support output slots"
    );

    // Provide a correctly-shaped slot
    let mut slots = vec![Tensor::new(vec![0.0f32; 3], vec![3])];
    op.execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots should succeed with matching slot");

    assert_eq!(slots[0].data, vec![1.0f32, 2.0, 3.0]);
    assert_eq!(slots[0].shape, vec![3]);
}

/// execute_into_slots on IdentityOp with empty slots returns an error.
#[test]
fn test_execute_into_slots_identity_empty_slots_error() {
    use oxionnx_core::{Attributes, Node, OpContext, OpKind, Operator, Tensor};
    use oxionnx_ops::registry::misc_ops::IdentityOp;

    let input = Tensor::new(vec![1.0f32, 2.0], vec![2]);
    let node = Node {
        op: OpKind::Identity,
        name: "id".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = OpContext {
        node: &node,
        inputs: vec![Some(&input)],
        outer_scope: None,
        registry: None,
    };

    let op = IdentityOp;
    let mut slots: Vec<Tensor> = vec![];
    let result = op.execute_into_slots(&ctx, &mut slots);
    assert!(
        result.is_err(),
        "execute_into_slots with empty slots must return error for IdentityOp"
    );
}

/// execute_into_slots on AddOp writes sum into a pre-allocated slot.
#[test]
fn test_execute_into_slots_add_op() {
    use oxionnx_core::{Attributes, Node, OpContext, OpKind, Operator, Tensor};
    use oxionnx_ops::registry::math_ops::AddOp;

    let a = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0f32, 5.0, 6.0], vec![3]);
    let node = Node {
        op: OpKind::Add,
        name: "add".to_string(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = OpContext {
        node: &node,
        inputs: vec![Some(&a), Some(&b)],
        outer_scope: None,
        registry: None,
    };

    let op = AddOp;
    assert!(
        op.supports_output_slots(),
        "AddOp must support output slots"
    );

    let mut slots = vec![Tensor::new(vec![0.0f32; 3], vec![3])];
    op.execute_into_slots(&ctx, &mut slots)
        .expect("execute_into_slots Add should succeed");

    assert_eq!(slots[0].data, vec![5.0f32, 7.0, 9.0]);
    assert_eq!(slots[0].shape, vec![3]);
}

/// clear() removes both inputs and outputs from a binding.
#[test]
fn test_io_binding_clear_all() {
    let mut binding = IoBinding::new();
    binding.bind_input("x", Tensor::new(vec![1.0f32], vec![1]));
    binding.bind_output("y", Tensor::new(vec![0.0f32], vec![1]));

    assert_eq!(binding.input_names().count(), 1);
    assert_eq!(binding.output_names().count(), 1);

    binding.clear();

    assert_eq!(binding.input_names().count(), 0, "inputs cleared");
    assert_eq!(binding.output_names().count(), 0, "outputs cleared");
}

// ── F.8 / SessionRunState pool-reuse regression tests ────────────────────────

/// Run a single Add node 100 times via `session.run()` and verify every output
/// is correct. Exercises the `SessionRunState` pool-reuse path: buffers
/// released after each iteration are reclaimed by the pool and reused in the
/// next, so any aliasing or stale-data bug would manifest as wrong values.
#[test]
fn test_pool_reuse_across_consecutive_runs() {
    use oxionnx_core::Graph;

    let add_node = oxionnx_core::Node {
        op: oxionnx_core::OpKind::Add,
        name: "add".to_string(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        attrs: oxionnx_core::Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![add_node],
        input_names: vec!["a".to_string()],
        output_names: vec!["c".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    // Constant "b" weight so only "a" is a runtime input.
    let mut weights = HashMap::new();
    weights.insert(
        "b".to_string(),
        Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]),
    );

    let session = Session::from_graph(graph, weights).expect("session build");

    for i in 0..100_usize {
        let factor = (i + 1) as f32;
        let a_data = vec![factor, factor * 2.0, factor * 3.0];
        let expected = vec![factor + 1.0, factor * 2.0 + 2.0, factor * 3.0 + 3.0];

        let mut inputs = HashMap::new();
        inputs.insert("a", Tensor::new(a_data, vec![3]));

        let outputs = session
            .run(&inputs)
            .unwrap_or_else(|e| panic!("run failed at iteration {i}: {e}"));

        let c = outputs
            .get("c")
            .unwrap_or_else(|| panic!("output 'c' missing at iteration {i}"));
        assert_eq!(
            c.data, expected,
            "pool reuse: wrong output at iteration {i}"
        );
        assert_eq!(c.shape, vec![3], "shape mismatch at iteration {i}");
    }
}

/// Regression guard for the `SessionRunState` refactor: build a two-node
/// chain (`Add` followed by `Relu`) and verify the output matches expected
/// values. Uses values that straddle zero so `Relu` actually zeroes elements.
///
/// Graph:  [a, b] -> Add -> "mid" -> Relu -> "out"
///
/// Input:  a = [-2, -1, 1, 2],  b (weight) = [1, 1, 1, 1]
/// After Add:  mid = [-1, 0, 2, 3]
/// After Relu: out = [0, 0, 2, 3]
#[test]
fn test_session_run_correctness_after_f8_refactor() {
    use oxionnx_core::Graph;

    let add_node = oxionnx_core::Node {
        op: oxionnx_core::OpKind::Add,
        name: "add".to_string(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["mid".to_string()],
        attrs: oxionnx_core::Attributes::default(),
    };
    let relu_node = oxionnx_core::Node {
        op: oxionnx_core::OpKind::Relu,
        name: "relu".to_string(),
        inputs: vec!["mid".to_string()],
        outputs: vec!["out".to_string()],
        attrs: oxionnx_core::Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![add_node, relu_node],
        input_names: vec!["a".to_string()],
        output_names: vec!["out".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    // Constant addend "b" = [1, 1, 1, 1]
    let mut weights = HashMap::new();
    weights.insert(
        "b".to_string(),
        Tensor::new(vec![1.0f32, 1.0, 1.0, 1.0], vec![4]),
    );

    let session = Session::from_graph(graph, weights).expect("session build");

    // a = [-2, -1, 1, 2] → Add → [-1, 0, 2, 3] → Relu → [0, 0, 2, 3]
    let mut inputs = HashMap::new();
    inputs.insert("a", Tensor::new(vec![-2.0f32, -1.0, 1.0, 2.0], vec![4]));

    let outputs = session.run(&inputs).expect("session run");
    let out = outputs.get("out").expect("output 'out' should be present");

    assert_eq!(out.shape, vec![4], "output shape must be [4]");
    assert_eq!(
        out.data,
        vec![0.0f32, 0.0, 2.0, 3.0],
        "Add+Relu correctness: expected [0, 0, 2, 3]"
    );
}
