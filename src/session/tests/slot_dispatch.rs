use super::super::types::OptLevel;
use super::super::Session;
use super::super::SessionBuilder;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Verify the Phase F execute_into_slots dispatch path for ReluOp.
///
/// ReluOp overrides `supports_output_slots() -> true`, so when static shapes
/// are resolved, `execute_node_with_inplace` should use the slot-writing path
/// rather than the normal allocate-then-return path. The results must be
/// numerically identical either way.
#[test]
fn test_phase_f_execute_into_slots_relu() {
    let node = Node {
        op: OpKind::Relu,
        name: "relu_slot".to_string(),
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

    // Build session with pool enabled so the slot path can acquire from pool
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_memory_pool(true)
        .build_from_graph(graph, HashMap::new())
        .expect("build session for Phase F test");

    let input = Tensor::new(vec![-3.0, -1.0, 0.0, 2.0, 5.0], vec![1, 5]);
    let result = session.run_one("x", input).expect("run Phase F relu");
    let y = result.get("y").expect("output y");

    // Relu: max(x, 0)
    assert_eq!(y.data, vec![0.0, 0.0, 0.0, 2.0, 5.0]);
    assert_eq!(y.shape, vec![1, 5]);
}

/// Verify that IoBinding's take/put helpers work correctly.
#[test]
fn test_io_binding_take_put_helpers() {
    use crate::IoBinding;

    let mut binding = IoBinding::new();
    let t = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    binding.bind_output("out", t.clone());

    // take should remove it
    let taken = binding.take_output_buffer("out");
    assert!(taken.is_some());
    assert_eq!(taken.as_ref().map(|t| &t.data), Some(&t.data));
    // After take, it should be gone
    assert!(binding.take_output_buffer("out").is_none());

    // put should re-insert it
    binding.put_output_buffer("out".to_string(), t.clone());
    assert!(binding.get_output("out").is_some());
}

/// Verify that run_with_binding uses the new put/take helpers and
/// correctly reuses pre-allocated buffers.
#[test]
fn test_run_with_binding_uses_new_helpers() {
    use crate::IoBinding;

    let node = Node {
        op: OpKind::Relu,
        name: "relu_binding".to_string(),
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
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build");

    let mut binding = IoBinding::new();
    binding.bind_input("x", Tensor::new(vec![-1.0, 3.0, -2.0], vec![1, 3]));
    // Pre-allocate output buffer with correct shape
    binding.bind_output("y", Tensor::new(vec![0.0; 3], vec![1, 3]));

    session
        .run_with_binding(&mut binding)
        .expect("run_with_binding");

    let y = binding.get_output("y").expect("output");
    // Relu of [-1, 3, -2] = [0, 3, 0]
    assert_eq!(y.data, vec![0.0, 3.0, 0.0]);
}

/// Helper: build a single-op session with memory pool enabled.
fn single_op_pooled_session(op: OpKind, input_name: &str, output_name: &str) -> Session {
    let node = Node {
        name: format!("{:?}_slot_test", op),
        op,
        inputs: vec![input_name.to_string()],
        outputs: vec![output_name.to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node],
        input_names: vec![input_name.to_string()],
        output_names: vec![output_name.to_string()],
        ..Default::default()
    };
    SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_memory_pool(true)
        .build_from_graph(graph, HashMap::new())
        .expect("build single-op pooled session")
}

/// Helper: build a two-input op session with memory pool enabled.
fn binary_op_pooled_session(
    op: OpKind,
    input_a: &str,
    input_b: &str,
    output_name: &str,
) -> Session {
    let node = Node {
        name: format!("{:?}_slot_test", op),
        op,
        inputs: vec![input_a.to_string(), input_b.to_string()],
        outputs: vec![output_name.to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![node],
        input_names: vec![input_a.to_string()],
        output_names: vec![output_name.to_string()],
        ..Default::default()
    };
    let mut weights = HashMap::new();
    weights.insert(
        input_b.to_string(),
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
    );
    SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_memory_pool(true)
        .build_from_graph(graph, weights)
        .expect("build binary-op pooled session")
}

/// Phase F: Sigmoid slot path produces sigmoid(x) = 1/(1+exp(-x)).
#[test]
fn test_phase_f_execute_into_slots_sigmoid() {
    let session = single_op_pooled_session(OpKind::Sigmoid, "x", "y");
    let input = Tensor::new(vec![0.0f32, 2.0, -2.0], vec![3]);
    let result = session.run_one("x", input).expect("run Phase F sigmoid");
    let y = result.get("y").expect("output y");
    assert_eq!(y.shape, vec![3]);
    // sigmoid(0) = 0.5
    assert!((y.data[0] - 0.5f32).abs() < 1e-5, "sigmoid(0) ≈ 0.5");
    // sigmoid(2) ≈ 0.8808
    assert!(
        (y.data[1] - 0.880_797f32).abs() < 1e-4,
        "sigmoid(2) ≈ 0.8808"
    );
    // All values in (0, 1)
    for &v in &y.data {
        assert!(
            (0.0..=1.0).contains(&v),
            "sigmoid output must be in [0, 1], got {v}"
        );
    }
}

/// Phase F: Tanh slot path produces tanh(x).
#[test]
fn test_phase_f_execute_into_slots_tanh() {
    let session = single_op_pooled_session(OpKind::Tanh, "x", "y");
    let input = Tensor::new(vec![0.0f32, 1.0, -1.0], vec![3]);
    let result = session.run_one("x", input).expect("run Phase F tanh");
    let y = result.get("y").expect("output y");
    assert_eq!(y.shape, vec![3]);
    assert!((y.data[0] - 0.0f32).abs() < 1e-6, "tanh(0) = 0");
    assert!((y.data[1] - 0.761_594f32).abs() < 1e-4, "tanh(1) ≈ 0.7616");
    assert!(
        (y.data[2] + 0.761_594f32).abs() < 1e-4,
        "tanh(-1) ≈ -0.7616"
    );
}

/// Phase F: Identity slot path copies input unchanged.
#[test]
fn test_phase_f_execute_into_slots_identity() {
    let session = single_op_pooled_session(OpKind::Identity, "x", "y");
    let input = Tensor::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]);
    let result = session
        .run_one("x", input.clone())
        .expect("run Phase F identity");
    let y = result.get("y").expect("output y");
    assert_eq!(y.data, input.data);
    assert_eq!(y.shape, vec![2, 2]);
}

/// Phase F: Add slot path with pre-allocated shape-matched buffer.
#[test]
fn test_phase_f_execute_into_slots_add() {
    let session = binary_op_pooled_session(OpKind::Add, "x", "w", "y");
    let input = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3]);
    let result = session.run_one("x", input).expect("run Phase F add");
    let y = result.get("y").expect("output y");
    // input [1, 2, 3] + weight [1, 2, 3] = [2, 4, 6]
    assert_eq!(y.data, vec![2.0, 4.0, 6.0]);
    assert_eq!(y.shape, vec![3]);
}

/// Phase F: Mul slot path with weight tensor.
#[test]
fn test_phase_f_execute_into_slots_mul() {
    let session = binary_op_pooled_session(OpKind::Mul, "x", "w", "y");
    let input = Tensor::new(vec![2.0f32, 3.0, 4.0], vec![3]);
    let result = session.run_one("x", input).expect("run Phase F mul");
    let y = result.get("y").expect("output y");
    // input [2, 3, 4] * weight [1, 2, 3] = [2, 6, 12]
    assert_eq!(y.data, vec![2.0, 6.0, 12.0]);
}

/// Phase F: Abs slot path produces element-wise absolute values.
#[test]
fn test_phase_f_execute_into_slots_abs() {
    let session = single_op_pooled_session(OpKind::Abs, "x", "y");
    let input = Tensor::new(vec![-3.0f32, 0.0, 4.0, -7.5], vec![4]);
    let result = session.run_one("x", input).expect("run Phase F abs");
    let y = result.get("y").expect("output y");
    assert_eq!(y.data, vec![3.0, 0.0, 4.0, 7.5]);
}

/// Phase F: Exp slot path produces e^x.
#[test]
fn test_phase_f_execute_into_slots_exp() {
    let session = single_op_pooled_session(OpKind::Exp, "x", "y");
    let input = Tensor::new(vec![0.0f32, 1.0, -1.0], vec![3]);
    let result = session.run_one("x", input).expect("run Phase F exp");
    let y = result.get("y").expect("output y");
    assert_eq!(y.shape, vec![3]);
    // exp(0) = 1.0
    assert!(
        (y.data[0] - 1.0f32).abs() < 1e-6,
        "exp(0) = 1, got {}",
        y.data[0]
    );
    // exp(1) ≈ 2.71828
    assert!((y.data[1] - std::f32::consts::E).abs() < 1e-5, "exp(1) ≈ e");
    // exp(-1) ≈ 0.36788
    assert!(
        (y.data[2] - 1.0 / std::f32::consts::E).abs() < 1e-5,
        "exp(-1) ≈ 1/e"
    );
}

/// Phase F: Slot path with wrong pre-allocated shape reallocates gracefully.
///
/// The slot contract allows the op to reallocate `*slot = new_tensor` when
/// the shapes don't match.  This test verifies no panic occurs and output
/// is still numerically correct.
#[test]
fn test_phase_f_slot_shape_mismatch_reallocates() {
    let session = single_op_pooled_session(OpKind::Relu, "x", "y");
    // Provide a larger input than any pre-allocated shape the pool might have seen.
    let large_data: Vec<f32> = (0..1000).map(|i| i as f32 - 500.0).collect();
    let input = Tensor::new(large_data, vec![100, 10]);
    let result = session
        .run_one("x", input)
        .expect("run Phase F slot mismatch");
    let y = result.get("y").expect("output y");
    assert_eq!(y.shape, vec![100, 10]);
    for (i, &v) in y.data.iter().enumerate() {
        let original = i as f32 - 500.0;
        let expected = if original < 0.0 { 0.0 } else { original };
        assert!(
            (v - expected).abs() < 1e-6,
            "relu mismatch at index {i}: got {v}, expected {expected}"
        );
    }
}

/// Phase F: Chain of slot-capable ops (Relu → Sigmoid) both use slot path.
///
/// Verifies that the slot path correctly chains: the pre-allocated output
/// from Relu becomes the input to Sigmoid's slot, both sharing the pool.
#[test]
fn test_phase_f_slot_chain_relu_sigmoid() {
    let relu = Node {
        op: OpKind::Relu,
        name: "relu".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["mid".to_string()],
        attrs: Attributes::default(),
    };
    let sigmoid = Node {
        op: OpKind::Sigmoid,
        name: "sigmoid".to_string(),
        inputs: vec!["mid".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };
    let graph = Graph {
        nodes: vec![relu, sigmoid],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_memory_pool(true)
        .build_from_graph(graph, HashMap::new())
        .expect("build relu→sigmoid chain session");

    let input = Tensor::new(vec![-2.0f32, 0.0, 2.0], vec![3]);
    let result = session.run_one("x", input).expect("run Phase F chain");
    let y = result.get("y").expect("output y");
    assert_eq!(y.shape, vec![3]);
    // relu(-2) = 0 → sigmoid(0) = 0.5
    assert!((y.data[0] - 0.5f32).abs() < 1e-5, "sigmoid(relu(-2)) ≈ 0.5");
    // relu(0) = 0 → sigmoid(0) = 0.5
    assert!((y.data[1] - 0.5f32).abs() < 1e-5, "sigmoid(relu(0)) ≈ 0.5");
    // relu(2) = 2 → sigmoid(2) ≈ 0.8808
    assert!(
        (y.data[2] - 0.880_797f32).abs() < 1e-4,
        "sigmoid(relu(2)) ≈ 0.8808"
    );
}

/// Phase F: slot path with profiling enabled records node profiles.
///
/// When the slot dispatch path is taken, profiling data must still be
/// collected and accessible via `session.profiling_results()`.
#[test]
fn test_phase_f_slot_path_profiling() {
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_memory_pool(true)
        .with_profiling()
        .build_from_graph(
            Graph {
                nodes: vec![Node {
                    op: OpKind::Relu,
                    name: "relu_profiled".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["y".to_string()],
                    attrs: Attributes::default(),
                }],
                input_names: vec!["x".to_string()],
                output_names: vec!["y".to_string()],
                ..Default::default()
            },
            HashMap::new(),
        )
        .expect("build profiled slot session");

    let input = Tensor::new(vec![-1.0f32, 2.0, -3.0], vec![3]);
    session.run_one("x", input).expect("run");

    let profiles = session.profiling_results().expect("profiling enabled");
    assert!(!profiles.is_empty(), "profiles must not be empty after run");
    let relu_profile = profiles
        .iter()
        .find(|p| p.node_name == "relu_profiled")
        .expect("relu_profiled must appear in profiles");
    assert!(
        !relu_profile.op_type.is_empty(),
        "op_type must be non-empty"
    );
}
