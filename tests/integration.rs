//! Integration tests for oxionnx using `Session::from_graph()`.

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, Session, Tensor};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

fn make_node_with_attrs(
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

// ── Test 1: Identity graph ──────────────────────────────────────────────────

#[test]
fn test_identity_graph() {
    let graph = Graph {
        nodes: vec![make_node(OpKind::Identity, "id0", &["x"], &["y"])],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
    };
    let weights: HashMap<String, Tensor> = HashMap::new();

    let session = Session::from_graph(graph, weights).expect("build session");
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let outputs = session.run_one("x", input.clone()).expect("run");

    let y = outputs.get("y").expect("output y");
    assert_eq!(y.shape, vec![1, 3]);
    assert_eq!(y.data, vec![1.0, 2.0, 3.0]);
}

// ── Test 2: Add two constants ───────────────────────────────────────────────

#[test]
fn test_add_two_constants() {
    let graph = Graph {
        nodes: vec![make_node(OpKind::Add, "add0", &["a", "b"], &["sum"])],
        input_names: vec![],
        output_names: vec!["sum".to_string()],
    };
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("a".to_string(), Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));
    weights.insert("b".to_string(), Tensor::new(vec![4.0, 5.0, 6.0], vec![3]));

    let session = Session::builder()
        .with_optimization_level(oxionnx::OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build session");
    let inputs: HashMap<&str, Tensor> = HashMap::new();
    let outputs = session.run(&inputs).expect("run");

    let sum = outputs.get("sum").expect("output sum");
    assert_eq!(sum.shape, vec![3]);
    assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);
}

// ── Test 3: Simple linear layer (MatMul + Add) ─────────────────────────────

#[test]
fn test_linear_layer() {
    // x [1,3] @ W [3,2] => mm [1,2], then mm + b [2] => out [1,2]
    let graph = Graph {
        nodes: vec![
            make_node(OpKind::MatMul, "matmul0", &["x", "W"], &["mm"]),
            make_node(OpKind::Add, "add0", &["mm", "b"], &["out"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["out".to_string()],
    };

    // W = [[1, 0],
    //      [0, 1],
    //      [1, 1]]  shape [3, 2]
    let w_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let b_data = vec![0.5, -0.5];
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("W".to_string(), Tensor::new(w_data, vec![3, 2]));
    weights.insert("b".to_string(), Tensor::new(b_data, vec![2]));

    let session = Session::from_graph(graph, weights).expect("build session");
    // x = [1, 2, 3]  => mm = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    // out = [4+0.5, 5-0.5] = [4.5, 4.5]
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let outputs = session.run_one("x", x).expect("run");

    let out = outputs.get("out").expect("output out");
    assert_eq!(out.shape, vec![1, 2]);
    let expected = [4.5, 4.5];
    for (a, b) in out.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }
}

// ── Test 4: Conv2D + ReLU ───────────────────────────────────────────────────

#[test]
fn test_conv2d_relu() {
    // Conv attributes
    let mut conv_attrs = Attributes::default();
    conv_attrs
        .int_lists
        .insert("strides".to_string(), vec![1, 1]);
    conv_attrs
        .int_lists
        .insert("pads".to_string(), vec![0, 0, 0, 0]);
    conv_attrs
        .int_lists
        .insert("dilations".to_string(), vec![1, 1]);
    conv_attrs.ints.insert("group".to_string(), 1);

    let graph = Graph {
        nodes: vec![
            make_node_with_attrs(
                OpKind::Conv,
                "conv0",
                &["input", "conv_w"],
                &["conv_out"],
                conv_attrs,
            ),
            make_node(OpKind::Relu, "relu0", &["conv_out"], &["relu_out"]),
        ],
        input_names: vec!["input".to_string()],
        output_names: vec!["relu_out".to_string()],
    };

    // Input: 5x5 image, all ones
    let input_data = vec![1.0_f32; 25];
    // Kernel: 3x3, all ones except center is -10 (to produce some negatives)
    let mut kernel_data = vec![1.0_f32; 9];
    kernel_data[4] = -10.0; // center element

    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert(
        "conv_w".to_string(),
        Tensor::new(kernel_data, vec![1, 1, 3, 3]),
    );

    let session = Session::from_graph(graph, weights).expect("build session");
    let input = Tensor::new(input_data, vec![1, 1, 5, 5]);
    let outputs = session.run_one("input", input).expect("run");

    let relu_out = outputs.get("relu_out").expect("output relu_out");
    assert_eq!(relu_out.shape, vec![1, 1, 3, 3]);

    // Each conv output = sum of 9 kernel values * 1.0 = (8*1 + 1*(-10)) = -2
    // After ReLU: max(0, -2) = 0
    for &v in &relu_out.data {
        assert!(v >= 0.0, "ReLU output should be non-negative, got {v}");
    }
    // With all-ones input and our kernel, every output should be 0
    for &v in &relu_out.data {
        assert!((v - 0.0).abs() < 1e-6, "expected 0.0, got {v}");
    }
}

// ── Test 5: Multi-output Split ──────────────────────────────────────────────

#[test]
fn test_split() {
    let mut split_attrs = Attributes::default();
    split_attrs.ints.insert("axis".to_string(), 1);
    split_attrs
        .int_lists
        .insert("split".to_string(), vec![3, 3]);

    let graph = Graph {
        nodes: vec![make_node_with_attrs(
            OpKind::Split,
            "split0",
            &["x"],
            &["a", "b"],
            split_attrs,
        )],
        input_names: vec!["x".to_string()],
        output_names: vec!["a".to_string(), "b".to_string()],
    };
    let weights: HashMap<String, Tensor> = HashMap::new();

    let session = Session::from_graph(graph, weights).expect("build session");
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
    let outputs = session.run_one("x", x).expect("run");

    let a = outputs.get("a").expect("output a");
    let b = outputs.get("b").expect("output b");
    assert_eq!(a.shape, vec![1, 3]);
    assert_eq!(b.shape, vec![1, 3]);
    assert_eq!(a.data, vec![1.0, 2.0, 3.0]);
    assert_eq!(b.data, vec![4.0, 5.0, 6.0]);
}

// ── Test 6: Session builder + from_graph, model_info, export_dot ────────────

#[test]
fn test_session_builder_and_introspection() {
    let graph = Graph {
        nodes: vec![
            make_node(OpKind::MatMul, "matmul_node", &["x", "W"], &["mm"]),
            make_node(OpKind::Relu, "relu_node", &["mm"], &["out"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["out".to_string()],
    };
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("W".to_string(), Tensor::new(vec![1.0; 6], vec![3, 2]));

    let session = Session::builder()
        .with_optimization_level(oxionnx::OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build session");

    // model_info
    let info = session.model_info();
    assert!(info.node_count >= 2, "expected at least 2 nodes");
    assert!(
        info.op_histogram.contains_key("MatMul"),
        "histogram should contain MatMul"
    );
    assert!(
        info.op_histogram.contains_key("Relu"),
        "histogram should contain Relu"
    );
    assert_eq!(info.parameter_count, 6); // W has 6 elements

    // export_dot
    let dot = session.export_dot();
    assert!(dot.contains("digraph"), "DOT should start with digraph");
    assert!(
        dot.contains("matmul_node"),
        "DOT should mention matmul_node"
    );
    assert!(dot.contains("relu_node"), "DOT should mention relu_node");

    // input/output names
    assert_eq!(session.input_names(), &["x"]);
    assert_eq!(session.output_names(), &["out"]);
}

// ── Test 7: Profiling ───────────────────────────────────────────────────────

#[test]
fn test_profiling() {
    let graph = Graph {
        nodes: vec![
            make_node(OpKind::Identity, "id_a", &["x"], &["mid"]),
            make_node(OpKind::Identity, "id_b", &["mid"], &["y"]),
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
    };
    let weights: HashMap<String, Tensor> = HashMap::new();

    let session = Session::builder()
        .with_profiling()
        .build_from_graph(graph, weights)
        .expect("build session");

    // Before running, profiling data should be empty
    let initial = session.profiling_results().expect("profiling enabled");
    assert!(initial.is_empty(), "no profiles before first run");

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let _outputs = session.run_one("x", x).expect("run");

    let profiles = session.profiling_results().expect("profiling enabled");
    assert!(
        !profiles.is_empty(),
        "profiling results should not be empty after run"
    );

    for p in &profiles {
        assert!(!p.node_name.is_empty(), "node_name should not be empty");
        assert!(!p.op_type.is_empty(), "op_type should not be empty");
    }

    // Verify we captured both nodes
    let names: Vec<&str> = profiles.iter().map(|p| p.node_name.as_str()).collect();
    assert!(names.contains(&"id_a"), "profiles should contain id_a");
    assert!(names.contains(&"id_b"), "profiles should contain id_b");
}
