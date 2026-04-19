use super::super::types::OptLevel;
use super::super::Session;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Helper: build a graph with input_infos that have symbolic dims.
/// Graph: input [batch_size, 3] -> Relu -> output [batch_size, 3]
fn build_dynamic_relu_graph() -> (Graph, HashMap<String, Tensor>) {
    use oxionnx_core::{DType, TensorInfo};

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
        input_infos: vec![TensorInfo {
            name: "input".to_string(),
            dtype: DType::F32,
            shape: vec![None, Some(3)],
            dim_params: vec![Some("batch_size".to_string()), None],
        }],
        output_infos: vec![TensorInfo {
            name: "output".to_string(),
            dtype: DType::F32,
            shape: vec![None, Some(3)],
            dim_params: vec![Some("batch_size".to_string()), None],
        }],
        ..Default::default()
    };

    (graph, HashMap::new())
}

/// Helper: build a graph with two symbolic dims (batch_size and seq_len).
/// Graph: input [batch_size, seq_len] -> Relu -> output [batch_size, seq_len]
fn build_dynamic_batch_seq_graph() -> (Graph, HashMap<String, Tensor>) {
    use oxionnx_core::{DType, TensorInfo};

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
        input_infos: vec![TensorInfo {
            name: "input".to_string(),
            dtype: DType::F32,
            shape: vec![None, None],
            dim_params: vec![Some("batch_size".to_string()), Some("seq_len".to_string())],
        }],
        output_infos: vec![],
        ..Default::default()
    };

    (graph, HashMap::new())
}

/// Helper: build a graph with two inputs sharing the same "batch_size" symbol.
/// input_a [batch_size, 3] -> Relu -> out_a
/// input_b [batch_size, 5] -> Relu -> out_b
fn build_dual_input_dynamic_graph() -> (Graph, HashMap<String, Tensor>) {
    use oxionnx_core::{DType, TensorInfo};

    let node_a = Node {
        op: OpKind::Relu,
        name: "relu_a".to_string(),
        inputs: vec!["input_a".to_string()],
        outputs: vec!["out_a".to_string()],
        attrs: Attributes::default(),
    };
    let node_b = Node {
        op: OpKind::Relu,
        name: "relu_b".to_string(),
        inputs: vec!["input_b".to_string()],
        outputs: vec!["out_b".to_string()],
        attrs: Attributes::default(),
    };

    let graph = Graph {
        nodes: vec![node_a, node_b],
        input_names: vec!["input_a".to_string(), "input_b".to_string()],
        output_names: vec!["out_a".to_string(), "out_b".to_string()],
        input_infos: vec![
            TensorInfo {
                name: "input_a".to_string(),
                dtype: DType::F32,
                shape: vec![None, Some(3)],
                dim_params: vec![Some("batch_size".to_string()), None],
            },
            TensorInfo {
                name: "input_b".to_string(),
                dtype: DType::F32,
                shape: vec![None, Some(5)],
                dim_params: vec![Some("batch_size".to_string()), None],
            },
        ],
        output_infos: vec![],
        ..Default::default()
    };

    (graph, HashMap::new())
}

#[test]
fn test_dynamic_batch_size() {
    let (graph, weights) = build_dynamic_relu_graph();
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build");

    // batch=1
    let input1 = Tensor::new(vec![-1.0, 2.0, 3.0], vec![1, 3]);
    let out1 = session.run_one("input", input1).expect("batch=1 run");
    let y1 = out1.get("output").expect("output");
    assert_eq!(y1.shape, vec![1, 3]);
    assert_eq!(y1.data, vec![0.0, 2.0, 3.0]);

    // batch=4 on the same session
    let input4 = Tensor::new(
        vec![
            -1.0, 2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 9.0, 10.0, -11.0, 12.0,
        ],
        vec![4, 3],
    );
    let out4 = session.run_one("input", input4).expect("batch=4 run");
    let y4 = out4.get("output").expect("output");
    assert_eq!(y4.shape, vec![4, 3]);
    assert_eq!(
        y4.data,
        vec![0.0, 2.0, 3.0, 4.0, 0.0, 6.0, 0.0, 8.0, 9.0, 10.0, 0.0, 12.0]
    );

    // Verify dynamic_dims updated
    let dims = session.dynamic_dims();
    assert_eq!(dims.get("batch_size"), Some(&4));
}

#[test]
fn test_dynamic_seq_len() {
    let (graph, weights) = build_dynamic_batch_seq_graph();
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build");

    // seq_len=2
    let input_s2 = Tensor::new(vec![-1.0, 2.0, 3.0, -4.0], vec![2, 2]);
    let out_s2 = session.run_one("input", input_s2).expect("seq=2 run");
    let y_s2 = out_s2.get("output").expect("output");
    assert_eq!(y_s2.shape, vec![2, 2]);

    // seq_len=5
    let input_s5 = Tensor::new(
        vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0],
        vec![2, 5],
    );
    let out_s5 = session.run_one("input", input_s5).expect("seq=5 run");
    let y_s5 = out_s5.get("output").expect("output");
    assert_eq!(y_s5.shape, vec![2, 5]);
    assert_eq!(
        y_s5.data,
        vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0]
    );

    let dims = session.dynamic_dims();
    assert_eq!(dims.get("batch_size"), Some(&2));
    assert_eq!(dims.get("seq_len"), Some(&5));
}

#[test]
fn test_shape_validation_rank_error() {
    let (graph, weights) = build_dynamic_relu_graph();
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build");

    // Expected rank 2 [batch_size, 3], provide rank 3
    let bad_input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let result = session.run_one("input", bad_input);
    assert!(result.is_err());
    let err_msg = format!("{}", result.expect_err("should error"));
    assert!(
        err_msg.contains("rank"),
        "Error should mention rank: {err_msg}"
    );
}

#[test]
fn test_shape_validation_dim_error() {
    let (graph, weights) = build_dynamic_relu_graph();
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build");

    // Expected [batch_size, 3], provide [2, 5] — dim 1 is static=3 but got 5
    let bad_input = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vec![2, 5],
    );
    let result = session.run_one("input", bad_input);
    assert!(result.is_err());
    let err_msg = format!("{}", result.expect_err("should error"));
    assert!(
        err_msg.contains("static dim"),
        "Error should mention static dim: {err_msg}"
    );
}

#[test]
fn test_shape_validation_symbolic_consistency() {
    let (graph, weights) = build_dual_input_dynamic_graph();
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build");

    // input_a has batch_size=2, input_b has batch_size=3 — should fail
    let input_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let input_b = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
        vec![3, 5],
    );

    let mut inputs = HashMap::new();
    inputs.insert("input_a", input_a);
    inputs.insert("input_b", input_b);
    let result = session.run(&inputs);
    assert!(result.is_err());
    let err_msg = format!("{}", result.expect_err("should error"));
    assert!(
        err_msg.contains("inconsistent") || err_msg.contains("conflicting"),
        "Error should mention inconsistency: {err_msg}"
    );
}

#[test]
fn test_resolve_dynamic_shapes_basic() {
    use oxionnx_core::{DType, TensorInfo};

    let infos = vec![
        TensorInfo {
            name: "x".to_string(),
            dtype: DType::F32,
            shape: vec![None, Some(768)],
            dim_params: vec![Some("batch_size".to_string()), None],
        },
        TensorInfo {
            name: "y".to_string(),
            dtype: DType::F32,
            shape: vec![None, None],
            dim_params: vec![Some("batch_size".to_string()), Some("seq_len".to_string())],
        },
    ];

    let tensor_x = Tensor::new(vec![0.0; 4 * 768], vec![4, 768]);
    let tensor_y = Tensor::new(vec![0.0; 4 * 128], vec![4, 128]);
    let mut inputs: HashMap<&str, &Tensor> = HashMap::new();
    inputs.insert("x", &tensor_x);
    inputs.insert("y", &tensor_y);

    let dim_map = Session::resolve_dynamic_shapes(&infos, &inputs).expect("resolve should succeed");
    assert_eq!(dim_map.get("batch_size"), Some(&4));
    assert_eq!(dim_map.get("seq_len"), Some(&128));
}

#[test]
fn test_run_after_shape_change() {
    let (graph, weights) = build_dynamic_relu_graph();
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_memory_pool(true)
        .build_from_graph(graph, weights)
        .expect("build");

    // Run with batch=2
    let input2 = Tensor::new(vec![-1.0, 2.0, 3.0, -4.0, 5.0, -6.0], vec![2, 3]);
    let out2 = session.run_one("input", input2).expect("batch=2");
    let y2 = out2.get("output").expect("output");
    assert_eq!(y2.data, vec![0.0, 2.0, 3.0, 0.0, 5.0, 0.0]);

    // Run with batch=1 — shapes changed, results should still be correct
    let input1 = Tensor::new(vec![10.0, -20.0, 30.0], vec![1, 3]);
    let out1 = session.run_one("input", input1).expect("batch=1");
    let y1 = out1.get("output").expect("output");
    assert_eq!(y1.data, vec![10.0, 0.0, 30.0]);
    assert_eq!(y1.shape, vec![1, 3]);

    // Run with batch=3 — shapes changed again
    let input3 = Tensor::new(
        vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0],
        vec![3, 3],
    );
    let out3 = session.run_one("input", input3).expect("batch=3");
    let y3 = out3.get("output").expect("output");
    assert_eq!(y3.shape, vec![3, 3]);
    assert_eq!(y3.data, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0]);
}

#[test]
fn test_dynamic_shape_memory_reallocation() {
    let (graph, weights) = build_dynamic_relu_graph();
    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_memory_pool(true)
        .build_from_graph(graph, weights)
        .expect("build");

    // Small batch
    let small = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let out_small = session.run_one("input", small).expect("small batch");
    assert_eq!(out_small.get("output").expect("output").shape, vec![1, 3]);

    // Large batch — pool should handle the size increase
    let large_data: Vec<f32> = (0..300).map(|i| i as f32 - 150.0).collect();
    let large = Tensor::new(large_data, vec![100, 3]);
    let out_large = session.run_one("input", large).expect("large batch");
    let y_large = out_large.get("output").expect("output");
    assert_eq!(y_large.shape, vec![100, 3]);

    // Verify output correctness: ReLU maps negatives to 0
    for (i, &val) in y_large.data.iter().enumerate() {
        let original = i as f32 - 150.0;
        let expected = if original < 0.0 { 0.0 } else { original };
        assert!(
            (val - expected).abs() < 1e-6,
            "Mismatch at index {i}: got {val}, expected {expected}"
        );
    }

    // Back to small — the pool still works
    let small2 = Tensor::new(vec![-5.0, 10.0, -15.0], vec![1, 3]);
    let out_small2 = session.run_one("input", small2).expect("small batch 2");
    let y2 = out_small2.get("output").expect("output");
    assert_eq!(y2.data, vec![0.0, 10.0, 0.0]);
}
