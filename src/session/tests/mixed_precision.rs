use super::super::types::OptLevel;
use super::super::Session;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Basic mixed precision: Relu → Add → MatMul → Relu graph.
/// Verify output matches f32-only within tolerance.
#[test]
fn test_mixed_precision_relu_add_matmul_relu() {
    // Graph: input [2,3] → Relu → relu_out → Add(relu_out, bias) → add_out
    //        → MatMul(add_out, weight [3,2]) → mm_out → Relu → output [2,2]
    let relu1 = Node {
        op: OpKind::Relu,
        name: "relu1".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        attrs: Attributes::default(),
    };
    let add = Node {
        op: OpKind::Add,
        name: "add1".to_string(),
        inputs: vec!["relu_out".to_string(), "bias".to_string()],
        outputs: vec!["add_out".to_string()],
        attrs: Attributes::default(),
    };
    let matmul = Node {
        op: OpKind::MatMul,
        name: "matmul1".to_string(),
        inputs: vec!["add_out".to_string(), "weight".to_string()],
        outputs: vec!["mm_out".to_string()],
        attrs: Attributes::default(),
    };
    let relu2 = Node {
        op: OpKind::Relu,
        name: "relu2".to_string(),
        inputs: vec!["mm_out".to_string()],
        outputs: vec!["output".to_string()],
        attrs: Attributes::default(),
    };

    let graph = Graph {
        nodes: vec![relu1, add, matmul, relu2],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    weights.insert(
        "bias".to_string(),
        Tensor::new(vec![0.5, 0.5, 0.5], vec![3]),
    );
    weights.insert(
        "weight".to_string(),
        Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]),
    );

    // Run with mixed precision
    let session_mp = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_mixed_precision(true)
        .build_from_graph(graph.clone(), weights.clone())
        .expect("build mixed precision session");

    // Run without mixed precision (reference)
    let session_f32 = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build f32 session");

    let input = Tensor::new(vec![-1.0, 2.0, 0.5, 3.0, -0.5, 1.0], vec![2, 3]);
    let out_mp = session_mp.run_one("input", input.clone()).expect("run mp");
    let out_f32 = session_f32.run_one("input", input).expect("run f32");

    let mp_data = &out_mp.get("output").expect("mp output").data;
    let f32_data = &out_f32.get("output").expect("f32 output").data;

    // Mixed precision should match f32 within tolerance (f16 has ~0.1% relative error)
    assert_eq!(mp_data.len(), f32_data.len());
    for (i, (&mp_val, &f32_val)) in mp_data.iter().zip(f32_data.iter()).enumerate() {
        let abs_err = (mp_val - f32_val).abs();
        let rel_tol = f32_val.abs() * 0.01 + 0.01; // 1% relative + 0.01 absolute
        assert!(
            abs_err < rel_tol,
            "Output[{i}]: mp={mp_val}, f32={f32_val}, err={abs_err} > tol={rel_tol}"
        );
    }
}

/// Verify f16-safe ops actually execute in f16 by checking profiling data.
#[test]
fn test_mixed_precision_profiling_shows_f16() {
    let relu = Node {
        op: OpKind::Relu,
        name: "relu1".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        attrs: Attributes::default(),
    };
    let add = Node {
        op: OpKind::Add,
        name: "add1".to_string(),
        inputs: vec!["relu_out".to_string(), "bias".to_string()],
        outputs: vec!["output".to_string()],
        attrs: Attributes::default(),
    };

    let graph = Graph {
        nodes: vec![relu, add],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    weights.insert(
        "bias".to_string(),
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
    );

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_profiling()
        .with_mixed_precision(true)
        .build_from_graph(graph, weights)
        .expect("build");

    let input = Tensor::new(vec![-1.0, 2.0, 3.0], vec![1, 3]);
    let _outputs = session.run_one("input", input).expect("run");

    let profiles = session.profiling_results().expect("profiling enabled");
    assert_eq!(profiles.len(), 2);
    // Relu has native f16 path — profiled as "Relu(f16)"
    assert_eq!(profiles[0].op_type, "Relu(f16)");
    // Add has native f16 path — profiled as "Add(f16)"
    assert_eq!(profiles[1].op_type, "Add(f16)");
}

/// Verify MatMul uses f32 accumulation even with mixed precision enabled.
#[test]
fn test_mixed_precision_matmul_stays_f32() {
    let matmul = Node {
        op: OpKind::MatMul,
        name: "mm".to_string(),
        inputs: vec!["input".to_string(), "weight".to_string()],
        outputs: vec!["output".to_string()],
        attrs: Attributes::default(),
    };

    let graph = Graph {
        nodes: vec![matmul],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    weights.insert(
        "weight".to_string(),
        Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
    );

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_profiling()
        .with_mixed_precision(true)
        .build_from_graph(graph, weights)
        .expect("build");

    let input = Tensor::new(vec![3.0, 7.0, 5.0, 11.0], vec![2, 2]);
    let outputs = session.run_one("input", input.clone()).expect("run");

    let out = outputs.get("output").expect("output");
    // Identity matrix multiplication: result == input (exact f32)
    assert_eq!(out.data, vec![3.0, 7.0, 5.0, 11.0]);

    // Profiling should show "MatMul" (not "MatMul(f16)")
    let profiles = session.profiling_results().expect("profiling enabled");
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].op_type, "MatMul");
}

/// Verify mixed precision session builds and runs without error.
#[test]
fn test_mixed_precision_builder() {
    let session = Session::builder()
        .with_mixed_precision(true)
        .load_from_bytes(&[]);
    assert!(session.is_ok());
    let session = session.expect("should build");
    assert!(session.mixed_precision);
}

/// Verify f16 rounding for ops without native f16 path (e.g., Softmax).
#[test]
fn test_mixed_precision_f16_rounding_fallback() {
    // Softmax is f16-safe but has no native f16 path in execute_elementwise_f16.
    // With mixed precision, its output should be rounded to f16 precision.
    let softmax = Node {
        op: OpKind::Softmax,
        name: "sm".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["output".to_string()],
        attrs: Attributes::default(),
    };

    let graph = Graph {
        nodes: vec![softmax],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };

    let session_mp = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_mixed_precision(true)
        .build_from_graph(graph.clone(), HashMap::new())
        .expect("build mp");

    let session_f32 = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build f32");

    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let out_mp = session_mp.run_one("input", input.clone()).expect("run mp");
    let out_f32 = session_f32.run_one("input", input).expect("run f32");

    let mp_data = &out_mp.get("output").expect("mp output").data;
    let f32_data = &out_f32.get("output").expect("f32 output").data;

    // Outputs should be close but mp data should be f16-rounded
    for (&mp_val, &f32_val) in mp_data.iter().zip(f32_data.iter()) {
        let abs_err = (mp_val - f32_val).abs();
        assert!(abs_err < 0.01, "mp={mp_val}, f32={f32_val}, err={abs_err}");
        // Verify mp_val is exactly representable in f16
        let roundtrip = half::f16::from_f32(mp_val).to_f32();
        assert_eq!(
            mp_val, roundtrip,
            "mp output should be exactly f16-representable"
        );
    }
}

/// Chain of consecutive f16-safe ops: Relu → Add → Sigmoid.
/// Verifies multi-op f16 execution works end-to-end.
#[test]
fn test_mixed_precision_consecutive_f16_ops() {
    let relu = Node {
        op: OpKind::Relu,
        name: "relu".to_string(),
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        attrs: Attributes::default(),
    };
    let add = Node {
        op: OpKind::Add,
        name: "add".to_string(),
        inputs: vec!["relu_out".to_string(), "bias".to_string()],
        outputs: vec!["add_out".to_string()],
        attrs: Attributes::default(),
    };
    let sigmoid = Node {
        op: OpKind::Sigmoid,
        name: "sig".to_string(),
        inputs: vec!["add_out".to_string()],
        outputs: vec!["output".to_string()],
        attrs: Attributes::default(),
    };

    let graph = Graph {
        nodes: vec![relu, add, sigmoid],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    weights.insert(
        "bias".to_string(),
        Tensor::new(vec![-0.5, 0.0, 0.5], vec![3]),
    );

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_profiling()
        .with_mixed_precision(true)
        .build_from_graph(graph, weights)
        .expect("build");

    let input = Tensor::new(vec![-2.0, 1.0, 3.0], vec![1, 3]);
    let outputs = session.run_one("input", input).expect("run");
    let out = outputs.get("output").expect("output");

    // All ops are f16-safe and have native f16 paths
    let profiles = session.profiling_results().expect("profiling");
    assert_eq!(profiles.len(), 3);
    assert_eq!(profiles[0].op_type, "Relu(f16)");
    assert_eq!(profiles[1].op_type, "Add(f16)");
    assert_eq!(profiles[2].op_type, "Sigmoid(f16)");

    // Output should be sigmoid values in [0, 1]
    for &v in &out.data {
        assert!((0.0..=1.0).contains(&v), "sigmoid output {v} out of range");
    }
}
