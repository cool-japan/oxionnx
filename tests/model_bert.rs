//! Synthetic BERT-tiny end-to-end inference tests for oxionnx.
//!
//! Builds a BERT-tiny-like architecture programmatically (no ONNX file needed)
//! and runs inference through the session API.
//!
//! BERT-tiny config:
//!   hidden=128, heads=2, intermediate=512, layers=2, seq_len=16, vocab_size=100

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, NodeProfile, OpKind, OptLevel, Session, Tensor};

const HIDDEN: usize = 128;
const SEQ_LEN: usize = 16;
const INTERMEDIATE: usize = 512;
const NUM_LAYERS: usize = 2;

/// Generate a deterministic tensor with small values in approximately [-1, 1].
fn det_tensor(shape: &[usize], seed: u32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(seed).wrapping_add(17)) as f32;
            (x % 200.0 - 100.0) * 0.01
        })
        .collect();
    Tensor::new(data, shape.to_vec())
}

/// Helper: create a node with default attributes.
fn make_node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

/// Helper: create a node with custom attributes.
fn make_node_attrs(
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

/// Build the full BERT-tiny graph, returning the Graph, weights, and output tensor name.
///
/// Architecture per layer:
///   Self-Attention (single-head simplified):
///     Q = MatMul(input, Wq), K = MatMul(input, Wk), V = MatMul(input, Wv)
///     scores = Softmax(MatMul(Q, Transpose(K)) / sqrt(hidden))
///     attn = MatMul(scores, V)
///     proj = MatMul(attn, Wo)
///     res1 = Add(input, proj)
///   FFN:
///     ffn1 = Gelu(MatMul(res1, W1))
///     ffn2 = MatMul(ffn1, W2)
///     layer_out = Add(res1, ffn2)
fn build_bert_tiny_graph() -> (Graph, HashMap<String, Tensor>, String) {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();
    let mut node_id: u32 = 0;

    let mut make_name = |prefix: &str| -> String {
        node_id += 1;
        format!("{}_{}", prefix, node_id)
    };

    // Current activation name flowing through the network.
    // Input is [1, SEQ_LEN, HIDDEN] - skip embedding lookup, test transformer layers.
    let mut current = "input_embed".to_string();

    for layer in 0..NUM_LAYERS {
        let prefix = format!("L{}", layer);

        // === Self-Attention (simplified single-head) ===

        // Q = MatMul(current, Wq)
        let wq_name = format!("{}_Wq", prefix);
        weights.insert(
            wq_name.clone(),
            det_tensor(&[HIDDEN, HIDDEN], layer as u32 * 100 + 1),
        );
        let q_name = make_name("Q");
        let q_out = format!("{}_out", q_name);
        nodes.push(make_node(
            OpKind::MatMul,
            &q_name,
            &[&current, &wq_name],
            &[&q_out],
        ));

        // K = MatMul(current, Wk)
        let wk_name = format!("{}_Wk", prefix);
        weights.insert(
            wk_name.clone(),
            det_tensor(&[HIDDEN, HIDDEN], layer as u32 * 100 + 2),
        );
        let k_name = make_name("K");
        let k_out = format!("{}_out", k_name);
        nodes.push(make_node(
            OpKind::MatMul,
            &k_name,
            &[&current, &wk_name],
            &[&k_out],
        ));

        // V = MatMul(current, Wv)
        let wv_name = format!("{}_Wv", prefix);
        weights.insert(
            wv_name.clone(),
            det_tensor(&[HIDDEN, HIDDEN], layer as u32 * 100 + 3),
        );
        let v_name = make_name("V");
        let v_out = format!("{}_out", v_name);
        nodes.push(make_node(
            OpKind::MatMul,
            &v_name,
            &[&current, &wv_name],
            &[&v_out],
        ));

        // Transpose K: [1, seq, hidden] -> [1, hidden, seq]
        let kt_name = make_name("KT");
        let kt_out = format!("{}_out", kt_name);
        let mut transpose_attrs = Attributes::default();
        transpose_attrs
            .int_lists
            .insert("perm".into(), vec![0, 2, 1]);
        nodes.push(make_node_attrs(
            OpKind::Transpose,
            &kt_name,
            &[&k_out],
            &[&kt_out],
            transpose_attrs,
        ));

        // scores = MatMul(Q, K^T) -> [1, seq, seq]
        let scores_name = make_name("scores");
        let scores_out = format!("{}_out", scores_name);
        nodes.push(make_node(
            OpKind::MatMul,
            &scores_name,
            &[&q_out, &kt_out],
            &[&scores_out],
        ));

        // Scale by 1/sqrt(hidden) using Div
        let scale_name = format!("{}_scale", prefix);
        let scale_val = (HIDDEN as f32).sqrt();
        weights.insert(scale_name.clone(), Tensor::new(vec![scale_val], vec![1]));
        let scaled_name = make_name("scaled_scores");
        let scaled_out = format!("{}_out", scaled_name);
        nodes.push(make_node(
            OpKind::Div,
            &scaled_name,
            &[&scores_out, &scale_name],
            &[&scaled_out],
        ));

        // Softmax on last axis
        let softmax_name = make_name("softmax");
        let softmax_out = format!("{}_out", softmax_name);
        let mut softmax_attrs = Attributes::default();
        softmax_attrs.ints.insert("axis".into(), -1);
        nodes.push(make_node_attrs(
            OpKind::Softmax,
            &softmax_name,
            &[&scaled_out],
            &[&softmax_out],
            softmax_attrs,
        ));

        // attn = MatMul(softmax, V) -> [1, seq, hidden]
        let attn_name = make_name("attn");
        let attn_out = format!("{}_out", attn_name);
        nodes.push(make_node(
            OpKind::MatMul,
            &attn_name,
            &[&softmax_out, &v_out],
            &[&attn_out],
        ));

        // Output projection: proj = MatMul(attn, Wo)
        let wo_name = format!("{}_Wo", prefix);
        weights.insert(
            wo_name.clone(),
            det_tensor(&[HIDDEN, HIDDEN], layer as u32 * 100 + 4),
        );
        let proj_name = make_name("proj");
        let proj_out = format!("{}_out", proj_name);
        nodes.push(make_node(
            OpKind::MatMul,
            &proj_name,
            &[&attn_out, &wo_name],
            &[&proj_out],
        ));

        // Residual: res1 = Add(current, proj)
        let res1_name = make_name("res1");
        let res1_out = format!("{}_out", res1_name);
        nodes.push(make_node(
            OpKind::Add,
            &res1_name,
            &[&current, &proj_out],
            &[&res1_out],
        ));

        // === Feed-Forward Network ===

        // ffn1 = MatMul(res1, W1) -> [1, seq, intermediate]
        let w1_name = format!("{}_W1", prefix);
        weights.insert(
            w1_name.clone(),
            det_tensor(&[HIDDEN, INTERMEDIATE], layer as u32 * 100 + 5),
        );
        let ffn1_name = make_name("ffn1");
        let ffn1_out = format!("{}_out", ffn1_name);
        nodes.push(make_node(
            OpKind::MatMul,
            &ffn1_name,
            &[&res1_out, &w1_name],
            &[&ffn1_out],
        ));

        // Gelu activation
        let gelu_name = make_name("gelu");
        let gelu_out = format!("{}_out", gelu_name);
        nodes.push(make_node(
            OpKind::Gelu,
            &gelu_name,
            &[&ffn1_out],
            &[&gelu_out],
        ));

        // ffn2 = MatMul(gelu, W2) -> [1, seq, hidden]
        let w2_name = format!("{}_W2", prefix);
        weights.insert(
            w2_name.clone(),
            det_tensor(&[INTERMEDIATE, HIDDEN], layer as u32 * 100 + 6),
        );
        let ffn2_name = make_name("ffn2");
        let ffn2_out = format!("{}_out", ffn2_name);
        nodes.push(make_node(
            OpKind::MatMul,
            &ffn2_name,
            &[&gelu_out, &w2_name],
            &[&ffn2_out],
        ));

        // Residual: layer_out = Add(res1, ffn2)
        let res2_name = make_name("res2");
        let res2_out = format!("{}_out", res2_name);
        nodes.push(make_node(
            OpKind::Add,
            &res2_name,
            &[&res1_out, &ffn2_out],
            &[&res2_out],
        ));

        current = res2_out;
    }

    let graph = Graph {
        nodes,
        input_names: vec!["input_embed".into()],
        output_names: vec![current.clone()],
        ..Default::default()
    };

    (graph, weights, current)
}

/// Validate that a BERT output tensor has the expected shape, contains no NaN/Inf,
/// and values are within a reasonable magnitude.
fn validate_bert_output(output: &Tensor, output_name: &str) {
    // Shape must be [1, SEQ_LEN, HIDDEN]
    assert_eq!(
        output.shape,
        vec![1, SEQ_LEN, HIDDEN],
        "unexpected output shape for '{}'",
        output_name
    );

    // No NaN or Inf
    assert!(
        output.data.iter().all(|v| v.is_finite()),
        "output '{}' contains NaN or Inf",
        output_name
    );

    // Values should not have exploded
    let max_abs = output.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e6,
        "output '{}' values exploded: max_abs={}",
        output_name,
        max_abs
    );
}

// ── Test 1: Basic BERT-tiny inference ───────────────────────────────────────

#[test]
fn test_bert_tiny_synthetic() {
    let (graph, weights, output_name) = build_bert_tiny_graph();

    let session = Session::from_graph(graph, weights).expect("BERT build failed");

    // Verify model info
    let info = session.model_info();
    assert!(
        info.node_count > 0,
        "model should have at least one node after optimization"
    );
    assert!(
        info.parameter_count > 0,
        "model should have weight parameters"
    );

    // Input: [1, SEQ_LEN, HIDDEN]
    let input = det_tensor(&[1, SEQ_LEN, HIDDEN], 42);
    let outputs = session
        .run_one("input_embed", input)
        .expect("BERT run failed");
    let output = outputs
        .get(&output_name)
        .expect("output tensor not found in results");

    validate_bert_output(output, &output_name);

    // Run a second time to verify determinism
    let input2 = det_tensor(&[1, SEQ_LEN, HIDDEN], 42);
    let outputs2 = session
        .run_one("input_embed", input2)
        .expect("BERT second run failed");
    let output2 = outputs2
        .get(&output_name)
        .expect("output tensor not found in second run");

    assert_eq!(
        output.data, output2.data,
        "two runs with the same input should produce identical output"
    );
}

// ── Test 2: BERT-tiny with profiling ────────────────────────────────────────

#[test]
fn test_bert_tiny_with_profiling() {
    let (graph, weights, output_name) = build_bert_tiny_graph();

    let session = Session::builder()
        .with_optimization_level(OptLevel::All)
        .with_profiling()
        .build_from_graph(graph, weights)
        .expect("BERT profiling build failed");

    let input = det_tensor(&[1, SEQ_LEN, HIDDEN], 99);
    let outputs = session
        .run_one("input_embed", input)
        .expect("BERT profiling run failed");
    let output = outputs.get(&output_name).expect("output tensor not found");

    validate_bert_output(output, &output_name);

    // Check profiling results
    let profiles: Vec<NodeProfile> = session
        .profiling_results()
        .expect("profiling should be enabled");
    assert!(
        !profiles.is_empty(),
        "profiling should have recorded at least one node"
    );

    // Every profiled node should have a non-empty op_type
    for profile in &profiles {
        assert!(
            !profile.op_type.is_empty(),
            "node '{}' has empty op_type",
            profile.node_name
        );
        assert!(
            !profile.output_shapes.is_empty(),
            "node '{}' has no output shapes",
            profile.node_name
        );
    }

    // We expect MatMul nodes to be present (the main compute ops)
    let matmul_count = profiles.iter().filter(|p| p.op_type == "MatMul").count();
    assert!(
        matmul_count > 0,
        "expected at least one MatMul in profiling data, got 0"
    );

    // Total profiled time should be non-zero
    let total_ns: u128 = profiles.iter().map(|p| p.duration.as_nanos()).sum();
    assert!(
        total_ns > 0,
        "total profiled time should be greater than zero"
    );
}

// ── Test 3: BERT-tiny with parallel execution ───────────────────────────────

#[test]
fn test_bert_tiny_with_parallel() {
    let (graph, weights, output_name) = build_bert_tiny_graph();

    // Build sequential session
    let (graph_seq, weights_seq, _) = build_bert_tiny_graph();
    let session_seq = Session::builder()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph_seq, weights_seq)
        .expect("sequential build failed");

    // Build parallel session
    let session_par = Session::builder()
        .with_optimization_level(OptLevel::All)
        .with_parallel_execution(true)
        .build_from_graph(graph, weights)
        .expect("parallel build failed");

    let input = det_tensor(&[1, SEQ_LEN, HIDDEN], 77);

    let outputs_seq = session_seq
        .run_one("input_embed", input.clone())
        .expect("sequential run failed");
    let outputs_par = session_par
        .run_one("input_embed", input)
        .expect("parallel run failed");

    let out_seq = outputs_seq
        .get(&output_name)
        .expect("sequential output not found");
    let out_par = outputs_par
        .get(&output_name)
        .expect("parallel output not found");

    validate_bert_output(out_seq, &output_name);
    validate_bert_output(out_par, &output_name);

    // Sequential and parallel should produce the same shape
    assert_eq!(
        out_seq.shape, out_par.shape,
        "sequential and parallel output shapes differ"
    );

    // Values should be very close (floating-point reorder may cause tiny differences)
    let max_diff = out_seq
        .data
        .iter()
        .zip(out_par.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Allow small tolerance for floating-point reordering in parallel execution
    assert!(
        max_diff < 1e-3,
        "sequential vs parallel max diff = {} (expected < 1e-3)",
        max_diff
    );
}
