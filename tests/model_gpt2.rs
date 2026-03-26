//! Synthetic GPT-2-like end-to-end inference test.
//!
//! Builds a simplified GPT-2 transformer architecture from scratch using
//! `Session::from_graph()` and verifies the forward pass produces correctly-shaped,
//! finite outputs.

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

// ── Constants ────────────────────────────────────────────────────────────────

const HIDDEN: usize = 64;
const SEQ_LEN: usize = 8;
const HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS; // 32
const INTERMEDIATE: usize = 256;
const VOCAB_SIZE: usize = 50;
const NUM_LAYERS: usize = 2;

// ── Helpers ──────────────────────────────────────────────────────────────────

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

fn shape_tensor(dims: &[i64]) -> Tensor {
    let len = dims.len();
    Tensor::new(dims.iter().map(|&d| d as f32).collect(), vec![len])
}

fn transpose_attrs(perm: &[i64]) -> Attributes {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("perm".to_string(), perm.to_vec());
    attrs
}

fn softmax_attrs(axis: i64) -> Attributes {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), axis);
    attrs
}

// ── GPT-2 Builder ────────────────────────────────────────────────────────────

struct Gpt2Builder {
    nodes: Vec<Node>,
    weights: HashMap<String, Tensor>,
    counter: usize,
}

impl Gpt2Builder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            weights: HashMap::new(),
            counter: 0,
        }
    }

    fn next_id(&mut self) -> usize {
        let id = self.counter;
        self.counter += 1;
        id
    }

    /// MatMul(input, weight) -> output name
    fn add_matmul(&mut self, input: &str, w_shape: &[usize], seed: u32, prefix: &str) -> String {
        let id = self.next_id();
        let w_name = format!("{prefix}_w{id}");
        let out_name = format!("{prefix}_mm{id}");

        self.weights
            .insert(w_name.clone(), det_tensor(w_shape, seed));
        self.nodes.push(make_node(
            OpKind::MatMul,
            &format!("{prefix}_matmul{id}"),
            &[input, &w_name],
            &[&out_name],
        ));
        out_name
    }

    /// Add(a, b) -> output name
    fn add_add(&mut self, a: &str, b: &str, prefix: &str) -> String {
        let id = self.next_id();
        let out_name = format!("{prefix}_add{id}");
        self.nodes.push(make_node(
            OpKind::Add,
            &format!("{prefix}_add_node{id}"),
            &[a, b],
            &[&out_name],
        ));
        out_name
    }

    /// Reshape -> output name
    fn add_reshape(&mut self, input: &str, target: &[i64], prefix: &str) -> String {
        let id = self.next_id();
        let shape_name = format!("{prefix}_rs{id}_shape");
        let out_name = format!("{prefix}_rs{id}");

        self.weights
            .insert(shape_name.clone(), shape_tensor(target));
        self.nodes.push(make_node(
            OpKind::Reshape,
            &format!("{prefix}_reshape{id}"),
            &[input, &shape_name],
            &[&out_name],
        ));
        out_name
    }

    /// Transpose -> output name
    fn add_transpose(&mut self, input: &str, perm: &[i64], prefix: &str) -> String {
        let id = self.next_id();
        let out_name = format!("{prefix}_tp{id}");
        self.nodes.push(make_node_with_attrs(
            OpKind::Transpose,
            &format!("{prefix}_transpose{id}"),
            &[input],
            &[&out_name],
            transpose_attrs(perm),
        ));
        out_name
    }

    /// Softmax -> output name
    fn add_softmax(&mut self, input: &str, axis: i64, prefix: &str) -> String {
        let id = self.next_id();
        let out_name = format!("{prefix}_sm{id}");
        self.nodes.push(make_node_with_attrs(
            OpKind::Softmax,
            &format!("{prefix}_softmax{id}"),
            &[input],
            &[&out_name],
            softmax_attrs(axis),
        ));
        out_name
    }

    /// Gelu -> output name
    fn add_gelu(&mut self, input: &str, prefix: &str) -> String {
        let id = self.next_id();
        let out_name = format!("{prefix}_gelu{id}");
        self.nodes.push(make_node(
            OpKind::Gelu,
            &format!("{prefix}_gelu_node{id}"),
            &[input],
            &[&out_name],
        ));
        out_name
    }

    /// Mul(a, scalar_tensor) -> output name (for scaling by 1/sqrt(d_k))
    fn add_mul_scalar(&mut self, input: &str, scalar: f32, prefix: &str) -> String {
        let id = self.next_id();
        let s_name = format!("{prefix}_scale{id}");
        let out_name = format!("{prefix}_mul{id}");

        self.weights
            .insert(s_name.clone(), Tensor::new(vec![scalar], vec![1]));
        self.nodes.push(make_node(
            OpKind::Mul,
            &format!("{prefix}_mul_node{id}"),
            &[input, &s_name],
            &[&out_name],
        ));
        out_name
    }

    /// Multi-head self-attention block.
    ///
    /// Input: [1, seq_len, hidden]
    /// Output: [1, seq_len, hidden]
    ///
    /// Q = input @ Wq   [1, seq, hidden]
    /// K = input @ Wk
    /// V = input @ Wv
    /// Reshape to [1, seq, heads, head_dim]
    /// Transpose to [1, heads, seq, head_dim]
    /// scores = Q @ K^T / sqrt(head_dim)  -> [1, heads, seq, seq]
    /// attn = Softmax(scores)
    /// context = attn @ V  -> [1, heads, seq, head_dim]
    /// Transpose back to [1, seq, heads, head_dim]
    /// Reshape to [1, seq, hidden]
    /// output = context @ Wo
    fn add_attention(&mut self, input: &str, layer: usize, seed_base: u32) -> String {
        let pfx = format!("l{layer}_attn");

        // Q, K, V projections: [1, seq, hidden] @ [hidden, hidden] -> [1, seq, hidden]
        let q = self.add_matmul(input, &[HIDDEN, HIDDEN], seed_base, &pfx);
        let k = self.add_matmul(input, &[HIDDEN, HIDDEN], seed_base.wrapping_add(10), &pfx);
        let v = self.add_matmul(input, &[HIDDEN, HIDDEN], seed_base.wrapping_add(20), &pfx);

        // Reshape: [1, seq, hidden] -> [1, seq, heads, head_dim]
        let q_4d = self.add_reshape(
            &q,
            &[1, SEQ_LEN as i64, HEADS as i64, HEAD_DIM as i64],
            &pfx,
        );
        let k_4d = self.add_reshape(
            &k,
            &[1, SEQ_LEN as i64, HEADS as i64, HEAD_DIM as i64],
            &pfx,
        );
        let v_4d = self.add_reshape(
            &v,
            &[1, SEQ_LEN as i64, HEADS as i64, HEAD_DIM as i64],
            &pfx,
        );

        // Transpose: [1, seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let q_t = self.add_transpose(&q_4d, &[0, 2, 1, 3], &pfx);
        let k_t = self.add_transpose(&k_4d, &[0, 2, 1, 3], &pfx);
        let v_t = self.add_transpose(&v_4d, &[0, 2, 1, 3], &pfx);

        // K^T: [1, heads, seq, head_dim] -> [1, heads, head_dim, seq]
        let k_transposed = self.add_transpose(&k_t, &[0, 1, 3, 2], &pfx);

        // scores = Q @ K^T: [1, heads, seq, seq]
        let scores = {
            let id = self.next_id();
            let out_name = format!("{pfx}_scores{id}");
            self.nodes.push(make_node(
                OpKind::MatMul,
                &format!("{pfx}_scores_mm{id}"),
                &[&q_t, &k_transposed],
                &[&out_name],
            ));
            out_name
        };

        // Scale: scores * (1 / sqrt(head_dim))
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let scaled = self.add_mul_scalar(&scores, scale, &pfx);

        // Softmax over last axis (axis=-1)
        let attn_weights = self.add_softmax(&scaled, -1, &pfx);

        // context = attn @ V: [1, heads, seq, head_dim]
        let context = {
            let id = self.next_id();
            let out_name = format!("{pfx}_ctx{id}");
            self.nodes.push(make_node(
                OpKind::MatMul,
                &format!("{pfx}_ctx_mm{id}"),
                &[&attn_weights, &v_t],
                &[&out_name],
            ));
            out_name
        };

        // Transpose back: [1, heads, seq, head_dim] -> [1, seq, heads, head_dim]
        let ctx_t = self.add_transpose(&context, &[0, 2, 1, 3], &pfx);

        // Reshape: [1, seq, heads, head_dim] -> [1, seq, hidden]
        let ctx_flat = self.add_reshape(&ctx_t, &[1, SEQ_LEN as i64, HIDDEN as i64], &pfx);

        // Output projection: [1, seq, hidden] @ [hidden, hidden] -> [1, seq, hidden]
        self.add_matmul(
            &ctx_flat,
            &[HIDDEN, HIDDEN],
            seed_base.wrapping_add(30),
            &pfx,
        )
    }

    /// Feed-forward network block.
    ///
    /// Input: [1, seq_len, hidden]
    /// Output: [1, seq_len, hidden]
    ///
    /// hidden -> intermediate (expand) -> Gelu -> hidden (contract)
    fn add_ffn(&mut self, input: &str, layer: usize, seed_base: u32) -> String {
        let pfx = format!("l{layer}_ffn");

        // Up-project: [1, seq, hidden] @ [hidden, intermediate] -> [1, seq, intermediate]
        let up = self.add_matmul(input, &[HIDDEN, INTERMEDIATE], seed_base, &pfx);

        // Gelu activation
        let act = self.add_gelu(&up, &pfx);

        // Down-project: [1, seq, intermediate] @ [intermediate, hidden] -> [1, seq, hidden]
        self.add_matmul(
            &act,
            &[INTERMEDIATE, HIDDEN],
            seed_base.wrapping_add(50),
            &pfx,
        )
    }

    /// Full transformer layer: attention + residual + FFN + residual
    fn add_transformer_layer(&mut self, input: &str, layer: usize, seed_base: u32) -> String {
        // Self-attention
        let attn_out = self.add_attention(input, layer, seed_base);

        // Residual connection
        let pfx = format!("l{layer}");
        let post_attn = self.add_add(input, &attn_out, &pfx);

        // FFN
        let ffn_out = self.add_ffn(&post_attn, layer, seed_base.wrapping_add(500));

        // Residual connection
        self.add_add(&post_attn, &ffn_out, &pfx)
    }
}

// ── Build full GPT-2-like model ──────────────────────────────────────────────

fn build_gpt2() -> (Graph, HashMap<String, Tensor>) {
    let mut b = Gpt2Builder::new();

    // Input: [1, SEQ_LEN, HIDDEN] (skip token embedding for simplicity)
    let mut current = "input".to_string();

    // Transformer layers
    for layer in 0..NUM_LAYERS {
        let seed = (layer as u32 + 1) * 1000;
        current = b.add_transformer_layer(&current, layer, seed);
    }

    // Language model head: [1, seq, hidden] @ [hidden, vocab] -> [1, seq, vocab]
    let logits = b.add_matmul(&current, &[HIDDEN, VOCAB_SIZE], 9999, "lm_head");

    let graph = Graph {
        nodes: b.nodes,
        input_names: vec!["input".to_string()],
        output_names: vec![logits],
    };

    (graph, b.weights)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn test_gpt2_like_synthetic() {
    let (graph, weights) = build_gpt2();

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build GPT-2 session");

    let input = det_tensor(&[1, SEQ_LEN, HIDDEN], 7);
    let outputs = session.run_one("input", input).expect("run GPT-2");

    assert_eq!(outputs.len(), 1);

    let logits = outputs.values().next().expect("output tensor");

    // Shape: [1, SEQ_LEN, VOCAB_SIZE]
    assert_eq!(logits.shape, vec![1, SEQ_LEN, VOCAB_SIZE]);

    // No NaN or Inf
    for (i, &v) in logits.data.iter().enumerate() {
        assert!(v.is_finite(), "GPT-2 output[{i}] is not finite: {v}");
    }

    // Values in reasonable range
    let max_abs = logits.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e8,
        "GPT-2 output has unexpectedly large values: max_abs = {max_abs}"
    );
}

#[test]
fn test_gpt2_output_shape() {
    let (graph, weights) = build_gpt2();

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("build GPT-2 session");

    let input = det_tensor(&[1, SEQ_LEN, HIDDEN], 42);
    let outputs = session.run_one("input", input).expect("run GPT-2");

    let logits = outputs.values().next().expect("output tensor");
    assert_eq!(
        logits.shape,
        vec![1, SEQ_LEN, VOCAB_SIZE],
        "GPT-2 output shape should be [1, {SEQ_LEN}, {VOCAB_SIZE}]"
    );

    // Total elements
    assert_eq!(
        logits.data.len(),
        SEQ_LEN * VOCAB_SIZE,
        "GPT-2 output element count mismatch"
    );
}

#[test]
fn test_gpt2_with_profiling() {
    let (graph, weights) = build_gpt2();

    let session = Session::builder()
        .with_optimization_level(OptLevel::None)
        .with_profiling()
        .build_from_graph(graph, weights)
        .expect("build GPT-2 session with profiling");

    // Before run: profiling empty
    let initial = session.profiling_results().expect("profiling enabled");
    assert!(initial.is_empty(), "no profiles before first run");

    let input = det_tensor(&[1, SEQ_LEN, HIDDEN], 7);
    let outputs = session.run_one("input", input).expect("run GPT-2");

    // Verify output is valid
    let logits = outputs.values().next().expect("output tensor");
    assert_eq!(logits.shape, vec![1, SEQ_LEN, VOCAB_SIZE]);

    // Profiling results should be populated
    let profiles = session.profiling_results().expect("profiling enabled");
    assert!(
        !profiles.is_empty(),
        "profiling results should not be empty after run"
    );

    // Check that we have MatMul, Softmax, Gelu in the profile
    let op_types: Vec<&str> = profiles.iter().map(|p| p.op_type.as_str()).collect();
    assert!(
        op_types.contains(&"MatMul"),
        "profiles should contain MatMul ops"
    );
    assert!(
        op_types.contains(&"Softmax"),
        "profiles should contain Softmax ops"
    );
    assert!(
        op_types.contains(&"Gelu"),
        "profiles should contain Gelu ops"
    );

    // Every profiled node should have a name and non-negative duration
    for p in &profiles {
        assert!(!p.node_name.is_empty(), "node_name should not be empty");
        assert!(!p.op_type.is_empty(), "op_type should not be empty");
    }
}

#[test]
fn test_gpt2_deterministic() {
    let (graph1, weights1) = build_gpt2();
    let (graph2, weights2) = build_gpt2();

    let session1 = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph1, weights1)
        .expect("build session 1");
    let session2 = Session::builder()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph2, weights2)
        .expect("build session 2");

    let input1 = det_tensor(&[1, SEQ_LEN, HIDDEN], 7);
    let input2 = det_tensor(&[1, SEQ_LEN, HIDDEN], 7);

    let out1 = session1.run_one("input", input1).expect("run 1");
    let out2 = session2.run_one("input", input2).expect("run 2");

    let v1 = &out1.values().next().expect("out1").data;
    let v2 = &out2.values().next().expect("out2").data;

    assert_eq!(v1.len(), v2.len());
    for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "Determinism broken at index {i}: {a} vs {b}"
        );
    }
}
