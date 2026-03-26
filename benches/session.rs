//! End-to-end session benchmarks for oxionnx.

use criterion::{criterion_group, criterion_main, Criterion};
use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, SessionBuilder, Tensor};
use std::collections::HashMap;
use std::hint::black_box;

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

/// Build a chain of `num_layers` linear layers: MatMul -> Add -> Relu.
/// Input shape: [1, dim], each layer maps dim -> dim.
/// Returns (graph, weights, input_name, output_name).
fn build_linear_chain(
    num_layers: usize,
    dim: usize,
) -> (Graph, HashMap<String, Tensor>, String, String) {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();

    let input_name = "input".to_string();
    let mut prev_output = input_name.clone();

    for i in 0..num_layers {
        let w_name = format!("W_{i}");
        let b_name = format!("b_{i}");
        let mm_out = format!("mm_{i}");
        let add_out = format!("add_{i}");
        let relu_out = format!("relu_{i}");

        // Weight matrix [dim, dim] with small values
        let w_data: Vec<f32> = (0..dim * dim).map(|j| 0.01 * (j % 97) as f32).collect();
        weights.insert(w_name.clone(), Tensor::new(w_data, vec![dim, dim]));

        // Bias [dim]
        let b_data: Vec<f32> = (0..dim).map(|j| 0.001 * j as f32).collect();
        weights.insert(b_name.clone(), Tensor::new(b_data, vec![dim]));

        nodes.push(make_node(
            OpKind::MatMul,
            &format!("matmul_{i}"),
            &[&prev_output, &w_name],
            &[&mm_out],
        ));
        nodes.push(make_node(
            OpKind::Add,
            &format!("add_{i}"),
            &[&mm_out, &b_name],
            &[&add_out],
        ));
        nodes.push(make_node(
            OpKind::Relu,
            &format!("relu_{i}"),
            &[&add_out],
            &[&relu_out],
        ));

        prev_output = relu_out;
    }

    let output_name = prev_output.clone();
    let graph = Graph {
        nodes,
        input_names: vec![input_name.clone()],
        output_names: vec![output_name.clone()],
    };

    (graph, weights, input_name, output_name)
}

// ── Session construction benchmark ───────────────────────────────────────────

fn bench_session_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Session_construction");

    let num_layers = 7; // 7 layers * 3 nodes = 21 nodes
    let dim = 128;
    let (graph, weights, _, _) = build_linear_chain(num_layers, dim);

    group.bench_function("21_node_graph", |bencher| {
        bencher.iter(|| {
            let g = graph.clone();
            let w = weights.clone();
            let _ = Session::from_graph(black_box(g), black_box(w));
        });
    });

    group.finish();
}

// ── Session inference benchmark ──────────────────────────────────────────────

fn bench_session_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("Session_inference");

    let num_layers = 5;
    let dim = 128;
    let (graph, weights, input_name, _) = build_linear_chain(num_layers, dim);

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, weights)
        .expect("failed to build session");

    let input_data: Vec<f32> = (0..dim).map(|i| 0.01 * i as f32).collect();
    let input_tensor = Tensor::new(input_data, vec![1, dim]);

    group.bench_function("5_layers_dim128", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session.run(black_box(&inputs));
        });
    });

    group.finish();
}

// ── Memory pool comparison benchmark ─────────────────────────────────────────

fn bench_memory_pool_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory_pool");

    let num_layers = 5;
    let dim = 128;

    let (graph_no_pool, weights_no_pool, input_name, _) = build_linear_chain(num_layers, dim);
    let (graph_pool, weights_pool, _, _) = build_linear_chain(num_layers, dim);

    let session_no_pool = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .with_memory_pool(false)
        .build_from_graph(graph_no_pool, weights_no_pool)
        .expect("failed to build session without pool");

    let session_with_pool = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .with_memory_pool(true)
        .build_from_graph(graph_pool, weights_pool)
        .expect("failed to build session with pool");

    let input_data: Vec<f32> = (0..dim).map(|i| 0.01 * i as f32).collect();
    let input_tensor = Tensor::new(input_data, vec![1, dim]);

    group.bench_function("without_pool", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session_no_pool.run(black_box(&inputs));
        });
    });

    group.bench_function("with_pool", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session_with_pool.run(black_box(&inputs));
        });
    });

    group.finish();
}

// ── Helpers for attributed nodes ─────────────────────────────────────────────

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

/// Build a single transformer block graph (simplified self-attention + FFN).
///
/// Architecture per block:
///   LayerNorm -> MatMul(Q) + MatMul(K) + MatMul(V) ->
///   Attention(Q*K^T / sqrt(d_k), softmax, *V) ->
///   MatMul(out_proj) -> Add(residual) ->
///   LayerNorm -> MatMul(ffn1) -> GELU -> MatMul(ffn2) -> Add(residual)
///
/// Dims: batch=1, seq_len, hidden_dim
fn build_transformer_block(
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> (Graph, HashMap<String, Tensor>, String, String) {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();

    let input_name = "input".to_string();
    let total = seq_len * hidden_dim;

    // ── LayerNorm 1 ─────────────────────────────────────────────────────────
    let ln1_scale_name = "ln1_scale";
    let ln1_bias_name = "ln1_bias";
    let ln1_out = "ln1_out";
    weights.insert(
        ln1_scale_name.to_string(),
        Tensor::new(vec![1.0; hidden_dim], vec![hidden_dim]),
    );
    weights.insert(
        ln1_bias_name.to_string(),
        Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
    );
    {
        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".to_string(), -1);
        attrs.floats.insert("epsilon".to_string(), 1e-5);
        nodes.push(make_node_with_attrs(
            OpKind::LayerNorm,
            "ln1",
            &[&input_name, ln1_scale_name, ln1_bias_name],
            &[ln1_out],
            attrs,
        ));
    }

    // ── Q, K, V projections ─────────────────────────────────────────────────
    for (label, out_name) in &[("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")] {
        let w_name = format!("{label}_weight");
        let b_name = format!("{label}_bias");
        let mm_out = format!("{label}_mm");
        let w_data: Vec<f32> = (0..hidden_dim * hidden_dim)
            .map(|j| 0.02 * ((j % 53) as f32 - 26.0) / 26.0)
            .collect();
        weights.insert(
            w_name.clone(),
            Tensor::new(w_data, vec![hidden_dim, hidden_dim]),
        );
        let b_data: Vec<f32> = vec![0.0; hidden_dim];
        weights.insert(b_name.clone(), Tensor::new(b_data, vec![hidden_dim]));
        nodes.push(make_node(
            OpKind::MatMul,
            &format!("{label}_matmul"),
            &[ln1_out, &w_name],
            &[&mm_out],
        ));
        nodes.push(make_node(
            OpKind::Add,
            &format!("{label}_add"),
            &[&mm_out, &b_name],
            &[out_name],
        ));
    }

    // ── Attention: scores = Q * K^T  ────────────────────────────────────────
    // Transpose K: [1, seq, hidden] -> [1, hidden, seq]
    {
        let mut attrs = Attributes::default();
        attrs.int_lists.insert("perm".to_string(), vec![0, 2, 1]);
        nodes.push(make_node_with_attrs(
            OpKind::Transpose,
            "k_transpose",
            &["k_proj"],
            &["k_t"],
            attrs,
        ));
    }
    nodes.push(make_node(
        OpKind::MatMul,
        "attn_scores",
        &["q_proj", "k_t"],
        &["attn_raw"],
    ));
    // Scale by 1/sqrt(hidden_dim) using Mul with a constant
    let scale_val = 1.0 / (hidden_dim as f32).sqrt();
    let scale_data = vec![scale_val; seq_len * seq_len];
    weights.insert(
        "attn_scale".to_string(),
        Tensor::new(scale_data, vec![1, seq_len, seq_len]),
    );
    nodes.push(make_node(
        OpKind::Mul,
        "attn_scale_mul",
        &["attn_raw", "attn_scale"],
        &["attn_scaled"],
    ));
    // Softmax
    {
        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".to_string(), -1);
        nodes.push(make_node_with_attrs(
            OpKind::Softmax,
            "attn_softmax",
            &["attn_scaled"],
            &["attn_probs"],
            attrs,
        ));
    }
    // Attention output = probs * V
    nodes.push(make_node(
        OpKind::MatMul,
        "attn_output",
        &["attn_probs", "v_proj"],
        &["attn_out"],
    ));

    // ── Output projection + residual ────────────────────────────────────────
    let out_w_data: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|j| 0.02 * ((j % 41) as f32 - 20.0) / 20.0)
        .collect();
    weights.insert(
        "out_proj_w".to_string(),
        Tensor::new(out_w_data, vec![hidden_dim, hidden_dim]),
    );
    weights.insert(
        "out_proj_b".to_string(),
        Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
    );
    nodes.push(make_node(
        OpKind::MatMul,
        "out_proj_mm",
        &["attn_out", "out_proj_w"],
        &["out_proj_mm_out"],
    ));
    nodes.push(make_node(
        OpKind::Add,
        "out_proj_add",
        &["out_proj_mm_out", "out_proj_b"],
        &["out_proj_out"],
    ));
    // Residual add
    nodes.push(make_node(
        OpKind::Add,
        "residual1",
        &[&input_name, "out_proj_out"],
        &["residual1_out"],
    ));

    // ── LayerNorm 2 ─────────────────────────────────────────────────────────
    weights.insert(
        "ln2_scale".to_string(),
        Tensor::new(vec![1.0; hidden_dim], vec![hidden_dim]),
    );
    weights.insert(
        "ln2_bias".to_string(),
        Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
    );
    {
        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".to_string(), -1);
        attrs.floats.insert("epsilon".to_string(), 1e-5);
        nodes.push(make_node_with_attrs(
            OpKind::LayerNorm,
            "ln2",
            &["residual1_out", "ln2_scale", "ln2_bias"],
            &["ln2_out"],
            attrs,
        ));
    }

    // ── FFN: MatMul -> GELU -> MatMul ───────────────────────────────────────
    let ffn1_w_data: Vec<f32> = (0..hidden_dim * ffn_dim)
        .map(|j| 0.02 * ((j % 67) as f32 - 33.0) / 33.0)
        .collect();
    weights.insert(
        "ffn1_w".to_string(),
        Tensor::new(ffn1_w_data, vec![hidden_dim, ffn_dim]),
    );
    weights.insert(
        "ffn1_b".to_string(),
        Tensor::new(vec![0.0; ffn_dim], vec![ffn_dim]),
    );
    nodes.push(make_node(
        OpKind::MatMul,
        "ffn1_mm",
        &["ln2_out", "ffn1_w"],
        &["ffn1_mm_out"],
    ));
    nodes.push(make_node(
        OpKind::Add,
        "ffn1_add",
        &["ffn1_mm_out", "ffn1_b"],
        &["ffn1_out"],
    ));
    nodes.push(make_node(
        OpKind::Gelu,
        "ffn_gelu",
        &["ffn1_out"],
        &["gelu_out"],
    ));

    let ffn2_w_data: Vec<f32> = (0..ffn_dim * hidden_dim)
        .map(|j| 0.02 * ((j % 59) as f32 - 29.0) / 29.0)
        .collect();
    weights.insert(
        "ffn2_w".to_string(),
        Tensor::new(ffn2_w_data, vec![ffn_dim, hidden_dim]),
    );
    weights.insert(
        "ffn2_b".to_string(),
        Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
    );
    nodes.push(make_node(
        OpKind::MatMul,
        "ffn2_mm",
        &["gelu_out", "ffn2_w"],
        &["ffn2_mm_out"],
    ));
    nodes.push(make_node(
        OpKind::Add,
        "ffn2_add",
        &["ffn2_mm_out", "ffn2_b"],
        &["ffn2_out"],
    ));
    // Residual add
    nodes.push(make_node(
        OpKind::Add,
        "residual2",
        &["residual1_out", "ffn2_out"],
        &["output"],
    ));

    let _ = total; // suppress unused warning
    let graph = Graph {
        nodes,
        input_names: vec![input_name.clone()],
        output_names: vec!["output".to_string()],
    };

    (graph, weights, input_name, "output".to_string())
}

// ── Transformer block benchmark ─────────────────────────────────────────────

fn bench_transformer_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transformer_block");

    let seq_len = 64;
    let hidden_dim = 256;
    let ffn_dim = hidden_dim * 4; // standard 4x expansion

    let (graph, weights, input_name, _) = build_transformer_block(seq_len, hidden_dim, ffn_dim);

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, weights)
        .expect("failed to build transformer session");

    let input_data: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| 0.01 * ((i % 97) as f32 - 48.0) / 48.0)
        .collect();
    let input_tensor = Tensor::new(input_data, vec![1, seq_len, hidden_dim]);

    group.bench_function("seq64_dim256_ffn1024", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session.run(black_box(&inputs));
        });
    });

    group.finish();
}

// ── ResNet-like benchmark (simplified linear blocks) ────────────────────────

/// Build a simplified ResNet-like model: chains of MatMul->Add->Relu with
/// skip connections every 2 layers.
fn build_resnet_like(
    num_blocks: usize,
    dim: usize,
) -> (Graph, HashMap<String, Tensor>, String, String) {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();

    let input_name = "input".to_string();
    let mut prev_output = input_name.clone();

    for blk in 0..num_blocks {
        let skip_input = prev_output.clone();

        // Two linear layers per block
        for layer in 0..2 {
            let idx = blk * 2 + layer;
            let w_name = format!("W_{idx}");
            let b_name = format!("b_{idx}");
            let mm_out = format!("mm_{idx}");
            let add_out = format!("add_{idx}");
            let relu_out = format!("relu_{idx}");

            let w_data: Vec<f32> = (0..dim * dim)
                .map(|j| 0.01 * ((j % 97) as f32 - 48.0) / 48.0)
                .collect();
            weights.insert(w_name.clone(), Tensor::new(w_data, vec![dim, dim]));
            weights.insert(b_name.clone(), Tensor::new(vec![0.001; dim], vec![dim]));

            nodes.push(make_node(
                OpKind::MatMul,
                &format!("matmul_{idx}"),
                &[&prev_output, &w_name],
                &[&mm_out],
            ));
            nodes.push(make_node(
                OpKind::Add,
                &format!("add_{idx}"),
                &[&mm_out, &b_name],
                &[&add_out],
            ));
            nodes.push(make_node(
                OpKind::Relu,
                &format!("relu_{idx}"),
                &[&add_out],
                &[&relu_out],
            ));

            prev_output = relu_out;
        }

        // Skip connection: add residual
        let skip_out = format!("skip_{blk}");
        nodes.push(make_node(
            OpKind::Add,
            &format!("skip_add_{blk}"),
            &[&prev_output, &skip_input],
            &[&skip_out],
        ));
        prev_output = skip_out;
    }

    let output_name = prev_output;
    let graph = Graph {
        nodes,
        input_names: vec![input_name.clone()],
        output_names: vec![output_name.clone()],
    };

    (graph, weights, input_name, output_name)
}

fn bench_resnet_like(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResNet_like");

    let num_blocks = 4;
    let dim = 256;

    let (graph, weights, input_name, _) = build_resnet_like(num_blocks, dim);

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, weights)
        .expect("failed to build resnet-like session");

    let input_data: Vec<f32> = (0..dim)
        .map(|i| 0.01 * ((i % 97) as f32 - 48.0) / 48.0)
        .collect();
    let input_tensor = Tensor::new(input_data, vec![1, dim]);

    group.bench_function("4_blocks_dim256", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session.run(black_box(&inputs));
        });
    });

    group.finish();
}

// ── BERT-like benchmark (stacked transformer blocks) ────────────────────────

fn bench_bert_like(c: &mut Criterion) {
    let mut group = c.benchmark_group("BERT_like");

    // 4-layer transformer encoder with smaller dims for benchmarking
    let seq_len = 32;
    let hidden_dim = 256;
    let ffn_dim = hidden_dim * 4;
    let num_layers = 4;

    // Build stacked transformer blocks by chaining them
    let mut all_nodes = Vec::new();
    let mut all_weights = HashMap::new();
    let input_name = "input".to_string();
    let mut prev_output = input_name.clone();

    for layer_idx in 0..num_layers {
        let prefix = format!("L{layer_idx}_");

        // LayerNorm 1
        let ln1_scale = format!("{prefix}ln1_scale");
        let ln1_bias = format!("{prefix}ln1_bias");
        let ln1_out = format!("{prefix}ln1_out");
        all_weights.insert(
            ln1_scale.clone(),
            Tensor::new(vec![1.0; hidden_dim], vec![hidden_dim]),
        );
        all_weights.insert(
            ln1_bias.clone(),
            Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
        );
        {
            let mut attrs = Attributes::default();
            attrs.ints.insert("axis".to_string(), -1);
            attrs.floats.insert("epsilon".to_string(), 1e-5);
            all_nodes.push(make_node_with_attrs(
                OpKind::LayerNorm,
                &format!("{prefix}ln1"),
                &[&prev_output, &ln1_scale, &ln1_bias],
                &[&ln1_out],
                attrs,
            ));
        }

        // Q, K, V projections
        for label in &["q", "k", "v"] {
            let w_name = format!("{prefix}{label}_w");
            let b_name = format!("{prefix}{label}_b");
            let mm_out = format!("{prefix}{label}_mm");
            let proj_out = format!("{prefix}{label}_proj");
            let w_data: Vec<f32> = (0..hidden_dim * hidden_dim)
                .map(|j| 0.02 * ((j % 53) as f32 - 26.0) / 26.0)
                .collect();
            all_weights.insert(
                w_name.clone(),
                Tensor::new(w_data, vec![hidden_dim, hidden_dim]),
            );
            all_weights.insert(
                b_name.clone(),
                Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
            );
            all_nodes.push(make_node(
                OpKind::MatMul,
                &format!("{prefix}{label}_matmul"),
                &[&ln1_out, &w_name],
                &[&mm_out],
            ));
            all_nodes.push(make_node(
                OpKind::Add,
                &format!("{prefix}{label}_add"),
                &[&mm_out, &b_name],
                &[&proj_out],
            ));
        }

        // Simplified attention: Q*K^T -> scale -> softmax -> *V
        let k_t = format!("{prefix}k_t");
        {
            let mut attrs = Attributes::default();
            attrs.int_lists.insert("perm".to_string(), vec![0, 2, 1]);
            all_nodes.push(make_node_with_attrs(
                OpKind::Transpose,
                &format!("{prefix}k_transpose"),
                &[&format!("{prefix}k_proj")],
                &[&k_t],
                attrs,
            ));
        }
        let attn_raw = format!("{prefix}attn_raw");
        all_nodes.push(make_node(
            OpKind::MatMul,
            &format!("{prefix}attn_scores"),
            &[&format!("{prefix}q_proj"), &k_t],
            &[&attn_raw],
        ));

        let scale_name = format!("{prefix}attn_scale");
        let scale_val = 1.0 / (hidden_dim as f32).sqrt();
        all_weights.insert(
            scale_name.clone(),
            Tensor::new(
                vec![scale_val; seq_len * seq_len],
                vec![1, seq_len, seq_len],
            ),
        );
        let attn_scaled = format!("{prefix}attn_scaled");
        all_nodes.push(make_node(
            OpKind::Mul,
            &format!("{prefix}attn_scale_mul"),
            &[&attn_raw, &scale_name],
            &[&attn_scaled],
        ));

        let attn_probs = format!("{prefix}attn_probs");
        {
            let mut attrs = Attributes::default();
            attrs.ints.insert("axis".to_string(), -1);
            all_nodes.push(make_node_with_attrs(
                OpKind::Softmax,
                &format!("{prefix}attn_softmax"),
                &[&attn_scaled],
                &[&attn_probs],
                attrs,
            ));
        }

        let attn_out = format!("{prefix}attn_out");
        all_nodes.push(make_node(
            OpKind::MatMul,
            &format!("{prefix}attn_output"),
            &[&attn_probs, &format!("{prefix}v_proj")],
            &[&attn_out],
        ));

        // Output projection + residual
        let out_w = format!("{prefix}out_w");
        let out_b = format!("{prefix}out_b");
        let out_mm = format!("{prefix}out_mm");
        let out_add = format!("{prefix}out_add");
        let residual1 = format!("{prefix}res1");
        let w_data: Vec<f32> = (0..hidden_dim * hidden_dim)
            .map(|j| 0.02 * ((j % 41) as f32 - 20.0) / 20.0)
            .collect();
        all_weights.insert(
            out_w.clone(),
            Tensor::new(w_data, vec![hidden_dim, hidden_dim]),
        );
        all_weights.insert(
            out_b.clone(),
            Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
        );
        all_nodes.push(make_node(
            OpKind::MatMul,
            &format!("{prefix}out_mm"),
            &[&attn_out, &out_w],
            &[&out_mm],
        ));
        all_nodes.push(make_node(
            OpKind::Add,
            &format!("{prefix}out_add"),
            &[&out_mm, &out_b],
            &[&out_add],
        ));
        all_nodes.push(make_node(
            OpKind::Add,
            &format!("{prefix}res1"),
            &[&prev_output, &out_add],
            &[&residual1],
        ));

        // LayerNorm 2
        let ln2_scale = format!("{prefix}ln2_scale");
        let ln2_bias = format!("{prefix}ln2_bias");
        let ln2_out = format!("{prefix}ln2_out");
        all_weights.insert(
            ln2_scale.clone(),
            Tensor::new(vec![1.0; hidden_dim], vec![hidden_dim]),
        );
        all_weights.insert(
            ln2_bias.clone(),
            Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
        );
        {
            let mut attrs = Attributes::default();
            attrs.ints.insert("axis".to_string(), -1);
            attrs.floats.insert("epsilon".to_string(), 1e-5);
            all_nodes.push(make_node_with_attrs(
                OpKind::LayerNorm,
                &format!("{prefix}ln2"),
                &[&residual1, &ln2_scale, &ln2_bias],
                &[&ln2_out],
                attrs,
            ));
        }

        // FFN
        let ffn1_w = format!("{prefix}ffn1_w");
        let ffn1_b = format!("{prefix}ffn1_b");
        let ffn1_mm = format!("{prefix}ffn1_mm");
        let ffn1_add_out = format!("{prefix}ffn1_add");
        let gelu_out = format!("{prefix}gelu");
        let ffn2_w = format!("{prefix}ffn2_w");
        let ffn2_b = format!("{prefix}ffn2_b");
        let ffn2_mm = format!("{prefix}ffn2_mm");
        let ffn2_add_out = format!("{prefix}ffn2_add");
        let residual2 = format!("{prefix}res2");

        let w1_data: Vec<f32> = (0..hidden_dim * ffn_dim)
            .map(|j| 0.02 * ((j % 67) as f32 - 33.0) / 33.0)
            .collect();
        all_weights.insert(
            ffn1_w.clone(),
            Tensor::new(w1_data, vec![hidden_dim, ffn_dim]),
        );
        all_weights.insert(
            ffn1_b.clone(),
            Tensor::new(vec![0.0; ffn_dim], vec![ffn_dim]),
        );
        all_nodes.push(make_node(
            OpKind::MatMul,
            &format!("{prefix}ffn1_mm"),
            &[&ln2_out, &ffn1_w],
            &[&ffn1_mm],
        ));
        all_nodes.push(make_node(
            OpKind::Add,
            &format!("{prefix}ffn1_add"),
            &[&ffn1_mm, &ffn1_b],
            &[&ffn1_add_out],
        ));
        all_nodes.push(make_node(
            OpKind::Gelu,
            &format!("{prefix}ffn_gelu"),
            &[&ffn1_add_out],
            &[&gelu_out],
        ));

        let w2_data: Vec<f32> = (0..ffn_dim * hidden_dim)
            .map(|j| 0.02 * ((j % 59) as f32 - 29.0) / 29.0)
            .collect();
        all_weights.insert(
            ffn2_w.clone(),
            Tensor::new(w2_data, vec![ffn_dim, hidden_dim]),
        );
        all_weights.insert(
            ffn2_b.clone(),
            Tensor::new(vec![0.0; hidden_dim], vec![hidden_dim]),
        );
        all_nodes.push(make_node(
            OpKind::MatMul,
            &format!("{prefix}ffn2_mm"),
            &[&gelu_out, &ffn2_w],
            &[&ffn2_mm],
        ));
        all_nodes.push(make_node(
            OpKind::Add,
            &format!("{prefix}ffn2_add"),
            &[&ffn2_mm, &ffn2_b],
            &[&ffn2_add_out],
        ));
        all_nodes.push(make_node(
            OpKind::Add,
            &format!("{prefix}res2"),
            &[&residual1, &ffn2_add_out],
            &[&residual2],
        ));

        prev_output = residual2;
    }

    let output_name = prev_output;
    let graph = Graph {
        nodes: all_nodes,
        input_names: vec![input_name.clone()],
        output_names: vec![output_name.clone()],
    };

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph, all_weights)
        .expect("failed to build BERT-like session");

    let input_data: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| 0.01 * ((i % 97) as f32 - 48.0) / 48.0)
        .collect();
    let input_tensor = Tensor::new(input_data, vec![1, seq_len, hidden_dim]);

    group.bench_function("4_layers_seq32_dim256", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session.run(black_box(&inputs));
        });
    });

    group.finish();
}

// ── Criterion setup ──────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_session_construction,
    bench_session_inference,
    bench_memory_pool_comparison,
    bench_transformer_block,
    bench_resnet_like,
    bench_bert_like,
);
criterion_main!(benches);
