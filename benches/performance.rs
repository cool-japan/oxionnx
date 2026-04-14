//! Performance comparison benchmark suite for OxiONNX.
//!
//! Simulates realistic model workloads (ResNet-50 backbone, BERT-base attention)
//! and provides operator microbenchmarks at production-relevant sizes.
//! Run with: `cargo bench --bench performance`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, SessionBuilder, Tensor};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

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

fn sequential_tensor(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| 0.001 * (i % 1000) as f32).collect();
    Tensor::new(data, shape.to_vec())
}

fn constant_tensor(shape: &[usize], val: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new(vec![val; n], shape.to_vec())
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. ResNet-50 Backbone Simulation
// ═════════════════════════════════════════════════════════════════════════════
//
// Simulates: Conv2D(3→64, 7×7, stride=2) → BatchNorm → ReLU → MaxPool
//            → 4 residual blocks (MatMul + Add + ReLU each)
// Input: batch=1, 3 channels, 224×224

fn build_resnet50_backbone() -> (Graph, HashMap<String, Tensor>, String) {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();

    let input_name = "image";

    // ── Stage 1: Conv2D(3→64, 7×7, stride=2, pad=3) ────────────────────────
    // Input: [1, 3, 224, 224] → Output: [1, 64, 112, 112]
    let conv1_w: Vec<f32> = (0..64 * 3 * 7 * 7)
        .map(|j| 0.01 * ((j % 97) as f32 - 48.0) / 48.0)
        .collect();
    weights.insert(
        "conv1_w".to_string(),
        Tensor::new(conv1_w, vec![64, 3, 7, 7]),
    );
    weights.insert("conv1_b".to_string(), Tensor::new(vec![0.0; 64], vec![64]));
    {
        let mut attrs = Attributes::default();
        attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
        attrs.int_lists.insert("pads".to_string(), vec![3, 3, 3, 3]);
        attrs
            .int_lists
            .insert("kernel_shape".to_string(), vec![7, 7]);
        attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
        attrs.ints.insert("group".to_string(), 1);
        nodes.push(make_node_with_attrs(
            OpKind::Conv,
            "conv1",
            &[input_name, "conv1_w", "conv1_b"],
            &["conv1_out"],
            attrs,
        ));
    }

    // ── BatchNorm ───────────────────────────────────────────────────────────
    weights.insert(
        "bn1_scale".to_string(),
        Tensor::new(vec![1.0; 64], vec![64]),
    );
    weights.insert("bn1_bias".to_string(), Tensor::new(vec![0.0; 64], vec![64]));
    weights.insert("bn1_mean".to_string(), Tensor::new(vec![0.0; 64], vec![64]));
    weights.insert("bn1_var".to_string(), Tensor::new(vec![1.0; 64], vec![64]));
    nodes.push(make_node(
        OpKind::BatchNorm,
        "bn1",
        &["conv1_out", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"],
        &["bn1_out"],
    ));

    // ── ReLU ────────────────────────────────────────────────────────────────
    nodes.push(make_node(
        OpKind::Relu,
        "relu1",
        &["bn1_out"],
        &["relu1_out"],
    ));

    // ── MaxPool(3×3, stride=2, pad=1) ───────────────────────────────────────
    // [1, 64, 112, 112] → [1, 64, 56, 56]
    {
        let mut attrs = Attributes::default();
        attrs
            .int_lists
            .insert("kernel_shape".to_string(), vec![3, 3]);
        attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
        attrs.int_lists.insert("pads".to_string(), vec![1, 1, 1, 1]);
        nodes.push(make_node_with_attrs(
            OpKind::MaxPool,
            "maxpool",
            &["relu1_out"],
            &["pool_out"],
            attrs,
        ));
    }

    // ── 4 Residual Blocks ───────────────────────────────────────────────────
    // Each block: flatten spatially → MatMul(dim→dim) + Add + ReLU → reshape back
    // For benchmarking, we keep it as flattened [1, 64*56*56] → MatMul →
    // reshape back. But since real ResNets use conv, we simulate the compute
    // density with MatMul chains on the flattened feature maps.
    //
    // Actually, to keep this realistic with the ops we have, let's do
    // 4 blocks of: Add(residual) + ReLU on the pooled output [1, 64, 56, 56]

    let mut prev_out = "pool_out".to_string();

    for blk in 0..4 {
        let skip_input = prev_out.clone();

        // Simulate residual block computation with element-wise ops
        // Two Add(weight) + ReLU layers per block
        for layer in 0..2 {
            let idx = blk * 2 + layer;
            let w_name = format!("res_w_{idx}");
            let add_out = format!("res_add_{idx}");
            let relu_out = format!("res_relu_{idx}");

            // Weight tensor with same shape as feature map [1, 64, 56, 56]
            let w_data: Vec<f32> = (0..64 * 56 * 56)
                .map(|j| 0.001 * ((j % 97) as f32 - 48.0) / 48.0)
                .collect();
            weights.insert(w_name.clone(), Tensor::new(w_data, vec![1, 64, 56, 56]));

            nodes.push(make_node(
                OpKind::Add,
                &format!("res_add_{idx}"),
                &[&prev_out, &w_name],
                &[&add_out],
            ));
            nodes.push(make_node(
                OpKind::Relu,
                &format!("res_relu_{idx}"),
                &[&add_out],
                &[&relu_out],
            ));
            prev_out = relu_out;
        }

        // Skip connection
        let skip_out = format!("skip_{blk}");
        nodes.push(make_node(
            OpKind::Add,
            &format!("skip_{blk}"),
            &[&prev_out, &skip_input],
            &[&skip_out],
        ));
        prev_out = skip_out;
    }

    let graph = Graph {
        nodes,
        input_names: vec![input_name.to_string()],
        output_names: vec![prev_out],
        ..Default::default()
    };

    (graph, weights, input_name.to_string())
}

fn bench_resnet50_backbone(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResNet50_backbone");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let (graph, weights, input_name) = build_resnet50_backbone();

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .with_memory_pool(true)
        .build_from_graph(graph, weights)
        .expect("failed to build resnet50 backbone session");

    // Input: [1, 3, 224, 224]
    let input_tensor = sequential_tensor(&[1, 3, 224, 224]);

    group.bench_function("batch1_224x224", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session.run(black_box(&inputs));
        });
    });

    group.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. BERT-base Attention Simulation
// ═════════════════════════════════════════════════════════════════════════════
//
// MatMul(batch=1, seq=128, hidden=768) for Q,K,V projections
// → reshape/transpose → batched MatMul (12 heads) → Softmax
// → batched MatMul → reshape → output projection

fn build_bert_attention() -> (Graph, HashMap<String, Tensor>, String) {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();

    let seq_len = 128;
    let hidden = 768;
    let input_name = "hidden_states";

    // ── Q, K, V Projections: [1, 128, 768] × [768, 768] → [1, 128, 768] ──
    for (label, out_name) in &[("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")] {
        let w_name = format!("{label}_weight");
        let b_name = format!("{label}_bias");
        let mm_out = format!("{label}_mm");

        let w_data: Vec<f32> = (0..hidden * hidden)
            .map(|j| 0.02 * ((j % 53) as f32 - 26.0) / 26.0)
            .collect();
        weights.insert(w_name.clone(), Tensor::new(w_data, vec![hidden, hidden]));
        weights.insert(b_name.clone(), Tensor::new(vec![0.0; hidden], vec![hidden]));

        nodes.push(make_node(
            OpKind::MatMul,
            &format!("{label}_matmul"),
            &[input_name, &w_name],
            &[&mm_out],
        ));
        nodes.push(make_node(
            OpKind::Add,
            &format!("{label}_bias_add"),
            &[&mm_out, &b_name],
            &[out_name],
        ));
    }

    // ── Attention scores: Q × K^T ──────────────────────────────────────────
    // Transpose K: [1, 128, 768] → [1, 768, 128]
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

    // Q × K^T: [1, 128, 768] × [1, 768, 128] → [1, 128, 128]
    nodes.push(make_node(
        OpKind::MatMul,
        "attn_scores",
        &["q_proj", "k_t"],
        &["attn_raw"],
    ));

    // Scale: divide by sqrt(768) ≈ 27.7
    let scale_val = 1.0 / (hidden as f32).sqrt();
    weights.insert(
        "attn_scale".to_string(),
        Tensor::new(
            vec![scale_val; seq_len * seq_len],
            vec![1, seq_len, seq_len],
        ),
    );
    nodes.push(make_node(
        OpKind::Mul,
        "attn_scale_mul",
        &["attn_raw", "attn_scale"],
        &["attn_scaled"],
    ));

    // Softmax over last axis
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

    // attn_probs × V: [1, 128, 128] × [1, 128, 768] → [1, 128, 768]
    nodes.push(make_node(
        OpKind::MatMul,
        "attn_output",
        &["attn_probs", "v_proj"],
        &["attn_out"],
    ));

    // ── Output projection: [1, 128, 768] × [768, 768] → [1, 128, 768] ────
    let out_w_data: Vec<f32> = (0..hidden * hidden)
        .map(|j| 0.02 * ((j % 41) as f32 - 20.0) / 20.0)
        .collect();
    weights.insert(
        "out_proj_w".to_string(),
        Tensor::new(out_w_data, vec![hidden, hidden]),
    );
    weights.insert(
        "out_proj_b".to_string(),
        Tensor::new(vec![0.0; hidden], vec![hidden]),
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
        &["output"],
    ));

    // ── Residual + LayerNorm ────────────────────────────────────────────────
    nodes.push(make_node(
        OpKind::Add,
        "residual_add",
        &[input_name, "output"],
        &["residual_out"],
    ));

    weights.insert(
        "ln_scale".to_string(),
        Tensor::new(vec![1.0; hidden], vec![hidden]),
    );
    weights.insert(
        "ln_bias".to_string(),
        Tensor::new(vec![0.0; hidden], vec![hidden]),
    );
    {
        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".to_string(), -1);
        attrs.floats.insert("epsilon".to_string(), 1e-5);
        nodes.push(make_node_with_attrs(
            OpKind::LayerNorm,
            "layer_norm",
            &["residual_out", "ln_scale", "ln_bias"],
            &["ln_out"],
            attrs,
        ));
    }

    let graph = Graph {
        nodes,
        input_names: vec![input_name.to_string()],
        output_names: vec!["ln_out".to_string()],
        ..Default::default()
    };

    (graph, weights, input_name.to_string())
}

fn bench_bert_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("BERT_attention");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let (graph, weights, input_name) = build_bert_attention();

    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .with_memory_pool(true)
        .build_from_graph(graph, weights)
        .expect("failed to build BERT attention session");

    // Input: [1, 128, 768]
    let input_tensor = sequential_tensor(&[1, 128, 768]);

    group.bench_function("seq128_hidden768", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session.run(black_box(&inputs));
        });
    });

    group.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Operator Microbenchmarks at Realistic Sizes
// ═════════════════════════════════════════════════════════════════════════════

fn bench_matmul_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perf_MatMul");

    for &size in &[512, 1024, 2048] {
        let a = sequential_tensor(&[size, size]);
        let b = sequential_tensor(&[size, size]);

        let sample_size = if size >= 2048 { 10 } else { 20 };
        group.sample_size(sample_size);

        group.bench_with_input(BenchmarkId::new("square", size), &size, |bencher, _| {
            bencher.iter(|| {
                let _ = oxionnx_ops::math::matmul(black_box(&a), black_box(&b));
            });
        });
    }

    group.finish();
}

fn bench_conv2d_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perf_Conv2D");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    // batch=1, 64 channels, 56×56, 3×3 kernel, stride=1, pad=1
    let input = sequential_tensor(&[1, 64, 56, 56]);
    let kernel = sequential_tensor(&[64, 64, 3, 3]);
    let bias = constant_tensor(&[64], 0.0);

    group.bench_function("64ch_56x56_k3_s1_p1", |bencher| {
        bencher.iter(|| {
            oxionnx_ops::conv::conv2d(
                black_box(&input),
                black_box(&kernel),
                Some(black_box(&bias)),
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                1,
            )
        });
    });

    group.finish();
}

fn bench_softmax_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perf_Softmax");

    // [1, 128, 768] — BERT hidden state softmax
    let x = sequential_tensor(&[1, 128, 768]);

    group.bench_function("1x128x768", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::nn::softmax(black_box(&x), -1);
        });
    });

    group.finish();
}

fn bench_layer_norm_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perf_LayerNorm");

    // [1, 128, 768] — BERT sequence
    let x = sequential_tensor(&[1, 128, 768]);
    let scale = constant_tensor(&[768], 1.0);
    let bias = constant_tensor(&[768], 0.0);

    group.bench_function("1x128x768", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::nn::layer_norm(
                black_box(&x),
                black_box(&scale),
                Some(black_box(&bias)),
                1e-5,
                -1,
            );
        });
    });

    group.finish();
}

fn bench_gelu_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perf_GELU");

    // 100K elements
    let x = sequential_tensor(&[100_000]);

    group.bench_function("100k_elements", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::nn::gelu(black_box(&x));
        });
    });

    group.finish();
}

fn bench_add_broadcast_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perf_Add_broadcast");

    // [1, 128, 768] + [768] — broadcast add
    let a = sequential_tensor(&[1, 128, 768]);
    let b = sequential_tensor(&[768]);

    group.bench_function("1x128x768_plus_768", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::math::add(black_box(&a), black_box(&b));
        });
    });

    group.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Optimization Pass Benchmark
// ═════════════════════════════════════════════════════════════════════════════
//
// Measure session load time with and without optimization passes
// for a realistically-sized graph.

fn build_optimizable_graph(
    num_layers: usize,
    dim: usize,
) -> (Graph, HashMap<String, Tensor>, String) {
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

        let w_data: Vec<f32> = (0..dim * dim)
            .map(|j| 0.01 * ((j % 97) as f32 - 48.0) / 48.0)
            .collect();
        weights.insert(w_name.clone(), Tensor::new(w_data, vec![dim, dim]));
        weights.insert(b_name.clone(), Tensor::new(vec![0.001; dim], vec![dim]));

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

        // Add a dead branch every other layer to exercise dead code elimination
        if i % 2 == 0 {
            let dead_out = format!("dead_{i}");
            nodes.push(make_node(
                OpKind::Relu,
                &format!("dead_relu_{i}"),
                &[&add_out],
                &[&dead_out],
            ));
        }

        prev_output = relu_out;
    }

    let graph = Graph {
        nodes,
        input_names: vec![input_name.clone()],
        output_names: vec![prev_output],
        ..Default::default()
    };

    (graph, weights, input_name)
}

fn bench_optimization_passes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimization_passes");
    group.sample_size(50);

    let num_layers = 20;
    let dim = 256;

    // Benchmark session construction WITHOUT optimization
    group.bench_function("load_no_optimization_20layers", |bencher| {
        let (graph, weights, _) = build_optimizable_graph(num_layers, dim);
        bencher.iter(|| {
            let g = graph.clone();
            let w = weights.clone();
            let _ = SessionBuilder::new()
                .with_optimization_level(OptLevel::None)
                .build_from_graph(black_box(g), black_box(w));
        });
    });

    // Benchmark session construction WITH full optimization
    group.bench_function("load_full_optimization_20layers", |bencher| {
        let (graph, weights, _) = build_optimizable_graph(num_layers, dim);
        bencher.iter(|| {
            let g = graph.clone();
            let w = weights.clone();
            let _ = SessionBuilder::new()
                .with_optimization_level(OptLevel::All)
                .build_from_graph(black_box(g), black_box(w));
        });
    });

    // Benchmark inference speed: no opt vs full opt
    let (graph_none, weights_none, input_name) = build_optimizable_graph(num_layers, dim);
    let (graph_all, weights_all, _) = build_optimizable_graph(num_layers, dim);

    let session_none = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .build_from_graph(graph_none, weights_none)
        .expect("build none");

    let session_all = SessionBuilder::new()
        .with_optimization_level(OptLevel::All)
        .build_from_graph(graph_all, weights_all)
        .expect("build all");

    let input_data: Vec<f32> = (0..dim).map(|i| 0.01 * i as f32).collect();
    let input_tensor = Tensor::new(input_data, vec![1, dim]);

    group.bench_function("inference_no_opt_20layers", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session_none.run(black_box(&inputs));
        });
    });

    group.bench_function("inference_full_opt_20layers", |bencher| {
        bencher.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert(input_name.as_str(), input_tensor.clone());
            let _ = session_all.run(black_box(&inputs));
        });
    });

    group.finish();
}

// ── Criterion setup ──────────────────────────────────────────────────────────

criterion_group!(
    model_workloads,
    bench_resnet50_backbone,
    bench_bert_attention,
);

criterion_group!(
    operator_microbenchmarks,
    bench_matmul_realistic,
    bench_conv2d_realistic,
    bench_softmax_realistic,
    bench_layer_norm_realistic,
    bench_gelu_realistic,
    bench_add_broadcast_realistic,
);

criterion_group!(optimization, bench_optimization_passes);

criterion_main!(model_workloads, operator_microbenchmarks, optimization);
