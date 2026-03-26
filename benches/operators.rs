//! Per-operator microbenchmarks for oxionnx.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxionnx_core::Tensor;
use std::hint::black_box;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Create a tensor filled with a constant value.
fn constant_tensor(shape: &[usize], val: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new(vec![val; n], shape.to_vec())
}

/// Create a tensor with sequential values (0.001 * i) to avoid degenerate cases.
fn sequential_tensor(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| 0.001 * i as f32).collect();
    Tensor::new(data, shape.to_vec())
}

// ── MatMul benchmarks ────────────────────────────────────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMul");
    for size in [128, 512, 1024] {
        let a = sequential_tensor(&[size, size]);
        let b = sequential_tensor(&[size, size]);
        group.bench_with_input(BenchmarkId::new("square", size), &size, |bencher, _| {
            bencher.iter(|| {
                let _ = oxionnx_ops::math::matmul(black_box(&a), black_box(&b));
            });
        });
    }
    group.finish();
}

// ── Conv2D benchmarks ────────────────────────────────────────────────────────

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("Conv2D");

    // Input: [1, 64, 56, 56], kernel: [64, 64, 3, 3], stride=1, pad=1
    let input = sequential_tensor(&[1, 64, 56, 56]);
    let kernel = sequential_tensor(&[64, 64, 3, 3]);
    let bias = constant_tensor(&[64], 0.0);

    group.bench_function("64ch_56x56_k3", |bencher| {
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

// ── Softmax benchmarks ───────────────────────────────────────────────────────

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("Softmax");

    let imagenet = sequential_tensor(&[1, 1000]);
    group.bench_function("imagenet_1x1000", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::nn::softmax(black_box(&imagenet), -1);
        });
    });

    let gpt2 = sequential_tensor(&[1, 50257]);
    group.bench_function("gpt2_1x50257", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::nn::softmax(black_box(&gpt2), -1);
        });
    });

    group.finish();
}

// ── LayerNorm benchmarks ─────────────────────────────────────────────────────

fn bench_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("LayerNorm");

    // BERT hidden: [1, 512, 768]
    let x = sequential_tensor(&[1, 512, 768]);
    let scale = constant_tensor(&[768], 1.0);
    let bias = constant_tensor(&[768], 0.0);

    group.bench_function("bert_1x512x768", |bencher| {
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

// ── Element-wise op benchmarks ───────────────────────────────────────────────

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("Elementwise");

    let a = sequential_tensor(&[1, 512, 768]);
    let b = sequential_tensor(&[1, 512, 768]);

    group.bench_function("add_1x512x768", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::math::add(black_box(&a), black_box(&b));
        });
    });

    group.bench_function("mul_1x512x768", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::math::mul(black_box(&a), black_box(&b));
        });
    });

    group.bench_function("relu_1x512x768", |bencher| {
        bencher.iter(|| {
            let _ = oxionnx_ops::nn::relu(black_box(&a));
        });
    });

    group.finish();
}

// ── Criterion setup ──────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_matmul,
    bench_conv2d,
    bench_softmax,
    bench_layer_norm,
    bench_elementwise,
);
criterion_main!(benches);
