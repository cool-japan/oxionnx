# OxiONNX

**Pure Rust ONNX Inference Engine -- Zero C/C++ Dependencies**

[![CI](https://github.com/cool-japan/oxionnx/actions/workflows/ci.yml/badge.svg)](https://github.com/cool-japan/oxionnx/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/oxionnx.svg)](https://crates.io/crates/oxionnx)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

OxiONNX is a high-performance ONNX inference engine written in pure Rust.
It supports 147 ONNX operators, GPU acceleration via wgpu, SIMD optimization,
and runs on any platform including WebAssembly.

**30,000+ lines of Rust | 590+ tests | 0 clippy warnings**

## Features

- **Pure Rust** -- Zero C/C++/Fortran dependencies. Safe, portable, auditable.
- **147 ONNX operators** -- Math, NN, Conv, Shape, Indexing, Comparison, RNN, Attention, ML
- **GPU acceleration** -- wgpu compute shaders for MatMul, Softmax, ReLU, etc.
- **SIMD optimization** -- NEON (aarch64) and AVX2 (x86_64) for element-wise ops
- **Multi-dtype** -- f32, f16, bf16, i8, i32, i64 with automatic type promotion
- **INT8 quantization** -- Quantized MatMul with per-channel scale/zero-point
- **Mixed precision** -- f16 activations with f32 accumulation
- **Graph optimization** -- Constant folding, operator fusion, CSE, dead code elimination
- **Memory efficiency** -- Arena allocator, buffer pooling, strided tensor views
- **Streaming inference** -- Token-by-token generation for autoregressive models
- **Async execution** -- Non-blocking inference via `run_async()`
- **Control flow** -- If/Loop/Scan operators with nested subgraph execution
- **Model encryption** -- AES-GCM encrypted model files
- **WebAssembly** -- Run in the browser via wasm-bindgen
- **no_std** -- Core types work without std (alloc only)
- **Session caching** -- Save/load pre-optimized graphs to skip re-optimization

## Quick Start

```rust
use oxionnx::{Session, Tensor};
use std::collections::HashMap;

// Load model
let session = Session::from_file("model.onnx".as_ref())?;

// Prepare input
let mut inputs = HashMap::new();
inputs.insert("input", Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]));

// Run inference
let outputs = session.run(&inputs)?;
println!("{:?}", outputs);
```

## Session Builder

```rust
use oxionnx::{Session, OptLevel};

let session = Session::builder()
    .with_optimization_level(OptLevel::All)
    .with_memory_pool(true)
    .with_parallel_execution(true)
    .with_profiling()
    .load("model.onnx".as_ref())?;
```

## Supported Operators

OxiONNX implements 147 operators across the full ONNX specification:

| Category | Count | Examples |
|----------|-------|---------|
| Math | 38 | MatMul, Gemm, Add, Mul, Pow, Sqrt, Reduce*, Trig |
| Neural Network | 35 | Relu, Sigmoid, Softmax, LayerNorm, BatchNorm, GELU, SiLU |
| Convolution | 8 | Conv, ConvTranspose, MaxPool, AveragePool, GlobalAvgPool |
| Shape | 14 | Reshape, Transpose, Concat, Slice, Split, Flatten |
| Indexing | 12 | Gather, Scatter, Where, OneHot, Compress, Unique |
| Comparison | 13 | Equal, Greater, Less, And, Or, Not, IsInf, IsNaN |
| RNN/Attention | 8 | LSTM, GRU, Attention, MultiHeadAttention, RotaryEmbedding |
| ONNX-ML | 12 | LinearClassifier, TreeEnsemble, SVM, Normalizer, TfIdf |
| Control Flow | 3 | If, Loop, Scan |
| Quantization | 4 | QuantizeLinear, DequantizeLinear, QLinearMatMul, QLinearConv |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `gpu` | GPU acceleration via wgpu |
| `simd` | SIMD-accelerated element-wise ops |
| `encryption` | AES-GCM model encryption |
| `mmap` | Memory-mapped weight loading |
| `wasm` | WebAssembly browser bindings |

## Architecture

```
oxionnx (root)           -- Session, optimizer, execution engine
  oxionnx-core           -- Tensor, DType, Graph, Operator trait, OnnxError
  oxionnx-ops            -- 147 operator implementations
  oxionnx-proto          -- Pure Rust ONNX protobuf parser
  oxionnx-gpu            -- wgpu compute backend (optional)
```

## License

Apache-2.0

## Author

COOLJAPAN OU (Team Kitasan)
