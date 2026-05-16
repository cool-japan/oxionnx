# OxiONNX

**Pure Rust ONNX Inference Engine -- Zero C/C++ Dependencies**

[![Crates.io](https://img.shields.io/crates/v/oxionnx.svg)](https://crates.io/crates/oxionnx)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

OxiONNX is a high-performance ONNX inference engine written in pure Rust.
It supports 165 ONNX operators, GPU acceleration via wgpu, SIMD optimization,
and runs on any platform including WebAssembly.

**63,460 lines of Rust | 1,179 tests | 0 clippy warnings**

## Features

- **Pure Rust** -- Zero C/C++/Fortran dependencies. Safe, portable, auditable.
- **165 ONNX operators** -- Math, NN, Conv, Shape, Indexing, Comparison, RNN, Attention, ML
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
- **Native dtype dispatch** -- `run_typed()` path executes 40+ operators natively (no f32 round-trip) via `TypedOpContext`; MatMul natively handles F32/F16/BF16/I8→I32/I32 dtypes
- **DirectML backend** -- Windows D3D12 execution provider (`directml` feature) with CPU fallback on other platforms
- **Zero-copy output reuse** -- All 121 operators support pre-allocated output slot reuse via `execute_into_slots`; 52 operators have hand-coded zero-copy kernels (Gather, ScatterND, ScatterElements, shape/pool/elementwise ops) — no memcpy, pointer-identity across inference runs with `IoBinding`

## Status

| Crate | Status | Tests |
|-------|--------|-------|
| `oxionnx` (root) | Alpha | 525 passing |
| `oxionnx-core` | Stable | 36 passing |
| `oxionnx-ops` | Alpha | 554 passing |
| `oxionnx-proto` | Stable | 37 passing |
| `oxionnx-gpu` | Alpha | 17 passing |
| `oxionnx-cuda` | Partial | 4 passing (GEMM/elementwise/softmax via OxiCUDA; Conv stubbed) |
| `oxionnx-directml` | Planned | 4 passing (Windows scaffold; HLSL shaders defined but not yet bound) |
| `oxionnx-coreml` | Partial | 2 passing (CoreML session bridge; macOS only) |

**Total: 1,179 tests passing, 0 clippy warnings, 63,460 SLoC**

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

OxiONNX implements 165 ONNX operators (plus 21 aliases including the `ai.onnx.ml.*` domain)

| Category | Count | Examples |
|----------|-------|---------|
| Math | 46 | MatMul, Gemm, Add, Mul, Pow, Sqrt, Reduce* (incl. L1/L2/LogSum/LogSumExp/SumSquare), Trig, ArgMax/Min, CumSum, TopK, BitShift, VariadicMin/Max/Mean/Sum |
| Neural Network | 33 | Relu, Sigmoid, Softmax, LayerNorm, BatchNorm, GELU, SiLU, Mish, GroupNorm, InstanceNorm, RmsNorm, Hardmax, Shrink |
| Convolution / Pool | 8 | Conv, ConvTranspose, MaxPool, AveragePool, GlobalAvgPool, GlobalMaxPool, Pad, Resize |
| Shape | 14 | Reshape, Transpose, Concat, Slice, Split, Flatten, Tile, DepthToSpace, SpaceToDepth, ReverseSequence, Size, Expand, Squeeze, Unsqueeze |
| Indexing / Quant | 11 | Gather, GatherElements, GatherND, Scatter, ScatterND, Where, OneHot, Compress, Unique, QuantizeLinear, DequantizeLinear |
| Comparison / Logic | 25 | Equal, Greater, Less, And, Or, Not, Xor, Bitwise* (And/Or/Xor/Not), IsInf, IsNaN, NonZero, Cast, Constant, Einsum, ConstantOfShape, EyeLike, Trilu, Identity, Shape, NonMaxSuppression |
| RNN / Attention | 7 | LSTM, GRU, Attention, MultiHeadAttention, RotaryEmbedding, GridSample, RoiAlign |
| DSP | 7 | DFT, STFT, HannWindow, HammingWindow, BlackmanWindow, MelWeightMatrix, Bernoulli |
| Control Flow | 3 | If, Loop, Scan |
| ONNX-ML | 11 | LinearClassifier, LinearRegressor, TreeEnsembleClassifier/Regressor, SVMClassifier/Regressor, Normalizer, Scaler, LabelEncoder, TfIdfVectorizer, StringNormalizer |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `gpu` | GPU acceleration via wgpu |
| `simd` | SIMD-accelerated element-wise ops |
| `encryption` | AES-GCM model encryption |
| `cuda` | CUDA GPU acceleration via OxiCUDA |
| `mmap` | Memory-mapped weight loading |
| `wasm` | WebAssembly browser bindings |
| `ndarray` | ndarray interop for Tensor conversion |
| `directml` | DirectML GPU acceleration (Windows, via D3D12) |

## Architecture

```
oxionnx (root)           -- Session, optimizer, execution engine
  oxionnx-core           -- Tensor, DType, Graph, Operator trait, OnnxError
  oxionnx-ops            -- 165 operator implementations
  oxionnx-proto          -- Pure Rust ONNX protobuf parser
  oxionnx-gpu            -- wgpu compute backend (optional)
  oxionnx-cuda           -- CUDA dispatch layer via OxiCUDA (optional)
  oxionnx-directml       -- DirectML execution provider for Windows D3D12 (optional)
  oxionnx-coreml         -- CoreML execution provider for macOS/iOS (optional)
```

## Performance

OxiONNX is a pure Rust implementation with no C/C++ BLAS dependency.
Run `cargo bench --bench performance` to measure on your hardware.

### Operator Microbenchmarks

| Operation | Size | Implementation | Notes |
|-----------|------|----------------|-------|
| MatMul | 512×512 | `matrixmultiply` crate | Run `cargo bench` to measure |
| MatMul | 1024×1024 | `matrixmultiply` crate | Run `cargo bench` to measure |
| MatMul | 2048×2048 | `matrixmultiply` crate | Run `cargo bench` to measure |
| Conv2D | 64ch, 56×56, 3×3 | im2col + matmul | Run `cargo bench` to measure |
| Softmax | [1, 128, 768] | Numerically stable (log-sum-exp) | Run `cargo bench` to measure |
| LayerNorm | [1, 128, 768] | Fused mean/var + scale/bias | Run `cargo bench` to measure |
| GELU | 100K elements | SIMD-accelerated (with `simd` feature) | Run `cargo bench` to measure |
| Add (broadcast) | [1, 128, 768] + [768] | Auto-broadcast | Run `cargo bench` to measure |

### End-to-End Model Workloads

| Workload | Description | Notes |
|----------|-------------|-------|
| ResNet-50 backbone | Conv(3→64, 7×7) → BN → ReLU → MaxPool → 4 residual blocks | batch=1, 224×224 input |
| BERT attention | Q/K/V projections → scaled dot-product attention → output proj | seq=128, hidden=768 |
| Transformer block | LayerNorm → Attention → FFN(GELU) → Residual | Stacked 4-layer encoder |
| Optimization passes | Session load with/without graph optimization | 20-layer graph with dead code |

### Performance Characteristics

- **Pure Rust, zero C/BLAS**: All computation uses `matrixmultiply` (pure Rust BLAS-like) and hand-written kernels
- **SIMD**: Optional NEON (aarch64) and AVX2 (x86_64) acceleration for element-wise ops via `--features simd`
- **Graph optimization**: Constant folding, operator fusion, CSE, and dead code elimination reduce runtime overhead
- **Memory pooling**: Buffer reuse across inference calls reduces allocation pressure
- **Parallelism**: Rayon-based parallel execution of independent graph branches

> **Comparison note**: OxiONNX prioritizes portability and safety (pure Rust, no unsafe in ops).
> For absolute peak throughput, C++ runtimes like onnxruntime (with MKL/cuDNN) will be faster
> on operations dominated by BLAS. OxiONNX targets use cases where pure Rust, WebAssembly
> compatibility, and zero native dependencies are more important than raw FLOPS.

## License

Apache-2.0

## Author

COOLJAPAN OU (Team Kitasan)
