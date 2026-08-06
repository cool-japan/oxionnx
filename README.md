# OxiONNX

**Pure Rust ONNX Inference Engine -- Zero C/C++ Dependencies**

[![Crates.io](https://img.shields.io/crates/v/oxionnx.svg)](https://crates.io/crates/oxionnx)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

OxiONNX is a high-performance ONNX inference engine written in pure Rust.
It supports 188 ONNX operators, GPU acceleration via wgpu, SIMD optimization,
and runs on any platform including WebAssembly.

**150,561 lines of Rust | 2,946 tests | 0 clippy warnings**

## Features

- **Pure Rust** -- Zero C/C++/Fortran dependencies. Safe, portable, auditable.
- **188 ONNX operators** -- Math, NN, Conv, Shape, Indexing, Comparison, RNN, Attention, ML; real-world detection models run, including YOLOv8 and YOLO11 (opset 11+)
- **GPU acceleration** -- wgpu compute shaders for MatMul, Softmax, ReLU, etc.
- **SIMD optimization** -- NEON (aarch64) and AVX2 (x86_64) for element-wise ops
- **Multi-dtype** -- f32, f16, bf16, i8, i32, i64 with automatic type promotion
- **INT8 quantization** -- Quantized MatMul with per-channel scale/zero-point
- **Mixed precision** -- f16 activations with f32 accumulation
- **Graph optimization** -- Constant folding, operator fusion, CSE, dead code elimination
- **Memory efficiency** -- Arena allocator, buffer pooling, strided tensor views
- **Streaming inference** -- Token-by-token generation for autoregressive models: `session.generate(prompt, GenerationConfig)` returns a `TokenStream` iterator that runs one forward pass per `next()`, feeds the model's `present.*` key/value outputs back in as the next step's `past.*` inputs, and stops on EOS, on a token cap, or on cancellation. Greedy (argmax) selection only -- temperature / top-k / top-p / beam search are deliberately out of scope; set `emit_logits` and sample outside the crate. No tokenizer: token ids in, token ids out
- **Async execution** -- Non-blocking inference via `Arc::clone(&session).run_async(inputs)`, which starts the model on a `std::thread` immediately and returns a `RunFuture`. Executor-agnostic (no async-runtime dependency at all): `.await` it under tokio/async-std/smol, or drive it with the crate's own dependency-free `block_on`. `spawn_run()` returns a blocking `RunHandle` for callers with no executor. The receiver is `Arc<Self>` because the worker thread outlives the call; thread-per-inference is the right tool for one long inference, not for many small concurrent ones
- **Cancellation** -- `SessionBuilder::with_session_cancellation(token)` makes every operator the model uses check a `CancellationToken` before it runs, so `run()` unwinds with `OnnxError::Cancelled` at the first node boundary after `token.cancel()` -- on the sequential path, the rayon parallel path, and inside `If`/`Loop`/`Scan` bodies. The token is **session-scoped**: cancelling stops every run in flight on that session. For per-request cancellation of a generation, use `GenerationConfig::with_cancellation`, which is checked between decode steps. Nodes claimed by a GPU execution provider are dispatched before the registry and are not cancellation points
- **Control flow** -- If/Loop/Scan operators with nested subgraph execution
- **ONNX local functions** -- `FunctionProto` bodies are inlined into the graph at load time, in both the eager and streaming parsers, so models built from reusable function definitions execute like any other graph
- **Rank-generic convolution** -- `Conv` / `ConvTranspose` support 1D/2D/3D spatial ranks through a shared N-D im2col path (`pads` uses the ONNX `[begin_0..begin_r-1, end_0..end_r-1]` layout, which for `r == 2` is exactly the classic `[top, left, bottom, right]` array); the 2D case keeps its dedicated fast path
- **Rank-0 (scalar) tensor support** -- `Tensor`/`TensorView` represent a true ONNX scalar (`shape: vec![]`, one element) instead of silently promoting every rank-reducing result to shape `[1]`; `Det` and the loss ops (`NegativeLogLikelihoodLoss`/`SoftmaxCrossEntropyLoss` with `reduction=mean|sum`) emit it end-to-end today, confirmed through shape resolution and output-slot allocation -- most other rank-reducing ops (e.g. an all-axes `Reduce*`) still promote to `[1]`, a known tracked gap
- **Opset-aware execution** -- `Softmax`/`LogSoftmax`/`Hardmax` branch on the model's declared `ai.onnx` opset (parsed from `opset_import`) instead of hardcoding opset-13+ semantics, so a pre-13 model gets the spec's default-axis-1-and-flatten-to-2D contract rather than the post-13 per-axis one
- **Einsum ellipsis and broadcasting** -- the equation parser handles numpy-compatible `...` tokens (e.g. `...ij,...jk->...ik` for broadcast batched matmul) with numpy's right-aligned broadcasting rule, and a label shared across operands broadcasts when one side's extent is `1`; large contractions lower to `matrixmultiply::sgemm` via greedy pairwise decomposition instead of a scalar loop nest
- **Model encryption** -- AES-GCM encrypted model files, keyed with CSPRNG-derived nonces
- **WebAssembly** -- Run in the browser via wasm-bindgen
- **no_std** -- Core types work without std (alloc only)
- **Session caching** -- `session.save_optimized(path)` writes the **post-optimization** graph (nodes, rewritten weight table, value-info, model metadata, nested subgraphs) in a version-tagged, length-prefixed pure-Rust binary format; `Session::load_optimized(path)` / `SessionBuilder::load_optimized(path)` rebuilds it at `OptLevel::None`, so constant folding, CSE, fusion and dead-node elimination do not run again (the test suite proves this by counting operator executions during load: exactly zero). The encoding is deterministic, so a cache file can be content-hashed; a truncated, foreign or wrong-version file is always a typed `OnnxError::Parse`. Runtime settings (threads, providers, profiling, memory pool) are **not** cached -- they come from the builder that loads it
- **Native dtype dispatch** -- `run_typed()` path executes 40+ operators natively (no f32 round-trip) via `TypedOpContext`; MatMul natively handles F32/F16/BF16/I8→I32/I32 dtypes
- **DirectML backend** -- Windows D3D12 execution provider (`directml` feature) with CPU fallback on other platforms; opt-in (`OXIONNX_DIRECTML=1` / `.with_directml(true)`), compile- and lint-verified for Windows and proven on Linux against a CPU oracle, but not yet executed on GPU hardware
- **Zero-copy output reuse** -- Operators write into pre-allocated output slots via `execute_into_slots`; a large subset of hot operators (elementwise, activations, normalization, reduce, pooling, shape, indexing, attention, conv, RNN) have hand-coded zero-copy slot-write kernels that avoid the intermediate copy and preserve pointer identity across inference runs with `IoBinding`. Operators without a hand-coded kernel fall back to a correct copy-based default (`execute()` then `copy_from_slice`)
- **Graph introspection** -- Enumerate a model's compute nodes (op type, inputs, outputs, attributes) via `Session::nodes()` / `NodeInfo`

## Status

| Crate | Status | Tests (pre-hardening-program baseline*) |
|-------|--------|-------|
| `oxionnx` (root) | Alpha | 613 passing |
| `oxionnx-core` | Stable | 36 passing |
| `oxionnx-ops` | Alpha | 624 passing |
| `oxionnx-proto` | Stable | 42 passing |
| `oxionnx-gpu` | Alpha | 17 passing |
| `oxionnx-cuda` | Partial | 10 passing (GEMM/elementwise/softmax via OxiCUDA; Conv still stubbed and not advertised as supported) |
| `oxionnx-directml` | Implemented (opt-in; GPU path not yet hardware-verified) | 242 tests, all Linux-executed or cross-target type-checked. Dual backend — DirectML operators + HLSL/D3D12 compute fallback — routing 15 ops: MatMul, Gemm, Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Softmax, ReduceSum, ReduceMean, ReduceMax, ReduceMin, Conv; kernels compile/lint-verified for Windows and proven on Linux vs a CPU oracle, but not yet run on GPU hardware |
| `oxionnx-coreml` | Alpha | 8 passing on non-Apple hosts (compiles to a stub); 26 passing + 7 skipped on macOS/iOS/tvOS/visionOS (predict/predict_raw/predict_features, compute-plan + model metadata) |

\* Per-crate figures predate the v0.1.5 hardening program (every crate gained tests during it, by an uneven amount) and do not sum to the total below; they are load-bearing only for relative crate maturity, not an absolute current count.

**Total: 2,946 tests passing on Linux (`cargo nextest run --workspace --all-features`; the exact count drifts between runs as the workspace evolves under active development). 0 clippy warnings on the host target. Platform-gated suites run only on their target OS — `oxionnx-coreml`'s Apple-only paths and `oxionnx-directml`'s Windows FFI tests are not included here — so no single machine runs the entire cross-platform set. 150,561 lines of Rust (126,388 excluding blanks/comments).**

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

OxiONNX implements 188 ONNX operators (plus 15 aliases: short-forms like `LayerNorm`/`RMSNorm`/`Silu`/`CeLU`, plus the 11-name `ai.onnx.ml.*` domain) -- 203 op-type strings resolve through the registry in total.

| Category | Count | Examples |
|----------|-------|---------|
| Math | 47 | MatMul, Gemm, Add, Mul, Pow, Sqrt, Reduce* (incl. L1/L2/LogSum/LogSumExp/SumSquare), Trig, ArgMax/Min, CumSum, TopK, BitShift, VariadicMin/Max/Mean/Sum, Det |
| Neural Network | 35 | Relu, Sigmoid, Softmax, LayerNorm, BatchNorm, GELU, SiLU, Mish, GroupNorm, InstanceNorm, RmsNorm, Hardmax, Shrink, NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss |
| Convolution / Pool | 15 | Conv, ConvTranspose (rank-generic: 1D/2D/3D), MaxPool, AveragePool, GlobalAvgPool, GlobalMaxPool, Pad, Resize, LRN, LpPool, GlobalLpPool, MaxUnpool, MaxRoiPool, Upsample, Col2Im |
| Shape | 15 | Reshape, Transpose, Concat, Slice, Split, Flatten, Tile, DepthToSpace, SpaceToDepth, ReverseSequence, Size, Expand, Squeeze, Unsqueeze, CenterCropPad |
| Indexing / Quant | 16 | Gather, GatherElements, GatherND, Scatter, ScatterND, Where, OneHot, Compress, Unique, QuantizeLinear, DequantizeLinear, QLinearConv, QLinearMatMul, MatMulInteger, ConvInteger, DynamicQuantizeLinear |
| Comparison / Logic | 26 | Equal, Greater, Less, And, Or, Not, Xor, Bitwise* (And/Or/Xor/Not), IsInf, IsNaN, NonZero, Cast, CastLike, Constant, Einsum, ConstantOfShape, EyeLike, Trilu, Identity, Shape, NonMaxSuppression |
| RNN / Attention | 8 | RNN, LSTM, GRU, Attention, MultiHeadAttention, RotaryEmbedding, GridSample, RoiAlign |
| DSP | 7 | DFT, STFT, HannWindow, HammingWindow, BlackmanWindow, MelWeightMatrix, Bernoulli |
| Control Flow | 3 | If, Loop, Scan |
| ONNX-ML | 11 | LinearClassifier, LinearRegressor, TreeEnsembleClassifier/Regressor, SVMClassifier/Regressor, Normalizer, Scaler, LabelEncoder, TfIdfVectorizer, StringNormalizer |
| Random / Generator | 5 | RandomNormal, RandomUniform, RandomNormalLike, RandomUniformLike, Multinomial |

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
| `coreml` | CoreML execution provider (Apple Silicon: macOS/iOS/tvOS/visionOS) |

## Architecture

```
oxionnx (root)           -- Session, optimizer, execution engine
  oxionnx-core           -- Tensor, DType, Graph, Operator trait, OnnxError
  oxionnx-ops            -- 188 operator implementations
  oxionnx-proto          -- Pure Rust ONNX protobuf parser
  oxionnx-gpu            -- wgpu compute backend (optional)
  oxionnx-cuda           -- CUDA dispatch layer via OxiCUDA (optional)
  oxionnx-directml       -- DirectML execution provider for Windows D3D12 (optional)
  oxionnx-coreml         -- CoreML execution provider for macOS/iOS/tvOS/visionOS (optional)
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

> **Comparison note**: OxiONNX prioritizes portability and safety: pure Rust with zero
> C/C++/Fortran dependencies, built on memory-safe pure-Rust crates. It is not `unsafe`-free —
> the ops crate confines `unsafe` to a few documented sites: the call into `matrixmultiply`
> (itself pure Rust) that backs MatMul/Conv in the default build, plus optional SIMD intrinsics
> compiled only under `--features simd`. In the default (non-`simd`) build `oxionnx-ops` is
> `#![deny(unsafe_code)]`, with those `matrixmultiply` call sites the only explicitly-allowed exceptions.
> For absolute peak throughput, C++ runtimes like onnxruntime (with MKL/cuDNN) will be faster
> on operations dominated by BLAS. OxiONNX targets use cases where pure Rust, WebAssembly
> compatibility, and zero native dependencies are more important than raw FLOPS.

## License

Apache-2.0

## Author

COOLJAPAN OU (Team Kitasan)
