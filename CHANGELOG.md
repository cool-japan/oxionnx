# Changelog

All notable changes to OxiONNX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-04-14

### Added
- RNN operators: GRU, LSTM, RNN, and sequence processing ops (`oxionnx-ops/src/rnn.rs`)
- Extended ML operators: SVM, tree ensemble, decision tree classifiers/regressors (`oxionnx-ops/src/ml_svm.rs`, `ml_tree.rs`)
- Attention operators: expanded multi-head attention and transformer support (`oxionnx-ops/src/attention.rs`)
- Indexing operators: additional gather/scatter/slice variants (`oxionnx-ops/src/indexing.rs`)
- Quantized inference ops: extended quantized operator coverage (`oxionnx-ops/src/quantized.rs`)
- Spatial operators: additional convolution and pooling variants (`oxionnx-ops/src/spatial.rs`)

### Improved
- Fusion optimizer: major expansion of graph fusion rules and pattern matching (`src/optimizer/fusion.rs`)
- Session API: improved execution flow and error handling (`src/session.rs`)
- GPU compute context: refined WebGPU context management and kernel dispatch (`oxionnx-gpu/src/context.rs`)
- GPU shaders: additional WGSL shader kernels (`oxionnx-gpu/src/shaders.rs`)
- CUDA backend: conv, elementwise, and quantized op refinements (`oxionnx-cuda/src/conv.rs`, `elementwise.rs`)
- Proto streaming parser and mmap loader improvements (`oxionnx-proto/src/streaming_parser.rs`, `mmap_loader.rs`)
- SIMD ops: extended SIMD-accelerated operator paths (`oxionnx-ops/src/simd_ops.rs`)
- Math and NN ops: expanded coverage (`oxionnx-ops/src/math.rs`, `nn.rs`)

## [0.1.0] - 2026-03-26

### Added
- Initial release of OxiONNX pure Rust ONNX inference engine
- 147 ONNX operator implementations
- GPU acceleration via wgpu (MatMul, Softmax, ReLU, Sigmoid, GELU, Reduce)
- SIMD optimization (NEON aarch64, AVX2 x86_64)
- Graph optimization: constant folding, operator fusion, CSE, dead code elimination
- Multi-dtype support (DType enum, TypedTensor, TensorStorage)
- INT8 quantized MatMul with per-channel scale/zero-point
- Mixed-precision inference (f16 activations, f32 accumulation)
- Control flow operators: If, Loop, Scan with nested subgraphs
- Streaming token generation for autoregressive models
- Async inference API (run_async)
- Session serialization (save/load pre-optimized graphs)
- Memory-mapped weight loading (mmap feature)
- Tensor arena allocator with buffer pooling
- Strided tensor views (zero-copy transpose, slice, squeeze)
- Broadcasting iterator (zero-allocation)
- AES-GCM model encryption (encryption feature)
- WebAssembly support via wasm-bindgen
- no_std support for oxionnx-core
- Execution provider abstraction with CPU/GPU fallback chain
- ONNX-ML operators: Linear, TreeEnsemble, SVM, Normalizer, Scaler, LabelEncoder, TfIdf
- Model metadata extraction API
- Model validation and schema checking
- Graph diff utility for debugging optimizations
- Model pruning (strip unused weights)
- Symbolic shape propagation
- Opset coverage report generator
- Benchmark-based CPU/GPU path selection
- Numerical tolerance validation
- CI pipeline (GitHub Actions)
- Property-based testing with proptest
- Fuzz testing for protobuf parser
- 595 tests, 0 clippy warnings

[0.1.1]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.0
