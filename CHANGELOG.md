# Changelog

All notable changes to OxiONNX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[0.1.0]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.0
