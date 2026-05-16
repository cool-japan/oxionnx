# Changelog

All notable changes to OxiONNX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.3] - 2026-05-16

### Added

- feat: `ProviderKind::DirectMl` variant (behind `directml` feature) — `ProviderKind` enum now covers CPU, GPU (wgpu), CUDA, and DirectML
- feat: `SessionBuilder::with_provider_kinds()` — non-trivial ordered execution provider list that routes dispatch at runtime; CPU is always the implicit terminal fallback
- feat: `SessionBuilder::provider_kinds()` accessor — returns the configured provider list for introspection
- feat: `Session::provider_kinds()` accessor — returns the provider list stored in the session
- feat: provider-list dispatch loop in `run_sequential_inner` — when `providers` is non-empty, each node is attempted through the list in order; falls back to legacy heuristic path when list is empty (backward compatible)
- feat: `try_provider_list_dispatch` helper — encapsulates CUDA / DirectML / GPU / CPU provider dispatch for the provider-list path; all providers return `None`-graceful results with CPU fallback guaranteed
- `ProviderKind` re-exported from `oxionnx` crate root

## [0.1.2] - 2026-04-19

### Added
- Phase D — Operator-Native TypedTensor Dispatch: `native_dtypes()` + `execute_typed()` opt-in hooks on the `Operator` trait; 40 pilot operators (23 math, 14 NN, Identity, Cast, Reshape) execute natively without f32 round-trips; all other operators fall back transparently via a correct default implementation
- Phase E — DirectML Execution Provider (Windows): new `oxionnx-directml` subcrate with `DirectMLContext::try_new()` cross-platform shim; Windows D3D12 context skeleton; HLSL compute shader scaffolds for MatMul, Add, Mul, Relu, Sigmoid; feature-gated behind `directml`; CPU fallback on non-Windows and unsupported ops
- Phase F — Operator-Level IOBinding Reuse: `supports_output_slots()` + `execute_into_slots()` opt-in hooks on the `Operator` trait; `SizeClassPool::acquire()` integrated on the slot path; pilot implementations for 40 operators; `IoBinding::take_output_buffer` / `put_output_buffer` helpers for caller-owned buffer management
- `OnnxError::DTypeMismatch` — new error variant for dtype validation in the typed dispatch path
- `TypedOpContext<'a>` — parallel context struct for typed operator dispatch
- Phase F complete — F.10 remainder sweep: all 121 operators opt into `supports_output_slots = true`; full slot-write path with pointer-identity pool reuse across the entire operator set
- Phase F.12 — zero-copy `execute_into_slots` hand-coded bodies for 30 operators: 22 pilot ops (Phase D/F pilot), plus 9 shape ops (Squeeze/Unsqueeze/Flatten/Expand/Split/Tile/DepthToSpace/SpaceToDepth/ReverseSequence), 12 elementwise NN ops (Clip/LeakyRelu/PRelu/HardSigmoid/Celu/Elu/Selu/ThresholdedRelu/LpNorm/MeanVarianceNorm/Hardmax/Shrink), 6 conv/pool ops (MaxPool/AveragePool/GlobalAveragePool/GlobalMaxPool/Pad/Resize), and Gather/ScatterND/ScatterElements
- D.3 scoped slice — `MatMulOp::execute_typed` with native dispatch for F32, F16, BF16, I8→I32, I32 dtypes; `native_dtypes()` declared; INT8×INT8→I32 triple-loop GEMM kernel; F16/BF16 kernels with f32 accumulator — no round-trip through f32 for quantized or half-precision matmul
- `oxionnx-ops/src/conv.rs` refactored into six focused submodules (`conv2d`, `im2col`, `winograd`, `pooling`, `tests`) to clear headroom for D.3 Conv native typed kernels (v0.1.9+)
- Integration test `full_slot_coverage_smoke.rs`: pointer-identity across 100 iterations + pool reuse assertions for the slot-write infrastructure

### Improved
- 100 new tests across typed dispatch, IOBinding, DirectML EP, and session slot paths (1074 total)
- Operator trait is fully backward-compatible: all four new methods carry correct default implementations

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

[0.1.3]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.3
[0.1.2]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.0
