# OxiONNX 0.1.1 -- Pure Rust ONNX Inference Engine

**Repository:** `cool-japan/oxionnx`
**License:** Apache-2.0
**Author:** COOLJAPAN OU (Team Kitasan)

**Current stats (2026-04-14):** ~47,829 SLoC (Rust code) | 167 OpKind variants | 1,023 tests passing | workspace layout (6 crates)
**Dependencies:** `half`, `matrixmultiply`, `bytemuck`, `rayon` (non-wasm), `tracing`, optional `wgpu`/`pollster` (gpu feature), optional `oxicuda-*` (cuda feature)
**Zero C/C++ dependencies.**

---

## 1. Architecture -- Workspace Migration & Plugin System

- [x] Migrate to Cargo workspace with subcrates:
  - [x] `oxionnx-core` -- `OnnxError`, `Tensor`, dtype enums, trait definitions
  - [x] `oxionnx-ops` -- all operator implementations (depends on `oxionnx-core`)
  - [x] `oxionnx-gpu` -- wgpu compute backend (depends on `oxionnx-core`)
  - [x] `oxionnx-proto` -- protobuf parser and ONNX model structures
  - [x] Root `oxionnx` crate re-exports everything; feature flags gate subcrates
- [x] Define `Operator` trait with `fn eval(&self, inputs: &[&Tensor], attrs: &Attributes) -> Result<Vec<Tensor>, OnnxError>`
- [x] Build trait-based operator registry (`HashMap<String, Box<dyn Operator>>`)
- [x] Allow user-supplied custom operators via registry insertion before session run
- [x] Version each subcrate under workspace inheritance (`*.workspace = true`)
- [x] Enforce `#![deny(unsafe_code)]` in `oxionnx-ops` and `oxionnx-proto`

---

## 2. Session & Inference Engine (`session.rs`)

### Graph Optimization Passes

- [x] Constant folding -- evaluate subgraphs with all-constant inputs at load time
- [x] Dead node elimination -- prune nodes whose outputs are never consumed
- [x] Operator fusion:
  - [x] Conv + BatchNormalization fusion
  - [x] MatMul + Add fusion (linear layer bias folding into single `sgemm` call with `beta=1`)
  - [x] Conv + Relu / Conv + Clip fusion
  - [x] Consecutive Transpose cancellation
  - [x] LayerNorm fusion -- replace the 7-op `mean->sub->pow->mean->add->sqrt->div->mul->add` pattern with a single fused kernel
- [x] Shape inference pre-pass -- resolve static shapes before execution
- [x] Common subexpression elimination

### Memory Planning

- [x] Static memory planner -- compute lifetime intervals for every intermediate tensor
- [x] Buffer reuse -- assign non-overlapping lifetimes to the same allocation
- [x] Tensor arena allocator -- single pre-allocated block, bump-pointer sub-allocation
- [x] Peak memory estimation API (`session.estimated_memory_bytes()`)
- [x] Activation memory pool -- pre-allocate `Vec<f32>` buffers, hand out and return instead of `Vec::new()` per activation

### Execution

- [x] Multi-threaded execution -- identify independent branches in the DAG; execute in parallel with rayon
- [x] Topological-level parallelism -- group nodes by topological depth; run each level in parallel
- [x] Streaming / chunked inference for sequence models (process in fixed-length windows)
- [x] Async execution API -- `session.run_async()` returning a future
- [x] Session serialization -- save pre-optimized graph to disk; reload without re-optimization
- [x] Execution provider abstraction (CPU, GPU, future backends) with fallback chain
- [x] In-place element-wise ops -- `Add`, `Mul`, `ReLU`, `GELU` when output shape == input shape and input has no other consumers

---

## 3. Tensor Module (`tensor.rs`)

### Multi-dtype Support

- [x] Promote dtype to a first-class enum: `F32`, `F16`, `BF16`, `F64`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`, `Bool`, `String`
- [x] Store tensor data as `enum TensorStorage { F32(Vec<f32>), F16(Vec<half::f16>), ... }` or type-erased byte buffer with dtype tag
- [x] Implement per-dtype dispatch for all element-wise operations
- [x] Automatic type promotion rules (e.g., `I8 + F32 -> F32`)
- [x] Mixed-precision inference support (F16 compute with F32 accumulation)
- [x] f16 runtime inference path -- store activations as `half::f16`, execute element-wise ops in f16; only promote to f32 for matmul accumulation
- [x] INT8 quantized MatMul -- `Tensor<i8>` with per-channel `scale` + `zero_point`

### Strided Views & Zero-Copy

- [x] Strided tensor view -- `TensorView { data: &[u8], shape, strides, offset, dtype }`
- [x] Zero-copy `transpose()`, `slice()`, `squeeze()`, `unsqueeze()` via stride manipulation
- [x] Contiguous check and lazy materialization (only copy when needed)
- [x] Broadcasting iterator that walks strides without allocating expanded tensors

### Performance

- [x] SIMD-accelerated element-wise ops (add, mul, relu, etc.) using `std::simd` or manual intrinsics behind feature flag
- [x] Memory-mapped tensor storage -- `mmap` large weight files, access on demand
- [x] Tensor arena allocator integration (shared with session memory planner)
- [x] Batch-aware tensor layout (NCHW vs NHWC) with layout conversion utilities

---

## 4. Protobuf Parser (`proto.rs`)

- [x] Streaming parser for multi-GB ONNX models (avoid loading entire protobuf into memory)
- [x] ONNX external data support -- weights stored in separate `.bin` files with relative path resolution
- [x] Opset version negotiation -- read `opset_import`, map to supported op versions, report unsupported ops before execution
- [x] ONNX-ML operator support:
  - [x] TreeEnsembleClassifier / TreeEnsembleRegressor
  - [x] SVMClassifier / SVMRegressor
  - [x] LinearClassifier / LinearRegressor
  - [x] Normalizer, Scaler, LabelEncoder
- [x] Model metadata extraction API (`model.metadata_props`, `model.doc_string`, producer info)
- [x] ONNX training info parsing (for fine-tuning use cases)
- [x] Validation of parsed model against ONNX spec constraints

---

## 5. Graph Module (`graph.rs`)

### Control Flow & Subgraphs

- [x] Subgraph support for `If` operator (then_branch / else_branch attributes)
- [x] Subgraph support for `Loop` operator (body attribute with loop-carried dependencies)
- [x] Subgraph support for `Scan` operator (sequential processing with state)
- [x] Proper scope resolution -- subgraph access to outer-scope values
- [x] Nested subgraph support (subgraphs within subgraphs)

### Shape Inference

- [x] Static shape inference pass at graph construction time
- [x] Dynamic / symbolic shape propagation (e.g., `batch_size` remains symbolic until runtime)
- [x] Shape inference for all 75+ implemented operators (partial - significantly expanded)
- [x] Shape error reporting with node name and operator context

### Visualization & Debugging

- [x] Export graph to DOT format for Graphviz rendering
- [x] Node-level execution time profiling (attach timing info per node)
- [x] Graph diff utility -- compare two graphs for debugging optimization passes
- [x] Operator schema validation -- check each node's inputs/outputs against ONNX operator spec

---

## 6. Model Loading (`model.rs`)

- [x] Lazy weight loading via `mmap` -- do not read weight data until first inference
- [x] Model encryption / decryption -- AES-GCM encrypted model files, key provided at load time (pure Rust via `aes-gcm` crate)
- [x] Model format versioning -- detect ONNX IR version, warn on unsupported versions
- [x] ONNX model zoo compatibility testing:
  - [x] ResNet-18 / ResNet-50 (test harness + synthetic test created)
  - [x] MobileNetV2 (test harness ready, download script provided)
  - [x] BERT-tiny / BERT-base (synthetic BERT-tiny test passing)
  - [x] GPT-2 (117M) (synthetic GPT-2 test passing)
  - [x] YOLOv8-nano (test harness ready)
  - [x] Whisper-tiny (test harness ready)
- [x] Model pruning -- strip unused initializers and nodes at load time
- [x] Model size reporting API (parameter count, weight bytes, graph node count)

---

## 7. Operators -- Completed & Pending

### Completed

- [x] RMSNorm
- [x] ReduceSum, ReduceMax, ReduceMin, ReduceProd
- [x] Split
- [x] ArgMax, ArgMin
- [x] SiLU, Swish
- [x] HardSigmoid, HardSwish
- [x] CumSum
- [x] Range
- [x] Tile
- [x] QuantizeLinear, DequantizeLinear
- [x] ScatterElements, ScatterND
- [x] GlobalAveragePool, GlobalMaxPool
- [x] TopK
- [x] Proper Cast implementation
- [x] Typed error enum (`OnnxError`)
- [x] Edition fix (`edition = "2021"`)

### Pending Operators

See [oxionnx-ops/](oxionnx-ops/) for operator implementations.

High-priority operators not yet tracked above:

- [x] Attention / MultiHeadAttention (ONNX contrib / custom fused op)
- [x] GroupNormalization (opset 18+)
- [x] RotaryEmbedding (for transformer models)
- [x] Einsum
- [x] GRU, LSTM, RNN (recurrent operators)
- [x] NonMaxSuppression (object detection post-processing)
- [x] RoiAlign (object detection)
- [x] GridSample
- [x] ConvTranspose (1D, 2D, 3D)
- [x] DepthToSpace, SpaceToDepth
- [x] Compress
- [x] Unique
- [x] StringNormalizer, TfIdfVectorizer (text operators)
- [x] DFT (opset 17)
- [x] STFT (opset 17)
- [x] HannWindow (opset 17)
- [x] HammingWindow (opset 17)
- [x] BlackmanWindow (opset 17)
- [x] MelWeightMatrix (opset 17)
- [x] Bernoulli (opset 15)
- [x] ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare
- [x] BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot (opset 18)
- [x] Size, Hardmax, Shrink

---

## 8. GPU Backend (`gpu/`)

See [oxionnx-gpu/](oxionnx-gpu/) for the GPU backend.

Summary of key items:

- [x] Shader library -- WGSL compute shaders for MatMul, Conv2D, element-wise ops, Softmax, Reduction, Attention
- [x] Automatic CPU-to-GPU fallback when an operator has no GPU kernel
- [x] GPU memory pool -- reuse `wgpu::Buffer` allocations across inference calls
- [x] Host-device transfer minimization -- keep tensors on GPU between consecutive GPU-capable nodes
- [x] Tiled MatMul with shared memory for large dimensions
- [x] WebGPU compatibility for wasm32 targets
- [x] Benchmark GPU vs CPU paths for each operator; auto-select fastest

---

## 9. Testing & Quality

### Unit & Integration Tests

- [x] Integration test suite in `/tests/` -- build minimal valid ONNX protobuf bytes in test code and verify end-to-end `Session::run()`
- [x] Per-operator unit tests with reference values from ONNX runtime or NumPy
- [x] Batch dimension tests (batch=1, batch=N, dynamic batch)
- [x] Edge case tests (empty tensors, scalar inputs, very large shapes)
- [x] Property-based testing with `proptest` for tensor operations (associativity, commutativity, broadcast correctness)
- [x] Fuzz testing for protobuf parser (malformed `.onnx` files)
- [x] Dilated + grouped Conv2D tests -- `dilation > 1` and `group > 1` correctness

### Conformance & Model Tests

- [x] ONNX backend test suite integration (node-level conformance)
- [x] BERT-tiny end-to-end inference correctness test (synthetic architecture, determinism + profiling)
- [x] ResNet-18 end-to-end inference correctness test (synthetic architecture with residual blocks)
- [x] GPT-2 (117M) end-to-end inference correctness test (synthetic transformer + LM head)
- [x] Numerical tolerance validation (max absolute error, relative error thresholds)
- [x] Opset version coverage report (which ops at which opset versions pass)

### Benchmarks & CI

- [x] Benchmark suite with `criterion` -- per-operator microbenchmarks
- [x] End-to-end model inference latency benchmarks
- [x] Memory usage benchmarks (peak RSS tracking)
- [x] CI pipeline (GitHub Actions):
  - [x] `cargo test` on Linux, macOS, Windows
  - [x] `cargo test --features gpu` with software renderer
  - [x] `cargo clippy -- -D warnings`
  - [x] `cargo doc --no-deps` zero warnings
  - [x] `cargo fmt --check`
  - [x] Benchmark regression detection
- [x] Code coverage tracking (tarpaulin or llvm-cov)

---

## 10. API & Ergonomics

### Rust API

- [x] Builder pattern for `Session` configuration:
  ```rust
  Session::builder()
      .with_optimization_level(OptLevel::All)
      .with_threads(4)
      .with_execution_provider(ExecutionProvider::Cpu)
      .load("model.onnx")?
  ```
- [x] Input/output shape introspection API -- `input_info()` / `output_info()` return `TensorInfo` with dtype, static shape, and symbolic `dim_params`
- [x] Symbolic shape API -- `TensorInfo::symbolic_shape()` returns `Vec<Dim>` where `Dim` is `Static(usize)` / `Symbol(String)` / `Unknown`
- [x] Model metadata API -- `Session::metadata()` returns `ModelMetadata` with `producer_name`, `producer_version`, `domain`, `ir_version`, `opset_imports`, `custom_metadata`
- [x] Multi-dtype inference API -- `Session::run_typed()` accepts/returns `TypedTensor` (i64, f16, bf16, i32, bool, …) via internal f32 conversion
- [x] `inputs_typed!` macro for ergonomic multi-dtype input construction
- [x] IOBinding -- `IoBinding` / `Session::run_with_binding()` for zero-allocation repeated inference (output buffers reused via `copy_from_slice` when shape matches)
- [x] Model profiling API -- per-node execution times, memory usage breakdown
- [x] Typed inference API with compile-time shape checking (advanced, optional)
- [x] `no_std` support for `oxionnx-core` (alloc only, no file I/O)
- [x] Opset version validation -- warn (not error) when loaded model's opset > tested range

### Async & Streaming

- [x] `async` inference API for non-blocking execution
- [x] Streaming token generation API for autoregressive models (yield tokens as produced)
- [x] Cancellation token support for long-running inference

### Foreign Bindings

- [x] `wasm-bindgen` feature -- run in browser via WebAssembly
- [x] ~~Python bindings via PyO3~~ (out of scope: requires CPython C dependency; reference doc at /tmp/oxionnx-python-reference.md)
- [x] Pre-built binaries for Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x86_64) (CI workflow created)

---

## 11. crates.io Publish Preparation

- [x] Top-level README with:
  - [x] Feature overview and architecture diagram
  - [x] Supported opset table (opset version, operator name, status)
  - [x] Performance comparison table (vs onnxruntime, tract, etc.)
  - [x] Quickstart code example
- [x] Keywords: `onnx`, `inference`, `machine-learning`, `neural-network`, `pure-rust`
- [x] Categories: `science`, `algorithms`, `wasm`
- [x] `cargo doc --no-deps` with zero warnings across all subcrates
- [x] All public items documented with doc comments
- [x] `cargo publish --dry-run` passes for every subcrate (oxionnx-core verified; others blocked only by unpublished deps)
- [x] Dependency audit -- ensure all deps are well-maintained and compatible
- [x] MSRV policy documented (minimum supported Rust version)
- [x] CHANGELOG.md maintained per release

---

## 12. Performance & Correctness Enhancements (v0.1.2)

### RNN Operator Completeness
- [x] Fix `sequence_lens` parameter -- variable-length sequences now correctly masked per batch element
- [x] LSTM peephole connections (W_ci, W_co, W_cf weight tensors for gates)
- [x] Per-gate activation function selection (`activations` attribute: Sigmoid, Tanh, Relu)
- [x] Comprehensive LSTM/GRU unit tests (14 tests: bidirectional, variable-length, peephole, custom activations)

### Flash Attention (Pure Rust)
- [x] Block-wise tiled attention computation with O(1) extra memory per block
- [x] Causal masking support (lower-triangular, with early-exit on future blocks)
- [x] Online softmax (numerically stable, single-pass with running max/sum)
- [x] Configurable block sizes (Br, Bc) for tuning across hardware
- [x] Multi-head flash attention with head split/merge
- [x] Short-sequence fallback to standard SDPA
- [x] 16 tests (small match, causal, batch, stability at 256 tokens, block boundaries)

### Multi-Query / Grouped-Query Attention
- [x] Multi-Query Attention (MQA) -- single K/V head shared across Q heads via modular indexing
- [x] Grouped-Query Attention (GQA) -- K/V head groups with configurable group size
- [x] ALiBi positional bias support (geometric slopes, absolute distance bias)
- [x] Comprehensive attention operator tests (22 tests: SDPA, MHA, MQA, GQA, ALiBi, causal, edge cases, numerical stability)

### SIMD Horizontal Reductions
- [x] SIMD `reduce_sum_f32` (NEON vaddvq_f32, AVX2 horizontal add, Kahan-compensated scalar)
- [x] SIMD `reduce_max_f32` / `reduce_min_f32` (NEON vmaxvq/vminvq, AVX2 horizontal max/min)
- [x] SIMD `dot_product_f32` (NEON vfmaq_f32, AVX2 _mm256_fmadd_ps)
- [x] SIMD `reduce_mean_f32`
- [x] Integration with ReduceSum/ReduceMax/ReduceMin/ReduceMean operator dispatch (#[cfg(feature = "simd")])
- [x] 28 tests covering known values, large arrays, edge cases, NaN/Inf boundaries

### Unique Operator Axis Mode
- [x] Implement `axis` parameter for `Unique` operator (per-slice uniqueness along axis)
- [x] Support sorted/unsorted modes, negative axis, 3D tensors
- [x] 11 tests (rows, columns, sorted, negative axis, all-same, all-distinct, 3D, error cases)

### Optimizer Fusion Enhancements
- [x] Conv + Add + ReLU fusion (ResNet block shortcut pattern → ConvAddRelu fused op)
- [x] Gather + Gather composition (compose indices for consecutive gathers on same axis)
- [x] Softmax + Dropout elimination (inference-mode dropout is identity)
- [x] Transpose + Reshape simplification (identity transpose before reshape, flatten patterns)
- [x] 19 tests across all fusion patterns

### Quantization Enhancements
- [x] Asymmetric quantization (non-zero `zero_point`, per-tensor and per-channel)
- [x] QLinearConv -- fully quantized INT8 convolution with im2col, per-channel scales, grouped support
- [x] Dynamic quantization (runtime scale/zero_point computation, uint8 range)
- [x] Optimized fully_quantized_matmul with precomputed row/column sums for zero_point correction
- [x] 17 tests (asymmetric roundtrip, QLinearConv 1×1/3×3/grouped/per-channel, dynamic quant)

### Mixed Precision Auto-Conversion
- [x] f16-safe operator classification (40+ ops: activations, norm, softmax, shape, attention)
- [x] f32-required operator classification (20+ ops: MatMul, Gemm, Conv, reductions, Pow/Exp/Log)
- [x] Native f16 execution for element-wise ops (Relu, Add, Mul, Sub, Sigmoid, Tanh, Neg, Abs)
- [x] f16 precision rounding for f16-safe ops without native paths
- [x] Session run integration with profiling (ops marked with "(f16)" suffix)
- [x] 36 tests (classification, f16 ops, broadcasting, end-to-end, profiling)

---

## 13. Performance & Architecture Enhancements (v0.1.3)

### KV Cache for Autoregressive Inference
- [x] Add `past_key_values: Option<&[(Tensor, Tensor)]>` parameter to SDPA, MHA, Flash Attention
- [x] Incremental KV cache: concatenate new K/V with cached past along sequence dimension
- [x] Cache-aware attention: compute Q against full (past+current) K/V
- [x] KV cache management: ring buffer for long sequences exceeding max_seq_len
- [x] Integration with session streaming API for token-by-token generation
- [x] Benchmarks: GPT-2 generation latency with/without KV cache

### Softmax / LayerNorm SIMD Acceleration
- [x] SIMD-accelerated softmax inner loop (exp vectorization, NEON/AVX2)
- [x] SIMD-accelerated LayerNorm (vectorized mean, variance, normalize)
- [x] SIMD-accelerated GroupNorm (reuse LayerNorm SIMD path)
- [x] Integration with `#[cfg(feature = "simd")]` dispatch in nn.rs

### GPU Shader Expansion
- [x] LayerNorm WGSL compute shader
- [x] BatchNorm WGSL compute shader
- [x] Transpose WGSL compute shader
- [x] ReduceMean WGSL compute shader
- [x] GPU dispatch table expansion in session/gpu_dispatch.rs

### Conv2D Cache-Blocked im2col + Winograd F(2,3)
- [x] Cache-blocked im2col: process output in spatial tiles for L1 cache locality
- [x] Winograd F(2,3) transform for 3×3 stride=1 dilation=1 convolutions (2.25× fewer multiplications)
- [x] Auto-select: Winograd for 3×3 s1d1, cache-blocked im2col otherwise

### Memory Pool Improvements
- [x] Enable memory pool by default in SessionBuilder
- [x] Size-class bucketing: tiny (<512B), small (<4KB), medium (<256KB), large
- [x] Fragmentation metric: track wasted bytes, trigger compaction above threshold
- [x] Pool statistics API: alloc count, reuse count, peak usage, fragmentation ratio

### Robustness & Configuration
- [x] Replace all .expect()/.unwrap() in production code with proper Result chaining
- [x] Thread pool: implement per-session rayon::ThreadPool via ThreadPoolBuilder
- [x] Thread count configuration: honor with_intra_threads(N) for op-internal parallelism

---

## 14. Performance & Runtime Enhancements (v0.1.4)

### SIMD Elementwise Expansion + Batched MatMul Parallelism
- [x] SIMD `simd_sub` / `simd_div` -- NEON/AVX2/scalar paths for subtraction and division
- [x] SIMD `simd_neg` / `simd_abs` / `simd_sqrt` / `simd_log` -- unary SIMD kernels
- [x] Integrate new SIMD ops into elementwise dispatch in math.rs (Sub, Div, Neg, Abs, Sqrt, Log)
- [x] Parallelize batched MatMul with rayon (batch dim ≥ 4 → par_iter over batch slices)
- [x] SIMD fast path for small MatMul (M×K < 64): avoid function call overhead, inline multiply-accumulate
- [x] Tests: 20+ covering SIMD correctness, batched matmul parallelism, edge cases

### KV Cache Robustness + Error Propagation
- [x] Eliminate all unwrap()/expect() calls in kv_cache.rs (25+ instances) with proper Result chaining
- [x] Add KvCacheError variant to OnnxError for cache-specific failures (bounds, shape, head count)
- [x] Fix unwrap chains in cached_attention and cached_flash_attention (attention.rs, flash.rs)
- [x] Add cache overflow recovery: auto-evict oldest entries when ring buffer wraps
- [x] Tests: verify all 13 existing KV cache tests still pass + 6 new error-path tests

### Dynamic Shape Runtime Resolution
- [x] Resolve symbolic dimensions at runtime from actual input tensor shapes (batch_size, seq_len)
- [x] Support varying batch sizes between consecutive `session.run()` calls
- [x] Shape validation: check input shapes against model's expected symbolic layout
- [x] Automatic intermediate tensor re-planning when input shapes change (lazy replanning)
- [x] Tests: 12+ (dynamic batch, dynamic seq_len, shape mismatch errors, re-planning)

### Critical-Path Operator Scheduling
- [x] Graph cost model: estimate per-op cost from output volume × op-type weight table
- [x] Critical-path scheduling: longest-remaining-path first within each topological level
- [x] Priority queue execution: schedule ready-ops by estimated cost (heaviest first for CPUs)
- [x] Integration with existing topological-depth parallelism in session/run.rs
- [x] Tests: 10+ (cost estimation, scheduling order, correctness, regression)

### SIMD-Accelerated im2col + Conv2D Pack
- [x] Vectorize im2col data packing loop (NEON vld1q/vst1q, AVX2 _mm256_load/_mm256_store)
- [x] Pack weight matrix for cache-friendly GEMM access pattern (row-major → panel layout)
- [x] Stride-aware SIMD im2col for common stride=1 case (sequential memory → SIMD copy)
- [x] Tests: 8+ (SIMD im2col correctness, pack/unpack roundtrip, performance regression)

### Execution Provider Operator-Level Routing
- [x] OperatorPlacement trait: per-op GPU/CPU routing decision at runtime
- [x] Size-threshold auto-placement: GPU for large MatMul/Conv (output > 64KB), CPU for small ops
- [x] Provider fallback chain: GPU → CPU with automatic data transfer management
- [x] Session builder API: `.with_op_placement(OpPlacement::Auto)` / `.with_op_placement(OpPlacement::Manual(map))`
- [x] Tests: 8+ (auto-placement, manual override, fallback, threshold tuning)

---

## 15. Future Roadmap (Deferred)

### Phase D — Operator-Native TypedTensor Dispatch
Full multi-dtype dispatch at the operator layer (80+ ops × 13 dtypes). Currently all
inference runs through f32 internally; `run_typed` performs input→f32 and f32→output
conversions. Native dispatch would avoid these round-trips for pure-integer or f16 models.
- **Scope**: ~1040 monomorphizations; requires rearchitecting the `Operator` trait
- **Deferred reason**: `run_typed` f32 conversion covers 90% of practical use cases
- **Trigger**: user demand for lossless i64 tensors > 2^24 range without precision loss

### Phase E — DirectML Execution Provider (Windows)
Full implementation of DirectML (Direct Machine Learning, D3D12-based) execution provider
for Windows GPU acceleration.
- **Current state**: stub in `execution_providers.rs` (no-op, compiles)
- **Dependencies**: `windows` crate, `d3d12` bindings — Windows-only
- **Scope**: ~2 weeks of platform-specific work; separate initiative
- **Trigger**: Windows-first deployment requirements from downstream projects

### Phase F — Operator-Level IOBinding Reuse
Extend `IoBinding` to allow individual operators to write directly into pre-allocated output
buffers (skip the intermediate `HashMap<String, Tensor>` in `run_internal`). Requires
changing the `Operator::execute` return contract.

---

## Cross-References

- **Operator roadmap:** [oxionnx-ops/](oxionnx-ops/)
- **GPU backend roadmap:** [oxionnx-gpu/](oxionnx-gpu/)
