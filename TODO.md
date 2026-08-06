# OxiONNX 0.1.5 -- Pure Rust ONNX Inference Engine

**Repository:** `cool-japan/oxionnx`
**License:** Apache-2.0
**Author:** COOLJAPAN OU (Team Kitasan)

**Current stats (2026-08-06):** ~126,388 SLoC (Rust code, tokei) | 190 OpKind variants | 188 registered operators (203 op-type strings incl. aliases) | 2,946 tests passing | workspace layout (8 crates)
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
- [x] Streaming / chunked inference for sequence models -- token-by-token generation via `session.generate(prompt, GenerationConfig)` (implemented v0.1.5 wave 2; see Async & Streaming below for the exact API)
- [x] Async execution API -- `Arc<Session>::run_async()` returning a `RunFuture` (+ `spawn_run()`/`block_on()`) (implemented v0.1.5 wave 2)
- [x] Session serialization -- `save_optimized()`/`load_optimized()`, version-tagged binary, reload at `OptLevel::None` (implemented v0.1.5 wave 2)
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
- [ ] Host-device transfer minimization -- keep tensors on GPU between consecutive GPU-capable nodes (never actually implemented: `GpuTensorTracker` promised this but nothing ever called `store`/`take`/`is_on_gpu` outside its own unit test; deleted as dead code in v0.1.5 wave 2 rather than left as a false claim. Real keep-on-GPU chaining needs `Tensor` to carry a device buffer, or the session executor to hold a residency map, plus buffer-taking variants of every `gpu_*` entry point -- none of which is a change local to `oxionnx-gpu`)
- [x] Tiled MatMul with shared memory for large dimensions
- [ ] WebGPU compatibility for wasm32 targets (honestly declined as of v0.1.5 wave 2: `GpuContext::try_new`/`try_new_async` return `None` on wasm32 at context-creation time, so the CPU path runs directly. Before this, a wasm32 context still uploaded inputs, encoded a pass, and called `queue.submit` for every node, then discarded the result because blocking `map_async` readback is impossible in the browser -- pure overhead that could never produce a value. Restoring real browser acceleration needs an `async` variant of every `gpu_*` entry point plus a `wasm-bindgen-futures` bridge, a public API split, not a local fix)
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

- [x] `async` inference API for non-blocking execution -- `Arc<Session>::run_async()` / `spawn_run()` / dependency-free `block_on()` (implemented v0.1.5 wave 2)
- [x] Streaming token generation API for autoregressive models (yield tokens as produced) -- `session.generate(prompt, GenerationConfig)` returns a `TokenStream`, feeding `present.*` back in as the next step's `past.*` (implemented v0.1.5 wave 2)
- [x] Cancellation token support -- `SessionBuilder::with_session_cancellation()` (session-scoped) and `GenerationConfig::with_cancellation()` (per-generation) (implemented v0.1.5 wave 2)

### Foreign Bindings

- [x] `wasm-bindgen` feature -- run in browser via WebAssembly
- [x] ~~Python bindings via PyO3~~ (out of scope: requires CPython C dependency; reference doc at /tmp/oxionnx-python-reference.md)
- [x] Pre-built binaries for Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x86_64) (CI workflow created)

---

## 11. crates.io Publish Preparation

- [x] Top-level README with:
  - [x] Feature overview and architecture diagram
  - [ ] Supported opset table (opset version, operator name, status) -- not in the README as a static table; `oxionnx::opset_coverage` exists as a runtime API instead (see Testing & Quality above). README ships an operator/category table without a per-opset breakdown
  - [ ] Performance comparison table (vs onnxruntime, tract, etc.) -- deliberately not shipped; replaced with an honest prose "Comparison note" plus `cargo bench` pointers, since no reproducible cross-engine numbers exist to publish (see the perf reports under Section 17 wave 2 for why: measurements swung 2-3x under concurrent-agent load and the A/B harnesses that produced clean numbers were throwaway)
  - [x] Quickstart code example (compile-checked against the current `Session`/`SessionBuilder`/`Tensor` API as of v0.1.5 wave 3)
- [x] Keywords: `onnx`, `inference`, `deep-learning`, `machine-learning`, `pure-rust` (root `Cargo.toml`; corrected wording -- previously listed as `neural-network` instead of `deep-learning`)
- [x] Categories: `science`, `algorithms` (root `Cargo.toml`; corrected count -- previously claimed a third `wasm` category that was never actually set on the root crate)
- [x] `cargo doc --no-deps` with zero warnings across all subcrates
- [x] All public items documented with doc comments
- [x] `cargo publish --dry-run` passes for every subcrate (oxionnx-core verified; others blocked only by unpublished deps)
- [x] Dependency audit -- ensure all deps are well-maintained and compatible
- [x] MSRV policy documented (minimum supported Rust version)
- [x] CHANGELOG.md maintained per release

---

## 12. Performance & Correctness Enhancements (v0.1.3)

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

## 14b. Subgraph Parsing + Dispatch Wiring (v0.1.4)

### Proto-layer subgraph parsing
- [x] `AttributeValue.g: Option<Box<GraphProto>>` — field to carry a parsed subgraph (types.rs; `Box` breaks the recursive type cycle)
- [x] `parse_attribute` field 6 — call `parse_graph(b)?` instead of discarding bytes (parser.rs:186-189); fixes both the eager and streaming parsers (streaming delegates to `parse_node`→`parse_attribute`)
- [x] `build_subgraph` in model.rs — converts `GraphProto` → runtime `Graph`; local initializers become synthesised `Constant` nodes prepended to `graph.nodes` (no `Graph` struct change)
- [x] `convert_attributes` graph arm — inserts `build_subgraph` result into `Attributes.graphs` before `attr_type` dispatch; mutual recursion handles nested subgraphs
- [x] Tests: `test_subgraph_attribute_parsed` (parser round-trip), `test_subgraph_attribute_wired_into_graph` (build_subgraph → Attributes.graphs)

### Session dispatch wiring
- [x] `OpContext.weights: Option<&'a HashMap<String, Tensor>>` — new field; model weights passed by reference so subgraph nodes resolve initializer names without cloning
- [x] `dispatch_node` (3 paths: slot-write, inplace, default) — `outer_scope: Some(state.as_map())`, `weights: Some(&self.weights)`, `registry: Some(&self.registry)`
- [x] `parallel.rs` rayon path — same wiring for `weights` and `registry`
- [x] `IfOp`/`LoopOp`/`ScanOp` — pass `ctx.weights.unwrap_or(&empty)` to `execute_subgraph` (zero allocation; outer-model initializers now reachable from inside branches/loops)

### End-to-end tests
- [x] `tests/control_flow_e2e.rs` — 9 tests: `If` true/false branch, `Loop` accumulate, `Loop` zero-iterations, `Loop` with outer weight, `Scan` element-wise, `Scan` with running state, `If` sequential subgraph ops, `If` subgraph-local initializer

---

## 15. Future Roadmap (Deferred)

### Phase D — Operator-Native TypedTensor Dispatch
> **Promoted to v0.1.5 — see Section 16 for tracking.**
Full multi-dtype dispatch at the operator layer (80+ ops × 13 dtypes). Currently all
inference runs through f32 internally; `run_typed` performs input→f32 and f32→output
conversions. Native dispatch would avoid these round-trips for pure-integer or f16 models.
- **Scope**: ~1040 monomorphizations; requires rearchitecting the `Operator` trait
- **Deferred reason**: `run_typed` f32 conversion covers 90% of practical use cases
- **Trigger**: user demand for lossless i64 tensors > 2^24 range without precision loss

### Phase E — DirectML Execution Provider (Windows)
> **Promoted to v0.1.5 — see Section 16 for tracking.**
Full implementation of DirectML (Direct Machine Learning, D3D12-based) execution provider
for Windows GPU acceleration.
- **Current state**: stub in `execution_providers.rs` (no-op, compiles)
- **Dependencies**: `windows` crate, `d3d12` bindings — Windows-only
- **Scope**: ~2 weeks of platform-specific work; separate initiative
- **Trigger**: Windows-first deployment requirements from downstream projects

### Phase F — Operator-Level IOBinding Reuse
> **Promoted to v0.1.5 — see Section 16 for tracking.**
Extend `IoBinding` to allow individual operators to write directly into pre-allocated output
buffers (skip the intermediate `HashMap<String, Tensor>` in `run_internal`). Requires
changing the `Operator::execute` return contract.

---

## 16. v0.1.5 — Phase D, E, F Promotion (In Progress)

### Phase D — Operator-Native TypedTensor Dispatch (pilot)
- [x] Infrastructure: `TypedOpContext`, `native_dtypes()`, `execute_typed()` hooks on `Operator` trait (all with backward-compat defaults) — `oxionnx-core/src/operator.rs`, `operator_typed.rs`, `operator_slots.rs`
- [x] Pilot: 40 operators declare `native_dtypes()` + `execute_typed()` (Add, Sub, Mul, Div, Neg, Sqrt, Relu, Sigmoid, Tanh, Gelu, Exp, Log, Abs, Erf, Identity, Cast, Reshape, + 23 more)
- [x] Typed arithmetic wired: `typed_add`, `typed_sub`, `typed_mul`, `typed_div`, `typed_relu`, `typed_sigmoid`, `typed_tanh`, `typed_gelu`, `typed_exp` from `typed_ops.rs` used in `execute_typed` bodies
- [x] Session `run_typed()`: initial pilot used an f32 fallback for every node, with per-node typed dispatch deferred to v0.1.6 -- **superseded by the v0.1.6 follow-up directly below**, which shipped; current `run_typed()` (`src/session/run/typed.rs`) carries `TypedTensor` intermediates and dispatches through `execute_typed` per-node whenever every present input's dtype is in that operator's `native_dtypes()`, falling back to a surgical f32 cast only for the inputs/ops that need it. Confirmed by reading the current source, not just this checklist
- [x] v0.1.6 follow-up: Rewrite `run_typed` to carry `TypedTensor` intermediates; per-node dispatch via `native_dtypes()` (skip f32 round-trip)
- [x] v0.1.6 follow-up: Native integer arithmetic in `typed_ops.rs` (Add/Sub/Mul/Div for I8/I16/I32/I64 without f32 intermediate)
- [x] v0.1.6 follow-up: Native dispatch for Conv, MatMul, Gemm (heavy hand-written kernels — conv.rs, rnn.rs, attention.rs)
  - [x] **v0.1.8 scoped slice: MatMul-only native typed dispatch** (planned 2026-04-18) — `native_dtypes()` for I8/I32/F16/BF16; `execute_typed` with INT8×INT8→I32 kernel, F16 and BF16 triple-loop kernels (f32 accumulator). No f32 round-trip. Precedent pattern for Gemm/Conv/Attention in v0.1.9+.
  - [x] **v0.1.9 scoped slice: GemmOp native typed dispatch** (planned 2026-04-18) — `native_dtypes()` for F32/F16/BF16/I8/I32; `execute_typed` with I8×I8→I32 kernel, I32×I32→I32, F16 and BF16 triple-loop kernels. Kernels in `math_typed.rs` (new module also housing v0.1.8 matmul typed kernels). No f32 round-trip.
  - [x] **v0.1.10 scoped slice: AttentionOp + MultiHeadAttentionOp native typed dispatch (F16/BF16)** (planned 2026-04-18) — `native_dtypes()` for F32/F16/BF16; `execute_typed` with f32-accumulator SDPA and MHA kernels (softmax in f32 for F16 numerical stability). Kernels in `attention/typed.rs` (new module). No f32 round-trip for Q/K/V. Prerequisite splitrs on attention.rs also shipped in v0.1.10.
  - [x] **v0.1.10+ scoped slice: ConvOp + ConvTransposeOp native typed dispatch (F16/BF16)** — `native_dtypes()` for F32/F16/BF16; `execute_typed` with cast-compute-cast kernels (f32 accumulator). New module `conv_typed.rs`. No f32 round-trip for input/weight.
  - [x] **v0.1.10+ scoped slice: LSTMOp + GRUOp native typed dispatch (F16/BF16)** — `native_dtypes()` for F32/F16/BF16; `execute_typed` with cast-compute-cast kernels. New module `rnn_typed.rs`. Multi-output (3 outputs for LSTM, 2 for GRU). No f32 round-trip.

### Phase E — DirectML Execution Provider (Windows) — COMPLETE (Wave 3 + Wave 4)

> **Reconciled 2026-07-11.** The three redundant `/stub-check` tracking sections below (Wave 3
> roadmap, "Stubs to implement" 2026-06-12, "Stubs to implement" 2026-06-22) all described the
> *same* work — turning the `oxionnx-directml` scaffold into a real execution provider. That work
> is now done. The 19 duplicate unchecked items are collapsed here.
>
> **Verification honesty (load-bearing):** there is no Windows host and no D3D12 GPU in this
> environment, so the GPU code path **cannot be executed here** and is **not hardware-verified**.
> What *is* verified: every Windows FFI line is compile- and lint-clean via
> `cargo clippy --target x86_64-pc-windows-gnu` (the crate had never even compiled for Windows
> before — `context.rs` imported `CreateEventW` without the `Win32_Security` feature), and every
> shader/operator *algorithm* is proven on Linux against a CPU oracle (`reference.rs`). Whether the
> D3D12 barrier sequences, descriptor bindings, and fence waits produce correct results on real
> silicon is settled only by `examples/directml_self_check.rs` + `OXIONNX_DIRECTML_VERIFY=1` on a
> Windows box. **Activation is opt-in** (`OXIONNX_DIRECTML=1` / `.with_directml(true)`, default OFF)
> precisely because a GPU kernel bug returns plausible-but-wrong numbers rather than crashing.

- [x] New subcrate `oxionnx-directml` with cross-platform shim (non-Windows: `try_new() -> None`) and Windows `#[cfg(target_os = "windows")]` D3D12 context skeleton
- [x] Feature flag `directml` wired: root `Cargo.toml`, session field (`Session::dml`), session init, dispatch block (CUDA → DirectML → wgpu → CPU priority order)
- [x] `DirectMLExecutionProvider` preserved as ort-compat no-op (with-feature: real factory path)
- [x] **Dual backend, DML-first:** shared D3D12 device/queue/fence/event; genuine `IDMLDevice` operators when `DirectML.dll` is present (resolved via `LoadLibraryW`/`GetProcAddress` so its absence falls back rather than failing process launch), else an HLSL/D3D12 compute engine (`D3DCompile` at runtime), else CPU
- [x] **Platform-neutral core, Linux-tested:** `plan.rs`/`layout.rs` do all shape validation, `u32` range-checking, dispatch-grid + DML descriptor layout math; `reference.rs` is a shader-faithful CPU oracle; `hlsl.rs` holds the shader sources
- [x] `context.rs`: real D3D12 device context (DXGI adapter enumeration skipping WARP unless `OXIONNX_DIRECTML_ALLOW_WARP=1`, feature-level 12_0→11_0 probe, compute queue, fence + `CreateEventW`), all COM state behind a `Mutex` with a documented `unsafe impl Send` so `Session: Sync` holds on Windows (pinned by `assert_send_sync::<Session>()`)
- [x] Compile & bind the MatMul HLSL compute shader — 2-D×2-D only; the transposed dispatch-grid doc-comment bug fixed and pinned by `hlsl_grid_is_not_transposed`
- [x] Compile & bind element-wise HLSL shaders (Add, Sub, Mul, Div, Relu, Sigmoid, Tanh)
- [x] Extend dispatch to Conv, Softmax, Reduce{Sum,Mean,Max,Min} — Softmax/Reduce on both HLSL and DML paths; Conv on the genuine-DML path only (`DML_CONVOLUTION`, cross-correlation mode to match ONNX), HLSL declines Conv to CPU
- [x] Observability: `Ok(None)`=declined vs `Err`=failed no longer conflated (killed the double `.ok()`-swallow); `OXIONNX_DIRECTML_VERIFY=1` shadow-compares every GPU result against the oracle; `OXIONNX_DIRECTML_STRICT=1` turns silent CPU-fallback into a hard error
- [x] End-to-end Windows CI job (windows-latest build+test) + a Linux→Windows cross-clippy job (incl. the `Session: Sync` gate) added inside `.github/workflows.disabled/` (CI stays disabled per user policy)
- [x] `is_supported_op` routes exactly the 15 claimed ops (`MatMul, Gemm, Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Softmax, ReduceSum, ReduceMean, ReduceMax, ReduceMin, Conv`); kept in lockstep with `dispatch::route` by test
- [x] 242 crate tests (all Linux-executed or cross-target type-checked); `examples/directml_self_check.rs` shipped as the hardware acceptance gate

### Phase F — Operator-Level IOBinding Reuse (pilot)
- [x] `execute_into_slots()` + `supports_output_slots()` hooks on `Operator` trait (backward-compat defaults)
- [x] Dispatch path wired in `execute_node_with_inplace` (sequential path): if op supports slots AND static output shape known → pre-allocate from pool + write via `execute_into_slots`
- [x] Pilot: 40 operators implement `execute_into_slots` (same set as Phase D pilot)
- [x] `IoBinding` helpers: `take_output_buffer` / `put_output_buffer` for future pointer-identity guarantee
- [x] `SizeClassPool::acquire()` called on Phase F path (pool is now acquire + release, not drain-only)
- [x] v0.1.6 follow-up: `SessionRunState` with pool-aware insert/release — wraps `HashMap<String, Tensor>` with `SizeClassPool` integration; pointer-identity for IoBinding outputs via `take_output_buffer`/`put_output_buffer`. (Slot-indexed Vec variant deferred to F.11 in Proposed follow-ups.)
- [x] v0.1.6 follow-up: Phase F pilot for parallel rayon branch (currently uses CPU-allocate path only)
- [x] v0.1.6 follow-up: Phase F for remaining 107 operators (all non-pilot operators still use default copy path)

---

## Cross-References

- **Operator roadmap:** [oxionnx-ops/](oxionnx-ops/)
- **GPU backend roadmap:** [oxionnx-gpu/](oxionnx-gpu/)

---

## Pure-Rust enhancements (planned 2026-06-08)

- [x] ONNX Reshape `allowzero` (opset 14+) (planned 2026-06-08)
  - **Goal:** Reshape honors the `allowzero` attribute; allowzero=0 (default) keeps `0`→copy-input-dim; allowzero=1 treats `0` as a literal zero-size dim (NumPy) and rejects the ambiguous `0`+`-1` combination.
  - **Design:** read `allowzero` (i64, default 0) in ReshapeOp; thread a bool into the shape-resolution helper in `shape/basic.rs`; literal-0 + a typed error on `0`&`-1`; preserve the default path exactly.
  - **Files:** oxionnx-ops/src/registry/shape_ops/reshape_ops.rs, oxionnx-ops/src/shape/basic.rs
  - **Tests:** allowzero=0 copy (regression), allowzero=1 literal-zero, allowzero=1 `0`+`-1` error, `-1` inference unchanged, allowzero=1 no-zero ≡ default.
  - **Risk:** `-1`×`0` interaction — covered by typed error + tests.
- [x] `TrainingInfo.initialization_graph` (ONNX training IR) (planned 2026-06-08)
  - **Goal:** parse and expose `TrainingInfoProto.initialization` (protobuf field 1), currently skipped with a "rarely used" comment.
  - **Design:** add `pub initialization_graph: Option<GraphProto>`; parse field 1 via the existing `parse_graph` helper in `parse_training_info`.
  - **Files:** oxionnx-proto/src/types.rs, oxionnx-proto/src/parser.rs
  - **Tests:** synthetic TrainingInfoProto with an init graph → `Some`; absence → `None`.
  - **Risk:** minimal — mirrors the existing `training_graph` parse.
- [x] wasm `console_error_panic_hook` (planned 2026-06-08)
  - **Goal:** wasm panics forward to the browser console instead of the current no-op in `wasm_init()`.
  - **Design:** optional workspace dependency gated by the existing `wasm` feature (`dep:console_error_panic_hook`); call `set_once()` in `wasm_init()`; native default build untouched.
  - **Files:** Cargo.toml, src/wasm.rs
  - **Verify:** wasm32-unknown-unknown compile-check + native clippy clean; Pure-Rust default closure preserved.
  - **Risk:** none on the native path; in-browser runtime test is out of the macOS-gate scope.

---

## Proposed follow-ups (deferred from v0.1.6 run, 2026-04-17)

- **D.3 — Native dispatch for Conv / Attention / RNN** (**COMPLETE**): MatMul (v0.1.8), Gemm (v0.1.9), AttentionOp+MHA (v0.1.10), ConvOp+ConvTransposeOp+LSTMOp+GRUOp (v0.1.10+), Attention SIMD/NEON/AVX2 (v0.1.10+) — all shipped. No remaining items.
## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check) — SUPERSEDED

> These three items (context, MATMUL dispatch, elementwise shaders) were duplicates of the
> DirectML Wave 3 work now marked complete under **Phase E** above. Closed 2026-07-11.

- **E.4 — DirectML real HLSL compilation** (**COMPLETE**): `D3DCompile` runtime path implemented in `backend/d3d12/shader.rs`; compile-verified for Windows via cross-target clippy. Runtime correctness needs a Windows GPU (`self_check`).
- **E.5 — Windows DirectML CI job** (**COMPLETE**, stays disabled): windows-latest build+test job + Linux→Windows cross-clippy job added inside `.github/workflows.disabled/`. Not re-enabled — CI remains off per user policy.
- **E.6 — DirectML Conv/Softmax/Reduce kernels** (**COMPLETE**): shipped Wave 4 — Softmax/Reduce on both HLSL and DML paths, Conv on the genuine-DML path (`DML_CONVOLUTION`).
- **E.7 — DirectML Q4/Q8 support** (deferred): quantized DirectML kernels remain future work; the f32 `run()` path is the only surface DirectML sees today. Genuinely blocked on hardware access to validate.
- **F.11 — Slot-indexed Vec<Tensor> SessionRunState** (v0.1.8): Replace the HashMap backing with a `Vec<Tensor>` + name→index lookup table. Would save the HashMap hash computation per node (~121 ops per run). Requires a mirror change in `TypedSessionRunState` and care around `bound_outputs` ownership. Defer until a real-world workload shows HashMap cost as material.
- **F.12 — Zero-copy per-op `execute_into_slots` for non-pilot hot ops** (**COMPLETE**): 22 hot ops shipped hand-coded slot-write bodies in v0.1.6. v0.1.7 added 27 more: shape_ops (Squeeze/Unsqueeze/Flatten/Expand/Split/Tile/DepthToSpace/SpaceToDepth/ReverseSequence), nn_ops (Clip/LeakyRelu/PRelu/HardSigmoid/Celu/Elu/Selu/ThresholdedRelu/LpNorm/MeanVarianceNorm/Hardmax/Shrink), conv_ops (MaxPool/AveragePool/GlobalAveragePool/GlobalMaxPool/Pad/Resize). v0.1.8 added 3 more: Gather, ScatterND, ScatterElements — subtotal 52. v0.1.9 added 2 more: Conv, ConvTranspose — subtotal 54. v0.1.10 added 2 more: LSTM, GRU — subtotal 56. v0.1.4 added 2 more: AttentionOp, MultiHeadAttentionOp — **total 58 of 121 ops** with hand-coded slot-write bodies. Phase F operator slot-write sweep fully closed. (The "121" denominator is the registry size as of v0.1.4; it has since grown to 188 as of the v0.1.5 hardening program — see Section 17 — and slot-write coverage for those newer ops has not been re-audited against this count.)
- **F.13 — F.12 remaining complex ops** (**COMPLETE**, v0.1.4): AttentionOp and MultiHeadAttentionOp shipped hand-coded `execute_into_slots` bodies (`registry/rnn_ops/attention.rs`). Backed by new allocation-free kernels `sdpa_into`, `sdpa_output_shape`, `reshape_from_heads_into`, `multi_head_attention_into` extracted from `attention/core.rs`. SIMD zeroing bug for `seq_q > 1` fixed as part of this work. 21 new tests in `oxionnx-ops/tests/output_slots_attention_test.rs`.
- **F.14 — Remaining 47 ops slot-write sweep** (**COMPLETE**, v0.1.4): Added true zero-copy `execute_into_slots` bodies for 47 additional operators, bringing the total to **105 of 121 ops** with hand-coded slot-write bodies. Covered: normalization ops (LayerNorm, GroupNorm, BatchNorm, RmsNorm, InstanceNorm) and activations (Softmax, LogSoftmax) via new `_into` kernel variants in `nn/normalization.rs`; reduce ops (ReduceSum/Mean/Max/Min/Prod/L1/L2/LogSum/LogSumExp/SumSquare) via new `reduce_with_into`/`reduce_output_shape` primitives in `math/reduce.rs`; ArgMax, ArgMin, CumSum, TopK (2-output) via new `_into` variants in `math/argminmax.rs` and `math/topk.rs`; variadic ops (Min/Max/Mean/Sum); comparison binary ops (Equal/Greater/GreaterOrEqual/Less/LessOrEqual/And/Or/Xor) via macro update; bitwise binary ops (BitwiseAnd/Or/Xor) via macro update; unary ops (Not, IsInf, IsNaN, BitwiseNot) with inline; shape/utility ops (Shape, Size, Constant, ConstantOfShape, EyeLike, Trilu, Einsum). 41 new tests in `oxionnx-ops/tests/output_slots_f14_test.rs`. Remaining 16 ops without slot bodies are variable-output (NonZero, Range, Compress, Unique, NonMaxSuppression, GatherND, GatherElements, Where), type-sensitive (QuantizeLinear, DequantizeLinear, OneHot), ML ops (11 LinearClassifier/TreeEnsemble/SVM/etc.), and spatial ops (RotaryEmbedding/GridSample/RoiAlign).

## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check) — SUPERSEDED

> All six DirectML items below (context acquisition, MATMUL, and the ADD/MUL/RELU/SIGMOID HLSL
> shaders) were the same Wave 3 work now complete under **Phase E** above. The note's premise held
> up exactly: the dispatch/device-acquisition code was implementable cross-platform behind the
> Windows cfg gates and is compile-verified there; end-to-end GPU correctness still needs a Windows
> D3D12 device (`examples/directml_self_check.rs`). Closed 2026-07-11.

---

## 17. Production-Grade Hardening Program (2026-08-05, v0.1.5)

> 12-lens exhaustive audit (spec conformance ×3, engine, proto robustness, panics, GPU, CUDA/CoreML,
> stubs, API/release, performance, test gaps) produced **232 findings**. Executed as 3 waves of
> parallel subagent implementation with file-ownership partitioning.

### Wave 1 — Critical/High correctness (163 findings, 16 domains) — COMPLETE (2026-08-05)
- [x] A proto-parser: checked pos+len everywhere, recursion depth limit, alloc clamping, correct TensorProto field numbers/encodings, group/unpacked-field handling (eager + streaming consistent)
- [x] B proto-model: dtype-aware weight decode (no silent zero-fill), STRINGS attrs wired, external-data sandbox + offset validation, dims validation
- [x] C shape-ops: Slice negative/steps/sentinels, Pad axes/negative/wrap, checked axis normalization across Concat/Split/Flatten/Unsqueeze/Transpose/Reshape, Split zero-size chunks
- [x] D indexing-quant: Where real broadcast, Scatter reduction attr + bounds + negative idx, Gather consistency, Quantize/Dequantize per-axis + dtype-derived saturation
- [x] E conv-pool: auto_pad everywhere, ceil_mode/dilations in real kernels, ConvTranspose output_shape, checked out-shape math
- [x] F resize: real cubic, nearest_mode variants, coordinate_transformation_modes, antialias, no silent fallbacks
- [x] G activations-misc: Gelu approximate, Clip opset-6 attrs, Cast truncate+saturate, Mod floored, Bitwise value-preserving, Reduce noop_with_empty_axes, ArgMax select_last_index, Shape start/end, Equal exact, Dropout mask, GroupNorm per-group scale
- [x] H ml-ops: TreeEnsemble cycle guard + NaN routing + MIN/MAX aggregates, TfIdf ngram_counts + batching, SVM Platt scaling, ML 1-D input shape, LabelEncoder
- [x] I rnn-attention: GRU linear_before_reset default fix, LSTM/GRU clip + layout, SDPA mask broadcast, Attention is_causal, GridSample string attrs, RoiAlign coord mode
- [x] J control-flow-dsp: Loop scan-output stacking, Scan output axes/directions, DFT axis, STFT errors, NMS max_boxes=0
- [x] K optimizer: fusion soundness (multi-consumer/graph-output guards), CSE non-commutative + fingerprint, Conv+Clip input bounds, constant-fold guards, OptLevel gating
- [x] L session-run: unknown-op = typed error (no silent skip), parallel-path outer_scope/mixed-precision, capture lifetimes, shape-resolution race, run_typed weight clone
- [x] M gpu-dispatch: decline-to-CPU gating (batch matmul, softmax axis, reduce keepdims, elementwise shapes, fused conv), unified support lists
- [x] N gpu-backend: 65535-workgroup clamping, buffer-size limits, error scopes -> CPU fallback, LeakyRelu alpha, readback timeout
- [x] O cuda: reduce >256 fix, batch matmul, softmax axis, attrs, Gemm bias broadcast, OXIONNX_CUDA_VERIFY shadow gate
- [x] P core-hardening: topo-sort underflow, no_std repair, CSPRNG nonces, `Tensor::try_new` (unconditional data/shape validation, including release builds; `Tensor::new` itself is unchanged and still `debug_assert`-only, by design, for callers who can guarantee the invariant statically), deny.toml

### Wave 2 — Hard features + performance — COMPLETE (2026-08-06)

> Stitch wave (between W1/W2) landed: STRINGS/TENSORS/GRAPHS attr wiring, streaming fallible decode,
> static-0 dims, external-data base_path through subgraphs, no_std repair for oxionnx-core,
> opset plumbing (OpContext carries model opset; pre-13 Softmax/Hardmax), shape-inference/kernel
> consistency (auto_pad/ceil_mode/output_shape/keep_aspect_ratio), Pad registry rewiring,
> GridSample align_corners, GRU clip/layout wiring, Gelu simd cross-path parity.
- [x] Register implemented-but-unregistered ops (QLinear* family, RNN); OpKind wiring
- [x] Conv1D/Conv3D real support (`conv::conv`/`conv::conv_transpose` are now rank-generic N-D entry points; the 2D fast path is unchanged and still the common case)
- [x] Opset-version plumbing into OpContext (pre-13 Softmax/Hardmax semantics)
- [x] run_async / cancellation / streaming: reconcile claims vs implementation
- [x] Einsum ellipsis + performance
- [x] Missing-op batch: LRN, CastLike, Upsample, LpPool, GlobalLpPool, MaxUnpool, MaxRoiPool, Col2Im, BitShift, Random*/Multinomial
- [x] CPU perf package: small-M matmul via sgemm, attention/flash parallelism + sgemm, KV-cache in-place append, broadcast without materialization, conv threading, winograd filter cache, stride-walk transpose/reduce, hashbrown maps, run-loop clone elimination
- [x] GPU perf: adapter limits, softmax workgroup reduction, pool byte budget, conv buffer reuse; wasm32 readback
- [x] Rank-0 scalar representation design fix

### Wave 3 — Tests, docs, release polish — COMPLETE (2026-08-06)
- [x] Test gaps: axes-as-input forms, IoBinding pointer identity, TreeEnsemble/SVM modes, typed dispatch, rank-0 broadcast, SIMD-vs-scalar equivalence, negative/panic-safety tests
- [x] `#[non_exhaustive]` on public error enums -- landed on `OnnxError` (`oxionnx-core`), `CoreMLError`, `CudaError`, the DirectML error type, `oxionnx-proto`'s reader-error type, and the execution-provider enum (`src/execution_providers.rs`)
- [ ] `missing_docs` warnings -- `#![warn(missing_docs)]` landed on `oxionnx-cuda`, `oxionnx-directml`, `oxionnx-coreml`; not yet confirmed on `oxionnx-core`/`oxionnx-ops`/`oxionnx-proto`/`oxionnx-gpu`/root
- [x] README/CHANGELOG/TODO reconciliation (claims == reality) -- this pass (T8-docs-release)
- [x] Tracing on load path -- `src/session/loading.rs` carries `tracing::debug_span!`/`info_span!` for parse/build/optimize/shape-inference plus `debug!`/`info!` events
- [x] Final gates: full nextest, clippy -D warnings, cargo doc, cargo deny check bans, fmt

### Program result (2026-08-06) — COMPLETE

Final gates: **2946 workspace tests / 2946 passed** (15 skipped: hardware-gated + perf probes),
`clippy --all-features --all-targets -D warnings` clean, `cargo fmt --check` clean,
`cargo deny check bans` ok, `cargo doc --no-deps` zero warnings.
Two extra fix waves beyond the plan (final-fixes + micro-close) closed every bug the Wave-3
test authors pinned (DepthToSpace/SpaceToDepth blocksize guards incl. slot paths + negative-attr
boundary, Cast unknown-dtype, RNN/LSTM/GRU direction validation, ArgMax/Reduce axis validation,
Einsum slot-shape masking, PROBIT precision, TreeEnsemble SOFTMAX_ZERO ordering, SVM kernel/
post_transform enum validation + Platt-mode label selection, load_mmap metadata, typed integer
exactness for Neg/Ceil/Floor/Round/Sign/Abs incl. a Round tie-detection epsilon bug, MHA typed
out_shape broadcast, rank-0 full-reduce completion CPU/GPU/DirectML-consistent).

Known documented micro-residuals (deliberate, low-impact): `is_full_reduction` tolerates
out-of-range axes (unreachable — every reduce entry point now validates first); LogOp
`default_typed_via_f32` F32-tagging vs ExpOp's dtype-preserving cast (cosmetic inconsistency);
missing_docs backlog (250–592 undocumented public items per crate — measured, deferred);
one stale comment pointer in shape/spatial.rs.

## 18. Post-Hardening GPU/async_run Crash-Hang Fix (2026-08-06) — COMPLETE

Found only once a real Vulkan adapter was actually reachable in the dev sandbox (the Wave-3 gates
above ran with no adapter present, so this class of bug was invisible to them): `run_async`/
`spawn_run` worker threads could end up holding the session's last `Arc`, running `GpuContext`'s
`Drop` (a live `wgpu::Device`/`Queue`/`Instance`) as the worker's last act before the OS thread
exited. NVIDIA's Vulkan ICD shares thread-affine state with its EGL/GLSI core for at least some
teardown paths; destroying on a different thread than the one that created it produced a `SIGSEGV`
inside the driver, sometimes while holding a global driver mutex the process's own `exit()` path
then blocked on forever (presenting as a hang, not a crash). A second, independent shape of the
same root cause: even with creation/destruction pinned to one thread, process exit could still race
a still-in-flight teardown against the driver's own `atexit`-registered `dlclose()`.
- [x] `src/session/async_run.rs`: `Shared::_session_keepalive` — an independent `Arc<Session>`
      clone held by `Shared` itself, so a worker thread's own capture is (almost) never the last
      reference standing
- [x] `src/session/gpu_owner.rs` (new): routes **both** `GpuContext` creation and destruction
      through one dedicated, process-lifetime thread, so a given context's creation and destruction
      are always on the same thread as each other (a "destruction-only" dedicated thread was tried
      first and made things categorically worse — 30/30 reproducible `SIGSEGV` — because it turned
      an occasional creator/destroyer mismatch into a guaranteed one; documented in-code so it is
      not retried)
- [x] `atexit`-registered quiescence hook (LIFO-ordered ahead of the driver's own handler), with a
      debounced re-check (`ACTIVE_WORKERS` + `IN_FLIGHT`, not a single-instant sample) closing a
      TOCTOU gap the first, undebounced version had
- [x] `oxionnx-gpu/src/context/types.rs`: `impl Drop for GpuContext` polls the device before its
      fields' automatic drops
- [x] Verification: the 3 originally-hanging tests at 90/90 and 150/150 clean across repeated
      stress runs; full `oxionnx` crate suite at 1014/1014 with the fix in, versus 1012–1013/1014
      before it (residual native-driver crashes/hangs, non-deterministic across which specific test
      in the same module they landed on)
- [x] `tests/w2_session_cache.rs::loading_a_cache_is_measurably_cheaper_than_optimizing_from_scratch`
      stabilized against scheduler jitter (batched `RUNS`-per-round timing + a real margin instead
      of a bare `<` — see the sibling defense already in `tests/w2_cancellation.rs`) — this did
      **not** fix the deeper issue logged as item 19 below, which surfaced afterward once a real
      adapter was consistently present for this test too
- [ ] Not yet investigated: whether this same cross-thread teardown hazard reaches `oxionnx-cuda`'s
      `CudaContext`/`oxicuda-driver` path the same way `oxionnx-gpu`'s `GpuContext` did — that stack
      is opt-in gated (`OXIONNX_CUDA=1`) and off in every gate run so far, so it has never been
      exercised against a real device long enough to say either way

## 19. Deferred — GPU Context Caching/Pooling Across Sessions

Discovered while re-diagnosing item 18's timing test with a real adapter finally in place:
`Session::build_from_graph` (the function both `Session::from_graph` and
`Session::from_optimized_bytes` funnel through) unconditionally calls `gpu_owner::try_new()` —
a full `wgpu::Device`/`Queue`/`Instance` acquisition plus rebuilding 20+ cached compute pipelines —
on **every single session construction**, whether built fresh or loaded from a cache, with no
reuse or sharing of a `GpuContext` across `Session` instances. This is not new in v0.1.5; it was
already true of the original `crate::gpu::GpuContext::try_new()` call site before item 18's
`gpu_owner` indirection existed. It went unnoticed because every prior gate run either had no
adapter reachable (fast `None` path) or didn't happen to time repeated session construction under
`--all-features`.

Confirmed causally (not just suspected) by rerunning the cache-timing test with `--no-default-features`:
build 23.4ms / load 2.5ms, 9.3x speed-up — versus build 1.70s / load 1.78s, 0.95x with `gpu`
enabled and a real adapter present, i.e. GPU context acquisition cost dominates and is paid equally
by both the "build" and "load" paths, erasing the actual signal either is meant to measure.

Likely a genuine production cost too, not just a test artifact: any application that constructs
more than one `Session` with the `gpu` feature on pays this full device+pipeline-rebuild cost per
session, unconditionally.

- [ ] Decide the caching unit: process-wide singleton `GpuContext` shared (`Arc`) across every
      `Session`, vs. a bounded pool keyed by adapter/feature requirements, vs. explicit
      opt-in reuse via a builder option — needs a decision on whether two `Session`s with
      different runtime knobs (mixed-precision, memory pool, etc.) can safely share one
      `GpuContext`, since pipelines are currently built assuming exclusive ownership
      by whichever `Session` created them
- [ ] If sharing an `Arc<GpuContext>`, `gpu_owner`'s one-context-per-owner-thread-round-trip
      model needs revisiting — reference counting means "destroy" is no longer 1:1 with
      "the `Session` that requested creation," which item 18's create/destroy-same-thread
      invariant assumed
- [ ] Re-benchmark `tests/w2_session_cache.rs`'s timing test (and any other session-construction
      timing assumptions elsewhere in the suite) once pooling lands — the current stabilized
      version (item 18) has a loose enough margin to survive either outcome, but should be
      revisited to assert something meaningful again once GPU context creation is no longer the
      dominant cost
