# OxiONNX 0.1.8 -- Pure Rust ONNX Inference Engine

**Repository:** `cool-japan/oxionnx`
**License:** Apache-2.0
**Author:** COOLJAPAN OU (Team Kitasan)

**Current stats (2026-08-14):** ~159,672 SLoC (Rust code, tokei) | 190 OpKind variants | 189 registered operators (204 op-type strings incl. aliases) | 3,595 tests passing (all-features; 3,289 with default features) | workspace layout (8 crates)
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
- [~] Host-device transfer minimization -- **weights: done (session-lifetime); activations: done (run-scoped), shipped v0.1.6.** A graph's initializers (conv `W`/`B`, `Gemm` `B`/`C`) are uploaded once per session and bound from a device-side residency cache on every later dispatch (`oxionnx-gpu/src/context/resident.rs`, keyed by initializer name from `src/session/gpu_dispatch.rs`) -- 502.7 MB of InSwapper-128 weights that used to cross the bus every frame now cross it once. Activations: `src/session/gpu_activations.rs`'s `RunActivations` now lets a GPU node's output stay in its device buffer for the next GPU consumer to bind in place, instead of a read-back + re-upload at every node boundary -- toggle via `Session::activation_residency_enabled()`/`set_activation_residency()`. Scope is deliberately narrower than weight residency, which is why the checkbox stays partial: it lasts only for the one `run()` call rather than across calls, a name is excluded entirely unless *every* consumer can bind it resident (`op_accepts_resident_slot`), and values captured into an `If`/`Loop` subgraph are always excluded since the subgraph body runs on the CPU. `GpuTensorTracker`, which promised the whole thing and was called by nothing, was deleted as dead code in v0.1.5 wave 2 -- its replacement is actually wired into the dispatcher and covered by 5 end-to-end tests (`src/session/tests/gpu_activation.rs`).
- [x] Tiled MatMul with shared memory for large dimensions
- [~] WebGPU compatibility for wasm32 targets -- **context + kernels: done; browser (`WasmSession`) wiring and in-browser verification: not started.** Honestly declined as of v0.1.5 wave 2: `GpuContext::try_new`/`try_new_async` returned `None` on wasm32 at context-creation time, so the CPU path ran directly -- before that, a wasm32 context still uploaded inputs, encoded a pass, and called `queue.submit` for every node, then discarded the result because blocking `map_async` readback is impossible in the browser, pure overhead that could never produce a value. Fixed in v0.1.6 for `try_new_async` specifically (the synchronous `try_new` still returns `None` on wasm32 by design): it now acquires a real `wgpu::Backends::BROWSER_WEBGPU` adapter, every kernel's read-back is an `async fn` awaiting a genuine `map_async` promise (`device_guard::read_back_web`, `#[cfg(target_arch = "wasm32")]`-gated), and `Session::enable_gpu_async()`/`Session::run_gpu_async()` are the entry points that drive them. Proven only on native so far -- `pollster::block_on`-driven tests (`src/session/run/sequential_async.rs`, `src/session/gpu_dispatch_tests.rs`, `src/session/tests/gpu_{activation,f16}.rs`) exercise the async entry points and run loop, but on native `enable_gpu_async` takes the blocking `try_new` branch and read-back goes through `read_back_blocking`, not `BROWSER_WEBGPU` acquisition or `read_back_web` -- so those two browser-specific pieces have zero runtime coverage anywhere. `src/wasm.rs`'s `WasmSession` (what JS callers actually use) has no GPU/async-GPU references at all and still calls the synchronous `Session::run`, and there is no `wasm-bindgen-test` anywhere in the repo (`scripts/check_wasm.sh` only builds the CPU-only `wasm` feature) -- so wiring `WasmSession` to the async path and validating it in a real browser remains open work
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

- [x] `wasm-bindgen` feature -- run in browser via WebAssembly. `cargo check`/`build --target wasm32-unknown-unknown --features wasm` compiled since v0.1.4, but CPU inference actually *worked* at runtime only from v0.1.6: `oxifft`'s default `threading` feature was a `compile_error!` on wasm32 (worked around by `oxionnx-ops` splitting it per-target), and `std::time::Instant`/`SystemTime` panic unconditionally on that target at the first timed load/dispatch call (fixed via `web_time`-backed `time_compat` modules in `oxionnx` and `oxionnx-ops`). `run_async`/`spawn_run`/`block_on` (thread-per-inference) remain native-only -- wasm32-unknown-unknown cannot spawn OS threads -- and are now a compile-time-absent API on that target rather than a runtime panic; call `Session::run` synchronously in the browser. `+simd128` (matrixmultiply's `v128` `sgemm` kernel) measured ~3-4.3x over scalar. `gpu` + `wasm` together now build: `Session`'s `Send + Sync` compile-time assertion (`src/session/mod.rs`) is wasm32-exempt, and an async WebGPU execution path exists (`Session::enable_gpu_async`/`run_gpu_async`, `GpuContext::try_new_async` acquiring a `wgpu::Backends::BROWSER_WEBGPU` adapter, kernels awaiting a real `map_async` read-back) -- proven on native targets only so far (see the WebGPU-compatibility entry under Section 8, GPU Backend, above for the native/browser coverage split); `WasmSession` (`src/wasm.rs`) does not yet call any of it and still runs every GPU-eligible node on the CPU
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

## 16. v0.1.5 — Phase D, E, F Promotion — COMPLETE

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

## 20. v0.1.6 — GPU Residency, Async WebGPU, InstanceNorm Fusion (2026-08-11) — COMPLETE

Shipped work for this release, summarized from `CHANGELOG.md`'s `## [0.1.6]` section (the
authoritative source — see there for full detail).

- [x] `oxionnx-gpu/src/context/resident.rs`: `ResidentBuffers` — session-lifetime GPU weight
      residency; `Conv`'s `W`/`B` and `Gemm`'s `B`/`C` initializers now upload once per session
      instead of once per dispatch (502.7 MB/frame → once, InSwapper-128)
- [x] `src/session/gpu_activations.rs`: `RunActivations` (new) — run-scoped GPU activation
      residency; a GPU node's output can now stay device-resident for the next GPU consumer
      instead of a read-back + re-upload at every node boundary; toggle via
      `Session::set_activation_residency()`
- [x] `src/session/run/sequential_async.rs` (new, ~932 lines) + `gpu_dispatch.rs`'s
      `try_gpu_dispatch_async`: `Session::run_gpu_async`/`enable_gpu_async` — a genuinely async
      GPU execution path (awaited, not thread-blocked); proven on native targets only so far
- [x] `oxionnx-gpu`: `GpuContext::try_new_async` now acquires a real
      `wgpu::Backends::BROWSER_WEBGPU` adapter, and kernel read-back is a real `async fn` via
      `device_guard::read_back_web`'s `map_async` — `gpu` + `wasm32` now builds together
      (`Session: Send + Sync` assertion is wasm32-exempt); not yet wired into `WasmSession`
- [x] wasm32-unknown-unknown CPU inference **actually works at runtime now**, not just compiles
      (every prior release's claim was compile-only): `oxifft`'s default `threading` feature
      target-split per-arch, a new `time_compat` module replacing `Instant`/`SystemTime` panics
      on wasm32, `session::async_run` now `#[cfg(not(target_arch = "wasm32"))]` (compile-time
      absent instead of a runtime panic)
- [x] `src/optimizer/fusion/instance_norm.rs`: `fuse_instance_norm` — recognizes decomposed
      AdaIN-style 8-node normalization chains (the only form PyTorch can export for this case)
      and collapses them into a single `OxiInstanceNorm` op (12 chains in InSwapper-128); matching
      GPU kernel in `oxionnx-gpu/src/shaders/instance_norm.rs`
- [x] `oxionnx-gpu/src/shaders/conv2d.rs`: `gpu_conv2d_implicit` — a direct implicit-GEMM Conv2D
      kernel replacing a CPU/GPU hybrid that measured slower than the plain CPU operator
      (~692 GFLOP/s at InSwapper's 128×128 decoder layer, M3)
- [x] `oxionnx-gpu`: four new WGSL kernels — `broadcast_binary.rs` (NumPy-style broadcast for
      Add/Sub/Mul/Div), `gemm.rs`'s `gpu_gemm_nt` (transposed-B Gemm), `prelu.rs`, `resize.rs`
      (bilinear/nearest) — plus `f16_variant.rs` (half-precision kernels derived from the f32
      source by textual substitution) and `kernel_support.rs` (compiled-pipeline caching)
- [x] `src/session/gpu_residency.rs`: `gpu_min_transfer_elements`/`ResidencyTier` — a measured
      two-tier size gate deciding whether a node is worth dispatching to the GPU at all, closing
      regressions of up to 36x that weight residency alone would have reopened
- [x] `oxionnx-gpu/src/context/budget.rs`: `TrackedBuffer`/`GpuMemoryBudget` (new) — the reusable
      buffer pool no longer leaks device memory without bound (`WebBuffer::drop` is a no-op on
      wgpu 29.0.4); every allocation is now checked against a live-byte budget (1.5 GiB default)
      before creation
- [x] `src/session/gpu_dispatch.rs`: `normalize_reduce_axes`/`normalize_single_reduce_axis` fix —
      GPU `Reduce*` no longer wraps a negative axis into a huge `usize`, and a full reduction now
      reports the correct rank-0 output shape instead of `[1]`
- [x] `oxionnx-ops`'s `conv2d.rs`: CPU im2col workspace capped at 64 MiB
      (`IM2COL_WORKSPACE_MAX_BYTES`) — output columns now processed in bit-identical blocks
      instead of scaling unbounded with kernel volume × output area (a wasm32 OOM risk)
- [x] `oxionnx-gpu`: WGSL `reflect`-mode `Pad` kernel fix — `reflect_coord`'s modulo assumed
      WGSL's `%` returns a sign-of-dividend remainder on a negative operand; Vulkan/NVIDIA
      returns the unsigned two's-complement remainder instead, producing wrong values in the
      leading padding region
- [x] `oxionnx-coreml`: on-disk `.mlpackage` compile cache (`compile_cache.rs`, content-keyed by
      path/length/mtime) fixes an unbounded leak (7,408 orphaned `.mlmodelc` trees / 857 GB
      measured) and cuts a warm three-model load from ~4.34s to ~0.14s (M3); output extraction
      (`array_read.rs`'s `CopyPlan`) fused into a single pass, 2.7x faster on SCRFD's padded
      outputs

## 21. `oxionnx-cuda` GEMM/Reduce cross-stream readback race (2026-08-12) — COMPLETE

Investigation into Linux+NVIDIA underperformance vs. `oxiface`'s CoreML path traced a real
"CUDA MatMul returns wrong numbers" bug, reproduced on-device (RTX A4000, sm_86) before fixing:
`matmul::cuda_matmul` (`M=1,K=25088,N=512`: 439/512 elements wrong; `64x64x64` all-ones:
3456/4096 wrong — every one reading back `0.0` instead of the correct value).

- **Root cause:** `oxicuda_dnn::DnnHandle` deliberately gives its internal `BlasHandle` its own
  CUDA stream (so BLAS and DNN launches can overlap), but `matmul.rs` and `reduce.rs` dispatched
  through `ctx.dnn.blas()` and then synchronized `ctx.dnn.stream()` — the *other*, empty stream —
  before reading results back. Every `oxicuda-driver` stream is `CU_STREAM_NON_BLOCKING`, so
  the host could read `C` back before the BLAS-stream kernel had finished. `reduce.rs` carried a
  comment asserting `reduce_axis` "launches on the same stream `ctx.dnn` owns" — false; corrected.
- **Fix:** new `oxicuda_dnn::handle::DnnHandle::synchronize_all()` (waits for both streams,
  documented with the full rationale); `matmul.rs`/`reduce.rs` now call it instead of
  `ctx.dnn.stream().synchronize()`.
- **Companion fix (upstream, `oxicuda-blas`):** the same investigation found `GemmDispatcher`
  severely under-provisions skinny/small-`M` GEMM launches (ArcFace/InSwapper's dominant
  `1x512`-ish shapes capped at ~512 threads on a device that can schedule ~70k) — a real
  ~17.6x-measured perf bug, though *not* the source of the wrong-numbers report above (that
  kernel's grid-stride loop covers every element correctly regardless of grid size; verified
  on-device before concluding this). Fixed via a genuine two-pass split-K launch. Full writeup:
  `oxicuda`'s `TODO.md`, "GEMM skinny/small-M occupancy + cross-stream readback race".
- **New coverage:** `oxionnx-cuda/tests/matmul_shape_sweep_gpu.rs` — the exact repro shapes above
  plus an M-sweep, element-for-element against `reference::ref_matmul`, through `cuda_matmul` end
  to end (fails against the pre-fix code with the exact wrong-element counts quoted above, passes
  after); `oxicuda-dnn`'s `gpu_tests/handle_sync.rs`.
- Developed against a local `oxicuda` checkout (`[patch.crates-io]` path patch) — since
  published: `oxicuda-driver`/`-memory`/`-blas`/`-dnn`/`-ptx`/`-launch` 0.5.5 is live on
  crates.io (`Cargo.lock` confirms a `registry+…crates.io-index` source, no `[patch]` section
  remains in this workspace's `Cargo.toml`), so this fix ships to every consumer, not just this
  dev tree.

## 22. `oxionnx-cuda` on-device regression-test audit for §21 (2026-08-12) — COMPLETE

Follow-up task: confirm §21's fixes (GEMM/reduce cross-stream race, cross-thread `CudaContext`
affinity) actually have thorough, real, on-device regression coverage through this crate's own
public dispatch, and that the coverage is consistently organised/gated — not just that a test
file exists somewhere.

- **Real gating bug found and fixed:** `tests/matmul_shape_sweep_gpu.rs` (added by §21) had no
  `required-features` gate, unlike every other on-device suite in this crate
  (`#[cfg(feature = "gpu-tests")]` in `src/lib.rs`/`src/conv.rs`). Confirmed on this machine: a
  bare `cargo test -p oxionnx-cuda` — no feature flag, no `OXIONNX_CUDA` env var — silently ran
  3 real on-device tests against the RTX A4000, contradicting this crate's own documented
  contract ("default `cargo test`/CI never touches a GPU", `context` module docs). Fixed with
  `required-features = ["gpu-tests"]` `[[test]]` entries in `Cargo.toml` for both on-device
  integration-test files; `matmul_shape_sweep_gpu.rs`'s no-device behaviour also aligned from a
  quiet runtime skip to the crate's dominant `.expect()`-and-fail convention (a `gpu-tests`-gated
  run with no GPU is a misconfigured invocation, matching `lib.rs`/`conv.rs`'s existing rule).
- **New coverage — cross-thread, third dispatch path:** `lib.rs`'s cross-thread suite
  (`build_context_on_a_thread_that_then_exits`) covered MatMul (BLAS/GEMM handle) and Relu
  (direct PTX-kernel launch); added `conv_dispatch_succeeds_from_a_different_thread_than_construction`
  covering `conv::cuda_conv`'s `oxicuda_dnn` engine-dispatch path (a third, distinct kernel-launch
  mechanism `activate_context` must also cover), a hand-verified 1x1-with-bias case through
  `try_cuda_dispatch` end to end.
- **New file — live shadow-verification wiring:** `tests/verify_path_gpu.rs`, gated the same way,
  self-validating (fails loudly, not silently, if run without `OXIONNX_CUDA_VERIFY=1` actually
  live — see `require_verify_enabled`). One test per `verify_or_fallback` call site in
  `try_cuda_dispatch` (MatMul, binary-elementwise, reduce, unary-elementwise, Softmax, and —
  once the sibling ref-conv task landed mid-session and wired `Conv` into the same gate — Conv
  too), each checked independently against the matching `reference::ref_*` oracle from outside
  `try_cuda_dispatch`, not just trusting its internal agreement. No dedicated file previously
  ran with `OXIONNX_CUDA_VERIFY=1` live and checked per-arm; the only prior coverage was
  incidental (the MatMul/Relu cross-thread tests happen to route through the same gate).
- **Verified for real, repeatedly, on this machine's RTX A4000:** `OXIONNX_CUDA=1
  OXIONNX_CUDA_VERIFY=1 cargo test -p oxionnx-cuda --features gpu-tests` — 110 lib tests + 3
  (`matmul_shape_sweep_gpu`) + 6 (`verify_path_gpu`, after the sibling Conv case landed) + 1
  doc-test, all green, across 7+ consecutive full runs (including once under
  `OXIONNX_CUDA_STRICT=1`, and a `cargo fmt --check`/`cargo clippy -D warnings` pass). One
  transient failure was observed mid-session (`ref_conv_adds_bias_once_per_output_channel` and,
  cascading from it via the newly-landed `verify_or_fallback("Conv", ...)` wiring, the new Conv
  cross-thread test) while the sibling ref-conv task's bias handling was mid-edit in the shared
  working tree; both settled to consistently green immediately after and stayed green over many
  subsequent reruns — not a bug in §21's fixes or in this task's own additions.
- Same `oxicuda` fix as §21 — now published as 0.5.5 (see §21's note); no longer dev-only.

## 23. `oxionnx-cuda`: `Conv` advertised as CUDA-supported (2026-08-12) — COMPLETE

The final wiring step of the Linux/NVIDIA investigation's CUDA-convolution track: with
`conv::cuda_conv` implemented (direct dispatch to `oxicuda-dnn`'s three validated forward
engines) and `reference::ref_conv` wired into the `Conv` arm's `verify_or_fallback` by the two
sibling tasks, `is_supported_op` was flipped to claim `OpKind::Conv`. Until this landed the
predicate was a hard `false`, so `oxionnx::execution_providers::decide_placement` never routed a
convolution to CUDA and the working implementation was unreachable from production inference —
only from direct `try_cuda_dispatch`/`cuda_conv` calls, i.e. from tests.

- **The flip:** `OpKind::Conv` added to `is_supported_op`'s `matches!`; its doc-comment table row
  changed from **no** to **yes** with the direct-dispatch note (`Conv1x1` / `DepthwiseConv` /
  `ImplicitGemmConv`, never `conv_forward`'s Winograd-capable auto-selector); the doctest example
  inverted; asymmetric `pads` named in "Necessary, not sufficient" as the per-node decline that
  survives the op-kind claim.
- **Self-consistency tests re-pinned:** `claimable_ops()` gained `Conv` (arity 25 → 26);
  `oracle_covers_every_op_the_unary_binary_and_reduce_dispatch_arms_claim` renamed to
  `oracle_covers_every_op_try_cuda_dispatch_can_claim` and given a `CONV_OPS` family with its own
  `ref_conv`-based check. `ref_conv` takes no `OpKind` and returns a plain `Vec<f32>`, so the
  `.is_some()` probe the unary/binary/reduce families use does not apply; instead the branch runs
  the oracle on a hand-computed problem (`1*1 + 2*10 + 3*100 + 4*1000 + bias 5 = 4326`, decade-
  separated so a dropped bias, a transposed filter, or a zeroed stub each land on a visibly
  different number).
- **Contradicted tests replaced, not deleted:** `conv::tests::conv_is_not_advertised_as_supported`
  → `conv_is_advertised_as_supported`; `lib.rs`'s `conv_has_an_arm_but_is_not_claimable` →
  `conv_is_advertised_as_supported`. Four *downstream* tests in the `oxionnx` crate itself also
  asserted the old behaviour and went red on the flip — `execution_providers::auto_never_routes_
  conv_to_cuda`, `provider_supports_op_cuda_delegates_to_the_cuda_crate`,
  `session::run::sequential::auto_never_gates_conv_to_cuda`,
  `session::run::parallel::conv_is_never_planned_onto_cuda` — each inverted to assert that CUDA
  now claims/heads the chain for `Conv`. Their original point (placement consults each backend's
  *own* predicate, not the wgpu-flavoured `is_gpu_capable`) was preserved by moving the exemplar
  from `Conv` to `ReduceMean`, where the two predicates still genuinely disagree.
- **New on-device coverage:** `conv_claimed_by_the_pre_filter_is_actually_dispatched` walks the
  full production sequence (`is_supported_op` → `try_cuda_dispatch` → `reference::ref_conv`
  comparison) on the oxiface workhorse shape (3x3, stride 1, pad 1, bias, multi-channel →
  `ImplicitGemmConv`); `advertised_conv_still_declines_the_configurations_it_cannot_compute`
  pins the other half of the contract — asymmetric `pads` must come back `Ok(None)`, never a
  tensor computed for padding the model did not ask for.
- **Falsifiability checked, not assumed** (this codebase's catalogued failure mode is tests that
  pass without testing anything): reverting only the `matches!` line makes 7 tests fail
  (2 × `conv_is_advertised_as_supported`, `is_supported_op_matches_dispatch_arms`, both new
  on-device tests, `conv_verify_path_agrees_live_on_real_hardware`, the doctest); making the
  decline-test's `pads` symmetric makes it fail (so the decline is attributable to the asymmetry,
  not to something incidental about the node); gutting `ref_conv`'s bias makes the new oracle
  branch fail with `left: [4321.0], right: [4326.0]`.
- **Stale docs corrected:** `conv.rs`'s "Why `Conv` is still not advertised" section replaced with
  "Advertised as CUDA-supported"; its stale `tests/conv_verify_gpu.rs` path (a file that never
  existed) corrected to `tests/verify_path_gpu.rs`; `verify_path_gpu.rs`'s module docs no longer
  claim the `Conv` arm is unreachable from placement, and its Conv test now *asserts* the
  predicate rather than describing it. `oxionnx-cuda/README.md`'s "Conv — not accelerated …
  `oxicuda-dnn`'s convolution engines have stubbed GEMM phases" row rewritten (that verdict on
  `oxicuda-dnn` no longer holds), 25 → 26 ops, and the 0.1.5 history paragraph marked as history.
  Root `README.md`'s "Conv still stubbed and not advertised" corrected — per-crate *test counts*
  there deliberately left alone, since they are release figures the footnote sums to a total and
  publishing is out of scope for this session.
- **Verified on this machine's RTX A4000** (sm_86, driver 550.144.03): `OXIONNX_CUDA=1
  OXIONNX_CUDA_VERIFY=1 cargo test -p oxionnx-cuda --features gpu-tests` — 112 lib + 3
  (`matmul_shape_sweep_gpu`) + 6 (`verify_path_gpu`) + 1 doc-test, all green; plain
  `cargo test -p oxionnx-cuda` 101 + 1 green; `cargo test -p oxionnx --features cuda` all green
  (493 lib + every integration binary); `cargo fmt --check` and `cargo clippy --all-targets`
  clean on both crates; `cargo check --workspace` green in the downstream `oxiface` tree.
- Same `oxicuda` fix as §21/§22 — now published as 0.5.5 (see §21's note); no longer dev-only.

## 24. `oxionnx-cuda` on-device tests: restore the OxiCUDA skip convention (2026-08-13) — COMPLETE

`cargo nextest run --all-features` on a CPU-only host (this Mac, Apple M3) failed 30 tests, all
in `oxionnx-cuda`, every one panicking with `"gpu-tests requires a real CUDA device -- run on a
CUDA-capable host"`. Cargo has no "all features except X", so `--all-features` unavoidably
switches `gpu-tests` on, and §22 had made that fatal without a device.

- **§22's premise was wrong.** §22 aligned `matmul_shape_sweep_gpu.rs` away from a quiet runtime
  skip and onto `.expect()`-and-fail, described as "the crate's dominant convention" and, in
  `Cargo.toml`, as mirroring `oxicuda-driver`/`oxicuda-blas`/`oxicuda-dnn`. OxiCUDA does the
  opposite: `oxicuda-blas/src/gpu_tests.rs` states "Every device test returns early (skips) when
  no CUDA device is present, so the suite stays green on CPU-only machines", and both it and
  `tests/gemm_shape_sweep_gpu.rs` use `Option`-returning fixtures (`gpu_fixture()`,
  `try_handle()`) with `let Some(..) = .. else { eprintln!(..); return; }`. Verified empirically:
  `cargo nextest run -p oxicuda-blas -p oxicuda-dnn -p oxicuda-driver --all-features` on this M3
  is 2753 passed / 0 failed. So this was a divergence from OxiCUDA, not a mirror of it.
- **Restored the skip convention** across all five on-device files — `src/dispatch_tests.rs`
  (`build_context_on_a_thread_that_then_exits` plus two inline `CudaContext::try_new_with`
  sites), `src/conv.rs` (`gpu_ctx`, consumed by `run_case_and_compare`, which skips per `tag`),
  `tests/matmul_shape_sweep_gpu.rs`, `tests/verify_path_gpu.rs`, `tests/batched_matmul_gpu.rs`.
  Every fixture now returns `Option<CudaContext>`; every test skips by name. Note
  `CudaContext::try_new_with` already returned `Option`, not `Result` — the old `.expect()` was
  reading as a `Result` idiom but never was one.
- **`verify_path_gpu.rs` gate ordering.** Its six tests called `require_verify_enabled()` (asserts
  `OXIONNX_CUDA_VERIFY=1` live) *before* acquiring the device, so on a CPU-only host they failed
  on the env-var assert rather than the device. The device check now runs first: no device →
  skip. The env-var assert is deliberately still loud on a host that *does* have a GPU, since
  there a missing var means the run proves nothing — that half of §22's design is preserved.
- `examples/dispatch_bench.rs` already skipped gracefully (`"no CUDA device -- run this on a
  CUDA-capable host"`); left as is. `required-features = ["gpu-tests"]` on the `[[test]]`/
  `[[example]]` targets is untouched — §22's *other* finding was real and still holds: a plain
  `cargo test -p oxionnx-cuda` must not touch a GPU on a CUDA-capable host.
- **Verified on this M3 Mac:** `cargo nextest run -p oxionnx-cuda --features gpu-tests` — 148
  run, 148 passed, 0 failed (previously 148 run / 30 failed; the feature adds exactly those 30
  on-device tests to the 118 that build without it). Under `--no-capture`, exactly 30 distinct
  `"no CUDA device present, skipping <name>"` lines are emitted — the same 30 that used to fail,
  confirming each test genuinely reaches its skip rather than having had its body orphaned by
  the refactor. Full workspace `cargo nextest run --workspace --all-features` — 3296 run, 3296
  passed, 18 skipped, 0 failed (was 30 failed).
  `cargo fmt --check` and `cargo clippy --all-targets --all-features -D warnings` clean.
- Still to re-verify on the RTX A4000 box: that the suite genuinely *runs* (not skips) there,
  i.e. all 148 execute for real — the skip path is by construction invisible on a GPU host, so
  a green run there must be checked for test count, not just exit status.

## 25. v0.1.7 release prep — full validation (2026-08-14) — COMPLETE

`/runall 0.1.7` full-mode pipeline: `/changelog-gen` → `/release-check` → `/final-call` →
`/readme`, run end to end on this M3 Mac. Ties together §21-24 above (all landed within this
same release cycle, after v0.1.6's §20) plus the `is_supported_op` op-family growth documented
in `CHANGELOG.md`'s `## [0.1.7]` entry (25 → 40 CUDA-accelerated ops).

- **Full workspace validation, this run:** `RUSTFLAGS="-C debuginfo=0" CARGO_INCREMENTAL=0
  cargo nextest run --no-fail-fast --workspace` — 3,289 run, 3,289 passed, 19 skipped, 0 failed.
  Same with `--all-features` — 3,595 run, 3,595 passed, 19 skipped, 0 failed. Both zero
  compiler warnings and zero nextest config warnings. `cargo clippy --all-features --all-targets
  -- -D warnings` clean. `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps` clean.
  Doc tests (`~/work/doctest-parallel.sh`): 15 passed, 0 failed, 8/8 crates. `cargo audit`: 0
  vulnerabilities. `cargo deny check bans`: ok. `cargo +nightly udeps --all-targets
  --all-features`: no unused dependencies. `unwrap()` audit (brace-matching classifier, not a
  naive grep): 0 production-code call sites across all 8 crates — every hit lands inside a
  `#[cfg(test)]`-gated region.
- **Docs fixed as part of this run:** `oxionnx-cuda/README.md`'s "Accelerated operators" table
  was stale at 26 ops (Conv-era) against the crate's actual 40 (re-verified directly against
  `is_supported_op`'s `matches!` arm) — six new rows added (Pooling, Resize, Pad, Data movement,
  Shape (zero-cost), PRelu) and the Binary row corrected (channel-broadcast and scalar-broadcast
  are no longer "same-shape only"). `src/session/run/parallel.rs`'s doc comment had the same
  stale "26"; corrected. This TODO.md's own §21-23 said the `oxicuda` cross-stream-race /
  Conv-wiring fix was "dev-only … not yet published" — `Cargo.lock` now shows
  `oxicuda-driver`/`-memory`/`-blas`/`-dnn`/`-ptx`/`-launch` 0.5.5 sourced from
  `registry+…crates.io-index` with no `[patch.crates-io]` section anywhere in this workspace's
  `Cargo.toml`, so that fix ships to every consumer now, not just this dev tree; corrected in
  place rather than left to read as still-pending.
- **Publish readiness:** `cargo publish --dry-run --allow-dirty` per crate in topological order
  (`oxionnx-core` → `oxionnx-coreml`/`oxionnx-cuda`/`oxionnx-ops`/`oxionnx-proto` →
  `oxionnx-directml`/`oxionnx-gpu` → `oxionnx`) — `oxionnx-core` (Tier 0, no internal deps)
  packages cleanly (18 files, 167.6 KiB); every dependent crate fails identically with `failed to
  select a version for the requirement oxionnx-core = "^0.1.7"` because that version is not yet
  on crates.io — the expected pre-publish chicken-and-egg for a fresh multi-tier release, not a
  packaging defect (confirmed: all seven failures share this one root cause, no other error
  shape).
- **Not addressed this run, tracked for later:** `oxionnx-cuda/src/reference.rs` is 1,953 lines
  — 47 lines under this project's 2,000-line-per-file cap. Not a violation yet, but the closest
  file to the ceiling in the workspace; worth a `splitrs` pass in a future cycle before it forces
  one mid-change. `oxionnx-gpu/README.md` already documents `GpuTuning` and
  `try_new_diagnosed()` (verified: both are covered in its own text), but not yet the
  multi-context `PipelineCache` fix (the `BindGroupLayout does not exist` panic from opening a
  second `GpuContext` in one process) or activation-buffer recycling — both real 0.1.7
  capabilities (see `CHANGELOG.md`'s Fixed/Changed sections) — added later in the cycle than the
  README's last content pass; worth a follow-up addition, scoped separately from this run.
