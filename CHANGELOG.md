# Changelog

All notable changes to OxiONNX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.4] - 2026-06-08

### Added
- feat: **Subgraph parsing + end-to-end execution for `If`, `Loop`, `Scan` operators** — ONNX models containing control-flow nodes now parse and execute correctly from real `.onnx` files. Previously the `If`/`Loop`/`Scan` kernels existed and passed in-memory unit tests but were never reachable from a loaded model due to two disconnects:
  1. `oxionnx-proto`: `AttributeValue` now carries `g: Option<Box<GraphProto>>` for graph-valued attributes (wire field 6); `parse_attribute` parses nested subgraphs via the existing `parse_graph` recursively (both eager and streaming parser paths fixed simultaneously). `build_subgraph` in `model.rs` converts a nested `GraphProto` into a runtime `Graph`, synthesising `Constant` nodes for subgraph-local initializers without touching the 48-literal `Graph` struct.
  2. `oxionnx-core`: `OpContext` gains a `weights: Option<&'a HashMap<String, Tensor>>` field so model weights can be accessed by subgraph nodes without cloning. `Session::dispatch_node` now populates `outer_scope`, `weights`, and `registry` in all three dispatch paths (slot-write, inplace, default-execute) and the parallel rayon path.
  3. `oxionnx-ops` control-flow kernels (`IfOp`, `LoopOp`, `ScanOp`) pass `ctx.weights` through to `execute_subgraph`, giving subgraph nodes access to outer-model initializers at zero allocation cost.
- feat: 9 new end-to-end integration tests (`tests/control_flow_e2e.rs`): `If` true/false branch, `Loop` accumulation, `Loop` zero-iteration, `Loop` with outer-scope weight, `Scan` element-wise map, `Scan` with running-state, `If` with sequential subgraph ops, `If` with subgraph-local initializer.
- feat: 2 new `oxionnx-proto` unit tests: `test_subgraph_attribute_parsed` (parser round-trip), `test_subgraph_attribute_wired_into_graph` (build_subgraph populates Attributes.graphs).
- perf: **Phase F.13 complete** — zero-copy `execute_into_slots` for `AttentionOp` and `MultiHeadAttentionOp`, the last two ops in the Phase F sweep. Backed by `sdpa_into` / `sdpa_output_shape` / `reshape_from_heads_into` / `multi_head_attention_into` in `attention/core.rs` (allocation-free kernels; existing public APIs unchanged). The `no-outproj` MHA path now skips the `concat` allocation entirely via `reshape_from_heads_into`; the `with-outproj` path writes the projected output directly into the caller's pooled buffer. Phase F slot-write sweep closed at 58 of 121 ops.
- fix: SIMD path in `scaled_dot_product_attention` did not zero `row_out` between query rows — `weighted_sum_v` accumulates, so for `seq_q > 1` with the `simd` feature enabled the result was incorrect. The extraction into `sdpa_into` naturally fixed this by writing directly into per-row slices of the caller buffer (no shared `row_out` temp at all).
- feat: 21 new tests in `oxionnx-ops/tests/output_slots_attention_test.rs`: AttentionOp and MultiHeadAttentionOp slot-write correctness vs `execute`, broadcast mask, custom scale, `seq_k ≠ seq_q`, 4D Q, second-call resize (grow and shrink), qkv-projection, out-projection with/without bias, error parity, batch > 1.

- perf: **Phase F.14 complete** — zero-copy `execute_into_slots` for 47 additional operators, bringing the total to 105 of 121 ops with hand-coded slot-write bodies. True zero-copy (no output Vec allocation): LayerNorm/GroupNorm/BatchNorm/RmsNorm/InstanceNorm/Softmax/LogSoftmax via new `_into` kernel variants in `nn/normalization.rs`; Not/IsInf/IsNaN/BitwiseNot/Shape/Size/Constant/ConstantOfShape inline into slot; CumSum inline (same-size as input). Reduced-allocation: ReduceSum/Mean/Max/Min/Prod/L1/L2/LogSum/LogSumExp/SumSquare via new `reduce_with_into`+`reduce_output_shape` in `math/reduce.rs`; ArgMax/ArgMin via `arg_reduce_into`; TopK (2 outputs) via `top_k_into`; variadic ops and comparison/bitwise binary ops via macro update (call+move avoiding Vec<Tensor> wrapper). 41 new tests in `output_slots_f14_test.rs`.
- feat: `NodeInfo` and `Session::nodes()`: public per-node graph introspection — enumerate a model's compute nodes (name, op type, inputs, outputs, and a deterministic attribute summary) in topological order; `NodeInfo` re-exported as `oxionnx::NodeInfo` (#2).
- `ReshapeOp` now reads the ONNX `allowzero` attribute: when `allowzero=0` (default) a `0` in the target shape copies the corresponding input dimension; when `allowzero=1` a `0` is a literal zero-size dimension. A new `resolve_reshape(input_dims, numel, shape, allowzero)` helper is re-exported from `oxionnx-ops::shape`. 7 new unit tests cover all `allowzero` code paths (`test_reshape_allowzero_*` in `oxionnx-ops/src/shape/tests.rs`).
- `console_error_panic_hook = "0.1.7"` added as an optional workspace dependency wired into the `wasm` feature for improved panic messages in WebAssembly environments.

### Changed
- Updated CUDA / OxiCUDA dependencies to 0.1.8.
- `oxifft` updated from 0.3 to 0.3.2.
- wasm32 compatibility: thread-pool initialization in `src/session/loading.rs` is now skipped under `#[cfg(target_arch = "wasm32")]`; all rayon imports and parallel-execution functions in `src/session/run/parallel.rs` are likewise gated with `#[cfg(not(target_arch = "wasm32"))]`. WASM sessions now build cleanly without a thread pool.

### Fixed
- `SplitOp` now honors the opset-1..12 `split` attribute for chunk sizes when no `split` input tensor is present, fixing YOLO11 (opset 11) models that failed with `reshape: element count mismatch (33600 vs [1, 128, 20, 20])` in the C2PSA attention block (#1).

---

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

[0.1.4]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.4
[0.1.3]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.3
[0.1.2]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxionnx/releases/tag/v0.1.0
