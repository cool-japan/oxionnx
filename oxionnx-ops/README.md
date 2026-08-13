# oxionnx-ops

Operator implementations for OxiONNX -- 189 ONNX operators in Pure Rust (204 including
`ai.onnx.ml.*` and legacy-name aliases).

This crate contains the actual computation logic for every ONNX operator supported
by OxiONNX. Each operator implements the `oxionnx_core::Operator` trait and is
registered via `default_registry()`. Implementations are checked against ONNX spec edge
cases rather than just the common case -- e.g. `Slice`'s negative `start`/`end`/`step`,
`Pad`'s opset-18 `axes` input and `wrap` mode, `GRU`'s default `linear_before_reset=0`
gate ordering, `ScatterElements`/`ScatterND`'s `reduction` attribute, per-channel
`QuantizeLinear`/`DequantizeLinear` via the `axis` attribute, and `TreeEnsemble*`'s
cycle-guard/NaN-routing/MIN-MAX aggregate handling.

## Operator Categories

- **Math** -- MatMul, Gemm, Add, Sub, Mul, Div, Pow, Sqrt, trig functions, reductions (Sum, Mean, Max, Min, Prod), TopK, CumSum, Einsum, Det, and more.
- **Neural Network** -- Softmax, Relu, Sigmoid, Tanh, Gelu, SiLU, LayerNorm, BatchNorm, GroupNorm, RMSNorm, HardSigmoid, HardSwish, Dropout, LRN. `Softmax`/`LogSoftmax`/`Hardmax` are opset-aware (pre-13 default-axis-1-plus-flatten-to-2D vs. opset-13+ direct per-axis reduction).
- **Convolution** -- Conv, ConvTranspose (rank-generic 1D/2D/3D via `conv::conv`/`conv::conv_transpose`, plus a dedicated 2D im2col/Winograd fast path), MaxPool, AveragePool, GlobalAveragePool, LpPool, GlobalLpPool, MaxUnpool, MaxRoiPool, Col2Im, CenterCropPad.
- **Shape** -- Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Concat, Slice, Gather, Scatter, Split, Expand, Tile, Pad, CastLike, and more.
- **Comparison** -- Equal, Greater, Less, Where, Not, And, Or, Xor.
- **Control Flow** -- If, Loop, Scan with subgraph execution.
- **RNN** -- LSTM, GRU, RNN.
- **ML** -- LinearClassifier, LinearRegressor, SVMClassifier [Partial -- kernel/`support_vectors` mode only; `linear` mode (no `support_vectors`) returns `Unsupported`], SVMRegressor, TreeEnsembleClassifier, TreeEnsembleRegressor, Normalizer, Scaler, LabelEncoder, TfIdfVectorizer, StringNormalizer.
- **Attention** -- Attention (fused), MultiHeadAttention, RotaryEmbedding.
- **Spatial** -- GridSample, RoiAlign.
- **Quantized** -- QLinearConv, QLinearMatMul, MatMulInteger, ConvInteger, QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear.
- **Resize** -- Resize with nearest/linear/cubic coordinate transforms and multiple `nearest_mode`/`coordinate_transformation_mode` variants; Upsample (legacy predecessor).
- **NMS** -- NonMaxSuppression.
- **Audio / DSP** -- DFT, STFT, BlackmanWindow, HannWindow, HammingWindow, MelWeightMatrix, Bernoulli.
- **Generators** -- RandomNormal, RandomUniform, RandomNormalLike, RandomUniformLike, Multinomial.
- **Loss** -- NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss.

## Usage

```toml
[dependencies]
oxionnx-ops = "0.1.8"
```

```rust
use oxionnx_ops::default_registry;

// Build a registry pre-populated with all supported operators
let registry = default_registry();

// Look up an operator by its ONNX op_type
let relu_op = registry.get("Relu").expect("Relu should be registered");
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `simd`  | No      | Enables hand-tuned SIMD kernels for performance-critical operators. |
| `wasm-threads` | No | wasm32-only: restores rayon's intra-operator parallelism in the browser. Requires a nightly `-Z build-std` rebuild of `std` with `-C target-feature=+atomics,+bulk-memory`, a cross-origin-isolated page (`SharedArrayBuffer`), and a `wasm-bindgen-rayon` `initThreadPool()` call before the first operator runs. Inert on non-wasm32 targets (no dependency or `cfg` change). |

On non-wasm targets, `rayon` is used automatically for parallel execution of
data-parallel operators.

## Native Dtype Dispatch

40+ operators implement `native_dtypes()` and `execute_typed()`, enabling `run_typed()` to skip
f32 round-trips for F16, BF16, I32, I64, and other dtypes. Includes MatMul with native dispatch
for F32/F16/BF16/I8→I32/I32 dtypes (no f32 round-trip). The remaining operators fall back
transparently to the f32 path via the default implementation on the `Operator` trait.
See `oxionnx_core::TypedOpContext` for the dispatch API.

## IOBinding Slot Reuse (Phase F)

Most operators in the registry opt into `supports_output_slots = true`. 130 operators implement
hand-coded `execute_into_slots` bodies (no memcpy, writes directly into pre-allocated output
buffers), including shape ops (Squeeze, Unsqueeze, Flatten, Expand, Split, Tile, DepthToSpace,
SpaceToDepth, ReverseSequence), elementwise NN (Clip, LeakyRelu, PRelu, HardSigmoid, Celu, Elu,
Selu, ThresholdedRelu, LpNorm, MeanVarianceNorm, Hardmax, Shrink), conv/pool (MaxPool,
AveragePool, GlobalAveragePool, GlobalMaxPool, Pad, Resize), indexing (Gather, ScatterND,
ScatterElements), and the arithmetic/comparison/bitwise/reduce/variadic families (Add/Sub/Mul/
Div/Pow, Sqrt and the trig functions, Equal/Greater/And/Or/Xor, BitwiseAnd/Or/Xor, the
`Reduce*` family, `Variadic{Min,Max,Mean,Sum}`) that share a handful of generic macros, each of
which generates its own `execute_into_slots` body. Use `IoBinding` to bind output buffers;
pointer identity is preserved across inference runs when the output shape is stable.

## Einsum

The equation parser supports numpy-compatible ellipsis (`...`) tokens (e.g.
`...ij,...jk->...ik` for broadcast batched matmul), right-aligning and broadcasting each
operand's ellipsis axes the way numpy does. A named label shared across operands now
broadcasts when one side's extent is `1`, matching `numpy.einsum`, instead of erroring on
any extent mismatch. Large contractions lower to `matrixmultiply::sgemm` through a greedy
pairwise decomposition; contractions at or below a small FLOP threshold still run through
the allocation-free scalar interpreter, which also serves as the GEMM path's test oracle.

## Performance

Small-`M` `MatMul` and attention/flash-attention are routed through `sgemm`
(`matrixmultiply`) instead of a scalar loop; KV-cache append is in-place instead of
reallocating; broadcast operands avoid materializing the expanded tensor; convolution's
im2col path is threaded and cache-blocked, with a persistent Winograd filter-transform
cache reused across calls.

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
