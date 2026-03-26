# oxionnx-ops

Operator implementations for OxiONNX -- 147 ONNX operators in Pure Rust.

This crate contains the actual computation logic for every ONNX operator supported
by OxiONNX. Each operator implements the `oxionnx_core::Operator` trait and is
registered via `default_registry()`.

## Operator Categories

- **Math** -- MatMul, Gemm, Add, Sub, Mul, Div, Pow, Sqrt, trig functions, reductions (Sum, Mean, Max, Min, Prod), TopK, CumSum, Einsum, and more.
- **Neural Network** -- Softmax, Relu, Sigmoid, Tanh, Gelu, SiLU, LayerNorm, BatchNorm, GroupNorm, RMSNorm, HardSigmoid, HardSwish, Dropout, LRN.
- **Convolution** -- Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool.
- **Shape** -- Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Concat, Slice, Gather, Scatter, Split, Expand, Tile, Pad, and more.
- **Comparison** -- Equal, Greater, Less, Where, Not, And, Or, Xor.
- **Control Flow** -- If, Loop, Scan with subgraph execution.
- **RNN** -- LSTM, GRU.
- **ML** -- LinearClassifier, LinearRegressor, SVMClassifier, TreeEnsembleClassifier, TreeEnsembleRegressor, Normalizer, Scaler, LabelEncoder, OneHotEncoder.
- **Attention** -- Multi-head attention (fused).
- **Quantized** -- QLinearConv, QLinearMatMul, QuantizeLinear, DequantizeLinear.
- **Resize** -- Resize with nearest/linear/cubic coordinate transforms.
- **NMS** -- NonMaxSuppression.

## Usage

```toml
[dependencies]
oxionnx-ops = "0.1.0"
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

On non-wasm targets, `rayon` is used automatically for parallel execution of
data-parallel operators.

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
