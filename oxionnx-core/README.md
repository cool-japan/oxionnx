# oxionnx-core

Core types for OxiONNX -- Tensor, Graph, OpKind, Operator trait, and error types.

This crate provides the foundational data structures and abstractions used throughout
the OxiONNX inference engine. It is `no_std`-compatible (with the `std` feature disabled)
and has minimal dependencies.

## Key Types

- **`Tensor`** -- N-dimensional tensor with f32 storage, shape, strides, and layout support (NCHW/NHWC/RowMajor).
- **`DType`** / **`TypedTensor`** -- Multi-dtype tensor support covering F32, F16, BF16, F64, I8/I16/I32/I64, U8/U16/U32/U64, and Bool.
- **`Graph`** -- Represents an ONNX computation graph as a list of `Node`s with input/output names.
- **`OpKind`** -- Enum of all supported ONNX operators (167).
- **`Operator`** trait -- Stateless interface for operator implementations; receives an `OpContext` with resolved inputs.
- **`OperatorRegistry`** -- Maps ONNX op_type strings to `Operator` trait objects.
- **`TypedOpContext`** -- Context struct for typed operator dispatch (Phase D); parallels `OpContext` but carries `TypedTensor` inputs.
- **`OnnxError`** -- Unified error type for the engine.

## Usage

```toml
[dependencies]
oxionnx-core = "0.1.2"
```

```rust
use oxionnx_core::{Tensor, Graph, OpKind, DType};

// Create a 2x3 tensor
let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
assert_eq!(t.shape, vec![2, 3]);

// Layout conversion
use oxionnx_core::{nchw_to_nhwc, nhwc_to_nchw};
let img = Tensor::new(vec![0.0; 24], vec![1, 3, 2, 4]); // [N,C,H,W]
let nhwc = nchw_to_nhwc(&img).expect("layout conversion");
assert_eq!(nhwc.shape, vec![1, 2, 4, 3]); // [N,H,W,C]
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std`   | Yes     | Enables standard library support. Disable for `no_std` environments. |
| `ndarray` | No    | ndarray interop for Tensor conversion (from_ndarray, to_ndarray, as_ndarray_view) |

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
