# oxionnx-core

Core types for OxiONNX -- Tensor, Graph, OpKind, Operator trait, and error types.

This crate provides the foundational data structures and abstractions used throughout
the OxiONNX inference engine. It is `no_std`-compatible (with the `std` feature disabled)
and has minimal dependencies; the `no_std` build is compile-verified via
`cargo check -p oxionnx-core --no-default-features`.

## Key Types

- **`Tensor`** -- N-dimensional tensor with f32 storage, shape, strides, and layout support (NCHW/NHWC/RowMajor). A true ONNX rank-0 (scalar) result is represented as `shape: vec![]` with one element instead of being promoted to `[1]`; `Det` and the two loss operators (`reduction=mean|sum`) emit rank 0 end-to-end today, while most other rank-reducing ops (e.g. `Squeeze` to a scalar, an all-axes `Reduce*`) still promote to `[1]` -- a known, tracked gap, not a regression.
- **`Tensor::try_new`** -- Fallible constructor validating `data.len() == shape.iter().product()` unconditionally, including release builds; returns a typed `OnnxError::ShapeMismatch` instead of risking a panic or corruption from untrusted model bytes. `Tensor::new`'s signature and infallible behavior (invariant checked via `debug_assert` only) are unchanged, for callers who can guarantee the invariant statically.
- **`DType`** / **`TypedTensor`** -- Multi-dtype tensor support covering F32, F16, BF16, F64, I8/I16/I32/I64, U8/U16/U32/U64, and Bool.
- **`Graph`** -- Represents an ONNX computation graph as a list of `Node`s with input/output names; topological sort no longer underflows when a node's output name collides with a known input/initializer name.
- **`OpKind`** -- Enum of all supported ONNX operators (190 named variants, plus an `Unknown(String)` catch-all for unrecognized op_types).
- **`Operator`** trait -- Stateless interface for operator implementations; receives an `OpContext` with resolved inputs.
- **`OperatorRegistry`** -- Maps ONNX op_type strings to `Operator` trait objects.
- **`TypedOpContext`** -- Context struct for typed operator dispatch (Phase D); parallels `OpContext` but carries `TypedTensor` inputs.
- **`OnnxError`** -- Unified error type for the engine; `#[non_exhaustive]` (as are the workspace's other error/config enums), so new failure modes can be added without a semver break.

## Usage

```toml
[dependencies]
oxionnx-core = "0.1.7"
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
