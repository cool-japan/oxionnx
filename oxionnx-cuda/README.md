# oxionnx-cuda

[![crates.io](https://img.shields.io/crates/v/oxionnx-cuda.svg)](https://crates.io/crates/oxionnx-cuda)
[![docs.rs](https://docs.rs/oxionnx-cuda/badge.svg)](https://docs.rs/oxionnx-cuda)

CUDA-accelerated dispatch layer for the [OxiONNX](https://github.com/cool-japan/oxionnx)
pure-Rust ONNX inference engine.

## Overview

`oxionnx-cuda` provides GPU-accelerated execution of ONNX operators via the
OxiCUDA stack (`oxicuda-driver`, `oxicuda-blas`, `oxicuda-dnn`, `oxicuda-ptx`,
`oxicuda-launch`). It sits at the highest priority in the three-tier dispatch
chain used by `oxionnx::Session`:

```
CUDA (highest priority)
  +-- try_cuda_dispatch -> Ok(Some(results))   <- GPU handled it
      +-- Ok(None)                             <- fall back to wgpu / CPU
wgpu GPU dispatch
CPU dispatch
```

When no CUDA device is available, `CudaContext::try_new()` returns `None` and
the session silently falls back to the wgpu or CPU backend — no crash, no
configuration required.

## Accelerated operators

| Category         | Operators                                                         |
|------------------|-------------------------------------------------------------------|
| Linear algebra   | MatMul, Gemm (batched, transposed) — via `oxicuda_blas::gemm`     |
| Convolution      | Conv — stubbed (returns `Ok(None)`, falls back to CPU; GEMM engine pending) |
| Unary activation | Relu, Sigmoid, Gelu, Tanh, Exp, Sqrt, Abs, Neg, Log, Ceil, Floor, HardSigmoid, HardSwish, SiLU, Softplus, LeakyRelu (15 ops via PTX) |
| Binary           | Add, Sub, Mul, Div (same-shape only)                              |
| Reduction        | ReduceSum, ReduceMax (single axis only)                           |
| Normalization    | Softmax (last-axis, row_size <= 1024)                             |

Unsupported or unrecognised operators return `Ok(None)` so the caller falls
back automatically.

## Feature flags

`oxionnx-cuda` itself exposes no feature flags. The crate is activated in the
parent `oxionnx` workspace crate via the `cuda` feature:

| Feature | Description |
|---------|-------------|
| `cuda`  | (on `oxionnx`) Enables `oxionnx-cuda` and CUDA dispatch in `Session`. |

## Usage

Add the parent crate with the `cuda` feature to your `Cargo.toml`:

```toml
[dependencies]
oxionnx = { version = "0.1", features = ["cuda"] }
```

If you need to use the CUDA dispatch layer directly:

```toml
[dependencies]
oxionnx-cuda = "0.1"
```

### Basic example

```rust
use oxionnx_cuda::CudaContext;

// Returns None if no CUDA device is present — no panic, no unwrap required.
if let Some(ctx) = CudaContext::try_new() {
    println!("CUDA device ready: {:?}", ctx.driver_context());
}
```

In practice, CUDA dispatch is invoked automatically by `oxionnx::Session` when
the `cuda` feature is enabled and a compatible GPU is present. Direct use of
`try_cuda_dispatch` is only needed when embedding the CUDA backend into a
custom inference loop.

### Error handling

All CUDA errors are represented by `CudaError` (re-exported from
`CudaDispatchError`). The variants cover driver initialisation failures, BLAS
and DNN operation errors, PTX compilation errors, unsupported configurations,
and shape mismatches. Each variant implements `std::error::Error` and converts
to `OnnxError::Internal` via a `From` impl so the session layer never needs to
handle CUDA errors directly.

## Requirements

- Rust 1.75 or later
- NVIDIA GPU with a driver that supports the CUDA runtime used by OxiCUDA
- The OxiCUDA crates (`oxicuda-*`) must be present in the workspace or
  available on crates.io

A missing or incompatible CUDA installation is not a hard error at build time:
the crate compiles on any platform, and `CudaContext::try_new()` returns `None`
at runtime when no device is available.

## Part of the OxiONNX workspace

This crate is a member of the
[oxionnx workspace](https://github.com/cool-japan/oxionnx).

Other workspace members:

- `oxionnx-core` — tensor types, graph representation, error types
- `oxionnx-ops` — CPU operator kernels
- `oxionnx-proto` — ONNX protobuf parser
- `oxionnx-gpu` — wgpu/WebGPU backend
- `oxionnx` — top-level session API

## License

Licensed under `Apache-2.0`.

Copyright COOLJAPAN OU (Team Kitasan).
