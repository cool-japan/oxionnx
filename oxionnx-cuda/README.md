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

When no CUDA device is available, or when CUDA has not been activated (it is
opt-in — see [Activation, shadow verification, and strict
mode](#activation-shadow-verification-and-strict-mode) below),
`CudaContext::try_new()` returns `None` and the session silently falls back
to the wgpu or CPU backend — no crash.

## Accelerated operators

| Category         | Operators                                                         |
|------------------|-------------------------------------------------------------------|
| Linear algebra   | MatMul, Gemm — batched with numpy-style batch broadcasting (e.g. 3-D activations × a 2-D weight), `transA`/`transB`, and a bias epilogue for `[]`/`[N]`/`[M,N]`-shaped `C` — via `oxicuda_blas::gemm` |
| Convolution      | Conv — not accelerated. `cuda_conv` unconditionally declines to the CPU: `oxicuda-dnn`'s convolution engines have stubbed GEMM phases that would silently produce wrong numbers, so no CUDA kernel is wired up rather than shipping one that's fast and wrong. |
| Unary activation | Relu, Sigmoid, Gelu, Tanh, Exp, Sqrt, Abs, Neg, Log, Ceil, Floor, HardSigmoid, HardSwish, SiLU, Softplus, LeakyRelu (16 ops via PTX) |
| Binary           | Add, Sub, Mul, Div (same-shape only)                              |
| Reduction        | ReduceSum, ReduceMax (single axis, any axis length)                |
| Normalization    | Softmax — last axis only; a non-default `axis` attribute declines to the CPU instead of computing the wrong axis; row width ≤ 1024 |

Unsupported or unrecognised operators, and unsupported configurations of an
otherwise-accelerated operator, return `Ok(None)` so the caller falls back to
wgpu/CPU automatically — never a wrong answer, only a missed acceleration.
`LeakyRelu` and `HardSigmoid` accelerate only when their `alpha`/`beta`
attributes equal the ONNX defaults (`alpha=0.01` for `LeakyRelu`;
`alpha=0.2`, `beta=0.5` for `HardSigmoid`), and `Gelu` only when
`approximate="tanh"` is set explicitly — each PTX kernel hard-codes one
constant configuration with no launch-time override, so a node asking for
anything else declines to the attribute-aware CPU kernel instead of silently
computing the wrong constant.

`oxionnx_cuda::is_supported_op(op: &OpKind) -> bool` is a cheap, pure,
allocation-free predicate reporting exactly which of the 25 ops above
`try_cuda_dispatch` is capable of claiming (`Conv` and everything else are
`false`). Placement logic in the `oxionnx` workspace crate consults it before
paying for an upload/dispatch/readback; returning `true` is necessary but
not sufficient — an individual node's shape or attributes can still put it
outside what the kernel handles, in which case dispatch itself declines.

## Activation, shadow verification, and strict mode

Acquiring a CUDA device at all, shadow-verifying every dispatched op against
a CPU oracle, and turning a verification mismatch into a hard error are
three independent switches, all opt-in and all off by default:

| Environment variable  | Default | Effect |
|------------------------|---------|--------|
| `OXIONNX_CUDA`         | off     | `CudaContext::try_new()` returns `None` — and every node runs on the CPU — until this is set, even on a machine with a working, otherwise-usable CUDA device. |
| `OXIONNX_CUDA_VERIFY`  | off     | Shadow-compares every CUDA-dispatched op's output, element by element and within tolerance, against a from-scratch CPU oracle (naive loops, `f64` accumulation) before trusting it. A mismatch is treated as a kernel *failure* — the GPU's numbers are discarded, not returned, and the node falls back to the CPU. |
| `OXIONNX_CUDA_STRICT`  | off     | Promotes a shadow-verification mismatch to a hard `CudaError::Verify` instead of a silent CPU fallback. Only has an effect when `OXIONNX_CUDA_VERIFY=1`. |

Any of `1`/`true`/`yes`/`on` (case-insensitive) enables a flag; unset, empty,
`0`/`false`/`no`/`off` disable it; anything else unrecognised is also treated
as enabled, on the theory that a typo is a request for the feature, not
silent old behavior.

`OXIONNX_CUDA_VERIFY` is new in 0.1.5, closing a gap relative to
`oxionnx-directml`, which already shipped the equivalent
`OXIONNX_DIRECTML_VERIFY` — before 0.1.5 this crate had no opt-in
verification gate at all, and several of the correctness bugs fixed in the
same release (below) shipped silently to every CUDA user by default. The
gate itself is real, exercised-by-unit-tests logic: the comparison,
tolerance handling, and per-op oracle formulas are all covered. But this
repository has no CUDA-capable host, so — like the rest of this crate — the
gate has never been run against real silicon in this codebase's own CI.
Turn it on and run it on real hardware, across more than one input shape per
op, before relying on a CUDA build in production.

Fixed in 0.1.5: batch-broadcast `MatMul`/`Gemm` operands, `ReduceSum`/
`ReduceMax` silently truncating to the first 256 elements of the axis,
`Softmax` ignoring the ONNX `axis` attribute, `LeakyRelu`/`HardSigmoid`
ignoring the node's `alpha`/`beta`, and `Gemm`'s bias epilogue dropping bias
for shapes other than `[N]`/`[M,N]`. The dead, permanently-unreachable
`Conv` GPU path (already disabled behind a bare `if true`, since
`oxicuda-dnn`'s conv engines produced wrong numbers) was deleted outright
rather than left in the tree as misleading dead code.

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

// Returns None unless `OXIONNX_CUDA=1` is set in the environment (activation
// is opt-in, see above) -- and also returns None if no CUDA device is
// present. No panic, no unwrap required either way.
if let Some(ctx) = CudaContext::try_new() {
    println!("CUDA device ready: {:?}", ctx.driver_context());
}
```

To bypass the environment-variable gate explicitly, use
`CudaContext::try_new_with(oxionnx_cuda::context::Activation::Enabled)`.

In practice, CUDA dispatch is invoked automatically by `oxionnx::Session` when
the `cuda` feature is enabled, `OXIONNX_CUDA=1` is set, and a compatible GPU
is present. Direct use of `try_cuda_dispatch` is only needed when embedding
the CUDA backend into a custom inference loop.

### Error handling

All CUDA errors are represented by `CudaError` (re-exported from
`CudaDispatchError`). The variants cover driver initialisation failures, BLAS
and DNN operation errors, PTX compilation errors, unsupported configurations,
shape mismatches, and — new in 0.1.5 — a `Verify` variant carrying a
shadow-verification mismatch report, only reachable under
`OXIONNX_CUDA_STRICT=1`. Each variant implements `std::error::Error` and
converts to `OnnxError::Internal` via a `From` impl so the session layer
never needs to handle CUDA errors directly. `CudaError` is
`#[non_exhaustive]`, so an exhaustive downstream `match` needs a wildcard arm.

## Requirements

- Rust 1.75 or later
- NVIDIA GPU with a driver that supports the CUDA runtime used by OxiCUDA
- The OxiCUDA crates (`oxicuda-*`) must be present in the workspace or
  available on crates.io — 0.1.5 depends on `oxicuda-driver`/`-memory`/
  `-blas`/`-dnn`/`-ptx`/`-launch` 0.5.3 (up from 0.1.8)
- `OXIONNX_CUDA=1` set in the environment to activate acquisition at all
  (see [Activation, shadow verification, and strict
  mode](#activation-shadow-verification-and-strict-mode) above)

A missing or incompatible CUDA installation is not a hard error at build time:
the crate compiles on any platform, and `CudaContext::try_new()` returns
`None` at runtime when no device is available, or when it has not been
activated.

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
