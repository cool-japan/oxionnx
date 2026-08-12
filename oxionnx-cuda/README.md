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
| Convolution      | Conv — NCHW forward, dispatched *directly* to one of three `oxicuda-dnn` engines: `Conv1x1` (1x1 filter, unit stride and dilation, no padding), `DepthwiseConv` (`groups == in_channels == out_channels`), otherwise `ImplicitGemmConv` (arbitrary stride, dilation and grouping). Deliberately does **not** go through `oxicuda_dnn::conv::api::conv_forward`, whose auto-selector can route into the Winograd path. Asymmetric `pads` declines to the CPU rather than silently computing with one side's padding value — ONNX allows `[top, left, bottom, right]` to differ per side, `ConvProblem` carries one value per spatial dimension — as do non-4-D shapes and group/channel mismatches. |
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
allocation-free predicate reporting exactly which of the 26 ops above
`try_cuda_dispatch` is capable of claiming (everything else is `false`).
Placement logic in the `oxionnx` workspace crate consults it before
paying for an upload/dispatch/readback; returning `true` is necessary but
not sufficient — an individual node's shape or attributes can still put it
outside what the kernel handles, in which case dispatch itself declines.
`Conv` is the sharpest example: it is advertised, and a convolution with
asymmetric padding is still declined at dispatch time.

## Session-lifetime caches

A `CudaContext` is built once per `oxionnx::Session` and shared by every
dispatch that session makes, so it is also where anything that should outlive a
single node lives:

* **A device-buffer pool.** Operand and output buffers are borrowed from a
  size-classed free list and returned when the dispatch ends, instead of a
  `cuMemAlloc`/`cuMemFree` pair per operand per node per frame.
* **Weight residency.** A `MatMul`/`Gemm` operand or a `Conv` filter/bias that
  resolves out of the session's *initializer* map — not out of this run's
  intermediates — is uploaded once and reused thereafter. Bytes that never
  change stop crossing the bus every frame.
* **Compiled PTX modules.** The elementwise and softmax kernels this crate
  generates itself are JIT-compiled once per context rather than once per
  dispatch.

Every copy also rides the same stream as the kernel that consumes it, so a
dispatch performs exactly one host/device fence instead of one per operand.

Measured on an RTX A4000 (sm_86, driver 550.144.03, CUDA 12.4) with
`cargo run -p oxionnx-cuda --features gpu-tests --release --example
dispatch_bench`, steady state, median of three runs:

| dispatch                                    | before   | after   |       |
|---------------------------------------------|---------:|--------:|------:|
| batched MatMul, `[4,64,128] x [4,128,64]`    | 1.66 ms  | 0.10 ms | 17x   |
| batched MatMul, `[16,64,128] x [16,128,64]`  | 5.89 ms  | 0.34 ms | 17x   |
| batched MatMul, `[4,256,256] x [4,256,256]`  | 8.32 ms  | 0.68 ms | 12x   |
| ArcFace head, `[1,25088] x [25088,512]`      | 43.65 ms | 5.56 ms | 7.9x  |
| broadcast batch, `[8,64,128] x [128,64]`     | 2.93 ms  | 0.58 ms | 5.0x  |
| InSwapper AdaIN, `[1,512] x [512,2048]`      | 1.04 ms  | 0.31 ms | 3.4x  |
| Conv 3x3, 64ch at 64x64, with bias           | 1.08 ms  | 0.65 ms | 1.7x  |
| Softmax `[1024,512]`                         | 1.27 ms  | 0.70 ms | 1.8x  |
| ReduceSum `[1024,512]` axis 1                | 0.54 ms  | 0.22 ms | 2.5x  |

The three batched-MatMul rows are the batched-dispatch change (one
upload/launch/readback for the whole batch, replacing one *complete* round trip
per batch slice); the rest is the pooling, residency and stream ordering above.

Two consequences worth knowing:

* **A session holds device memory between runs.** Resident weights plus idle
  pooled buffers; `CudaContext::cached_device_bytes()` reports how much, and
  `CudaContext::release_device_caches()` frees all of it (the next dispatch
  simply re-allocates and re-uploads). The pool is bounded — buffers beyond its
  budget are freed on check-in rather than retained.
* **Initializers must actually be invariant.** `try_cuda_dispatch` treats a
  name found in `weights` as denoting the same bytes for as long as the context
  lives, which is what `oxionnx::Session` guarantees by construction. A direct
  caller that swaps weight maps under one context should call
  `release_device_caches()` in between; see `try_cuda_dispatch`'s own docs for
  the check that usually catches it anyway.

`CudaContext::cache_counters()` reports what the caches did. The number that
matters is `weight_bytes_uploaded`: as a delta across a steady-state frame it
must be **zero**, and `tests/batched_matmul_gpu.rs` asserts exactly that.

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
`oxicuda-dnn`'s conv engines produced wrong numbers *at the time*) was
deleted outright rather than left in the tree as misleading dead code.
That verdict on `oxicuda-dnn` no longer holds and this paragraph is history,
not current status: `Conv` has since been re-implemented as the direct
`Conv1x1`/`DepthwiseConv`/`ImplicitGemmConv` dispatch described above, each
engine validated on real hardware against an independent CPU oracle, and is
advertised by `is_supported_op`. See the "Accelerated operators" table.

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
- `oxionnx-directml` — DirectML (Windows D3D12) execution provider
- `oxionnx-coreml` — Apple CoreML execution provider
- `oxionnx` — top-level session API

## License

Licensed under `Apache-2.0`.

Copyright COOLJAPAN OU (Team Kitasan).
