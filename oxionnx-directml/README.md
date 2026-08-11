# oxionnx-directml

DirectML / Direct3D 12 execution provider for [OxiONNX](https://github.com/cool-japan/oxionnx).

On Windows it dispatches a small set of ONNX nodes to the GPU. On every other platform
it is a fully-typed, zero-overhead no-op: `DirectMLContext::try_new()` returns `None`,
`try_directml_dispatch()` returns `Ok(None)`, and the CPU path handles everything.

---

## ⚠️ Verification status — read this first

This crate has **never been executed on a GPU**, and there is no Windows host or D3D12
device in this project's CI. Nothing below is "supported" in the sense of "we ran it and
it produced the right numbers". Be precise about what is and is not known:

| Layer | Status |
|---|---|
| Shape validation, dispatch grids, buffer sizing, DML descriptor math (`plan.rs`, `layout.rs`) | ✅ **Unit-tested on Linux.** Part of the crate's 242 tests, incl. property tests and a CPU-oracle cross-check against `oxionnx-ops`. |
| HLSL sources, root-constant layouts, cbuffer field order (`hlsl.rs`) | ✅ **Structurally tested** (cbuffer width, field order, entry points, index recovery, bounds guards). The HLSL is **never parsed** by anything in CI. |
| Windows FFI glue (`backend/d3d12/*`, `backend/dml/*`) | ⚠️ **Compile- and lint-checked only**, via `cargo clippy --target x86_64-pc-windows-gnu -- -D warnings`. Never linked, never run. |
| Actual GPU execution, numerical correctness, barrier completeness, COM refcount balance | ❌ **Unverified.** See "The hardware gate" below. |

**Do not read the operator table below as "this works."** Read it as "this is implemented
and type-checks; the numbers have never been checked against a GPU."

Because of this, **the GPU path is off by default even on Windows.** You must opt in with
`OXIONNX_DIRECTML=1` (or `SessionBuilder::with_directml(true)`). See "Activation" below for
why: a GPU kernel bug does not crash, it returns plausible-looking wrong numbers.

### The hardware gate

`DirectMLContext::self_check()` runs the real GPU path on fixed inputs and diffs every
result against the CPU oracle in `reference.rs`, element by element. It is the **only**
mechanism that can validate this crate's Windows-only code.

```console
set OXIONNX_DIRECTML=1
cargo run -p oxionnx-directml --example directml_self_check

REM On a Windows VM with no GPU — the only place this code can be exercised without
REM hardware.  WARP is Microsoft's *conformant* software D3D12 implementation, so it
REM catches a wrong index, a bad root-signature slot or a malformed tensor descriptor just
REM as well as silicon does.  What it cannot catch is the class of bug that is correct on
REM one vendor's part and garbage on another's — a missing UAV barrier being the classic.
set OXIONNX_DIRECTML_ALLOW_WARP=1
cargo run -p oxionnx-directml --example directml_self_check
```

Exit codes: `0` pass · `1` **the GPU returned wrong numbers** · `2` no context (nothing was
tested — *not* a pass) · `3` the GPU path failed before producing anything to compare.

If you have a Windows machine with a D3D12 GPU: run it, and paste the report into the PR.
Until someone does, this crate's GPU path is *unproven*.

---

## Activation

Every switch is off by default, and each defaults off for a reason.

| Variable | Default | Effect |
|---|---|---|
| `OXIONNX_DIRECTML=1` | **off** | Acquire a GPU at all. Without it `DirectMLContext::try_new()` returns `None`, the session holds `dml: None`, and every node runs on the CPU. `SessionBuilder::with_directml(true)` is the programmatic equivalent. |
| `OXIONNX_DIRECTML_VERIFY=1` | off | Shadow-compare **every** dispatched op against the CPU oracle. A mismatch is treated as a kernel *failure*: the wrong numbers are **discarded**, not returned. Roughly doubles the cost of every claimed node — a diagnostic mode, not a production one. |
| `OXIONNX_DIRECTML_STRICT=1` | off | A kernel *failure* becomes a hard `Err` instead of a silent CPU fallback. A *declined* op still falls back — declining is not failing. |
| `OXIONNX_DIRECTML_ALLOW_WARP=1` | off | Permit the software (WARP) adapter. Skipped by default: WARP is a CPU rasteriser, so a "GPU" backend silently running on it would be *slower* than the tuned CPU kernels it exists to beat. |

All four accept `1`/`true`/`yes`/`on`; anything unrecognised is treated as **enabled**, on
the grounds that a user who typed `=please` has asked for the feature and silently ignoring
them would hand back a false all-clear.

### Declined vs. failed

The distinction is load-bearing, and conflating it is how a dead GPU masquerades as a
working one:

| Outcome | Meaning | Router |
|---|---|---|
| `Ok(Some(t))` | the GPU computed it | returned |
| `Declined` | **not ours** — this op/shape is outside what the backend expresses. Normal and expected. | `Ok(None)` → CPU, logged at `debug` |
| `ShapeMismatch` | **your model is broken** — the CPU op would fail too | `Ok(None)` → CPU, which raises the real diagnostic |
| anything else | **the GPU broke** | logged at `error!`, then `Ok(None)` → CPU — or `Err` under `OXIONNX_DIRECTML_STRICT` |

A genuine failure is never silent. Inference stays correct (the CPU runs the node), but it
is reported at `error!` every time, because "your GPU provider has been dead since process
start" is not something a user should have to infer from a stopwatch.

---

## Architecture: two backends, DirectML first

```
DirectMLContext                       (Send + Sync, Mutex-guarded)
  └─ Mutex<Backend>
       ├─ D3d12Core     device + COMPUTE queue + allocator + list + fence + event
       └─ Engine
            ├─ DmlEngine    IDMLDevice — genuine DirectML operators
            └─ HlslEngine   D3DCompile'd compute shaders — the fallback
```

`try_new()` builds the shared D3D12 core, then tries `IDMLDevice`; if `DirectML.dll` is
not present it falls back to compiling the HLSL in `hlsl.rs` at run time with
`D3DCompile` (FXC, `cs_5_0` — inbox on every supported Windows, no SDK needed).

`DMLCreateDevice` is resolved with `LoadLibraryW` + `GetProcAddress`, **never statically
linked**. A static import is resolved by the loader at *process start*, so a missing
`DirectML.dll` would make the host process fail to **launch** — and the HLSL fallback,
which exists precisely for that case, would be unreachable.

### Almost all the logic is platform-neutral

| Module | Compiled on | Owns |
|---|---|---|
| `plan.rs` | every target | shape validation, dispatch-grid math, buffer sizing, root constants |
| `layout.rs` | every target | DML tensor descriptors, strides, `TotalTensorSizeInBytes`, the operator cache key |
| `reference.rs` | every target | the CPU oracle the GPU path is diffed against |
| `hlsl.rs` | every target | the shader sources |
| `backend/*` | Windows only | thin FFI glue, and nothing else |

That split is deliberate: it is what makes a Linux box a useful place to find bugs in a
Windows GPU backend.

---

## Enabling

```toml
[dependencies]
oxionnx = { version = "0.1.6", features = ["directml"] }
```

## Usage

The provider is wired in by the session; you do not construct it yourself.

```rust,no_run
use oxionnx::Session;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// With the `directml` feature enabled, the session acquires a DirectML context on Windows
// *when the user has opted in* (`OXIONNX_DIRECTML=1`), and falls back to CPU everywhere and
// everywhen else.  The GPU path is off by default — see "Activation".
let session = Session::load("model.onnx")?;
# Ok(())
# }
```

To query what the provider will claim, without building a context:

```rust
use oxionnx_directml::is_supported_op;
use oxionnx_core::graph::OpKind;

assert!(is_supported_op(&OpKind::MatMul));
assert!(!is_supported_op(&OpKind::Identity));
```

---

## Claimed operators

`is_supported_op` returns `true` for exactly these, and the router claims exactly these.
**"Claimed" is not "verified"** — every row here is implemented and type-checked; not one
has been run against a GPU. See the verification status at the top of this file.

| Operator | Accepted shapes | Otherwise |
|---|---|---|
| `MatMul` | **2-D × 2-D only** | declined → CPU |
| `Gemm` | **2-D × 2-D**, α/β/transA/transB, optional broadcast `C` | declined → CPU |
| `Add` | **identical shapes only** | declined → CPU |
| `Sub` | **identical shapes only** | declined → CPU |
| `Mul` | **identical shapes only** | declined → CPU |
| `Div` | **identical shapes only** | declined → CPU |
| `Relu` | rank ≤ 4, non-empty | declined → CPU |
| `Sigmoid` | rank ≤ 4, non-empty | declined → CPU |
| `Tanh` | rank ≤ 4, non-empty | declined → CPU |
| `Softmax` | **single axis**, default `-1` (opset-13 semantics) | declined → CPU |
| `ReduceSum` | **single resolved axis only** (empty/multi-axis `axes` declines), `keepdims` default `1` | declined → CPU |
| `ReduceMean` | **single resolved axis only** (empty/multi-axis `axes` declines), `keepdims` default `1` | declined → CPU |
| `ReduceMax` | **single resolved axis only** (empty/multi-axis `axes` declines), `keepdims` default `1` | declined → CPU |
| `ReduceMin` | **single resolved axis only** (empty/multi-axis `axes` declines), `keepdims` default `1` | declined → CPU |
| `Conv` | **rank-4 (2-D) only**, `auto_pad=NOTSET`, optional bias `B`, strides/dilations/pads/group at ONNX defaults (1/1/0/1) | declined → CPU |

Every operand must be non-empty (`numel() > 0`) — a `[0, 128]` tensor is routine after an
empty batch, and `CreateCommittedResource` with `Width = 0` fails outright.

### Why the restrictions are this tight

A *declined* op is not a failure. It returns `Ok(None)`, and `oxionnx-ops`' tuned CPU
kernel runs and produces the right answer. Declining is free and correct; guessing is not.

* **MatMul is 2-D × 2-D.** ONNX `MatMul` allows N-D operands with broadcast batch dims,
  and the shape math for that is easy — which is the trap. `MATMUL_HLSL` indexes
  `A[AOff + row*K + k]`, and until a backend is *shown on hardware* to walk the batch
  offsets correctly, accepting `[8,128,64] × [64,32]` means returning a tensor of the
  wrong shape filled with the wrong numbers. The CPU kernel one line away gets it right.
* **The binary ops require identical shapes.** Broadcasting is declined even when the
  operands are perfectly broadcastable. The kernels are index-parallel `C[i] = A[i] ⊕ B[i]`
  with no notion of a shape; dispatching `max(a.numel(), b.numel())` threads over a
  `[2,3,4]` + `[1,4]` pair reads past the end of the 4-element operand and returns a
  correctly-*shaped* tensor full of garbage. No bounds check fires. No shape-only test
  catches it.

Both restrictions are enforced in `plan.rs`, on every platform, with tests that pin them.
Lifting either one means implementing the broadcast (both mechanisms — CPU expansion and
DirectML 0-strides — are already written and tested in `plan.rs` / `layout.rs`), *and*
running `self_check` on real hardware. In that order.

The nearest still-declined neighbours are `Identity`, `LogSoftmax`, `ReduceProd` and
`ConvTranspose` — each one attribute or operator-descriptor away from something already
claimed, which is exactly why `is_supported_op_does_not_over_claim` names each of them
individually:

* **`Identity`** has no dispatch arm at all — there is no kernel to call, let alone decline.
* **`LogSoftmax`** is a distinct DirectML activation, not `Softmax` followed by a CPU
  `log` — no plan expresses it yet.
* **`ReduceProd`** would be a fifth `ReduceKind`; the enum, and the router's match arm,
  carry only `Sum`, `Mean`, `Max`, `Min`.
* **`ConvTranspose`** needs a transposed-convolution descriptor. `ConvPlan` and the
  DirectML op builder only construct `DML_CONVOLUTION_OPERATOR_DESC` with
  `DML_CONVOLUTION_DIRECTION_FORWARD`.

Each stays unclaimed until it gets its own plan and router arm, and — like every op
above — a `self_check` run against real hardware.

---

## Platform notes

* **Windows**: `try_new()` attempts D3D12 device acquisition and returns `Some(ctx)` only
  if it succeeds. A machine with no D3D12 adapter returns `None` and runs on the CPU —
  that is not an error.
* **macOS / Linux / WASM**: `try_new()` always returns `None`. `is_active()` is a
  monomorphic `false`, so `try_directml_dispatch` folds away entirely.
* The `windows` dependency is target-gated
  (`[target.'cfg(target_os = "windows")'.dependencies]`) and is FFI to the OS, not a
  C/C++ build dependency. The crate remains pure Rust — nothing here invokes a C compiler.

## Development

```console
# Both targets must be green.  The second one is where Windows bugs are found.
cargo clippy -p oxionnx-directml --all-targets -- -D warnings
cargo clippy --target x86_64-pc-windows-gnu -p oxionnx-directml --all-targets -- -D warnings
cargo nextest run -p oxionnx-directml
```

`clippy --target x86_64-pc-windows-gnu` type-checks and lints every
`#[cfg(target_os = "windows")]` line without a linker or a Windows host. It is this
crate's primary gate, and it is a genuinely strong one — it is how the missing
`Win32_Security` feature (without which the crate did not compile for Windows *at all*)
was found.

## License

Apache-2.0
