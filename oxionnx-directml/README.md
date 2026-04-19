# oxionnx-directml

DirectML execution provider for OxiONNX — Windows D3D12 GPU backend.

This crate provides a feature-gated DirectML execution provider that dispatches
selected ONNX operators to the Windows Direct3D 12 compute pipeline. On all other
platforms it compiles as a transparent no-op: `DirectMLContext::try_new()` returns
`None` and `try_directml_dispatch()` returns `Ok(None)`, allowing the CPU path to
handle all operations.

## Architecture

```
oxionnx (root)
  └── oxionnx-directml  (feature = "directml")
       ├── context.rs   — D3D12 device + command queue init (Windows only)
       ├── dispatch.rs  — per-op routing to HLSL compute kernels
       └── kernels/
            ├── matmul.rs     — MatMul HLSL shader dispatch
            └── elementwise.rs — Add, Mul, Relu, Sigmoid HLSL kernels
```

## Enabling the Feature

```toml
[dependencies]
oxionnx = { version = "0.1.2", features = ["directml"] }
```

## Usage

```rust
use oxionnx::{Session, ExecutionProvider};
use oxionnx::execution_providers::DirectMLExecutionProvider;

// Build a session that prefers DirectML on Windows (falls back to CPU elsewhere)
let session = Session::builder()
    .with_execution_provider(DirectMLExecutionProvider::default().build())
    .load("model.onnx".as_ref())?;
```

## Supported Operators (v0.1.2)

| Operator | Status |
|----------|--------|
| MatMul   | Scaffold (HLSL defined; kernel binding v0.1.6) |
| Add      | Scaffold (HLSL defined; kernel binding v0.1.6) |
| Mul      | Scaffold (HLSL defined; kernel binding v0.1.6) |
| Relu     | Scaffold (HLSL defined; kernel binding v0.1.6) |
| Sigmoid  | Scaffold (HLSL defined; kernel binding v0.1.6) |

All unsupported operations fall back transparently to the CPU execution path.

## Platform Notes

- **Windows**: `DirectMLContext::try_new()` initializes a D3D12 device and returns `Some(ctx)`.
- **macOS / Linux / WASM**: `DirectMLContext::try_new()` always returns `None`; the entire crate compiles as a no-op with zero overhead.
- The `windows` crate dependency is target-gated: `[target.'cfg(target_os = "windows")'.dependencies]`.

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
