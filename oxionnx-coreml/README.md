# oxionnx-coreml

Apple **CoreML** execution provider for the OxiONNX inference engine —
load pre-converted `.mlpackage` (or `.mlmodelc`) bundles and run inference
through Apple's whole-graph scheduler (Apple Neural Engine, integrated
GPU, or CPU).

This is a **thin runtime crate**, not a per-op kernel registry.  CoreML's
whole-graph optimiser is what produces the headline speedups; per-op
dispatch would give them back.

| Model       | Speedup vs CPU (M3) | ANE engagement |
| :---------- | :------------------ | :------------- |
| ArcFace     | 17.4×               | 97 %           |
| SCRFD       | 25.3×               | 97 %           |
| InSwapper   | 7.91×               | 100 %          |

## Quick start

```rust,ignore
use std::collections::HashMap;
use oxionnx_coreml::{MlPackageModel, MlComputeUnits};
use oxionnx_core::Tensor;

let model = MlPackageModel::load(
    "/tmp/w600k_r50.mlpackage",
    MlComputeUnits::All,
)?;

let mut inputs = HashMap::new();
inputs.insert(
    model.input_names().into_iter().next().unwrap(),
    Tensor::new(vec![0.0_f32; 3 * 112 * 112], vec![1, 3, 112, 112]),
);

let outputs = model.predict(&inputs)?;
let summary = model.compute_plan_summary()?;
println!("ANE engagement: {:.1}%", summary.ane_fraction() * 100.0);
```

## Public API

* [`MlPackageModel::load`] — load a `.mlpackage` (compiled at load time)
  or an existing `.mlmodelc` bundle.  Returns `CoreMLError::Io` for
  missing paths and `CoreMLError::Framework` for any `NSError` the
  framework raises during load.
* [`MlPackageModel::predict`] — run inference; takes `&self`, so callers
  may share the model behind `Arc` and run concurrent predictions.
* [`MlPackageModel::compute_plan_summary`] — diagnostics; reports the
  number of program ops the runtime placed on each compute unit.
* [`MlComputeUnits`] — `CpuOnly` / `CpuAndGpu` / `CpuAndAne` / `All`.
* [`CoreMLError`] — `thiserror`-derived error type.

## Platform support

Implemented for macOS (`#[cfg(target_os = "macos")]`).  The crate compiles
on every other target as a stub that surfaces
`CoreMLError::UnsupportedPlatform` from every entry point.

## Tests

The unit tests that hit the framework load a real `.mlpackage` from the
filesystem and are gated with `#[ignore]` so the workspace test run does
not require model files.  To run them manually:

```bash
# Convert the OxiFace ArcFace model to /tmp/w600k_r50.mlpackage first.
cargo test -p oxionnx-coreml -- --ignored --test-threads=1
```

The non-ignored tests cover the platform stub, error mapping for missing
files, and the documented `load_from_bytes` rejection.

## License

Apache-2.0.  Copyright COOLJAPAN OU (Team Kitasan).
