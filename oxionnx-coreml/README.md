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

Model metadata (author, license, version, ...) is a one-liner:

```rust,ignore
// continuing from `model` above
let metadata = model.model_metadata()?;
if let Some(version) = metadata.get("version") {
    println!("model version: {version}");
}
```

## Public API

### Loading

* [`MlPackageModel::load`] — load a `.mlpackage` (compiled **once**, then
  served from the persistent compile cache described below) or an
  existing `.mlmodelc` bundle (loaded as-is, nothing to compile).
  Returns `CoreMLError::Io` for missing paths and
  `CoreMLError::Framework` for any `NSError` the framework raises during
  load.
* [`MlPackageModel::ensure_compiled`] — compile a `.mlpackage` into the
  cache and return the resulting `.mlmodelc` path *without* loading it,
  for installers and first-run setup steps that want to pay the compile
  before the first request rather than inside it.
* [`MlPackageModel::load_from_bytes`] — always returns
  `CoreMLError::UnsupportedFormat`.  `.mlpackage`/`.mlmodelc` bundles
  are directory trees, not a single serialized blob, so there is no
  bytes-based loading path; materialize the bundle to disk and call
  [`MlPackageModel::load`] instead.  Provided for API parity with
  `Session::from_bytes`.

#### Compile cache

`MLModel::compileModelAtURL_error:` compiles a `.mlpackage` into a fresh
UUID-named directory under `$TMPDIR` on **every** call, and never reuses
or deletes it.  This crate therefore compiles once and moves the result
into a stable, content-keyed location:

```text
$OXIONNX_COREML_CACHE_DIR                 # if set and non-empty
$HOME/Library/Caches/oxionnx-coreml       # otherwise
<system temp dir>/oxionnx-coreml          # last resort
    └── <bundle stem>-<fingerprint>.mlmodelc
```

The fingerprint folds the relative path, length and mtime of every file
in the source bundle, so replacing a `.mlpackage` produces a new entry
rather than reusing a stale compile.  Entries are self-describing and
the cache is safe to delete wholesale at any time; nothing is pruned
automatically.  Concurrent processes racing on the same key each install
via an atomic rename, so a reader sees either a complete entry or none.
If the cache cannot be used at all (read-only volume, full disk) the
load still succeeds — it falls back to the framework's temporary
directory.

Beyond skipping the compile itself, the stable path is what lets Apple's
own ANE program cache hit across process launches: on an M3, loading
ArcFace + SCRFD + InSwapper costs ~4.3 s from a never-seen path and
~0.14 s once the cache is warm.

### Inference

`predict`, `predict_raw`, and `predict_features` all share the same
input handling: every input `Tensor`'s data is handed to CoreML **by
pointer**, not copied, via `MLMultiArray`'s
`initWithDataPointer_shape_dataType_strides_deallocator_error`
initializer — the constructed array never escapes the synchronous call
that built it, so this is safe without paying for a copy.

* [`MlPackageModel::predict`] — run inference; takes `&self`, so callers
  may share the model behind `Arc` and run concurrent predictions (see
  `tests/concurrent_predict.rs`).  Returns `f32` `Tensor`s for every
  declared `MLMultiArray` output (`Float16` sources are up-converted);
  non-array outputs raise `CoreMLError::MissingOutput` pointing callers
  at `predict_features`.
* [`MlPackageModel::predict_raw`] — same input contract as `predict`,
  but returns `HashMap<String, RawArray>` with the output dtype
  **preserved** (no `Float16` → `f32` up-conversion) — for
  CoreML→CoreML pipelines that need the exact bytes an upstream model
  produced, or where fp16 precision must survive the hop.
* [`MlPackageModel::predict_features`] — same input contract, but
  returns `HashMap<String, FeatureOutput>` covering **every** CoreML
  feature type, not just `MLMultiArray`: images (decoded from a
  `CVPixelBuffer` for the `OneComponent8`, `32BGRA`,
  `OneComponent16Half`, and `OneComponent32Float` pixel formats —
  anything else raises `CoreMLError::UnsupportedPixelFormat`), sequences
  (`Int64`/`String` elements), dictionaries (stringified keys, `f64`
  values), strings, `Int64`s, and `Double`s.
* [`MlPackageModel::warm_up`] — pre-execute one prediction with
  caller-supplied dummy inputs and discard the outputs, paying CoreML's
  first-call specialization cost (kernel JIT, ANE program compile,
  scratch-buffer allocation) up front instead of inside the hot path.

### Introspection

* [`MlPackageModel::input_names`] / [`MlPackageModel::output_names`] —
  the model's declared input/output feature names, sorted
  lexicographically (the underlying `NSDictionary` does not preserve
  insertion order).  Note: `coremltools` rewrites ONNX output names to
  `var_NNNN` during conversion — stable per converted `.mlpackage`, but
  not the original ONNX names.
* [`MlPackageModel::compute_plan_summary`] — diagnostics; reports the
  number of program ops the runtime placed on each compute unit (ANE /
  GPU / CPU / unknown), folded across every function the `MLProgram`
  declares.
* [`MlPackageModel::compute_plan_breakdown`] — the same traversal as
  `compute_plan_summary`, but keyed by each operation's `operatorName`
  (e.g. `gather`, `conv`) instead of folded into one flat total —
  diagnoses *which* op kinds got kicked off the ANE.  Every entry's
  fields sum back to `compute_plan_summary`'s totals for the same
  model.
* [`MlPackageModel::model_metadata`] — `description` / `version` /
  `author` / `license`, plus any creator-defined key/value pairs under
  a `creator.<key>` prefix.  A `.mlpackage` with no metadata returns an
  empty map, never an error.

### Types

* [`MlComputeUnits`] — `CpuOnly` / `CpuAndGpu` / `CpuAndAne` / `All`.
* [`ComputePlanSummary`] — `ane_ops` / `gpu_ops` / `cpu_ops` /
  `unknown_ops` counts, plus `total_ops()`, `compute_ops()`,
  `ane_fraction()`, and `merge()` (the accumulation primitive
  `compute_plan_breakdown` uses so its per-operator totals always
  reconcile with `compute_plan_summary`'s flat totals).
* [`RawArray`] / [`MlArrayDtype`] — `predict_raw`'s dtype-tagged output:
  `shape`, `dtype`, and a tightly packed C-contiguous `data: Vec<u8>`.
  `MlArrayDtype` mirrors `F32` / `F16` / `F64` / `I32` / `I8`, though
  only `F32`/`F16` sources are actually produced today.
* [`FeatureOutput`] / [`SequenceValue`] — `predict_features`'s
  per-output enum (`MultiArray` / `Image` / `Sequence` / `Dictionary` /
  `String` / `Int64` / `Double`) and the `Int64`/`String` element
  payload of a `Sequence`.
* [`CoreMLError`] — `thiserror`-derived error type; `predict_features`
  adds two variants, `UnsupportedFeatureType` (no portable
  representation, e.g. `MLFeatureTypeState`) and
  `UnsupportedPixelFormat` (an image output outside the four supported
  pixel formats).

## Platform support

Implemented for macOS, iOS, tvOS, and visionOS
(`#[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos"))]`)
— every Apple platform CoreML itself ships on, gated behind a single
`cfg`.  Validation differs by platform, though:

* **macOS** is the primary platform this crate's own test suite
  actually executes on (26 tests run by default — see
  [Tests](#tests) below).
* **iOS** (`aarch64-apple-ios`) is compile-validated directly by this
  crate — `cargo build`, `cargo clippy`, and `cargo test --no-run` all
  succeed for that target — but the compiled tests are not *executed*
  here (that needs an iOS simulator or device).
* **tvOS** and **visionOS** share the identical CoreML API surface via
  the same `cfg` gate, but are Rust tier-3 targets (nightly +
  `-Zbuild-std`) that could not be independently compile-validated on
  this development machine (see `TODO.md`) — treat them as "supported
  via the same code path," not "tested."

The crate compiles on every other target (Linux, Windows, wasm, …) as a
stub that surfaces `CoreMLError::UnsupportedPlatform` from every entry
point, so downstream code that conditionally enables this provider does
not need to fence its own code paths.

## Tests

43 tests run by default.  8 additional tests need a real `.mlpackage`
and are gated with `#[ignore]` so the workspace test run does not
require model files.  Every one of them reads its bundle path from the
`OXIONNX_COREML_TEST_MODEL` environment variable and skips gracefully
(prints a note, does not fail) when it is unset or points nowhere:

```bash
OXIONNX_COREML_TEST_MODEL=/path/to/w600k_r50.mlpackage \
    cargo test -p oxionnx-coreml -- --ignored --test-threads=1
```

They cover the ArcFace load/predict/metadata/compute-plan round trips
(`src/package/tests.rs`), the concurrency stress test in
`tests/concurrent_predict.rs` (8 threads × 20 `predict` calls against
one shared `Arc<MlPackageModel>`), and the compile-cache end-to-end test
in `tests/compile_cache.rs` (two consecutive resolutions compile once,
produce one cache entry, and leave nothing in `$TMPDIR`).

The non-ignored tests cover the stride-normalizing output readers
against deliberately padded `MLMultiArray` layouts — including SCRFD's
exact `[N, 1] / [N, 4] / [N, 10]` shapes with 32-element row padding —
cross-checked against a naive reference gather; the compile cache's
keying, race resolution, eviction and degradation paths;
`predict_features`'s dispatch across every supported CoreML feature type
(including all four pixel formats); `model_metadata`'s defensive
string-extraction helpers; the
`compute_plan_summary`/`compute_plan_breakdown` merge and reconciliation
logic; the no-copy/owned-array deallocator paths; error mapping for
missing files; and the documented `load_from_bytes` rejection.

## License

Apache-2.0.  Copyright COOLJAPAN OU (Team Kitasan).
