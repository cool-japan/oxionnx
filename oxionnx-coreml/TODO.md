# oxionnx-coreml — TODO

Tracker for known gaps in the initial 0.1.3 release. Every tracked item
below is now resolved as of 0.1.5 (see each item's own planned/resolved
date) — `predict_raw`, `predict_features`, `compute_plan_breakdown`,
`model_metadata`, the concurrency stress test, and iOS/tvOS/visionOS
support all shipped this release. 0.1.6 then shipped further work that
was never tracked here as a discrete item — `MlPackageModel::ensure_compiled`,
the persistent on-disk compile cache (`package/compile_cache.rs`, fixing
the every-load `$TMPDIR` recompile-and-leak), and the single-pass
`array_read.rs` output extraction (stride walk and dtype conversion fused
into one `getBytesWithHandler:` pass) — see `CHANGELOG.md`'s `[0.1.6]`
section for the full writeup. Remaining open ideas live under
[Proposed follow-ups](#proposed-follow-ups) below.

## I/O performance

- [x] Switch input construction to the no-copy
      `MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error`
      path.  The current implementation calls
      `initWithShape_dataType_error` and `copy_nonoverlapping`s the slice
      into CoreML-owned storage; for large inputs (e.g. 1×3×640×640 SCRFD
      = 4.9 MB) the copy is a meaningful fraction of the per-frame budget.
      The wrinkle is that the deallocator block needs to keep the source
      `Vec<f32>` alive for the lifetime of the `MLMultiArray` — design
      a `block2::RcBlock`-based holder that takes ownership of the
      buffer and frees it from the deallocator callback. (planned 2026-07-09)
  - **Goal:** `predict()` stops copying each input into CoreML-owned storage; inputs are handed to CoreML by pointer.
  - **Design:** Replace `initWithShape_dataType_error` + `copy_nonoverlapping` in `multi_array_from_f32` with `initWithDataPointer_shape_dataType_strides_deallocator_error`, building a C-contiguous strides `NSArray<NSNumber>` alongside the existing shape array. Two correct sub-cases: (a) in-`predict`, the array never escapes the call (consumed synchronously by `predictionFromFeatures_error` and dropped before return) — pass a pointer into the borrowed `&[f32]` with `deallocator: None` since CoreML never owns/frees it; (b) a new `multi_array_from_owned(data: Vec<f32>, shape)` moves the `Vec` into a `block2::RcBlock::new(move |_ptr| drop(owned))` and passes `Some(&*rc)` for cases where the array must outlive the source (used by `predict_raw`).
  - **Files:** `oxionnx-coreml/src/package.rs` (`multi_array_from_f32`, new `multi_array_from_owned`).
  - **Prerequisites:** none.
  - **Tests:** `multi_array_from_owned` round-trip (build from Vec, read back via `getBytesWithHandler`, assert values); deallocator-fires test (`Arc<AtomicBool>` captured in the RcBlock, drop the array, assert set); strides-builder pure-logic unit test. No model file needed.
  - **Risk:** use-after-free if the borrowed-pointer array escapes `predict()` — mitigated by keeping the borrowed-pointer variant strictly non-escaping and documenting the SAFETY invariant; the owned/RcBlock variant is used for anything that must outlive the call.

- [x] `predict` currently up-converts Float16 outputs to f32 inside the
      runtime.  When a downstream consumer wants the raw fp16 (e.g. for
      another CoreML model in a pipeline), expose a `predict_raw` variant
      that returns `MLMultiArray` handles directly. (planned 2026-07-09)
  - **Goal:** a `predict` variant that preserves output dtype (no Float16→f32 up-conversion) for CoreML→CoreML pipelines.
  - **Design:** add a portable `RawArray { shape: Vec<usize>, dtype: MlArrayDtype, data: Vec<u8> }` (`MlArrayDtype`: F32/F16/I32/I8/F64) defined outside the macOS cfg split so the stub compiles too. Add `predict_raw(&self, inputs) -> Result<HashMap<String, RawArray>>`. Extract the stride/contiguity logic already in `tensor_from_multi_array` into a shared `read_raw_bytes` helper reused by both `predict` (bytes + f32 convert) and `predict_raw` (bytes as-is, dtype preserved). Chose portable bytes over leaking `Retained<MLMultiArray>` to keep objc2 out of the cross-platform public API; a zero-copy handle variant is noted as future work.
  - **Files:** `oxionnx-coreml/src/package.rs` (`RawArray`, `MlArrayDtype`, `predict_raw`, `read_raw_bytes` refactor), `oxionnx-coreml/src/lib.rs` (re-export), stub twin for `predict_raw`/`RawArray`.
  - **Prerequisites:** none.
  - **Tests:** `#[ignore]` model-driven test (env-var `OXIONNX_COREML_TEST_MODEL` path, graceful skip if unset/missing) asserting an fp16 output keeps `dtype==F16` and correct byte length; unit test that `RawArray` for an f32 output matches `predict()`'s f32 tensor bytes.
  - **Risk:** dtype coverage beyond F32/F16 — error cleanly via `UnsupportedOutputDtype`, same as today.

## Functional gaps

- [x] No support for non-`MLMultiArray` features yet.  `MLImage`,
      `MLSequence`, and `MLDictionary` features will surface as
      `CoreMLError::MissingOutput` (the `multiArrayValue()` getter
      returns nil for non-array features).  The SCRFD / ArcFace /
      InSwapper sub-gates are all multi-array, so this is non-blocking
      for OxiFace. (planned 2026-07-09)
  - **Goal:** image/sequence/dictionary/scalar outputs no longer mis-surface as `MissingOutput`; expose them through a typed enum.
  - **Design:** new portable `FeatureOutput { MultiArray(Tensor), Image(Tensor), Sequence(SequenceValue), Dictionary(HashMap<String,f64>), String(String), Int64(i64), Double(f64) }` (+ `SequenceValue::{Int64(Vec<i64>), String(Vec<String>)}`). Add `predict_features(&self) -> Result<HashMap<String, FeatureOutput>>` dispatching on `fv.r#type()` (`MLFeatureType`). Image path decodes standard `CVPixelBuffer` formats (OneComponent8, 32BGRA, OneComponent16Half/Float) into a `Tensor`, erroring clearly on exotic formats. `predict()`'s existing `multiArrayValue()`-nil error message now points to `predict_features()`. `predict()`'s return type is unchanged (array-only, non-breaking).
  - **Files:** `oxionnx-coreml/src/package.rs` (enum + `predict_features` + extractors), `oxionnx-coreml/src/error.rs` (optional `UnsupportedFeatureType` variant), `oxionnx-coreml/src/lib.rs` (re-exports), stub twins.
  - **Prerequisites:** none.
  - **Tests:** construct synthetic `MLFeatureValue`s directly (`featureValueWithInt64`/`Double`/`String`/`MultiArray`, dictionary/sequence constructors) — no model needed — assert correct `FeatureOutput` dispatch. Image path `#[ignore]` (needs a real image-output model).
  - **Risk:** image-format matrix is open-ended — bounded to standard formats + explicit error otherwise. No current OxiFace consumer needs non-array outputs; this is completeness work, not a live blocker.

- [x] `compute_plan_summary` only counts the `main` function.
      `MLProgram` models can declare additional functions
      (e.g. for stateful submodels); they are silently ignored.  Audit
      whether any of the OxiFace targets ship multi-function programs.
      (resolved 2026-07-09: the silent-ignore bug is fixed as part of the
      `compute_plan_breakdown` work in Diagnostics below — all program
      functions are now iterated, not just `main`. The OxiFace-specific
      audit sub-clause is a separate investigation needing external model
      artifacts; split out to Proposed follow-ups.)

- [x] `load_from_bytes` returns `UnsupportedFormat`.  We could implement
      it by extracting a temporary directory and calling `load`, but the
      `.mlpackage` bundle is intrinsically a directory tree — bytes-only
      loading would require the caller to serialize that bundle first.
      Document this clearly rather than papering over the limitation. (planned 2026-07-09)
  - **Goal:** the limitation is documented clearly on the API itself, per the item's own resolution — no behavior change.
  - **Design:** expand the doc comment on `load_from_bytes` explaining the directory-bundle constraint and pointing callers to `load(path)`; sharpen the `UnsupportedFormat` message text accordingly.
  - **Files:** `oxionnx-coreml/src/package.rs` (`load_from_bytes` doc + error message).
  - **Prerequisites:** none.
  - **Tests:** existing `load_from_bytes_returns_unsupported_format` test continues to cover this; no new test needed (doc-only change).
  - **Risk:** none — no behavior change.

## Diagnostics

- [x] `ComputePlanSummary` only returns a 4-bucket histogram.  The
      spike examples additionally report per-`operatorName`
      breakdowns that diagnose *which* op kinds got kicked off the ANE
      (e.g. `gather`, `scatter_along_axis`).  Add a richer
      `compute_plan_breakdown` that returns
      `HashMap<String, ComputePlanSummary>` keyed by operator name. (planned 2026-07-09)
  - **Goal:** per-operator device-placement breakdown, and stop silently ignoring non-`main` `MLProgram` functions (folds in the code portion of the "compute_plan_summary only counts main" item above — see that item's resolution note).
  - **Design:** `operatorName` is already read in the compute-plan traversal but discarded. Refactor the per-op classification in `compute_plan_summary` into a shared routine that accumulates into both the flat `ComputePlanSummary` and a new `HashMap<String, ComputePlanSummary>` keyed by op name. Add `compute_plan_breakdown(&self) -> Result<HashMap<String, ComputePlanSummary>>`. Replace the hard-coded `"main"`-only function lookup with iteration over all keys of `program.functions()`, so multi-function `MLProgram` models are fully counted in both the summary and the breakdown. The async `MLComputePlan`/`PlanSlot`/`Condvar` marshaling is left untouched — only the post-load traversal changes.
  - **Files:** `oxionnx-coreml/src/package.rs` (`compute_plan_summary` refactor + `compute_plan_breakdown`), `oxionnx-coreml/src/compute.rs` (helper, if warranted), `oxionnx-coreml/src/lib.rs`, stub twin.
  - **Prerequisites:** none.
  - **Tests:** `#[ignore]` model-driven test (env-var path, graceful skip) asserting breakdown keys are a subset of real op names and per-op counts sum to the flat summary's totals; pure-logic test of the accumulator merge if extracted as a standalone function.
  - **Risk:** async bridge is delicate — scope changes strictly to the post-load traversal, not the loading/marshaling code.

- [x] No way to surface MLModel metadata (model author, license,
      version, custom user-defined keys) yet.  Add a
      `model_metadata() -> HashMap<String, String>` method that walks
      `MLModelDescription::metadata`. (planned 2026-07-09)
  - **Goal:** `model_metadata(&self) -> Result<HashMap<String, String>>` exposing author/license/version/description/creator-defined metadata.
  - **Design:** `MLModelDescription::metadata()` returns `NSDictionary<MLModelMetadataKey, AnyObject>` — values are `AnyObject`, not guaranteed `NSString`. For the 4 string keys (`MLModelAuthorKey`, `MLModelLicenseKey`, `MLModelVersionStringKey`, `MLModelDescriptionKey`) look up and downcast defensively to `NSString` (never `unwrap`), inserting under stable snake keys (`author`/`license`/`version`/`description`). `MLModelCreatorDefinedKey` maps to a nested `NSDictionary<NSString,NSString>` — flatten into the map with a `creator.` prefix. Each `MLModel*Key` static is a weakly-linked `Option<&'static NSString>` — skip `None` gracefully.
  - **Files:** `oxionnx-coreml/src/package.rs` (new method, pattern-matched on `collect_io_names`), `oxionnx-coreml/src/lib.rs`, stub twin.
  - **Prerequisites:** none.
  - **Tests:** `#[ignore]` model-driven test (env-var path, graceful skip) asserting the map is returned and, where the fixture has known metadata, values match; non-macOS stub test asserting `UnsupportedPlatform`.
  - **Risk:** value-type assumptions — downcast defensively throughout, never `unwrap()`.

## Threading

- [x] `MlPackageModel` is documented as `Send + Sync` (matching
      Apple's MLModel guarantees) but we have no stress test that
      actually exercises concurrent `predict` calls.  Add a benchmark
      that runs N threads against a single Arc<MlPackageModel>. (planned 2026-07-09)
  - **Goal:** actually exercise the documented `Send + Sync` contract with concurrent `predict` calls.
  - **Design:** new `oxionnx-coreml/tests/concurrent_predict.rs` (first `tests/` file in this crate; no new dependency — `std::thread` + `std::sync::Arc`). Model path from env (`OXIONNX_COREML_TEST_MODEL`); if unset or missing, print a skip note and return gracefully (no panic, no hardcoded absolute path — improves on the existing `.expect()` convention). Spawn N threads (e.g. 8) each running M predictions against a shared `Arc<MlPackageModel>`; assert every result has the expected output shape and no thread panicked. `#[ignore]` + `#[cfg(target_os = "macos")]`.
  - **Files:** new `oxionnx-coreml/tests/concurrent_predict.rs`.
  - **Prerequisites:** none.
  - **Tests:** the test itself is the deliverable.
  - **Risk:** needs a real model to actually run — env-var + graceful skip keeps default `nextest`/CI green either way. A criterion throughput benchmark is noted as a follow-up (would need new dev-deps + a `benches/` dir this crate doesn't have yet).

## Future cross-platform work

- [x] Consider iOS / tvOS / visionOS support.  CoreML APIs are
      identical, only the `target_os` cfg gates need broadening.
      Confirm `objc2-core-ml` actually builds on those targets first. (planned 2026-07-09)
  - **Goal:** the CoreML impl compiles for Apple mobile targets, not just macOS.
  - **Design:** broaden every `#[cfg(target_os = "macos")]` gate to `#[cfg(any(target_os="macos", target_os="ios", target_os="tvos", target_os="visionos"))]` (matches the platform list already named in `error.rs`'s doc comment; surgical vs. `target_vendor="apple"` which would also pull in watchOS). Coordinated sites: the `Cargo.toml` macOS-gated dependency-table key, the `macos_impl`/`stub_impl` module gates and their `pub use` re-exports in `package.rs`, `compute.rs`'s `to_native()` gate, and the two test-module gates. The `not(target_os="macos")` stub gates become the negation of the broadened set.
  - **Files:** `oxionnx-coreml/Cargo.toml`, `oxionnx-coreml/src/package.rs`, `oxionnx-coreml/src/compute.rs`. (Umbrella-crate gates in `oxionnx/src/lib.rs` and `oxionnx/Cargo.toml` are out of scope for this crate-local TODO — split out to Proposed follow-ups.)
  - **Prerequisites:** `rustup target add aarch64-apple-ios` (self-buildable — a rustc target download, not external infrastructure).
  - **Tests:** `cargo build --target aarch64-apple-ios -p oxionnx-coreml` clean, proving the FFI compiles on a non-macOS Apple target. tvOS/visionOS are tier-3 (nightly + `-Zbuild-std`) and cannot be stable-validated on this machine; they share identical CoreML APIs with iOS so the gate includes them, but they're only compile-validated transitively (documented as such, not silently assumed).
  - **Risk:** breaking a currently-green target — macOS keeps matching, Linux/wasm still hit the stub; only the newly-added Apple targets exercise the FFI, behind an explicit opt-in `rustup target add`. If the iOS build fails, narrow the predicate back and report as a deviation rather than forcing it green.

## Proposed follow-ups

- **OxiFace multi-function program audit** (split from the former "`compute_plan_summary` only counts `main`" item, resolved 2026-07-09): confirm whether any shipped OxiFace `.mlpackage` targets (SCRFD / ArcFace / InSwapper, or future additions) actually declare multi-function `MLProgram`s. The code-level bug (silently ignoring non-`main` functions) is already fixed — `compute_plan_summary`/`compute_plan_breakdown` now iterate every function. This follow-up is purely investigative and needs access to the actual OxiFace model artifacts, which are not present in this repo/environment.
- **`predict_raw` zero-copy `Retained<MLMultiArray>` handle variant** (noted during the 2026-07-09 `predict_raw` implementation): the shipped `predict_raw` returns portable `RawArray` bytes (dtype-preserving, no f32 up-conversion) to keep objc2 out of the cross-platform public API. A future zero-copy variant returning `Retained<MLMultiArray>` handles directly would avoid the byte copy entirely for CoreML→CoreML pipelines, at the cost of an objc2-typed public API (macOS-only surface).
- **Concurrent-`predict` criterion benchmark** (noted during the 2026-07-09 concurrency stress-test work): `tests/concurrent_predict.rs` covers correctness (N threads, shared `Arc<MlPackageModel>`, no panics, correct shapes) but not throughput. A criterion-based benchmark would need new `[dev-dependencies]` + a `benches/` dir, neither of which exist in this crate yet.
- **Umbrella-crate cfg gates for Apple mobile** (noted during the 2026-07-09 iOS/tvOS/visionOS work): `oxionnx/src/lib.rs:38,57` and `oxionnx/Cargo.toml`'s macOS-gated dev-dependency table are still macOS-only; broadening them to iOS/tvOS/visionOS is out of scope for this crate-local TODO and left for a follow-up pass on the umbrella crate.
