# oxionnx-coreml — TODO

Tracker for known gaps in the initial 0.1.3 release.

## I/O performance

- [ ] Switch input construction to the no-copy
      `MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error`
      path.  The current implementation calls
      `initWithShape_dataType_error` and `copy_nonoverlapping`s the slice
      into CoreML-owned storage; for large inputs (e.g. 1×3×640×640 SCRFD
      = 4.9 MB) the copy is a meaningful fraction of the per-frame budget.
      The wrinkle is that the deallocator block needs to keep the source
      `Vec<f32>` alive for the lifetime of the `MLMultiArray` — design
      a `block2::RcBlock`-based holder that takes ownership of the
      buffer and frees it from the deallocator callback.

- [ ] `predict` currently up-converts Float16 outputs to f32 inside the
      runtime.  When a downstream consumer wants the raw fp16 (e.g. for
      another CoreML model in a pipeline), expose a `predict_raw` variant
      that returns `MLMultiArray` handles directly.

## Functional gaps

- [ ] No support for non-`MLMultiArray` features yet.  `MLImage`,
      `MLSequence`, and `MLDictionary` features will surface as
      `CoreMLError::MissingOutput` (the `multiArrayValue()` getter
      returns nil for non-array features).  The SCRFD / ArcFace /
      InSwapper sub-gates are all multi-array, so this is non-blocking
      for OxiFace.

- [ ] `compute_plan_summary` only counts the `main` function.
      `MLProgram` models can declare additional functions
      (e.g. for stateful submodels); they are silently ignored.  Audit
      whether any of the OxiFace targets ship multi-function programs.

- [ ] `load_from_bytes` returns `UnsupportedFormat`.  We could implement
      it by extracting a temporary directory and calling `load`, but the
      `.mlpackage` bundle is intrinsically a directory tree — bytes-only
      loading would require the caller to serialize that bundle first.
      Document this clearly rather than papering over the limitation.

## Diagnostics

- [ ] `ComputePlanSummary` only returns a 4-bucket histogram.  The
      spike examples additionally report per-`operatorName`
      breakdowns that diagnose *which* op kinds got kicked off the ANE
      (e.g. `gather`, `scatter_along_axis`).  Add a richer
      `compute_plan_breakdown` that returns
      `HashMap<String, ComputePlanSummary>` keyed by operator name.

- [ ] No way to surface MLModel metadata (model author, license,
      version, custom user-defined keys) yet.  Add a
      `model_metadata() -> HashMap<String, String>` method that walks
      `MLModelDescription::metadata`.

## Threading

- [ ] `MlPackageModel` is documented as `Send + Sync` (matching
      Apple's MLModel guarantees) but we have no stress test that
      actually exercises concurrent `predict` calls.  Add a benchmark
      that runs N threads against a single Arc<MlPackageModel>.

## Future cross-platform work

- [ ] Consider iOS / tvOS / visionOS support.  CoreML APIs are
      identical, only the `target_os` cfg gates need broadening.
      Confirm `objc2-core-ml` actually builds on those targets first.
