//! macOS CoreML runtime implementation — `MlPackageModel`, the
//! `predict`/`predict_raw`/`predict_features`/`model_metadata` methods,
//! and every private helper that crosses the objc2 FFI boundary.
//!
//! Split out of `package.rs` by line-count policy (< 2000 lines/file);
//! this is a pure relocation — the module nesting (this file's contents
//! are still the direct child module `super::macos_impl` of `package`)
//! is unchanged, so every `pub(super)` visibility relationship among
//! `package`, `macos_impl` and the sibling `tests` module still resolves
//! exactly as it did when this was one file.

use super::*;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

use block2::StackBlock;

/// Hand-off slot for the asynchronous `MLComputePlan` completion
/// handler.  Pointer fields hold raw `*mut MLComputePlan` /
/// `*mut NSError` (encoded as `usize` so the struct itself is
/// trivially `Send + Sync`).  The block bumps the framework retain
/// count before stashing the pointer; the receiving side calls
/// `Retained::from_raw` to claim that +1.
#[derive(Default)]
struct PlanSlotInner {
    status: u8,
    plan_ptr: usize,
    err_ptr: usize,
}

#[derive(Default)]
struct PlanSlot {
    lock: Mutex<PlanSlotInner>,
    cvar: Condvar,
}
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputePlan, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureType, MLFeatureValue,
    MLModel, MLModelAuthorKey, MLModelConfiguration, MLModelCreatorDefinedKey,
    MLModelDescriptionKey, MLModelLicenseKey, MLModelStructureProgram,
    MLModelStructureProgramOperation, MLModelVersionStringKey, MLMultiArray, MLMultiArrayDataType,
    MLSequence,
};
use objc2_core_video::{
    kCVPixelFormatType_32BGRA, kCVPixelFormatType_OneComponent16Half,
    kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_OneComponent8, kCVReturnSuccess,
    CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferIsPlanar, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress,
};
use objc2_foundation::{
    NSArray, NSDictionary, NSError, NSNumber, NSObject, NSObjectProtocol, NSString, NSURL,
};

/// Loaded CoreML model + cached input/output names.
///
/// See the crate-level documentation for usage.
pub struct MlPackageModel {
    model: Retained<MLModel>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    /// Path to the *compiled* `.mlmodelc` directory — kept for
    /// [`compute_plan_summary`] which has to reload the bundle through
    /// `MLComputePlan`.
    compiled_path: PathBuf,
    /// Original compute-units policy supplied at load time — also reused
    /// by [`compute_plan_summary`].
    compute_units: MlComputeUnits,
}

// SAFETY: Apple documents MLModel as thread-safe.  Concurrent
// `predictionFromFeatures_error:` calls on the same instance are
// explicitly supported.  The `compiled_path` and metadata fields are
// immutable after construction.
unsafe impl Send for MlPackageModel {}
unsafe impl Sync for MlPackageModel {}

impl MlPackageModel {
    /// Load a `.mlpackage` (will be compiled) or `.mlmodelc` (loaded as-is).
    pub fn load(path: impl AsRef<Path>, compute_units: MlComputeUnits) -> Result<Self> {
        let path = path.as_ref();
        // Surface a clean error before crossing the FFI boundary.
        if !path.exists() {
            return Err(CoreMLError::Io {
                path: path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "model bundle does not exist",
                ),
            });
        }
        let compiled = compile_if_needed(path)?;
        let url = nsurl_for_dir(&compiled);
        let cfg = unsafe { MLModelConfiguration::new() };
        unsafe { cfg.setComputeUnits(compute_units.to_native()) };
        let model = unsafe {
            MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg)
                .map_err(nserror_to_coreml)?
        };

        let (input_names, output_names) = collect_io_names(&model);

        Ok(Self {
            model,
            input_names,
            output_names,
            compiled_path: compiled,
            compute_units,
        })
    }

    /// Load a model from an in-memory byte buffer.
    ///
    /// **Always unsupported for this runtime** — `.mlpackage` (and
    /// compiled `.mlmodelc`) bundles are *directory trees*
    /// (`Manifest.json`, `Data/`, per-weight blob files, ...), not a
    /// single serialized blob, so there is no "bytes" form to parse
    /// here.  A caller holding an in-memory representation of a
    /// packaged model (e.g. a `.zip`/`.tar` archive of a
    /// `.mlpackage`) must first materialize it back into a directory
    /// on disk before this runtime can load it.
    ///
    /// Always returns [`CoreMLError::UnsupportedFormat`].  Use
    /// [`load`](Self::load) with a path to the bundle directory
    /// instead.  Provided (returning a clean error rather than being
    /// absent) for API parity with `Session::from_bytes`.
    pub fn load_from_bytes(_bytes: &[u8], _compute_units: MlComputeUnits) -> Result<Self> {
        Err(CoreMLError::UnsupportedFormat(
            "MlPackageModel::load_from_bytes has no bytes-based loading path: \
             .mlpackage/.mlmodelc are directory bundles, not a single-file \
             format — write the bundle to a directory and call \
             MlPackageModel::load(path) instead",
        ))
    }

    /// Names of the model's input features, in declaration order
    /// (sorted lexicographically — the underlying `NSDictionary` does
    /// not preserve insertion order).
    pub fn input_names(&self) -> Vec<String> {
        self.input_names.clone()
    }

    /// Names of the model's output features.
    ///
    /// Note: `coremltools` rewrites ONNX output names to `var_NNNN`
    /// during conversion.  These names are stable for a given converted
    /// `.mlpackage` but are *not* the original ONNX names.
    pub fn output_names(&self) -> Vec<String> {
        self.output_names.clone()
    }

    /// Pre-execute one prediction with caller-supplied dummy inputs and
    /// discard the outputs.
    ///
    /// First-call latency for a freshly-loaded `.mlpackage` includes
    /// CoreML's per-graph specialisation (kernel JIT, ANE program
    /// compile, scratch-buffer allocation).  On Apple M3 with the
    /// InSwapper bundle this overhead is in the same order of
    /// magnitude as steady-state inference (1.0–1.5×) — when the
    /// workload is dispatched across N rayon workers, each worker
    /// pays the warm-up cost on its first call, which surfaces as a
    /// 3× per-frame slowdown for the first batch.
    ///
    /// Calling `warm_up` once at session-construction time pays this
    /// cost up front (synchronously, on the constructing thread)
    /// instead of lazily inside the hot path.
    ///
    /// `input_template` must already match the model's declared input
    /// shape — the caller is responsible for producing zero-filled
    /// (or otherwise neutral) tensors of the right rank.  Returns
    /// `Ok(())` on success, or any error that
    /// [`predict`](Self::predict) would surface.
    pub fn warm_up(&self, input_template: &HashMap<String, Tensor>) -> Result<()> {
        let _outs = self.predict(input_template)?;
        Ok(())
    }

    /// Validate `inputs` against the model's declared input names,
    /// build the corresponding `MLMultiArray`s (via
    /// [`multi_array_from_f32`]), wrap them in an
    /// `MLDictionaryFeatureProvider`, and run
    /// `predictionFromFeatures_error`.
    ///
    /// Shared by [`predict`](Self::predict),
    /// [`predict_raw`](Self::predict_raw) and
    /// [`predict_features`](Self::predict_features) — the three
    /// differ only in how they walk the *output* feature values
    /// afterward, so this factors out the one unsafe
    /// `predictionFromFeatures_error` call site all three would
    /// otherwise duplicate.
    fn run_prediction(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<Retained<ProtocolObject<dyn MLFeatureProvider>>> {
        // Validate input set up front so we can return a clean error
        // rather than a CoreML "missing feature" NSError.
        for name in &self.input_names {
            if !inputs.contains_key(name) {
                return Err(CoreMLError::InputMismatch(format!(
                    "missing input feature '{name}'",
                )));
            }
        }

        // Build MLMultiArrays in the same order as input_names so we keep
        // ownership of the Retained<MLMultiArray> until prediction is done.
        let mut arrays: Vec<(String, Retained<MLMultiArray>)> =
            Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let t = inputs.get(name).ok_or_else(|| {
                CoreMLError::InputMismatch(format!("missing input feature '{name}'"))
            })?;
            let arr = multi_array_from_f32(&t.data, &t.shape)?;
            arrays.push((name.clone(), arr));
        }

        // Build the MLDictionaryFeatureProvider from the (name, array) pairs.
        let provider = make_provider(&arrays)?;
        let p_obj: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*provider);

        unsafe {
            self.model
                .predictionFromFeatures_error(p_obj)
                .map_err(nserror_to_coreml)
        }
    }

    /// Run inference.
    ///
    /// `inputs` is keyed by the names returned by [`input_names`](Self::input_names);
    /// every required input must be present.  Extra entries are ignored.
    ///
    /// Returns a map keyed by [`output_names`](Self::output_names).
    pub fn predict(&self, inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let outputs = self.run_prediction(inputs)?;

        // Pull every declared output back as an f32 Tensor.
        let mut result = HashMap::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let key = NSString::from_str(name);
            let fv = unsafe { outputs.featureValueForName(&key) }
                .ok_or_else(|| CoreMLError::MissingOutput(name.clone()))?;
            let arr = unsafe { fv.multiArrayValue() }.ok_or_else(|| {
                CoreMLError::MissingOutput(format!(
                    "{name} (not MLMultiArray — use predict_features() instead to \
                     retrieve image/sequence/dictionary/scalar outputs)"
                ))
            })?;
            let tensor = tensor_from_multi_array(&arr)?;
            result.insert(name.clone(), tensor);
        }
        Ok(result)
    }

    /// Run inference and return each output's contents
    /// **dtype-preserving** — no `Float16` → `f32` up-conversion — for
    /// CoreML→CoreML pipelines where a downstream model wants the
    /// exact bytes an upstream model produced, or where fp16
    /// precision must survive the hop.
    ///
    /// Input handling and validation are identical to
    /// [`predict`](Self::predict) (same `inputs` contract, same
    /// error conditions); only output extraction differs — each
    /// declared output is read via the shared `read_raw_bytes` core
    /// instead of the `f32`-converting `tensor_from_multi_array`.
    ///
    /// Dtype coverage matches `predict`'s exactly: only `Float32`
    /// and `Float16` source arrays are supported today, everything
    /// else raises [`CoreMLError::UnsupportedOutputDtype`].
    pub fn predict_raw(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, RawArray>> {
        let outputs = self.run_prediction(inputs)?;

        let mut result = HashMap::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let key = NSString::from_str(name);
            let fv = unsafe { outputs.featureValueForName(&key) }
                .ok_or_else(|| CoreMLError::MissingOutput(name.clone()))?;
            let arr = unsafe { fv.multiArrayValue() }.ok_or_else(|| {
                CoreMLError::MissingOutput(format!(
                    "{name} (not MLMultiArray — use predict_features() instead to \
                     retrieve image/sequence/dictionary/scalar outputs)"
                ))
            })?;
            let raw = read_raw_bytes(&arr)?;
            result.insert(name.clone(), raw);
        }
        Ok(result)
    }

    /// Run inference and return **every** declared output through the
    /// typed [`FeatureOutput`] enum, regardless of `MLFeatureType` —
    /// unlike [`predict`](Self::predict) and
    /// [`predict_raw`](Self::predict_raw), which only ever surface
    /// `MLMultiArray`-shaped outputs and reject anything else with
    /// [`CoreMLError::MissingOutput`].
    ///
    /// Input handling and validation are identical to
    /// [`predict`](Self::predict) (same `inputs` contract, same error
    /// conditions).
    ///
    /// # Image decoding bound
    ///
    /// `MLFeatureTypeImage` outputs are decoded from their backing
    /// `CVPixelBuffer` for four standard, non-planar pixel formats
    /// only:
    ///
    /// | `pixelFormatType`     | Result [`Tensor`] shape | Element value |
    /// | :--------------------- | :----------------------- | :------------ |
    /// | `OneComponent8`       | `[height, width]`        | raw byte, `0.0..=255.0`, unnormalized |
    /// | `32BGRA`              | `[height, width, 4]`     | raw byte per channel, memory order B, G, R, A |
    /// | `OneComponent16Half`  | `[height, width]`        | `f16` sample converted to `f32` |
    /// | `OneComponent32Float` | `[height, width]`        | `f32` sample verbatim |
    ///
    /// Any other pixel format (planar layouts included — none of the
    /// four above ever are) raises
    /// [`CoreMLError::UnsupportedPixelFormat`] rather than silently
    /// misinterpreting the buffer's bytes.  This bound is deliberate:
    /// `CVPixelBuffer`'s format matrix is open-ended (dozens of YUV,
    /// Bayer and wide-gamut variants), and no OxiFace sub-gate model
    /// today produces an image-typed output — extending this list is
    /// straightforward follow-up work if/when a concrete format is
    /// needed, but guessing at an unverified decode would be worse
    /// than a clear error.
    ///
    /// # Other feature types
    ///
    /// * `MultiArray` — identical extraction to `predict`'s outputs.
    /// * `Sequence` — `Int64` or `String` element sequences only (the
    ///   only two element types `MLSequence` defines).
    /// * `Dictionary` — keys are stringified (`NSString` used
    ///   directly; anything else via its Objective-C `-description`,
    ///   which covers the `NSNumber` keys Apple's own contract
    ///   guarantees `dictionaryValue()` may otherwise produce),
    ///   values read as `f64`.
    /// * `String`, `Int64`, `Double` — read directly.
    /// * `Invalid` / `State` — CoreML defines these `MLFeatureType`s,
    ///   but neither has a portable representation here (a model
    ///   never legitimately *declares* an `Invalid`-typed output, and
    ///   `State` is a mutable inference-time buffer, not a value) —
    ///   both raise [`CoreMLError::UnsupportedFeatureType`].
    pub fn predict_features(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, FeatureOutput>> {
        let outputs = self.run_prediction(inputs)?;

        let mut result = HashMap::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let key = NSString::from_str(name);
            let fv = unsafe { outputs.featureValueForName(&key) }
                .ok_or_else(|| CoreMLError::MissingOutput(name.clone()))?;
            let out = feature_value_to_output(&fv)?;
            result.insert(name.clone(), out);
        }
        Ok(result)
    }

    /// Report the per-device op-count breakdown for this model under the
    /// configured compute-units policy, folded into one flat total
    /// across **every** function the `MLProgram` declares — not just
    /// `"main"` (a model with only a `"main"` function, the common
    /// case, sees no behavior change from this).  See
    /// [`compute_plan_breakdown`](Self::compute_plan_breakdown) for
    /// the equivalent per-`operatorName` view of this identical
    /// traversal.
    ///
    /// Reloads the bundle through `MLComputePlan` (the framework does
    /// not let us introspect a live `MLModel`) via the private
    /// `load_compute_plan` helper; see that method's doc comment (in
    /// this module's source) for the async-to-sync bridging details.
    pub fn compute_plan_summary(&self) -> Result<ComputePlanSummary> {
        let (summary, _breakdown) = self.compute_plan_summary_and_breakdown()?;
        Ok(summary)
    }

    /// Report the per-`operatorName` device-placement breakdown for
    /// this model under the configured compute-units policy — the
    /// same traversal [`compute_plan_summary`](Self::compute_plan_summary)
    /// folds into one flat total, but keyed by each operation's
    /// `operatorName` instead. Operations sharing a name (e.g. three
    /// separate `"gather"` ops scattered across the graph) accumulate
    /// into that name's single [`ComputePlanSummary`] entry via
    /// [`ComputePlanSummary::merge`].
    ///
    /// Iterates every function the `MLProgram` declares, exactly like
    /// [`compute_plan_summary`](Self::compute_plan_summary) — so
    /// summing every entry's fields across the returned map always
    /// reconciles exactly with that method's totals for the same
    /// model: both are produced by the identical per-operation
    /// classification pass (`classify_operation`) over the identical
    /// set of functions (`accumulate_program_operations`), just
    /// accumulated into a different shape.
    ///
    /// Reloads the bundle through `MLComputePlan`; see the private
    /// `load_compute_plan` helper's doc comment (in this module's
    /// source) for the async-to-sync bridging details.
    pub fn compute_plan_breakdown(&self) -> Result<HashMap<String, ComputePlanSummary>> {
        let (_summary, breakdown) = self.compute_plan_summary_and_breakdown()?;
        Ok(breakdown)
    }

    /// Shared core for
    /// [`compute_plan_summary`](Self::compute_plan_summary) and
    /// [`compute_plan_breakdown`](Self::compute_plan_breakdown):
    /// [`load_compute_plan`](Self::load_compute_plan) the bundle,
    /// resolve its `MLProgram` tree (erroring cleanly if this model
    /// is not an `MLProgram` — e.g. a legacy NeuralNetwork-typed
    /// `.mlmodel`), and hand off to `accumulate_program_operations`
    /// for the actual per-operation classification and accumulation.
    fn compute_plan_summary_and_breakdown(
        &self,
    ) -> Result<(ComputePlanSummary, HashMap<String, ComputePlanSummary>)> {
        let plan = self.load_compute_plan()?;
        let structure = unsafe { plan.modelStructure() };
        let program = unsafe { structure.program() }.ok_or_else(|| {
            CoreMLError::ComputePlan("not an MLProgram (no program tree)".to_string())
        })?;
        Ok(accumulate_program_operations(&plan, &program))
    }

    /// Asynchronously load the `MLComputePlan` for this model's
    /// compiled bundle under its configured compute-units policy.
    /// The async completion handler is bridged to a synchronous
    /// return via a `Condvar`-backed hand-off slot, with a 60-second
    /// timeout.
    ///
    /// This is purely the load step — callers are responsible for
    /// walking the returned plan's `modelStructure()` tree
    /// themselves (see
    /// [`compute_plan_summary_and_breakdown`](Self::compute_plan_summary_and_breakdown)).
    fn load_compute_plan(&self) -> Result<Retained<MLComputePlan>> {
        let url = nsurl_for_dir(&self.compiled_path);
        let cfg = unsafe { MLModelConfiguration::new() };
        unsafe { cfg.setComputeUnits(self.compute_units.to_native()) };

        // `Retained<...>` is not `Send`, so we cannot stash it in an Arc
        // for cross-thread transfer.  Instead we transfer the raw
        // CoreML pointers (which are bag-of-bits Send by default) and
        // re-`Retained::retain` on the receiving thread.  Status uses
        // signed encoding: 0 = pending, 1 = success (plan_ptr valid),
        // 2 = framework error (err_ptr valid), 3 = neither pointer set.
        let slot: Arc<PlanSlot> = Arc::new(PlanSlot::default());
        let slot_clone = slot.clone();

        let block = StackBlock::new(move |plan: *mut MLComputePlan, err: *mut NSError| {
            // Block-arg pointers are autoreleased by the framework.
            // We convert each into a `Retained<...>` (bumping the
            // refcount via objc_retain) and then immediately leak it
            // back to a raw pointer with `Retained::into_raw`.  The
            // receiving thread reclaims the +1 with `from_raw`.  This
            // ensures the object survives autorelease-pool draining
            // between the callback and the wait completion.
            let (status, plan_p, err_p): (u8, usize, usize) = if !plan.is_null() {
                match unsafe { Retained::retain(plan) } {
                    Some(p) => (1, Retained::into_raw(p) as usize, 0),
                    None => (3, 0, 0),
                }
            } else if !err.is_null() {
                match unsafe { Retained::retain(err) } {
                    Some(e) => (2, 0, Retained::into_raw(e) as usize),
                    None => (3, 0, 0),
                }
            } else {
                (3, 0, 0)
            };
            if let Ok(mut g) = slot_clone.lock.lock() {
                g.status = status;
                g.plan_ptr = plan_p;
                g.err_ptr = err_p;
                slot_clone.cvar.notify_all();
            }
        });

        unsafe {
            MLComputePlan::loadContentsOfURL_configuration_completionHandler(&url, &cfg, &block);
        }

        let guard = slot
            .lock
            .lock()
            .map_err(|e| CoreMLError::ComputePlan(format!("mutex poisoned: {e}")))?;
        let timeout = std::time::Duration::from_secs(60);
        let (mut guard, wait_res) = slot
            .cvar
            .wait_timeout_while(guard, timeout, |g| g.status == 0)
            .map_err(|e| CoreMLError::ComputePlan(format!("condvar wait: {e}")))?;
        if wait_res.timed_out() {
            return Err(CoreMLError::ComputePlan(
                "MLComputePlan load timed out after 60s".to_string(),
            ));
        }
        let status = guard.status;
        let plan_addr = guard.plan_ptr;
        let err_addr = guard.err_ptr;
        // Reset under the lock before dropping it.
        guard.status = 0;
        guard.plan_ptr = 0;
        guard.err_ptr = 0;
        drop(guard);

        let plan = match status {
            1 => {
                // SAFETY: the block retained the plan once already; we
                // claim that +1 reference here without bumping again.
                let plan_ptr = plan_addr as *mut MLComputePlan;
                unsafe { Retained::from_raw(plan_ptr) }.ok_or_else(|| {
                    CoreMLError::ComputePlan(
                        "completion handler returned null plan pointer".to_string(),
                    )
                })?
            }
            2 => {
                let err_ptr = err_addr as *mut NSError;
                let err = unsafe { Retained::from_raw(err_ptr) }.ok_or_else(|| {
                    CoreMLError::ComputePlan(
                        "completion handler returned null error pointer".to_string(),
                    )
                })?;
                return Err(nserror_to_coreml(err));
            }
            _ => {
                return Err(CoreMLError::ComputePlan(
                    "completion handler invoked with neither plan nor error".to_string(),
                ))
            }
        };
        Ok(plan)
    }

    /// Model-level metadata: description, author, license, version,
    /// and any creator-defined key/value pairs the `.mlpackage`
    /// embeds.
    ///
    /// Reads `MLModelDescription::metadata()`
    /// (`NSDictionary<MLModelMetadataKey, AnyObject>` — values are
    /// `AnyObject`, *not* guaranteed `NSString`, hence the defensive
    /// downcasts this performs) and projects the four well-known
    /// string keys plus the nested creator-defined dictionary into a
    /// single flat map:
    ///
    /// | Result key        | Source                                                        |
    /// | :------------------ | :-------------------------------------------------------------- |
    /// | `"description"`   | `MLModelDescriptionKey`                                       |
    /// | `"version"`       | `MLModelVersionStringKey`                                     |
    /// | `"author"`        | `MLModelAuthorKey`                                            |
    /// | `"license"`       | `MLModelLicenseKey`                                           |
    /// | `"creator.<key>"` | each entry of `MLModelCreatorDefinedKey`'s nested dictionary |
    ///
    /// Every key is optional in a CoreML model — a `.mlpackage` with
    /// no metadata at all returns `Ok` with an empty map, never an
    /// error.  Each of the five `MLModel*Key` statics is itself
    /// weakly-linked (`Option<&'static NSString>`, `None` on a
    /// CoreML framework revision that predates the key) and skipped
    /// gracefully when absent; likewise a value present under a
    /// known key but of an unexpected runtime class is skipped
    /// rather than mis-decoded, per this crate's no-`unwrap` policy.
    pub fn model_metadata(&self) -> Result<HashMap<String, String>> {
        let desc = unsafe { self.model.modelDescription() };
        let metadata = unsafe { desc.metadata() };

        let mut out = HashMap::new();
        insert_metadata_string(
            &metadata,
            unsafe { MLModelDescriptionKey },
            "description",
            &mut out,
        );
        insert_metadata_string(
            &metadata,
            unsafe { MLModelVersionStringKey },
            "version",
            &mut out,
        );
        insert_metadata_string(&metadata, unsafe { MLModelAuthorKey }, "author", &mut out);
        insert_metadata_string(&metadata, unsafe { MLModelLicenseKey }, "license", &mut out);

        if let Some(key) = unsafe { MLModelCreatorDefinedKey } {
            if let Some(creator_obj) = metadata.objectForKey(key) {
                // The nested dictionary's own generic parameters are
                // erased at the Objective-C runtime level (CoreML's
                // own signature only promises `AnyObject`), so the
                // only representable downcast target is the fully
                // erased `NSDictionary<AnyObject, AnyObject>` — see
                // `any_object_key_to_string`'s doc for why both keys
                // and values can share the same stringify helper.
                if let Ok(creator_dict) =
                    creator_obj.downcast::<NSDictionary<AnyObject, AnyObject>>()
                {
                    let keys = creator_dict.allKeys();
                    for i in 0..keys.len() {
                        let k = keys.objectAtIndex(i);
                        let Some(v) = creator_dict.objectForKey(&k) else {
                            // `k` came from this same dictionary's own
                            // `allKeys()`; a missing value would mean
                            // the dictionary mutated out from under
                            // us mid-iteration, which an immutable
                            // `NSDictionary` snapshot never does.
                            // Skip defensively rather than treat this
                            // as fatal.
                            continue;
                        };
                        let key_str = any_object_key_to_string(k);
                        let val_str = any_object_key_to_string(v);
                        out.insert(format!("creator.{key_str}"), val_str);
                    }
                }
            }
        }

        Ok(out)
    }
}

// ────────────────────────────────────────────────────────────────────── //
// Helpers — every helper that crosses the FFI boundary lives here.       //
// ────────────────────────────────────────────────────────────────────── //

// ────────────────────────────────────────────────────────────────────── //
// `compute_plan_summary` / `compute_plan_breakdown` support —           //
// operation classification and the shared function/block traversal      //
// both public entry points fold into their respective accumulator       //
// shape.                                                                 //
// ────────────────────────────────────────────────────────────────────── //

/// Which compute device CoreML placed a single `MLProgram` operation
/// on — the classification [`classify_operation`] produces.  Shared
/// by both [`MlPackageModel::compute_plan_summary`]'s flat
/// accumulation and [`MlPackageModel::compute_plan_breakdown`]'s
/// per-operator-name accumulation, so the two can never disagree on
/// how a given operation is bucketed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpDevice {
    /// Placed on the Apple Neural Engine.
    Ane,
    /// Placed on the integrated GPU.
    Gpu,
    /// Fell back to CPU.
    Cpu,
    /// No reported device — `const` ops (data placement, not actual
    /// compute) and any operation the framework itself could not
    /// determine a device usage for both land here.
    Unknown,
}

impl OpDevice {
    /// A single-operation [`ComputePlanSummary`] delta with exactly
    /// one field set to `1` — the unit
    /// [`accumulate_program_operations`] folds via
    /// [`ComputePlanSummary::merge`] once per classified operation,
    /// into both the flat running total and the per-operator-name
    /// breakdown entry.
    fn as_summary_delta(self) -> ComputePlanSummary {
        let mut delta = ComputePlanSummary::default();
        match self {
            Self::Ane => delta.ane_ops = 1,
            Self::Gpu => delta.gpu_ops = 1,
            Self::Cpu => delta.cpu_ops = 1,
            Self::Unknown => delta.unknown_ops = 1,
        }
        delta
    }
}

/// Classify one `MLProgram` operation's device placement under
/// `plan`. Single shared routine behind both
/// [`MlPackageModel::compute_plan_summary`]'s flat accumulation and
/// [`MlPackageModel::compute_plan_breakdown`]'s per-operator-name
/// accumulation — `opname` is passed in rather than re-read from `op`
/// because every caller has already read it (to key the breakdown
/// map), so re-reading it here would be redundant FFI traffic.
///
/// `const` ops carry no compute device (they're data placement, not
/// actual compute) — bucketed as [`OpDevice::Unknown`], same as an
/// operation the framework itself reports no device usage for.
fn classify_operation(
    plan: &MLComputePlan,
    op: &MLModelStructureProgramOperation,
    opname: &str,
) -> OpDevice {
    if opname == "const" {
        return OpDevice::Unknown;
    }
    match unsafe { plan.computeDeviceUsageForMLProgramOperation(op) } {
        Some(usage) => {
            let dev = unsafe { usage.preferredComputeDevice() };
            // SAFETY: ProtocolObject is #[repr(C)] with an
            // `inner: AnyObject` first field, so the pointer cast is
            // sound.
            let dev_obj: &AnyObject = unsafe { &*(&*dev as *const _ as *const AnyObject) };
            let class_name = dev_obj
                .class()
                .name()
                .to_str()
                .unwrap_or("UnknownDeviceClass")
                .to_string();
            if class_name.contains("NeuralEngine") {
                OpDevice::Ane
            } else if class_name.contains("GPU") {
                OpDevice::Gpu
            } else if class_name.contains("CPU") {
                OpDevice::Cpu
            } else {
                OpDevice::Unknown
            }
        }
        None => OpDevice::Unknown,
    }
}

/// Walk every operation of **every** function declared in `program`,
/// classifying each with [`classify_operation`] and merging its
/// one-operation [`ComputePlanSummary`] delta (via
/// [`ComputePlanSummary::merge`]) into both a flat running total and
/// a per-`operatorName` breakdown map.
///
/// Iterating every key of `program.functions()` — rather than only
/// looking up `"main"` — is the fix for `compute_plan_summary`'s
/// former silent-ignore behavior on multi-function `MLProgram`
/// models (e.g. stateful submodels declare auxiliary functions
/// alongside `main`).  For the common case (a model with only a
/// `"main"` function) this produces identical totals to visiting
/// `"main"` alone, since there is nothing else to iterate.
fn accumulate_program_operations(
    plan: &MLComputePlan,
    program: &MLModelStructureProgram,
) -> (ComputePlanSummary, HashMap<String, ComputePlanSummary>) {
    let mut summary = ComputePlanSummary::default();
    let mut breakdown: HashMap<String, ComputePlanSummary> = HashMap::new();

    let funcs = unsafe { program.functions() };
    let func_names = funcs.allKeys();
    for fi in 0..func_names.len() {
        let fname = func_names.objectAtIndex(fi);
        let Some(func) = funcs.objectForKey(&fname) else {
            // `fname` came from this same dictionary's own
            // `allKeys()`; a missing value would mean the dictionary
            // mutated out from under us mid-iteration, which an
            // immutable `NSDictionary` snapshot never does. Skip
            // defensively rather than treat this as fatal.
            continue;
        };
        let blk = unsafe { func.block() };
        let ops = unsafe { blk.operations() };
        let n = ops.len();
        for i in 0..n {
            let op: Retained<MLModelStructureProgramOperation> = ops.objectAtIndex(i);
            let opname = unsafe { op.operatorName() }.to_string();
            let delta = classify_operation(plan, &op, &opname).as_summary_delta();
            summary.merge(&delta);
            breakdown.entry(opname).or_default().merge(&delta);
        }
    }

    (summary, breakdown)
}

fn nsurl_for_dir(path: &Path) -> Retained<NSURL> {
    let s = NSString::from_str(&path.to_string_lossy());
    // .mlpackage and .mlmodelc are both directory bundles.
    NSURL::fileURLWithPath_isDirectory(&s, true)
}

fn nserror_to_coreml(err: Retained<NSError>) -> CoreMLError {
    let code = err.code() as i64;
    let msg = err.localizedDescription().to_string();
    CoreMLError::Framework { code, message: msg }
}

/// Compile a `.mlpackage` to `.mlmodelc` (cached in the system tmp dir
/// by the framework).  `.mlmodelc` paths are returned as-is.
fn compile_if_needed(path: &Path) -> Result<PathBuf> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext == "mlmodelc" {
        return Ok(path.to_path_buf());
    }
    let url = nsurl_for_dir(path);
    // `compileModelAtURL_error:` is the legacy synchronous compile API.
    // The newer async variant returns the same compiled URL but is
    // harder to bridge.  Suppress the deprecation warning locally.
    let compiled: Retained<NSURL> = unsafe {
        #[allow(deprecated)]
        MLModel::compileModelAtURL_error(&url).map_err(nserror_to_coreml)?
    };
    let s = compiled.path().ok_or_else(|| {
        CoreMLError::Internal("compiled NSURL has no filesystem path".to_string())
    })?;
    Ok(PathBuf::from(s.to_string()))
}

fn collect_io_names(model: &MLModel) -> (Vec<String>, Vec<String>) {
    let desc = unsafe { model.modelDescription() };
    let in_dict = unsafe { desc.inputDescriptionsByName() };
    let out_dict = unsafe { desc.outputDescriptionsByName() };
    let inputs = collect_dict_keys(&in_dict);
    let outputs = collect_dict_keys(&out_dict);
    (inputs, outputs)
}

fn collect_dict_keys<V>(dict: &NSDictionary<NSString, V>) -> Vec<String>
where
    V: objc2::Message,
{
    let keys = dict.allKeys();
    let mut out = Vec::with_capacity(keys.len());
    for i in 0..keys.len() {
        let k = keys.objectAtIndex(i);
        out.push(k.to_string());
    }
    out.sort();
    out
}

/// C-contiguous ("row-major") per-dimension strides for `shape`, in
/// *elements* (not bytes) — the same convention `MLMultiArray::strides()`
/// uses.  The last dimension has stride 1; each preceding dimension's
/// stride is the product of every following dimension's size.
fn c_contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let rank = shape.len();
    let mut strides = vec![0isize; rank];
    if rank > 0 {
        strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as isize;
        }
    }
    strides
}

/// Build the `shape` and C-contiguous `strides` `NSNumber` arrays
/// shared by every `MLMultiArray` constructor in this module.
fn build_shape_and_strides_arrays(
    shape: &[usize],
) -> (Retained<NSArray<NSNumber>>, Retained<NSArray<NSNumber>>) {
    let shape_numbers: Vec<Retained<NSNumber>> = shape
        .iter()
        .map(|d| NSNumber::new_isize(*d as isize))
        .collect();
    let shape_arr: Retained<NSArray<NSNumber>> = NSArray::from_retained_slice(&shape_numbers);

    let stride_numbers: Vec<Retained<NSNumber>> = c_contiguous_strides(shape)
        .iter()
        .map(|s| NSNumber::new_isize(*s))
        .collect();
    let strides_arr: Retained<NSArray<NSNumber>> = NSArray::from_retained_slice(&stride_numbers);

    (shape_arr, strides_arr)
}

/// Build a Float32 `MLMultiArray` that **aliases** the caller's slice —
/// no copy.  The returned array's backing buffer is a raw pointer into
/// `data`; CoreML is told (via `deallocator: None`) that it does not
/// own this memory and must never free it.
///
/// # Non-escaping contract
///
/// This is a private helper called only from
/// [`MlPackageModel::predict`] and [`MlPackageModel::predict_raw`],
/// both of which:
/// 1. construct every input array synchronously within the call,
/// 2. hand every array to `predictionFromFeatures_error` (which reads
///    — never writes — input features; CoreML allocates its own
///    output storage unless the caller opts into `MLPredictionOptions`
///    output-backing, which we never do here), and
/// 3. drop every constructed array (via the local `arrays: Vec<...>`
///    going out of scope at the end of the call) before returning.
///
/// The returned `Retained<MLMultiArray>` must **never** be stored
/// beyond the synchronous call that built it, sent across an
/// await/thread boundary, or otherwise allowed to outlive `data`.  For
/// an ownership-transferring variant that may outlive its
/// constructing frame, see `multi_array_from_owned`.
pub(super) fn multi_array_from_f32(
    data: &[f32],
    shape: &[usize],
) -> Result<Retained<MLMultiArray>> {
    let expected_count: usize = shape.iter().product();
    if expected_count != data.len() {
        return Err(CoreMLError::InputMismatch(format!(
            "input tensor has {} elements but declared shape {shape:?} implies {expected_count}",
            data.len(),
        )));
    }
    let (shape_arr, strides_arr) = build_shape_and_strides_arrays(shape);

    // SAFETY:
    // * `data_pointer` is derived from `data: &[f32]`, which Rust
    //   guarantees is non-null and well-aligned for `f32` even when
    //   empty — `NonNull::new` below is therefore infallible in
    //   practice; we still handle `None` defensively rather than
    //   unwrap.
    // * Validity/lifetime: per the "Non-escaping contract" above, the
    //   constructed `MLMultiArray` never escapes the synchronous
    //   `predict`/`predict_raw` call that (transitively, via this
    //   function) constructs it, so the pointer stays valid for the
    //   array's entire lifetime — `data` (borrowed from the caller's
    //   `&HashMap<String, Tensor>` argument) outlives that call by
    //   construction of Rust's borrow checker.
    // * No-write aliasing: `data` is handed to CoreML strictly as an
    //   *input* feature.  `predictionFromFeatures_error` reads input
    //   features to compute outputs into separately-allocated
    //   storage; it does not write back through an input's data
    //   pointer unless the caller explicitly registers it as an
    //   output backing via `MLPredictionOptions` (which this runtime
    //   never does).  This preserves the invariant that nothing
    //   mutates memory behind a `&[f32]` shared borrow.
    // * `deallocator: None` is therefore correct: CoreML does not own
    //   `data`'s allocation and must not attempt to free it, whether
    //   at `Retained` drop time inside `predict`/`predict_raw`, or
    //   earlier if CoreML releases its own internal reference to the
    //   input feature sooner.
    let data_pointer =
        core::ptr::NonNull::new(data.as_ptr().cast_mut().cast::<core::ffi::c_void>()).ok_or_else(
            || CoreMLError::Internal("input slice pointer was unexpectedly null".to_string()),
        )?;
    let arr = unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            data_pointer,
            &shape_arr,
            MLMultiArrayDataType::Float32,
            &strides_arr,
            None,
        )
        .map_err(nserror_to_coreml)?
    };
    Ok(arr)
}

/// Build a Float32 `MLMultiArray` that **takes ownership** of `data` —
/// no copy, and (unlike [`multi_array_from_f32`]) the returned array
/// may be stored, returned, or otherwise allowed to outlive the frame
/// that constructed it.
///
/// `data` is moved into a heap-allocated `block2::RcBlock` deallocator
/// closure; CoreML frees the backing allocation (by dropping the
/// `Vec`) exactly when the `MLMultiArray`'s last reference is
/// released, whether that happens synchronously in this frame or long
/// after it returns.
///
/// `predict` and `predict_raw` are both fully synchronous and never
/// need an escaping array, so [`multi_array_from_f32`] suffices for
/// them today — this function is currently exercised only by this
/// module's tests, kept as reviewed, tested infrastructure for a
/// future escaping consumer (e.g. a zero-copy
/// `Retained<MLMultiArray>`-returning `predict_raw` variant; see
/// `TODO.md`'s "Proposed follow-ups").  `#[cfg(test)]`-gated so an
/// as-yet-unconsumed helper does not trip the crate's `-D warnings`
/// dead-code gate: its only caller today is `owned_array_tests`,
/// below.
#[cfg(test)]
fn multi_array_from_owned(mut data: Vec<f32>, shape: &[usize]) -> Result<Retained<MLMultiArray>> {
    let expected_count: usize = shape.iter().product();
    if expected_count != data.len() {
        return Err(CoreMLError::InputMismatch(format!(
            "owned tensor has {} elements but declared shape {shape:?} implies {expected_count}",
            data.len(),
        )));
    }
    let (shape_arr, strides_arr) = build_shape_and_strides_arrays(shape);

    // Pointer into `data`'s heap allocation, captured *before* `data`
    // is moved below. Moving a `Vec<f32>` relocates only its 3-word
    // (ptr, len, cap) descriptor; the heap buffer that pointer refers
    // to never moves, so this pointer remains valid after the move.
    let data_pointer = core::ptr::NonNull::new(data.as_mut_ptr().cast::<core::ffi::c_void>())
        .ok_or_else(|| {
            CoreMLError::Internal("owned tensor pointer was unexpectedly null".to_string())
        })?;

    // `block2::RcBlock::new` requires an `Fn`, not `FnOnce` — but
    // freeing `data` is an inherently one-shot, consuming operation.
    // We square this circle with `Mutex<Option<Vec<f32>>>`: `Fn`'s
    // `&self` receiver only ever needs shared access to *take* the
    // `Vec` out through interior mutability.  A real mutex (rather
    // than `Cell`) is used deliberately: `block2` 0.6.2's `IntoBlock`
    // impl does not require `Send`/`Sync` on the closure (see its
    // `traits.rs`, which has a standing `// TODO: Add + Send, + Sync`
    // comment) — nothing in the type system stops CoreML's runtime
    // from invoking this deallocator on a thread other than the one
    // that built it.  `Cell` would compile but would be relying on an
    // *unenforced* single-invocation, single-thread assumption;
    // `Mutex` keeps the one-time take-and-drop sound even if that
    // assumption is ever violated, at the cost of one uncontended
    // lock on a deallocation path that is never hot.
    let owned: Mutex<Option<Vec<f32>>> = Mutex::new(Some(data));
    let dealloc = block2::RcBlock::new(move |_ptr: core::ptr::NonNull<core::ffi::c_void>| {
        // `_ptr` is CoreML's view of `data_pointer` above; we don't
        // need it because `owned` already knows its own
        // pointer/length/capacity.  A poisoned mutex (only possible
        // if a panic somehow unwound through a previous invocation)
        // still yields a usable guard via `into_inner` — we're only
        // ever freeing memory here, never observing a torn value.
        let mut guard = owned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        drop(guard.take());
    });

    // SAFETY:
    // * `data_pointer` is valid, non-null, and aligned for `f32`
    //   (derived from a live `Vec<f32>`'s own pointer).
    // * Validity/lifetime: `owned` (holding the only remaining handle
    //   to `data`'s allocation) is moved into `dealloc`, which is
    //   moved into the `MLMultiArray` as its deallocator — so the
    //   allocation stays alive for exactly as long as CoreML needs
    //   it, and is freed deterministically (via `Drop`) the one time
    //   CoreML invokes the deallocator: never before, and never
    //   leaked after.
    // * `deallocator: Some(&dealloc)` is correct here (unlike
    //   `multi_array_from_f32`'s `None`) because this function's
    //   whole purpose is transferring ownership of `data` to the
    //   `MLMultiArray`.
    let arr = unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            data_pointer,
            &shape_arr,
            MLMultiArrayDataType::Float32,
            &strides_arr,
            Some(&dealloc),
        )
        .map_err(nserror_to_coreml)?
    };
    Ok(arr)
}

fn make_provider(
    arrays: &[(String, Retained<MLMultiArray>)],
) -> Result<Retained<MLDictionaryFeatureProvider>> {
    let keys_owned: Vec<Retained<NSString>> =
        arrays.iter().map(|(n, _)| NSString::from_str(n)).collect();
    let vals_owned: Vec<Retained<MLFeatureValue>> = arrays
        .iter()
        .map(|(_, a)| unsafe { MLFeatureValue::featureValueWithMultiArray(a) })
        .collect();
    let key_refs: Vec<&NSString> = keys_owned.iter().map(|k| &**k).collect();
    let val_refs: Vec<&MLFeatureValue> = vals_owned.iter().map(|v| &**v).collect();
    let dict: Retained<NSDictionary<NSString, MLFeatureValue>> =
        NSDictionary::from_slices(&key_refs, &val_refs);
    // SAFETY: NSDictionary is structurally identical regardless of its
    // Rust type parameters — Objective-C dictionaries hold `id` values.
    // The init signature wants AnyObject; we reinterpret the typed
    // pointer.  This pattern is replicated from the spike code.
    let dict_any: &NSDictionary<NSString, AnyObject> = unsafe {
        &*(dict.as_ref() as *const NSDictionary<NSString, MLFeatureValue>
            as *const NSDictionary<NSString, AnyObject>)
    };
    unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            dict_any,
        )
        .map_err(nserror_to_coreml)
    }
}

/// Map a source `MLMultiArrayDataType` to the portable
/// [`MlArrayDtype`] tag plus its per-element byte width, or reject it
/// with [`CoreMLError::UnsupportedOutputDtype`].
///
/// Only `Float32` and `Float16` are supported today — exactly the set
/// `tensor_from_multi_array` has always accepted.  `MLMultiArrayDataType`
/// is a newtype-wrapped integer with associated consts (not a real
/// Rust `enum`), so a catch-all arm is required for exhaustiveness.
fn dtype_and_width(dt: MLMultiArrayDataType) -> Result<(MlArrayDtype, usize)> {
    match dt {
        MLMultiArrayDataType::Float32 => Ok((MlArrayDtype::F32, core::mem::size_of::<f32>())),
        MLMultiArrayDataType::Float16 => Ok((MlArrayDtype::F16, core::mem::size_of::<u16>())),
        _ => Err(CoreMLError::UnsupportedOutputDtype(format!("{dt:?}"))),
    }
}

/// Extract an `MLMultiArray`'s contents verbatim into a portable
/// [`RawArray`] — dtype preserved (no `Float16` → `f32`
/// up-conversion), shape normalized to a tightly-packed C-contiguous
/// byte run regardless of CoreML's internal strides.
///
/// This is the shared extraction core for both
/// `tensor_from_multi_array` (which further converts the bytes to
/// `f32`) and [`MlPackageModel::predict_raw`] (which returns the
/// bytes as-is).
///
/// **Stride-aware copy.**  CoreML may allocate output buffers with
/// non-C-contiguous strides for ANE / GPU alignment.  In practice this
/// shows up on SCRFD: an output declared as shape `[800, 1]` is laid
/// out with strides `[32, 1]` (each row padded to 32 elements for
/// 64-byte cache-line alignment).  A naive `copy_nonoverlapping(N)`
/// from `dataPointer()` reads padding bytes and silently scrambles the
/// data — the symptom in OxiFace's `--device coreml` SCRFD detector
/// was zero-detection on real face images even though `coremltools`'
/// Python `predict()` returned correct values for the same model.
///
/// We copy the per-element *bytes* via
/// `byte_ptr.offset(idx · stride · elem_bytes)` where `idx` is the
/// C-major destination index decomposed via the declared `shape`,
/// multiplied with the array's reported `strides()`.  On the
/// C-contiguous fast path this is a single bulk `copy_nonoverlapping`;
/// on the strided slow path it walks element by element.  Every copy
/// here is `u8`-to-`u8` (never a typed pointer cast), so there is no
/// alignment hazard regardless of `dtype`'s natural alignment — unlike
/// interpreting the bytes as `f32`/`u16`, which
/// `tensor_from_multi_array`'s conversion pass does safely via
/// `from_ne_bytes` rather than a pointer cast, precisely to avoid
/// assuming a `Vec<u8>`'s heap allocation is `f32`/`u16`-aligned (Rust
/// only guarantees `align_of::<u8>() == 1` for it).
pub(super) fn read_raw_bytes(arr: &MLMultiArray) -> Result<RawArray> {
    let shape = read_shape(arr);
    let strides = read_strides(arr);
    let dt = unsafe { arr.dataType() };
    let (dtype, elem_bytes) = dtype_and_width(dt)?;

    // C-contiguous element count = product of shape (NOT `arr.count()`,
    // which can include stride padding on some allocations).
    let n_c_contig: usize = shape.iter().product();
    let c_strides = c_contiguous_strides(&shape);
    let is_c_contiguous = strides == c_strides;

    // Allocate the destination outside the closure; the closure writes
    // through a raw pointer.  The buffer is alive for the entire
    // function (and therefore for the entire synchronous handler
    // call).
    let mut out: Vec<u8> = vec![0u8; n_c_contig * elem_bytes];
    let dst_ptr: *mut u8 = out.as_mut_ptr();

    // `block2::StackBlock` requires `Fn`, so the "did the handler run"
    // signal goes through a `Cell<bool>`.  `getBytesWithHandler` is
    // documented to always invoke its handler synchronously exactly
    // once, but we don't trust that blindly — see the
    // `Err(Internal(...))` fallback below.
    let invoked: std::cell::Cell<bool> = std::cell::Cell::new(false);
    let invoked_ref: &std::cell::Cell<bool> = &invoked;
    let shape_ref: &Vec<usize> = &shape;
    let strides_ref: &Vec<isize> = &strides;

    let handler = block2::StackBlock::new(
        |bytes: core::ptr::NonNull<core::ffi::c_void>, _size: isize| {
            let base = bytes.as_ptr() as *const u8;
            if is_c_contiguous {
                // Fast path — bulk byte copy.  `base` and `dst_ptr`
                // are both `u8`-typed, so this has no alignment
                // requirement beyond `align_of::<u8>() == 1`,
                // regardless of the logical element width.
                unsafe {
                    std::ptr::copy_nonoverlapping(base, dst_ptr, n_c_contig * elem_bytes);
                }
            } else {
                // Stride-aware slow path.  Walk a multi-dimensional
                // index in C-major order; translate each step to a
                // source *byte* offset using the declared
                // (element-unit) strides, then copy exactly one
                // element's worth of raw bytes — never interpreting
                // them as a typed value, so there's no alignment
                // hazard here either.
                let mut idx: Vec<usize> = vec![0; shape_ref.len()];
                for dst in 0..n_c_contig {
                    let mut src_offset: isize = 0;
                    for d in 0..shape_ref.len() {
                        src_offset += idx[d] as isize * strides_ref[d];
                    }
                    unsafe {
                        let src = base.offset(src_offset * elem_bytes as isize);
                        let dst_elem = dst_ptr.add(dst * elem_bytes);
                        std::ptr::copy_nonoverlapping(src, dst_elem, elem_bytes);
                    }
                    // Increment idx in C-major order (innermost first).
                    for d in (0..shape_ref.len()).rev() {
                        idx[d] += 1;
                        if idx[d] < shape_ref[d] {
                            break;
                        }
                        idx[d] = 0;
                    }
                }
            }
            invoked_ref.set(true);
        },
    );

    unsafe { arr.getBytesWithHandler(&handler) };

    if invoked.get() {
        Ok(RawArray {
            shape,
            dtype,
            data: out,
        })
    } else {
        Err(CoreMLError::Internal(
            "getBytesWithHandler did not invoke its handler".to_string(),
        ))
    }
}

/// Convert an output `MLMultiArray` to a tightly-packed C-contiguous
/// `f32` [`Tensor`].  `Float32` sources copy through unchanged;
/// `Float16` sources are up-converted per-element.  Built on
/// [`read_raw_bytes`] — see that function's doc comment for the
/// stride-aware extraction and the SCRFD cache-line-padding story
/// that motivates it.
pub(super) fn tensor_from_multi_array(arr: &MLMultiArray) -> Result<Tensor> {
    let raw = read_raw_bytes(arr)?;
    let data: Vec<f32> = match raw.dtype {
        MlArrayDtype::F32 => raw
            .data
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        MlArrayDtype::F16 => raw
            .data
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32())
            .collect(),
        MlArrayDtype::F64 | MlArrayDtype::I32 | MlArrayDtype::I8 => {
            // Unreachable in practice: `read_raw_bytes` calls
            // `dtype_and_width`, which only ever returns `Ok` for
            // F32/F16 — anything else already returned
            // `Err(UnsupportedOutputDtype)` above. Surfaced as an
            // `Internal` error rather than `unreachable!()` so a
            // future change to `dtype_and_width`'s coverage fails
            // loudly as a `Result`, not a panic.
            return Err(CoreMLError::Internal(format!(
                "read_raw_bytes produced dtype {:?}, but only F32/F16 are ever returned \
                 for MLMultiArray sources — this indicates an internal invariant violation",
                raw.dtype
            )));
        }
    };
    Ok(Tensor::new(data, raw.shape))
}

fn read_shape(arr: &MLMultiArray) -> Vec<usize> {
    let s = unsafe { arr.shape() };
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = s.objectAtIndex(i);
        // NSNumber::longLongValue is the safe path for any integer
        // underlying type.
        let iv = v.longLongValue();
        out.push(if iv < 0 { 0 } else { iv as usize });
    }
    out
}

/// Read `MLMultiArray::strides()` as a plain `Vec<isize>` (in elements,
/// not bytes — same convention CoreML uses).
fn read_strides(arr: &MLMultiArray) -> Vec<isize> {
    let s = unsafe { arr.strides() };
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = s.objectAtIndex(i);
        out.push(v.longLongValue() as isize);
    }
    out
}

// ────────────────────────────────────────────────────────────────────── //
// `predict_features` support — MLFeatureValue dispatch, dictionary /    //
// sequence extraction, and the CVPixelBuffer image decoder.             //
// ────────────────────────────────────────────────────────────────────── //

/// Dispatch a single `MLFeatureValue` to the matching [`FeatureOutput`]
/// variant, based on `fv.r#type()`.  Factored out of
/// [`MlPackageModel::predict_features`] (`pub(super)` rather than
/// fully private) so it can be exercised directly against hand-built
/// `MLFeatureValue`s in tests, without needing a loaded model.
pub(super) fn feature_value_to_output(fv: &MLFeatureValue) -> Result<FeatureOutput> {
    match unsafe { fv.r#type() } {
        MLFeatureType::MultiArray => {
            let arr = unsafe { fv.multiArrayValue() }.ok_or_else(|| {
                CoreMLError::Internal(
                    "MLFeatureValue reported type MultiArray but multiArrayValue() was nil"
                        .to_string(),
                )
            })?;
            let tensor = tensor_from_multi_array(&arr)?;
            Ok(FeatureOutput::MultiArray(tensor))
        }
        MLFeatureType::Int64 => Ok(FeatureOutput::Int64(unsafe { fv.int64Value() })),
        MLFeatureType::Double => Ok(FeatureOutput::Double(unsafe { fv.doubleValue() })),
        MLFeatureType::String => {
            let s = unsafe { fv.stringValue() };
            Ok(FeatureOutput::String(s.to_string()))
        }
        MLFeatureType::Dictionary => {
            let dict = unsafe { fv.dictionaryValue() };
            Ok(FeatureOutput::Dictionary(dictionary_value_to_map(&dict)))
        }
        MLFeatureType::Sequence => {
            let seq = unsafe { fv.sequenceValue() }.ok_or_else(|| {
                CoreMLError::Internal(
                    "MLFeatureValue reported type Sequence but sequenceValue() was nil".to_string(),
                )
            })?;
            let sv = sequence_value_to_portable(&seq)?;
            Ok(FeatureOutput::Sequence(sv))
        }
        MLFeatureType::Image => {
            let pixel_buffer = unsafe { fv.imageBufferValue() }.ok_or_else(|| {
                CoreMLError::Internal(
                    "MLFeatureValue reported type Image but imageBufferValue() was nil".to_string(),
                )
            })?;
            let tensor = tensor_from_pixel_buffer(&pixel_buffer)?;
            Ok(FeatureOutput::Image(tensor))
        }
        other => Err(CoreMLError::UnsupportedFeatureType(format!(
            "MLFeatureType({}) is none of Int64/Double/String/MultiArray/Dictionary/\
             Sequence/Image, which is everything predict_features's dispatch supports — \
             MLFeatureTypeInvalid (0) and MLFeatureTypeState (8) have no portable \
             FeatureOutput representation",
            other.0
        ))),
    }
}

/// Flatten an `MLFeatureValue`'s `dictionaryValue()`
/// (`NSDictionary<AnyObject, NSNumber>` — Apple guarantees keys are
/// always `NSNumber` or `NSString`, see
/// `MLFeatureValue::featureValueWithDictionary_error`'s own doc) into
/// a portable `HashMap<String, f64>`.  Keys are stringified via
/// [`any_object_key_to_string`]; if two distinct keys ever stringified
/// to the same Rust `String` (which Apple's key-type contract makes
/// vanishingly unlikely in practice), the later-iterated value wins,
/// matching `HashMap::insert`'s normal overwrite semantics.
fn dictionary_value_to_map(dict: &NSDictionary<AnyObject, NSNumber>) -> HashMap<String, f64> {
    let keys = dict.allKeys();
    let mut out = HashMap::with_capacity(keys.len());
    for i in 0..keys.len() {
        let key_obj = keys.objectAtIndex(i);
        let Some(value) = dict.objectForKey(&key_obj) else {
            // `key_obj` came from this same dictionary's own
            // `allKeys()`; a missing value would mean the dictionary
            // mutated out from under us mid-iteration, which an
            // immutable `NSDictionary` snapshot never does. Skip
            // defensively rather than treat this as fatal.
            continue;
        };
        out.insert(any_object_key_to_string(key_obj), value.doubleValue());
    }
    out
}

/// Read an `MLSequence`'s elements into the matching [`SequenceValue`]
/// variant, per its own `r#type()` (`MLFeatureTypeInt64` or
/// `MLFeatureTypeString` — the only two element types `MLSequence`
/// defines).
fn sequence_value_to_portable(seq: &MLSequence) -> Result<SequenceValue> {
    match unsafe { seq.r#type() } {
        MLFeatureType::Int64 => {
            let nums = unsafe { seq.int64Values() };
            let mut out = Vec::with_capacity(nums.len());
            for i in 0..nums.len() {
                out.push(nums.objectAtIndex(i).longLongValue());
            }
            Ok(SequenceValue::Int64(out))
        }
        MLFeatureType::String => {
            let strs = unsafe { seq.stringValues() };
            let mut out = Vec::with_capacity(strs.len());
            for i in 0..strs.len() {
                out.push(strs.objectAtIndex(i).to_string());
            }
            Ok(SequenceValue::String(out))
        }
        other => Err(CoreMLError::UnsupportedFeatureType(format!(
            "MLSequence element MLFeatureType({}) is not supported — MLSequence only ever \
             holds Int64 or String elements per Apple's own contract",
            other.0
        ))),
    }
}

/// Best-effort textual form of an `NSDictionary`/metadata key or value
/// typed as `AnyObject`.  `NSString` values stringify directly;
/// anything else (most commonly `NSNumber`, which
/// `MLFeatureValue::featureValueWithDictionary_error`'s own contract
/// guarantees for non-string dictionary keys) falls back to the
/// object's Objective-C `-description`, via
/// [`any_object_description`] — which also makes this helper equally
/// usable for `MLModelCreatorDefinedKey`'s nested metadata dictionary
/// (see [`MlPackageModel::model_metadata`]), whose keys *and* values
/// are both `AnyObject`.
pub(super) fn any_object_key_to_string(key: Retained<AnyObject>) -> String {
    match key.downcast::<NSString>() {
        Ok(s) => s.to_string(),
        Err(obj) => any_object_description(&obj),
    }
}

/// `-description` of an arbitrary `AnyObject`, going through the
/// universal `NSObjectProtocol` every Objective-C object (whether
/// `NSObject`- or `NSProxy`-rooted) implements.
///
/// # Safety justification
///
/// `ProtocolObject::from_ref`'s safe, generic constructor cannot be
/// used here: it requires `dyn NSObjectProtocol: ImplementedBy<T>`,
/// which `objc2` only implements for `T: NSObjectProtocol` — a bound
/// a bare `AnyObject` does not (and structurally cannot) satisfy,
/// since `AnyObject` represents a value of *unknown* class. We
/// therefore reinterpret the reference directly: `ProtocolObject<P>`
/// is `#[repr(C)]` with an `AnyObject` first field and a
/// `PhantomData<P>` second field (zero-sized for every `P`, including
/// unsized `dyn` types), so it has the exact same layout as a bare
/// `AnyObject` regardless of `P` — the same justification
/// `compute_plan_summary`'s device-class lookup uses for the
/// mirror-image cast (`ProtocolObject<dyn MLComputeDeviceProtocol>`
/// -> `AnyObject`, above). The one obligation the type system cannot
/// check for us is whether the pointee truly responds to
/// `-description`; every Cocoa object does (it is part of the
/// universal `NSObject`/`NSProxy` root contract), so this holds for
/// any value that reaches this function.
fn any_object_description(obj: &AnyObject) -> String {
    let proto: &ProtocolObject<dyn NSObjectProtocol> =
        unsafe { &*(obj as *const AnyObject as *const ProtocolObject<dyn NSObjectProtocol>) };
    let desc: Retained<NSObject> = proto.description();
    // `NSObjectProtocol::description`'s own doc guarantees the
    // runtime value is always actually an `NSString` (the return
    // type is narrowed to `NSObject` only because the protocol lives
    // in `objc2`, without an `objc2-foundation` dependency to name
    // `NSString` directly) — but we still downcast defensively
    // rather than `Retained::cast_unchecked`, per this crate's
    // no-`unwrap` policy.
    desc.downcast::<NSString>()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "<no description>".to_string())
}

/// Look `key` up in `metadata` (`NSDictionary<NSString, AnyObject>` —
/// `MLModelMetadataKey` is a plain `NSString` type alias), and if
/// present *and* actually an `NSString`, insert its Rust string under
/// `dest_key` into `out`.  "`key` is `None`" (the weakly-linked
/// symbol is absent on this CoreML framework revision), "not present
/// in `metadata`", and "present but not an `NSString`" are all
/// treated identically: the entry is simply omitted — never an
/// error, never a panic.
pub(super) fn insert_metadata_string(
    metadata: &NSDictionary<NSString, AnyObject>,
    key: Option<&NSString>,
    dest_key: &str,
    out: &mut HashMap<String, String>,
) {
    let Some(key) = key else { return };
    let Some(value) = metadata.objectForKey(key) else {
        return;
    };
    if let Ok(s) = value.downcast::<NSString>() {
        out.insert(dest_key.to_string(), s.to_string());
    }
}

/// Decode an `MLFeatureTypeImage` output's backing `CVPixelBuffer`
/// into a `Tensor`, for the bounded set of standard pixel formats
/// [`MlPackageModel::predict_features`] documents. See that method's
/// doc comment for the exact format → shape/layout table; this is
/// the implementation those guarantees describe.
fn tensor_from_pixel_buffer(pixel_buffer: &CVPixelBuffer) -> Result<Tensor> {
    let format = CVPixelBufferGetPixelFormatType(pixel_buffer);

    // Reject planar layouts up front, before ever locking the
    // buffer — none of the four formats predict_features supports
    // are ever planar (each is either a single one-component plane
    // or one interleaved/chunky plane), so `true` here always
    // indicates an exotic pixel format this decoder does not
    // attempt.
    if CVPixelBufferIsPlanar(pixel_buffer) {
        return Err(CoreMLError::UnsupportedPixelFormat(format!(
            "planar CVPixelBuffer layouts are not supported by predict_features's image \
             decoder (pixelFormatType {format} / \"{}\")",
            four_char_code(format),
        )));
    }

    let width = CVPixelBufferGetWidth(pixel_buffer);
    let height = CVPixelBufferGetHeight(pixel_buffer);

    // SAFETY: `CVPixelBufferLockBaseAddress`/`UnlockBaseAddress` are
    // the only two `CVPixelBuffer` entry points genuinely marked
    // `unsafe` in `objc2-core-video` — every `Get*`/`Is*` accessor
    // used in this function is a safe wrapper. Apple's own contract
    // is that `CVPixelBufferGetBaseAddress` may only be called while
    // the buffer is locked, which Rust's type system cannot express;
    // the explicit `unsafe` acknowledgement of that precondition is
    // localized inside `PixelBufferLockGuard::lock`, below, whose
    // `Drop` impl guarantees the matching unlock runs on every exit
    // path out of this function — including the
    // unsupported-pixel-format error return further down, which
    // happens while still holding the lock.
    let _lock = PixelBufferLockGuard::lock(pixel_buffer, CVPixelBufferLockFlags::ReadOnly)?;

    let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
    let base = CVPixelBufferGetBaseAddress(pixel_buffer);
    let base = core::ptr::NonNull::new(base.cast::<u8>()).ok_or_else(|| {
        CoreMLError::Internal(
            "CVPixelBufferGetBaseAddress returned null after a successful lock".to_string(),
        )
    })?;

    // `kCVPixelFormatType_*` are bare (non-path-qualified) `OSType`
    // constants using Apple's `k`-prefixed naming, not Rust's
    // `SCREAMING_SNAKE_CASE` — matching them as bare identifiers in
    // `match` pattern position triggers rustc's `non_upper_case_globals`
    // lint (it cannot statically tell "match this constant" apart
    // from "bind a fresh variable named like this constant" without
    // the naming heuristic it warns about). Plain equality
    // comparisons in expression position have no such ambiguity, so
    // an `if`/`else if` chain sidesteps the warning entirely while
    // behaving identically to the equivalent `match`.
    if format == kCVPixelFormatType_OneComponent8 {
        let data =
            copy_pixel_rows_to_f32(base, bytes_per_row, width, height, 1, 1, |b| b[0] as f32)?;
        Ok(Tensor::new(data, vec![height, width]))
    } else if format == kCVPixelFormatType_32BGRA {
        let data =
            copy_pixel_rows_to_f32(base, bytes_per_row, width, height, 4, 1, |b| b[0] as f32)?;
        Ok(Tensor::new(data, vec![height, width, 4]))
    } else if format == kCVPixelFormatType_OneComponent16Half {
        let data = copy_pixel_rows_to_f32(base, bytes_per_row, width, height, 1, 2, |b| {
            half::f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32()
        })?;
        Ok(Tensor::new(data, vec![height, width]))
    } else if format == kCVPixelFormatType_OneComponent32Float {
        let data = copy_pixel_rows_to_f32(base, bytes_per_row, width, height, 1, 4, |b| {
            f32::from_ne_bytes([b[0], b[1], b[2], b[3]])
        })?;
        Ok(Tensor::new(data, vec![height, width]))
    } else {
        Err(CoreMLError::UnsupportedPixelFormat(format!(
            "CVPixelBuffer pixelFormatType {format} / \"{}\" is not one of the standard \
             formats predict_features's image decoder supports (OneComponent8, 32BGRA, \
             OneComponent16Half, OneComponent32Float)",
            four_char_code(format),
        )))
    }
}

/// RAII guard ensuring `CVPixelBufferUnlockBaseAddress` always runs
/// exactly once for a successful `CVPixelBufferLockBaseAddress`, no
/// matter which path a caller exits by (including early `?`
/// returns). Apple's docs require lock/unlock calls to use the same
/// [`CVPixelBufferLockFlags`] on both ends ("Non-symmetrical usage of
/// this flag will result in undefined behavior"); storing the flags
/// this guard was locked with and reusing them in `Drop` keeps every
/// lock/unlock pair symmetrical by construction, so a future added
/// return path can never forget to unlock.
struct PixelBufferLockGuard<'a> {
    buffer: &'a CVPixelBuffer,
    flags: CVPixelBufferLockFlags,
}

impl<'a> PixelBufferLockGuard<'a> {
    /// Lock `buffer` with `flags` (`ReadOnly` for every production
    /// read path in this module; tests that need to *write*
    /// synthetic pixel data pass the empty/read-write flag set
    /// instead).
    fn lock(buffer: &'a CVPixelBuffer, flags: CVPixelBufferLockFlags) -> Result<Self> {
        // SAFETY: `CVPixelBufferLockBaseAddress` has no precondition
        // beyond `buffer` being a valid, live `CVPixelBuffer`
        // reference — which Rust's `&CVPixelBuffer` guarantees — and
        // matching every lock with exactly one unlock using the same
        // flags, which this guard's `Drop` impl discharges.
        let ret = unsafe { CVPixelBufferLockBaseAddress(buffer, flags) };
        if ret != kCVReturnSuccess {
            return Err(CoreMLError::Internal(format!(
                "CVPixelBufferLockBaseAddress failed with CVReturn {ret}"
            )));
        }
        Ok(Self { buffer, flags })
    }
}

impl Drop for PixelBufferLockGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: matches the successful lock this guard was
        // constructed from, with the same flags Apple's docs require
        // for symmetry. Best-effort: `Drop` cannot return a
        // `Result`, and a failing unlock here would indicate a
        // CoreVideo-internal problem outside this runtime's control
        // — there is nothing actionable to do beyond not panicking.
        let _ = unsafe { CVPixelBufferUnlockBaseAddress(self.buffer, self.flags) };
    }
}

/// Copy `height` rows of `elems_per_pixel`-element, `elem_bytes`-wide
/// pixels out of a **locked** `CVPixelBuffer`'s base address,
/// skipping whatever row padding `bytes_per_row` carries beyond each
/// row's logical `width * elems_per_pixel * elem_bytes` — the same
/// "declared shape may not match the buffer's real stride" hazard
/// [`read_raw_bytes`] documents for `MLMultiArray` (the concrete
/// motivating case there was SCRFD's cache-line-padded output;
/// `CVPixelBuffer` rows are padded for the same alignment reasons),
/// mirrored here for `CVPixelBuffer`. `decode_elem` converts one
/// element's raw bytes (always exactly `elem_bytes` long) to `f32`.
fn copy_pixel_rows_to_f32(
    base: core::ptr::NonNull<u8>,
    bytes_per_row: usize,
    width: usize,
    height: usize,
    elems_per_pixel: usize,
    elem_bytes: usize,
    decode_elem: impl Fn(&[u8]) -> f32,
) -> Result<Vec<f32>> {
    let row_logical_bytes = width
        .checked_mul(elems_per_pixel)
        .and_then(|v| v.checked_mul(elem_bytes))
        .ok_or_else(|| {
            CoreMLError::Internal("CVPixelBuffer row byte count overflowed usize".to_string())
        })?;
    if row_logical_bytes > bytes_per_row {
        return Err(CoreMLError::Internal(format!(
            "CVPixelBuffer row needs {row_logical_bytes} bytes but bytesPerRow is only \
             {bytes_per_row} — refusing to read past the row's own stride",
        )));
    }

    let total_elems = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(elems_per_pixel))
        .ok_or_else(|| {
            CoreMLError::Internal("CVPixelBuffer element count overflowed usize".to_string())
        })?;
    let mut out = Vec::with_capacity(total_elems);
    for row in 0..height {
        let row_offset = row.checked_mul(bytes_per_row).ok_or_else(|| {
            CoreMLError::Internal("CVPixelBuffer row offset overflowed usize".to_string())
        })?;
        // SAFETY: `base` is the locked buffer's base address, live
        // for at least as long as the `PixelBufferLockGuard` this
        // function's caller holds (which outlives this whole call).
        // `row < height` and `row_logical_bytes <= bytes_per_row`
        // (checked above) together bound the read to at most
        // `(height - 1) * bytes_per_row + bytes_per_row ==
        // height * bytes_per_row` bytes from `base` — exactly the
        // allocation size `CVPixelBufferGetBytesPerRow`'s own
        // documentation guarantees for a non-planar buffer
        // ("bytesPerRow * height will cover the entire image"),
        // which this function's only caller
        // (`tensor_from_pixel_buffer`) has already confirmed via
        // `CVPixelBufferIsPlanar`.
        let row_ptr = unsafe { base.as_ptr().add(row_offset) };
        let row_slice = unsafe { core::slice::from_raw_parts(row_ptr, row_logical_bytes) };
        for pixel_bytes in row_slice.chunks_exact(elem_bytes) {
            out.push(decode_elem(pixel_bytes));
        }
    }
    Ok(out)
}

/// Render a `CVPixelBuffer` `pixelFormatType` `OSType` as its
/// 4-character code (e.g. `0x42475241` -> `"BGRA"`), falling back to
/// an escaped hex byte for anything outside the printable-ASCII
/// range — some CoreVideo formats intentionally use non-printable
/// leading bytes. Purely for clearer error messages; never used for
/// dispatch logic.
fn four_char_code(v: u32) -> String {
    v.to_be_bytes()
        .iter()
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                (b as char).to_string()
            } else {
                format!("\\x{b:02x}")
            }
        })
        .collect()
}

/// Unit tests for `multi_array_from_owned` — nested inside
/// `macos_impl` (rather than the crate's usual root-level `mod
/// tests`) because `multi_array_from_owned` is `#[cfg(test)]`-gated
/// and module-private; a descendant module gets access for free
/// without widening its visibility.
#[cfg(test)]
mod owned_array_tests {
    use super::*;

    /// Build an array from an owned `Vec<f32>`, read it back, and
    /// confirm the round-tripped values and shape match.
    #[test]
    fn multi_array_from_owned_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let arr = multi_array_from_owned(data, &[2, 2])
            .expect("multi_array_from_owned should build a 2x2 Float32 MLMultiArray");
        let tensor = tensor_from_multi_array(&arr)
            .expect("tensor_from_multi_array should read back the array just built");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    /// `multi_array_from_owned`'s own `Mutex<Option<Vec<f32>>>` state
    /// is private to its closure and can't be observed from outside,
    /// so this test exercises the *identical* mechanism it uses
    /// (`initWithDataPointer_shape_dataType_strides_deallocator_error`
    /// plus a `block2::RcBlock` that takes-and-drops an owned
    /// `Vec<f32>` via a `Mutex`) with an `Arc<AtomicBool>` flag added,
    /// to verify CoreML actually invokes a `block2::RcBlock`
    /// deallocator when an owned-backing `MLMultiArray`'s last
    /// reference is released.
    ///
    /// NOTE: Apple does not document *when* (synchronously vs.
    /// deferred to an autorelease-pool drain) a deallocator fires
    /// relative to the Rust-visible `drop`. Empirically, on this
    /// machine's macOS/CoreML, a bare `drop` of a directly-owned
    /// `Retained<MLMultiArray>` (no autorelease pool involved —
    /// `objc2`'s `Retained::drop` releases directly via
    /// `objc_release`, it does not autorelease first) fires the
    /// deallocator synchronously, which this test asserts. If a
    /// future OS revision defers this, the fix is to drain an
    /// autorelease pool between the drop and the assertion, not to
    /// change `multi_array_from_owned` itself.
    #[test]
    fn multi_array_from_owned_deallocator_fires_on_drop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let fired = Arc::new(AtomicBool::new(false));
        let fired_for_block = Arc::clone(&fired);

        let mut data = vec![0.0f32; 4];
        let (shape_arr, strides_arr) = build_shape_and_strides_arrays(&[2, 2]);
        let data_pointer = core::ptr::NonNull::new(data.as_mut_ptr().cast::<core::ffi::c_void>())
            .expect("Vec::as_mut_ptr is never null");
        let owned: Mutex<Option<Vec<f32>>> = Mutex::new(Some(data));
        let dealloc = block2::RcBlock::new(move |_ptr: core::ptr::NonNull<core::ffi::c_void>| {
            let mut guard = owned
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            drop(guard.take());
            fired_for_block.store(true, Ordering::SeqCst);
        });

        // SAFETY: mirrors `multi_array_from_owned`'s own SAFETY
        // argument — `data_pointer` is valid/aligned/non-null, and
        // `Some(&dealloc)` is correct because this test-local
        // MLMultiArray takes ownership of `data` via the mutex.
        let arr = unsafe {
            MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                MLMultiArray::alloc(),
                data_pointer,
                &shape_arr,
                MLMultiArrayDataType::Float32,
                &strides_arr,
                Some(&dealloc),
            )
        }
        .expect("initWithDataPointer_shape_dataType_strides_deallocator_error should succeed");

        assert!(
            !fired.load(Ordering::SeqCst),
            "deallocator must not fire before the array is dropped"
        );
        drop(arr);

        assert!(
            fired.load(Ordering::SeqCst),
            "deallocator did not fire synchronously on drop; CoreML may be deferring \
             release to an autorelease pool — see the NOTE on this test"
        );
    }
}
