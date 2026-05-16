//! `MlPackageModel` — load a pre-converted `.mlpackage` (or `.mlmodelc`) and
//! run inference through Apple's CoreML runtime.
//!
//! The runtime is a thin wrapper over `MLModel::predictionFromFeatures_error`.
//! It deliberately exposes only a graph-level interface — there is no
//! per-op dispatch surface, because CoreML's whole-graph scheduler is what
//! delivers the 17–25× speedups on Apple Silicon (per the OxiFace
//! ArcFace / SCRFD / InSwapper sub-gates).
//!
//! ## Threading
//!
//! [`MLModel`] is documented as thread-safe by Apple — multiple threads may
//! call `predictionFromFeatures_error:` concurrently on the same instance.
//! We expose this as `predict(&self, ...)` and provide manual `Send` + `Sync`
//! impls for [`MlPackageModel`] (the `Retained<MLModel>` field does not
//! auto-derive these, since `objc2` cannot tell from the type alone).
//!
//! ## I/O details
//!
//! * Inputs are projected into `MLMultiArray` instances (Float32) by
//!   `copy_nonoverlapping` from the supplied [`oxionnx_core::Tensor::data`]
//!   slice.  The `MLMultiArray` owns its backing buffer; the `Tensor` is no
//!   longer referenced once the call returns.
//! * Outputs are read via `getBytesWithHandler:` (the modern, non-deprecated
//!   path).  Both `Float32` and `Float16` outputs are accepted — the latter
//!   is up-converted to `f32` for the returned `Tensor`.

use std::collections::HashMap;
use std::path::Path;

use crate::compute::{ComputePlanSummary, MlComputeUnits};
use crate::error::{CoreMLError, Result};

#[cfg(target_os = "macos")]
mod macos_impl {
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
        MLComputePlan, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
        MLModelConfiguration, MLModelStructureProgramOperation, MLMultiArray, MLMultiArrayDataType,
    };
    use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

    use oxionnx_core::Tensor;

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

        /// `.mlpackage` is a *directory* bundle — there is no in-memory
        /// equivalent.  Provided for API parity with `Session::from_bytes`.
        pub fn load_from_bytes(_bytes: &[u8], _compute_units: MlComputeUnits) -> Result<Self> {
            Err(CoreMLError::UnsupportedFormat(
                "MlPackageModel requires a directory path; load_from_bytes is not supported",
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

        /// Run inference.
        ///
        /// `inputs` is keyed by the names returned by [`input_names`](Self::input_names);
        /// every required input must be present.  Extra entries are ignored.
        ///
        /// Returns a map keyed by [`output_names`](Self::output_names).
        pub fn predict(&self, inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
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
            let p_obj: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*provider);

            let outputs = unsafe {
                self.model
                    .predictionFromFeatures_error(p_obj)
                    .map_err(nserror_to_coreml)?
            };

            // Pull every declared output back as an f32 Tensor.
            let mut result = HashMap::with_capacity(self.output_names.len());
            for name in &self.output_names {
                let key = NSString::from_str(name);
                let fv = unsafe { outputs.featureValueForName(&key) }
                    .ok_or_else(|| CoreMLError::MissingOutput(name.clone()))?;
                let arr = unsafe { fv.multiArrayValue() }.ok_or_else(|| {
                    CoreMLError::MissingOutput(format!("{name} (not MLMultiArray)"))
                })?;
                let tensor = tensor_from_multi_array(&arr)?;
                result.insert(name.clone(), tensor);
            }
            Ok(result)
        }

        /// Report the per-device op-count breakdown for this model under the
        /// configured compute-units policy.
        ///
        /// Reloads the bundle through `MLComputePlan` (the framework does
        /// not let us introspect a live `MLModel`).  The async completion
        /// handler is bridged to a synchronous wait via `Condvar`, with a
        /// 60-second timeout.
        pub fn compute_plan_summary(&self) -> Result<ComputePlanSummary> {
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
                MLComputePlan::loadContentsOfURL_configuration_completionHandler(
                    &url, &cfg, &block,
                );
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

            let structure = unsafe { plan.modelStructure() };
            let program = unsafe { structure.program() }.ok_or_else(|| {
                CoreMLError::ComputePlan("not an MLProgram (no program tree)".to_string())
            })?;
            let funcs = unsafe { program.functions() };
            let main_key = NSString::from_str("main");
            let main_func = funcs.objectForKey(&main_key).ok_or_else(|| {
                CoreMLError::ComputePlan("program has no 'main' function".to_string())
            })?;
            let blk = unsafe { main_func.block() };
            let ops = unsafe { blk.operations() };
            let n = ops.len();

            let mut summary = ComputePlanSummary::default();
            for i in 0..n {
                let op: Retained<MLModelStructureProgramOperation> = ops.objectAtIndex(i);
                let opname = unsafe { op.operatorName() }.to_string();
                // `const` ops have no compute device — bucket them as unknown.
                if opname == "const" {
                    summary.unknown_ops += 1;
                    continue;
                }
                match unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) } {
                    Some(usage) => {
                        let dev = unsafe { usage.preferredComputeDevice() };
                        // ProtocolObject is #[repr(C)] with `inner: AnyObject`
                        // first — the pointer cast is sound.
                        let dev_obj: &AnyObject = unsafe {
                            // SAFETY: ProtocolObject is #[repr(C)] with
                            // an `inner: AnyObject` first field, so the
                            // pointer cast is sound.
                            &*(&*dev as *const _ as *const AnyObject)
                        };
                        let class_name = dev_obj
                            .class()
                            .name()
                            .to_str()
                            .unwrap_or("UnknownDeviceClass")
                            .to_string();
                        if class_name.contains("NeuralEngine") {
                            summary.ane_ops += 1;
                        } else if class_name.contains("GPU") {
                            summary.gpu_ops += 1;
                        } else if class_name.contains("CPU") {
                            summary.cpu_ops += 1;
                        } else {
                            summary.unknown_ops += 1;
                        }
                    }
                    None => {
                        summary.unknown_ops += 1;
                    }
                }
            }
            Ok(summary)
        }
    }

    // ────────────────────────────────────────────────────────────────────── //
    // Helpers — every helper that crosses the FFI boundary lives here.       //
    // ────────────────────────────────────────────────────────────────────── //

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

    /// Build a Float32 `MLMultiArray` from a borrowed slice — copies into
    /// fresh CoreML-owned storage.  Avoids the lifetime hazards of the
    /// `initWithDataPointer_*` zero-copy variant; the I/O cost is dominated
    /// by neural inference anyway.
    fn multi_array_from_f32(data: &[f32], shape: &[usize]) -> Result<Retained<MLMultiArray>> {
        let shape_numbers: Vec<Retained<NSNumber>> = shape
            .iter()
            .map(|d| NSNumber::new_isize(*d as isize))
            .collect();
        let shape_arr: Retained<NSArray<NSNumber>> = NSArray::from_retained_slice(&shape_numbers);
        let arr = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &shape_arr,
                MLMultiArrayDataType::Float32,
            )
            .map_err(nserror_to_coreml)?
        };
        let count = unsafe { arr.count() } as usize;
        if count != data.len() {
            return Err(CoreMLError::InputMismatch(format!(
                "MLMultiArray element count {count} does not match supplied tensor of {} elements",
                data.len()
            )));
        }
        // SAFETY: `dataPointer` is deprecated for *read* access on modern
        // CoreML, but for *write* into a freshly-allocated MLMultiArray it
        // remains the documented contract: the buffer is contiguous Float32
        // when we asked for Float32.  We just allocated it, so no reader
        // exists yet.
        #[allow(deprecated)]
        unsafe {
            let ptr = arr.dataPointer().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
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

    /// Convert an `MLMultiArray` of Float32 / Float16 to an owned f32
    /// [`Tensor`].  Other dtypes raise [`CoreMLError::UnsupportedOutputDtype`].
    ///
    /// Uses `getBytesWithHandler:` (the documented modern API) — the
    /// supplied block is invoked synchronously with a buffer pointer that
    /// is valid only for the duration of the call.  We capture only `Copy`
    /// values inside the block (raw pointers, the dtype, and a `&Cell` for
    /// the status) so the closure satisfies `block2::StackBlock`'s
    /// `Clone` bound.
    /// Convert an output `MLMultiArray` to a tightly-packed C-contiguous f32
    /// [`Tensor`].
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
    /// We read the per-element value via `byte_ptr.offset(idx · stride · sizeof(elem))`
    /// where `idx` is the C-major destination index decomposed via the
    /// declared `shape`, multiplied with the array's reported `strides()`.
    /// On the C-contiguous fast path this still vectorises to a
    /// `copy_nonoverlapping`; on the strided slow path it walks element by
    /// element.
    fn tensor_from_multi_array(arr: &MLMultiArray) -> Result<Tensor> {
        let shape = read_shape(arr);
        let strides = read_strides(arr);
        let dt = unsafe { arr.dataType() };

        // C-contiguous element count = product of shape (NOT `arr.count()`,
        // which can include stride padding on some allocations).
        let n_c_contig: usize = shape.iter().product::<usize>();

        // Compute what *would-be* C-contiguous strides for this shape.
        let rank = shape.len();
        let mut c_strides: Vec<isize> = vec![0; rank];
        if rank > 0 {
            c_strides[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                c_strides[i] = c_strides[i + 1] * shape[i + 1] as isize;
            }
        }
        let is_c_contiguous = strides == c_strides;

        // Allocate the destination outside the closure; the closure writes
        // through a raw pointer.  The buffer is alive for the entire
        // function (and therefore for the entire synchronous handler call).
        let mut out: Vec<f32> = vec![0.0_f32; n_c_contig];
        let dst_ptr: *mut f32 = out.as_mut_ptr();

        // We need to capture `shape`, `strides`, and `is_c_contiguous` into
        // the block.  `block2::StackBlock` requires `Fn`, so values that
        // change per-iteration go through a `Cell` of `u8` (the status word
        // — encoding 0 = never invoked, 1 = success, 2 = unsupported dtype).
        // The `shape`/`strides` slices are read-only for the duration of
        // the block, so they're captured by reference into the closure.
        let status: std::cell::Cell<u8> = std::cell::Cell::new(0);
        let status_ref: &std::cell::Cell<u8> = &status;
        let shape_ref: &Vec<usize> = &shape;
        let strides_ref: &Vec<isize> = &strides;

        let handler = block2::StackBlock::new(
            |bytes: core::ptr::NonNull<core::ffi::c_void>, _size: isize| {
                let elem_bytes: usize = match dt {
                    MLMultiArrayDataType::Float32 => core::mem::size_of::<f32>(),
                    MLMultiArrayDataType::Float16 => core::mem::size_of::<u16>(),
                    _ => {
                        status_ref.set(2);
                        return;
                    }
                };
                let base = bytes.as_ptr() as *const u8;
                if is_c_contiguous {
                    // Fast path — bulk copy.
                    unsafe {
                        match dt {
                            MLMultiArrayDataType::Float32 => {
                                let p = base as *const f32;
                                std::ptr::copy_nonoverlapping(p, dst_ptr, n_c_contig);
                            }
                            MLMultiArrayDataType::Float16 => {
                                let p = base as *const u16;
                                for i in 0..n_c_contig {
                                    let raw = *p.add(i);
                                    *dst_ptr.add(i) = half::f16::from_bits(raw).to_f32();
                                }
                            }
                            _ => unreachable!("dtype-checked above"),
                        }
                    }
                } else {
                    // Stride-aware slow path.  Walk a multi-dimensional
                    // index in C-major order; translate each step to a
                    // source byte offset using the declared strides.
                    let mut idx: Vec<usize> = vec![0; shape_ref.len()];
                    for dst in 0..n_c_contig {
                        let mut src_offset: isize = 0;
                        for d in 0..shape_ref.len() {
                            src_offset += idx[d] as isize * strides_ref[d];
                        }
                        unsafe {
                            let byte_ptr = base.offset(src_offset * elem_bytes as isize);
                            let v = match dt {
                                MLMultiArrayDataType::Float32 => *(byte_ptr as *const f32),
                                MLMultiArrayDataType::Float16 => {
                                    let raw = *(byte_ptr as *const u16);
                                    half::f16::from_bits(raw).to_f32()
                                }
                                _ => unreachable!(),
                            };
                            *dst_ptr.add(dst) = v;
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
                status_ref.set(1);
            },
        );

        unsafe { arr.getBytesWithHandler(&handler) };

        match status.get() {
            1 => Ok(Tensor::new(out, shape)),
            2 => Err(CoreMLError::UnsupportedOutputDtype(format!("{dt:?}"))),
            _ => Err(CoreMLError::Internal(
                "getBytesWithHandler did not invoke its handler".to_string(),
            )),
        }
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
}

#[cfg(target_os = "macos")]
pub use macos_impl::MlPackageModel;

// ──────────────────────────────────────────────────────────────────────────── //
// Non-macOS stub — preserves the API surface so dependent crates compile     //
// portably.  Every method short-circuits to `UnsupportedPlatform`.            //
// ──────────────────────────────────────────────────────────────────────────── //

#[cfg(not(target_os = "macos"))]
mod stub_impl {
    use super::*;

    /// Stub that always fails on non-macOS targets.  Present so callers can
    /// share code between platforms behind `#[cfg(feature = "coreml")]`.
    pub struct MlPackageModel {
        _private: (),
    }

    impl MlPackageModel {
        /// Always returns [`CoreMLError::UnsupportedPlatform`].
        pub fn load(_path: impl AsRef<Path>, _compute_units: MlComputeUnits) -> Result<Self> {
            Err(CoreMLError::UnsupportedPlatform)
        }

        /// Always returns [`CoreMLError::UnsupportedPlatform`].
        pub fn load_from_bytes(_bytes: &[u8], _compute_units: MlComputeUnits) -> Result<Self> {
            Err(CoreMLError::UnsupportedPlatform)
        }

        /// Always returns an empty list — the stub holds no model.
        pub fn input_names(&self) -> Vec<String> {
            Vec::new()
        }

        /// Always returns an empty list — the stub holds no model.
        pub fn output_names(&self) -> Vec<String> {
            Vec::new()
        }

        /// Always returns [`CoreMLError::UnsupportedPlatform`].
        pub fn predict(
            &self,
            _inputs: &HashMap<String, oxionnx_core::Tensor>,
        ) -> Result<HashMap<String, oxionnx_core::Tensor>> {
            Err(CoreMLError::UnsupportedPlatform)
        }

        /// Always returns [`CoreMLError::UnsupportedPlatform`].
        pub fn warm_up(
            &self,
            _input_template: &HashMap<String, oxionnx_core::Tensor>,
        ) -> Result<()> {
            Err(CoreMLError::UnsupportedPlatform)
        }

        /// Always returns [`CoreMLError::UnsupportedPlatform`].
        pub fn compute_plan_summary(&self) -> Result<ComputePlanSummary> {
            Err(CoreMLError::UnsupportedPlatform)
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use stub_impl::MlPackageModel;

// ──────────────────────────────────────────────────────────────────────────── //
// Tests — every test that hits the framework is `#[ignore]` because the      //
// bundles live outside the source tree.  Manual run command:                 //
//                                                                            //
//     cargo test -p oxionnx-coreml -- --ignored --test-threads=1             //
// ──────────────────────────────────────────────────────────────────────────── //

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use oxionnx_core::Tensor;

    const ARCFACE_PATH: &str = "/tmp/w600k_r50.mlpackage";

    /// Smoke test: model loads and surfaces a single input + single output.
    /// Requires `/tmp/w600k_r50.mlpackage` from the OxiFace ArcFace sub-gate.
    #[test]
    #[ignore]
    fn test_load_arcface() {
        let model = MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All)
            .expect("load /tmp/w600k_r50.mlpackage (run the OxiFace conversion script first)");
        assert_eq!(
            model.input_names().len(),
            1,
            "ArcFace has exactly one input"
        );
        assert_eq!(
            model.output_names().len(),
            1,
            "ArcFace has exactly one output"
        );
    }

    /// End-to-end roundtrip: 1×3×112×112 input -> 1×512 embedding.
    #[test]
    #[ignore]
    fn test_predict_arcface_returns_512_dim_embedding() {
        let model = MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All)
            .expect("load arcface .mlpackage");
        let input_name = model
            .input_names()
            .into_iter()
            .next()
            .expect("at least one input");
        let output_name = model
            .output_names()
            .into_iter()
            .next()
            .expect("at least one output");

        let n = 3 * 112 * 112;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) / 1000.0).collect();
        let tensor = Tensor::new(data, vec![1, 3, 112, 112]);
        let mut inputs = HashMap::new();
        inputs.insert(input_name, tensor);
        let outputs = model.predict(&inputs).expect("prediction");
        let out = outputs
            .get(&output_name)
            .expect("declared output present in result map");
        assert_eq!(out.data.len(), 512, "ArcFace embedding dimension");
        assert_eq!(out.shape.iter().product::<usize>(), 512, "shape sanity");
    }

    /// Confirm the compute-plan introspection actually places work on the ANE
    /// for the ArcFace model (the headline finding from the sub-gate: 97 %
    /// of compute ops on ANE).
    #[test]
    #[ignore]
    fn test_compute_plan_reports_ane_engagement() {
        let model = MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All)
            .expect("load arcface .mlpackage");
        let summary = model.compute_plan_summary().expect("compute plan");
        assert!(
            summary.ane_ops > 0,
            "expected ArcFace to engage the ANE, got {summary:?}",
        );
        let frac = summary.ane_fraction();
        assert!(
            frac > 0.5,
            "ArcFace should run majority on ANE, got fraction {frac}",
        );
    }

    /// load_from_bytes is intentionally unsupported and must surface a
    /// clean error instead of panicking or attempting to parse the bytes.
    #[test]
    fn load_from_bytes_returns_unsupported_format() {
        let r = MlPackageModel::load_from_bytes(&[], MlComputeUnits::All);
        match r {
            Err(CoreMLError::UnsupportedFormat(_)) => {}
            Err(other) => panic!("expected UnsupportedFormat, got {other:?}"),
            Ok(_) => panic!("expected UnsupportedFormat, got Ok"),
        }
    }

    /// load() of a non-existent path must produce CoreMLError::Io rather
    /// than crossing into Objective-C with a bogus path.
    #[test]
    fn load_missing_path_returns_io_error() {
        let r = MlPackageModel::load(
            "/tmp/this/path/does/not/exist.mlpackage",
            MlComputeUnits::All,
        );
        match r {
            Err(CoreMLError::Io { .. }) => {}
            Err(other) => panic!("expected Io, got {other:?}"),
            Ok(_) => panic!("expected Io, got Ok"),
        }
    }
}

#[cfg(all(test, not(target_os = "macos")))]
mod stub_tests {
    use super::*;

    /// Every operation on the non-macOS stub must return UnsupportedPlatform.
    #[test]
    fn stub_load_returns_unsupported_platform() {
        let r = MlPackageModel::load("anywhere", MlComputeUnits::All);
        assert!(matches!(r, Err(CoreMLError::UnsupportedPlatform)));
    }

    #[test]
    fn stub_load_from_bytes_returns_unsupported_platform() {
        let r = MlPackageModel::load_from_bytes(&[], MlComputeUnits::All);
        assert!(matches!(r, Err(CoreMLError::UnsupportedPlatform)));
    }
}
