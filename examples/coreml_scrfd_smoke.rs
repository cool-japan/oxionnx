//! CoreML smoke benchmark for SCRFD (det_10g) — sub-gate for OxiFace's
//! face-detection acceleration plan.
//!
//! Loads a `.mlpackage` (compiles to `.mlmodelc` first), runs the model with
//! `MLComputeUnits::All` so CoreML can pick ANE/GPU/CPU, then runs the same
//! input with `MLComputeUnits::CPUOnly` for an apples-to-apples speedup.
//! Reports:
//!  * median latency (CoreML All vs CoreML CPUOnly)
//!  * per-output element counts (must match SCRFD-640 contract)
//!  * MLComputePlan: which compute device CoreML actually picked per op
//!
//! Output names are discovered at runtime (do NOT hardcode — coremltools
//! rewrites them to `var_<id>`, and the rewrite is not deterministic across
//! conversions).
//!
//! Build / run (macOS only):
//!     cargo run --release --example coreml_scrfd_smoke -- \
//!         /tmp/det_10g.mlpackage
#![allow(clippy::too_many_lines, deprecated)]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("coreml_scrfd_smoke only runs on macOS (CoreML).");
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::run();
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Condvar, Mutex};

    /// Sync-bridge slot: holds a pending plan/error result for the condvar protocol.
    /// `Retained<MLComputePlan>` is not `Send`/`Sync` due to Objective-C retain semantics
    /// (retain/release must be balanced on the originating thread), so we use `Arc`
    /// here only as an intra-thread sync handle — the block fires on the same run-loop
    /// thread before we inspect the value, matching Apple's documented usage pattern.
    type ComputePlanSlot = Arc<(
        Mutex<Option<Result<Retained<MLComputePlan>, Retained<NSError>>>>,
        Condvar,
    )>;
    use std::time::Instant;

    use block2::StackBlock;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2::AnyThread;
    use objc2_core_ml::{
        MLComputePlan, MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider,
        MLFeatureValue, MLModel, MLModelConfiguration, MLModelStructureProgramOperation,
        MLMultiArray, MLMultiArrayDataType,
    };
    use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

    const N_WARMUP: usize = 5;
    const N_ITERS: usize = 50;
    const INPUT_NAME_CORE_ML: &str = "input_1"; // coremltools rewrote 'input.1'
    const N_INPUT: usize = 3 * 640 * 640;

    // Expected per-stride anchor counts for SCRFD-10g at 640x640 (2 anchors/cell):
    //   stride 8  : 80 * 80 * 2 = 12800
    //   stride 16 : 40 * 40 * 2 =  3200
    //   stride 32 : 20 * 20 * 2 =   800
    // Heads: cls (1), bbox (4), kps (10).
    const EXPECTED_OUTPUT_COUNTS: &[(&str, usize)] = &[
        ("stride8.cls", 12800),
        ("stride16.cls", 3200),
        ("stride32.cls", 800),
        ("stride8.bbox", 12800 * 4),
        ("stride16.bbox", 3200 * 4),
        ("stride32.bbox", 800 * 4),
        ("stride8.kps", 12800 * 10),
        ("stride16.kps", 3200 * 10),
        ("stride32.kps", 800 * 10),
    ];
    const EXPECTED_TOTAL_ELEMS: usize = 252_000;

    // ----------------------------------------------------------------------------
    // Utility
    // ----------------------------------------------------------------------------

    fn deterministic_input() -> Vec<f32> {
        // Same shape as ArcFace template: ramp / 1000.0. Not a real face — we are
        // measuring perf and ANE engagement, not accuracy.
        (0..N_INPUT).map(|i| (i as f32) / 100_000.0).collect()
    }

    fn ns_str(s: &str) -> Retained<NSString> {
        NSString::from_str(s)
    }

    fn nsurl_for_dir(path: &Path) -> Retained<NSURL> {
        let s = ns_str(&path.to_string_lossy());
        NSURL::fileURLWithPath_isDirectory(&s, true)
    }

    fn median(samples: &mut [f64]) -> f64 {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        samples[samples.len() / 2]
    }

    // ----------------------------------------------------------------------------
    // CoreML: compile mlpackage -> mlmodelc, load, configure compute units = All
    // ----------------------------------------------------------------------------

    fn compile_if_needed(path: &Path) -> PathBuf {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext == "mlmodelc" {
            return path.to_path_buf();
        }
        let url = nsurl_for_dir(path);
        let compiled: Retained<NSURL> = unsafe {
            #[allow(deprecated)]
            match MLModel::compileModelAtURL_error(&url) {
                Ok(u) => u,
                Err(e) => panic!("compileModelAtURL failed: {:?}", e),
            }
        };
        let s = compiled.path().expect("compiled URL has no path");
        PathBuf::from(s.to_string())
    }

    fn load_model(mlmodelc_path: &Path, units: MLComputeUnits) -> Retained<MLModel> {
        let url = nsurl_for_dir(mlmodelc_path);
        let cfg = unsafe { MLModelConfiguration::new() };
        unsafe { cfg.setComputeUnits(units) };
        unsafe {
            match MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg) {
                Ok(m) => m,
                Err(e) => panic!("load mlmodelc failed: {:?}", e),
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Build an MLMultiArray from an f32 slice
    // ----------------------------------------------------------------------------

    fn multi_array_from_f32_owned(data: &[f32], shape: &[usize]) -> Retained<MLMultiArray> {
        let shape_arr: Retained<NSArray<NSNumber>> = NSArray::from_retained_slice(
            &shape
                .iter()
                .map(|d| NSNumber::new_isize(*d as isize))
                .collect::<Vec<_>>(),
        );
        let arr = unsafe {
            match MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &shape_arr,
                MLMultiArrayDataType::Float32,
            ) {
                Ok(a) => a,
                Err(e) => panic!("MLMultiArray init failed: {:?}", e),
            }
        };
        let ptr = unsafe { arr.dataPointer() }.as_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        arr
    }

    // ----------------------------------------------------------------------------
    // MLDictionaryFeatureProvider with one input
    // ----------------------------------------------------------------------------

    fn make_provider(name: &str, arr: &MLMultiArray) -> Retained<MLDictionaryFeatureProvider> {
        let key = ns_str(name);
        let val: Retained<MLFeatureValue> =
            unsafe { MLFeatureValue::featureValueWithMultiArray(arr) };
        let keys = [&*key];
        let vals = [&*val];
        let dict: Retained<NSDictionary<NSString, MLFeatureValue>> =
            NSDictionary::from_slices(&keys, &vals);
        let dict_any: &NSDictionary<NSString, AnyObject> = unsafe {
            &*(dict.as_ref() as *const NSDictionary<NSString, MLFeatureValue>
                as *const NSDictionary<NSString, AnyObject>)
        };
        unsafe {
            match MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                dict_any,
            ) {
                Ok(p) => p,
                Err(e) => panic!("FeatureProvider init failed: {:?}", e),
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Discover output feature names from an MLModel.
    // ----------------------------------------------------------------------------

    fn discover_output_names(model: &MLModel) -> Vec<String> {
        let desc = unsafe { model.modelDescription() };
        let dict = unsafe { desc.outputDescriptionsByName() };
        let keys = dict.allKeys();
        let mut out = Vec::with_capacity(keys.len());
        for i in 0..keys.len() {
            let k = keys.objectAtIndex(i);
            out.push(k.to_string());
        }
        // Sort for stable iteration order across runs.
        out.sort();
        out
    }

    // ----------------------------------------------------------------------------
    // Run prediction once, return Vec<(name, Vec<f32>)> with all outputs.
    // Handles both Float32 and Float16 dtypes (mlprogram sometimes lowers).
    // ----------------------------------------------------------------------------

    /// Read an `MLMultiArray` into a tightly-packed C-contiguous f32 Vec, honoring
    /// the array's reported strides.  CoreML may allocate outputs with strides
    /// optimised for ANE / GPU dispatch (e.g. shape `[800, 1]` with strides
    /// `[32, 1]` — each "row" padded to 32 elements for cache-line alignment),
    /// in which case a naive `copy_nonoverlapping` on `dataPointer()` reads
    /// padding bytes and badly mis-aligns the result.  This is the same logic
    /// used by `oxionnx_coreml::package::tensor_from_multi_array` — kept here
    /// as a parallel implementation so the smoke test stays self-contained.
    fn extract_array(arr: &MLMultiArray) -> Vec<f32> {
        let dt = unsafe { arr.dataType() };
        let shape_ns = unsafe { arr.shape() };
        let strides_ns = unsafe { arr.strides() };
        let shape: Vec<usize> = (0..shape_ns.len())
            .map(|i| shape_ns.objectAtIndex(i).longLongValue() as usize)
            .collect();
        let strides: Vec<isize> = (0..strides_ns.len())
            .map(|i| strides_ns.objectAtIndex(i).longLongValue() as isize)
            .collect();
        let n_c_contig: usize = shape.iter().product::<usize>();
        let p = unsafe { arr.dataPointer() }.as_ptr();
        let mut out = vec![0.0_f32; n_c_contig];

        let rank = shape.len();
        let mut c_strides: Vec<isize> = vec![0; rank];
        if rank > 0 {
            c_strides[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                c_strides[i] = c_strides[i + 1] * shape[i + 1] as isize;
            }
        }
        let is_c_contiguous = strides == c_strides;

        if is_c_contiguous {
            match dt {
                MLMultiArrayDataType::Float32 => {
                    let pf = p as *const f32;
                    unsafe {
                        std::ptr::copy_nonoverlapping(pf, out.as_mut_ptr(), n_c_contig);
                    }
                }
                MLMultiArrayDataType::Float16 => {
                    let p16 = p as *const u16;
                    for (i, slot) in out.iter_mut().enumerate() {
                        let raw = unsafe { *p16.add(i) };
                        *slot = half::f16::from_bits(raw).to_f32();
                    }
                }
                _ => panic!("unexpected output dtype {:?}", dt),
            }
        } else {
            let elem_bytes: isize = match dt {
                MLMultiArrayDataType::Float32 => 4,
                MLMultiArrayDataType::Float16 => 2,
                _ => panic!("unexpected output dtype {:?}", dt),
            };
            let mut idx = vec![0usize; rank];
            for dst_slot in out.iter_mut() {
                let mut src_offset: isize = 0;
                for d in 0..rank {
                    src_offset += idx[d] as isize * strides[d];
                }
                *dst_slot = unsafe {
                    let byte_ptr = (p as *const u8).offset(src_offset * elem_bytes);
                    match dt {
                        MLMultiArrayDataType::Float32 => *(byte_ptr as *const f32),
                        MLMultiArrayDataType::Float16 => {
                            let raw = *(byte_ptr as *const u16);
                            half::f16::from_bits(raw).to_f32()
                        }
                        _ => unreachable!(),
                    }
                };
                for d in (0..rank).rev() {
                    idx[d] += 1;
                    if idx[d] < shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        }
        out
    }

    fn predict_all(
        model: &MLModel,
        input: &MLMultiArray,
        output_names: &[String],
    ) -> Vec<(String, Vec<f32>)> {
        let provider = make_provider(INPUT_NAME_CORE_ML, input);
        let p_obj: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*provider);
        let outputs = unsafe {
            match model.predictionFromFeatures_error(p_obj) {
                Ok(o) => o,
                Err(e) => panic!("prediction failed: {:?}", e),
            }
        };
        let mut result = Vec::with_capacity(output_names.len());
        for name in output_names {
            let key = ns_str(name);
            let fv = unsafe { outputs.featureValueForName(&key) }
                .unwrap_or_else(|| panic!("missing output feature {name}"));
            let arr = unsafe { fv.multiArrayValue() }
                .unwrap_or_else(|| panic!("output {name} not a MultiArray"));
            result.push((name.clone(), extract_array(&arr)));
        }
        result
    }

    // ----------------------------------------------------------------------------
    // MLComputePlan introspection
    // ----------------------------------------------------------------------------

    fn introspect_compute_plan(
        mlmodelc_path: &Path,
        units: MLComputeUnits,
    ) -> Option<(usize, usize)> {
        println!(
            "\n[MLComputePlan] Loading plan for {} ...",
            mlmodelc_path.display()
        );
        let url = nsurl_for_dir(mlmodelc_path);
        let cfg = unsafe { MLModelConfiguration::new() };
        unsafe { cfg.setComputeUnits(units) };

        // SAFETY: Retained<MLComputePlan> is not Send/Sync due to ObjC retain semantics;
        // Arc is used here solely as a shared intra-thread handle between the
        // StackBlock callback and the waiting thread below.
        #[allow(clippy::arc_with_non_send_sync)]
        let slot: ComputePlanSlot = Arc::new((Mutex::new(None), Condvar::new()));
        let slot_clone = slot.clone();

        let block = StackBlock::new(move |plan: *mut MLComputePlan, err: *mut NSError| {
            let res = if !plan.is_null() {
                Ok(unsafe { Retained::retain(plan) }.expect("retain plan"))
            } else if !err.is_null() {
                Err(unsafe { Retained::retain(err) }.expect("retain err"))
            } else {
                return;
            };
            let (lock, cvar) = &*slot_clone;
            let mut guard = lock.lock().expect("slot mutex poisoned");
            *guard = Some(res);
            cvar.notify_all();
        });

        unsafe {
            MLComputePlan::loadContentsOfURL_configuration_completionHandler(&url, &cfg, &block);
        }

        let (lock, cvar) = &*slot;
        let guard = lock.lock().expect("slot mutex poisoned");
        let timeout = std::time::Duration::from_secs(30);
        let (mut guard, wait_res) = cvar
            .wait_timeout_while(guard, timeout, |g| g.is_none())
            .expect("cv wait");
        if wait_res.timed_out() {
            println!("[MLComputePlan]   timed out after 30s — skipping introspection");
            return None;
        }
        let plan = match guard.take() {
            Some(Ok(p)) => p,
            Some(Err(e)) => {
                println!("[MLComputePlan]   load error: {:?}", e);
                return None;
            }
            None => {
                println!("[MLComputePlan]   no result — skipping");
                return None;
            }
        };

        let structure = unsafe { plan.modelStructure() };
        let program = match unsafe { structure.program() } {
            Some(p) => p,
            None => {
                println!("[MLComputePlan]   not an MLProgram — skipping");
                return None;
            }
        };
        let funcs = unsafe { program.functions() };
        let main_key = ns_str("main");
        let main_func = match funcs.objectForKey(&main_key) {
            Some(f) => f,
            None => {
                println!("[MLComputePlan]   program has no 'main' function — skipping");
                return None;
            }
        };
        let blk = unsafe { main_func.block() };
        let ops = unsafe { blk.operations() };
        let n = ops.len();
        let mut by_dev: HashMap<String, usize> = HashMap::new();
        let mut by_dev_compute: HashMap<String, usize> = HashMap::new();
        let mut samples: Vec<(String, String)> = Vec::new();
        let mut compute_op_total: usize = 0;
        for i in 0..n {
            let op: Retained<MLModelStructureProgramOperation> = ops.objectAtIndex(i);
            let opname = unsafe { op.operatorName() }.to_string();
            let dev_name = match unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) } {
                Some(usage) => {
                    let dev = unsafe { usage.preferredComputeDevice() };
                    let dev_obj: &AnyObject = unsafe { &*(dev.as_ref() as *const _) };
                    let class: &objc2::runtime::AnyClass = dev_obj.class();
                    class.name().to_str().unwrap_or("?").to_string()
                }
                None => "Unknown".to_string(),
            };
            *by_dev.entry(dev_name.clone()).or_insert(0) += 1;
            // For the "real" engagement %, exclude const / load ops which have no
            // assigned compute device (they're handled by the runtime as data
            // placement, not as compute).
            if opname != "const" {
                *by_dev_compute.entry(dev_name.clone()).or_insert(0) += 1;
                compute_op_total += 1;
            }
            if samples.len() < 10 {
                samples.push((opname, dev_name));
            }
        }
        println!("[MLComputePlan]   total ops: {n}");
        let mut entries: Vec<_> = by_dev.iter().collect();
        entries.sort_by(|a, b| b.1.cmp(a.1));
        for (dev, count) in &entries {
            let pct = (**count as f64) * 100.0 / (n as f64);
            println!("[MLComputePlan]   {dev}: {count} ops ({pct:.1}%)");
        }
        println!("[MLComputePlan]   first ops (operatorName -> device):");
        for (op, dev) in &samples {
            println!("[MLComputePlan]     {op:<28} -> {dev}");
        }

        println!(
            "[MLComputePlan]   --- excluding 'const' ops ({} compute ops) ---",
            compute_op_total
        );
        let mut centries: Vec<_> = by_dev_compute.iter().collect();
        centries.sort_by(|a, b| b.1.cmp(a.1));
        for (dev, count) in &centries {
            let pct = (**count as f64) * 100.0 / (compute_op_total as f64);
            println!("[MLComputePlan]   compute {dev}: {count} ops ({pct:.1}%)");
        }

        let ane_compute = centries
            .iter()
            .filter(|(d, _)| d.contains("NeuralEngine"))
            .map(|(_, c)| **c)
            .sum::<usize>();
        let ane_pct = (ane_compute as f64) * 100.0 / (compute_op_total as f64);
        println!(
        "[MLComputePlan]   *** ANE-assigned compute ops: {ane_compute}/{compute_op_total} ({ane_pct:.1}%) ***"
    );
        Some((ane_compute, compute_op_total))
    }

    // ----------------------------------------------------------------------------
    // CoreML benchmark
    // ----------------------------------------------------------------------------

    fn run_coreml(
        mlmodelc_path: &Path,
        units: MLComputeUnits,
        input: &[f32],
    ) -> (Vec<(String, Vec<f32>)>, f64) {
        println!("\n[CoreML] Loading model with units = {:?}", units);
        let model = load_model(mlmodelc_path, units);
        let output_names = discover_output_names(&model);
        println!(
            "[CoreML] discovered {} output features: {:?}",
            output_names.len(),
            output_names
        );

        let arr = multi_array_from_f32_owned(input, &[1, 3, 640, 640]);
        let mut last = Vec::new();
        for _ in 0..N_WARMUP {
            last = predict_all(&model, &arr, &output_names);
        }
        let mut samples = Vec::with_capacity(N_ITERS);
        for _ in 0..N_ITERS {
            let t0 = Instant::now();
            last = predict_all(&model, &arr, &output_names);
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(&mut samples);
        println!("[CoreML] median: {med:.2} ms over {N_ITERS} iters (warmup={N_WARMUP})");
        let total_elems: usize = last.iter().map(|(_, v)| v.len()).sum();
        println!("[CoreML] total output elements: {}", total_elems);
        for (name, v) in &last {
            println!("[CoreML]   {name:<10} -> {} elems", v.len());
        }
        (last, med)
    }

    // ----------------------------------------------------------------------------
    // Validate per-stride output element counts (regardless of which CoreML name
    // got assigned to which head — match by element count).
    // ----------------------------------------------------------------------------

    fn validate_outputs(outs: &[(String, Vec<f32>)]) {
        let total_elems: usize = outs.iter().map(|(_, v)| v.len()).sum();
        assert_eq!(
            total_elems, EXPECTED_TOTAL_ELEMS,
            "total output elements mismatch (got {}, want {})",
            total_elems, EXPECTED_TOTAL_ELEMS
        );

        let mut got_counts: Vec<usize> = outs.iter().map(|(_, v)| v.len()).collect();
        got_counts.sort();
        let mut want_counts: Vec<usize> = EXPECTED_OUTPUT_COUNTS.iter().map(|(_, n)| *n).collect();
        want_counts.sort();
        assert_eq!(
            got_counts, want_counts,
            "per-output element counts do not match SCRFD-640 contract"
        );
        println!(
            "[validate] all 9 outputs match SCRFD-640 contract (total {EXPECTED_TOTAL_ELEMS} f32)"
        );
    }

    // ----------------------------------------------------------------------------
    // main
    // ----------------------------------------------------------------------------

    pub fn run() {
        let mut args = std::env::args().skip(1);
        let mlpkg = PathBuf::from(
            args.next()
                .unwrap_or_else(|| "/tmp/det_10g.mlpackage".to_string()),
        );

        println!("=== OxiFace CoreML SCRFD sub-gate ===");
        println!("Model: {}", mlpkg.display());

        let mlmodelc = compile_if_needed(&mlpkg);
        println!("Compiled mlmodelc: {}", mlmodelc.display());

        let plan = introspect_compute_plan(&mlmodelc, MLComputeUnits::All);

        let input = deterministic_input();
        assert_eq!(input.len(), N_INPUT);

        let (coreml_out, coreml_ms) = run_coreml(&mlmodelc, MLComputeUnits::All, &input);
        validate_outputs(&coreml_out);

        let (_cpu_out, cml_cpu_ms) = run_coreml(&mlmodelc, MLComputeUnits::CPUOnly, &input);

        println!("\n=========== RESULTS (SCRFD-10g, 1x3x640x640) ===========");
        println!("CoreML(All)     median: {coreml_ms:>8.2} ms");
        println!("CoreML(CPUOnly) median: {cml_cpu_ms:>8.2} ms");
        println!("Speedup CPUOnly/All: {:.2}x", cml_cpu_ms / coreml_ms);
        if let Some((ane, total)) = plan {
            let pct = (ane as f64) * 100.0 / (total as f64);
            println!("ANE engagement: {ane}/{total} ops ({pct:.1}%)");
        }
        println!("===============================");
    }
} // mod macos_impl
