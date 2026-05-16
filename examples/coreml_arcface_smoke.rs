//! CoreML vs CPU smoke benchmark for ArcFace (w600k_r50) — kill-switch gate
//! for OxiFace's CoreML/ANE acceleration plan.
//!
//! Loads a `.mlpackage` (compiles to `.mlmodelc` first), runs the model with
//! `MLComputeUnits::All` so CoreML can pick ANE/GPU/CPU, then runs the same
//! input through the CPU oxionnx Session and compares:
//!  * median latency (CPU vs CoreML)
//!  * cosine similarity of the 512-dim embedding
//!  * MLComputePlan: which compute device CoreML actually picked per op
//!
//! Build / run (macOS only, default features only — does not need `gpu`):
//!     cargo run --release --example coreml_arcface_smoke -- \
//!         /tmp/w600k_r50.mlpackage /tmp/w600k_r50.onnx
//!
//! The ONNX path is optional. When omitted, only the CoreML side runs.
#![allow(clippy::too_many_lines, deprecated)]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("coreml_arcface_smoke only runs on macOS (CoreML).");
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

    use oxionnx::{Session, Tensor};

    const N_WARMUP: usize = 5;
    const N_ITERS: usize = 50;
    const INPUT_NAME_CORE_ML: &str = "input_1"; // coremltools rewrote 'input.1'
    const OUTPUT_NAME_CORE_ML: &str = "var_1110"; // coremltools rewrote '1110'
    const INPUT_NAME_ONNX: &str = "input.1";
    const EMBED_DIM: usize = 512;
    const N_INPUT: usize = 3 * 112 * 112;

    // ----------------------------------------------------------------------------
    // Utility
    // ----------------------------------------------------------------------------

    fn deterministic_input() -> Vec<f32> {
        (0..N_INPUT).map(|i| (i as f32) / 1000.0).collect()
    }

    fn ns_str(s: &str) -> Retained<NSString> {
        NSString::from_str(s)
    }

    fn nsurl_for_dir(path: &Path) -> Retained<NSURL> {
        let s = ns_str(&path.to_string_lossy());
        // mlpackage AND mlmodelc are both directories.
        NSURL::fileURLWithPath_isDirectory(&s, true)
    }

    fn median(samples: &mut [f64]) -> f64 {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        samples[samples.len() / 2]
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        dot / (na.sqrt() * nb.sqrt())
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
    // Build an MLMultiArray from a borrowed f32 slice (no copy, with deallocator
    // no-op because we keep the buffer alive across the call).
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
        // initWithDictionary_error wants NSDictionary<NSString, AnyObject>; the storage
        // is structurally identical (NSDictionary holds Objective-C `id` regardless of
        // the Rust generic). Reinterpret the typed pointer for the FFI call.
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
    // Run prediction once, return f32 output as Vec<f32>
    // ----------------------------------------------------------------------------

    fn predict_once(model: &MLModel, input: &MLMultiArray) -> Vec<f32> {
        let provider = make_provider(INPUT_NAME_CORE_ML, input);
        let p_obj: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*provider);
        let outputs = unsafe {
            match model.predictionFromFeatures_error(p_obj) {
                Ok(o) => o,
                Err(e) => panic!("prediction failed: {:?}", e),
            }
        };
        let key = ns_str(OUTPUT_NAME_CORE_ML);
        let fv = unsafe { outputs.featureValueForName(&key) }.expect("missing output feature");
        let arr = unsafe { fv.multiArrayValue() }.expect("output not a MultiArray");
        let n = unsafe { arr.count() } as usize;
        let mut out = vec![0.0_f32; n];
        let p = unsafe { arr.dataPointer() }.as_ptr() as *const f32;
        let dt = unsafe { arr.dataType() };
        if dt == MLMultiArrayDataType::Float32 {
            unsafe { std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), n) };
        } else if dt == MLMultiArrayDataType::Float16 {
            // convert via half crate -> f32
            let p16 = p as *const u16;
            for (i, slot) in out.iter_mut().enumerate() {
                let raw = unsafe { *p16.add(i) };
                *slot = half::f16::from_bits(raw).to_f32();
            }
        } else {
            panic!("unexpected output dtype {:?}", dt);
        }
        out
    }

    // ----------------------------------------------------------------------------
    // MLComputePlan introspection (sync-bridge over the async block API).
    // Reports: per-operation operatorName + preferred device class name.
    // ----------------------------------------------------------------------------

    fn introspect_compute_plan(mlmodelc_path: &Path, units: MLComputeUnits) {
        println!(
            "\n[MLComputePlan] Loading plan for {} ...",
            mlmodelc_path.display()
        );
        let url = nsurl_for_dir(mlmodelc_path);
        let cfg = unsafe { MLModelConfiguration::new() };
        unsafe { cfg.setComputeUnits(units) };

        // Sync bridge: condvar + slot for plan/error.
        // SAFETY: Retained<MLComputePlan> is not Send/Sync due to ObjC retain semantics;
        // Arc is used here solely as a shared intra-thread handle between the
        // StackBlock callback and the waiting thread below.
        #[allow(clippy::arc_with_non_send_sync)]
        let slot: ComputePlanSlot = Arc::new((Mutex::new(None), Condvar::new()));
        let slot_clone = slot.clone();

        let block = StackBlock::new(move |plan: *mut MLComputePlan, err: *mut NSError| {
            // Block-arg pointers from Apple are *not* owned by us — they are
            // autoreleased. We must retain them to extend their lifetime past
            // the block.
            let res = if !plan.is_null() {
                Ok(unsafe { Retained::retain(plan) }.expect("retain plan"))
            } else if !err.is_null() {
                Err(unsafe { Retained::retain(err) }.expect("retain err"))
            } else {
                return; // signal nothing — main thread will time out
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
        let mut guard = lock.lock().expect("slot mutex poisoned");
        let timeout = std::time::Duration::from_secs(30);
        let (g2, wait_res) = cvar
            .wait_timeout_while(guard, timeout, |g| g.is_none())
            .expect("cv wait");
        guard = g2;
        if wait_res.timed_out() {
            println!("[MLComputePlan]   timed out after 30s — skipping introspection");
            return;
        }
        let plan = match guard.take() {
            Some(Ok(p)) => p,
            Some(Err(e)) => {
                println!("[MLComputePlan]   load error: {:?}", e);
                return;
            }
            None => {
                println!("[MLComputePlan]   no result — skipping");
                return;
            }
        };

        // Walk the program: program -> functions["main"] -> block -> operations[]
        let structure = unsafe { plan.modelStructure() };
        let program = match unsafe { structure.program() } {
            Some(p) => p,
            None => {
                println!("[MLComputePlan]   not an MLProgram (no program tree) — skipping");
                return;
            }
        };
        let funcs = unsafe { program.functions() };
        let main_key = ns_str("main");
        let main_func = match funcs.objectForKey(&main_key) {
            Some(f) => f,
            None => {
                println!("[MLComputePlan]   program has no 'main' function — skipping");
                return;
            }
        };
        let blk = unsafe { main_func.block() };
        let ops = unsafe { blk.operations() };
        let n = ops.len();
        let mut by_dev: HashMap<String, usize> = HashMap::new();
        let mut samples: Vec<(String, String)> = Vec::new();
        for i in 0..n {
            let op: Retained<MLModelStructureProgramOperation> = ops.objectAtIndex(i);
            let opname = unsafe { op.operatorName() }.to_string();
            let dev_name = match unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) } {
                Some(usage) => {
                    let dev = unsafe { usage.preferredComputeDevice() };
                    // `dev` is `Retained<ProtocolObject<dyn MLComputeDeviceProtocol>>`.
                    // `ProtocolObject` is `#[repr(C)]` with `inner: AnyObject` first,
                    // so a pointer cast is sound.
                    let dev_obj: &AnyObject = unsafe { &*(dev.as_ref() as *const _) };
                    let class: &objc2::runtime::AnyClass = dev_obj.class();
                    class.name().to_str().unwrap_or("?").to_string()
                }
                None => "Unknown".to_string(),
            };
            *by_dev.entry(dev_name.clone()).or_insert(0) += 1;
            if samples.len() < 6 {
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
        println!("[MLComputePlan]   first ops:");
        for (op, dev) in &samples {
            println!("[MLComputePlan]     {op:<28} -> {dev}");
        }

        let ane_count = entries
            .iter()
            .filter(|(d, _)| d.contains("NeuralEngine"))
            .map(|(_, c)| **c)
            .sum::<usize>();
        let ane_pct = (ane_count as f64) * 100.0 / (n as f64);
        println!("[MLComputePlan]   *** ANE-assigned ops: {ane_count}/{n} ({ane_pct:.1}%) ***");
    }

    // ----------------------------------------------------------------------------
    // CPU baseline via oxionnx
    // ----------------------------------------------------------------------------

    fn run_cpu_baseline(onnx_path: &Path, input: &[f32]) -> (Vec<f32>, f64) {
        println!("\n[CPU] Loading oxionnx Session: {}", onnx_path.display());
        let sess = match Session::from_file(onnx_path) {
            Ok(s) => s,
            Err(e) => panic!("oxionnx Session::from_file failed: {:?}", e),
        };

        let mk_inputs = |data: Vec<f32>| -> HashMap<&'static str, Tensor> {
            let t = Tensor::new(data, vec![1, 3, 112, 112]);
            let mut h: HashMap<&'static str, Tensor> = HashMap::new();
            h.insert(INPUT_NAME_ONNX, t);
            h
        };

        // Warm-up
        let mut last_out = Vec::new();
        for _ in 0..N_WARMUP {
            let inputs = mk_inputs(input.to_vec());
            let out = match sess.run(&inputs) {
                Ok(o) => o,
                Err(e) => panic!("oxionnx run failed: {:?}", e),
            };
            last_out = out.into_iter().next().expect("no output").1.data.clone();
        }

        let mut samples = Vec::with_capacity(N_ITERS);
        for _ in 0..N_ITERS {
            let inputs = mk_inputs(input.to_vec());
            let t0 = Instant::now();
            let out = match sess.run(&inputs) {
                Ok(o) => o,
                Err(e) => panic!("oxionnx run failed: {:?}", e),
            };
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_out = out.into_iter().next().expect("no output").1.data.clone();
        }
        let med = median(&mut samples);
        println!("[CPU] median: {med:.2} ms over {N_ITERS} iters (warmup={N_WARMUP})");
        (last_out, med)
    }

    // ----------------------------------------------------------------------------
    // CoreML benchmark
    // ----------------------------------------------------------------------------

    fn run_coreml(mlmodelc_path: &Path, units: MLComputeUnits, input: &[f32]) -> (Vec<f32>, f64) {
        println!("\n[CoreML] Loading model with units = {:?}", units);
        let model = load_model(mlmodelc_path, units);
        let arr = multi_array_from_f32_owned(input, &[1, 3, 112, 112]);
        let mut last = Vec::new();
        for _ in 0..N_WARMUP {
            last = predict_once(&model, &arr);
        }
        let mut samples = Vec::with_capacity(N_ITERS);
        for _ in 0..N_ITERS {
            let t0 = Instant::now();
            last = predict_once(&model, &arr);
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(&mut samples);
        println!(
        "[CoreML] median: {med:.2} ms over {N_ITERS} iters (warmup={N_WARMUP}); output dim = {}",
        last.len()
    );
        (last, med)
    }

    // ----------------------------------------------------------------------------
    // main
    // ----------------------------------------------------------------------------

    pub fn run() {
        let mut args = std::env::args().skip(1);
        let mlpkg = PathBuf::from(
            args.next()
                .unwrap_or_else(|| "/tmp/w600k_r50.mlpackage".to_string()),
        );
        let onnx = args.next().map(PathBuf::from);

        println!("=== OxiFace CoreML kill-switch gate ===");
        println!("Model: {}", mlpkg.display());

        let mlmodelc = compile_if_needed(&mlpkg);
        println!("Compiled mlmodelc: {}", mlmodelc.display());

        introspect_compute_plan(&mlmodelc, MLComputeUnits::All);

        let input = deterministic_input();
        assert_eq!(input.len(), N_INPUT);

        let (coreml_out, coreml_ms) = run_coreml(&mlmodelc, MLComputeUnits::All, &input);
        assert_eq!(coreml_out.len(), EMBED_DIM, "expected 512-dim embedding");

        // Baseline 1: CoreML with CPUOnly (apples-to-apples, same numerics path).
        let (cml_cpu_out, cml_cpu_ms) = run_coreml(&mlmodelc, MLComputeUnits::CPUOnly, &input);
        let cs_cml_cpu_vs_all = cosine(&cml_cpu_out, &coreml_out);

        // Baseline 2 (optional): oxionnx CPU Session — may not run if oxionnx
        // doesn't yet support every op in this graph (broadcast etc.).
        let (oxn_cpu_out, oxn_cpu_ms) = match onnx.as_ref() {
            Some(p) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_cpu_baseline(p, &input)
            })) {
                Ok(v) => v,
                Err(_) => {
                    println!(
                        "[CPU] oxionnx baseline panicked — graph contains ops oxionnx \
                     does not yet support. Falling back to CoreML CPUOnly only."
                    );
                    (Vec::new(), 0.0)
                }
            },
            None => {
                println!("[CPU] (oxionnx baseline skipped: no ONNX path passed)");
                (Vec::new(), 0.0)
            }
        };

        println!("\n=========== RESULTS ===========");
        println!("CoreML(All)     median: {coreml_ms:>8.2} ms");
        println!("CoreML(CPUOnly) median: {cml_cpu_ms:>8.2} ms");
        println!("Speedup CPUOnly/All: {:.2}x", cml_cpu_ms / coreml_ms);
        println!("Cosine sim CPUOnly vs All (CoreML): {cs_cml_cpu_vs_all:.6}");
        if !oxn_cpu_out.is_empty() {
            println!("oxionnx CPU      median: {oxn_cpu_ms:>8.2} ms");
            println!(
                "Speedup oxionnx_CPU/CoreML(All): {:.2}x",
                oxn_cpu_ms / coreml_ms
            );
            let cs = cosine(
                &oxn_cpu_out[..EMBED_DIM.min(oxn_cpu_out.len())],
                &coreml_out,
            );
            println!("Cosine sim oxionnx_CPU vs CoreML(All): {cs:.6}");
        }
        println!("===============================");
    }
} // mod macos_impl
