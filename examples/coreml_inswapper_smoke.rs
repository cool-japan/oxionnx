//! CoreML smoke benchmark for InSwapper-128 (the actual face-swap model) —
//! final sub-gate for OxiFace's CoreML/ANE acceleration plan.
//!
//! Two-input model:
//!   - `target` : 1x3x128x128 f32  (target face image, [0,1] range)
//!   - `source` : 1x512    f32     (L2-normalized ArcFace embedding)
//!
//! Output (renamed by coremltools):
//!   - `var_1144` : 1x3x128x128 f32 (swapped face)
//!
//! Reads pre-saved reference inputs and the ORT-CPU reference output from
//! /tmp/inswapper_{target,source,ort_output}.npy so we can do a max-abs-diff
//! correctness check (image output, not embedding — cosine sim is wrong here).
//!
//! Build / run (macOS only):
//!     cargo run --release --example coreml_inswapper_smoke -- \
//!         /tmp/inswapper_128.mlpackage
#![allow(clippy::too_many_lines, deprecated)]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("coreml_inswapper_smoke only runs on macOS (CoreML).");
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
    const N_ITERS: usize = 30; // fewer than ArcFace because each iter is ~1 sec on CPU
    const INPUT_NAME_TARGET: &str = "target";
    const INPUT_NAME_SOURCE: &str = "source";
    const N_TARGET: usize = 3 * 128 * 128;
    const N_SOURCE: usize = 512;
    const N_OUTPUT: usize = 3 * 128 * 128;
    // Tolerance for fp16 lowering (mlprogram lowers weights to fp16 on macOS14).
    const MAX_ABS_DIFF_THRESHOLD: f32 = 0.05;

    // ----------------------------------------------------------------------------
    // NPY loader — minimal: header parser + raw f32 little-endian payload.
    // Supports the exact layout numpy.save writes for an f32 contiguous array.
    // ----------------------------------------------------------------------------

    fn load_npy_f32(path: &Path) -> Vec<f32> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => panic!("failed to read npy {}: {:?}", path.display(), e),
        };
        if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
            panic!("not a NPY file: {}", path.display());
        }
        let major = bytes[6];
        let minor = bytes[7];
        let header_len: usize = match (major, minor) {
            (1, 0) => u16::from_le_bytes([bytes[8], bytes[9]]) as usize + 10,
            (2, _) | (3, _) => {
                let l = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
                l + 12
            }
            _ => panic!("unsupported NPY version {}.{}", major, minor),
        };
        let header_str = std::str::from_utf8(
            &bytes[bytes.iter().position(|&b| b == b'{').unwrap_or(0)..header_len],
        )
        .expect("npy header utf8");
        if !header_str.contains("'<f4'") && !header_str.contains("'|f4'") {
            panic!(
                "expected float32 npy ('<f4'); got header: {}",
                header_str.trim()
            );
        }
        if header_str.contains("'fortran_order': True") {
            panic!("fortran_order=True NPYs not supported");
        }
        let payload = &bytes[header_len..];
        if payload.len() % 4 != 0 {
            panic!("npy payload not aligned to f32 stride");
        }
        let n = payload.len() / 4;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 4;
            let v = f32::from_le_bytes([
                payload[off],
                payload[off + 1],
                payload[off + 2],
                payload[off + 3],
            ]);
            out.push(v);
        }
        out
    }

    // ----------------------------------------------------------------------------
    // Utility
    // ----------------------------------------------------------------------------

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
    // CoreML: compile mlpackage -> mlmodelc, load with chosen compute units
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
    // MLMultiArray helpers
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
    // Multi-input feature provider
    // ----------------------------------------------------------------------------

    fn make_provider_pairs(
        pairs: &[(&str, &MLMultiArray)],
    ) -> Retained<MLDictionaryFeatureProvider> {
        let keys_owned: Vec<Retained<NSString>> = pairs.iter().map(|(n, _)| ns_str(n)).collect();
        let vals_owned: Vec<Retained<MLFeatureValue>> = pairs
            .iter()
            .map(|(_, a)| unsafe { MLFeatureValue::featureValueWithMultiArray(a) })
            .collect();
        let key_refs: Vec<&NSString> = keys_owned.iter().map(|k| &**k).collect();
        let val_refs: Vec<&MLFeatureValue> = vals_owned.iter().map(|v| &**v).collect();

        let dict: Retained<NSDictionary<NSString, MLFeatureValue>> =
            NSDictionary::from_slices(&key_refs, &val_refs);
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
    // Discover output feature names
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
        out.sort();
        out
    }

    // ----------------------------------------------------------------------------
    // Run prediction; pull single output as Vec<f32>
    // ----------------------------------------------------------------------------

    fn extract_array(arr: &MLMultiArray) -> Vec<f32> {
        let n = unsafe { arr.count() } as usize;
        let mut out = vec![0.0_f32; n];
        let p = unsafe { arr.dataPointer() }.as_ptr();
        let dt = unsafe { arr.dataType() };
        if dt == MLMultiArrayDataType::Float32 {
            let pf = p as *const f32;
            unsafe { std::ptr::copy_nonoverlapping(pf, out.as_mut_ptr(), n) };
        } else if dt == MLMultiArrayDataType::Float16 {
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

    fn predict_once(
        model: &MLModel,
        target: &MLMultiArray,
        source: &MLMultiArray,
        output_name: &str,
    ) -> Vec<f32> {
        let provider =
            make_provider_pairs(&[(INPUT_NAME_TARGET, target), (INPUT_NAME_SOURCE, source)]);
        let p_obj: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*provider);
        let outputs = unsafe {
            match model.predictionFromFeatures_error(p_obj) {
                Ok(o) => o,
                Err(e) => panic!("prediction failed: {:?}", e),
            }
        };
        let key = ns_str(output_name);
        let fv = unsafe { outputs.featureValueForName(&key) }
            .unwrap_or_else(|| panic!("missing output feature {output_name}"));
        let arr = unsafe { fv.multiArrayValue() }
            .unwrap_or_else(|| panic!("output {output_name} not a MultiArray"));
        extract_array(&arr)
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
        let timeout = std::time::Duration::from_secs(60);
        let (mut guard, wait_res) = cvar
            .wait_timeout_while(guard, timeout, |g| g.is_none())
            .expect("cv wait");
        if wait_res.timed_out() {
            println!("[MLComputePlan]   timed out after 60s — skipping introspection");
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
        let mut op_dev_pairs: Vec<(String, String)> = Vec::with_capacity(n);
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
            if opname != "const" {
                *by_dev_compute.entry(dev_name.clone()).or_insert(0) += 1;
                compute_op_total += 1;
            }
            op_dev_pairs.push((opname, dev_name));
        }
        println!("[MLComputePlan]   total ops: {n}");
        let mut entries: Vec<_> = by_dev.iter().collect();
        entries.sort_by(|a, b| b.1.cmp(a.1));
        for (dev, count) in &entries {
            let pct = (**count as f64) * 100.0 / (n as f64);
            println!("[MLComputePlan]   {dev}: {count} ops ({pct:.1}%)");
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

        // Per-operatorName breakdown for the non-ANE devices — diagnoses which
        // op kinds got kicked off ANE.
        let mut non_ane_by_op: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for (op, dev) in &op_dev_pairs {
            if op == "const" {
                continue;
            }
            if dev.contains("NeuralEngine") {
                continue;
            }
            non_ane_by_op
                .entry(op.clone())
                .or_default()
                .entry(dev.clone())
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
        if !non_ane_by_op.is_empty() {
            println!("[MLComputePlan]   --- non-ANE compute ops by operatorName ---");
            let mut sorted_ops: Vec<_> = non_ane_by_op.iter().collect();
            sorted_ops.sort_by(|a, b| {
                b.1.values()
                    .sum::<usize>()
                    .cmp(&a.1.values().sum::<usize>())
            });
            for (op, devs) in sorted_ops {
                let total: usize = devs.values().sum();
                let detail: Vec<String> = devs.iter().map(|(d, c)| format!("{d}={c}")).collect();
                println!(
                    "[MLComputePlan]     {op:<28} total={total:<3} ({})",
                    detail.join(", ")
                );
            }
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
        target: &[f32],
        source: &[f32],
    ) -> (Vec<f32>, f64) {
        println!("\n[CoreML] Loading model with units = {:?}", units);
        let model = load_model(mlmodelc_path, units);
        let output_names = discover_output_names(&model);
        println!(
            "[CoreML] discovered {} output features: {:?}",
            output_names.len(),
            output_names
        );
        if output_names.len() != 1 {
            panic!("expected exactly 1 output, got {}", output_names.len());
        }
        let output_name = output_names[0].clone();

        let target_arr = multi_array_from_f32_owned(target, &[1, 3, 128, 128]);
        let source_arr = multi_array_from_f32_owned(source, &[1, 512]);

        let mut last = Vec::new();
        for _ in 0..N_WARMUP {
            last = predict_once(&model, &target_arr, &source_arr, &output_name);
        }
        let mut samples = Vec::with_capacity(N_ITERS);
        for _ in 0..N_ITERS {
            let t0 = Instant::now();
            last = predict_once(&model, &target_arr, &source_arr, &output_name);
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
    // Numerical correctness: max abs diff vs ORT reference
    // ----------------------------------------------------------------------------

    fn numerical_check(coreml_out: &[f32], ort_ref: &[f32], label: &str) {
        if coreml_out.len() != ort_ref.len() {
            panic!(
                "[{}] output length mismatch: coreml={}, ort={}",
                label,
                coreml_out.len(),
                ort_ref.len()
            );
        }
        let mut max_abs = 0.0_f32;
        let mut max_idx = 0_usize;
        let mut sum_sq = 0.0_f64;
        for (i, (a, b)) in coreml_out.iter().zip(ort_ref.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
            sum_sq += (d as f64) * (d as f64);
        }
        let rmse = (sum_sq / (coreml_out.len() as f64)).sqrt();
        println!(
        "[{label}] max_abs_diff = {max_abs:.6} at idx {max_idx} (coreml={:.4}, ort={:.4}); rmse = {rmse:.6e}",
        coreml_out[max_idx], ort_ref[max_idx]
    );
        if max_abs > MAX_ABS_DIFF_THRESHOLD {
            println!(
            "[{label}] WARNING: max_abs_diff {max_abs} exceeds threshold {MAX_ABS_DIFF_THRESHOLD}"
        );
        }
    }

    // ----------------------------------------------------------------------------
    // main
    // ----------------------------------------------------------------------------

    pub fn run() {
        let mut args = std::env::args().skip(1);
        let mlpkg = PathBuf::from(
            args.next()
                .unwrap_or_else(|| "/tmp/inswapper_128.mlpackage".to_string()),
        );

        println!("=== OxiFace CoreML InSwapper-128 sub-gate ===");
        println!("Model: {}", mlpkg.display());

        let mlmodelc = compile_if_needed(&mlpkg);
        println!("Compiled mlmodelc: {}", mlmodelc.display());

        let plan = introspect_compute_plan(&mlmodelc, MLComputeUnits::All);

        let target = load_npy_f32(Path::new("/tmp/inswapper_target.npy"));
        let source = load_npy_f32(Path::new("/tmp/inswapper_source.npy"));
        let ort_ref = load_npy_f32(Path::new("/tmp/inswapper_ort_output.npy"));
        assert_eq!(target.len(), N_TARGET, "target NPY size unexpected");
        assert_eq!(source.len(), N_SOURCE, "source NPY size unexpected");
        assert_eq!(ort_ref.len(), N_OUTPUT, "ort_ref NPY size unexpected");

        let (cml_all_out, cml_all_ms) =
            run_coreml(&mlmodelc, MLComputeUnits::All, &target, &source);
        numerical_check(&cml_all_out, &ort_ref, "All_vs_ORT");

        let (cml_cpu_out, cml_cpu_ms) =
            run_coreml(&mlmodelc, MLComputeUnits::CPUOnly, &target, &source);
        numerical_check(&cml_cpu_out, &ort_ref, "CPU_vs_ORT");

        println!("\n=========== RESULTS (InSwapper-128) ===========");
        println!("CoreML(All)     median: {cml_all_ms:>8.2} ms");
        println!("CoreML(CPUOnly) median: {cml_cpu_ms:>8.2} ms");
        println!(
            "Speedup CPUOnly/All:           {:.2}x",
            cml_cpu_ms / cml_all_ms
        );
        if let Some((ane, total)) = plan {
            let pct = (ane as f64) * 100.0 / (total as f64);
            println!("ANE engagement: {ane}/{total} ops ({pct:.1}%)");
        }
        println!("===============================================");
    }
} // mod macos_impl
