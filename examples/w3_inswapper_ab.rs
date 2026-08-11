//! [w3-ab] The real InSwapper-128 forward pass through
//! [`Session::run_gpu_async`], measured under the 2x2 of **activation
//! residency** and **half-precision compute**.
//!
//! Wave 1 measured residency on a synthetic four-op chain and Wave 2 measured
//! `f16` on a synthetic conv stack. Neither had ever run the model the claims
//! are *about*. This does: it loads `inswapper_128.onnx`, feeds it a
//! `[1, 3, 128, 128]` image blob in `[0, 1]` and a unit-norm `[1, 512]`
//! identity latent — the magnitudes OxiFace actually passes — and reports, for
//! each of the four combinations:
//!
//! * whole-run wall clock, median of the measured iterations. A run is
//!   `run_gpu_async`, which ends in a real read-back of every graph output into
//!   host memory, so nothing here can be credited to work the driver has not
//!   finished;
//! * where the graph ran (`gpu_nodes` vs `cpu_nodes`);
//! * every activation-residency counter, plus the resident weight footprint;
//! * **per-frame host↔device bytes**, as
//!   `uploaded_bytes` delta + `readback_bytes` + `activation_readback_bytes`.
//!   The upload delta is the context's own cumulative counter, of which the
//!   two per-run upload counters are subsets — adding those on top would
//!   double-count and flatter the residency ratio;
//! * output PSNR against the all-OFF arm, with the saturation split that says
//!   whether that comparison had any power to detect a difference.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features gpu --example w3_inswapper_ab -- <model.onnx> [rot0..rot3|solo0..solo3]
//! ```
//!
//! The model path comes from `argv[1]`, or from `OXIONNX_INSWAPPER_MODEL`. It
//! is a ~550 MB download, never a repository fixture, so nothing here hardcodes
//! a location.
//!
//! `rotN` rotates the within-iteration combo order by `N`; the order also
//! advances every iteration, so no combo is systematically the one that follows
//! another's cache state. Run all four rotations and report the spread across
//! processes — the best single pair is not the result.
//!
//! `soloN` runs **only** combo `N`, never touching the other arms' pipelines or
//! weight formats. That is the only configuration in which
//! `gpu_resident_bytes` is an honest footprint for that mode: a session that
//! has served both formats holds an `f32` *and* an `f16` copy of every
//! initializer, and reports their sum.

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("built without --features gpu; there is no device path to measure");
}

#[cfg(feature = "gpu")]
fn main() {
    gpu_impl::run();
}

#[cfg(feature = "gpu")]
mod gpu_impl {
    use oxionnx::execution_providers::OpPlacement;
    use oxionnx::session::gpu_residency::{take_run_stats, GpuRunStats};
    use oxionnx::tensor::Tensor;
    use oxionnx::Session;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Instant;

    /// `(activation residency, f16 compute)`, in the order the table prints.
    /// Index 0 is the reference arm: both mechanisms off, i.e. the code path as
    /// it behaved before either wave.
    const COMBOS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

    /// Measured iterations per combo. Odd, so the median is a real element of
    /// the sample rather than a value that was never observed.
    const ITERS: usize = 21;

    /// Runs per combo before the clock starts. Each residency mode and each
    /// weight format compiles its own pipelines and populates its own half of
    /// the weight cache; a cold first run would otherwise be charged to
    /// whichever arm happened to go first.
    const WARMUP: usize = 3;

    /// A PSNR taken over values pinned at a rail proves nothing, so anything
    /// this close to `±1` is counted as saturated and reported separately.
    ///
    /// The check earns its place by being *able* to fail: an earlier harness on
    /// this model warned that the graph ends in a saturating `Tanh`. Measured,
    /// it does not — `inswapper_128`'s output lands in `[0.0026, 0.9920]` with
    /// **0%** of its 49 152 elements saturated, so the PSNR column below is
    /// discriminating over the whole tensor. Left in place because that is a
    /// property of this model under these inputs, not a guarantee.
    const SATURATION: f32 = 0.999_9;

    fn label(combo: (bool, bool)) -> String {
        format!(
            "res {} / f16 {}",
            if combo.0 { "ON " } else { "OFF" },
            if combo.1 { "ON " } else { "OFF" }
        )
    }

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        values.get(values.len() / 2).copied().unwrap_or(f64::NAN)
    }

    fn model_path(args: &[String]) -> Option<PathBuf> {
        let explicit = args
            .get(1)
            .filter(|s| !s.starts_with("rot") && !s.starts_with("solo"))
            .cloned()
            .or_else(|| std::env::var("OXIONNX_INSWAPPER_MODEL").ok())?;
        let path = PathBuf::from(explicit);
        path.is_file().then_some(path)
    }

    /// A fixed LCG, so every process and every arm sees byte-identical input
    /// without pulling an RNG crate into the dev-dependencies.
    struct Lcg(u64);

    impl Lcg {
        fn next_unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as f32 / (1u64 << 31) as f32
        }
    }

    /// An image blob in `[0, 1]` — OxiFace divides the aligned RGB crop by 255
    /// before it reaches the model, so this is the distribution the graph is
    /// actually calibrated for.
    fn image_blob(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut lcg = Lcg(seed);
        let data = (0..n).map(|_| lcg.next_unit()).collect();
        Tensor::new(data, shape.to_vec())
    }

    /// A unit-norm identity latent. OxiFace computes it as
    /// `normalize(embedding @ emap)`, so its L2 norm is exactly 1 — about
    /// `0.044` per element at 512 dimensions.
    ///
    /// This matters for the measurement, not just for realism: an
    /// out-of-distribution latent (an unnormalised LCG fill has norm ~6.5, 6.5x
    /// too large) pushes the graph's output toward its rails, and a PSNR gate
    /// over a saturated output is vacuous. At the real magnitude the measured
    /// output saturates 0% — see [`SATURATION`].
    fn identity_latent(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut lcg = Lcg(seed);
        let mut data: Vec<f32> = (0..n).map(|_| lcg.next_unit() - 0.5).collect();
        let norm = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut data {
                *v /= norm;
            }
        }
        Tensor::new(data, shape.to_vec())
    }

    /// Peak-signal-to-noise ratio at a stated `peak`.
    ///
    /// `None` means the two buffers are bit-identical — printed as such rather
    /// than as an infinity, because `inf dB` in a table reads like a bug.
    fn psnr(reference: &[f32], candidate: &[f32], peak: f64) -> Option<f64> {
        if reference.len() != candidate.len() || reference.is_empty() {
            return None;
        }
        let mse = reference
            .iter()
            .zip(candidate)
            .map(|(a, b)| {
                let d = f64::from(*a) - f64::from(*b);
                d * d
            })
            .sum::<f64>()
            / reference.len() as f64;
        (mse > 0.0).then(|| 10.0 * (peak * peak / mse).log10())
    }

    /// One measured run: the numbers that vary run to run, kept per iteration
    /// so the spread is visible instead of averaged away.
    struct Sample {
        wall_ms: f64,
        uploaded: u64,
        stats: GpuRunStats,
        /// `None` when this run was bit-identical to the reference.
        psnr_peak2: Option<f64>,
    }

    impl Sample {
        /// Bytes that crossed the bus in this run, both directions.
        fn moved_bytes(&self) -> u64 {
            self.uploaded
                .saturating_add(self.stats.readback_bytes)
                .saturating_add(self.stats.activation_readback_bytes)
        }
    }

    /// Resolve the graph's inputs from the model's own type information and
    /// build a tensor of the right magnitude for each.
    ///
    /// Assignment is by name first (`source` is InSwapper's identity latent),
    /// then by rank — a rank-2 input is the latent, a rank-4 one is the image
    /// blob. Nothing is positional: the order `input_info` reports is the
    /// model's, not a contract.
    fn build_inputs(session: &Session) -> Option<Vec<(String, Tensor)>> {
        let mut owned = Vec::new();
        for (index, info) in session.input_info().iter().enumerate() {
            let dims: Option<Vec<usize>> = info
                .symbolic_shape()
                .into_iter()
                .map(|d| match d {
                    oxionnx::graph::Dim::Static(n) => Some(n),
                    _ => None,
                })
                .collect();
            let dims = dims?;
            let seed = 0x5eed + index as u64;
            let is_latent = info.name.contains("source") || dims.len() == 2;
            let tensor = if is_latent {
                identity_latent(&dims, seed)
            } else {
                image_blob(&dims, seed)
            };
            println!(
                "  input {:<10} {:?}  {}",
                info.name,
                dims,
                if is_latent {
                    "unit-norm identity latent"
                } else {
                    "image blob in [0, 1]"
                }
            );
            owned.push((info.name.clone(), tensor));
        }
        (!owned.is_empty()).then_some(owned)
    }

    /// Apply a combo's toggles, reporting whether both actually took.
    fn arm(session: &Session, combo: (bool, bool)) -> bool {
        session.set_activation_residency(combo.0) == combo.0
            && session.set_f16_compute(combo.1) == combo.1
    }

    /// Run once and collect everything that describes the run.
    fn measure(
        session: &Session,
        inputs: &HashMap<&str, Tensor>,
        reference: Option<&HashMap<String, Tensor>>,
    ) -> Option<(HashMap<String, Tensor>, Sample)> {
        let before = session.gpu_uploaded_bytes();
        let start = Instant::now();
        let outputs = pollster::block_on(session.run_gpu_async(inputs)).ok()?;
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        let stats = take_run_stats();
        let uploaded = session.gpu_uploaded_bytes().saturating_sub(before);

        // Worst PSNR across the graph's outputs, so a model with more than one
        // cannot hide a bad one behind a good one.
        let mut psnr_peak2 = None;
        if let Some(reference) = reference {
            for (name, want) in reference {
                let Some(got) = outputs.get(name) else {
                    continue;
                };
                if let Some(db) = psnr(&want.data, &got.data, 2.0) {
                    psnr_peak2 = Some(psnr_peak2.map_or(db, |worst: f64| worst.min(db)));
                }
            }
        }

        Some((
            outputs,
            Sample {
                wall_ms,
                uploaded,
                stats,
                psnr_peak2,
            },
        ))
    }

    fn report(combo: (bool, bool), samples: &[Sample], resident_bytes: u64, has_reference: bool) {
        let Some(last) = samples.last() else {
            return;
        };
        let mut walls: Vec<f64> = samples.iter().map(|s| s.wall_ms).collect();
        let min = walls.iter().copied().fold(f64::MAX, f64::min);
        let max = walls.iter().copied().fold(f64::MIN, f64::max);
        let med = median(&mut walls);
        let mut moved: Vec<f64> = samples.iter().map(|s| mib(s.moved_bytes())).collect();
        let moved_med = median(&mut moved);
        let worst_weight_upload = samples
            .iter()
            .map(|s| s.stats.weight_upload_bytes)
            .max()
            .unwrap_or(0);
        let s = &last.stats;

        println!("\n--- {} ---", label(combo));
        println!(
            "  wall clock          med {med:.1} ms   min {min:.1}   max {max:.1}  (n={})",
            samples.len()
        );
        println!(
            "  nodes               gpu {} / cpu {}",
            s.gpu_nodes, s.cpu_nodes
        );
        println!(
            "  host<->device       {:.1} MiB/frame  (up {:.1} + down {:.1} + act-down {:.1})",
            moved_med,
            mib(last.uploaded),
            mib(s.readback_bytes),
            mib(s.activation_readback_bytes)
        );
        println!(
            "  readbacks           {} ({:.1} MiB)",
            s.readbacks,
            mib(s.readback_bytes)
        );
        println!(
            "  resident_outputs    {}   resident_operands {}",
            s.resident_outputs, s.resident_operands
        );
        println!(
            "  activation_bytes_saved   {:.1} MiB (gross)",
            mib(s.activation_bytes_saved)
        );
        println!(
            "  activation_readbacks     {} ({:.1} MiB)",
            s.activation_readbacks,
            mib(s.activation_readback_bytes)
        );
        println!(
            "  activation_uploads       {} ({:.1} MiB)",
            s.activation_uploads,
            mib(s.activation_upload_bytes)
        );
        println!(
            "  activation_peak_bytes    {:.1} MiB",
            mib(s.activation_peak_bytes)
        );
        println!(
            "  resident weight bytes    {:.1} MiB   (session-wide; sums every format served)",
            mib(resident_bytes)
        );
        println!(
            "  weight_upload_bytes      {worst_weight_upload} (worst measured run; non-zero = the weight cache is thrashing and this row's timing is not steady state)"
        );
        let worst_psnr = samples
            .iter()
            .filter_map(|s| s.psnr_peak2)
            .fold(None, |worst: Option<f64>, db| {
                Some(worst.map_or(db, |w| w.min(db)))
            });
        match (has_reference, worst_psnr) {
            (false, _) => println!(
                "  PSNR vs all-OFF          not measured in this process (no reference arm)"
            ),
            (true, Some(db)) => println!(
                "  PSNR vs all-OFF          {db:.2} dB (peak 2.0) / {:.2} dB (peak 1.0), worst of {} runs",
                db - 6.020_6,
                samples.len()
            ),
            (true, None) => {
                println!("  PSNR vs all-OFF          bit-identical on every measured run")
            }
        }
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let Some(path) = model_path(&args) else {
            println!(
                "usage: w3_inswapper_ab <inswapper_128.onnx> [rot0..rot3|solo0..solo3]\n\
                 (or set OXIONNX_INSWAPPER_MODEL); the model is a ~550 MB download, not a fixture"
            );
            return;
        };
        let mode = args
            .iter()
            .skip(1)
            .find(|s| s.starts_with("rot") || s.starts_with("solo"))
            .cloned()
            .unwrap_or_else(|| "rot0".to_string());
        let index = mode
            .trim_start_matches("solo")
            .trim_start_matches("rot")
            .parse::<usize>()
            .unwrap_or(0)
            % COMBOS.len();
        let solo = mode.starts_with("solo");
        let arms: Vec<usize> = if solo {
            vec![index]
        } else {
            (0..COMBOS.len()).collect()
        };

        println!(
            "model {}\nmode {mode} ({})",
            path.display(),
            if solo {
                format!("solo: only {} ever runs", label(COMBOS[index]))
            } else {
                "all four combos, order rotated per iteration".to_string()
            }
        );

        let mut session = match Session::builder()
            .with_op_placement(OpPlacement::Auto {
                gpu_threshold_bytes: 4096,
            })
            .load(&path)
        {
            Ok(session) => session,
            Err(e) => {
                println!("could not load the model: {e}");
                return;
            }
        };
        let Some(owned) = build_inputs(&session) else {
            println!("skip: the model does not declare fully static input shapes");
            return;
        };
        let inputs: HashMap<&str, Tensor> = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor.clone()))
            .collect();

        if !pollster::block_on(session.enable_gpu_async()) {
            println!("skip: no GPU adapter available");
            return;
        }
        if !session.f16_compute_supported() && arms.iter().any(|&i| COMBOS[i].1) {
            println!("skip: adapter does not support shader-f16, so half the table is unreachable");
            return;
        }

        // Warm every arm this process will measure — and only those, which is
        // what makes `solo` a clean footprint reading.
        for &i in &arms {
            if !arm(&session, COMBOS[i]) {
                println!("skip: the toggles would not take the state {:?}", COMBOS[i]);
                return;
            }
            for _ in 0..WARMUP {
                if pollster::block_on(session.run_gpu_async(&inputs)).is_err() {
                    println!("skip: a warm-up run failed for {}", label(COMBOS[i]));
                    return;
                }
                let _ = take_run_stats();
            }
        }

        // The reference: combo 0, both mechanisms off. Taken after warm-up so
        // it is a steady-state result, and taken once so every arm is compared
        // against the same bytes.
        //
        // A `solo` process that is not measuring combo 0 deliberately does
        // **not** take it. Running the reference would dispatch the whole graph
        // in `f32` and leave an `f32` copy of every initializer in the weight
        // cache, which is exactly the contamination `solo` exists to avoid —
        // the first version of this harness did that and reported the *sum* of
        // both formats as the "f16-only" footprint. PSNR for those arms comes
        // from the `rot` processes, which do hold a reference.
        let reference = if arms.contains(&0) {
            if !arm(&session, COMBOS[0]) {
                println!("skip: could not arm the reference combo");
                return;
            }
            let Some((reference, _)) = measure(&session, &inputs, None) else {
                println!("skip: the reference run failed");
                return;
            };
            let mut total = 0usize;
            let mut saturated = 0usize;
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for tensor in reference.values() {
                for &v in &tensor.data {
                    total += 1;
                    if v.abs() > SATURATION {
                        saturated += 1;
                    }
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let pct = if total > 0 {
                100.0 * saturated as f64 / total as f64
            } else {
                0.0
            };
            println!(
                "\nreference output: n={total} range [{lo:.4}, {hi:.4}]  saturated(|v|>{SATURATION}) {saturated} ({pct:.1}%)"
            );
            Some(reference)
        } else {
            println!(
                "\nno reference arm in this process: PSNR is not measured here, so that the \
                 resident-weight figure below is one format's footprint and not two"
            );
            None
        };

        // Measured loop. Every iteration visits each armed combo once, in an
        // order that advances with the iteration and with the process's `rot`,
        // so no arm is systematically second.
        let mut samples: HashMap<usize, Vec<Sample>> =
            arms.iter().map(|&i| (i, Vec::new())).collect();
        let mut resident: HashMap<usize, u64> = HashMap::new();
        for iteration in 0..ITERS {
            for step in 0..arms.len() {
                let i = arms[(step + iteration + index) % arms.len()];
                if !arm(&session, COMBOS[i]) {
                    println!("skip: the toggles stopped taking mid-measurement");
                    return;
                }
                let Some((_, sample)) = measure(&session, &inputs, reference.as_ref()) else {
                    println!("skip: a measured run failed for {}", label(COMBOS[i]));
                    return;
                };
                resident.insert(i, session.gpu_resident_bytes());
                if let Some(bucket) = samples.get_mut(&i) {
                    bucket.push(sample);
                }
            }
        }

        if let Some(error) = session.gpu_device_error() {
            println!("\nDEVICE DEGRADED during the comparison: {error}");
            println!("every number above is a CPU fallback, not a measurement of the device");
            return;
        }

        for &i in &arms {
            let (Some(bucket), Some(&bytes)) = (samples.get(&i), resident.get(&i)) else {
                continue;
            };
            report(COMBOS[i], bucket, bytes, reference.is_some());
        }

        // Ratios against the reference arm, when this process measured it.
        if let (Some(base), false) = (samples.get(&0), solo) {
            let mut base_walls: Vec<f64> = base.iter().map(|s| s.wall_ms).collect();
            let mut base_moved: Vec<f64> = base.iter().map(|s| s.moved_bytes() as f64).collect();
            let base_wall = median(&mut base_walls);
            let base_bytes = median(&mut base_moved);
            println!("\n--- vs the all-OFF arm ---");
            println!(
                "{:<20} {:>12} {:>12} {:>14} {:>12}",
                "combo", "med ms", "speedup", "MiB/frame", "traffic cut"
            );
            for &i in &arms {
                let Some(bucket) = samples.get(&i) else {
                    continue;
                };
                let mut walls: Vec<f64> = bucket.iter().map(|s| s.wall_ms).collect();
                let mut moved: Vec<f64> = bucket.iter().map(|s| s.moved_bytes() as f64).collect();
                let wall = median(&mut walls);
                let bytes = median(&mut moved);
                println!(
                    "{:<20} {wall:>12.1} {:>11.3}x {:>14.1} {:>11.2}x",
                    label(COMBOS[i]),
                    base_wall / wall,
                    bytes / (1024.0 * 1024.0),
                    base_bytes / bytes.max(1.0)
                );
            }
        }

        // Is the PSNR column above worth anything? Perturb the identity latent
        // and require the same comparison to light up. Without this, a table of
        // "bit-identical" rows could equally mean the outputs never depended on
        // the input at all.
        //
        // Skipped without a reference — a `solo` process has no PSNR column to
        // qualify, and running combo 0 here purely to build one would leave the
        // other format resident after all.
        let Some(reference) = reference else {
            return;
        };
        let mut perturbed: HashMap<&str, Tensor> = HashMap::new();
        for (name, tensor) in &owned {
            let mut t = tensor.clone();
            if t.shape.len() == 2 || name.contains("source") {
                if let Some(first) = t.data.first_mut() {
                    *first += 100.0;
                }
            }
            perturbed.insert(name.as_str(), t);
        }
        if !arm(&session, COMBOS[0]) {
            println!("\nsensitivity check skipped: could not re-arm the reference combo");
            return;
        }
        match pollster::block_on(session.run_gpu_async(&perturbed)) {
            Ok(shifted) => {
                let mut moved = 0.0f32;
                for (name, want) in &reference {
                    if let Some(got) = shifted.get(name) {
                        moved = moved.max(
                            want.data
                                .iter()
                                .zip(&got.data)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f32, f32::max),
                        );
                    }
                }
                let verdict = if moved > 1e-3 {
                    "the PSNR column is discriminating"
                } else {
                    "VACUOUS: the output did not move, so no PSNR above proves anything"
                };
                println!("\nsensitivity (source[0] += 100): max_abs_diff {moved:.3e} — {verdict}");
            }
            Err(e) => println!("\nsensitivity check failed to run: {e}"),
        }
        let _ = take_run_stats();
    }
}
