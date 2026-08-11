//! [r3a] InSwapper forward through `Session::run_gpu_async`, measured.
//!
//! Reports the wave's headline table: total wall clock, read-back count and
//! bytes, nodes on GPU vs CPU, and the top CPU-fallback op types by time —
//! against a `CpuOnly`-pinned reference run of the same session, which also
//! serves as the numerical check.
//!
//! The model is a 277 MB download, not a repository fixture: this exits with
//! a message when it is absent. Point `OXIONNX_INSWAPPER_MODEL` at a copy to
//! run it from elsewhere.
//!
//! ```text
//! cargo run --release --features gpu --example r3a_inswapper_residency
//! ```

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("built without --features gpu; nothing to measure");
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

    fn model_path() -> Option<PathBuf> {
        if let Ok(explicit) = std::env::var("OXIONNX_INSWAPPER_MODEL") {
            let path = PathBuf::from(explicit);
            return path.is_file().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home)
            .join(".oxiface")
            .join("models")
            .join("web")
            .join("inswapper_128_fp16.onnx");
        path.is_file().then_some(path)
    }

    /// Deterministic pseudo-random tensor — a fixed LCG, so both runs see
    /// byte-identical input without pulling in an RNG crate.
    fn lcg_tensor(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed;
        let data = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::new(data, shape.to_vec())
    }

    fn report(label: &str, stats: &GpuRunStats, wall: std::time::Duration) {
        let total_nodes = stats.gpu_nodes + stats.cpu_nodes;
        println!("\n--- {label} ---");
        println!(
            "  wall clock          {:>10.1} ms",
            wall.as_secs_f64() * 1e3
        );
        println!(
            "  nodes               {:>10}  (gpu {} / cpu {})",
            total_nodes, stats.gpu_nodes, stats.cpu_nodes
        );
        println!(
            "  readbacks           {:>10}  ({:.1} MiB)",
            stats.readbacks,
            stats.readback_bytes as f64 / (1024.0 * 1024.0)
        );
        let gpu_total: std::time::Duration = stats.gpu_time_by_op.values().sum();
        let cpu_total: std::time::Duration = stats.cpu_time_by_op.values().sum();
        println!(
            "  attributed time     gpu {:.1} ms / cpu {:.1} ms",
            gpu_total.as_secs_f64() * 1e3,
            cpu_total.as_secs_f64() * 1e3
        );
        println!("  top CPU fallbacks by time:");
        for (op, count, dur) in stats.top_cpu_fallbacks(5) {
            println!(
                "     {op:<20} {count:>4} nodes  {:>9.2} ms",
                dur.as_secs_f64() * 1e3
            );
        }
        let mut gpu_rows: Vec<(&String, &std::time::Duration)> =
            stats.gpu_time_by_op.iter().collect();
        gpu_rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        println!("  top GPU ops by time:");
        for (op, dur) in gpu_rows.iter().take(5) {
            println!("     {op:<20}       {:>9.2} ms", dur.as_secs_f64() * 1e3);
        }
    }

    pub fn run() {
        let Some(path) = model_path() else {
            println!("inswapper_128_fp16.onnx not installed; nothing to measure");
            return;
        };

        // Graph inputs, read from the model rather than hardcoded.
        //
        // NOTE: on native, `Session::load` attaches a wgpu device at build
        // time (see `session::loading`), so "did not call enable_gpu_async"
        // is NOT a CPU reference — the first measurement of this harness was
        // wrong for exactly that reason. The only honest reference is a
        // session built with `OpPlacement::CpuOnly`, which `decide_placement`
        // short-circuits for every op and size. The two sessions are built
        // and dropped in sequence so peak memory stays at one copy of the
        // ~550 MB weight map.
        let specs = {
            let probe = match Session::builder().load(&path) {
                Ok(s) => s,
                Err(e) => {
                    println!("could not load the model: {e}");
                    return;
                }
            };
            let specs: Vec<(String, Vec<usize>)> = probe
                .input_info()
                .iter()
                .filter_map(|info| {
                    let dims: Option<Vec<usize>> = info
                        .symbolic_shape()
                        .into_iter()
                        .map(|d| match d {
                            oxionnx::graph::Dim::Static(n) => Some(n),
                            _ => None,
                        })
                        .collect();
                    dims.map(|d| (info.name.clone(), d))
                })
                .collect();
            specs
        };
        println!("model inputs: {specs:?}");
        let owned: Vec<(String, Tensor)> = specs
            .iter()
            .enumerate()
            .map(|(i, (name, shape))| (name.clone(), lcg_tensor(shape, 0x5eed + i as u64)))
            .collect();
        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        for (name, tensor) in &owned {
            inputs.insert(name.as_str(), tensor.clone());
        }

        // ── Reference: every node pinned to the CPU ────────────────────
        let (cpu_out, cpu_wall, cpu_stats) = {
            let cpu_session = match Session::builder()
                .with_op_placement(OpPlacement::CpuOnly)
                .load(&path)
            {
                Ok(s) => s,
                Err(e) => {
                    println!("could not load the CPU reference model: {e}");
                    return;
                }
            };
            // Warm once so page faults on the freshly-parsed weight map are
            // not charged to the measured run, matching the GPU arm.
            let _ = pollster::block_on(cpu_session.run_gpu_async(&inputs));
            let _ = take_run_stats();
            let t0 = Instant::now();
            let out = match pollster::block_on(cpu_session.run_gpu_async(&inputs)) {
                Ok(o) => o,
                Err(e) => {
                    println!("CPU reference run failed: {e}");
                    return;
                }
            };
            let wall = t0.elapsed();
            let stats = take_run_stats();
            report("CPU only (OpPlacement::CpuOnly)", &stats, wall);
            assert_eq!(
                stats.gpu_nodes, 0,
                "CpuOnly placement must not dispatch a single node to the GPU"
            );
            (out, wall, stats)
        };

        let mut session = match Session::builder()
            .with_op_placement(OpPlacement::Auto {
                gpu_threshold_bytes: 4096,
            })
            .load(&path)
        {
            Ok(s) => s,
            Err(e) => {
                println!("could not load the model: {e}");
                return;
            }
        };

        // ── GPU: Auto placement, device attached ────────────────────────────
        let has_gpu = pollster::block_on(session.enable_gpu_async());
        if !has_gpu {
            println!("\nno GPU adapter on this machine; stopping after the CPU reference");
            return;
        }
        // One warm-up run: first touch compiles pipelines and, once the
        // weight cache lands, populates it. Steady state is what matters.
        let _ = pollster::block_on(session.run_gpu_async(&inputs));
        let _ = take_run_stats();

        let t1 = Instant::now();
        let gpu_out = match pollster::block_on(session.run_gpu_async(&inputs)) {
            Ok(o) => o,
            Err(e) => {
                println!("GPU run failed: {e}");
                return;
            }
        };
        let gpu_wall = t1.elapsed();
        let gpu_stats = take_run_stats();
        report("GPU (Auto, warm)", &gpu_stats, gpu_wall);

        // ── Numerical agreement ─────────────────────────────────────────────
        println!("\n--- agreement ---");
        let mut worst = 0.0f32;
        let mut worst_name = String::new();
        for (name, cpu_tensor) in &cpu_out {
            let Some(gpu_tensor) = gpu_out.get(name) else {
                println!("  output {name} missing from the GPU run");
                continue;
            };
            assert_eq!(
                cpu_tensor.shape, gpu_tensor.shape,
                "output {name}: shape differs"
            );
            let d = cpu_tensor
                .data
                .iter()
                .zip(gpu_tensor.data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if d > worst {
                worst = d;
                worst_name = name.clone();
            }
            // The graph ends in `Tanh`, which saturates: a comparison that
            // only ever sees saturated values would report a diff of zero no
            // matter what the convolutions did. Report how much of the output
            // is actually discriminating, so "bit-identical" can be read
            // correctly.
            let saturated = cpu_tensor.data.iter().filter(|v| v.abs() > 0.999_9).count();
            let distinct = cpu_tensor
                .data
                .iter()
                .filter(|v| v.abs() <= 0.999_9)
                .count();
            println!(
                "  {name:<24} n={:<9} max_abs_diff={d:.3e}  saturated={saturated} unsaturated={distinct}",
                cpu_tensor.data.len()
            );
        }
        println!("  worst: {worst:.3e} on {worst_name}");

        // ── Is that agreement meaningful, or is the comparison vacuous? ─────
        //
        // A max-diff of exactly zero across a 20-conv model is a strong claim,
        // and the way it is usually wrong is that the comparison never had the
        // power to detect anything. So prove it does: rerun the GPU session on
        // a deliberately perturbed input and require the *same* comparison to
        // light up.
        //
        // `source` (the identity embedding, feeding all 12 AdaIN heads) is
        // used rather than `target`, because this model is measurably
        // insensitive to an individual `target` pixel — perturbing
        // `target[0]` by +100 changes no output element at all, while the same
        // change to `source[0]` moves the output by ~1.0. That is a property
        // of the model and this synthetic input, not of the harness, and it is
        // exactly the sort of thing that makes a naive sensitivity check
        // silently useless.
        let mut perturbed: HashMap<&str, Tensor> = HashMap::new();
        for (name, tensor) in &owned {
            let mut t = tensor.clone();
            if name == "source" {
                t.data[0] += 100.0;
            }
            perturbed.insert(name.as_str(), t);
        }
        if let Ok(shifted) = pollster::block_on(session.run_gpu_async(&perturbed)) {
            let mut moved = 0.0f32;
            for (name, cpu_tensor) in &cpu_out {
                if let Some(s) = shifted.get(name) {
                    moved = moved.max(
                        cpu_tensor
                            .data
                            .iter()
                            .zip(s.data.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max),
                    );
                }
            }
            println!("  sensitivity check (source[0] += 100): max_abs_diff={moved:.3e}");
            assert!(
                moved > 1e-3,
                "the CPU-vs-GPU comparison above is vacuous: perturbing the \
                 input changed nothing, so a diff of {worst:.3e} proves nothing",
            );
        }
        let _ = take_run_stats();

        println!(
            "\nspeedup vs CPU-only: {:.2}x  ({:.1} ms -> {:.1} ms)",
            cpu_wall.as_secs_f64() / gpu_wall.as_secs_f64(),
            cpu_wall.as_secs_f64() * 1e3,
            gpu_wall.as_secs_f64() * 1e3,
        );

        // ── Per-op crossover: where is the GPU actually winning? ────────────
        //
        // This is the table the threshold work is calibrated from. `where`
        // reads GPU when the node ran on the GPU in the Auto run, CPU when it
        // declined or was never offered.
        println!("\n--- per-op: CPU-only vs Auto, same 154 nodes ---");
        println!(
            "{:<20} {:>5} {:>12} {:>12} {:>10}  ran on",
            "op", "n", "cpu ms", "auto ms", "gpu/cpu"
        );
        let mut ops: Vec<String> = cpu_stats.cpu_time_by_op.keys().cloned().collect();
        ops.sort();
        let mut gpu_slower = Vec::new();
        for op in ops {
            let cpu_ms = cpu_stats
                .cpu_time_by_op
                .get(&op)
                .map_or(0.0, |d| d.as_secs_f64() * 1e3);
            let n = cpu_stats.cpu_count_by_op.get(&op).copied().unwrap_or(0);
            let (auto_ms, ran_on) = match gpu_stats.gpu_time_by_op.get(&op) {
                Some(d) => (d.as_secs_f64() * 1e3, "GPU"),
                None => (
                    gpu_stats
                        .cpu_time_by_op
                        .get(&op)
                        .map_or(0.0, |d| d.as_secs_f64() * 1e3),
                    "cpu",
                ),
            };
            let ratio = if cpu_ms > 0.0 {
                auto_ms / cpu_ms
            } else {
                f64::NAN
            };
            println!("{op:<20} {n:>5} {cpu_ms:>12.2} {auto_ms:>12.2} {ratio:>10.2}  {ran_on}");
            if ran_on == "GPU" && ratio > 1.0 {
                gpu_slower.push((op.clone(), n, cpu_ms, auto_ms, ratio));
            }
        }
        if gpu_slower.is_empty() {
            println!("\nevery GPU-dispatched op type beat its CPU kernel");
        } else {
            println!("\nGPU-dispatched op types that are SLOWER than the CPU kernel:");
            for (op, n, cpu_ms, auto_ms, ratio) in &gpu_slower {
                println!(
                    "   {op:<18} {n:>4} nodes  cpu {cpu_ms:>8.2} ms  gpu {auto_ms:>8.2} ms  ({ratio:.2}x slower)"
                );
            }
        }
    }
}
