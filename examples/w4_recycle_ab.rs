//! \[w4\] What activation-buffer **recycling** does to a frame: pool hit rate,
//! allocation churn, and whether the idle pooled total is a steady state or a
//! leak.
//!
//! # The decision this reports on
//!
//! Wave 1 destroyed a device-resident activation at its last consumer and left
//! `GpuContext::recycle_device_tensor` unused, with an open question: does
//! returning the allocation to the reusable-buffer pool beat handing it back to
//! the driver? Wave 4 measured it, as an **interleaved, paired** A/B inside one
//! process — one session, one graph, the two dispositions alternating per
//! iteration with the within-iteration order flipped every iteration and again
//! between processes, so a host under uncontrolled background load cancels in
//! the ratio. 25 measured pairs per process, 4 processes per case.
//!
//! ```text
//! (a) InSwapper-128, residency ON + f16 ON      paired median recycle/destroy
//!     proc 1  rot0   426.22 / 431.06 ms  min 413.87 / 424.69    0.9779  19/25
//!     proc 2  rot1   425.85 / 435.04 ms  min 419.56 / 426.96    0.9805  24/25
//!     proc 3  rot0   433.13 / 442.29 ms  min 419.82 / 431.49    0.9764  23/25
//!     proc 4  rot1   493.35 / 504.49 ms  min 479.70 / 494.15    0.9762  24/25
//!     pool    recycle  7 alloc + 87 reuse per frame (96.1% hit)
//!             destroy 86 alloc +  8 reuse per frame ( 4.9% hit)
//!     memory  recycle pooled 84.59 MiB / live 324.34 MiB
//!             destroy pooled  0.38 MiB / live 240.14 MiB   (weights 239.75)
//!
//! (b) Conv seed + 48 unary element-wise nodes at [1, 16, 32, 32]
//!     proc 1  rot0     4.36 /   6.02 ms  min   4.01 /   5.79    0.7242  25/25
//!     proc 2  rot1     5.05 /   6.02 ms  min   4.12 /   5.62    0.8054  25/25
//!     proc 3  rot0     5.23 /   5.89 ms  min   4.16 /   5.73    0.8889  23/25
//!     proc 4  rot1     4.26 /   4.98 ms  min   3.93 /   4.50    0.8514  25/25
//!     pool    recycle  1 alloc + 48 reuse per frame (98.9% hit)
//!             destroy 47 alloc +  2 reuse per frame ( 3.1% hit)
//! ```
//!
//! Recycling won both cases and the outputs were **byte-identical in all 200
//! measured pairs**, so the engine recycles unconditionally and the switch that
//! produced this table is gone — one code path, not a runtime option. What is
//! left here is the half that stays useful: the counters that say the mechanism
//! is working on *this* device, and the pooled-byte trend that says it is
//! bounded. Reproducing the A/B itself needs the two-line switch reinstated on
//! `GpuContext`; the browser wave may well want to, because a WebGPU
//! `createBuffer` is not a Metal one and (b) is where the difference lives.
//!
//! # The memory this costs, stated exactly
//!
//! InSwapper's pool settles holding **64 of its 64 permitted entries** at
//! 84.59 MiB, against a 256 MiB byte budget: it walks up to the *count* bound
//! and is then held there by LRU eviction. That is a ceiling, not a trend, and
//! it is the honest way to describe it — "the pooled total stopped growing" is
//! true but would be true of a leak that had merely reached its cap. The small
//! chain returns as many buffers per frame as it requests and needs 2 entries
//! and 0.12 MiB, so the ceiling is a property of graphs that hand the pool more
//! than they ask of it. Either way the bytes are reclaimable: `reclaim_for`
//! empties idle entries before any allocation is declined.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features gpu --example w4_recycle_ab -- real <inswapper_128.onnx>
//! cargo run --release --features gpu --example w4_recycle_ab -- synth
//! ```
//!
//! The model path may also come from `OXIONNX_INSWAPPER_MODEL`. It is a ~550 MB
//! download, never a repository fixture, so nothing here hardcodes a location.
//! **No absolute number this prints is comparable with one taken at another
//! time** — the host's background load moves them all. The hit rate, the
//! allocations per frame and the flatness of the trend are what carry meaning.

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
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Instant;

    use oxionnx::execution_providers::OpPlacement;
    use oxionnx::graph::{Attributes, Graph, Node, OpKind};
    use oxionnx::session::gpu_residency::take_run_stats;
    use oxionnx::tensor::Tensor;
    use oxionnx::Session;

    /// Measured iterations. Odd, so the median is a real element of the sample.
    const ITERS: usize = 25;

    /// Runs before the clock starts. The pool reaches its steady state during
    /// them; a cold pool would charge the first measured frame with allocations
    /// the mechanism exists to make once.
    const WARMUP: usize = 5;

    /// Unary element-wise nodes chained behind the synthetic graph's `Conv`.
    const CHAIN_LEN: usize = 48;

    /// Channels and spatial extent of the synthetic chain: 16 * 32 * 32 =
    /// 16 384 `f32` = 64 KiB per activation. Above `MIN_GPU_DISPATCH_BYTES`
    /// (4 KiB) and above `RESIDENT_DISPATCH_FLOOR` (256 elements), so every
    /// node in the chain is genuinely offered to the device; small enough that
    /// no node is compute-bound.
    const SYNTH_C: usize = 16;
    const SYNTH_HW: usize = 32;

    /// Input channels of the seeding `Conv`, which is **not** part of what is
    /// being measured and exists only to put a value on the device.
    ///
    /// It has to clear `oxionnx_gpu`'s own `GPU_THRESHOLD` (`m * k * n >= 10M`
    /// on the implicit-GEMM shape) or it declines to the CPU and the whole
    /// chain stays in the transferring tier, where every memory-bound op
    /// declines at every size — measured, the first version of this harness did
    /// exactly that and reported `gpu 0 / cpu 49`. At `[1, 128, 32, 32]` in and
    /// `[16, 128, 3, 3]` weights the shape is `1024 x 1152 x 16` = 18.9M, which
    /// clears it. Only the seed is wide; every activation this is about is
    /// 64 KiB.
    const SYNTH_C_IN: usize = 128;

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        values.get(values.len() / 2).copied().unwrap_or(f64::NAN)
    }

    fn minimum(values: &[f64]) -> f64 {
        values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// A fixed LCG, so every process sees byte-identical input without pulling
    /// an RNG crate into the dev-dependencies.
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

    fn image_blob(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut lcg = Lcg(seed);
        let data = (0..n).map(|_| lcg.next_unit()).collect();
        Tensor::new(data, shape.to_vec())
    }

    /// A unit-norm identity latent — the magnitude OxiFace actually passes. See
    /// `w3_inswapper_ab` for why an out-of-distribution latent would push the
    /// graph's output to its rails.
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

    /// Deterministic, signed and non-monotonic: a flat fill would hide a buffer
    /// bound at the wrong offset, and an all-positive one would make the
    /// rectifiers in the chain the identity.
    fn fill(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
                ((x % 37) as f32) * 0.041 - 0.75
            })
            .collect()
    }

    /// `Conv` followed by [`CHAIN_LEN`] unary element-wise nodes, every
    /// intermediate shaped `[1, SYNTH_C, SYNTH_HW, SYNTH_HW]`.
    ///
    /// The ops alternate `LeakyRelu`/`Tanh` rather than repeating one: no
    /// fusion pass in this engine claims either, and alternating makes an
    /// accidental fold visible as a drop in the node count printed below.
    fn synth_graph() -> (Graph, HashMap<String, Tensor>) {
        let mut nodes = Vec::with_capacity(CHAIN_LEN + 1);
        let mut conv_attrs = Attributes::default();
        conv_attrs
            .int_lists
            .insert("pads".to_string(), vec![1, 1, 1, 1]);
        conv_attrs
            .int_lists
            .insert("strides".to_string(), vec![1, 1]);
        conv_attrs
            .int_lists
            .insert("dilations".to_string(), vec![1, 1]);
        conv_attrs.ints.insert("group".to_string(), 1);
        nodes.push(Node {
            op: OpKind::Conv,
            name: "conv".to_string(),
            inputs: vec![
                "x".to_string(),
                "conv.weight".to_string(),
                "conv.bias".to_string(),
            ],
            outputs: vec!["h0".to_string()],
            attrs: conv_attrs,
        });
        for i in 0..CHAIN_LEN {
            let op = if i % 2 == 0 {
                OpKind::LeakyRelu
            } else {
                OpKind::Tanh
            };
            let mut attrs = Attributes::default();
            if matches!(op, OpKind::LeakyRelu) {
                attrs.floats.insert("alpha".to_string(), 0.2);
            }
            nodes.push(Node {
                op,
                name: format!("act{i}"),
                inputs: vec![format!("h{i}")],
                outputs: vec![format!("h{}", i + 1)],
                attrs,
            });
        }
        let graph = Graph {
            nodes,
            input_names: vec!["x".to_string()],
            output_names: vec![format!("h{CHAIN_LEN}")],
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "conv.weight".to_string(),
            Tensor::new(
                fill(SYNTH_C * SYNTH_C_IN * 3 * 3, 13),
                vec![SYNTH_C, SYNTH_C_IN, 3, 3],
            ),
        );
        weights.insert(
            "conv.bias".to_string(),
            Tensor::new(fill(SYNTH_C, 19), vec![SYNTH_C]),
        );
        (graph, weights)
    }

    /// What one run produced, plus everything about the pool that it moved.
    struct Sample {
        wall_ms: f64,
        /// Buffer requests this run had to forward to the driver.
        allocations: u64,
        /// Buffer requests this run served from an idle pooled entry.
        reuses: u64,
        /// Idle pooled bytes after the run.
        pooled: u64,
        /// Total live device bytes after the run.
        live: u64,
        resident_outputs: usize,
        gpu_nodes: usize,
        cpu_nodes: usize,
    }

    fn measure(
        session: &Session,
        inputs: &HashMap<&str, Tensor>,
    ) -> Option<(HashMap<String, Tensor>, Sample)> {
        let allocs_before = session.gpu_pool_allocations();
        let reuses_before = session.gpu_pool_reuses();
        let start = Instant::now();
        let outputs = pollster::block_on(session.run_gpu_async(inputs)).ok()?;
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        let stats = take_run_stats();
        Some((
            outputs,
            Sample {
                wall_ms,
                allocations: session.gpu_pool_allocations().saturating_sub(allocs_before),
                reuses: session.gpu_pool_reuses().saturating_sub(reuses_before),
                pooled: session.gpu_pooled_bytes(),
                live: session.gpu_live_bytes(),
                resident_outputs: stats.resident_outputs,
                gpu_nodes: stats.gpu_nodes,
                cpu_nodes: stats.cpu_nodes,
            },
        ))
    }

    /// The first output tensor by sorted name — a stable choice, so successive
    /// runs are compared over the same values.
    fn canonical_output(outputs: &HashMap<String, Tensor>) -> Option<(&str, &[f32])> {
        let mut names: Vec<&String> = outputs.keys().collect();
        names.sort();
        let name = names.first()?;
        outputs
            .get(*name)
            .map(|tensor| (name.as_str(), tensor.data.as_slice()))
    }

    /// Run [`ITERS`] measured frames and report the pool's behaviour across
    /// them, including whether successive frames agree to the bit.
    fn profile(session: &Session, inputs: &HashMap<&str, Tensor>) {
        for _ in 0..WARMUP {
            if pollster::block_on(session.run_gpu_async(inputs)).is_err() {
                println!("skip: a warm-up run failed");
                return;
            }
            let _ = take_run_stats();
        }

        let mut samples = Vec::with_capacity(ITERS);
        let mut reference: Option<Vec<f32>> = None;
        let mut mismatches = 0usize;
        for _ in 0..ITERS {
            let Some((out, sample)) = measure(session, inputs) else {
                println!("skip: a measured run failed");
                return;
            };
            samples.push(sample);
            let Some((name, data)) = canonical_output(&out) else {
                println!("skip: the graph produced no comparable output");
                return;
            };
            match &reference {
                None => reference = Some(data.to_vec()),
                Some(want) => {
                    if want.as_slice() != data {
                        mismatches += 1;
                        if mismatches == 1 {
                            println!(
                                "  !! output '{name}' changed between frames — a recycled buffer \
                                 is being read before it is written"
                            );
                        }
                    }
                }
            }
        }

        let Some(last) = samples.last() else {
            return;
        };
        let mut walls: Vec<f64> = samples.iter().map(|s| s.wall_ms).collect();
        let min = minimum(&walls);
        let med = median(&mut walls);
        let allocations: u64 = samples.iter().map(|s| s.allocations).sum();
        let reuses: u64 = samples.iter().map(|s| s.reuses).sum();
        let requests = allocations.saturating_add(reuses);
        let mut alloc_per_run: Vec<f64> = samples.iter().map(|s| s.allocations as f64).collect();
        let mut reuse_per_run: Vec<f64> = samples.iter().map(|s| s.reuses as f64).collect();

        println!();
        println!("  wall clock   med {med:8.2} ms   min {min:8.2} ms   (n={ITERS})");
        println!(
            "  nodes        gpu {} / cpu {}   resident_outputs {}",
            last.gpu_nodes, last.cpu_nodes, last.resident_outputs
        );
        println!(
            "  pool         alloc/run med {:5.0}   reuse/run med {:5.0}   hit rate {:.1}%",
            median(&mut alloc_per_run),
            median(&mut reuse_per_run),
            if requests > 0 {
                100.0 * reuses as f64 / requests as f64
            } else {
                0.0
            }
        );
        println!(
            "  bit identity {}/{} frames agreed with the first",
            samples.len() - 1 - mismatches,
            samples.len() - 1
        );

        // Boundedness is a trend claim: a single final reading cannot tell a
        // steady state from slow growth, so print every frame's pooled total.
        print!("  pooled MiB   ");
        for sample in &samples {
            print!("{:.2} ", mib(sample.pooled));
        }
        println!();
        println!(
            "  live         {:.2} MiB   pooled {:.2} MiB   resident weights {:.2} MiB",
            mib(last.live),
            mib(last.pooled),
            mib(session.gpu_resident_bytes())
        );
        let (held, cap) = session.gpu_pooled_buffers();
        let grew = samples
            .first()
            .is_some_and(|first| last.pooled > first.pooled);
        println!(
            "  bounds       {held}/{cap} buffers held, {:.2}/{:.2} MiB — pooled bytes {} \
             across the measured frames",
            mib(last.pooled),
            mib(session.gpu_pool_byte_budget()),
            if grew { "GREW" } else { "did not grow" }
        );
    }

    fn model_path(args: &[String]) -> Option<PathBuf> {
        let explicit = args
            .get(2)
            .cloned()
            .or_else(|| std::env::var("OXIONNX_INSWAPPER_MODEL").ok())?;
        let path = PathBuf::from(explicit);
        path.is_file().then_some(path)
    }

    /// Resolve the graph's inputs from the model's own type information. See
    /// `w3_inswapper_ab::build_inputs`, which this mirrors.
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
            owned.push((info.name.clone(), tensor));
        }
        (!owned.is_empty()).then_some(owned)
    }

    fn run_real(args: &[String]) {
        let Some(path) = model_path(args) else {
            println!(
                "usage: w4_recycle_ab real <inswapper_128.onnx>\n\
                 (or set OXIONNX_INSWAPPER_MODEL); the model is a ~550 MB download, not a fixture"
            );
            return;
        };
        println!("real InSwapper-128 -- {}", path.display());
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
        // The shipping configuration, and the only one the decision was about.
        if !session.set_activation_residency(true) {
            println!("skip: activation residency would not take");
            return;
        }
        if !session.set_f16_compute(true) {
            println!("skip: adapter does not support shader-f16");
            return;
        }
        println!("configuration: residency ON + f16 ON");
        profile(&session, &inputs);
        if let Some(err) = session.gpu_device_error() {
            println!("  !! device degraded during the measurement: {err}");
        }
    }

    fn run_synth() {
        let (graph, weights) = synth_graph();
        // The crate default is `CpuOnly`, under which wgpu is never offered a
        // node and the whole chain runs on the host — measuring nothing.
        let mut session = match Session::builder()
            .with_op_placement(OpPlacement::Auto {
                gpu_threshold_bytes: 4096,
            })
            .build_from_graph(graph, weights)
        {
            Ok(session) => session,
            Err(e) => {
                println!("could not build the synthetic graph: {e}");
                return;
            }
        };
        let x = image_blob(&[1, SYNTH_C_IN, SYNTH_HW, SYNTH_HW], 7);
        let inputs: HashMap<&str, Tensor> = HashMap::from([("x", x)]);
        if !pollster::block_on(session.enable_gpu_async()) {
            println!("skip: no GPU adapter available");
            return;
        }
        if !session.set_activation_residency(true) {
            println!("skip: activation residency would not take");
            return;
        }
        println!(
            "synthetic chain -- Conv + {CHAIN_LEN} unary element-wise nodes at \
             [1, {SYNTH_C}, {SYNTH_HW}, {SYNTH_HW}] ({} KiB per activation)",
            SYNTH_C * SYNTH_HW * SYNTH_HW * 4 / 1024
        );
        profile(&session, &inputs);
        if let Some(err) = session.gpu_device_error() {
            println!("  !! device degraded during the measurement: {err}");
        }
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let mode = args.get(1).map(String::as_str).unwrap_or("synth");
        println!("mode {mode}   ({ITERS} measured frames, {WARMUP} warm-up frames)");
        match mode {
            "real" => run_real(&args),
            "synth" => run_synth(),
            other => println!("unknown mode '{other}'; expected 'real' or 'synth'"),
        }
    }
}
