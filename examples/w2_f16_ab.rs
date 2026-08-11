//! [w2-f16] Native A/B of half-precision compute on a conv-heavy synthetic
//! graph approximating InSwapper's shapes.
//!
//! Measures **whole-run wall clock**, where a run is
//! `Session::run_gpu_async` — which ends in a real read-back of the graph's
//! output into host memory, so nothing here can be credited to work the driver
//! has not actually finished.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features gpu --example w2_f16_ab -- [ab|ba] [channels] [layers]
//! ```
//!
//! The first argument sets which mode runs first *within* each interleaved
//! pair. Run the example several times in both orders — the driver comment at
//! the bottom of this file has the exact loop — so that a warm-up asymmetry
//! (first-touch page faults, pipeline compilation, thermal state) cannot be
//! mistaken for a speedup. Each process prints its own min/median for both
//! modes; the honest number is the spread across processes, not the best pair.

use std::collections::HashMap;
use std::time::Instant;

use oxionnx::execution_providers::OpPlacement;
use oxionnx::graph::{Attributes, Graph, Node, OpKind};
use oxionnx::tensor::Tensor;
use oxionnx::Session;

/// Spatial extent of every convolution's activation — InSwapper's bottleneck
/// resolution.
const HW: usize = 32;

/// The Gemm chain's shape, taken from InSwapper's AdaIN heads
/// (`[64, 512] x [2048, 512]^T`). 134 MFLOP each, so both clear the session's
/// 10 MFLOP Gemm dispatch gate.
const GEMM_M: usize = 64;
const GEMM_K: usize = 512;
const GEMM_N: usize = 2048;

fn fill(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            (((x % 37) as f32) * 0.041 - 0.75) * scale
        })
        .collect()
}

fn conv_node(name: &str, input: &str, output: &str) -> Node {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("pads".to_string(), vec![1, 1, 1, 1]);
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);
    Node {
        op: OpKind::Conv,
        name: name.to_string(),
        inputs: vec![
            input.to_string(),
            format!("{name}.weight"),
            format!("{name}.bias"),
        ],
        outputs: vec![output.to_string()],
        attrs,
    }
}

fn gemm_node(name: &str, input: &str, output: &str) -> Node {
    let mut attrs = Attributes::default();
    attrs.ints.insert("transB".to_string(), 1);
    attrs.floats.insert("alpha".to_string(), 1.0);
    attrs.floats.insert("beta".to_string(), 1.0);
    Node {
        op: OpKind::Gemm,
        name: name.to_string(),
        inputs: vec![input.to_string(), format!("{name}.b"), format!("{name}.c")],
        outputs: vec![output.to_string()],
        attrs,
    }
}

/// `layers` convolutions with a `Relu` between each pair, then two `Gemm`
/// nodes fed by a reshape of the last activation — the AdaIN-head shape.
///
/// Returns the graph, its initializers, and the FLOP count of one forward pass.
fn conv_graph(channels: usize, layers: usize) -> (Graph, HashMap<String, Tensor>, f64) {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();
    let mut flops = 0.0f64;

    for layer in 0..layers {
        let input = if layer == 0 {
            "x".to_string()
        } else {
            format!("a{layer}")
        };
        let name = format!("conv{}", layer + 1);
        let conv_out = format!("h{}", layer + 1);
        nodes.push(conv_node(&name, &input, &conv_out));
        nodes.push(Node {
            op: OpKind::Relu,
            name: format!("relu{}", layer + 1),
            inputs: vec![conv_out],
            outputs: vec![format!("a{}", layer + 1)],
            attrs: Attributes::default(),
        });
        let seed = 13 + layer as u32 * 17;
        weights.insert(
            format!("{name}.weight"),
            Tensor::new(
                fill(channels * channels * 9, seed, 0.05),
                vec![channels, channels, 3, 3],
            ),
        );
        weights.insert(
            format!("{name}.bias"),
            Tensor::new(fill(channels, seed + 3, 0.1), vec![channels]),
        );
        // 2 * M * N * K, with M = C_out, N = OH*OW, K = C_in*9.
        flops += 2.0 * (channels * HW * HW * channels * 9) as f64;
    }

    // Two Gemm nodes on a separate `[GEMM_M, GEMM_K]` input — InSwapper's AdaIN
    // style heads, whose `B` is an initializer read as `B^T`.
    //
    // A separate input rather than a flattening of the last activation: a
    // `[1, channels*HW*HW]` flattening makes `B` a `[512, 131072]` matrix
    // (268 MB), which exceeds `max_storage_buffer_binding_size` and declines to
    // the CPU — so the Gemm nodes would be in the graph and never on the GPU.
    // These shapes are InSwapper's actual ones and clear the 10 MFLOP gate.
    for (index, (input, output, n, k)) in
        [("z", "g1", GEMM_N, GEMM_K), ("g1", "g2", GEMM_K, GEMM_N)]
            .iter()
            .enumerate()
    {
        let name = format!("gemm{}", index + 1);
        nodes.push(gemm_node(&name, input, output));
        weights.insert(
            format!("{name}.b"),
            Tensor::new(fill(n * k, 71 + index as u32, 0.01), vec![*n, *k]),
        );
        weights.insert(
            format!("{name}.c"),
            Tensor::new(fill(*n, 91 + index as u32, 0.1), vec![*n]),
        );
        flops += 2.0 * (GEMM_M * n * k) as f64;
    }

    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string(), "z".to_string()],
        // Both chains are read back, so neither can be optimized away and the
        // measured wall clock covers a real host-visible result for each.
        output_names: vec![format!("a{layers}"), "g2".to_string()],
        ..Default::default()
    };
    (graph, weights, flops)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let f16_first = args.get(1).map_or(true, |s| s != "ba");
    let channels: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let layers: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);

    let (graph, weights, flops) = conv_graph(channels, layers);
    let Ok(mut session) = Session::builder()
        .with_op_placement(OpPlacement::Auto {
            gpu_threshold_bytes: 65_536,
        })
        .build_from_graph(graph, weights)
    else {
        println!("skip: could not build the session");
        return;
    };
    if !pollster::block_on(session.enable_gpu_async()) {
        println!("skip: no GPU adapter available");
        return;
    }
    if !session.f16_compute_supported() {
        println!("skip: adapter does not support shader-f16");
        return;
    }

    let mut inputs = HashMap::new();
    inputs.insert(
        "x",
        Tensor::new(fill(channels * HW * HW, 5, 1.0), vec![1, channels, HW, HW]),
    );
    inputs.insert(
        "z",
        Tensor::new(fill(GEMM_M * GEMM_K, 7, 1.0), vec![GEMM_M, GEMM_K]),
    );

    // Warm both modes: the weight cache is session-lifetime and each mode
    // compiles its own pipeline, so a cold first run in either mode would be
    // charged to that mode.
    //
    // Note what this leaves behind, because it bounds the result: from here on
    // the context holds **both** an f32 and an f16 copy of every weight. So the
    // f16 arm gets none of the halved-resident-footprint benefit a
    // single-mode session would see, and both arms run under the combined
    // memory pressure. Whatever speedup this prints is therefore a conservative
    // lower bound on what a browser page that only ever runs one mode would get.
    for enabled in [false, true] {
        if session.set_f16_compute(enabled) != enabled {
            println!("skip: the f16 toggle would not take the state {enabled}");
            return;
        }
        for _ in 0..3 {
            if pollster::block_on(session.run_gpu_async(&inputs)).is_err() {
                println!("skip: a warm-up run failed");
                return;
            }
        }
    }

    // `f64::NAN` marks a run that failed; `main` bails out below rather than
    // averaging a sentinel into the result.
    let run_once = |enabled: bool| -> f64 {
        if session.set_f16_compute(enabled) != enabled {
            return f64::NAN;
        }
        let start = Instant::now();
        let Ok(outputs) = pollster::block_on(session.run_gpu_async(&inputs)) else {
            return f64::NAN;
        };
        // Touch the read-back values so nothing above can be elided, and so the
        // measurement provably includes the host-visible result.
        let checksum: f32 = outputs
            .values()
            .flat_map(|t| t.data.iter())
            .fold(0.0f32, |acc, v| acc + v);
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if checksum.is_finite() {
            elapsed
        } else {
            f64::NAN
        }
    };

    let iters = 15;
    let mut off = Vec::with_capacity(iters);
    let mut on = Vec::with_capacity(iters);
    for i in 0..iters {
        // Alternate the within-pair order every iteration as well, so neither
        // mode is systematically the one that follows the other's cache state.
        if (i % 2 == 0) == f16_first {
            on.push(run_once(true));
            off.push(run_once(false));
        } else {
            off.push(run_once(false));
            on.push(run_once(true));
        }
    }

    if off.iter().chain(&on).any(|v| v.is_nan()) {
        println!("skip: at least one measured run failed");
        return;
    }

    let gflop = flops / 1e9;
    let (off_min, on_min) = (
        off.iter().copied().fold(f64::MAX, f64::min),
        on.iter().copied().fold(f64::MAX, f64::min),
    );
    let (off_med, on_med) = (median(&mut off), median(&mut on));
    println!(
        "order={} channels={channels} layers={layers} {gflop:.2} GFLOP/run",
        if f16_first { "f16-first" } else { "f32-first" }
    );
    println!(
        "  f16 OFF: min {off_min:.2} ms  med {off_med:.2} ms  -> {:.0} GFLOP/s",
        gflop / (off_med / 1e3)
    );
    println!(
        "  f16 ON : min {on_min:.2} ms  med {on_med:.2} ms  -> {:.0} GFLOP/s",
        gflop / (on_med / 1e3)
    );
    println!(
        "  speedup: {:.3}x on the median, {:.3}x on the min",
        off_med / on_med,
        off_min / on_min
    );
    println!(
        "  resident weight bytes (both formats held after an A/B): {}",
        session.gpu_resident_bytes()
    );
}
