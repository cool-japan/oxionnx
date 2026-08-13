//! Per-dispatch cost of [`oxionnx_cuda::try_cuda_dispatch`] **with and
//! without** CUDA graph replay, measured against each other in one process.
//!
//! Run on a CUDA-capable host with:
//!
//! ```text
//! cargo run -p oxionnx-cuda --features gpu-tests --release --example graph_dispatch_bench
//! ```
//!
//! # Why this exists next to `dispatch_bench`
//!
//! Two reasons, and the second is not optional on a shared machine:
//!
//! * **It A/Bs the same shape.** `dispatch_bench` reports one number per case;
//!   the question here is the *difference* between two code paths for one
//!   case, which needs both measured under conditions as close to identical as
//!   they can be made.
//! * **It cancels drift.** This GPU is shared. Between two separate benchmark
//!   *processes* the device's clock state and another tenant's load can move a
//!   measurement by more than the effect being measured — observed here as the
//!   same untouched case (`relu 1M`) reporting 1.49 ms in one run and 2.78 ms
//!   in the next. So the two paths are interleaved in **alternating blocks**
//!   inside a single process, and the statistic reported is the **minimum**
//!   per-iteration time over all blocks. Contention can only ever add time, so
//!   the minimum is the closest available estimate of the uncontended cost,
//!   and taking it for both paths under the same interleaving makes the two
//!   comparable even while the machine is busy.
//!
//! Toggling between the paths uses
//! [`CudaContext::set_graph_capture`](oxionnx_cuda::CudaContext::set_graph_capture),
//! so both blocks run against one context, one set of resident weights, and
//! one warmed buffer pool.

use std::collections::HashMap;
use std::time::Instant;

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::Tensor;
use oxionnx_cuda::context::Activation;
use oxionnx_cuda::{try_cuda_dispatch, CudaContext};

/// Untimed dispatches before any block is timed.
const WARMUP: usize = 20;

/// Timed dispatches per block.
const BLOCK: usize = 40;

/// Alternating (graphs-off, graphs-on) block pairs per case.
///
/// More rounds is strictly better for the minimum-of-blocks statistic this
/// reports: every extra round is another chance for each path to be observed
/// during a lull. Eleven is where the numbers stopped moving between runs on
/// this machine.
const ROUNDS: usize = 11;

/// How long to keep the device busy before the first measurement, so its
/// clocks are out of the idle P-state. See `dispatch_bench`'s equivalent.
const DEVICE_SPIN_UP: std::time::Duration = std::time::Duration::from_millis(1500);

fn make_node(op: OpKind, inputs: &[&str]) -> Node {
    Node {
        op,
        name: "bench".to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    }
}

/// Deterministic pseudo-random operand data (the LCG this crate's on-device
/// tests use).
fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = f64::from((state >> 32) as u32) / 4_294_967_296.0;
            (unit * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Time one block of `BLOCK` dispatches, in microseconds per call.
fn time_block(
    ctx: &CudaContext,
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
) -> Option<f64> {
    let start = Instant::now();
    for _ in 0..BLOCK {
        match try_cuda_dispatch(node, weights, intermediates, ctx) {
            Ok(Some(_)) => {}
            Ok(None) => return None,
            Err(_) => return None,
        }
    }
    Some(start.elapsed().as_secs_f64() * 1e6 / BLOCK as f64)
}

/// Interleaved A/B for one node, printing both minima and the delta.
fn bench(
    label: &str,
    ctx: &CudaContext,
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
) {
    // Warm both paths: resident weights, the buffer pool, every compiled
    // kernel, and — for the graph path — the recording itself, which must not
    // land inside a timed block.
    for on in [false, true] {
        ctx.set_graph_capture(on);
        for _ in 0..WARMUP {
            match try_cuda_dispatch(node, weights, intermediates, ctx) {
                Ok(Some(_)) => {}
                Ok(None) => {
                    println!("{label:<32} DECLINED (not accelerated for this configuration)");
                    ctx.set_graph_capture(false);
                    return;
                }
                Err(e) => {
                    println!("{label:<32} ERROR: {e}");
                    ctx.set_graph_capture(false);
                    return;
                }
            }
        }
    }
    let (recorded, poisoned) = ctx.graph_stats();

    let mut off_blocks = Vec::with_capacity(ROUNDS);
    let mut on_blocks = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        ctx.set_graph_capture(false);
        let Some(off) = time_block(ctx, node, weights, intermediates) else {
            println!("{label:<32} ERROR during the graphs-off block");
            return;
        };
        ctx.set_graph_capture(true);
        let Some(on) = time_block(ctx, node, weights, intermediates) else {
            println!("{label:<32} ERROR during the graphs-on block");
            return;
        };
        off_blocks.push(off);
        on_blocks.push(on);
    }
    ctx.set_graph_capture(false);

    // Both statistics, because they answer different questions and a report
    // that quoted only one would be choosing its answer: the **minimum** is the
    // best available estimate of the uncontended cost, and the **median** is
    // what a run on this (shared) machine actually experiences. Where they
    // disagree in sign, neither number is a result.
    let (off_min, off_med) = min_and_median(&mut off_blocks);
    let (on_min, on_med) = min_and_median(&mut on_blocks);
    println!(
        "{label:<32} {off_min:>8.2} {on_min:>8.2} {:>+7.1}% {off_med:>8.2} {on_med:>8.2} \
         {:>+7.1}%   {recorded} rec, {poisoned} poisoned",
        (on_min - off_min) / off_min * 100.0,
        (on_med - off_med) / off_med * 100.0,
    );
}

/// Minimum and median of a block-time sample (sorts `blocks` in place).
fn min_and_median(blocks: &mut [f64]) -> (f64, f64) {
    blocks.sort_by(f64::total_cmp);
    let min = blocks.first().copied().unwrap_or(f64::NAN);
    let median = blocks.get(blocks.len() / 2).copied().unwrap_or(f64::NAN);
    (min, median)
}

/// Keep the device busy so its clocks are up before the first case is timed.
fn spin_up_device(ctx: &CudaContext) {
    let (m, k, n) = (256usize, 256, 256);
    let mut inter = HashMap::new();
    inter.insert(
        "a".to_string(),
        Tensor::new(pseudo_random(m * k, 999), vec![m, k]),
    );
    inter.insert(
        "b".to_string(),
        Tensor::new(pseudo_random(k * n, 998), vec![k, n]),
    );
    let node = make_node(OpKind::MatMul, &["a", "b"]);
    let weights = HashMap::new();
    let deadline = Instant::now() + DEVICE_SPIN_UP;
    while Instant::now() < deadline {
        if try_cuda_dispatch(&node, &weights, &inter, ctx).is_err() {
            return;
        }
    }
}

/// One `[m, k] @ [k, n]` case whose right operand is a graph initializer under
/// a **distinct** name (see `dispatch_bench`'s header for why the name must be
/// unique per case).
fn weighted_matmul(
    label: &str,
    ctx: &CudaContext,
    weight_name: &str,
    m: usize,
    k: usize,
    n: usize,
) {
    let mut weights = HashMap::new();
    weights.insert(
        weight_name.to_string(),
        Tensor::new(pseudo_random(k * n, 3), vec![k, n]),
    );
    let mut inter = HashMap::new();
    inter.insert(
        "a".to_string(),
        Tensor::new(pseudo_random(m * k, 4), vec![m, k]),
    );
    let node = make_node(OpKind::MatMul, &["a", weight_name]);
    bench(label, ctx, &node, &weights, &inter);
}

fn main() {
    let Some(ctx) = CudaContext::try_new_with(Activation::Enabled) else {
        eprintln!("no CUDA device -- run this on a CUDA-capable host");
        return;
    };
    spin_up_device(&ctx);

    println!(
        "{:<32} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "case", "off min", "on min", "delta", "off med", "on med", "delta"
    );
    println!("{}", "-".repeat(105));

    // ── The repeated GEMM shapes of the face pipeline ─────────────────────
    // ArcFace's embedding head: one per detected face per frame.
    weighted_matmul(
        "arcface head 1x25088x512",
        &ctx,
        "w_arcface_head",
        1,
        25088,
        512,
    );
    // InSwapper's AdaIN projection: twelve per frame.
    weighted_matmul("inswapper adain 1x512x2048", &ctx, "w_adain", 1, 512, 2048);
    // A smaller projection, where per-launch overhead is the largest share of
    // the total and graphs therefore have the most to remove.
    weighted_matmul("small proj 1x512x512", &ctx, "w_small", 1, 512, 512);
    weighted_matmul("small proj 1x256x256", &ctx, "w_tiny", 1, 256, 256);
    // Compute-bound, for contrast: nothing here is launch-overhead.
    weighted_matmul("square 128x512x512", &ctx, "w_square", 128, 512, 512);

    // ── Batched MatMul: many launches per dispatch ────────────────────────
    // `gemm_strided_batched` is a host-side loop of one launch per batch
    // element, so a graph collapses `batch` submissions into one — the case
    // with the most launch overhead available to remove.
    for (batch, m, k, n) in [(4usize, 64usize, 128usize, 64usize), (16, 64, 128, 64)] {
        let mut inter = HashMap::new();
        inter.insert(
            "a".to_string(),
            Tensor::new(pseudo_random(batch * m * k, 7), vec![batch, m, k]),
        );
        inter.insert(
            "b".to_string(),
            Tensor::new(pseudo_random(batch * k * n, 8), vec![batch, k, n]),
        );
        let node = make_node(OpKind::MatMul, &["a", "b"]);
        bench(
            &format!("batched matmul b={batch} 64x128x64"),
            &ctx,
            &node,
            &HashMap::new(),
            &inter,
        );
    }

    println!();
    let (recorded, poisoned) = ctx.graph_stats();
    println!("recorded graphs: {recorded} ({poisoned} poisoned)");
    println!(
        "device memory held by the residency caches: {:.1} MiB",
        ctx.cached_device_bytes() as f64 / (1024.0 * 1024.0)
    );
}
