//! Wall-clock cost of *repeated* [`oxionnx_cuda::try_cuda_dispatch`] calls.
//!
//! Run on a CUDA-capable host with:
//!
//! ```text
//! cargo run -p oxionnx-cuda --features gpu-tests --release --example dispatch_bench
//! ```
//!
//! # Why this measures repetition rather than a single call
//!
//! The workload this crate exists for is a video pipeline: the same three
//! graphs (SCRFD, ArcFace, InSwapper) run once per frame, for hundreds of
//! frames, so every node's dispatch happens hundreds of times with *identical*
//! weights and identical shapes. A benchmark that times one call measures
//! mostly first-call costs (context warm-up, kernel JIT) that a real run pays
//! once; a benchmark that times the steady state measures what actually adds up.
//!
//! So each case below runs `WARMUP` untimed dispatches first — enough to fill
//! every module/kernel cache and to let the buffer pool reach its steady-state
//! size — and only then times `ITERS` more. The reported number is per-call
//! wall clock in that steady state.
//!
//! Timing is honest about the device: every `try_cuda_dispatch` path
//! synchronizes its stream(s) and reads the result back to host memory before
//! returning, so the returned `Tensor` is proof the work completed. No extra
//! fence is needed (and none would help — there is nothing left in flight).
//!
//! # The weight-residency cases are the point of the split
//!
//! Cases whose name ends in `[w]` put the second operand in the *weights* map
//! (a graph initializer, whose bytes are invariant for the session) rather than
//! in `intermediates`. That is the exact distinction
//! `try_cuda_dispatch`'s residency keying keys on, so running the same shape
//! both ways isolates what weight residency buys from what buffer pooling buys.

use std::collections::HashMap;
use std::time::Instant;

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::Tensor;
use oxionnx_cuda::context::Activation;
use oxionnx_cuda::{try_cuda_dispatch, CudaContext};

/// Untimed dispatches before the clock starts, per case.
const WARMUP: usize = 10;

/// Timed dispatches per case.
const ITERS: usize = 50;

/// How long to keep the device busy before the first measurement.
///
/// An idle NVIDIA GPU sits in its lowest power state (`P8`, ~15 W on this
/// machine's A4000) and takes a noticeable fraction of a second to ramp its
/// clocks once work arrives. Measured without this, the first case in the
/// table ran up to **2x slower** than the same case measured later in the same
/// process — an artefact of the clock ramp that would otherwise be attributed
/// to whatever happened to be measured first. Spinning the device up front
/// makes the cases comparable to each other and the run comparable to the next
/// run.
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

/// Deterministic pseudo-random operand data (the same LCG this crate's
/// on-device tests use), so a run is reproducible and no value is degenerate.
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

/// Time `ITERS` steady-state dispatches of one node, reporting ms/call.
fn bench(
    label: &str,
    ctx: &CudaContext,
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
) {
    for _ in 0..WARMUP {
        match try_cuda_dispatch(node, weights, intermediates, ctx) {
            Ok(Some(_)) => {}
            Ok(None) => {
                println!("{label:<34} DECLINED (not accelerated for this configuration)");
                return;
            }
            Err(e) => {
                println!("{label:<34} ERROR: {e}");
                return;
            }
        }
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        if let Err(e) = try_cuda_dispatch(node, weights, intermediates, ctx) {
            println!("{label:<34} ERROR: {e}");
            return;
        }
    }
    let per_call = start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
    println!("{label:<34} {per_call:>9.3} ms/call");
}

/// Keep the device busy for [`DEVICE_SPIN_UP`] so its clocks are up before the
/// first case is timed. See that constant for why this is not optional.
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

fn main() {
    let Some(ctx) = CudaContext::try_new_with(Activation::Enabled) else {
        eprintln!("no CUDA device -- run this on a CUDA-capable host");
        return;
    };
    spin_up_device(&ctx);
    println!("{:<34} {:>9}", "case", "steady state");
    println!("{}", "-".repeat(46));

    // ── MatMul, 2-D, activation x activation ──────────────────────────────
    // The confirmed measurement shape from the investigation this work came
    // out of. Both operands are intermediates, so nothing here is resident:
    // this case isolates allocation + upload/readback overhead.
    {
        let (m, k, n) = (128usize, 512, 512);
        let mut inter = HashMap::new();
        inter.insert(
            "a".to_string(),
            Tensor::new(pseudo_random(m * k, 1), vec![m, k]),
        );
        inter.insert(
            "b".to_string(),
            Tensor::new(pseudo_random(k * n, 2), vec![k, n]),
        );
        let node = make_node(OpKind::MatMul, &["a", "b"]);
        bench("matmul 128x512x512", &ctx, &node, &HashMap::new(), &inter);
    }

    // ── MatMul, 2-D, activation x weight ──────────────────────────────────
    // Identical arithmetic, but B is a graph initializer: the difference
    // between this line and the one above is exactly what weight residency
    // removes (a 1 MiB re-upload per call).
    {
        let (m, k, n) = (128usize, 512, 512);
        let mut weights = HashMap::new();
        weights.insert(
            "b".to_string(),
            Tensor::new(pseudo_random(k * n, 2), vec![k, n]),
        );
        let mut inter = HashMap::new();
        inter.insert(
            "a".to_string(),
            Tensor::new(pseudo_random(m * k, 1), vec![m, k]),
        );
        let node = make_node(OpKind::MatMul, &["a", "b"]);
        bench("matmul 128x512x512 [w]", &ctx, &node, &weights, &inter);
    }

    // ── ArcFace's embedding head ──────────────────────────────────────────
    // `[1, 25088] @ [25088, 512]`: a 49 MiB weight against a 98 KiB
    // activation. The most weight-upload-dominated shape in the workload.
    {
        let (m, k, n) = (1usize, 25088, 512);
        let mut weights = HashMap::new();
        weights.insert(
            "b".to_string(),
            Tensor::new(pseudo_random(k * n, 3), vec![k, n]),
        );
        let mut inter = HashMap::new();
        inter.insert(
            "a".to_string(),
            Tensor::new(pseudo_random(m * k, 4), vec![m, k]),
        );
        let node = make_node(OpKind::MatMul, &["a", "b"]);
        bench(
            "arcface head 1x25088x512 [w]",
            &ctx,
            &node,
            &weights,
            &inter,
        );
    }

    // ── InSwapper's AdaIN projection ──────────────────────────────────────
    // `[1, 512] @ [512, 2048]`, of which the graph runs 12 per frame.
    {
        let (m, k, n) = (1usize, 512, 2048);
        let mut weights = HashMap::new();
        weights.insert(
            "b".to_string(),
            Tensor::new(pseudo_random(k * n, 5), vec![k, n]),
        );
        let mut inter = HashMap::new();
        inter.insert(
            "a".to_string(),
            Tensor::new(pseudo_random(m * k, 6), vec![m, k]),
        );
        let node = make_node(OpKind::MatMul, &["a", "b"]);
        bench(
            "inswapper adain 1x512x2048 [w]",
            &ctx,
            &node,
            &weights,
            &inter,
        );
    }

    // ── Batched MatMul ────────────────────────────────────────────────────
    // The per-slice loop this work replaces: `batch` independent GEMMs, each
    // of which used to be its own upload/GEMM/readback round trip.
    for batch in [4usize, 16] {
        let (m, k, n) = (64usize, 128, 64);
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

    // ── Batched MatMul with large slices ──────────────────────────────────
    // Both operands batched, so this cannot collapse and takes the
    // strided-batch launch. Deliberately large per slice: `gemm_strided_batched`
    // runs `GemmTemplate`'s *naive* kernel, whereas the per-slice loop this
    // replaced ran the tuned `GemmDispatcher` one. If the naive kernel were
    // going to lose the arithmetic back, it would be here rather than at the
    // small sizes above — so this case is the guard on that trade, not a
    // showcase.
    {
        let (batch, m, k, n) = (4usize, 256usize, 256, 256);
        let mut inter = HashMap::new();
        inter.insert(
            "a".to_string(),
            Tensor::new(pseudo_random(batch * m * k, 17), vec![batch, m, k]),
        );
        inter.insert(
            "b".to_string(),
            Tensor::new(pseudo_random(batch * k * n, 18), vec![batch, k, n]),
        );
        let node = make_node(OpKind::MatMul, &["a", "b"]);
        bench(
            "batched matmul b=4 256^3",
            &ctx,
            &node,
            &HashMap::new(),
            &inter,
        );
    }

    // ── Broadcast batched MatMul ──────────────────────────────────────────
    // `[8, 64, 128] @ [128, 64]`: B broadcasts across the batch, which is the
    // stride-0 case of the strided-batch dispatch.
    {
        let (batch, m, k, n) = (8usize, 64usize, 128, 64);
        let mut weights = HashMap::new();
        weights.insert(
            "b".to_string(),
            Tensor::new(pseudo_random(k * n, 9), vec![k, n]),
        );
        let mut inter = HashMap::new();
        inter.insert(
            "a".to_string(),
            Tensor::new(pseudo_random(batch * m * k, 10), vec![batch, m, k]),
        );
        let node = make_node(OpKind::MatMul, &["a", "b"]);
        bench("broadcast batched b=8 [w]", &ctx, &node, &weights, &inter);
    }

    // ── Conv: the workload's dominant op ──────────────────────────────────
    // 3x3, stride 1, pad 1, 64 channels at 64x64 -- an InSwapper/SCRFD
    // workhorse. Weight and bias are initializers.
    {
        let (n, c, h, w, out_c) = (1usize, 64usize, 64usize, 64usize, 64usize);
        let mut weights = HashMap::new();
        weights.insert(
            "w".to_string(),
            Tensor::new(pseudo_random(out_c * c * 9, 11), vec![out_c, c, 3, 3]),
        );
        weights.insert(
            "bias".to_string(),
            Tensor::new(pseudo_random(out_c, 12), vec![out_c]),
        );
        let mut inter = HashMap::new();
        inter.insert(
            "x".to_string(),
            Tensor::new(pseudo_random(n * c * h * w, 13), vec![n, c, h, w]),
        );
        let mut node = make_node(OpKind::Conv, &["x", "w", "bias"]);
        node.attrs
            .int_lists
            .insert("strides".to_string(), vec![1, 1]);
        node.attrs
            .int_lists
            .insert("pads".to_string(), vec![1, 1, 1, 1]);
        node.attrs
            .int_lists
            .insert("dilations".to_string(), vec![1, 1]);
        node.attrs.ints.insert("group".to_string(), 1);
        bench("conv 3x3 64ch 64x64 [w]", &ctx, &node, &weights, &inter);
    }

    // ── Elementwise / reduce / softmax ────────────────────────────────────
    {
        let len = 1 << 20;
        let mut inter = HashMap::new();
        inter.insert(
            "x".to_string(),
            Tensor::new(pseudo_random(len, 14), vec![len]),
        );
        bench(
            "relu 1M",
            &ctx,
            &make_node(OpKind::Relu, &["x"]),
            &HashMap::new(),
            &inter,
        );

        let mut binary = HashMap::new();
        binary.insert(
            "x".to_string(),
            Tensor::new(pseudo_random(len, 14), vec![len]),
        );
        binary.insert(
            "y".to_string(),
            Tensor::new(pseudo_random(len, 15), vec![len]),
        );
        bench(
            "add 1M",
            &ctx,
            &make_node(OpKind::Add, &["x", "y"]),
            &HashMap::new(),
            &binary,
        );

        let mut rows = HashMap::new();
        rows.insert(
            "x".to_string(),
            Tensor::new(pseudo_random(1024 * 512, 16), vec![1024, 512]),
        );
        let mut softmax_node = make_node(OpKind::Softmax, &["x"]);
        softmax_node.attrs.ints.insert("axis".to_string(), -1);
        bench(
            "softmax 1024x512",
            &ctx,
            &softmax_node,
            &HashMap::new(),
            &rows,
        );

        let mut reduce_node = make_node(OpKind::ReduceSum, &["x"]);
        reduce_node
            .attrs
            .int_lists
            .insert("axes".to_string(), vec![1]);
        bench(
            "reducesum 1024x512 axis1",
            &ctx,
            &reduce_node,
            &HashMap::new(),
            &rows,
        );
    }

    // ── What the caches did ───────────────────────────────────────────────
    //
    // The timings above are the claim; this is the mechanism, and the second
    // line is the one that matters. A steady-state frame must upload **zero**
    // weight bytes: every initializer crossed the bus during warm-up and never
    // again. Anything else means some operand is not being keyed, or is
    // conflicting with another under the same name — a cache that looks busy
    // while re-uploading everything. (Pinned as an assertion, not just
    // printed, by `tests/batched_matmul_gpu.rs`.)
    let before = ctx.cache_counters();
    {
        let (m, k, n) = (128usize, 512, 512);
        let mut weights = HashMap::new();
        weights.insert(
            "b".to_string(),
            Tensor::new(pseudo_random(k * n, 2), vec![k, n]),
        );
        let mut inter = HashMap::new();
        inter.insert(
            "a".to_string(),
            Tensor::new(pseudo_random(m * k, 1), vec![m, k]),
        );
        let node = make_node(OpKind::MatMul, &["a", "b"]);
        // Warm this exact identity, then measure one further steady-state call.
        let _ = try_cuda_dispatch(&node, &weights, &inter, &ctx);
        let warm = ctx.cache_counters();
        let _ = try_cuda_dispatch(&node, &weights, &inter, &ctx);
        let delta = ctx.cache_counters().since(warm);
        println!();
        println!("one steady-state dispatch:");
        println!(
            "  pool: {} reused, {} allocated, {} evicted",
            delta.pool_hits, delta.pool_allocs, delta.pool_evictions
        );
        println!(
            "  weights: {} hits, {} misses, {} bytes uploaded",
            delta.weight_hits, delta.weight_misses, delta.weight_bytes_uploaded
        );
    }
    let session = ctx.cache_counters().since(before);
    println!(
        "since the caches warmed: {} weight bytes uploaded in total",
        session.weight_bytes_uploaded
    );
    println!(
        "device memory held by the caches: {:.1} MiB",
        ctx.cached_device_bytes() as f64 / (1024.0 * 1024.0)
    );
}
