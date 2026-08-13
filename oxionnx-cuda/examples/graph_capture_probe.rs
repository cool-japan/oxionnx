//! Hardware probe for driver-backed CUDA stream capture / graph replay.
//!
//! Run on a CUDA-capable host with:
//!
//! ```text
//! cargo run -p oxionnx-cuda --features gpu-tests --release --example graph_capture_probe
//! ```
//!
//! # What this proves, and why it comes before any integration
//!
//! [`oxicuda_driver::graph::StreamGraphCapture`] claims to drive the real
//! `cuStreamBeginCapture_v2` / `cuStreamEndCapture` / `cuGraphInstantiate`
//! path. Before wiring graph replay into `oxionnx-cuda`'s dispatch layer, this
//! example establishes — on this machine, against this driver — six separate
//! facts, each of which the integration depends on and none of which can be
//! taken on faith:
//!
//! 1. **A multi-kernel capture is driver-backed.** Two *real* launches (a tuned
//!    `oxicuda-blas` GEMM and an `oxicuda-ptx` elementwise kernel) issued on a
//!    non-default stream between `begin` and `end` produce a `GraphExec` with
//!    `is_driver_backed() == true` and a node count of at least 2.
//! 2. **Capture records rather than executes.** The output buffer still holds
//!    its sentinel after `end()` and before the first replay.
//! 3. **Replay is numerically exact.** The replayed result matches the same
//!    computation run through ordinary launches, bit for bit — and stays exact
//!    across replays with the *input* rewritten in between, which is the
//!    property that makes a cached graph usable for a per-frame workload.
//! 4. **Replay is cheaper than re-issuing.** Per-iteration wall time for
//!    `ITERS` graph replays versus `ITERS` ordinary launch pairs, measured both
//!    with a per-iteration synchronise (what a real dispatch pays) and without
//!    one (which isolates host-side submission cost).
//! 5. **…including in a round-trip shape.** The same comparison with an H2D
//!    upload before and a D2H readback after, on the same stream — the shape a
//!    real [`oxionnx_cuda::try_cuda_dispatch`] actually has. Whether a
//!    `cuGraphLaunch` is cheaper *in that position* is a different question
//!    from whether it is cheaper alone, and the answer differs, so it is
//!    measured rather than inferred.
//! 6. **Convolution is measured too, and separately.** `crate::conv` issues a
//!    single kernel per dispatch, so its recording holds exactly one node —
//!    the case with the least launch overhead available for a graph to remove.
//!    [`CONV_SHAPES`] is what decided that `conv.rs` is not worth integrating.
//!
//! It also probes the failure modes the integration must survive: a
//! `cuMemAlloc` mid-capture (which is exactly what `oxicuda-blas`'s split-K
//! GEMM path used to do — see [`SHAPES`]) and an un-`end`ed capture being
//! dropped.
//!
//! Everything here talks to `oxicuda-*` directly rather than through
//! [`oxionnx_cuda::try_cuda_dispatch`]: the point is to characterise the
//! *primitive*, with none of this crate's dispatch logic in the way.

use std::sync::Arc;
use std::time::Instant;

use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::{level3::gemm_api::gemm, Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_dnn::conv::descriptor::ConvProblem;
use oxicuda_dnn::conv::fprop::implicit_gemm::ImplicitGemmConv;
use oxicuda_dnn::handle::DnnHandle;
use oxicuda_dnn::types::{TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_driver::ffi::{CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CU_STREAM_CAPTURE_STATUS_ACTIVE};
use oxicuda_driver::graph::StreamGraphCapture;
use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{grid_size_for, Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::{
    ir::PtxType,
    templates::elementwise::{ElementwiseOp, ElementwiseTemplate},
};

/// Untimed iterations before the clock starts.
const WARMUP: usize = 20;

/// Timed iterations per measured case.
const ITERS: usize = 200;

/// Elementwise block size, matching `crate::elementwise`'s.
const BLOCK_SIZE: u32 = 256;

/// Sentinel written into the intermediate buffer before a capture, so "capture
/// did not execute" is observable rather than assumed.
const SENTINEL: f32 = -12_345.5;

/// `(label, m, k, n)` cases, chosen to straddle `oxicuda-blas`'s split-K rule
/// (`m*n < 65536 && k >= 512`, see `GemmDispatcher::should_use_split_k_workspace`).
///
/// That rule is not a detail here: the split-K path allocates and frees its
/// reduction workspace *inside* the GEMM call, and `cuMemAlloc` is one of the
/// calls the driver forbids during capture. Every GEMM shape the face pipeline
/// actually repeats — ArcFace's `[1, 25088] @ [25088, 512]` head, InSwapper's
/// `[1, 512] @ [512, 2048]` AdaIN projections — lands on the split-K side of
/// that line, so a probe that only measured the single-pass side would prove
/// the primitive on precisely the shapes the workload does not have.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    // Single pass: m*n == 65536, exactly at the threshold, so NOT split-K.
    ("single-pass 256x256x256", 256, 256, 256),
    // Split-K: InSwapper's AdaIN projection, 12 per frame.
    ("split-k 1x512x2048 (inswapper adain)", 1, 512, 2048),
    // Split-K: ArcFace's embedding head.
    ("split-k 1x25088x512 (arcface head)", 1, 25088, 512),
];

/// `(label, in_channels, out_channels, spatial)` 3x3 stride-1 pad-1
/// convolutions, sized after InSwapper's 128x128 generator: a full-resolution
/// early block, a mid-resolution block, and the widest bottleneck block.
///
/// Convolution is the dominant op of the workload by node count, so whether
/// graph replay helps *here* decides whether it is worth wiring into
/// `conv.rs` at all. It is probed through the same `ImplicitGemmConv` engine
/// `crate::conv` dispatches to for a general 3x3.
const CONV_SHAPES: &[(&str, u32, u32, u32)] = &[
    ("conv3x3 64->64 @128x128 (inswapper)", 64, 64, 128),
    ("conv3x3 256->256 @32x32 (inswapper)", 256, 256, 32),
    ("conv3x3 512->512 @16x16 (inswapper)", 512, 512, 16),
];

/// Deterministic pseudo-random data (the LCG this crate's on-device tests use).
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

/// Device-side state for one probed shape.
struct Case {
    m: usize,
    k: usize,
    n: usize,
    d_a: DeviceBuffer<f32>,
    d_b: DeviceBuffer<f32>,
    d_c: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
}

impl Case {
    fn new(m: usize, k: usize, n: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut d_a = DeviceBuffer::<f32>::alloc(m * k)?;
        let mut d_b = DeviceBuffer::<f32>::alloc(k * n)?;
        d_a.copy_from_host(&pseudo_random(m * k, 101))?;
        d_b.copy_from_host(&pseudo_random(k * n, 202))?;
        Ok(Self {
            m,
            k,
            n,
            d_a,
            d_b,
            d_c: DeviceBuffer::<f32>::alloc(m * n)?,
            d_out: DeviceBuffer::<f32>::alloc(m * n)?,
        })
    }

    fn out_len(&self) -> usize {
        self.m * self.n
    }

    /// The two-kernel body this probe captures and replays: a GEMM into `d_c`,
    /// then a ReLU of `d_c` into `d_out`, both on `stream`.
    ///
    /// Issued identically whether a capture is active or not — which is the
    /// whole point: a captured graph must be the *same* work the normal path
    /// issues.
    fn issue(
        &mut self,
        blas: &BlasHandle,
        relu: &Kernel,
        stream: &Stream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let desc_a = MatrixDesc::<f32>::from_buffer(
            &self.d_a,
            self.m as u32,
            self.k as u32,
            Layout::RowMajor,
        )?;
        let desc_b = MatrixDesc::<f32>::from_buffer(
            &self.d_b,
            self.k as u32,
            self.n as u32,
            Layout::RowMajor,
        )?;
        let mut desc_c = MatrixDescMut::<f32>::from_buffer(
            &mut self.d_c,
            self.m as u32,
            self.n as u32,
            Layout::RowMajor,
        )?;
        gemm(
            blas,
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0_f32,
            &desc_a,
            &desc_b,
            0.0_f32,
            &mut desc_c,
        )?;

        let n_elems = u32::try_from(self.out_len())?;
        let params = LaunchParams::new(
            Dim3::from(grid_size_for(n_elems, BLOCK_SIZE)),
            Dim3::from(BLOCK_SIZE),
        );
        let args = (
            self.d_c.as_device_ptr(),
            self.d_out.as_device_ptr(),
            n_elems,
        );
        relu.launch(&params, stream, &args)?;
        Ok(())
    }
}

/// Largest absolute difference between two equal-length slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Probe one shape end to end, printing what it found. Returns `Err` only for
/// an environment failure; a shape that cannot be captured is *reported*, not
/// fatal — characterising that is half of what this probe is for.
fn probe_shape(
    label: &str,
    (m, k, n): (usize, usize, usize),
    blas: &BlasHandle,
    relu: &Kernel,
    stream: &Stream,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {label} ===");
    let mut case = Case::new(m, k, n)?;
    let len = case.out_len();

    // Reference: the same body through ordinary launches.
    case.issue(blas, relu, stream)?;
    stream.synchronize()?;
    let mut reference = vec![0.0_f32; len];
    case.d_out.copy_to_host(&mut reference)?;

    // ── Fact 1 + 2: capture is driver-backed and does not execute ──────────
    case.d_c.copy_from_host(&vec![SENTINEL; len])?;
    let capture = StreamGraphCapture::begin(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let status = capture.capture_status()?;
    if status != CU_STREAM_CAPTURE_STATUS_ACTIVE {
        return Err(format!("stream reports capture status {status}, expected ACTIVE").into());
    }
    let issued = case.issue(blas, relu, stream);
    let exec = match issued {
        Ok(()) => capture.end()?,
        Err(e) => {
            // Abandon the capture; `StreamGraphCapture::drop` ends it and
            // destroys whatever partial graph the driver hands back, leaving
            // the stream usable.
            drop(capture);
            println!("  NOT CAPTURABLE: {e}");
            // Prove the stream survived the abandoned capture.
            case.issue(blas, relu, stream)?;
            stream.synchronize()?;
            let mut after = vec![0.0_f32; len];
            case.d_out.copy_to_host(&mut after)?;
            println!(
                "  stream still usable after the abort (max abs diff vs reference {:e})",
                max_abs_diff(&after, &reference)
            );
            return Ok(());
        }
    };
    println!(
        "  capture: driver_backed={} nodes={}",
        exec.is_driver_backed(),
        exec.node_count()
    );
    if !exec.is_driver_backed() {
        return Err("captured graph is NOT driver-backed".into());
    }
    if exec.node_count() < 2 {
        return Err(format!(
            "capture recorded {} node(s); expected at least 2 (GEMM + ReLU)",
            exec.node_count()
        )
        .into());
    }
    let mut out_host = vec![0.0_f32; len];
    case.d_c.copy_to_host(&mut out_host)?;
    if !out_host
        .iter()
        .all(|&v| (v - SENTINEL).abs() < f32::EPSILON)
    {
        return Err("capture EXECUTED the recorded work; it must only record".into());
    }
    println!("  capture recorded without executing: yes");

    // ── Fact 3: replay is numerically exact, repeatedly ────────────────────
    exec.launch(stream)?;
    stream.synchronize()?;
    case.d_out.copy_to_host(&mut out_host)?;
    let first_diff = max_abs_diff(&out_host, &reference);
    println!("  replay vs normal launch, max abs diff: {first_diff:e}");
    if first_diff != 0.0 {
        return Err(
            format!("replayed graph differs from the normal path by {first_diff:e}").into(),
        );
    }

    // Rewrite the *input* between replays: the graph holds pointers, not
    // values, so a new A must produce a new (and correct) answer.
    case.d_a.copy_from_host(&pseudo_random(m * k, 303))?;
    exec.launch(stream)?;
    stream.synchronize()?;
    let mut replay2 = vec![0.0_f32; len];
    case.d_out.copy_to_host(&mut replay2)?;
    case.issue(blas, relu, stream)?;
    stream.synchronize()?;
    let mut normal2 = vec![0.0_f32; len];
    case.d_out.copy_to_host(&mut normal2)?;
    let second_diff = max_abs_diff(&replay2, &normal2);
    println!("  replay tracks a rewritten input, max abs diff: {second_diff:e}");
    if second_diff != 0.0 {
        return Err("replay did not track the rewritten input".into());
    }
    if max_abs_diff(&replay2, &reference) == 0.0 {
        return Err("replay produced the OLD answer for a new input".into());
    }

    // ── Fact 4: per-iteration cost, replay vs re-issue ─────────────────────
    for _ in 0..WARMUP {
        case.issue(blas, relu, stream)?;
        stream.synchronize()?;
        exec.launch(stream)?;
        stream.synchronize()?;
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        case.issue(blas, relu, stream)?;
        stream.synchronize()?;
    }
    let normal_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let start = Instant::now();
    for _ in 0..ITERS {
        exec.launch(stream)?;
        stream.synchronize()?;
    }
    let graph_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let start = Instant::now();
    for _ in 0..ITERS {
        case.issue(blas, relu, stream)?;
    }
    stream.synchronize()?;
    let normal_submit_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let start = Instant::now();
    for _ in 0..ITERS {
        exec.launch(stream)?;
    }
    stream.synchronize()?;
    let graph_submit_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // ── Fact 5: the same comparison in a *round-trip* shape ────────────────
    //
    // A real `try_cuda_dispatch` does not launch in isolation: it uploads this
    // frame's activation, launches, reads the result back, and synchronises,
    // all on one stream. That surrounds the launches (or the `cuGraphLaunch`)
    // with async memcpys, and whether a graph is cheaper *in that position* is
    // a different question from whether it is cheaper on its own — one that
    // the integrated benchmark answers differently, so it is measured here
    // rather than inferred.
    let host_a = pseudo_random(m * k, 505);
    let mut readback = vec![0.0_f32; len];

    for _ in 0..WARMUP {
        case.d_a.copy_from_host_async(&host_a, stream)?;
        exec.launch(stream)?;
        case.d_out.copy_to_host_async(&mut readback, stream)?;
        stream.synchronize()?;
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        case.d_a.copy_from_host_async(&host_a, stream)?;
        case.issue(blas, relu, stream)?;
        case.d_out.copy_to_host_async(&mut readback, stream)?;
        stream.synchronize()?;
    }
    let normal_round_trip_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let start = Instant::now();
    for _ in 0..ITERS {
        case.d_a.copy_from_host_async(&host_a, stream)?;
        exec.launch(stream)?;
        case.d_out.copy_to_host_async(&mut readback, stream)?;
        stream.synchronize()?;
    }
    let graph_round_trip_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    println!("  normal launches + sync : {normal_us:>8.2} us/iter");
    println!("  graph replay    + sync : {graph_us:>8.2} us/iter");
    println!(
        "    -> {:+.1}% wall",
        (graph_us - normal_us) / normal_us * 100.0
    );
    println!("  normal submit (no sync): {normal_submit_us:>8.2} us/iter");
    println!("  graph  submit (no sync): {graph_submit_us:>8.2} us/iter");
    println!(
        "    -> {:+.1}% host submission",
        (graph_submit_us - normal_submit_us) / normal_submit_us * 100.0
    );
    println!("  normal H2D+work+D2H+sync: {normal_round_trip_us:>8.2} us/iter");
    println!("  graph  H2D+work+D2H+sync: {graph_round_trip_us:>8.2} us/iter");
    println!(
        "    -> {:+.1}% round trip",
        (graph_round_trip_us - normal_round_trip_us) / normal_round_trip_us * 100.0
    );
    Ok(())
}

/// Probe one 3x3 stride-1 pad-1 convolution the same way [`probe_shape`]
/// probes a GEMM, through `oxicuda-dnn`'s `ImplicitGemmConv` — the engine
/// `crate::conv` dispatches a general 3x3 to.
///
/// Convolution rides `DnnHandle`'s own stream rather than the BLAS one (the
/// two are separate; see `DnnHandle::build`), so this builds its own handle.
fn probe_conv(
    label: &str,
    in_channels: u32,
    out_channels: u32,
    spatial: u32,
    ctx: &Arc<Context>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {label} ===");
    let dnn = DnnHandle::new(ctx)?;
    let stream = dnn.stream();

    let problem = ConvProblem {
        batch: 1,
        in_channels,
        in_dims: vec![spatial, spatial],
        out_channels,
        filter_dims: vec![3, 3],
        padding: vec![1, 1],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        groups: 1,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    };
    let out_dims = problem.output_dims()?;
    let in_len = (in_channels * spatial * spatial) as usize;
    let filter_len = (out_channels * in_channels * 9) as usize;
    let out_len = (out_channels * out_dims[0] * out_dims[1]) as usize;

    let mut d_in = DeviceBuffer::<f32>::alloc(in_len)?;
    let mut d_filter = DeviceBuffer::<f32>::alloc(filter_len)?;
    let mut d_out = DeviceBuffer::<f32>::alloc(out_len)?;
    d_in.copy_from_host(&pseudo_random(in_len, 401))?;
    d_filter.copy_from_host(&pseudo_random(filter_len, 402))?;

    let in_desc = TensorDesc::<f32>::nchw(
        &d_in,
        1,
        in_channels,
        problem.in_dims[0],
        problem.in_dims[1],
    )?;
    let filter_desc = TensorDesc::<f32>::nchw(&d_filter, out_channels, in_channels, 3, 3)?;
    let engine = ImplicitGemmConv::new(problem, dnn.sm_version());

    // A closure cannot hold `&mut d_out` alongside the reads above, so the
    // output descriptor is rebuilt per issue — exactly as `crate::conv` does.
    let issue = |d_out: &mut DeviceBuffer<f32>| -> Result<(), Box<dyn std::error::Error>> {
        let mut out_desc =
            TensorDescMut::<f32>::nchw(d_out, 1, out_channels, out_dims[0], out_dims[1])?;
        engine.execute(&dnn, &in_desc, &filter_desc, None, &mut out_desc)?;
        Ok(())
    };

    issue(&mut d_out)?;
    stream.synchronize()?;
    let mut reference = vec![0.0_f32; out_len];
    d_out.copy_to_host(&mut reference)?;

    d_out.copy_from_host(&vec![SENTINEL; out_len])?;
    let capture = StreamGraphCapture::begin(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let recorded = issue(&mut d_out);
    let exec = match recorded {
        Ok(()) => capture.end()?,
        Err(e) => {
            drop(capture);
            println!("  NOT CAPTURABLE: {e}");
            return Ok(());
        }
    };
    println!(
        "  capture: driver_backed={} nodes={}",
        exec.is_driver_backed(),
        exec.node_count()
    );
    let mut probe = vec![0.0_f32; out_len];
    d_out.copy_to_host(&mut probe)?;
    if !probe.iter().all(|&v| (v - SENTINEL).abs() < f32::EPSILON) {
        return Err("conv capture EXECUTED the recorded work".into());
    }

    exec.launch(stream)?;
    stream.synchronize()?;
    d_out.copy_to_host(&mut probe)?;
    let diff = max_abs_diff(&probe, &reference);
    println!("  replay vs normal launch, max abs diff: {diff:e}");
    if diff != 0.0 {
        return Err(format!("replayed conv graph differs by {diff:e}").into());
    }

    for _ in 0..WARMUP {
        issue(&mut d_out)?;
        stream.synchronize()?;
        exec.launch(stream)?;
        stream.synchronize()?;
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        issue(&mut d_out)?;
        stream.synchronize()?;
    }
    let normal_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let start = Instant::now();
    for _ in 0..ITERS {
        exec.launch(stream)?;
        stream.synchronize()?;
    }
    let graph_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    println!("  normal launch + sync : {normal_us:>9.2} us/iter");
    println!("  graph replay  + sync : {graph_us:>9.2} us/iter");
    println!(
        "    -> {:+.1}% wall",
        (graph_us - normal_us) / normal_us * 100.0
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    oxicuda_driver::init()?;
    let device = Device::get(0)?;
    let ctx = Arc::new(Context::new(&device)?);
    ctx.set_current()?;

    // The BLAS handle owns the stream everything below rides. `Stream::new`
    // sets `CU_STREAM_NON_BLOCKING`, which capture requires (the legacy default
    // stream cannot be captured).
    let blas = BlasHandle::with_stream(&ctx, Stream::new(&ctx)?)?;
    let stream = blas.stream();

    let template = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F32, blas.sm_version());
    let kernel_name = template.kernel_name();
    let module = Arc::new(Module::from_ptx(&template.generate()?)?);
    let relu = Kernel::from_module(module, &kernel_name)?;

    // Spin the device's clocks up so the first shape is not measured in P8.
    {
        let mut warm = Case::new(256, 256, 256)?;
        let deadline = Instant::now() + std::time::Duration::from_millis(1500);
        while Instant::now() < deadline {
            warm.issue(&blas, &relu, stream)?;
            stream.synchronize()?;
        }
    }

    for &(label, m, k, n) in SHAPES {
        probe_shape(label, (m, k, n), &blas, &relu, stream)?;
    }

    for &(label, in_channels, out_channels, spatial) in CONV_SHAPES {
        probe_conv(label, in_channels, out_channels, spatial, &ctx)?;
    }

    // ── Failure mode A: a device allocation mid-capture ────────────────────
    //
    // Establishes what the driver does about the hazard by name, since the
    // integration's safety argument rests on it: an allocation that is freed
    // before replay would leave the graph reading released memory.
    {
        println!("\n=== failure modes ===");
        let cap = StreamGraphCapture::begin(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let alloc_result = DeviceBuffer::<f32>::alloc(1024);
        match &alloc_result {
            Ok(_) => println!("  mid-capture cuMemAlloc: SUCCEEDED (driver permits it)"),
            Err(e) => println!("  mid-capture cuMemAlloc: rejected -- {e}"),
        }
        drop(alloc_result);
        match cap.end() {
            Ok(exec) => println!(
                "    capture still ended, nodes={} driver_backed={}",
                exec.node_count(),
                exec.is_driver_backed()
            ),
            Err(e) => println!("    capture was invalidated: {e}"),
        }
    }

    // ── Failure mode B: dropping an un-ended capture ───────────────────────
    {
        let mut case = Case::new(64, 64, 64)?;
        let cap = StreamGraphCapture::begin(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        drop(cap);
        case.issue(&blas, &relu, stream)?;
        stream.synchronize()?;
        println!("  dropped capture leaves the stream usable: yes");
    }

    Ok(())
}

fn main() {
    match run() {
        Ok(()) => println!("\nprobe: OK"),
        Err(e) => {
            eprintln!("probe FAILED: {e}");
            std::process::exit(1);
        }
    }
}
