//! [c3] Conv2D throughput: direct implicit-GEMM kernel vs the hybrid im2col
//! path it replaces vs the rayon CPU operator.
//!
//! Run:
//!     cargo run -p oxionnx-gpu --example c3_conv2d_gflops
//!
//! (No `--release` needed: this workspace's `[profile.dev]` is already
//! `opt-level = 3`.)
//!
//! # What is measured, and why the breakdown is there
//!
//! Every number is **end to end** — operand upload, dispatch, read-back — the
//! same cost production code pays per call, exactly as
//! `h2_tiled_gemm_gflops.rs` measures the tiled GEMM.
//!
//! The three paths do the same FLOPs. What differs is what they move and what
//! they do on the host, so a bare speedup number would not say *why* one wins.
//! Each case therefore also reports:
//!
//! * **staged bytes per call** — the direct kernel stages the input and the
//!   weights; the hybrid stages the weights and an `[K, N]` column matrix that
//!   is `kH*kW` times the input. For InSwapper's dominant layer that is 4.0 MB
//!   vs 37.7 MB, per call, twelve times per frame.
//! * **host im2col ms** — pure CPU work the hybrid does before it can dispatch
//!   at all, and which the direct kernel does not do.
//! * **staging memcpy ms** for each of those buffers, measured with the same
//!   `create_buffer_init` the kernels use.
//!
//! If the direct kernel wins, that breakdown says which of the three
//! differences paid for it. If it does not, the same breakdown is the
//! bottleneck analysis.

use std::time::Instant;

use oxionnx_core::Tensor;
use oxionnx_gpu::compute::gpu_conv2d_hybrid_async;
use oxionnx_gpu::shaders::{gpu_conv2d_implicit, ConvActivation};
use oxionnx_gpu::GpuContext;
use oxionnx_gpu::TensorSource;
use wgpu::util::DeviceExt;

const WARMUP: usize = 2;
/// `median` indexes at `len / 2`, which panics on an empty slice — guard the
/// constant itself so it cannot be lowered to zero without a compile error.
const _ITERS_MUST_BE_NONZERO: () = assert!(ITERS > 0);
const ITERS: usize = 5;
/// The CPU reference is slow enough at these shapes that five samples is pure
/// wall-clock waste; two is enough to see if the first was an outlier.
const CPU_ITERS: usize = 2;

/// One benchmark shape, in the same terms as the model dump it came from.
struct Case {
    label: &'static str,
    /// `[N, C_in, H, W]`.
    input: [usize; 4],
    /// `[C_out, C_in, kH, kW]`.
    weight: [usize; 4],
    strides: [usize; 2],
    /// ONNX order: `[top, left, bottom, right]`.
    pads: [usize; 4],
}

const CASES: &[Case] = &[
    Case {
        // Twelve of these per InSwapper frame — the model's dominant cost.
        label: "InSwapper bottleneck  Conv[1024,1024,3,3] s1 p0 @ 34x34 -> 32x32  (x12/frame)",
        input: [1, 1024, 34, 34],
        weight: [1024, 1024, 3, 3],
        strides: [1, 1],
        pads: [0, 0, 0, 0],
    },
    Case {
        // The single most expensive node in InSwapper: 19.33 GMAC.
        label: "InSwapper decoder     Conv[256,512,3,3] s1 p1 @ 128x128",
        input: [1, 512, 128, 128],
        weight: [256, 512, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
    },
    Case {
        // Four of these per ArcFace frame; representative of the trunk.
        label: "ArcFace trunk         Conv[64,64,3,3] s1 p1 @ 56x56  (x4/frame)",
        input: [1, 64, 56, 56],
        weight: [64, 64, 3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
    },
];

/// Deterministic, signed, non-monotonic fill — see `h2_tiled_gemm_gflops.rs`.
fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            ((x % 23) as f32) * 0.037 - 0.4
        })
        .collect()
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// `[K, N]` column matrix for one batch element, exactly as the hybrid path
/// builds it on the host before every dispatch. Reproduced here (rather than
/// instrumented inside `compute.rs`) so the cost can be attributed without
/// changing the path being measured.
#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    oh: usize,
    ow: usize,
    col: &mut [f32],
) {
    let col_cols = oh * ow;
    let mut row = 0usize;
    for ic in 0..c_in {
        let plane = &input[ic * h * w..][..h * w];
        for ky in 0..kh {
            for kx in 0..kw {
                for oy in 0..oh {
                    let iy = (oy * strides[0] + ky) as isize - pads[0] as isize;
                    let base = row * col_cols + oy * ow;
                    if iy < 0 || iy >= h as isize {
                        for ox in 0..ow {
                            col[base + ox] = 0.0;
                        }
                        continue;
                    }
                    let iy = iy as usize;
                    for ox in 0..ow {
                        let ix = (ox * strides[1] + kx) as isize - pads[1] as isize;
                        col[base + ox] = if ix >= 0 && ix < w as isize {
                            plane[iy * w + ix as usize]
                        } else {
                            0.0
                        };
                    }
                }
                row += 1;
            }
        }
    }
}

/// Median wall time of staging `len` f32s into a fresh storage buffer, the way
/// every kernel in this crate uploads an operand.
fn staging_ms(ctx: &GpuContext, len: usize) -> f64 {
    let data = vec![0.5f32; len];
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("c3_staging_probe"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let elapsed = t.elapsed().as_secs_f64();
        drop(buf);
        samples.push(elapsed);
    }
    median(&mut samples) * 1000.0
}

fn report(name: &str, ms: Option<f64>, gflop: f64, baseline: Option<f64>) {
    match ms {
        None => println!("    {name:<22} [declined]"),
        Some(ms) => {
            let speedup = baseline.map_or(String::new(), |b| format!("   {:.2}x", b / ms));
            println!(
                "    {name:<22} {:>10.3} ms   {:>8.1} GFLOP/s{speedup}",
                ms,
                gflop / (ms / 1000.0) / 1e9
            );
        }
    }
}

fn run_case(ctx: &GpuContext, case: &Case) {
    let [n, c_in, h, w] = case.input;
    let [c_out, _, kh, kw] = case.weight;
    let oh = (h + case.pads[0] + case.pads[2] - kh) / case.strides[0] + 1;
    let ow = (w + case.pads[1] + case.pads[3] - kw) / case.strides[1] + 1;
    let (m, k, n_gemm) = (c_out, c_in * kh * kw, oh * ow);
    // One multiply and one add per MAC.
    let gflop = 2.0 * n as f64 * m as f64 * k as f64 * n_gemm as f64;

    let input = Tensor::new(fill(n * c_in * h * w, 2_654_435_761), case.input.to_vec());
    let weight = Tensor::new(fill(c_out * k, 40_503), case.weight.to_vec());
    let bias = Tensor::new(fill(c_out, 97_711), vec![c_out]);

    println!("\n  {}", case.label);
    println!(
        "    GEMM M={m} K={k} N={n_gemm}   {:.2} GFLOP   out={:?}",
        gflop / 1e9,
        [n, c_out, oh, ow]
    );

    // --- direct implicit-GEMM kernel ---
    let mut direct = None;
    for _ in 0..WARMUP {
        if gpu_conv2d_implicit(
            ctx,
            &input,
            &weight,
            Some(&bias),
            case.strides,
            case.pads,
            [1, 1],
            1,
            ConvActivation::Relu,
        )
        .is_none()
        {
            break;
        }
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let out = gpu_conv2d_implicit(
            ctx,
            &input,
            &weight,
            Some(&bias),
            case.strides,
            case.pads,
            [1, 1],
            1,
            ConvActivation::Relu,
        );
        let elapsed = t.elapsed().as_secs_f64();
        if out.is_none() {
            samples.clear();
            break;
        }
        samples.push(elapsed);
    }
    if !samples.is_empty() {
        direct = Some(median(&mut samples) * 1000.0);
    }

    // --- hybrid: host im2col + GPU GEMM + host bias ---
    let mut hybrid = None;
    for _ in 0..WARMUP {
        if pollster::block_on(gpu_conv2d_hybrid_async(
            ctx,
            TensorSource::tensor(&input),
            &weight,
            Some(&bias),
            case.strides,
            case.pads,
            [1, 1],
            1,
        ))
        .is_none()
        {
            break;
        }
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let out = pollster::block_on(gpu_conv2d_hybrid_async(
            ctx,
            TensorSource::tensor(&input),
            &weight,
            Some(&bias),
            case.strides,
            case.pads,
            [1, 1],
            1,
        ));
        let elapsed = t.elapsed().as_secs_f64();
        if out.is_none() {
            samples.clear();
            break;
        }
        samples.push(elapsed);
    }
    if !samples.is_empty() {
        hybrid = Some(median(&mut samples) * 1000.0);
    }

    // --- rayon CPU operator ---
    let _ = oxionnx_ops::conv::conv2d(
        &input,
        &weight,
        Some(&bias),
        case.strides,
        case.pads,
        [1, 1],
        1,
    );
    let mut samples = Vec::with_capacity(CPU_ITERS);
    for _ in 0..CPU_ITERS {
        let t = Instant::now();
        let out = oxionnx_ops::conv::conv2d(
            &input,
            &weight,
            Some(&bias),
            case.strides,
            case.pads,
            [1, 1],
            1,
        );
        samples.push(t.elapsed().as_secs_f64());
        std::hint::black_box(&out);
    }
    let cpu = median(&mut samples) * 1000.0;

    // --- where the difference comes from ---
    let in_len = n * c_in * h * w;
    let weight_len = c_out * k;
    let col_len = k * n_gemm;
    let mut col = vec![0.0f32; col_len];
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        im2col(
            &input.data,
            c_in,
            h,
            w,
            kh,
            kw,
            case.strides,
            case.pads,
            oh,
            ow,
            &mut col,
        );
        samples.push(t.elapsed().as_secs_f64());
    }
    let im2col_ms = median(&mut samples) * 1000.0;

    // The hybrid's GEMM stage in isolation: the crate's tiled MatMul on the
    // already-built column matrix, `[M, K] x [K, N]`. This is the floor the
    // hybrid path could ever reach — remove *all* of its host im2col and
    // staging and this is what is left — and therefore the number that says
    // whether the direct kernel's remaining time is addressing overhead or
    // simply the same multiply.
    let mut gemm = None;
    for _ in 0..WARMUP {
        if oxionnx_gpu::gpu_matmul_tiled(ctx, &weight.data, &col, m, k, n_gemm).is_none() {
            break;
        }
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let out = oxionnx_gpu::gpu_matmul_tiled(ctx, &weight.data, &col, m, k, n_gemm);
        let elapsed = t.elapsed().as_secs_f64();
        if out.is_none() {
            samples.clear();
            break;
        }
        samples.push(elapsed);
    }
    if !samples.is_empty() {
        gemm = Some(median(&mut samples) * 1000.0);
    }
    drop(col);

    report("direct (this kernel)", direct, gflop, None);
    report("hybrid (im2col+GEMM)", hybrid, gflop, direct);
    report("rayon CPU", Some(cpu), gflop, direct);
    report("[ref] tiled GEMM only", gemm, gflop, direct);

    println!(
        "    staged/call: direct {:.1} MB (input {:.1} + weight {:.1})   \
         hybrid {:.1} MB (col {:.1} + weight {:.1})   ratio {:.2}x",
        mib((in_len + weight_len) * 4),
        mib(in_len * 4),
        mib(weight_len * 4),
        mib((col_len + weight_len) * 4),
        mib(col_len * 4),
        mib(weight_len * 4),
        (col_len + weight_len) as f64 / (in_len + weight_len) as f64,
    );
    println!(
        "    host im2col {im2col_ms:>8.3} ms   staging memcpy: input {:>7.3} ms  \
         weight {:>7.3} ms  col {:>7.3} ms",
        staging_ms(ctx, in_len),
        staging_ms(ctx, weight_len),
        staging_ms(ctx, col_len),
    );
}

fn main() {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("ERROR: GpuContext::try_new() returned None — no wgpu adapter available.");
        std::process::exit(1);
    };

    println!("[c3] oxionnx-gpu Conv2D: direct implicit-GEMM vs hybrid im2col vs rayon CPU");
    println!(
        "warmup={WARMUP} gpu_iters={ITERS} cpu_iters={CPU_ITERS}  \
         (end-to-end: upload + dispatch + blocking read-back)"
    );
    println!(
        "host parallelism: {}",
        std::thread::available_parallelism().map_or_else(|_| "?".into(), |p| p.to_string())
    );
    println!(
        "device limits: max_storage_binding {:.0} MiB, max_buffer {:.0} MiB, max_wg/dim {}",
        mib(ctx.limits.max_storage_buffer_binding_size as usize),
        mib(ctx.limits.max_buffer_size as usize),
        ctx.limits.max_workgroups_per_dimension,
    );

    // The kernel caches its compiled pipeline per device; the first call pays
    // the WGSL compile. Measure it explicitly rather than letting it hide in
    // the first sample of the first case.
    let probe_in = Tensor::new(fill(16 * 8 * 8, 3), vec![1, 16, 8, 8]);
    let probe_w = Tensor::new(fill(64 * 16 * 9, 5), vec![64, 16, 3, 3]);
    let t = Instant::now();
    let _ = gpu_conv2d_implicit(
        &ctx,
        &probe_in,
        &probe_w,
        None,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
    );
    let cold = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let _ = gpu_conv2d_implicit(
        &ctx,
        &probe_in,
        &probe_w,
        None,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
    );
    let warm = t.elapsed().as_secs_f64() * 1000.0;
    println!("pipeline: first call {cold:.3} ms, cached call {warm:.3} ms (WGSL compile is the difference)");

    for case in CASES {
        run_case(&ctx, case);
    }

    println!("\nDone.");
}
