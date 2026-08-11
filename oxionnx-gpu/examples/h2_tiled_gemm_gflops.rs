//! GFLOP/s benchmark for `oxionnx-gpu`'s tiled MatMul kernel (Agent H2).
//!
//! Standalone from `examples/oxionnx_gpu_smoke.rs` at the workspace root (a
//! parallel wave owns that file this cycle) -- measures the *tiled* kernel
//! specifically, at the two shapes the H2 task calls out (1024^3, 2048^3),
//! through the public `gpu_matmul_tiled` entry point so the number reflects
//! the same end-to-end cost (buffer upload + dispatch + blocking read-back)
//! production code pays on every call.
//!
//! Run:
//!     cargo run -p oxionnx-gpu --release --example h2_tiled_gemm_gflops
//!
//! (Or without `--release`; this workspace's `[profile.dev]` already sets
//! `opt-level = 3`, so a plain `cargo run -p oxionnx-gpu --example ...` gives
//! a meaningful, if not identical, number without a full LTO release build.)

use std::time::Instant;

use oxionnx_gpu::{gpu_matmul_tiled, GpuContext};

const WARMUP: usize = 3;
// median() below indexes this slice at len/2, which panics on an empty
// slice -- guard the const itself (rather than adding a runtime check to
// every call site) so ITERS can never be lowered to 0 without a compile
// error.
const _ITERS_MUST_BE_NONZERO: () = assert!(ITERS > 0);
const ITERS: usize = 10;

/// Deterministic, signed, non-monotonic fill -- see `h2_tiled_matmul_gemm.rs`
/// for why this shape of data is preferred over a plain `i % small` ramp.
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

fn bench_case(ctx: &GpuContext, m: usize, k: usize, n: usize) {
    let a = fill(m * k, 2_654_435_761);
    let b = fill(k * n, 40_503);

    for _ in 0..WARMUP {
        if gpu_matmul_tiled(ctx, &a, &b, m, k, n).is_none() {
            println!("  {m}x{k}x{n}: [skip] gpu_matmul_tiled returned None (no adapter, or below its size gates)");
            return;
        }
    }

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let out = gpu_matmul_tiled(ctx, &a, &b, m, k, n);
        let elapsed = t.elapsed().as_secs_f64();
        if out.is_none() {
            println!("  {m}x{k}x{n}: [skip] gpu_matmul_tiled returned None mid-run");
            return;
        }
        samples.push(elapsed);
    }

    let med = median(&mut samples);
    // multiply + add per MAC
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let gflops = flops / med / 1e9;
    println!(
        "  {m}x{k}x{n}: median={:>9.3} ms   {:>8.2} GFLOP/s   (min={:.3} ms, max={:.3} ms)",
        med * 1000.0,
        gflops,
        samples.iter().cloned().fold(f64::INFINITY, f64::min) * 1000.0,
        samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1000.0,
    );
}

fn main() {
    let Some(ctx) = GpuContext::try_new() else {
        eprintln!("ERROR: GpuContext::try_new() returned None -- no wgpu adapter available.");
        std::process::exit(1);
    };
    println!("oxionnx-gpu tiled MatMul GFLOP/s (Agent H2)");
    println!(
        "warmup={WARMUP} iters={ITERS} (end-to-end: upload + dispatch + blocking read-back)\n"
    );

    bench_case(&ctx, 1024, 1024, 1024);
    bench_case(&ctx, 2048, 2048, 2048);

    println!("\nDone.");
}
