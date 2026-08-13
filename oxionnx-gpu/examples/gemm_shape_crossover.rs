//! Where the GPU stops beating the CPU for a GEMM, measured rather than assumed.
//!
//! This is the harness behind the table in
//! [`oxionnx_gpu::context::tuning`] — the one that produced
//! `GpuTuning::gemm_min_mac` and `GpuTuning::gemm_min_intensity`. Re-run it on
//! any new target before changing either number.
//!
//! ```text
//! cargo run --release -p oxionnx-gpu --example gemm_shape_crossover
//! ```
//!
//! # Three columns, because there are three different questions
//!
//! * `gpu_mm`  — `gpu_matmul`: both operands upload, the result reads back.
//! * `gpu_gem` — `gpu_gemm_nt_resident_async` with a warm weight cache: `B`
//!   does **not** cross the bus. This is the regime a video pipeline runs in,
//!   where the same model weights are dispatched hundreds of times.
//! * the CPU reference for each, `oxionnx_ops::math::{matmul, gemm}`
//!   (rayon + `matrixmultiply`).
//!
//! The gap between the two GPU columns at small `m` *is* the case for
//! residency, and it is why `GemmWeightTraffic` exists.
//!
//! # Methodology, and why it is not a plain loop
//!
//! The reference box's RTX A4000 idles at 210 MHz of a 2100 MHz boost and
//! refuses `nvidia-smi -lgc` (a virtualized GPU: "the current user does not
//! have permission to change clocks"). An interleaved CPU/GPU measurement on
//! such a part measures the clock ramp, not the kernel — an early version of
//! this sweep reported the same shape as both 3.3 ms and 10.6 ms depending on
//! how much CPU work ran between dispatches. So each shape gets an
//! uninterrupted GPU burst to bring the clocks up, the CPU phase runs
//! separately, and `min` (fully-boosted) is reported next to the median.
//!
//! Every entry point is called through its public form, so the numbers include
//! everything production pays: buffer allocation, upload, dispatch, fence and
//! read-back.

use std::time::Instant;

use oxionnx_core::Tensor;
use oxionnx_gpu::{gpu_matmul, GpuContext, GpuTuning, WeightKeys};

/// Dispatches that only warm the clocks; not measured.
const SPINUP: usize = 40;
/// Measured dispatches per round.
const ITERS: usize = 40;
/// Rounds, so a transient (another process on the GPU, a page fault storm)
/// shows up as a spread between `min` and the median rather than as the answer.
const ROUNDS: usize = 3;

/// Deterministic, signed, non-monotonic fill — see `h2_tiled_matmul_gemm.rs`
/// for why this beats an `i % small` ramp for numerical shapes.
fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            ((x % 23) as f32) * 0.037 - 0.4
        })
        .collect()
}

/// `(min, median)` in milliseconds.
fn stats(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (samples[0] * 1e3, samples[samples.len() / 2] * 1e3)
}

fn main() {
    let ctx = match GpuContext::try_new_diagnosed() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("no GPU: {err}");
            return;
        }
    };
    // The sweep must not be filtered by the very gate it exists to calibrate.
    let installed = *ctx.tuning();
    let mut ctx = ctx;
    ctx.set_tuning(GpuTuning::PARITY);
    println!("adapter class: {}", installed.class.as_str());
    println!(
        "installed gate: mac >= {} (cached {}), intensity >= {}",
        installed.gemm_min_mac, installed.gemm_min_mac_cached, installed.gemm_min_intensity
    );
    println!("ratios are GPU/CPU: > 1.00 means the GPU LOST\n");
    println!(
        "{:>6} {:>6} {:>6} {:>13} {:>5} | {:>8} {:>8} {:>6} | {:>8} {:>8} {:>6} | {:>6}",
        "m", "k", "n", "m*k*n", "I", "cpu_mm", "gpu_mm", "g/c", "cpu_gem", "gpu_gem", "g/c", "gate"
    );

    let cases: [(usize, usize); 6] = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        // ArcFace's embedding head.
        (25_088, 512),
        // InSwapper's AdaIN style heads.
        (512, 2048),
    ];
    let ms = [1usize, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256];

    for (k, n) in cases {
        for m in ms {
            let Some(mac) = GpuTuning::gemm_mac(m, k, n) else {
                continue;
            };
            // Keep the sweep's own wall clock bounded; the interesting region
            // is entirely below this.
            if mac > 5_000_000_000 {
                continue;
            }
            let intensity = GpuTuning::gemm_intensity(m, k, n).unwrap_or(0);

            let a = fill(m * k, 2_654_435_761);
            let b = fill(k * n, 40_503);
            let at = Tensor::new(a.clone(), vec![m, k]);
            let bt = Tensor::new(b.clone(), vec![k, n]);
            // `gpu_gemm_nt` reads B as [N, K] — the `nn.Linear` layout.
            let btn = Tensor::new(b.clone(), vec![n, k]);
            let key = format!("sweep_{k}_{n}");
            let keys = WeightKeys::new(Some(key.as_str()), None);

            let (mut c_mm, mut g_mm, mut c_ge, mut g_ge) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut declined = false;

            for _ in 0..ROUNDS {
                for _ in 0..SPINUP {
                    declined |= gpu_matmul(&ctx, &a, &b, m, k, n).is_none();
                }
                for _ in 0..ITERS {
                    let t = Instant::now();
                    let out = gpu_matmul(&ctx, &a, &b, m, k, n);
                    g_mm.push(t.elapsed().as_secs_f64());
                    declined |= out.is_none();
                    std::hint::black_box(out);
                }
                for _ in 0..SPINUP {
                    declined |= pollster::block_on(oxionnx_gpu::gpu_gemm_nt_resident_async(
                        &ctx, &a, m, k, &b, n, None, 1.0, 1.0, keys,
                    ))
                    .is_none();
                }
                for _ in 0..ITERS {
                    let t = Instant::now();
                    let out = pollster::block_on(oxionnx_gpu::gpu_gemm_nt_resident_async(
                        &ctx, &a, m, k, &b, n, None, 1.0, 1.0, keys,
                    ));
                    g_ge.push(t.elapsed().as_secs_f64());
                    declined |= out.is_none();
                    std::hint::black_box(out);
                }
                for _ in 0..8 {
                    std::hint::black_box(oxionnx_ops::math::matmul(&at, &bt).expect("cpu matmul"));
                }
                for _ in 0..20 {
                    let t = Instant::now();
                    std::hint::black_box(oxionnx_ops::math::matmul(&at, &bt).expect("cpu matmul"));
                    c_mm.push(t.elapsed().as_secs_f64());
                }
                for _ in 0..8 {
                    std::hint::black_box(
                        oxionnx_ops::math::gemm(&at, &btn, None, 1.0, 1.0, false, true)
                            .expect("cpu gemm"),
                    );
                }
                for _ in 0..20 {
                    let t = Instant::now();
                    std::hint::black_box(
                        oxionnx_ops::math::gemm(&at, &btn, None, 1.0, 1.0, false, true)
                            .expect("cpu gemm"),
                    );
                    c_ge.push(t.elapsed().as_secs_f64());
                }
            }

            let (cm, _) = stats(&mut c_mm);
            let (cg, _) = stats(&mut c_ge);
            let (gm, _) = stats(&mut g_mm);
            let (gg, _) = stats(&mut g_ge);
            // What the *installed* gate would have decided for this shape.
            let gate = match (
                installed.gemm_admits(m, k, n, oxionnx_gpu::GemmWeightTraffic::PerDispatch),
                installed.gemm_admits(m, k, n, oxionnx_gpu::GemmWeightTraffic::Cached),
            ) {
                (true, true) => "both",
                (false, true) => "cache",
                (true, false) => "up!?",
                (false, false) => "none",
            };
            println!(
                "{m:>6} {k:>6} {n:>6} {mac:>13} {intensity:>5} | {cm:>8.3} {gm:>8.3} {:>6.2} | \
                 {cg:>8.3} {gg:>8.3} {:>6.2} | {gate:>6}{}",
                gm / cm,
                gg / cg,
                if declined {
                    "  [kernel declined — result not meaningful]"
                } else {
                    ""
                }
            );
        }
        println!();
    }
}
