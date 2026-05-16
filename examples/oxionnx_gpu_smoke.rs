//! GPU vs CPU micro-benchmark for oxionnx.
//!
//! Compares `oxionnx::gpu::gpu_matmul` and `oxionnx::gpu::gpu_conv2d` against
//! the CPU baselines (`matrixmultiply::sgemm` for matmul,
//! `oxionnx_ops::conv::conv2d` for conv2d). Sizes are picked to match
//! face-swap workloads (InSwapper / ArcFace / SCRFD layers).
//!
//! Build / run:
//!     cargo run --release --features gpu --example oxionnx_gpu_smoke
//!
//! Without the `gpu` feature this example prints a message and exits.

#[cfg(feature = "gpu")]
mod gpu_impl {
    use std::time::Instant;

    use oxionnx::gpu::{gpu_conv2d, gpu_matmul, GpuContext};
    use oxionnx::Tensor;

    const WARMUP: usize = 5;
    const ITERS: usize = 20;

    // --------------------------------------------------------------------------
    // CPU baselines
    // --------------------------------------------------------------------------

    pub fn cpu_matmul_mm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        // Row-major: rsa=k, csa=1; rsb=n, csb=1; rsc=n, csc=1
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        c
    }

    pub fn cpu_conv2d_baseline(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        oxionnx_ops::conv::conv2d(input, weight, bias, [1, 1], [1, 1, 1, 1], [1, 1], 1)
    }

    // --------------------------------------------------------------------------
    // Timing helpers
    // --------------------------------------------------------------------------

    pub struct Stats {
        pub min_ms: f64,
        pub median_ms: f64,
        pub mean_ms: f64,
    }

    pub fn stats(samples: &[f64]) -> Stats {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ms = sorted[sorted.len() / 2];
        let min_ms = sorted[0];
        let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
        Stats {
            min_ms,
            median_ms,
            mean_ms,
        }
    }

    pub fn time_loop<F>(label: &str, mut f: F) -> Stats
    where
        F: FnMut(),
    {
        for _ in 0..WARMUP {
            f();
        }
        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            f();
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let s = stats(&samples);
        println!(
            "  {:<28} min={:>9.3} ms  median={:>9.3} ms  mean={:>9.3} ms",
            label, s.min_ms, s.median_ms, s.mean_ms
        );
        s
    }

    // --------------------------------------------------------------------------
    // MatMul cases
    // --------------------------------------------------------------------------

    pub fn run_matmul_case(ctx: &GpuContext, label: &str, m: usize, k: usize, n: usize) {
        let flops = m * k * n;
        println!(
            "\n[MatMul] {label}  M={m} K={k} N={n}  ({:.1}M FLOPs)",
            flops as f64 / 1e6
        );

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 13) as f32) * 0.015 - 0.09)
            .collect();

        // Sanity check (one shot)
        let cpu_once = cpu_matmul_mm(&a, &b, m, k, n);
        let gpu_once = gpu_matmul(ctx, &a, &b, m, k, n);
        match &gpu_once {
            Some(g) => {
                let mut max_err = 0.0_f32;
                for (gv, cv) in g.iter().zip(cpu_once.iter()) {
                    let e = (gv - cv).abs();
                    if e > max_err {
                        max_err = e;
                    }
                }
                println!("  [verify] max|gpu-cpu|={max_err:.4}");
            }
            None => {
                println!(
                    "  [skip] gpu_matmul returned None (size below 10M FLOP threshold or no adapter)"
                );
                return;
            }
        }

        let cpu = time_loop("CPU matrixmultiply::sgemm", || {
            let _ = cpu_matmul_mm(&a, &b, m, k, n);
        });
        let gpu = time_loop("GPU gpu_matmul (incl readback)", || {
            let _ = gpu_matmul(ctx, &a, &b, m, k, n);
        });

        let speedup = cpu.median_ms / gpu.median_ms;
        println!("  >>> speedup (CPU/GPU median): {speedup:.2}x");
    }

    // --------------------------------------------------------------------------
    // Conv2D cases
    // --------------------------------------------------------------------------

    pub fn run_conv_case(ctx: &GpuContext, label: &str, n: usize, c: usize, h: usize, w: usize) {
        println!("\n[Conv2D] {label}  in=[{n},{c},{h},{w}]  weight=[{c},{c},3,3]  pad=1, stride=1");
        let input_data: Vec<f32> = (0..n * c * h * w)
            .map(|i| ((i % 19) as f32) * 0.001)
            .collect();
        let weight_data: Vec<f32> = (0..c * c * 3 * 3)
            .map(|i| ((i % 11) as f32) * 0.01)
            .collect();
        let bias_data: Vec<f32> = vec![0.0; c];

        let input = Tensor::new(input_data, vec![n, c, h, w]);
        let weight = Tensor::new(weight_data, vec![c, c, 3, 3]);
        let bias = Tensor::new(bias_data, vec![c]);

        // Probe to verify GPU path is engaged
        let gpu_once = gpu_conv2d(
            ctx,
            &input,
            &weight,
            Some(&bias),
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            1,
        );
        if gpu_once.is_none() {
            println!("  [skip] gpu_conv2d returned None (im2col GEMM below threshold)");
            return;
        }

        let cpu = time_loop("CPU oxionnx_ops::conv::conv2d", || {
            let _ = cpu_conv2d_baseline(&input, &weight, Some(&bias));
        });
        let gpu = time_loop("GPU gpu_conv2d (im2col+GPU GEMM)", || {
            let _ = gpu_conv2d(
                ctx,
                &input,
                &weight,
                Some(&bias),
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                1,
            );
        });

        let speedup = cpu.median_ms / gpu.median_ms;
        println!("  >>> speedup (CPU/GPU median): {speedup:.2}x");
    }

    // --------------------------------------------------------------------------
    // Entry
    // --------------------------------------------------------------------------

    pub fn run() {
        println!("oxionnx GPU vs CPU smoke benchmark");
        println!("warmup={WARMUP} iters={ITERS}");

        let ctx = match GpuContext::try_new() {
            Some(c) => c,
            None => {
                eprintln!(
                    "ERROR: GpuContext::try_new() returned None — no wgpu adapter available."
                );
                std::process::exit(1);
            }
        };
        println!("GPU context acquired (wgpu adapter ready).");

        // ---- MatMul cases ----
        // Square 1024 (tiled kernel, ~1 GFLOP) — well above threshold.
        run_matmul_case(&ctx, "square 1024 (tiled)", 1024, 1024, 1024);
        // ArcFace fc-like: batch=1, but M<32 forces basic kernel; ~12.8M FLOPs (borderline).
        run_matmul_case(
            &ctx,
            "ArcFace fc 1x25088x512 (basic, borderline)",
            1,
            25088,
            512,
        );
        // Larger fc: still M=1 so basic kernel; ~100M FLOPs.
        // (1x25088x4096 would need >256MB buffer for B; wgpu rejects it. Use 4096x4096
        // which fits and exercises the tiled kernel anyway.)
        run_matmul_case(&ctx, "Mid fc 32x4096x4096 (tiled)", 32, 4096, 4096);
        // Square 512 — 134M FLOPs, above threshold, tiled.
        run_matmul_case(&ctx, "square 512 (tiled)", 512, 512, 512);
        // Square 2048 — 8.6 GFLOPs, well above threshold, tiled — should show GPU win.
        run_matmul_case(&ctx, "square 2048 (tiled)", 2048, 2048, 2048);

        // ---- Conv2D cases ----
        // InSwapper-like middle layer: [1,256,16,16] * [256,256,3,3]
        // im2col GEMM = [256, 256*9=2304, 16*16=256] ≈ 150M FLOPs.
        run_conv_case(&ctx, "InSwapper mid (256ch, 16x16)", 1, 256, 16, 16);
        // Conv 64ch / 56x56 — same shape as the existing CPU bench
        run_conv_case(&ctx, "ResNet stage1 (64ch, 56x56)", 1, 64, 56, 56);

        println!("\nDone.");
    }
}

#[cfg(feature = "gpu")]
fn main() {
    gpu_impl::run();
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("This example requires the 'gpu' feature. Rebuild with --features=gpu");
}
