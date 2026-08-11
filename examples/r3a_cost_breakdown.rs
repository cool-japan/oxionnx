//! [r3a] Where does a GPU node's wall clock actually go?
//!
//! Decomposes per-dispatch cost into (a) fixed per-call overhead, (b) WGSL
//! pipeline construction, (c) host<->device transfer, before deciding whether
//! residency or pipeline caching is the right lever.
//!
//! The trick is comparing two kernels that differ in exactly one property:
//!
//! * `gpu_relu_async` — pipeline is **pre-built once** in `GpuContext`.
//! * `gpu_pad_async`  — pipeline is **rebuilt on every call**
//!   (`kernel_support`'s documented convention).
//!
//! At the same element count both move the same bytes and do trivial
//! arithmetic, so `pad - relu` is the per-call pipeline construction cost.
//!
//! ```text
//! cargo run --release --features gpu --example r3a_cost_breakdown
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
    use oxionnx::gpu::{
        gpu_conv2d_fused, gpu_instance_norm, gpu_pad, gpu_relu, ConvActivation, GpuContext, PadMode,
    };
    use oxionnx::tensor::Tensor;
    use std::time::Instant;

    const ITERS: usize = 12;

    /// Median of `ITERS` timed calls, in milliseconds, after one warm-up.
    fn median_ms(mut f: impl FnMut() -> bool) -> f64 {
        if !f() {
            return f64::NAN;
        }
        let mut samples: Vec<f64> = (0..ITERS)
            .map(|_| {
                let t = Instant::now();
                let ok = f();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if ok {
                    ms
                } else {
                    f64::NAN
                }
            })
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        samples[samples.len() / 2]
    }

    pub fn run() {
        let Some(ctx) = GpuContext::try_new() else {
            println!("no GPU adapter; nothing to measure");
            return;
        };

        println!("== per-call fixed cost: cached pipeline (relu) vs per-call pipeline (pad) ==");
        println!(
            "{:>12}  {:>12}  {:>12}  {:>12}",
            "elements", "relu ms", "pad ms", "pad-relu ms"
        );
        // Pad with 0 on every side: identical element count in and out, so the
        // only difference from relu is the pipeline construction and the
        // slightly richer index math.
        for &(n, c, h, w) in &[
            (1usize, 1usize, 8usize, 8usize),
            (1, 64, 32, 32),
            (1, 1024, 32, 32),
            (1, 512, 64, 64),
        ] {
            let len = n * c * h * w;
            let data = vec![0.5f32; len];
            let shape = [n, c, h, w];
            let relu_ms = median_ms(|| gpu_relu(&ctx, &data).is_some());
            let pad_ms = median_ms(|| {
                gpu_pad(&ctx, &data, &shape, 0, 0, 0, 0, PadMode::Reflect, 0.0).is_some()
            });
            println!(
                "{len:>12}  {relu_ms:>12.3}  {pad_ms:>12.3}  {:>12.3}",
                pad_ms - relu_ms
            );
        }

        println!("\n== transfer scaling (relu: upload n + trivial compute + readback n) ==");
        println!(
            "{:>12}  {:>10}  {:>12}  {:>14}",
            "elements", "MiB moved", "ms", "GiB/s (2n)"
        );
        for &len in &[1024usize, 262_144, 1_048_576, 4_194_304, 16_777_216] {
            let data = vec![0.5f32; len];
            let ms = median_ms(|| gpu_relu(&ctx, &data).is_some());
            let moved = (len as f64 * 4.0 * 2.0) / (1024.0 * 1024.0);
            println!(
                "{len:>12}  {:>10.1}  {ms:>12.3}  {:>14.2}",
                moved / 2.0,
                moved / 1024.0 / (ms / 1e3)
            );
        }

        println!("\n== InSwapper's real shapes ==");
        // The bottleneck conv: [1,1024,32,32] * [1024,1024,3,3], x12 per frame.
        let input = Tensor::new(vec![0.25f32; 1024 * 34 * 34], vec![1, 1024, 34, 34]);
        let weight = Tensor::new(vec![0.01f32; 1024 * 1024 * 3 * 3], vec![1024, 1024, 3, 3]);
        let bias = Tensor::new(vec![0.0f32; 1024], vec![1024]);
        let conv_ms = median_ms(|| {
            gpu_conv2d_fused(
                &ctx,
                &input,
                &weight,
                Some(&bias),
                [1, 1],
                [0, 0, 0, 0],
                [1, 1],
                1,
                ConvActivation::None,
            )
            .is_some()
        });
        let weight_mib = (1024.0 * 1024.0 * 9.0 * 4.0) / (1024.0 * 1024.0);
        println!(
            "conv [1,1024,34,34] x [1024,1024,3,3]   {conv_ms:>8.3} ms   (weight alone = {weight_mib:.1} MiB)"
        );

        // How long does just *uploading* that weight take? Measured as a relu
        // over the same element count, minus the readback of the same size —
        // i.e. one direction is ~half the relu time at that size.
        let weight_len = 1024 * 1024 * 9;
        let wdata = vec![0.01f32; weight_len];
        let relu_weight_ms = median_ms(|| gpu_relu(&ctx, &wdata).is_some());
        println!(
            "relu over the same element count       {relu_weight_ms:>8.3} ms   (= upload + readback of {weight_mib:.1} MiB)"
        );

        // Pad at InSwapper's real pad shape, x14 per frame.
        let pad_in = vec![0.5f32; 1024 * 32 * 32];
        let pad_ms = median_ms(|| {
            gpu_pad(
                &ctx,
                &pad_in,
                &[1, 1024, 32, 32],
                1,
                1,
                1,
                1,
                PadMode::Reflect,
                0.0,
            )
            .is_some()
        });
        println!("pad  [1,1024,32,32] +1 reflect         {pad_ms:>8.3} ms   (x14 per frame)");

        let in_data = vec![0.5f32; 1024 * 32 * 32];
        let in_ms =
            median_ms(|| gpu_instance_norm(&ctx, &in_data, &[1, 1024, 32, 32], 1e-8).is_some());
        println!("instance_norm [1,1024,32,32]           {in_ms:>8.3} ms   (x12 per frame)");
    }
}
