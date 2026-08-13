//! Where the element-wise, normalization, reduction, transpose and softmax
//! kernels stop beating their CPU counterparts — measured, with every operand
//! transferring.
//!
//! The harness behind the second table in [`oxionnx_gpu::context::tuning`], and
//! the reason four of those kernels' native floors are `usize::MAX` rather than
//! a large number: on an RTX A4000 they lose at *every* size tried, across three
//! orders of magnitude, and the loss is structural rather than a threshold that
//! wants nudging (both sides are linear in `n`; the GPU's constant is an order
//! of magnitude worse because it moves the same bytes over a 4–6 GB/s bus that
//! the CPU reads once from many-channel DDR).
//!
//! ```text
//! cargo run --release -p oxionnx-gpu --example kernel_crossover
//! ```
//!
//! # What this does *not* measure
//!
//! The resident regime. Every dispatch here uploads its operands and reads its
//! result back, because that is the regime the size floors govern — an operand
//! already in a device buffer skips them entirely
//! (`context::activation::skips_size_threshold`). A kernel that loses every row
//! of this table can still be the right choice when its input is already on the
//! device and its consumer is another GPU node; that is what activation
//! residency is for, and it is a different measurement.
//!
//! Same clock-ramp methodology as `gemm_shape_crossover.rs`: uninterrupted GPU
//! bursts, CPU phase separate, `min` reported.

use std::time::Instant;

use oxionnx_core::Tensor;
use oxionnx_gpu::{
    gpu_add, gpu_batch_norm, gpu_layer_norm, gpu_reduce_sum, gpu_relu, gpu_softmax, gpu_transpose,
    GpuContext, GpuTuning,
};

const SPINUP: usize = 40;
const ITERS: usize = 40;
const ROUNDS: usize = 3;

fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            ((x % 23) as f32) * 0.037 - 0.4
        })
        .collect()
}

fn best(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[0] * 1e3
}

/// Burst a closure: `SPINUP` unmeasured calls to bring the clocks up, then
/// `ITERS` measured ones, appended to `out`.
fn burst<T>(f: impl Fn() -> T, out: &mut Vec<f64>) {
    for _ in 0..SPINUP {
        std::hint::black_box(f());
    }
    for _ in 0..ITERS {
        let t = Instant::now();
        std::hint::black_box(f());
        out.push(t.elapsed().as_secs_f64());
    }
}

fn row(label: &str, size: usize, cpu: f64, gpu: f64) {
    println!(
        "{label:<22} {size:>12} | {cpu:>9.3} {gpu:>9.3} {:>7.2}",
        gpu / cpu
    );
}

fn main() {
    let ctx = match GpuContext::try_new_diagnosed() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("no GPU: {err}");
            return;
        }
    };
    let installed = *ctx.tuning();
    // Measure the kernels, not the policy — otherwise every row below the
    // installed floor would report "declined" and the sweep could never
    // re-derive that floor.
    let mut ctx = ctx;
    ctx.set_tuning(GpuTuning::PARITY);
    println!("adapter class: {}", installed.class.as_str());
    println!("installed floors: {installed:?}");
    println!("ratios are GPU/CPU: > 1.00 means the GPU LOST\n");
    println!(
        "{:<22} {:>12} | {:>9} {:>9} {:>7}",
        "kernel", "elements", "cpu ms", "gpu ms", "g/c"
    );

    for n in [100_000usize, 250_000, 1_000_000, 4_000_000, 16_000_000] {
        let data = fill(n, 2_654_435_761);
        let t = Tensor::new(data.clone(), vec![n]);
        let (mut c, mut g) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            burst(|| gpu_relu(&ctx, &data), &mut g);
            burst(|| oxionnx_ops::nn::relu(&t), &mut c);
        }
        row("gpu_relu (unary EW)", n, best(&mut c), best(&mut g));
    }
    println!();

    for n in [100_000usize, 250_000, 1_000_000, 4_000_000, 16_000_000] {
        let a = fill(n, 2_654_435_761);
        let b = fill(n, 40_503);
        let at = Tensor::new(a.clone(), vec![n]);
        let bt = Tensor::new(b.clone(), vec![n]);
        let (mut c, mut g) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            burst(|| gpu_add(&ctx, &a, &b), &mut g);
            burst(|| oxionnx_ops::math::add(&at, &bt), &mut c);
        }
        row("gpu_add (binary EW)", n, best(&mut c), best(&mut g));
    }
    println!();

    // Reductions are gated on the *output* count, so vary that.
    for rows in [50_000usize, 500_000, 2_000_000, 8_000_000] {
        let shape = vec![rows, 4];
        let data = fill(rows * 4, 2_654_435_761);
        let t = Tensor::new(data.clone(), shape.clone());
        let (mut c, mut g) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            burst(|| gpu_reduce_sum(&ctx, &data, 1, &shape), &mut g);
            burst(|| oxionnx_ops::math::reduce_sum(&t, &[1], false), &mut c);
        }
        row("gpu_reduce_sum (out)", rows, best(&mut c), best(&mut g));
    }
    println!();

    for rows in [128usize, 512, 2048, 8192, 32768] {
        let cols = 512usize;
        let shape = vec![rows, cols];
        let data = fill(rows * cols, 2_654_435_761);
        let scale = fill(cols, 7);
        let bias = fill(cols, 11);
        let (t, st, bt) = (
            Tensor::new(data.clone(), shape.clone()),
            Tensor::new(scale.clone(), vec![cols]),
            Tensor::new(bias.clone(), vec![cols]),
        );
        let (mut c, mut g) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            burst(
                || gpu_layer_norm(&ctx, &data, &shape, &scale, &bias, 1e-5),
                &mut g,
            );
            burst(
                || oxionnx_ops::nn::layer_norm(&t, &st, Some(&bt), 1e-5, -1),
                &mut c,
            );
        }
        row("gpu_layer_norm", rows * cols, best(&mut c), best(&mut g));
    }
    println!();

    for h in [32usize, 64, 128, 256] {
        let shape = vec![2usize, 32, h, h];
        let total = 2 * 32 * h * h;
        let data = fill(total, 2_654_435_761);
        let (p, q, mean) = (fill(32, 7), fill(32, 11), fill(32, 13));
        let var: Vec<f32> = fill(32, 17).iter().map(|v| v.abs() + 1.0).collect();
        let t = Tensor::new(data.clone(), shape.clone());
        let (pt, qt, mt, vt) = (
            Tensor::new(p.clone(), vec![32]),
            Tensor::new(q.clone(), vec![32]),
            Tensor::new(mean.clone(), vec![32]),
            Tensor::new(var.clone(), vec![32]),
        );
        let (mut c, mut g) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            burst(
                || gpu_batch_norm(&ctx, &data, &shape, &p, &q, &mean, &var, 1e-5),
                &mut g,
            );
            burst(
                || oxionnx_ops::nn::batch_norm(&t, &pt, &qt, &mt, &vt, 1e-5),
                &mut c,
            );
        }
        row("gpu_batch_norm", total, best(&mut c), best(&mut g));
    }
    println!();

    for a in [256usize, 512, 1024, 2048, 4096] {
        let shape = vec![a, a];
        let data = fill(a * a, 2_654_435_761);
        let t = Tensor::new(data.clone(), shape.clone());
        let (mut c, mut g) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            burst(|| gpu_transpose(&ctx, &data, &shape, &[1, 0]), &mut g);
            burst(|| oxionnx_ops::shape::transpose(&t, &[1, 0]), &mut c);
        }
        row("gpu_transpose", a * a, best(&mut c), best(&mut g));
    }
    println!();

    // Softmax is gated on the row length *and* the total, because a long row in
    // a short batch clears the first and still loses; both are varied here.
    for (rows, cols) in [
        (64usize, 1024usize),
        (256, 1024),
        (1024, 1024),
        (4096, 1024),
        (16384, 1024),
    ] {
        let shape = vec![rows, cols];
        let data = fill(rows * cols, 2_654_435_761);
        let t = Tensor::new(data.clone(), shape.clone());
        let (mut c, mut g) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            burst(|| gpu_softmax(&ctx, &data, &shape), &mut g);
            burst(|| oxionnx_ops::nn::softmax(&t, -1), &mut c);
        }
        row(
            &format!("gpu_softmax {rows}x{cols}"),
            rows * cols,
            best(&mut c),
            best(&mut g),
        );
    }
}
