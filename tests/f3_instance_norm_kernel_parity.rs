//! CPU ⟷ GPU parity for the fused `OxiInstanceNorm` kernel.
//!
//! `oxionnx-gpu` cannot depend on `oxionnx-ops`, so the WGSL kernel's own unit
//! tests compare against a small reimplementation of the CPU arithmetic that
//! lives beside the shader. That is a useful independent check of the shader,
//! but it is not a check that the two *shipped* kernels agree: change
//! `oxi_instance_norm`'s variance formula or its default epsilon and those
//! tests keep passing against the stale copy.
//!
//! This file is the real comparison. The root crate depends on `oxionnx-ops`
//! unconditionally and on `oxionnx-gpu` under the `gpu` feature, so it is the
//! first place both kernels are reachable at once.
//!
//! Requires `--features gpu`; skips when no wgpu adapter is reachable.

#![cfg(feature = "gpu")]

use oxionnx::gpu::shaders::gpu_instance_norm;
use oxionnx::gpu::GpuContext;
use oxionnx::Tensor;
use oxionnx_ops::registry::oxi_instance_norm::oxi_instance_norm;

/// Deterministic input with a non-trivial per-plane mean and spread, so a
/// kernel that normalised the wrong region would visibly disagree.
fn ramp(shape: &[usize], scale: f32, offset: f32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| offset + scale * ((i % 29) as f32 - 14.0) + 0.5 * (i as f32).sin())
        .collect()
}

/// `atol + rtol * |expected|`: the normalised output has zero mean per plane,
/// so many elements sit near zero where a pure relative bound is meaningless.
fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= atol + rtol * e.abs(),
            "{label}: element {i}: gpu {a} vs cpu {e}"
        );
    }
}

/// The shipped CPU kernel and the shipped GPU kernel must agree, across the
/// 256-thread workgroup boundary the shader's strided reduction pivots on.
///
/// The GPU kernel carries no minimum-size gate (it follows `kernel_support`'s
/// convention, not `normalization.rs`'s thresholds), which is what lets these
/// small shapes actually dispatch rather than returning a vacuous `None`.
#[test]
fn cpu_and_gpu_kernels_agree_across_the_workgroup_boundary() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    if ctx.is_degraded() {
        return;
    }

    for (shape, eps) in [
        (vec![1usize, 3, 5, 7], 1e-8f32), // 35 per plane: far below WG_SIZE
        (vec![1usize, 2, 16, 16], 1e-8),  // exactly WG_SIZE
        (vec![1usize, 4, 20, 37], 1e-6),  // 740: not a multiple of WG_SIZE
        (vec![2usize, 8, 32, 32], 1e-5),  // 1024 per plane, 16 planes
        (vec![2usize, 4, 50], 1e-5),      // rank 3: one spatial axis
    ] {
        let data = ramp(&shape, 1.3, 4.0);
        let cpu =
            oxi_instance_norm(&Tensor::new(data.clone(), shape.clone()), eps).expect("cpu kernel");
        let gpu = gpu_instance_norm(&ctx, &data, &shape, eps).unwrap_or_else(|| {
            panic!(
                "an adapter is present and {shape:?} is a valid shape, so the GPU \
                 kernel must dispatch — a None here would make this test vacuous"
            )
        });
        assert_close(&gpu, &cpu.data, 1e-5, 1e-5, &format!("shape {shape:?}"));
    }
}

/// Both kernels must read `epsilon` the same way: it is the only thing keeping
/// a zero-variance plane finite, and the only free parameter the fused node
/// carries.
#[test]
fn cpu_and_gpu_kernels_agree_on_epsilon() {
    let ctx = match GpuContext::try_new() {
        Some(ctx) => ctx,
        None => return,
    };
    if ctx.is_degraded() {
        return;
    }

    let shape = vec![1usize, 2, 8, 8];
    for eps in [0.0f32, 1e-8, 1e-5, 1.0, 100.0] {
        let data = ramp(&shape, 2.0, -6.0);
        let cpu =
            oxi_instance_norm(&Tensor::new(data.clone(), shape.clone()), eps).expect("cpu kernel");
        let gpu = match gpu_instance_norm(&ctx, &data, &shape, eps) {
            Some(gpu) => gpu,
            None => panic!("GPU kernel declined a valid shape at eps={eps}"),
        };
        assert_close(&gpu, &cpu.data, 1e-5, 1e-5, &format!("eps {eps}"));
    }

    // A constant plane: variance is exactly zero, so the result is decided
    // entirely by epsilon on both sides.
    let flat = vec![7.0f32; 128];
    let cpu = oxi_instance_norm(&Tensor::new(flat.clone(), shape.clone()), 1e-8).expect("cpu");
    let gpu = match gpu_instance_norm(&ctx, &flat, &shape, 1e-8) {
        Some(gpu) => gpu,
        None => panic!("GPU kernel declined a constant plane"),
    };
    assert!(gpu.iter().all(|v| v.is_finite()), "{gpu:?}");
    assert_close(&gpu, &cpu.data, 1e-5, 1e-5, "constant plane");
}
