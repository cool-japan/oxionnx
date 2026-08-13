//! CPU oracles for this wave's elementwise/normalization ops (`Add`/`Sub`/
//! `Mul`/`Div`'s channel-broadcast path, `PRelu`, `BatchNormalization`,
//! `OxiInstanceNorm`).
//!
//! Kept in a file of its own — included into [`mod@crate::reference`]
//! verbatim via `#[path]` — rather than appended to `reference.rs` itself,
//! purely to avoid growing an already-large file further (this workspace's
//! own "single file under 2000 lines" refactor policy). Mirrors
//! `reference_data_ops.rs`, the identical companion-file pattern the
//! data-movement CUDA op wave (`MaxPool`/`AveragePool`/`Resize`/`Pad`/
//! `Slice`/`Concat`) already established for exactly this reason — see that
//! file's own header, and `lib.rs`'s `#[path = "dispatch_tests.rs"] mod
//! dispatch_tests;` / `conv.rs`'s `#[path = "conv_tests.rs"] mod tests;` for
//! the same convention elsewhere in this crate. Every function here is
//! reachable as `crate::reference::ref_*`, exactly like every oracle
//! `reference.rs` defines directly.
//!
//! # Independent of `oxionnx-ops`, and of the kernels under test
//!
//! Same rule as every oracle in `reference.rs` itself (see that module's own
//! "why this does not depend on `oxionnx-ops`"): every formula below is
//! reimplemented from scratch against the exact shape each dispatch module
//! (`crate::broadcast`, `crate::prelu`, `crate::norm`) documents its kernel
//! as computing, never called from the CPU operator library and never
//! shared with the CUDA kernel/PTX it grades. [`ref_binary_broadcast`] is
//! the one exception worth naming: it calls back into `reference.rs`'s own
//! [`super::ref_binary`] for the per-element `Add`/`Sub`/`Mul`/`Div`
//! formula, which is fine — that function is itself independent of the GPU
//! kernel, just shared plumbing between the exact-shape and broadcast paths'
//! oracles, exactly as `crate::broadcast`'s kernel and
//! `elementwise::cuda_binary_elementwise_bound`'s kernel share no code
//! either but compute the same four formulas.
//!
//! # Deliberately serial
//!
//! None of the functions below has an inner accumulation loop past a single
//! `(n, c)` plane (`ref_oxi_instance_norm`'s two passes over `spatial`
//! elements — a few thousand for every real model in this pipeline). None of
//! them are the "349 GFLOP of scalar work per frame" pathology `rayon` was
//! added to `reference.rs` to fix (see that module's own "Parallelism"
//! section), so none of them pull `rayon` in at all.

use oxionnx_core::graph::OpKind;

use super::ref_binary;

/// The channel/scalar-broadcast binary formula [`crate::broadcast`]'s kernel
/// computes: for every element `i` of `full`, combine it with
/// `small[(i / spatial) % channels]` (or `small[0]` when `small.len() ==
/// 1`), in the order `reverse` selects.
///
/// `reverse == false` computes `full[i] OP small[idx]`; `reverse == true`
/// computes `small[idx] OP full[i]` — see `crate::broadcast::reverse_for`
/// for why only `Sub`/`Div` ever need `true`.
///
/// Returns `None` — no formula, `shadow_verify` skips the check rather than
/// comparing against a substitute — when `channels` or `spatial` is `0`
/// (either would divide/index by zero), `small`'s length is neither
/// `channels` nor `1`, or `op` has no [`ref_binary`] formula.
#[must_use]
pub fn ref_binary_broadcast(
    op: &OpKind,
    full: &[f32],
    small: &[f32],
    channels: usize,
    spatial: usize,
    reverse: bool,
) -> Option<Vec<f32>> {
    if channels == 0 || spatial == 0 {
        return None;
    }
    if small.len() != channels && small.len() != 1 {
        return None;
    }
    full.iter()
        .enumerate()
        .map(|(i, &full_val)| {
            let channel = (i / spatial) % channels;
            let small_idx = if small.len() == 1 { 0 } else { channel };
            let small_val = small[small_idx];
            if reverse {
                ref_binary(op, small_val, full_val)
            } else {
                ref_binary(op, full_val, small_val)
            }
        })
        .collect()
}

/// The per-channel `PRelu` formula `crate::prelu`'s kernel computes:
/// `x >= 0 ? x : slope[idx] * x`, `idx = (i / spatial) % channels` (or `0`
/// when `slope.len() == 1`).
///
/// `f64` intermediate arithmetic, matching `reference::ref_unary`'s
/// discipline.
///
/// Returns `None` when `channels`/`spatial` is `0`, or `slope`'s length is
/// neither `channels` nor `1` — mirrors `crate::prelu::prelu_plan`'s own
/// decline rule, so every shape this oracle is ever asked about is one the
/// kernel itself would have accepted.
#[must_use]
pub fn ref_prelu(x: &[f32], slope: &[f32], channels: usize, spatial: usize) -> Option<Vec<f32>> {
    if channels == 0 || spatial == 0 {
        return None;
    }
    if slope.len() != channels && slope.len() != 1 {
        return None;
    }
    Some(
        x.iter()
            .enumerate()
            .map(|(i, &xv)| {
                let channel = (i / spatial) % channels;
                let idx = if slope.len() == 1 { 0 } else { channel };
                let xf = f64::from(xv);
                let y = if xf >= 0.0 {
                    xf
                } else {
                    f64::from(slope[idx]) * xf
                };
                y as f32
            })
            .collect(),
    )
}

/// The `BatchNormalization` (inference) formula `crate::norm`'s kernel
/// computes: `y = scale[c] * (x - mean[c]) / sqrt(var[c] + eps) + bias[c]`,
/// `c = (i / spatial) % channels`, matching
/// `oxionnx-ops::nn::normalization::batch_norm` element for element.
///
/// `f64` intermediate arithmetic throughout (including the `sqrt` and the
/// division), matching `reference::ref_unary`'s discipline.
///
/// Returns `None` when `channels`/`spatial` is `0`, or any of
/// `scale`/`bias`/`mean`/`var` is shorter than `channels`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ref_batch_norm(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    channels: usize,
    spatial: usize,
    epsilon: f32,
) -> Option<Vec<f32>> {
    if channels == 0 || spatial == 0 {
        return None;
    }
    if scale.len() < channels
        || bias.len() < channels
        || mean.len() < channels
        || var.len() < channels
    {
        return None;
    }
    let eps = f64::from(epsilon);
    Some(
        x.iter()
            .enumerate()
            .map(|(i, &xv)| {
                let c = (i / spatial) % channels;
                let inv_std = 1.0 / (f64::from(var[c]) + eps).sqrt();
                let y = (f64::from(xv) - f64::from(mean[c])) * inv_std * f64::from(scale[c])
                    + f64::from(bias[c]);
                y as f32
            })
            .collect(),
    )
}

/// The `OxiInstanceNorm` formula `crate::norm`'s kernel computes: two-pass
/// per-`(n, c)`-plane mean/variance, then `(x - mean) / sqrt(var + eps)` —
/// no affine term. Matches `oxionnx-ops::registry::oxi_instance_norm`'s
/// `normalize_plane` element for element, in `f64` throughout (matching
/// `reference::ref_unary`'s discipline; the CPU operator itself accumulates
/// in `f32`, so this oracle is *more* precise than what it grades, exactly
/// the relationship `reference::ref_conv`/`reference::ref_matmul` already
/// have with their GPU kernels).
///
/// `shape` is `[N, C, d1, ...]`; the plane size is `product(shape[2..])`.
/// Returns `None` when `shape` has rank `< 3` or the flattened plane size is
/// `0` — mirrors `crate::norm::oxi_instance_norm_plan`'s own decline rule.
#[must_use]
pub fn ref_oxi_instance_norm(x: &[f32], shape: &[usize], epsilon: f32) -> Option<Vec<f32>> {
    if shape.len() < 3 {
        return None;
    }
    let spatial: usize = shape[2..].iter().product();
    if spatial == 0 {
        return None;
    }
    let eps = f64::from(epsilon);
    let mut out = vec![0.0_f32; x.len()];
    for (plane_in, plane_out) in x.chunks(spatial).zip(out.chunks_mut(spatial)) {
        let mut sum = 0.0_f64;
        for &v in plane_in {
            sum += f64::from(v);
        }
        let mean = sum / spatial as f64;
        let mut var_sum = 0.0_f64;
        for &v in plane_in {
            let d = f64::from(v) - mean;
            var_sum += d * d;
        }
        let var = var_sum / spatial as f64;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (dst, &src) in plane_out.iter_mut().zip(plane_in.iter()) {
            *dst = ((f64::from(src) - mean) * inv_std) as f32;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ref_binary_broadcast ─────────────────────────────────────────────────

    #[test]
    fn ref_binary_broadcast_per_channel_mul_forward() {
        // full = [1,2,1,1,2,2] as [N=1,C=2,H=2? no -- keep it simple]:
        // channels=2, spatial=2, full has 4 elements: [c0,c0,c1,c1] pattern.
        let full = [1.0_f32, 2.0, 3.0, 4.0]; // channel 0: [1,2], channel 1: [3,4]
        let small = [10.0_f32, 100.0]; // per-channel multiplier
        let out = ref_binary_broadcast(&OpKind::Mul, &full, &small, 2, 2, false).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 300.0, 400.0]);
    }

    #[test]
    fn ref_binary_broadcast_scalar_add() {
        let full = [1.0_f32, 2.0, 3.0, 4.0];
        let scalar = [100.0_f32];
        // channels=1 forces small_idx=0 for every element, regardless of spatial.
        let out = ref_binary_broadcast(&OpKind::Add, &full, &scalar, 1, 4, false).unwrap();
        assert_eq!(out, vec![101.0, 102.0, 103.0, 104.0]);
    }

    #[test]
    fn ref_binary_broadcast_sub_reverse_computes_small_minus_full() {
        let full = [1.0_f32, 2.0];
        let small = [10.0_f32];
        let forward = ref_binary_broadcast(&OpKind::Sub, &full, &small, 1, 2, false).unwrap();
        let reverse = ref_binary_broadcast(&OpKind::Sub, &full, &small, 1, 2, true).unwrap();
        assert_eq!(forward, vec![1.0 - 10.0, 2.0 - 10.0]);
        assert_eq!(reverse, vec![10.0 - 1.0, 10.0 - 2.0]);
    }

    #[test]
    fn ref_binary_broadcast_declines_zero_channels_or_spatial() {
        assert_eq!(
            ref_binary_broadcast(&OpKind::Add, &[1.0], &[1.0], 0, 1, false),
            None
        );
        assert_eq!(
            ref_binary_broadcast(&OpKind::Add, &[1.0], &[1.0], 1, 0, false),
            None
        );
    }

    #[test]
    fn ref_binary_broadcast_declines_a_mismatched_small_length() {
        // small has 3 elements, matching neither channels (2) nor 1.
        assert_eq!(
            ref_binary_broadcast(
                &OpKind::Add,
                &[1.0, 2.0, 3.0, 4.0],
                &[1.0, 2.0, 3.0],
                2,
                2,
                false
            ),
            None
        );
    }

    // ── ref_prelu ────────────────────────────────────────────────────────────

    #[test]
    fn ref_prelu_hand_verified_per_channel() {
        // channels=2, spatial=2: channel 0 = [1.0,-2.0], channel 1 = [3.0,-4.0].
        let x = [1.0_f32, -2.0, 3.0, -4.0];
        let slope = [0.1_f32, 0.5];
        let out = ref_prelu(&x, &slope, 2, 2).unwrap();
        // x >= 0 passes through; x < 0 scales by that channel's slope.
        assert_eq!(out, vec![1.0, -0.2, 3.0, -2.0]);
    }

    #[test]
    fn ref_prelu_scalar_slope_applies_everywhere() {
        let x = [1.0_f32, -2.0, 3.0, -4.0];
        let out = ref_prelu(&x, &[0.5], 2, 2).unwrap();
        assert_eq!(out, vec![1.0, -1.0, 3.0, -2.0]);
    }

    #[test]
    fn ref_prelu_declines_mismatched_slope_length() {
        assert_eq!(ref_prelu(&[1.0, -2.0], &[0.1, 0.2, 0.3], 2, 1), None);
    }

    #[test]
    fn ref_prelu_declines_zero_channels_or_spatial() {
        assert_eq!(ref_prelu(&[1.0], &[0.1], 0, 1), None);
        assert_eq!(ref_prelu(&[1.0], &[0.1], 1, 0), None);
    }

    // ── ref_batch_norm ───────────────────────────────────────────────────────

    #[test]
    fn ref_batch_norm_hand_verified() {
        // One channel, one spatial element: y = scale*(x-mean)/sqrt(var+eps)+bias.
        // x=5, mean=1, var=3, eps=1 -> sqrt(4)=2 -> (5-1)/2=2 -> *scale(2)+bias(1)=5.
        let out = ref_batch_norm(&[5.0], &[2.0], &[1.0], &[1.0], &[3.0], 1, 1, 1.0).unwrap();
        assert!((out[0] - 5.0).abs() < 1.0e-6, "got {}", out[0]);
    }

    #[test]
    fn ref_batch_norm_identity_affine_and_zero_stats_is_a_pure_rescale() {
        // scale=1, bias=0, mean=0, var=eps_complement so inv_std=1: y == x.
        let x = [1.0_f32, -2.0, 3.5];
        let out = ref_batch_norm(&x, &[1.0], &[0.0], &[0.0], &[0.0], 1, 3, 1.0).unwrap();
        // var=0, eps=1 -> inv_std = 1/sqrt(1) = 1 -> y = x.
        for (a, b) in out.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1.0e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn ref_batch_norm_indexes_channel_from_the_flat_position() {
        // channels=2, spatial=2: [c0,c0,c1,c1]. Distinct scale per channel.
        let x = [0.0_f32, 0.0, 0.0, 0.0];
        let out = ref_batch_norm(
            &x,
            &[10.0, 20.0],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            2,
            2,
            1.0, // var+eps = 1, inv_std = 1
        )
        .unwrap();
        // y = scale*(0-0)*1 + bias = bias, per channel.
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn ref_batch_norm_declines_short_operands() {
        assert_eq!(
            ref_batch_norm(&[1.0, 2.0], &[1.0], &[0.0], &[0.0], &[1.0], 2, 1, 1e-5),
            None
        );
    }

    // ── ref_oxi_instance_norm ────────────────────────────────────────────────

    #[test]
    fn ref_oxi_instance_norm_hand_verified_constant_plane() {
        // A constant plane has zero variance: (x - mean) == 0 everywhere.
        let x = [5.0_f32; 8];
        let out = ref_oxi_instance_norm(&x, &[1, 2, 2, 2], 1e-8).unwrap();
        assert!(out.iter().all(|&v| v.abs() < 1e-6), "{out:?}");
    }

    #[test]
    fn ref_oxi_instance_norm_each_plane_has_zero_mean_and_unit_variance() {
        let x: Vec<f32> = (0..32).map(|i| (i % 7) as f32 - 3.0).collect();
        let shape = [2usize, 4, 2, 2]; // 8 planes of 4 elements each
        let out = ref_oxi_instance_norm(&x, &shape, 1e-12).unwrap();
        let spatial = 4;
        for plane in out.chunks(spatial) {
            let mean: f32 = plane.iter().sum::<f32>() / spatial as f32;
            let var: f32 =
                plane.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / spatial as f32;
            assert!(mean.abs() < 1e-4, "plane mean {mean}");
            assert!((var - 1.0).abs() < 1e-3, "plane var {var}");
        }
    }

    #[test]
    fn ref_oxi_instance_norm_matches_hand_computed_two_element_plane() {
        // Single (n,c) plane [1,3]: mean=2, var=mean((1-2)^2,(3-2)^2)=1.
        // eps=0 -> inv_std=1 -> normalized = [(1-2), (3-2)] = [-1, 1].
        let out = ref_oxi_instance_norm(&[1.0, 3.0], &[1, 1, 1, 2], 0.0).unwrap();
        assert!((out[0] - (-1.0)).abs() < 1.0e-5, "{out:?}");
        assert!((out[1] - 1.0).abs() < 1.0e-5, "{out:?}");
    }

    #[test]
    fn ref_oxi_instance_norm_declines_rank_below_three() {
        assert_eq!(ref_oxi_instance_norm(&[1.0, 2.0], &[1, 2], 1e-5), None);
    }

    #[test]
    fn ref_oxi_instance_norm_declines_a_zero_spatial_extent() {
        assert_eq!(ref_oxi_instance_norm(&[], &[1, 2, 0, 3], 1e-5), None);
    }
}
