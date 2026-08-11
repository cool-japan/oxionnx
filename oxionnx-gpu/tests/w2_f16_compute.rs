//! [w2-f16] Device-backed checks for the opt-in half-precision compute path.
//!
//! Five claims, each mapped to a mandate:
//!
//! * **Quality.** `conv2d` (both the implicit kernel and the fused entry point)
//!   and `gemm_nt` in `f16` must stay within 55 dB PSNR of their own `f32`
//!   result on realistic-magnitude tensors. The measured figures are printed,
//!   because the number is the evidence, not the pass/fail.
//! * **Cache format keying.** Flipping the toggle between two runs of one
//!   context must never serve one kernel the other's bytes. Both runs must be
//!   correct, and the `f32` copy must still be resident (and still hit) after
//!   the `f16` copy exists.
//! * **Unsupported feature.** A context whose device lacks `SHADER_F16` reports
//!   the mode off no matter what was requested, and every kernel stays on the
//!   `f32` path.
//! * **Residency interplay.** An `f16` convolution consuming a device-resident
//!   input and producing a `Device`-placed output must still be correct.
//! * **Byte reduction.** The resident weight bytes with the toggle on must be
//!   half those with it off.
//!
//! Every test prints a `skip: ...` line and returns when no adapter is present.

use oxionnx_core::Tensor;
use oxionnx_gpu::context::activation::{OutputPlacement, TensorSource};
use oxionnx_gpu::{
    gpu_conv2d_fused_placed_async, gpu_conv2d_implicit_placed_async, gpu_gemm_nt_placed_async,
    ConvActivation, GpuContext, WeightKeys,
};

// ── shapes ──────────────────────────────────────────────────────────────
//
// Big enough to clear `compute.rs`'s 10 MFLOP dispatch gate (the fused entry
// point applies it) and to make a K-deep reduction's error meaningful, small
// enough that the whole file runs in seconds.
const C: usize = 64;
const HW: usize = 32;
const K: usize = 3;

/// Realistic-magnitude fill: roughly normal, clamped into `[-2, 2]`.
///
/// A uniform ramp would understate `f16`'s error near zero and overstate it at
/// the extremes; the sum of three decorrelated sawtooths is cheap, deterministic
/// and bell-ish, which is what a trained weight tensor actually looks like.
fn normalish(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let i = i as u32;
            let a = (i.wrapping_mul(seed) % 1009) as f32 / 1009.0;
            let b = (i.wrapping_mul(seed ^ 0x9E37_79B9) % 761) as f32 / 761.0;
            let c = (i.wrapping_mul(seed.rotate_left(7)) % 397) as f32 / 397.0;
            ((a + b + c) / 1.5 - 1.0) * 2.0
        })
        .collect()
}

/// The same fill with a per-channel scale sweeping four orders of magnitude.
///
/// `f16` has ~11 bits of mantissa but a *much* narrower exponent range than
/// `f32`, so a tensor whose channels differ wildly in magnitude is the case
/// where a naive half-precision kernel falls apart. Accumulating in `f32` is
/// what keeps it from doing so, and this is the fill that checks it.
fn channel_scaled(len: usize, channels: usize, seed: u32) -> Vec<f32> {
    let per = len / channels.max(1);
    normalish(len, seed)
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let ch = i.checked_div(per).unwrap_or(0);
            v * 10f32.powi((ch % 5) as i32 - 2)
        })
        .collect()
}

/// Peak-signal-to-noise ratio of `got` against `want`, in dB.
///
/// `f32::INFINITY` for a bit-identical pair, which is what a toggle-off
/// comparison must produce.
fn psnr(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "PSNR needs matching lengths");
    let peak = want.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    let mse = want
        .iter()
        .zip(got)
        .map(|(&w, &g)| (f64::from(w) - f64::from(g)).powi(2))
        .sum::<f64>()
        / want.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (peak * peak / mse).log10()
}

/// Largest absolute deviation, and the reference's own peak magnitude.
fn max_abs_error(got: &[f32], want: &[f32]) -> (f64, f64) {
    let peak = want.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    let err = want.iter().zip(got).fold(0.0f64, |m, (&w, &g)| {
        m.max((f64::from(w) - f64::from(g)).abs())
    });
    (err, peak)
}

/// The quality bar, and the sanity bound that goes with it.
///
/// 55 dB is the mandated floor; the design lands at 70-85 dB because every
/// accumulator is `f32`. The max-abs bound is relative to the reference's own
/// peak, because an absolute one is meaningless without knowing the scale.
fn assert_quality(case: &str, got: &[f32], want: &[f32]) {
    let db = psnr(got, want);
    let (err, peak) = max_abs_error(got, want);
    println!("  {case}: PSNR {db:.1} dB, max|err| {err:.3e} (peak {peak:.3})");
    assert!(
        db >= 55.0,
        "{case}: PSNR {db:.1} dB is below the 55 dB gate"
    );
    assert!(
        err <= 0.02 * peak.max(1e-6),
        "{case}: max abs error {err:.3e} exceeds 2% of the peak {peak:.3}"
    );
}

/// A context, or `None` with the mandated skip line.
fn context(test: &str) -> Option<GpuContext> {
    let Some(ctx) = GpuContext::try_new() else {
        println!("skip: no GPU adapter available ({test})");
        return None;
    };
    Some(ctx)
}

/// A context that can actually run the half-precision kernels.
fn f16_context(test: &str) -> Option<GpuContext> {
    let ctx = context(test)?;
    if !ctx.f16_compute_supported() {
        println!("skip: adapter does not support shader-f16 ({test})");
        return None;
    }
    Some(ctx)
}

fn conv_operands(seed: u32, scaled: bool) -> (Tensor, Tensor, Tensor) {
    let in_len = C * HW * HW;
    let w_len = C * C * K * K;
    let input = Tensor::new(
        if scaled {
            channel_scaled(in_len, C, seed)
        } else {
            normalish(in_len, seed)
        },
        vec![1, C, HW, HW],
    );
    let weight = Tensor::new(normalish(w_len, seed ^ 0x5bf0_3635), vec![C, C, K, K]);
    let bias = Tensor::new(normalish(C, seed ^ 0x27d4_eb2f), vec![C]);
    (input, weight, bias)
}

/// Run the implicit conv once, with the toggle in the given state.
fn run_conv(
    ctx: &GpuContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    keys: WeightKeys<'_>,
    f16: bool,
) -> Vec<f32> {
    assert_eq!(
        ctx.set_f16_compute(f16),
        f16,
        "the toggle must take the state this test asked for"
    );
    let out = pollster::block_on(gpu_conv2d_implicit_placed_async(
        ctx,
        TensorSource::tensor(input),
        weight,
        Some(bias),
        keys,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
        OutputPlacement::Host,
    ))
    .expect("the conv kernel must dispatch this shape");
    out.into_vec()
        .expect("a Host placement returns host values")
}

// ── (b) quality gate ────────────────────────────────────────────────────

#[test]
fn conv2d_f16_meets_the_quality_gate() {
    let Some(ctx) = f16_context("conv2d_f16_meets_the_quality_gate") else {
        return;
    };
    println!("conv2d implicit, [1,{C},{HW},{HW}] x [{C},{C},{K},{K}]:");
    for (label, scaled) in [("normal magnitudes", false), ("channel-scaled", true)] {
        let (input, weight, bias) = conv_operands(0x2545_f491, scaled);
        let want = run_conv(&ctx, &input, &weight, &bias, WeightKeys::default(), false);
        let got = run_conv(&ctx, &input, &weight, &bias, WeightKeys::default(), true);
        assert_quality(label, &got, &want);
    }
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );
}

#[test]
fn conv2d_fused_f16_meets_the_quality_gate() {
    let Some(ctx) = f16_context("conv2d_fused_f16_meets_the_quality_gate") else {
        return;
    };
    println!("conv2d fused (bias + activation in the epilogue):");
    // Every fused activation, so the epilogue — which stays entirely in f32 —
    // is exercised on the f16 path as well as the f32 one.
    for (label, act) in [
        ("relu", ConvActivation::Relu),
        ("leaky_relu(0.2)", ConvActivation::LeakyRelu(0.2)),
        (
            "clip(-1, 6)",
            ConvActivation::Clip {
                min: -1.0,
                max: 6.0,
            },
        ),
    ] {
        let (input, weight, bias) = conv_operands(0x9e37_79b9, false);
        let run = |f16: bool| {
            assert_eq!(ctx.set_f16_compute(f16), f16);
            pollster::block_on(gpu_conv2d_fused_placed_async(
                &ctx,
                TensorSource::tensor(&input),
                &weight,
                Some(&bias),
                WeightKeys::default(),
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                1,
                act,
                OutputPlacement::Host,
            ))
            .expect("the fused conv entry point must dispatch this shape")
            .into_vec()
            .expect("Host placement")
        };
        let want = run(false);
        let got = run(true);
        assert_quality(label, &got, &want);
    }
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );
}

#[test]
fn gemm_nt_f16_meets_the_quality_gate() {
    let Some(ctx) = f16_context("gemm_nt_f16_meets_the_quality_gate") else {
        return;
    };
    // InSwapper's AdaIN head shape: A [m, k], B [n, k] read as B^T.
    let (m, k, n) = (32usize, 512usize, 512usize);
    println!("gemm_nt, [{m},{k}] x [{n},{k}]^T:");
    for (label, scaled) in [("normal magnitudes", false), ("channel-scaled B", true)] {
        let a = normalish(m * k, 0x1234_5678);
        let b = if scaled {
            channel_scaled(n * k, n, 0x8765_4321)
        } else {
            normalish(n * k, 0x8765_4321)
        };
        let c = normalish(n, 0xabcd_ef01);
        let a_shape = [m, k];
        let run = |f16: bool| {
            assert_eq!(ctx.set_f16_compute(f16), f16);
            pollster::block_on(gpu_gemm_nt_placed_async(
                &ctx,
                TensorSource::host(&a, &a_shape),
                m,
                k,
                &b,
                n,
                Some(&c),
                1.0,
                1.0,
                WeightKeys::default(),
                OutputPlacement::Host,
            ))
            .expect("the gemm kernel must dispatch this shape")
            .into_vec()
            .expect("Host placement")
        };
        let want = run(false);
        let got = run(true);
        assert_quality(label, &got, &want);
    }
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );
}

// ── (c) toggle-flip mid-session, and the cache format key ───────────────

/// The regression mandate 3 exists for: with a *keyed* weight, flipping the
/// toggle between runs must produce two correct results, not one correct and
/// one built from bytes read at the wrong width.
///
/// The sequence is deliberately f32 -> f16 -> f32: the last run is what proves
/// the `f16` copy did not evict or overwrite the `f32` one.
#[test]
fn flipping_the_toggle_mid_session_never_crosses_the_formats() {
    let Some(ctx) = f16_context("flipping_the_toggle_mid_session_never_crosses_the_formats") else {
        return;
    };
    let (input, weight, bias) = conv_operands(0x0517_3f2a, false);
    let keys = WeightKeys::new(Some("conv.weight"), Some("conv.bias"));

    // 1. f32, cold: this is the reference every later run is measured against.
    let f32_reference = run_conv(&ctx, &input, &weight, &bias, keys, false);
    assert!(
        ctx.is_resident("conv.weight"),
        "the weight must be resident"
    );
    let f32_only_bytes = ctx.resident_bytes();
    assert_eq!(ctx.resident_len(), 2, "one weight and one bias identity");

    // 2. f16: a *second* copy of the same identity, not a replacement.
    let f16_result = run_conv(&ctx, &input, &weight, &bias, keys, true);
    let both_bytes = ctx.resident_bytes();
    assert_eq!(
        ctx.resident_len(),
        2,
        "two formats of one weight are still one identity"
    );
    assert!(
        both_bytes > f32_only_bytes,
        "the f16 copy must be an additional allocation ({f32_only_bytes} -> {both_bytes})"
    );
    assert_quality("f16 against the f32 reference", &f16_result, &f32_reference);

    // 3. f32 again. If the cache were keyed by name alone, this would either
    //    read the f16 bytes as f32 (garbage) or re-upload every frame. It must
    //    do neither: same bits as run 1, and no new bytes at all.
    let uploads_before = ctx.uploaded_bytes();
    let counters_before = ctx.resident_counters();
    let f32_again = run_conv(&ctx, &input, &weight, &bias, keys, false);
    assert_eq!(
        f32_again, f32_reference,
        "returning to f32 must reproduce the f32 result bit for bit"
    );
    let delta = ctx.resident_counters().since(counters_before);
    assert_eq!(
        delta.uploaded_bytes, 0,
        "the f32 copy must still be resident and still hit"
    );
    assert!(delta.hits >= 2, "weight and bias must both hit: {delta:?}");
    assert_eq!(
        ctx.resident_bytes(),
        both_bytes,
        "no format's copy may be evicted by the other"
    );

    // 4. And back to f16 — also a pure hit now.
    let counters_before = ctx.resident_counters();
    let f16_again = run_conv(&ctx, &input, &weight, &bias, keys, true);
    assert_eq!(
        f16_again, f16_result,
        "the second f16 run must reproduce the first bit for bit"
    );
    let delta = ctx.resident_counters().since(counters_before);
    assert_eq!(
        delta.uploaded_bytes, 0,
        "the f16 copy must be resident too, not re-uploaded per flip"
    );
    println!(
        "  resident bytes: f32 only {f32_only_bytes}, both formats {both_bytes}, \
         total uploads since step 3: {}",
        ctx.uploaded_bytes() - uploads_before
    );
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );
}

/// The byte-reduction claim, isolated: two contexts, one weight each.
///
/// Separate contexts rather than one, because the point is the *steady-state*
/// resident footprint of a session that only ever runs in one mode — which is
/// what a browser page actually pays.
#[test]
fn a_resident_f16_weight_is_half_the_bytes() {
    let Some(ctx_a) = f16_context("a_resident_f16_weight_is_half_the_bytes") else {
        return;
    };
    let Some(ctx_b) = f16_context("a_resident_f16_weight_is_half_the_bytes") else {
        return;
    };
    let (input, weight, bias) = conv_operands(0x3c6e_f35f, false);
    let keys = WeightKeys::new(Some("conv.weight"), None);

    let _ = run_conv(&ctx_a, &input, &weight, &bias, keys, false);
    let _ = run_conv(&ctx_b, &input, &weight, &bias, keys, true);

    let f32_bytes = ctx_a.resident_bytes();
    let f16_bytes = ctx_b.resident_bytes();
    let expected = (C * C * K * K * 4) as u64;
    println!(
        "  resident weight bytes: f32 {f32_bytes}, f16 {f16_bytes} (weight is {expected} B as f32)"
    );
    assert_eq!(f32_bytes, expected, "the f32 copy is 4 bytes per element");
    assert_eq!(
        f16_bytes,
        expected / 2,
        "the f16 copy must be exactly half the bytes"
    );
}

// ── (d) unsupported / toggle-off ────────────────────────────────────────

/// The effective state must equal the device's own feature bit, and the
/// default must be off on every device.
///
/// The `!supported` branch cannot be produced on hardware that supports the
/// feature, so it is pinned deviceless by
/// `context::weight_format::tests::support_is_the_authority_over_the_request`.
/// What *this* test adds is that the reported support is the device's actual
/// feature bit rather than a hopeful constant.
#[test]
fn support_matches_the_device_and_the_default_is_off() {
    let Some(ctx) = context("support_matches_the_device_and_the_default_is_off") else {
        return;
    };
    let device_says = ctx.device.features().contains(wgpu::Features::SHADER_F16);
    assert_eq!(
        ctx.f16_compute_supported(),
        device_says,
        "reported support must be the device's own feature bit"
    );
    assert!(
        !ctx.f16_compute_enabled(),
        "half precision must never be silently on"
    );
    assert_eq!(
        ctx.set_f16_compute(true),
        device_says,
        "set must report the effective state, which support decides"
    );
    assert_eq!(ctx.f16_compute_enabled(), device_says);
    assert!(!ctx.set_f16_compute(false));
    assert!(!ctx.f16_compute_enabled());
    println!("  device reports shader-f16: {device_says}");
}

/// With the toggle off, two runs of the same kernel are bit-identical — and
/// stay so after the `f16` path has been exercised in between.
#[test]
fn the_toggle_off_path_is_bit_identical_before_and_after_f16() {
    let Some(ctx) = context("the_toggle_off_path_is_bit_identical_before_and_after_f16") else {
        return;
    };
    let (input, weight, bias) = conv_operands(0x7f4a_7c15, false);
    let keys = WeightKeys::new(Some("w"), Some("b"));

    // Never touched: this is the "before your change" reference.
    let pristine = pollster::block_on(gpu_conv2d_implicit_placed_async(
        &ctx,
        TensorSource::tensor(&input),
        &weight,
        Some(&bias),
        keys,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
        OutputPlacement::Host,
    ))
    .expect("dispatch")
    .into_vec()
    .expect("Host placement");

    // Explicitly off must equal never-touched.
    assert!(!ctx.set_f16_compute(false));
    let explicit_off = run_conv(&ctx, &input, &weight, &bias, keys, false);
    assert_eq!(
        explicit_off, pristine,
        "an explicit `false` must be indistinguishable from never asking"
    );

    if ctx.f16_compute_supported() {
        let _ = run_conv(&ctx, &input, &weight, &bias, keys, true);
        let back_off = run_conv(&ctx, &input, &weight, &bias, keys, false);
        assert_eq!(
            back_off, pristine,
            "the f32 path must be untouched by having run the f16 one"
        );
    }
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );
}

// ── (e) residency interplay ─────────────────────────────────────────────

/// An `f16` convolution whose input is already on the device and whose output
/// stays there. Both halves of Wave 1's residency contract, on the new path.
#[test]
fn f16_conv_honours_device_input_and_device_output() {
    let Some(ctx) = f16_context("f16_conv_honours_device_input_and_device_output") else {
        return;
    };
    let (input, weight, bias) = conv_operands(0x1d87_2b41, false);
    let keys = WeightKeys::new(Some("w"), Some("b"));

    // The f32 reference, entirely host-side.
    let reference = run_conv(&ctx, &input, &weight, &bias, keys, false);

    // Now: f16, input resident, output left on the device.
    assert!(ctx.set_f16_compute(true));
    let resident_input = ctx
        .upload_device_tensor("test_input", &input.data, &input.shape)
        .expect("the input must upload as a device tensor");
    let device_out = pollster::block_on(gpu_conv2d_implicit_placed_async(
        &ctx,
        TensorSource::Device(&resident_input),
        &weight,
        Some(&bias),
        keys,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        ConvActivation::None,
        OutputPlacement::Device,
    ))
    .expect("the f16 kernel must accept a device input")
    .into_device()
    .expect("a Device placement must leave the result on the device");

    assert_eq!(
        device_out.shape(),
        &[1, C, HW, HW],
        "a device-placed result still carries its shape"
    );
    let got = pollster::block_on(oxionnx_gpu::read_device_tensor_async(&ctx, &device_out))
        .expect("reading the device tensor back must succeed");
    println!("f16 conv, device input -> device output:");
    assert_quality("device-resident f16", &got.data, &reference);
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );
}
