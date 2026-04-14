//! Integration tests for the 7 Audio/DSP operators:
//! HannWindow, HammingWindow, BlackmanWindow, DFT, STFT, MelWeightMatrix, Bernoulli.
//!
//! Tests use the direct OpContext construction pattern (same as the internal
//! unit tests in `dsp.rs`) because `oxionnx-ops` integration tests do not have
//! access to the `Session` / `Graph` API from the root crate.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::dsp::{
    BernoulliOp, BlackmanWindowOp, DFTOp, HammingWindowOp, HannWindowOp, MelWeightMatrixOp, STFTOp,
};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        registry: None,
    }
}

fn dummy_node(op: OpKind) -> Node {
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn node_with_int_attrs(op: OpKind, pairs: &[(&str, i64)]) -> Node {
    let mut n = dummy_node(op);
    for &(k, v) in pairs {
        n.attrs.ints.insert(k.to_string(), v);
    }
    n
}

fn node_with_float_attrs(op: OpKind, pairs: &[(&str, f32)]) -> Node {
    let mut n = dummy_node(op);
    for &(k, v) in pairs {
        n.attrs.floats.insert(k.to_string(), v);
    }
    n
}

/// Scalar tensor holding a single f32 value (used for i64-compatible scalars).
fn scalar_tensor(val: f32) -> Tensor {
    Tensor::new(vec![val], vec![1])
}

// ── Window tests ──────────────────────────────────────────────────────────────

/// HannWindow periodic with size=8.
/// expected[n] = 0.5 - 0.5 * cos(2π*n/8)
#[test]
fn test_hann_window_periodic() {
    use std::f32::consts::PI;
    let n = 8usize;
    let size_t = scalar_tensor(n as f32);
    // periodic = 1 (default, but set explicitly for clarity)
    let node = node_with_int_attrs(OpKind::HannWindow, &[("periodic", 1)]);
    let ctx = make_ctx(&node, vec![Some(&size_t)]);
    let out = HannWindowOp
        .execute(&ctx)
        .expect("HannWindow periodic execute failed");

    assert_eq!(out.len(), 1, "expected exactly one output tensor");
    assert_eq!(out[0].shape, vec![n], "shape mismatch");

    for i in 0..n {
        let expected = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
        let actual = out[0].data[i];
        assert!(
            (actual - expected).abs() < 1e-6,
            "HannWindow periodic idx {i}: got {actual}, expected {expected}",
        );
    }
}

/// HannWindow symmetric (periodic=0) with size=8.
/// expected[n] = 0.5 - 0.5 * cos(2π*n/(N-1))
#[test]
fn test_hann_window_symmetric() {
    use std::f32::consts::PI;
    let n = 8usize;
    let size_t = scalar_tensor(n as f32);
    let node = node_with_int_attrs(OpKind::HannWindow, &[("periodic", 0)]);
    let ctx = make_ctx(&node, vec![Some(&size_t)]);
    let out = HannWindowOp
        .execute(&ctx)
        .expect("HannWindow symmetric execute failed");

    assert_eq!(out[0].shape, vec![n]);
    let denom = (n - 1) as f32;
    for i in 0..n {
        let expected = 0.5 - 0.5 * (2.0 * PI * i as f32 / denom).cos();
        let actual = out[0].data[i];
        assert!(
            (actual - expected).abs() < 1e-6,
            "HannWindow symmetric idx {i}: got {actual}, expected {expected}",
        );
    }
}

/// HammingWindow periodic with size=8.
/// expected[n] = 0.54347826 - 0.45652174 * cos(2π*n/8)
#[test]
fn test_hamming_window() {
    use std::f32::consts::PI;
    const ALPHA: f32 = 0.543_478_26;
    const BETA: f32 = 0.456_521_74;
    let n = 8usize;
    let size_t = scalar_tensor(n as f32);
    let node = node_with_int_attrs(OpKind::HammingWindow, &[("periodic", 1)]);
    let ctx = make_ctx(&node, vec![Some(&size_t)]);
    let out = HammingWindowOp
        .execute(&ctx)
        .expect("HammingWindow execute failed");

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape, vec![n]);
    for i in 0..n {
        let expected = ALPHA - BETA * (2.0 * PI * i as f32 / n as f32).cos();
        let actual = out[0].data[i];
        assert!(
            (actual - expected).abs() < 1e-5,
            "HammingWindow idx {i}: got {actual}, expected {expected}",
        );
    }
}

/// BlackmanWindow periodic with size=8.
/// expected[n] = 0.42 - 0.5*cos(2π*n/8) + 0.08*cos(4π*n/8)
#[test]
fn test_blackman_window() {
    use std::f32::consts::PI;
    let n = 8usize;
    let size_t = scalar_tensor(n as f32);
    let node = node_with_int_attrs(OpKind::BlackmanWindow, &[("periodic", 1)]);
    let ctx = make_ctx(&node, vec![Some(&size_t)]);
    let out = BlackmanWindowOp
        .execute(&ctx)
        .expect("BlackmanWindow execute failed");

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape, vec![n]);
    for i in 0..n {
        let x = 2.0 * PI * i as f32 / n as f32;
        let expected = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
        let actual = out[0].data[i];
        assert!(
            (actual - expected).abs() < 1e-5,
            "BlackmanWindow idx {i}: got {actual}, expected {expected}",
        );
    }
}

// ── DFT tests ─────────────────────────────────────────────────────────────────

/// DFT on a sine wave at bin k=1 of length N=8: x[n] = sin(2π*n/8).
/// After a forward DFT the magnitude should peak at bin k=1.
/// Input shape: [1, 8, 1], output shape: [1, 8, 2].
#[test]
fn test_dft_sine_wave_peak() {
    use std::f32::consts::PI;
    let n = 8usize;
    let data: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect();
    let input = Tensor::new(data, vec![1, n, 1]);
    let node = dummy_node(OpKind::DFT);
    let ctx = make_ctx(&node, vec![Some(&input), None]);
    let out = DFTOp.execute(&ctx).expect("DFT sine-wave execute failed");

    assert_eq!(out.len(), 1);
    // Output shape [1, 8, 2]
    assert_eq!(out[0].shape, vec![1, n, 2]);

    // Compute magnitudes for each bin.
    let mag: Vec<f32> = (0..n)
        .map(|k| {
            let re = out[0].data[k * 2];
            let im = out[0].data[k * 2 + 1];
            (re * re + im * im).sqrt()
        })
        .collect();

    // Bin 1 and bin N-1=7 are the conjugate pair for a real sine; bin 1 >= bin 0 and bin 3.
    assert!(
        mag[1] > mag[0],
        "magnitude at bin 1 should exceed bin 0: mag[1]={}, mag[0]={}",
        mag[1],
        mag[0]
    );
    assert!(
        mag[1] > mag[3],
        "magnitude at bin 1 should exceed bin 3: mag[1]={}, mag[3]={}",
        mag[1],
        mag[3]
    );
}

/// Forward DFT then inverse DFT recovers the original real signal within 1e-4.
/// Input shape [1, 4, 1].
#[test]
fn test_dft_inverse_roundtrip() {
    let orig_vals = vec![1.0f32, 2.0, 3.0, 4.0];
    let n = orig_vals.len();
    let input = Tensor::new(orig_vals.clone(), vec![1, n, 1]);

    // Forward DFT (no onesided).
    let fwd_node = dummy_node(OpKind::DFT);
    let fwd_ctx = make_ctx(&fwd_node, vec![Some(&input), None]);
    let fwd = DFTOp.execute(&fwd_ctx).expect("DFT forward execute failed");
    // Shape [1, n, 2].
    assert_eq!(fwd[0].shape, vec![1, n, 2]);

    // Inverse DFT.
    let inv_node = node_with_int_attrs(OpKind::DFT, &[("inverse", 1)]);
    let inv_ctx = make_ctx(&inv_node, vec![Some(&fwd[0]), None]);
    let back = DFTOp.execute(&inv_ctx).expect("IDFT execute failed");
    // Shape [1, n, 2].
    assert_eq!(back[0].shape, vec![1, n, 2]);

    // Real parts should recover the original signal.
    for (i, &expected) in orig_vals.iter().enumerate() {
        let re = back[0].data[i * 2];
        assert!(
            (re - expected).abs() < 1e-4,
            "roundtrip idx {i}: re={re}, expected={expected}",
        );
    }
}

/// DFT onesided flag: output bins = N/2 + 1.
#[test]
fn test_dft_onesided_shape() {
    let n = 8usize;
    let data = vec![1.0f32; n];
    let input = Tensor::new(data, vec![1, n, 1]);
    let node = node_with_int_attrs(OpKind::DFT, &[("onesided", 1)]);
    let ctx = make_ctx(&node, vec![Some(&input), None]);
    let out = DFTOp.execute(&ctx).expect("DFT onesided execute failed");

    // onesided → n/2 + 1 = 5 bins, shape [1, 5, 2]
    assert_eq!(out[0].shape, vec![1, 5, 2]);
}

// ── STFT test ─────────────────────────────────────────────────────────────────

/// STFT output shape check.
/// Signal [1, 16, 1], frame_step=4, frame_length=8, onesided=1.
/// n_frames = (16 - 8) / 4 + 1 = 3
/// n_dft    = floor(8/2) + 1   = 5
/// Expected output shape: [1, 3, 5, 2]
#[test]
fn test_stft_output_shape() {
    let signal = Tensor::new(vec![1.0f32; 16], vec![1, 16, 1]);
    let frame_step_t = scalar_tensor(4.0);
    let window = Tensor::new(vec![1.0f32; 8], vec![8]);
    let frame_length_t = scalar_tensor(8.0);

    let node = node_with_int_attrs(OpKind::STFT, &[("onesided", 1)]);
    let ctx = make_ctx(
        &node,
        vec![
            Some(&signal),
            Some(&frame_step_t),
            Some(&window),
            Some(&frame_length_t),
        ],
    );
    let out = STFTOp.execute(&ctx).expect("STFT execute failed");

    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].shape,
        vec![1, 3, 5, 2],
        "STFT shape mismatch: got {:?}",
        out[0].shape,
    );
}

/// STFT without window (rectangular window implicit), two-sided output.
#[test]
fn test_stft_two_sided_shape() {
    let signal = Tensor::new(vec![0.5f32; 24], vec![1, 24, 1]);
    let frame_step_t = scalar_tensor(4.0);
    // No window (pass None for inputs[2]), frame_length=8 via inputs[3].
    let frame_length_t = scalar_tensor(8.0);

    // onesided=0 → all frame_length bins
    let node = node_with_int_attrs(OpKind::STFT, &[("onesided", 0)]);
    let ctx = make_ctx(
        &node,
        vec![
            Some(&signal),
            Some(&frame_step_t),
            None,
            Some(&frame_length_t),
        ],
    );
    let out = STFTOp.execute(&ctx).expect("STFT two-sided execute failed");

    // n_frames = (24 - 8) / 4 + 1 = 5
    // n_dft    = 8  (two-sided)
    assert_eq!(out[0].shape, vec![1, 5, 8, 2]);
}

// ── MelWeightMatrix tests ─────────────────────────────────────────────────────

/// MelWeightMatrix output shape and per-column energy > 0.
///
/// Parameters chosen so that each mel bin actually overlaps with at least one
/// spectrogram bin:
///   num_mel_bins=4, dft_length=256, sample_rate=8000,
///   lower_edge=60.0, upper_edge=3900.0.
/// Expected output shape: [dft_length/2+1, num_mel_bins] = [129, 4].
///
/// Note: with very low DFT resolution (e.g. dft_length=16, sr=16 kHz), spec
/// bins are 1000 Hz apart, which is wider than the first few mel bands. The
/// correct behaviour is that those bands then have zero weight — so the test
/// uses a coarser mel bank with a dense enough spectrogram.
#[test]
fn test_mel_weight_matrix_shape_and_sum() {
    let num_mel = scalar_tensor(4.0);
    let dft_len = scalar_tensor(256.0);
    let sample_rate = scalar_tensor(8_000.0);
    let lower = scalar_tensor(60.0);
    let upper = scalar_tensor(3_900.0);

    let node = dummy_node(OpKind::MelWeightMatrix);
    let ctx = make_ctx(
        &node,
        vec![
            Some(&num_mel),
            Some(&dft_len),
            Some(&sample_rate),
            Some(&lower),
            Some(&upper),
        ],
    );
    let out = MelWeightMatrixOp
        .execute(&ctx)
        .expect("MelWeightMatrix execute failed");

    assert_eq!(out.len(), 1);
    // num_spectrogram_bins = 256/2 + 1 = 129
    assert_eq!(
        out[0].shape,
        vec![129, 4],
        "MelWeightMatrix shape mismatch: {:?}",
        out[0].shape,
    );

    // All weights non-negative.
    for &v in &out[0].data {
        assert!(v >= 0.0, "negative weight: {v}");
    }

    // Each mel-bin column should accumulate some energy (sum > 0).
    // With 129 spec bins covering 0–4000 Hz and 4 mel bands from 60 Hz to
    // 3900 Hz, every band overlaps multiple bins.
    let num_spec_bins = 129usize;
    let num_mel_bins = 4usize;
    for m in 0..num_mel_bins {
        let col_sum: f32 = (0..num_spec_bins)
            .map(|s| out[0].data[s * num_mel_bins + m])
            .sum();
        assert!(
            col_sum > 0.0,
            "mel bin {m} has zero total weight (col_sum={col_sum})",
        );
    }
}

/// MelWeightMatrix with larger dimensions produces correct shape [257, 40].
#[test]
fn test_mel_weight_matrix_large_shape() {
    let num_mel = scalar_tensor(40.0);
    let dft_len = scalar_tensor(512.0);
    let sample_rate = scalar_tensor(16_000.0);
    let lower = scalar_tensor(0.0);
    let upper = scalar_tensor(8_000.0);

    let node = dummy_node(OpKind::MelWeightMatrix);
    let ctx = make_ctx(
        &node,
        vec![
            Some(&num_mel),
            Some(&dft_len),
            Some(&sample_rate),
            Some(&lower),
            Some(&upper),
        ],
    );
    let out = MelWeightMatrixOp
        .execute(&ctx)
        .expect("MelWeightMatrix large execute failed");

    // num_spectrogram_bins = 512/2 + 1 = 257
    assert_eq!(out[0].shape, vec![257, 40]);
}

// ── Bernoulli tests ───────────────────────────────────────────────────────────

/// Two runs with identical seed produce identical output.
#[test]
fn test_bernoulli_seed_reproducible() {
    let n = 100usize;
    let probs = Tensor::new(vec![0.5f32; n], vec![n]);

    // The seed is stored in attrs.floats["seed"] as a non-zero f32.
    let node = node_with_float_attrs(OpKind::Bernoulli, &[("seed", 42.0)]);

    let ctx1 = make_ctx(&node, vec![Some(&probs)]);
    let out1 = BernoulliOp.execute(&ctx1).expect("Bernoulli run 1 failed");

    let ctx2 = make_ctx(&node, vec![Some(&probs)]);
    let out2 = BernoulliOp.execute(&ctx2).expect("Bernoulli run 2 failed");

    assert_eq!(out1[0].shape, vec![n]);
    assert_eq!(out2[0].shape, vec![n]);
    assert_eq!(
        out1[0].data, out2[0].data,
        "Bernoulli with same seed should produce identical outputs",
    );
}

/// With a large input and p=0.5, ~50 % of outputs should be 1 (±5 %).
#[test]
fn test_bernoulli_probability_distribution() {
    let n = 10_000usize;
    let probs = Tensor::new(vec![0.5f32; n], vec![n]);

    // Use a time-seeded run (seed=0.0 triggers system-time seed in implementation).
    let node = dummy_node(OpKind::Bernoulli);
    let ctx = make_ctx(&node, vec![Some(&probs)]);
    let out = BernoulliOp
        .execute(&ctx)
        .expect("Bernoulli distribution test failed");

    assert_eq!(out[0].shape, vec![n]);

    // All outputs should be 0.0 or 1.0.
    for &v in &out[0].data {
        assert!(
            v == 0.0 || v == 1.0,
            "unexpected Bernoulli output value: {v}"
        );
    }

    let ones: usize = out[0].data.iter().filter(|&&v| v == 1.0).count();
    let frac = ones as f32 / n as f32;
    assert!(
        (0.45..=0.55).contains(&frac),
        "Bernoulli p=0.5 fraction of 1s out of range: {frac} (expected ~0.5)",
    );
}

/// Bernoulli with p=0.0 should always output 0.
#[test]
fn test_bernoulli_all_zero_prob() {
    let n = 50usize;
    let probs = Tensor::new(vec![0.0f32; n], vec![n]);
    let node = node_with_float_attrs(OpKind::Bernoulli, &[("seed", 7.0)]);
    let ctx = make_ctx(&node, vec![Some(&probs)]);
    let out = BernoulliOp
        .execute(&ctx)
        .expect("Bernoulli p=0 execute failed");

    for (i, &v) in out[0].data.iter().enumerate() {
        assert_eq!(v, 0.0, "Bernoulli p=0 should always output 0, idx {i}: {v}");
    }
}

/// Bernoulli with p=1.0 should always output 1.
#[test]
fn test_bernoulli_all_one_prob() {
    let n = 50usize;
    let probs = Tensor::new(vec![1.0f32; n], vec![n]);
    let node = node_with_float_attrs(OpKind::Bernoulli, &[("seed", 13.0)]);
    let ctx = make_ctx(&node, vec![Some(&probs)]);
    let out = BernoulliOp
        .execute(&ctx)
        .expect("Bernoulli p=1 execute failed");

    for (i, &v) in out[0].data.iter().enumerate() {
        assert_eq!(v, 1.0, "Bernoulli p=1 should always output 1, idx {i}: {v}");
    }
}

/// Bernoulli output shape matches input shape (pass-through shape).
#[test]
fn test_bernoulli_output_shape_matches_input() {
    // Shape [3, 4]
    let probs = Tensor::new(vec![0.3f32; 12], vec![3, 4]);
    let node = node_with_float_attrs(OpKind::Bernoulli, &[("seed", 99.0)]);
    let ctx = make_ctx(&node, vec![Some(&probs)]);
    let out = BernoulliOp
        .execute(&ctx)
        .expect("Bernoulli shape test failed");

    assert_eq!(out[0].shape, vec![3, 4], "Bernoulli output shape mismatch");
}
