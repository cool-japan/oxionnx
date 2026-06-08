//! Unit tests for all DSP operators.

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::{OpContext, Tensor};

use super::bernoulli::BernoulliOp;
use super::dft::DFTOp;
use super::mel::MelWeightMatrixOp;
use super::stft::STFTOp;
use super::window::{BlackmanWindowOp, HammingWindowOp, HannWindowOp};
use oxionnx_core::operator::Operator;

fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
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

fn node_with_attrs(op: OpKind, periodic: i64) -> Node {
    let mut n = dummy_node(op);
    n.attrs.ints.insert("periodic".into(), periodic);
    n
}

// ── Window function tests ─────────────────────────────────────────────────────

#[test]
fn hann_window_periodic_8() {
    let size_t = Tensor::new(vec![8.0], vec![1]);
    let node = dummy_node(OpKind::HannWindow);
    let ctx = make_ctx(&node, vec![Some(&size_t)]);
    let out = HannWindowOp.execute(&ctx).expect("HannWindow failed");
    assert_eq!(out[0].shape, vec![8]);
    // First and last samples of a periodic Hann window should both be ~0.
    assert!((out[0].data[0]).abs() < 1e-6, "w[0] should be ~0");
}

#[test]
fn hamming_window_symmetric_4() {
    let size_t = Tensor::new(vec![4.0], vec![1]);
    let node = node_with_attrs(OpKind::HammingWindow, 0); // symmetric
    let ctx = make_ctx(&node, vec![Some(&size_t)]);
    let out = HammingWindowOp.execute(&ctx).expect("HammingWindow failed");
    assert_eq!(out[0].shape, vec![4]);
    // Symmetric Hamming with N=4, denom = N-1 = 3.
    // w[0] = alpha - beta*cos(0) = 0.54347826 - 0.45652174 = 0.08695652
    let w0 = out[0].data[0];
    let expected = 0.543_478_26_f32 - 0.456_521_74_f32;
    assert!(
        (w0 - expected).abs() < 1e-4,
        "w[0]={w0} expected={expected}"
    );
    // w[1]: 0.54347826 - 0.45652174*cos(2pi/3) ≈ 0.5435 + 0.2283 ≈ 0.7717
    let w1 = out[0].data[1];
    let expected1 =
        0.543_478_26_f32 - 0.456_521_74_f32 * (2.0 * std::f32::consts::PI / 3.0_f32).cos();
    assert!(
        (w1 - expected1).abs() < 1e-4,
        "w[1]={w1} expected={expected1}"
    );
}

#[test]
fn blackman_window_shape() {
    let size_t = Tensor::new(vec![16.0], vec![1]);
    let node = dummy_node(OpKind::BlackmanWindow);
    let ctx = make_ctx(&node, vec![Some(&size_t)]);
    let out = BlackmanWindowOp
        .execute(&ctx)
        .expect("BlackmanWindow failed");
    assert_eq!(out[0].shape, vec![16]);
}

// ── DFT tests ─────────────────────────────────────────────────────────────────

#[test]
fn dft_real_signal_dc() {
    // A constant real signal of all 1s → DC bin should equal N, others ~0.
    let n = 8usize;
    let data: Vec<f32> = vec![1.0; n];
    let input = Tensor::new(data, vec![1, n]);
    let node = dummy_node(OpKind::DFT);
    let ctx = make_ctx(&node, vec![Some(&input), None]);
    let out = DFTOp.execute(&ctx).expect("DFT failed");
    // Shape: [1, 8, 2]
    assert_eq!(out[0].shape, vec![1, n, 2]);
    let re_dc = out[0].data[0];
    let im_dc = out[0].data[1];
    assert!((re_dc - n as f32).abs() < 1e-3, "DC re={re_dc}");
    assert!(im_dc.abs() < 1e-3, "DC im={im_dc}");
}

#[test]
fn dft_roundtrip() {
    // Forward DFT then inverse DFT should recover original signal.
    let n = 8usize;
    let orig: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input = Tensor::new(orig.clone(), vec![1, n]);
    let node = dummy_node(OpKind::DFT);
    let ctx = make_ctx(&node, vec![Some(&input), None]);
    let fwd = DFTOp.execute(&ctx).expect("DFT forward failed");

    // fwd[0]: shape [1, 8, 2] — treat as complex input for inverse.
    // The inverse DFT input must be provided as complex (last dim = 2).
    let spectrum = &fwd[0];

    let mut inv_node = dummy_node(OpKind::DFT);
    inv_node.attrs.ints.insert("inverse".into(), 1);
    let ctx2 = make_ctx(&inv_node, vec![Some(spectrum), None]);
    let back = DFTOp.execute(&ctx2).expect("IDFT failed");
    // Shape: [1, 8, 2]
    assert_eq!(back[0].shape, vec![1, n, 2]);
    for (i, &expected) in orig.iter().enumerate() {
        let re = back[0].data[i * 2];
        assert!(
            (re - expected).abs() < 1e-3,
            "sample {i}: re={re} expected={expected}"
        );
    }
}

#[test]
fn dft_onesided() {
    let n = 8usize;
    let data: Vec<f32> = vec![1.0; n];
    let input = Tensor::new(data, vec![1, n]);
    let mut node = dummy_node(OpKind::DFT);
    node.attrs.ints.insert("onesided".into(), 1);
    let ctx = make_ctx(&node, vec![Some(&input), None]);
    let out = DFTOp.execute(&ctx).expect("DFT onesided failed");
    // onesided → n/2+1 = 5 bins
    assert_eq!(out[0].shape, vec![1, 5, 2]);
}

// ── STFT tests ─────────────────────────────────────────────────────────────────

#[test]
fn stft_basic_shape() {
    // Signal: [1, 16], frame_step=4, window=[8] (all ones), frame_length=8, onesided=1
    let signal = Tensor::new(vec![1.0f32; 16], vec![1, 16]);
    let frame_step_t = Tensor::new(vec![4.0], vec![1]);
    let window = Tensor::new(vec![1.0f32; 8], vec![8]);
    let frame_length_t = Tensor::new(vec![8.0], vec![1]);
    let mut node = dummy_node(OpKind::STFT);
    node.attrs.ints.insert("onesided".into(), 1);
    let ctx = make_ctx(
        &node,
        vec![
            Some(&signal),
            Some(&frame_step_t),
            Some(&window),
            Some(&frame_length_t),
        ],
    );
    let out = STFTOp.execute(&ctx).expect("STFT failed");
    // n_frames = (16 - 8) / 4 + 1 = 3
    // n_dft    = 8/2 + 1 = 5  (onesided)
    assert_eq!(out[0].shape, vec![1, 3, 5, 2]);
}

// ── MelWeightMatrix tests ──────────────────────────────────────────────────────

#[test]
fn mel_weight_matrix_shape() {
    let num_mel = Tensor::new(vec![40.0], vec![1]);
    let dft_len = Tensor::new(vec![512.0], vec![1]);
    let sample_rate = Tensor::new(vec![16000.0], vec![1]);
    let lower = Tensor::new(vec![0.0], vec![1]);
    let upper = Tensor::new(vec![8000.0], vec![1]);
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
        .expect("MelWeightMatrix failed");
    // num_spectrogram_bins = 512/2+1 = 257
    assert_eq!(out[0].shape, vec![257, 40]);
    // All weights should be non-negative.
    for &v in &out[0].data {
        assert!(v >= 0.0, "negative weight: {v}");
    }
}

#[test]
fn mel_weight_matrix_non_negative_and_bounded() {
    let num_mel = Tensor::new(vec![20.0], vec![1]);
    let dft_len = Tensor::new(vec![256.0], vec![1]);
    let sample_rate = Tensor::new(vec![8000.0], vec![1]);
    let lower = Tensor::new(vec![80.0], vec![1]);
    let upper = Tensor::new(vec![3000.0], vec![1]);
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
        .expect("MelWeightMatrix failed");
    for &v in &out[0].data {
        assert!(
            (0.0..=(1.0 + f32::EPSILON)).contains(&v),
            "weight out of [0,1]: {v}"
        );
    }
}

// ── Bernoulli tests ───────────────────────────────────────────────────────────

#[test]
fn bernoulli_always_zero() {
    // Probabilities all 0.0 → all samples should be 0.
    let probs = Tensor::new(vec![0.0f32; 100], vec![100]);
    let mut node = dummy_node(OpKind::Bernoulli);
    node.attrs.floats.insert("seed".into(), 42.0);
    let ctx = make_ctx(&node, vec![Some(&probs)]);
    let out = BernoulliOp.execute(&ctx).expect("Bernoulli failed");
    assert_eq!(out[0].shape, vec![100]);
    for &v in &out[0].data {
        assert_eq!(v, 0.0, "expected 0, got {v}");
    }
}

#[test]
fn bernoulli_always_one() {
    // Probabilities all 1.0 → all samples should be 1.
    let probs = Tensor::new(vec![1.0f32; 100], vec![100]);
    let mut node = dummy_node(OpKind::Bernoulli);
    node.attrs.floats.insert("seed".into(), 42.0);
    let ctx = make_ctx(&node, vec![Some(&probs)]);
    let out = BernoulliOp.execute(&ctx).expect("Bernoulli failed");
    for &v in &out[0].data {
        assert_eq!(v, 1.0, "expected 1, got {v}");
    }
}

#[test]
fn bernoulli_seeded_reproducible() {
    // Same seed → same output.
    let probs = Tensor::new(vec![0.5f32; 50], vec![50]);
    let mut node = dummy_node(OpKind::Bernoulli);
    node.attrs.floats.insert("seed".into(), 7.0);

    let ctx1 = make_ctx(&node, vec![Some(&probs)]);
    let out1 = BernoulliOp.execute(&ctx1).expect("Bernoulli run1 failed");

    let ctx2 = make_ctx(&node, vec![Some(&probs)]);
    let out2 = BernoulliOp.execute(&ctx2).expect("Bernoulli run2 failed");

    assert_eq!(
        out1[0].data, out2[0].data,
        "Seeded Bernoulli not reproducible"
    );
}

#[test]
fn bernoulli_shape_preserved() {
    let probs = Tensor::new(vec![0.5f32; 12], vec![3, 4]);
    let node = dummy_node(OpKind::Bernoulli);
    let ctx = make_ctx(&node, vec![Some(&probs)]);
    let out = BernoulliOp.execute(&ctx).expect("Bernoulli failed");
    assert_eq!(out[0].shape, vec![3, 4]);
}
