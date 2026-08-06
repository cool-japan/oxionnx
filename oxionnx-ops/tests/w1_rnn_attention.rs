//! Wave-1 regression tests for the RNN / attention / spatial domain.
//!
//! Every expected value in this file was produced by an independent NumPy
//! implementation of the corresponding ONNX equations (see the comment above
//! each constant) and then inlined as `f32`.

use oxionnx_core::operator::{OpContext, Operator};
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    OnnxError, Tensor,
};
use oxionnx_ops::attention::scaled_dot_product_attention;
use oxionnx_ops::registry::rnn_ops::{AttentionOp, GridSampleOp, LSTMOp, RoiAlignOp};
use oxionnx_ops::rnn::{gru_ext, lstm_ext, simple_rnn_ext, RnnExtras};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn node(op: OpKind) -> Node {
    Node {
        name: "w1".into(),
        op,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn ctx<'a>(n: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node: n,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

#[track_caller]
fn assert_all_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length {} != expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: got {a}, expected {e} (tol {tol})"
        );
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── [a1-2] GRU linear_before_reset = 0 ──────────────────────────────────────

/// seq=2, batch=1, input=1, hidden=2 with a deliberately non-uniform `Rr` so the
/// two reset gates differ; that is exactly the case the old
/// `rt[j] * Σ_k Rh[j][k]·h[k]` indexing got wrong.
fn gru_fixture() -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    let x = Tensor::new(vec![0.7, -0.4], vec![2, 1, 1]);
    // W [1, 3*2, 1] — gate order z, r, h
    let w = Tensor::new(vec![0.5, -0.3, 0.9, 0.2, -0.6, 0.4], vec![1, 6, 1]);
    // R [1, 3*2, 2]
    let r = Tensor::new(
        vec![
            0.10, -0.20, // Rz row 0
            0.30, 0.40, // Rz row 1
            1.50, -1.20, // Rr row 0
            -0.90, 1.10, // Rr row 1
            0.25, 0.35, // Rh row 0
            -0.45, 0.55, // Rh row 1
        ],
        vec![1, 6, 2],
    );
    let b = Tensor::new(
        vec![
            0.05, -0.05, 0.10, -0.10, 0.15, -0.15, // Wbz, Wbr, Wbh
            0.02, -0.02, 0.03, -0.03, 0.04, -0.04, // Rbz, Rbr, Rbh
        ],
        vec![1, 12],
    );
    let h0 = Tensor::new(vec![0.6, -0.8], vec![1, 1, 2]);
    (x, w, r, b, h0)
}

#[test]
fn gru_linear_before_reset_zero_matches_onnx_equation() -> Result<(), OnnxError> {
    let (x, w, r, b, h0) = gru_fixture();

    // NumPy: ht = tanh(Wh·xt + (rt ⊙ h_{t-1})·Rhᵀ + Wbh + Rbh)
    const Y_LBR0: [f32; 4] = [0.34309703, -0.46389404, 0.38139734, -0.4728859];
    const YH_LBR0: [f32; 2] = [0.38139734, -0.4728859];
    // NumPy: ht = tanh(Wh·xt + rt ⊙ (h_{t-1}·Rhᵀ + Rbh) + Wbh)
    const Y_LBR1: [f32; 4] = [0.27552935, -0.32757753, 0.32270557, -0.36026663];

    let (y0, yh0) = gru_ext(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        Some(&h0),
        2,
        "forward",
        false,
        None,
        RnnExtras::default(),
    )?;
    assert_eq!(y0.shape, vec![2, 1, 1, 2]);
    assert_all_close(&y0.data, &Y_LBR0, 1e-5, "GRU lbr=0 Y");
    assert_all_close(&yh0.data, &YH_LBR0, 1e-5, "GRU lbr=0 Y_h");

    let (y1, _) = gru_ext(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        Some(&h0),
        2,
        "forward",
        true,
        None,
        RnnExtras::default(),
    )?;
    assert_all_close(&y1.data, &Y_LBR1, 1e-5, "GRU lbr=1 Y");

    // The two modes must be measurably different — the old code collapsed
    // lbr=0 into lbr=1 up to bias placement.
    assert!(
        max_abs_diff(&y0.data, &y1.data) > 0.05,
        "lbr=0 and lbr=1 must differ, got {}",
        max_abs_diff(&y0.data, &y1.data)
    );
    Ok(())
}

// ── [a1-9 / a11-13] clip ────────────────────────────────────────────────────

#[test]
fn gru_clip_clamps_activation_inputs() -> Result<(), OnnxError> {
    let (x, w, r, b, h0) = gru_fixture();
    // NumPy with np.clip(pre_activation, -0.1, 0.1)
    const Y_CLIP: [f32; 4] = [0.26764315, -0.43234026, 0.18295145, -0.26245812];

    let extras = RnnExtras {
        clip: 0.1,
        ..RnnExtras::default()
    };
    let (y, _) = gru_ext(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        Some(&h0),
        2,
        "forward",
        false,
        None,
        extras,
    )?;
    assert_all_close(&y.data, &Y_CLIP, 1e-5, "GRU clip=0.1 Y");
    Ok(())
}

/// seq=2, batch=2, input=2, hidden=2 LSTM fixture with large weights so `clip`
/// and `HardSigmoid` both bite.
fn lstm_fixture() -> (Tensor, Tensor, Tensor, Tensor) {
    let x = Tensor::new(
        vec![0.5, 1.589, 1.103, -1.099, -0.799, 1.494, -1.979, 1.285],
        vec![2, 2, 2],
    );
    let w = Tensor::new(
        vec![
            1.188, -0.128, -0.788, -0.886, -0.981, -0.22, 0.018, 0.214, 1.982, 1.171, 0.489, 1.956,
            -1.139, -1.359, 0.45, -1.824,
        ],
        vec![1, 8, 2],
    );
    let r = Tensor::new(
        vec![
            -1.857, 0.06, -0.135, 1.669, 0.517, 0.056, -0.013, -1.01, -1.953, -1.23, 0.768, -1.198,
            -0.522, -1.985, 1.32, -1.382,
        ],
        vec![1, 8, 2],
    );
    let b = Tensor::new(
        vec![
            -0.93, 1.521, 0.039, 1.389, 0.559, 0.967, -1.634, 0.165, 0.031, 1.485, -0.555, 0.393,
            -1.763, -0.449, -0.708, -1.399,
        ],
        vec![1, 16],
    );
    (x, w, r, b)
}

/// NumPy LSTM, `clip` absent.
const LSTM_Y_PLAIN: [f32; 8] = [
    -0.07351618,
    -0.5775728,
    -0.11233048,
    0.5569395,
    -0.13346821,
    -0.8537763,
    -0.03772701,
    -0.2689566,
];
const LSTM_YH_PLAIN: [f32; 4] = [-0.13346821, -0.8537763, -0.03772701, -0.2689566];

#[test]
fn lstm_clip_clamps_activation_inputs() -> Result<(), OnnxError> {
    let (x, w, r, b) = lstm_fixture();
    // NumPy with np.clip(pre_activation, -0.5, 0.5)
    const Y_CLIP: [f32; 8] = [
        -0.06520848,
        -0.17426972,
        -0.10569993,
        0.17426972,
        -0.11176171,
        -0.27109867,
        -0.17163788,
        -0.06733412,
    ];
    const YC_CLIP: [f32; 4] = [-0.24033679, -0.46669903, -0.28306726, -0.10859925];

    let (y_plain, _, _) = lstm_ext(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        2,
        "forward",
        None,
        RnnExtras::default(),
    )?;
    assert_all_close(&y_plain.data, &LSTM_Y_PLAIN, 1e-5, "LSTM plain Y");

    let extras = RnnExtras {
        clip: 0.5,
        ..RnnExtras::default()
    };
    let (y, _, y_c) = lstm_ext(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        2,
        "forward",
        None,
        extras,
    )?;
    assert_all_close(&y.data, &Y_CLIP, 1e-5, "LSTM clip=0.5 Y");
    assert_all_close(&y_c.data, &YC_CLIP, 1e-5, "LSTM clip=0.5 Y_c");
    assert!(
        max_abs_diff(&y.data, &y_plain.data) > 0.4,
        "clip must change the result"
    );
    Ok(())
}

// ── [a1-9] activation list ──────────────────────────────────────────────────

#[test]
fn lstm_hard_sigmoid_gate_activation() -> Result<(), OnnxError> {
    let (x, w, r, b) = lstm_fixture();
    // NumPy with f = HardSigmoid(alpha=0.2, beta=0.5) on i/o/f gates.
    const Y_HARDSIG: [f32; 8] = [
        -0.08659133,
        -0.5828494,
        -0.12162952,
        0.5633261,
        -0.13836867,
        -0.90681684,
        -0.0,
        -0.26774582,
    ];

    let acts = ["HardSigmoid", "Tanh", "Tanh"];
    let (y, _, _) = lstm_ext(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        2,
        "forward",
        Some(&acts),
        RnnExtras::default(),
    )?;
    assert_all_close(&y.data, &Y_HARDSIG, 1e-5, "LSTM HardSigmoid Y");
    assert!(
        max_abs_diff(&y.data, &LSTM_Y_PLAIN) > 0.01,
        "HardSigmoid must differ from Sigmoid"
    );
    Ok(())
}

#[test]
fn lstm_unknown_activation_is_rejected() {
    let (x, w, r, b) = lstm_fixture();
    let acts = ["NotAnActivation", "Tanh", "Tanh"];
    let err = lstm_ext(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        2,
        "forward",
        Some(&acts),
        RnnExtras::default(),
    )
    .expect_err("unknown activation must be rejected, not silently treated as Tanh");
    assert!(
        matches!(err, OnnxError::Unsupported(_)),
        "expected Unsupported, got {err:?}"
    );
}

#[test]
fn rnn_activation_alpha_is_applied() -> Result<(), OnnxError> {
    // h_1 = LeakyRelu(x·Wᵀ) with x = -1, W = 1 → alpha * -1.
    let x = Tensor::new(vec![-1.0], vec![1, 1, 1]);
    let w = Tensor::new(vec![1.0], vec![1, 1, 1]);
    let r = Tensor::new(vec![0.0], vec![1, 1, 1]);
    let acts = ["LeakyRelu"];

    // Default alpha = 0.01.
    let (y_default, _) = simple_rnn_ext(
        &x,
        &w,
        &r,
        None,
        None,
        None,
        1,
        "forward",
        Some(&acts),
        RnnExtras::default(),
    )?;
    assert_all_close(&y_default.data, &[-0.01], 1e-6, "LeakyRelu default alpha");

    // activation_alpha = [0.25].
    let alphas = [0.25f32];
    let extras = RnnExtras {
        activation_alpha: &alphas,
        ..RnnExtras::default()
    };
    let (y_alpha, _) = simple_rnn_ext(
        &x,
        &w,
        &r,
        None,
        None,
        None,
        1,
        "forward",
        Some(&acts),
        extras,
    )?;
    assert_all_close(&y_alpha.data, &[-0.25], 1e-6, "LeakyRelu alpha=0.25");
    Ok(())
}

// ── [a11-13] layout = 1 ─────────────────────────────────────────────────────

#[test]
fn lstm_layout_one_is_batch_major() -> Result<(), OnnxError> {
    let (_, w, r, b) = lstm_fixture();
    // Same data as `lstm_fixture` but transposed to [batch, seq, input].
    let x_bm = Tensor::new(
        vec![0.5, 1.589, -0.799, 1.494, 1.103, -1.099, -1.979, 1.285],
        vec![2, 2, 2],
    );
    // NumPy: transpose(Y[seq, dir, batch, hidden]) → [batch, seq, dir, hidden]
    const Y_LAYOUT1: [f32; 8] = [
        -0.07351618,
        -0.5775728,
        -0.13346821,
        -0.8537763,
        -0.11233048,
        0.5569395,
        -0.03772701,
        -0.2689566,
    ];

    let extras = RnnExtras {
        layout: 1,
        ..RnnExtras::default()
    };
    let (y, y_h, y_c) = lstm_ext(
        &x_bm,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        2,
        "forward",
        None,
        extras,
    )?;
    assert_eq!(y.shape, vec![2, 2, 1, 2], "layout=1 Y shape");
    assert_eq!(y_h.shape, vec![2, 1, 2], "layout=1 Y_h shape");
    assert_eq!(y_c.shape, vec![2, 1, 2], "layout=1 Y_c shape");
    assert_all_close(&y.data, &Y_LAYOUT1, 1e-5, "LSTM layout=1 Y");
    // Y_h is [batch, num_dir, hidden]; with num_dir = 1 that is the same flat
    // ordering as the layout=0 [num_dir, batch, hidden].
    assert_all_close(&y_h.data, &LSTM_YH_PLAIN, 1e-5, "LSTM layout=1 Y_h");
    Ok(())
}

#[test]
fn lstm_layout_out_of_range_is_rejected() {
    let (x, w, r, b) = lstm_fixture();
    let extras = RnnExtras {
        layout: 2,
        ..RnnExtras::default()
    };
    let err = lstm_ext(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        2,
        "forward",
        None,
        extras,
    )
    .expect_err("layout=2 must be rejected");
    assert!(
        matches!(err, OnnxError::InvalidModel(_)),
        "expected InvalidModel, got {err:?}"
    );
}

#[test]
fn lstm_op_reads_clip_and_layout_attributes() -> Result<(), OnnxError> {
    let (_, w, r, b) = lstm_fixture();
    let x_bm = Tensor::new(
        vec![0.5, 1.589, -0.799, 1.494, 1.103, -1.099, -1.979, 1.285],
        vec![2, 2, 2],
    );
    let mut n = node(OpKind::LSTM);
    n.attrs.ints.insert("hidden_size".into(), 2);
    n.attrs.ints.insert("layout".into(), 1);
    let c = ctx(&n, vec![Some(&x_bm), Some(&w), Some(&r), Some(&b)]);
    let out = LSTMOp.execute(&c)?;
    assert_eq!(out[0].shape, vec![2, 2, 1, 2], "LSTMOp layout=1 Y shape");

    // Same node, but now with clip — the result must change.
    let mut n_clip = node(OpKind::LSTM);
    n_clip.attrs.ints.insert("hidden_size".into(), 2);
    n_clip.attrs.ints.insert("layout".into(), 1);
    n_clip.attrs.floats.insert("clip".into(), 0.5);
    let c_clip = ctx(&n_clip, vec![Some(&x_bm), Some(&w), Some(&r), Some(&b)]);
    let out_clip = LSTMOp.execute(&c_clip)?;
    assert!(
        max_abs_diff(&out[0].data, &out_clip[0].data) > 0.4,
        "LSTMOp must honour the clip attribute"
    );
    Ok(())
}

#[test]
fn lstm_truncated_weights_error_instead_of_panic() {
    let (x, _, r, b) = lstm_fixture();
    // W is one row short of [1, 8, 2].
    let bad_w = Tensor::new(vec![0.0f32; 14], vec![1, 7, 2]);
    let err = lstm_ext(
        &x,
        &bad_w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        2,
        "forward",
        None,
        RnnExtras::default(),
    )
    .expect_err("a truncated W must produce a typed error");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

// ── [a1-4] SDPA mask broadcasting ───────────────────────────────────────────

fn mha_shaped_qkv() -> (Tensor, Tensor, Tensor) {
    // [batch=2, heads=2, seq=3, dim=2]
    let q = Tensor::new(
        vec![
            -0.743, -0.001, 0.203, -0.943, -0.704, 0.856, -0.859, -0.74, 0.897, 0.244, -0.262,
            0.023, 0.326, -0.449, -0.724, 0.576, 0.341, 0.025, 0.633, 0.098, 0.962, -0.591, 0.107,
            -0.033,
        ],
        vec![2, 2, 3, 2],
    );
    let k = Tensor::new(
        vec![
            -0.293, 0.183, -0.529, 0.604, 0.735, -0.742, -0.066, -0.446, -0.834, 0.792, -0.14,
            -0.705, 0.347, -0.596, 0.803, -0.566, -0.934, -0.598, -0.309, -0.062, 0.812, 0.395,
            -0.321, -0.966,
        ],
        vec![2, 2, 3, 2],
    );
    let v = Tensor::new(
        vec![
            -0.68, 0.993, -0.081, 0.382, -0.891, -0.932, 0.692, 0.176, -0.383, -0.365, -0.822,
            -0.655, -0.951, 0.678, -0.067, -0.746, 0.478, -0.609, -0.876, 0.197, 0.792, -0.946,
            0.61, -0.62,
        ],
        vec![2, 2, 3, 2],
    );
    (q, k, v)
}

#[test]
fn sdpa_mask_broadcasts_over_heads() -> Result<(), OnnxError> {
    let (q, k, v) = mha_shaped_qkv();
    // [B=2, 1, S_q=3, S_k=3]: batch 0 masks key 2, batch 1 masks key 0.
    let mut mask_data = vec![0.0f32; 18];
    for i in 0..3 {
        mask_data[i * 3 + 2] = -1e9;
        mask_data[9 + i * 3] = -1e9;
    }
    let mask = Tensor::new(mask_data, vec![2, 1, 3, 3]);

    // NumPy softmax((q·kᵀ)/√d + mask)·v
    const EXPECTED: [f32; 24] = [
        -0.3620007,
        0.66863006,
        -0.42722654,
        0.7351626,
        -0.32538238,
        0.6312782,
        0.20309344,
        -0.07004507,
        0.22755517,
        -0.05773456,
        0.11094657,
        -0.11641852,
        0.1529934,
        -0.6906989,
        0.3177904,
        -0.64927286,
        0.14918026,
        -0.6916574,
        0.72756946,
        -0.83059144,
        0.71015745,
        -0.7994029,
        0.7034548,
        -0.7873971,
    ];

    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None)?;
    assert_eq!(out.shape, vec![2, 2, 3, 2]);
    assert_all_close(&out.data, &EXPECTED, 1e-5, "SDPA [B,1,Sq,Sk] mask");

    // The old code dropped the mask for 6 of the 8 (batch, head) slices.
    let unmasked = scaled_dot_product_attention(&q, &k, &v, None, None)?;
    assert!(
        max_abs_diff(&out.data, &unmasked.data) > 0.5,
        "the mask must actually be applied to every head"
    );
    Ok(())
}

#[test]
fn sdpa_key_padding_mask_broadcasts_over_queries() -> Result<(), OnnxError> {
    let (q, k, v) = mha_shaped_qkv();
    // [B=2, 1, 1, S_k=3] key-padding mask.
    let mask = Tensor::new(vec![0.0, -1e9, 0.0, 0.0, 0.0, -1e9], vec![2, 1, 1, 3]);
    // NumPy reference.
    const EXPECTED: [f32; 24] = [
        -0.7577152,
        0.28398675,
        -0.8239649,
        -0.32042396,
        -0.73382473,
        0.5019451,
        -0.13312386,
        -0.27689162,
        -0.03034507,
        -0.2204787,
        -0.06859464,
        -0.24147302,
        -0.48789048,
        -0.06800448,
        -0.557693,
        0.04443761,
        -0.48460814,
        -0.07329185,
        0.17531027,
        -0.5234122,
        0.1900592,
        -0.53351897,
        -0.011093,
        -0.3956791,
    ];
    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None)?;
    assert_all_close(&out.data, &EXPECTED, 1e-5, "SDPA [B,1,1,Sk] mask");
    Ok(())
}

#[test]
fn sdpa_rejects_non_broadcastable_mask() {
    let (q, k, v) = mha_shaped_qkv();
    // seq_k = 3 but the mask claims 4 keys.
    let bad = Tensor::new(vec![0.0f32; 2 * 3 * 4], vec![2, 3, 4]);
    let err = scaled_dot_product_attention(&q, &k, &v, Some(&bad), None)
        .expect_err("a mask that cannot broadcast must be an error, not silently dropped");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );

    // Leading dim 3 matches neither batch (2) nor heads (2) nor 1.
    let bad_lead = Tensor::new(vec![0.0f32; 3 * 3 * 3], vec![3, 3, 3]);
    let err = scaled_dot_product_attention(&q, &k, &v, Some(&bad_lead), None)
        .expect_err("a mask with an incompatible leading dim must be an error");
    assert!(
        matches!(err, OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
}

// ── [a11-1] Attention is_causal ─────────────────────────────────────────────

#[test]
fn attention_op_honours_is_causal() -> Result<(), OnnxError> {
    let q = Tensor::new(
        vec![
            -0.829, -0.526, 0.603, 0.164, -0.812, -0.134, -0.042, -0.681, 0.469, -0.773, -0.218,
            0.033,
        ],
        vec![4, 3],
    );
    let k = Tensor::new(
        vec![
            -0.139, 0.174, 0.476, 0.913, -0.432, 0.297, 0.392, -0.415, -0.997, 0.947, -0.403,
            -0.372,
        ],
        vec![4, 3],
    );
    let v = Tensor::new(
        vec![
            0.783, 0.17, -0.057, 0.547, -0.939, 0.414, -0.252, -0.818, 0.321, 0.863, -0.586, 0.26,
        ],
        vec![4, 3],
    );
    // NumPy softmax with the strictly-upper triangle set to -inf.
    const CAUSAL: [f32; 12] = [
        0.783,
        0.17,
        -0.057,
        0.6418484,
        -0.49329308,
        0.22470517,
        0.40389317,
        -0.53748214,
        0.23194164,
        0.47773775,
        -0.48223728,
        0.20825376,
    ];
    const NON_CAUSAL: [f32; 12] = [
        0.53214043,
        -0.44635424,
        0.1956354,
        0.4549248,
        -0.6091895,
        0.2612391,
        0.515798,
        -0.5493081,
        0.2387807,
        0.47773775,
        -0.48223728,
        0.20825376,
    ];

    let mut n = node(OpKind::Attention);
    n.attrs.ints.insert("is_causal".into(), 1);
    let c = ctx(&n, vec![Some(&q), Some(&k), Some(&v)]);
    let out = AttentionOp.execute(&c)?;
    assert_eq!(out[0].shape, vec![4, 3]);
    assert_all_close(&out[0].data, &CAUSAL, 1e-5, "Attention is_causal=1");

    let n_plain = node(OpKind::Attention);
    let c_plain = ctx(&n_plain, vec![Some(&q), Some(&k), Some(&v)]);
    let out_plain = AttentionOp.execute(&c_plain)?;
    assert_all_close(&out_plain[0].data, &NON_CAUSAL, 1e-5, "Attention default");

    // Row 0 must see only key 0 under causal masking.
    assert_all_close(
        &out[0].data[..3],
        &v.data[..3],
        1e-5,
        "causal row 0 == v[0]",
    );
    Ok(())
}

#[test]
fn attention_op_is_causal_matches_explicit_mask() -> Result<(), OnnxError> {
    let q = Tensor::new(vec![0.3, -0.7, 0.2, 0.9, -0.4, 0.1], vec![3, 2]);
    let k = Tensor::new(vec![0.5, 0.1, -0.2, 0.8, 0.7, -0.6], vec![3, 2]);
    let v = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, -1.0, 2.0], vec![3, 2]);

    let mut n = node(OpKind::Attention);
    n.attrs.ints.insert("is_causal".into(), 1);
    let out = AttentionOp.execute(&ctx(&n, vec![Some(&q), Some(&k), Some(&v)]))?;

    let neg = -1.0e9f32;
    let mask = Tensor::new(
        vec![0.0, neg, neg, 0.0, 0.0, neg, 0.0, 0.0, 0.0],
        vec![3, 3],
    );
    let reference = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None)?;
    assert_all_close(
        &out[0].data,
        &reference.data,
        1e-5,
        "is_causal vs explicit lower-triangular mask",
    );
    Ok(())
}

// ── [a1-1] GridSample string attributes ─────────────────────────────────────

fn grid_sample_inputs() -> (Tensor, Tensor) {
    // 3x3 feature map holding 1..9.
    let input = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
    );
    // Four (x, y) grid points: centre, far bottom-right, far top-left, inside.
    let grid = Tensor::new(
        vec![0.0, 0.0, 2.0, 2.0, -2.0, -2.0, 0.9, -0.9],
        vec![1, 1, 4, 2],
    );
    (input, grid)
}

#[test]
fn grid_sample_reads_string_mode_and_padding_mode() -> Result<(), OnnxError> {
    let (input, grid) = grid_sample_inputs();

    // NumPy/torch reference, align_corners = 1.
    const NEAREST_BORDER: [f32; 4] = [5.0, 9.0, 1.0, 3.0];
    const NEAREST_ZEROS: [f32; 4] = [5.0, 0.0, 0.0, 3.0];
    const BILINEAR_BORDER: [f32; 4] = [5.0, 9.0, 1.0, 3.2];

    let mut n = node(OpKind::GridSample);
    n.attrs.ints.insert("align_corners".into(), 1);
    n.attrs.strings.insert("mode".into(), "nearest".into());
    n.attrs
        .strings
        .insert("padding_mode".into(), "border".into());
    let out = GridSampleOp.execute(&ctx(&n, vec![Some(&input), Some(&grid)]))?;
    assert_eq!(out[0].shape, vec![1, 1, 1, 4]);
    assert_all_close(&out[0].data, &NEAREST_BORDER, 1e-5, "nearest/border");

    let mut n = node(OpKind::GridSample);
    n.attrs.ints.insert("align_corners".into(), 1);
    n.attrs.strings.insert("mode".into(), "nearest".into());
    let out = GridSampleOp.execute(&ctx(&n, vec![Some(&input), Some(&grid)]))?;
    assert_all_close(&out[0].data, &NEAREST_ZEROS, 1e-5, "nearest/zeros");

    // Opset-20 spelling of bilinear is "linear".
    let mut n = node(OpKind::GridSample);
    n.attrs.ints.insert("align_corners".into(), 1);
    n.attrs.strings.insert("mode".into(), "linear".into());
    n.attrs
        .strings
        .insert("padding_mode".into(), "border".into());
    let out = GridSampleOp.execute(&ctx(&n, vec![Some(&input), Some(&grid)]))?;
    assert_all_close(&out[0].data, &BILINEAR_BORDER, 1e-5, "linear/border");

    // The default (bilinear/zeros) must differ from nearest/border, i.e. the
    // string attributes really changed the kernel that ran.
    let n = {
        let mut n = node(OpKind::GridSample);
        n.attrs.ints.insert("align_corners".into(), 1);
        n
    };
    let default_out = GridSampleOp.execute(&ctx(&n, vec![Some(&input), Some(&grid)]))?;
    assert!(
        max_abs_diff(&default_out[0].data, &NEAREST_BORDER) > 1.0,
        "default must not equal nearest/border"
    );
    Ok(())
}

#[test]
fn grid_sample_rejects_unknown_mode() {
    let (input, grid) = grid_sample_inputs();
    let mut n = node(OpKind::GridSample);
    n.attrs.strings.insert("mode".into(), "quintic".into());
    let err = GridSampleOp
        .execute(&ctx(&n, vec![Some(&input), Some(&grid)]))
        .expect_err("an unknown GridSample mode must be rejected");
    assert!(
        matches!(err, OnnxError::InvalidModel(_)),
        "expected InvalidModel, got {err:?}"
    );

    let mut n = node(OpKind::GridSample);
    n.attrs.strings.insert("padding_mode".into(), "wrap".into());
    let err = GridSampleOp
        .execute(&ctx(&n, vec![Some(&input), Some(&grid)]))
        .expect_err("an unknown GridSample padding_mode must be rejected");
    assert!(
        matches!(err, OnnxError::InvalidModel(_)),
        "expected InvalidModel, got {err:?}"
    );
}

// ── [a1-12] RoiAlign coordinate_transformation_mode ─────────────────────────

#[test]
fn roi_align_coordinate_transformation_mode() -> Result<(), OnnxError> {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let input = Tensor::new(data, vec![1, 1, 4, 4]);
    let rois = Tensor::new(vec![0.0, 0.0, 4.0, 4.0], vec![1, 4]);
    let batch_indices = Tensor::new(vec![0.0], vec![1]);

    // NumPy port of onnxruntime's RoiAlign for both coordinate modes.
    const HALF_PIXEL: [f32; 4] = [2.5, 4.5, 10.5, 12.5];
    const OUTPUT_HALF_PIXEL: [f32; 4] = [5.0, 6.75, 12.0, 13.75];

    let base_attrs = |n: &mut Node| {
        n.attrs.ints.insert("output_height".into(), 2);
        n.attrs.ints.insert("output_width".into(), 2);
        n.attrs.ints.insert("sampling_ratio".into(), 2);
    };

    // Absent attribute → ONNX default `half_pixel`.
    let mut n = node(OpKind::RoiAlign);
    base_attrs(&mut n);
    let out = RoiAlignOp.execute(&ctx(
        &n,
        vec![Some(&input), Some(&rois), Some(&batch_indices)],
    ))?;
    assert_eq!(out[0].shape, vec![1, 1, 2, 2]);
    assert_all_close(&out[0].data, &HALF_PIXEL, 1e-5, "RoiAlign default");

    let mut n = node(OpKind::RoiAlign);
    base_attrs(&mut n);
    n.attrs
        .strings
        .insert("coordinate_transformation_mode".into(), "half_pixel".into());
    let out = RoiAlignOp.execute(&ctx(
        &n,
        vec![Some(&input), Some(&rois), Some(&batch_indices)],
    ))?;
    assert_all_close(&out[0].data, &HALF_PIXEL, 1e-5, "RoiAlign half_pixel");

    let mut n = node(OpKind::RoiAlign);
    base_attrs(&mut n);
    n.attrs.strings.insert(
        "coordinate_transformation_mode".into(),
        "output_half_pixel".into(),
    );
    let out = RoiAlignOp.execute(&ctx(
        &n,
        vec![Some(&input), Some(&rois), Some(&batch_indices)],
    ))?;
    assert_all_close(
        &out[0].data,
        &OUTPUT_HALF_PIXEL,
        1e-5,
        "RoiAlign output_half_pixel",
    );
    Ok(())
}

#[test]
fn roi_align_rejects_out_of_range_batch_index() {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let input = Tensor::new(data, vec![1, 1, 4, 4]);
    let rois = Tensor::new(vec![0.0, 0.0, 4.0, 4.0], vec![1, 4]);
    let batch_indices = Tensor::new(vec![7.0], vec![1]);

    let mut n = node(OpKind::RoiAlign);
    n.attrs.ints.insert("output_height".into(), 2);
    n.attrs.ints.insert("output_width".into(), 2);
    let err = RoiAlignOp
        .execute(&ctx(
            &n,
            vec![Some(&input), Some(&rois), Some(&batch_indices)],
        ))
        .expect_err("an out-of-range batch index must be an error, not a panic");
    assert!(
        matches!(err, OnnxError::InvalidModel(_)),
        "expected InvalidModel, got {err:?}"
    );
}
