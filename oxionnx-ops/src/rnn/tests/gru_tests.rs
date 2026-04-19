//! GRU kernel unit tests.

use oxionnx_core::{OnnxError, Tensor};

use crate::rnn::gru;

// ── GRU tests ────────────────────────────────────────────────────────────────

/// Single timestep GRU with hand-computed values.
#[test]
fn test_gru_forward_hand_computed() -> Result<(), OnnxError> {
    let batch = 1;
    let input_size = 3;
    let hidden_size = 4;
    let seq_len = 1;

    let x = Tensor::new(vec![0.5, -0.3, 0.8], vec![seq_len, batch, input_size]);

    let mut w_data = vec![0.0f32; 12 * 3];
    for g in 0..3 {
        for j in 0..hidden_size {
            w_data[(g * hidden_size + j) * input_size + (j % input_size)] = 0.1;
        }
    }
    let w = Tensor::new(w_data.clone(), vec![1, 12, 3]);
    let r = Tensor::new(vec![0.02f32; 12 * 4], vec![1, 12, 4]);
    let b = Tensor::new(vec![0.0f32; 24], vec![1, 24]);

    let (y, y_h) = gru(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        hidden_size,
        "forward",
        false,
        None,
    )?;

    assert_eq!(y.shape, vec![1, 1, 1, 4]);
    assert_eq!(y_h.shape, vec![1, 1, 4]);

    // Hand-compute: z_t = sigmoid(wx_z), ht_cand = tanh(wx_h), h = (1-z)*h_cand
    let x_vals = [0.5f32, -0.3, 0.8];
    let mut wx = [0.0f32; 12];
    for row in 0..12 {
        for k in 0..3 {
            wx[row] += x_vals[k] * w_data[row * 3 + k];
        }
    }
    for j in 0..hidden_size {
        let z_val = 1.0 / (1.0 + (-wx[j]).exp());
        let h_cand = wx[2 * hidden_size + j].tanh();
        let h_expected = (1.0 - z_val) * h_cand;
        assert!(
            (y_h.data[j] - h_expected).abs() < 1e-5,
            "h[{j}]: expected {h_expected}, got {}",
            y_h.data[j]
        );
    }

    for (a, b_val) in y_h.data.iter().zip(y.data.iter()) {
        assert!((a - b_val).abs() < 1e-6);
    }

    Ok(())
}

/// Bidirectional GRU: shapes + both directions non-zero.
#[test]
fn test_gru_bidirectional() -> Result<(), OnnxError> {
    let seq_len = 3;
    let batch = 2;
    let input_size = 2;
    let hidden_size = 4;

    let x = Tensor::new(
        vec![0.1f32; seq_len * batch * input_size],
        vec![seq_len, batch, input_size],
    );
    let w = Tensor::new(vec![0.1f32; 2 * 12 * 2], vec![2, 12, 2]);
    let r = Tensor::new(vec![0.02f32; 2 * 12 * 4], vec![2, 12, 4]);

    let (y, y_h) = gru(
        &x,
        &w,
        &r,
        None,
        None,
        None,
        hidden_size,
        "bidirectional",
        false,
        None,
    )?;

    assert_eq!(y.shape, vec![3, 2, 2, 4]);
    assert_eq!(y_h.shape, vec![2, 2, 4]);

    let bh = batch * hidden_size;
    let dir0: f32 = y_h.data[..bh].iter().map(|v| v.abs()).sum();
    let dir1: f32 = y_h.data[bh..].iter().map(|v| v.abs()).sum();
    assert!(dir0 > 1e-6, "GRU fwd should be non-zero");
    assert!(dir1 > 1e-6, "GRU bwd should be non-zero");

    Ok(())
}

/// GRU with sequence_lens: variable-length masking.
#[test]
fn test_gru_sequence_lens() -> Result<(), OnnxError> {
    let seq_len = 4;
    let batch = 2;
    let input_size = 2;
    let hidden_size = 3;

    let x = Tensor::new(
        vec![0.1f32; seq_len * batch * input_size],
        vec![seq_len, batch, input_size],
    );
    let w = Tensor::new(vec![0.1f32; 9 * 2], vec![1, 9, 2]);
    let r = Tensor::new(vec![0.02f32; 9 * 3], vec![1, 9, 3]);

    // batch[0] length=1, batch[1] length=4
    let sl = Tensor::new(vec![1.0, 4.0], vec![2]);

    let (y, y_h) = gru(
        &x,
        &w,
        &r,
        None,
        Some(&sl),
        None,
        hidden_size,
        "forward",
        false,
        None,
    )?;

    // batch[0]: t>=1 → zero
    let bh = batch * hidden_size;
    for t in 1..seq_len {
        let off = t * bh;
        for j in 0..hidden_size {
            assert!(
                y.data[off + j].abs() < 1e-10,
                "gru batch[0] t={t} j={j} should be zero"
            );
        }
    }

    // Y_h[batch0] matches Y[t=0, batch0]
    let y_t0_b0 = &y.data[..hidden_size];
    let yh_b0 = &y_h.data[..hidden_size];
    for j in 0..hidden_size {
        assert!((y_t0_b0[j] - yh_b0[j]).abs() < 1e-6);
    }

    Ok(())
}

/// GRU: linear_before_reset modes.
#[test]
fn test_gru_linear_before_reset() {
    let seq_len = 2;
    let batch = 1;
    let input_size = 2;
    let hidden_size = 3;

    let x = Tensor::new(vec![0.5, -0.3, 0.2, 0.1], vec![seq_len, batch, input_size]);
    let w = Tensor::new(vec![0.1f32; 9 * 2], vec![1, 9, 2]);
    let r = Tensor::new(vec![0.05f32; 9 * 3], vec![1, 9, 3]);
    let b = Tensor::new(vec![0.0f32; 18], vec![1, 18]);

    let (y1, _) = gru(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        hidden_size,
        "forward",
        false,
        None,
    )
    .expect("GRU lbr=false");
    let (y2, _) = gru(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        hidden_size,
        "forward",
        true,
        None,
    )
    .expect("GRU lbr=true");

    assert_eq!(y1.shape, y2.shape);
    // With zero init, rt * 0 = 0 → same
    let diff: f32 = y1
        .data
        .iter()
        .zip(&y2.data)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff < 1e-5,
        "with zero init, lbr shouldn't matter, diff={diff}"
    );
}
