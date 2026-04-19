//! Edge case tests for LSTM, GRU, and simple RNN kernels.

use oxionnx_core::{OnnxError, Tensor};

use crate::rnn::{gru, lstm, simple_rnn};

// ── Edge cases ────────────────────────────────────────────────────────────────

/// Minimal dims: batch=1, seq=1, hidden=1.
#[test]
fn test_edge_unit_dims() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0], vec![1, 1, 1]);

    // LSTM
    let w_l = Tensor::new(vec![0.5; 4], vec![1, 4, 1]);
    let r_l = Tensor::new(vec![0.1; 4], vec![1, 4, 1]);
    let b_l = Tensor::new(vec![0.0; 8], vec![1, 8]);
    let (y_l, yh_l, yc_l) = lstm(
        &x,
        &w_l,
        &r_l,
        Some(&b_l),
        None,
        None,
        None,
        None,
        1,
        "forward",
        None,
    )?;
    assert_eq!(y_l.shape, vec![1, 1, 1, 1]);
    assert_eq!(yh_l.shape, vec![1, 1, 1]);
    assert_eq!(yc_l.shape, vec![1, 1, 1]);
    assert!(yh_l.data[0].is_finite());
    assert!(yh_l.data[0].abs() > 1e-10);

    // GRU
    let w_g = Tensor::new(vec![0.5; 3], vec![1, 3, 1]);
    let r_g = Tensor::new(vec![0.1; 3], vec![1, 3, 1]);
    let b_g = Tensor::new(vec![0.0; 6], vec![1, 6]);
    let (y_g, yh_g) = gru(
        &x,
        &w_g,
        &r_g,
        Some(&b_g),
        None,
        None,
        1,
        "forward",
        false,
        None,
    )?;
    assert_eq!(y_g.shape, vec![1, 1, 1, 1]);
    assert_eq!(yh_g.shape, vec![1, 1, 1]);
    assert!(yh_g.data[0].is_finite());

    // Simple RNN
    let w_r = Tensor::new(vec![0.5], vec![1, 1, 1]);
    let r_r = Tensor::new(vec![0.1], vec![1, 1, 1]);
    let b_r = Tensor::new(vec![0.0; 2], vec![1, 2]);
    let (y_r, yh_r) = simple_rnn(&x, &w_r, &r_r, Some(&b_r), None, None, 1, "forward", "Tanh")?;
    assert_eq!(y_r.shape, vec![1, 1, 1, 1]);
    assert_eq!(yh_r.shape, vec![1, 1, 1]);
    assert!(yh_r.data[0].is_finite());

    Ok(())
}

/// sequence_lens=0 → all zeros.
#[test]
fn test_sequence_lens_zero() -> Result<(), OnnxError> {
    let seq_len = 3;
    let batch = 1;
    let input_size = 2;
    let hidden_size = 2;

    let x = Tensor::new(
        vec![1.0f32; seq_len * batch * input_size],
        vec![seq_len, batch, input_size],
    );
    let w = Tensor::new(vec![0.1f32; 8 * 2], vec![1, 8, 2]);
    let r = Tensor::new(vec![0.02f32; 8 * 2], vec![1, 8, 2]);
    let sl = Tensor::new(vec![0.0], vec![1]);

    let (y, y_h, y_c) = lstm(
        &x,
        &w,
        &r,
        None,
        Some(&sl),
        None,
        None,
        None,
        hidden_size,
        "forward",
        None,
    )?;

    for v in &y.data {
        assert!(v.abs() < 1e-10, "Y should be zero when seq_lens=0");
    }
    for v in &y_h.data {
        assert!(v.abs() < 1e-10, "Y_h should be zero when seq_lens=0");
    }
    for v in &y_c.data {
        assert!(v.abs() < 1e-10, "Y_c should be zero when seq_lens=0");
    }

    Ok(())
}

/// LSTM with non-zero initial states.
#[test]
fn test_lstm_initial_state() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![0.0; 2], vec![1, 1, 2]);
    let w = Tensor::new(vec![0.0f32; 8 * 2], vec![1, 8, 2]);
    let r = Tensor::new(vec![0.1f32; 8 * 2], vec![1, 8, 2]);

    let h0 = Tensor::new(vec![0.5, -0.3], vec![1, 1, 2]);
    let c0 = Tensor::new(vec![1.0, -0.5], vec![1, 1, 2]);

    let (_, y_h, _) = lstm(
        &x,
        &w,
        &r,
        None,
        None,
        Some(&h0),
        Some(&c0),
        None,
        2,
        "forward",
        None,
    )?;

    let (_, yh_zero, _) = lstm(&x, &w, &r, None, None, None, None, None, 2, "forward", None)?;

    let diff: f32 = y_h
        .data
        .iter()
        .zip(&yh_zero.data)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-5, "non-zero init should differ, diff={diff}");

    Ok(())
}

/// Bidirectional LSTM with sequence_lens.
#[test]
fn test_lstm_bidirectional_sequence_lens() -> Result<(), OnnxError> {
    let seq_len = 4;
    let batch = 2;
    let input_size = 2;
    let hidden_size = 2;

    let x = Tensor::new(
        vec![0.1f32; seq_len * batch * input_size],
        vec![seq_len, batch, input_size],
    );
    let w = Tensor::new(vec![0.1f32; 2 * 8 * 2], vec![2, 8, 2]);
    let r = Tensor::new(vec![0.02f32; 2 * 8 * 2], vec![2, 8, 2]);
    let sl = Tensor::new(vec![2.0, 4.0], vec![2]);

    let (y, y_h, y_c) = lstm(
        &x,
        &w,
        &r,
        None,
        Some(&sl),
        None,
        None,
        None,
        hidden_size,
        "bidirectional",
        None,
    )?;

    assert_eq!(y.shape, vec![4, 2, 2, 2]);
    assert_eq!(y_h.shape, vec![2, 2, 2]);
    assert_eq!(y_c.shape, vec![2, 2, 2]);

    let num_dir = 2;
    let bh = batch * hidden_size;

    // Forward (d=0): batch[0] t>=2 → zero
    for t in 2..seq_len {
        let off = t * num_dir * bh;
        for j in 0..hidden_size {
            assert!(
                y.data[off + j].abs() < 1e-10,
                "bidir fwd batch[0] t={t} j={j} should be 0"
            );
        }
    }

    // Reverse (d=1): batch[0] t>=2 → zero
    for t in 2..seq_len {
        let off = t * num_dir * bh + bh;
        for j in 0..hidden_size {
            assert!(
                y.data[off + j].abs() < 1e-10,
                "bidir rev batch[0] t={t} j={j} should be 0"
            );
        }
    }

    Ok(())
}
