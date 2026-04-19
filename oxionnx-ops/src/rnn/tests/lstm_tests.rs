//! LSTM kernel unit tests.

use oxionnx_core::{OnnxError, Tensor};

use crate::rnn::lstm;

// ── LSTM tests ──────────────────────────────────────────────────────────────

/// Single timestep LSTM with hand-computed gate values.
#[test]
fn test_lstm_forward_hand_computed() -> Result<(), OnnxError> {
    let batch = 1;
    let input_size = 3;
    let hidden_size = 4;
    let seq_len = 1;

    let x = Tensor::new(vec![0.5, -0.3, 0.8], vec![seq_len, batch, input_size]);

    // W: [1, 16, 3] – sparse diagonal-like pattern
    let mut w_data = vec![0.0f32; 16 * 3];
    for g in 0..4 {
        for j in 0..hidden_size {
            w_data[(g * hidden_size + j) * input_size + (j % input_size)] = 0.1;
        }
    }
    let w = Tensor::new(w_data.clone(), vec![1, 16, 3]);
    let r = Tensor::new(vec![0.02f32; 16 * 4], vec![1, 16, 4]);
    let b = Tensor::new(vec![0.0f32; 32], vec![1, 32]);

    let (y, y_h, y_c) = lstm(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        hidden_size,
        "forward",
        None,
    )?;

    assert_eq!(y.shape, vec![1, 1, 1, 4]);
    assert_eq!(y_h.shape, vec![1, 1, 4]);
    assert_eq!(y_c.shape, vec![1, 1, 4]);

    // Hand-compute: zero init h/c, x = [0.5, -0.3, 0.8]
    let x_vals = [0.5f32, -0.3, 0.8];
    let mut wx = [0.0f32; 16];
    for row in 0..16 {
        for k in 0..3 {
            wx[row] += x_vals[k] * w_data[row * 3 + k];
        }
    }
    for j in 0..hidden_size {
        let i_val = 1.0 / (1.0 + (-wx[j]).exp());
        let o_val = 1.0 / (1.0 + (-wx[hidden_size + j]).exp());
        let f_val = 1.0 / (1.0 + (-wx[2 * hidden_size + j]).exp());
        let c_cand = wx[3 * hidden_size + j].tanh();
        let c_new = f_val * 0.0 + i_val * c_cand;
        let h_new = o_val * c_new.tanh();

        assert!(
            (y_h.data[j] - h_new).abs() < 1e-5,
            "h[{j}]: expected {h_new}, got {}",
            y_h.data[j]
        );
        assert!(
            (y_c.data[j] - c_new).abs() < 1e-5,
            "c[{j}]: expected {c_new}, got {}",
            y_c.data[j]
        );
    }

    // y_h must match Y[0]
    for (a, b_val) in y_h.data.iter().zip(y.data.iter()) {
        assert!((a - b_val).abs() < 1e-6);
    }

    Ok(())
}

/// Multi-timestep LSTM: evolving state + shapes.
#[test]
fn test_lstm_multi_step() -> Result<(), OnnxError> {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 4;
    let seq_len = 5;

    let x = Tensor::new(
        vec![0.1f32; seq_len * batch * input_size],
        vec![seq_len, batch, input_size],
    );
    let w = Tensor::new(vec![0.05f32; 16 * input_size], vec![1, 16, input_size]);
    let r = Tensor::new(vec![0.02f32; 16 * hidden_size], vec![1, 16, hidden_size]);

    let (y, y_h, y_c) = lstm(
        &x,
        &w,
        &r,
        None,
        None,
        None,
        None,
        None,
        hidden_size,
        "forward",
        None,
    )?;

    assert_eq!(y.shape, vec![seq_len, 1, batch, hidden_size]);
    assert_eq!(y_h.shape, vec![1, batch, hidden_size]);
    assert_eq!(y_c.shape, vec![1, batch, hidden_size]);

    // Last timestep of Y must match Y_h
    let bh = batch * hidden_size;
    let last_off = (seq_len - 1) * bh;
    for i in 0..bh {
        assert!((y.data[last_off + i] - y_h.data[i]).abs() < 1e-6);
    }

    // States must evolve over time
    let diff: f32 = y.data[0..bh]
        .iter()
        .zip(&y.data[last_off..last_off + bh])
        .map(|(a, b_val)| (a - b_val).abs())
        .sum();
    assert!(diff > 1e-7, "states should evolve, diff={diff}");

    Ok(())
}

/// Bidirectional LSTM: shapes and both directions produce output.
#[test]
fn test_lstm_bidirectional() -> Result<(), OnnxError> {
    let seq_len = 3;
    let batch = 2;
    let input_size = 4;
    let hidden_size = 5;

    let x = Tensor::new(
        vec![0.1f32; seq_len * batch * input_size],
        vec![seq_len, batch, input_size],
    );
    let w = Tensor::new(vec![0.01f32; 2 * 20 * 4], vec![2, 20, 4]);
    let r = Tensor::new(vec![0.01f32; 2 * 20 * 5], vec![2, 20, 5]);

    let (y, y_h, y_c) = lstm(
        &x,
        &w,
        &r,
        None,
        None,
        None,
        None,
        None,
        hidden_size,
        "bidirectional",
        None,
    )?;

    assert_eq!(y.shape, vec![3, 2, 2, 5]);
    assert_eq!(y_h.shape, vec![2, 2, 5]);
    assert_eq!(y_c.shape, vec![2, 2, 5]);

    // Both directions should have non-zero output
    let bh = batch * hidden_size;
    let dir0_sum: f32 = y_h.data[..bh].iter().map(|v| v.abs()).sum();
    let dir1_sum: f32 = y_h.data[bh..].iter().map(|v| v.abs()).sum();
    assert!(dir0_sum > 1e-6, "forward dir should be non-zero");
    assert!(dir1_sum > 1e-6, "reverse dir should be non-zero");

    Ok(())
}

/// LSTM with sequence_lens: variable-length masking.
#[test]
fn test_lstm_sequence_lens() -> Result<(), OnnxError> {
    let seq_len = 4;
    let batch = 2;
    let input_size = 2;
    let hidden_size = 3;

    // Distinct input per timestep
    let mut x_data = Vec::new();
    for t in 0..seq_len {
        for b in 0..batch {
            for _i in 0..input_size {
                x_data.push(0.1 * (t as f32 + 1.0) + 0.01 * (b as f32));
            }
        }
    }
    let x = Tensor::new(x_data, vec![seq_len, batch, input_size]);
    let w = Tensor::new(vec![0.1f32; 12 * 2], vec![1, 12, 2]);
    let r = Tensor::new(vec![0.02f32; 12 * 3], vec![1, 12, 3]);

    // batch[0] length=2, batch[1] length=4 (full)
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
        "forward",
        None,
    )?;
    assert_eq!(y.shape, vec![4, 1, 2, 3]);

    // batch[0]: Y[t>=2] must be zero
    let bh = batch * hidden_size;
    for t in 2..seq_len {
        let off = t * bh;
        for j in 0..hidden_size {
            assert!(
                y.data[off + j].abs() < 1e-10,
                "batch[0] t={t} j={j} should be zero, got {}",
                y.data[off + j]
            );
        }
    }

    // batch[1] last valid step should be non-zero
    let off = 3 * bh + hidden_size;
    let sum: f32 = y.data[off..off + hidden_size].iter().map(|v| v.abs()).sum();
    assert!(sum > 1e-7, "batch[1] t=3 should be non-zero");

    // Y_h[batch0] must match Y[t=1, batch0]
    let y_t1_b0 = &y.data[1 * bh..1 * bh + hidden_size];
    let yh_b0 = &y_h.data[..hidden_size];
    for j in 0..hidden_size {
        assert!(
            (y_t1_b0[j] - yh_b0[j]).abs() < 1e-6,
            "Y_h[b=0] should match Y[t=1,b=0], j={j}"
        );
    }

    // Y_h[batch1] must match Y[t=3, batch1]
    let y_t3_b1 = &y.data[3 * bh + hidden_size..3 * bh + 2 * hidden_size];
    let yh_b1 = &y_h.data[hidden_size..2 * hidden_size];
    for j in 0..hidden_size {
        assert!(
            (y_t3_b1[j] - yh_b1[j]).abs() < 1e-6,
            "Y_h[b=1] should match Y[t=3,b=1], j={j}"
        );
    }

    assert_eq!(y_c.shape, vec![1, 2, 3]);

    // Also test with reverse direction
    let (y_rev, _, _) = lstm(
        &x,
        &w,
        &r,
        None,
        Some(&sl),
        None,
        None,
        None,
        hidden_size,
        "reverse",
        None,
    )?;
    // batch[0] reverse: t>=2 should be zero
    for t in 2..seq_len {
        let off = t * bh;
        for j in 0..hidden_size {
            assert!(
                y_rev.data[off + j].abs() < 1e-10,
                "reverse batch[0] t={t} j={j} should be zero"
            );
        }
    }

    Ok(())
}

/// LSTM with peephole connections.
#[test]
fn test_lstm_peephole() -> Result<(), OnnxError> {
    let seq_len = 2;
    let batch = 1;
    let input_size = 2;
    let hidden_size = 3;

    let x = Tensor::new(vec![0.5, -0.3, 0.2, 0.1], vec![seq_len, batch, input_size]);
    let w = Tensor::new(vec![0.1f32; 12 * 2], vec![1, 12, 2]);
    let r = Tensor::new(vec![0.02f32; 12 * 3], vec![1, 12, 3]);
    let b = Tensor::new(vec![0.0f32; 24], vec![1, 24]);

    // Without peephole
    let (_, y_h_no_p, y_c_no_p) = lstm(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        hidden_size,
        "forward",
        None,
    )?;

    // With peephole: P = [1, 9] (3*hidden_size)
    let p = Tensor::new(vec![0.1f32; 9], vec![1, 9]);
    let (_, y_h_p, y_c_p) = lstm(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        Some(&p),
        hidden_size,
        "forward",
        None,
    )?;

    // After step 0 c != 0, so step 1 peephole kicks in → results differ.
    let diff_h: f32 = y_h_p
        .data
        .iter()
        .zip(&y_h_no_p.data)
        .map(|(a, b_val)| (a - b_val).abs())
        .sum();
    let diff_c: f32 = y_c_p
        .data
        .iter()
        .zip(&y_c_no_p.data)
        .map(|(a, b_val)| (a - b_val).abs())
        .sum();
    assert!(diff_h > 1e-6, "peephole should change h, diff={diff_h}");
    assert!(diff_c > 1e-6, "peephole should change c, diff={diff_c}");

    // Single step from zero init:
    //   P_i * c_prev = 0,  P_f * c_prev = 0  → i_t, f_t, c_t identical
    //   P_o * c_new  ≠ 0  (c_new = i_t * cell_candidate ≠ 0) → o_t / h_t differ
    let x1 = Tensor::new(vec![0.5, -0.3], vec![1, 1, 2]);
    let (_, yh1_no, yc1_no) = lstm(
        &x1,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        None,
        hidden_size,
        "forward",
        None,
    )?;
    let (_, yh1_p, yc1_p) = lstm(
        &x1,
        &w,
        &r,
        Some(&b),
        None,
        None,
        None,
        Some(&p),
        hidden_size,
        "forward",
        None,
    )?;
    // Cell state is unaffected (P_i and P_f terms are zero with c_prev=0).
    for j in 0..hidden_size {
        assert!(
            (yc1_no.data[j] - yc1_p.data[j]).abs() < 1e-7,
            "single step zero init: cell state unaffected by peephole, j={j}"
        );
    }
    // Output gate uses new C_t (non-zero), so h_t differs even in the first step.
    let diff_h1: f32 = yh1_p
        .data
        .iter()
        .zip(&yh1_no.data)
        .map(|(a, b_val)| (a - b_val).abs())
        .sum();
    assert!(
        diff_h1 > 1e-6,
        "output gate peephole must affect single-step h, diff={diff_h1}"
    );

    Ok(())
}

/// LSTM with custom activations.
#[test]
fn test_lstm_activations() -> Result<(), OnnxError> {
    let seq_len = 2;
    let batch = 1;
    let input_size = 2;
    let hidden_size = 3;

    let x = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![seq_len, batch, input_size]);
    let w = Tensor::new(vec![0.1f32; 12 * 2], vec![1, 12, 2]);
    let r = Tensor::new(vec![0.01f32; 12 * 3], vec![1, 12, 3]);

    let (_, yh_default, _) = lstm(
        &x,
        &w,
        &r,
        None,
        None,
        None,
        None,
        None,
        hidden_size,
        "forward",
        None,
    )?;

    // Custom: [Relu, Sigmoid, Tanh]
    let acts: &[&str] = &["Relu", "Sigmoid", "Tanh"];
    let (_, yh_custom, _) = lstm(
        &x,
        &w,
        &r,
        None,
        None,
        None,
        None,
        None,
        hidden_size,
        "forward",
        Some(acts),
    )?;

    let diff: f32 = yh_default
        .data
        .iter()
        .zip(&yh_custom.data)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-6, "custom activations should differ, diff={diff}");

    for v in &yh_custom.data {
        assert!(v.is_finite(), "custom activation output must be finite");
    }

    Ok(())
}
