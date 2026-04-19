//! Simple (Elman) RNN kernel unit tests.

use oxionnx_core::{OnnxError, Tensor};

use crate::rnn::simple_rnn;

// ── Simple RNN tests ──────────────────────────────────────────────────────────

/// RNN tanh with hand-computed values (2 timesteps).
#[test]
fn test_rnn_tanh_hand_computed() -> Result<(), OnnxError> {
    let batch = 1;
    let input_size = 3;
    let hidden_size = 4;
    let seq_len = 2;

    let x = Tensor::new(
        vec![0.5, -0.3, 0.8, 0.2, 0.1, -0.4],
        vec![seq_len, batch, input_size],
    );

    let mut w_data = vec![0.0f32; hidden_size * input_size];
    for j in 0..hidden_size {
        w_data[j * input_size + (j % input_size)] = 0.1;
    }
    let w = Tensor::new(w_data.clone(), vec![1, hidden_size, input_size]);
    let r_data = vec![0.02f32; hidden_size * hidden_size];
    let r = Tensor::new(r_data.clone(), vec![1, hidden_size, hidden_size]);
    let b = Tensor::new(vec![0.0f32; 2 * hidden_size], vec![1, 2 * hidden_size]);

    let (y, y_h) = simple_rnn(
        &x,
        &w,
        &r,
        Some(&b),
        None,
        None,
        hidden_size,
        "forward",
        "Tanh",
    )?;

    assert_eq!(y.shape, vec![seq_len, 1, batch, hidden_size]);
    assert_eq!(y_h.shape, vec![1, batch, hidden_size]);

    // Step 0: h_prev=0
    let x0 = [0.5f32, -0.3, 0.8];
    let mut wx0 = vec![0.0f32; hidden_size];
    for j in 0..hidden_size {
        for k in 0..input_size {
            wx0[j] += x0[k] * w_data[j * input_size + k];
        }
    }
    let h0: Vec<f32> = wx0.iter().map(|v| v.tanh()).collect();
    for j in 0..hidden_size {
        assert!(
            (y.data[j] - h0[j]).abs() < 1e-5,
            "step0 h[{j}]: expected {}, got {}",
            h0[j],
            y.data[j]
        );
    }

    // Step 1: h_prev=h0
    let x1 = [0.2f32, 0.1, -0.4];
    let mut wx1 = vec![0.0f32; hidden_size];
    for j in 0..hidden_size {
        for k in 0..input_size {
            wx1[j] += x1[k] * w_data[j * input_size + k];
        }
    }
    let mut rh1 = vec![0.0f32; hidden_size];
    for j in 0..hidden_size {
        for k in 0..hidden_size {
            rh1[j] += h0[k] * r_data[j * hidden_size + k];
        }
    }
    let h1: Vec<f32> = (0..hidden_size).map(|j| (wx1[j] + rh1[j]).tanh()).collect();
    for j in 0..hidden_size {
        assert!(
            (y_h.data[j] - h1[j]).abs() < 1e-5,
            "step1 h[{j}]: expected {}, got {}",
            h1[j],
            y_h.data[j]
        );
    }

    Ok(())
}

/// Simple RNN reverse + Relu.
#[test]
fn test_rnn_reverse_relu() -> Result<(), OnnxError> {
    let seq_len = 3;
    let batch = 1;
    let input_size = 2;
    let hidden_size = 3;

    let x = Tensor::new(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        vec![seq_len, batch, input_size],
    );
    let w = Tensor::new(
        vec![0.1f32; hidden_size * input_size],
        vec![1, hidden_size, input_size],
    );
    let r = Tensor::new(
        vec![0.02f32; hidden_size * hidden_size],
        vec![1, hidden_size, hidden_size],
    );

    let (y_fwd, _) = simple_rnn(&x, &w, &r, None, None, None, hidden_size, "forward", "Relu")?;
    let (y_rev, _) = simple_rnn(&x, &w, &r, None, None, None, hidden_size, "reverse", "Relu")?;

    assert_eq!(y_fwd.shape, y_rev.shape);

    let diff: f32 = y_fwd
        .data
        .iter()
        .zip(&y_rev.data)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "forward and reverse should differ, diff={diff}"
    );

    // Relu outputs non-negative
    for v in &y_fwd.data {
        assert!(*v >= -1e-10, "Relu >= 0, got {v}");
    }
    for v in &y_rev.data {
        assert!(*v >= -1e-10, "Relu >= 0, got {v}");
    }

    Ok(())
}

/// Simple RNN with sequence_lens.
#[test]
fn test_rnn_sequence_lens() -> Result<(), OnnxError> {
    let seq_len = 5;
    let batch = 2;
    let input_size = 2;
    let hidden_size = 3;

    let x = Tensor::new(
        vec![0.1f32; seq_len * batch * input_size],
        vec![seq_len, batch, input_size],
    );
    let w = Tensor::new(
        vec![0.1f32; hidden_size * input_size],
        vec![1, hidden_size, input_size],
    );
    let r = Tensor::new(
        vec![0.02f32; hidden_size * hidden_size],
        vec![1, hidden_size, hidden_size],
    );

    // batch[0] length=3, batch[1] length=5
    let sl = Tensor::new(vec![3.0, 5.0], vec![2]);

    let (y, y_h) = simple_rnn(
        &x,
        &w,
        &r,
        None,
        Some(&sl),
        None,
        hidden_size,
        "forward",
        "Tanh",
    )?;

    let bh = batch * hidden_size;

    // batch[0]: t>=3 → zero
    for t in 3..seq_len {
        let off = t * bh;
        for j in 0..hidden_size {
            assert!(
                y.data[off + j].abs() < 1e-10,
                "rnn batch[0] t={t} j={j} should be 0"
            );
        }
    }

    // Y_h[batch0] == Y[t=2, batch0]
    let y_t2_b0 = &y.data[2 * bh..2 * bh + hidden_size];
    let yh_b0 = &y_h.data[..hidden_size];
    for j in 0..hidden_size {
        assert!((y_t2_b0[j] - yh_b0[j]).abs() < 1e-6);
    }

    // Y_h[batch1] == Y[t=4, batch1]
    let y_t4_b1 = &y.data[4 * bh + hidden_size..4 * bh + 2 * hidden_size];
    let yh_b1 = &y_h.data[hidden_size..2 * hidden_size];
    for j in 0..hidden_size {
        assert!((y_t4_b1[j] - yh_b1[j]).abs() < 1e-6);
    }

    Ok(())
}
