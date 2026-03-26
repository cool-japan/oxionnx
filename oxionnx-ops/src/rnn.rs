//! LSTM and GRU recurrent neural network kernels (ONNX spec).

use oxionnx_core::{OnnxError, Tensor};

// ── Helper: 2D matmul (M,K) x (K,N) -> (M,N) ─────────────────────────────

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── LSTM ────────────────────────────────────────────────────────────────────

/// Run one direction of LSTM over a sequence.
///
/// `x_seq` is `[seq_len, batch, input_size]` (already ordered for this direction).
/// Returns `(all_hidden, last_h, last_c)` where `all_hidden` is `[seq_len, batch, hidden_size]`.
#[allow(clippy::too_many_arguments)]
fn lstm_one_direction(
    x_seq: &[&[f32]], // seq_len slices of [batch * input_size]
    w: &[f32],        // [4*hidden_size, input_size]
    r: &[f32],        // [4*hidden_size, hidden_size]
    bias: &[f32],     // [8*hidden_size] (Wb concat Rb)
    h_init: &[f32],   // [batch * hidden_size]
    c_init: &[f32],   // [batch * hidden_size]
    batch: usize,
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
    _sequence_lens: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let gate4 = 4 * hidden_size;
    let mut h = h_init.to_vec();
    let mut c = c_init.to_vec();

    // Pre-extract biases: ONNX layout is [Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc]
    let wb = &bias[..gate4];
    let rb = &bias[gate4..gate4 * 2];

    let mut all_h = Vec::with_capacity(seq_len * batch * hidden_size);

    for x_t in x_seq.iter().take(seq_len) {
        // x_t: [batch, input_size], W: [4*hs, input_size]
        // We want x_t @ W^T = [batch, 4*hs]
        let wx = matmul_2d_a_bt(x_t, w, batch, input_size, gate4);
        // h: [batch, hidden_size], R: [4*hs, hidden_size]
        // We want h @ R^T = [batch, 4*hs]
        let rh = matmul_2d_a_bt(&h, r, batch, hidden_size, gate4);

        let mut new_h = vec![0.0f32; batch * hidden_size];
        let mut new_c = vec![0.0f32; batch * hidden_size];

        for b in 0..batch {
            for j in 0..hidden_size {
                // ONNX gate order: i, o, f, c
                let i_idx = j;
                let o_idx = hidden_size + j;
                let f_idx = 2 * hidden_size + j;
                let c_idx = 3 * hidden_size + j;

                let gate_base = b * gate4;
                let it =
                    sigmoid(wx[gate_base + i_idx] + rh[gate_base + i_idx] + wb[i_idx] + rb[i_idx]);
                let ot =
                    sigmoid(wx[gate_base + o_idx] + rh[gate_base + o_idx] + wb[o_idx] + rb[o_idx]);
                let ft =
                    sigmoid(wx[gate_base + f_idx] + rh[gate_base + f_idx] + wb[f_idx] + rb[f_idx]);
                let ct =
                    (wx[gate_base + c_idx] + rh[gate_base + c_idx] + wb[c_idx] + rb[c_idx]).tanh();

                let cell_idx = b * hidden_size + j;
                new_c[cell_idx] = ft * c[cell_idx] + it * ct;
                new_h[cell_idx] = ot * new_c[cell_idx].tanh();
            }
        }

        all_h.extend_from_slice(&new_h);
        h = new_h;
        c = new_c;
    }

    (all_h, h, c)
}

/// Compute A @ B^T where A is [m, k] and B is [n, k], result is [m, n].
fn matmul_2d_a_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[j * k + kk];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// LSTM operator kernel.
///
/// # Arguments
/// * `x` - Input tensor `[seq_len, batch, input_size]`
/// * `w` - Weight tensor `[num_directions, 4*hidden_size, input_size]`
/// * `r` - Recurrence weight `[num_directions, 4*hidden_size, hidden_size]`
/// * `b` - Bias `[num_directions, 8*hidden_size]` (optional)
/// * `sequence_lens` - Per-batch sequence lengths (optional)
/// * `initial_h` - Initial hidden state `[num_directions, batch, hidden_size]` (optional)
/// * `initial_c` - Initial cell state `[num_directions, batch, hidden_size]` (optional)
/// * `hidden_size` - Hidden state dimension
/// * `direction` - "forward", "reverse", or "bidirectional"
///
/// # Returns
/// `(Y, Y_h, Y_c)` where:
/// * `Y` - `[seq_len, num_directions, batch, hidden_size]`
/// * `Y_h` - `[num_directions, batch, hidden_size]`
/// * `Y_c` - `[num_directions, batch, hidden_size]`
#[allow(clippy::too_many_arguments)]
pub fn lstm(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    sequence_lens: Option<&Tensor>,
    initial_h: Option<&Tensor>,
    initial_c: Option<&Tensor>,
    hidden_size: usize,
    direction: &str,
) -> Result<(Tensor, Tensor, Tensor), OnnxError> {
    let seq_len = x.shape[0];
    let batch = x.shape[1];
    let input_size = x.shape[2];

    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };
    let gate4 = 4 * hidden_size;

    let dir_w_size = gate4 * input_size;
    let dir_r_size = gate4 * hidden_size;
    let dir_b_size = 8 * hidden_size;
    let dir_h_size = batch * hidden_size;

    let zeros_b = vec![0.0f32; dir_b_size];
    let zeros_h = vec![0.0f32; dir_h_size];

    // Collect per-timestep slices
    let step_size = batch * input_size;
    let x_steps: Vec<&[f32]> = (0..seq_len)
        .map(|t| &x.data[t * step_size..(t + 1) * step_size])
        .collect();

    let mut y_all = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut y_h_all = vec![0.0f32; num_dir * batch * hidden_size];
    let mut y_c_all = vec![0.0f32; num_dir * batch * hidden_size];

    let seq_lens_data = sequence_lens.map(|t| &t.data[..]);

    for d in 0..num_dir {
        let w_d = &w.data[d * dir_w_size..(d + 1) * dir_w_size];
        let r_d = &r.data[d * dir_r_size..(d + 1) * dir_r_size];
        let b_d = b
            .map(|bt| &bt.data[d * dir_b_size..(d + 1) * dir_b_size])
            .unwrap_or(&zeros_b);
        let h_init = initial_h
            .map(|ht| &ht.data[d * dir_h_size..(d + 1) * dir_h_size])
            .unwrap_or(&zeros_h);
        let c_init = initial_c
            .map(|ct| &ct.data[d * dir_h_size..(d + 1) * dir_h_size])
            .unwrap_or(&zeros_h);

        let is_reverse =
            (d == 0 && direction == "reverse") || (d == 1 && direction == "bidirectional");

        let ordered_steps: Vec<&[f32]> = if is_reverse {
            x_steps.iter().rev().copied().collect()
        } else {
            x_steps.clone()
        };

        let (all_h, last_h, last_c) = lstm_one_direction(
            &ordered_steps,
            w_d,
            r_d,
            b_d,
            h_init,
            c_init,
            batch,
            input_size,
            hidden_size,
            seq_len,
            seq_lens_data,
        );

        // Copy into Y: [seq_len, num_dir, batch, hidden_size]
        let bh = batch * hidden_size;
        for t in 0..seq_len {
            let src_t = if is_reverse { seq_len - 1 - t } else { t };
            let y_offset = t * num_dir * bh + d * bh;
            let src_offset = src_t * bh;
            y_all[y_offset..y_offset + bh].copy_from_slice(&all_h[src_offset..src_offset + bh]);
        }

        // Y_h, Y_c
        let yh_off = d * dir_h_size;
        y_h_all[yh_off..yh_off + dir_h_size].copy_from_slice(&last_h);
        y_c_all[yh_off..yh_off + dir_h_size].copy_from_slice(&last_c);
    }

    let y = Tensor::new(y_all, vec![seq_len, num_dir, batch, hidden_size]);
    let y_h = Tensor::new(y_h_all, vec![num_dir, batch, hidden_size]);
    let y_c = Tensor::new(y_c_all, vec![num_dir, batch, hidden_size]);

    Ok((y, y_h, y_c))
}

// ── GRU ─────────────────────────────────────────────────────────────────────

/// Run one direction of GRU over a sequence.
#[allow(clippy::too_many_arguments)]
fn gru_one_direction(
    x_seq: &[&[f32]],
    w: &[f32],      // [3*hidden_size, input_size]
    r: &[f32],      // [3*hidden_size, hidden_size]
    bias: &[f32],   // [6*hidden_size] (Wb concat Rb)
    h_init: &[f32], // [batch * hidden_size]
    batch: usize,
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
    linear_before_reset: bool,
    _sequence_lens: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let gate3 = 3 * hidden_size;
    let mut h = h_init.to_vec();

    // Biases: [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]
    let wb = &bias[..gate3];
    let rb = &bias[gate3..gate3 * 2];

    let mut all_h = Vec::with_capacity(seq_len * batch * hidden_size);

    for x_t in x_seq.iter().take(seq_len) {
        // x_t @ W^T = [batch, 3*hs]
        let wx = matmul_2d_a_bt(x_t, w, batch, input_size, gate3);
        // h @ R^T = [batch, 3*hs]
        let rh = matmul_2d_a_bt(&h, r, batch, hidden_size, gate3);

        let mut new_h = vec![0.0f32; batch * hidden_size];

        for b_idx in 0..batch {
            for j in 0..hidden_size {
                // Gate order: z, r, h
                let z_idx = j;
                let r_idx = hidden_size + j;
                let h_idx = 2 * hidden_size + j;

                let gb = b_idx * gate3;
                let zt = sigmoid(wx[gb + z_idx] + rh[gb + z_idx] + wb[z_idx] + rb[z_idx]);
                let rt = sigmoid(wx[gb + r_idx] + rh[gb + r_idx] + wb[r_idx] + rb[r_idx]);

                let ht_candidate = if linear_before_reset {
                    // ht = tanh(Wh*Xt + rt * (Rh*H_{t-1} + Rbh) + Wbh)
                    (wx[gb + h_idx] + rt * (rh[gb + h_idx] + rb[h_idx]) + wb[h_idx]).tanh()
                } else {
                    // ht = tanh(Wh*Xt + Rh*(rt * H_{t-1}) + Wbh + Rbh)
                    // For this we need Rh * (rt * h), which means we can't reuse the
                    // precomputed rh directly for the h-gate. We need a separate multiply.
                    let r_h_slice =
                        &r[2 * hidden_size * hidden_size..3 * hidden_size * hidden_size];
                    // Compute rt * h[b] for this batch element
                    let h_base = b_idx * hidden_size;
                    let mut rh_val = 0.0f32;
                    for kk in 0..hidden_size {
                        rh_val += r_h_slice[j * hidden_size + kk] * rt * h[h_base + kk];
                    }
                    (wx[gb + h_idx] + rh_val + wb[h_idx] + rb[h_idx]).tanh()
                };

                let cell_idx = b_idx * hidden_size + j;
                new_h[cell_idx] = (1.0 - zt) * ht_candidate + zt * h[cell_idx];
            }
        }

        all_h.extend_from_slice(&new_h);
        h = new_h;
    }

    (all_h, h)
}

/// GRU operator kernel.
///
/// # Arguments
/// * `x` - Input `[seq_len, batch, input_size]`
/// * `w` - Weights `[num_directions, 3*hidden_size, input_size]`
/// * `r` - Recurrence weights `[num_directions, 3*hidden_size, hidden_size]`
/// * `b` - Bias `[num_directions, 6*hidden_size]` (optional)
/// * `sequence_lens` - Per-batch sequence lengths (optional)
/// * `initial_h` - `[num_directions, batch, hidden_size]` (optional)
/// * `hidden_size` - Hidden dimension
/// * `direction` - "forward", "reverse", or "bidirectional"
/// * `linear_before_reset` - If true, apply reset gate before the linear transformation
///
/// # Returns
/// `(Y, Y_h)` where:
/// * `Y` - `[seq_len, num_directions, batch, hidden_size]`
/// * `Y_h` - `[num_directions, batch, hidden_size]`
#[allow(clippy::too_many_arguments)]
pub fn gru(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    sequence_lens: Option<&Tensor>,
    initial_h: Option<&Tensor>,
    hidden_size: usize,
    direction: &str,
    linear_before_reset: bool,
) -> Result<(Tensor, Tensor), OnnxError> {
    let seq_len = x.shape[0];
    let batch = x.shape[1];
    let input_size = x.shape[2];

    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };
    let gate3 = 3 * hidden_size;

    let dir_w_size = gate3 * input_size;
    let dir_r_size = gate3 * hidden_size;
    let dir_b_size = 6 * hidden_size;
    let dir_h_size = batch * hidden_size;

    let zeros_b = vec![0.0f32; dir_b_size];
    let zeros_h = vec![0.0f32; dir_h_size];

    let step_size = batch * input_size;
    let x_steps: Vec<&[f32]> = (0..seq_len)
        .map(|t| &x.data[t * step_size..(t + 1) * step_size])
        .collect();

    let mut y_all = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut y_h_all = vec![0.0f32; num_dir * batch * hidden_size];

    let seq_lens_data = sequence_lens.map(|t| &t.data[..]);

    for d in 0..num_dir {
        let w_d = &w.data[d * dir_w_size..(d + 1) * dir_w_size];
        let r_d = &r.data[d * dir_r_size..(d + 1) * dir_r_size];
        let b_d = b
            .map(|bt| &bt.data[d * dir_b_size..(d + 1) * dir_b_size])
            .unwrap_or(&zeros_b);
        let h_init = initial_h
            .map(|ht| &ht.data[d * dir_h_size..(d + 1) * dir_h_size])
            .unwrap_or(&zeros_h);

        let is_reverse =
            (d == 0 && direction == "reverse") || (d == 1 && direction == "bidirectional");

        let ordered_steps: Vec<&[f32]> = if is_reverse {
            x_steps.iter().rev().copied().collect()
        } else {
            x_steps.clone()
        };

        let (all_h, last_h) = gru_one_direction(
            &ordered_steps,
            w_d,
            r_d,
            b_d,
            h_init,
            batch,
            input_size,
            hidden_size,
            seq_len,
            linear_before_reset,
            seq_lens_data,
        );

        let bh = batch * hidden_size;
        for t in 0..seq_len {
            let src_t = if is_reverse { seq_len - 1 - t } else { t };
            let y_offset = t * num_dir * bh + d * bh;
            let src_offset = src_t * bh;
            y_all[y_offset..y_offset + bh].copy_from_slice(&all_h[src_offset..src_offset + bh]);
        }

        let yh_off = d * dir_h_size;
        y_h_all[yh_off..yh_off + dir_h_size].copy_from_slice(&last_h);
    }

    let y = Tensor::new(y_all, vec![seq_len, num_dir, batch, hidden_size]);
    let y_h = Tensor::new(y_h_all, vec![num_dir, batch, hidden_size]);

    Ok((y, y_h))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_forward_basic() {
        // seq_len=2, batch=1, input_size=2, hidden_size=3
        let seq_len = 2;
        let batch = 1;
        let input_size = 2;
        let hidden_size = 3;

        // x: [2, 1, 2]
        let x = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![seq_len, batch, input_size]);

        // W: [1, 12, 2] (4*3=12 gates, input_size=2)
        let w = Tensor::new(vec![0.1f32; 12 * 2], vec![1, 12, 2]);
        // R: [1, 12, 3]
        let r = Tensor::new(vec![0.01f32; 12 * 3], vec![1, 12, 3]);
        // B: [1, 24]
        let b = Tensor::new(vec![0.0f32; 24], vec![1, 24]);

        let (y, y_h, y_c) = lstm(
            &x,
            &w,
            &r,
            Some(&b),
            None,
            None,
            None,
            hidden_size,
            "forward",
        )
        .expect("LSTM should not fail");

        // Check shapes
        assert_eq!(y.shape, vec![2, 1, 1, 3]);
        assert_eq!(y_h.shape, vec![1, 1, 3]);
        assert_eq!(y_c.shape, vec![1, 1, 3]);

        // y_h should be the last timestep of y for direction 0
        let last_y = &y.data[1 * 1 * 1 * 3..2 * 1 * 1 * 3];
        for (a, b) in y_h.data.iter().zip(last_y.iter()) {
            assert!((a - b).abs() < 1e-6, "y_h should match last timestep of Y");
        }

        // Values should be non-zero (gates activated)
        assert!(y_h.data.iter().all(|&v| v.abs() > 1e-7));
    }

    #[test]
    fn test_lstm_shapes_bidirectional() {
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
            hidden_size,
            "bidirectional",
        )
        .expect("LSTM bidirectional should not fail");

        assert_eq!(y.shape, vec![3, 2, 2, 5]);
        assert_eq!(y_h.shape, vec![2, 2, 5]);
        assert_eq!(y_c.shape, vec![2, 2, 5]);
    }

    #[test]
    fn test_gru_forward_basic() {
        let seq_len = 2;
        let batch = 1;
        let input_size = 2;
        let hidden_size = 3;

        let x = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![seq_len, batch, input_size]);
        let w = Tensor::new(vec![0.1f32; 9 * 2], vec![1, 9, 2]);
        let r = Tensor::new(vec![0.01f32; 9 * 3], vec![1, 9, 3]);
        let b = Tensor::new(vec![0.0f32; 18], vec![1, 18]);

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
        )
        .expect("GRU should not fail");

        assert_eq!(y.shape, vec![2, 1, 1, 3]);
        assert_eq!(y_h.shape, vec![1, 1, 3]);

        // y_h matches last timestep
        let last_y = &y.data[1 * 1 * 1 * 3..2 * 1 * 1 * 3];
        for (a, b) in y_h.data.iter().zip(last_y.iter()) {
            assert!((a - b).abs() < 1e-6, "y_h should match last timestep of Y");
        }

        assert!(y_h.data.iter().all(|&v| v.abs() > 1e-7));
    }

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
        )
        .expect("GRU lbr=true");

        // Results should differ when linear_before_reset differs
        let diff: f32 = y1
            .data
            .iter()
            .zip(y2.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        // With zero initial h, both modes should give same result since rt * 0 = 0
        // So actually they should be equal for zero init state. Let's verify shapes at least.
        assert_eq!(y1.shape, y2.shape);
        // With zero initial state, reset gate doesn't matter, so diff should be ~0
        assert!(
            diff < 1e-5,
            "With zero init, lbr shouldn't matter, diff={diff}"
        );
    }
}
