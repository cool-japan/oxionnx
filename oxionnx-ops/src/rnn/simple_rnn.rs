//! Simple (Elman) RNN operator kernel (ONNX spec).
//!
//! Supports forward, reverse, and bidirectional modes; optional bias;
//! configurable activation function; and variable-length sequences.

use oxionnx_core::{OnnxError, Tensor};

use super::common::{matmul_2d_a_bt, step_is_valid, Activation};

// ── Simple RNN ──────────────────────────────────────────────────────────────

/// Run one direction of a simple (Elman) RNN over a sequence.
///
/// `h_t = activation(x_t @ W^T + h_{t-1} @ R^T + Wb + Rb)`
#[allow(clippy::too_many_arguments)]
fn simple_rnn_one_direction(
    x_seq: &[&[f32]],
    w: &[f32],      // [hidden_size, input_size]
    r: &[f32],      // [hidden_size, hidden_size]
    bias: &[f32],   // [2*hidden_size] (Wb concat Rb)
    h_init: &[f32], // [batch * hidden_size]
    batch: usize,
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
    activation: Activation,
    sequence_lens: Option<&[usize]>,
    is_reverse: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut h = h_init.to_vec();

    let wb = &bias[..hidden_size];
    let rb = &bias[hidden_size..2 * hidden_size];

    let mut last_valid_h = h_init.to_vec();
    let mut all_h = vec![0.0f32; seq_len * batch * hidden_size];

    for (t, x_t) in x_seq.iter().enumerate().take(seq_len) {
        // x_t @ W^T = [batch, hidden_size]
        let wx = matmul_2d_a_bt(x_t, w, batch, input_size, hidden_size);
        // h @ R^T = [batch, hidden_size]
        let rh = matmul_2d_a_bt(&h, r, batch, hidden_size, hidden_size);

        let mut new_h = vec![0.0f32; batch * hidden_size];

        for b_idx in 0..batch {
            let base = b_idx * hidden_size;
            if step_is_valid(t, b_idx, seq_len, sequence_lens, is_reverse) {
                for j in 0..hidden_size {
                    let idx = base + j;
                    new_h[idx] = activation.apply(wx[idx] + rh[idx] + wb[j] + rb[j]);
                }
                last_valid_h[base..base + hidden_size]
                    .copy_from_slice(&new_h[base..base + hidden_size]);

                let h_off = t * batch * hidden_size + base;
                all_h[h_off..h_off + hidden_size].copy_from_slice(&new_h[base..base + hidden_size]);
            } else {
                new_h[base..base + hidden_size].copy_from_slice(&h[base..base + hidden_size]);
            }
        }

        h = new_h;
    }

    (all_h, last_valid_h)
}

/// Simple (Elman) RNN operator kernel.
///
/// # Inputs
/// * `x` – `[seq_len, batch, input_size]`
/// * `w` – `[num_directions, hidden_size, input_size]`
/// * `r` – `[num_directions, hidden_size, hidden_size]`
/// * `b` – `[num_directions, 2*hidden_size]` (optional)
/// * `sequence_lens` – Per-batch lengths (optional)
/// * `initial_h` – `[num_directions, batch, hidden_size]` (optional)
/// * `hidden_size` – Hidden dimension
/// * `direction` – `"forward"`, `"reverse"`, or `"bidirectional"`
/// * `activation` – `"Tanh"`, `"Relu"`, `"Sigmoid"`
///
/// # Returns
/// `(Y, Y_h)` shaped `[seq_len, num_dir, batch, hs]`, `[num_dir, batch, hs]`
#[allow(clippy::too_many_arguments)]
pub fn simple_rnn(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    sequence_lens: Option<&Tensor>,
    initial_h: Option<&Tensor>,
    hidden_size: usize,
    direction: &str,
    activation: &str,
) -> Result<(Tensor, Tensor), OnnxError> {
    let seq_len = x.shape[0];
    let batch = x.shape[1];
    let input_size = x.shape[2];

    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };

    let dir_w_size = hidden_size * input_size;
    let dir_r_size = hidden_size * hidden_size;
    let dir_b_size = 2 * hidden_size;
    let dir_h_size = batch * hidden_size;

    let zeros_b = vec![0.0f32; dir_b_size];
    let zeros_h = vec![0.0f32; dir_h_size];

    let seq_lens: Option<Vec<usize>> =
        sequence_lens.map(|t| t.data.iter().take(batch).map(|&v| v as usize).collect());
    let seq_lens_ref = seq_lens.as_deref();

    let step_size = batch * input_size;
    let x_steps: Vec<&[f32]> = (0..seq_len)
        .map(|t| &x.data[t * step_size..(t + 1) * step_size])
        .collect();

    let mut y_all = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut y_h_all = vec![0.0f32; num_dir * batch * hidden_size];

    let act = Activation::from_name(activation);

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

        let (all_h, last_h) = simple_rnn_one_direction(
            &ordered_steps,
            w_d,
            r_d,
            b_d,
            h_init,
            batch,
            input_size,
            hidden_size,
            seq_len,
            act,
            seq_lens_ref,
            is_reverse,
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
