//! LSTM, GRU, and simple RNN kernels (ONNX spec).
//!
//! Supports:
//! - Variable-length sequences via `sequence_lens`
//! - LSTM peephole connections (optional 8th input P)
//! - Per-gate activation functions (ONNX `activations` attribute)
//! - Forward, reverse, and bidirectional modes

use oxionnx_core::{OnnxError, Tensor};

// ── Activation ──────────────────────────────────────────────────────────────

/// Supported activation functions for RNN gates.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Activation {
    Sigmoid,
    Tanh,
    Relu,
}

impl Activation {
    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
        }
    }

    fn from_name(s: &str) -> Self {
        match s {
            "Sigmoid" => Activation::Sigmoid,
            "Relu" => Activation::Relu,
            _ => Activation::Tanh,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

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

/// Check whether processing step `t` is valid for batch element `b`.
///
/// For forward: valid when `t < sequence_lens[b]`.
/// For reverse: the reversed input processes original timestep `(seq_len-1-t)`,
/// which is valid when `(seq_len-1-t) < sequence_lens[b]`, i.e. `t >= seq_len - lens[b]`.
fn step_is_valid(
    t: usize,
    b: usize,
    seq_len: usize,
    sequence_lens: Option<&[usize]>,
    is_reverse: bool,
) -> bool {
    match sequence_lens {
        None => true,
        Some(lens) => {
            let len_b = if b < lens.len() { lens[b] } else { seq_len };
            if is_reverse {
                len_b >= seq_len || t >= (seq_len - len_b)
            } else {
                t < len_b
            }
        }
    }
}

// ── LSTM ────────────────────────────────────────────────────────────────────

/// Run one direction of LSTM over a sequence.
///
/// Returns `(all_hidden, last_h, last_c)` where `all_hidden` is flat
/// `[seq_len, batch, hidden_size]` with zeros for masked timesteps.
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
    sequence_lens: Option<&[usize]>,
    is_reverse: bool,
    peephole: Option<&[f32]>,     // [3*hidden_size]: P_i, P_o, P_f
    activations: [Activation; 3], // [gate, cell_candidate, output_transform]
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let gate4 = 4 * hidden_size;
    let mut h = h_init.to_vec();
    let mut c = c_init.to_vec();

    // Pre-extract biases: ONNX layout [Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc]
    let wb = &bias[..gate4];
    let rb = &bias[gate4..gate4 * 2];

    // Track the last valid hidden/cell state per batch element (for Y_h / Y_c).
    let mut last_valid_h = h_init.to_vec();
    let mut last_valid_c = c_init.to_vec();

    let mut all_h = vec![0.0f32; seq_len * batch * hidden_size];

    let act_gate = activations[0];
    let act_cell = activations[1];
    let act_out = activations[2];

    for (t, x_t) in x_seq.iter().enumerate().take(seq_len) {
        // x_t @ W^T = [batch, 4*hs]
        let wx = matmul_2d_a_bt(x_t, w, batch, input_size, gate4);
        // h @ R^T = [batch, 4*hs]
        let rh = matmul_2d_a_bt(&h, r, batch, hidden_size, gate4);

        let mut new_h = vec![0.0f32; batch * hidden_size];
        let mut new_c = vec![0.0f32; batch * hidden_size];

        for b_idx in 0..batch {
            let base = b_idx * hidden_size;
            if step_is_valid(t, b_idx, seq_len, sequence_lens, is_reverse) {
                for j in 0..hidden_size {
                    // ONNX gate order: i, o, f, c
                    let i_idx = j;
                    let o_idx = hidden_size + j;
                    let f_idx = 2 * hidden_size + j;
                    let c_idx = 3 * hidden_size + j;
                    let gb = b_idx * gate4;
                    let cell_idx = base + j;

                    // Peephole for input and forget gates (use C_{t-1})
                    let p_i = peephole.map_or(0.0, |p| p[j] * c[cell_idx]);
                    let p_f = peephole.map_or(0.0, |p| p[2 * hidden_size + j] * c[cell_idx]);

                    let it = act_gate
                        .apply(wx[gb + i_idx] + rh[gb + i_idx] + wb[i_idx] + rb[i_idx] + p_i);
                    let ft = act_gate
                        .apply(wx[gb + f_idx] + rh[gb + f_idx] + wb[f_idx] + rb[f_idx] + p_f);
                    let ct =
                        act_cell.apply(wx[gb + c_idx] + rh[gb + c_idx] + wb[c_idx] + rb[c_idx]);

                    new_c[cell_idx] = ft * c[cell_idx] + it * ct;

                    // Peephole for output gate (uses NEW C_t)
                    let p_o = peephole.map_or(0.0, |p| p[hidden_size + j] * new_c[cell_idx]);

                    let ot = act_gate
                        .apply(wx[gb + o_idx] + rh[gb + o_idx] + wb[o_idx] + rb[o_idx] + p_o);

                    new_h[cell_idx] = ot * act_out.apply(new_c[cell_idx]);
                }
                // Update last valid state for this batch element
                last_valid_h[base..base + hidden_size]
                    .copy_from_slice(&new_h[base..base + hidden_size]);
                last_valid_c[base..base + hidden_size]
                    .copy_from_slice(&new_c[base..base + hidden_size]);

                // Write to output (valid timestep)
                let h_off = t * batch * hidden_size + base;
                all_h[h_off..h_off + hidden_size].copy_from_slice(&new_h[base..base + hidden_size]);
            } else {
                // Invalid timestep: freeze internal state, output stays 0
                new_h[base..base + hidden_size].copy_from_slice(&h[base..base + hidden_size]);
                new_c[base..base + hidden_size].copy_from_slice(&c[base..base + hidden_size]);
            }
        }

        h = new_h;
        c = new_c;
    }

    (all_h, last_valid_h, last_valid_c)
}

/// LSTM operator kernel.
///
/// # Inputs (matching ONNX spec order)
/// * `x` – `[seq_len, batch, input_size]`
/// * `w` – `[num_directions, 4*hidden_size, input_size]`
/// * `r` – `[num_directions, 4*hidden_size, hidden_size]`
/// * `b` – `[num_directions, 8*hidden_size]` (optional)
/// * `sequence_lens` – Per-batch lengths (optional)
/// * `initial_h` – `[num_directions, batch, hidden_size]` (optional)
/// * `initial_c` – `[num_directions, batch, hidden_size]` (optional)
/// * `peephole` – `[num_directions, 3*hidden_size]` (optional, P_i/P_o/P_f)
/// * `hidden_size` – Hidden dimension
/// * `direction` – `"forward"`, `"reverse"`, or `"bidirectional"`
/// * `activations` – Override gate activations. Default `["Sigmoid","Tanh","Tanh"]` per dir.
///
/// # Returns
/// `(Y, Y_h, Y_c)` shaped
/// `[seq_len, num_dir, batch, hs]`, `[num_dir, batch, hs]`, `[num_dir, batch, hs]`
#[allow(clippy::too_many_arguments)]
pub fn lstm(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    sequence_lens: Option<&Tensor>,
    initial_h: Option<&Tensor>,
    initial_c: Option<&Tensor>,
    peephole: Option<&Tensor>,
    hidden_size: usize,
    direction: &str,
    activations: Option<&[&str]>,
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
    let dir_p_size = 3 * hidden_size;

    let zeros_b = vec![0.0f32; dir_b_size];
    let zeros_h = vec![0.0f32; dir_h_size];

    // Convert sequence_lens from f32 to usize.
    let seq_lens: Option<Vec<usize>> =
        sequence_lens.map(|t| t.data.iter().take(batch).map(|&v| v as usize).collect());
    let seq_lens_ref = seq_lens.as_deref();

    // Collect per-timestep slices.
    let step_size = batch * input_size;
    let x_steps: Vec<&[f32]> = (0..seq_len)
        .map(|t| &x.data[t * step_size..(t + 1) * step_size])
        .collect();

    let mut y_all = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut y_h_all = vec![0.0f32; num_dir * batch * hidden_size];
    let mut y_c_all = vec![0.0f32; num_dir * batch * hidden_size];

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
        let p_d = peephole.map(|pt| &pt.data[d * dir_p_size..(d + 1) * dir_p_size]);

        let is_reverse =
            (d == 0 && direction == "reverse") || (d == 1 && direction == "bidirectional");

        let ordered_steps: Vec<&[f32]> = if is_reverse {
            x_steps.iter().rev().copied().collect()
        } else {
            x_steps.clone()
        };

        // Parse activations for this direction (3 per direction).
        let acts = parse_lstm_activations(activations, d);

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
            seq_lens_ref,
            is_reverse,
            p_d,
            acts,
        );

        // Copy into Y: [seq_len, num_dir, batch, hidden_size].
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

/// Parse LSTM activations for direction `d` from the optional attribute.
/// Default: [Sigmoid, Tanh, Tanh].
fn parse_lstm_activations(activations: Option<&[&str]>, direction_idx: usize) -> [Activation; 3] {
    match activations {
        Some(acts) => {
            let off = direction_idx * 3;
            [
                acts.get(off)
                    .map_or(Activation::Sigmoid, |s| Activation::from_name(s)),
                acts.get(off + 1)
                    .map_or(Activation::Tanh, |s| Activation::from_name(s)),
                acts.get(off + 2)
                    .map_or(Activation::Tanh, |s| Activation::from_name(s)),
            ]
        }
        None => [Activation::Sigmoid, Activation::Tanh, Activation::Tanh],
    }
}

// ── GRU ─────────────────────────────────────────────────────────────────────

/// Run one direction of GRU over a sequence.
///
/// Returns `(all_hidden, last_h)`.
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
    sequence_lens: Option<&[usize]>,
    is_reverse: bool,
    activations: [Activation; 2], // [gate, hidden_candidate]
) -> (Vec<f32>, Vec<f32>) {
    let gate3 = 3 * hidden_size;
    let mut h = h_init.to_vec();

    // Biases: [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]
    let wb = &bias[..gate3];
    let rb = &bias[gate3..gate3 * 2];

    let mut last_valid_h = h_init.to_vec();
    let mut all_h = vec![0.0f32; seq_len * batch * hidden_size];

    let act_gate = activations[0];
    let act_hidden = activations[1];

    for (t, x_t) in x_seq.iter().enumerate().take(seq_len) {
        // x_t @ W^T = [batch, 3*hs]
        let wx = matmul_2d_a_bt(x_t, w, batch, input_size, gate3);
        // h @ R^T = [batch, 3*hs]
        let rh = matmul_2d_a_bt(&h, r, batch, hidden_size, gate3);

        let mut new_h = vec![0.0f32; batch * hidden_size];

        for b_idx in 0..batch {
            let base = b_idx * hidden_size;
            if step_is_valid(t, b_idx, seq_len, sequence_lens, is_reverse) {
                for j in 0..hidden_size {
                    // Gate order: z, r, h
                    let z_idx = j;
                    let r_idx = hidden_size + j;
                    let h_idx = 2 * hidden_size + j;

                    let gb = b_idx * gate3;
                    let zt =
                        act_gate.apply(wx[gb + z_idx] + rh[gb + z_idx] + wb[z_idx] + rb[z_idx]);
                    let rt =
                        act_gate.apply(wx[gb + r_idx] + rh[gb + r_idx] + wb[r_idx] + rb[r_idx]);

                    let ht_candidate = if linear_before_reset {
                        // ht = g(Wh*Xt + rt ⊙ (Rh*H_{t-1} + Rbh) + Wbh)
                        act_hidden
                            .apply(wx[gb + h_idx] + rt * (rh[gb + h_idx] + rb[h_idx]) + wb[h_idx])
                    } else {
                        // ht = g(Wh*Xt + Rh*(rt ⊙ H_{t-1}) + Wbh + Rbh)
                        let r_h_slice =
                            &r[2 * hidden_size * hidden_size..3 * hidden_size * hidden_size];
                        let h_base = b_idx * hidden_size;
                        let mut rh_val = 0.0f32;
                        for kk in 0..hidden_size {
                            rh_val += r_h_slice[j * hidden_size + kk] * rt * h[h_base + kk];
                        }
                        act_hidden.apply(wx[gb + h_idx] + rh_val + wb[h_idx] + rb[h_idx])
                    };

                    let cell_idx = base + j;
                    new_h[cell_idx] = (1.0 - zt) * ht_candidate + zt * h[cell_idx];
                }
                last_valid_h[base..base + hidden_size]
                    .copy_from_slice(&new_h[base..base + hidden_size]);

                let h_off = t * batch * hidden_size + base;
                all_h[h_off..h_off + hidden_size].copy_from_slice(&new_h[base..base + hidden_size]);
            } else {
                // Freeze state, output stays 0
                new_h[base..base + hidden_size].copy_from_slice(&h[base..base + hidden_size]);
            }
        }

        h = new_h;
    }

    (all_h, last_valid_h)
}

/// GRU operator kernel.
///
/// # Inputs
/// * `x` – `[seq_len, batch, input_size]`
/// * `w` – `[num_directions, 3*hidden_size, input_size]`
/// * `r` – `[num_directions, 3*hidden_size, hidden_size]`
/// * `b` – `[num_directions, 6*hidden_size]` (optional)
/// * `sequence_lens` – Per-batch lengths (optional)
/// * `initial_h` – `[num_directions, batch, hidden_size]` (optional)
/// * `hidden_size` – Hidden dimension
/// * `direction` – `"forward"`, `"reverse"`, or `"bidirectional"`
/// * `linear_before_reset` – Reset gate mode
/// * `activations` – Override activations. Default `["Sigmoid","Tanh"]` per dir.
///
/// # Returns
/// `(Y, Y_h)` shaped `[seq_len, num_dir, batch, hs]`, `[num_dir, batch, hs]`
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
    activations: Option<&[&str]>,
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

    let seq_lens: Option<Vec<usize>> =
        sequence_lens.map(|t| t.data.iter().take(batch).map(|&v| v as usize).collect());
    let seq_lens_ref = seq_lens.as_deref();

    let step_size = batch * input_size;
    let x_steps: Vec<&[f32]> = (0..seq_len)
        .map(|t| &x.data[t * step_size..(t + 1) * step_size])
        .collect();

    let mut y_all = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut y_h_all = vec![0.0f32; num_dir * batch * hidden_size];

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

        let acts = parse_gru_activations(activations, d);

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
            seq_lens_ref,
            is_reverse,
            acts,
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

/// Parse GRU activations for direction `d`. Default: [Sigmoid, Tanh].
fn parse_gru_activations(activations: Option<&[&str]>, direction_idx: usize) -> [Activation; 2] {
    match activations {
        Some(acts) => {
            let off = direction_idx * 2;
            [
                acts.get(off)
                    .map_or(Activation::Sigmoid, |s| Activation::from_name(s)),
                acts.get(off + 1)
                    .map_or(Activation::Tanh, |s| Activation::from_name(s)),
            ]
        }
        None => [Activation::Sigmoid, Activation::Tanh],
    }
}

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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LSTM tests ──────────────────────────────────────────────────────

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

    // ── GRU tests ───────────────────────────────────────────────────────

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

    // ── Simple RNN tests ────────────────────────────────────────────────

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

    // ── Edge cases ──────────────────────────────────────────────────────

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
}
