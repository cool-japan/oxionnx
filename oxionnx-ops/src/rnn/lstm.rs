//! LSTM operator kernel (ONNX spec).
//!
//! Supports forward, reverse, and bidirectional modes; optional peephole
//! connections; per-gate activation overrides; and variable-length sequences.

use oxionnx_core::{OnnxError, Tensor};

use super::common::{matmul_2d_a_bt, step_is_valid, Activation};

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

/// Caller-provided output slots for zero-copy LSTM computation.
///
/// The caller pre-allocates each buffer and passes mutable references here.
/// `lstm_into` resizes each Vec if needed but never replaces the backing
/// allocation (pointer stability across repeated calls).
pub(crate) struct LstmOutputSlots<'a> {
    /// All hidden outputs: flat `[seq_len * num_dir * batch * hidden_size]`
    pub y: &'a mut Vec<f32>,
    /// Last hidden state: flat `[num_dir * batch * hidden_size]`
    pub y_h: &'a mut Vec<f32>,
    /// Last cell state: flat `[num_dir * batch * hidden_size]`
    pub y_c: &'a mut Vec<f32>,
}

/// Core LSTM computation writing directly into caller-provided output buffers.
///
/// All inputs mirror `lstm()`. The three output tensors are written into `slots`
/// in-place. No new `Vec<f32>` is allocated for outputs; only the internal
/// per-direction working buffers (`h`, `c`, `wx`, `rh`, `new_h`, `new_c`)
/// that belong to the computation kernel itself are allocated.
///
/// Existing `lstm()` delegates here to avoid duplication.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lstm_into(
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
    slots: LstmOutputSlots<'_>,
) -> Result<(), OnnxError> {
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

    // Resize output buffers if necessary — never replace them (pointer stability).
    let y_len = seq_len * num_dir * batch * hidden_size;
    let yh_len = num_dir * batch * hidden_size;
    if slots.y.len() != y_len {
        slots.y.resize(y_len, 0.0f32);
    }
    if slots.y_h.len() != yh_len {
        slots.y_h.resize(yh_len, 0.0f32);
    }
    if slots.y_c.len() != yh_len {
        slots.y_c.resize(yh_len, 0.0f32);
    }

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
            slots.y[y_offset..y_offset + bh].copy_from_slice(&all_h[src_offset..src_offset + bh]);
        }

        // Y_h, Y_c
        let yh_off = d * dir_h_size;
        slots.y_h[yh_off..yh_off + dir_h_size].copy_from_slice(&last_h);
        slots.y_c[yh_off..yh_off + dir_h_size].copy_from_slice(&last_c);
    }

    Ok(())
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
    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };

    // Allocate output buffers, then delegate to lstm_into.
    let mut y_data = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut y_h_data = vec![0.0f32; num_dir * batch * hidden_size];
    let mut y_c_data = vec![0.0f32; num_dir * batch * hidden_size];

    let y_shape = vec![seq_len, num_dir, batch, hidden_size];
    let y_h_shape = vec![num_dir, batch, hidden_size];
    let y_c_shape = vec![num_dir, batch, hidden_size];

    lstm_into(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        initial_c,
        peephole,
        hidden_size,
        direction,
        activations,
        LstmOutputSlots {
            y: &mut y_data,
            y_h: &mut y_h_data,
            y_c: &mut y_c_data,
        },
    )?;

    let y = Tensor::new(y_data, y_shape);
    let y_h = Tensor::new(y_h_data, y_h_shape);
    let y_c = Tensor::new(y_c_data, y_c_shape);

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
