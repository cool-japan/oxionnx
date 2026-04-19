//! GRU operator kernel (ONNX spec).
//!
//! Supports forward, reverse, and bidirectional modes; optional bias;
//! linear-before-reset mode; per-gate activation overrides; and variable-length sequences.

use oxionnx_core::{OnnxError, Tensor};

use super::common::{matmul_2d_a_bt, step_is_valid, Activation};

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

/// Caller-provided output slots for zero-copy GRU computation.
///
/// The caller pre-allocates each buffer and passes mutable references here.
/// `gru_into` resizes each Vec if needed but never replaces the backing
/// allocation (pointer stability across repeated calls).
pub(crate) struct GruOutputSlots<'a> {
    /// All hidden outputs: flat `[seq_len * num_dir * batch * hidden_size]`
    pub y: &'a mut Vec<f32>,
    /// Last hidden state: flat `[num_dir * batch * hidden_size]`
    pub y_h: &'a mut Vec<f32>,
}

/// Core GRU computation writing directly into caller-provided output buffers.
///
/// All inputs mirror `gru()`. The two output tensors are written into `slots`
/// in-place. No new `Vec<f32>` is allocated for outputs; only the internal
/// per-direction working buffers that belong to the computation kernel are allocated.
///
/// Existing `gru()` delegates here to avoid duplication.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gru_into(
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
    slots: GruOutputSlots<'_>,
) -> Result<(), OnnxError> {
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

    // Resize output buffers if necessary — never replace them (pointer stability).
    let y_len = seq_len * num_dir * batch * hidden_size;
    let yh_len = num_dir * batch * hidden_size;
    if slots.y.len() != y_len {
        slots.y.resize(y_len, 0.0f32);
    }
    if slots.y_h.len() != yh_len {
        slots.y_h.resize(yh_len, 0.0f32);
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
            slots.y[y_offset..y_offset + bh].copy_from_slice(&all_h[src_offset..src_offset + bh]);
        }

        let yh_off = d * dir_h_size;
        slots.y_h[yh_off..yh_off + dir_h_size].copy_from_slice(&last_h);
    }

    Ok(())
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
    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };

    // Allocate output buffers, then delegate to gru_into.
    let mut y_data = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut y_h_data = vec![0.0f32; num_dir * batch * hidden_size];

    let y_shape = vec![seq_len, num_dir, batch, hidden_size];
    let y_h_shape = vec![num_dir, batch, hidden_size];

    gru_into(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        hidden_size,
        direction,
        linear_before_reset,
        activations,
        GruOutputSlots {
            y: &mut y_data,
            y_h: &mut y_h_data,
        },
    )?;

    let y = Tensor::new(y_data, y_shape);
    let y_h = Tensor::new(y_h_data, y_h_shape);

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
