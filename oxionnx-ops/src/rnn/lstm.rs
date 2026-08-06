//! LSTM operator kernel (ONNX spec).
//!
//! Supports forward, reverse, and bidirectional modes; optional peephole
//! connections; per-gate activation overrides; `clip`; `layout`; and
//! variable-length sequences.

use oxionnx_core::{OnnxError, Tensor};

use super::common::{
    clip_val, matmul_2d_a_bt, resolve_activation, step_is_valid, validate_direction,
    validate_rnn_shapes, Activation, RnnExtras, RnnShapeCheck,
};
use super::layout;

// ── LSTM ────────────────────────────────────────────────────────────────────

/// Parameters of a single LSTM direction pass.
struct LstmDirParams<'a> {
    w: &'a [f32],      // [4*hidden_size, input_size]
    r: &'a [f32],      // [4*hidden_size, hidden_size]
    bias: &'a [f32],   // [8*hidden_size] (Wb concat Rb)
    h_init: &'a [f32], // [batch * hidden_size]
    c_init: &'a [f32], // [batch * hidden_size]
    batch: usize,
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
    sequence_lens: Option<&'a [usize]>,
    is_reverse: bool,
    peephole: Option<&'a [f32]>,  // [3*hidden_size]: P_i, P_o, P_f
    activations: [Activation; 3], // [gate, cell_candidate, output_transform]
    clip: f32,
}

/// Run one direction of LSTM over a sequence.
///
/// Returns `(all_hidden, last_h, last_c)` where `all_hidden` is flat
/// `[seq_len, batch, hidden_size]` with zeros for masked timesteps.
fn lstm_one_direction(x_seq: &[&[f32]], p: &LstmDirParams<'_>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden_size = p.hidden_size;
    let batch = p.batch;
    let seq_len = p.seq_len;
    let gate4 = 4 * hidden_size;
    let mut h = p.h_init.to_vec();
    let mut c = p.c_init.to_vec();

    // Pre-extract biases: ONNX layout [Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc]
    let wb = &p.bias[..gate4];
    let rb = &p.bias[gate4..gate4 * 2];

    // Track the last valid hidden/cell state per batch element (for Y_h / Y_c).
    let mut last_valid_h = p.h_init.to_vec();
    let mut last_valid_c = p.c_init.to_vec();

    let mut all_h = vec![0.0f32; seq_len * batch * hidden_size];

    let act_gate = p.activations[0];
    let act_cell = p.activations[1];
    let act_out = p.activations[2];
    let clip = p.clip;

    // `new_h`/`new_c` are allocated once and ping-ponged with `h`/`c` via
    // `mem::swap` at the end of the timestep loop, instead of a fresh `Vec`
    // every timestep: for every `(b_idx, j)` cell, exactly one of the
    // `if`/`else` branches below unconditionally overwrites it each
    // iteration (the `if` branch assigns every `new_c[cell_idx]` /
    // `new_h[cell_idx]` it touches; the `else` branch `copy_from_slice`s the
    // full `[base, base + hidden_size)` range), so there is nothing to clear
    // before reuse — the swapped-in stale contents are never read.
    let mut new_h = vec![0.0f32; batch * hidden_size];
    let mut new_c = vec![0.0f32; batch * hidden_size];

    for (t, x_t) in x_seq.iter().enumerate().take(seq_len) {
        // x_t @ W^T = [batch, 4*hs]
        let wx = matmul_2d_a_bt(x_t, p.w, batch, p.input_size, gate4);
        // h @ R^T = [batch, 4*hs]
        let rh = matmul_2d_a_bt(&h, p.r, batch, hidden_size, gate4);

        for b_idx in 0..batch {
            let base = b_idx * hidden_size;
            if step_is_valid(t, b_idx, seq_len, p.sequence_lens, p.is_reverse) {
                for j in 0..hidden_size {
                    // ONNX gate order: i, o, f, c
                    let i_idx = j;
                    let o_idx = hidden_size + j;
                    let f_idx = 2 * hidden_size + j;
                    let c_idx = 3 * hidden_size + j;
                    let gb = b_idx * gate4;
                    let cell_idx = base + j;

                    // Peephole for input and forget gates (use C_{t-1})
                    let p_i = p.peephole.map_or(0.0, |pp| pp[j] * c[cell_idx]);
                    let p_f = p
                        .peephole
                        .map_or(0.0, |pp| pp[2 * hidden_size + j] * c[cell_idx]);

                    let it = act_gate.apply(clip_val(
                        wx[gb + i_idx] + rh[gb + i_idx] + wb[i_idx] + rb[i_idx] + p_i,
                        clip,
                    ));
                    let ft = act_gate.apply(clip_val(
                        wx[gb + f_idx] + rh[gb + f_idx] + wb[f_idx] + rb[f_idx] + p_f,
                        clip,
                    ));
                    let ct = act_cell.apply(clip_val(
                        wx[gb + c_idx] + rh[gb + c_idx] + wb[c_idx] + rb[c_idx],
                        clip,
                    ));

                    new_c[cell_idx] = ft * c[cell_idx] + it * ct;

                    // Peephole for output gate (uses NEW C_t)
                    let p_o = p
                        .peephole
                        .map_or(0.0, |pp| pp[hidden_size + j] * new_c[cell_idx]);

                    let ot = act_gate.apply(clip_val(
                        wx[gb + o_idx] + rh[gb + o_idx] + wb[o_idx] + rb[o_idx] + p_o,
                        clip,
                    ));

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

        std::mem::swap(&mut h, &mut new_h);
        std::mem::swap(&mut c, &mut new_c);
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

/// Seq-major (`layout = 0`) LSTM core.
#[allow(clippy::too_many_arguments)]
fn lstm_into_seq_major(
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
    extras: RnnExtras<'_>,
    slots: LstmOutputSlots<'_>,
) -> Result<(), OnnxError> {
    validate_direction("LSTM", direction)?;
    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };
    let sizes = validate_rnn_shapes(RnnShapeCheck {
        op: "LSTM",
        x,
        w,
        r,
        b,
        initial_states: &[initial_h, initial_c],
        hidden_size,
        gates: 4,
        num_dir,
    })?;

    let seq_len = x.shape[0];
    let batch = x.shape[1];
    let input_size = x.shape[2];

    let (dir_w_size, dir_r_size, dir_b_size, dir_h_size) = (sizes.w, sizes.r, sizes.b, sizes.h);
    let dir_p_size = 3 * hidden_size;
    if let Some(pt) = peephole {
        if pt.data.len() < num_dir * dir_p_size {
            return Err(OnnxError::ShapeMismatch(format!(
                "LSTM: input P holds {} elements but {num_dir} direction(s) with \
                 hidden_size={hidden_size} need {}",
                pt.data.len(),
                num_dir * dir_p_size
            )));
        }
    }

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
        let acts = parse_lstm_activations(activations, &extras, d)?;

        let (all_h, last_h, last_c) = lstm_one_direction(
            &ordered_steps,
            &LstmDirParams {
                w: w_d,
                r: r_d,
                bias: b_d,
                h_init,
                c_init,
                batch,
                input_size,
                hidden_size,
                seq_len,
                sequence_lens: seq_lens_ref,
                is_reverse,
                peephole: p_d,
                activations: acts,
                clip: extras.clip,
            },
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

/// Core LSTM computation writing directly into caller-provided output buffers.
///
/// All inputs mirror `lstm()`; the three output tensors are written into `slots`
/// in-place. No new `Vec<f32>` is allocated for outputs on the `layout = 0` path;
/// only the internal per-direction working buffers (`h`, `c`, `wx`, `rh`,
/// `new_h`, `new_c`) that belong to the computation kernel itself are allocated.
///
/// With `layout = 1` the inputs are converted to seq-major, the kernel runs, and
/// the outputs are written back in batch-major order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lstm_into_ext(
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
    extras: RnnExtras<'_>,
    slots: LstmOutputSlots<'_>,
) -> Result<(), OnnxError> {
    layout::validate_layout("LSTM", extras.layout)?;
    if extras.layout == 0 {
        return lstm_into_seq_major(
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
            extras,
            slots,
        );
    }

    // layout == 1: [batch, seq, ...] in, [batch, seq, num_dir, hidden] out.
    let x_sm = layout::x_to_seq_major("LSTM", x)?;
    let ih_sm = initial_h
        .map(|t| layout::state_to_dir_major("LSTM", t))
        .transpose()?;
    let ic_sm = initial_c
        .map(|t| layout::state_to_dir_major("LSTM", t))
        .transpose()?;

    let seq_len = x_sm.shape[0];
    let batch = x_sm.shape[1];
    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };

    let mut y_tmp = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut yh_tmp = vec![0.0f32; num_dir * batch * hidden_size];
    let mut yc_tmp = vec![0.0f32; num_dir * batch * hidden_size];

    lstm_into_seq_major(
        &x_sm,
        w,
        r,
        b,
        sequence_lens,
        ih_sm.as_ref(),
        ic_sm.as_ref(),
        peephole,
        hidden_size,
        direction,
        activations,
        extras,
        LstmOutputSlots {
            y: &mut y_tmp,
            y_h: &mut yh_tmp,
            y_c: &mut yc_tmp,
        },
    )?;

    if slots.y.len() != y_tmp.len() {
        slots.y.resize(y_tmp.len(), 0.0f32);
    }
    if slots.y_h.len() != yh_tmp.len() {
        slots.y_h.resize(yh_tmp.len(), 0.0f32);
    }
    if slots.y_c.len() != yc_tmp.len() {
        slots.y_c.resize(yc_tmp.len(), 0.0f32);
    }
    layout::y_to_batch_major(&y_tmp, seq_len, num_dir, batch, hidden_size, slots.y);
    layout::state_to_batch_major(&yh_tmp, num_dir, batch, hidden_size, slots.y_h);
    layout::state_to_batch_major(&yc_tmp, num_dir, batch, hidden_size, slots.y_c);

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
    lstm_ext(
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
        RnnExtras::default(),
    )
}

/// LSTM operator kernel with the optional ONNX attributes of [`RnnExtras`]
/// (`clip`, `layout`, `activation_alpha`, `activation_beta`).
#[allow(clippy::too_many_arguments)]
pub fn lstm_ext(
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
    extras: RnnExtras<'_>,
) -> Result<(Tensor, Tensor, Tensor), OnnxError> {
    layout::validate_layout("LSTM", extras.layout)?;
    if x.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "LSTM: X must be 3D, got {:?}",
            x.shape
        )));
    }
    let (seq_len, batch) = if extras.layout == 1 {
        (x.shape[1], x.shape[0])
    } else {
        (x.shape[0], x.shape[1])
    };
    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };

    let mut y_data = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut y_h_data = vec![0.0f32; num_dir * batch * hidden_size];
    let mut y_c_data = vec![0.0f32; num_dir * batch * hidden_size];

    let (y_shape, state_shape) =
        layout::output_shapes(extras.layout, seq_len, num_dir, batch, hidden_size);

    lstm_into_ext(
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
        extras,
        LstmOutputSlots {
            y: &mut y_data,
            y_h: &mut y_h_data,
            y_c: &mut y_c_data,
        },
    )?;

    let y = Tensor::new(y_data, y_shape);
    let y_h = Tensor::new(y_h_data, state_shape.clone());
    let y_c = Tensor::new(y_c_data, state_shape);

    Ok((y, y_h, y_c))
}

/// Parse LSTM activations for direction `d` from the optional attribute.
/// Default: [Sigmoid, Tanh, Tanh].
///
/// An activation name outside the ONNX list is rejected with a typed error
/// instead of silently becoming `Tanh`.
fn parse_lstm_activations(
    activations: Option<&[&str]>,
    extras: &RnnExtras<'_>,
    direction_idx: usize,
) -> Result<[Activation; 3], OnnxError> {
    let off = direction_idx * 3;
    let alphas = extras.activation_alpha;
    let betas = extras.activation_beta;
    Ok([
        resolve_activation(activations, alphas, betas, off, Activation::SIGMOID)?,
        resolve_activation(activations, alphas, betas, off + 1, Activation::TANH)?,
        resolve_activation(activations, alphas, betas, off + 2, Activation::TANH)?,
    ])
}
