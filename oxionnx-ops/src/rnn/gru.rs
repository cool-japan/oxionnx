//! GRU operator kernel (ONNX spec).
//!
//! Supports forward, reverse, and bidirectional modes; optional bias;
//! linear-before-reset mode; per-gate activation overrides; `clip`; `layout`;
//! and variable-length sequences.

use oxionnx_core::{OnnxError, Tensor};

use super::common::{
    clip_val, matmul_2d_a_bt, resolve_activation, step_is_valid, validate_direction,
    validate_rnn_shapes, Activation, RnnExtras, RnnShapeCheck,
};
use super::layout;

// ── GRU ─────────────────────────────────────────────────────────────────────

/// Parameters of a single GRU direction pass.
struct GruDirParams<'a> {
    w: &'a [f32],      // [3*hidden_size, input_size]
    r: &'a [f32],      // [3*hidden_size, hidden_size]
    bias: &'a [f32],   // [6*hidden_size] (Wb concat Rb)
    h_init: &'a [f32], // [batch * hidden_size]
    batch: usize,
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
    linear_before_reset: bool,
    sequence_lens: Option<&'a [usize]>,
    is_reverse: bool,
    activations: [Activation; 2], // [gate, hidden_candidate]
    clip: f32,
}

/// Run one direction of GRU over a sequence.
///
/// Returns `(all_hidden, last_h)`.
fn gru_one_direction(x_seq: &[&[f32]], p: &GruDirParams<'_>) -> (Vec<f32>, Vec<f32>) {
    let hidden_size = p.hidden_size;
    let batch = p.batch;
    let seq_len = p.seq_len;
    let gate3 = 3 * hidden_size;
    let mut h = p.h_init.to_vec();

    // Biases: [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]
    let wb = &p.bias[..gate3];
    let rb = &p.bias[gate3..gate3 * 2];

    // Rh block of R: rows [2*hidden_size, 3*hidden_size).
    let r_h_slice = &p.r[2 * hidden_size * hidden_size..3 * hidden_size * hidden_size];

    let mut last_valid_h = p.h_init.to_vec();
    let mut all_h = vec![0.0f32; seq_len * batch * hidden_size];

    let act_gate = p.activations[0];
    let act_hidden = p.activations[1];
    let clip = p.clip;

    // Reset gates for the whole hidden vector of the current batch element.
    // Allocated once; `linear_before_reset = 0` needs `rt[k]` for every k, not
    // just the output unit j (see the ONNX equation for `ht`).
    let mut rt_vec = vec![0.0f32; hidden_size];

    // `new_h` is allocated once and ping-ponged with `h` via `mem::swap` at
    // the end of the timestep loop, instead of a fresh `Vec` every timestep:
    // for every `(b_idx, j)` cell, exactly one of the `if`/`else` branches
    // below unconditionally overwrites it each iteration (the `if` branch
    // assigns every `new_h[cell_idx]` it touches; the `else` branch
    // `copy_from_slice`s the full `[base, base + hidden_size)` range), so
    // there is nothing to clear before reuse — the swapped-in stale contents
    // are never read.
    let mut new_h = vec![0.0f32; batch * hidden_size];

    for (t, x_t) in x_seq.iter().enumerate().take(seq_len) {
        // x_t @ W^T = [batch, 3*hs]
        let wx = matmul_2d_a_bt(x_t, p.w, batch, p.input_size, gate3);
        // h @ R^T = [batch, 3*hs]
        let rh = matmul_2d_a_bt(&h, p.r, batch, hidden_size, gate3);

        for b_idx in 0..batch {
            let base = b_idx * hidden_size;
            if step_is_valid(t, b_idx, seq_len, p.sequence_lens, p.is_reverse) {
                let gb = b_idx * gate3;

                // Pass 1: reset gate for every hidden unit of this batch element.
                for (j, rt_slot) in rt_vec.iter_mut().enumerate() {
                    let r_idx = hidden_size + j;
                    *rt_slot = act_gate.apply(clip_val(
                        wx[gb + r_idx] + rh[gb + r_idx] + wb[r_idx] + rb[r_idx],
                        clip,
                    ));
                }

                // Pass 2: update gate + hidden candidate.
                for j in 0..hidden_size {
                    // Gate order: z, r, h
                    let z_idx = j;
                    let h_idx = 2 * hidden_size + j;

                    let zt = act_gate.apply(clip_val(
                        wx[gb + z_idx] + rh[gb + z_idx] + wb[z_idx] + rb[z_idx],
                        clip,
                    ));

                    let ht_candidate = if p.linear_before_reset {
                        // ht = g(Wh*Xt + rt ⊙ (Rh*H_{t-1} + Rbh) + Wbh)
                        act_hidden.apply(clip_val(
                            wx[gb + h_idx] + rt_vec[j] * (rh[gb + h_idx] + rb[h_idx]) + wb[h_idx],
                            clip,
                        ))
                    } else {
                        // ht = g(Wh*Xt + Rh*(rt ⊙ H_{t-1}) + Wbh + Rbh)
                        //    = g(Wh*Xt + Σ_k Rh[j][k]·rt[k]·h[k] + Wbh + Rbh)
                        let row = &r_h_slice[j * hidden_size..(j + 1) * hidden_size];
                        let h_row = &h[base..base + hidden_size];
                        let mut rh_val = 0.0f32;
                        for kk in 0..hidden_size {
                            rh_val += row[kk] * rt_vec[kk] * h_row[kk];
                        }
                        act_hidden.apply(clip_val(
                            wx[gb + h_idx] + rh_val + wb[h_idx] + rb[h_idx],
                            clip,
                        ))
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

        std::mem::swap(&mut h, &mut new_h);
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
    gru_into_ext(
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
        RnnExtras::default(),
        slots,
    )
}

/// Seq-major (`layout = 0`) GRU core.
#[allow(clippy::too_many_arguments)]
fn gru_into_seq_major(
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
    extras: RnnExtras<'_>,
    slots: GruOutputSlots<'_>,
) -> Result<(), OnnxError> {
    validate_direction("GRU", direction)?;
    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };
    let sizes = validate_rnn_shapes(RnnShapeCheck {
        op: "GRU",
        x,
        w,
        r,
        b,
        initial_states: &[initial_h],
        hidden_size,
        gates: 3,
        num_dir,
    })?;

    let seq_len = x.shape[0];
    let batch = x.shape[1];
    let input_size = x.shape[2];

    let (dir_w_size, dir_r_size, dir_b_size, dir_h_size) = (sizes.w, sizes.r, sizes.b, sizes.h);

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

        let acts = parse_gru_activations(activations, &extras, d)?;

        let (all_h, last_h) = gru_one_direction(
            &ordered_steps,
            &GruDirParams {
                w: w_d,
                r: r_d,
                bias: b_d,
                h_init,
                batch,
                input_size,
                hidden_size,
                seq_len,
                linear_before_reset,
                sequence_lens: seq_lens_ref,
                is_reverse,
                activations: acts,
                clip: extras.clip,
            },
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

/// GRU computation honouring the optional ONNX attributes in [`RnnExtras`].
///
/// With `layout = 0` this is exactly [`gru_into`]. With `layout = 1` the inputs
/// are converted to seq-major, the kernel runs, and the outputs are written back
/// in batch-major order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gru_into_ext(
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
    extras: RnnExtras<'_>,
    slots: GruOutputSlots<'_>,
) -> Result<(), OnnxError> {
    layout::validate_layout("GRU", extras.layout)?;
    if extras.layout == 0 {
        return gru_into_seq_major(
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
            extras,
            slots,
        );
    }

    // layout == 1: [batch, seq, ...] in, [batch, seq, num_dir, hidden] out.
    let x_sm = layout::x_to_seq_major("GRU", x)?;
    let ih_sm = initial_h
        .map(|t| layout::state_to_dir_major("GRU", t))
        .transpose()?;

    let seq_len = x_sm.shape[0];
    let batch = x_sm.shape[1];
    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };

    let mut y_tmp = vec![0.0f32; seq_len * num_dir * batch * hidden_size];
    let mut yh_tmp = vec![0.0f32; num_dir * batch * hidden_size];

    gru_into_seq_major(
        &x_sm,
        w,
        r,
        b,
        sequence_lens,
        ih_sm.as_ref(),
        hidden_size,
        direction,
        linear_before_reset,
        activations,
        extras,
        GruOutputSlots {
            y: &mut y_tmp,
            y_h: &mut yh_tmp,
        },
    )?;

    if slots.y.len() != y_tmp.len() {
        slots.y.resize(y_tmp.len(), 0.0f32);
    }
    if slots.y_h.len() != yh_tmp.len() {
        slots.y_h.resize(yh_tmp.len(), 0.0f32);
    }
    layout::y_to_batch_major(&y_tmp, seq_len, num_dir, batch, hidden_size, slots.y);
    layout::state_to_batch_major(&yh_tmp, num_dir, batch, hidden_size, slots.y_h);

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
    gru_ext(
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
        RnnExtras::default(),
    )
}

/// GRU operator kernel with the optional ONNX attributes of [`RnnExtras`]
/// (`clip`, `layout`, `activation_alpha`, `activation_beta`).
#[allow(clippy::too_many_arguments)]
pub fn gru_ext(
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
    extras: RnnExtras<'_>,
) -> Result<(Tensor, Tensor), OnnxError> {
    layout::validate_layout("GRU", extras.layout)?;
    if x.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "GRU: X must be 3D, got {:?}",
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

    let (y_shape, y_h_shape) =
        layout::output_shapes(extras.layout, seq_len, num_dir, batch, hidden_size);

    gru_into_ext(
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
        extras,
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
///
/// An activation name outside the ONNX list is rejected with a typed error
/// instead of silently becoming `Tanh`.
fn parse_gru_activations(
    activations: Option<&[&str]>,
    extras: &RnnExtras<'_>,
    direction_idx: usize,
) -> Result<[Activation; 2], OnnxError> {
    let off = direction_idx * 2;
    let alphas = extras.activation_alpha;
    let betas = extras.activation_beta;
    Ok([
        resolve_activation(activations, alphas, betas, off, Activation::SIGMOID)?,
        resolve_activation(activations, alphas, betas, off + 1, Activation::TANH)?,
    ])
}
