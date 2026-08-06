//! Simple (Elman) RNN operator kernel (ONNX spec).
//!
//! Supports forward, reverse, and bidirectional modes; optional bias;
//! configurable activation function; `clip`; `layout`; and variable-length
//! sequences.

use oxionnx_core::{OnnxError, Tensor};

use super::common::{
    clip_val, matmul_2d_a_bt, resolve_activation, step_is_valid, validate_direction,
    validate_rnn_shapes, Activation, RnnExtras, RnnShapeCheck,
};
use super::layout;

// ── Simple RNN ──────────────────────────────────────────────────────────────

/// Parameters of a single RNN direction pass.
struct RnnDirParams<'a> {
    w: &'a [f32],      // [hidden_size, input_size]
    r: &'a [f32],      // [hidden_size, hidden_size]
    bias: &'a [f32],   // [2*hidden_size] (Wb concat Rb)
    h_init: &'a [f32], // [batch * hidden_size]
    batch: usize,
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
    activation: Activation,
    sequence_lens: Option<&'a [usize]>,
    is_reverse: bool,
    clip: f32,
}

/// Run one direction of a simple (Elman) RNN over a sequence.
///
/// `h_t = activation(clip(x_t @ W^T + h_{t-1} @ R^T + Wb + Rb))`
fn simple_rnn_one_direction(x_seq: &[&[f32]], p: &RnnDirParams<'_>) -> (Vec<f32>, Vec<f32>) {
    let hidden_size = p.hidden_size;
    let batch = p.batch;
    let seq_len = p.seq_len;
    let mut h = p.h_init.to_vec();

    let wb = &p.bias[..hidden_size];
    let rb = &p.bias[hidden_size..2 * hidden_size];

    let mut last_valid_h = p.h_init.to_vec();
    let mut all_h = vec![0.0f32; seq_len * batch * hidden_size];
    let clip = p.clip;

    for (t, x_t) in x_seq.iter().enumerate().take(seq_len) {
        // x_t @ W^T = [batch, hidden_size]
        let wx = matmul_2d_a_bt(x_t, p.w, batch, p.input_size, hidden_size);
        // h @ R^T = [batch, hidden_size]
        let rh = matmul_2d_a_bt(&h, p.r, batch, hidden_size, hidden_size);

        let mut new_h = vec![0.0f32; batch * hidden_size];

        for b_idx in 0..batch {
            let base = b_idx * hidden_size;
            if step_is_valid(t, b_idx, seq_len, p.sequence_lens, p.is_reverse) {
                for j in 0..hidden_size {
                    let idx = base + j;
                    new_h[idx] = p
                        .activation
                        .apply(clip_val(wx[idx] + rh[idx] + wb[j] + rb[j], clip));
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
/// * `activation` – Any ONNX RNN activation name (`"Tanh"`, `"Relu"`, …).
///   Applied to every direction. An unknown name is a typed error.
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
    // One entry per direction so bidirectional keeps using the same activation.
    let acts = [activation, activation];
    simple_rnn_ext(
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        hidden_size,
        direction,
        Some(&acts),
        RnnExtras::default(),
    )
}

/// Simple (Elman) RNN kernel with per-direction activations and the optional
/// ONNX attributes of [`RnnExtras`] (`clip`, `layout`, `activation_alpha`,
/// `activation_beta`).
#[allow(clippy::too_many_arguments)]
pub fn simple_rnn_ext(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    sequence_lens: Option<&Tensor>,
    initial_h: Option<&Tensor>,
    hidden_size: usize,
    direction: &str,
    activations: Option<&[&str]>,
    extras: RnnExtras<'_>,
) -> Result<(Tensor, Tensor), OnnxError> {
    layout::validate_layout("RNN", extras.layout)?;

    // Convert batch-major inputs once; the kernel below is seq-major.
    let x_owned;
    let x_sm = if extras.layout == 1 {
        x_owned = layout::x_to_seq_major("RNN", x)?;
        &x_owned
    } else {
        x
    };
    let ih_owned = if extras.layout == 1 {
        initial_h
            .map(|t| layout::state_to_dir_major("RNN", t))
            .transpose()?
    } else {
        None
    };
    let initial_h = if extras.layout == 1 {
        ih_owned.as_ref()
    } else {
        initial_h
    };

    // Must run before `num_dir` is computed below: `direction` is compared
    // against `"reverse"`/`"bidirectional"` both here and again per-direction
    // (`is_reverse`, further down) with no other validation, so an
    // unrecognized string (typo, wrong case, ...) used to silently fall
    // through to plain forward-only execution instead of erroring — the same
    // gap `validate_direction` already closed for `LSTM`/`GRU` at the top of
    // `lstm_into_seq_major`/`gru_into_seq_major`. One call here covers both
    // downstream comparisons.
    validate_direction("RNN", direction)?;
    let num_dir: usize = if direction == "bidirectional" { 2 } else { 1 };
    let sizes = validate_rnn_shapes(RnnShapeCheck {
        op: "RNN",
        x: x_sm,
        w,
        r,
        b,
        initial_states: &[initial_h],
        hidden_size,
        gates: 1,
        num_dir,
    })?;

    let seq_len = x_sm.shape[0];
    let batch = x_sm.shape[1];
    let input_size = x_sm.shape[2];

    let (dir_w_size, dir_r_size, dir_b_size, dir_h_size) = (sizes.w, sizes.r, sizes.b, sizes.h);

    let zeros_b = vec![0.0f32; dir_b_size];
    let zeros_h = vec![0.0f32; dir_h_size];

    let seq_lens: Option<Vec<usize>> =
        sequence_lens.map(|t| t.data.iter().take(batch).map(|&v| v as usize).collect());
    let seq_lens_ref = seq_lens.as_deref();

    let step_size = batch * input_size;
    let x_steps: Vec<&[f32]> = (0..seq_len)
        .map(|t| &x_sm.data[t * step_size..(t + 1) * step_size])
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

        // One activation per direction.
        let act = resolve_activation(
            activations,
            extras.activation_alpha,
            extras.activation_beta,
            d,
            Activation::TANH,
        )?;

        let (all_h, last_h) = simple_rnn_one_direction(
            &ordered_steps,
            &RnnDirParams {
                w: w_d,
                r: r_d,
                bias: b_d,
                h_init,
                batch,
                input_size,
                hidden_size,
                seq_len,
                activation: act,
                sequence_lens: seq_lens_ref,
                is_reverse,
                clip: extras.clip,
            },
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

    let (y_shape, y_h_shape) =
        layout::output_shapes(extras.layout, seq_len, num_dir, batch, hidden_size);

    if extras.layout == 1 {
        let mut y_bm = vec![0.0f32; y_all.len()];
        let mut yh_bm = vec![0.0f32; y_h_all.len()];
        layout::y_to_batch_major(&y_all, seq_len, num_dir, batch, hidden_size, &mut y_bm);
        layout::state_to_batch_major(&y_h_all, num_dir, batch, hidden_size, &mut yh_bm);
        y_all = y_bm;
        y_h_all = yh_bm;
    }

    let y = Tensor::new(y_all, y_shape);
    let y_h = Tensor::new(y_h_all, y_h_shape);

    Ok((y, y_h))
}
