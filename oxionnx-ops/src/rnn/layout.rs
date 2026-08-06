//! ONNX `layout` attribute (opset 14) support for `RNN` / `GRU` / `LSTM`.
//!
//! `layout = 0` (the default) is the seq-major form the kernels compute in:
//!
//! | tensor       | layout 0                                    | layout 1                                    |
//! |--------------|---------------------------------------------|---------------------------------------------|
//! | `X`          | `[seq, batch, input]`                       | `[batch, seq, input]`                       |
//! | `initial_h`  | `[num_dir, batch, hidden]`                  | `[batch, num_dir, hidden]`                  |
//! | `initial_c`  | `[num_dir, batch, hidden]`                  | `[batch, num_dir, hidden]`                  |
//! | `Y`          | `[seq, num_dir, batch, hidden]`             | `[batch, seq, num_dir, hidden]`             |
//! | `Y_h`, `Y_c` | `[num_dir, batch, hidden]`                  | `[batch, num_dir, hidden]`                  |
//!
//! This module converts the batch-major (`layout = 1`) form to and from the
//! seq-major form so the kernels themselves stay layout-agnostic.

use oxionnx_core::{OnnxError, Tensor};

/// Reject any `layout` value outside the ONNX-defined `{0, 1}`.
pub(super) fn validate_layout(op: &str, layout: i64) -> Result<(), OnnxError> {
    if layout == 0 || layout == 1 {
        Ok(())
    } else {
        Err(OnnxError::InvalidModel(format!(
            "{op}: layout must be 0 ([seq, batch, ...]) or 1 ([batch, seq, ...]), got {layout}"
        )))
    }
}

/// `X`: `[batch, seq, input]` → `[seq, batch, input]`.
pub(super) fn x_to_seq_major(op: &str, x: &Tensor) -> Result<Tensor, OnnxError> {
    if x.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: layout=1 requires X of rank 3 [batch_size, seq_length, input_size], got {:?}",
            x.shape
        )));
    }
    let (batch, seq_len, input_size) = (x.shape[0], x.shape[1], x.shape[2]);
    let needed = batch
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(input_size))
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: X shape overflows usize")))?;
    if x.data.len() < needed {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: X holds {} elements but shape {:?} needs {needed}",
            x.data.len(),
            x.shape
        )));
    }

    let mut out = vec![0.0f32; needed];
    for b in 0..batch {
        for t in 0..seq_len {
            let src = (b * seq_len + t) * input_size;
            let dst = (t * batch + b) * input_size;
            out[dst..dst + input_size].copy_from_slice(&x.data[src..src + input_size]);
        }
    }
    Ok(Tensor::new(out, vec![seq_len, batch, input_size]))
}

/// `initial_h` / `initial_c`: `[batch, num_dir, hidden]` → `[num_dir, batch, hidden]`.
pub(super) fn state_to_dir_major(op: &str, t: &Tensor) -> Result<Tensor, OnnxError> {
    if t.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: layout=1 requires initial_h/initial_c of rank 3 \
             [batch_size, num_directions, hidden_size], got {:?}",
            t.shape
        )));
    }
    let (batch, num_dir, hidden) = (t.shape[0], t.shape[1], t.shape[2]);
    let needed = batch
        .checked_mul(num_dir)
        .and_then(|v| v.checked_mul(hidden))
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: state shape overflows usize")))?;
    if t.data.len() < needed {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: initial state holds {} elements but shape {:?} needs {needed}",
            t.data.len(),
            t.shape
        )));
    }

    let mut out = vec![0.0f32; needed];
    for b in 0..batch {
        for d in 0..num_dir {
            let src = (b * num_dir + d) * hidden;
            let dst = (d * batch + b) * hidden;
            out[dst..dst + hidden].copy_from_slice(&t.data[src..src + hidden]);
        }
    }
    Ok(Tensor::new(out, vec![num_dir, batch, hidden]))
}

/// `Y`: `[seq, num_dir, batch, hidden]` → `[batch, seq, num_dir, hidden]`.
///
/// `dst` must already hold `seq_len * num_dir * batch * hidden` elements.
pub(super) fn y_to_batch_major(
    src: &[f32],
    seq_len: usize,
    num_dir: usize,
    batch: usize,
    hidden: usize,
    dst: &mut [f32],
) {
    for t in 0..seq_len {
        for d in 0..num_dir {
            for b in 0..batch {
                let s = ((t * num_dir + d) * batch + b) * hidden;
                let o = ((b * seq_len + t) * num_dir + d) * hidden;
                dst[o..o + hidden].copy_from_slice(&src[s..s + hidden]);
            }
        }
    }
}

/// `Y_h` / `Y_c`: `[num_dir, batch, hidden]` → `[batch, num_dir, hidden]`.
///
/// `dst` must already hold `num_dir * batch * hidden` elements.
pub(super) fn state_to_batch_major(
    src: &[f32],
    num_dir: usize,
    batch: usize,
    hidden: usize,
    dst: &mut [f32],
) {
    for d in 0..num_dir {
        for b in 0..batch {
            let s = (d * batch + b) * hidden;
            let o = (b * num_dir + d) * hidden;
            dst[o..o + hidden].copy_from_slice(&src[s..s + hidden]);
        }
    }
}

/// Output shapes for `Y`, `Y_h`/`Y_c` under the requested layout.
pub(super) fn output_shapes(
    layout: i64,
    seq_len: usize,
    num_dir: usize,
    batch: usize,
    hidden: usize,
) -> (Vec<usize>, Vec<usize>) {
    if layout == 1 {
        (
            vec![batch, seq_len, num_dir, hidden],
            vec![batch, num_dir, hidden],
        )
    } else {
        (
            vec![seq_len, num_dir, batch, hidden],
            vec![num_dir, batch, hidden],
        )
    }
}
