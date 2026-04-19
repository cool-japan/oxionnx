//! STFT operator: Short-Time Fourier Transform.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::dft::dft_1d;
use super::helpers::scalar_i64;

pub struct STFTOp;
impl Operator for STFTOp {
    fn op_type(&self) -> &str {
        "STFT"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        // inputs[0]: signal  [B, T] or [B, T, 1]
        // inputs[1]: frame_step  scalar i64
        // inputs[2]: optional window  [frame_length]
        // inputs[3]: optional frame_length  scalar i64
        let signal = ctx.input(0)?;
        let frame_step_t = ctx.input(1)?;
        let frame_step = scalar_i64(frame_step_t, "STFT/frame_step")? as usize;
        let window_tensor = ctx.optional_input(2);
        let frame_length_opt = ctx.optional_input(3);

        let onesided = ctx.attrs().i("onesided", 1) != 0;

        // Signal shape: [B, T] or [B, T, 1].
        let (batch, time_len) = match signal.shape.len() {
            2 => (signal.shape[0], signal.shape[1]),
            3 if signal.shape[2] <= 1 => (signal.shape[0], signal.shape[1]),
            3 => {
                return Err(OnnxError::ShapeMismatch(format!(
                    "STFT: signal last dim must be 1, got {}",
                    signal.shape[2]
                )));
            }
            d => {
                return Err(OnnxError::ShapeMismatch(format!(
                    "STFT: signal must be 2-D or 3-D, got {d}-D"
                )));
            }
        };

        // Resolve frame_length.
        let frame_length: usize = if let Some(fl_t) = frame_length_opt {
            scalar_i64(fl_t, "STFT/frame_length")? as usize
        } else if let Some(wt) = window_tensor {
            if wt.numel() > 0 {
                // The window tensor shape gives us the frame length.
                wt.shape
                    .last()
                    .copied()
                    .ok_or_else(|| OnnxError::ShapeMismatch("STFT: empty window tensor".into()))?
            } else {
                return Err(OnnxError::ShapeMismatch(
                    "STFT: frame_length not provided and window tensor is empty".into(),
                ));
            }
        } else {
            return Err(OnnxError::ShapeMismatch(
                "STFT: frame_length not provided and no window tensor".into(),
            ));
        };

        if frame_step == 0 {
            return Err(OnnxError::ShapeMismatch(
                "STFT: frame_step must be > 0".into(),
            ));
        }
        if frame_length == 0 {
            return Err(OnnxError::ShapeMismatch(
                "STFT: frame_length must be > 0".into(),
            ));
        }
        if frame_length > time_len {
            return Err(OnnxError::ShapeMismatch(format!(
                "STFT: frame_length ({frame_length}) > signal length ({time_len})"
            )));
        }

        let n_frames = (time_len - frame_length) / frame_step + 1;
        let n_dft = if onesided {
            frame_length / 2 + 1
        } else {
            frame_length
        };

        // Pre-fetch window coefficients if supplied.
        let window_data: Option<&[f32]> = window_tensor.and_then(|wt| {
            if wt.numel() > 0 {
                Some(wt.data.as_slice())
            } else {
                None
            }
        });

        let out_size = batch * n_frames * n_dft * 2;
        let mut out_data = vec![0.0f32; out_size];

        for b in 0..batch {
            // Flat offset into the signal for this batch.
            let sig_stride = if signal.shape.len() == 3 {
                time_len * signal.shape[2]
            } else {
                time_len
            };
            let sig_base = b * sig_stride;

            for frame_idx in 0..n_frames {
                let t_start = frame_idx * frame_step;

                // Build real-valued frame, apply optional window.
                let mut frame = Vec::with_capacity(frame_length * 2);
                for k in 0..frame_length {
                    let sample_idx = sig_base + t_start + k;
                    let sample = signal.data[sample_idx];
                    let windowed = if let Some(wnd) = window_data {
                        sample * wnd[k]
                    } else {
                        sample
                    };
                    frame.push(windowed);
                    frame.push(0.0); // imaginary part = 0
                }

                // DFT.
                let spectrum = dft_1d(&frame, frame_length, false);

                // Write n_dft complex samples into output.
                let dst_base = (b * n_frames * n_dft + frame_idx * n_dft) * 2;
                let copy_len = n_dft * 2;
                out_data[dst_base..dst_base + copy_len].copy_from_slice(&spectrum[..copy_len]);
            }
        }

        let out_shape = vec![batch, n_frames, n_dft, 2];
        Ok(vec![Tensor::new(out_data, out_shape)])
    }

    fn supports_output_slots(&self) -> bool {
        true
    }
}
