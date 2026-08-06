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
        // inputs[0]: signal  [B, T] or [B, T, 1] (real) or [B, T, 2] (complex)
        // inputs[1]: frame_step  scalar i64
        // inputs[2]: optional window  [frame_length]
        // inputs[3]: optional frame_length  scalar i64
        let signal = ctx.input(0)?;
        let frame_step_t = ctx.input(1)?;
        let frame_step = scalar_i64(frame_step_t, "STFT/frame_step")? as usize;
        let window_tensor = ctx.optional_input(2);
        let frame_length_opt = ctx.optional_input(3);

        let onesided = ctx.attrs().i("onesided", 1) != 0;

        // Signal shape: [B, T] (real), [B, T, 1] (real), or [B, T, 2]
        // (complex: interleaved real/imaginary components in the trailing
        // axis, matching the ONNX DFT input convention).
        let (batch, time_len, is_complex) = match signal.shape.len() {
            2 => (signal.shape[0], signal.shape[1], false),
            3 if signal.shape[2] == 2 => (signal.shape[0], signal.shape[1], true),
            3 if signal.shape[2] <= 1 => (signal.shape[0], signal.shape[1], false),
            3 => {
                return Err(OnnxError::ShapeMismatch(format!(
                    "STFT: signal last dim must be 1 or 2, got {}",
                    signal.shape[2]
                )));
            }
            d => {
                return Err(OnnxError::ShapeMismatch(format!(
                    "STFT: signal must be 2-D or 3-D, got {d}-D"
                )));
            }
        };

        // ONNX spec: "if the input or window tensors are complex, then
        // onesided output is not possible" -- the conjugate-symmetry
        // shortcut onesided relies on only holds for a real-valued signal.
        if is_complex && onesided {
            return Err(OnnxError::ShapeMismatch(
                "STFT: onesided=1 is not valid when the signal is complex \
                 (last dim size 2); onesided output requires a real signal"
                    .into(),
            ));
        }

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

        // Pre-fetch window coefficients if supplied, validating its length
        // against frame_length up front. Without this check, a window
        // shorter than frame_length would panic (index out of bounds) in
        // the frame-building loop below instead of returning a typed error.
        let window_data: Option<&[f32]> = match window_tensor {
            Some(wt) if wt.numel() > 0 => {
                if wt.numel() != frame_length {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "STFT: window length ({}) does not match frame_length ({})",
                        wt.numel(),
                        frame_length
                    )));
                }
                Some(wt.data.as_slice())
            }
            _ => None,
        };

        let n_frames = (time_len - frame_length) / frame_step + 1;
        let n_dft = if onesided {
            frame_length / 2 + 1
        } else {
            frame_length
        };

        let out_size = batch * n_frames * n_dft * 2;
        let mut out_data = vec![0.0f32; out_size];

        // Per-sample width in the flat `signal.data` buffer: 2 for an
        // interleaved (re, im) complex signal, 1 for a real signal (whether
        // via an explicit trailing dim of size 1, or the 2-D shorthand with
        // no trailing dim at all).
        let sample_width = if is_complex { 2 } else { 1 };

        for b in 0..batch {
            // Flat offset into the signal for this batch.
            let sig_stride = time_len * sample_width;
            let sig_base = b * sig_stride;

            for frame_idx in 0..n_frames {
                let t_start = frame_idx * frame_step;

                // Build the (re, im) frame, applying the optional real-valued
                // window to both components.
                let mut frame = Vec::with_capacity(frame_length * 2);
                for k in 0..frame_length {
                    let sample_idx = sig_base + (t_start + k) * sample_width;
                    let re = signal.data[sample_idx];
                    let im = if is_complex {
                        signal.data[sample_idx + 1]
                    } else {
                        0.0
                    };
                    let wnd_coeff = window_data.map(|wnd| wnd[k]).unwrap_or(1.0);
                    frame.push(re * wnd_coeff);
                    frame.push(im * wnd_coeff);
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
        OpContext {
            node,
            inputs,
            outer_scope: None,
            weights: None,
            registry: None,
        }
    }

    fn dummy_node(op: OpKind) -> Node {
        Node {
            name: "test".into(),
            op,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: Attributes::default(),
        }
    }

    /// Regression for a1-18/a3-10: a window shorter than frame_length must
    /// return a typed ShapeMismatch error, not panic (index out of bounds).
    #[test]
    fn stft_window_shorter_than_frame_length_errors_not_panics() {
        let signal = Tensor::new(vec![1.0f32; 16], vec![1, 16]);
        let frame_step_t = Tensor::new(vec![4.0], vec![1]);
        // window has 4 elements but frame_length (explicit input) says 8.
        let window = Tensor::new(vec![1.0f32; 4], vec![4]);
        let frame_length_t = Tensor::new(vec![8.0], vec![1]);
        let node = dummy_node(OpKind::STFT);
        let ctx = make_ctx(
            &node,
            vec![
                Some(&signal),
                Some(&frame_step_t),
                Some(&window),
                Some(&frame_length_t),
            ],
        );
        let result = STFTOp.execute(&ctx);
        assert!(result.is_err(), "mismatched window/frame_length must error");
    }

    /// Regression for a1-18: onesided=1 with a complex signal is spec-illegal
    /// ("if the input or window tensors are complex, then onesided output is
    /// not possible") and must be a typed error.
    #[test]
    fn stft_complex_signal_with_onesided_errors() {
        let signal = Tensor::new(vec![1.0f32; 32], vec![1, 16, 2]);
        let frame_step_t = Tensor::new(vec![4.0], vec![1]);
        let frame_length_t = Tensor::new(vec![8.0], vec![1]);
        let mut node = dummy_node(OpKind::STFT);
        node.attrs.ints.insert("onesided".into(), 1);
        let ctx = make_ctx(
            &node,
            vec![
                Some(&signal),
                Some(&frame_step_t),
                None,
                Some(&frame_length_t),
            ],
        );
        let result = STFTOp.execute(&ctx);
        assert!(result.is_err(), "onesided + complex signal must error");
    }

    /// Regression for a1-18: a complex input signal `[B, T, 2]` must be
    /// accepted (onesided=0) and both real/imaginary components must flow
    /// into the per-frame DFT, verified against `numpy.fft.fft` on a
    /// complex-valued input.
    #[test]
    fn stft_complex_signal_matches_numpy_fft() {
        // Complex signal: 4 samples, re = [1,2,3,4], im = [0,1,0,-1].
        // One frame (frame_length=4, frame_step=4, no window), onesided=0.
        // numpy reference:
        //   x = np.array([1+0j, 2+1j, 3+0j, 4-1j])
        //   np.fft.fft(x) == [10+0j, 0+2j, -2+0j, -4-2j]
        let signal_data = vec![
            1.0, 0.0, // re,im sample0
            2.0, 1.0, // sample1
            3.0, 0.0, // sample2
            4.0, -1.0, // sample3
        ];
        let signal = Tensor::new(signal_data, vec![1, 4, 2]);
        let frame_step_t = Tensor::new(vec![4.0], vec![1]);
        let frame_length_t = Tensor::new(vec![4.0], vec![1]);
        let mut node = dummy_node(OpKind::STFT);
        node.attrs.ints.insert("onesided".into(), 0);
        let ctx = make_ctx(
            &node,
            vec![
                Some(&signal),
                Some(&frame_step_t),
                None,
                Some(&frame_length_t),
            ],
        );
        let out = STFTOp.execute(&ctx).expect("complex STFT should succeed");
        // n_frames = (4-4)/4+1 = 1; n_dft = 4 (two-sided) -> shape [1,1,4,2]
        assert_eq!(out[0].shape, vec![1, 1, 4, 2]);
        let expected = [(10.0_f32, 0.0), (0.0, 2.0), (-2.0, 0.0), (-4.0, -2.0)];
        for (k, &(re, im)) in expected.iter().enumerate() {
            assert!(
                (out[0].data[k * 2] - re).abs() < 1e-4,
                "bin {k} re: got {}, expected {re}",
                out[0].data[k * 2]
            );
            assert!(
                (out[0].data[k * 2 + 1] - im).abs() < 1e-4,
                "bin {k} im: got {}, expected {im}",
                out[0].data[k * 2 + 1]
            );
        }
    }
}
