//! DFT operator: 1-D Discrete Fourier Transform (forward / inverse).

use oxifft::Complex;
use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::helpers::{complex_to_interleaved, interleaved_to_complex, resolve_dft_length};

/// Perform a 1-D FFT/IFFT on a slice of complex pairs (interleaved re/im).
///
/// `signal` must have length exactly `n * 2` (n complex samples interleaved).
/// Returns `n` complex samples as interleaved re/im.
pub(super) fn dft_1d(signal_interleaved: &[f32], n: usize, inverse: bool) -> Vec<f32> {
    let mut c: Vec<Complex<f32>> = interleaved_to_complex(signal_interleaved);
    // Pad or truncate to exactly n samples.
    c.resize(n, Complex::new(0.0, 0.0));

    let result = if inverse {
        oxifft::ifft::<f32>(&c)
    } else {
        oxifft::fft::<f32>(&c)
    };
    complex_to_interleaved(&result)
}

pub struct DFTOp;
impl Operator for DFTOp {
    fn op_type(&self) -> &str {
        "DFT"
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let inverse = ctx.attrs().i("inverse", 0) != 0;
        let onesided = if inverse {
            false // ONNX spec: onesided is ignored when inverse=true
        } else {
            ctx.attrs().i("onesided", 0) != 0
        };

        // Input shape is either:
        //   [batch, signal_len]         — real signal
        //   [batch, signal_len, 1]      — real signal (channel dim = 1)
        //   [batch, signal_len, 2]      — complex signal (channel dim = 2)
        let ndim = input.shape.len();
        let (batch, signal_len, is_complex) = match ndim {
            2 => {
                let b = input.shape[0];
                let s = input.shape[1];
                (b, s, false)
            }
            3 => {
                let b = input.shape[0];
                let s = input.shape[1];
                let ch = input.shape[2];
                if ch == 2 {
                    (b, s, true)
                } else if ch == 1 {
                    (b, s, false)
                } else {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "DFT: last dim must be 1 or 2, got {ch}"
                    )));
                }
            }
            _ => {
                return Err(OnnxError::ShapeMismatch(format!(
                    "DFT: expected 2-D or 3-D input, got {ndim}-D"
                )));
            }
        };

        let n = resolve_dft_length(ctx, signal_len)?;
        let out_len = if onesided { n / 2 + 1 } else { n };

        let mut out_data = vec![0.0f32; batch * out_len * 2];

        // Stride between successive batch slices in the input.
        let in_stride = if is_complex {
            signal_len * 2
        } else {
            signal_len
        };

        for b in 0..batch {
            // Build an interleaved complex buffer of length n for this batch.
            let frame_interleaved: Vec<f32> = if is_complex {
                // Already interleaved re/im pairs.
                let src = &input.data[b * in_stride..(b * in_stride + signal_len * 2)];
                let mut v = src.to_vec();
                // Pad or truncate to n samples (each sample = 2 f32s).
                v.resize(n * 2, 0.0);
                v
            } else {
                // Real-only: imaginary part = 0.
                let src = &input.data[b * in_stride..(b * in_stride + signal_len)];
                let mut v = Vec::with_capacity(n * 2);
                for &re in src.iter().take(n) {
                    v.push(re);
                    v.push(0.0);
                }
                // Pad with zeros if src was shorter than n.
                while v.len() < n * 2 {
                    v.push(0.0);
                }
                v
            };

            let result = dft_1d(&frame_interleaved, n, inverse);

            // Write only the first out_len complex samples.
            let dst_base = b * out_len * 2;
            let copy_len = out_len * 2;
            out_data[dst_base..dst_base + copy_len].copy_from_slice(&result[..copy_len]);
        }

        let out_shape = vec![batch, out_len, 2];
        Ok(vec![Tensor::new(out_data, out_shape)])
    }
}
