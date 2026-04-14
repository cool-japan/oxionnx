//! Audio/Signal Processing operators: DFT, STFT, window functions, MelWeightMatrix, Bernoulli.

use std::time::{SystemTime, UNIX_EPOCH};

use oxifft::Complex;
use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

// ──────────────────────────────────────────────────────────────────────────────
// Internal conversion helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Convert an interleaved `[re0, im0, re1, im1, ...]` slice to `Vec<Complex<f32>>`.
fn interleaved_to_complex(data: &[f32]) -> Vec<Complex<f32>> {
    data.chunks_exact(2)
        .map(|pair| Complex::new(pair[0], pair[1]))
        .collect()
}

/// Flatten a `Vec<Complex<f32>>` to interleaved `[re0, im0, re1, im1, ...]`.
fn complex_to_interleaved(cs: &[Complex<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(cs.len() * 2);
    for c in cs {
        out.push(c.re);
        out.push(c.im);
    }
    out
}

/// Extract a scalar i64 from a tensor (first element cast to i64).
fn scalar_i64(t: &Tensor, label: &str) -> Result<i64, OnnxError> {
    t.data
        .first()
        .copied()
        .map(|v| v as i64)
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{label}: empty scalar tensor")))
}

/// Extract a scalar f32 from a tensor (first element).
fn scalar_f32(t: &Tensor, label: &str) -> Result<f32, OnnxError> {
    t.data
        .first()
        .copied()
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{label}: empty scalar tensor")))
}

// ──────────────────────────────────────────────────────────────────────────────
// Window function helpers
// ──────────────────────────────────────────────────────────────────────────────

enum WindowKind {
    Hann,
    Hamming,
    Blackman,
}

/// Compute a window of `size` samples.
///
/// - `periodic = true`  → denominator = N   (DFT use-case, opset default)
/// - `periodic = false` → denominator = N-1 (filter-design use-case)
fn window_generic(size: usize, periodic: bool, kind: &WindowKind) -> Vec<f32> {
    use std::f32::consts::PI;
    if size == 0 {
        return Vec::new();
    }
    let denom = if periodic {
        size as f32
    } else {
        (size - 1) as f32
    };
    (0..size)
        .map(|n| {
            let n_f = n as f32;
            match kind {
                WindowKind::Hann => 0.5 - 0.5 * (2.0 * PI * n_f / denom).cos(),
                WindowKind::Hamming => 0.543_478_26 - 0.456_521_74 * (2.0 * PI * n_f / denom).cos(),
                WindowKind::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * n_f / denom).cos()
                        + 0.08 * (4.0 * PI * n_f / denom).cos()
                }
            }
        })
        .collect()
}

/// Shared logic for all three window-function operators.
fn execute_window_op(
    ctx: &OpContext<'_>,
    kind: &WindowKind,
    op_name: &str,
) -> Result<Vec<Tensor>, OnnxError> {
    let size_t = ctx.input(0)?;
    let size = scalar_i64(size_t, &format!("{op_name}/size"))? as usize;

    // periodic attr: 1 = periodic (DFT), 0 = symmetric (filter). Default = 1.
    let periodic = ctx.attrs().i("periodic", 1) != 0;

    let data = window_generic(size, periodic, kind);
    let out = Tensor::new(data, vec![size]);
    Ok(vec![out])
}

// ──────────────────────────────────────────────────────────────────────────────
// HannWindow
// ──────────────────────────────────────────────────────────────────────────────

pub struct HannWindowOp;
impl Operator for HannWindowOp {
    fn op_type(&self) -> &str {
        "HannWindow"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        execute_window_op(ctx, &WindowKind::Hann, "HannWindow")
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// HammingWindow
// ──────────────────────────────────────────────────────────────────────────────

pub struct HammingWindowOp;
impl Operator for HammingWindowOp {
    fn op_type(&self) -> &str {
        "HammingWindow"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        execute_window_op(ctx, &WindowKind::Hamming, "HammingWindow")
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BlackmanWindow
// ──────────────────────────────────────────────────────────────────────────────

pub struct BlackmanWindowOp;
impl Operator for BlackmanWindowOp {
    fn op_type(&self) -> &str {
        "BlackmanWindow"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        execute_window_op(ctx, &WindowKind::Blackman, "BlackmanWindow")
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DFT — 1-D Discrete Fourier Transform (forward / inverse)
// ──────────────────────────────────────────────────────────────────────────────

/// Perform a 1-D FFT/IFFT on a slice of complex pairs (interleaved re/im).
///
/// `signal` must have length exactly `n * 2` (n complex samples interleaved).
/// Returns `n` complex samples as interleaved re/im.
fn dft_1d(signal_interleaved: &[f32], n: usize, inverse: bool) -> Vec<f32> {
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

/// Resolve the DFT length N from optional input[1] or from signal shape.
fn resolve_dft_length(ctx: &OpContext<'_>, signal_len: usize) -> Result<usize, OnnxError> {
    if let Some(t) = ctx.optional_input(1) {
        let n = scalar_i64(t, "DFT/dft_length")?;
        if n <= 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "DFT: dft_length must be positive, got {n}"
            )));
        }
        Ok(n as usize)
    } else {
        Ok(signal_len)
    }
}

pub struct DFTOp;
impl Operator for DFTOp {
    fn op_type(&self) -> &str {
        "DFT"
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

// ──────────────────────────────────────────────────────────────────────────────
// STFT — Short-Time Fourier Transform
// ──────────────────────────────────────────────────────────────────────────────

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
}

// ──────────────────────────────────────────────────────────────────────────────
// MelWeightMatrix
// ──────────────────────────────────────────────────────────────────────────────

/// Convert Hz to HTK Mel scale.
#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert HTK Mel to Hz.
#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10_f32.powf(mel / 2595.0) - 1.0)
}

/// Uniformly-spaced values from `start` to `stop` (inclusive), `count` points.
fn linspace_f32(start: f32, stop: f32, count: usize) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f32;
    (0..count).map(|i| start + i as f32 * step).collect()
}

/// Triangular mel-filter weight for a single spectrogram bin frequency.
///
/// The filter is a triangle between `lower` (0) → `center` (peak) → `upper` (0).
#[inline]
fn triangular_weight(bin_hz: f32, lower: f32, center: f32, upper: f32) -> f32 {
    if bin_hz <= lower || bin_hz >= upper {
        0.0
    } else if bin_hz < center {
        (bin_hz - lower) / (center - lower).max(f32::EPSILON)
    } else {
        (upper - bin_hz) / (upper - center).max(f32::EPSILON)
    }
}

pub struct MelWeightMatrixOp;
impl Operator for MelWeightMatrixOp {
    fn op_type(&self) -> &str {
        "MelWeightMatrix"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        // All five inputs are scalar tensors (may be f32 or integral f32).
        let num_mel_bins = scalar_f32(ctx.input(0)?, "MelWeightMatrix/num_mel_bins")? as usize;
        let dft_length = scalar_f32(ctx.input(1)?, "MelWeightMatrix/dft_length")? as usize;
        let sample_rate = scalar_f32(ctx.input(2)?, "MelWeightMatrix/sample_rate")?;
        let lower_edge_hertz = scalar_f32(ctx.input(3)?, "MelWeightMatrix/lower_edge_hertz")?;
        let upper_edge_hertz = scalar_f32(ctx.input(4)?, "MelWeightMatrix/upper_edge_hertz")?;

        if num_mel_bins == 0 || dft_length == 0 {
            return Err(OnnxError::ShapeMismatch(
                "MelWeightMatrix: num_mel_bins and dft_length must be positive".into(),
            ));
        }
        if lower_edge_hertz >= upper_edge_hertz {
            return Err(OnnxError::ShapeMismatch(
                "MelWeightMatrix: lower_edge_hertz must be < upper_edge_hertz".into(),
            ));
        }

        let num_spectrogram_bins = dft_length / 2 + 1;

        // Spectrogram bin frequencies: [0, sample_rate/2] linearly.
        let spec_bins_hz = linspace_f32(0.0, sample_rate / 2.0, num_spectrogram_bins);

        // Mel band edges: num_mel_bins + 2 points (lower, num_mel_bins centers, upper).
        let lower_mel = hz_to_mel(lower_edge_hertz);
        let upper_mel = hz_to_mel(upper_edge_hertz);
        let mel_pts = linspace_f32(lower_mel, upper_mel, num_mel_bins + 2);
        let mel_hz: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();

        // Build weight matrix: shape [num_spectrogram_bins, num_mel_bins].
        let mut data = vec![0.0f32; num_spectrogram_bins * num_mel_bins];
        for s in 0..num_spectrogram_bins {
            let bin_hz = spec_bins_hz[s];
            for m in 0..num_mel_bins {
                let lower = mel_hz[m];
                let center = mel_hz[m + 1];
                let upper = mel_hz[m + 2];
                data[s * num_mel_bins + m] = triangular_weight(bin_hz, lower, center, upper);
            }
        }

        let out_shape = vec![num_spectrogram_bins, num_mel_bins];
        Ok(vec![Tensor::new(data, out_shape)])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Bernoulli
// ──────────────────────────────────────────────────────────────────────────────

/// A simple xorshift64* PRNG — no external crates.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // xorshift must not start with 0.
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self(s)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // Use upper 24 bits for mantissa (avoids lower-bit patterns in xorshift).
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

pub struct BernoulliOp;
impl Operator for BernoulliOp {
    fn op_type(&self) -> &str {
        "Bernoulli"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let probs = ctx.input(0)?;

        // Resolve seed: if attr "seed" is present and non-zero, use it.
        // Otherwise seed from system time.
        let seed_attr = ctx.attrs().f("seed", 0.0);
        let rng_seed: u64 = if seed_attr != 0.0 {
            // Reinterpret the f32 bits as a u64 seed.
            (seed_attr.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15)
        } else {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E3779B97F4A7C15)
        };

        let mut rng = Xorshift64::new(rng_seed);
        let data: Vec<f32> = probs
            .data
            .iter()
            .map(|&p| if rng.next_f32() < p { 1.0 } else { 0.0 })
            .collect();

        Ok(vec![Tensor::new(data, probs.shape.clone())])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
        OpContext {
            node,
            inputs,
            outer_scope: None,
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

    fn node_with_attrs(op: OpKind, periodic: i64) -> Node {
        let mut n = dummy_node(op);
        n.attrs.ints.insert("periodic".into(), periodic);
        n
    }

    // ── Window function tests ─────────────────────────────────────────────────

    #[test]
    fn hann_window_periodic_8() {
        let size_t = Tensor::new(vec![8.0], vec![1]);
        let node = dummy_node(OpKind::HannWindow);
        let ctx = make_ctx(&node, vec![Some(&size_t)]);
        let out = HannWindowOp.execute(&ctx).expect("HannWindow failed");
        assert_eq!(out[0].shape, vec![8]);
        // First and last samples of a periodic Hann window should both be ~0.
        assert!((out[0].data[0]).abs() < 1e-6, "w[0] should be ~0");
    }

    #[test]
    fn hamming_window_symmetric_4() {
        let size_t = Tensor::new(vec![4.0], vec![1]);
        let node = node_with_attrs(OpKind::HammingWindow, 0); // symmetric
        let ctx = make_ctx(&node, vec![Some(&size_t)]);
        let out = HammingWindowOp.execute(&ctx).expect("HammingWindow failed");
        assert_eq!(out[0].shape, vec![4]);
        // Symmetric Hamming with N=4, denom = N-1 = 3.
        // w[0] = alpha - beta*cos(0) = 0.54347826 - 0.45652174 = 0.08695652
        let w0 = out[0].data[0];
        let expected = 0.543_478_26_f32 - 0.456_521_74_f32;
        assert!(
            (w0 - expected).abs() < 1e-4,
            "w[0]={w0} expected={expected}"
        );
        // w[1]: 0.54347826 - 0.45652174*cos(2pi/3) ≈ 0.5435 + 0.2283 ≈ 0.7717
        let w1 = out[0].data[1];
        let expected1 =
            0.543_478_26_f32 - 0.456_521_74_f32 * (2.0 * std::f32::consts::PI / 3.0_f32).cos();
        assert!(
            (w1 - expected1).abs() < 1e-4,
            "w[1]={w1} expected={expected1}"
        );
    }

    #[test]
    fn blackman_window_shape() {
        let size_t = Tensor::new(vec![16.0], vec![1]);
        let node = dummy_node(OpKind::BlackmanWindow);
        let ctx = make_ctx(&node, vec![Some(&size_t)]);
        let out = BlackmanWindowOp
            .execute(&ctx)
            .expect("BlackmanWindow failed");
        assert_eq!(out[0].shape, vec![16]);
    }

    // ── DFT tests ─────────────────────────────────────────────────────────────

    #[test]
    fn dft_real_signal_dc() {
        // A constant real signal of all 1s → DC bin should equal N, others ~0.
        let n = 8usize;
        let data: Vec<f32> = vec![1.0; n];
        let input = Tensor::new(data, vec![1, n]);
        let node = dummy_node(OpKind::DFT);
        let ctx = make_ctx(&node, vec![Some(&input), None]);
        let out = DFTOp.execute(&ctx).expect("DFT failed");
        // Shape: [1, 8, 2]
        assert_eq!(out[0].shape, vec![1, n, 2]);
        let re_dc = out[0].data[0];
        let im_dc = out[0].data[1];
        assert!((re_dc - n as f32).abs() < 1e-3, "DC re={re_dc}");
        assert!(im_dc.abs() < 1e-3, "DC im={im_dc}");
    }

    #[test]
    fn dft_roundtrip() {
        // Forward DFT then inverse DFT should recover original signal.
        let n = 8usize;
        let orig: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let input = Tensor::new(orig.clone(), vec![1, n]);
        let node = dummy_node(OpKind::DFT);
        let ctx = make_ctx(&node, vec![Some(&input), None]);
        let fwd = DFTOp.execute(&ctx).expect("DFT forward failed");

        // fwd[0]: shape [1, 8, 2] — treat as complex input for inverse.
        // The inverse DFT input must be provided as complex (last dim = 2).
        let spectrum = &fwd[0];

        let mut inv_node = dummy_node(OpKind::DFT);
        inv_node.attrs.ints.insert("inverse".into(), 1);
        let ctx2 = make_ctx(&inv_node, vec![Some(spectrum), None]);
        let back = DFTOp.execute(&ctx2).expect("IDFT failed");
        // Shape: [1, 8, 2]
        assert_eq!(back[0].shape, vec![1, n, 2]);
        for (i, &expected) in orig.iter().enumerate() {
            let re = back[0].data[i * 2];
            assert!(
                (re - expected).abs() < 1e-3,
                "sample {i}: re={re} expected={expected}"
            );
        }
    }

    #[test]
    fn dft_onesided() {
        let n = 8usize;
        let data: Vec<f32> = vec![1.0; n];
        let input = Tensor::new(data, vec![1, n]);
        let mut node = dummy_node(OpKind::DFT);
        node.attrs.ints.insert("onesided".into(), 1);
        let ctx = make_ctx(&node, vec![Some(&input), None]);
        let out = DFTOp.execute(&ctx).expect("DFT onesided failed");
        // onesided → n/2+1 = 5 bins
        assert_eq!(out[0].shape, vec![1, 5, 2]);
    }

    // ── STFT tests ─────────────────────────────────────────────────────────────

    #[test]
    fn stft_basic_shape() {
        // Signal: [1, 16], frame_step=4, window=[8] (all ones), frame_length=8, onesided=1
        let signal = Tensor::new(vec![1.0f32; 16], vec![1, 16]);
        let frame_step_t = Tensor::new(vec![4.0], vec![1]);
        let window = Tensor::new(vec![1.0f32; 8], vec![8]);
        let frame_length_t = Tensor::new(vec![8.0], vec![1]);
        let mut node = dummy_node(OpKind::STFT);
        node.attrs.ints.insert("onesided".into(), 1);
        let ctx = make_ctx(
            &node,
            vec![
                Some(&signal),
                Some(&frame_step_t),
                Some(&window),
                Some(&frame_length_t),
            ],
        );
        let out = STFTOp.execute(&ctx).expect("STFT failed");
        // n_frames = (16 - 8) / 4 + 1 = 3
        // n_dft    = 8/2 + 1 = 5  (onesided)
        assert_eq!(out[0].shape, vec![1, 3, 5, 2]);
    }

    // ── MelWeightMatrix tests ──────────────────────────────────────────────────

    #[test]
    fn mel_weight_matrix_shape() {
        let num_mel = Tensor::new(vec![40.0], vec![1]);
        let dft_len = Tensor::new(vec![512.0], vec![1]);
        let sample_rate = Tensor::new(vec![16000.0], vec![1]);
        let lower = Tensor::new(vec![0.0], vec![1]);
        let upper = Tensor::new(vec![8000.0], vec![1]);
        let node = dummy_node(OpKind::MelWeightMatrix);
        let ctx = make_ctx(
            &node,
            vec![
                Some(&num_mel),
                Some(&dft_len),
                Some(&sample_rate),
                Some(&lower),
                Some(&upper),
            ],
        );
        let out = MelWeightMatrixOp
            .execute(&ctx)
            .expect("MelWeightMatrix failed");
        // num_spectrogram_bins = 512/2+1 = 257
        assert_eq!(out[0].shape, vec![257, 40]);
        // All weights should be non-negative.
        for &v in &out[0].data {
            assert!(v >= 0.0, "negative weight: {v}");
        }
    }

    #[test]
    fn mel_weight_matrix_non_negative_and_bounded() {
        let num_mel = Tensor::new(vec![20.0], vec![1]);
        let dft_len = Tensor::new(vec![256.0], vec![1]);
        let sample_rate = Tensor::new(vec![8000.0], vec![1]);
        let lower = Tensor::new(vec![80.0], vec![1]);
        let upper = Tensor::new(vec![3000.0], vec![1]);
        let node = dummy_node(OpKind::MelWeightMatrix);
        let ctx = make_ctx(
            &node,
            vec![
                Some(&num_mel),
                Some(&dft_len),
                Some(&sample_rate),
                Some(&lower),
                Some(&upper),
            ],
        );
        let out = MelWeightMatrixOp
            .execute(&ctx)
            .expect("MelWeightMatrix failed");
        for &v in &out[0].data {
            assert!(
                v >= 0.0 && v <= 1.0 + f32::EPSILON,
                "weight out of [0,1]: {v}"
            );
        }
    }

    // ── Bernoulli tests ───────────────────────────────────────────────────────

    #[test]
    fn bernoulli_always_zero() {
        // Probabilities all 0.0 → all samples should be 0.
        let probs = Tensor::new(vec![0.0f32; 100], vec![100]);
        let mut node = dummy_node(OpKind::Bernoulli);
        node.attrs.floats.insert("seed".into(), 42.0);
        let ctx = make_ctx(&node, vec![Some(&probs)]);
        let out = BernoulliOp.execute(&ctx).expect("Bernoulli failed");
        assert_eq!(out[0].shape, vec![100]);
        for &v in &out[0].data {
            assert_eq!(v, 0.0, "expected 0, got {v}");
        }
    }

    #[test]
    fn bernoulli_always_one() {
        // Probabilities all 1.0 → all samples should be 1.
        let probs = Tensor::new(vec![1.0f32; 100], vec![100]);
        let mut node = dummy_node(OpKind::Bernoulli);
        node.attrs.floats.insert("seed".into(), 42.0);
        let ctx = make_ctx(&node, vec![Some(&probs)]);
        let out = BernoulliOp.execute(&ctx).expect("Bernoulli failed");
        for &v in &out[0].data {
            assert_eq!(v, 1.0, "expected 1, got {v}");
        }
    }

    #[test]
    fn bernoulli_seeded_reproducible() {
        // Same seed → same output.
        let probs = Tensor::new(vec![0.5f32; 50], vec![50]);
        let mut node = dummy_node(OpKind::Bernoulli);
        node.attrs.floats.insert("seed".into(), 7.0);

        let ctx1 = make_ctx(&node, vec![Some(&probs)]);
        let out1 = BernoulliOp.execute(&ctx1).expect("Bernoulli run1 failed");

        let ctx2 = make_ctx(&node, vec![Some(&probs)]);
        let out2 = BernoulliOp.execute(&ctx2).expect("Bernoulli run2 failed");

        assert_eq!(
            out1[0].data, out2[0].data,
            "Seeded Bernoulli not reproducible"
        );
    }

    #[test]
    fn bernoulli_shape_preserved() {
        let probs = Tensor::new(vec![0.5f32; 12], vec![3, 4]);
        let node = dummy_node(OpKind::Bernoulli);
        let ctx = make_ctx(&node, vec![Some(&probs)]);
        let out = BernoulliOp.execute(&ctx).expect("Bernoulli failed");
        assert_eq!(out[0].shape, vec![3, 4]);
    }
}
