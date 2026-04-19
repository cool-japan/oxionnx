//! MelWeightMatrix operator: triangular mel-filter weight matrix construction.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::helpers::scalar_f32;

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

    fn supports_output_slots(&self) -> bool {
        true
    }
}
