//! Internal conversion and scalar-extraction helpers shared across DSP operators.

use oxifft::Complex;
use oxionnx_core::{OnnxError, OpContext, Tensor};

/// Convert an interleaved `[re0, im0, re1, im1, ...]` slice to `Vec<Complex<f32>>`.
pub(super) fn interleaved_to_complex(data: &[f32]) -> Vec<Complex<f32>> {
    data.chunks_exact(2)
        .map(|pair| Complex::new(pair[0], pair[1]))
        .collect()
}

/// Flatten a `Vec<Complex<f32>>` to interleaved `[re0, im0, re1, im1, ...]`.
pub(super) fn complex_to_interleaved(cs: &[Complex<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(cs.len() * 2);
    for c in cs {
        out.push(c.re);
        out.push(c.im);
    }
    out
}

/// Extract a scalar i64 from a tensor (first element cast to i64).
pub(super) fn scalar_i64(t: &Tensor, label: &str) -> Result<i64, OnnxError> {
    t.data
        .first()
        .copied()
        .map(|v| v as i64)
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{label}: empty scalar tensor")))
}

/// Extract a scalar f32 from a tensor (first element).
pub(super) fn scalar_f32(t: &Tensor, label: &str) -> Result<f32, OnnxError> {
    t.data
        .first()
        .copied()
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{label}: empty scalar tensor")))
}

/// Resolve the DFT length N from optional input[1] or from signal shape.
pub(super) fn resolve_dft_length(
    ctx: &OpContext<'_>,
    signal_len: usize,
) -> Result<usize, OnnxError> {
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
