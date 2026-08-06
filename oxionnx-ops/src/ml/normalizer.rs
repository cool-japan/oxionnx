//! Normalizer ONNX-ML operator implementation.

use oxionnx_core::{OnnxError, OpContext, Tensor};

use super::shape::batch_dims;

/// Normalization mode selected by the `norm` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Norm {
    Max,
    L1,
    L2,
}

impl Norm {
    /// Parse the `norm` attribute. The ONNX-ML schema default is `MAX`.
    fn parse(s: &str) -> Result<Self, OnnxError> {
        match s {
            "MAX" | "" => Ok(Self::Max),
            "L1" => Ok(Self::L1),
            "L2" => Ok(Self::L2),
            other => Err(OnnxError::InvalidModel(format!(
                "Normalizer: unknown norm '{other}' (expected MAX, L1 or L2)"
            ))),
        }
    }
}

/// ONNX-ML Normalizer operator.
///
/// Normalizes each sample (row) of the input independently. A 1-D `[C]` input
/// is a single sample with `C` features, so it is normalized as one vector.
pub fn normalizer(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();
    let norm = Norm::parse(attrs.s("norm"))?;

    let (n, features) = batch_dims(x, "Normalizer")?;

    let mut data = x.data.clone();

    for i in 0..n {
        let offset = i * features;
        let row = &mut data[offset..offset + features];

        match norm {
            Norm::Max => {
                let max_abs = row.iter().copied().fold(0.0f32, |acc, v| acc.max(v.abs()));
                if max_abs > 0.0 {
                    for v in row.iter_mut() {
                        *v /= max_abs;
                    }
                }
            }
            Norm::L1 => {
                let sum_abs: f32 = row.iter().map(|v| v.abs()).sum();
                if sum_abs > 0.0 {
                    for v in row.iter_mut() {
                        *v /= sum_abs;
                    }
                }
            }
            Norm::L2 => {
                let sum_sq: f32 = row.iter().map(|v| v * v).sum();
                let norm_val = sum_sq.sqrt();
                if norm_val > 0.0 {
                    for v in row.iter_mut() {
                        *v /= norm_val;
                    }
                }
            }
        }
    }

    Ok(vec![Tensor::new(data, x.shape.clone())])
}
