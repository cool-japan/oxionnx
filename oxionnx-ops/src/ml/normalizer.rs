//! Normalizer ONNX-ML operator implementation.

use oxionnx_core::{OnnxError, OpContext, Tensor};

/// ONNX-ML Normalizer operator.
///
/// Normalizes each row of the input tensor.
/// `norm`: "MAX", "L1", "L2"
pub fn normalizer(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();
    let norm = attrs.s("norm");

    // Treat as 2D: [N, features]
    let n = x.shape[0];
    let features = x.numel() / n;

    let mut data = x.data.clone();

    for i in 0..n {
        let offset = i * features;
        let row = &mut data[offset..offset + features];

        match norm {
            "MAX" => {
                let max_abs = row.iter().copied().fold(0.0f32, |acc, v| acc.max(v.abs()));
                if max_abs > 0.0 {
                    for v in row.iter_mut() {
                        *v /= max_abs;
                    }
                }
            }
            "L1" => {
                let sum_abs: f32 = row.iter().map(|v| v.abs()).sum();
                if sum_abs > 0.0 {
                    for v in row.iter_mut() {
                        *v /= sum_abs;
                    }
                }
            }
            _ => {
                // Default to L2
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
