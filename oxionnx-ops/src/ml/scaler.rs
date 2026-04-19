//! Scaler ONNX-ML operator implementation.

use oxionnx_core::{OnnxError, OpContext, Tensor};

/// ONNX-ML Scaler operator.
///
/// Output = (X - offset) * scale
/// offset and scale are per-feature vectors that broadcast along the feature axis.
pub fn scaler(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let offset = attrs.float_lists.get("offset").cloned().unwrap_or_default();
    let scale = attrs.float_lists.get("scale").cloned().unwrap_or_default();

    let n = x.shape[0];
    let features = x.numel() / n;

    let mut data = x.data.clone();
    for i in 0..n {
        for f in 0..features {
            let idx = i * features + f;
            let off = if f < offset.len() { offset[f] } else { 0.0 };
            let sc = if f < scale.len() { scale[f] } else { 1.0 };
            data[idx] = (data[idx] - off) * sc;
        }
    }

    Ok(vec![Tensor::new(data, x.shape.clone())])
}
