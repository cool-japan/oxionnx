//! Scaler ONNX-ML operator implementation.

use oxionnx_core::{OnnxError, OpContext, Tensor};

use super::shape::batch_dims;

/// ONNX-ML Scaler operator.
///
/// `Y = (X - offset) * scale`
///
/// Per the ONNX-ML schema, `offset` and `scale` are "length of features or
/// length 1"; a length-1 list broadcasts across every feature column. Any
/// other length is a malformed model and yields a typed error.
pub fn scaler(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let offset = attrs
        .float_lists
        .get("offset")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let scale = attrs
        .float_lists
        .get("scale")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let (n, features) = batch_dims(x, "Scaler")?;
    validate_len(offset, features, "offset")?;
    validate_len(scale, features, "scale")?;

    let mut data = x.data.clone();
    for i in 0..n {
        let row = i * features;
        for f in 0..features {
            let off = param_at(offset, f, 0.0);
            let sc = param_at(scale, f, 1.0);
            let idx = row + f;
            data[idx] = (data[idx] - off) * sc;
        }
    }

    Ok(vec![Tensor::new(data, x.shape.clone())])
}

/// Reject parameter lists that are neither absent, scalar, nor per-feature.
fn validate_len(values: &[f32], features: usize, name: &str) -> Result<(), OnnxError> {
    if values.is_empty() || values.len() == 1 || values.len() == features {
        Ok(())
    } else {
        Err(OnnxError::InvalidModel(format!(
            "Scaler: '{name}' has {} entries; expected 1 or {features} (one per feature)",
            values.len()
        )))
    }
}

/// Fetch the parameter for feature `f`, broadcasting a length-1 list.
#[inline]
fn param_at(values: &[f32], f: usize, default: f32) -> f32 {
    if values.len() == 1 {
        values.first().copied().unwrap_or(default)
    } else {
        values.get(f).copied().unwrap_or(default)
    }
}
