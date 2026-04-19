//! LabelEncoder ONNX-ML operator implementation.

use oxionnx_core::{OnnxError, OpContext, Tensor};

/// ONNX-ML LabelEncoder operator.
///
/// Maps input values to output values using lookup tables.
/// Supports int-to-int and float-to-float mappings.
pub fn label_encoder(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let keys_int64s = attrs.ints("keys_int64s");
    let values_int64s = attrs.ints("values_int64s");
    let keys_floats = attrs
        .float_lists
        .get("keys_floats")
        .cloned()
        .unwrap_or_default();
    let values_floats = attrs
        .float_lists
        .get("values_floats")
        .cloned()
        .unwrap_or_default();
    let default_int64 = attrs.i("default_int64", -1);
    let default_float = attrs.f("default_float", 0.0);

    let mut data = Vec::with_capacity(x.numel());

    if !keys_int64s.is_empty() && !values_int64s.is_empty() {
        // Int-to-int mapping: input f32 values are treated as integers
        for &val in &x.data {
            let key = val as i64;
            let found = keys_int64s
                .iter()
                .position(|&k| k == key)
                .map(|pos| {
                    if pos < values_int64s.len() {
                        values_int64s[pos] as f32
                    } else {
                        default_int64 as f32
                    }
                })
                .unwrap_or(default_int64 as f32);
            data.push(found);
        }
    } else if !keys_floats.is_empty() && !values_floats.is_empty() {
        // Float-to-float mapping
        for &val in &x.data {
            let found = keys_floats
                .iter()
                .position(|&k| (k - val).abs() < f32::EPSILON)
                .map(|pos| {
                    if pos < values_floats.len() {
                        values_floats[pos]
                    } else {
                        default_float
                    }
                })
                .unwrap_or(default_float);
            data.push(found);
        }
    } else {
        // No mapping defined: pass through with default
        data.extend(x.data.iter().map(|_| default_float));
    }

    Ok(vec![Tensor::new(data, x.shape.clone())])
}
