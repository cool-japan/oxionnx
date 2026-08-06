//! LabelEncoder ONNX-ML operator implementation.

use std::collections::HashMap;

use oxionnx_core::{OnnxError, OpContext, Tensor};

/// Keys of a LabelEncoder lookup table, mapped to the output value.
///
/// Key and value attributes are chosen independently, so every combination the
/// f32 tensor runtime can represent (int→int, int→float, float→int,
/// float→float, tensor→tensor) is supported.
enum KeyMap {
    /// `keys_int64s` (or an integral `keys_tensor`): input is truncated to i64.
    Ints(HashMap<i64, f32>),
    /// `keys_floats` (or a float `keys_tensor`): input is matched bitwise.
    Floats(HashMap<u32, f32>),
}

impl KeyMap {
    #[inline]
    fn lookup(&self, value: f32, default: f32) -> f32 {
        if value.is_nan() {
            // NaN never compares equal to a key, and `NaN as i64` saturates to
            // 0 which would spuriously match the key 0.
            return default;
        }
        match self {
            Self::Ints(map) => map.get(&(value as i64)).copied().unwrap_or(default),
            Self::Floats(map) => map.get(&float_key(value)).copied().unwrap_or(default),
        }
    }
}

/// Canonical bit pattern of a float key (`-0.0` and `+0.0` are the same key).
#[inline]
fn float_key(value: f32) -> u32 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

/// ONNX-ML LabelEncoder operator.
///
/// Maps input values to output values through a lookup table built from the
/// `keys_*` / `values_*` attribute pair. Keys and values may use different
/// types (`keys_int64s` + `values_floats` is emitted by ordinal/target
/// encoders). Unrepresentable combinations (string keys or values) and
/// inconsistent tables produce a typed error instead of silent defaults.
pub fn label_encoder(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    // ── Values (the mapping target decides which default applies) ──────────
    let values_int64s = attrs.ints("values_int64s");
    let values_floats = attrs
        .float_lists
        .get("values_floats")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let values_tensor = attrs.tensors.get("values_tensor");

    if !attrs.string_list("values_strings").is_empty() {
        return Err(OnnxError::Unsupported(
            "LabelEncoder: 'values_strings' output is not representable in an f32 tensor".into(),
        ));
    }

    let (values, default) = if let Some(tensor) = values_tensor {
        let default = attrs
            .tensors
            .get("default_tensor")
            .and_then(|t| t.data.first().copied())
            .unwrap_or(attrs.f("default_float", -0.0));
        (tensor.data.clone(), default)
    } else if !values_int64s.is_empty() {
        let default = attrs.i("default_int64", -1) as f32;
        (values_int64s.iter().map(|&v| v as f32).collect(), default)
    } else if !values_floats.is_empty() {
        (values_floats.to_vec(), attrs.f("default_float", -0.0))
    } else {
        return Err(OnnxError::InvalidModel(
            "LabelEncoder: no 'values_*' attribute provided".into(),
        ));
    };

    // ── Keys ───────────────────────────────────────────────────────────────
    let keys_int64s = attrs.ints("keys_int64s");
    let keys_floats = attrs
        .float_lists
        .get("keys_floats")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let keys_tensor = attrs.tensors.get("keys_tensor");

    if !attrs.string_list("keys_strings").is_empty() {
        return Err(OnnxError::Unsupported(
            "LabelEncoder: 'keys_strings' input is not representable in an f32 tensor".into(),
        ));
    }

    let key_count = if let Some(tensor) = keys_tensor {
        tensor.data.len()
    } else if !keys_int64s.is_empty() {
        keys_int64s.len()
    } else if !keys_floats.is_empty() {
        keys_floats.len()
    } else {
        return Err(OnnxError::InvalidModel(
            "LabelEncoder: no 'keys_*' attribute provided".into(),
        ));
    };

    if key_count != values.len() {
        return Err(OnnxError::InvalidModel(format!(
            "LabelEncoder: {key_count} keys but {} values",
            values.len()
        )));
    }

    // The first registration of a duplicate key wins, matching the insertion
    // order semantics of a hash map built front-to-back.
    let key_map = if let Some(tensor) = keys_tensor {
        let mut map = HashMap::with_capacity(key_count);
        for (&k, &v) in tensor.data.iter().zip(values.iter()) {
            map.entry(float_key(k)).or_insert(v);
        }
        KeyMap::Floats(map)
    } else if !keys_int64s.is_empty() {
        let mut map = HashMap::with_capacity(key_count);
        for (&k, &v) in keys_int64s.iter().zip(values.iter()) {
            map.entry(k).or_insert(v);
        }
        KeyMap::Ints(map)
    } else {
        let mut map = HashMap::with_capacity(key_count);
        for (&k, &v) in keys_floats.iter().zip(values.iter()) {
            map.entry(float_key(k)).or_insert(v);
        }
        KeyMap::Floats(map)
    };

    let data: Vec<f32> = x.data.iter().map(|&v| key_map.lookup(v, default)).collect();

    Ok(vec![Tensor::new(data, x.shape.clone())])
}
