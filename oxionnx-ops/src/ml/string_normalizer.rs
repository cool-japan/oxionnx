//! StringNormalizer ONNX-ML operator implementation.

use oxionnx_core::{OnnxError, OpContext, Tensor};

/// ONNX-ML StringNormalizer operator.
///
/// Since oxionnx uses f32 tensors, strings are encoded as sequences of byte values:
/// each f32 element represents one byte of UTF-8 data, with 0.0 acting as a string
/// delimiter (null terminator).
///
/// Attributes:
///   - `case_change_action`: `"LOWER"`, `"UPPER"`, or `"NONE"` (default `"NONE"`)
///   - `is_case_sensitive`: int (default 1) — controls stopword matching case sensitivity
///   - `locale`: string (optional, currently unused; Pure Rust, no ICU)
///   - `stopwords`: string list — words to filter out
///
/// Input 0: X \[N\] — f32 tensor encoding null-terminated UTF-8 strings
/// Output 0: Y — f32 tensor with normalized strings (same encoding)
pub fn string_normalizer(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let case_action = attrs.s("case_change_action");
    let is_case_sensitive = attrs.i("is_case_sensitive", 1) != 0;
    let stopwords = attrs.string_list("stopwords");

    // ── Step 1: Decode f32 byte stream into strings ─────────────────────
    // Each f32 is treated as one byte value. Strings are delimited by 0.0.
    let mut strings: Vec<String> = Vec::new();
    let mut current_bytes: Vec<u8> = Vec::new();

    for &val in &x.data {
        let byte_val = val as u8;
        if byte_val == 0 {
            // Null terminator — flush the current string
            let s = String::from_utf8(current_bytes.clone()).unwrap_or_default();
            strings.push(s);
            current_bytes.clear();
        } else {
            current_bytes.push(byte_val);
        }
    }
    // If the stream doesn't end with a null terminator, flush remaining bytes
    if !current_bytes.is_empty() {
        let s = String::from_utf8(current_bytes).unwrap_or_default();
        strings.push(s);
    }

    // ── Step 2: Filter stopwords ────────────────────────────────────────
    if !stopwords.is_empty() {
        strings.retain(|s| {
            if is_case_sensitive {
                !stopwords.iter().any(|sw| sw == s)
            } else {
                let s_lower = s.to_lowercase();
                !stopwords.iter().any(|sw| sw.to_lowercase() == s_lower)
            }
        });
    }

    // ── Step 3: Apply case transformation ───────────────────────────────
    for s in &mut strings {
        match case_action {
            "LOWER" => *s = s.to_lowercase(),
            "UPPER" => *s = s.to_uppercase(),
            _ => {} // "NONE" or empty — no change
        }
    }

    // ── Step 4: Re-encode strings back to f32 tensor ────────────────────
    let mut output: Vec<f32> = Vec::new();
    for (i, s) in strings.iter().enumerate() {
        for &b in s.as_bytes() {
            output.push(b as f32);
        }
        // Add null terminator between strings (but not after the last one)
        if i + 1 < strings.len() {
            output.push(0.0);
        }
    }

    let len = output.len();
    Ok(vec![Tensor::new(output, vec![len])])
}
