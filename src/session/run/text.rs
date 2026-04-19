use crate::tensor::Tensor;
use crate::OnnxError;

use super::super::Session;

impl Session {
    /// Text correction helper: tokenize -> run -> detokenize.
    /// Character-level tokenization (Unicode codepoint IDs).
    pub fn correct_text(&self, text: &str) -> Result<String, OnnxError> {
        let ids: Vec<f32> = text.chars().map(|c| c as u32 as f32).collect();
        let n = ids.len();
        let input = Tensor::new(ids, vec![1, n]);

        let outputs = self.run_one("input_ids", input)?;

        if let Some(out) = outputs.values().next() {
            let chars: String = out
                .data
                .iter()
                .filter_map(|&v| char::from_u32(v as u32).filter(|&c| c != '\0'))
                .collect();
            Ok(chars)
        } else {
            Ok(text.to_string())
        }
    }
}
