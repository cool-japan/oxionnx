//! Token-by-token generation for autoregressive models.
//!
//! [`TokenStream`] is the decode loop of a GPT-style model expressed as a plain
//! [`Iterator`]: each `next()` runs the graph once, picks the next token, feeds
//! the model's `present.*` key/value outputs back in as the next step's
//! `past.*` inputs, and yields the token. A caller streams it straight to a
//! socket:
//!
//! ```no_run
//! use oxionnx::{GenerationConfig, KvCacheBinding, Session};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let session = Session::from_file("gpt2.onnx".as_ref())?;
//! let config = GenerationConfig::default()
//!     .with_max_new_tokens(64)
//!     .with_eos_token_id(50256)
//!     .with_kv_cache(vec![KvCacheBinding::empty(
//!         "past_key_values.0.key",
//!         "present.0.key",
//!         vec![1, 12, 0, 64],
//!     )]);
//!
//! for step in session.generate(&[15496, 11, 616, 1438, 318], config)? {
//!     let step = step?;
//!     print!("{} ", step.token);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Scope — what this is and is not
//!
//! It is the **plumbing**: the step loop, the cache feedback, the stop
//! conditions, and greedy (argmax) token selection. It is deliberately not a
//! sampler zoo — no temperature, no top-k/top-p, no beam search, no repetition
//! penalty. Those are decisions about *text*, they need an RNG whose seeding
//! becomes part of the public contract, and none of them can be verified
//! against a reference the way the plumbing can. [`GenerationConfig::emit_logits`]
//! hands the caller the raw logits row so a sampler can live outside this crate
//! without re-running the model.
//!
//! It also does no tokenization: tokens go in as ids and come out as ids.
//!
//! # Cancellation is per-generation here
//!
//! [`GenerationConfig::cancellation`] is checked **between decode steps**, and
//! it belongs to the stream rather than to the session — so two generations
//! running concurrently on one `Arc<Session>` can be cancelled independently.
//! (That is the opposite of
//! [`crate::SessionBuilder::with_session_cancellation`], which is necessarily
//! session-wide; see [`crate::session::cancellation`].) A session token, if one
//! is bound, is honoured too — it stops the run mid-graph.

use crate::tensor::Tensor;
use crate::{CancellationToken, OnnxError, Session};
use std::collections::HashMap;

/// One `past.* ← present.*` feedback edge of a model's key/value cache.
///
/// A GPT-2 export has two of these per layer (key and value), so a 12-layer
/// model needs 24 bindings.
#[derive(Debug, Clone)]
pub struct KvCacheBinding {
    /// Graph input the cache is fed to, e.g. `past_key_values.0.key`.
    pub past_input: String,
    /// Graph output the updated cache is read from, e.g. `present.0.key`.
    pub present_output: String,
    /// What to feed on the very first step, before any cache exists.
    ///
    /// For a standard export this is an all-zero tensor whose sequence axis is
    /// `0` — see [`KvCacheBinding::empty`].
    pub initial: Tensor,
}

impl KvCacheBinding {
    /// A binding whose first step feeds a zero-filled tensor of `initial_shape`.
    ///
    /// Pass the model's declared past shape with the sequence axis set to `0`
    /// (`[batch, heads, 0, head_dim]` for a GPT-2 export), which is exactly what
    /// an empty cache means.
    #[must_use]
    pub fn empty(
        past_input: impl Into<String>,
        present_output: impl Into<String>,
        initial_shape: Vec<usize>,
    ) -> Self {
        let numel: usize = initial_shape.iter().product();
        Self {
            past_input: past_input.into(),
            present_output: present_output.into(),
            initial: Tensor::new(vec![0.0; numel], initial_shape),
        }
    }

    /// A binding seeded with an explicit tensor — for models that cannot accept
    /// a zero-length cache, or for resuming a previously captured one.
    #[must_use]
    pub fn seeded(
        past_input: impl Into<String>,
        present_output: impl Into<String>,
        initial: Tensor,
    ) -> Self {
        Self {
            past_input: past_input.into(),
            present_output: present_output.into(),
            initial,
        }
    }
}

/// How a [`TokenStream`] drives the model.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Graph input that receives the token ids. Default `"input_ids"`.
    pub input_ids: String,
    /// Graph output holding the logits. Default `"logits"`.
    pub logits: String,
    /// Graph input for the attention mask, when the model declares one.
    ///
    /// Fed as a row of `1.0` covering prompt + generated tokens.
    pub attention_mask: Option<String>,
    /// Graph input for absolute positions, when the model declares one.
    pub position_ids: Option<String>,
    /// The model's key/value cache feedback edges. Empty means the model has no
    /// cache, in which case every step re-feeds the whole sequence.
    pub kv_cache: Vec<KvCacheBinding>,
    /// Upper bound on generated tokens. The stream always ends here even if the
    /// model never emits the EOS token.
    pub max_new_tokens: usize,
    /// Token that ends generation. It is *yielded* and then the stream ends.
    pub eos_token_id: Option<i64>,
    /// Include the logits row each token was chosen from in [`StreamStep`].
    ///
    /// Off by default: for a 50k vocabulary this is a 200 KB copy per token.
    pub emit_logits: bool,
    /// Per-generation cancellation, checked between steps.
    pub cancellation: Option<CancellationToken>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            input_ids: "input_ids".to_string(),
            logits: "logits".to_string(),
            attention_mask: None,
            position_ids: None,
            kv_cache: Vec::new(),
            max_new_tokens: 32,
            eos_token_id: None,
            emit_logits: false,
            cancellation: None,
        }
    }
}

impl GenerationConfig {
    /// Rename the token-id input.
    #[must_use]
    pub fn with_input_ids(mut self, name: impl Into<String>) -> Self {
        self.input_ids = name.into();
        self
    }

    /// Rename the logits output.
    #[must_use]
    pub fn with_logits(mut self, name: impl Into<String>) -> Self {
        self.logits = name.into();
        self
    }

    /// Feed an all-ones attention mask to `name`.
    #[must_use]
    pub fn with_attention_mask(mut self, name: impl Into<String>) -> Self {
        self.attention_mask = Some(name.into());
        self
    }

    /// Feed absolute positions to `name`.
    #[must_use]
    pub fn with_position_ids(mut self, name: impl Into<String>) -> Self {
        self.position_ids = Some(name.into());
        self
    }

    /// Set the key/value cache feedback edges.
    #[must_use]
    pub fn with_kv_cache(mut self, bindings: Vec<KvCacheBinding>) -> Self {
        self.kv_cache = bindings;
        self
    }

    /// Cap the number of generated tokens.
    #[must_use]
    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    /// Stop after this token is produced.
    #[must_use]
    pub fn with_eos_token_id(mut self, id: i64) -> Self {
        self.eos_token_id = Some(id);
        self
    }

    /// Attach the logits row to every [`StreamStep`].
    #[must_use]
    pub fn with_emit_logits(mut self, emit: bool) -> Self {
        self.emit_logits = emit;
        self
    }

    /// Make this generation cancellable between steps.
    #[must_use]
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }
}

/// One generated token and its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamStep {
    /// The chosen token id.
    pub token: i64,
    /// 0-based position among the tokens this stream generated.
    pub index: usize,
    /// The logits row `token` was the argmax of, when
    /// [`GenerationConfig::emit_logits`] is set.
    pub logits: Option<Vec<f32>>,
}

/// A lazy, cancellable decode loop over one prompt.
///
/// Created by [`Session::generate`]. Each `next()` is exactly one forward pass;
/// nothing runs until the iterator is advanced.
pub struct TokenStream<'s> {
    session: &'s Session,
    config: GenerationConfig,
    /// Prompt ids, fixed for the life of the stream.
    prompt: Vec<i64>,
    /// Tokens produced so far.
    generated: Vec<i64>,
    /// Live cache, keyed by the graph input each entry is fed to.
    past: HashMap<String, Tensor>,
    /// Set by EOS, by the token cap, or by a failed step — a stream never
    /// resumes after it has ended.
    finished: bool,
}

impl Session {
    /// Start a token-by-token generation over `prompt`.
    ///
    /// Nothing executes until the returned [`TokenStream`] is advanced.
    ///
    /// # Errors
    ///
    /// [`OnnxError::InvalidModel`] if the configuration names an input or output
    /// this model does not declare — checked once, up front, rather than
    /// failing on the first step.
    pub fn generate(
        &self,
        prompt: &[i64],
        config: GenerationConfig,
    ) -> Result<TokenStream<'_>, OnnxError> {
        if prompt.is_empty() {
            return Err(OnnxError::InvalidModel(
                "generate: the prompt must contain at least one token".to_string(),
            ));
        }
        self.require_input(&config.input_ids, "input_ids")?;
        self.require_output(&config.logits, "logits")?;
        if let Some(name) = &config.attention_mask {
            self.require_input(name, "attention_mask")?;
        }
        if let Some(name) = &config.position_ids {
            self.require_input(name, "position_ids")?;
        }
        let mut past = HashMap::with_capacity(config.kv_cache.len());
        for binding in &config.kv_cache {
            self.require_input(&binding.past_input, "kv cache past")?;
            self.require_output(&binding.present_output, "kv cache present")?;
            past.insert(binding.past_input.clone(), binding.initial.clone());
        }
        Ok(TokenStream {
            session: self,
            config,
            prompt: prompt.to_vec(),
            generated: Vec::new(),
            past,
            finished: false,
        })
    }

    /// Run a generation to completion and return just the generated token ids.
    ///
    /// # Errors
    ///
    /// The first error any step produces, including
    /// [`OnnxError::Cancelled`].
    pub fn generate_tokens(
        &self,
        prompt: &[i64],
        config: GenerationConfig,
    ) -> Result<Vec<i64>, OnnxError> {
        self.generate(prompt, config)?
            .map(|step| step.map(|step| step.token))
            .collect()
    }

    fn require_input(&self, name: &str, role: &str) -> Result<(), OnnxError> {
        if self.input_names().iter().any(|n| n == name) {
            return Ok(());
        }
        Err(OnnxError::InvalidModel(format!(
            "generate: no graph input named '{name}' (configured as {role}); this model declares {:?}",
            self.input_names()
        )))
    }

    fn require_output(&self, name: &str, role: &str) -> Result<(), OnnxError> {
        if self.output_names().iter().any(|n| n == name) {
            return Ok(());
        }
        Err(OnnxError::InvalidModel(format!(
            "generate: no graph output named '{name}' (configured as {role}); this model declares {:?}",
            self.output_names()
        )))
    }
}

impl TokenStream<'_> {
    /// Tokens generated so far.
    #[must_use]
    pub fn generated(&self) -> &[i64] {
        &self.generated
    }

    /// The prompt this stream was started from.
    #[must_use]
    pub fn prompt(&self) -> &[i64] {
        &self.prompt
    }

    /// Has the stream ended (EOS, token cap, cancellation or error)?
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Does this model carry its own key/value cache?
    fn uses_cache(&self) -> bool {
        !self.config.kv_cache.is_empty()
    }

    /// The ids fed to the model on this step.
    ///
    /// With a cache, the first step primes it with the whole prompt and every
    /// later step feeds exactly the one token the cache does not yet cover.
    /// Without a cache the model has no memory, so the whole sequence is
    /// re-fed each time — which is quadratic, and the reason the cache exists.
    fn step_ids(&self) -> Vec<i64> {
        if !self.uses_cache() || self.generated.is_empty() {
            let mut ids = self.prompt.clone();
            ids.extend_from_slice(&self.generated);
            return ids;
        }
        match self.generated.last() {
            Some(&last) => vec![last],
            // Unreachable: `generated` is non-empty on this branch.
            None => self.prompt.clone(),
        }
    }

    /// Run one forward pass and choose one token.
    fn step(&mut self) -> Result<StreamStep, OnnxError> {
        let ids = self.step_ids();
        let total_len = self.prompt.len() + self.generated.len();
        let seq_len = ids.len();

        let mut owned: HashMap<String, Tensor> = HashMap::new();
        owned.insert(
            self.config.input_ids.clone(),
            Tensor::new(ids.iter().map(|&id| id as f32).collect(), vec![1, seq_len]),
        );
        if let Some(name) = &self.config.attention_mask {
            owned.insert(
                name.clone(),
                Tensor::new(vec![1.0; total_len], vec![1, total_len]),
            );
        }
        if let Some(name) = &self.config.position_ids {
            let first = total_len - seq_len;
            let positions = (first..total_len).map(|p| p as f32).collect();
            owned.insert(name.clone(), Tensor::new(positions, vec![1, seq_len]));
        }

        let mut inputs: HashMap<&str, &Tensor> =
            owned.iter().map(|(k, v)| (k.as_str(), v)).collect();
        for (name, tensor) in &self.past {
            inputs.insert(name.as_str(), tensor);
        }

        let outputs = self.session.run_internal(&inputs)?;

        // Feed the updated cache forward before anything can fail on the logits,
        // so a malformed logits output does not leave a half-updated cache.
        let mut next_past = HashMap::with_capacity(self.config.kv_cache.len());
        for binding in &self.config.kv_cache {
            let present = outputs.get(&binding.present_output).ok_or_else(|| {
                OnnxError::InvalidModel(format!(
                    "generate: the model produced no output '{}' to refresh the cache input '{}'",
                    binding.present_output, binding.past_input
                ))
            })?;
            next_past.insert(binding.past_input.clone(), present.clone());
        }

        let logits = outputs.get(&self.config.logits).ok_or_else(|| {
            OnnxError::InvalidModel(format!(
                "generate: the model produced no output '{}'",
                self.config.logits
            ))
        })?;
        let row = last_position_logits(logits, &self.config.logits)?;
        let token = argmax(row);

        self.past = next_past;
        let index = self.generated.len();
        self.generated.push(token);
        if self.config.eos_token_id == Some(token) {
            self.finished = true;
        }
        Ok(StreamStep {
            token,
            index,
            logits: self.config.emit_logits.then(|| row.to_vec()),
        })
    }
}

impl Iterator for TokenStream<'_> {
    type Item = Result<StreamStep, OnnxError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.generated.len() >= self.config.max_new_tokens {
            return None;
        }
        // Between steps, not inside one: a step that has started always finishes
        // (or fails on its own terms), so the cache is never left half-updated.
        if let Some(token) = &self.config.cancellation {
            if token.is_cancelled() {
                self.finished = true;
                return Some(Err(OnnxError::Cancelled(format!(
                    "generation cancelled after {} token(s)",
                    self.generated.len()
                ))));
            }
        }
        match self.step() {
            Ok(step) => Some(Ok(step)),
            Err(e) => {
                self.finished = true;
                Some(Err(e))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            return (0, Some(0));
        }
        (
            0,
            Some(
                self.config
                    .max_new_tokens
                    .saturating_sub(self.generated.len()),
            ),
        )
    }
}

/// The vocabulary row belonging to the **last** sequence position.
///
/// Accepts the three shapes real exports produce: `[batch, seq, vocab]`,
/// `[seq, vocab]` and `[vocab]`. Anything else is a typed error rather than a
/// guess — silently reading the wrong row would produce plausible-looking
/// garbage.
fn last_position_logits<'t>(logits: &'t Tensor, name: &str) -> Result<&'t [f32], OnnxError> {
    let vocab = *logits.shape.last().ok_or_else(|| {
        OnnxError::ShapeMismatch(format!("generate: logits output '{name}' is a scalar"))
    })?;
    if vocab == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "generate: logits output '{name}' has an empty vocabulary axis"
        )));
    }
    if logits.data.len() < vocab || logits.data.len() % vocab != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "generate: logits output '{name}' has {} elements, which is not a whole number of \
             rows of {vocab}",
            logits.data.len()
        )));
    }
    match logits.shape.len() {
        1..=3 => Ok(&logits.data[logits.data.len() - vocab..]),
        rank => Err(OnnxError::ShapeMismatch(format!(
            "generate: logits output '{name}' has rank {rank}; expected [vocab], [seq, vocab] or \
             [batch, seq, vocab]"
        ))),
    }
}

/// Index of the largest element; ties go to the lowest index.
///
/// `NaN` never wins a comparison, so a row that is entirely `NaN` yields `0`
/// rather than an arbitrary index — deterministic, and the caller sees a
/// degenerate token instead of a panic.
fn argmax(row: &[f32]) -> i64 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &value) in row.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best = i;
        }
    }
    best as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_breaks_ties_towards_the_lowest_index() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-5.0, -1.0, -9.0]), 1);
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(argmax(&[f32::NAN, 0.5]), 1);
    }

    #[test]
    fn the_last_row_is_taken_from_every_accepted_logits_rank() {
        let rank3 = Tensor::new(vec![1.0, 2.0, 9.0, 3.0], vec![1, 2, 2]);
        assert_eq!(
            last_position_logits(&rank3, "logits").expect("rank 3"),
            &[9.0, 3.0]
        );
        let rank2 = Tensor::new(vec![1.0, 2.0, 9.0, 3.0], vec![2, 2]);
        assert_eq!(
            last_position_logits(&rank2, "logits").expect("rank 2"),
            &[9.0, 3.0]
        );
        let rank1 = Tensor::new(vec![4.0, 5.0], vec![2]);
        assert_eq!(
            last_position_logits(&rank1, "logits").expect("rank 1"),
            &[4.0, 5.0]
        );
    }

    #[test]
    fn a_degenerate_logits_shape_is_a_typed_error() {
        let scalar = Tensor::new(vec![1.0], vec![]);
        assert!(matches!(
            last_position_logits(&scalar, "logits"),
            Err(OnnxError::ShapeMismatch(_))
        ));
        let rank4 = Tensor::new(vec![1.0, 2.0], vec![1, 1, 1, 2]);
        assert!(matches!(
            last_position_logits(&rank4, "logits"),
            Err(OnnxError::ShapeMismatch(_))
        ));
    }
}
