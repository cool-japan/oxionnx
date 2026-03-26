//! ONNX-ML domain operator implementations.
//!
//! Covers LinearClassifier, LinearRegressor, Normalizer, Scaler, LabelEncoder,
//! TreeEnsembleClassifier, TreeEnsembleRegressor, SVMClassifier, and SVMRegressor.

use oxionnx_core::{OnnxError, OpContext, Tensor};

// ── Post-transform helpers ──────────────────────────────────────────────────

/// Apply softmax row-wise to a [N, C] buffer stored in row-major order.
fn softmax_rows(data: &mut [f32], n: usize, c: usize) {
    for row in 0..n {
        let offset = row * c;
        let row_slice = &mut data[offset..offset + c];
        let max_val = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row_slice.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row_slice.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Apply logistic (sigmoid) element-wise.
fn logistic_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Apply probit transform element-wise (approximate inverse of the standard normal CDF).
/// Uses the Abramowitz & Stegun rational approximation.
fn probit_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        // Clamp to (0, 1) to avoid infinities
        let p = v.clamp(1e-7, 1.0 - 1e-7);
        // Approximate probit via the rational approximation of the inverse normal CDF
        // Using the Beasley-Springer-Moro algorithm (simplified)
        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };
        // Rational approximation constants
        let c0 = 2.515_517_f32;
        let c1 = 0.802_853_f32;
        let c2 = 0.010_328_f32;
        let d1 = 1.432_788_f32;
        let d2 = 0.189_269_f32;
        let d3 = 0.001_308_f32;
        let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
        *v = if p < 0.5 { -result } else { result };
    }
}

/// Post-transform enumeration used by ML operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostTransform {
    None,
    Softmax,
    SoftmaxZero,
    Logistic,
    Probit,
}

impl PostTransform {
    /// Parse a post-transform string into the enum variant.
    pub fn parse(s: &str) -> Self {
        match s {
            "SOFTMAX" => Self::Softmax,
            "SOFTMAX_ZERO" => Self::SoftmaxZero,
            "LOGISTIC" => Self::Logistic,
            "PROBIT" => Self::Probit,
            _ => Self::None,
        }
    }
}

/// Apply a post-transform to a row-major [N, C] score buffer.
pub fn apply_post_transform(data: &mut [f32], n: usize, c: usize, transform: PostTransform) {
    match transform {
        PostTransform::Softmax => softmax_rows(data, n, c),
        PostTransform::SoftmaxZero => softmax_zero_rows(data, n, c),
        PostTransform::Logistic => logistic_inplace(data),
        PostTransform::Probit => probit_inplace(data),
        PostTransform::None => {}
    }
}

/// Apply softmax-zero: like softmax but zero entries remain zero.
fn softmax_zero_rows(data: &mut [f32], n: usize, c: usize) {
    for row in 0..n {
        let offset = row * c;
        let row_slice = &mut data[offset..offset + c];
        let max_val = row_slice
            .iter()
            .copied()
            .filter(|&v| v != 0.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row_slice.iter_mut() {
            if *v != 0.0 {
                *v = (*v - max_val).exp();
                sum += *v;
            }
        }
        if sum > 0.0 {
            for v in row_slice.iter_mut() {
                if *v != 0.0 {
                    *v /= sum;
                }
            }
        }
    }
}

// ── LinearClassifier ────────────────────────────────────────────────────────

/// ONNX-ML LinearClassifier operator.
///
/// Input 0: X \[N, features\]
/// Output 0: predicted labels (as f32)
/// Output 1: class scores \[N, num_classes\]
pub fn linear_classifier(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let coefficients = attrs
        .float_lists
        .get("coefficients")
        .ok_or_else(|| OnnxError::Parse("LinearClassifier: missing 'coefficients'".into()))?;

    let intercepts = attrs
        .float_lists
        .get("intercepts")
        .cloned()
        .unwrap_or_default();

    let class_labels_ints = attrs.ints("classlabels_ints");
    let multi_class = attrs.i("multi_class", 0); // 0 = one-vs-rest
    let post_transform = attrs.s("post_transform");

    // Determine dimensions
    let n = x.shape[0];
    let features = if x.shape.len() > 1 {
        x.shape[1]
    } else {
        x.numel() / n
    };

    // Number of classes: from class labels or inferred from coefficients
    let num_classes = if !class_labels_ints.is_empty() {
        class_labels_ints.len()
    } else {
        // coefficients length = num_classes * features (for multi_class)
        // or (num_classes - 1) * features for binary one-vs-rest
        let raw_targets = coefficients.len() / features;
        if raw_targets == 0 {
            return Err(OnnxError::ShapeMismatch(
                "LinearClassifier: coefficient count does not match features".into(),
            ));
        }
        raw_targets
    };

    // For binary one-vs-rest with single set of coefficients, we have 1 target
    let num_targets = coefficients.len() / features;
    let is_binary_ovr = multi_class == 0 && num_targets == 1 && num_classes == 2;

    // Compute raw scores: scores[i, j] = dot(X[i], W[j]) + bias[j]
    let score_cols = if is_binary_ovr { 1 } else { num_targets };
    let mut scores = vec![0.0f32; n * score_cols];

    for i in 0..n {
        for j in 0..score_cols {
            let mut val = 0.0f32;
            let w_offset = j * features;
            let x_offset = i * features;
            for f in 0..features {
                val += x.data[x_offset + f] * coefficients[w_offset + f];
            }
            if j < intercepts.len() {
                val += intercepts[j];
            }
            scores[i * score_cols + j] = val;
        }
    }

    // Expand binary one-vs-rest to 2-class scores
    let (final_scores, final_cols) = if is_binary_ovr {
        let mut expanded = vec![0.0f32; n * 2];
        for i in 0..n {
            let s = scores[i];
            expanded[i * 2] = -s; // class 0 score
            expanded[i * 2 + 1] = s; // class 1 score
        }
        (expanded, 2usize)
    } else {
        (scores, score_cols)
    };

    let mut result_scores = final_scores;

    // Apply post-transform
    match post_transform {
        "SOFTMAX" => softmax_rows(&mut result_scores, n, final_cols),
        "LOGISTIC" => logistic_inplace(&mut result_scores),
        "PROBIT" => probit_inplace(&mut result_scores),
        _ => {} // "NONE" or empty
    }

    // Compute predicted labels via argmax
    let mut labels = vec![0.0f32; n];
    for (i, label) in labels.iter_mut().enumerate() {
        let row_offset = i * final_cols;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..final_cols {
            if result_scores[row_offset + j] > best_val {
                best_val = result_scores[row_offset + j];
                best_idx = j;
            }
        }
        // Map to class label if available
        if !class_labels_ints.is_empty() && best_idx < class_labels_ints.len() {
            *label = class_labels_ints[best_idx] as f32;
        } else {
            *label = best_idx as f32;
        }
    }

    let label_tensor = Tensor::new(labels, vec![n]);
    let score_tensor = Tensor::new(result_scores, vec![n, final_cols]);

    Ok(vec![label_tensor, score_tensor])
}

// ── LinearRegressor ─────────────────────────────────────────────────────────

/// ONNX-ML LinearRegressor operator.
///
/// Input 0: X \[N, features\]
/// Output 0: Y \[N, targets\]
pub fn linear_regressor(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let coefficients = attrs
        .float_lists
        .get("coefficients")
        .ok_or_else(|| OnnxError::Parse("LinearRegressor: missing 'coefficients'".into()))?;

    let intercepts = attrs
        .float_lists
        .get("intercepts")
        .cloned()
        .unwrap_or_default();

    let post_transform = attrs.s("post_transform");

    let n = x.shape[0];
    let features = if x.shape.len() > 1 {
        x.shape[1]
    } else {
        x.numel() / n
    };

    // Number of targets
    let targets_attr = attrs.i("targets", 0);
    let num_targets = if targets_attr > 0 {
        targets_attr as usize
    } else {
        // Infer from coefficients
        let t = coefficients.len() / features;
        if t == 0 {
            1
        } else {
            t
        }
    };

    // Compute Y = X * W^T + bias
    let mut output = vec![0.0f32; n * num_targets];
    for i in 0..n {
        for j in 0..num_targets {
            let mut val = 0.0f32;
            let w_offset = j * features;
            let x_offset = i * features;
            for f in 0..features {
                if w_offset + f < coefficients.len() {
                    val += x.data[x_offset + f] * coefficients[w_offset + f];
                }
            }
            if j < intercepts.len() {
                val += intercepts[j];
            }
            output[i * num_targets + j] = val;
        }
    }

    // Apply post-transform
    match post_transform {
        "LOGISTIC" => logistic_inplace(&mut output),
        "SOFTMAX" => softmax_rows(&mut output, n, num_targets),
        "PROBIT" => probit_inplace(&mut output),
        _ => {} // "NONE", "LINEAR", or empty
    }

    Ok(vec![Tensor::new(output, vec![n, num_targets])])
}

// ── Normalizer ──────────────────────────────────────────────────────────────

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

// ── Scaler ──────────────────────────────────────────────────────────────────

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

// ── LabelEncoder ────────────────────────────────────────────────────────────

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

// ── TfIdfVectorizer ─────────────────────────────────────────────────────

/// ONNX-ML TfIdfVectorizer operator.
///
/// Converts a sequence of token IDs into TF, IDF, or TFIDF feature vectors.
///
/// Input 0: X `[N]` or `[N, 1]` — token ID sequence (as f32)
///
/// Attributes:
///   - mode: "TF" | "IDF" | "TFIDF"
///   - min_gram_length: i64
///   - max_gram_length: i64
///   - max_skip_count: i64
///   - ngram_counts: int_list — number of ngrams per gram length
///   - ngram_indexes: int_list — output index for each ngram
///   - pool_int64s: int_list — flattened ngram token IDs
///   - weights: float_list (optional) — IDF weights per output index
///
/// Output 0: Y `[output_size]` — feature vector
pub fn tfidf_vectorizer(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let mode = attrs.s("mode"); // "TF", "IDF", or "TFIDF"
    let min_gram_length = attrs.i("min_gram_length", 1) as usize;
    let max_gram_length = attrs.i("max_gram_length", 1) as usize;
    let max_skip_count = attrs.i("max_skip_count", 0) as usize;
    let ngram_counts = attrs.ints("ngram_counts");
    let ngram_indexes = attrs.ints("ngram_indexes");
    let pool_int64s = attrs.ints("pool_int64s");
    let weights = attrs
        .float_lists
        .get("weights")
        .cloned()
        .unwrap_or_default();

    // Flatten input to 1D token sequence
    let tokens: Vec<i64> = x.data.iter().map(|&v| v as i64).collect();
    let seq_len = tokens.len();

    // Determine output size from max of ngram_indexes + 1
    let output_size = ngram_indexes
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);

    // Build a lookup from ngram token tuple -> ngram_index in output
    // ngram_counts tells us how many ngrams exist for each gram length.
    // The pool_int64s is flattened: for gram_length g, each ngram uses g consecutive entries.
    // ngram_counts[k] = number of ngrams of length (k + min_gram_length)
    let mut ngram_map: Vec<(Vec<i64>, usize)> = Vec::new();
    let mut pool_offset = 0usize;
    let mut index_offset = 0usize;

    for (k, &count) in ngram_counts.iter().enumerate() {
        let gram_len = min_gram_length + k;
        let count = count as usize;
        for _ in 0..count {
            if pool_offset + gram_len <= pool_int64s.len() && index_offset < ngram_indexes.len() {
                let ngram: Vec<i64> = pool_int64s[pool_offset..pool_offset + gram_len].to_vec();
                let out_idx = ngram_indexes[index_offset] as usize;
                ngram_map.push((ngram, out_idx));
            }
            pool_offset += gram_len;
            index_offset += 1;
        }
    }

    // Count matched ngrams
    let mut counts = vec![0.0f32; output_size];

    // For each gram length in [min_gram_length, max_gram_length], slide over the input
    for gram_len in min_gram_length..=max_gram_length {
        if gram_len == 0 || gram_len > seq_len {
            continue;
        }

        // Generate all skip-gram combinations for this gram_len
        // A skip-gram of length g with max_skip_count s:
        // pick g tokens from the sequence where consecutive picks are separated
        // by at most s tokens (i.e., gap 0..=s between consecutive picked positions).
        // For skip_count=0, this is just contiguous ngrams.
        if max_skip_count == 0 {
            // Simple contiguous ngrams
            for start in 0..=seq_len.saturating_sub(gram_len) {
                let ngram: Vec<i64> = tokens[start..start + gram_len].to_vec();
                // Look up in ngram_map
                for (pattern, out_idx) in &ngram_map {
                    if pattern.len() == gram_len && *pattern == ngram && *out_idx < output_size {
                        counts[*out_idx] += 1.0;
                    }
                }
            }
        } else {
            // Skip-grams: enumerate all valid position tuples
            // Positions: p[0], p[1], ..., p[gram_len-1]
            // Constraints: p[0] >= 0, p[k] - p[k-1] in 1..=(max_skip_count+1)
            for start in 0..seq_len {
                generate_skipgrams(
                    &tokens,
                    start,
                    gram_len,
                    max_skip_count,
                    seq_len,
                    &ngram_map,
                    output_size,
                    &mut counts,
                );
            }
        }
    }

    // Apply mode
    let mut output = vec![0.0f32; output_size];
    match mode {
        "TF" => {
            output.copy_from_slice(&counts);
        }
        "IDF" => {
            for (i, &c) in counts.iter().enumerate() {
                if c > 0.0 {
                    output[i] = if i < weights.len() { weights[i] } else { 1.0 };
                }
            }
        }
        _ => {
            // Default to TFIDF: count * weight
            for (i, &c) in counts.iter().enumerate() {
                let w = if i < weights.len() { weights[i] } else { 1.0 };
                output[i] = c * w;
            }
        }
    }

    Ok(vec![Tensor::new(output, vec![output_size])])
}

/// Generate skip-grams starting at a given position and accumulate counts.
#[allow(clippy::too_many_arguments)]
fn generate_skipgrams(
    tokens: &[i64],
    start: usize,
    gram_len: usize,
    max_skip: usize,
    seq_len: usize,
    ngram_map: &[(Vec<i64>, usize)],
    output_size: usize,
    counts: &mut [f32],
) {
    // Use an iterative stack-based approach to avoid deep recursion
    let mut positions = vec![0usize; gram_len];
    positions[0] = start;

    let mut depth = 1usize;

    if gram_len == 1 {
        // Single token ngram
        let ngram = vec![tokens[start]];
        for (pattern, out_idx) in ngram_map {
            if pattern.len() == 1 && *pattern == ngram && *out_idx < output_size {
                counts[*out_idx] += 1.0;
            }
        }
        return;
    }

    // next_pos[depth] tracks the next position to try at each depth level
    let mut next_pos = vec![0usize; gram_len];
    next_pos[1] = start + 1;

    loop {
        if depth >= gram_len {
            // We have a complete tuple, check it
            let ngram: Vec<i64> = positions[..gram_len].iter().map(|&p| tokens[p]).collect();
            for (pattern, out_idx) in ngram_map {
                if pattern.len() == gram_len && *pattern == ngram && *out_idx < output_size {
                    counts[*out_idx] += 1.0;
                }
            }
            depth -= 1;
            if depth == 0 {
                break;
            }
            continue;
        }

        let prev = positions[depth - 1];
        let pos = next_pos[depth];
        let max_pos = (prev + max_skip + 1).min(seq_len.saturating_sub(1));

        if pos > max_pos {
            // Backtrack
            if depth <= 1 {
                break;
            }
            depth -= 1;
            continue;
        }

        positions[depth] = pos;
        next_pos[depth] = pos + 1;
        depth += 1;
        if depth < gram_len {
            next_pos[depth] = pos + 1;
        }
    }
}

/// ONNX-ML StringNormalizer operator (stub — returns Unsupported).
///
/// String tensors are not supported in oxionnx, so this always returns an error.
pub fn string_normalizer(_ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    Err(OnnxError::Unsupported(
        "StringNormalizer requires string tensor support which is not available".into(),
    ))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    /// Helper to build a minimal OpContext for testing.
    fn make_context<'a>(
        op: OpKind,
        inputs: Vec<Option<&'a Tensor>>,
        attrs: Attributes,
    ) -> (Node, Vec<Option<&'a Tensor>>) {
        let node = Node {
            op,
            name: "test_node".to_string(),
            inputs: vec![],
            outputs: vec![],
            attrs,
        };
        (node, inputs)
    }

    fn ctx_from<'a>(node: &'a Node, inputs: &'a [Option<&'a Tensor>]) -> OpContext<'a> {
        OpContext {
            node,
            inputs: inputs.to_vec(),
            outer_scope: None,
            registry: None,
        }
    }

    #[test]
    fn test_linear_classifier_2class() {
        // 2 classes, 2 features, 3 samples
        // W = [[1, 0], [0, 1]], bias = [0, 0]
        // For one-vs-rest binary: single set of coefficients
        // Use multi-class = 1 (multinomial) with 2 targets
        let x = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5], vec![3, 2]);

        let mut attrs = Attributes::default();
        attrs
            .float_lists
            .insert("coefficients".into(), vec![1.0, 0.0, 0.0, 1.0]);
        attrs
            .float_lists
            .insert("intercepts".into(), vec![0.0, 0.0]);
        attrs
            .int_lists
            .insert("classlabels_ints".into(), vec![0, 1]);
        attrs.ints.insert("multi_class".into(), 1);
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::LinearClassifier, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = linear_classifier(&ctx).expect("linear_classifier failed");

        assert_eq!(result.len(), 2);

        // Labels: argmax of scores
        let labels = &result[0];
        assert_eq!(labels.shape, vec![3]);
        // Sample 0: [1, 0] -> class 0
        assert!((labels.data[0] - 0.0).abs() < 1e-5);
        // Sample 1: [0, 1] -> class 1
        assert!((labels.data[1] - 1.0).abs() < 1e-5);
        // Sample 2: [0.5, 0.5] -> either (tie), argmax picks first => class 0
        // Actually both are 0.5, so first one wins
        assert!((labels.data[2] - 0.0).abs() < 1e-5);

        // Scores
        let scores = &result[1];
        assert_eq!(scores.shape, vec![3, 2]);
        assert!((scores.data[0] - 1.0).abs() < 1e-5); // sample 0, class 0
        assert!((scores.data[1] - 0.0).abs() < 1e-5); // sample 0, class 1
    }

    #[test]
    fn test_linear_classifier_softmax() {
        let x = Tensor::new(vec![2.0, 1.0], vec![1, 2]);

        let mut attrs = Attributes::default();
        attrs
            .float_lists
            .insert("coefficients".into(), vec![1.0, 0.0, 0.0, 1.0]);
        attrs
            .float_lists
            .insert("intercepts".into(), vec![0.0, 0.0]);
        attrs
            .int_lists
            .insert("classlabels_ints".into(), vec![0, 1]);
        attrs.ints.insert("multi_class".into(), 1);
        attrs
            .strings
            .insert("post_transform".into(), "SOFTMAX".into());

        let (node, inputs) = make_context(OpKind::LinearClassifier, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = linear_classifier(&ctx).expect("softmax classifier failed");

        let scores = &result[1];
        // After softmax, scores should sum to 1.0
        let sum: f32 = scores.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Class 0 score > class 1 score (raw: 2.0 vs 1.0)
        assert!(scores.data[0] > scores.data[1]);
    }

    #[test]
    fn test_linear_classifier_binary_ovr() {
        // Binary one-vs-rest: single set of coefficients
        let x = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![2, 2]);

        let mut attrs = Attributes::default();
        // Single target coefficients
        attrs
            .float_lists
            .insert("coefficients".into(), vec![1.0, 0.0]);
        attrs.float_lists.insert("intercepts".into(), vec![0.0]);
        attrs
            .int_lists
            .insert("classlabels_ints".into(), vec![0, 1]);
        attrs.ints.insert("multi_class".into(), 0); // one-vs-rest
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::LinearClassifier, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = linear_classifier(&ctx).expect("binary ovr failed");

        let labels = &result[0];
        // Sample 0: dot([1,0], [1,0]) = 1 > 0, so class 1
        assert!((labels.data[0] - 1.0).abs() < 1e-5);
        // Sample 1: dot([-1,0], [1,0]) = -1 < 0, so class 0
        assert!((labels.data[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_regressor() {
        // 2 samples, 3 features, 1 target
        // W = [1, 2, 3], bias = [1]
        let x = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 0.0, 0.0], vec![2, 3]);

        let mut attrs = Attributes::default();
        attrs
            .float_lists
            .insert("coefficients".into(), vec![1.0, 2.0, 3.0]);
        attrs.float_lists.insert("intercepts".into(), vec![1.0]);
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::LinearRegressor, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = linear_regressor(&ctx).expect("linear_regressor failed");

        assert_eq!(result.len(), 1);
        let y = &result[0];
        assert_eq!(y.shape, vec![2, 1]);
        // Sample 0: 1*1 + 2*1 + 3*1 + 1 = 7
        assert!((y.data[0] - 7.0).abs() < 1e-5);
        // Sample 1: 1*2 + 2*0 + 3*0 + 1 = 3
        assert!((y.data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_regressor_multi_target() {
        // 1 sample, 2 features, 2 targets
        // W = [[1, 0], [0, 1]], bias = [1, 2]
        let x = Tensor::new(vec![3.0, 4.0], vec![1, 2]);

        let mut attrs = Attributes::default();
        attrs
            .float_lists
            .insert("coefficients".into(), vec![1.0, 0.0, 0.0, 1.0]);
        attrs
            .float_lists
            .insert("intercepts".into(), vec![1.0, 2.0]);
        attrs.ints.insert("targets".into(), 2);
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::LinearRegressor, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = linear_regressor(&ctx).expect("multi-target regressor failed");

        let y = &result[0];
        assert_eq!(y.shape, vec![1, 2]);
        // Target 0: 1*3 + 0*4 + 1 = 4
        assert!((y.data[0] - 4.0).abs() < 1e-5);
        // Target 1: 0*3 + 1*4 + 2 = 6
        assert!((y.data[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalizer_max() {
        let x = Tensor::new(vec![3.0, -4.0, 1.0, 2.0], vec![2, 2]);

        let mut attrs = Attributes::default();
        attrs.strings.insert("norm".into(), "MAX".into());

        let (node, inputs) = make_context(OpKind::Normalizer, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = normalizer(&ctx).expect("normalizer MAX failed");

        let y = &result[0];
        // Row 0: max_abs = 4, [3/4, -4/4] = [0.75, -1.0]
        assert!((y.data[0] - 0.75).abs() < 1e-5);
        assert!((y.data[1] - (-1.0)).abs() < 1e-5);
        // Row 1: max_abs = 2, [1/2, 2/2] = [0.5, 1.0]
        assert!((y.data[2] - 0.5).abs() < 1e-5);
        assert!((y.data[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalizer_l1() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let mut attrs = Attributes::default();
        attrs.strings.insert("norm".into(), "L1".into());

        let (node, inputs) = make_context(OpKind::Normalizer, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = normalizer(&ctx).expect("normalizer L1 failed");

        let y = &result[0];
        // Row 0: sum_abs = 3, [1/3, 2/3]
        assert!((y.data[0] - 1.0 / 3.0).abs() < 1e-5);
        assert!((y.data[1] - 2.0 / 3.0).abs() < 1e-5);
        // Row 1: sum_abs = 7, [3/7, 4/7]
        assert!((y.data[2] - 3.0 / 7.0).abs() < 1e-5);
        assert!((y.data[3] - 4.0 / 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalizer_l2() {
        let x = Tensor::new(vec![3.0, 4.0], vec![1, 2]);

        let mut attrs = Attributes::default();
        attrs.strings.insert("norm".into(), "L2".into());

        let (node, inputs) = make_context(OpKind::Normalizer, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = normalizer(&ctx).expect("normalizer L2 failed");

        let y = &result[0];
        // norm = 5, [3/5, 4/5] = [0.6, 0.8]
        assert!((y.data[0] - 0.6).abs() < 1e-5);
        assert!((y.data[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_scaler() {
        // 2 samples, 3 features
        let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);

        let mut attrs = Attributes::default();
        attrs
            .float_lists
            .insert("offset".into(), vec![10.0, 20.0, 30.0]);
        attrs
            .float_lists
            .insert("scale".into(), vec![0.1, 0.2, 0.3]);

        let (node, inputs) = make_context(OpKind::Scaler, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = scaler(&ctx).expect("scaler failed");

        let y = &result[0];
        assert_eq!(y.shape, vec![2, 3]);
        // Sample 0: (10-10)*0.1=0, (20-20)*0.2=0, (30-30)*0.3=0
        assert!((y.data[0] - 0.0).abs() < 1e-5);
        assert!((y.data[1] - 0.0).abs() < 1e-5);
        assert!((y.data[2] - 0.0).abs() < 1e-5);
        // Sample 1: (40-10)*0.1=3, (50-20)*0.2=6, (60-30)*0.3=9
        assert!((y.data[3] - 3.0).abs() < 1e-5);
        assert!((y.data[4] - 6.0).abs() < 1e-5);
        assert!((y.data[5] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_label_encoder_int_to_int() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 99.0], vec![4]);

        let mut attrs = Attributes::default();
        attrs.int_lists.insert("keys_int64s".into(), vec![1, 2, 3]);
        attrs
            .int_lists
            .insert("values_int64s".into(), vec![10, 20, 30]);
        attrs.ints.insert("default_int64".into(), -1);

        let (node, inputs) = make_context(OpKind::LabelEncoder, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = label_encoder(&ctx).expect("label_encoder failed");

        let y = &result[0];
        assert_eq!(y.shape, vec![4]);
        assert!((y.data[0] - 10.0).abs() < 1e-5);
        assert!((y.data[1] - 20.0).abs() < 1e-5);
        assert!((y.data[2] - 30.0).abs() < 1e-5);
        assert!((y.data[3] - (-1.0)).abs() < 1e-5); // default
    }

    #[test]
    fn test_label_encoder_float_to_float() {
        let x = Tensor::new(vec![1.5, 2.5, 9.9], vec![3]);

        let mut attrs = Attributes::default();
        attrs
            .float_lists
            .insert("keys_floats".into(), vec![1.5, 2.5]);
        attrs
            .float_lists
            .insert("values_floats".into(), vec![100.0, 200.0]);
        attrs.floats.insert("default_float".into(), -999.0);

        let (node, inputs) = make_context(OpKind::LabelEncoder, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = label_encoder(&ctx).expect("label_encoder float failed");

        let y = &result[0];
        assert!((y.data[0] - 100.0).abs() < 1e-5);
        assert!((y.data[1] - 200.0).abs() < 1e-5);
        assert!((y.data[2] - (-999.0)).abs() < 1e-5); // default
    }

    // -----------------------------------------------------------------------
    // TfIdfVectorizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tfidf_vectorizer_tf() {
        // Simple unigram TF mode
        // Tokens: [1, 2, 1, 3, 2, 1]
        // Vocabulary (unigrams): 1 -> index 0, 2 -> index 1, 3 -> index 2
        // Expected counts: token 1 appears 3 times, token 2 appears 2 times, token 3 appears 1 time
        let x = Tensor::new(vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0], vec![6]);

        let mut attrs = Attributes::default();
        attrs.strings.insert("mode".into(), "TF".into());
        attrs.ints.insert("min_gram_length".into(), 1);
        attrs.ints.insert("max_gram_length".into(), 1);
        attrs.ints.insert("max_skip_count".into(), 0);
        // 3 unigrams
        attrs.int_lists.insert("ngram_counts".into(), vec![3]);
        // output index for each ngram
        attrs
            .int_lists
            .insert("ngram_indexes".into(), vec![0, 1, 2]);
        // the ngram pool: token 1, token 2, token 3
        attrs.int_lists.insert("pool_int64s".into(), vec![1, 2, 3]);

        let (node, inputs) = make_context(OpKind::TfIdfVectorizer, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = tfidf_vectorizer(&ctx).expect("tfidf_vectorizer TF failed");

        let y = &result[0];
        assert_eq!(y.shape, vec![3]);
        assert!((y.data[0] - 3.0).abs() < 1e-5); // token 1 count
        assert!((y.data[1] - 2.0).abs() < 1e-5); // token 2 count
        assert!((y.data[2] - 1.0).abs() < 1e-5); // token 3 count
    }

    #[test]
    fn test_tfidf_vectorizer_idf() {
        // IDF mode: presence * weight
        // Tokens: [1, 2, 1]
        // Vocabulary: 1 -> idx 0, 2 -> idx 1, 3 -> idx 2
        // Weights: [0.5, 1.5, 2.0]
        // Token 1 present -> output[0] = 0.5
        // Token 2 present -> output[1] = 1.5
        // Token 3 absent  -> output[2] = 0.0
        let x = Tensor::new(vec![1.0, 2.0, 1.0], vec![3]);

        let mut attrs = Attributes::default();
        attrs.strings.insert("mode".into(), "IDF".into());
        attrs.ints.insert("min_gram_length".into(), 1);
        attrs.ints.insert("max_gram_length".into(), 1);
        attrs.ints.insert("max_skip_count".into(), 0);
        attrs.int_lists.insert("ngram_counts".into(), vec![3]);
        attrs
            .int_lists
            .insert("ngram_indexes".into(), vec![0, 1, 2]);
        attrs.int_lists.insert("pool_int64s".into(), vec![1, 2, 3]);
        attrs
            .float_lists
            .insert("weights".into(), vec![0.5, 1.5, 2.0]);

        let (node, inputs) = make_context(OpKind::TfIdfVectorizer, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = tfidf_vectorizer(&ctx).expect("tfidf_vectorizer IDF failed");

        let y = &result[0];
        assert_eq!(y.shape, vec![3]);
        assert!((y.data[0] - 0.5).abs() < 1e-5); // token 1 present
        assert!((y.data[1] - 1.5).abs() < 1e-5); // token 2 present
        assert!((y.data[2] - 0.0).abs() < 1e-5); // token 3 absent
    }

    #[test]
    fn test_tfidf_vectorizer_bigram() {
        // Bigram matching in TF mode
        // Tokens: [1, 2, 3, 1, 2]
        // Bigrams: [1,2] -> idx 0, [2,3] -> idx 1, [3,1] -> idx 2
        // Occurrences: [1,2] appears at pos 0 and pos 3 -> count 2
        //              [2,3] appears at pos 1 -> count 1
        //              [3,1] appears at pos 2 -> count 1
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0], vec![5]);

        let mut attrs = Attributes::default();
        attrs.strings.insert("mode".into(), "TF".into());
        attrs.ints.insert("min_gram_length".into(), 2);
        attrs.ints.insert("max_gram_length".into(), 2);
        attrs.ints.insert("max_skip_count".into(), 0);
        // 3 bigrams
        attrs.int_lists.insert("ngram_counts".into(), vec![3]);
        attrs
            .int_lists
            .insert("ngram_indexes".into(), vec![0, 1, 2]);
        // flattened bigrams: [1,2], [2,3], [3,1]
        attrs
            .int_lists
            .insert("pool_int64s".into(), vec![1, 2, 2, 3, 3, 1]);

        let (node, inputs) = make_context(OpKind::TfIdfVectorizer, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = tfidf_vectorizer(&ctx).expect("tfidf_vectorizer bigram failed");

        let y = &result[0];
        assert_eq!(y.shape, vec![3]);
        assert!((y.data[0] - 2.0).abs() < 1e-5); // [1,2] count
        assert!((y.data[1] - 1.0).abs() < 1e-5); // [2,3] count
        assert!((y.data[2] - 1.0).abs() < 1e-5); // [3,1] count
    }
}
