//! TfIdfVectorizer ONNX-ML operator implementation.

use oxionnx_core::{OnnxError, OpContext, Tensor};

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
