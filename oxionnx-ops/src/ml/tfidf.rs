//! TfIdfVectorizer ONNX-ML operator implementation.

use std::collections::HashMap;

use oxionnx_core::{OnnxError, OpContext, Tensor};

/// Weighting criterion selected by the `mode` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Raw occurrence counts.
    Tf,
    /// Presence (0/1) scaled by `weights`.
    Idf,
    /// Occurrence counts scaled by `weights`.
    TfIdf,
}

impl Mode {
    fn parse(s: &str) -> Result<Self, OnnxError> {
        match s {
            "TF" => Ok(Self::Tf),
            "IDF" => Ok(Self::Idf),
            "TFIDF" => Ok(Self::TfIdf),
            other => Err(OnnxError::InvalidModel(format!(
                "TfIdfVectorizer: mode '{other}' is unrecognized, expected TF, IDF or TFIDF"
            ))),
        }
    }
}

/// A registered n-gram: `Some(output_index)` for a complete pool entry,
/// `None` for a prefix that only leads to longer entries.
type NgramMap = HashMap<Vec<i64>, Option<usize>>;

/// ONNX-ML TfIdfVectorizer operator.
///
/// Extracts n-gram / skip-gram counts from a token-id sequence.
///
/// Input 0: X `[C]` — one sequence — or `[N, C]` — a batch of N sequences.
///
/// Attributes:
///   - `mode`: "TF" | "IDF" | "TFIDF"
///   - `min_gram_length` / `max_gram_length`: inclusive gram-size window
///   - `max_skip_count`: maximum number of tokens skipped between gram items
///   - `ngram_counts`: **start index into `pool_int64s`** of the (k+1)-grams
///   - `ngram_indexes`: output column of the i-th n-gram of the pool
///   - `pool_int64s`: flattened n-gram token ids
///   - `weights` (optional): per-output-column IDF weight
///
/// Output 0: Y `[output_size]` for a 1-D input, `[N, output_size]` for a
/// batched input.
pub fn tfidf_vectorizer(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let mode = Mode::parse(attrs.s("mode"))?;
    let min_gram_length = attrs.i("min_gram_length", 1);
    let max_gram_length = attrs.i("max_gram_length", 1);
    let max_skip_count = attrs.i("max_skip_count", 0);

    if min_gram_length <= 0 {
        return Err(OnnxError::InvalidModel(format!(
            "TfIdfVectorizer: min_gram_length must be positive, got {min_gram_length}"
        )));
    }
    if max_gram_length < min_gram_length {
        return Err(OnnxError::InvalidModel(format!(
            "TfIdfVectorizer: max_gram_length ({max_gram_length}) < min_gram_length ({min_gram_length})"
        )));
    }
    if max_skip_count < 0 {
        return Err(OnnxError::InvalidModel(format!(
            "TfIdfVectorizer: max_skip_count must be non-negative, got {max_skip_count}"
        )));
    }
    let min_gram_length = min_gram_length as usize;
    let max_gram_length = max_gram_length as usize;
    // Saturating: a corrupt `max_skip_count` near i64::MAX must not wrap.
    let max_skip_distance = (max_skip_count as usize).saturating_add(1);

    let ngram_counts = attrs.ints("ngram_counts");
    let ngram_indexes = attrs.ints("ngram_indexes");
    let pool_int64s = attrs.ints("pool_int64s");
    let weights = attrs
        .float_lists
        .get("weights")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    if !attrs.string_list("pool_strings").is_empty() {
        return Err(OnnxError::Unsupported(
            "TfIdfVectorizer: 'pool_strings' requires string tensors".into(),
        ));
    }

    // Output width is the largest ngram index + 1.
    let mut output_size = 0usize;
    for &idx in ngram_indexes {
        if idx < 0 {
            return Err(OnnxError::InvalidModel(
                "TfIdfVectorizer: negative ngram_indexes entry".into(),
            ));
        }
        output_size = output_size.max(idx as usize + 1);
    }

    let (rows, row_len, out_shape) = input_layout(x, output_size)?;

    let ngram_map = build_ngram_map(
        ngram_counts,
        ngram_indexes,
        pool_int64s,
        min_gram_length,
        max_gram_length,
    )?;

    let mut output = vec![0.0f32; rows * output_size];
    if output_size == 0 || ngram_map.is_empty() || row_len == 0 {
        return Ok(vec![Tensor::new(output, out_shape)]);
    }

    // Scratch buffers reused across rows so the scan allocates nothing.
    let mut tokens: Vec<i64> = Vec::with_capacity(row_len);
    let mut gram: Vec<i64> = Vec::with_capacity(max_gram_length);

    for row in 0..rows {
        let start = row * row_len;
        tokens.clear();
        tokens.extend(x.data[start..start + row_len].iter().map(|&v| v as i64));

        let counts = &mut output[row * output_size..(row + 1) * output_size];
        scan_row(
            &tokens,
            &ngram_map,
            min_gram_length,
            max_gram_length,
            max_skip_distance,
            &mut gram,
            counts,
            mode,
            weights,
        );
    }

    Ok(vec![Tensor::new(output, out_shape)])
}

/// Resolve `(rows, row_len, output_shape)` from the input shape.
///
/// A 1-D `[C]` (or scalar) input is a single sequence and keeps a 1-D
/// `[output_size]` output; `[N, C]` yields `[N, output_size]`.
fn input_layout(x: &Tensor, output_size: usize) -> Result<(usize, usize, Vec<usize>), OnnxError> {
    let (rows, row_len, shape) = match x.shape.len() {
        0 => (1usize, x.data.len().min(1), vec![output_size]),
        1 => (1usize, x.shape[0], vec![output_size]),
        2 => (x.shape[0], x.shape[1], vec![x.shape[0], output_size]),
        rank => {
            return Err(OnnxError::ShapeMismatch(format!(
                "TfIdfVectorizer: input must be [C] or [N, C], got rank {rank}"
            )))
        }
    };

    let needed = rows.checked_mul(row_len).ok_or_else(|| {
        OnnxError::ShapeMismatch("TfIdfVectorizer: input size overflows usize".into())
    })?;
    if x.data.len() < needed {
        return Err(OnnxError::ShapeMismatch(format!(
            "TfIdfVectorizer: input holds {} elements but shape {:?} requires {needed}",
            x.data.len(),
            x.shape
        )));
    }
    if rows.checked_mul(output_size).is_none() {
        return Err(OnnxError::ShapeMismatch(
            "TfIdfVectorizer: output size overflows usize".into(),
        ));
    }

    Ok((rows, row_len, shape))
}

/// Build the n-gram lookup table from the pool.
///
/// `ngram_counts[k]` is the index into `pool_int64s` at which the
/// (k+1)-grams start; the bucket ends where the next one starts (or at the end
/// of the pool). The running n-gram ordinal indexes `ngram_indexes`, and it
/// advances across buckets whose gram size falls outside
/// `[min_gram_length, max_gram_length]` even though those are not registered.
fn build_ngram_map(
    ngram_counts: &[i64],
    ngram_indexes: &[i64],
    pool_int64s: &[i64],
    min_gram_length: usize,
    max_gram_length: usize,
) -> Result<NgramMap, OnnxError> {
    let mut map: NgramMap = HashMap::new();
    let total_items = pool_int64s.len();
    let mut ngram_id = 0usize;

    for (bucket, &start) in ngram_counts.iter().enumerate() {
        let gram_size = bucket + 1;
        let end = ngram_counts
            .get(bucket + 1)
            .copied()
            .unwrap_or(total_items as i64);
        if start < 0 || end < start || end > total_items as i64 {
            return Err(OnnxError::InvalidModel(format!(
                "TfIdfVectorizer: ngram_counts bucket {bucket} spans [{start}, {end}) which is out of bounds for a pool of {total_items} items"
            )));
        }
        let (start, end) = (start as usize, end as usize);
        let items = end - start;
        if items == 0 {
            continue;
        }
        if items % gram_size != 0 {
            return Err(OnnxError::InvalidModel(format!(
                "TfIdfVectorizer: {items} pool items do not compose whole {gram_size}-grams"
            )));
        }
        let ngrams = items / gram_size;

        if gram_size >= min_gram_length && gram_size <= max_gram_length {
            for g in 0..ngrams {
                let base = start + g * gram_size;
                let out_idx = *ngram_indexes.get(ngram_id + g).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "TfIdfVectorizer: ngram_indexes has {} entries but the pool declares at least {}",
                        ngram_indexes.len(),
                        ngram_id + g + 1
                    ))
                })?;
                // Register every prefix so the scan can stop early, then the
                // complete n-gram with its output column.
                for len in 1..gram_size {
                    map.entry(pool_int64s[base..base + len].to_vec())
                        .or_insert(None);
                }
                let entry = map
                    .entry(pool_int64s[base..base + gram_size].to_vec())
                    .or_insert(None);
                if entry.is_none() {
                    *entry = Some(out_idx as usize);
                }
            }
        }
        ngram_id += ngrams;
    }

    Ok(map)
}

/// Accumulate the n-gram hits of one token row into `counts`.
///
/// Mirrors onnxruntime: each generated gram uses a single fixed skip distance
/// (1..=max_skip_count+1) between consecutive items, and unigrams are counted
/// only on the first pass since they cannot be affected by skipping.
#[allow(clippy::too_many_arguments)]
fn scan_row(
    tokens: &[i64],
    ngram_map: &NgramMap,
    min_gram_length: usize,
    max_gram_length: usize,
    max_skip_distance: usize,
    gram: &mut Vec<i64>,
    counts: &mut [f32],
    mode: Mode,
    weights: &[f32],
) {
    let seq_len = tokens.len();
    let mut start_gram_size = min_gram_length;
    // A skip distance of `seq_len` already exhausts the row, so scanning
    // further is pure waste (and keeps the stride multiplication in range).
    let max_skip_distance = max_skip_distance.min(seq_len.max(1));

    for skip_distance in 1..=max_skip_distance {
        for start in 0..seq_len {
            // Not enough room left for even the shortest gram.
            if start.saturating_add(skip_distance.saturating_mul(start_gram_size - 1)) >= seq_len {
                break;
            }

            gram.clear();
            let mut pos = start;
            let mut gram_size = 1usize;
            while gram_size <= max_gram_length && pos < seq_len {
                gram.push(tokens[pos]);
                match ngram_map.get(gram.as_slice()) {
                    None => break,
                    Some(hit) => {
                        if gram_size >= start_gram_size {
                            if let Some(out_idx) = *hit {
                                if let Some(slot) = counts.get_mut(out_idx) {
                                    accumulate(slot, out_idx, mode, weights);
                                }
                            }
                        }
                    }
                }
                gram_size += 1;
                pos += skip_distance;
            }
        }

        // Unigrams are independent of the skip distance: count them once.
        if start_gram_size == 1 {
            start_gram_size = 2;
            if start_gram_size > max_gram_length {
                break;
            }
        }
    }
}

/// Apply the weighting criterion for one n-gram hit.
#[inline]
fn accumulate(slot: &mut f32, out_idx: usize, mode: Mode, weights: &[f32]) {
    match mode {
        Mode::Tf => *slot += 1.0,
        Mode::Idf => *slot = weights.get(out_idx).copied().unwrap_or(1.0),
        Mode::TfIdf => *slot += weights.get(out_idx).copied().unwrap_or(1.0),
    }
}
