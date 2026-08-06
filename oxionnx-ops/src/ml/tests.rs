//! Tests for ONNX-ML operator implementations.

use super::{
    label_encoder, linear_classifier, linear_regressor, normalizer, scaler, string_normalizer,
    tfidf_vectorizer,
};
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    OpContext, Tensor,
};

/// Helper to build a minimal OpContext for testing.
fn make_context(
    op: OpKind,
    inputs: Vec<Option<&Tensor>>,
    attrs: Attributes,
) -> (Node, Vec<Option<&Tensor>>) {
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
        weights: None,
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
    // Unigrams start at pool index 0 (ngram_counts holds pool offsets).
    attrs.int_lists.insert("ngram_counts".into(), vec![0]);
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
    // Unigrams start at pool index 0 (ngram_counts holds pool offsets).
    attrs.int_lists.insert("ngram_counts".into(), vec![0]);
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
    // No unigrams (bucket 0 spans [0, 0)); the bigrams start at pool index 0.
    attrs.int_lists.insert("ngram_counts".into(), vec![0, 0]);
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

// -----------------------------------------------------------------------
// StringNormalizer tests
// -----------------------------------------------------------------------

/// Helper: encode a slice of strings into a null-separated f32 tensor.
fn encode_strings(strings: &[&str]) -> Tensor {
    let mut data: Vec<f32> = Vec::new();
    for (i, s) in strings.iter().enumerate() {
        for &b in s.as_bytes() {
            data.push(b as f32);
        }
        if i + 1 < strings.len() {
            data.push(0.0); // null separator
        }
    }
    let len = data.len();
    Tensor::new(data, vec![len])
}

/// Helper: decode a f32 tensor back into strings (split on 0.0).
fn decode_strings(tensor: &Tensor) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    for &v in &tensor.data {
        let b = v as u8;
        if b == 0 {
            result.push(String::from_utf8(current.clone()).unwrap_or_default());
            current.clear();
        } else {
            current.push(b);
        }
    }
    if !current.is_empty() {
        result.push(String::from_utf8(current).unwrap_or_default());
    }
    result
}

#[test]
fn test_string_normalizer_lowercase() {
    let x = encode_strings(&["Hello", "WORLD", "Foo"]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "LOWER".into());

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("string_normalizer LOWER failed");

    let decoded = decode_strings(&result[0]);
    assert_eq!(decoded, vec!["hello", "world", "foo"]);
}

#[test]
fn test_string_normalizer_uppercase() {
    let x = encode_strings(&["Hello", "world"]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "UPPER".into());

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("string_normalizer UPPER failed");

    let decoded = decode_strings(&result[0]);
    assert_eq!(decoded, vec!["HELLO", "WORLD"]);
}

#[test]
fn test_string_normalizer_none_action() {
    let x = encode_strings(&["Hello", "World"]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "NONE".into());

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("string_normalizer NONE failed");

    let decoded = decode_strings(&result[0]);
    assert_eq!(decoded, vec!["Hello", "World"]);
}

#[test]
fn test_string_normalizer_stopwords_case_sensitive() {
    let x = encode_strings(&["hello", "the", "world", "The"]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "NONE".into());
    attrs.ints.insert("is_case_sensitive".into(), 1);
    attrs
        .string_lists
        .insert("stopwords".into(), vec!["the".into()]);

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("stopwords case-sensitive failed");

    let decoded = decode_strings(&result[0]);
    // "the" removed, "The" kept (case sensitive)
    assert_eq!(decoded, vec!["hello", "world", "The"]);
}

#[test]
fn test_string_normalizer_stopwords_case_insensitive() {
    let x = encode_strings(&["hello", "the", "world", "The"]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "NONE".into());
    attrs.ints.insert("is_case_sensitive".into(), 0);
    attrs
        .string_lists
        .insert("stopwords".into(), vec!["the".into()]);

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("stopwords case-insensitive failed");

    let decoded = decode_strings(&result[0]);
    // Both "the" and "The" removed (case insensitive)
    assert_eq!(decoded, vec!["hello", "world"]);
}

#[test]
fn test_string_normalizer_stopwords_with_case_change() {
    // Stopwords are filtered BEFORE case change is applied
    let x = encode_strings(&["Hello", "a", "World"]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "UPPER".into());
    attrs.ints.insert("is_case_sensitive".into(), 1);
    attrs
        .string_lists
        .insert("stopwords".into(), vec!["a".into()]);

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("stopwords + case change failed");

    let decoded = decode_strings(&result[0]);
    assert_eq!(decoded, vec!["HELLO", "WORLD"]);
}

#[test]
fn test_string_normalizer_empty_input() {
    // Empty tensor
    let x = Tensor::new(vec![], vec![0]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "LOWER".into());

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("empty input failed");

    assert!(result[0].data.is_empty());
}

#[test]
fn test_string_normalizer_single_string() {
    let x = encode_strings(&["OnlyOne"]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "LOWER".into());

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("single string failed");

    let decoded = decode_strings(&result[0]);
    assert_eq!(decoded, vec!["onlyone"]);
}

#[test]
fn test_string_normalizer_all_stopwords_removed() {
    let x = encode_strings(&["a", "the"]);

    let mut attrs = Attributes::default();
    attrs
        .strings
        .insert("case_change_action".into(), "NONE".into());
    attrs
        .string_lists
        .insert("stopwords".into(), vec!["a".into(), "the".into()]);

    let (node, inputs) = make_context(OpKind::StringNormalizer, vec![Some(&x)], attrs);
    let ctx = ctx_from(&node, &inputs);
    let result = string_normalizer(&ctx).expect("all stopwords removed failed");

    assert!(result[0].data.is_empty());
}
