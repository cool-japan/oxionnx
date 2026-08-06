//! Wave-1 correctness tests for the ai.onnx.ml operator family.
//!
//! Covers TfIdfVectorizer's `ngram_counts` / batching semantics, Scaler
//! broadcasting, the 1-D `[C]` input convention shared by every ONNX-ML
//! operator, and the LabelEncoder key/value type matrix.

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::{OnnxError, OpContext, Tensor};
use oxionnx_ops::ml::{label_encoder, linear_regressor, normalizer, scaler, tfidf_vectorizer};

fn node_with(op: OpKind, attrs: Attributes) -> Node {
    Node {
        op,
        name: "w1_ml".to_string(),
        inputs: vec![],
        outputs: vec![],
        attrs,
    }
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

fn run(
    op: OpKind,
    attrs: Attributes,
    x: &Tensor,
    f: fn(&OpContext<'_>) -> Result<Vec<Tensor>, OnnxError>,
) -> Result<Vec<Tensor>, OnnxError> {
    let node = node_with(op, attrs);
    let inputs = vec![Some(x)];
    let ctx = ctx_from(&node, &inputs);
    f(&ctx)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length: {actual:?}");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "index {i}: expected {e}, got {a} ({actual:?})"
        );
    }
}

// ── [a3-3] TfIdfVectorizer ─────────────────────────────────────────────────

/// The canonical ONNX spec pool: four unigrams then three bigrams.
///
/// `ngram_counts = [0, 4]` means the 1-grams start at pool index 0 and the
/// 2-grams start at pool index 4.
fn spec_pool_attrs(mode: &str, min_gram: i64, max_gram: i64, skip: i64) -> Attributes {
    let mut attrs = Attributes::default();
    attrs.strings.insert("mode".into(), mode.into());
    attrs.ints.insert("min_gram_length".into(), min_gram);
    attrs.ints.insert("max_gram_length".into(), max_gram);
    attrs.ints.insert("max_skip_count".into(), skip);
    attrs.int_lists.insert("ngram_counts".into(), vec![0, 4]);
    attrs
        .int_lists
        .insert("ngram_indexes".into(), vec![0, 1, 2, 3, 4, 5, 6]);
    attrs
        .int_lists
        .insert("pool_int64s".into(), vec![2, 3, 5, 4, 5, 6, 7, 8, 6, 7]);
    attrs
}

#[test]
fn ngram_counts_are_pool_start_offsets() {
    // Unigrams {2,3,5,4} -> columns 0..3, bigrams {[5,6],[7,8],[6,7]} -> 4..6.
    // Tokens 1 2 3 5 4 5 6 7 8 6 7:
    //   unigram 2 x1, 3 x1, 5 x2, 4 x1
    //   bigram [5,6] x1, [7,8] x1, [6,7] x2
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 5.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 7.0],
        vec![11],
    );

    let attrs = spec_pool_attrs("TF", 1, 2, 0);
    let out = run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer).expect("tfidf");
    assert_eq!(out[0].shape, vec![7]);
    assert_close(&out[0].data, &[1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]);
}

#[test]
fn only_bigrams_are_counted_when_min_gram_length_is_two() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 5.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 7.0],
        vec![11],
    );

    let attrs = spec_pool_attrs("TF", 2, 2, 0);
    let out = run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer).expect("tfidf");
    // The unigram columns stay zero even though the tokens occur.
    assert_close(&out[0].data, &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
}

#[test]
fn batched_input_produces_one_row_per_sample() {
    // [2, 6] input -> [2, 7] output; n-grams never straddle the row boundary.
    let x = Tensor::new(
        vec![
            1.0, 1.0, 3.0, 3.0, 3.0, 7.0, //
            8.0, 6.0, 7.0, 5.0, 6.0, 8.0,
        ],
        vec![2, 6],
    );

    let attrs = spec_pool_attrs("TF", 1, 2, 0);
    let out = run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer).expect("tfidf");
    assert_eq!(out[0].shape, vec![2, 7]);
    // Row 0: token 3 x3 (column 1); no bigram of the pool occurs.
    // Row 1: token 5 x1 (column 2), 6 x2 (nothing), 7 x1, 8 x2 -> unigram
    // columns are {2,3,5,4} so only 5 hits column 2.
    // Bigrams: [6,7] at index 1 -> column 6, [5,6] at index 3 -> column 4.
    assert_close(
        &out[0].data,
        &[
            0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ],
    );
}

#[test]
fn skip_grams_use_one_fixed_skip_distance() {
    // Tokens: 5 9 6 -> with max_skip_count = 1 the pair (5, 6) is generated at
    // skip distance 2, so bigram [5,6] (column 4) is counted once.
    let x = Tensor::new(vec![5.0, 9.0, 6.0], vec![3]);

    let attrs = spec_pool_attrs("TF", 2, 2, 1);
    let out = run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer).expect("tfidf");
    assert_close(&out[0].data, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn skip_grams_are_not_generated_without_skip_budget() {
    let x = Tensor::new(vec![5.0, 9.0, 6.0], vec![3]);

    let attrs = spec_pool_attrs("TF", 2, 2, 0);
    let out = run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer).expect("tfidf");
    assert_close(&out[0].data, &[0.0; 7]);
}

#[test]
fn idf_mode_reports_presence_times_weight() {
    let x = Tensor::new(vec![5.0, 5.0, 6.0, 7.0], vec![4]);

    let mut attrs = spec_pool_attrs("IDF", 1, 2, 0);
    attrs
        .float_lists
        .insert("weights".into(), vec![0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    let out = run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer).expect("tfidf");
    // Token 5 present (column 2, weight 2.0) even though it occurs twice;
    // bigrams [5,6] (column 4, weight 3.0) and [6,7] (column 6, weight 4.0).
    assert_close(&out[0].data, &[0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]);
}

#[test]
fn tfidf_mode_scales_counts_by_weight() {
    let x = Tensor::new(vec![5.0, 5.0, 6.0, 7.0], vec![4]);

    let mut attrs = spec_pool_attrs("TFIDF", 1, 2, 0);
    attrs
        .float_lists
        .insert("weights".into(), vec![0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    let out = run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer).expect("tfidf");
    // Token 5 occurs twice: 2 * 2.0 = 4.0; the bigrams occur once each.
    assert_close(&out[0].data, &[0.0, 0.0, 4.0, 0.0, 3.0, 0.0, 4.0]);
}

#[test]
fn unknown_mode_is_a_typed_error() {
    let x = Tensor::new(vec![1.0], vec![1]);
    let attrs = spec_pool_attrs("COUNT", 1, 1, 0);
    assert!(matches!(
        run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn out_of_range_ngram_counts_is_a_typed_error() {
    let x = Tensor::new(vec![1.0], vec![1]);
    let mut attrs = spec_pool_attrs("TF", 1, 2, 0);
    attrs.int_lists.insert("ngram_counts".into(), vec![0, 99]);
    assert!(matches!(
        run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn rank_three_input_is_rejected() {
    let x = Tensor::new(vec![1.0; 8], vec![2, 2, 2]);
    let attrs = spec_pool_attrs("TF", 1, 2, 0);
    assert!(matches!(
        run(OpKind::TfIdfVectorizer, attrs, &x, tfidf_vectorizer),
        Err(OnnxError::ShapeMismatch(_))
    ));
}

// ── [a3-14] Scaler broadcasting ────────────────────────────────────────────

#[test]
fn scaler_broadcasts_length_one_offset_and_scale() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let mut attrs = Attributes::default();
    attrs.float_lists.insert("offset".into(), vec![5.0]);
    attrs.float_lists.insert("scale".into(), vec![2.0]);

    let out = run(OpKind::Scaler, attrs, &x, scaler).expect("scaler");
    assert_eq!(out[0].shape, vec![2, 3]);
    assert_close(&out[0].data, &[-8.0, -6.0, -4.0, -2.0, 0.0, 2.0]);
}

#[test]
fn scaler_keeps_per_feature_parameters() {
    let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);

    let mut attrs = Attributes::default();
    attrs
        .float_lists
        .insert("offset".into(), vec![10.0, 20.0, 30.0]);
    attrs
        .float_lists
        .insert("scale".into(), vec![0.1, 0.2, 0.3]);

    let out = run(OpKind::Scaler, attrs, &x, scaler).expect("scaler");
    assert_close(&out[0].data, &[0.0, 0.0, 0.0, 3.0, 6.0, 9.0]);
}

#[test]
fn scaler_rejects_a_parameter_of_the_wrong_length() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);

    let mut attrs = Attributes::default();
    attrs.float_lists.insert("offset".into(), vec![1.0, 2.0]);
    attrs.float_lists.insert("scale".into(), vec![1.0]);

    assert!(matches!(
        run(OpKind::Scaler, attrs, &x, scaler),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn scaler_treats_one_dimensional_input_as_one_sample() {
    // [C] means one sample: the per-feature parameters line up with the values.
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

    let mut attrs = Attributes::default();
    attrs
        .float_lists
        .insert("offset".into(), vec![1.0, 2.0, 3.0]);
    attrs
        .float_lists
        .insert("scale".into(), vec![1.0, 2.0, 3.0]);

    let out = run(OpKind::Scaler, attrs, &x, scaler).expect("scaler");
    assert_eq!(out[0].shape, vec![3]);
    assert_close(&out[0].data, &[0.0, 0.0, 0.0]);
}

// ── [a3-15] 1-D input handling ─────────────────────────────────────────────

#[test]
fn normalizer_l2_normalizes_a_one_dimensional_sample_as_one_vector() {
    let x = Tensor::new(vec![3.0, 4.0], vec![2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("norm".into(), "L2".into());

    let out = run(OpKind::Normalizer, attrs, &x, normalizer).expect("normalizer");
    assert_eq!(out[0].shape, vec![2]);
    assert_close(&out[0].data, &[0.6, 0.8]);
}

#[test]
fn normalizer_defaults_to_max_norm() {
    // The ONNX-ML schema default for `norm` is MAX.
    let x = Tensor::new(vec![3.0, -4.0], vec![1, 2]);

    let attrs = Attributes::default();
    let out = run(OpKind::Normalizer, attrs, &x, normalizer).expect("normalizer");
    assert_close(&out[0].data, &[0.75, -1.0]);
}

#[test]
fn normalizer_rejects_an_unknown_norm() {
    let x = Tensor::new(vec![3.0, 4.0], vec![2]);

    let mut attrs = Attributes::default();
    attrs.strings.insert("norm".into(), "L3".into());

    assert!(matches!(
        run(OpKind::Normalizer, attrs, &x, normalizer),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn linear_regressor_treats_one_dimensional_input_as_one_sample() {
    // W = [1, 2, 3], bias = [1]; x = [1, 1, 1] must give a single 7.0.
    let x = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);

    let mut attrs = Attributes::default();
    attrs
        .float_lists
        .insert("coefficients".into(), vec![1.0, 2.0, 3.0]);
    attrs.float_lists.insert("intercepts".into(), vec![1.0]);
    attrs.strings.insert("post_transform".into(), "NONE".into());

    let out = run(OpKind::LinearRegressor, attrs, &x, linear_regressor).expect("regressor");
    assert_eq!(out[0].shape, vec![1, 1]);
    assert_close(&out[0].data, &[7.0]);
}

// ── [a3-19] LabelEncoder type combinations ─────────────────────────────────

#[test]
fn label_encoder_maps_int_keys_to_float_values() {
    let x = Tensor::new(vec![1.0, 2.0, 7.0], vec![3]);

    let mut attrs = Attributes::default();
    attrs.int_lists.insert("keys_int64s".into(), vec![1, 2]);
    attrs
        .float_lists
        .insert("values_floats".into(), vec![0.5, 1.5]);
    attrs.floats.insert("default_float".into(), -9.0);

    let out = run(OpKind::LabelEncoder, attrs, &x, label_encoder).expect("label_encoder");
    assert_close(&out[0].data, &[0.5, 1.5, -9.0]);
}

#[test]
fn label_encoder_maps_float_keys_to_int_values() {
    let x = Tensor::new(vec![1.5, 2.5, 0.0], vec![3]);

    let mut attrs = Attributes::default();
    attrs
        .float_lists
        .insert("keys_floats".into(), vec![1.5, 2.5]);
    attrs
        .int_lists
        .insert("values_int64s".into(), vec![100, 200]);
    attrs.ints.insert("default_int64".into(), -1);

    let out = run(OpKind::LabelEncoder, attrs, &x, label_encoder).expect("label_encoder");
    assert_close(&out[0].data, &[100.0, 200.0, -1.0]);
}

#[test]
fn label_encoder_supports_tensor_keys_and_values() {
    let x = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);

    let mut attrs = Attributes::default();
    attrs
        .tensors
        .insert("keys_tensor".into(), Tensor::new(vec![4.0, 5.0], vec![2]));
    attrs.tensors.insert(
        "values_tensor".into(),
        Tensor::new(vec![40.0, 50.0], vec![2]),
    );
    attrs
        .tensors
        .insert("default_tensor".into(), Tensor::new(vec![-7.0], vec![1]));

    let out = run(OpKind::LabelEncoder, attrs, &x, label_encoder).expect("label_encoder");
    assert_close(&out[0].data, &[40.0, 50.0, -7.0]);
}

#[test]
fn label_encoder_rejects_mismatched_key_and_value_counts() {
    let x = Tensor::new(vec![1.0], vec![1]);

    let mut attrs = Attributes::default();
    attrs.int_lists.insert("keys_int64s".into(), vec![1, 2, 3]);
    attrs
        .float_lists
        .insert("values_floats".into(), vec![0.5, 1.5]);

    assert!(matches!(
        run(OpKind::LabelEncoder, attrs, &x, label_encoder),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn label_encoder_rejects_string_keys() {
    let x = Tensor::new(vec![1.0], vec![1]);

    let mut attrs = Attributes::default();
    attrs
        .string_lists
        .insert("keys_strings".into(), vec!["a".into()]);
    attrs.int_lists.insert("values_int64s".into(), vec![1]);

    assert!(matches!(
        run(OpKind::LabelEncoder, attrs, &x, label_encoder),
        Err(OnnxError::Unsupported(_))
    ));
}

#[test]
fn label_encoder_without_a_mapping_is_a_typed_error() {
    let x = Tensor::new(vec![1.0], vec![1]);

    assert!(matches!(
        run(
            OpKind::LabelEncoder,
            Attributes::default(),
            &x,
            label_encoder
        ),
        Err(OnnxError::InvalidModel(_))
    ));
}

#[test]
fn label_encoder_maps_nan_input_to_the_default() {
    let x = Tensor::new(vec![f32::NAN, 0.0], vec![2]);

    let mut attrs = Attributes::default();
    attrs.float_lists.insert("keys_floats".into(), vec![0.0]);
    attrs.float_lists.insert("values_floats".into(), vec![1.0]);
    attrs.floats.insert("default_float".into(), -3.0);

    let out = run(OpKind::LabelEncoder, attrs, &x, label_encoder).expect("label_encoder");
    assert_close(&out[0].data, &[-3.0, 1.0]);
}
