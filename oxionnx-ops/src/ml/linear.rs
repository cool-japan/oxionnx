//! LinearClassifier and LinearRegressor ONNX-ML operator implementations.

use oxionnx_core::{OnnxError, OpContext, Tensor};

use super::post_transform::{logistic_inplace, probit_inplace, softmax_rows};
use super::shape::batch_dims;

/// ONNX-ML LinearClassifier operator.
///
/// Input 0: X \[N, features\] (a 1-D \[C\] input is one sample with C features)
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
    let (n, features) = batch_dims(x, "LinearClassifier")?;
    if features == 0 {
        return Err(OnnxError::ShapeMismatch(
            "LinearClassifier: input has zero features".into(),
        ));
    }

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
                // Coefficient lists shorter than num_targets * features are
                // malformed; treat the missing weights as zero rather than
                // indexing out of bounds.
                val +=
                    x.data[x_offset + f] * coefficients.get(w_offset + f).copied().unwrap_or(0.0);
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

/// ONNX-ML LinearRegressor operator.
///
/// Input 0: X \[N, features\] (a 1-D \[C\] input is one sample with C features)
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

    let (n, features) = batch_dims(x, "LinearRegressor")?;
    if features == 0 {
        return Err(OnnxError::ShapeMismatch(
            "LinearRegressor: input has zero features".into(),
        ));
    }

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
