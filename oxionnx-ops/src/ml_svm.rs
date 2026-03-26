//! ONNX-ML SVM operator implementations.
//!
//! Covers SVMClassifier and SVMRegressor.

use oxionnx_core::{OnnxError, OpContext, Tensor};

use crate::ml::{apply_post_transform, PostTransform};

// ── Kernel helpers ─────────────────────────────────────────────────────────

/// Kernel type for SVM operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelType {
    Linear,
    Poly,
    Rbf,
    Sigmoid,
}

impl KernelType {
    fn parse(s: &str) -> Self {
        match s {
            "POLY" => Self::Poly,
            "RBF" => Self::Rbf,
            "SIGMOID" => Self::Sigmoid,
            _ => Self::Linear,
        }
    }
}

/// Compute kernel value between sample x and support vector sv.
#[inline]
fn kernel_value(
    kernel: KernelType,
    x: &[f32],
    sv: &[f32],
    gamma: f32,
    coef0: f32,
    degree: f32,
) -> f32 {
    match kernel {
        KernelType::Linear => dot(x, sv),
        KernelType::Rbf => {
            let diff_sq: f32 = x
                .iter()
                .zip(sv.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            (-gamma * diff_sq).exp()
        }
        KernelType::Poly => {
            let d = gamma * dot(x, sv) + coef0;
            d.powf(degree)
        }
        KernelType::Sigmoid => {
            let d = gamma * dot(x, sv) + coef0;
            d.tanh()
        }
    }
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ── SVMClassifier ──────────────────────────────────────────────────────────

/// ONNX-ML SVMClassifier operator.
///
/// Input 0: X \[N, features\]
/// Output 0: predicted labels \[N\] (as f32)
/// Output 1: scores \[N, num_classes\]
pub fn svm_classifier(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let kernel_type_str = attrs.s("kernel_type");
    let kernel = KernelType::parse(kernel_type_str);

    let kernel_params = attrs
        .float_lists
        .get("kernel_params")
        .cloned()
        .unwrap_or_default();
    let gamma = if !kernel_params.is_empty() {
        kernel_params[0]
    } else {
        0.0
    };
    let coef0 = if kernel_params.len() > 1 {
        kernel_params[1]
    } else {
        0.0
    };
    let degree = if kernel_params.len() > 2 {
        kernel_params[2]
    } else {
        3.0
    };

    let support_vectors_flat = attrs
        .float_lists
        .get("support_vectors")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let coefficients = attrs
        .float_lists
        .get("coefficients")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let rho = attrs
        .float_lists
        .get("rho")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let class_labels = attrs.ints("classlabels_int64s");
    let vectors_per_class = attrs.ints("vectors_per_class");

    let post_transform_str = attrs.s("post_transform");
    let post_transform = PostTransform::parse(post_transform_str);

    let n = x.shape[0];
    let features = if x.shape.len() > 1 {
        x.shape[1]
    } else {
        x.numel() / n
    };

    // Number of support vectors
    let n_sv = if features > 0 && !support_vectors_flat.is_empty() {
        support_vectors_flat.len() / features
    } else {
        0
    };

    let num_classes = if !class_labels.is_empty() {
        class_labels.len()
    } else if !vectors_per_class.is_empty() {
        vectors_per_class.len()
    } else {
        2 // default binary
    };

    // Allocate output scores
    let mut all_scores = vec![0.0f32; n * num_classes];

    if num_classes == 2 {
        // Binary classification: single decision function
        // score = sum(coeff_i * kernel(x, sv_i)) - rho[0]
        // (note: ONNX uses -rho convention for the bias)
        let rho_val = if !rho.is_empty() { rho[0] } else { 0.0 };

        for sample_idx in 0..n {
            let x_start = sample_idx * features;
            let x_slice = &x.data[x_start..x_start + features];

            let mut decision = 0.0f32;
            for sv_idx in 0..n_sv {
                let sv_start = sv_idx * features;
                let sv_end = sv_start + features;
                if sv_end > support_vectors_flat.len() {
                    break;
                }
                let sv_slice = &support_vectors_flat[sv_start..sv_end];
                let k = kernel_value(kernel, x_slice, sv_slice, gamma, coef0, degree);
                let coeff = if sv_idx < coefficients.len() {
                    coefficients[sv_idx]
                } else {
                    0.0
                };
                decision += coeff * k;
            }
            decision -= rho_val;

            let score_offset = sample_idx * 2;
            // Class 0 score = -decision, class 1 score = decision
            // (positive decision => class 1)
            all_scores[score_offset] = -decision;
            all_scores[score_offset + 1] = decision;
        }
    } else {
        // Multi-class: one-vs-one voting
        // Number of binary classifiers = C*(C-1)/2
        let n_binary = num_classes * (num_classes - 1) / 2;

        // Build class offsets from vectors_per_class
        let mut class_offsets = Vec::with_capacity(num_classes + 1);
        class_offsets.push(0usize);
        for &vpc in vectors_per_class.iter() {
            let last = class_offsets.last().copied().unwrap_or(0);
            class_offsets.push(last + vpc as usize);
        }

        for sample_idx in 0..n {
            let x_start = sample_idx * features;
            let x_slice = &x.data[x_start..x_start + features];

            // Compute all kernel values once
            let mut kvals = vec![0.0f32; n_sv];
            for (sv_idx, kval) in kvals.iter_mut().enumerate() {
                let sv_start = sv_idx * features;
                let sv_end = sv_start + features;
                if sv_end > support_vectors_flat.len() {
                    break;
                }
                let sv_slice = &support_vectors_flat[sv_start..sv_end];
                *kval = kernel_value(kernel, x_slice, sv_slice, gamma, coef0, degree);
            }

            // One-vs-one voting
            let mut votes = vec![0.0f32; num_classes];
            let mut pair_idx = 0usize;
            for i in 0..num_classes {
                for j in (i + 1)..num_classes {
                    if pair_idx >= n_binary {
                        break;
                    }
                    let rho_val = if pair_idx < rho.len() {
                        rho[pair_idx]
                    } else {
                        0.0
                    };

                    // Decision for pair (i, j):
                    // sum over SVs of class i: coeff[row j-1][sv] * k(x, sv)
                    // + sum over SVs of class j: coeff[row i][sv] * k(x, sv)
                    // - rho[pair_idx]
                    let mut decision = 0.0f32;

                    // SVs for class i
                    if i < class_offsets.len() - 1 {
                        let start = class_offsets[i];
                        let end = class_offsets[i + 1];
                        for sv_idx in start..end {
                            if sv_idx >= kvals.len() {
                                break;
                            }
                            // Coefficient row for pair involving class j
                            let coeff_idx = (j - 1) * n_sv + sv_idx;
                            let c = if coeff_idx < coefficients.len() {
                                coefficients[coeff_idx]
                            } else {
                                0.0
                            };
                            decision += c * kvals[sv_idx];
                        }
                    }

                    // SVs for class j
                    if j < class_offsets.len() - 1 {
                        let start = class_offsets[j];
                        let end = class_offsets[j + 1];
                        for sv_idx in start..end {
                            if sv_idx >= kvals.len() {
                                break;
                            }
                            let coeff_idx = i * n_sv + sv_idx;
                            let c = if coeff_idx < coefficients.len() {
                                coefficients[coeff_idx]
                            } else {
                                0.0
                            };
                            decision += c * kvals[sv_idx];
                        }
                    }

                    decision -= rho_val;

                    if decision > 0.0 {
                        votes[i] += 1.0;
                    } else {
                        votes[j] += 1.0;
                    }

                    pair_idx += 1;
                }
            }

            let score_offset = sample_idx * num_classes;
            all_scores[score_offset..(num_classes + score_offset)]
                .copy_from_slice(&votes[..num_classes]);
        }
    }

    // Apply post-transform
    apply_post_transform(&mut all_scores, n, num_classes, post_transform);

    // Compute predicted labels via argmax
    let mut labels = vec![0.0f32; n];
    for (i, label) in labels.iter_mut().enumerate() {
        let row_offset = i * num_classes;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..num_classes {
            if all_scores[row_offset + j] > best_val {
                best_val = all_scores[row_offset + j];
                best_idx = j;
            }
        }
        if !class_labels.is_empty() && best_idx < class_labels.len() {
            *label = class_labels[best_idx] as f32;
        } else {
            *label = best_idx as f32;
        }
    }

    let label_tensor = Tensor::new(labels, vec![n]);
    let score_tensor = Tensor::new(all_scores, vec![n, num_classes]);

    Ok(vec![label_tensor, score_tensor])
}

// ── SVMRegressor ───────────────────────────────────────────────────────────

/// ONNX-ML SVMRegressor operator.
///
/// Input 0: X \[N, features\]
/// Output 0: Y \[N, n_targets\]
pub fn svm_regressor(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let kernel_type_str = attrs.s("kernel_type");
    let kernel = KernelType::parse(kernel_type_str);

    let kernel_params = attrs
        .float_lists
        .get("kernel_params")
        .cloned()
        .unwrap_or_default();
    let gamma = if !kernel_params.is_empty() {
        kernel_params[0]
    } else {
        0.0
    };
    let coef0 = if kernel_params.len() > 1 {
        kernel_params[1]
    } else {
        0.0
    };
    let degree = if kernel_params.len() > 2 {
        kernel_params[2]
    } else {
        3.0
    };

    let support_vectors_flat = attrs
        .float_lists
        .get("support_vectors")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let coefficients = attrs
        .float_lists
        .get("coefficients")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let rho = attrs
        .float_lists
        .get("rho")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let n_supports = attrs.i("n_supports", 0) as usize;
    let one_class = attrs.i("one_class", 0);
    let post_transform_str = attrs.s("post_transform");
    let post_transform = PostTransform::parse(post_transform_str);

    let n = x.shape[0];
    let features = if x.shape.len() > 1 {
        x.shape[1]
    } else {
        x.numel() / n
    };

    let n_sv = if n_supports > 0 {
        n_supports
    } else if features > 0 && !support_vectors_flat.is_empty() {
        support_vectors_flat.len() / features
    } else {
        0
    };

    // Number of targets: inferred from coefficients and support vectors
    let n_targets = coefficients
        .len()
        .checked_div(n_sv)
        .and_then(|t| if t == 0 { None } else { Some(t) })
        .unwrap_or(1);

    let mut output = vec![0.0f32; n * n_targets];

    for sample_idx in 0..n {
        let x_start = sample_idx * features;
        let x_slice = &x.data[x_start..x_start + features];
        let out_offset = sample_idx * n_targets;

        for target in 0..n_targets {
            let mut val = 0.0f32;
            for sv_idx in 0..n_sv {
                let sv_start = sv_idx * features;
                let sv_end = sv_start + features;
                if sv_end > support_vectors_flat.len() {
                    break;
                }
                let sv_slice = &support_vectors_flat[sv_start..sv_end];
                let k = kernel_value(kernel, x_slice, sv_slice, gamma, coef0, degree);
                let coeff_idx = target * n_sv + sv_idx;
                let c = if coeff_idx < coefficients.len() {
                    coefficients[coeff_idx]
                } else {
                    0.0
                };
                val += c * k;
            }
            // Add rho (bias)
            let rho_val = if target < rho.len() { rho[target] } else { 0.0 };
            val += rho_val;

            // For one_class SVM, output is the sign
            if one_class != 0 {
                val = if val > 0.0 { 1.0 } else { -1.0 };
            }

            output[out_offset + target] = val;
        }
    }

    // Apply post-transform
    apply_post_transform(&mut output, n, n_targets, post_transform);

    Ok(vec![Tensor::new(output, vec![n, n_targets])])
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

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
    fn test_svm_classifier_binary_linear() {
        // Binary SVM with LINEAR kernel.
        // 2 support vectors, 2 features.
        // SV0 = [1, 0], SV1 = [0, 1]
        // coefficients = [1.0, -1.0]
        // rho = [0.0]
        // Decision = 1.0 * dot(x, [1,0]) + (-1.0) * dot(x, [0,1]) - 0.0
        //          = x[0] - x[1]
        // If decision > 0: class 1, else class 0

        let x = Tensor::new(
            vec![
                2.0, 1.0, // sample 0: decision = 2 - 1 = 1 > 0 => class 1
                1.0, 3.0, // sample 1: decision = 1 - 3 = -2 < 0 => class 0
            ],
            vec![2, 2],
        );

        let mut attrs = Attributes::default();
        attrs.strings.insert("kernel_type".into(), "LINEAR".into());
        attrs
            .float_lists
            .insert("support_vectors".into(), vec![1.0, 0.0, 0.0, 1.0]);
        attrs
            .float_lists
            .insert("coefficients".into(), vec![1.0, -1.0]);
        attrs.float_lists.insert("rho".into(), vec![0.0]);
        attrs
            .int_lists
            .insert("classlabels_int64s".into(), vec![0, 1]);
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::SVMClassifier, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = svm_classifier(&ctx).expect("svm_classifier failed");

        assert_eq!(result.len(), 2);

        let labels = &result[0];
        assert_eq!(labels.shape, vec![2]);
        // Sample 0: decision = 1 > 0 => class 1
        assert!((labels.data[0] - 1.0).abs() < 1e-5);
        // Sample 1: decision = -2 < 0 => class 0
        assert!((labels.data[1] - 0.0).abs() < 1e-5);

        let scores = &result[1];
        assert_eq!(scores.shape, vec![2, 2]);
        // Sample 0: [-1.0, 1.0]
        assert!((scores.data[0] - (-1.0)).abs() < 1e-5);
        assert!((scores.data[1] - 1.0).abs() < 1e-5);
        // Sample 1: [2.0, -2.0]
        assert!((scores.data[2] - 2.0).abs() < 1e-5);
        assert!((scores.data[3] - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_svm_regressor_linear() {
        // Simple linear SVM regression.
        // 2 support vectors, 2 features.
        // SV0 = [1, 0], SV1 = [0, 1]
        // coefficients = [0.5, 0.5]
        // rho = [1.0]
        // Output = 0.5 * dot(x, [1,0]) + 0.5 * dot(x, [0,1]) + 1.0
        //        = 0.5 * x[0] + 0.5 * x[1] + 1.0

        let x = Tensor::new(
            vec![
                2.0, 4.0, // sample 0: 0.5*2 + 0.5*4 + 1 = 4.0
                0.0, 0.0, // sample 1: 0 + 0 + 1 = 1.0
            ],
            vec![2, 2],
        );

        let mut attrs = Attributes::default();
        attrs.strings.insert("kernel_type".into(), "LINEAR".into());
        attrs
            .float_lists
            .insert("support_vectors".into(), vec![1.0, 0.0, 0.0, 1.0]);
        attrs
            .float_lists
            .insert("coefficients".into(), vec![0.5, 0.5]);
        attrs.float_lists.insert("rho".into(), vec![1.0]);
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::SVMRegressor, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = svm_regressor(&ctx).expect("svm_regressor failed");

        assert_eq!(result.len(), 1);
        let y = &result[0];
        assert_eq!(y.shape, vec![2, 1]);
        assert!((y.data[0] - 4.0).abs() < 1e-5);
        assert!((y.data[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_svm_classifier_rbf_kernel() {
        // Binary SVM with RBF kernel.
        // 1 support vector at origin [0, 0], coefficient = 1.0
        // gamma = 1.0, rho = -0.5
        // kernel(x, [0,0]) = exp(-1.0 * ||x||^2)
        // decision = 1.0 * exp(-||x||^2) - (-0.5) = exp(-||x||^2) + 0.5

        let x = Tensor::new(
            vec![
                0.0, 0.0, // sample 0: exp(0) + 0.5 = 1.5 > 0 => class 1
                10.0, 10.0, // sample 1: exp(-200) + 0.5 ≈ 0.5 > 0 => class 1
                0.5, 0.5, // sample 2: exp(-0.5) + 0.5 ≈ 1.107 > 0 => class 1
            ],
            vec![3, 2],
        );

        let mut attrs = Attributes::default();
        attrs.strings.insert("kernel_type".into(), "RBF".into());
        attrs
            .float_lists
            .insert("kernel_params".into(), vec![1.0, 0.0, 3.0]); // gamma=1.0
        attrs
            .float_lists
            .insert("support_vectors".into(), vec![0.0, 0.0]);
        attrs.float_lists.insert("coefficients".into(), vec![1.0]);
        attrs.float_lists.insert("rho".into(), vec![-0.5]);
        attrs
            .int_lists
            .insert("classlabels_int64s".into(), vec![0, 1]);
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::SVMClassifier, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = svm_classifier(&ctx).expect("svm_classifier rbf failed");

        let labels = &result[0];
        // All samples should be class 1 since decision is always positive
        // (exp(-||x||^2) is always positive, plus 0.5)
        for i in 0..3 {
            assert!(
                (labels.data[i] - 1.0).abs() < 1e-5,
                "sample {i}: expected class 1, got {}",
                labels.data[i]
            );
        }

        // Verify the actual score for sample 0
        let scores = &result[1];
        // decision = exp(0) + 0.5 = 1.5
        // score[class1] = 1.5, score[class0] = -1.5
        assert!((scores.data[1] - 1.5).abs() < 1e-5);
        assert!((scores.data[0] - (-1.5)).abs() < 1e-5);
    }
}
