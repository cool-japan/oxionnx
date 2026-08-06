//! ONNX-ML SVM operator implementations.
//!
//! Covers SVMClassifier and SVMRegressor.
//!
//! The classifier follows libsvm (and therefore onnxruntime): pairwise
//! decision values are `sum(coef * kernel) + rho`. Without Platt scaling the
//! predicted label is the class with the most one-versus-one votes. When the
//! Platt scaling coefficients `prob_a` / `prob_b` are present, the score
//! output instead holds calibrated probabilities obtained by pairwise
//! coupling *and* the predicted label is the argmax of those coupled
//! probabilities — libsvm's `svm_predict_probability` semantics, which
//! onnxruntime follows. The two selections can disagree: a class can carry a
//! plurality of pairwise votes yet still lose the probability-weighted
//! comparison once the pairwise margins are folded in (see
//! `ml_svm/tests.rs`).

use oxionnx_core::{OnnxError, OpContext, Tensor};

use crate::ml::{apply_post_transform, batch_dims, PostTransform};

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
    /// Parse a `kernel_type` attribute string.
    ///
    /// An absent attribute (`attrs.s` returns `""`) resolves to the ONNX
    /// default of `LINEAR`, same as an explicit `"LINEAR"`. Any other,
    /// unrecognized value is a malformed model, not a silent fallback to
    /// `LINEAR` -- the same "bad enum falls through to a default variant"
    /// pitfall guarded against elsewhere in this crate (`NodeMode::parse`,
    /// `Aggregate::parse` in `ml_tree.rs`).
    fn parse(s: &str, op: &str) -> Result<Self, OnnxError> {
        match s.trim() {
            "" | "LINEAR" => Ok(Self::Linear),
            "POLY" => Ok(Self::Poly),
            "RBF" => Ok(Self::Rbf),
            "SIGMOID" => Ok(Self::Sigmoid),
            other => Err(OnnxError::InvalidModel(format!(
                "{op}: unrecognized kernel_type '{other}' (expected LINEAR, POLY, RBF or SIGMOID)"
            ))),
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

/// Kernel hyper-parameters packed in the `kernel_params` attribute.
struct KernelParams {
    kernel: KernelType,
    gamma: f32,
    coef0: f32,
    degree: f32,
}

impl KernelParams {
    fn parse(attrs: &oxionnx_core::graph::Attributes, op: &str) -> Result<Self, OnnxError> {
        let params = attrs
            .float_lists
            .get("kernel_params")
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        Ok(Self {
            kernel: KernelType::parse(attrs.s("kernel_type"), op)?,
            gamma: params.first().copied().unwrap_or(0.0),
            coef0: params.get(1).copied().unwrap_or(0.0),
            degree: params.get(2).copied().unwrap_or(3.0),
        })
    }

    #[inline]
    fn eval_kernel(&self, x: &[f32], sv: &[f32]) -> f32 {
        kernel_value(self.kernel, x, sv, self.gamma, self.coef0, self.degree)
    }
}

// ── Platt scaling / pairwise coupling ──────────────────────────────────────

/// Numerically stable logistic, matching onnxruntime's `ComputeLogistic`.
#[inline]
fn compute_logistic(val: f32) -> f32 {
    let v = 1.0 / (1.0 + (-val.abs()).exp());
    if val < 0.0 {
        1.0 - v
    } else {
        v
    }
}

/// libsvm's `sigmoid_predict`: `1 / (1 + exp(a * score + b))`.
#[inline]
fn sigmoid_probability(score: f32, prob_a: f32, prob_b: f32) -> f32 {
    1.0 - compute_logistic(score * prob_a + prob_b)
}

/// Scratch buffers for the pairwise coupling, allocated once per call.
struct CouplingScratch {
    /// `k x k` pairwise probability matrix.
    pairwise: Vec<f32>,
    /// `k x k` working matrix of the quadratic program.
    q: Vec<f32>,
    /// `k` element gradient buffer.
    qp: Vec<f32>,
}

impl CouplingScratch {
    fn new(k: usize) -> Self {
        Self {
            pairwise: vec![0.0; k * k],
            q: vec![0.0; k * k],
            qp: vec![0.0; k],
        }
    }
}

/// Multi-class probability estimates from pairwise probabilities.
///
/// Method 2 of Wu, Lin & Weng, "Probability Estimates for Multi-class
/// Classification by Pairwise Coupling" (2004), as implemented by libsvm and
/// onnxruntime.
fn multiclass_probability(k: usize, scratch: &mut CouplingScratch, p: &mut [f32]) {
    if k == 0 || p.len() < k {
        return;
    }
    let r = &scratch.pairwise;
    let q = &mut scratch.q;
    let qp = &mut scratch.qp;
    q.fill(0.0);

    let eps = 0.005 / k as f32;
    for i in 0..k {
        p[i] = 1.0 / k as f32;
        for j in 0..i {
            q[i * k + i] += r[j * k + i] * r[j * k + i];
            q[i * k + j] = q[j * k + i];
        }
        for j in (i + 1)..k {
            q[i * k + i] += r[j * k + i] * r[j * k + i];
            q[i * k + j] = -r[j * k + i] * r[i * k + j];
        }
    }

    for _ in 0..100 {
        // Recompute Qp and pQp for numerical accuracy.
        let mut pqp = 0.0f32;
        for i in 0..k {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += q[i * k + j] * p[j];
            }
            qp[i] = acc;
            pqp += p[i] * acc;
        }

        let max_error = qp
            .iter()
            .take(k)
            .map(|&v| (v - pqp).abs())
            .fold(0.0f32, f32::max);
        if max_error < eps {
            break;
        }

        for i in 0..k {
            let q_ii = q[i * k + i];
            if q_ii == 0.0 {
                continue;
            }
            let diff = (-qp[i] + pqp) / q_ii;
            p[i] += diff;
            let denom = 1.0 + diff;
            if denom == 0.0 {
                continue;
            }
            pqp = (pqp + diff * (diff * q_ii + 2.0 * qp[i])) / denom / denom;
            for j in 0..k {
                qp[j] = (qp[j] + diff * q[i * k + j]) / denom;
                p[j] /= denom;
            }
        }
    }
}

// ── SVMClassifier ──────────────────────────────────────────────────────────

/// ONNX-ML SVMClassifier operator.
///
/// Input 0: X \[N, features\] (a 1-D \[C\] input is one sample with C features)
/// Output 0: predicted labels \[N\] (as f32)
/// Output 1: scores \[N, C\] — probabilities when `prob_a`/`prob_b` are
/// present, otherwise the raw one-versus-one decision values
/// (\[N, C*(C-1)/2\] for more than two classes, \[N, 2\] for two).
pub fn svm_classifier(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    const OP: &str = "SVMClassifier";

    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let params = KernelParams::parse(attrs, OP)?;
    let support_vectors = attrs
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
    let prob_a = attrs
        .float_lists
        .get("prob_a")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let prob_b = attrs
        .float_lists
        .get("prob_b")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let class_labels = attrs.ints("classlabels_int64s");
    let vectors_per_class = attrs.ints("vectors_per_class");
    let post_transform = PostTransform::parse(attrs.s("post_transform"), OP)?;

    let (n, features) = batch_dims(x, OP)?;

    let num_classes = if !class_labels.is_empty() {
        class_labels.len()
    } else if !attrs.string_list("classlabels_strings").is_empty() {
        attrs.string_list("classlabels_strings").len()
    } else if !vectors_per_class.is_empty() {
        vectors_per_class.len()
    } else {
        2
    };
    if num_classes < 2 {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: at least two classes are required, got {num_classes}"
        )));
    }
    // onnxruntime rejects models with 65536 or more classes; the pairwise
    // classifier count is quadratic in the class count.
    if num_classes >= 65536 {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: {num_classes} classes exceed the supported maximum"
        )));
    }
    let pair_count = num_classes * (num_classes - 1) / 2;

    if support_vectors.is_empty() {
        return Err(OnnxError::Unsupported(
            "SVMClassifier: linear mode (no 'support_vectors') is not implemented".into(),
        ));
    }
    if features == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{OP}: input has zero features"
        )));
    }

    // Support vector count: the class partition when available, otherwise
    // inferred from the flattened support vector matrix.
    let mut class_offsets: Vec<usize> = Vec::with_capacity(num_classes + 1);
    let n_sv = if vectors_per_class.is_empty() {
        support_vectors.len() / features
    } else {
        if vectors_per_class.len() != num_classes {
            return Err(OnnxError::InvalidModel(format!(
                "{OP}: 'vectors_per_class' has {} entries but the model declares {num_classes} classes",
                vectors_per_class.len()
            )));
        }
        let mut total = 0usize;
        class_offsets.push(0);
        for (i, &count) in vectors_per_class.iter().enumerate() {
            if count < 0 {
                return Err(OnnxError::InvalidModel(format!(
                    "{OP}: 'vectors_per_class[{i}]' is negative ({count})"
                )));
            }
            total = total.checked_add(count as usize).ok_or_else(|| {
                OnnxError::InvalidModel(format!("{OP}: 'vectors_per_class' sum overflows usize"))
            })?;
            class_offsets.push(total);
        }
        total
    };
    if n_sv == 0 {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: the model declares no support vectors"
        )));
    }
    let needed_support = n_sv.checked_mul(features).ok_or_else(|| {
        OnnxError::InvalidModel(format!("{OP}: support vector matrix size overflows usize"))
    })?;
    if support_vectors.len() < needed_support {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: 'support_vectors' holds {} values, expected at least {needed_support}",
            support_vectors.len()
        )));
    }
    if rho.len() < pair_count {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: 'rho' holds {} values but {pair_count} one-versus-one classifiers are declared",
            rho.len()
        )));
    }
    if num_classes > 2 && class_offsets.is_empty() {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: 'vectors_per_class' is required for {num_classes}-class models"
        )));
    }

    let have_proba = !prob_a.is_empty();
    if have_proba && (prob_a.len() < pair_count || prob_b.len() < pair_count) {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: 'prob_a'/'prob_b' need {pair_count} entries, got {} and {}",
            prob_a.len(),
            prob_b.len()
        )));
    }

    // Score layout follows onnxruntime: probabilities give one column per
    // class, raw scores give one column per one-versus-one classifier (with
    // the binary case widened to two columns).
    let score_cols = if have_proba {
        num_classes
    } else if num_classes > 2 {
        pair_count
    } else {
        2
    };

    let score_count = n.checked_mul(score_cols).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!("{OP}: score buffer size overflows usize"))
    })?;
    let mut all_scores = vec![0.0f32; score_count];
    let mut labels: Vec<f32> = Vec::with_capacity(n);

    let mut kvals = vec![0.0f32; n_sv];
    let mut decisions = vec![0.0f32; pair_count];
    let mut votes = vec![0u32; num_classes];
    let mut scratch = CouplingScratch::new(if have_proba { num_classes } else { 0 });

    for sample_idx in 0..n {
        let x_start = sample_idx * features;
        let x_slice = &x.data[x_start..x_start + features];

        for (sv_idx, kval) in kvals.iter_mut().enumerate() {
            let sv_start = sv_idx * features;
            *kval = params.eval_kernel(x_slice, &support_vectors[sv_start..sv_start + features]);
        }

        votes.iter_mut().for_each(|v| *v = 0);

        if num_classes == 2 {
            // Single classifier: both classes share coefficient row 0.
            let mut sum = 0.0f32;
            for (sv_idx, &k) in kvals.iter().enumerate() {
                sum += coefficients.get(sv_idx).copied().unwrap_or(0.0) * k;
            }
            sum += rho[0];
            decisions[0] = sum;
            votes[if sum > 0.0 { 0 } else { 1 }] += 1;
        } else {
            let mut pair_idx = 0usize;
            for i in 0..(num_classes - 1) {
                let i_row = i * n_sv;
                for j in (i + 1)..num_classes {
                    let j_row = (j - 1) * n_sv;
                    let mut sum = 0.0f32;

                    // Support vectors of class i weighted by row (j - 1).
                    let i_end = class_offsets[i + 1].min(n_sv);
                    for (sv_idx, &k) in kvals.iter().enumerate().take(i_end).skip(class_offsets[i])
                    {
                        sum += coefficients.get(j_row + sv_idx).copied().unwrap_or(0.0) * k;
                    }
                    // Support vectors of class j weighted by row i.
                    let j_end = class_offsets[j + 1].min(n_sv);
                    for (sv_idx, &k) in kvals.iter().enumerate().take(j_end).skip(class_offsets[j])
                    {
                        sum += coefficients.get(i_row + sv_idx).copied().unwrap_or(0.0) * k;
                    }

                    sum += rho[pair_idx];
                    decisions[pair_idx] = sum;
                    votes[if sum > 0.0 { i } else { j }] += 1;
                    pair_idx += 1;
                }
            }
        }

        let score_offset = sample_idx * score_cols;
        let row = &mut all_scores[score_offset..score_offset + score_cols];
        if have_proba {
            // Platt-scale every pairwise decision, then couple the pairwise
            // probabilities into per-class estimates.
            let mut pair_idx = 0usize;
            for i in 0..(num_classes - 1) {
                for j in (i + 1)..num_classes {
                    let raw = sigmoid_probability(
                        decisions[pair_idx],
                        prob_a[pair_idx],
                        prob_b[pair_idx],
                    );
                    let clamped = raw.clamp(1.0e-7, 1.0 - 1.0e-7);
                    scratch.pairwise[i * num_classes + j] = clamped;
                    scratch.pairwise[j * num_classes + i] = 1.0 - clamped;
                    pair_idx += 1;
                }
            }
            multiclass_probability(num_classes, &mut scratch, row);
        } else if num_classes > 2 {
            row.copy_from_slice(&decisions);
        } else {
            row[0] = -decisions[0];
            row[1] = decisions[0];
        }

        // The predicted label: libsvm's `svm_predict_probability` (which
        // onnxruntime follows) selects the class with the highest *coupled
        // probability* whenever Platt scaling is available, in preference to
        // the raw one-versus-one vote count `svm_predict` would use. The two
        // can disagree — a class can carry a plurality of pairwise votes yet
        // still lose the probability-weighted comparison once the pairwise
        // margins are folded in (see
        // `ml_svm/tests.rs::multiclass_probability_argmax_can_disagree_with_votes`).
        // Without `prob_a`/`prob_b` there is no probability estimate to
        // select from, so votes decide, exactly as `svm_predict` does.
        let best_idx = if have_proba {
            let mut idx = 0usize;
            let mut best = row[0];
            for (class_idx, &p) in row.iter().enumerate().skip(1) {
                if p > best {
                    best = p;
                    idx = class_idx;
                }
            }
            idx
        } else {
            let mut idx = 0usize;
            let mut best = votes[0];
            for (class_idx, &count) in votes.iter().enumerate().skip(1) {
                if count > best {
                    best = count;
                    idx = class_idx;
                }
            }
            idx
        };
        labels.push(match class_labels.get(best_idx) {
            Some(&label) => label as f32,
            None => best_idx as f32,
        });
    }

    apply_post_transform(&mut all_scores, n, score_cols, post_transform);

    Ok(vec![
        Tensor::new(labels, vec![n]),
        Tensor::new(all_scores, vec![n, score_cols]),
    ])
}

// ── SVMRegressor ───────────────────────────────────────────────────────────

/// ONNX-ML SVMRegressor operator.
///
/// Input 0: X \[N, features\] (a 1-D \[C\] input is one sample with C features)
/// Output 0: Y \[N, n_targets\]
pub fn svm_regressor(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    const OP: &str = "SVMRegressor";

    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let params = KernelParams::parse(attrs, OP)?;
    let support_vectors = attrs
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

    let n_supports = attrs.i("n_supports", 0);
    if n_supports < 0 {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: 'n_supports' is negative ({n_supports})"
        )));
    }
    let one_class = attrs.i("one_class", 0);
    let post_transform = PostTransform::parse(attrs.s("post_transform"), OP)?;

    let (n, features) = batch_dims(x, OP)?;

    let n_sv = if n_supports > 0 {
        n_supports as usize
    } else {
        support_vectors.len().checked_div(features).unwrap_or(0)
    };
    if n_sv > 0 {
        let needed_support = n_sv.checked_mul(features).ok_or_else(|| {
            OnnxError::InvalidModel(format!("{OP}: support vector matrix size overflows usize"))
        })?;
        if support_vectors.len() < needed_support {
            return Err(OnnxError::InvalidModel(format!(
                "{OP}: 'support_vectors' holds {} values, expected at least {needed_support}",
                support_vectors.len()
            )));
        }
    }

    // Number of targets: inferred from coefficients and support vectors.
    let n_targets = coefficients
        .len()
        .checked_div(n_sv)
        .filter(|&t| t != 0)
        .unwrap_or(1);

    let output_count = n.checked_mul(n_targets).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!("{OP}: output buffer size overflows usize"))
    })?;
    let mut output = vec![0.0f32; output_count];

    for sample_idx in 0..n {
        let x_start = sample_idx * features;
        let x_slice = &x.data[x_start..x_start + features];
        let out_offset = sample_idx * n_targets;

        for target in 0..n_targets {
            let mut val = 0.0f32;
            for sv_idx in 0..n_sv {
                let sv_start = sv_idx * features;
                let k =
                    params.eval_kernel(x_slice, &support_vectors[sv_start..sv_start + features]);
                let coeff_idx = target * n_sv + sv_idx;
                val += coefficients.get(coeff_idx).copied().unwrap_or(0.0) * k;
            }
            val += rho.get(target).copied().unwrap_or(0.0);

            // For one-class SVM the output is the sign of the decision value.
            if one_class != 0 {
                val = if val > 0.0 { 1.0 } else { -1.0 };
            }

            output[out_offset + target] = val;
        }
    }

    apply_post_transform(&mut output, n, n_targets, post_transform);

    Ok(vec![Tensor::new(output, vec![n, n_targets])])
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
