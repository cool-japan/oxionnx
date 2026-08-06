//! Shared reduction core for `NegativeLogLikelihoodLoss` and
//! `SoftmaxCrossEntropyLoss` (the latter computes `log_softmax` over its raw
//! `scores` input, then delegates to the same core this module implements).

use oxionnx_core::{Attributes, OnnxError, Tensor};

/// Parse the `reduction` and `ignore_index` attributes shared by both loss
/// operators.
///
/// `ignore_index` is read directly off the attribute map (not through
/// [`Attributes::i`]'s fill-in-a-default accessor) so that "absent" and
/// "explicitly set to some value that happens to equal the fill-in default"
/// can never be confused -- there is no such collision risk here since the
/// spec's `ignore_index` has no default (it means "ignore nothing" only when
/// truly absent).
pub(crate) fn parse_loss_attrs<'a>(
    attrs: &'a Attributes,
    op: &str,
) -> Result<(&'a str, Option<i64>), OnnxError> {
    let reduction = match attrs.s("reduction") {
        "" | "mean" => "mean",
        "sum" => "sum",
        "none" => "none",
        other => {
            return Err(OnnxError::InvalidModel(format!(
                "{op}: unknown reduction '{other}' (expected 'mean', 'sum' or 'none')"
            )))
        }
    };
    let ignore_index = attrs.ints.get("ignore_index").copied();
    Ok((reduction, ignore_index))
}

/// The shared `NegativeLogLikelihoodLoss` computation.
///
/// `log_probs`: shape `[N, C, d1, .., dk]` (already log-probabilities -- for
/// `NegativeLogLikelihoodLoss` this is the raw `input`, unchanged per spec;
/// for `SoftmaxCrossEntropyLoss` the caller has already applied
/// `log_softmax(scores, axis=1)`).
/// `target`: shape `[N, d1, .., dk]`, class indices (float-encoded, this
/// engine's usual integer-in-an-f32-lane convention).
/// `weight`: optional, shape `[C]`.
///
/// Per-element loss is `-weight[c] * log_probs[n, c, d1, .., dk]` where
/// `c = target[n, d1, .., dk]`; an element whose target equals
/// `ignore_index` contributes `0` and is excluded from both the `sum` and
/// the weight-sum used by `mean` -- so a `mean` reduction over a batch that
/// is *entirely* ignored is `0.0 / 0.0 = NaN`, matching IEEE 754 float
/// division and (checked against `onnx.reference` 1.21) the ONNX reference
/// implementation's own behavior for that input; no special-casing needed.
pub(crate) fn nll_loss(
    log_probs: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: &str,
    ignore_index: Option<i64>,
    op: &str,
) -> Result<Tensor, OnnxError> {
    let rank = log_probs.ndim();
    if rank < 2 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: input must have rank >= 2 ([N, C, ...]), got shape {:?}",
            log_probs.shape
        )));
    }
    let n = log_probs.shape[0];
    let c = log_probs.shape[1];
    let spatial = &log_probs.shape[2..];
    let s: usize = spatial.iter().product();

    let mut expected_target_shape = Vec::with_capacity(1 + spatial.len());
    expected_target_shape.push(n);
    expected_target_shape.extend_from_slice(spatial);
    if target.shape != expected_target_shape {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: target shape {:?} does not match the expected {:?} (input shape {:?} \
             without its class axis)",
            target.shape, expected_target_shape, log_probs.shape
        )));
    }
    if let Some(w) = weight {
        if w.numel() != c {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: weight has {} elements, expected {c} (one per class)",
                w.numel()
            )));
        }
    }

    let mut unreduced = vec![0.0_f32; n * s];
    let mut weight_used = vec![0.0_f32; n * s];
    for ni in 0..n {
        for si in 0..s {
            let flat = ni * s + si;
            let t_val = target.data[flat];
            if !t_val.is_finite() {
                return Err(OnnxError::InvalidModel(format!(
                    "{op}: target contains a non-finite class index ({t_val})"
                )));
            }
            let t_idx = t_val as i64;
            if ignore_index == Some(t_idx) {
                continue; // unreduced/weight_used stay 0 and are excluded below
            }
            if t_idx < 0 || t_idx as usize >= c {
                return Err(OnnxError::InvalidModel(format!(
                    "{op}: target class index {t_idx} out of range [0, {c})"
                )));
            }
            let ci = t_idx as usize;
            let w = weight.map_or(1.0, |wt| wt.data[ci]);
            let input_flat = (ni * c + ci) * s + si;
            unreduced[flat] = -w * log_probs.data[input_flat];
            weight_used[flat] = w;
        }
    }

    match reduction {
        "none" => Ok(Tensor::new(unreduced, expected_target_shape)),
        "sum" => Ok(Tensor::rank0(unreduced.iter().sum())),
        "mean" => {
            let total: f32 = unreduced.iter().sum();
            let wsum: f32 = weight_used.iter().sum();
            Ok(Tensor::rank0(total / wsum))
        }
        _ => unreachable!("reduction is validated by parse_loss_attrs"),
    }
}
