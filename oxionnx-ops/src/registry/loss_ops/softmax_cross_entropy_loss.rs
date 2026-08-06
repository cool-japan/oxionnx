//! `SoftmaxCrossEntropyLoss` operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::nll_core::{nll_loss, parse_loss_attrs};

/// ONNX `SoftmaxCrossEntropyLoss` (opset 12+): `log_softmax(scores, axis=1)`
/// followed by the same reduction `NegativeLogLikelihoodLoss` implements
/// (both delegate to the same shared `nll_loss` core in this `loss_ops`
/// module).
///
/// Inputs: `scores` (raw logits, shape `[N, C, d1, .., dk]` -- unlike
/// `NegativeLogLikelihoodLoss`, these are **not** yet log-probabilities),
/// `labels` (class indices, `[N, d1, .., dk]`), optional `weights` (`[C]`).
/// Attributes: `reduction`, `ignore_index` (identical contract to
/// `NegativeLogLikelihoodLoss`).
///
/// Outputs: `output` (the loss) and, when the node declares a second output,
/// `log_prob` -- `log_softmax(scores, axis=1)`, the same shape as `scores`.
pub struct SoftmaxCrossEntropyLossOp;

impl Operator for SoftmaxCrossEntropyLossOp {
    fn op_type(&self) -> &str {
        "SoftmaxCrossEntropyLoss"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let scores = ctx.input(0)?;
        let labels = ctx.input(1)?;
        let weight = ctx.optional_input(2);
        let (reduction, ignore_index) = parse_loss_attrs(ctx.attrs(), "SoftmaxCrossEntropyLoss")?;

        let log_prob = crate::nn::log_softmax(scores, 1)?;
        let loss = nll_loss(
            &log_prob,
            labels,
            weight,
            reduction,
            ignore_index,
            "SoftmaxCrossEntropyLoss",
        )?;

        let mut outputs = vec![loss];
        if ctx.node.outputs.len() > 1 {
            outputs.push(log_prob);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        scores: &Tensor,
        labels: &Tensor,
        weight: Option<&Tensor>,
        reduction: &str,
        ignore_index: Option<i64>,
        num_outputs: usize,
    ) -> Result<Vec<Tensor>, OnnxError> {
        let mut attrs = oxionnx_core::Attributes::default();
        attrs
            .strings
            .insert("reduction".into(), reduction.to_string());
        if let Some(ig) = ignore_index {
            attrs.ints.insert("ignore_index".into(), ig);
        }
        let mut inputs = vec!["scores".into(), "labels".into()];
        if weight.is_some() {
            inputs.push("weight".into());
        }
        let mut outputs = vec!["y".into()];
        if num_outputs > 1 {
            outputs.push("log_prob".into());
        }
        let node = oxionnx_core::Node {
            name: "scel".into(),
            op: oxionnx_core::OpKind::SoftmaxCrossEntropyLoss,
            inputs,
            outputs,
            attrs,
        };
        let mut op_inputs = vec![Some(scores), Some(labels)];
        if let Some(w) = weight {
            op_inputs.push(Some(w));
        }
        let ctx = OpContext {
            node: &node,
            inputs: op_inputs,
            outer_scope: None,
            weights: None,
            registry: None,
        };
        SoftmaxCrossEntropyLossOp.execute(&ctx)
    }

    fn probe() -> (Tensor, Tensor) {
        let scores = Tensor::new(
            vec![
                1.0, 2.0, 0.5, //
                0.2, 0.1, 3.0, //
                1.5, 1.5, 1.5,
            ],
            vec![3, 3],
        );
        let labels = Tensor::new(vec![1.0, 2.0, 0.0], vec![3]);
        (scores, labels)
    }

    /// Reference: `onnx.reference` `SoftmaxCrossEntropyLoss` (opset 21).
    #[test]
    fn mean_matches_onnx_reference() {
        let (scores, labels) = probe();
        let out = run(&scores, &labels, None, "mean", None, 1).expect("run");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, Vec::<usize>::new());
        assert!((out[0].data[0] - 0.557_527_5).abs() < 1e-5);
    }

    #[test]
    fn sum_matches_onnx_reference() {
        let (scores, labels) = probe();
        let out = run(&scores, &labels, None, "sum", None, 1).expect("run");
        assert!((out[0].data[0] - 1.672_582_6).abs() < 1e-5);
    }

    #[test]
    fn none_matches_onnx_reference() {
        let (scores, labels) = probe();
        let out = run(&scores, &labels, None, "none", None, 1).expect("run");
        assert_eq!(out[0].shape, vec![3]);
        let expected = [0.464_368_82_f32, 0.109_601_45, 1.098_612_3];
        for (a, e) in out[0].data.iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
        }
    }

    /// The optional second (`log_prob`) output is `log_softmax(scores,
    /// axis=1)`, and is only produced when the node declares it.
    #[test]
    fn second_output_is_log_softmax_and_is_optional() {
        let (scores, labels) = probe();

        let one_output = run(&scores, &labels, None, "mean", None, 1).expect("run");
        assert_eq!(
            one_output.len(),
            1,
            "log_prob must not be computed/returned unrequested"
        );

        let two_outputs = run(&scores, &labels, None, "mean", None, 2).expect("run");
        assert_eq!(two_outputs.len(), 2);
        assert!((two_outputs[0].data[0] - 0.557_527_5).abs() < 1e-5);
        assert_eq!(two_outputs[1].shape, vec![3, 3]);
        let expected_log_prob = [
            -1.464_368_8_f32,
            -0.464_368_82,
            -1.964_368_8,
            -2.909_601_4,
            -3.009_601_6,
            -0.109_601_45,
            -1.098_612_3,
            -1.098_612_3,
            -1.098_612_3,
        ];
        for (a, e) in two_outputs[1].data.iter().zip(expected_log_prob) {
            assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
        }
    }

    #[test]
    fn weight_and_ignore_index_match_onnx_reference() {
        let (scores, labels) = probe();
        let w = Tensor::new(vec![2.0, 1.0, 0.5], vec![3]);
        let mean = run(&scores, &labels, Some(&w), "mean", Some(2), 1).expect("run");
        assert!((mean[0].data[0] - 0.887_197_8).abs() < 1e-5);
        let none = run(&scores, &labels, Some(&w), "none", Some(2), 1).expect("run");
        let expected = [0.464_368_82_f32, 0.0, 2.197_224_6];
        for (a, e) in none[0].data.iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
        }
    }
}
