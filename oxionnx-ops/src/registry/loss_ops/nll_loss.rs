//! `NegativeLogLikelihoodLoss` operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::nll_core::{nll_loss, parse_loss_attrs};

/// ONNX `NegativeLogLikelihoodLoss` (opset 12+).
///
/// Inputs: `input` (log-probabilities, shape `[N, C, d1, .., dk]`), `target`
/// (class indices, shape `[N, d1, .., dk]`), optional `weight` (`[C]`).
/// Attributes: `reduction` (`"mean"` default, `"sum"`, `"none"`),
/// `ignore_index` (optional). See the `nll_loss` core (in this `loss_ops`
/// module) for the exact reduction formula, including the
/// `ignore_index`/weighting interaction.
pub struct NegativeLogLikelihoodLossOp;

impl Operator for NegativeLogLikelihoodLossOp {
    fn op_type(&self) -> &str {
        "NegativeLogLikelihoodLoss"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let target = ctx.input(1)?;
        let weight = ctx.optional_input(2);
        let (reduction, ignore_index) = parse_loss_attrs(ctx.attrs(), "NegativeLogLikelihoodLoss")?;
        Ok(vec![nll_loss(
            input,
            target,
            weight,
            reduction,
            ignore_index,
            "NegativeLogLikelihoodLoss",
        )?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        input: &Tensor,
        target: &Tensor,
        weight: Option<&Tensor>,
        reduction: &str,
        ignore_index: Option<i64>,
    ) -> Result<Tensor, OnnxError> {
        let mut attrs = oxionnx_core::Attributes::default();
        attrs
            .strings
            .insert("reduction".into(), reduction.to_string());
        if let Some(ig) = ignore_index {
            attrs.ints.insert("ignore_index".into(), ig);
        }
        let mut inputs = vec!["input".into(), "target".into()];
        if weight.is_some() {
            inputs.push("weight".into());
        }
        let node = oxionnx_core::Node {
            name: "nll".into(),
            op: oxionnx_core::OpKind::NegativeLogLikelihoodLoss,
            inputs,
            outputs: vec!["y".into()],
            attrs,
        };
        let mut op_inputs = vec![Some(input), Some(target)];
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
        Ok(NegativeLogLikelihoodLossOp.execute(&ctx)?.remove(0))
    }

    fn probe() -> (Tensor, Tensor) {
        let x = Tensor::new(
            vec![
                -1.2, -0.5, -2.1, -3.0, //
                -0.1, -2.5, -1.8, -0.9, //
                -3.3, -0.2, -1.1, -2.7,
            ],
            vec![3, 4],
        );
        let target = Tensor::new(vec![1.0, 0.0, 2.0], vec![3]);
        (x, target)
    }

    /// Reference: `onnx.reference` `NegativeLogLikelihoodLoss` (opset 21).
    #[test]
    fn basic_mean_matches_onnx_reference() {
        let (x, target) = probe();
        let out = run(&x, &target, None, "mean", None).expect("run");
        assert_eq!(out.shape, Vec::<usize>::new(), "mean reduction is rank-0");
        assert!((out.data[0] - 0.566_666_66).abs() < 1e-5);
    }

    #[test]
    fn basic_sum_matches_onnx_reference() {
        let (x, target) = probe();
        let out = run(&x, &target, None, "sum", None).expect("run");
        assert_eq!(out.shape, Vec::<usize>::new(), "sum reduction is rank-0");
        assert!((out.data[0] - 1.7).abs() < 1e-5);
    }

    #[test]
    fn basic_none_matches_onnx_reference() {
        let (x, target) = probe();
        let out = run(&x, &target, None, "none", None).expect("run");
        assert_eq!(out.shape, vec![3]);
        assert_eq!(out.data, vec![0.5, 0.1, 1.1]);
    }

    #[test]
    fn weighted_mean_and_sum_match_onnx_reference() {
        let (x, target) = probe();
        let w = Tensor::new(vec![1.0, 2.0, 0.5, 3.0], vec![4]);
        let mean = run(&x, &target, Some(&w), "mean", None).expect("run");
        assert!((mean.data[0] - 0.471_428_6).abs() < 1e-5);
        let sum = run(&x, &target, Some(&w), "sum", None).expect("run");
        assert!((sum.data[0] - 1.65).abs() < 1e-5);
        let none = run(&x, &target, Some(&w), "none", None).expect("run");
        assert_eq!(none.data, vec![1.0, 0.1, 0.55]);
    }

    #[test]
    fn ignore_index_excludes_matching_targets_matches_onnx_reference() {
        let (x, target) = probe();
        // target[1] == 0 == ignore_index.
        let mean = run(&x, &target, None, "mean", Some(0)).expect("run");
        assert!((mean.data[0] - 0.8).abs() < 1e-5);
        let sum = run(&x, &target, None, "sum", Some(0)).expect("run");
        assert!((sum.data[0] - 1.6).abs() < 1e-5);
        let none = run(&x, &target, None, "none", Some(0)).expect("run");
        assert_eq!(none.data, vec![0.5, 0.0, 1.1]);
    }

    #[test]
    fn weight_and_ignore_index_together_matches_onnx_reference() {
        let (x, target) = probe();
        let w = Tensor::new(vec![1.0, 2.0, 0.5, 3.0], vec![4]);
        let mean = run(&x, &target, Some(&w), "mean", Some(0)).expect("run");
        assert!((mean.data[0] - 0.62).abs() < 1e-5);
    }

    /// `ignore_index` is compared against the *raw* target value, not a
    /// class-normalized one, so a negative `ignore_index` (the PyTorch-export
    /// convention, e.g. `-100` as the "no label here" sentinel) must still
    /// match and exclude that element -- not be rejected as an out-of-range
    /// class index. Reference: `onnx.reference`.
    #[test]
    fn negative_ignore_index_matches_onnx_reference() {
        let (x, _) = probe();
        let target = Tensor::new(vec![1.0, -100.0, 2.0], vec![3]);
        let none = run(&x, &target, None, "none", Some(-100)).expect("run");
        assert_eq!(none.data, vec![0.5, 0.0, 1.1]);
        let mean = run(&x, &target, None, "mean", Some(-100)).expect("run");
        assert!((mean.data[0] - 0.8).abs() < 1e-5);
    }

    /// `reduction="mean"` over a batch that is *entirely* ignored divides
    /// `0.0 / 0.0`, which is `NaN` under IEEE 754 -- and matches what
    /// `onnx.reference` 1.21 itself returns for this exact input (checked
    /// empirically; it is not merely this engine's choice of convention).
    #[test]
    fn all_targets_ignored_mean_is_nan_matching_onnx_reference() {
        let (x, _) = probe();
        let target = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]);
        let mean = run(&x, &target, None, "mean", Some(0)).expect("run");
        assert!(mean.data[0].is_nan(), "expected NaN, got {}", mean.data[0]);
        let sum = run(&x, &target, None, "sum", Some(0)).expect("run");
        assert_eq!(sum.data[0], 0.0);
        let none = run(&x, &target, None, "none", Some(0)).expect("run");
        assert_eq!(none.data, vec![0.0, 0.0, 0.0]);
    }

    /// N-D case (`N=2, C=3, d1=2`): reference `onnx.reference`.
    #[test]
    fn nd_case_matches_onnx_reference() {
        let x = Tensor::new(
            vec![
                -0.5, -1.2, -2.0, -0.3, -1.5, -1.0, //
                -1.1, -0.4, -0.9, -2.2, -1.8, -0.6,
            ],
            vec![2, 3, 2],
        );
        let target = Tensor::new(vec![0.0, 2.0, 1.0, 0.0], vec![2, 2]);
        let none = run(&x, &target, None, "none", None).expect("run");
        assert_eq!(none.shape, vec![2, 2]);
        assert_eq!(none.data, vec![0.5, 1.0, 0.9, 0.4]);
        let mean = run(&x, &target, None, "mean", None).expect("run");
        assert!((mean.data[0] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn unknown_reduction_errors() {
        let (x, target) = probe();
        let err = run(&x, &target, None, "bogus", None).expect_err("must error");
        assert!(format!("{err}").contains("reduction"), "got: {err}");
    }

    #[test]
    fn out_of_range_target_errors() {
        let (x, _) = probe();
        let target = Tensor::new(vec![1.0, 0.0, 99.0], vec![3]);
        let err = run(&x, &target, None, "mean", None).expect_err("must error");
        assert!(format!("{err}").contains("out of range"), "got: {err}");
    }
}
