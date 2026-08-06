//! `Multinomial` operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::rng::{resolve_seed, Rng};

/// ONNX `Multinomial` (opset 7+): draw `sample_size` class indices per batch
/// row from the categorical distribution implied by `input`'s unnormalized
/// log-probabilities (logits).
///
/// Input: `input`, shape `[batch_size, class_size]`. Attributes:
/// `sample_size` (default 1), `seed` (optional; see the `rng` submodule for
/// the distributional-not-bitwise contract this shares with the rest of the
/// `Random*` family). Output: `[batch_size, sample_size]`, holding the
/// sampled class index (an integer value stored in an `f32` lane, this
/// engine's usual convention for integer-valued tensors -- e.g.
/// `ArgMax`/`GatherND` indices).
///
/// # Sampling algorithm
///
/// For each batch row, `input` is a vector of *log*-probabilities (logits),
/// so it is first turned into a proper probability distribution via a
/// numerically-stable softmax (subtract the row max before exponentiating),
/// then a category is drawn by inverse-CDF sampling: build the cumulative
/// distribution and find the first index whose cumulative probability
/// exceeds a uniform draw `u ~ Uniform(0, 1)`.
pub struct MultinomialOp;

impl Operator for MultinomialOp {
    fn op_type(&self) -> &str {
        "Multinomial"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        if x.ndim() != 2 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Multinomial: input must be 2-D [batch_size, class_size], got shape {:?}",
                x.shape
            )));
        }
        let batch = x.shape[0];
        let classes = x.shape[1];
        if classes == 0 {
            return Err(OnnxError::InvalidModel(
                "Multinomial: class_size must be > 0".into(),
            ));
        }

        let attrs = ctx.attrs();
        let sample_size_raw = attrs.i("sample_size", 1);
        if sample_size_raw < 0 {
            return Err(OnnxError::InvalidModel(format!(
                "Multinomial: 'sample_size' must be >= 0, got {sample_size_raw}"
            )));
        }
        let sample_size = sample_size_raw as usize;

        let mut rng = Rng::new(resolve_seed(attrs, &ctx.node.name), 0);
        let mut out = Vec::with_capacity(batch * sample_size);
        let mut cdf = vec![0.0_f32; classes];

        for b in 0..batch {
            let logits = &x.data[b * classes..(b + 1) * classes];
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !max_logit.is_finite() {
                return Err(OnnxError::InvalidModel(format!(
                    "Multinomial: row {b} has no finite logit"
                )));
            }
            let mut sum = 0.0_f32;
            for (dst, &v) in cdf.iter_mut().zip(logits.iter()) {
                let e = (v - max_logit).exp();
                sum += e;
                *dst = sum; // running total; normalize to a CDF below
            }
            for v in cdf.iter_mut() {
                *v /= sum;
            }
            // Guard float round-off so the very last draw can never fail to
            // find an index (the true CDF must reach exactly 1.0).
            if let Some(last) = cdf.last_mut() {
                *last = 1.0;
            }

            for _ in 0..sample_size {
                let u = rng.next_f32();
                let idx = cdf.partition_point(|&c| c <= u).min(classes - 1);
                out.push(idx as f32);
            }
        }

        Ok(vec![Tensor::new(out, vec![batch, sample_size])])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(x: &Tensor, sample_size: i64, seed: f32) -> Tensor {
        let mut attrs = oxionnx_core::Attributes::default();
        attrs.ints.insert("sample_size".into(), sample_size);
        attrs.floats.insert("seed".into(), seed);
        let node = oxionnx_core::Node {
            name: "mn".into(),
            op: oxionnx_core::OpKind::Multinomial,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attrs,
        };
        let ctx = OpContext {
            node: &node,
            inputs: vec![Some(x)],
            outer_scope: None,
            weights: None,
            registry: None,
        };
        MultinomialOp.execute(&ctx).expect("execute").remove(0)
    }

    /// A hugely lopsided 2-class distribution (`softmax([0, 10])` puts
    /// ~99.995% of the mass on class 1): with 4000 draws the empirical
    /// frequency of class 1 must be overwhelmingly dominant, or the sampler
    /// is not respecting the logits (e.g. sampling uniformly, or with a
    /// reversed CDF).
    #[test]
    fn multinomial_respects_a_lopsided_distribution() {
        let x = Tensor::new(vec![0.0, 10.0], vec![1, 2]);
        let out = run(&x, 4000, 11.0);
        assert_eq!(out.shape, vec![1, 4000]);
        let frac_class1 = out.data.iter().filter(|&&v| v == 1.0).count() as f64 / 4000.0;
        assert!(
            frac_class1 > 0.98,
            "class 1 (softmax mass ~0.99995) drawn only {frac_class1} of the time"
        );
        // every sample must be a valid class index
        assert!(out.data.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    /// A uniform 4-class distribution (equal logits): with a large sample
    /// every class must appear roughly `n/4` times.
    #[test]
    fn multinomial_uniform_logits_are_roughly_balanced() {
        let x = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let out = run(&x, 8000, 5.0);
        let mut counts = [0usize; 4];
        for &v in &out.data {
            let idx = v as usize;
            assert!(idx < 4, "class index {idx} out of range");
            counts[idx] += 1;
        }
        // Expected count per class = 2000; binomial std dev = sqrt(8000 *
        // 0.25 * 0.75) ~= 38.7. A 300-wide band is ~7.7 SE, comfortably
        // non-flaky while still catching a badly skewed sampler.
        for (idx, &c) in counts.iter().enumerate() {
            assert!(
                (1700..2300).contains(&c),
                "class {idx} count {c} far from the expected ~2000"
            );
        }
    }

    /// Two independent batch rows with different distributions must not
    /// cross-contaminate: row 0 always draws class 0, row 1 always class 2.
    #[test]
    fn multinomial_batches_are_independent() {
        let x = Tensor::new(
            vec![
                20.0, -20.0, -20.0, //
                -20.0, -20.0, 20.0,
            ],
            vec![2, 3],
        );
        let out = run(&x, 50, 3.0);
        assert_eq!(out.shape, vec![2, 50]);
        assert!(out.data[0..50].iter().all(|&v| v == 0.0));
        assert!(out.data[50..100].iter().all(|&v| v == 2.0));
    }

    #[test]
    fn multinomial_same_seed_is_reproducible() {
        let x = Tensor::new(vec![1.0, 2.0, 0.5, 3.0], vec![1, 4]);
        let a = run(&x, 200, 77.0);
        let b = run(&x, 200, 77.0);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn multinomial_rejects_non_2d_input() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let mut attrs = oxionnx_core::Attributes::default();
        attrs.ints.insert("sample_size".into(), 1);
        let node = oxionnx_core::Node {
            name: "mn".into(),
            op: oxionnx_core::OpKind::Multinomial,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attrs,
        };
        let ctx = OpContext {
            node: &node,
            inputs: vec![Some(&x)],
            outer_scope: None,
            weights: None,
            registry: None,
        };
        let err = MultinomialOp.execute(&ctx).expect_err("rank 1 must error");
        assert!(format!("{err}").contains("2-D"), "got: {err}");
    }
}
