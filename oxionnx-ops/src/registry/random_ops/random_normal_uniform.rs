//! `RandomNormal`, `RandomUniform`, `RandomNormalLike`, `RandomUniformLike`.

use oxionnx_core::{Attributes, OnnxError, OpContext, Operator, Tensor};

use super::rng::{resolve_seed, Rng};

/// Read the required `shape` (`ints`) attribute of `RandomNormal`/`RandomUniform`.
fn read_shape_attr(attrs: &Attributes, op: &str) -> Result<Vec<usize>, OnnxError> {
    let ints = attrs.ints("shape");
    if ints.is_empty() {
        return Err(OnnxError::InvalidModel(format!(
            "{op}: missing required 'shape' attribute"
        )));
    }
    ints.iter()
        .map(|&d| {
            if d < 0 {
                Err(OnnxError::InvalidModel(format!(
                    "{op}: 'shape' entries must be non-negative, got {d}"
                )))
            } else {
                Ok(d as usize)
            }
        })
        .collect()
}

/// ONNX `RandomUniform` (opset 1+): `shape` is a required attribute; no
/// inputs. Samples are drawn i.i.d. `Uniform(low, high)`.
pub struct RandomUniformOp;
impl Operator for RandomUniformOp {
    fn op_type(&self) -> &str {
        "RandomUniform"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let shape = read_shape_attr(attrs, "RandomUniform")?;
        let low = attrs.f("low", 0.0);
        let high = attrs.f("high", 1.0);
        let mut rng = Rng::new(resolve_seed(attrs, &ctx.node.name), 0);
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.uniform(low, high)).collect();
        Ok(vec![Tensor::new(data, shape)])
    }
}

/// ONNX `RandomUniformLike` (opset 1+): `shape` is taken from input 0.
pub struct RandomUniformLikeOp;
impl Operator for RandomUniformLikeOp {
    fn op_type(&self) -> &str {
        "RandomUniformLike"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let attrs = ctx.attrs();
        let low = attrs.f("low", 0.0);
        let high = attrs.f("high", 1.0);
        let mut rng = Rng::new(resolve_seed(attrs, &ctx.node.name), 0);
        let data: Vec<f32> = (0..x.numel()).map(|_| rng.uniform(low, high)).collect();
        Ok(vec![Tensor::new(data, x.shape.clone())])
    }
}

/// ONNX `RandomNormal` (opset 1+): `shape` is a required attribute; no
/// inputs. Samples are drawn i.i.d. `Normal(mean, scale)` (`scale` is the
/// standard deviation, per the ONNX attribute name).
pub struct RandomNormalOp;
impl Operator for RandomNormalOp {
    fn op_type(&self) -> &str {
        "RandomNormal"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let shape = read_shape_attr(attrs, "RandomNormal")?;
        let mean = attrs.f("mean", 0.0);
        let scale = attrs.f("scale", 1.0);
        let mut rng = Rng::new(resolve_seed(attrs, &ctx.node.name), 0);
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| mean + scale * rng.next_standard_normal())
            .collect();
        Ok(vec![Tensor::new(data, shape)])
    }
}

/// ONNX `RandomNormalLike` (opset 1+): `shape` is taken from input 0.
pub struct RandomNormalLikeOp;
impl Operator for RandomNormalLikeOp {
    fn op_type(&self) -> &str {
        "RandomNormalLike"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let attrs = ctx.attrs();
        let mean = attrs.f("mean", 0.0);
        let scale = attrs.f("scale", 1.0);
        let mut rng = Rng::new(resolve_seed(attrs, &ctx.node.name), 0);
        let data: Vec<f32> = (0..x.numel())
            .map(|_| mean + scale * rng.next_standard_normal())
            .collect();
        Ok(vec![Tensor::new(data, x.shape.clone())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_no_inputs(node: &oxionnx_core::Node) -> OpContext<'_> {
        OpContext {
            node,
            inputs: Vec::new(),
            outer_scope: None,
            weights: None,
            registry: None,
        }
    }

    fn node_with_attrs(op: oxionnx_core::OpKind, attrs: Attributes) -> oxionnx_core::Node {
        oxionnx_core::Node {
            name: "rnd".into(),
            op,
            inputs: Vec::new(),
            outputs: vec!["y".into()],
            attrs,
        }
    }

    fn shape_attrs(shape: &[i64]) -> Attributes {
        let mut a = Attributes::default();
        a.int_lists.insert("shape".into(), shape.to_vec());
        a
    }

    fn mean_std(data: &[f32]) -> (f64, f64) {
        let n = data.len() as f64;
        let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    }

    #[test]
    fn random_uniform_respects_shape_and_range() {
        let mut attrs = shape_attrs(&[5, 4000]);
        attrs.floats.insert("low".into(), 2.0);
        attrs.floats.insert("high".into(), 5.0);
        attrs.floats.insert("seed".into(), 1.0);
        let node = node_with_attrs(oxionnx_core::OpKind::RandomUniform, attrs);
        let out = RandomUniformOp
            .execute(&ctx_no_inputs(&node))
            .expect("execute");
        assert_eq!(out[0].shape, vec![5, 4000]);
        assert_eq!(out[0].data.len(), 20_000);
        for &v in &out[0].data {
            assert!((2.0..5.0).contains(&v), "sample {v} outside [2, 5)");
        }
        // Law of large numbers: sample mean of Uniform(2,5) -> 3.5, std ->
        // sqrt((5-2)^2/12) ~= 0.866. With n = 20000 the standard error of the
        // mean is ~0.0061; 10 SE is a wide, non-flaky bound that still catches
        // a broken generator (e.g. one that always returns `low`, or samples
        // outside the range).
        let (mean, _) = mean_std(&out[0].data);
        assert!((mean - 3.5).abs() < 0.06, "sample mean {mean} far from 3.5");
    }

    #[test]
    fn random_normal_matches_mean_and_scale() {
        let mut attrs = shape_attrs(&[20_000]);
        attrs.floats.insert("mean".into(), 1.0);
        attrs.floats.insert("scale".into(), 2.0);
        attrs.floats.insert("seed".into(), 7.0);
        let node = node_with_attrs(oxionnx_core::OpKind::RandomNormal, attrs);
        let out = RandomNormalOp
            .execute(&ctx_no_inputs(&node))
            .expect("execute");
        assert_eq!(out[0].data.len(), 20_000);
        let (mean, std) = mean_std(&out[0].data);
        // SE(mean) = scale/sqrt(n) ~= 0.014; SE(std) is of similar order.
        // 0.15 is comfortably wide but would catch a mean/scale mixup.
        assert!((mean - 1.0).abs() < 0.15, "sample mean {mean} far from 1.0");
        assert!((std - 2.0).abs() < 0.15, "sample std {std} far from 2.0");
    }

    #[test]
    fn random_uniform_same_seed_is_reproducible() {
        let attrs = {
            let mut a = shape_attrs(&[100]);
            a.floats.insert("seed".into(), 42.0);
            a
        };
        let node = node_with_attrs(oxionnx_core::OpKind::RandomUniform, attrs.clone());
        let out1 = RandomUniformOp.execute(&ctx_no_inputs(&node)).unwrap();
        let node2 = node_with_attrs(oxionnx_core::OpKind::RandomUniform, attrs);
        let out2 = RandomUniformOp.execute(&ctx_no_inputs(&node2)).unwrap();
        assert_eq!(out1[0].data, out2[0].data);
    }

    #[test]
    fn random_uniform_explicit_seed_zero_is_still_deterministic_and_not_all_low() {
        // Regression guard: a naive `attrs.f("seed", 0.0)` accessor cannot
        // distinguish "seed explicitly set to 0.0" from "seed absent" and
        // would (depending on the fallback branch) either reseed from the
        // wall clock every call or treat 0.0 as a sentinel; this op must
        // treat an explicit 0.0 exactly like any other seed value.
        let attrs = {
            let mut a = shape_attrs(&[256]);
            a.floats.insert("seed".into(), 0.0);
            a
        };
        let node = node_with_attrs(oxionnx_core::OpKind::RandomUniform, attrs.clone());
        let out1 = RandomUniformOp.execute(&ctx_no_inputs(&node)).unwrap();
        let node2 = node_with_attrs(oxionnx_core::OpKind::RandomUniform, attrs);
        let out2 = RandomUniformOp.execute(&ctx_no_inputs(&node2)).unwrap();
        assert_eq!(out1[0].data, out2[0].data);
        // Not every sample identically equal to `low` (which a broken
        // "seed 0 means unseeded/degenerate" path could produce).
        assert!(out1[0].data.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn random_uniform_like_takes_shape_from_input() {
        let x = Tensor::zeros(&[2, 3, 4]);
        let mut attrs = Attributes::default();
        attrs.floats.insert("seed".into(), 3.0);
        let node = oxionnx_core::Node {
            name: "rnd".into(),
            op: oxionnx_core::OpKind::RandomUniformLike,
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
        let out = RandomUniformLikeOp.execute(&ctx).expect("execute");
        assert_eq!(out[0].shape, vec![2, 3, 4]);
        for &v in &out[0].data {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn random_normal_like_takes_shape_from_input() {
        let x = Tensor::zeros(&[10, 10]);
        let mut attrs = Attributes::default();
        attrs.floats.insert("seed".into(), 9.0);
        let node = oxionnx_core::Node {
            name: "rnd".into(),
            op: oxionnx_core::OpKind::RandomNormalLike,
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
        let out = RandomNormalLikeOp.execute(&ctx).expect("execute");
        assert_eq!(out[0].shape, vec![10, 10]);
        assert_eq!(out[0].data.len(), 100);
    }

    #[test]
    fn random_normal_missing_shape_errors() {
        let node = node_with_attrs(oxionnx_core::OpKind::RandomNormal, Attributes::default());
        let err = RandomNormalOp
            .execute(&ctx_no_inputs(&node))
            .expect_err("missing shape must error");
        assert!(format!("{err}").contains("shape"), "got: {err}");
    }
}
