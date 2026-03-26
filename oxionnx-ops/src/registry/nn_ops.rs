//! Operator trait implementations for neural network operations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::nn;

// ── Simple unary activations (return Tensor directly) ───────────────────────

macro_rules! unary_nn_op {
    ($name:ident, $op_type:expr, $func:path) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                Ok(vec![$func(ctx.input(0)?)])
            }
        }
    };
}

/// Unary NN op with in-place support via a per-element closure.
macro_rules! unary_nn_op_inplace {
    ($name:ident, $op_type:expr, $func:path, $inplace_fn:expr) => {
        pub struct $name;
        impl Operator for $name {
            fn op_type(&self) -> &str {
                $op_type
            }
            fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
                Ok(vec![$func(ctx.input(0)?)])
            }
            fn supports_inplace(&self) -> bool {
                true
            }
            fn execute_inplace(
                &self,
                mut input: Tensor,
                _ctx: &OpContext<'_>,
            ) -> Result<Vec<Tensor>, OnnxError> {
                let f: fn(f32) -> f32 = $inplace_fn;
                for x in input.data.iter_mut() {
                    *x = f(*x);
                }
                Ok(vec![input])
            }
        }
    };
}

unary_nn_op_inplace!(ReluOp, "Relu", nn::relu, |x: f32| x.max(0.0));
unary_nn_op_inplace!(SigmoidOp, "Sigmoid", nn::sigmoid, |x: f32| {
    1.0 / (1.0 + (-x).exp())
});
unary_nn_op_inplace!(TanhOp, "Tanh", nn::tanh_op, f32::tanh);
unary_nn_op_inplace!(GeluOp, "Gelu", nn::gelu, |x: f32| {
    x * 0.5 * (1.0 + (x * 0.797_884_6 * (1.0 + 0.044715 * x * x)).tanh())
});
unary_nn_op_inplace!(SiLUOp, "SiLU", nn::silu, |x: f32| {
    x / (1.0 + (-x).exp())
});
unary_nn_op!(HardSwishOp, "HardSwish", nn::hard_swish);
unary_nn_op!(SoftplusOp, "Softplus", nn::softplus);
unary_nn_op!(SoftsignOp, "Softsign", nn::softsign);
unary_nn_op!(MishOp, "Mish", nn::mish);
unary_nn_op!(DropoutOp, "Dropout", nn::dropout);

// ── Erf (custom inline implementation) ─────────────────────────────────────

/// Polynomial approximation of error function (Abramowitz & Stegun 7.1.26)
fn erf_approx(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0f32 } else { -1.0f32 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

pub struct ErfOp;
impl Operator for ErfOp {
    fn op_type(&self) -> &str {
        "Erf"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let data: Vec<f32> = x.data.iter().map(|&v| erf_approx(v)).collect();
        Ok(vec![Tensor::new(data, x.shape.clone())])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = erf_approx(*x);
        }
        Ok(vec![input])
    }
}

// ── Abs, Log, Exp (inline unary ops) ───────────────────────────────────────

pub struct AbsOp;
impl Operator for AbsOp {
    fn op_type(&self) -> &str {
        "Abs"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        Ok(vec![Tensor::new(
            x.data.iter().map(|v| v.abs()).collect(),
            x.shape.clone(),
        )])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = x.abs();
        }
        Ok(vec![input])
    }
}

pub struct LogOp;
impl Operator for LogOp {
    fn op_type(&self) -> &str {
        "Log"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        Ok(vec![Tensor::new(
            x.data.iter().map(|v| v.ln()).collect(),
            x.shape.clone(),
        )])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = x.ln();
        }
        Ok(vec![input])
    }
}

pub struct ExpOp;
impl Operator for ExpOp {
    fn op_type(&self) -> &str {
        "Exp"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        Ok(vec![Tensor::new(
            x.data.iter().map(|v| v.exp()).collect(),
            x.shape.clone(),
        )])
    }
    fn supports_inplace(&self) -> bool {
        true
    }
    fn execute_inplace(
        &self,
        mut input: Tensor,
        _ctx: &OpContext<'_>,
    ) -> Result<Vec<Tensor>, OnnxError> {
        for x in input.data.iter_mut() {
            *x = x.exp();
        }
        Ok(vec![input])
    }
}

// ── Clip ────────────────────────────────────────────────────────────────────

pub struct ClipOp;
impl Operator for ClipOp {
    fn op_type(&self) -> &str {
        "Clip"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let min_val = ctx
            .optional_input(1)
            .map(|t| t.data[0])
            .unwrap_or(f32::NEG_INFINITY);
        let max_val = ctx
            .optional_input(2)
            .map(|t| t.data[0])
            .unwrap_or(f32::INFINITY);
        Ok(vec![Tensor::new(
            x.data.iter().map(|&v| v.clamp(min_val, max_val)).collect(),
            x.shape.clone(),
        )])
    }
}

// ── Softmax ─────────────────────────────────────────────────────────────────

pub struct SoftmaxOp;
impl Operator for SoftmaxOp {
    fn op_type(&self) -> &str {
        "Softmax"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axis = ctx.attrs().i("axis", -1);
        Ok(vec![nn::softmax(ctx.input(0)?, axis)?])
    }
}

// ── LogSoftmax ──────────────────────────────────────────────────────────────

pub struct LogSoftmaxOp;
impl Operator for LogSoftmaxOp {
    fn op_type(&self) -> &str {
        "LogSoftmax"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axis = ctx.attrs().i("axis", -1);
        Ok(vec![nn::log_softmax(ctx.input(0)?, axis)?])
    }
}

// ── LayerNorm ───────────────────────────────────────────────────────────────

pub struct LayerNormOp;
impl Operator for LayerNormOp {
    fn op_type(&self) -> &str {
        "LayerNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let eps = attrs.f("epsilon", 1e-5);
        let axis = attrs.i("axis", -1);
        Ok(vec![nn::layer_norm(x, scale, bias, eps, axis)?])
    }
}

// ── GroupNorm ───────────────────────────────────────────────────────────────

pub struct GroupNormOp;
impl Operator for GroupNormOp {
    fn op_type(&self) -> &str {
        "GroupNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let num_groups = attrs.i("num_groups", 1) as usize;
        let eps = attrs.f("epsilon", 1e-5);
        Ok(vec![nn::group_norm(x, scale, bias, num_groups, eps)?])
    }
}

// ── BatchNorm ───────────────────────────────────────────────────────────────

pub struct BatchNormOp;
impl Operator for BatchNormOp {
    fn op_type(&self) -> &str {
        "BatchNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.input(2)?;
        let mean = ctx.input(3)?;
        let var = ctx.input(4)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        Ok(vec![nn::batch_norm(x, scale, bias, mean, var, eps)?])
    }
}

// ── RMSNorm ─────────────────────────────────────────────────────────────────

pub struct RmsNormOp;
impl Operator for RmsNormOp {
    fn op_type(&self) -> &str {
        "SimplifiedLayerNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let eps = ctx.attrs().f("epsilon", 1e-6);
        Ok(vec![nn::rms_norm(x, scale, eps)?])
    }
}

// ── LeakyRelu ───────────────────────────────────────────────────────────────

pub struct LeakyReluOp;
impl Operator for LeakyReluOp {
    fn op_type(&self) -> &str {
        "LeakyRelu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let alpha = ctx.attrs().f("alpha", 0.01);
        Ok(vec![nn::leaky_relu(ctx.input(0)?, alpha)])
    }
}

// ── PRelu ───────────────────────────────────────────────────────────────────

pub struct PReluOp;
impl Operator for PReluOp {
    fn op_type(&self) -> &str {
        "PRelu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![nn::prelu(ctx.input(0)?, ctx.input(1)?)])
    }
}

// ── HardSigmoid ─────────────────────────────────────────────────────────────

pub struct HardSigmoidOp;
impl Operator for HardSigmoidOp {
    fn op_type(&self) -> &str {
        "HardSigmoid"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let alpha = attrs.f("alpha", 0.2);
        let beta = attrs.f("beta", 0.5);
        Ok(vec![nn::hard_sigmoid(ctx.input(0)?, alpha, beta)])
    }
}

// ── Celu ────────────────────────────────────────────────────────────────────

pub struct CeluOp;
impl Operator for CeluOp {
    fn op_type(&self) -> &str {
        "Celu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let alpha = ctx.attrs().f("alpha", 1.0);
        Ok(vec![nn::celu(ctx.input(0)?, alpha)])
    }
}

// ── Elu ─────────────────────────────────────────────────────────────────────

pub struct EluOp;
impl Operator for EluOp {
    fn op_type(&self) -> &str {
        "Elu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let alpha = ctx.attrs().f("alpha", 1.0);
        Ok(vec![nn::elu(ctx.input(0)?, alpha)])
    }
}

// ── Selu ────────────────────────────────────────────────────────────────────

pub struct SeluOp;
impl Operator for SeluOp {
    fn op_type(&self) -> &str {
        "Selu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let alpha = attrs.f("alpha", 1.6732632);
        let gamma = attrs.f("gamma", 1.050_701);
        Ok(vec![nn::selu(ctx.input(0)?, alpha, gamma)])
    }
}

// ── ThresholdedRelu ─────────────────────────────────────────────────────────

pub struct ThresholdedReluOp;
impl Operator for ThresholdedReluOp {
    fn op_type(&self) -> &str {
        "ThresholdedRelu"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let alpha = ctx.attrs().f("alpha", 1.0);
        Ok(vec![nn::thresholded_relu(ctx.input(0)?, alpha)])
    }
}

// ── InstanceNorm ────────────────────────────────────────────────────────────

pub struct InstanceNormOp;
impl Operator for InstanceNormOp {
    fn op_type(&self) -> &str {
        "InstanceNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.input(2)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        Ok(vec![nn::instance_norm(x, scale, bias, eps)?])
    }
}

// ── LpNorm ──────────────────────────────────────────────────────────────────

pub struct LpNormOp;
impl Operator for LpNormOp {
    fn op_type(&self) -> &str {
        "LpNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let axis = attrs.i("axis", -1);
        let p = attrs.i("p", 2);
        Ok(vec![nn::lp_norm(ctx.input(0)?, axis, p)?])
    }
}

// ── MeanVarianceNormalization ───────────────────────────────────────────────

pub struct MeanVarianceNormalizationOp;
impl Operator for MeanVarianceNormalizationOp {
    fn op_type(&self) -> &str {
        "MeanVarianceNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axes_list = ctx.attrs().ints("axes");
        let axes = if axes_list.is_empty() {
            vec![0, 2, 3]
        } else {
            axes_list.to_vec()
        };
        Ok(vec![nn::mean_variance_normalization(ctx.input(0)?, &axes)?])
    }
}
