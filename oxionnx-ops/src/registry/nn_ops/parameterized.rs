//! Parameterized activation operator implementations: Clip, Softmax,
//! LogSoftmax, LeakyRelu, PRelu, HardSigmoid, Celu, Elu, Selu,
//! ThresholdedRelu, Hardmax, Shrink.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::nn;

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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let min_val = ctx
            .optional_input(1)
            .map(|t| t.data[0])
            .unwrap_or(f32::NEG_INFINITY);
        let max_val = ctx
            .optional_input(2)
            .map(|t| t.data[0])
            .unwrap_or(f32::INFINITY);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = x.clamp(min_val, max_val);
        }
        Ok(())
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
    fn supports_output_slots(&self) -> bool {
        true
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
    fn supports_output_slots(&self) -> bool {
        true
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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let alpha = ctx.attrs().f("alpha", 0.01);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = if x >= 0.0 { x } else { alpha * x };
        }
        Ok(())
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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let slope = ctx.input(1)?;
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        slots[0].data.copy_from_slice(&input.data);

        let slope_numel = slope.numel();
        if slope_numel == 1 {
            let alpha = slope.data[0];
            for v in slots[0].data.iter_mut() {
                if *v < 0.0 {
                    *v *= alpha;
                }
            }
        } else if input.ndim() >= 2 {
            let c = slope_numel;
            let spatial: usize = if input.ndim() > 2 {
                input.shape[2..].iter().product()
            } else {
                1
            };
            let batch_n = input.shape[0];
            let x_c = input.shape[1];
            if x_c == c {
                for ni in 0..batch_n {
                    for ci in 0..c {
                        let alpha = slope.data[ci];
                        for si in 0..spatial {
                            let idx = ni * c * spatial + ci * spatial + si;
                            if slots[0].data[idx] < 0.0 {
                                slots[0].data[idx] *= alpha;
                            }
                        }
                    }
                }
            } else {
                for (i, v) in slots[0].data.iter_mut().enumerate() {
                    if *v < 0.0 {
                        *v *= slope.data[i % slope_numel];
                    }
                }
            }
        } else {
            for (i, v) in slots[0].data.iter_mut().enumerate() {
                if *v < 0.0 {
                    *v *= slope.data[i % slope_numel];
                }
            }
        }
        Ok(())
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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let alpha = ctx.attrs().f("alpha", 0.2);
        let beta = ctx.attrs().f("beta", 0.5);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = (alpha * x + beta).clamp(0.0, 1.0);
        }
        Ok(())
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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let alpha = ctx.attrs().f("alpha", 1.0);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = if x >= 0.0 {
                x
            } else {
                alpha * ((x / alpha).exp() - 1.0)
            };
        }
        Ok(())
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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let alpha = ctx.attrs().f("alpha", 1.0);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) };
        }
        Ok(())
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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let alpha = ctx.attrs().f("alpha", 1.6732632);
        let gamma = ctx.attrs().f("gamma", 1.050_701);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = gamma * if x > 0.0 { x } else { alpha * x.exp() - alpha };
        }
        Ok(())
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
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let alpha = ctx.attrs().f("alpha", 1.0);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = if x > alpha { x } else { 0.0 };
        }
        Ok(())
    }
}

// ── Hardmax ──────────────────────────────────────────────────────────────────

pub struct HardmaxOp;
impl Operator for HardmaxOp {
    fn op_type(&self) -> &str {
        "Hardmax"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let axis = ctx.attrs().i("axis", -1);
        Ok(vec![nn::hardmax(x, axis)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let axis = ctx.attrs().i("axis", -1);
        let ndim = input.ndim();
        let ax = if axis < 0 {
            (axis + ndim as i64) as usize
        } else {
            axis as usize
        };
        if ax >= ndim {
            return Err(OnnxError::from(format!(
                "hardmax: axis {axis} out of range for {ndim}D tensor"
            )));
        }
        let n = input.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        // Must zero everything first: output is one-hot, all non-argmax positions are 0.
        slots[0].data.fill(0.0);

        let outer: usize = input.shape[..ax].iter().product::<usize>().max(1);
        let inner: usize = input.shape[ax + 1..].iter().product::<usize>().max(1);
        let axis_len = input.shape[ax];

        for o in 0..outer {
            for i in 0..inner {
                let mut best_k = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for k in 0..axis_len {
                    let idx = o * axis_len * inner + k * inner + i;
                    if input.data[idx] > best_v {
                        best_v = input.data[idx];
                        best_k = k;
                    }
                }
                slots[0].data[o * axis_len * inner + best_k * inner + i] = 1.0;
            }
        }
        Ok(())
    }
}

// ── Shrink ───────────────────────────────────────────────────────────────────

pub struct ShrinkOp;
impl Operator for ShrinkOp {
    fn op_type(&self) -> &str {
        "Shrink"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let bias = ctx.attrs().f("bias", 0.0);
        let lambd = ctx.attrs().f("lambd", 0.5);
        Ok(vec![nn::shrink(x, bias, lambd)])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let bias = ctx.attrs().f("bias", 0.0);
        let lambd = ctx.attrs().f("lambd", 0.5);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = if x < -lambd {
                x + bias
            } else if x > lambd {
                x - bias
            } else {
                0.0
            };
        }
        Ok(())
    }
}
