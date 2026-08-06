//! Parameterized activation operator implementations: Clip, Softmax,
//! LogSoftmax, LeakyRelu, PRelu, HardSigmoid, Celu, Elu, Selu,
//! ThresholdedRelu, Hardmax, Shrink.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::nn;

// ── Clip ────────────────────────────────────────────────────────────────────

/// Resolve Clip's min/max bounds. Opset-11+ carries them as optional inputs 1/2;
/// opset-6 (and any opset-11+ node that omits the inputs) carries them as the
/// `min`/`max` float attributes instead, defaulting to +/-inf as the spec requires.
/// A present-but-empty bound tensor (malformed model) also falls back to the
/// attribute/default rather than indexing an empty `data` slice.
fn clip_bounds(ctx: &OpContext<'_>) -> (f32, f32) {
    let min_val = ctx
        .optional_input(1)
        .and_then(|t| t.data.first().copied())
        .unwrap_or_else(|| ctx.attrs().f("min", f32::NEG_INFINITY));
    let max_val = ctx
        .optional_input(2)
        .and_then(|t| t.data.first().copied())
        .unwrap_or_else(|| ctx.attrs().f("max", f32::INFINITY));
    (min_val, max_val)
}

/// `v.clamp(min, max)`, but never panics on a malformed model whose
/// min > max (`f32::clamp` asserts `min <= max` and panics otherwise). Chained
/// `max` then `min` matches numpy's/onnxruntime's own clip implementation
/// order, so a well-formed `min <= max` gives byte-identical results to
/// `clamp`, and a malformed `min > max` degrades to a well-defined constant
/// (every element becomes `max`) instead of crashing the process.
fn clip_one(v: f32, min_val: f32, max_val: f32) -> f32 {
    v.max(min_val).min(max_val)
}

pub struct ClipOp;
impl Operator for ClipOp {
    fn op_type(&self) -> &str {
        "Clip"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let (min_val, max_val) = clip_bounds(ctx);
        Ok(vec![Tensor::new(
            x.data
                .iter()
                .map(|&v| clip_one(v, min_val, max_val))
                .collect(),
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
        let (min_val, max_val) = clip_bounds(ctx);
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        for (dst, &x) in slots[0].data.iter_mut().zip(input.data.iter()) {
            *dst = clip_one(x, min_val, max_val);
        }
        Ok(())
    }
}

// ── Softmax family: the opset-13 contract change ────────────────────────────

/// The opset at which `Softmax`/`LogSoftmax`/`Hardmax` changed contract.
const SOFTMAX_FAMILY_OPSET_13: i64 = 13;

/// Resolve which contract `Softmax` / `LogSoftmax` / `Hardmax` executes under,
/// returning `(axis, coerce_to_2d)`.
///
/// ONNX redefined all three at **opset 13**:
///
/// * **opsets 1–12** — `axis` defaults to **1** and names the point at which the
///   input is *coerced to 2D*: the tensor becomes
///   `[prod(shape[..axis]), prod(shape[axis..])]`, the reduction runs across the
///   whole flattened trailing block, and the result is reshaped back.  So an
///   opset-11 `Softmax` on `[1,3,4,4]` normalises over 3·4·4 = 48 values.
/// * **opsets 13+** — `axis` defaults to **-1** and names the single axis that is
///   reduced, independently for every other coordinate.  The same tensor
///   normalises over 3 values.
///
/// The two regimes agree only when the coercion point is the last axis (or the
/// rank is ≤ 2) — which is why the 2D classification tail that ends most models
/// never noticed the difference, and why the divergence only shows on rank > 2.
///
/// The model's declared opset arrives via [`OpContext::opset`]; a graph that
/// declares none reports `DEFAULT_OPSET` and therefore gets the current contract.
fn softmax_family_regime(ctx: &OpContext<'_>) -> (i64, bool) {
    if ctx.opset() < SOFTMAX_FAMILY_OPSET_13 {
        (ctx.attrs().i("axis", 1), true)
    } else {
        (ctx.attrs().i("axis", -1), false)
    }
}

/// Resolve a possibly-negative `axis` against `ndim`.
///
/// ONNX accepts `[-r, r-1]` for every op in this family, in both regimes. A
/// malformed model outside that range (including the pre-opset-13 default of 1
/// applied to a rank-1 tensor) gets a typed error naming the offending axis
/// rather than a wrapped index or a panic.
fn resolve_family_axis(op: &str, axis: i64, ndim: usize) -> Result<usize, OnnxError> {
    let resolved = if axis < 0 { axis + ndim as i64 } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        return Err(OnnxError::from(format!(
            "{op}: axis {axis} out of range for {ndim}D tensor"
        )));
    }
    Ok(resolved as usize)
}

/// The pre-opset-13 2D coercion of `x` at `ax`: `[prod(shape[..ax]), prod(shape[ax..])]`.
///
/// Row-major storage makes the coercion a pure reinterpretation of the buffer —
/// element order is untouched — so the existing N-D kernels compute the legacy
/// contract exactly by running on this view with `axis = 1`, and the caller
/// restores the original shape afterwards.
fn coerce_2d(x: &Tensor, ax: usize) -> Tensor {
    let rows: usize = x.shape[..ax].iter().product();
    let row_len: usize = x.shape[ax..].iter().product();
    Tensor::new(x.data.clone(), vec![rows, row_len])
}

// ── Softmax ─────────────────────────────────────────────────────────────────

pub struct SoftmaxOp;
impl Operator for SoftmaxOp {
    fn op_type(&self) -> &str {
        "Softmax"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let (axis, coerce) = softmax_family_regime(ctx);
        let ax = resolve_family_axis("softmax", axis, x.ndim())?;
        if coerce {
            let flat = nn::softmax(&coerce_2d(x, ax), 1)?;
            return Ok(vec![Tensor::new(flat.data, x.shape.clone())]);
        }
        Ok(vec![nn::softmax(x, ax as i64)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal("SoftmaxOp: no output slots".into()));
        }
        let x = ctx.input(0)?;
        let (axis, coerce) = softmax_family_regime(ctx);
        // Validate before touching the slot: a rejected node must leave the
        // caller's pre-allocated buffer exactly as it found it.
        let ax = resolve_family_axis("softmax", axis, x.ndim())?;
        let n = x.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&x.shape);
        if coerce {
            nn::normalization::softmax_into(&coerce_2d(x, ax), 1, &mut slots[0].data)?;
        } else {
            nn::normalization::softmax_into(x, ax as i64, &mut slots[0].data)?;
        }
        Ok(())
    }
}

// ── LogSoftmax ──────────────────────────────────────────────────────────────

pub struct LogSoftmaxOp;
impl Operator for LogSoftmaxOp {
    fn op_type(&self) -> &str {
        "LogSoftmax"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let (axis, coerce) = softmax_family_regime(ctx);
        let ax = resolve_family_axis("log_softmax", axis, x.ndim())?;
        if coerce {
            let flat = nn::log_softmax(&coerce_2d(x, ax), 1)?;
            return Ok(vec![Tensor::new(flat.data, x.shape.clone())]);
        }
        Ok(vec![nn::log_softmax(x, ax as i64)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal("LogSoftmaxOp: no output slots".into()));
        }
        let x = ctx.input(0)?;
        let (axis, coerce) = softmax_family_regime(ctx);
        // Validate before touching the slot: a rejected node must leave the
        // caller's pre-allocated buffer exactly as it found it.
        let ax = resolve_family_axis("log_softmax", axis, x.ndim())?;
        let n = x.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&x.shape);
        if coerce {
            nn::normalization::log_softmax_into(&coerce_2d(x, ax), 1, &mut slots[0].data)?;
        } else {
            nn::normalization::log_softmax_into(x, ax as i64, &mut slots[0].data)?;
        }
        Ok(())
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
        let (axis, coerce) = softmax_family_regime(ctx);
        let ax = resolve_family_axis("hardmax", axis, x.ndim())?;
        // A zero-sized tensor has no element for the reduction to elect, and
        // `nn::hardmax` writes its one-hot marker unconditionally — on an empty
        // buffer that is an out-of-bounds index, i.e. a panic on a legal (dynamic
        // batch 0) input. Empty in, empty out.
        if x.numel() == 0 {
            return Ok(vec![Tensor::new(Vec::new(), x.shape.clone())]);
        }
        if coerce {
            let flat = nn::hardmax(&coerce_2d(x, ax), 1)?;
            return Ok(vec![Tensor::new(flat.data, x.shape.clone())]);
        }
        Ok(vec![nn::hardmax(x, ax as i64)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal("HardmaxOp: no output slots".into()));
        }
        let input = ctx.input(0)?;
        let (axis, coerce) = softmax_family_regime(ctx);
        let ndim = input.ndim();
        let ax = resolve_family_axis("hardmax", axis, ndim)?;

        // Mirror `execute()`'s guard above: a zero-sized tensor has no element
        // for the reduction to elect. `outer`/`inner` below are `.max(1)`-clamped
        // in the non-coerce (opset 13+) branch to make the legitimate
        // empty-product case work (rank-1, or `ax` flush against a shape
        // boundary) -- but that clamp cannot tell that case apart from a
        // genuine zero-size dim elsewhere in the shape (e.g. `[0,3,4]` or
        // `[2,3,0]` with `ax=1`), and would otherwise send the loop below
        // indexing into `input.data`, which is correctly zero-length for a
        // zero-numel input: an out-of-bounds panic on a legal (e.g. dynamic
        // batch 0) input instead of the correct empty result. Empty in, empty
        // out -- same contract as `execute()`.
        if input.numel() == 0 {
            slots[0].data.clear();
            slots[0].shape.clone_from(&input.shape);
            return Ok(());
        }

        // The pre-opset-13 contract coerces to `[prod(shape[..ax]), prod(shape[ax..])]`
        // and picks one winner per *row* of the coerced matrix; the opset-13+
        // contract picks one winner per slice along `ax`.  Both are the same loop
        // over (outer, axis_len, inner) — the coercion is exactly the case where
        // `axis_len` swallows every trailing dimension and `inner` collapses to 1.
        let (outer, axis_len, inner) = if coerce {
            let rows: usize = input.shape[..ax].iter().product();
            let row_len: usize = input.shape[ax..].iter().product();
            (rows, row_len, 1usize)
        } else {
            (
                input.shape[..ax].iter().product::<usize>().max(1),
                input.shape[ax],
                input.shape[ax + 1..].iter().product::<usize>().max(1),
            )
        };

        let n = input.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        // Must zero everything first: output is one-hot, all non-argmax positions are 0.
        slots[0].data.fill(0.0);

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
                // A zero-length reduction has no winner to mark; without this the
                // `best_k = 0` fallback would write a 1.0 into the next row.
                if axis_len > 0 {
                    slots[0].data[o * axis_len * inner + best_k * inner + i] = 1.0;
                }
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
