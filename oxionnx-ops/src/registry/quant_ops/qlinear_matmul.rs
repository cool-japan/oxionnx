//! `QLinearMatMul` and `MatMulInteger` operator implementations.

use oxionnx_core::{
    DType, OnnxError, OpContext, Operator, Tensor, TensorStorage, TypedOpContext, TypedTensor,
};

use super::{project_to_f32, saturate_i32, scale_input, tensor_to_i32, zero_point_lanes, SatRange};

/// The `[..., M, K] x [..., K, N]` shape agreement of a (possibly batched)
/// integer matmul, with the ONNX/NumPy 1-D promotion rules applied.
struct MatMulShape {
    /// Number of independent `[M, K] x [K, N]` products.
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    /// Result shape after undoing any 1-D promotion.
    out_shape: Vec<usize>,
    /// Whether `a`'s batch dimensions are broadcast (`batch_a == 1`).
    a_broadcast: bool,
    /// Whether `b`'s batch dimensions are broadcast (`batch_b == 1`).
    b_broadcast: bool,
}

/// Resolve the shapes of an integer matmul.
///
/// Supports the full ONNX contract this engine's quantized graphs need:
/// 2-D x 2-D, 1-D promotion on either side, and stacked batches where one side
/// may carry a single (broadcast) batch.
fn resolve_shapes(a: &[usize], b: &[usize], op: &str) -> Result<MatMulShape, OnnxError> {
    if a.is_empty() || b.is_empty() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: inputs must have rank >= 1, got {a:?} and {b:?}"
        )));
    }
    // 1-D promotion: a `[K]` becomes `[1, K]` on the left / `[K, 1]` on the
    // right, and the promoted axis is removed from the result again.
    let a_1d = a.len() == 1;
    let b_1d = b.len() == 1;
    let a_dims: Vec<usize> = if a_1d { vec![1, a[0]] } else { a.to_vec() };
    let b_dims: Vec<usize> = if b_1d { vec![b[0], 1] } else { b.to_vec() };

    let m = a_dims[a_dims.len() - 2];
    let k = a_dims[a_dims.len() - 1];
    let k_b = b_dims[b_dims.len() - 2];
    let n = b_dims[b_dims.len() - 1];
    if k != k_b {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: inner dimensions disagree: {k} vs {k_b} (shapes {a:?} and {b:?})"
        )));
    }

    let batch_a: usize = a_dims[..a_dims.len() - 2].iter().product();
    let batch_b: usize = b_dims[..b_dims.len() - 2].iter().product();
    let batch = match (batch_a, batch_b) {
        (x, y) if x == y => x,
        (1, y) => y,
        (x, 1) => x,
        (x, y) => {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: batch dimensions disagree ({x} vs {y}) and neither is 1 \
                 (shapes {a:?} and {b:?})"
            )))
        }
    };

    // Batch dimensions of the result come from whichever side is not broadcast.
    let batch_prefix: &[usize] = if batch_a >= batch_b {
        &a_dims[..a_dims.len() - 2]
    } else {
        &b_dims[..b_dims.len() - 2]
    };
    let mut out_shape: Vec<usize> = batch_prefix.to_vec();
    if !a_1d {
        out_shape.push(m);
    }
    if !b_1d {
        out_shape.push(n);
    }

    Ok(MatMulShape {
        batch,
        m,
        k,
        n,
        out_shape,
        a_broadcast: batch_a == 1 && batch != 1,
        b_broadcast: batch_b == 1 && batch != 1,
    })
}

/// Zero point lane for row/column `idx` of a per-tensor or per-row/column list.
#[inline]
fn zp_at(lanes: &[i32], idx: usize) -> i32 {
    if lanes.len() == 1 {
        lanes[0]
    } else {
        lanes.get(idx).copied().unwrap_or(0)
    }
}

/// `Σ (a - a_zp) * (b - b_zp)` for every output element, in `i64`.
///
/// `a_zp` may be per-tensor or per-row of `A` (ONNX allows a 1-D `[M]` zero
/// point for `MatMulInteger`); `b_zp` may be per-tensor or per-column of `B`.
fn integer_matmul(
    a: &[i32],
    b: &[i32],
    a_zp: &[i32],
    b_zp: &[i32],
    shape: &MatMulShape,
) -> Vec<i64> {
    let MatMulShape { batch, m, k, n, .. } = *shape;
    let mut out = vec![0_i64; batch * m * n];
    for bi in 0..batch {
        let a_base = if shape.a_broadcast { 0 } else { bi * m * k };
        let b_base = if shape.b_broadcast { 0 } else { bi * k * n };
        for i in 0..m {
            let a_zp_i = i64::from(zp_at(a_zp, i));
            for j in 0..n {
                let b_zp_j = i64::from(zp_at(b_zp, j));
                let mut sum = 0_i64;
                for p in 0..k {
                    let av = i64::from(a[a_base + i * k + p]) - a_zp_i;
                    let bv = i64::from(b[b_base + p * n + j]) - b_zp_j;
                    sum += av * bv;
                }
                out[bi * m * n + i * n + j] = sum;
            }
        }
    }
    out
}

/// Validate a zero-point input's length against the axis it may vary along.
///
/// ONNX allows `a_zero_point` to be a scalar or a 1-D tensor of size `M` (one
/// per row of `A`), and `b_zero_point` a scalar or a 1-D tensor of size `N`.
/// Any other length is a malformed model: silently reading `0` for the missing
/// lanes would produce a plausible-looking but wrong result.
fn check_zero_point(
    lanes: &[i32],
    expected: usize,
    op: &str,
    which: &str,
) -> Result<(), OnnxError> {
    if lanes.len() == 1 || lanes.len() == expected {
        return Ok(());
    }
    Err(OnnxError::ShapeMismatch(format!(
        "{op}: {which} has {} entries, expected 1 (per tensor) or {expected}",
        lanes.len()
    )))
}

/// Validate that a decoded tensor's element count matches its declared shape.
fn check_len(t: &Tensor, op: &str, which: &str) -> Result<(), OnnxError> {
    let expected: usize = t.shape.iter().product();
    if t.data.len() != expected {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: {which} data length {} does not match shape {:?}",
            t.data.len(),
            t.shape
        )));
    }
    Ok(())
}

// ── MatMulInteger ───────────────────────────────────────────────────────────

/// ONNX `MatMulInteger` (opset 10+): `y = (A - a_zp) @ (B - b_zp)` in int32.
///
/// No scales, no saturation to 8 bits — the output is a raw int32 accumulator.
pub struct MatMulIntegerOp;

impl Operator for MatMulIntegerOp {
    fn op_type(&self) -> &str {
        "MatMulInteger"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let op = "MatMulInteger";
        let a = ctx.input(0)?;
        let b = ctx.input(1)?;
        check_len(a, op, "A")?;
        check_len(b, op, "B")?;
        let a_zp = zero_point_lanes(ctx.optional_input(2), "MatMulInteger: a_zero_point")?;
        let b_zp = zero_point_lanes(ctx.optional_input(3), "MatMulInteger: b_zero_point")?;

        let shape = resolve_shapes(&a.shape, &b.shape, op)?;
        check_zero_point(&a_zp, shape.m, op, "a_zero_point")?;
        check_zero_point(&b_zp, shape.n, op, "b_zero_point")?;
        let a_i32 = tensor_to_i32(a, "MatMulInteger: A")?;
        let b_i32 = tensor_to_i32(b, "MatMulInteger: B")?;
        let acc = integer_matmul(&a_i32, &b_i32, &a_zp, &b_zp, &shape);
        let data: Vec<f32> = acc.into_iter().map(saturate_i32).collect();
        Ok(vec![Tensor::new(data, shape.out_shape)])
    }
}

// ── QLinearMatMul ───────────────────────────────────────────────────────────

/// ONNX `QLinearMatMul` (opset 10+).
///
/// ```text
/// y = saturate(round(((a - a_zp) @ (b - b_zp)) * (a_scale * b_scale / y_scale)) + y_zp)
/// ```
///
/// Inputs, in order: `a, a_scale, a_zero_point, b, b_scale, b_zero_point,
/// y_scale, y_zero_point`. See `SatRange::infer` for the uint8-vs-int8
/// output range rule.
pub struct QLinearMatMulOp;

impl QLinearMatMulOp {
    /// `range_override`, when `Some`, replaces [`SatRange::infer`]'s
    /// value-based cascade with the exact range [`SatRange::for_dtype`]
    /// resolved from `y_zero_point`'s declared dtype — see
    /// [`Operator::execute_typed`] below, the only caller that can supply
    /// one.
    fn run(
        &self,
        ctx: &OpContext<'_>,
        range_override: Option<SatRange>,
    ) -> Result<Tensor, OnnxError> {
        let op = "QLinearMatMul";
        let a = ctx.input(0)?;
        let a_scale = scale_input(ctx.input(1)?, "QLinearMatMul: a_scale")?;
        let a_zp = zero_point_lanes(ctx.optional_input(2), "QLinearMatMul: a_zero_point")?;
        let b = ctx.input(3)?;
        let b_scale = scale_input(ctx.input(4)?, "QLinearMatMul: b_scale")?;
        let b_zp = zero_point_lanes(ctx.optional_input(5), "QLinearMatMul: b_zero_point")?;
        let y_scale = scale_input(ctx.input(6)?, "QLinearMatMul: y_scale")?;
        let y_zp_lanes = zero_point_lanes(ctx.optional_input(7), "QLinearMatMul: y_zero_point")?;
        check_len(a, op, "a")?;
        check_len(b, op, "b")?;

        let shape = resolve_shapes(&a.shape, &b.shape, op)?;
        check_zero_point(&a_zp, shape.m, op, "a_zero_point")?;
        check_zero_point(&b_zp, shape.n, op, "b_zero_point")?;
        let a_i32 = tensor_to_i32(a, "QLinearMatMul: a")?;
        let b_i32 = tensor_to_i32(b, "QLinearMatMul: b")?;
        let acc = integer_matmul(&a_i32, &b_i32, &a_zp, &b_zp, &shape);

        let y_zp = *y_zp_lanes.first().unwrap_or(&0);
        let range = range_override.unwrap_or_else(|| SatRange::infer(&y_zp_lanes, &a_zp));
        // f32 combined scale, f64 product, ties-to-even rounding — the ONNX
        // reference's exact evaluation order.
        let combined = a_scale * b_scale / y_scale;
        let data: Vec<f32> = acc
            .into_iter()
            .map(|v| range.requantize(v, combined, y_zp))
            .collect();
        Ok(Tensor::new(data, shape.out_shape))
    }
}

impl Operator for QLinearMatMulOp {
    fn op_type(&self) -> &str {
        "QLinearMatMul"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        Ok(vec![self.run(ctx, None)?])
    }

    /// `a`/`b`/every zero point may arrive as `I8` or `U8`; every scale is
    /// `F32`. Declaring both here is what lets `execute_typed` below run at
    /// all on `Session::run_typed` — and, crucially, is what makes
    /// `y_zero_point`'s *declared* dtype visible to it in the first place
    /// (see `SatRange::for_dtype`). `I32` matches [`super::QLinearConvOp`]'s
    /// list for consistency; `QLinearMatMul` has no `I32` input of its own so
    /// it never actually gates on that entry — same result either way,
    /// through `execute_typed`'s f32 projection.
    ///
    /// Cost note: this also moves `run_typed`'s weight-sourced inputs (`b`
    /// is typically a static weight matrix, when it is a model initializer)
    /// from a zero-copy borrow to a once-per-run clone — see
    /// [`super::QLinearConvOp::native_dtypes`]'s doc comment for the full
    /// explanation; `Session::run` is unaffected.
    fn native_dtypes(&self) -> &'static [DType] {
        &[DType::I8, DType::U8, DType::I32, DType::F32]
    }

    /// Same computation as [`Self::execute`] (the kernel operates on f32-lane
    /// [`Tensor`]s regardless — this crate's universal quantized-value
    /// convention, see the module doc comment), except `y_zero_point`'s
    /// *declared* dtype is read from the `TypedTensor` **before** the
    /// f32 projection erases it, and threaded through as an explicit
    /// `SatRange` instead of leaving `Self::run` to fall back to
    /// `SatRange::infer`'s value-based (and, in the ambiguous band,
    /// union-clamped) cascade.
    fn execute_typed(&self, ctx: &TypedOpContext<'_>) -> Result<Vec<TypedTensor>, OnnxError> {
        // Input 7 is `y_zero_point` (see `Self::run`).
        let range_override = ctx.input(7).and_then(|t| SatRange::for_dtype(t.dtype()));

        let owned = project_to_f32(ctx);
        let refs: Vec<Option<&Tensor>> = owned.iter().map(|o| o.as_ref()).collect();
        let f32_ctx = OpContext {
            node: ctx.node,
            inputs: refs,
            outer_scope: None,
            weights: None,
            registry: ctx.registry,
        };
        let out = self.run(&f32_ctx, range_override)?;
        Ok(vec![TypedTensor::new(
            TensorStorage::F32(out.data),
            out.shape,
        )])
    }
}
