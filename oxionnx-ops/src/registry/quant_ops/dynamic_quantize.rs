//! `DynamicQuantizeLinear` operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::round_ties_even_f32;

/// ONNX `DynamicQuantizeLinear` (opset 11+).
///
/// Computes uint8 quantization parameters from the data itself and applies
/// them, producing **three** outputs: `y`, `y_scale`, `y_zero_point`.
///
/// ```text
/// qmin, qmax   = 0, 255
/// max_x        = maximum(0, max(x))          // the range always spans 0 so
/// min_x        = minimum(0, min(x))          // that 0 is exactly representable
/// y_scale      = (max_x - min_x) / (qmax - qmin)
/// y_zero_point = round(clamp(qmin - min_x / y_scale, qmin, qmax))
/// y            = clamp(round(x / y_scale) + y_zero_point, qmin, qmax)
/// ```
///
/// Two details are easy to get wrong and are pinned by
/// `tests/w2_quantized_ops_e2e.rs` against `onnx.reference`:
///
/// * the zero point is added **after** rounding (`round(x/s) + zp`), not
///   inside it — for a negative half-way value like `-127.5` the two orders
///   differ by one whole quantization step, and
/// * rounding is **ties-to-even** (`np.rint`), not half-away-from-zero, so
///   `25.5` quantizes to `26` while `24.5` quantizes to `24`.
///
/// The degenerate all-equal input (`max_x == min_x`, e.g. an all-zero tensor)
/// would divide by zero; following `onnx.reference`, the numerator is replaced
/// by `1.0`, giving `y_scale = 1/255` and an all-zero output.
///
/// `y_scale` and `y_zero_point` are declared rank-0 scalars, and are emitted as
/// genuine rank-0 tensors — shape `[]`, via [`Tensor::rank0`] — not as the
/// rank-1 `[1]` of the legacy [`Tensor::scalar`]. Consumers that read `data[0]`
/// (`QuantizeLinear`, `MatMulInteger`, …) are unaffected either way; the
/// difference shows up at a following `Shape` node, which must report a
/// length-0 vector for these outputs.
pub struct DynamicQuantizeLinearOp;

/// `qmin` / `qmax` of the uint8 target type this operator is defined for.
const QMIN: f32 = 0.0;
const QMAX: f32 = 255.0;

impl Operator for DynamicQuantizeLinearOp {
    fn op_type(&self) -> &str {
        "DynamicQuantizeLinear"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        if x.data.is_empty() {
            return Err(OnnxError::InvalidModel(
                "DynamicQuantizeLinear: input is empty".into(),
            ));
        }
        if let Some(bad) = x.data.iter().find(|v| !v.is_finite()) {
            return Err(OnnxError::InvalidModel(format!(
                "DynamicQuantizeLinear: input contains a non-finite value ({bad})"
            )));
        }

        let max_x = x
            .data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0);
        let min_x = x
            .data
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .min(0.0);
        let span = if max_x == min_x { 1.0 } else { max_x - min_x };
        let y_scale = span / (QMAX - QMIN);

        let initial_zp = QMIN - min_x / y_scale;
        let y_zero_point = round_ties_even_f32(initial_zp.clamp(QMIN, QMAX));

        let data: Vec<f32> = x
            .data
            .iter()
            .map(|&v| (round_ties_even_f32(v / y_scale) + y_zero_point).clamp(QMIN, QMAX))
            .collect();

        Ok(vec![
            Tensor::new(data, x.shape.clone()),
            Tensor::rank0(y_scale),
            Tensor::rank0(y_zero_point),
        ])
    }
}
