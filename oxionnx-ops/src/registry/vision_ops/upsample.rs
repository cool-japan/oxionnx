//! `Upsample` operator implementation (deprecated since opset 10).

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::resize::{resize_into, ResizeOptions};

/// ONNX `Upsample` (opsets 7–9, deprecated in favour of `Resize` at opset 10).
///
/// Still present in essentially every opset ≤ 9 detection / segmentation export
/// in the model zoo, which is why it is implemented rather than rejected.
///
/// # Mapping onto `Resize`
///
/// `Upsample` is `Resize` with the **`asymmetric`** coordinate transformation
/// and the **`floor`** nearest rule — *not* `Resize`'s defaults
/// (`half_pixel` / `round_prefer_floor`). The difference is visible on the
/// smallest possible case: doubling `[a, b]` gives `[a, a, b, b]` under
/// `asymmetric` (which is `np.repeat`, what `Upsample` means) but
/// `[a, a, a, b]` under `half_pixel`. Verified against
/// `onnx.reference`'s `Upsample`-9 in `tests/w2_vision_ops_e2e.rs`.
///
/// # Where `scales` lives
///
/// * `Upsample`-7 carries `scales` as a **float-list attribute**.
/// * `Upsample`-9/10 carries it as the **second input**.
///
/// Both are accepted, keyed on which one is actually present rather than on
/// `ctx.opset()` — an exporter that writes the opset-7 form under a later
/// declared opset (or vice versa) is common enough in the wild that presence is
/// the more reliable signal.
///
/// `mode` accepts `nearest` (the default) and `linear`/`bilinear`; ONNX names
/// the 2-D case `linear`, and `bilinear` is normalized to it because older
/// exporters emit that spelling.
pub struct UpsampleOp;

impl UpsampleOp {
    fn run(&self, ctx: &OpContext<'_>, out: &mut Tensor) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;

        // `scales` as an input (opset 9/10) wins over the attribute (opset 7)
        // when both are present; a model carrying both is malformed anyway and
        // the input is the newer, more specific form.
        let scales: Vec<f32> = match ctx.optional_input(1).filter(|t| t.numel() > 0) {
            Some(t) => t.data.clone(),
            None => {
                let attr = ctx.attrs();
                let from_attr = attr
                    .float_lists
                    .get("scales")
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                if from_attr.is_empty() {
                    return Err(OnnxError::InvalidModel(
                        "Upsample: no scales given (expected the `scales` input of \
                         Upsample-9/10 or the `scales` attribute of Upsample-7)"
                            .into(),
                    ));
                }
                from_attr.to_vec()
            }
        };

        if scales.len() != input.ndim() {
            return Err(OnnxError::ShapeMismatch(format!(
                "Upsample: scales has {} entries but the input has rank {}",
                scales.len(),
                input.ndim()
            )));
        }
        for (axis, &s) in scales.iter().enumerate() {
            if !s.is_finite() || s <= 0.0 {
                return Err(OnnxError::InvalidModel(format!(
                    "Upsample: scales[{axis}] must be finite and > 0, got {s}"
                )));
            }
        }

        let raw_mode = ctx.attrs().s("mode");
        let mode = match raw_mode {
            "" | "nearest" => "nearest",
            "linear" | "bilinear" => "linear",
            "cubic" | "bicubic" => "cubic",
            other => {
                return Err(OnnxError::Unsupported(format!(
                    "Upsample: unknown mode '{other}' (expected nearest, linear or cubic)"
                )))
            }
        };

        let opts = ResizeOptions {
            mode,
            coordinate_transformation_mode: "asymmetric",
            nearest_mode: "floor",
            ..ResizeOptions::default()
        };
        resize_into(input, Some(&scales), None, &opts, out)
    }
}

impl Operator for UpsampleOp {
    fn op_type(&self) -> &str {
        "Upsample"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let mut out = Tensor::zeros(&[0]);
        self.run(ctx, &mut out)?;
        Ok(vec![out])
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let [slot] = slots else {
            return Err(OnnxError::Internal(format!(
                "UpsampleOp: expected 1 output slot, got {}",
                slots.len()
            )));
        };
        self.run(ctx, slot)
    }
}
