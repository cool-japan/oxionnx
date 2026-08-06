//! ResizeOp operator implementation.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::resize::{resize_into, ResizeOptions};

// ── Resize ──────────────────────────────────────────────────────────────────

/// Positional input index of the optional `roi` tensor (opset 11+).
const ROI_INPUT: usize = 1;
/// Positional input index of the optional `scales` tensor (opset 11+).
const SCALES_INPUT: usize = 2;
/// Positional input index of the optional `sizes` tensor (opset 11+).
const SIZES_INPUT: usize = 3;

/// Read an optional input, treating an empty tensor as "not supplied".
///
/// ONNX allows a trailing optional input to be passed as an empty tensor when a
/// later one is used, so `numel() == 0` must be treated exactly like absent.
fn non_empty<'a>(ctx: &OpContext<'a>, idx: usize) -> Option<&'a Tensor> {
    ctx.optional_input(idx).filter(|t| t.numel() > 0)
}

/// Convert the `sizes` tensor (stored as f32 in this engine) into dimensions.
fn read_sizes(raw: &[f32]) -> Result<Vec<usize>, OnnxError> {
    raw.iter()
        .map(|&v| {
            if !v.is_finite() || v < 0.0 || v > usize::MAX as f32 {
                Err(OnnxError::InvalidModel(format!(
                    "Resize: sizes entry {v} is not a valid non-negative dimension"
                )))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

/// The three optional tensor inputs of `Resize`, already unpacked.
#[derive(Default)]
struct ResizeInputs {
    scales: Option<Vec<f32>>,
    /// Raw `sizes` values, still to be validated as dimensions.
    sizes: Option<Vec<f32>>,
    roi: Option<Vec<f32>>,
}

/// Resolve the `scales` / `sizes` / `roi` inputs, tolerating the opset-10 layout.
///
/// Resize-10 has only `(X, scales)`; Resize-11 and later insert `roi` at index 1,
/// so a two-input node is ambiguous. It is disambiguated by length: `roi` carries
/// `2 * rank` values whereas `scales` carries `rank`.
fn read_tensor_inputs(ctx: &OpContext<'_>, rank: usize) -> ResizeInputs {
    let roi = non_empty(ctx, ROI_INPUT);
    let scales = non_empty(ctx, SCALES_INPUT);
    let sizes = non_empty(ctx, SIZES_INPUT);
    if scales.is_none() && sizes.is_none() {
        if let Some(candidate) = roi {
            if ctx.inputs.len() <= 2 && candidate.numel() == rank {
                return ResizeInputs {
                    scales: Some(candidate.data.clone()),
                    ..ResizeInputs::default()
                };
            }
        }
    }
    ResizeInputs {
        scales: scales.map(|t| t.data.clone()),
        sizes: sizes.map(|t| t.data.clone()),
        roi: roi.map(|t| t.data.clone()),
    }
}

/// Build the option set for one Resize node from its attributes.
fn read_options<'a>(
    ctx: &'a OpContext<'a>,
    axes: &'a [i64],
    roi: Option<&'a [f32]>,
) -> ResizeOptions<'a> {
    let attrs = ctx.attrs();
    let defaults = ResizeOptions::default();
    let pick = |name: &str, fallback: &'a str| -> &'a str {
        let v = attrs.s(name);
        if v.is_empty() {
            fallback
        } else {
            v
        }
    };
    ResizeOptions {
        mode: pick("mode", defaults.mode),
        coordinate_transformation_mode: pick(
            "coordinate_transformation_mode",
            defaults.coordinate_transformation_mode,
        ),
        nearest_mode: pick("nearest_mode", defaults.nearest_mode),
        keep_aspect_ratio_policy: pick(
            "keep_aspect_ratio_policy",
            defaults.keep_aspect_ratio_policy,
        ),
        cubic_coeff_a: attrs.f("cubic_coeff_a", defaults.cubic_coeff_a),
        extrapolation_value: attrs.f("extrapolation_value", defaults.extrapolation_value),
        exclude_outside: attrs.i("exclude_outside", 0) != 0,
        antialias: attrs.i("antialias", 0) != 0,
        axes: if axes.is_empty() { None } else { Some(axes) },
        roi,
    }
}

/// ONNX `Resize` (opset 19/21 semantics).
pub struct ResizeOp;

impl ResizeOp {
    fn run(&self, ctx: &OpContext<'_>, out: &mut Tensor) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let raw = read_tensor_inputs(ctx, input.ndim());
        let sizes = raw.sizes.as_deref().map(read_sizes).transpose()?;
        let axes: Vec<i64> = ctx.attrs().ints("axes").to_vec();
        let opts = read_options(ctx, &axes, raw.roi.as_deref());
        resize_into(input, raw.scales.as_deref(), sizes.as_deref(), &opts, out)
    }
}

impl Operator for ResizeOp {
    fn op_type(&self) -> &str {
        "Resize"
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
                "ResizeOp: expected 1 output slot, got {}",
                slots.len()
            )));
        };
        self.run(ctx, slot)
    }
}
