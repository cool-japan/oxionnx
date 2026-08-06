//! `Conv` kernel — 2-D convolution.  **Platform-neutral** — no `#[cfg]`, no FFI.
//!
//! Same three-step shape as [`crate::kernels::matmul`]: build a [`ConvPlan`] from the
//! operand shapes and the node's `strides` / `pads` / `dilations` / `group` attributes;
//! hand it plus the raw `&[f32]` slices to the backend; check the length, shadow-verify,
//! wrap into a `Tensor` of the plan's output shape.
//!
//! # Conv is DirectML-only — the HLSL engine declines it
//!
//! There is deliberately no Conv HLSL shader (see [`crate::plan::nn::ConvPlan`]): a correct,
//! performant convolution kernel is a wholly different animal from the naive elementwise
//! shaders, so the HLSL backend returns [`crate::DirectMLError::Declined`] for `Conv` and
//! only the genuine DirectML metacommand handles it.  That decline is the *backend's* to
//! make — this kernel is engine-agnostic and calls [`Backend::conv`] exactly as the matmul
//! kernel calls [`Backend::matmul`]; when the HLSL engine is active the call declines and
//! `dispatch::route` falls the node back to the CPU.
//!
//! # `auto_pad` is refused, not guessed
//!
//! ONNX `auto_pad` other than `NOTSET` (`SAME_UPPER`, `SAME_LOWER`, `VALID`) makes the pad
//! amounts implicit, derived from the input and stride at run time.  Rather than reproduce
//! that inference — and risk a plausible, wrong padding — this kernel declines any
//! non-`NOTSET` `auto_pad` to the CPU operator, which implements it.  Only explicit `pads`
//! (the `NOTSET` case, including an absent attribute) reach the GPU.

use oxionnx_core::Tensor;

use crate::backend::Backend;
use crate::error::{DirectMLError, Result};
use crate::kernels::matmul::{check_len, verified};
use crate::plan::ConvPlan;
use crate::reference;

/// ONNX 2-D `Conv`.
///
/// `input` is `[N, C_in, H, W]`, `weight` is `[C_out, C_in/group, kH, kW]`, and `bias`,
/// when present, is `[C_out]`.  `strides` / `pads` / `dilations` are the raw ONNX attribute
/// lists (each empty for the ONNX default), and `group` / `auto_pad` are the raw attribute
/// values.
///
/// # Errors
/// [`DirectMLError::Declined`] when `auto_pad` is not `NOTSET`, the input or weight is not
/// rank 4 (this backend's [`ConvPlan`] only expresses the 2-D case — see its own docs), an
/// attribute list carries a negative entry or an unexpected length, the kernel does not fit
/// the padded input, the tensor is empty, or a size overflows `u32` — each routes to a CPU
/// fallback.
/// [`DirectMLError::ShapeMismatch`] when a channel/group constraint is violated (the CPU
/// operator rejects the same input).  Anything else is a genuine GPU failure.
#[allow(clippy::too_many_arguments)] // Mirrors ONNX `Conv`'s attribute set 1:1, like `dml_gemm`.
pub(crate) fn dml_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: &[i64],
    pads: &[i64],
    dilations: &[i64],
    group: i64,
    auto_pad: &str,
    backend: &Backend,
) -> Result<Tensor> {
    // ONNX spells "explicit pads" as either an absent `auto_pad` or `auto_pad = NOTSET`.
    // Every other value makes padding implicit; we decline rather than infer it.
    if !matches!(auto_pad, "" | "NOTSET") {
        return Err(DirectMLError::Declined(format!(
            "Conv: auto_pad={auto_pad:?} makes padding implicit; declining to the CPU operator, \
             which implements SAME_UPPER/SAME_LOWER/VALID"
        )));
    }

    let strides = to_usizes("Conv strides", strides)?;
    let pads = to_usizes("Conv pads", pads)?;
    let dilations = to_usizes("Conv dilations", dilations)?;
    let group = usize::try_from(group)
        .map_err(|_| DirectMLError::Declined(format!("Conv: negative group {group}")))?;

    let plan = ConvPlan::conv(
        &input.shape,
        &weight.shape,
        bias.map(|t| t.shape.as_slice()),
        &strides,
        &pads,
        &dilations,
        group,
    )?;

    let bias_data = bias.map(|t| t.data.as_slice());
    let gpu = backend.conv(&plan, &input.data, &weight.data, bias_data)?;

    check_len("Conv", gpu.len(), plan.output_elems()?)?;

    if reference::verify_enabled() {
        let comparison = reference::verify_conv(&plan, &input.data, &weight.data, bias_data, &gpu)?;
        verified("Conv", &comparison)?;
    }

    Ok(Tensor::new(gpu, plan.output_shape.clone()))
}

/// Widen an ONNX integer attribute list to `usize`, **declining** any negative entry.
///
/// ONNX `strides` / `pads` / `dilations` are non-negative by definition; a negative value
/// is a malformed node, and mapping it to `0` (or wrapping it via `as usize`) would hand the
/// GPU a plausible, wrong dispatch instead of falling the node back to the CPU.  So a
/// negative entry is a [`DirectMLError::Declined`], and the length itself is left for
/// [`ConvPlan::conv`] to range-check (it accepts the ONNX default of an empty list).
///
/// # Errors
/// [`DirectMLError::Declined`] when any entry is negative.
fn to_usizes(what: &str, values: &[i64]) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .map_err(|_| DirectMLError::Declined(format!("{what}: negative entry {v}")))
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::to_usizes;
    use crate::error::DirectMLError;
    use crate::plan::ConvPlan;

    fn declined(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::Declined(_))
    }

    #[test]
    fn to_usizes_widens_a_valid_list_and_declines_a_negative_one() {
        assert_eq!(to_usizes("strides", &[2, 2]).unwrap(), vec![2, 2]);
        assert_eq!(to_usizes("pads", &[]).unwrap(), Vec::<usize>::new());
        assert!(declined(&to_usizes("dilations", &[-1, 1]).unwrap_err()));
    }

    #[test]
    fn the_default_attribute_lists_produce_the_standard_output_shape() {
        // Empty strides/pads/dilations and group 1 are the ONNX defaults the router passes
        // when a `Conv` node omits them: stride 1, no pad, dilation 1.
        let plan = ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[], &[], &[], 1).unwrap();
        assert_eq!(plan.output_shape, vec![1, 4, 3, 3]);
        assert!(!plan.has_bias);
    }

    #[test]
    fn a_bad_group_or_bad_attribute_length_is_declined() {
        // group 0 (the `usize::try_from` of a negative group also lands here as a decline).
        assert!(declined(
            &ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[], &[], &[], 0).unwrap_err()
        ));
        // strides must be 0 or 2 entries.
        assert!(declined(
            &ConvPlan::conv(&[1, 1, 5, 5], &[4, 1, 3, 3], None, &[1], &[], &[], 1).unwrap_err()
        ));
    }
}
