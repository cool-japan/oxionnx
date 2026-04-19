//! Typed (F16/BF16) conv kernels for ConvOp and ConvTransposeOp.
//!
//! Computation runs in f32 accumulator for numerical stability. The caller is
//! responsible for computing the output shape before calling these helpers.
//! F32 path is not duplicated here — it delegates to the existing `execute()` logic.

use oxionnx_core::Tensor;

use crate::conv;

// ── Parameter bundles ────────────────────────────────────────────────────────

/// Convolution kernel parameters shared between conv2d and the typed wrappers.
pub(crate) struct Conv2dParams {
    pub strides: [usize; 2],
    pub pads: [usize; 4],
    pub dilations: [usize; 2],
    pub group: usize,
}

/// Fused activation parameters (from optimizer fusion).
pub(crate) struct FusedActivation<'a> {
    pub activation: &'a str,
    pub min: f32,
    pub max: f32,
}

/// ConvTranspose kernel parameters.
pub(crate) struct ConvTranspose2dParams {
    pub strides: [usize; 2],
    pub pads: [usize; 4],
    pub output_padding: [usize; 2],
    pub dilations: [usize; 2],
    pub group: usize,
}

/// Typed input bundle for Conv2d: raw bits + shapes.
pub(crate) struct Conv2dInputs<'a> {
    pub input_bits: &'a [u16],
    pub input_shape: &'a [usize],
    pub weight_bits: &'a [u16],
    pub weight_shape: &'a [usize],
    pub bias_bits: Option<&'a [u16]>,
}

/// Typed input bundle for ConvTranspose2d: raw bits + shapes.
pub(crate) struct ConvTranspose2dInputs<'a> {
    pub input_bits: &'a [u16],
    pub input_shape: &'a [usize],
    pub weight_bits: &'a [u16],
    pub weight_shape: &'a [usize],
    pub bias_bits: Option<&'a [u16]>,
}

// ── ConvOp helpers ──────────────────────────────────────────────────────────

/// Bundled f32 inputs for the inner conv2d computation.
struct Conv2dF32Inputs<'a> {
    input: &'a [f32],
    input_shape: &'a [usize],
    weight: &'a [f32],
    weight_shape: &'a [usize],
    bias: Option<&'a [f32]>,
}

/// Core f32 computation shared by both dtype paths for Conv2d.
///
/// Takes already-promoted f32 slices and runs the existing `conv2d_into` kernel.
/// Fused activation is applied to the output buffer before returning.
fn conv2d_f32_inner(
    f32_inputs: &Conv2dF32Inputs<'_>,
    params: &Conv2dParams,
    act: &FusedActivation<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) {
    let input_tensor = Tensor::new(f32_inputs.input.to_vec(), f32_inputs.input_shape.to_vec());
    let weight_tensor = Tensor::new(f32_inputs.weight.to_vec(), f32_inputs.weight_shape.to_vec());
    let bias_tensor = f32_inputs
        .bias
        .map(|b| Tensor::new(b.to_vec(), vec![f32_inputs.weight_shape[0]]));
    conv::conv2d_into(
        &input_tensor,
        &weight_tensor,
        bias_tensor.as_ref(),
        params.strides,
        params.pads,
        params.dilations,
        params.group,
        out,
        out_shape,
    );
    // Apply fused activation (mirrors ConvOp::execute exactly)
    if act.activation == "relu" {
        for v in out.iter_mut() {
            *v = v.max(0.0);
        }
    } else if act.activation == "clip" {
        for v in out.iter_mut() {
            *v = v.clamp(act.min, act.max);
        }
    }
}

/// Cast f16 bit-packed inputs to f32, run conv2d, cast output back to f16 bits.
fn conv2d_half_inner<F, G>(
    inputs: &Conv2dInputs<'_>,
    from_bits: F,
    to_bits: G,
    params: &Conv2dParams,
    act: &FusedActivation<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) where
    F: Fn(u16) -> f32,
    G: Fn(f32) -> u16,
{
    let input_f32: Vec<f32> = inputs.input_bits.iter().map(|&b| from_bits(b)).collect();
    let weight_f32: Vec<f32> = inputs.weight_bits.iter().map(|&b| from_bits(b)).collect();
    let bias_f32: Option<Vec<f32>> = inputs
        .bias_bits
        .map(|bs| bs.iter().map(|&b| from_bits(b)).collect());

    let out_len = out_shape.iter().product();
    let mut buf = vec![0.0_f32; out_len];
    let f32_inputs = Conv2dF32Inputs {
        input: &input_f32,
        input_shape: inputs.input_shape,
        weight: &weight_f32,
        weight_shape: inputs.weight_shape,
        bias: bias_f32.as_deref(),
    };
    conv2d_f32_inner(&f32_inputs, params, act, &mut buf, out_shape);
    for (ob, &val) in out_bits.iter_mut().zip(buf.iter()) {
        *ob = to_bits(val);
    }
}

/// F16 Conv2D kernel.
///
/// Inputs/weights are `Vec<u16>` f16 raw bits; computation in f32; output back
/// to f16 raw bits.
pub(crate) fn conv2d_f16(
    inputs: &Conv2dInputs<'_>,
    params: &Conv2dParams,
    act: &FusedActivation<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) {
    conv2d_half_inner(
        inputs,
        |b| half::f16::from_bits(b).to_f32(),
        |v| half::f16::from_f32(v).to_bits(),
        params,
        act,
        out_bits,
        out_shape,
    );
}

/// BF16 Conv2D kernel.
///
/// Inputs/weights are `Vec<u16>` bf16 raw bits; computation in f32; output back
/// to bf16 raw bits.
pub(crate) fn conv2d_bf16(
    inputs: &Conv2dInputs<'_>,
    params: &Conv2dParams,
    act: &FusedActivation<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) {
    conv2d_half_inner(
        inputs,
        |b| half::bf16::from_bits(b).to_f32(),
        |v| half::bf16::from_f32(v).to_bits(),
        params,
        act,
        out_bits,
        out_shape,
    );
}

// ── ConvTransposeOp helpers ─────────────────────────────────────────────────

/// Bundled f32 inputs for the inner conv_transpose2d computation.
struct ConvTranspose2dF32Inputs<'a> {
    input: &'a [f32],
    input_shape: &'a [usize],
    weight: &'a [f32],
    weight_shape: &'a [usize],
    bias: Option<&'a [f32]>,
}

/// Core f32 computation shared by both dtype paths for ConvTranspose2d.
///
/// Takes already-promoted f32 slices and runs the existing `conv_transpose2d_into`
/// kernel. ConvTransposeOp has no fused activation.
fn conv_transpose2d_f32_inner(
    f32_inputs: &ConvTranspose2dF32Inputs<'_>,
    params: &ConvTranspose2dParams,
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<(), String> {
    let input_tensor = Tensor::new(f32_inputs.input.to_vec(), f32_inputs.input_shape.to_vec());
    let weight_tensor = Tensor::new(f32_inputs.weight.to_vec(), f32_inputs.weight_shape.to_vec());
    let bias_tensor = f32_inputs.bias.map(|b| {
        let c_out = f32_inputs.weight_shape[1] * params.group;
        Tensor::new(b.to_vec(), vec![c_out])
    });
    conv::conv_transpose2d_into(
        &input_tensor,
        &weight_tensor,
        bias_tensor.as_ref(),
        &params.strides,
        &params.pads,
        &params.output_padding,
        &params.dilations,
        params.group,
        out,
        out_shape,
    )
}

/// Cast half-precision inputs to f32, run ConvTranspose2d, cast output back.
fn conv_transpose2d_half_inner<F, G>(
    inputs: &ConvTranspose2dInputs<'_>,
    from_bits: F,
    to_bits: G,
    params: &ConvTranspose2dParams,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), String>
where
    F: Fn(u16) -> f32,
    G: Fn(f32) -> u16,
{
    let input_f32: Vec<f32> = inputs.input_bits.iter().map(|&b| from_bits(b)).collect();
    let weight_f32: Vec<f32> = inputs.weight_bits.iter().map(|&b| from_bits(b)).collect();
    let bias_f32: Option<Vec<f32>> = inputs
        .bias_bits
        .map(|bs| bs.iter().map(|&b| from_bits(b)).collect());

    let out_len = out_shape.iter().product();
    let mut buf = vec![0.0_f32; out_len];
    let f32_inputs = ConvTranspose2dF32Inputs {
        input: &input_f32,
        input_shape: inputs.input_shape,
        weight: &weight_f32,
        weight_shape: inputs.weight_shape,
        bias: bias_f32.as_deref(),
    };
    conv_transpose2d_f32_inner(&f32_inputs, params, &mut buf, out_shape)?;
    for (ob, &val) in out_bits.iter_mut().zip(buf.iter()) {
        *ob = to_bits(val);
    }
    Ok(())
}

/// F16 ConvTranspose2D kernel.
pub(crate) fn conv_transpose2d_f16(
    inputs: &ConvTranspose2dInputs<'_>,
    params: &ConvTranspose2dParams,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), String> {
    conv_transpose2d_half_inner(
        inputs,
        |b| half::f16::from_bits(b).to_f32(),
        |v| half::f16::from_f32(v).to_bits(),
        params,
        out_bits,
        out_shape,
    )
}

/// BF16 ConvTranspose2D kernel.
pub(crate) fn conv_transpose2d_bf16(
    inputs: &ConvTranspose2dInputs<'_>,
    params: &ConvTranspose2dParams,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), String> {
    conv_transpose2d_half_inner(
        inputs,
        |b| half::bf16::from_bits(b).to_f32(),
        |v| half::bf16::from_f32(v).to_bits(),
        params,
        out_bits,
        out_shape,
    )
}
