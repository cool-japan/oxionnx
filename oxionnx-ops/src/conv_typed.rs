//! Typed (F16/BF16) conv kernels for ConvOp and ConvTransposeOp.
//!
//! Computation runs in an f32 accumulator for numerical stability, through the
//! same rank-generic kernels the f32 path uses — so 1D / 2D / 3D convolutions
//! and transposed convolutions are all supported in half precision too. The
//! caller is responsible for computing the output shape before calling these
//! helpers. The F32 path is not duplicated here — it delegates to the existing
//! `execute()` logic.

use oxionnx_core::OnnxError;

use crate::conv::{self, ConvParams, ConvTransposeParams};

// ── Parameter bundles ────────────────────────────────────────────────────────

/// Fused activation parameters (from optimizer fusion).
pub(crate) struct FusedActivation<'a> {
    pub activation: &'a str,
    pub min: f32,
    pub max: f32,
}

/// Typed input bundle for Conv / ConvTranspose: raw bits + shapes.
pub(crate) struct ConvInputs<'a> {
    pub input_bits: &'a [u16],
    pub input_shape: &'a [usize],
    pub weight_bits: &'a [u16],
    pub weight_shape: &'a [usize],
    pub bias_bits: Option<&'a [u16]>,
}

/// Promote half-precision bits to f32 and split into the pieces the kernels take.
struct Promoted {
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Promoted {
    fn new<F: Fn(u16) -> f32>(inputs: &ConvInputs<'_>, from_bits: F) -> Self {
        Self {
            input: inputs.input_bits.iter().map(|&b| from_bits(b)).collect(),
            weight: inputs.weight_bits.iter().map(|&b| from_bits(b)).collect(),
            bias: inputs
                .bias_bits
                .map(|bs| bs.iter().map(|&b| from_bits(b)).collect()),
        }
    }
}

// ── ConvOp helpers ──────────────────────────────────────────────────────────

/// Cast half-precision inputs to f32, run the N-D convolution, cast back.
fn conv_half_inner<F, G>(
    inputs: &ConvInputs<'_>,
    from_bits: F,
    to_bits: G,
    params: &ConvParams<'_>,
    act: &FusedActivation<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), OnnxError>
where
    F: Fn(u16) -> f32,
    G: Fn(f32) -> u16,
{
    let promoted = Promoted::new(inputs, from_bits);
    let out_len = out_shape.iter().product();
    let mut buf = vec![0.0_f32; out_len];
    conv::conv_into(
        &promoted.input,
        inputs.input_shape,
        &promoted.weight,
        inputs.weight_shape,
        promoted.bias.as_deref(),
        params,
        &mut buf,
        out_shape,
    )?;
    // Apply fused activation (mirrors ConvOp::execute exactly)
    if act.activation == "relu" {
        for v in buf.iter_mut() {
            *v = v.max(0.0);
        }
    } else if act.activation == "clip" {
        // Same NaN / inverted-bound handling as `apply_fused_activation`:
        // `f32::clamp` asserts `min <= max`, which a hand-authored node can
        // violate.
        let lo = if act.min.is_nan() {
            f32::NEG_INFINITY
        } else {
            act.min
        };
        let hi = if act.max.is_nan() {
            f32::INFINITY
        } else {
            act.max
        };
        if lo <= hi {
            for v in buf.iter_mut() {
                *v = v.clamp(lo, hi);
            }
        }
    }
    for (ob, &val) in out_bits.iter_mut().zip(buf.iter()) {
        *ob = to_bits(val);
    }
    Ok(())
}

/// F16 N-D convolution kernel.
///
/// Inputs/weights are `[u16]` f16 raw bits; computation in f32; output back to
/// f16 raw bits.
pub(crate) fn conv_f16(
    inputs: &ConvInputs<'_>,
    params: &ConvParams<'_>,
    act: &FusedActivation<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), OnnxError> {
    conv_half_inner(
        inputs,
        |b| half::f16::from_bits(b).to_f32(),
        |v| half::f16::from_f32(v).to_bits(),
        params,
        act,
        out_bits,
        out_shape,
    )
}

/// BF16 N-D convolution kernel.
///
/// Inputs/weights are `[u16]` bf16 raw bits; computation in f32; output back
/// to bf16 raw bits.
pub(crate) fn conv_bf16(
    inputs: &ConvInputs<'_>,
    params: &ConvParams<'_>,
    act: &FusedActivation<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), OnnxError> {
    conv_half_inner(
        inputs,
        |b| half::bf16::from_bits(b).to_f32(),
        |v| half::bf16::from_f32(v).to_bits(),
        params,
        act,
        out_bits,
        out_shape,
    )
}

// ── ConvTransposeOp helpers ─────────────────────────────────────────────────

/// Cast half-precision inputs to f32, run the N-D transposed convolution,
/// cast back. ConvTranspose has no fused activation.
fn conv_transpose_half_inner<F, G>(
    inputs: &ConvInputs<'_>,
    from_bits: F,
    to_bits: G,
    params: &ConvTransposeParams<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), OnnxError>
where
    F: Fn(u16) -> f32,
    G: Fn(f32) -> u16,
{
    let promoted = Promoted::new(inputs, from_bits);
    let out_len = out_shape.iter().product();
    let mut buf = vec![0.0_f32; out_len];
    conv::conv_transpose_into(
        &promoted.input,
        inputs.input_shape,
        &promoted.weight,
        inputs.weight_shape,
        promoted.bias.as_deref(),
        params,
        &mut buf,
        out_shape,
    )?;
    for (ob, &val) in out_bits.iter_mut().zip(buf.iter()) {
        *ob = to_bits(val);
    }
    Ok(())
}

/// F16 N-D transposed convolution kernel.
pub(crate) fn conv_transpose_f16(
    inputs: &ConvInputs<'_>,
    params: &ConvTransposeParams<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), OnnxError> {
    conv_transpose_half_inner(
        inputs,
        |b| half::f16::from_bits(b).to_f32(),
        |v| half::f16::from_f32(v).to_bits(),
        params,
        out_bits,
        out_shape,
    )
}

/// BF16 N-D transposed convolution kernel.
pub(crate) fn conv_transpose_bf16(
    inputs: &ConvInputs<'_>,
    params: &ConvTransposeParams<'_>,
    out_bits: &mut [u16],
    out_shape: &[usize],
) -> Result<(), OnnxError> {
    conv_transpose_half_inner(
        inputs,
        |b| half::bf16::from_bits(b).to_f32(),
        |v| half::bf16::from_f32(v).to_bits(),
        params,
        out_bits,
        out_shape,
    )
}
