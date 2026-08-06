//! ConvOp and ConvTransposeOp operator implementations.
//!
//! Both operators are rank-generic: they accept `[N, C, d_0, …, d_{r-1}]` for
//! any spatial rank `r >= 1` (Conv1D for audio/TCN models, Conv2D for vision,
//! Conv3D for video/volumetric), resolve `auto_pad` at every rank and dispatch
//! to `crate::conv`, which keeps a specialised rank-2 kernel, lowers rank 1
//! onto it and runs a generic im2col + GEMM for rank ≥ 3.
//!
//! The shared spatial-attribute helpers — `auto_pad` resolution, stride /
//! dilation / pad validation and the output-extent formulas — live in
//! [`crate::conv::spatial`] so the kernels, the operator wrappers and the
//! engine's shape-inference pass resolve the ONNX geometry through exactly one
//! implementation.

use oxionnx_core::{Attributes, OnnxError, OpContext, Operator, Tensor};

use crate::conv;
use crate::conv::spatial::{
    self, parse_auto_pad, read_group, read_nonneg_spatial, read_pads, read_positive_spatial,
    resolve_pads, spatial_rank, AutoPad,
};

// ── Conv ────────────────────────────────────────────────────────────────────

/// Fully resolved Conv geometry: attributes validated and `auto_pad` applied.
struct ConvGeometry {
    strides: Vec<usize>,
    /// `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]`.
    pads: Vec<usize>,
    dilations: Vec<usize>,
    group: usize,
    /// Full `[N, F, o_0, …]` output shape.
    out_shape: Vec<usize>,
}

impl ConvGeometry {
    /// Read + validate every spatial attribute of a `Conv` node.
    ///
    /// `weight_shape` supplies the kernel extents used by `auto_pad`.
    fn from_attrs(
        attrs: &Attributes,
        input_shape: &[usize],
        weight_shape: &[usize],
    ) -> Result<Self, OnnxError> {
        let rank = spatial_rank(input_shape, "Conv", "input")?;
        if weight_shape.len() != input_shape.len() {
            return Err(OnnxError::ShapeMismatch(format!(
                "Conv: weight rank {} must equal input rank {} ([F, C/group, k_0, ...])",
                weight_shape.len(),
                input_shape.len()
            )));
        }
        let strides = read_positive_spatial(attrs.ints("strides"), rank, 1, "strides", "Conv")?;
        let dilations =
            read_positive_spatial(attrs.ints("dilations"), rank, 1, "dilations", "Conv")?;
        let group = read_group(attrs, "Conv")?;
        // `kernel_shape` is redundant for Conv (W carries the extents) but a
        // model may still declare it; a disagreement would make `auto_pad`
        // derive a padding the kernel does not use.
        let kernel_attr = attrs.ints("kernel_shape");
        if !kernel_attr.is_empty() {
            let declared = spatial::read_kernel_shape(kernel_attr, rank, "Conv")?;
            if declared != weight_shape[2..] {
                return Err(OnnxError::ShapeMismatch(format!(
                    "Conv: kernel_shape {declared:?} disagrees with weight spatial dims {:?}",
                    &weight_shape[2..]
                )));
            }
        }
        // Per the spec W is [M, C/group, k_0, ...]; a mismatch would make the
        // grouped im2col read past the end of the input buffer.
        let c_in = input_shape[1];
        let c_per_group = weight_shape[1];
        if c_per_group.checked_mul(group) != Some(c_in) {
            return Err(OnnxError::ShapeMismatch(format!(
                "Conv: input channels {c_in} != weight input channels {c_per_group} * group {group}"
            )));
        }
        if weight_shape[0] % group != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Conv: output channels {} not divisible by group {group}",
                weight_shape[0]
            )));
        }
        let explicit = read_pads(attrs.ints("pads"), rank, "Conv")?;
        let auto_pad = parse_auto_pad(attrs.s("auto_pad"), "Conv")?;
        let pads = resolve_pads(
            auto_pad,
            &input_shape[2..],
            &weight_shape[2..],
            &strides,
            &dilations,
            &explicit,
        );
        let out_shape = spatial::compute_conv_out_shape(
            "Conv",
            input_shape,
            weight_shape,
            &strides,
            &pads,
            &dilations,
        )?;
        Ok(Self {
            strides,
            pads,
            dilations,
            group,
            out_shape,
        })
    }

    /// Borrowed kernel parameters for `crate::conv`.
    fn params(&self) -> conv::ConvParams<'_> {
        conv::ConvParams {
            strides: &self.strides,
            pads: &self.pads,
            dilations: &self.dilations,
            group: self.group,
        }
    }
}

/// Apply the optimizer's fused activation, if any, in place.
fn apply_fused_activation(attrs: &Attributes, data: &mut [f32]) {
    let activation = attrs.s("activation");
    if activation == "relu" {
        for v in data.iter_mut() {
            *v = v.max(0.0);
        }
    } else if activation == "clip" {
        let min_val = attrs.f("activation_min", f32::NEG_INFINITY);
        let max_val = attrs.f("activation_max", f32::INFINITY);
        // `f32::clamp` has a real (non-debug) `assert!(min <= max)` and treats
        // either bound being NaN as failing that assert too — both reachable
        // from a hand-authored Conv node's fused-activation attributes (the
        // optimizer itself never emits such attributes). Match ONNX Clip
        // semantics instead: a NaN bound is unbounded on that side, and an
        // otherwise-inverted [min, max] passes the data through unclamped
        // rather than panicking.
        let lo = if min_val.is_nan() {
            f32::NEG_INFINITY
        } else {
            min_val
        };
        let hi = if max_val.is_nan() {
            f32::INFINITY
        } else {
            max_val
        };
        if lo <= hi {
            for v in data.iter_mut() {
                *v = v.clamp(lo, hi);
            }
        }
    }
}

pub struct ConvOp;
impl Operator for ConvOp {
    fn op_type(&self) -> &str {
        "Conv"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let weight = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let geo = ConvGeometry::from_attrs(attrs, &input.shape, &weight.shape)?;

        let out_len: usize = geo.out_shape.iter().product();
        let mut data = vec![0.0_f32; out_len];
        conv::conv_into(
            &input.data,
            &input.shape,
            &weight.data,
            &weight.shape,
            bias.map(|b| b.data.as_slice()),
            &geo.params(),
            &mut data,
            &geo.out_shape,
        )?;

        // Apply fused activation if set by the optimizer
        apply_fused_activation(attrs, &mut data);

        Ok(vec![Tensor::new(data, geo.out_shape)])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &oxionnx_core::OpContext<'_>,
        slots: &mut [oxionnx_core::Tensor],
    ) -> Result<(), oxionnx_core::OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal(
                "ConvOp: expected at least 1 output slot, got 0".into(),
            ));
        }
        let input = ctx.input(0)?;
        let weight = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let geo = ConvGeometry::from_attrs(attrs, &input.shape, &weight.shape)?;

        let out_len: usize = geo.out_shape.iter().product();
        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        slots[0].shape.clone_from(&geo.out_shape);
        conv::conv_into(
            &input.data,
            &input.shape,
            &weight.data,
            &weight.shape,
            bias.map(|b| b.data.as_slice()),
            &geo.params(),
            &mut slots[0].data,
            &geo.out_shape,
        )?;

        // Apply fused activation in-place — mirrors execute() exactly.
        apply_fused_activation(attrs, &mut slots[0].data);

        Ok(())
    }

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, TypedTensor};

        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("ConvOp: missing input".into()))?;
        let weight = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("ConvOp: missing weight".into()))?;
        let bias = ctx.input(2);

        let geo = ConvGeometry::from_attrs(ctx.attrs(), &input.shape, &weight.shape)?;

        let act = crate::conv_typed::FusedActivation {
            activation: ctx.attrs().s("activation"),
            min: ctx.attrs().f("activation_min", f32::NEG_INFINITY),
            max: ctx.attrs().f("activation_max", f32::INFINITY),
        };

        let out_shape = geo.out_shape.clone();
        let out_len: usize = out_shape.iter().product();
        let params = geo.params();

        match (&input.storage, &weight.storage) {
            // ── F32: delegate to existing execute() logic ──
            (TensorStorage::F32(_), TensorStorage::F32(_)) => {
                oxionnx_core::default_typed_via_f32(self, ctx)
            }

            // ── F16 × F16 → F16 ──
            (TensorStorage::F16(ib), TensorStorage::F16(wb)) => {
                let bias_bits = if let Some(b) = bias {
                    match &b.storage {
                        TensorStorage::F16(bb) => Some(bb.as_slice()),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let inputs = crate::conv_typed::ConvInputs {
                    input_bits: ib,
                    input_shape: &input.shape,
                    weight_bits: wb,
                    weight_shape: &weight.shape,
                    bias_bits,
                };
                let mut out_bits = vec![0u16; out_len];
                crate::conv_typed::conv_f16(&inputs, &params, &act, &mut out_bits, &out_shape)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::F16(out_bits),
                    out_shape,
                )])
            }

            // ── BF16 × BF16 → BF16 ──
            (TensorStorage::BF16(ib), TensorStorage::BF16(wb)) => {
                let bias_bits = if let Some(b) = bias {
                    match &b.storage {
                        TensorStorage::BF16(bb) => Some(bb.as_slice()),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let inputs = crate::conv_typed::ConvInputs {
                    input_bits: ib,
                    input_shape: &input.shape,
                    weight_bits: wb,
                    weight_shape: &weight.shape,
                    bias_bits,
                };
                let mut out_bits = vec![0u16; out_len];
                crate::conv_typed::conv_bf16(&inputs, &params, &act, &mut out_bits, &out_shape)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::BF16(out_bits),
                    out_shape,
                )])
            }

            // ── Mixed / unsupported dtypes: fall back to f32 round-trip ──
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }
}

// ── ConvTranspose ───────────────────────────────────────────────────────────

/// Fully resolved ConvTranspose geometry.
///
/// Handles the three-way interaction the ONNX spec defines between `pads`,
/// `auto_pad` and `output_shape`:
///
/// * `output_shape` (when present) wins — the padding is *derived* from it as
///   `total = stride * (in - 1) + output_padding + ((k - 1) * dilation + 1) - out`,
///   split with the odd pixel at the end for `SAME_UPPER` and at the beginning
///   otherwise (ONNX Runtime parity).
/// * `auto_pad = SAME_UPPER/SAME_LOWER` without `output_shape` targets
///   `out = in * stride` and derives the padding the same way.
/// * `auto_pad = VALID` forces zero padding; `NOTSET` uses `pads` verbatim.
struct ConvTransposeGeometry {
    strides: Vec<usize>,
    pads: Vec<usize>,
    dilations: Vec<usize>,
    group: usize,
    /// Full `[N, C_out, o_0, …]` output shape.
    out_shape: Vec<usize>,
}

impl ConvTransposeGeometry {
    fn from_attrs(
        attrs: &Attributes,
        input_shape: &[usize],
        weight_shape: &[usize],
    ) -> Result<Self, OnnxError> {
        let rank = spatial_rank(input_shape, "ConvTranspose", "input")?;
        if weight_shape.len() != input_shape.len() {
            return Err(OnnxError::ShapeMismatch(format!(
                "ConvTranspose: weight rank {} must equal input rank {} \
                 ([C_in, C_out/group, k_0, ...])",
                weight_shape.len(),
                input_shape.len()
            )));
        }
        let strides =
            read_positive_spatial(attrs.ints("strides"), rank, 1, "strides", "ConvTranspose")?;
        let dilations = read_positive_spatial(
            attrs.ints("dilations"),
            rank,
            1,
            "dilations",
            "ConvTranspose",
        )?;
        let group = read_group(attrs, "ConvTranspose")?;
        let kernel_attr = attrs.ints("kernel_shape");
        if !kernel_attr.is_empty() {
            let declared = spatial::read_kernel_shape(kernel_attr, rank, "ConvTranspose")?;
            if declared != weight_shape[2..] {
                return Err(OnnxError::ShapeMismatch(format!(
                    "ConvTranspose: kernel_shape {declared:?} disagrees with weight spatial \
                     dims {:?}",
                    &weight_shape[2..]
                )));
            }
        }
        // Per the spec W is [C_in, C_out/group, k_0, ...]; a mismatch would
        // index past the end of the weight buffer during the scatter-accumulate.
        if input_shape[1] != weight_shape[0] {
            return Err(OnnxError::ShapeMismatch(format!(
                "ConvTranspose: input channels {} != weight input channels {}",
                input_shape[1], weight_shape[0]
            )));
        }
        if input_shape[1] % group != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "ConvTranspose: input channels {} not divisible by group {group}",
                input_shape[1]
            )));
        }
        let explicit = read_pads(attrs.ints("pads"), rank, "ConvTranspose")?;
        let auto_pad = parse_auto_pad(attrs.s("auto_pad"), "ConvTranspose")?;
        let output_padding = read_nonneg_spatial(
            attrs.ints("output_padding"),
            rank,
            "output_padding",
            "ConvTranspose",
        )?;
        let requested_out = read_output_shape_attr(attrs.ints("output_shape"), rank)?;

        let input_spatial = &input_shape[2..];
        let kernel = &weight_shape[2..];

        // Target spatial extent, when one is dictated by output_shape/auto_pad.
        let target: Option<Vec<usize>> = match (requested_out, auto_pad) {
            (Some(shape), _) => Some(shape),
            (None, AutoPad::SameUpper | AutoPad::SameLower) => Some(
                (0..rank)
                    .map(|axis| input_spatial[axis].saturating_mul(strides[axis]))
                    .collect(),
            ),
            (None, _) => None,
        };

        let pads = match target {
            None => {
                if auto_pad == AutoPad::Valid {
                    vec![0_usize; 2 * rank]
                } else {
                    explicit
                }
            }
            Some(target) => {
                let mut derived = vec![0_usize; 2 * rank];
                for axis in 0..rank {
                    let (begin, end) = derive_transpose_pads(
                        axis,
                        rank,
                        input_spatial[axis],
                        target[axis],
                        kernel[axis],
                        strides[axis],
                        dilations[axis],
                        output_padding[axis],
                        auto_pad == AutoPad::SameUpper,
                    )?;
                    derived[axis] = begin;
                    derived[axis + rank] = end;
                }
                derived
            }
        };

        let out_shape = spatial::compute_conv_transpose_out_shape(
            "ConvTranspose",
            input_shape,
            weight_shape,
            &strides,
            &pads,
            &output_padding,
            &dilations,
            group,
        )?;

        Ok(Self {
            strides,
            pads,
            dilations,
            group,
            out_shape,
        })
    }

    /// Borrowed kernel parameters for `crate::conv`.
    ///
    /// `output_padding` is already folded into `out_shape` by
    /// `compute_conv_transpose_out_shape`, so the kernel does not need it.
    fn params(&self) -> conv::ConvTransposeParams<'_> {
        conv::ConvTransposeParams {
            strides: &self.strides,
            pads: &self.pads,
            dilations: &self.dilations,
            group: self.group,
        }
    }
}

/// Read the optional `output_shape` attribute as the `rank` spatial extents.
///
/// Exporters emit either the spatial dims alone or the full `[N, C, o_0, …]`;
/// both forms are accepted, matching ONNX Runtime.
fn read_output_shape_attr(values: &[i64], rank: usize) -> Result<Option<Vec<usize>>, OnnxError> {
    let spatial: &[i64] = if values.is_empty() {
        return Ok(None);
    } else if values.len() == rank {
        values
    } else if values.len() == rank + 2 {
        &values[2..]
    } else {
        return Err(OnnxError::ShapeMismatch(format!(
            "ConvTranspose: output_shape must have {rank} (spatial) or {} (full) entries, got {}",
            rank + 2,
            values.len()
        )));
    };
    let mut out = vec![0_usize; rank];
    for (axis, slot) in out.iter_mut().enumerate() {
        let v = spatial[axis];
        if v < 1 {
            return Err(OnnxError::ShapeMismatch(format!(
                "ConvTranspose: output_shape spatial dim {axis} must be >= 1, got {v}"
            )));
        }
        *slot = v as usize;
    }
    Ok(Some(out))
}

/// Derive `(pad_begin, pad_end)` for one axis from a requested output extent.
#[allow(clippy::too_many_arguments)]
fn derive_transpose_pads(
    axis: usize,
    rank: usize,
    in_dim: usize,
    out_dim: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    output_padding: usize,
    same_upper: bool,
) -> Result<(usize, usize), OnnxError> {
    let label = spatial::axis_label(rank, axis);
    let natural = spatial::conv_transpose_natural_dim(
        "ConvTranspose",
        label,
        in_dim,
        stride,
        output_padding,
        kernel,
        dilation,
    )?;
    let total = natural.checked_sub(out_dim).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!(
            "ConvTranspose: requested output extent {out_dim} on spatial axis {axis} exceeds \
             the un-cropped extent {natural}"
        ))
    })?;
    let half = total / 2;
    // SAME_UPPER keeps the smaller half at the start (extra pixel cropped from
    // the end); every other mode crops the extra pixel from the start.
    if same_upper {
        Ok((half, total - half))
    } else {
        Ok((total - half, half))
    }
}

pub struct ConvTransposeOp;
impl Operator for ConvTransposeOp {
    fn op_type(&self) -> &str {
        "ConvTranspose"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let weight = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let geo = ConvTransposeGeometry::from_attrs(ctx.attrs(), &input.shape, &weight.shape)?;
        let out_len: usize = geo.out_shape.iter().product();
        let mut data = vec![0.0_f32; out_len];
        conv::conv_transpose_into(
            &input.data,
            &input.shape,
            &weight.data,
            &weight.shape,
            bias.map(|b| b.data.as_slice()),
            &geo.params(),
            &mut data,
            &geo.out_shape,
        )?;
        Ok(vec![Tensor::new(data, geo.out_shape)])
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
            return Err(OnnxError::Internal(
                "ConvTransposeOp: expected at least 1 output slot, got 0".into(),
            ));
        }
        let input = ctx.input(0)?;
        let weight = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let geo = ConvTransposeGeometry::from_attrs(ctx.attrs(), &input.shape, &weight.shape)?;

        let out_len: usize = geo.out_shape.iter().product();
        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        slots[0].shape.clone_from(&geo.out_shape);
        conv::conv_transpose_into(
            &input.data,
            &input.shape,
            &weight.data,
            &weight.shape,
            bias.map(|b| b.data.as_slice()),
            &geo.params(),
            &mut slots[0].data,
            &geo.out_shape,
        )?;

        // ConvTransposeOp has no fused activation.
        Ok(())
    }

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, TypedTensor};

        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("ConvTransposeOp: missing input".into()))?;
        let weight = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("ConvTransposeOp: missing weight".into()))?;
        let bias = ctx.input(2);

        let geo = ConvTransposeGeometry::from_attrs(ctx.attrs(), &input.shape, &weight.shape)?;
        let out_shape = geo.out_shape.clone();
        let out_len: usize = out_shape.iter().product();
        let params = geo.params();

        match (&input.storage, &weight.storage) {
            // ── F32: delegate to existing execute() logic ──
            (TensorStorage::F32(_), TensorStorage::F32(_)) => {
                oxionnx_core::default_typed_via_f32(self, ctx)
            }

            // ── F16 × F16 → F16 ──
            (TensorStorage::F16(ib), TensorStorage::F16(wb)) => {
                let bias_bits = if let Some(b) = bias {
                    match &b.storage {
                        TensorStorage::F16(bb) => Some(bb.as_slice()),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let inputs = crate::conv_typed::ConvInputs {
                    input_bits: ib,
                    input_shape: &input.shape,
                    weight_bits: wb,
                    weight_shape: &weight.shape,
                    bias_bits,
                };
                let mut out_bits = vec![0u16; out_len];
                crate::conv_typed::conv_transpose_f16(&inputs, &params, &mut out_bits, &out_shape)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::F16(out_bits),
                    out_shape,
                )])
            }

            // ── BF16 × BF16 → BF16 ──
            (TensorStorage::BF16(ib), TensorStorage::BF16(wb)) => {
                let bias_bits = if let Some(b) = bias {
                    match &b.storage {
                        TensorStorage::BF16(bb) => Some(bb.as_slice()),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let inputs = crate::conv_typed::ConvInputs {
                    input_bits: ib,
                    input_shape: &input.shape,
                    weight_bits: wb,
                    weight_shape: &weight.shape,
                    bias_bits,
                };
                let mut out_bits = vec![0u16; out_len];
                crate::conv_typed::conv_transpose_bf16(
                    &inputs,
                    &params,
                    &mut out_bits,
                    &out_shape,
                )?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::BF16(out_bits),
                    out_shape,
                )])
            }

            // ── Mixed / unsupported dtypes: fall back to f32 round-trip ──
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }
}
