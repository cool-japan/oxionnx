//! Shared N-D spatial geometry for `Conv`, `ConvTranspose`, `MaxPool` and
//! `AveragePool`.
//!
//! Every ONNX spatial operator resolves `auto_pad`, validates
//! `strides` / `dilations` / `pads` / `group` and derives its output extents
//! through the helpers here, so the kernels, the operator wrappers and the
//! engine's shape-inference pass can never disagree about the geometry.
//!
//! ## Conventions
//!
//! * A spatial tensor is `[N, C, d_0, …, d_{r-1}]`; `r` is its *spatial rank*
//!   (1 for `Conv1D`, 2 for `Conv2D`, 3 for `Conv3D`, …).
//! * `pads` always follows the ONNX layout
//!   `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]` (length `2 * r`).
//!   For `r == 2` that is exactly the `[top, left, bottom, right]` array the
//!   2D kernels take, so no reordering happens on the common path.
//! * `strides`, `dilations` and `output_padding` have one entry per spatial
//!   axis.

use oxionnx_core::{Attributes, OnnxError};

/// ONNX `auto_pad` padding mode (Conv, ConvTranspose, MaxPool, AveragePool).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AutoPad {
    /// Explicit `pads` attribute is authoritative.
    NotSet,
    /// Pad so `out = ceil(in / stride)`; odd padding goes to the end.
    SameUpper,
    /// Pad so `out = ceil(in / stride)`; odd padding goes to the beginning.
    SameLower,
    /// No padding at all.
    Valid,
}

/// Parse the `auto_pad` string attribute, rejecting unknown values.
pub(crate) fn parse_auto_pad(raw: &str, op: &str) -> Result<AutoPad, OnnxError> {
    match raw {
        "" | "NOTSET" => Ok(AutoPad::NotSet),
        "SAME_UPPER" => Ok(AutoPad::SameUpper),
        "SAME_LOWER" => Ok(AutoPad::SameLower),
        "VALID" => Ok(AutoPad::Valid),
        other => Err(OnnxError::Unsupported(format!(
            "{op}: unknown auto_pad value '{other}' \
             (expected NOTSET, SAME_UPPER, SAME_LOWER or VALID)"
        ))),
    }
}

/// Human-readable axis label used in error messages.
///
/// Rank 1/2/3 get the conventional `W` / `H,W` / `D,H,W` names (the 2D labels
/// match the pre-N-D messages verbatim); higher ranks fall back to the index.
pub(crate) fn axis_label(rank: usize, axis: usize) -> &'static str {
    const R1: [&str; 1] = ["W"];
    const R2: [&str; 2] = ["H", "W"];
    const R3: [&str; 3] = ["D", "H", "W"];
    const IDX: [&str; 8] = ["0", "1", "2", "3", "4", "5", "6", "7"];
    match rank {
        1 => R1.get(axis).copied().unwrap_or("?"),
        2 => R2.get(axis).copied().unwrap_or("?"),
        3 => R3.get(axis).copied().unwrap_or("?"),
        _ => IDX.get(axis).copied().unwrap_or("?"),
    }
}

/// Spatial rank of a `[N, C, d_0, …]` tensor, rejecting rank < 3.
pub(crate) fn spatial_rank(shape: &[usize], op: &str, what: &str) -> Result<usize, OnnxError> {
    if shape.len() < 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: {what} must have rank >= 3 ([N, C, d_0, ...]), got rank {}",
            shape.len()
        )));
    }
    Ok(shape.len() - 2)
}

/// Read a per-spatial-axis attribute (`strides`, `dilations`) whose entries
/// must be `>= 1`.
///
/// Entries the attribute does not supply fall back to `default` — that covers
/// an absent attribute (ONNX defaults it to all ones) and, deliberately, an
/// attribute *shorter* than the spatial rank. A short `strides` is malformed
/// per the spec, but this engine has always defaulted the missing axes rather
/// than rejecting the model, and both the operator and the planner must agree
/// on that (see `a_short_strides_attribute_defaults_the_missing_axis_instead_of_panicking`
/// in `tests/s4_engine_stitch.rs`). Entries beyond the spatial rank are ignored,
/// again matching the pre-N-D behaviour.
pub(crate) fn read_positive_spatial(
    values: &[i64],
    rank: usize,
    default: usize,
    name: &str,
    op: &str,
) -> Result<Vec<usize>, OnnxError> {
    let mut out = vec![default; rank];
    for (axis, slot) in out.iter_mut().enumerate() {
        if let Some(&v) = values.get(axis) {
            if v < 1 {
                return Err(OnnxError::ShapeMismatch(format!(
                    "{op}: {name}[{axis}] must be >= 1, got {v}"
                )));
            }
            *slot = v as usize;
        }
    }
    Ok(out)
}

/// Read a per-spatial-axis attribute whose entries must be `>= 0`
/// (`output_padding`). Missing entries default to 0; see
/// [`read_positive_spatial`] for why short attributes are tolerated.
pub(crate) fn read_nonneg_spatial(
    values: &[i64],
    rank: usize,
    name: &str,
    op: &str,
) -> Result<Vec<usize>, OnnxError> {
    let mut out = vec![0_usize; rank];
    for (axis, slot) in out.iter_mut().enumerate() {
        if let Some(&v) = values.get(axis) {
            if v < 0 {
                return Err(OnnxError::ShapeMismatch(format!(
                    "{op}: {name}[{axis}] must be >= 0, got {v}"
                )));
            }
            *slot = v as usize;
        }
    }
    Ok(out)
}

/// Read the `pads` attribute as `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]`.
///
/// Negative padding is invalid for Conv/ConvTranspose/Pool and previously
/// wrapped into an enormous `usize`; it is a typed error. Missing entries
/// default to 0 — see [`read_positive_spatial`] for why short attributes are
/// tolerated rather than rejected.
pub(crate) fn read_pads(values: &[i64], rank: usize, op: &str) -> Result<Vec<usize>, OnnxError> {
    let mut out = vec![0_usize; 2 * rank];
    for (idx, slot) in out.iter_mut().enumerate() {
        if let Some(&v) = values.get(idx) {
            if v < 0 {
                return Err(OnnxError::ShapeMismatch(format!(
                    "{op}: pads[{idx}] must be >= 0, got {v}"
                )));
            }
            *slot = v as usize;
        }
    }
    Ok(out)
}

/// Read the `group` attribute, requiring `>= 1` (`group == 0` used to divide
/// by zero inside the kernels).
pub(crate) fn read_group(attrs: &Attributes, op: &str) -> Result<usize, OnnxError> {
    let group = attrs.i("group", 1);
    if group < 1 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: group must be >= 1, got {group}"
        )));
    }
    Ok(group as usize)
}

/// Read the `kernel_shape` attribute, which is required for the pooling
/// operators (and optional but validated for `Conv`).
///
/// Unlike `strides` / `pads` / `dilations` there is no sensible default for a
/// *missing* entry, so an attribute shorter than the spatial rank — including
/// an absent one — is a typed error. Entries beyond the spatial rank are
/// ignored, matching both the pre-N-D kernel (`ks_v.len() < 2`) and the
/// planner's `read_kernel_shape`, so the two never disagree about which nodes
/// are well-formed.
pub(crate) fn read_kernel_shape(
    values: &[i64],
    rank: usize,
    op: &str,
) -> Result<Vec<usize>, OnnxError> {
    if values.len() < rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: kernel_shape requires {rank} spatial dims (one per input spatial axis), got {}",
            values.len()
        )));
    }
    let mut out = vec![1_usize; rank];
    for (axis, slot) in out.iter_mut().enumerate() {
        let v = values[axis];
        if v < 1 {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: kernel_shape[{axis}] must be >= 1, got {v}"
            )));
        }
        *slot = v as usize;
    }
    Ok(out)
}

/// SAME_UPPER / SAME_LOWER padding split for one axis.
///
/// Per the ONNX spec the target extent is `ceil(in / stride)`, the total
/// padding is `(out - 1) * stride + ((k - 1) * dilation + 1) - in` clamped at
/// zero, and the odd pixel goes to the end (`SAME_UPPER`) or the beginning
/// (`SAME_LOWER`).
fn same_pad_split(
    in_dim: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    lower: bool,
) -> (usize, usize) {
    let out_dim = in_dim.div_ceil(stride.max(1));
    let eff_k = kernel.saturating_sub(1).saturating_mul(dilation) + 1;
    let needed = out_dim
        .saturating_sub(1)
        .saturating_mul(stride)
        .saturating_add(eff_k)
        .saturating_sub(in_dim);
    let half = needed / 2;
    if lower {
        (needed - half, half)
    } else {
        (half, needed - half)
    }
}

/// Resolve the effective `[begin_0.., end_0..]` padding from `auto_pad` plus
/// the explicit `pads` attribute.
pub(crate) fn resolve_pads(
    auto_pad: AutoPad,
    input_spatial: &[usize],
    kernel: &[usize],
    strides: &[usize],
    dilations: &[usize],
    explicit: &[usize],
) -> Vec<usize> {
    let rank = input_spatial.len();
    match auto_pad {
        AutoPad::NotSet => explicit.to_vec(),
        AutoPad::Valid => vec![0_usize; 2 * rank],
        AutoPad::SameUpper | AutoPad::SameLower => {
            let lower = auto_pad == AutoPad::SameLower;
            let mut out = vec![0_usize; 2 * rank];
            for axis in 0..rank {
                let (begin, end) = same_pad_split(
                    input_spatial[axis],
                    kernel.get(axis).copied().unwrap_or(1),
                    strides.get(axis).copied().unwrap_or(1),
                    dilations.get(axis).copied().unwrap_or(1),
                    lower,
                );
                out[axis] = begin;
                out[axis + rank] = end;
            }
            out
        }
    }
}

/// Pooling / convolution output extent for one axis, honoring `ceil_mode` and
/// `dilations`.
///
/// Mirrors the ONNX / ONNX Runtime formula:
/// `out = floor_or_ceil((in + pad_begin + pad_end - ((k - 1) * dilation + 1)) / stride) + 1`,
/// followed by the ceil-mode correction that drops a trailing window which
/// would start inside the right-hand padding.
///
/// All arithmetic is checked: a padded extent smaller than the dilated kernel,
/// a zero stride/dilation or an overflowing size product yields a typed
/// [`OnnxError::ShapeMismatch`] instead of an unsigned underflow (which panics
/// in debug builds and wraps to a near-`usize::MAX` allocation in release).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pool_out_dim(
    op: &str,
    axis: &str,
    in_dim: usize,
    pad_begin: usize,
    pad_end: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    ceil_mode: bool,
) -> Result<usize, OnnxError> {
    if stride == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: strides[{axis}] must be >= 1, got 0"
        )));
    }
    if kernel == 0 || dilation == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: invalid kernel/dilation on axis {axis} (kernel={kernel}, dilation={dilation})"
        )));
    }
    let eff_k = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("{op}: kernel extent overflow on axis {axis}"))
        })?;
    let padded = in_dim
        .checked_add(pad_begin)
        .and_then(|v| v.checked_add(pad_end))
        .ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("{op}: padded extent overflow on axis {axis}"))
        })?;
    if padded < eff_k {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: padded input extent {padded} on axis {axis} is smaller than the dilated \
             kernel extent {eff_k} (kernel={kernel}, dilation={dilation})"
        )));
    }
    let span = padded - eff_k;
    let mut out = if ceil_mode {
        span.div_ceil(stride) + 1
    } else {
        span / stride + 1
    };
    // A ceil-mode window is only legal if it *starts* inside the input or the
    // left padding; drop the trailing window otherwise (ONNX Runtime parity).
    if ceil_mode && out > 1 && (out - 1).saturating_mul(stride) >= in_dim + pad_begin {
        out -= 1;
    }
    Ok(out)
}

/// Floor-mode convolution output extent for one axis (`ceil_mode` never
/// applies to `Conv`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_out_dim(
    op: &str,
    axis: &str,
    in_dim: usize,
    pad_begin: usize,
    pad_end: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
) -> Result<usize, OnnxError> {
    pool_out_dim(
        op, axis, in_dim, pad_begin, pad_end, kernel, dilation, stride, false,
    )
}

/// Natural transposed-convolution output extent for one axis:
/// `stride * (in - 1) + output_padding + ((k - 1) * dilation + 1) - pad_begin - pad_end`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose_out_dim(
    op: &str,
    axis: &str,
    in_dim: usize,
    stride: usize,
    output_padding: usize,
    kernel: usize,
    dilation: usize,
    pad_begin: usize,
    pad_end: usize,
) -> Result<usize, OnnxError> {
    let natural =
        conv_transpose_natural_dim(op, axis, in_dim, stride, output_padding, kernel, dilation)?;
    let total_pad = pad_begin.checked_add(pad_end).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!("{op}: padding overflow on axis {axis}"))
    })?;
    natural.checked_sub(total_pad).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!(
            "{op}: pads {total_pad} on axis {axis} exceed the un-cropped output extent {natural}"
        ))
    })
}

/// The un-cropped transposed-convolution extent, before `pads` is subtracted.
pub(crate) fn conv_transpose_natural_dim(
    op: &str,
    axis: &str,
    in_dim: usize,
    stride: usize,
    output_padding: usize,
    kernel: usize,
    dilation: usize,
) -> Result<usize, OnnxError> {
    if stride == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: strides[{axis}] must be >= 1, got 0"
        )));
    }
    if dilation == 0 || kernel == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: invalid kernel/dilation on axis {axis} (kernel={kernel}, dilation={dilation})"
        )));
    }
    if in_dim == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: input extent on axis {axis} must be >= 1, got 0"
        )));
    }
    let eff_k = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("{op}: kernel extent overflow on axis {axis}"))
        })?;
    stride
        .checked_mul(in_dim - 1)
        .and_then(|v| v.checked_add(output_padding))
        .and_then(|v| v.checked_add(eff_k))
        .ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("{op}: output extent overflow on axis {axis}"))
        })
}

/// Compute the full `[N, F, o_0, …]` output shape for an N-D convolution.
///
/// `input_shape` is `[N, C, d_0, …]`, `weight_shape` is `[F, C/group, k_0, …]`,
/// and `pads` is the resolved `2 * r` ONNX padding vector.
pub(crate) fn compute_conv_out_shape(
    op: &str,
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    dilations: &[usize],
) -> Result<Vec<usize>, OnnxError> {
    let rank = spatial_rank(input_shape, op, "input")?;
    if weight_shape.len() != input_shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: weight rank {} must equal input rank {} ([F, C/group, k_0, ...])",
            weight_shape.len(),
            input_shape.len()
        )));
    }
    if strides.len() != rank || dilations.len() != rank || pads.len() != 2 * rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: strides/dilations need {rank} entries and pads needs {} \
             (got {}/{}/{})",
            2 * rank,
            strides.len(),
            dilations.len(),
            pads.len()
        )));
    }
    let mut out_shape = Vec::with_capacity(rank + 2);
    out_shape.push(input_shape[0]);
    out_shape.push(weight_shape[0]);
    for axis in 0..rank {
        out_shape.push(conv_out_dim(
            op,
            axis_label(rank, axis),
            input_shape[axis + 2],
            pads[axis],
            pads[axis + rank],
            weight_shape[axis + 2],
            dilations[axis],
            strides[axis],
        )?);
    }
    Ok(out_shape)
}

/// Compute the full `[N, C_out, o_0, …]` output shape for an N-D transposed
/// convolution. `weight_shape` is `[C_in, C_out/group, k_0, …]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_conv_transpose_out_shape(
    op: &str,
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    output_padding: &[usize],
    dilations: &[usize],
    group: usize,
) -> Result<Vec<usize>, OnnxError> {
    let rank = spatial_rank(input_shape, op, "input")?;
    if weight_shape.len() != input_shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: weight rank {} must equal input rank {} ([C_in, C_out/group, k_0, ...])",
            weight_shape.len(),
            input_shape.len()
        )));
    }
    if group == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: group must be >= 1, got 0"
        )));
    }
    let c_out = weight_shape[1]
        .checked_mul(group)
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: output channel count overflows")))?;
    let mut out_shape = Vec::with_capacity(rank + 2);
    out_shape.push(input_shape[0]);
    out_shape.push(c_out);
    for axis in 0..rank {
        out_shape.push(conv_transpose_out_dim(
            op,
            axis_label(rank, axis),
            input_shape[axis + 2],
            strides.get(axis).copied().unwrap_or(1),
            output_padding.get(axis).copied().unwrap_or(0),
            weight_shape[axis + 2],
            dilations.get(axis).copied().unwrap_or(1),
            pads.get(axis).copied().unwrap_or(0),
            pads.get(axis + rank).copied().unwrap_or(0),
        )?);
    }
    Ok(out_shape)
}

/// Mixed-radix odometer over a spatial index vector.
///
/// Increments `idx` (last axis fastest) within `extents` and reports whether
/// the counter wrapped back to all-zero, i.e. whether the iteration is done.
/// Used by the generic N-D kernels to walk output / kernel positions without
/// recursion or per-rank specialisation.
#[inline]
pub(crate) fn odometer_next(idx: &mut [usize], extents: &[usize]) -> bool {
    for axis in (0..idx.len()).rev() {
        idx[axis] += 1;
        if idx[axis] < extents[axis] {
            return false;
        }
        idx[axis] = 0;
    }
    true
}
