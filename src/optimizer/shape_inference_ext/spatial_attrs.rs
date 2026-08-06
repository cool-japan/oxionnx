//! Shared spatial-attribute readers for Conv / ConvTranspose / pooling shape
//! inference.
//!
//! Every helper here mirrors — deliberately, function for function — the
//! validated readers the operators themselves use in
//! `oxionnx-ops/src/registry/conv_ops/conv.rs`. The two must stay in lock-step:
//! a planner that predicts a different extent than the kernel produces either
//! over-allocates a slot (best case, the slot is resized and the buffer-pool
//! reuse is wasted) or propagates a wrong "known shape" into fusion decisions
//! that size synthesised constants from it (worst case).
//!
//! Every function returns `Option`: shape inference is best-effort, so a
//! malformed or unsupported attribute means "no prediction", never a panic.
//! That matters concretely — the previous readers did `p as usize` on a
//! model-supplied `i64`, so a negative `pads` entry wrapped to ~2^64 and the
//! subsequent `input_dim + pads[..]` overflow-panicked in any build with
//! `debug_assertions` (which includes `cargo test`).

/// Read an N-entry spatial attribute (`strides`, `dilations`) whose entries
/// must be `>= 1`, filling missing trailing entries with `default`.
///
/// Filling rather than rejecting matches `read_positive_pair` on the kernel
/// side, which reads `values.get(axis)` per axis and leaves the default in
/// place when the attribute is short.
pub(crate) fn read_positive_spatial(
    values: &[i64],
    rank: usize,
    default: usize,
) -> Option<Vec<usize>> {
    let mut out = vec![default; rank];
    for (axis, slot) in out.iter_mut().enumerate() {
        if let Some(&v) = values.get(axis) {
            if v < 1 {
                return None;
            }
            *slot = usize::try_from(v).ok()?;
        }
    }
    Some(out)
}

/// Read a required N-entry kernel extent attribute (`kernel_shape`).
///
/// Unlike [`read_positive_spatial`] there is no default: pooling operators
/// require `kernel_shape`, and a short attribute is malformed.
pub(crate) fn read_kernel_shape(values: &[i64], rank: usize) -> Option<Vec<usize>> {
    if values.len() < rank {
        return None;
    }
    let mut out = Vec::with_capacity(rank);
    for &v in &values[..rank] {
        if v < 1 {
            return None;
        }
        out.push(usize::try_from(v).ok()?);
    }
    Some(out)
}

/// Read the `pads` attribute as `[begin_0, .., begin_{n-1}, end_0, .., end_{n-1}]`.
///
/// Missing trailing entries default to zero (kernel-side `read_pads_2d`
/// parity); a negative entry is rejected rather than wrapped.
pub(crate) fn read_pads(values: &[i64], rank: usize) -> Option<Vec<usize>> {
    let mut out = vec![0_usize; rank * 2];
    for (idx, slot) in out.iter_mut().enumerate() {
        if let Some(&v) = values.get(idx) {
            if v < 0 {
                return None;
            }
            *slot = usize::try_from(v).ok()?;
        }
    }
    Some(out)
}

/// SAME_UPPER / SAME_LOWER padding split for one spatial axis.
///
/// Per the ONNX spec the target extent is `ceil(in / stride)`, the total
/// padding is `(out - 1) * stride + ((k - 1) * dilation + 1) - in` clamped at
/// zero, and the odd pixel goes to the end (`SAME_UPPER`) or the beginning
/// (`SAME_LOWER`).
pub(crate) fn same_pad_split(
    in_dim: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    lower: bool,
) -> (usize, usize) {
    let out_dim = in_dim.div_ceil(stride.max(1));
    let effective_kernel = kernel.saturating_sub(1).saturating_mul(dilation) + 1;
    let needed = out_dim
        .saturating_sub(1)
        .saturating_mul(stride)
        .saturating_add(effective_kernel)
        .saturating_sub(in_dim);
    let half = needed / 2;
    if lower {
        (needed - half, half)
    } else {
        (half, needed - half)
    }
}

/// Resolve the effective per-axis padding from `auto_pad` plus the explicit
/// `pads` attribute, mirroring `resolve_pads_2d` on the kernel side.
///
/// Returns `None` for an `auto_pad` value the operators reject, so an
/// unrecognised mode yields no shape prediction instead of a guess.
pub(crate) fn resolve_pads(
    auto_pad: &str,
    input_spatial: &[usize],
    kernel: &[usize],
    strides: &[usize],
    dilations: &[usize],
    explicit: &[usize],
) -> Option<Vec<usize>> {
    let rank = input_spatial.len();
    if kernel.len() < rank || strides.len() < rank || dilations.len() < rank {
        return None;
    }
    match auto_pad {
        "" | "NOTSET" => {
            if explicit.len() != rank * 2 {
                return None;
            }
            Some(explicit.to_vec())
        }
        "VALID" => Some(vec![0_usize; rank * 2]),
        mode @ ("SAME_UPPER" | "SAME_LOWER") => {
            let lower = mode == "SAME_LOWER";
            let mut pads = vec![0_usize; rank * 2];
            for axis in 0..rank {
                let (begin, end) = same_pad_split(
                    input_spatial[axis],
                    kernel[axis],
                    strides[axis],
                    dilations[axis],
                    lower,
                );
                pads[axis] = begin;
                pads[axis + rank] = end;
            }
            Some(pads)
        }
        // Unknown auto_pad: the operator errors, so refuse to guess a shape.
        _ => None,
    }
}

/// Pooling output extent for one axis, honouring `ceil_mode` and `dilations`.
///
/// Mirrors `pool_out_dim` in `registry/conv_ops/conv.rs`, including the
/// ceil-mode correction that drops a trailing window which would *start* inside
/// the right-hand padding (ONNX Runtime parity). Without that correction the
/// planner over-predicts: `in = 5, k = 3, s = 3, pads = [0, 2], ceil_mode = 1`
/// gives 3 from the bare formula but the kernel produces 2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pool_out_dim(
    in_dim: usize,
    pad_begin: usize,
    pad_end: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    ceil_mode: bool,
) -> Option<usize> {
    let span = spatial_span(in_dim, pad_begin, pad_end, kernel, dilation)?;
    if stride == 0 {
        return None;
    }
    let mut out = if ceil_mode {
        span.div_ceil(stride).checked_add(1)?
    } else {
        (span / stride).checked_add(1)?
    };
    if ceil_mode && out > 1 && (out - 1).saturating_mul(stride) >= in_dim.saturating_add(pad_begin)
    {
        out -= 1;
    }
    Some(out)
}

/// Conv output extent for one axis (always floor — `Conv` has no `ceil_mode`).
pub(crate) fn conv_out_dim(
    in_dim: usize,
    pad_begin: usize,
    pad_end: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
) -> Option<usize> {
    pool_out_dim(in_dim, pad_begin, pad_end, kernel, dilation, stride, false)
}

/// `in + pad_begin + pad_end - ((k - 1) * dilation + 1)`, or `None` when the
/// dilated kernel does not fit (which the operators report as a `ShapeMismatch`).
fn spatial_span(
    in_dim: usize,
    pad_begin: usize,
    pad_end: usize,
    kernel: usize,
    dilation: usize,
) -> Option<usize> {
    if kernel == 0 || dilation == 0 {
        return None;
    }
    let effective_kernel = (kernel - 1).checked_mul(dilation)?.checked_add(1)?;
    let padded = in_dim.checked_add(pad_begin)?.checked_add(pad_end)?;
    padded.checked_sub(effective_kernel)
}

/// Transposed-convolution output extent for one axis:
/// `stride * (in - 1) + output_padding + ((k - 1) * dilation + 1) - pad_begin - pad_end`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose_out_dim(
    in_dim: usize,
    stride: usize,
    output_padding: usize,
    kernel: usize,
    dilation: usize,
    pad_begin: usize,
    pad_end: usize,
) -> Option<usize> {
    if in_dim == 0 || kernel == 0 || dilation == 0 || stride == 0 {
        return None;
    }
    let effective_kernel = (kernel - 1).checked_mul(dilation)?.checked_add(1)?;
    let natural = stride
        .checked_mul(in_dim - 1)?
        .checked_add(output_padding)?
        .checked_add(effective_kernel)?;
    natural
        .checked_sub(pad_begin)?
        .checked_sub(pad_end)
        .filter(|&out| out > 0)
}
