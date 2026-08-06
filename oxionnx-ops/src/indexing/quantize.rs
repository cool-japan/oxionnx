use oxionnx_core::Tensor;

use super::index_util::normalize_axis;

/// Round to the nearest integer, ties to even (banker's rounding) — the
/// rounding mode the ONNX `QuantizeLinear` spec mandates for
/// `round(x / y_scale)`. Rust's `f32::round()` is ties-away-from-zero,
/// which diverges exactly at the `.5` boundary (e.g. `2.5 -> 3` instead of
/// the spec-correct `2`), so it cannot be used directly here. Hand-rolled
/// rather than the standard-library `f32::round_ties_even` (stabilized in
/// Rust 1.77) to stay compatible with this workspace's `rust-version`
/// (1.75) MSRV.
#[inline]
fn round_ties_even(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let floor = v.floor();
    let diff = v - floor;
    if diff > 0.5 {
        floor + 1.0
    } else if diff < 0.5 {
        floor
    } else {
        // Exact tie: round to the neighboring even integer.
        if floor.rem_euclid(2.0) == 0.0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

/// The saturation range `QuantizeLinear` clamps its output into, inferred
/// from `y_zero_point`.
///
/// Per spec the output integer type is `y_zero_point`'s TensorProto dtype,
/// defaulting to **uint8** `[0, 255]` only when `y_zero_point` is absent
/// entirely. This crate's `Tensor` carries no dtype tag (everything is
/// logical f32), so when a zero-point *is* provided, its declared type
/// cannot be read directly — it is inferred from the zero-point's value:
///
/// - `y_zero_point` absent: **uint8** `[0, 255]`, zero-point 0 — the one
///   case the spec pins down unconditionally.
/// - Any provided zero-point component `> 127`: **uint8** `[0, 255]` — a
///   uint8-only value (int8 tops out at 127).
/// - Any provided zero-point otherwise (including exactly 0): **int8**
///   `[-128, 127]`. A zero-point of 0 is, in practice, overwhelmingly the
///   signature of *symmetric int8* quantization — the dominant real-world
///   scheme for weight quantization (ORT static quantization, PyTorch FX,
///   TensorRT, etc. all default weights to signed symmetric int8 with
///   zero-point 0). Treating an explicit zero-point of 0 as uint8 instead
///   would silently zero out every negative quantized weight (e.g.
///   `round(-0.5/scale) + 0 = -50` clamped to `0` instead of staying
///   `-50`) — a far more damaging failure than the reverse (an unusual
///   uint8-with-zero-point-0 scheme whose values happen to need `>127`
///   loses precision at the boundary instead of being silently zeroed).
fn saturation_range(zero_point: Option<&Tensor>) -> (f32, f32) {
    match zero_point {
        None => (0.0, 255.0),
        Some(zp) if zp.data.iter().any(|&v| v > 127.0) => (0.0, 255.0),
        Some(_) => (-128.0, 127.0),
    }
}

/// Compute the per-element quantization channel for flat index `flat` of a
/// tensor shaped `shape`, given an axis-normalized `axis` and the number of
/// scale/zero-point entries (`param_len`).
///
/// Per-tensor quantization (`param_len <= 1`) always maps to channel 0
/// regardless of shape/axis. Per-axis quantization maps every element to
/// its own coordinate along `axis`, computed from `axis`'s stride —
/// `(flat / inner_stride) % dim[axis]` — NOT `flat % param_len`, which is
/// only correct when `axis` happens to be the fastest-varying (last) one.
#[inline]
fn channel_index(flat: usize, shape: &[usize], axis: usize, param_len: usize) -> usize {
    if param_len <= 1 {
        return 0;
    }
    let inner_stride: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let axis_dim = shape[axis].max(1);
    (flat / inner_stride) % axis_dim
}

/// Resolve the quantization axis: only meaningful (and only validated) when
/// there is genuinely more than one scale/zero-point value to pick between —
/// per spec, `axis` is ignored entirely for per-tensor quantization.
fn resolve_axis(ndim: usize, axis: i64, param_len: usize, op: &str) -> Result<usize, String> {
    if param_len > 1 {
        normalize_axis(axis, ndim, op)
    } else {
        Ok(0)
    }
}

/// QuantizeLinear: `y = saturate(round_ties_even(x / y_scale) + y_zero_point)`.
///
/// Uses the ONNX default quantization axis (1); see [`quantize_linear_axis`]
/// for per-axis (per-channel) control.
pub fn quantize_linear(
    x: &Tensor,
    y_scale: &Tensor,
    y_zero_point: Option<&Tensor>,
) -> Result<Tensor, String> {
    quantize_linear_axis(x, y_scale, y_zero_point, 1)
}

/// QuantizeLinear with an explicit per-axis quantization `axis`.
///
/// `y_scale` (and `y_zero_point`, if present) may hold either a single
/// value (per-tensor quantization, `axis` ignored) or one value per
/// coordinate along `axis` (per-channel quantization — e.g. one scale per
/// output channel of a convolution weight, `axis=0`).
pub fn quantize_linear_axis(
    x: &Tensor,
    y_scale: &Tensor,
    y_zero_point: Option<&Tensor>,
    axis: i64,
) -> Result<Tensor, String> {
    let scale_len = y_scale.numel();
    if scale_len == 0 {
        return Err("quantize_linear: y_scale must have at least one element".into());
    }
    if y_zero_point.is_some_and(|t| t.numel() == 0) {
        return Err(
            "quantize_linear: y_zero_point must have at least one element when provided".into(),
        );
    }
    let zp_len = y_zero_point.map(Tensor::numel).unwrap_or(0);
    let ax = resolve_axis(x.ndim(), axis, scale_len.max(zp_len), "quantize_linear")?;
    let (lo, hi) = saturation_range(y_zero_point);

    let data: Vec<f32> = x
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let scale = y_scale.data[channel_index(i, &x.shape, ax, scale_len)];
            let zp = y_zero_point
                .map(|t| t.data[channel_index(i, &x.shape, ax, zp_len)])
                .unwrap_or(0.0);
            (round_ties_even(v / scale) + zp).clamp(lo, hi)
        })
        .collect();
    Ok(Tensor::new(data, x.shape.clone()))
}

/// DequantizeLinear: `y = (x - x_zero_point) * x_scale`.
///
/// Uses the ONNX default dequantization axis (1); see
/// [`dequantize_linear_axis`] for per-axis (per-channel) control.
pub fn dequantize_linear(
    x: &Tensor,
    x_scale: &Tensor,
    x_zero_point: Option<&Tensor>,
) -> Result<Tensor, String> {
    dequantize_linear_axis(x, x_scale, x_zero_point, 1)
}

/// DequantizeLinear with an explicit per-axis dequantization `axis` (see
/// [`quantize_linear_axis`] for the per-tensor vs per-channel rules).
pub fn dequantize_linear_axis(
    x: &Tensor,
    x_scale: &Tensor,
    x_zero_point: Option<&Tensor>,
    axis: i64,
) -> Result<Tensor, String> {
    let scale_len = x_scale.numel();
    if scale_len == 0 {
        return Err("dequantize_linear: x_scale must have at least one element".into());
    }
    if x_zero_point.is_some_and(|t| t.numel() == 0) {
        return Err(
            "dequantize_linear: x_zero_point must have at least one element when provided".into(),
        );
    }
    let zp_len = x_zero_point.map(Tensor::numel).unwrap_or(0);
    let ax = resolve_axis(x.ndim(), axis, scale_len.max(zp_len), "dequantize_linear")?;

    let data: Vec<f32> = x
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let scale = x_scale.data[channel_index(i, &x.shape, ax, scale_len)];
            let zp = x_zero_point
                .map(|t| t.data[channel_index(i, &x.shape, ax, zp_len)])
                .unwrap_or(0.0);
            (v - zp) * scale
        })
        .collect();
    Ok(Tensor::new(data, x.shape.clone()))
}
