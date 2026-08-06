//! ONNX `Resize` operator (opset-19 / opset-21 semantics).
//!
//! The implementation is *separable*: an N-D resize is performed as a sequence
//! of independent 1-D resampling passes, one per resized axis. This is exactly
//! how the ONNX reference implementation defines N-D interpolation
//! (`_interpolate_nd_with_x` recurses one axis at a time), so linear / cubic
//! results agree with the reference rather than only approximating it.
//!
//! Supported:
//! * `mode`: `nearest`, `linear`, `cubic` (plus the `bilinear` / `trilinear` /
//!   `bicubic` aliases some exporters emit).
//! * `coordinate_transformation_mode`: `half_pixel`, `half_pixel_symmetric`,
//!   `pytorch_half_pixel`, `align_corners`, `asymmetric`, `tf_crop_and_resize`,
//!   `tf_half_pixel_for_nn` (opset-10 legacy).
//! * `nearest_mode`: `round_prefer_floor` (default), `round_prefer_ceil`,
//!   `floor`, `ceil`.
//! * `cubic_coeff_a`, `exclude_outside`, `extrapolation_value`, `antialias`,
//!   `axes`, `keep_aspect_ratio_policy`, and the `roi` input.
//!
//! Anything not recognised produces a typed [`OnnxError`] — never a silent
//! substitution of a different kernel.

use oxionnx_core::{OnnxError, Tensor};

// ── Public options ──────────────────────────────────────────────────────────

/// Full attribute set of ONNX `Resize`, as borrowed strings/slices.
///
/// [`Default`] reproduces the ONNX defaults, so callers only override what the
/// node actually specifies.
#[derive(Debug, Clone, Copy)]
pub struct ResizeOptions<'a> {
    /// `mode` attribute: `nearest` | `linear` | `cubic`.
    pub mode: &'a str,
    /// `coordinate_transformation_mode` attribute.
    pub coordinate_transformation_mode: &'a str,
    /// `nearest_mode` attribute (only meaningful for `mode = "nearest"`).
    pub nearest_mode: &'a str,
    /// `keep_aspect_ratio_policy` attribute (only meaningful with `sizes`).
    pub keep_aspect_ratio_policy: &'a str,
    /// `cubic_coeff_a` attribute (only meaningful for `mode = "cubic"`).
    pub cubic_coeff_a: f32,
    /// `extrapolation_value` attribute (only used by `tf_crop_and_resize`).
    pub extrapolation_value: f32,
    /// `exclude_outside` attribute.
    pub exclude_outside: bool,
    /// `antialias` attribute (linear / cubic downscaling only).
    pub antialias: bool,
    /// `axes` attribute: subset of axes that `roi`/`scales`/`sizes` refer to.
    pub axes: Option<&'a [i64]>,
    /// `roi` input, laid out as `[start_0..start_k, end_0..end_k]`.
    pub roi: Option<&'a [f32]>,
}

impl Default for ResizeOptions<'_> {
    fn default() -> Self {
        Self {
            mode: "nearest",
            coordinate_transformation_mode: "half_pixel",
            nearest_mode: "round_prefer_floor",
            keep_aspect_ratio_policy: "stretch",
            cubic_coeff_a: -0.75,
            extrapolation_value: 0.0,
            exclude_outside: false,
            antialias: false,
            axes: None,
            roi: None,
        }
    }
}

/// Resize with the `mode` / `coordinate_transformation_mode` pair only.
///
/// Every other attribute takes its ONNX default. Errors on an unknown mode.
pub fn resize(
    input: &Tensor,
    scales: Option<&[f32]>,
    sizes: Option<&[usize]>,
    mode: &str,
    coord_transform: &str,
) -> Result<Tensor, OnnxError> {
    resize_with(
        input,
        scales,
        sizes,
        &ResizeOptions {
            mode,
            coordinate_transformation_mode: coord_transform,
            ..ResizeOptions::default()
        },
    )
}

/// Resize honouring the full opset-19 attribute set.
pub fn resize_with(
    input: &Tensor,
    scales: Option<&[f32]>,
    sizes: Option<&[usize]>,
    opts: &ResizeOptions<'_>,
) -> Result<Tensor, OnnxError> {
    let mut out = Tensor::zeros(&[0]);
    resize_into(input, scales, sizes, opts, &mut out)?;
    Ok(out)
}

/// Resize writing into a caller-owned output tensor (output-slot path).
///
/// `out` is resized in place, so a reused slot avoids reallocation.
pub fn resize_into(
    input: &Tensor,
    scales: Option<&[f32]>,
    sizes: Option<&[usize]>,
    opts: &ResizeOptions<'_>,
    out: &mut Tensor,
) -> Result<(), OnnxError> {
    let expected = checked_product(&input.shape)?;
    if input.data.len() != expected {
        return Err(OnnxError::ShapeMismatch(format!(
            "Resize: input has {} elements but shape {:?} implies {expected}",
            input.data.len(),
            input.shape,
        )));
    }

    let interp = parse_interp(opts.mode)?;
    let coord = parse_coord_mode(opts.coordinate_transformation_mode)?;
    let nearest_rule = parse_nearest_rule(opts.nearest_mode)?;
    if opts.antialias && interp == Interp::Nearest {
        return Err(OnnxError::Unsupported(
            "Resize: antialias=1 is not defined for mode='nearest'".to_string(),
        ));
    }
    if !opts.cubic_coeff_a.is_finite() {
        return Err(OnnxError::InvalidModel(
            "Resize: cubic_coeff_a must be finite".to_string(),
        ));
    }
    if !opts.extrapolation_value.is_finite() && coord == CoordMode::TfCropAndResize {
        return Err(OnnxError::InvalidModel(
            "Resize: extrapolation_value must be finite".to_string(),
        ));
    }

    let kernel = KernelParams {
        interp,
        nearest_rule,
        coord,
        cubic_a: opts.cubic_coeff_a,
        exclude_outside: opts.exclude_outside,
        antialias: opts.antialias,
    };
    let plan = resolve_plan(input, scales, sizes, opts, coord)?;

    let out_n = checked_product(&plan.out_shape)?;
    fit(out, out_n);
    out.shape.clone_from(&plan.out_shape);
    if out_n == 0 {
        return Ok(());
    }

    // Build one resampler per axis; identity passes are dropped.
    let rank = input.shape.len();
    let mut passes: Vec<(usize, AxisResampler)> = Vec::new();
    for d in 0..rank {
        let spec = AxisSpec {
            in_size: input.shape[d],
            out_size: plan.out_shape[d],
            scale: plan.scale[d],
            float_width: plan.float_width[d],
            roi_start: plan.roi[d].0,
            roi_end: plan.roi[d].1,
        };
        let resampler = build_axis(&spec, &kernel)?;
        if resampler.is_identity(spec.in_size) {
            continue;
        }
        passes.push((d, resampler));
    }

    if passes.is_empty() {
        if out.data.len() != input.data.len() {
            return Err(OnnxError::Internal(
                "Resize: identity plan produced a mismatched output length".to_string(),
            ));
        }
        out.data.copy_from_slice(&input.data);
    } else {
        run_passes(input, &plan.out_shape, &passes, out)?;
    }
    apply_extrapolation(&plan.out_shape, &passes, opts.extrapolation_value, out);
    Ok(())
}

// ── Mode parsing ────────────────────────────────────────────────────────────

/// Interpolation kernel selected by `mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interp {
    Nearest,
    Linear,
    Cubic,
}

/// Rounding rule selected by `nearest_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NearestRule {
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

/// Coordinate mapping selected by `coordinate_transformation_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoordMode {
    HalfPixel,
    HalfPixelSymmetric,
    PytorchHalfPixel,
    AlignCorners,
    Asymmetric,
    TfCropAndResize,
    TfHalfPixelForNn,
}

fn parse_interp(mode: &str) -> Result<Interp, OnnxError> {
    match mode {
        "" | "nearest" => Ok(Interp::Nearest),
        "linear" | "bilinear" | "trilinear" => Ok(Interp::Linear),
        "cubic" | "bicubic" => Ok(Interp::Cubic),
        other => Err(OnnxError::Unsupported(format!(
            "Resize: unsupported mode '{other}' (expected 'nearest', 'linear' or 'cubic')"
        ))),
    }
}

fn parse_nearest_rule(mode: &str) -> Result<NearestRule, OnnxError> {
    match mode {
        "" | "round_prefer_floor" => Ok(NearestRule::RoundPreferFloor),
        "round_prefer_ceil" => Ok(NearestRule::RoundPreferCeil),
        "floor" => Ok(NearestRule::Floor),
        "ceil" => Ok(NearestRule::Ceil),
        other => Err(OnnxError::Unsupported(format!(
            "Resize: unsupported nearest_mode '{other}'"
        ))),
    }
}

fn parse_coord_mode(mode: &str) -> Result<CoordMode, OnnxError> {
    match mode {
        "" | "half_pixel" => Ok(CoordMode::HalfPixel),
        "half_pixel_symmetric" => Ok(CoordMode::HalfPixelSymmetric),
        "pytorch_half_pixel" => Ok(CoordMode::PytorchHalfPixel),
        "align_corners" => Ok(CoordMode::AlignCorners),
        "asymmetric" => Ok(CoordMode::Asymmetric),
        "tf_crop_and_resize" => Ok(CoordMode::TfCropAndResize),
        "tf_half_pixel_for_nn" => Ok(CoordMode::TfHalfPixelForNn),
        other => Err(OnnxError::Unsupported(format!(
            "Resize: unsupported coordinate_transformation_mode '{other}'"
        ))),
    }
}

// ── Shape / scale resolution ────────────────────────────────────────────────

struct KernelParams {
    interp: Interp,
    nearest_rule: NearestRule,
    coord: CoordMode,
    cubic_a: f32,
    exclude_outside: bool,
    antialias: bool,
}

struct AxisSpec {
    in_size: usize,
    out_size: usize,
    /// Scale factor as seen by the coordinate transform (never derived from the
    /// *truncated* output size when the model supplied `scales`).
    scale: f32,
    /// `scale * in_size` before truncation — only `half_pixel_symmetric` needs it.
    float_width: f32,
    roi_start: f32,
    roi_end: f32,
}

struct ResizePlan {
    out_shape: Vec<usize>,
    scale: Vec<f32>,
    float_width: Vec<f32>,
    roi: Vec<(f32, f32)>,
}

fn checked_product(shape: &[usize]) -> Result<usize, OnnxError> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d).ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("Resize: shape {shape:?} overflows usize"))
        })
    })
}

fn fit(out: &mut Tensor, n: usize) {
    if out.data.len() != n {
        out.data.clear();
        out.data.resize(n, 0.0f32);
    }
}

/// Resolve `axes` into concrete, de-duplicated axis indices.
fn resolve_axes(rank: usize, axes: Option<&[i64]>) -> Result<Vec<usize>, OnnxError> {
    let Some(axes) = axes else {
        return Ok((0..rank).collect());
    };
    let mut seen = vec![false; rank];
    let mut out = Vec::with_capacity(axes.len());
    for &a in axes {
        let norm = if a < 0 { a + rank as i64 } else { a };
        if norm < 0 || norm >= rank as i64 {
            return Err(OnnxError::InvalidModel(format!(
                "Resize: axes entry {a} out of range for rank {rank}"
            )));
        }
        let idx = norm as usize;
        if seen[idx] {
            return Err(OnnxError::InvalidModel(format!(
                "Resize: axes contains duplicate axis {idx}"
            )));
        }
        seen[idx] = true;
        out.push(idx);
    }
    Ok(out)
}

fn resolve_roi(
    rank: usize,
    axes: &[usize],
    coord: CoordMode,
    roi: Option<&[f32]>,
) -> Result<Vec<(f32, f32)>, OnnxError> {
    let mut out = vec![(0.0f32, 1.0f32); rank];
    if coord != CoordMode::TfCropAndResize {
        // `roi` is ignored by every other coordinate transformation mode.
        return Ok(out);
    }
    let roi = roi.ok_or_else(|| {
        OnnxError::InvalidModel(
            "Resize: coordinate_transformation_mode='tf_crop_and_resize' requires the 'roi' input"
                .to_string(),
        )
    })?;
    if roi.len() != axes.len() * 2 {
        return Err(OnnxError::ShapeMismatch(format!(
            "Resize: roi has {} values but {} were expected (2 x {} axes)",
            roi.len(),
            axes.len() * 2,
            axes.len(),
        )));
    }
    for (i, &d) in axes.iter().enumerate() {
        let start = roi[i];
        let end = roi[axes.len() + i];
        if !start.is_finite() || !end.is_finite() {
            return Err(OnnxError::InvalidModel(
                "Resize: roi values must be finite".to_string(),
            ));
        }
        out[d] = (start, end);
    }
    Ok(out)
}

fn resolve_plan(
    input: &Tensor,
    scales: Option<&[f32]>,
    sizes: Option<&[usize]>,
    opts: &ResizeOptions<'_>,
    coord: CoordMode,
) -> Result<ResizePlan, OnnxError> {
    let rank = input.shape.len();
    let axes = resolve_axes(rank, opts.axes)?;
    let roi = resolve_roi(rank, &axes, coord, opts.roi)?;

    let mut out_shape = input.shape.clone();
    let mut scale = vec![1.0f32; rank];
    let mut float_width: Vec<f32> = input.shape.iter().map(|&d| d as f32).collect();

    match (scales, sizes) {
        (Some(_), Some(_)) => {
            return Err(OnnxError::InvalidModel(
                "Resize: only one of 'scales' and 'sizes' may be specified".to_string(),
            ));
        }
        (None, None) => {
            return Err(OnnxError::InvalidModel(
                "Resize: one of 'scales' or 'sizes' must be specified".to_string(),
            ));
        }
        (Some(scales), None) => {
            if scales.len() != axes.len() {
                return Err(OnnxError::ShapeMismatch(format!(
                    "Resize: scales has {} values but {} axes are resized",
                    scales.len(),
                    axes.len(),
                )));
            }
            for (i, &d) in axes.iter().enumerate() {
                let s = scales[i];
                if !s.is_finite() || s <= 0.0 {
                    return Err(OnnxError::InvalidModel(format!(
                        "Resize: scales[{i}] = {s} must be finite and positive"
                    )));
                }
                // Spec: output_dimension = floor(input_dimension * scale).
                // The product is evaluated in f32, matching onnxruntime.
                let width = input.shape[d] as f32 * s;
                let dim = width.floor();
                if !(0.0..=usize::MAX as f32).contains(&dim) {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "Resize: scales[{i}] = {s} yields an out-of-range output dimension"
                    )));
                }
                out_shape[d] = dim as usize;
                scale[d] = s;
                float_width[d] = width;
            }
        }
        (None, Some(sizes)) => {
            if sizes.len() != axes.len() {
                return Err(OnnxError::ShapeMismatch(format!(
                    "Resize: sizes has {} values but {} axes are resized",
                    sizes.len(),
                    axes.len(),
                )));
            }
            apply_sizes(
                input,
                sizes,
                &axes,
                opts.keep_aspect_ratio_policy,
                &mut out_shape,
                &mut scale,
                &mut float_width,
            )?;
        }
    }

    for (d, (&in_len, &out_len)) in input.shape.iter().zip(out_shape.iter()).enumerate() {
        if in_len == 0 && out_len != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Resize: axis {d} has input length 0, cannot produce {out_len} output elements",
            )));
        }
    }

    Ok(ResizePlan {
        out_shape,
        scale,
        float_width,
        roi,
    })
}

fn apply_sizes(
    input: &Tensor,
    sizes: &[usize],
    axes: &[usize],
    policy: &str,
    out_shape: &mut [usize],
    scale: &mut [f32],
    float_width: &mut [f32],
) -> Result<(), OnnxError> {
    match policy {
        "" | "stretch" => {
            for (i, &d) in axes.iter().enumerate() {
                out_shape[d] = sizes[i];
                scale[d] = if input.shape[d] == 0 {
                    1.0
                } else {
                    sizes[i] as f32 / input.shape[d] as f32
                };
                float_width[d] = sizes[i] as f32;
            }
            Ok(())
        }
        "not_larger" | "not_smaller" => {
            let mut common: Option<f32> = None;
            for (i, &d) in axes.iter().enumerate() {
                if input.shape[d] == 0 {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "Resize: keep_aspect_ratio_policy='{policy}' cannot use axis {d} of length 0"
                    )));
                }
                let ratio = sizes[i] as f32 / input.shape[d] as f32;
                common = Some(match (common, policy) {
                    (None, _) => ratio,
                    (Some(cur), "not_larger") => cur.min(ratio),
                    (Some(cur), _) => cur.max(ratio),
                });
            }
            let Some(k) = common else {
                // No axes to resize: nothing to adjust.
                return Ok(());
            };
            for &d in axes {
                let width = k * input.shape[d] as f32;
                // Spec: round_int rounds halfway cases up.
                let dim = (width + 0.5).floor();
                if !(0.0..=usize::MAX as f32).contains(&dim) {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "Resize: keep_aspect_ratio_policy='{policy}' yields an out-of-range size"
                    )));
                }
                out_shape[d] = dim as usize;
                scale[d] = k;
                float_width[d] = width;
            }
            Ok(())
        }
        other => Err(OnnxError::Unsupported(format!(
            "Resize: unsupported keep_aspect_ratio_policy '{other}'"
        ))),
    }
}

// ── Per-axis resampler ──────────────────────────────────────────────────────

/// Precomputed 1-D gather/blend table for one axis.
struct AxisResampler {
    out_size: usize,
    taps: usize,
    /// `out_size * taps` source indices, already clamped into `[0, in_size)`.
    indices: Vec<usize>,
    /// `out_size * taps` weights, parallel to `indices`.
    weights: Vec<f32>,
    /// Per-output flag: coordinate fell outside the input (tf_crop_and_resize).
    extrapolated: Vec<bool>,
    any_extrapolated: bool,
}

impl AxisResampler {
    /// True when the pass would copy the axis unchanged.
    fn is_identity(&self, in_size: usize) -> bool {
        if self.any_extrapolated || self.out_size != in_size {
            return false;
        }
        for j in 0..self.out_size {
            let mut matched = false;
            for k in 0..self.taps {
                let w = self.weights[j * self.taps + k];
                if w == 0.0 {
                    continue;
                }
                if matched || w != 1.0 || self.indices[j * self.taps + k] != j {
                    return false;
                }
                matched = true;
            }
            if !matched {
                return false;
            }
        }
        true
    }
}

/// Keys' cubic convolution kernel with free parameter `a` (`cubic_coeff_a`).
fn cubic_kernel(x: f32, a: f32) -> f32 {
    let ax = x.abs();
    if ax <= 1.0 {
        ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
    } else if ax < 2.0 {
        ((a * ax - 5.0 * a) * ax + 8.0 * a) * ax - 4.0 * a
    } else {
        0.0
    }
}

fn transform_coord(dst: usize, spec: &AxisSpec, coord: CoordMode) -> f32 {
    let x = dst as f32;
    let in_len = spec.in_size as f32;
    let out_len = spec.out_size as f32;
    match coord {
        CoordMode::HalfPixel => (x + 0.5) / spec.scale - 0.5,
        CoordMode::HalfPixelSymmetric => {
            let adjustment = if spec.float_width == 0.0 {
                1.0
            } else {
                out_len / spec.float_width
            };
            let offset = (in_len / 2.0) * (1.0 - adjustment);
            offset + (x + 0.5) / spec.scale - 0.5
        }
        CoordMode::PytorchHalfPixel => {
            if spec.out_size > 1 {
                (x + 0.5) / spec.scale - 0.5
            } else {
                0.0
            }
        }
        CoordMode::AlignCorners => {
            if spec.out_size <= 1 {
                0.0
            } else {
                x * (in_len - 1.0) / (out_len - 1.0)
            }
        }
        CoordMode::Asymmetric => x / spec.scale,
        CoordMode::TfHalfPixelForNn => (x + 0.5) / spec.scale,
        CoordMode::TfCropAndResize => {
            if spec.out_size > 1 {
                spec.roi_start * (in_len - 1.0)
                    + x * (spec.roi_end - spec.roi_start) * (in_len - 1.0) / (out_len - 1.0)
            } else {
                0.5 * (spec.roi_start + spec.roi_end) * (in_len - 1.0)
            }
        }
    }
}

/// Tap offsets for the interpolation window, as `start..start + count`.
///
/// The offsets are relative to `base`, where `x_original = base + ratio` with
/// `ratio` in `(0, 1]` — the convention used by the ONNX reference kernels.
fn tap_window(spec: &AxisSpec, kernel: &KernelParams) -> Result<(i64, usize), OnnxError> {
    let plain = match kernel.interp {
        Interp::Nearest => (0i64, 1usize),
        Interp::Linear => (0i64, 2usize),
        Interp::Cubic => (-1i64, 4usize),
    };
    if !kernel.antialias {
        return Ok(plain);
    }
    // Antialiasing stretches the filter by max(1, 1/scale) when downscaling.
    let filter_scale = spec.scale.min(1.0);
    if !filter_scale.is_finite() || filter_scale <= 0.0 {
        return Err(OnnxError::InvalidModel(format!(
            "Resize: antialias requires a positive scale, got {}",
            spec.scale
        )));
    }
    let support = match kernel.interp {
        Interp::Cubic => 2.0f32,
        _ => 1.0f32,
    };
    // ONNX reference kernels: start = floor(-support / scale) + 1 and the tap
    // range is [start, 2 - start), i.e. `2 - 2 * start` symmetric taps.
    let start_f = (-support / filter_scale).floor() + 1.0;
    let count_f = 2.0 - 2.0 * start_f;
    let limit = 4.0 * spec.in_size as f32 + 16.0;
    if !start_f.is_finite() || !count_f.is_finite() || count_f < 1.0 || count_f > limit {
        return Err(OnnxError::Unsupported(format!(
            "Resize: antialias filter width {count_f} is out of range for an axis of length {}",
            spec.in_size
        )));
    }
    let count = count_f as usize;
    spec.out_size.checked_mul(count).ok_or_else(|| {
        OnnxError::ShapeMismatch("Resize: antialias filter table overflows usize".to_string())
    })?;
    Ok((start_f as i64, count))
}

fn build_axis(spec: &AxisSpec, kernel: &KernelParams) -> Result<AxisResampler, OnnxError> {
    let out_size = spec.out_size;
    if out_size == 0 || spec.in_size == 0 {
        // Nothing to sample: an empty axis produces an empty axis (validated
        // upstream), and `dim_size - 1` is never evaluated.
        return Ok(AxisResampler {
            out_size,
            taps: 1,
            indices: vec![0; out_size],
            weights: vec![0.0; out_size],
            extrapolated: Vec::new(),
            any_extrapolated: false,
        });
    }
    let max_idx = (spec.in_size - 1) as i64;
    let (offset0, taps) = tap_window(spec, kernel)?;
    let total = out_size.checked_mul(taps).ok_or_else(|| {
        OnnxError::ShapeMismatch("Resize: resampler table overflows usize".to_string())
    })?;

    let mut indices = vec![0usize; total];
    let mut weights = vec![0.0f32; total];
    let mut extrapolated = vec![false; out_size];
    let mut any_extrapolated = false;
    let filter_scale = if kernel.antialias {
        spec.scale.min(1.0)
    } else {
        1.0
    };
    let mut raw = vec![0i64; taps];

    for (j, extrapolated_j) in extrapolated.iter_mut().enumerate() {
        let src = transform_coord(j, spec, kernel.coord);
        if !src.is_finite() {
            return Err(OnnxError::Arithmetic(format!(
                "Resize: coordinate transform produced {src} on an axis of length {}",
                spec.in_size
            )));
        }
        if kernel.coord == CoordMode::TfCropAndResize && (src < 0.0 || src > max_idx as f32) {
            *extrapolated_j = true;
            any_extrapolated = true;
        }
        // Every tap is clamped into [0, max_idx] regardless, so pinning the
        // coordinate to a bounded neighbourhood of the axis preserves the result
        // while keeping the i64 tap arithmetic away from saturation (a coordinate
        // of -1e38 would otherwise saturate to i64::MIN and then overflow).
        let guard = 2.0 * taps as f32 + 4.0;
        let src = src.clamp(-guard, max_idx as f32 + guard);
        let row = j * taps;
        if kernel.interp == Interp::Nearest {
            let picked = match kernel.nearest_rule {
                NearestRule::RoundPreferFloor => (src - 0.5).ceil(),
                NearestRule::RoundPreferCeil => (src + 0.5).floor(),
                NearestRule::Floor => src.floor(),
                NearestRule::Ceil => src.ceil(),
            };
            indices[row] = (picked as i64).clamp(0, max_idx) as usize;
            weights[row] = 1.0;
            continue;
        }

        // ratio lives in (0, 1]: an exactly integral coordinate is expressed as
        // (x - 1) + 1 so that the tap set stays symmetric, matching the ONNX
        // reference implementation.
        let floor = src.floor();
        let (base, ratio) = if floor == src {
            (src as i64 - 1, 1.0f32)
        } else {
            (floor as i64, src - floor)
        };

        let mut sum = 0.0f32;
        for k in 0..taps {
            let offset = offset0 + k as i64;
            let arg = (offset as f32 - ratio) * filter_scale;
            let w = match kernel.interp {
                Interp::Linear => (1.0 - arg.abs()).clamp(0.0, 1.0),
                Interp::Cubic => cubic_kernel(arg, kernel.cubic_a),
                Interp::Nearest => 1.0,
            };
            let idx = base + offset;
            raw[k] = idx;
            weights[row + k] = w;
            indices[row + k] = idx.clamp(0, max_idx) as usize;
            sum += w;
        }
        if kernel.exclude_outside {
            let mut kept = 0.0f32;
            for k in 0..taps {
                if raw[k] >= 0 && raw[k] <= max_idx {
                    kept += weights[row + k];
                }
            }
            // If *every* tap lies outside the tensor the exclusion would leave a
            // zero kernel (the ONNX reference divides by zero here); keep the
            // edge-clamped weights instead so the output stays the border value.
            if kept != 0.0 {
                for k in 0..taps {
                    if raw[k] < 0 || raw[k] > max_idx {
                        weights[row + k] = 0.0;
                    }
                }
                sum = kept;
            }
        }
        if (kernel.antialias || kernel.exclude_outside) && sum != 0.0 {
            let inv = 1.0 / sum;
            for w in &mut weights[row..row + taps] {
                *w *= inv;
            }
        }
    }

    Ok(AxisResampler {
        out_size,
        taps,
        indices,
        weights,
        extrapolated,
        any_extrapolated,
    })
}

// ── Separable execution ─────────────────────────────────────────────────────

/// Resample a single axis of a row-major buffer.
fn run_axis_pass(
    src: &[f32],
    src_shape: &[usize],
    axis: usize,
    rs: &AxisResampler,
    dst: &mut [f32],
) {
    let n_in = src_shape[axis];
    let outer: usize = src_shape[..axis].iter().product();
    let inner: usize = src_shape[axis + 1..].iter().product();
    let n_out = rs.out_size;
    let taps = rs.taps;
    let in_plane = n_in * inner;
    let out_plane = n_out * inner;

    for o in 0..outer {
        let sbase = o * in_plane;
        let dbase = o * out_plane;
        for j in 0..n_out {
            let row = j * taps;
            let dst_row = &mut dst[dbase + j * inner..dbase + (j + 1) * inner];
            let mut initialised = false;
            for k in 0..taps {
                let w = rs.weights[row + k];
                if w == 0.0 {
                    continue;
                }
                let s = sbase + rs.indices[row + k] * inner;
                let src_row = &src[s..s + inner];
                if !initialised {
                    initialised = true;
                    if w == 1.0 {
                        dst_row.copy_from_slice(src_row);
                    } else {
                        for (d, &v) in dst_row.iter_mut().zip(src_row.iter()) {
                            *d = w * v;
                        }
                    }
                } else {
                    for (d, &v) in dst_row.iter_mut().zip(src_row.iter()) {
                        *d += w * v;
                    }
                }
            }
            if !initialised {
                dst_row.fill(0.0);
            }
        }
    }
}

fn run_passes(
    input: &Tensor,
    out_shape: &[usize],
    passes: &[(usize, AxisResampler)],
    out: &mut Tensor,
) -> Result<(), OnnxError> {
    let mut cur_shape = input.shape.clone();
    let mut prev: Vec<f32> = Vec::new();
    for (step, (axis, rs)) in passes.iter().enumerate() {
        let src_owned = std::mem::take(&mut prev);
        let src: &[f32] = if step == 0 { &input.data } else { &src_owned };
        let last = step + 1 == passes.len();
        if last {
            let mut final_shape = cur_shape.clone();
            final_shape[*axis] = rs.out_size;
            if final_shape.as_slice() != out_shape || out.data.len() != checked_product(out_shape)?
            {
                return Err(OnnxError::Internal(format!(
                    "Resize: separable plan ended at shape {final_shape:?}, expected {out_shape:?}"
                )));
            }
            run_axis_pass(src, &cur_shape, *axis, rs, &mut out.data);
        } else {
            let mut next_shape = cur_shape.clone();
            next_shape[*axis] = rs.out_size;
            let n = checked_product(&next_shape)?;
            let mut dst = vec![0.0f32; n];
            run_axis_pass(src, &cur_shape, *axis, rs, &mut dst);
            prev = dst;
        }
        cur_shape[*axis] = rs.out_size;
    }
    Ok(())
}

/// Overwrite every output position whose coordinate fell outside the crop box.
fn apply_extrapolation(
    out_shape: &[usize],
    passes: &[(usize, AxisResampler)],
    extrapolation_value: f32,
    out: &mut Tensor,
) {
    for (axis, rs) in passes {
        if !rs.any_extrapolated {
            continue;
        }
        let outer: usize = out_shape[..*axis].iter().product();
        let inner: usize = out_shape[*axis + 1..].iter().product();
        let n_out = rs.out_size;
        for o in 0..outer {
            let base = o * n_out * inner;
            for (j, &flag) in rs.extrapolated.iter().enumerate() {
                if flag {
                    out.data[base + j * inner..base + (j + 1) * inner].fill(extrapolation_value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_nearest_2x() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = resize(&input, None, Some(&[1, 1, 4, 4]), "nearest", "asymmetric")
            .expect("resize failed");
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
        #[rustfmt::skip]
        assert_eq!(out.data, vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ]);
    }

    #[test]
    fn test_resize_bilinear_2x() {
        // 1x1x2x2 -> 1x1x4x4 with bilinear, align_corners
        let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 1, 2, 2]);
        let out = resize(&input, None, Some(&[1, 1, 4, 4]), "linear", "align_corners")
            .expect("resize failed");
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
        // corners should be preserved
        assert!((out.data[0] - 0.0).abs() < 1e-5);
        assert!((out.data[3] - 1.0).abs() < 1e-5);
        assert!((out.data[12] - 2.0).abs() < 1e-5);
        assert!((out.data[15] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_resize_pytorch_half_pixel() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = resize(
            &input,
            Some(&[1.0, 1.0, 2.0, 2.0]),
            None,
            "nearest",
            "pytorch_half_pixel",
        )
        .expect("resize failed");
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_unknown_mode_is_error() {
        let input = Tensor::new(vec![1.0, 2.0], vec![2]);
        let err = resize(&input, Some(&[2.0]), None, "quintic", "half_pixel")
            .expect_err("unknown mode must fail");
        assert!(matches!(err, OnnxError::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn test_zero_length_axis_does_not_panic() {
        let input = Tensor::new(Vec::new(), vec![1, 1, 0, 4]);
        let out = resize(
            &input,
            Some(&[1.0, 1.0, 1.0, 2.0]),
            None,
            "nearest",
            "half_pixel",
        )
        .expect("zero-length axis must not panic");
        assert_eq!(out.shape, vec![1, 1, 0, 8]);
        assert!(out.data.is_empty());
    }
}
