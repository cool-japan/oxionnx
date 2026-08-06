//! Integer N-D convolution shared by `QLinearConv` and `ConvInteger`.
//!
//! Both operators compute the same accumulator —
//! `Σ (x_q - x_zero_point) * (w_q - w_zero_point)` in exact integer
//! arithmetic — and differ only in what they do with it afterwards
//! (`QLinearConv` requantizes, `ConvInteger` emits int32 directly), so the
//! geometry resolution and the accumulation live here once.
//!
//! Geometry comes from [`crate::conv::spatial`], the *same* helpers the float
//! `Conv` / `MaxPool` operators use, so a quantized graph and its float twin
//! can never disagree on `auto_pad` resolution or an output extent — and the
//! kernel is rank-generic for free (1-D quantized TCNs, 2-D vision, 3-D
//! volumetric).

use oxionnx_core::{Attributes, OnnxError, Tensor};

use crate::conv::spatial::{
    compute_conv_out_shape, odometer_next, parse_auto_pad, read_group, read_pads,
    read_positive_spatial, resolve_pads, spatial_rank,
};

use super::{lane_to_i32, tensor_to_i32};

/// Fully resolved geometry of one integer convolution node.
pub(super) struct IntConvGeometry {
    pub strides: Vec<usize>,
    /// `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]`.
    pub pads: Vec<usize>,
    pub dilations: Vec<usize>,
    pub group: usize,
    /// Full `[N, M, o_0, …]` output shape.
    pub out_shape: Vec<usize>,
}

impl IntConvGeometry {
    /// Read and validate every spatial attribute, applying `auto_pad`.
    pub fn from_attrs(
        attrs: &Attributes,
        input_shape: &[usize],
        weight_shape: &[usize],
        op: &str,
    ) -> Result<Self, OnnxError> {
        let rank = spatial_rank(input_shape, op, "input")?;
        if weight_shape.len() != input_shape.len() {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: weight rank {} must equal input rank {} ([M, C/group, k_0, ...])",
                weight_shape.len(),
                input_shape.len()
            )));
        }
        let strides = read_positive_spatial(attrs.ints("strides"), rank, 1, "strides", op)?;
        let dilations = read_positive_spatial(attrs.ints("dilations"), rank, 1, "dilations", op)?;
        let group = read_group(attrs, op)?;

        let c_in = input_shape[1];
        let c_per_group = weight_shape[1];
        if c_per_group.checked_mul(group) != Some(c_in) {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: input channels {c_in} != weight input channels {c_per_group} * group {group}"
            )));
        }
        if weight_shape[0] % group != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: output channels {} not divisible by group {group}",
                weight_shape[0]
            )));
        }

        let explicit = read_pads(attrs.ints("pads"), rank, op)?;
        let auto_pad = parse_auto_pad(attrs.s("auto_pad"), op)?;
        let pads = resolve_pads(
            auto_pad,
            &input_shape[2..],
            &weight_shape[2..],
            &strides,
            &dilations,
            &explicit,
        );
        let out_shape =
            compute_conv_out_shape(op, input_shape, weight_shape, &strides, &pads, &dilations)?;
        Ok(Self {
            strides,
            pads,
            dilations,
            group,
            out_shape,
        })
    }
}

/// Zero points of one integer convolution, already decoded to `i32`.
pub(super) struct ConvZeroPoints {
    /// Per-tensor input zero point.
    pub x: i32,
    /// Per-tensor (`len == 1`) or per-output-channel weight zero points.
    pub w: Vec<i32>,
}

impl ConvZeroPoints {
    /// Weight zero point for output channel `oc`.
    #[inline]
    fn w_for(&self, oc: usize) -> i32 {
        if self.w.len() == 1 {
            self.w[0]
        } else {
            self.w[oc]
        }
    }
}

/// Result of an integer convolution: `[N, M, o_0, …]` accumulators in `i64`.
pub(super) struct IntConvOutput {
    pub acc: Vec<i64>,
    pub shape: Vec<usize>,
}

/// Row-major strides of the spatial axes of a `[N, C, d_0, …]` tensor.
fn spatial_strides(spatial: &[usize]) -> Vec<usize> {
    let rank = spatial.len();
    let mut strides = vec![1_usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * spatial[d + 1];
    }
    strides
}

/// Integer N-D convolution `Σ (x_q - x_zp) * (w_q - w_zp)`, optionally with an
/// int32 bias added into the accumulator (`QLinearConv`'s `B` input, which the
/// spec defines as already living in the `x_scale * w_scale` domain).
///
/// Accumulation is in `i64`: products reach `255 * 255` and a realistic kernel
/// volume pushes the sum past f32's exact-integer range, so an f32 accumulator
/// would silently drop low bits.
pub(super) fn integer_conv(
    x: &Tensor,
    w: &Tensor,
    zero_points: &ConvZeroPoints,
    bias: Option<&Tensor>,
    geo: &IntConvGeometry,
    op: &str,
) -> Result<IntConvOutput, OnnxError> {
    let rank = geo.strides.len();
    let batch = x.shape[0];
    let c_in = x.shape[1];
    let in_spatial = &x.shape[2..];
    let c_out = w.shape[0];
    let c_per_group = w.shape[1];
    let kernel = &w.shape[2..];
    let out_spatial = &geo.out_shape[2..];

    if zero_points.w.len() != 1 && zero_points.w.len() != c_out {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: w_zero_point has {} entries, expected 1 or {c_out} (per output channel)",
            zero_points.w.len()
        )));
    }
    let expected_x: usize = x.shape.iter().product();
    if x.data.len() != expected_x {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: input data length {} does not match shape {:?}",
            x.data.len(),
            x.shape
        )));
    }
    let expected_w: usize = w.shape.iter().product();
    if w.data.len() != expected_w {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: weight data length {} does not match shape {:?}",
            w.data.len(),
            w.shape
        )));
    }

    let bias_i64: Vec<i64> = match bias {
        None => Vec::new(),
        Some(b) => {
            if b.data.len() != c_out {
                return Err(OnnxError::ShapeMismatch(format!(
                    "{op}: bias has {} entries, expected {c_out}",
                    b.data.len()
                )));
            }
            let label = format!("{op}: bias");
            b.data
                .iter()
                .map(|&v| lane_to_i32(v, &label).map(i64::from))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    let x_i32 = tensor_to_i32(x, &format!("{op}: input"))?;
    let w_i32 = tensor_to_i32(w, &format!("{op}: weight"))?;

    let c_out_per_group = c_out / geo.group;
    let in_stride = spatial_strides(in_spatial);
    let in_plane: usize = in_spatial.iter().product();
    let out_plane: usize = out_spatial.iter().product();
    let k_volume: usize = kernel.iter().product();
    let x_zp = i64::from(zero_points.x);

    let mut acc = vec![0_i64; batch * c_out * out_plane];
    let mut oidx = vec![0_usize; rank];
    let mut kidx = vec![0_usize; rank];

    for n in 0..batch {
        for oc in 0..c_out {
            let g = oc / c_out_per_group;
            let w_zp = i64::from(zero_points.w_for(oc));
            let w_base = oc * c_per_group * k_volume;
            let out_base = (n * c_out + oc) * out_plane;
            let bias_term = bias_i64.get(oc).copied().unwrap_or(0);

            oidx.iter_mut().for_each(|v| *v = 0);
            for o_flat in 0..out_plane {
                let mut sum = bias_term;
                kidx.iter_mut().for_each(|v| *v = 0);
                for k_flat in 0..k_volume {
                    // Resolve this kernel tap's input coordinate; a tap that
                    // lands in the padding contributes exactly 0 because the
                    // zero point *is* the quantized encoding of real 0.
                    let mut in_off = 0_usize;
                    let mut inside = true;
                    for d in 0..rank {
                        let pos = oidx[d] * geo.strides[d] + kidx[d] * geo.dilations[d];
                        match pos.checked_sub(geo.pads[d]) {
                            Some(ip) if ip < in_spatial[d] => in_off += ip * in_stride[d],
                            _ => {
                                inside = false;
                                break;
                            }
                        }
                    }
                    if inside {
                        for ic in 0..c_per_group {
                            let in_c = g * c_per_group + ic;
                            let plane = (n * c_in + in_c) * in_plane;
                            let xv = i64::from(x_i32[plane + in_off]);
                            let wv = i64::from(w_i32[w_base + ic * k_volume + k_flat]);
                            sum += (xv - x_zp) * (wv - w_zp);
                        }
                    }
                    odometer_next(&mut kidx, kernel);
                }
                acc[out_base + o_flat] = sum;
                odometer_next(&mut oidx, out_spatial);
            }
        }
    }

    Ok(IntConvOutput {
        acc,
        shape: geo.out_shape.clone(),
    })
}
