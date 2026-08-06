//! Pooling: one rank-generic `MaxPool` / `AveragePool` kernel plus the global
//! reductions.
//!
//! There is exactly **one** implementation of each pooling reduction. It takes
//! a fully resolved [`PoolGeometry`] — `auto_pad` already applied, `ceil_mode`
//! and `dilations` already folded into the output extents — and is shared by:
//!
//! * the `MaxPool` / `AveragePool` operators in
//!   `crate::registry::conv_ops::pooling`, which build the geometry from the
//!   node attributes for any spatial rank, and
//! * the raw [`max_pool2d`] / [`avg_pool2d`] entry points, thin rank-2 compat
//!   wrappers that pin `dilations = 1` and `ceil_mode = false` (the pre-N-D
//!   behaviour benches and downstream callers rely on).
//!
//! The previous split — a legacy dilation-blind pair here and a second,
//! spec-complete pair inside the registry — is gone.

use oxionnx_core::{OnnxError, Tensor};

use super::spatial::{self, odometer_next};

// ── Geometry ────────────────────────────────────────────────────────────────

/// Fully resolved N-D pooling geometry.
///
/// `pads` uses the ONNX layout `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]`;
/// `out` holds the spatial output extents only (the `[N, C]` prefix comes from
/// the input shape).
#[derive(Clone, Debug)]
pub(crate) struct PoolGeometry {
    pub kernel: Vec<usize>,
    pub strides: Vec<usize>,
    pub pads: Vec<usize>,
    pub dilations: Vec<usize>,
    pub out: Vec<usize>,
}

impl PoolGeometry {
    /// Resolve the output extents for already-validated attribute vectors.
    ///
    /// Returns a typed [`OnnxError::ShapeMismatch`] for any combination with no
    /// valid window (zero stride, kernel larger than the padded input, an
    /// overflowing extent).
    pub(crate) fn resolve(
        op: &str,
        input_spatial: &[usize],
        kernel: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
        dilations: Vec<usize>,
        ceil_mode: bool,
    ) -> Result<Self, OnnxError> {
        let rank = input_spatial.len();
        if kernel.len() != rank
            || strides.len() != rank
            || dilations.len() != rank
            || pads.len() != 2 * rank
        {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: kernel_shape/strides/dilations need {rank} entries and pads needs {} \
                 (got {}/{}/{}/{})",
                2 * rank,
                kernel.len(),
                strides.len(),
                dilations.len(),
                pads.len()
            )));
        }
        let mut out = Vec::with_capacity(rank);
        for axis in 0..rank {
            out.push(spatial::pool_out_dim(
                op,
                spatial::axis_label(rank, axis),
                input_spatial[axis],
                pads[axis],
                pads[axis + rank],
                kernel[axis],
                dilations[axis],
                strides[axis],
                ceil_mode,
            )?);
        }
        Ok(Self {
            kernel,
            strides,
            pads,
            dilations,
            out,
        })
    }

    /// Spatial rank.
    pub(crate) fn rank(&self) -> usize {
        self.kernel.len()
    }

    /// `[N, C, o_0, …]` for the given input shape.
    pub(crate) fn out_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let mut shape = Vec::with_capacity(self.out.len() + 2);
        shape.push(input_shape.first().copied().unwrap_or(0));
        shape.push(input_shape.get(1).copied().unwrap_or(0));
        shape.extend_from_slice(&self.out);
        shape
    }

    /// Total number of output elements for the given input shape.
    pub(crate) fn out_len(&self, input_shape: &[usize]) -> usize {
        self.out_shape(input_shape).iter().product()
    }
}

/// Per-plane scratch describing how the last spatial axis is walked.
struct LastAxis {
    stride: usize,
    dilation: usize,
    pad: usize,
    extent: usize,
    kernel: usize,
}

/// Resolve the leading spatial axes of one kernel row.
///
/// Returns the in-plane offset contributed by axes `0..rank-1`, or `None` when
/// any of them lands outside the input (i.e. the whole row is padding).
fn leading_offset(
    geo: &PoolGeometry,
    in_spatial: &[usize],
    in_stride: &[usize],
    oidx: &[usize],
    kidx: &[usize],
) -> Option<usize> {
    let mut off = 0_usize;
    for d in 0..kidx.len() {
        let pos = oidx[d] * geo.strides[d] + kidx[d] * geo.dilations[d];
        let ip = pos.checked_sub(geo.pads[d])?;
        if ip >= in_spatial[d] {
            return None;
        }
        off += ip * in_stride[d];
    }
    Some(off)
}

/// Row-major strides of the spatial axes of a `[N, C, d_0, …]` tensor.
fn spatial_strides(in_spatial: &[usize]) -> Vec<usize> {
    let rank = in_spatial.len();
    let mut strides = vec![1_usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * in_spatial[d + 1];
    }
    strides
}

// ── MaxPool ─────────────────────────────────────────────────────────────────

/// Rank-generic MaxPool over `[N, C, d_0, …]`.
///
/// `indices` is written only when non-empty; each entry is the flattened index
/// of the winning input element, encoded row-major
/// (`((n*C + c) * d_0 + i_0) * d_1 + i_1 …`) or, for spatial rank 2 with
/// `column_major`, as `((n*C + c) * W + ix) * H + iy`.
///
/// A window that samples no in-bounds element at all (reachable at `ceil_mode`
/// edges) yields `f32::NEG_INFINITY` and index `0`, matching the pre-N-D
/// behaviour of both former implementations.
pub(crate) fn max_pool_into(
    input: &[f32],
    input_shape: &[usize],
    geo: &PoolGeometry,
    out: &mut [f32],
    indices: &mut [f32],
    column_major: bool,
) {
    let Some((n, c, in_spatial)) = split_shape(input_shape, geo) else {
        out.fill(f32::NEG_INFINITY);
        indices.fill(0.0_f32);
        return;
    };
    let rank = geo.rank();
    let last = rank - 1;
    let in_plane: usize = in_spatial.iter().product();
    let out_plane: usize = geo.out.iter().product();
    let in_stride = spatial_strides(in_spatial);
    let want_indices = !indices.is_empty();
    let last_axis = LastAxis {
        stride: geo.strides[last],
        dilation: geo.dilations[last],
        pad: geo.pads[last],
        extent: in_spatial[last],
        kernel: geo.kernel[last],
    };
    let k_outer: usize = geo.kernel[..last].iter().product();

    let mut oidx = vec![0_usize; rank];
    let mut kidx = vec![0_usize; last];

    for nc in 0..n * c {
        let plane = nc * in_plane;
        let out_base = nc * out_plane;
        oidx.iter_mut().for_each(|v| *v = 0);
        for oflat in 0..out_plane {
            let o_last = oidx[last];
            let mut max_val = f32::NEG_INFINITY;
            let mut best: Option<usize> = None;
            let mut first: Option<usize> = None;
            kidx.iter_mut().for_each(|v| *v = 0);
            for _ in 0..k_outer {
                if let Some(off) = leading_offset(geo, in_spatial, &in_stride, &oidx, &kidx) {
                    for kl in 0..last_axis.kernel {
                        let pos = o_last * last_axis.stride + kl * last_axis.dilation;
                        if pos < last_axis.pad {
                            continue;
                        }
                        let ip = pos - last_axis.pad;
                        if ip >= last_axis.extent {
                            continue;
                        }
                        let sample = off + ip;
                        if first.is_none() {
                            first = Some(sample);
                        }
                        let v = input[plane + sample];
                        if v > max_val {
                            max_val = v;
                            best = Some(sample);
                        }
                    }
                }
                if odometer_next(&mut kidx, &geo.kernel[..last]) {
                    break;
                }
            }
            let o = out_base + oflat;
            out[o] = max_val;
            if want_indices {
                indices[o] = match best.or(first) {
                    Some(sample) => {
                        let flat = if column_major && rank == 2 {
                            let width = in_spatial[1];
                            let iy = sample / width;
                            let ix = sample % width;
                            (nc * width + ix) * in_spatial[0] + iy
                        } else {
                            plane + sample
                        };
                        flat as f32
                    }
                    None => 0.0_f32,
                };
            }
            if odometer_next(&mut oidx, &geo.out) {
                break;
            }
        }
    }
}

// ── AveragePool ─────────────────────────────────────────────────────────────

/// Rank-generic AveragePool over `[N, C, d_0, …]`.
///
/// With `count_include_pad` the divisor is the full kernel volume; otherwise
/// only the in-bounds sampled positions are counted. A window that samples no
/// in-bounds element at all (reachable at `ceil_mode` edges) produces `0.0`
/// instead of dividing by zero.
///
/// NOTE — deliberate ONNX Runtime parity, do not "fix" to the PyTorch answer:
/// when `count_include_pad = 1` meets a `ceil_mode` window that hangs past the
/// declared right padding, ORT still divides by `prod(kernel)` while PyTorch
/// clamps the window to `in + pad_end` and divides by the clamped size. The
/// ONNX spec does not settle it and there is no node test, so we follow ORT.
pub(crate) fn avg_pool_into(
    input: &[f32],
    input_shape: &[usize],
    geo: &PoolGeometry,
    count_include_pad: bool,
    out: &mut [f32],
) {
    let Some((n, c, in_spatial)) = split_shape(input_shape, geo) else {
        out.fill(0.0_f32);
        return;
    };
    let rank = geo.rank();
    let last = rank - 1;
    let in_plane: usize = in_spatial.iter().product();
    let out_plane: usize = geo.out.iter().product();
    let in_stride = spatial_strides(in_spatial);
    let full_window: usize = geo.kernel.iter().product();
    let last_axis = LastAxis {
        stride: geo.strides[last],
        dilation: geo.dilations[last],
        pad: geo.pads[last],
        extent: in_spatial[last],
        kernel: geo.kernel[last],
    };
    let k_outer: usize = geo.kernel[..last].iter().product();

    let mut oidx = vec![0_usize; rank];
    let mut kidx = vec![0_usize; last];

    for nc in 0..n * c {
        let plane = nc * in_plane;
        let out_base = nc * out_plane;
        oidx.iter_mut().for_each(|v| *v = 0);
        for oflat in 0..out_plane {
            let o_last = oidx[last];
            let mut sum = 0.0_f32;
            let mut count = 0_usize;
            kidx.iter_mut().for_each(|v| *v = 0);
            for _ in 0..k_outer {
                if let Some(off) = leading_offset(geo, in_spatial, &in_stride, &oidx, &kidx) {
                    for kl in 0..last_axis.kernel {
                        let pos = o_last * last_axis.stride + kl * last_axis.dilation;
                        if pos < last_axis.pad {
                            continue;
                        }
                        let ip = pos - last_axis.pad;
                        if ip >= last_axis.extent {
                            continue;
                        }
                        sum += input[plane + off + ip];
                        count += 1;
                    }
                }
                if odometer_next(&mut kidx, &geo.kernel[..last]) {
                    break;
                }
            }
            let divisor = if count_include_pad {
                full_window
            } else {
                count
            };
            out[out_base + oflat] = if divisor > 0 {
                sum / divisor as f32
            } else {
                0.0_f32
            };
            if odometer_next(&mut oidx, &geo.out) {
                break;
            }
        }
    }
}

/// Split `[N, C, d_0, …]` into `(N, C, spatial)`, rejecting a shape whose
/// spatial rank disagrees with the geometry.
fn split_shape<'a>(
    input_shape: &'a [usize],
    geo: &PoolGeometry,
) -> Option<(usize, usize, &'a [usize])> {
    if geo.rank() == 0 || input_shape.len() != geo.rank() + 2 {
        return None;
    }
    Some((input_shape[0], input_shape[1], &input_shape[2..]))
}

// ── Rank-2 compatibility wrappers ───────────────────────────────────────────

/// 2D max pooling (floor output mode, dilation 1).
///
/// A thin wrapper over the shared `max_pool_into` kernel; the `MaxPool`
/// operator in the registry drives the same kernel with the full spec
/// (`auto_pad`, `ceil_mode`, `dilations`, `storage_order`, `Indices`) resolved
/// from the node attributes.
///
/// Degenerate parameters (non-4D input, zero stride, kernel larger than the
/// padded input) return an empty `[N, C, 0, 0]` tensor rather than panicking.
/// input: [N, C, H, W]
pub fn max_pool2d(
    input: &Tensor,
    kernel_shape: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
) -> Tensor {
    let Some(geo) = compat_geometry("MaxPool", &input.shape, kernel_shape, strides, pads) else {
        return degenerate_out(&input.shape);
    };
    let total = geo.out_len(&input.shape);
    let mut values = vec![f32::NEG_INFINITY; total];
    let mut no_indices: [f32; 0] = [];
    max_pool_into(
        &input.data,
        &input.shape,
        &geo,
        &mut values,
        &mut no_indices,
        false,
    );
    Tensor::new(values, geo.out_shape(&input.shape))
}

/// 2D average pooling (floor output mode, dilation 1).
///
/// A thin wrapper over the shared `avg_pool_into` kernel; the `AveragePool`
/// operator in the registry drives the same kernel with the full spec
/// (`auto_pad`, `ceil_mode`, `dilations`, `count_include_pad`) resolved from
/// the node attributes.
///
/// Degenerate parameters (non-4D input, zero stride, kernel larger than the
/// padded input) return an empty `[N, C, 0, 0]` tensor rather than panicking.
/// input: [N, C, H, W]
pub fn avg_pool2d(
    input: &Tensor,
    kernel_shape: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
    count_include_pad: bool,
) -> Tensor {
    let Some(geo) = compat_geometry("AveragePool", &input.shape, kernel_shape, strides, pads)
    else {
        return degenerate_out(&input.shape);
    };
    let total = geo.out_len(&input.shape);
    let mut values = vec![0.0_f32; total];
    avg_pool_into(
        &input.data,
        &input.shape,
        &geo,
        count_include_pad,
        &mut values,
    );
    Tensor::new(values, geo.out_shape(&input.shape))
}

/// Rank-2 geometry for the compat wrappers: floor mode, dilation 1, explicit
/// `[top, left, bottom, right]` padding.
fn compat_geometry(
    op: &str,
    input_shape: &[usize],
    kernel_shape: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
) -> Option<PoolGeometry> {
    if input_shape.len() != 4 {
        return None;
    }
    PoolGeometry::resolve(
        op,
        &input_shape[2..],
        kernel_shape.to_vec(),
        strides.to_vec(),
        pads.to_vec(),
        vec![1, 1],
        false,
    )
    .ok()
}

/// Empty result for a degenerate pooling request (documented, never a panic).
fn degenerate_out(input_shape: &[usize]) -> Tensor {
    if input_shape.len() != 4 {
        return Tensor::new(Vec::new(), vec![0, 0, 0, 0]);
    }
    Tensor::new(Vec::new(), vec![input_shape[0], input_shape[1], 0, 0])
}

// ── Global pooling ──────────────────────────────────────────────────────────

/// Global average pooling: reduce all spatial dimensions to 1.
/// Input: [N, C, d0, d1, ...] → Output: [N, C, 1, 1, ...]
pub fn global_avg_pool(x: &Tensor) -> Tensor {
    if x.ndim() < 3 {
        return x.clone();
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();
    let mut out = vec![0.0f32; n * c];
    for ni in 0..n {
        for ci in 0..c {
            let base = ni * c * spatial + ci * spatial;
            let sum: f32 = x.data[base..base + spatial].iter().sum();
            out[ni * c + ci] = sum / spatial as f32;
        }
    }
    let mut out_shape = vec![n, c];
    out_shape.extend(vec![1usize; x.ndim() - 2]);
    Tensor::new(out, out_shape)
}

/// Global max pooling: reduce all spatial dimensions to 1.
/// Input: [N, C, d0, d1, ...] → Output: [N, C, 1, 1, ...]
pub fn global_max_pool(x: &Tensor) -> Tensor {
    if x.ndim() < 3 {
        return x.clone();
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();
    let mut out = vec![f32::NEG_INFINITY; n * c];
    for ni in 0..n {
        for ci in 0..c {
            let base = ni * c * spatial + ci * spatial;
            let max_val = x.data[base..base + spatial]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            out[ni * c + ci] = max_val;
        }
    }
    let mut out_shape = vec![n, c];
    out_shape.extend(vec![1usize; x.ndim() - 2]);
    Tensor::new(out, out_shape)
}
