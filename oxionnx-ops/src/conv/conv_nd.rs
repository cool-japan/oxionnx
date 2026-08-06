//! Rank-generic convolution: `Conv1D`, `Conv2D`, `Conv3D` and beyond.
//!
//! The engine keeps one hand-tuned kernel — the rank-2 [`super::conv2d`] path
//! with its 1×1, Winograd, SIMD-im2col and parallel-GEMM specialisations — and
//! reaches every other rank through this module:
//!
//! * **rank 1** (`[N, C, W]`) is *lowered* to `[N, C, 1, W]` and run on the 2D
//!   kernel. With `kH = 1`, `pad_top = pad_bottom = 0`, `stride_h = 1` and
//!   `dilation_h = 1` the output height is `(1 + 0 + 0 - 1) / 1 + 1 = 1` and
//!   the im2col window degenerates to the 1D window, so the lowering is exact,
//!   not an approximation — and it costs no copy, because the buffers are
//!   re-interpreted, not reshaped. Keeping `W` as the *last* axis also keeps it
//!   contiguous, so the SIMD stride-1 im2col still applies.
//! * **rank ≥ 3** runs a generic im2col + GEMM built on an N-D gather.
//!
//! `pads` follows the ONNX layout `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]`
//! throughout; see [`super::spatial`].

use oxionnx_core::{OnnxError, Tensor};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use super::conv2d::{conv2d_into_slices, sgemm_maybe_parallel};
use super::spatial::{self, odometer_next};

/// Validated spatial parameters of an N-D convolution.
pub(crate) struct ConvParams<'a> {
    /// One stride per spatial axis.
    pub strides: &'a [usize],
    /// `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]`.
    pub pads: &'a [usize],
    /// One dilation per spatial axis.
    pub dilations: &'a [usize],
    /// Channel-group count (`1` for a dense convolution).
    pub group: usize,
}

/// Rank-generic convolution into a pre-allocated buffer.
///
/// `input` is `[N, C, d_0, …]`, `weight` is `[F, C/group, k_0, …]`, `bias` is
/// `[F]` and `out_shape` must be the shape
/// [`spatial::compute_conv_out_shape`] derives for the same parameters.
///
/// Returns a typed [`OnnxError`] — never panics — for any rank / channel /
/// buffer-length combination a malformed model can produce.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_into(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    params: &ConvParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<(), OnnxError> {
    let rank = validate(
        "Conv",
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        params,
        out,
        out_shape,
    )?;

    match rank {
        1 => {
            // [N, C, W] → [N, C, 1, W]; the buffers are untouched.
            let input4 = [input_shape[0], input_shape[1], 1, input_shape[2]];
            let weight4 = [weight_shape[0], weight_shape[1], 1, weight_shape[2]];
            let out4 = [out_shape[0], out_shape[1], 1, out_shape[2]];
            conv2d_into_slices(
                input,
                &input4,
                weight,
                &weight4,
                bias,
                [1, params.strides[0]],
                [0, params.pads[0], 0, params.pads[1]],
                [1, params.dilations[0]],
                params.group,
                out,
                &out4,
            );
            Ok(())
        }
        2 => {
            conv2d_into_slices(
                input,
                input_shape,
                weight,
                weight_shape,
                bias,
                [params.strides[0], params.strides[1]],
                [
                    params.pads[0],
                    params.pads[1],
                    params.pads[2],
                    params.pads[3],
                ],
                [params.dilations[0], params.dilations[1]],
                params.group,
                out,
                out_shape,
            );
            Ok(())
        }
        _ => {
            conv_nd_into(
                input,
                input_shape,
                weight,
                weight_shape,
                bias,
                params,
                out,
                out_shape,
            );
            Ok(())
        }
    }
}

/// Rank-generic convolution returning a fresh tensor.
///
/// `pads` uses the ONNX layout `[begin_0, …, end_{r-1}]`; `strides`,
/// `dilations` have one entry per spatial axis. Rank 2 with the classic
/// `[top, left, bottom, right]` array is exactly the same call, since that
/// array *is* the ONNX layout for `r == 2`.
pub fn conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: &[usize],
    pads: &[usize],
    dilations: &[usize],
    group: usize,
) -> Result<Tensor, OnnxError> {
    let out_shape = spatial::compute_conv_out_shape(
        "Conv",
        &input.shape,
        &weight.shape,
        strides,
        pads,
        dilations,
    )?;
    let out_len: usize = out_shape.iter().product();
    let mut data = vec![0.0_f32; out_len];
    conv_into(
        &input.data,
        &input.shape,
        &weight.data,
        &weight.shape,
        bias.map(|b| b.data.as_slice()),
        &ConvParams {
            strides,
            pads,
            dilations,
            group,
        },
        &mut data,
        &out_shape,
    )?;
    Ok(Tensor::new(data, out_shape))
}

/// Shared rank / channel / buffer validation for the N-D conv entry points.
#[allow(clippy::too_many_arguments)]
fn validate(
    op: &str,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    params: &ConvParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<usize, OnnxError> {
    let rank = spatial::spatial_rank(input_shape, op, "input")?;
    if weight_shape.len() != input_shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: weight rank {} must equal input rank {} ([F, C/group, k_0, ...])",
            weight_shape.len(),
            input_shape.len()
        )));
    }
    if out_shape.len() != input_shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: output rank {} must equal input rank {}",
            out_shape.len(),
            input_shape.len()
        )));
    }
    if params.group == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: group must be >= 1, got 0"
        )));
    }
    if params.strides.len() != rank || params.dilations.len() != rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: strides ({}) and dilations ({}) need {rank} entries",
            params.strides.len(),
            params.dilations.len()
        )));
    }
    if params.pads.len() != 2 * rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: pads needs {} entries, got {}",
            2 * rank,
            params.pads.len()
        )));
    }
    let c_in = input_shape[1];
    let c_out = weight_shape[0];
    let c_per_group = weight_shape[1];
    if c_per_group.checked_mul(params.group) != Some(c_in) {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: input channels {c_in} != weight input channels {c_per_group} * group {}",
            params.group
        )));
    }
    if c_out % params.group != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: output channels {c_out} not divisible by group {}",
            params.group
        )));
    }
    let in_len = checked_volume(input_shape, op, "input")?;
    let w_len = checked_volume(weight_shape, op, "weight")?;
    let out_len = checked_volume(out_shape, op, "output")?;
    if input.len() < in_len || weight.len() < w_len || out.len() < out_len {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: buffer too small for its shape (input {} < {in_len}, weight {} < {w_len}, \
             output {} < {out_len})",
            input.len(),
            weight.len(),
            out.len()
        )));
    }
    if let Some(b) = bias {
        if b.len() < c_out {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: bias has {} entries, expected {c_out}",
                b.len()
            )));
        }
    }
    Ok(rank)
}

/// Element count of a shape, rejecting an overflowing product.
fn checked_volume(shape: &[usize], op: &str, what: &str) -> Result<usize, OnnxError> {
    shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("{op}: {what} shape {shape:?} overflows usize"))
        })
}

/// Generic (rank ≥ 3) im2col + GEMM convolution.
///
/// Every argument has already been validated by [`validate`]; this function
/// performs no further checks and cannot panic on the values it accepts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_nd_into(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    params: &ConvParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) {
    let rank = input_shape.len() - 2;
    let n = input_shape[0];
    let c_in = input_shape[1];
    let c_out = weight_shape[0];
    let c_per_group = weight_shape[1];
    let in_spatial = &input_shape[2..];
    let kernel = &weight_shape[2..];
    let out_spatial = &out_shape[2..];

    let ksize: usize = kernel.iter().product();
    let col_rows = c_per_group * ksize;
    let col_cols: usize = out_spatial.iter().product();
    let c_out_per_group = c_out / params.group;
    let group = params.group;
    let pads_begin = &params.pads[..rank];

    let out_len: usize = out_shape.iter().product();
    if out_len == 0 || col_cols == 0 || col_rows == 0 {
        let fill_len = out_len.min(out.len());
        out[..fill_len].fill(0.0_f32);
        return;
    }

    let total_jobs = n * group;

    // Multiple independent (batch, group) slices → one rayon job each, exactly
    // as the rank-2 kernel does.
    #[cfg(not(target_arch = "wasm32"))]
    if total_jobs > 1 {
        let job = |idx: usize| -> Vec<f32> {
            let batch = idx / group;
            let g = idx % group;
            let mut col = vec![0.0f32; col_rows * col_cols];
            im2col_nd(
                input,
                c_in,
                in_spatial,
                g * c_per_group,
                c_per_group,
                kernel,
                params.strides,
                pads_begin,
                params.dilations,
                out_spatial,
                batch,
                &mut col,
            );
            let mut job_out = vec![0.0f32; c_out_per_group * col_cols];
            let w_off = g * c_out_per_group * col_rows;
            // SAFETY: `weight[w_off..]` is [c_out_per_group, col_rows],
            // `col` is [col_rows, col_cols] and `job_out` is exactly
            // [c_out_per_group, col_cols]; all row-major.
            #[allow(unsafe_code)]
            unsafe {
                matrixmultiply::sgemm(
                    c_out_per_group,
                    col_rows,
                    col_cols,
                    1.0,
                    weight[w_off..].as_ptr(),
                    col_rows as isize,
                    1,
                    col.as_ptr(),
                    col_cols as isize,
                    1,
                    0.0,
                    job_out.as_mut_ptr(),
                    col_cols as isize,
                    1,
                );
            }
            add_bias(bias, g, c_out_per_group, col_cols, &mut job_out);
            job_out
        };
        let job_outputs: Vec<Vec<f32>> = (0..total_jobs).into_par_iter().map(job).collect();
        for (idx, job_out) in job_outputs.into_iter().enumerate() {
            let batch = idx / group;
            let g = idx % group;
            let o_off = (batch * c_out + g * c_out_per_group) * col_cols;
            out[o_off..o_off + job_out.len()].copy_from_slice(&job_out);
        }
        return;
    }

    // Single slice (or WASM): parallelise inside the job instead.
    let mut col = vec![0.0f32; col_rows * col_cols];
    for batch in 0..n {
        for g in 0..group {
            im2col_nd_maybe_parallel(
                input,
                c_in,
                in_spatial,
                g * c_per_group,
                c_per_group,
                kernel,
                params.strides,
                pads_begin,
                params.dilations,
                out_spatial,
                batch,
                &mut col,
            );
            let w_off = g * c_out_per_group * col_rows;
            let o_off = (batch * c_out + g * c_out_per_group) * col_cols;
            sgemm_maybe_parallel(
                c_out_per_group,
                col_rows,
                col_cols,
                &weight[w_off..],
                &col,
                &mut out[o_off..],
            );
            add_bias(
                bias,
                g,
                c_out_per_group,
                col_cols,
                &mut out[o_off..o_off + c_out_per_group * col_cols],
            );
        }
    }
}

/// Add the per-output-channel bias of group `g` to a `[c_out_per_group, cols]`
/// result block.
fn add_bias(
    bias: Option<&[f32]>,
    g: usize,
    c_out_per_group: usize,
    cols: usize,
    block: &mut [f32],
) {
    let Some(b) = bias else {
        return;
    };
    for oc in 0..c_out_per_group {
        let bv = b.get(g * c_out_per_group + oc).copied().unwrap_or(0.0_f32);
        let start = oc * cols;
        for v in &mut block[start..start + cols] {
            *v += bv;
        }
    }
}

/// Build the N-D im2col column matrix for one (batch, group) slice.
///
/// Row `ic * prod(kernel) + kflat` holds input channel `in_c_start + ic` at
/// kernel offset `kflat` (row-major over the kernel axes); column `oflat` is
/// output position `oflat` (row-major over the output spatial axes). That is
/// exactly the layout `[F, C/group, k_0, …]` weights expect from a GEMM, and
/// for `rank == 2` it is byte-identical to what `im2col_adaptive` produces.
///
/// The last spatial axis is walked in an inner loop over a contiguous input
/// run, so the gather stays cache-friendly for the common 3D case.
#[allow(clippy::too_many_arguments)]
fn im2col_nd(
    input: &[f32],
    c_in: usize,
    in_spatial: &[usize],
    in_c_start: usize,
    c_per_group: usize,
    kernel: &[usize],
    strides: &[usize],
    pads_begin: &[usize],
    dilations: &[usize],
    out_spatial: &[usize],
    batch: usize,
    col: &mut [f32],
) {
    let rank = in_spatial.len();
    if rank == 0 {
        return;
    }
    let last = rank - 1;
    let ksize: usize = kernel.iter().product();
    let col_cols: usize = out_spatial.iter().product();
    let in_plane: usize = in_spatial.iter().product();
    if ksize == 0 || col_cols == 0 {
        return;
    }

    // Row-major strides of the input spatial axes.
    let mut in_stride = vec![1_usize; rank];
    for d in (0..last).rev() {
        in_stride[d] = in_stride[d + 1] * in_spatial[d + 1];
    }

    let out_last = out_spatial[last];
    let in_last = in_spatial[last];
    let stride_last = strides[last];
    let dilation_last = dilations[last];
    let pad_last = pads_begin[last];
    let outer_cols: usize = out_spatial[..last].iter().product();

    let mut kidx = vec![0_usize; rank];
    let mut oidx = vec![0_usize; last];

    for ic in 0..c_per_group {
        let plane_off = (batch * c_in + in_c_start + ic) * in_plane;
        kidx.iter_mut().for_each(|v| *v = 0);
        for kflat in 0..ksize {
            let base = (ic * ksize + kflat) * col_cols;
            oidx.iter_mut().for_each(|v| *v = 0);
            for outer in 0..outer_cols {
                // Resolve the leading spatial axes once per output row.
                let mut ok = true;
                let mut off = plane_off;
                for d in 0..last {
                    let pos = oidx[d] * strides[d] + kidx[d] * dilations[d];
                    if pos < pads_begin[d] {
                        ok = false;
                        break;
                    }
                    let ip = pos - pads_begin[d];
                    if ip >= in_spatial[d] {
                        ok = false;
                        break;
                    }
                    off += ip * in_stride[d];
                }
                let dst = base + outer * out_last;
                let run = &mut col[dst..dst + out_last];
                if !ok {
                    run.fill(0.0_f32);
                } else {
                    let k_off = kidx[last] * dilation_last;
                    for (o, slot) in run.iter_mut().enumerate() {
                        let pos = o * stride_last + k_off;
                        *slot = if pos >= pad_last && pos - pad_last < in_last {
                            input[off + (pos - pad_last)]
                        } else {
                            0.0_f32
                        };
                    }
                }
                // An empty odometer (rank 1) reports "wrapped" immediately, so
                // this also terminates the single-iteration rank-1 case.
                if odometer_next(&mut oidx, &out_spatial[..last]) {
                    break;
                }
            }
            odometer_next(&mut kidx, kernel);
        }
    }
}

/// [`im2col_nd`] split by input channel across rayon when the work justifies it.
///
/// Same disjoint-row-band argument as the rank-2 `im2col_maybe_parallel`: a
/// contiguous band of input channels owns a contiguous band of column-matrix
/// rows, and the gather is element-wise, so the result is byte-identical to
/// the sequential fill.
#[allow(clippy::too_many_arguments)]
fn im2col_nd_maybe_parallel(
    input: &[f32],
    c_in: usize,
    in_spatial: &[usize],
    in_c_start: usize,
    c_per_group: usize,
    kernel: &[usize],
    strides: &[usize],
    pads_begin: &[usize],
    dilations: &[usize],
    out_spatial: &[usize],
    batch: usize,
    col: &mut [f32],
) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let ksize: usize = kernel.iter().product();
        let col_cols: usize = out_spatial.iter().product();
        let num_threads = rayon::current_num_threads();
        if num_threads > 1 && c_per_group >= 2 && c_per_group * ksize * col_cols >= 64 * 64 * 64 {
            let chunk_ic = c_per_group.div_ceil(num_threads).max(1);
            let chunk_len = chunk_ic * ksize * col_cols;
            col.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(t, sub)| {
                    let first = t * chunk_ic;
                    if first >= c_per_group {
                        return;
                    }
                    let count = (c_per_group - first).min(chunk_ic);
                    im2col_nd(
                        input,
                        c_in,
                        in_spatial,
                        in_c_start + first,
                        count,
                        kernel,
                        strides,
                        pads_begin,
                        dilations,
                        out_spatial,
                        batch,
                        sub,
                    );
                });
            return;
        }
    }

    im2col_nd(
        input,
        c_in,
        in_spatial,
        in_c_start,
        c_per_group,
        kernel,
        strides,
        pads_begin,
        dilations,
        out_spatial,
        batch,
        col,
    );
}
