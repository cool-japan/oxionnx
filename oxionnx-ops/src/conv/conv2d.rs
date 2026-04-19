use oxionnx_core::Tensor;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use super::im2col::im2col_adaptive;
use super::winograd::conv2d_winograd_f2x3;

/// Compute the output shape for a 2D convolution.
///
/// `input_shape` must be `[N, C, H, W]`.
/// `weight_shape` must be `[F, C/group, kH, kW]`.
/// `pads` must be `[pad_top, pad_left, pad_bottom, pad_right]` (length 4).
pub(crate) fn compute_conv2d_out_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    dilations: &[usize],
) -> Vec<usize> {
    let n = input_shape[0];
    let h = input_shape[2];
    let w = input_shape[3];
    let f = weight_shape[0];
    let kh = weight_shape[2];
    let kw = weight_shape[3];
    let pad_top = pads.first().copied().unwrap_or(0);
    let pad_left = pads.get(1).copied().unwrap_or(0);
    let pad_bottom = pads.get(2).copied().unwrap_or(0);
    let pad_right = pads.get(3).copied().unwrap_or(0);
    let stride_h = strides.first().copied().unwrap_or(1);
    let stride_w = strides.get(1).copied().unwrap_or(1);
    let dilation_h = dilations.first().copied().unwrap_or(1);
    let dilation_w = dilations.get(1).copied().unwrap_or(1);
    let out_h = (h + pad_top + pad_bottom - dilation_h * (kh - 1) - 1) / stride_h + 1;
    let out_w = (w + pad_left + pad_right - dilation_w * (kw - 1) - 1) / stride_w + 1;
    vec![n, f, out_h, out_w]
}

/// Write conv2d result directly into a pre-allocated output buffer.
///
/// `out_shape` must be the result of `compute_conv2d_out_shape` for these inputs.
/// `out` must have length equal to `out_shape.iter().product()`.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn conv2d_into(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    out: &mut [f32],
    out_shape: &[usize],
) {
    let n = input.shape[0];
    let c_in = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let c_out = weight.shape[0];
    let c_per_group = weight.shape[1];
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    let oh = out_shape[2];
    let ow = out_shape[3];

    let c_out_per_group = c_out / group;
    let no_pad = pads == [0, 0, 0, 0];

    // ── 1×1 conv fast path: skip im2col entirely ──────────────────────────
    // When kernel is 1×1, stride=1, no padding, dilation=1: input channels
    // are already contiguous per spatial position → direct matmul.
    if kh == 1 && kw == 1 && strides == [1, 1] && no_pad && dilations == [1, 1] {
        conv2d_1x1_into(
            input,
            weight,
            bias,
            n,
            c_in,
            c_out,
            c_per_group,
            c_out_per_group,
            h,
            w,
            group,
            out,
        );
        return;
    }

    // ── Winograd F(2,3) fast path: 3×3 kernel, stride 1, dilation 1 ──────
    if kh == 3
        && kw == 3
        && strides == [1, 1]
        && dilations == [1, 1]
        && group == 1
        && oh >= 4
        && ow >= 4
        && pads[0] == pads[1]
        && pads[0] == pads[2]
        && pads[0] == pads[3]
    {
        let pad = pads[0];
        let bias_slice = bias.map(|b| b.data.as_slice());
        if let Ok(data) = conv2d_winograd_f2x3(
            &input.data,
            &weight.data,
            bias_slice,
            n,
            c_in,
            h,
            w,
            c_out,
            pad,
        ) {
            out.copy_from_slice(&data);
            return;
        }
    }

    let col_rows = c_per_group * kh * kw;
    let col_cols = oh * ow;
    let total_jobs = n * group;

    // Zero out the buffer before accumulating results.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    if total_jobs <= 1 {
        // ── Fast path: single job, write directly to output ────────────
        let mut col = vec![0.0f32; col_rows * col_cols];
        for batch in 0..n {
            for g in 0..group {
                let in_c_start = g * c_per_group;
                im2col_adaptive(
                    &input.data,
                    c_in,
                    h,
                    w,
                    in_c_start,
                    c_per_group,
                    kh,
                    kw,
                    strides,
                    pads,
                    dilations,
                    oh,
                    ow,
                    batch,
                    &mut col,
                );
                let w_off = g * c_out_per_group * col_rows;
                let o_off = (batch * c_out + g * c_out_per_group) * col_cols;
                unsafe {
                    matrixmultiply::sgemm(
                        c_out_per_group,
                        col_rows,
                        col_cols,
                        1.0,
                        weight.data[w_off..].as_ptr(),
                        col_rows as isize,
                        1,
                        col.as_ptr(),
                        col_cols as isize,
                        1,
                        0.0,
                        out[o_off..].as_mut_ptr(),
                        col_cols as isize,
                        1,
                    );
                }
                if let Some(b) = bias {
                    for oc in 0..c_out_per_group {
                        let bv = b.data[g * c_out_per_group + oc];
                        let start = o_off + oc * col_cols;
                        for j in 0..col_cols {
                            out[start + j] += bv;
                        }
                    }
                }
            }
        }
    } else {
        // ── Parallel path: multiple jobs ────────────────────────────────
        let job_out_size = c_out_per_group * col_cols;

        #[cfg(not(target_arch = "wasm32"))]
        let job_outputs: Vec<Vec<f32>> = (0..total_jobs)
            .into_par_iter()
            .map(|idx| {
                let batch = idx / group;
                let g = idx % group;
                conv2d_single_job(
                    input,
                    weight,
                    bias,
                    strides,
                    pads,
                    dilations,
                    c_in,
                    h,
                    w,
                    c_out,
                    c_per_group,
                    c_out_per_group,
                    kh,
                    kw,
                    oh,
                    ow,
                    col_rows,
                    col_cols,
                    batch,
                    g,
                )
            })
            .collect();

        #[cfg(target_arch = "wasm32")]
        let job_outputs: Vec<Vec<f32>> = (0..total_jobs)
            .map(|idx| {
                let batch = idx / group;
                let g = idx % group;
                conv2d_single_job(
                    input,
                    weight,
                    bias,
                    strides,
                    pads,
                    dilations,
                    c_in,
                    h,
                    w,
                    c_out,
                    c_per_group,
                    c_out_per_group,
                    kh,
                    kw,
                    oh,
                    ow,
                    col_rows,
                    col_cols,
                    batch,
                    g,
                )
            })
            .collect();

        for (idx, job_out) in job_outputs.into_iter().enumerate() {
            let batch = idx / group;
            let g = idx % group;
            let o_off = (batch * c_out + g * c_out_per_group) * col_cols;
            out[o_off..o_off + job_out_size].copy_from_slice(&job_out);
        }
    }
}

/// 2D convolution via im2col + GEMM with Rayon parallelization.
///
/// Each (batch, group) pair is independent and processed in parallel on
/// native targets.  On WASM (single-threaded), falls back to sequential.
///
/// input: [N, C_in, H, W]
/// weight: \[C_out, C_in/group, kH, kW\]
/// bias: \[C_out\] (optional)
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Tensor {
    let out_shape =
        compute_conv2d_out_shape(&input.shape, &weight.shape, &strides, &pads, &dilations);
    let out_len: usize = out_shape.iter().product();
    let mut data = vec![0.0_f32; out_len];
    conv2d_into(
        input, weight, bias, strides, pads, dilations, group, &mut data, &out_shape,
    );
    Tensor::new(data, out_shape)
}

/// Process one (batch, group) slice: im2col → sgemm → bias.
#[allow(clippy::too_many_arguments, unsafe_code)]
fn conv2d_single_job(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    c_in: usize,
    h: usize,
    w: usize,
    _c_out: usize,
    c_per_group: usize,
    c_out_per_group: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    col_rows: usize,
    col_cols: usize,
    batch: usize,
    g: usize,
) -> Vec<f32> {
    let mut col = vec![0.0f32; col_rows * col_cols];
    let in_c_start = g * c_per_group;
    im2col_adaptive(
        &input.data,
        c_in,
        h,
        w,
        in_c_start,
        c_per_group,
        kh,
        kw,
        strides,
        pads,
        dilations,
        oh,
        ow,
        batch,
        &mut col,
    );

    let w_off = g * c_out_per_group * col_rows;
    let job_size = c_out_per_group * col_cols;
    let mut out = vec![0.0f32; job_size];

    unsafe {
        matrixmultiply::sgemm(
            c_out_per_group,
            col_rows,
            col_cols,
            1.0,
            weight.data[w_off..].as_ptr(),
            col_rows as isize,
            1,
            col.as_ptr(),
            col_cols as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            col_cols as isize,
            1,
        );
    }

    if let Some(b) = bias {
        for oc in 0..c_out_per_group {
            let bias_val = b.data[g * c_out_per_group + oc];
            let row_start = oc * col_cols;
            for j in 0..col_cols {
                out[row_start + j] += bias_val;
            }
        }
    }

    out
}

/// 1×1 convolution fast path, writing directly into a pre-allocated buffer.
///
/// For 1×1 kernel with stride=1, no padding, dilation=1:
///   input  [N, C_in, H, W]  → treat as [N, C_in, H*W]
///   weight [C_out, C_in/g, 1, 1] → treat as [C_out/g, C_in/g]
///   output = weight × input_slice (matmul, no copy)
///
/// This saves allocating and filling the im2col column matrix.
#[allow(clippy::too_many_arguments, unsafe_code)]
fn conv2d_1x1_into(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    n: usize,
    c_in: usize,
    c_out: usize,
    c_per_group: usize,
    c_out_per_group: usize,
    h: usize,
    w: usize,
    group: usize,
    out: &mut [f32],
) {
    let spatial = h * w;

    // Threshold for parallel tiled sgemm: if the GEMM is large enough,
    // split output channels across rayon threads.
    #[cfg(not(target_arch = "wasm32"))]
    let parallel_threshold = 64 * 64 * 64; // M*K*N threshold
    #[cfg(not(target_arch = "wasm32"))]
    let num_threads = rayon::current_num_threads();

    for batch in 0..n {
        for g in 0..group {
            let in_c_start = g * c_per_group;
            let in_off = (batch * c_in + in_c_start) * spatial;
            let w_off = g * c_out_per_group * c_per_group;
            let o_off = (batch * c_out + g * c_out_per_group) * spatial;

            let m = c_out_per_group;
            let k = c_per_group;
            let nn = spatial;

            #[cfg(not(target_arch = "wasm32"))]
            {
                let flops = m * k * nn;
                if flops >= parallel_threshold && m >= num_threads * 2 {
                    // ── Parallel tiled sgemm: split M across threads ──
                    parallel_sgemm(
                        m,
                        k,
                        nn,
                        &weight.data[w_off..],
                        &input.data[in_off..],
                        &mut out[o_off..],
                    );
                } else {
                    unsafe {
                        matrixmultiply::sgemm(
                            m,
                            k,
                            nn,
                            1.0,
                            weight.data[w_off..].as_ptr(),
                            k as isize,
                            1,
                            input.data[in_off..].as_ptr(),
                            nn as isize,
                            1,
                            0.0,
                            out[o_off..].as_mut_ptr(),
                            nn as isize,
                            1,
                        );
                    }
                }
            }

            #[cfg(target_arch = "wasm32")]
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    nn,
                    1.0,
                    weight.data[w_off..].as_ptr(),
                    k as isize,
                    1,
                    input.data[in_off..].as_ptr(),
                    nn as isize,
                    1,
                    0.0,
                    out[o_off..].as_mut_ptr(),
                    nn as isize,
                    1,
                );
            }

            if let Some(b) = bias {
                for oc in 0..c_out_per_group {
                    let bv = b.data[g * c_out_per_group + oc];
                    let start = o_off + oc * spatial;
                    for j in 0..spatial {
                        out[start + j] += bv;
                    }
                }
            }
        }
    }
}

/// Split a large sgemm (C = A × B) by rows of A across rayon threads.
/// A: [m, k] row-major, B: [k, n] row-major, C: [m, n] row-major.
#[cfg(not(target_arch = "wasm32"))]
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
fn parallel_sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let num_threads = rayon::current_num_threads();
    let chunk = m.div_ceil(num_threads);

    // Each thread computes a disjoint horizontal tile of C.
    // Collect results then copy (avoids unsafe mutable aliasing).
    let tiles: Vec<(usize, Vec<f32>)> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let row_start = t * chunk;
            if row_start >= m {
                return None;
            }
            let row_end = (row_start + chunk).min(m);
            let tile_m = row_end - row_start;
            let mut tile = vec![0.0f32; tile_m * n];

            unsafe {
                matrixmultiply::sgemm(
                    tile_m,
                    k,
                    n,
                    1.0,
                    a[row_start * k..].as_ptr(),
                    k as isize,
                    1,
                    b.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    tile.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
            Some((row_start, tile))
        })
        .collect();

    for (row_start, tile) in tiles {
        let dst_off = row_start * n;
        c[dst_off..dst_off + tile.len()].copy_from_slice(&tile);
    }
}
