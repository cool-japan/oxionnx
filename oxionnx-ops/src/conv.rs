use oxionnx_core::Tensor;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// 2D convolution via im2col + GEMM with Rayon parallelization.
///
/// Each (batch, group) pair is independent and processed in parallel on
/// native targets.  On WASM (single-threaded), falls back to sequential.
///
/// input: [N, C_in, H, W]
/// weight: \[C_out, C_in/group, kH, kW\]
/// bias: \[C_out\] (optional)
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Tensor {
    let n = input.shape[0];
    let c_in = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let c_out = weight.shape[0];
    let c_per_group = weight.shape[1];
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    let oh = (h + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
    let ow = (w + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1] + 1;

    let c_out_per_group = c_out / group;
    let no_pad = pads == [0, 0, 0, 0];

    // ── 1×1 conv fast path: skip im2col entirely ──────────────────────
    // When kernel is 1×1, stride=1, no padding, dilation=1: input channels
    // are already contiguous per spatial position → direct matmul.
    if kh == 1 && kw == 1 && strides == [1, 1] && no_pad && dilations == [1, 1] {
        return conv2d_1x1(
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
        );
    }

    let col_rows = c_per_group * kh * kw;
    let col_cols = oh * ow;
    let total_jobs = n * group;

    let mut out = vec![0.0f32; n * c_out * oh * ow];

    if total_jobs <= 1 {
        // ── Fast path: single job, write directly to output ────────────
        let mut col = vec![0.0f32; col_rows * col_cols];
        for batch in 0..n {
            for g in 0..group {
                let in_c_start = g * c_per_group;
                im2col(
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

    Tensor::new(out, vec![n, c_out, oh, ow])
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
    im2col(
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

/// 1×1 convolution fast path: no im2col needed.
///
/// For 1×1 kernel with stride=1, no padding:
///   input  [N, C_in, H, W]  → treat as [N, C_in, H*W]
///   weight [C_out, C_in/g, 1, 1] → treat as [C_out/g, C_in/g]
///   output = weight × input_slice (matmul, no copy)
///
/// This saves allocating and filling the im2col column matrix.
#[allow(clippy::too_many_arguments, unsafe_code)]
fn conv2d_1x1(
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
) -> Tensor {
    let spatial = h * w;
    let mut out = vec![0.0f32; n * c_out * spatial];

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

    Tensor::new(out, vec![n, c_out, h, w])
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

/// Build the im2col column matrix for one (batch, group) slice.
///
/// Each column corresponds to one output spatial position (oy, ox).
/// Each row corresponds to one element of the flattened kernel window
/// (ic, ky, kx) where ic ∈ [0, c_per_group).
///
/// Layout: row-major [col_rows, col_cols] where
///   col_rows = c_per_group * kH * kW
///   col_cols = OH * OW
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    in_c_start: usize,
    c_per_group: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    oh: usize,
    ow: usize,
    batch: usize,
    col: &mut [f32],
) {
    let col_cols = oh * ow;
    let mut row = 0;
    for ic in 0..c_per_group {
        let in_c = in_c_start + ic;
        let in_plane = &input[(batch * c_in + in_c) * h * w..][..h * w];
        for ky in 0..kh {
            for kx in 0..kw {
                for oy in 0..oh {
                    let iy = (oy * strides[0] + ky * dilations[0]) as isize - pads[0] as isize;
                    if iy < 0 || iy >= h as isize {
                        // Entire row of output cols at this oy is zero (padding)
                        let base = row * col_cols + oy * ow;
                        for ox in 0..ow {
                            col[base + ox] = 0.0;
                        }
                        continue;
                    }
                    let iy = iy as usize;
                    let base = row * col_cols + oy * ow;
                    for ox in 0..ow {
                        let ix = (ox * strides[1] + kx * dilations[1]) as isize - pads[1] as isize;
                        col[base + ox] = if ix >= 0 && ix < w as isize {
                            in_plane[iy * w + ix as usize]
                        } else {
                            0.0
                        };
                    }
                }
                row += 1;
            }
        }
    }
}

/// 2D max pooling.
/// input: [N, C, H, W]
pub fn max_pool2d(
    input: &Tensor,
    kernel_shape: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
) -> Tensor {
    let n = input.shape[0];
    let c = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let [kh, kw] = kernel_shape;
    let oh = (h + pads[0] + pads[2] - kh) / strides[0] + 1;
    let ow = (w + pads[1] + pads[3] - kw) / strides[1] + 1;

    let mut out = vec![f32::NEG_INFINITY; n * c * oh * ow];

    for batch in 0..n {
        for ch in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut max_val = f32::NEG_INFINITY;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * strides[0] + ky) as isize - pads[0] as isize;
                            let ix = (ox * strides[1] + kx) as isize - pads[1] as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let iy = iy as usize;
                                let ix = ix as usize;
                                let idx = ((batch * c + ch) * h + iy) * w + ix;
                                if input.data[idx] > max_val {
                                    max_val = input.data[idx];
                                }
                            }
                        }
                    }
                    out[((batch * c + ch) * oh + oy) * ow + ox] = max_val;
                }
            }
        }
    }

    Tensor::new(out, vec![n, c, oh, ow])
}

/// 2D average pooling.
/// input: [N, C, H, W]
pub fn avg_pool2d(
    input: &Tensor,
    kernel_shape: [usize; 2],
    strides: [usize; 2],
    pads: [usize; 4],
    count_include_pad: bool,
) -> Tensor {
    let n = input.shape[0];
    let c = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let [kh, kw] = kernel_shape;
    let oh = (h + pads[0] + pads[2] - kh) / strides[0] + 1;
    let ow = (w + pads[1] + pads[3] - kw) / strides[1] + 1;

    let mut out = vec![0.0f32; n * c * oh * ow];

    for batch in 0..n {
        for ch in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * strides[0] + ky) as isize - pads[0] as isize;
                            let ix = (ox * strides[1] + kx) as isize - pads[1] as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                let iy = iy as usize;
                                let ix = ix as usize;
                                let idx = ((batch * c + ch) * h + iy) * w + ix;
                                sum += input.data[idx];
                                count += 1;
                            } else if count_include_pad {
                                count += 1;
                            }
                        }
                    }
                    let divisor = if count_include_pad { kh * kw } else { count };
                    out[((batch * c + ch) * oh + oy) * ow + ox] = if divisor > 0 {
                        sum / divisor as f32
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    Tensor::new(out, vec![n, c, oh, ow])
}

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

/// ConvTranspose2D: fractionally-strided convolution (deconvolution)
/// input: [N, C_in, H, W]
/// weight: [C_in, C_out/group, kH, kW]
/// output: [N, C_out, oH, oW]
/// oH = stride*(H-1) + output_padding + ((kH-1)*dilation + 1) - pad_top - pad_bottom
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4], // [top, left, bottom, right]
    output_padding: [usize; 2],
    dilations: [usize; 2],
    group: usize,
) -> Result<Tensor, String> {
    if input.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: input must be 4D, got {}D",
            input.ndim()
        ));
    }
    if weight.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: weight must be 4D, got {}D",
            weight.ndim()
        ));
    }

    let (n, c_in, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let c_out_per_group = weight.shape[1];
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    let c_out = c_out_per_group * group;
    let c_in_per_group = c_in / group;

    if c_in % group != 0 {
        return Err(format!(
            "conv_transpose2d: c_in ({}) not divisible by group ({})",
            c_in, group
        ));
    }

    let eff_kh = (kh - 1) * dilations[0] + 1;
    let eff_kw = (kw - 1) * dilations[1] + 1;
    let oh = strides[0] * (h - 1) + output_padding[0] + eff_kh - pads[0] - pads[2];
    let ow = strides[1] * (w - 1) + output_padding[1] + eff_kw - pads[1] - pads[3];

    let mut output = vec![0.0f32; n * c_out * oh * ow];

    // For each input element, scatter its contribution to the output
    for ni in 0..n {
        for g in 0..group {
            for ic in 0..c_in_per_group {
                let ci = g * c_in_per_group + ic;
                for iy in 0..h {
                    for ix in 0..w {
                        let in_val = input.data[((ni * c_in + ci) * h + iy) * w + ix];
                        for oc in 0..c_out_per_group {
                            let co = g * c_out_per_group + oc;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oy_raw = iy * strides[0] + ky * dilations[0];
                                    let ox_raw = ix * strides[1] + kx * dilations[1];
                                    if oy_raw < pads[0] || ox_raw < pads[1] {
                                        continue;
                                    }
                                    let oy = oy_raw - pads[0];
                                    let ox = ox_raw - pads[1];
                                    if oy >= oh || ox >= ow {
                                        continue;
                                    }
                                    let w_val = weight.data
                                        [((ci * c_out_per_group + oc) * kh + ky) * kw + kx];
                                    output[((ni * c_out + co) * oh + oy) * ow + ox] +=
                                        in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias
    if let Some(b) = bias {
        for ni in 0..n {
            for co in 0..c_out {
                let bias_val = b.data[co];
                for oy in 0..oh {
                    for ox in 0..ow {
                        output[((ni * c_out + co) * oh + oy) * ow + ox] += bias_val;
                    }
                }
            }
        }
    }

    Ok(Tensor::new(output, vec![n, c_out, oh, ow]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_identity_kernel() {
        // 1x1 identity kernel: output should equal input
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
        let out = conv2d(&input, &weight, None, [1, 1], [0, 0, 0, 0], [1, 1], 1);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_conv2d_3x3_edge_detect() {
        // 3x3 kernel on 4x4 input, no padding
        #[rustfmt::skip]
        let input = Tensor::new(vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ], vec![1, 1, 4, 4]);
        // simple sum kernel
        let weight = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
        let out = conv2d(&input, &weight, None, [1, 1], [0, 0, 0, 0], [1, 1], 1);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_conv2d_stride2() {
        let input = Tensor::new(vec![1.0; 16], vec![1, 1, 4, 4]);
        let weight = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, [2, 2], [0, 0, 0, 0], [1, 1], 1);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_conv2d_grouped() {
        // 2 groups, 2 input channels, 2 output channels
        let input = Tensor::new(
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            vec![1, 2, 2, 2],
        );
        // group 0: weight for channel 0, group 1: weight for channel 1
        let weight = Tensor::new(vec![1.0, 3.0], vec![2, 1, 1, 1]);
        let out = conv2d(&input, &weight, None, [1, 1], [0, 0, 0, 0], [1, 1], 2);
        assert_eq!(out.shape, vec![1, 2, 2, 2]);
        // group 0: 1.0 * 1.0 = 1.0, group 1: 3.0 * 2.0 = 6.0
        assert_eq!(&out.data[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&out.data[4..], &[6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_conv2d_with_bias() {
        let input = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
        let weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
        let bias = Tensor::new(vec![10.0], vec![1]);
        let out = conv2d(
            &input,
            &weight,
            Some(&bias),
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1,
        );
        assert_eq!(out.data, vec![11.0, 11.0, 11.0, 11.0]);
    }

    #[test]
    fn test_global_avg_pool() {
        // [1, 2, 2, 2]: channel 0 = [1,2,3,4], channel 1 = [5,6,7,8]
        #[rustfmt::skip]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 2, 2]);
        let out = global_avg_pool(&x);
        assert_eq!(out.shape, vec![1, 2, 1, 1]);
        assert!((out.data[0] - 2.5).abs() < 1e-5); // mean of [1,2,3,4]
        assert!((out.data[1] - 6.5).abs() < 1e-5); // mean of [5,6,7,8]
    }

    #[test]
    fn test_global_max_pool() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 2, 2],
        );
        let out = global_max_pool(&x);
        assert_eq!(out.shape, vec![1, 2, 1, 1]);
        assert_eq!(out.data[0], 4.0);
        assert_eq!(out.data[1], 8.0);
    }

    #[test]
    fn test_max_pool2d() {
        #[rustfmt::skip]
        let input = Tensor::new(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ], vec![1, 1, 4, 4]);
        let out = max_pool2d(&input, [2, 2], [2, 2], [0, 0, 0, 0]);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avg_pool2d() {
        #[rustfmt::skip]
        let input = Tensor::new(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ], vec![1, 1, 4, 4]);
        let out = avg_pool2d(&input, [2, 2], [2, 2], [0, 0, 0, 0], false);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // (1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5, (9+10+13+14)/4=11.5, (11+12+15+16)/4=13.5
        assert_eq!(out.data, vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_conv_transpose_basic() {
        // 1x1 kernel with stride 1 = identity-like
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
        let out = conv_transpose2d(
            &input,
            &weight,
            None,
            [1, 1],
            [0, 0, 0, 0],
            [0, 0],
            [1, 1],
            1,
        )
        .expect("conv_transpose basic failed");
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_conv_transpose_upsample() {
        // stride=2 upsamples
        let input = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
        let weight = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
        let out = conv_transpose2d(
            &input,
            &weight,
            None,
            [2, 2],
            [0, 0, 0, 0],
            [0, 0],
            [1, 1],
            1,
        )
        .expect("conv_transpose upsample failed");
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_conv_transpose_with_bias() {
        let input = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
        let weight = Tensor::new(vec![2.0], vec![1, 1, 1, 1]);
        let bias = Tensor::new(vec![3.0], vec![1]);
        let out = conv_transpose2d(
            &input,
            &weight,
            Some(&bias),
            [1, 1],
            [0, 0, 0, 0],
            [0, 0],
            [1, 1],
            1,
        )
        .expect("conv_transpose with bias failed");
        assert_eq!(out.data, vec![5.0]); // 1*2 + 3
    }

    #[test]
    fn test_conv_transpose_with_padding() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
        let out = conv_transpose2d(
            &input,
            &weight,
            None,
            [1, 1],
            [1, 1, 1, 1],
            [0, 0],
            [1, 1],
            1,
        )
        .expect("conv_transpose with padding failed");
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_conv_transpose_invalid_input() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
        let result = conv_transpose2d(
            &input,
            &weight,
            None,
            [1, 1],
            [0, 0, 0, 0],
            [0, 0],
            [1, 1],
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_conv_transpose_multi_channel() {
        // 2 input channels, 1 output channel
        let input = Tensor::new(
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            vec![1, 2, 2, 2],
        );
        let weight = Tensor::new(vec![1.0, 1.0], vec![2, 1, 1, 1]);
        let out = conv_transpose2d(
            &input,
            &weight,
            None,
            [1, 1],
            [0, 0, 0, 0],
            [0, 0],
            [1, 1],
            1,
        )
        .expect("conv_transpose multi channel failed");
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // Each output pixel: 1*1 + 2*1 = 3
        assert_eq!(out.data, vec![3.0, 3.0, 3.0, 3.0]);
    }
}
