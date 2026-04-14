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

    // ── Winograd F(2,3) fast path: 3×3 kernel, stride 1, dilation 1 ──
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
            return Tensor::new(data, vec![n, c_out, oh, ow]);
        }
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

/// Tile size for cache-blocked im2col.
const IM2COL_TILE: usize = 32;

/// Threshold (in output spatial positions) above which cache-blocked im2col is used.
const IM2COL_BLOCK_THRESHOLD: usize = 256;

/// Cache-blocked im2col: processes output positions in 2D spatial tiles
/// for better L1/L2 cache locality on large feature maps.
///
/// Produces the same column matrix as `im2col` but with better cache behavior
/// when oh*ow is large. Within each tile, the columns written per row fit in
/// fewer cache lines, reducing eviction on wide feature maps.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col_blocked(
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

    // Process output in 2D spatial tiles
    for oy_base in (0..oh).step_by(IM2COL_TILE) {
        let oy_end = (oy_base + IM2COL_TILE).min(oh);
        for ox_base in (0..ow).step_by(IM2COL_TILE) {
            let ox_end = (ox_base + IM2COL_TILE).min(ow);

            // For this tile, fill the corresponding columns of every row
            let mut row = 0;
            for ic in 0..c_per_group {
                let in_c = in_c_start + ic;
                let in_plane = &input[(batch * c_in + in_c) * h * w..][..h * w];
                for ky in 0..kh {
                    for kx in 0..kw {
                        for oy in oy_base..oy_end {
                            let iy =
                                (oy * strides[0] + ky * dilations[0]) as isize - pads[0] as isize;
                            let base = row * col_cols + oy * ow;
                            if iy < 0 || iy >= h as isize {
                                for ox in ox_base..ox_end {
                                    col[base + ox] = 0.0;
                                }
                                continue;
                            }
                            let iy = iy as usize;
                            for ox in ox_base..ox_end {
                                let ix = (ox * strides[1] + kx * dilations[1]) as isize
                                    - pads[1] as isize;
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
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SIMD-accelerated im2col for stride=1 dilation=1
// ═══════════════════════════════════════════════════════════════════════

/// Copy `src` to `dst` using SIMD wide loads/stores where available.
#[cfg(feature = "simd")]
#[inline]
fn simd_copy_f32(src: &[f32], dst: &mut [f32]) {
    let len = src.len();

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let mut i = 0;
        while i + 4 <= len {
            // SAFETY: bounds checked above, vld1q/vst1q operate on 4 f32s.
            unsafe {
                let v = vld1q_f32(src.as_ptr().add(i));
                vst1q_f32(dst.as_mut_ptr().add(i), v);
            }
            i += 4;
        }
        if i < len {
            dst[i..len].copy_from_slice(&src[i..len]);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            use core::arch::x86_64::*;
            let mut i = 0;
            while i + 8 <= len {
                // SAFETY: AVX2 confirmed, bounds checked; unaligned load/store.
                unsafe {
                    let v = _mm256_loadu_ps(src.as_ptr().add(i));
                    _mm256_storeu_ps(dst.as_mut_ptr().add(i), v);
                }
                i += 8;
            }
            if i < len {
                dst[i..len].copy_from_slice(&src[i..len]);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        dst.copy_from_slice(src);
    }
}

/// SIMD-accelerated im2col for stride=1, dilation=1 convolutions.
///
/// When stride=1 and dilation=1, each im2col row for a given (channel, ky, kx)
/// and output row `oy` is a contiguous horizontal strip of the input.  We compute
/// the valid range once per row and bulk-copy with SIMD, avoiding per-element
/// bounds checks.
#[cfg(feature = "simd")]
#[allow(clippy::too_many_arguments)]
pub fn im2col_simd_stride1(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    in_c_start: usize,
    c_per_group: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    out_h: usize,
    out_w: usize,
    batch: usize,
    col: &mut [f32],
) {
    let col_cols = out_h * out_w;
    let mut row = 0;

    for ic in 0..c_per_group {
        let in_c = in_c_start + ic;
        let plane_off = (batch * c_in + in_c) * h * w;

        for ky in 0..kh {
            for kx in 0..kw {
                for oy in 0..out_h {
                    let iy = oy as isize + ky as isize - pad_h as isize;
                    let dst_off = row * col_cols + oy * out_w;

                    if iy < 0 || iy >= h as isize {
                        // Entire segment is padding → zero-fill
                        for v in &mut col[dst_off..dst_off + out_w] {
                            *v = 0.0;
                        }
                        continue;
                    }
                    let iy = iy as usize;
                    let row_base = plane_off + iy * w;

                    // ix = ox + kx - pad_w  → valid when 0 <= ix < w
                    let ox_start = pad_w.saturating_sub(kx);
                    let ox_end = (w + pad_w).saturating_sub(kx).min(out_w);

                    // Left padding zeros
                    for v in &mut col[dst_off..dst_off + ox_start.min(out_w)] {
                        *v = 0.0;
                    }

                    // SIMD bulk copy of valid region
                    if ox_start < ox_end {
                        let src_start = row_base + ox_start + kx - pad_w;
                        let count = ox_end - ox_start;
                        simd_copy_f32(
                            &input[src_start..src_start + count],
                            &mut col[dst_off + ox_start..dst_off + ox_start + count],
                        );
                    }

                    // Right padding zeros
                    for v in &mut col[dst_off + ox_end..dst_off + out_w] {
                        *v = 0.0;
                    }
                }
                row += 1;
            }
        }
    }
}

/// Pack weight matrix into panel layout for cache-friendly GEMM access.
///
/// Input layout: `[rows, cols]` row-major (e.g. `[C_out, C_in * kH * kW]`).
/// Output: panels of `panel_width` rows stored column-major within each panel.
/// This pre-packing can be done once at model load time and cached.
pub fn pack_weights_panel(
    weights: &[f32],
    rows: usize,
    cols: usize,
    panel_width: usize,
) -> Vec<f32> {
    let num_panels = rows.div_ceil(panel_width);
    let mut packed = vec![0.0f32; num_panels * panel_width * cols];

    for panel in 0..num_panels {
        let row_start = panel * panel_width;
        let row_end = (row_start + panel_width).min(rows);
        let panel_off = panel * panel_width * cols;

        for col in 0..cols {
            for r in 0..panel_width {
                let src_row = row_start + r;
                packed[panel_off + col * panel_width + r] = if src_row < row_end {
                    weights[src_row * cols + col]
                } else {
                    0.0
                };
            }
        }
    }
    packed
}

/// Dispatch to cache-blocked im2col for large outputs, original for small.
/// When the `simd` feature is enabled and stride=1 dilation=1, uses
/// SIMD-accelerated bulk-copy im2col instead.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col_adaptive(
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
    #[cfg(feature = "simd")]
    if strides == [1, 1] && dilations == [1, 1] {
        im2col_simd_stride1(
            input,
            c_in,
            h,
            w,
            in_c_start,
            c_per_group,
            kh,
            kw,
            pads[0],
            pads[1],
            oh,
            ow,
            batch,
            col,
        );
        return;
    }

    if oh * ow >= IM2COL_BLOCK_THRESHOLD {
        im2col_blocked(
            input,
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
            col,
        );
    } else {
        im2col(
            input,
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
            col,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Winograd F(2,3) convolution
// ═══════════════════════════════════════════════════════════════════════

/// Winograd F(2,3) filter transform: U = G · g · G^T
///
/// Transforms a 3×3 filter into Winograd domain (4×4).
/// G = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]
#[inline]
fn winograd_filter_transform(g: &[f32]) -> [f32; 16] {
    // temp = G * g  (4×3)
    let mut temp = [0.0f32; 12];
    for j in 0..3 {
        temp[j] = g[j];
        temp[3 + j] = 0.5 * (g[j] + g[3 + j] + g[6 + j]);
        temp[6 + j] = 0.5 * (g[j] - g[3 + j] + g[6 + j]);
        temp[9 + j] = g[6 + j];
    }

    // U = temp * G^T  (4×4)
    let mut u = [0.0f32; 16];
    for i in 0..4 {
        let t0 = temp[i * 3];
        let t1 = temp[i * 3 + 1];
        let t2 = temp[i * 3 + 2];
        u[i * 4] = t0;
        u[i * 4 + 1] = 0.5 * (t0 + t1 + t2);
        u[i * 4 + 2] = 0.5 * (t0 - t1 + t2);
        u[i * 4 + 3] = t2;
    }
    u
}

/// Winograd F(2,3) input transform: V = B^T · d · B
///
/// Transforms a 4×4 input tile into Winograd domain.
/// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
#[inline]
fn winograd_input_transform(d: &[f32; 16]) -> [f32; 16] {
    // temp = B^T * d  (4×4)
    let mut temp = [0.0f32; 16];
    for j in 0..4 {
        temp[j] = d[j] - d[8 + j];
        temp[4 + j] = d[4 + j] + d[8 + j];
        temp[8 + j] = d[8 + j] - d[4 + j];
        temp[12 + j] = d[4 + j] - d[12 + j];
    }

    // V = temp * B  (4×4)
    let mut v = [0.0f32; 16];
    for i in 0..4 {
        let t0 = temp[i * 4];
        let t1 = temp[i * 4 + 1];
        let t2 = temp[i * 4 + 2];
        let t3 = temp[i * 4 + 3];
        v[i * 4] = t0 - t2;
        v[i * 4 + 1] = t1 + t2;
        v[i * 4 + 2] = t2 - t1;
        v[i * 4 + 3] = t1 - t3;
    }
    v
}

/// Winograd F(2,3) output transform: Y = A^T · M · A
///
/// Transforms a 4×4 Winograd-domain result back to a 2×2 output tile.
/// A^T = [[1,1,1,0],[0,1,-1,-1]]
#[inline]
fn winograd_output_transform(m: &[f32; 16]) -> [f32; 4] {
    // temp = A^T * M  (2×4)
    let mut temp = [0.0f32; 8];
    for j in 0..4 {
        temp[j] = m[j] + m[4 + j] + m[8 + j];
        temp[4 + j] = m[4 + j] - m[8 + j] - m[12 + j];
    }

    // Y = temp * A  (2×2)
    [
        temp[0] + temp[1] + temp[2],
        temp[1] - temp[2] - temp[3],
        temp[4] + temp[5] + temp[6],
        temp[5] - temp[6] - temp[7],
    ]
}

/// Winograd F(2,3) convolution for 3×3 kernels with stride=1, dilation=1.
///
/// Computes 2×2 output tiles from 4×4 input tiles, reducing multiplications
/// from 36 to 16 per output tile (2.25× fewer).
///
/// Only valid when: kh=kw=3, stride=1, dilation=1, group=1.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_winograd_f2x3(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    c: usize,
    ih: usize,
    iw: usize,
    oc: usize,
    pad: usize,
) -> Result<Vec<f32>, String> {
    if ih + 2 * pad < 3 || iw + 2 * pad < 3 {
        return Err("conv2d_winograd_f2x3: padded input too small for 3x3 kernel".to_string());
    }
    let oh = ih + 2 * pad - 2;
    let ow = iw + 2 * pad - 2;

    let expected_weight_len = oc * c * 9;
    if weight.len() < expected_weight_len {
        return Err(format!(
            "conv2d_winograd_f2x3: weight length {} < expected {}",
            weight.len(),
            expected_weight_len
        ));
    }

    // Number of 2×2 output tiles (ceiling division)
    let tile_h = oh.div_ceil(2);
    let tile_w = ow.div_ceil(2);

    // Pre-transform all filters: U[oc][c] each 4×4 = 16 floats
    let mut u_all = vec![0.0f32; oc * c * 16];
    for o in 0..oc {
        for i in 0..c {
            let g_start = (o * c + i) * 9;
            let u = winograd_filter_transform(&weight[g_start..g_start + 9]);
            let u_start = (o * c + i) * 16;
            u_all[u_start..u_start + 16].copy_from_slice(&u);
        }
    }

    let mut output = vec![0.0f32; n * oc * oh * ow];

    // Reusable buffer for transformed input tiles per channel
    let mut v_tiles = vec![[0.0f32; 16]; c];

    for batch in 0..n {
        for th in 0..tile_h {
            for tw in 0..tile_w {
                let oy_start = th * 2;
                let ox_start = tw * 2;
                let iy_base = oy_start as isize - pad as isize;
                let ix_base = ox_start as isize - pad as isize;

                // How many output rows/cols this tile actually produces
                let out_rows = if oy_start + 2 <= oh { 2 } else { oh - oy_start };
                let out_cols = if ox_start + 2 <= ow { 2 } else { ow - ox_start };

                // Transform input tiles for all channels
                for (ic, v_tile) in v_tiles.iter_mut().enumerate() {
                    let mut d = [0.0f32; 16];
                    let plane_off = (batch * c + ic) * ih * iw;
                    for dy in 0..4usize {
                        let iy = iy_base + dy as isize;
                        for dx in 0..4usize {
                            let ix = ix_base + dx as isize;
                            d[dy * 4 + dx] =
                                if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                                    input[plane_off + iy as usize * iw + ix as usize]
                                } else {
                                    0.0
                                };
                        }
                    }
                    *v_tile = winograd_input_transform(&d);
                }

                // For each output channel, accumulate in Winograd domain then transform back
                for o in 0..oc {
                    let mut m_acc = [0.0f32; 16];
                    for (ic, v_tile) in v_tiles.iter().enumerate() {
                        let u_start = (o * c + ic) * 16;
                        let u = &u_all[u_start..u_start + 16];
                        for k in 0..16 {
                            m_acc[k] += u[k] * v_tile[k];
                        }
                    }

                    let y = winograd_output_transform(&m_acc);

                    // Write output tile (handle edge tiles that produce < 2×2)
                    let out_plane = (batch * oc + o) * oh * ow;
                    for dy in 0..out_rows {
                        for dx in 0..out_cols {
                            output[out_plane + (oy_start + dy) * ow + (ox_start + dx)] =
                                y[dy * 2 + dx];
                        }
                    }
                }
            }
        }
    }

    // Add bias
    if let Some(b) = bias {
        for batch in 0..n {
            for (o, &bias_val) in b.iter().enumerate() {
                let plane_off = (batch * oc + o) * oh * ow;
                for j in 0..oh * ow {
                    output[plane_off + j] += bias_val;
                }
            }
        }
    }

    Ok(output)
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

    // ══════════════════════════════════════════════════════════════
    // Cache-blocked im2col tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_im2col_blocked_matches_original_3x3() {
        // Verify cache-blocked im2col produces identical output to original
        let c_in = 3;
        let h = 8;
        let w = 8;
        let kh = 3;
        let kw = 3;
        let strides = [1, 1];
        let pads = [1, 1, 1, 1];
        let dilations = [1, 1];
        let oh = (h + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
        let ow = (w + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1] + 1;
        let col_rows = c_in * kh * kw;
        let col_cols = oh * ow;

        let input: Vec<f32> = (0..c_in * h * w).map(|i| i as f32 * 0.1).collect();
        let mut col_orig = vec![0.0f32; col_rows * col_cols];
        let mut col_block = vec![0.0f32; col_rows * col_cols];

        im2col(
            &input,
            c_in,
            h,
            w,
            0,
            c_in,
            kh,
            kw,
            strides,
            pads,
            dilations,
            oh,
            ow,
            0,
            &mut col_orig,
        );
        im2col_blocked(
            &input,
            c_in,
            h,
            w,
            0,
            c_in,
            kh,
            kw,
            strides,
            pads,
            dilations,
            oh,
            ow,
            0,
            &mut col_block,
        );

        for i in 0..col_orig.len() {
            assert!(
                (col_orig[i] - col_block[i]).abs() < 1e-6,
                "mismatch at index {}: orig={}, blocked={}",
                i,
                col_orig[i],
                col_block[i]
            );
        }
    }

    #[test]
    fn test_im2col_blocked_matches_original_5x5() {
        let c_in = 2;
        let h = 12;
        let w = 12;
        let kh = 5;
        let kw = 5;
        let strides = [1, 1];
        let pads = [2, 2, 2, 2];
        let dilations = [1, 1];
        let oh = (h + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
        let ow = (w + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1] + 1;
        let col_rows = c_in * kh * kw;
        let col_cols = oh * ow;

        let input: Vec<f32> = (0..c_in * h * w).map(|i| (i as f32).sin()).collect();
        let mut col_orig = vec![0.0f32; col_rows * col_cols];
        let mut col_block = vec![0.0f32; col_rows * col_cols];

        im2col(
            &input,
            c_in,
            h,
            w,
            0,
            c_in,
            kh,
            kw,
            strides,
            pads,
            dilations,
            oh,
            ow,
            0,
            &mut col_orig,
        );
        im2col_blocked(
            &input,
            c_in,
            h,
            w,
            0,
            c_in,
            kh,
            kw,
            strides,
            pads,
            dilations,
            oh,
            ow,
            0,
            &mut col_block,
        );

        for i in 0..col_orig.len() {
            assert!(
                (col_orig[i] - col_block[i]).abs() < 1e-6,
                "5x5 mismatch at index {}: orig={}, blocked={}",
                i,
                col_orig[i],
                col_block[i]
            );
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Winograd F(2,3) tests
    // ══════════════════════════════════════════════════════════════

    /// Reference im2col-based conv2d for comparison (uses original im2col path,
    /// bypassing Winograd dispatch).
    #[allow(unsafe_code)]
    fn conv2d_reference(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        n: usize,
        c_in: usize,
        ih: usize,
        iw: usize,
        c_out: usize,
        kh: usize,
        kw: usize,
        pad: usize,
    ) -> Vec<f32> {
        let oh = ih + 2 * pad - kh + 1;
        let ow = iw + 2 * pad - kw + 1;
        let col_rows = c_in * kh * kw;
        let col_cols = oh * ow;
        let mut out = vec![0.0f32; n * c_out * oh * ow];
        let mut col = vec![0.0f32; col_rows * col_cols];

        for batch in 0..n {
            im2col(
                input,
                c_in,
                ih,
                iw,
                0,
                c_in,
                kh,
                kw,
                [1, 1],
                [pad, pad, pad, pad],
                [1, 1],
                oh,
                ow,
                batch,
                &mut col,
            );
            let o_off = batch * c_out * col_cols;
            unsafe {
                matrixmultiply::sgemm(
                    c_out,
                    col_rows,
                    col_cols,
                    1.0,
                    weight.as_ptr(),
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
                for oc in 0..c_out {
                    let bv = b[oc];
                    let start = o_off + oc * col_cols;
                    for j in 0..col_cols {
                        out[start + j] += bv;
                    }
                }
            }
        }
        out
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", label);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{}: mismatch at [{}]: {} vs {} (diff={})",
                label,
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    #[test]
    fn test_winograd_small_1x1x4x4() {
        // Minimal: [1,1,4,4] input, [1,1,3,3] weight, pad=0 → 2×2 output
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let weight: Vec<f32> = vec![1.0; 9];
        let expected = conv2d_reference(&input, &weight, None, 1, 1, 4, 4, 1, 3, 3, 0);
        let got =
            conv2d_winograd_f2x3(&input, &weight, None, 1, 1, 4, 4, 1, 0).expect("winograd small");
        assert_close(&expected, &got, 1e-4, "winograd_small_4x4");
    }

    #[test]
    fn test_winograd_medium_multichannel() {
        // [1,3,8,8] input, [16,3,3,3] weight, pad=1
        let n = 1;
        let c = 3;
        let ih = 8;
        let iw = 8;
        let oc = 16;
        let pad = 1;
        let input: Vec<f32> = (0..n * c * ih * iw)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let weight: Vec<f32> = (0..oc * c * 9).map(|i| (i as f32 * 0.03).cos()).collect();
        let expected = conv2d_reference(&input, &weight, None, n, c, ih, iw, oc, 3, 3, pad);
        let got = conv2d_winograd_f2x3(&input, &weight, None, n, c, ih, iw, oc, pad)
            .expect("winograd medium");
        assert_close(&expected, &got, 1e-4, "winograd_medium");
    }

    #[test]
    fn test_winograd_with_padding() {
        let input: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let weight: Vec<f32> = (0..9).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let expected = conv2d_reference(&input, &weight, None, 1, 1, 5, 5, 1, 3, 3, 1);
        let got =
            conv2d_winograd_f2x3(&input, &weight, None, 1, 1, 5, 5, 1, 1).expect("winograd pad");
        assert_close(&expected, &got, 1e-4, "winograd_with_padding");
    }

    #[test]
    fn test_winograd_with_bias() {
        let n = 1;
        let c = 2;
        let ih = 6;
        let iw = 6;
        let oc = 4;
        let pad = 1;
        let input: Vec<f32> = (0..n * c * ih * iw).map(|i| i as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..oc * c * 9).map(|i| i as f32 * 0.05).collect();
        let bias = vec![1.0, -0.5, 0.25, 3.0];
        let expected = conv2d_reference(&input, &weight, Some(&bias), n, c, ih, iw, oc, 3, 3, pad);
        let got = conv2d_winograd_f2x3(&input, &weight, Some(&bias), n, c, ih, iw, oc, pad)
            .expect("winograd bias");
        assert_close(&expected, &got, 1e-4, "winograd_with_bias");
    }

    #[test]
    fn test_winograd_multi_batch() {
        let n = 4;
        let c = 2;
        let ih = 6;
        let iw = 6;
        let oc = 3;
        let pad = 1;
        let input: Vec<f32> = (0..n * c * ih * iw)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let weight: Vec<f32> = (0..oc * c * 9).map(|i| (i as f32 * 0.11).cos()).collect();
        let expected = conv2d_reference(&input, &weight, None, n, c, ih, iw, oc, 3, 3, pad);
        let got = conv2d_winograd_f2x3(&input, &weight, None, n, c, ih, iw, oc, pad)
            .expect("winograd multi-batch");
        assert_close(&expected, &got, 1e-4, "winograd_multi_batch");
    }

    #[test]
    fn test_winograd_fallback_stride() {
        // stride != 1 → Winograd should NOT be selected; verify via conv2d dispatch
        let input = Tensor::new(
            (0..1 * 1 * 6 * 6).map(|i| i as f32).collect(),
            vec![1, 1, 6, 6],
        );
        let weight = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
        // stride=2 → should use im2col path, not Winograd
        let out = conv2d(&input, &weight, None, [2, 2], [0, 0, 0, 0], [1, 1], 1);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_winograd_fallback_dilation() {
        // dilation != 1 → should fallback to im2col
        let input = Tensor::new(
            (0..1 * 1 * 8 * 8).map(|i| i as f32).collect(),
            vec![1, 1, 8, 8],
        );
        let weight = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
        let out = conv2d(&input, &weight, None, [1, 1], [0, 0, 0, 0], [2, 2], 1);
        // dilation=2, kh=3 → effective kernel=5, oh=8-5+1=4, ow=4
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_winograd_small_input_3x3() {
        // Input exactly 3×3 with pad=0 → output 1×1 (less than 4×4)
        // Should NOT use Winograd (oh < 4), falls through to im2col
        let input = Tensor::new((0..9).map(|i| i as f32).collect(), vec![1, 1, 3, 3]);
        let weight = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
        let out = conv2d(&input, &weight, None, [1, 1], [0, 0, 0, 0], [1, 1], 1);
        assert_eq!(out.shape, vec![1, 1, 1, 1]);
        // Sum of 0..8 = 36
        assert!((out.data[0] - 36.0).abs() < 1e-5);
    }

    #[test]
    fn test_winograd_non_square() {
        // Non-square input [1,1,6,8]
        let ih = 6;
        let iw = 8;
        let pad = 1;
        let input: Vec<f32> = (0..ih * iw).map(|i| i as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..9).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let expected = conv2d_reference(&input, &weight, None, 1, 1, ih, iw, 1, 3, 3, pad);
        let got = conv2d_winograd_f2x3(&input, &weight, None, 1, 1, ih, iw, 1, pad)
            .expect("winograd non-square");
        assert_close(&expected, &got, 1e-4, "winograd_non_square");
    }

    #[test]
    fn test_winograd_skips_grouped_conv() {
        // Grouped conv (group=2) → Winograd dispatch requires group==1,
        // so this must use im2col and compute correctly.
        let input = Tensor::new(
            (0..1 * 4 * 6 * 6).map(|i| (i as f32 * 0.1).sin()).collect(),
            vec![1, 4, 6, 6],
        );
        // 4 output channels, 2 groups → 2 oc per group, 2 ic per group
        let weight = Tensor::new(
            (0..4 * 2 * 3 * 3)
                .map(|i| (i as f32 * 0.05).cos())
                .collect(),
            vec![4, 2, 3, 3],
        );
        let out_grouped = conv2d(&input, &weight, None, [1, 1], [1, 1, 1, 1], [1, 1], 2);
        assert_eq!(out_grouped.shape, vec![1, 4, 6, 6]);
        // Also verify via non-grouped single-group reference for each group half
        // Just check it doesn't crash and shape is correct
        assert_eq!(out_grouped.data.len(), 1 * 4 * 6 * 6);
    }

    #[test]
    fn test_winograd_dispatch_via_conv2d() {
        // Verify that conv2d with 3×3/stride=1/dilation=1/group=1/pad=1
        // produces correct results (it should dispatch to Winograd internally)
        let n = 2;
        let c = 3;
        let ih = 8;
        let iw = 8;
        let oc = 8;
        let pad = 1;
        let input_data: Vec<f32> = (0..n * c * ih * iw)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let weight_data: Vec<f32> = (0..oc * c * 9).map(|i| (i as f32 * 0.03).cos()).collect();
        let bias_data = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];

        // Direct Winograd call
        let winograd_out = conv2d_winograd_f2x3(
            &input_data,
            &weight_data,
            Some(&bias_data),
            n,
            c,
            ih,
            iw,
            oc,
            pad,
        )
        .expect("winograd dispatch");

        // Via conv2d (should auto-dispatch to Winograd)
        let input_t = Tensor::new(input_data.clone(), vec![n, c, ih, iw]);
        let weight_t = Tensor::new(weight_data.clone(), vec![oc, c, 3, 3]);
        let bias_t = Tensor::new(bias_data.clone(), vec![oc]);
        let conv2d_out = conv2d(
            &input_t,
            &weight_t,
            Some(&bias_t),
            [1, 1],
            [pad, pad, pad, pad],
            [1, 1],
            1,
        );

        assert_close(&winograd_out, &conv2d_out.data, 1e-5, "dispatch_matches");
    }
}
