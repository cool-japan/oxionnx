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
pub(crate) fn im2col(
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
pub(crate) const IM2COL_TILE: usize = 32;

/// Threshold (in output spatial positions) above which cache-blocked im2col is used.
pub(crate) const IM2COL_BLOCK_THRESHOLD: usize = 256;

/// Cache-blocked im2col: processes output positions in 2D spatial tiles
/// for better L1/L2 cache locality on large feature maps.
///
/// Produces the same column matrix as `im2col` but with better cache behavior
/// when oh*ow is large. Within each tile, the columns written per row fit in
/// fewer cache lines, reducing eviction on wide feature maps.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn im2col_blocked(
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
pub(crate) fn simd_copy_f32(src: &[f32], dst: &mut [f32]) {
    // Only the two hand-vectorised arms below read this; on every other
    // target (notably wasm32, where the `simd` feature is still on but this
    // function has no intrinsic path) the binding would be dead and would
    // trip `unused_variables` in an otherwise warning-free build.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
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
pub(crate) fn im2col_adaptive(
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
