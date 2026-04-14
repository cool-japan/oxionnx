//! Tests for SIMD-accelerated im2col and weight packing in conv.rs.

use oxionnx_core::Tensor;
use oxionnx_ops::conv;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Simple reference im2col (scalar, stride=1, dilation=1) for verification.
#[cfg(feature = "simd")]
fn im2col_reference(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let col_cols = out_h * out_w;
    let col_rows = c_in * kh * kw;
    let mut col = vec![0.0f32; col_rows * col_cols];
    let mut row = 0;
    for ic in 0..c_in {
        for ky in 0..kh {
            for kx in 0..kw {
                for oy in 0..out_h {
                    let iy = oy as isize + ky as isize - pad_h as isize;
                    for ox in 0..out_w {
                        let ix = ox as isize + kx as isize - pad_w as isize;
                        col[row * col_cols + oy * out_w + ox] =
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                input[ic * h * w + iy as usize * w + ix as usize]
                            } else {
                                0.0
                            };
                    }
                }
                row += 1;
            }
        }
    }
    col
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "{label}: mismatch at [{i}]: {x} vs {y} (diff={})",
            (x - y).abs()
        );
    }
}

// ── SIMD im2col tests ───────────────────────────────────────────────────

#[cfg(feature = "simd")]
mod simd_im2col {
    use super::*;

    #[test]
    fn test_simd_im2col_stride1_basic() {
        // 1×1×4×4 input, 3×3 kernel, no padding
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (c_in, h, w, kh, kw) = (1, 4, 4, 3, 3);
        let (pad_h, pad_w) = (0, 0);
        let out_h = h + 2 * pad_h - kh + 1; // 2
        let out_w = w + 2 * pad_w - kw + 1; // 2
        let col_rows = c_in * kh * kw;
        let col_cols = out_h * out_w;

        let expected = im2col_reference(&input, c_in, h, w, kh, kw, pad_h, pad_w, out_h, out_w);
        let mut col = vec![0.0f32; col_rows * col_cols];
        conv::im2col_simd_stride1(
            &input, c_in, h, w, 0, c_in, kh, kw, pad_h, pad_w, out_h, out_w, 0, &mut col,
        );
        assert_close(&expected, &col, 1e-6, "simd_im2col_basic");
    }

    #[test]
    fn test_simd_im2col_stride1_padded() {
        // 1×1×4×4 input, 3×3 kernel, padding=1
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (c_in, h, w, kh, kw) = (1, 4, 4, 3, 3);
        let (pad_h, pad_w) = (1, 1);
        let out_h = h + 2 * pad_h - kh + 1; // 4
        let out_w = w + 2 * pad_w - kw + 1; // 4
        let col_rows = c_in * kh * kw;
        let col_cols = out_h * out_w;

        let expected = im2col_reference(&input, c_in, h, w, kh, kw, pad_h, pad_w, out_h, out_w);
        let mut col = vec![0.0f32; col_rows * col_cols];
        conv::im2col_simd_stride1(
            &input, c_in, h, w, 0, c_in, kh, kw, pad_h, pad_w, out_h, out_w, 0, &mut col,
        );
        assert_close(&expected, &col, 1e-6, "simd_im2col_padded");
    }

    #[test]
    fn test_simd_im2col_stride1_multichannel() {
        // 1×3×8×8 input
        let (c_in, h, w, kh, kw) = (3, 8, 8, 3, 3);
        let (pad_h, pad_w) = (1, 1);
        let out_h = h + 2 * pad_h - kh + 1;
        let out_w = w + 2 * pad_w - kw + 1;
        let input: Vec<f32> = (0..c_in * h * w).map(|i| (i as f32 * 0.1).sin()).collect();
        let col_rows = c_in * kh * kw;
        let col_cols = out_h * out_w;

        let expected = im2col_reference(&input, c_in, h, w, kh, kw, pad_h, pad_w, out_h, out_w);
        let mut col = vec![0.0f32; col_rows * col_cols];
        conv::im2col_simd_stride1(
            &input, c_in, h, w, 0, c_in, kh, kw, pad_h, pad_w, out_h, out_w, 0, &mut col,
        );
        assert_close(&expected, &col, 1e-6, "simd_im2col_multichannel");
    }

    #[test]
    fn test_simd_im2col_matches_scalar() {
        // Large random-ish input: 1×16×32×32, 5×5 kernel, pad=2
        let (c_in, h, w, kh, kw) = (16, 32, 32, 5, 5);
        let (pad_h, pad_w) = (2, 2);
        let out_h = h + 2 * pad_h - kh + 1;
        let out_w = w + 2 * pad_w - kw + 1;
        let input: Vec<f32> = (0..c_in * h * w)
            .map(|i| (i as f32 * 0.017).sin() * 100.0)
            .collect();
        let col_rows = c_in * kh * kw;
        let col_cols = out_h * out_w;

        let expected = im2col_reference(&input, c_in, h, w, kh, kw, pad_h, pad_w, out_h, out_w);
        let mut col = vec![0.0f32; col_rows * col_cols];
        conv::im2col_simd_stride1(
            &input, c_in, h, w, 0, c_in, kh, kw, pad_h, pad_w, out_h, out_w, 0, &mut col,
        );
        assert_close(&expected, &col, 1e-5, "simd_im2col_matches_scalar");
    }
}

// ── Weight packing tests ────────────────────────────────────────────────

#[test]
fn test_weight_pack_roundtrip() {
    let rows = 10;
    let cols = 6;
    let panel_width = 4;
    let weights: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let packed = conv::pack_weights_panel(&weights, rows, cols, panel_width);

    let num_panels = rows.div_ceil(panel_width); // 3

    // Verify each element is accessible at the expected packed location
    for panel in 0..num_panels {
        let row_start = panel * panel_width;
        let row_end = (row_start + panel_width).min(rows);
        let panel_off = panel * panel_width * cols;
        for col in 0..cols {
            for r in 0..(row_end - row_start) {
                let orig = weights[(row_start + r) * cols + col];
                let got = packed[panel_off + col * panel_width + r];
                assert!(
                    (orig - got).abs() < 1e-6,
                    "pack mismatch panel={panel} col={col} r={r}: {orig} vs {got}"
                );
            }
        }
    }

    // Padding rows in last incomplete panel should be zero
    let last_panel = num_panels - 1;
    let last_row_end = rows;
    let last_row_start = last_panel * panel_width;
    let leftover = last_row_end - last_row_start; // 2
    let panel_off = last_panel * panel_width * cols;
    for col in 0..cols {
        for r in leftover..panel_width {
            let v = packed[panel_off + col * panel_width + r];
            assert!(v.abs() < 1e-6, "padding should be 0, got {v}");
        }
    }
}

// ── Full conv2d through SIMD path tests ─────────────────────────────────

#[test]
fn test_conv2d_simd_path_correctness() {
    // stride=1 dilation=1 → hits SIMD path when feature enabled
    #[rustfmt::skip]
    let input = Tensor::new(
        (0..1*1*6*6).map(|i| i as f32 * 0.1).collect(),
        vec![1, 1, 6, 6],
    );
    let weight = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
    let out = conv::conv2d(&input, &weight, None, [1, 1], [1, 1, 1, 1], [1, 1], 1);
    assert_eq!(out.shape, vec![1, 1, 6, 6]);

    // Verify a few known output values (sum of 3×3 window)
    // Center pixel (2,2) with pad=1: input rows [1..4] cols [1..4]
    // 0.7+0.8+0.9 + 1.3+1.4+1.5 + 1.9+2.0+2.1 = 12.6
    let center = out.data[2 * 6 + 2];
    assert!(
        (center - 12.6).abs() < 1e-3,
        "center mismatch: {center} vs 12.6"
    );
}

#[test]
fn test_conv2d_simd_vs_scalar() {
    // Compare SIMD conv2d (stride=1) vs stride=2 (uses scalar path) for consistency
    // Both should produce correct results; we verify stride=1 path against manual reference.
    let n = 1;
    let c_in = 3;
    let h = 8;
    let w = 8;
    let c_out = 4;

    let input_data: Vec<f32> = (0..n * c_in * h * w)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let weight_data: Vec<f32> = (0..c_out * c_in * 3 * 3)
        .map(|i| (i as f32 * 0.03).cos())
        .collect();
    let bias_data = vec![0.1, -0.2, 0.3, -0.4];

    let input = Tensor::new(input_data.clone(), vec![n, c_in, h, w]);
    let weight = Tensor::new(weight_data.clone(), vec![c_out, c_in, 3, 3]);
    let bias = Tensor::new(bias_data.clone(), vec![c_out]);

    // stride=1 path (uses SIMD when enabled) with pad=1
    let out_simd = conv::conv2d(
        &input,
        &weight,
        Some(&bias),
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
    );
    assert_eq!(out_simd.shape, vec![1, c_out, h, w]);

    // Naive reference: im2col with scalar loops + sgemm
    let oh = h;
    let ow = w;
    let col_rows = c_in * 3 * 3;
    let col_cols = oh * ow;
    let mut col = vec![0.0f32; col_rows * col_cols];
    // Scalar im2col
    let mut row = 0;
    for ic in 0..c_in {
        for ky in 0..3usize {
            for kx in 0..3usize {
                for oy in 0..oh {
                    let iy = oy as isize + ky as isize - 1;
                    for ox in 0..ow {
                        let ix = ox as isize + kx as isize - 1;
                        col[row * col_cols + oy * ow + ox] =
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                input_data[ic * h * w + iy as usize * w + ix as usize]
                            } else {
                                0.0
                            };
                    }
                }
                row += 1;
            }
        }
    }
    let mut ref_out = vec![0.0f32; c_out * col_cols];
    // SAFETY: test code, matmul dimensions verified above
    unsafe {
        matrixmultiply::sgemm(
            c_out,
            col_rows,
            col_cols,
            1.0,
            weight_data.as_ptr(),
            col_rows as isize,
            1,
            col.as_ptr(),
            col_cols as isize,
            1,
            0.0,
            ref_out.as_mut_ptr(),
            col_cols as isize,
            1,
        );
    }
    for oc in 0..c_out {
        let bv = bias_data[oc];
        for j in 0..col_cols {
            ref_out[oc * col_cols + j] += bv;
        }
    }

    assert_close(&ref_out, &out_simd.data, 1e-4, "conv2d_simd_vs_scalar");
}

#[test]
fn test_conv2d_simd_large() {
    // Larger regression test: 1×64×32×32, 128 filters 3×3, pad=1
    let (n, c_in, h, w) = (1, 64, 32, 32);
    let c_out = 128;
    let input = Tensor::new(
        (0..n * c_in * h * w)
            .map(|i| (i as f32 * 0.001).sin())
            .collect(),
        vec![n, c_in, h, w],
    );
    let weight = Tensor::new(
        (0..c_out * c_in * 9)
            .map(|i| (i as f32 * 0.0007).cos())
            .collect(),
        vec![c_out, c_in, 3, 3],
    );
    let bias = Tensor::new((0..c_out).map(|i| i as f32 * 0.01).collect(), vec![c_out]);

    let out = conv::conv2d(
        &input,
        &weight,
        Some(&bias),
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
    );
    assert_eq!(out.shape, vec![n, c_out, h, w]);
    assert_eq!(out.data.len(), n * c_out * h * w);

    // Sanity: no NaN or Inf
    for (i, &v) in out.data.iter().enumerate() {
        assert!(v.is_finite(), "non-finite at index {i}: {v}");
    }
}

/// Regression: basic conv2d operations still produce correct results.
#[test]
fn test_conv2d_existing_tests_still_pass() {
    // 1x1 identity kernel
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
    let out = conv::conv2d(&input, &weight, None, [1, 1], [0, 0, 0, 0], [1, 1], 1);
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);

    // 3x3 sum kernel on 4x4 with stride=2 (non-SIMD path)
    let input2 = Tensor::new(vec![1.0; 16], vec![1, 1, 4, 4]);
    let weight2 = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
    let out2 = conv::conv2d(&input2, &weight2, None, [2, 2], [0, 0, 0, 0], [1, 1], 1);
    assert_eq!(out2.shape, vec![1, 1, 2, 2]);
    assert_eq!(out2.data, vec![4.0, 4.0, 4.0, 4.0]);

    // With bias
    let input3 = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
    let weight3 = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
    let bias3 = Tensor::new(vec![10.0], vec![1]);
    let out3 = conv::conv2d(
        &input3,
        &weight3,
        Some(&bias3),
        [1, 1],
        [0, 0, 0, 0],
        [1, 1],
        1,
    );
    assert_eq!(out3.data, vec![11.0, 11.0, 11.0, 11.0]);
}
