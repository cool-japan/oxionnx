#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::identity_op
)]

use oxionnx_core::Tensor;

use crate::conv::conv2d::conv2d;
use crate::conv::im2col::{im2col, im2col_blocked};
use crate::conv::pooling::{avg_pool2d, global_avg_pool, global_max_pool, max_pool2d};
use crate::conv::transpose::conv_transpose2d;
use crate::conv::winograd::conv2d_winograd_f2x3;

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
    let got = conv2d_winograd_f2x3(&input, &weight, None, 1, 1, 5, 5, 1, 1).expect("winograd pad");
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

// ── Performance probe (run with `--run-ignored all`) ─────────────────────────

/// Compare the Winograd F(2,3) path against im2col + sgemm on realistic CNN
/// layer shapes. Not an assertion — a measurement aid; see the
/// `winograd_is_profitable` cost model in `conv2d.rs`.
#[test]
#[ignore = "timing probe, not a correctness assertion"]
fn perf_probe_winograd_vs_im2col() {
    use crate::conv::im2col::im2col_adaptive;
    use std::time::Instant;

    #[allow(unsafe_code)]
    fn im2col_sgemm(
        input: &[f32],
        weight: &[f32],
        c: usize,
        ih: usize,
        iw: usize,
        oc: usize,
        pad: usize,
    ) -> Vec<f32> {
        let oh = ih + 2 * pad - 2;
        let ow = iw + 2 * pad - 2;
        let col_rows = c * 9;
        let col_cols = oh * ow;
        let mut col = vec![0.0f32; col_rows * col_cols];
        im2col_adaptive(
            input,
            c,
            ih,
            iw,
            0,
            c,
            3,
            3,
            [1, 1],
            [pad, pad, pad, pad],
            [1, 1],
            oh,
            ow,
            0,
            &mut col,
        );
        let mut out = vec![0.0f32; oc * col_cols];
        unsafe {
            matrixmultiply::sgemm(
                oc,
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
                out.as_mut_ptr(),
                col_cols as isize,
                1,
            );
        }
        out
    }

    // (c, oc, h, w) — representative ResNet-ish 3x3 layers plus a small one.
    let cases = [
        (3_usize, 64_usize, 224_usize, 224_usize),
        (64, 64, 56, 56),
        (128, 128, 28, 28),
        (256, 256, 14, 14),
        (512, 512, 7, 7),
        (16, 16, 32, 32),
        (8, 8, 64, 64),
    ];
    for (c, oc, h, w) in cases {
        let input: Vec<f32> = (0..c * h * w).map(|i| (i as f32 * 0.001).sin()).collect();
        let weight: Vec<f32> = (0..oc * c * 9).map(|i| (i as f32 * 0.003).cos()).collect();
        let pad = 1;

        // Warm-up + timed runs.
        let _ = conv2d_winograd_f2x3(&input, &weight, None, 1, c, h, w, oc, pad);
        let t0 = Instant::now();
        let a = conv2d_winograd_f2x3(&input, &weight, None, 1, c, h, w, oc, pad)
            .expect("winograd probe");
        let wino = t0.elapsed();

        let _ = im2col_sgemm(&input, &weight, c, h, w, oc, pad);
        let t1 = Instant::now();
        let b = im2col_sgemm(&input, &weight, c, h, w, oc, pad);
        let gemm = t1.elapsed();

        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "c={c:4} oc={oc:4} {h:3}x{w:3}: winograd {:>10.3?}  im2col+sgemm {:>10.3?}  \
             ratio {:>6.2}x  maxdiff {max_diff:.3e}  oc*c={}",
            wino,
            gemm,
            wino.as_secs_f64() / gemm.as_secs_f64().max(1e-12),
            oc * c
        );
    }
}

// ── Rank-generic / specialised path parity ──────────────────────────────────
//
// Every performance specialisation in this module claims to be *bit-identical*
// to the path it replaces. These tests assert that claim on the raw bits, not
// within a tolerance — a difference of one ULP would already break the
// bit-exact integer cross-check `oxionnx-directml`'s `reference_vs_ops` runs
// against `conv2d`.

/// Deterministic, exactly-representable filler (integers scaled by 1/4).
fn ramp(n: usize, modulus: usize, offset: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % modulus) as f32 - offset as f32) * scale)
        .collect()
}

/// `(input_shape, weight_shape, strides, pads, dilations, group)` for a rank-2
/// parity case.
type Rank2Case = (
    [usize; 4],
    [usize; 4],
    [usize; 2],
    [usize; 4],
    [usize; 2],
    usize,
);

fn assert_bitwise(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &e)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "{label}[{i}]: {g} ({:#010x}) != {e} ({:#010x})",
            g.to_bits(),
            e.to_bits()
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[allow(unsafe_code)]
fn parallel_sgemm_is_bitwise_identical_to_one_sequential_call() {
    // `conv2d_into` splits the GEMM by rows of A across rayon on the batch-1,
    // group-1 path. That is only legitimate if matrixmultiply's K-blocking —
    // and hence the accumulation order of each output element — is independent
    // of M. Assert it rather than reason about it.
    use crate::conv::conv2d::parallel_sgemm;

    for &(m, k, n) in &[
        (64_usize, 27_usize, 196_usize),
        (128, 288, 49),
        (32, 64, 1024),
        (7, 13, 29),
        (1, 5, 3),
        (256, 4608, 49),
    ] {
        let a = ramp(m * k, 23, 11, 0.25);
        let b = ramp(k * n, 31, 15, 0.5);
        let mut par = vec![0.0f32; m * n];
        parallel_sgemm(m, k, n, &a, &b, &mut par, n);

        let mut seq = vec![0.0f32; m * n];
        // SAFETY: a is [m,k], b is [k,n], seq is [m,n], all row-major.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                seq.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        assert_bitwise(&par, &seq, &format!("parallel_sgemm {m}x{k}x{n}"));

        // Same call with a wider row stride — how the blocked im2col path
        // writes one column block into the columns of `out` it owns. The
        // result must still match, *and* the columns outside the `n`-wide
        // window must be untouched: they belong to the neighbouring column
        // blocks of the same convolution.
        for ldc in [n + 1, n + 7, 2 * n + 3] {
            let mut strided = vec![f32::NAN; m * ldc];
            parallel_sgemm(m, k, n, &a, &b, &mut strided, ldc);
            for row in 0..m {
                assert_bitwise(
                    &strided[row * ldc..row * ldc + n],
                    &seq[row * n..(row + 1) * n],
                    &format!("parallel_sgemm {m}x{k}x{n} ldc {ldc} row {row}"),
                );
                for (col, &v) in strided[row * ldc + n..(row + 1) * ldc].iter().enumerate() {
                    assert!(
                        v.is_nan(),
                        "parallel_sgemm ldc {ldc} wrote outside its window \
                         at row {row} column {}",
                        n + col
                    );
                }
            }
        }
    }
}

#[test]
fn conv1d_lowering_matches_the_equivalent_2d_convolution_bitwise() {
    // `[N, C, W]` is run as `[N, C, 1, W]`. With kH = 1, pad_h = 0, stride_h =
    // 1 and dilation_h = 1 the height axis contributes exactly one window, so
    // the lowering must be exact — not merely close.
    use crate::conv::conv_nd::{conv_into, ConvParams};

    let (n, c, w_in, f, kw) = (2_usize, 3_usize, 17_usize, 5_usize, 4_usize);
    let input = ramp(n * c * w_in, 19, 9, 0.25);
    let weight = ramp(f * c * kw, 13, 6, 0.5);
    let bias = ramp(f, 7, 3, 0.25);

    for &(stride, pad_b, pad_e, dilation) in &[
        (1_usize, 0_usize, 0_usize, 1_usize),
        (2, 2, 1, 1),
        (1, 3, 3, 2),
        (3, 1, 2, 2),
    ] {
        let eff_k = (kw - 1) * dilation + 1;
        let out_w = (w_in + pad_b + pad_e - eff_k) / stride + 1;

        let mut got = vec![0.0f32; n * f * out_w];
        conv_into(
            &input,
            &[n, c, w_in],
            &weight,
            &[f, c, kw],
            Some(&bias),
            &ConvParams {
                strides: &[stride],
                pads: &[pad_b, pad_e],
                dilations: &[dilation],
                group: 1,
            },
            &mut got,
            &[n, f, out_w],
        )
        .expect("1D conv");

        let mut want = vec![0.0f32; n * f * out_w];
        conv_into(
            &input,
            &[n, c, 1, w_in],
            &weight,
            &[f, c, 1, kw],
            Some(&bias),
            &ConvParams {
                strides: &[1, stride],
                pads: &[0, pad_b, 0, pad_e],
                dilations: &[1, dilation],
                group: 1,
            },
            &mut want,
            &[n, f, 1, out_w],
        )
        .expect("2D conv");

        assert_bitwise(
            &got,
            &want,
            &format!("conv1d s{stride} p{pad_b},{pad_e} d{dilation}"),
        );
    }
}

#[test]
fn generic_nd_conv_matches_the_rank2_kernel_bitwise() {
    // The rank-2 kernel (SIMD/blocked im2col + tuned GEMM) and the generic N-D
    // gather must agree exactly: im2col is a pure gather, and both feed the
    // same sgemm, so the only way to differ is a wrong index.
    use crate::conv::conv2d::conv2d_into_slices;
    use crate::conv::conv_nd::{conv_nd_into, ConvParams};

    let cases: &[Rank2Case] = &[
        ([1, 3, 9, 11], [4, 3, 3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 1),
        ([2, 4, 8, 8], [6, 2, 2, 3], [2, 1], [1, 2, 0, 1], [1, 1], 2),
        (
            [1, 2, 12, 10],
            [3, 2, 3, 2],
            [1, 2],
            [0, 0, 0, 0],
            [2, 3],
            1,
        ),
        ([1, 6, 7, 7], [6, 2, 3, 3], [1, 1], [2, 2, 2, 2], [1, 1], 3),
    ];
    for &(in_shape, w_shape, strides, pads, dilations, group) in cases {
        let input = ramp(in_shape.iter().product(), 17, 8, 0.25);
        let weight = ramp(w_shape.iter().product(), 11, 5, 0.5);
        let bias = ramp(w_shape[0], 5, 2, 0.25);
        let out_shape = crate::conv::spatial::compute_conv_out_shape(
            "Conv", &in_shape, &w_shape, &strides, &pads, &dilations,
        )
        .expect("valid geometry");
        let out_len: usize = out_shape.iter().product();

        let mut specialised = vec![0.0f32; out_len];
        conv2d_into_slices(
            &input,
            &in_shape,
            &weight,
            &w_shape,
            Some(&bias),
            strides,
            pads,
            dilations,
            group,
            &mut specialised,
            &out_shape,
        );

        let mut generic = vec![0.0f32; out_len];
        conv_nd_into(
            &input,
            &in_shape,
            &weight,
            &w_shape,
            Some(&bias),
            &ConvParams {
                strides: &strides,
                pads: &pads,
                dilations: &dilations,
                group,
            },
            &mut generic,
            &out_shape,
        );

        assert_bitwise(
            &generic,
            &specialised,
            &format!("conv {in_shape:?} g{group}"),
        );
    }
}

#[test]
fn conv_transpose_generic_matches_rank2_bitwise() {
    use crate::conv::spatial::compute_conv_transpose_out_shape;
    use crate::conv::transpose::{scatter_generic, scatter_rank2, ConvTransposeParams};

    let cases: &[Rank2Case] = &[
        ([1, 2, 5, 4], [2, 3, 3, 3], [1, 1], [0, 0, 0, 0], [1, 1], 1),
        ([2, 4, 4, 5], [4, 2, 2, 3], [2, 3], [1, 2, 1, 0], [1, 1], 2),
        ([1, 3, 6, 6], [3, 2, 3, 2], [1, 2], [0, 1, 2, 1], [2, 3], 1),
    ];
    for &(in_shape, w_shape, strides, pads, dilations, group) in cases {
        let input = ramp(in_shape.iter().product(), 23, 11, 0.25);
        let weight = ramp(w_shape.iter().product(), 13, 6, 0.5);
        let out_shape = compute_conv_transpose_out_shape(
            "ConvTranspose",
            &in_shape,
            &w_shape,
            &strides,
            &pads,
            &[0, 0],
            &dilations,
            group,
        )
        .expect("valid geometry");
        let out_len: usize = out_shape.iter().product();
        let params = ConvTransposeParams {
            strides: &strides,
            pads: &pads,
            dilations: &dilations,
            group,
        };

        let mut specialised = vec![0.0f32; out_len];
        scatter_rank2(
            &input,
            &in_shape,
            &weight,
            &w_shape,
            &params,
            &mut specialised,
            &out_shape,
        );
        let mut generic = vec![0.0f32; out_len];
        scatter_generic(
            &input,
            &in_shape,
            &weight,
            &w_shape,
            &params,
            &mut generic,
            &out_shape,
        );
        assert_bitwise(
            &generic,
            &specialised,
            &format!("conv_transpose {in_shape:?} g{group}"),
        );
    }
}

#[test]
fn pool_compat_wrappers_match_an_independent_naive_reference() {
    // `max_pool2d` / `avg_pool2d` are now thin wrappers over the shared N-D
    // kernel; pin their (floor-mode, dilation-1) behaviour against a
    // straightforward direct implementation so the unification cannot drift.
    let (n, c, h, w) = (2_usize, 3_usize, 7_usize, 6_usize);
    let data = ramp(n * c * h * w, 29, 14, 0.25);
    let input = Tensor::new(data.clone(), vec![n, c, h, w]);

    for &(kh, kw, sh, sw, pt, pl, pb, pr) in &[
        (
            2_usize, 2_usize, 2_usize, 2_usize, 0_usize, 0_usize, 0_usize, 0_usize,
        ),
        (3, 3, 1, 1, 1, 1, 1, 1),
        (3, 2, 2, 1, 1, 0, 0, 1),
    ] {
        let oh = (h + pt + pb - kh) / sh + 1;
        let ow = (w + pl + pr - kw) / sw + 1;
        let mut want_max = vec![f32::NEG_INFINITY; n * c * oh * ow];
        let mut want_avg_excl = vec![0.0f32; n * c * oh * ow];
        let mut want_avg_incl = vec![0.0f32; n * c * oh * ow];
        for nc in 0..n * c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for ky in 0..kh {
                        let iy = (oy * sh + ky) as isize - pt as isize;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = (ox * sw + kx) as isize - pl as isize;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let v = data[nc * h * w + iy as usize * w + ix as usize];
                            if v > best {
                                best = v;
                            }
                            sum += v;
                            count += 1;
                        }
                    }
                    let o = (nc * oh + oy) * ow + ox;
                    want_max[o] = best;
                    want_avg_excl[o] = if count > 0 { sum / count as f32 } else { 0.0 };
                    want_avg_incl[o] = sum / (kh * kw) as f32;
                }
            }
        }

        let got_max = max_pool2d(&input, [kh, kw], [sh, sw], [pt, pl, pb, pr]);
        assert_eq!(got_max.shape, vec![n, c, oh, ow]);
        assert_bitwise(&got_max.data, &want_max, "max_pool2d");

        let got_excl = avg_pool2d(&input, [kh, kw], [sh, sw], [pt, pl, pb, pr], false);
        assert_bitwise(&got_excl.data, &want_avg_excl, "avg_pool2d exclude");

        let got_incl = avg_pool2d(&input, [kh, kw], [sh, sw], [pt, pl, pb, pr], true);
        assert_bitwise(&got_incl.data, &want_avg_incl, "avg_pool2d include");
    }
}

#[test]
fn conv2d_into_does_not_depend_on_the_incoming_buffer_contents() {
    // The blanket zero-fill at the top of `conv2d_into` was removed because
    // every branch fully overwrites the output. Prove it by running each
    // dispatch branch twice from deliberately poisoned buffers.
    use crate::conv::conv2d::conv2d_into_slices;

    let cases: &[Rank2Case] = &[
        // 1×1 fast path
        ([2, 4, 5, 5], [6, 4, 1, 1], [1, 1], [0, 0, 0, 0], [1, 1], 1),
        // Winograd F(2,3) (small enough to stay under the cost gate)
        ([1, 2, 8, 8], [3, 2, 3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 1),
        // im2col single job
        ([1, 3, 9, 9], [4, 3, 3, 3], [2, 2], [1, 0, 0, 1], [1, 1], 1),
        // im2col multi job (batch and group)
        ([3, 4, 6, 7], [8, 2, 2, 3], [1, 1], [1, 1, 1, 1], [1, 1], 2),
    ];
    for &(in_shape, w_shape, strides, pads, dilations, group) in cases {
        let input = ramp(in_shape.iter().product(), 17, 8, 0.25);
        let weight = ramp(w_shape.iter().product(), 11, 5, 0.5);
        let bias = ramp(w_shape[0], 5, 2, 0.25);
        let out_shape = crate::conv::spatial::compute_conv_out_shape(
            "Conv", &in_shape, &w_shape, &strides, &pads, &dilations,
        )
        .expect("valid geometry");
        let out_len: usize = out_shape.iter().product();

        let mut zeroed = vec![0.0f32; out_len];
        let mut poisoned = vec![f32::NAN; out_len];
        for (i, v) in poisoned.iter_mut().enumerate() {
            *v = if i % 3 == 0 { f32::NAN } else { -1e30 };
        }
        for buf in [&mut zeroed, &mut poisoned] {
            conv2d_into_slices(
                &input,
                &in_shape,
                &weight,
                &w_shape,
                Some(&bias),
                strides,
                pads,
                dilations,
                group,
                buf,
                &out_shape,
            );
        }
        assert_bitwise(&poisoned, &zeroed, &format!("conv2d_into {in_shape:?}"));
    }
}

/// Timing probe for the batch-1 / group-1 parallelisation (see the report
/// note). Not a correctness assertion — run with `--run-ignored all`.
#[test]
#[ignore = "timing probe, not a correctness assertion"]
fn perf_probe_conv2d_batch1_group1() {
    use std::time::Instant;

    let cases = [
        (
            3_usize, 64_usize, 224_usize, 224_usize, 7_usize, 2_usize, 3_usize,
        ),
        (64, 64, 56, 56, 3, 1, 1),
        (128, 128, 28, 28, 3, 1, 1),
        (256, 256, 14, 14, 3, 1, 1),
        (512, 512, 7, 7, 3, 1, 1),
        (256, 512, 14, 14, 1, 1, 0),
    ];
    println!("threads = {}", rayon::current_num_threads());
    for (c, oc, h, w, k, s, p) in cases {
        let input = Tensor::new(ramp(c * h * w, 251, 125, 0.015625), vec![1, c, h, w]);
        let weight = Tensor::new(ramp(oc * c * k * k, 127, 63, 0.0078125), vec![oc, c, k, k]);
        // Report the *minimum* of many runs: this tree is built concurrently
        // by sibling agents, and under a 3x-oversubscribed machine the mean of
        // a threaded benchmark measures the load, not the kernel.
        let mut best = std::time::Duration::MAX;
        for _ in 0..15 {
            let t = Instant::now();
            let out = conv2d(&input, &weight, None, [s, s], [p, p, p, p], [1, 1], 1);
            let dt = t.elapsed();
            std::hint::black_box(&out);
            best = best.min(dt);
        }
        println!("c={c:4} oc={oc:4} {h:3}x{w:3} k{k} s{s} p{p}: {best:>12.3?}");
    }
}

// ── Column-blocked im2col ───────────────────────────────────────────────────
//
// Above `IM2COL_WORKSPACE_MAX_BYTES` the im2col workspace is built one column
// block at a time instead of whole (`inswapper_128` otherwise asks for a
// single 411 MB allocation per frame — an abort risk on wasm32, where a failed
// `WebAssembly.Memory.grow` kills the module and the high-water mark never
// comes back down). The claim is that this is *bit-identical*, not merely
// close: the gather is exact, and `matrixmultiply`'s N loop is its outermost
// loop, so splitting N externally cannot reorder any element's
// K-accumulation. Both halves of that claim are asserted below.

#[test]
#[allow(unsafe_code)]
fn sgemm_column_blocks_are_bitwise_identical_to_one_call() {
    // The load-bearing premise of the blocked im2col path: computing C's
    // columns in disjoint blocks — each `sgemm` writing into a strided view of
    // the full C — must reproduce one wide `sgemm` exactly.
    for &(m, k, n) in &[
        (3_usize, 6272_usize, 1024_usize),
        (256, 4608, 512),
        (64, 27, 196),
        (7, 13, 29),
        (1, 5, 3),
        (33, 129, 257),
    ] {
        let a = ramp(m * k, 23, 11, 0.25);
        let b = ramp(k * n, 31, 15, 0.5);

        let mut want = vec![0.0f32; m * n];
        // SAFETY: a is [m,k], b is [k,n], want is [m,n], all row-major.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                want.as_mut_ptr(),
                n as isize,
                1,
            );
        }

        for &block in &[1_usize, 2, 5, 64, 251, 1024] {
            if block > n {
                continue;
            }
            let mut got = vec![f32::NAN; m * n];
            let mut start = 0;
            while start < n {
                let width = block.min(n - start);
                // Gather the block's columns of B into a packed [k, width].
                let mut b_block = vec![0.0f32; k * width];
                for r in 0..k {
                    b_block[r * width..(r + 1) * width]
                        .copy_from_slice(&b[r * n + start..r * n + start + width]);
                }
                // SAFETY: `got` is [m, n] row-major; this writes the
                // `width` columns starting at `start`, row stride `n`.
                unsafe {
                    matrixmultiply::sgemm(
                        m,
                        k,
                        width,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        b_block.as_ptr(),
                        width as isize,
                        1,
                        0.0,
                        got[start..].as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
                start += width;
            }
            assert_bitwise(&got, &want, &format!("sgemm {m}x{k}x{n} block {block}"));
        }
    }
}

#[test]
fn im2col_block_cols_respects_the_workspace_cap() {
    use crate::conv::conv2d::{im2col_block_cols, IM2COL_WORKSPACE_MAX_BYTES};

    const CAP: usize = IM2COL_WORKSPACE_MAX_BYTES;

    // Under the cap the full column matrix is kept in one piece — the
    // historical code path, unchanged.
    assert_eq!(im2col_block_cols(147, 16384, 128, CAP), 16384);
    assert_eq!(im2col_block_cols(2304, 4096, 64, CAP), 4096);
    assert_eq!(im2col_block_cols(0, 4096, 64, CAP), 4096);
    assert_eq!(im2col_block_cols(9216, 0, 0, CAP), 0);

    // Every `inswapper_128` convolution that exceeds the cap, as
    // `(col_rows, col_cols, ow, expected_blocks)`.
    let over_cap = [
        (6272_usize, 16384_usize, 128_usize, 7_usize), // [3,128,7,7]  → 411 MB
        (4608, 16384, 128, 5),                         // [256,512,3,3] → 302 MB
        (9216, 4096, 64, 3),                           // [512,1024,3,3] → 151 MB
        (2304, 16384, 128, 3),                         // [128,256,3,3] → 151 MB
        (1152, 16384, 128, 2),                         // [256,128,3,3] →  76 MB
    ];
    for (col_rows, col_cols, ow, blocks) in over_cap {
        let bw = im2col_block_cols(col_rows, col_cols, ow, CAP);
        assert!(
            col_rows * bw * 4 <= CAP,
            "block {bw} of {col_rows} rows exceeds the {CAP}-byte cap"
        );
        assert_eq!(
            bw % ow,
            0,
            "block {bw} is not a whole number of output rows"
        );
        assert_eq!(col_cols.div_ceil(bw), blocks, "block count for {col_rows}");
    }

    // A single output row wider than the cap still yields a usable width.
    let bw = im2col_block_cols(1 << 20, 4096, 4096, CAP);
    assert!((1..=16).contains(&bw), "degenerate cap width: {bw}");
    assert!((1 << 20) * bw * 4 <= CAP);
    // A cap of zero cannot be honoured; one column is the floor.
    assert_eq!(im2col_block_cols(8, 100, 10, 0), 1);
}

/// Run one convolution with the workspace cap disabled (the monolithic im2col
/// reference) and again at each of `caps`, asserting the outputs match bit for
/// bit. Returns the number of column blocks the smallest cap produced.
fn assert_blocked_matches_monolithic(case: Rank2Case, caps: &[usize], label: &str) -> usize {
    use crate::conv::conv2d::{conv2d_into_slices_capped, im2col_block_cols};

    let (in_shape, w_shape, strides, pads, dilations, group) = case;
    let input = ramp(in_shape.iter().product(), 17, 8, 0.25);
    let weight = ramp(w_shape.iter().product(), 11, 5, 0.5);
    let bias = ramp(w_shape[0], 5, 2, 0.25);
    let out_shape = crate::conv::spatial::compute_conv_out_shape(
        "Conv", &in_shape, &w_shape, &strides, &pads, &dilations,
    )
    .expect("valid geometry");
    let out_len: usize = out_shape.iter().product();
    let (oh, ow) = (out_shape[2], out_shape[3]);
    let col_rows = w_shape[1] * w_shape[2] * w_shape[3];

    let mut want = vec![0.0f32; out_len];
    conv2d_into_slices_capped(
        &input,
        &in_shape,
        &weight,
        &w_shape,
        Some(&bias),
        strides,
        pads,
        dilations,
        group,
        &mut want,
        &out_shape,
        usize::MAX,
    );

    let mut blocks = 1;
    for &cap in caps {
        let bw = im2col_block_cols(col_rows, oh * ow, ow, cap);
        blocks = blocks.max((oh * ow).div_ceil(bw.max(1)));
        let mut got = vec![f32::NAN; out_len];
        conv2d_into_slices_capped(
            &input,
            &in_shape,
            &weight,
            &w_shape,
            Some(&bias),
            strides,
            pads,
            dilations,
            group,
            &mut got,
            &out_shape,
            cap,
        );
        assert_bitwise(&got, &want, &format!("{label} cap {cap} (width {bw})"));
    }
    blocks
}

#[test]
fn blocked_im2col_matches_the_monolithic_path_bitwise() {
    // Small shapes with the cap driven down far enough to force many blocks,
    // including caps too small to hold a whole output row so the gather has to
    // handle partial-row heads and tails. Covers strided, dilated, padded,
    // grouped and batched (multi-job) convolutions — i.e. both the single-job
    // and the `par_chunks_mut` job paths.
    let caps: &[usize] = &[usize::MAX, 1 << 20, 4096, 512, 64, 1];
    let cases: &[(&str, Rank2Case)] = &[
        (
            "padded 3x3",
            ([1, 3, 9, 11], [4, 3, 3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 1),
        ),
        (
            "stride 2x1 asymmetric pad",
            (
                [1, 4, 13, 12],
                [6, 4, 3, 3],
                [2, 1],
                [1, 2, 0, 1],
                [1, 1],
                1,
            ),
        ),
        (
            "dilated 2x3",
            (
                [1, 2, 12, 14],
                [3, 2, 3, 2],
                [1, 2],
                [0, 0, 0, 0],
                [2, 3],
                1,
            ),
        ),
        (
            "grouped x3 padded",
            ([1, 6, 9, 9], [9, 2, 3, 3], [1, 1], [2, 2, 2, 2], [1, 1], 3),
        ),
        (
            "batched grouped (multi job)",
            ([3, 4, 7, 8], [8, 2, 2, 3], [1, 1], [1, 1, 1, 1], [1, 1], 2),
        ),
        (
            "batched strided dilated (multi job)",
            (
                [2, 4, 15, 15],
                [4, 2, 3, 3],
                [2, 2],
                [1, 1, 1, 1],
                [2, 2],
                2,
            ),
        ),
        (
            "wide kernel, single output row",
            ([1, 3, 7, 20], [5, 3, 7, 3], [1, 1], [0, 0, 0, 0], [1, 1], 1),
        ),
    ];
    for &(label, case) in cases {
        let blocks = assert_blocked_matches_monolithic(case, caps, label);
        assert!(blocks > 1, "{label}: never actually blocked");
    }
}

#[test]
fn blocked_im2col_matches_at_the_production_cap() {
    // The same geometry as `inswapper_128`'s final convolution ([3,128,7,7]
    // over a 128×128 output, 411 MB of im2col) at half the output extent, so
    // it still trips the 64 MiB cap — 102.8 MB, two blocks — while keeping the
    // reference run affordable inside the test-suite.
    use crate::conv::conv2d::{im2col_block_cols, IM2COL_WORKSPACE_MAX_BYTES};

    let case: Rank2Case = ([1, 128, 70, 70], [3, 128, 7, 7], [1, 1], [0; 4], [1, 1], 1);
    let bw = im2col_block_cols(6272, 64 * 64, 64, IM2COL_WORKSPACE_MAX_BYTES);
    assert!(bw < 64 * 64, "shape no longer exercises blocking");
    let blocks = assert_blocked_matches_monolithic(
        case,
        &[IM2COL_WORKSPACE_MAX_BYTES],
        "inswapper final conv (half extent)",
    );
    assert_eq!(blocks, 2);
}

#[test]
#[ignore = "allocates the 411 MB / 302 MB monolithic references it compares against"]
fn blocked_im2col_matches_on_the_inswapper_worst_shapes() {
    // The two convolutions that motivated the cap, at full size:
    //   Conv_612  [3,128,7,7]   over [1,128,134,134] → 411.04 MB im2col
    //   Conv_594  [256,512,3,3] over [1,512,128,128] → 301.99 MB im2col
    let cases: &[(&str, Rank2Case)] = &[
        (
            "inswapper Conv_612 (411 MB)",
            (
                [1, 128, 134, 134],
                [3, 128, 7, 7],
                [1, 1],
                [0; 4],
                [1, 1],
                1,
            ),
        ),
        (
            "inswapper Conv_594 (302 MB)",
            (
                [1, 512, 128, 128],
                [256, 512, 3, 3],
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                1,
            ),
        ),
    ];
    for &(label, case) in cases {
        let blocks = assert_blocked_matches_monolithic(
            case,
            &[crate::conv::conv2d::IM2COL_WORKSPACE_MAX_BYTES],
            label,
        );
        assert!(blocks > 1, "{label}: never actually blocked");
    }
}

/// A/B timing probe for the workspace cap on the five `inswapper_128`
/// convolutions that exceed it, plus two that do not (which must take the
/// unchanged code path). Both sides run in one process off one build, since a
/// cross-build comparison on a machine shared with sibling agents measures the
/// load rather than the kernel. Run with `--run-ignored all`.
#[test]
#[ignore = "timing probe, not a correctness assertion"]
fn perf_probe_conv2d_im2col_blocking() {
    use crate::conv::conv2d::{conv2d_into_slices_capped, IM2COL_WORKSPACE_MAX_BYTES};
    use std::time::{Duration, Instant};

    /// `(label, input shape, weight shape, strides, pads)`.
    type ProbeCase = (&'static str, [usize; 4], [usize; 4], [usize; 2], [usize; 4]);

    let cases: &[ProbeCase] = &[
        (
            "Conv_612 411MB",
            [1, 128, 134, 134],
            [3, 128, 7, 7],
            [1, 1],
            [0; 4],
        ),
        (
            "Conv_594 302MB",
            [1, 512, 128, 128],
            [256, 512, 3, 3],
            [1, 1],
            [1, 1, 1, 1],
        ),
        (
            "Conv_590 151MB",
            [1, 1024, 64, 64],
            [512, 1024, 3, 3],
            [1, 1],
            [1, 1, 1, 1],
        ),
        (
            "Conv_596 151MB",
            [1, 256, 128, 128],
            [128, 256, 3, 3],
            [1, 1],
            [1, 1, 1, 1],
        ),
        (
            "Conv_042  76MB",
            [1, 128, 128, 128],
            [256, 128, 3, 3],
            [1, 1],
            [1, 1, 1, 1],
        ),
        (
            "Conv_062  38MB (under cap)",
            [1, 1024, 34, 34],
            [1024, 1024, 3, 3],
            [1, 1],
            [0; 4],
        ),
        (
            "Conv_040  10MB (under cap)",
            [1, 3, 134, 134],
            [128, 3, 7, 7],
            [1, 1],
            [0; 4],
        ),
    ];

    println!("threads = {}", rayon::current_num_threads());
    for &(label, in_shape, w_shape, strides, pads) in cases {
        let input = ramp(in_shape.iter().product(), 251, 125, 0.015625);
        let weight = ramp(w_shape.iter().product(), 127, 63, 0.0078125);
        let out_shape = crate::conv::spatial::compute_conv_out_shape(
            "Conv",
            &in_shape,
            &w_shape,
            &strides,
            &pads,
            &[1, 1],
        )
        .expect("valid geometry");
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len];

        // Interleave the two variants inside the repetition loop and report
        // the minimum: sibling agents build in this tree, and a load spike
        // that lands on one whole arm of a sequential A/B is indistinguishable
        // from a regression. Adjacent runs share the spike.
        let mut timing = [Duration::MAX; 2];
        for _ in 0..9 {
            for (slot, cap) in [usize::MAX, IM2COL_WORKSPACE_MAX_BYTES].iter().enumerate() {
                let t = Instant::now();
                conv2d_into_slices_capped(
                    &input,
                    &in_shape,
                    &weight,
                    &w_shape,
                    None,
                    strides,
                    pads,
                    [1, 1],
                    1,
                    &mut out,
                    &out_shape,
                    *cap,
                );
                let dt = t.elapsed();
                std::hint::black_box(&out);
                timing[slot] = timing[slot].min(dt);
            }
        }
        let ratio = timing[1].as_secs_f64() / timing[0].as_secs_f64().max(1e-12);
        let col_rows = w_shape[1] * w_shape[2] * w_shape[3];
        let col_cols = out_shape[2] * out_shape[3];
        let bw = crate::conv::conv2d::im2col_block_cols(
            col_rows,
            col_cols,
            out_shape[3],
            IM2COL_WORKSPACE_MAX_BYTES,
        );
        println!(
            "{label:28}: monolithic {:>10.3?}  blocked {:>10.3?}  {:+6.2}%   \
             workspace {:>6.1} MiB -> {:>5.1} MiB ({} blocks)",
            timing[0],
            timing[1],
            (ratio - 1.0) * 100.0,
            (col_rows * col_cols * 4) as f64 / (1024.0 * 1024.0),
            (col_rows * bw * 4) as f64 / (1024.0 * 1024.0),
            col_cols.div_ceil(bw),
        );
    }
}
