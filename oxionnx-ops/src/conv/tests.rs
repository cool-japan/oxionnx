#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::identity_op
)]

use oxionnx_core::Tensor;

use crate::conv::conv2d::conv2d;
use crate::conv::im2col::{im2col, im2col_blocked};
use crate::conv::pooling::{
    avg_pool2d, conv_transpose2d, global_avg_pool, global_max_pool, max_pool2d,
};
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
