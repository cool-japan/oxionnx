//! QLinearConv2d tests.

use oxionnx_core::Tensor;

use crate::quantized::qlinear_conv2d;

use super::common::{reference_conv2d, relative_error};

#[test]
fn test_qlinear_conv2d_1x1_kernel() {
    let x_f32 = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
            8.5,
        ],
        vec![1, 2, 3, 3],
    );
    let w_f32 = Tensor::new(
        vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.6, 0.2, -0.3],
        vec![4, 2, 1, 1],
    );
    let ref_out = reference_conv2d(
        &x_f32.data,
        1,
        2,
        3,
        3,
        &w_f32.data,
        4,
        1,
        1,
        None,
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    );
    let x_scale = 9.0 / 127.0;
    let x_zp: i8 = 0;
    let x_q_data: Vec<f32> = x_f32
        .data
        .iter()
        .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
        .collect();
    let x_q = Tensor::new(x_q_data, vec![1, 2, 3, 3]);
    let w_scale = vec![0.6 / 127.0];
    let w_zp = vec![0i8];
    let w_q_data: Vec<f32> = w_f32
        .data
        .iter()
        .map(|&v| (v / w_scale[0]).round().clamp(-128.0, 127.0))
        .collect();
    let w_q = Tensor::new(w_q_data, vec![4, 2, 1, 1]);
    let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
    let y_scale = expected_max / 127.0;
    let y_zp: i8 = 0;
    let result = qlinear_conv2d(
        &x_q,
        x_scale,
        x_zp,
        &w_q,
        &w_scale,
        &w_zp,
        y_scale,
        y_zp,
        None,
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    )
    .expect("qlinear_conv2d 1x1");
    assert_eq!(result.shape, vec![1, 4, 3, 3]);
    let deq_out: Vec<f32> = result
        .data
        .iter()
        .map(|&v| (v - y_zp as f32) * y_scale)
        .collect();
    let ref_tensor = Tensor::new(ref_out, vec![1, 4, 3, 3]);
    let deq_tensor = Tensor::new(deq_out, vec![1, 4, 3, 3]);
    let rel_err = relative_error(&deq_tensor, &ref_tensor);
    assert!(
        rel_err < 0.15,
        "QLinearConv 1x1 relative error {} too large",
        rel_err,
    );
}

#[test]
fn test_qlinear_conv2d_3x3_kernel() {
    let x_f32 = Tensor::new((0..16).map(|i| i as f32 * 0.5).collect(), vec![1, 1, 4, 4]);
    let w_f32 = Tensor::new(
        vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0],
        vec![1, 1, 3, 3],
    );
    let bias = Tensor::new(vec![0.5], vec![1]);
    let ref_out = reference_conv2d(
        &x_f32.data,
        1,
        1,
        4,
        4,
        &w_f32.data,
        1,
        3,
        3,
        Some(&bias.data),
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    );
    let x_max = 7.5f32;
    let x_scale = x_max / 127.0;
    let x_zp: i8 = 0;
    let x_q_data: Vec<f32> = x_f32
        .data
        .iter()
        .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
        .collect();
    let x_q = Tensor::new(x_q_data, vec![1, 1, 4, 4]);
    let w_max = 2.0f32;
    let w_scale = vec![w_max / 127.0];
    let w_zp = vec![0i8];
    let w_q_data: Vec<f32> = w_f32
        .data
        .iter()
        .map(|&v| (v / w_scale[0]).round().clamp(-128.0, 127.0))
        .collect();
    let w_q = Tensor::new(w_q_data, vec![1, 1, 3, 3]);
    let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
    let y_scale = if expected_max < 1e-10 {
        1e-10
    } else {
        expected_max / 127.0
    };
    let y_zp: i8 = 0;
    let result = qlinear_conv2d(
        &x_q,
        x_scale,
        x_zp,
        &w_q,
        &w_scale,
        &w_zp,
        y_scale,
        y_zp,
        Some(&bias),
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    )
    .expect("qlinear_conv2d 3x3");
    assert_eq!(result.shape, vec![1, 1, 2, 2]);
    let deq_out: Vec<f32> = result
        .data
        .iter()
        .map(|&v| (v - y_zp as f32) * y_scale)
        .collect();
    let ref_tensor = Tensor::new(ref_out, vec![1, 1, 2, 2]);
    let deq_tensor = Tensor::new(deq_out, vec![1, 1, 2, 2]);
    let rel_err = relative_error(&deq_tensor, &ref_tensor);
    assert!(
        rel_err < 0.2,
        "QLinearConv 3x3 relative error {} too large",
        rel_err,
    );
}

#[test]
fn test_qlinear_conv2d_grouped() {
    let x_f32 = Tensor::new(
        (0..36).map(|i| (i as f32 - 18.0) * 0.1).collect(),
        vec![1, 4, 3, 3],
    );
    let w_f32 = Tensor::new(
        vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.6, 0.2, -0.3],
        vec![4, 2, 1, 1],
    );
    let ref_out = reference_conv2d(
        &x_f32.data,
        1,
        4,
        3,
        3,
        &w_f32.data,
        4,
        1,
        1,
        None,
        &[1, 1],
        &[0, 0, 0, 0],
        2,
    );
    let x_max = x_f32
        .data
        .iter()
        .copied()
        .fold(0.0f32, |a, v| a.max(v.abs()));
    let x_scale = x_max / 127.0;
    let x_zp: i8 = 0;
    let x_q_data: Vec<f32> = x_f32
        .data
        .iter()
        .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
        .collect();
    let x_q = Tensor::new(x_q_data, vec![1, 4, 3, 3]);
    let w_max = 0.6f32;
    let w_scale = vec![w_max / 127.0];
    let w_zp = vec![0i8];
    let w_q_data: Vec<f32> = w_f32
        .data
        .iter()
        .map(|&v| (v / w_scale[0]).round().clamp(-128.0, 127.0))
        .collect();
    let w_q = Tensor::new(w_q_data, vec![4, 2, 1, 1]);
    let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
    let y_scale = if expected_max < 1e-10 {
        1e-10
    } else {
        expected_max / 127.0
    };
    let y_zp: i8 = 0;
    let result = qlinear_conv2d(
        &x_q,
        x_scale,
        x_zp,
        &w_q,
        &w_scale,
        &w_zp,
        y_scale,
        y_zp,
        None,
        &[1, 1],
        &[0, 0, 0, 0],
        2,
    )
    .expect("qlinear_conv2d grouped");
    assert_eq!(result.shape, vec![1, 4, 3, 3]);
    let deq_out: Vec<f32> = result
        .data
        .iter()
        .map(|&v| (v - y_zp as f32) * y_scale)
        .collect();
    let ref_tensor = Tensor::new(ref_out.clone(), vec![1, 4, 3, 3]);
    let deq_tensor = Tensor::new(deq_out, vec![1, 4, 3, 3]);
    let rel_err = relative_error(&deq_tensor, &ref_tensor);
    assert!(
        rel_err < 0.2,
        "QLinearConv grouped relative error {} too large",
        rel_err,
    );
}

#[test]
fn test_qlinear_conv2d_per_channel_scales() {
    let x_f32 = Tensor::new((0..9).map(|i| i as f32).collect(), vec![1, 1, 3, 3]);
    let w_f32 = Tensor::new(vec![0.5, -0.8], vec![2, 1, 1, 1]);
    let ref_out = reference_conv2d(
        &x_f32.data,
        1,
        1,
        3,
        3,
        &w_f32.data,
        2,
        1,
        1,
        None,
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    );
    let x_max = 8.0f32;
    let x_scale = x_max / 127.0;
    let x_zp: i8 = 0;
    let x_q_data: Vec<f32> = x_f32
        .data
        .iter()
        .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
        .collect();
    let x_q = Tensor::new(x_q_data, vec![1, 1, 3, 3]);
    let w_scale = vec![0.5 / 127.0, 0.8 / 127.0];
    let w_zp = vec![0i8, 0i8];
    let w_q_data: Vec<f32> = vec![
        (0.5f32 / w_scale[0]).round().clamp(-128.0, 127.0),
        (-0.8f32 / w_scale[1]).round().clamp(-128.0, 127.0),
    ];
    let w_q = Tensor::new(w_q_data, vec![2, 1, 1, 1]);
    let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
    let y_scale = if expected_max < 1e-10 {
        1e-10
    } else {
        expected_max / 127.0
    };
    let y_zp: i8 = 0;
    let result = qlinear_conv2d(
        &x_q,
        x_scale,
        x_zp,
        &w_q,
        &w_scale,
        &w_zp,
        y_scale,
        y_zp,
        None,
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    )
    .expect("qlinear_conv2d per-channel");
    assert_eq!(result.shape, vec![1, 2, 3, 3]);
    let deq_out: Vec<f32> = result
        .data
        .iter()
        .map(|&v| (v - y_zp as f32) * y_scale)
        .collect();
    let ref_tensor = Tensor::new(ref_out, vec![1, 2, 3, 3]);
    let deq_tensor = Tensor::new(deq_out, vec![1, 2, 3, 3]);
    let rel_err = relative_error(&deq_tensor, &ref_tensor);
    assert!(
        rel_err < 0.15,
        "QLinearConv per-channel relative error {} too large",
        rel_err,
    );
}

#[test]
fn test_qlinear_conv2d_with_nonzero_zp() {
    let x_f32 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let w_f32 = Tensor::new(vec![0.5], vec![1, 1, 1, 1]);
    let bias = Tensor::new(vec![0.1], vec![1]);
    let ref_out = reference_conv2d(
        &x_f32.data,
        1,
        1,
        2,
        2,
        &w_f32.data,
        1,
        1,
        1,
        Some(&bias.data),
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    );
    let x_scale = 4.0 / 255.0;
    let x_zp_f = (-1.0f32 / x_scale).round().clamp(-128.0, 127.0);
    let x_zp = x_zp_f as i8;
    let x_q_data: Vec<f32> = x_f32
        .data
        .iter()
        .map(|&v| (v / x_scale + x_zp_f).round().clamp(-128.0, 127.0))
        .collect();
    let x_q = Tensor::new(x_q_data, vec![1, 1, 2, 2]);
    let w_scale = vec![0.5 / 127.0];
    let w_zp = vec![3i8];
    let w_q_data: Vec<f32> = vec![(0.5 / w_scale[0] + w_zp[0] as f32)
        .round()
        .clamp(-128.0, 127.0)];
    let w_q = Tensor::new(w_q_data, vec![1, 1, 1, 1]);
    let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
    let y_scale = if expected_max < 1e-10 {
        1e-10
    } else {
        expected_max / 127.0
    };
    let y_zp: i8 = 5;
    let result = qlinear_conv2d(
        &x_q,
        x_scale,
        x_zp,
        &w_q,
        &w_scale,
        &w_zp,
        y_scale,
        y_zp,
        Some(&bias),
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    )
    .expect("qlinear_conv2d with nonzero zp");
    assert_eq!(result.shape, vec![1, 1, 2, 2]);
    let deq_out: Vec<f32> = result
        .data
        .iter()
        .map(|&v| (v - y_zp as f32) * y_scale)
        .collect();
    let ref_tensor = Tensor::new(ref_out, vec![1, 1, 2, 2]);
    let deq_tensor = Tensor::new(deq_out, vec![1, 1, 2, 2]);
    let rel_err = relative_error(&deq_tensor, &ref_tensor);
    assert!(
        rel_err < 0.25,
        "QLinearConv nonzero-zp relative error {} too large",
        rel_err,
    );
}

#[test]
fn test_qlinear_conv2d_shape_validation() {
    let x = Tensor::new(vec![1.0; 6], vec![2, 3]);
    let w = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
    let result = qlinear_conv2d(
        &x,
        1.0,
        0,
        &w,
        &[1.0],
        &[0],
        1.0,
        0,
        None,
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    );
    assert!(result.is_err());
    let x2 = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);
    let w2 = Tensor::new(vec![1.0; 3], vec![3]);
    let result2 = qlinear_conv2d(
        &x2,
        1.0,
        0,
        &w2,
        &[1.0],
        &[0],
        1.0,
        0,
        None,
        &[1, 1],
        &[0, 0, 0, 0],
        1,
    );
    assert!(result2.is_err());
}

#[test]
fn test_qlinear_conv2d_with_padding() {
    let x_f32 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let w_f32 = Tensor::new(
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![1, 1, 3, 3],
    );
    let ref_out = reference_conv2d(
        &x_f32.data,
        1,
        1,
        2,
        2,
        &w_f32.data,
        1,
        3,
        3,
        None,
        &[1, 1],
        &[1, 1, 1, 1],
        1,
    );
    let x_scale = 4.0 / 127.0;
    let x_zp: i8 = 0;
    let x_q_data: Vec<f32> = x_f32
        .data
        .iter()
        .map(|&v| (v / x_scale).round().clamp(-128.0, 127.0))
        .collect();
    let x_q = Tensor::new(x_q_data, vec![1, 1, 2, 2]);
    let w_scale = vec![1.0 / 127.0];
    let w_zp = vec![0i8];
    let w_q_data: Vec<f32> = w_f32
        .data
        .iter()
        .map(|&v| (v / w_scale[0]).round().clamp(-128.0, 127.0))
        .collect();
    let w_q = Tensor::new(w_q_data, vec![1, 1, 3, 3]);
    let expected_max = ref_out.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
    let y_scale = if expected_max < 1e-10 {
        1e-10
    } else {
        expected_max / 127.0
    };
    let y_zp: i8 = 0;
    let result = qlinear_conv2d(
        &x_q,
        x_scale,
        x_zp,
        &w_q,
        &w_scale,
        &w_zp,
        y_scale,
        y_zp,
        None,
        &[1, 1],
        &[1, 1, 1, 1],
        1,
    )
    .expect("qlinear_conv2d with padding");
    assert_eq!(result.shape, vec![1, 1, 2, 2]);
    let deq_out: Vec<f32> = result
        .data
        .iter()
        .map(|&v| (v - y_zp as f32) * y_scale)
        .collect();
    let ref_tensor = Tensor::new(ref_out, vec![1, 1, 2, 2]);
    let deq_tensor = Tensor::new(deq_out, vec![1, 1, 2, 2]);
    let rel_err = relative_error(&deq_tensor, &ref_tensor);
    assert!(
        rel_err < 0.15,
        "QLinearConv padded relative error {} too large",
        rel_err,
    );
}
