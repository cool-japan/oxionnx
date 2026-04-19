//! Unit tests for nn module operations.

use super::*;
use oxionnx_core::{OnnxError, Tensor};

#[test]
fn test_softmax_last_dim() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let y = softmax(&x, -1)?;
    let sum: f32 = y.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(y.data[2] > y.data[1] && y.data[1] > y.data[0]);
    Ok(())
}

#[test]
fn test_layer_norm() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let scale = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
    let bias = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]);
    let y = layer_norm(&x, &scale, Some(&bias), 1e-5, -1)?;
    // Each row should have mean≈0
    let mean0: f32 = y.data[..3].iter().sum::<f32>() / 3.0;
    assert!(mean0.abs() < 1e-5, "mean={mean0}");
    Ok(())
}

#[test]
fn test_gelu() {
    let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
    let y = gelu(&x);
    assert!((y.data[0]).abs() < 1e-6); // gelu(0) = 0
    assert!(y.data[1] > 0.0); // gelu(1) > 0
    assert!(y.data[2] < 0.0); // gelu(-1) < 0
}

#[test]
fn test_leaky_relu() {
    let x = Tensor::new(vec![2.0, -3.0, 0.0, -1.0], vec![4]);
    let y = leaky_relu(&x, 0.01);
    assert_eq!(y.data[0], 2.0);
    assert!((y.data[1] - (-0.03)).abs() < 1e-6);
    assert_eq!(y.data[2], 0.0);
    assert!((y.data[3] - (-0.01)).abs() < 1e-6);
}

#[test]
fn test_silu() {
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
    let y = silu(&x);
    assert!((y.data[0]).abs() < 1e-6);
    assert!(y.data[1] > 0.0 && y.data[1] < 1.0);
    assert!(y.data[2] > -0.5 && y.data[2] < 0.0);
}

#[test]
fn test_hard_sigmoid() {
    // clamp(alpha*x + beta, 0, 1) with alpha=0.2, beta=0.5
    let x = Tensor::new(vec![-10.0, 0.0, 10.0, 1.0], vec![4]);
    let y = hard_sigmoid(&x, 0.2, 0.5);
    assert_eq!(y.data[0], 0.0);
    assert!((y.data[1] - 0.5).abs() < 1e-6);
    assert_eq!(y.data[2], 1.0);
}

#[test]
fn test_hard_swish() {
    // hard_swish(0) = 0 * 0.5 = 0
    let x = Tensor::new(vec![0.0, 3.0, -3.0, 6.0], vec![4]);
    let y = hard_swish(&x);
    assert!((y.data[0]).abs() < 1e-6);
    assert!((y.data[1] - 3.0 * (3.0 / 6.0 + 0.5)).abs() < 1e-5);
    assert_eq!(y.data[2], 0.0); // -3: clamp(-3/6+0.5, 0, 1) = 0
    assert_eq!(y.data[3], 6.0); // 6: 6 * clamp(1.5, 0, 1) = 6
}

#[test]
fn test_rms_norm() -> Result<(), OnnxError> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let scale = Tensor::new(vec![1.0; 4], vec![4]);
    let y = rms_norm(&x, &scale, 1e-6)?;
    // Each row should have RMS ≈ 1
    let sq_mean: f32 = y.data.iter().map(|&v| v * v).sum::<f32>() / 4.0;
    assert!((sq_mean - 1.0).abs() < 1e-4, "sq_mean={sq_mean}");
    Ok(())
}

#[test]
fn test_prelu_per_channel() {
    // [1, 2, 2, 2] input, 2 channels
    #[rustfmt::skip]
    let x = Tensor::new(vec![
        1.0, -2.0, 3.0, -4.0,  // channel 0
        -1.0, 2.0, -3.0, 4.0,  // channel 1
    ], vec![1, 2, 2, 2]);
    let slope = Tensor::new(vec![0.1, 0.2], vec![2]);
    let y = prelu(&x, &slope);
    assert_eq!(y.data[0], 1.0);
    assert!((y.data[1] - (-0.2)).abs() < 1e-6); // -2 * 0.1
    assert_eq!(y.data[2], 3.0);
    assert!((y.data[3] - (-0.4)).abs() < 1e-6); // -4 * 0.1
    assert!((y.data[4] - (-0.2)).abs() < 1e-6); // -1 * 0.2
    assert_eq!(y.data[5], 2.0);
    assert!((y.data[6] - (-0.6)).abs() < 1e-6); // -3 * 0.2
    assert_eq!(y.data[7], 4.0);
}

#[test]
fn test_log_softmax() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let out = log_softmax(&t, -1).expect("log_softmax failed");
    // exp(log_softmax) should sum to ~1.0
    let sum: f32 = out.data.iter().map(|v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // All values should be negative (log of probability)
    assert!(out.data.iter().all(|v| *v <= 0.0));
}

#[test]
fn test_log_softmax_2d() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let out = log_softmax(&t, 1).expect("log_softmax failed");
    // Each row should sum to ~1.0 after exp
    let sum0: f32 = out.data[0..3].iter().map(|v| v.exp()).sum();
    let sum1: f32 = out.data[3..6].iter().map(|v| v.exp()).sum();
    assert!((sum0 - 1.0).abs() < 1e-5);
    assert!((sum1 - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_invalid_axis() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(log_softmax(&t, 5).is_err());
}

#[allow(clippy::approx_constant)]
#[test]
fn test_softplus() {
    let t = Tensor::new(vec![0.0, 1.0, -1.0, 20.0, -20.0], vec![5]);
    let out = softplus(&t);
    assert!((out.data[0] - 0.6931).abs() < 1e-3); // ln(2)
    assert!(out.data[3] > 19.9); // large x ~ x
    assert!(out.data[4] < 0.01); // large negative ~ 0
}

#[test]
fn test_softsign() {
    let t = Tensor::new(vec![0.0, 1.0, -1.0, 100.0], vec![4]);
    let out = softsign(&t);
    assert!((out.data[0]).abs() < 1e-6);
    assert!((out.data[1] - 0.5).abs() < 1e-6);
    assert!((out.data[2] + 0.5).abs() < 1e-6);
    assert!((out.data[3] - 100.0 / 101.0).abs() < 1e-4);
}

#[test]
fn test_mish() {
    let t = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
    let out = mish(&t);
    assert!((out.data[0]).abs() < 1e-6); // mish(0) = 0
                                         // mish(1) = 1 * tanh(ln(1+e)) ~ 0.8651
    assert!((out.data[1] - 0.8651).abs() < 1e-3);
}

#[test]
fn test_elu() {
    let t = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
    let out = elu(&t, 1.0);
    assert!((out.data[0] - 1.0).abs() < 1e-6);
    assert!((out.data[1]).abs() < 1e-6);
    // alpha*(exp(-1)-1) ~ -0.6321
    assert!((out.data[2] - ((-1.0_f32).exp() - 1.0)).abs() < 1e-4);
}

#[test]
fn test_elu_custom_alpha() {
    let t = Tensor::new(vec![-1.0], vec![1]);
    let out = elu(&t, 2.0);
    assert!((out.data[0] - 2.0 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-4);
}

#[test]
fn test_celu() {
    let t = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
    let out = celu(&t, 1.0);
    assert!((out.data[0] - 1.0).abs() < 1e-6);
    assert!((out.data[1]).abs() < 1e-6);
    // celu(-1, alpha=1) = 1*(exp(-1/1)-1) = exp(-1)-1 ~ -0.6321
    assert!((out.data[2] - ((-1.0_f32).exp() - 1.0)).abs() < 1e-4);
}

#[test]
fn test_celu_custom_alpha() {
    let t = Tensor::new(vec![-2.0], vec![1]);
    let out = celu(&t, 0.5);
    let expected = 0.5 * ((-2.0_f32 / 0.5).exp() - 1.0);
    assert!((out.data[0] - expected).abs() < 1e-4);
}

#[test]
fn test_selu() {
    let t = Tensor::new(vec![1.0, 0.0, -1.0], vec![3]);
    let alpha = 1.673_263_2_f32;
    let gamma = 1.050_701_f32;
    let out = selu(&t, alpha, gamma);
    assert!((out.data[0] - gamma).abs() < 1e-4);
    // selu(0) = gamma * (alpha*exp(0) - alpha) = gamma * 0 = 0
    assert!((out.data[1]).abs() < 1e-5);
}

#[test]
fn test_thresholded_relu() {
    let t = Tensor::new(vec![-1.0, 0.0, 0.5, 1.0, 2.0], vec![5]);
    let out = thresholded_relu(&t, 1.0);
    assert_eq!(out.data, vec![0.0, 0.0, 0.0, 0.0, 2.0]);
}

#[test]
fn test_instance_norm() {
    // [1, 2, 2, 2] - 1 batch, 2 channels, 2x2 spatial
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = Tensor::new(data, vec![1, 2, 2, 2]);
    let scale = Tensor::new(vec![1.0, 1.0], vec![2]);
    let bias = Tensor::new(vec![0.0, 0.0], vec![2]);
    let out = instance_norm(&x, &scale, &bias, 1e-5).expect("instance_norm failed");
    // Each channel should have approximately zero mean
    let ch0_mean: f32 = out.data[0..4].iter().sum::<f32>() / 4.0;
    assert!(ch0_mean.abs() < 1e-4);
    let ch1_mean: f32 = out.data[4..8].iter().sum::<f32>() / 4.0;
    assert!(ch1_mean.abs() < 1e-4);
}

#[test]
fn test_instance_norm_with_scale_bias() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = Tensor::new(data, vec![1, 2, 2, 2]);
    let scale = Tensor::new(vec![2.0, 3.0], vec![2]);
    let bias = Tensor::new(vec![1.0, -1.0], vec![2]);
    let out = instance_norm(&x, &scale, &bias, 1e-5).expect("instance_norm failed");
    // Channel 0 mean should be bias[0] = 1.0
    let ch0_mean: f32 = out.data[0..4].iter().sum::<f32>() / 4.0;
    assert!((ch0_mean - 1.0).abs() < 1e-3);
}

#[test]
fn test_instance_norm_too_few_dims() {
    let x = Tensor::new(vec![1.0, 2.0], vec![2]);
    let scale = Tensor::new(vec![1.0], vec![1]);
    let bias = Tensor::new(vec![0.0], vec![1]);
    assert!(instance_norm(&x, &scale, &bias, 1e-5).is_err());
}

#[test]
fn test_lp_norm_l2() {
    let t = Tensor::new(vec![3.0, 4.0], vec![2]);
    let out = lp_norm(&t, 0, 2).expect("lp_norm failed");
    // L2 norm of [3,4] = 5, so normalized = [0.6, 0.8]
    assert!((out.data[0] - 0.6).abs() < 1e-5);
    assert!((out.data[1] - 0.8).abs() < 1e-5);
}

#[test]
fn test_lp_norm_l1() {
    let t = Tensor::new(vec![3.0, -4.0], vec![2]);
    let out = lp_norm(&t, 0, 1).expect("lp_norm failed");
    // L1 norm = 7, so [3/7, -4/7]
    assert!((out.data[0] - 3.0 / 7.0).abs() < 1e-5);
    assert!((out.data[1] - (-4.0 / 7.0)).abs() < 1e-5);
}

#[test]
fn test_lp_norm_invalid_axis() {
    let t = Tensor::new(vec![1.0, 2.0], vec![2]);
    assert!(lp_norm(&t, 5, 2).is_err());
}

#[test]
fn test_lp_norm_2d() {
    // [2, 3] tensor, normalize along axis=1
    let t = Tensor::new(vec![3.0, 4.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]);
    let out = lp_norm(&t, 1, 2).expect("lp_norm failed");
    // Row 0: norm = 5, [0.6, 0.8, 0.0]
    assert!((out.data[0] - 0.6).abs() < 1e-5);
    assert!((out.data[1] - 0.8).abs() < 1e-5);
    assert!((out.data[2]).abs() < 1e-5);
    // Row 1: norm = 1, [1.0, 0.0, 0.0]
    assert!((out.data[3] - 1.0).abs() < 1e-5);
}

#[test]
fn test_mean_variance_normalization() {
    // Simple 4D case: [1, 2, 1, 2], axes=[0, 2, 3]
    let data = vec![1.0, 3.0, 5.0, 7.0];
    let x = Tensor::new(data, vec![1, 2, 1, 2]);
    let out = mean_variance_normalization(&x, &[0, 2, 3]).expect("mean_var_norm failed");
    // Channel 0 slice: [1, 3], mean=2, var=1, normalized=[-1, 1]
    assert!((out.data[0] - (-1.0)).abs() < 0.1);
    assert!((out.data[1] - 1.0).abs() < 0.1);
    // Channel 1 slice: [5, 7], mean=6, var=1, normalized=[-1, 1]
    assert!((out.data[2] - (-1.0)).abs() < 0.1);
    assert!((out.data[3] - 1.0).abs() < 0.1);
}

#[test]
fn test_mean_variance_normalization_default_axes() {
    // 4D [2, 1, 1, 1], axes=[0,2,3]
    let data = vec![2.0, 4.0];
    let x = Tensor::new(data, vec![2, 1, 1, 1]);
    let out = mean_variance_normalization(&x, &[0, 2, 3]).expect("mean_var_norm failed");
    // mean=3, var=1, normalized: [-1, 1]
    assert!((out.data[0] - (-1.0)).abs() < 0.1);
    assert!((out.data[1] - 1.0).abs() < 0.1);
}

#[test]
fn test_mean_variance_normalization_invalid_axis() {
    let x = Tensor::new(vec![1.0], vec![1]);
    assert!(mean_variance_normalization(&x, &[5]).is_err());
}

#[test]
fn test_dropout_identity() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let out = dropout(&t);
    assert_eq!(out.data, t.data);
}

// ── J-phase nn ops tests ────────────────────────────────────────────────

#[test]
fn test_hardmax_basic() {
    let x = Tensor::new(vec![1.0, 3.0, 2.0], vec![3]);
    let out = hardmax(&x, 0).expect("hardmax failed");
    assert_eq!(out.data, vec![0.0, 1.0, 0.0]);
}

#[test]
fn test_hardmax_negative_axis() {
    let x = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
    let out = hardmax(&x, -1).expect("hardmax failed");
    // row0: [1,3] → max at idx 1 → [0,1]
    // row1: [2,4] → max at idx 1 → [0,1]
    assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_shrink_basic() {
    let x = Tensor::new(vec![-2.0, -0.3, 0.0, 0.3, 2.0], vec![5]);
    let out = shrink(&x, 0.0, 0.5);
    // -2 < -0.5 → -2+0=-2; -0.3 in [-0.5, 0.5] → 0; 0 → 0; 0.3 → 0; 2 > 0.5 → 2-0=2
    assert!((out.data[0] - (-2.0)).abs() < 1e-5);
    assert!((out.data[1] - 0.0).abs() < 1e-5);
    assert!((out.data[4] - 2.0).abs() < 1e-5);
}
