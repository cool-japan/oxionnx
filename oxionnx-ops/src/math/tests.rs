use super::*;
use oxionnx_core::{OnnxError, Tensor};

#[test]
fn test_add_same_shape() -> Result<(), OnnxError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let c = add(&a, &b)?;
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn test_add_broadcast_scalar() -> Result<(), OnnxError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![10.0], vec![1]);
    let c = add(&a, &b)?;
    assert_eq!(c.data, vec![11.0, 12.0, 13.0]);
    Ok(())
}

#[test]
fn test_matmul_2x3_3x4() -> Result<(), OnnxError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![3, 4],
    );
    let c = matmul(&a, &b)?;
    assert_eq!(c.shape, vec![2, 4]);
    assert!((c.data[0] - 1.0).abs() < 1e-5);
    assert!((c.data[1] - 2.0).abs() < 1e-5);
    assert!((c.data[4] - 4.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_reduce_mean() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let m = reduce_mean(&t, &[1], false)?;
    assert_eq!(m.shape, vec![2]);
    assert!((m.data[0] - 2.0).abs() < 1e-5);
    assert!((m.data[1] - 5.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_reduce_sum() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let s = reduce_sum(&t, &[1], false)?;
    assert_eq!(s.shape, vec![2]);
    assert!((s.data[0] - 6.0).abs() < 1e-5);
    assert!((s.data[1] - 15.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_reduce_max_keepdims() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
    let m = reduce_max(&t, &[1], true)?;
    assert_eq!(m.shape, vec![2, 1]);
    assert_eq!(m.data, vec![5.0, 3.0]);
    Ok(())
}

#[test]
fn test_reduce_min() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
    let m = reduce_min(&t, &[1], false)?;
    assert_eq!(m.data, vec![1.0, 2.0]);
    Ok(())
}

#[test]
fn test_reduce_prod() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let p = reduce_prod(&t, &[1], false)?;
    assert_eq!(p.data, vec![2.0, 12.0]);
    Ok(())
}

#[test]
fn test_arg_max() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0, 4.0, 0.0], vec![2, 3]);
    let idx = arg_max(&t, 1, false, false)?;
    assert_eq!(idx.shape, vec![2]);
    assert_eq!(idx.data[0], 1.0); // max of [1,5,3] is at index 1
    assert_eq!(idx.data[1], 1.0); // max of [2,4,0] is at index 1
    Ok(())
}

#[test]
fn test_arg_min() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0, 4.0, 0.0], vec![2, 3]);
    let idx = arg_min(&t, 1, false, false)?;
    assert_eq!(idx.data[0], 0.0); // min of [1,5,3] is at index 0
    assert_eq!(idx.data[1], 2.0); // min of [2,4,0] is at index 2
    Ok(())
}

#[test]
fn test_cumsum_inclusive() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let c = cumsum(&t, 0, false, false)?;
    assert_eq!(c.data, vec![1.0, 3.0, 6.0, 10.0]);
    Ok(())
}

#[test]
fn test_cumsum_exclusive() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let c = cumsum(&t, 0, true, false)?;
    assert_eq!(c.data, vec![0.0, 1.0, 3.0, 6.0]);
    Ok(())
}

#[test]
fn test_range() -> Result<(), OnnxError> {
    let r = range(0.0, 5.0, 1.0)?;
    assert_eq!(r.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(r.shape, vec![5]);
    Ok(())
}

#[test]
fn test_range_negative_delta() -> Result<(), OnnxError> {
    let r = range(5.0, 0.0, -1.0)?;
    assert_eq!(r.data, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    Ok(())
}

#[test]
fn test_top_k_largest() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], vec![8]);
    let (vals, idxs) = top_k(&t, 3, 0, true, true)?;
    assert_eq!(vals.shape, vec![3]);
    assert!((vals.data[0] - 9.0).abs() < 1e-5);
    assert!((vals.data[1] - 6.0).abs() < 1e-5);
    assert!((vals.data[2] - 5.0).abs() < 1e-5);
    assert_eq!(idxs.data[0], 5.0); // 9 is at index 5
    Ok(())
}

#[test]
fn test_top_k_smallest() -> Result<(), OnnxError> {
    let t = Tensor::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);
    let (vals, _) = top_k(&t, 2, 0, false, true)?;
    assert_eq!(vals.shape, vec![2]);
    assert!((vals.data[0] - 1.0).abs() < 1e-5);
    assert!((vals.data[1] - 1.0).abs() < 1e-5);
    Ok(())
}

// ── Unary math ops tests ────────────────────────────────────────────────

#[test]
fn test_ceil() {
    let t = Tensor::new(vec![1.5, -1.5, 0.0, 2.3], vec![4]);
    let r = ceil(&t);
    assert_eq!(r.data, vec![2.0, -1.0, 0.0, 3.0]);
}

#[test]
fn test_floor_op() {
    let t = Tensor::new(vec![1.5, -1.5, 0.0, 2.9], vec![4]);
    let r = floor_op(&t);
    assert_eq!(r.data, vec![1.0, -2.0, 0.0, 2.0]);
}

#[test]
fn test_round_op() {
    let t = Tensor::new(vec![1.5, 2.5, 0.4, -0.6], vec![4]);
    let r = round_op(&t);
    // Rust uses banker's rounding: 1.5 -> 2.0, 2.5 -> 2.0
    assert_eq!(r.data, vec![2.0, 2.0, 0.0, -1.0]);
}

#[test]
fn test_sign() {
    let t = Tensor::new(vec![-3.0, 0.0, 5.0, -0.5], vec![4]);
    let r = sign(&t);
    assert_eq!(r.data, vec![-1.0, 0.0, 1.0, -1.0]);
}

#[test]
fn test_sin_cos_tan() {
    let t = Tensor::new(vec![0.0], vec![1]);
    let s = sin_op(&t);
    let c = cos_op(&t);
    let ta = tan_op(&t);
    assert!((s.data[0] - 0.0).abs() < 1e-5);
    assert!((c.data[0] - 1.0).abs() < 1e-5);
    assert!((ta.data[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_asin_acos_atan() {
    let t = Tensor::new(vec![0.0, 0.5], vec![2]);
    let as_r = asin_op(&t);
    let ac_r = acos_op(&t);
    let at_r = atan_op(&t);
    assert!((as_r.data[0] - 0.0).abs() < 1e-5);
    assert!((ac_r.data[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    assert!((at_r.data[0] - 0.0).abs() < 1e-5);
    // asin(0.5) = pi/6
    assert!((as_r.data[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-5);
}

#[test]
fn test_sinh_cosh() {
    let t = Tensor::new(vec![0.0, 1.0], vec![2]);
    let sh = sinh_op(&t);
    let ch = cosh_op(&t);
    assert!((sh.data[0] - 0.0).abs() < 1e-5);
    assert!((ch.data[0] - 1.0).abs() < 1e-5);
    assert!((sh.data[1] - 1.0_f32.sinh()).abs() < 1e-5);
    assert!((ch.data[1] - 1.0_f32.cosh()).abs() < 1e-5);
}

#[test]
fn test_asinh_acosh_atanh() {
    let t_asinh = Tensor::new(vec![0.0, 1.0], vec![2]);
    let r = asinh_op(&t_asinh);
    assert!((r.data[0] - 0.0).abs() < 1e-4);
    assert!((r.data[1] - 1.0_f32.asinh()).abs() < 1e-4);

    let t_acosh = Tensor::new(vec![1.0, 2.0], vec![2]);
    let r2 = acosh_op(&t_acosh);
    assert!((r2.data[0] - 0.0).abs() < 1e-4);
    assert!((r2.data[1] - 2.0_f32.acosh()).abs() < 1e-4);

    let t_atanh = Tensor::new(vec![0.0, 0.5], vec![2]);
    let r3 = atanh_op(&t_atanh);
    assert!((r3.data[0] - 0.0).abs() < 1e-5);
    assert!((r3.data[1] - 0.5_f32.atanh()).abs() < 1e-5);
}

// ── Binary math ops tests ───────────────────────────────────────────────

#[test]
fn test_mod_op_fmod() {
    let a = Tensor::new(vec![7.0, -7.0], vec![2]);
    let b = Tensor::new(vec![3.0], vec![1]);
    let r = mod_op(&a, &b, 1).expect("mod_op fmod failed");
    assert!((r.data[0] - 1.0).abs() < 1e-5); // 7 % 3 = 1
    assert!((r.data[1] - (-1.0)).abs() < 1e-5); // -7 % 3 = -1
}

// [a0-13] fmod=0 is ONNX's default and specifies numpy.mod semantics (result
// takes the sign of the divisor), NOT C's truncated modulo (sign of the
// dividend) -- this test previously asserted the latter, which was the bug.
// Renamed from `test_mod_op_truncated` to `test_mod_op_floored` to match
// what fmod=0 actually computes.
#[test]
fn test_mod_op_floored() {
    let a = Tensor::new(vec![7.0, -7.0], vec![2]);
    let b = Tensor::new(vec![3.0], vec![1]);
    let r = mod_op(&a, &b, 0).expect("mod_op floored failed");
    assert!((r.data[0] - 1.0).abs() < 1e-5); // 7 mod 3 = 1
    assert!((r.data[1] - 2.0).abs() < 1e-5); // -7 mod 3 = 2 (sign follows divisor, not dividend)
}

#[test]
fn test_bit_shift_left() {
    let x = Tensor::new(vec![1.0, 2.0, 4.0], vec![3]);
    let y = Tensor::new(vec![2.0], vec![1]);
    let r = bit_shift(&x, &y, "LEFT").expect("bit_shift left failed");
    assert_eq!(r.data, vec![4.0, 8.0, 16.0]);
}

#[test]
fn test_bit_shift_right() {
    let x = Tensor::new(vec![16.0, 8.0, 4.0], vec![3]);
    let y = Tensor::new(vec![2.0], vec![1]);
    let r = bit_shift(&x, &y, "RIGHT").expect("bit_shift right failed");
    assert_eq!(r.data, vec![4.0, 2.0, 1.0]);
}

// ── Variadic ops tests ──────────────────────────────────────────────────

#[test]
fn test_variadic_min() {
    let a = Tensor::new(vec![5.0, 2.0, 8.0], vec![3]);
    let b = Tensor::new(vec![3.0, 6.0, 1.0], vec![3]);
    let c = Tensor::new(vec![4.0, 1.0, 9.0], vec![3]);
    let r = variadic_min(&[&a, &b, &c]).expect("variadic_min failed");
    assert_eq!(r.data, vec![3.0, 1.0, 1.0]);
}

#[test]
fn test_variadic_max() {
    let a = Tensor::new(vec![5.0, 2.0, 8.0], vec![3]);
    let b = Tensor::new(vec![3.0, 6.0, 1.0], vec![3]);
    let c = Tensor::new(vec![4.0, 1.0, 9.0], vec![3]);
    let r = variadic_max(&[&a, &b, &c]).expect("variadic_max failed");
    assert_eq!(r.data, vec![5.0, 6.0, 9.0]);
}

#[test]
fn test_variadic_sum() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let c = Tensor::new(vec![7.0, 8.0, 9.0], vec![3]);
    let r = variadic_sum(&[&a, &b, &c]).expect("variadic_sum failed");
    assert_eq!(r.data, vec![12.0, 15.0, 18.0]);
}

#[test]
fn test_variadic_mean() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let c = Tensor::new(vec![7.0, 8.0, 9.0], vec![3]);
    let r = variadic_mean(&[&a, &b, &c]).expect("variadic_mean failed");
    assert!((r.data[0] - 4.0).abs() < 1e-5);
    assert!((r.data[1] - 5.0).abs() < 1e-5);
    assert!((r.data[2] - 6.0).abs() < 1e-5);
}

#[test]
fn test_variadic_empty() {
    assert!(variadic_min(&[]).is_err());
    assert!(variadic_max(&[]).is_err());
    assert!(variadic_sum(&[]).is_err());
    assert!(variadic_mean(&[]).is_err());
}

// ── J-phase reduce ops tests ────────────────────────────────────────────

#[test]
fn test_reduce_l1_basic() {
    let x = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
    let out = reduce_l1(&x, &[1], false).unwrap();
    // row0: |-1|+|2|=3, row1: |-3|+|4|=7
    assert_eq!(out.shape, vec![2]);
    assert!((out.data[0] - 3.0).abs() < 1e-5);
    assert!((out.data[1] - 7.0).abs() < 1e-5);
}

#[test]
fn test_reduce_l2_basic() {
    let x = Tensor::new(vec![3.0, 4.0], vec![2]);
    let out = reduce_l2(&x, &[], false).unwrap();
    assert!((out.data[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_reduce_log_sum_basic() {
    let x = Tensor::new(vec![1.0, 1.0], vec![2]);
    let out = reduce_log_sum(&x, &[], false).unwrap();
    assert!((out.data[0] - (2.0f32).ln()).abs() < 1e-5);
}

#[test]
fn test_reduce_sum_square_basic() {
    let x = Tensor::new(vec![2.0, 3.0], vec![2]);
    let out = reduce_sum_square(&x, &[], false).unwrap();
    assert!((out.data[0] - 13.0).abs() < 1e-5);
}

#[test]
fn test_reduce_log_sum_exp_stability() {
    // Naive exp(1000) overflows; stable impl must stay finite.
    // x = [1000, 1001, 1002], max = 1002
    // shifted = [-2, -1, 0]
    // result = 1002 + log(exp(-2) + exp(-1) + exp(0))
    let x = Tensor::new(vec![1000.0, 1001.0, 1002.0], vec![3]);
    let out = reduce_log_sum_exp(&x, &[], false).unwrap();
    let expected = 1002.0f32 + ((-2.0f32).exp() + (-1.0f32).exp() + 1.0f32).ln();
    assert!(
        (out.data[0] - expected).abs() < 1e-3,
        "got {}, expected {}",
        out.data[0],
        expected
    );
    // Also verify it is finite (the key stability property)
    assert!(out.data[0].is_finite(), "result must be finite");
}

// ── Batched MatMul parallel tests ───────────────────────────────────────

#[test]
fn test_batched_matmul_parallel() {
    // batch=8, each [2,3] @ [3,2] = [2,2]
    let batch = 8;
    let m = 2;
    let k = 3;
    let n = 2;
    let a_data: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let a = Tensor::new(a_data, vec![batch, m, k]);
    let b = Tensor::new(b_data, vec![batch, k, n]);
    let out = matmul(&a, &b).expect("matmul failed");
    assert_eq!(out.shape, vec![batch, m, n]);
    // Verify first batch manually: a[0] = [[0,0.1,0.2],[0.3,0.4,0.5]]
    // b[0] = [[0.5,0.6],[0.7,0.8],[0.9,1.0]]
    // c[0,0,0] = 0*0.5 + 0.1*0.7 + 0.2*0.9 = 0 + 0.07 + 0.18 = 0.25
    assert!(
        (out.data[0] - 0.25).abs() < 1e-4,
        "matmul batch 0 [0,0]={}",
        out.data[0]
    );
}

#[test]
fn test_batched_matmul_single_batch() {
    // batch=1 uses sequential path
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2]);
    let out = matmul(&a, &b).expect("matmul failed");
    assert_eq!(out.shape, vec![1, 2, 2]);
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert!((out.data[0] - 19.0).abs() < 1e-4);
    assert!((out.data[1] - 22.0).abs() < 1e-4);
    assert!((out.data[2] - 43.0).abs() < 1e-4);
    assert!((out.data[3] - 50.0).abs() < 1e-4);
}

#[test]
fn test_batched_matmul_large_batch() {
    // batch=32, [4,4] @ [4,4] identity check
    let batch = 32;
    let sz = 4;
    // Identity matrix tiled
    let mut eye = vec![0.0f32; sz * sz];
    for i in 0..sz {
        eye[i * sz + i] = 1.0;
    }
    let b_data: Vec<f32> = (0..batch).flat_map(|_| eye.iter().copied()).collect();
    let a_data: Vec<f32> = (0..batch * sz * sz).map(|i| (i as f32) * 0.01).collect();
    let a = Tensor::new(a_data.clone(), vec![batch, sz, sz]);
    let b = Tensor::new(b_data, vec![batch, sz, sz]);
    let out = matmul(&a, &b).expect("matmul failed");
    assert_eq!(out.shape, vec![batch, sz, sz]);
    // A @ I = A
    for (i, (&got, &expected)) in out.data.iter().zip(a_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "matmul identity [{i}]: got {got}, expected {expected}"
        );
    }
}
