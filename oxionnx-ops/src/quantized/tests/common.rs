//! Shared test helpers for quantized module tests.

use oxionnx_core::Tensor;

/// Helper: compute f32 matmul for reference.
pub fn f32_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a.data[i * k + p] * b.data[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Tensor::new(out, vec![m, n])
}

/// Helper: max absolute error between two tensors.
pub fn max_abs_error(a: &Tensor, b: &Tensor) -> f32 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Helper: relative error (Frobenius norm of difference / Frobenius norm of reference).
pub fn relative_error(result: &Tensor, reference: &Tensor) -> f32 {
    let diff_norm: f32 = result
        .data
        .iter()
        .zip(reference.data.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let ref_norm: f32 = reference.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if ref_norm < 1e-10 {
        diff_norm
    } else {
        diff_norm / ref_norm
    }
}

/// Helper: simple f32 conv2d for reference (no dilation).
#[allow(clippy::too_many_arguments)]
pub fn reference_conv2d(
    input: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    c_out: usize,
    kh: usize,
    kw: usize,
    bias: Option<&[f32]>,
    strides: &[usize],
    pads: &[usize],
    group: usize,
) -> Vec<f32> {
    let c_per_group = c_in / group;
    let c_out_per_group = c_out / group;
    let h_out = (h + pads[0] + pads[2] - kh) / strides[0] + 1;
    let w_out = (w + pads[1] + pads[3] - kw) / strides[1] + 1;
    let mut out = vec![0.0f32; n * c_out * h_out * w_out];
    for batch in 0..n {
        for g in 0..group {
            for oc in 0..c_out_per_group {
                let global_oc = g * c_out_per_group + oc;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for ic in 0..c_per_group {
                            let in_c = g * c_per_group + ic;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (oh * strides[0] + ky) as isize - pads[0] as isize;
                                    let ix = (ow * strides[1] + kx) as isize - pads[1] as isize;
                                    if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                        let x_val = input[(batch * c_in + in_c) * h * w
                                            + iy as usize * w
                                            + ix as usize];
                                        let w_val = weight
                                            [((global_oc * c_per_group + ic) * kh + ky) * kw + kx];
                                        sum += x_val * w_val;
                                    }
                                }
                            }
                        }
                        if let Some(b) = bias {
                            sum += b[global_oc];
                        }
                        out[(batch * c_out + global_oc) * h_out * w_out + oh * w_out + ow] = sum;
                    }
                }
            }
        }
    }
    out
}
