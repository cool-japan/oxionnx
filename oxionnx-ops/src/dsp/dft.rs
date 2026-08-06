//! DFT operator: Discrete Fourier Transform (forward / inverse), transformed
//! along an arbitrary axis of an N-D tensor.

use oxifft::Complex;
use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use super::helpers::{
    complex_to_interleaved, interleaved_to_complex, resolve_dft_length, scalar_i64,
};

/// Perform a 1-D FFT/IFFT on a slice of complex pairs (interleaved re/im).
///
/// `signal` must have length exactly `n * 2` (n complex samples interleaved).
/// Returns `n` complex samples as interleaved re/im.
pub(super) fn dft_1d(signal_interleaved: &[f32], n: usize, inverse: bool) -> Vec<f32> {
    let mut c: Vec<Complex<f32>> = interleaved_to_complex(signal_interleaved);
    // Pad or truncate to exactly n samples.
    c.resize(n, Complex::new(0.0, 0.0));

    let result = if inverse {
        oxifft::ifft::<f32>(&c)
    } else {
        oxifft::fft::<f32>(&c)
    };
    complex_to_interleaved(&result)
}

/// Row-major strides (in element units) for a shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Normalize a (possibly negative) DFT axis against a logical rank that
/// always counts the trailing real/imaginary component axis, even when that
/// axis is implicit (the 2-D real-signal shorthand `[batch, signal_len]`
/// with no explicit component dim). Mirrors the ONNX reference
/// implementation's `axis % rank(input)` wraparound (`onnx/reference/ops/
/// op_dft.py`), while rejecting an axis that resolves onto the reserved
/// component axis itself.
fn normalize_dft_axis(raw_axis: i64, logical_rank: usize) -> Result<usize, OnnxError> {
    let rank = logical_rank as i64;
    let normalized = raw_axis.rem_euclid(rank);
    if normalized == rank - 1 {
        return Err(OnnxError::ShapeMismatch(format!(
            "DFT: axis {raw_axis} resolves to the complex-component dimension \
             (logical rank={logical_rank}), which is not a valid DFT axis"
        )));
    }
    Ok(normalized as usize)
}

/// Resolve the `axis` parameter. Input[2] (opset 20+) takes precedence over
/// the `axis` attribute (opset 17-19), which in turn takes precedence over
/// the opset-20 spec default of -2 (the axis immediately preceding the
/// component axis -- for the common rank<=3 case this is identical to
/// opset 17's literal default of 1).
fn resolve_dft_axis(ctx: &OpContext<'_>, logical_rank: usize) -> Result<usize, OnnxError> {
    let raw_axis: i64 = if let Some(t) = ctx.optional_input(2) {
        scalar_i64(t, "DFT/axis")?
    } else if let Some(&a) = ctx.attrs().ints.get("axis") {
        a
    } else {
        -2
    };
    normalize_dft_axis(raw_axis, logical_rank)
}

pub struct DFTOp;
impl Operator for DFTOp {
    fn op_type(&self) -> &str {
        "DFT"
    }

    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let input = ctx.input(0)?;
        let inverse = ctx.attrs().i("inverse", 0) != 0;
        let onesided = if inverse {
            false // ONNX spec: onesided is ignored when inverse=true
        } else {
            ctx.attrs().i("onesided", 0) != 0
        };

        // Input shape is either:
        //   [..., signal_len]                — real signal shorthand, rank 2
        //                                       only (no explicit component axis)
        //   [..., signal_dim, 1]              — real signal (explicit component axis)
        //   [..., signal_dim, 2]              — complex signal (explicit component axis)
        // The transform may run along any axis except the trailing component
        // axis -- see the `axis` attribute (opset 17-19) / input[2] (opset
        // 20+), which may reference any outer dimension (including the batch
        // dimension), not just a hardcoded "signal" position.
        let ndim = input.shape.len();
        if ndim < 2 {
            return Err(OnnxError::ShapeMismatch(format!(
                "DFT: expected input rank >= 2, got {ndim}-D"
            )));
        }
        let (component_dim_present, is_complex) = if ndim == 2 {
            (false, false)
        } else {
            match input.shape[ndim - 1] {
                1 => (true, false),
                2 => (true, true),
                other => {
                    return Err(OnnxError::ShapeMismatch(format!(
                        "DFT: last dim must be 1 or 2, got {other}"
                    )));
                }
            }
        };

        let outer_shape: Vec<usize> = if component_dim_present {
            input.shape[..ndim - 1].to_vec()
        } else {
            input.shape.clone()
        };
        let outer_rank = outer_shape.len();
        // Logical rank always counts the (real or implicit) component axis,
        // matching the ONNX spec's `rank(input)` for axis normalization.
        let axis = resolve_dft_axis(ctx, outer_rank + 1)?;
        let signal_len = outer_shape[axis];

        let n = resolve_dft_length(ctx, signal_len)?;
        if n == 0 {
            return Err(OnnxError::ShapeMismatch(
                "DFT: resolved transform length (dft_length) is 0".into(),
            ));
        }
        let out_len = if onesided { n / 2 + 1 } else { n };

        let mut out_outer_shape = outer_shape.clone();
        out_outer_shape[axis] = out_len;

        let strides = row_major_strides(&outer_shape);
        let out_strides = row_major_strides(&out_outer_shape);
        let axis_stride = strides[axis];
        let axis_stride_out = out_strides[axis];
        // Raw-index multiplier: how many f32s wide one "outer element" is.
        // This is 1 whenever there is no complex pair to skip over --
        // whether because the component axis is absent (2-D shorthand) or
        // present with size 1 (real) -- and 2 when the component axis holds
        // an interleaved (re, im) pair.
        let component_size: usize = if is_complex { 2 } else { 1 };

        let total_out_outer: usize = out_outer_shape.iter().product();
        let mut out_data = vec![0.0f32; total_out_outer * 2];

        // Number of outer coordinate combinations excluding `axis` (an empty
        // product, when outer_rank == 1, correctly yields 1).
        let outer_iters: usize = outer_shape
            .iter()
            .enumerate()
            .filter(|&(d, _)| d != axis)
            .map(|(_, &dim)| dim)
            .product();

        let mut counter = vec![0usize; outer_rank];
        let mut interleaved: Vec<f32> = Vec::with_capacity(n * 2);
        for _ in 0..outer_iters {
            // `counter[axis]` is never incremented below, so it stays 0 and
            // contributes nothing here; the axis offset is added separately
            // via `k * axis_stride` in the gather/scatter loops.
            let base: usize = counter.iter().zip(&strides).map(|(&c, &s)| c * s).sum();
            let base_out: usize = counter.iter().zip(&out_strides).map(|(&c, &s)| c * s).sum();

            interleaved.clear();
            for k in 0..n {
                if k < signal_len {
                    let flat = (base + k * axis_stride) * component_size;
                    interleaved.push(input.data[flat]);
                    interleaved.push(if is_complex {
                        input.data[flat + 1]
                    } else {
                        0.0
                    });
                } else {
                    // Zero-pad when the requested transform length exceeds
                    // the signal length along `axis`.
                    interleaved.push(0.0);
                    interleaved.push(0.0);
                }
            }

            let result = dft_1d(&interleaved, n, inverse);
            for k in 0..out_len {
                let flat_out = (base_out + k * axis_stride_out) * 2;
                out_data[flat_out] = result[k * 2];
                out_data[flat_out + 1] = result[k * 2 + 1];
            }

            // Advance the odometer over every outer dim except `axis`.
            for d in (0..outer_rank).rev() {
                if d == axis {
                    continue;
                }
                counter[d] += 1;
                if counter[d] < outer_shape[d] {
                    break;
                }
                counter[d] = 0;
            }
        }

        // DFT output always carries an explicit trailing component axis of
        // size 2 (re, im), regardless of whether the input used the 2-D
        // real-signal shorthand.
        let mut out_shape = out_outer_shape;
        out_shape.push(2);
        Ok(vec![Tensor::new(out_data, out_shape)])
    }
}

#[cfg(test)]
mod axis_tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
        OpContext {
            node,
            inputs,
            outer_scope: None,
            weights: None,
            registry: None,
        }
    }

    fn dummy_node(op: OpKind) -> Node {
        Node {
            name: "test".into(),
            op,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: Attributes::default(),
        }
    }

    #[test]
    fn normalize_axis_default_v17_style_rank3() {
        // rank-3 canonical [batch, signal, component]: logical_rank=3.
        // -2 (the opset-20 default) must resolve to 1, matching opset-17's
        // literal attribute default.
        assert_eq!(normalize_dft_axis(-2, 3).unwrap(), 1);
        assert_eq!(normalize_dft_axis(1, 3).unwrap(), 1);
    }

    #[test]
    fn normalize_axis_rejects_component_axis() {
        // logical_rank=3: axis 2 (== -1) is the component axis, invalid.
        assert!(normalize_dft_axis(2, 3).is_err());
        assert!(normalize_dft_axis(-1, 3).is_err());
    }

    #[test]
    fn normalize_axis_wraps_large_values_like_reference() {
        // The ONNX reference implementation applies a plain `% rank` with no
        // range validation beyond landing on the component axis.
        // 100.rem_euclid(3) == 1; (-101).rem_euclid(3) == 1 (Python: 100 % 3
        // == 1, -101 % 3 == 1) -- both land on the same valid non-component
        // axis. (-100).rem_euclid(3) == 2 lands exactly on the component
        // axis and is covered separately by
        // `normalize_axis_rejects_component_axis`.
        assert_eq!(normalize_dft_axis(100, 3).unwrap(), 1);
        assert_eq!(normalize_dft_axis(-101, 3).unwrap(), 1);
    }

    #[test]
    fn dft_rejects_rank_below_2() {
        let input = Tensor::new(vec![1.0], vec![1]);
        let node = dummy_node(OpKind::DFT);
        let ctx = make_ctx(&node, vec![Some(&input), None]);
        assert!(DFTOp.execute(&ctx).is_err());
    }

    #[test]
    fn dft_rejects_bad_last_dim() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);
        let node = dummy_node(OpKind::DFT);
        let ctx = make_ctx(&node, vec![Some(&input), None]);
        assert!(DFTOp.execute(&ctx).is_err());
    }

    fn assert_close(actual: f32, expected: f64, msg: &str) {
        assert!(
            (actual as f64 - expected).abs() < 1e-4,
            "{msg}: got {actual}, expected {expected}"
        );
    }

    /// Regression for a1-17/a3-10: DFT along an explicit non-default axis
    /// (attribute, opset 17-19 style) on a rank-4 input `[1, 2, 3, 1]`
    /// (previously rejected outright -- only rank 2/3 was accepted).
    /// `axis=2` transforms the size-3 `d2` dimension; `onesided=1` keeps the
    /// first `3/2+1=2` bins. Reference: `numpy.fft.fft(x, axis=2)` with
    /// `x = np.arange(6).reshape(1,2,3)`.
    #[test]
    fn dft_axis_attribute_rank4_onesided() {
        let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 2, 3, 1]);
        let mut node = dummy_node(OpKind::DFT);
        node.attrs.ints.insert("axis".into(), 2);
        node.attrs.ints.insert("onesided".into(), 1);
        let ctx = make_ctx(&node, vec![Some(&input), None]);
        let out = DFTOp.execute(&ctx).expect("DFT axis=2 rank4 failed");
        // out_len = 3/2+1 = 2 -> shape [1, 2, 2, 2]
        assert_eq!(out[0].shape, vec![1, 2, 2, 2]);
        // b=0, d1=0: [(3, 0), (-1.5, 0.866025)]
        assert_close(out[0].data[0], 3.0, "d1=0 bin0 re");
        assert_close(out[0].data[1], 0.0, "d1=0 bin0 im");
        assert_close(out[0].data[2], -1.5, "d1=0 bin1 re");
        assert_close(out[0].data[3], 0.866_025, "d1=0 bin1 im");
        // b=0, d1=1: [(12, 0), (-1.5, 0.866025)]
        assert_close(out[0].data[4], 12.0, "d1=1 bin0 re");
        assert_close(out[0].data[5], 0.0, "d1=1 bin0 im");
        assert_close(out[0].data[6], -1.5, "d1=1 bin1 re");
        assert_close(out[0].data[7], 0.866_025, "d1=1 bin1 im");
    }

    /// Regression for a3-6/a3-10-style negative-axis handling, but for DFT:
    /// `axis` supplied as input[2] (opset 20+ style) with a spec-legal
    /// negative value. On the same `[1,2,3,1]` input, `axis=-3` normalizes
    /// against logical_rank=4 to outer dim 1 (`d1`, size 2). Reference:
    /// `numpy.fft.fft(x, axis=1)`.
    #[test]
    fn dft_axis_input_negative_rank4() {
        let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 2, 3, 1]);
        let axis_t = Tensor::new(vec![-3.0], vec![1]);
        let node = dummy_node(OpKind::DFT);
        let ctx = make_ctx(&node, vec![Some(&input), None, Some(&axis_t)]);
        let out = DFTOp
            .execute(&ctx)
            .expect("DFT axis=-3 (input) rank4 failed");
        // Full transform along dim1 (size 2, no onesided) -> shape [1, 2, 3, 2]
        assert_eq!(out[0].shape, vec![1, 2, 3, 2]);
        // Output layout is [batch=1, dim1(out_len=2), d2(3), component(2)]:
        // d2 stays a passthrough (untransformed) axis nested inside the
        // transformed dim1 axis. out[b, k, d2, :] = X[b, k, d2] where
        // X = numpy.fft.fft(x, axis=1).
        let x_fft_axis1 = [
            // k=0 (DC), d2=0,1,2
            (3.0_f64, 0.0),
            (5.0, 0.0),
            (7.0, 0.0),
            // k=1, d2=0,1,2
            (-3.0, 0.0),
            (-3.0, 0.0),
            (-3.0, 0.0),
        ];
        for k in 0..2 {
            for d2 in 0..3 {
                let base = (k * 3 + d2) * 2;
                let (re, im) = x_fft_axis1[k * 3 + d2];
                assert_close(out[0].data[base], re, &format!("k={k} d2={d2} re"));
                assert_close(out[0].data[base + 1], im, &format!("k={k} d2={d2} im"));
            }
        }
    }

    /// DFT along axis=0 (the leading/"batch" dimension itself) on a rank-3
    /// canonical input `[2, 3, 1]` -- the ONNX spec's valid axis range
    /// `[0, r-2]` explicitly permits axis 0, not just a hardcoded "signal"
    /// position. Reference: `numpy.fft.fft(y, axis=0)` with
    /// `y = [[1,2,3],[4,5,6]]`.
    #[test]
    fn dft_axis_zero_batch_dim() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);
        let mut node = dummy_node(OpKind::DFT);
        node.attrs.ints.insert("axis".into(), 0);
        let ctx = make_ctx(&node, vec![Some(&input), None]);
        let out = DFTOp.execute(&ctx).expect("DFT axis=0 failed");
        // Transform along dim0 (size 2) -> shape [2, 3, 2]
        assert_eq!(out[0].shape, vec![2, 3, 2]);
        // d1=0: [(5,0),(-3,0)]; d1=1: [(7,0),(-3,0)]; d1=2: [(9,0),(-3,0)]
        let expected = [
            (5.0_f64, 0.0),
            (7.0, 0.0),
            (9.0, 0.0),
            (-3.0, 0.0),
            (-3.0, 0.0),
            (-3.0, 0.0),
        ];
        for k in 0..2 {
            for d1 in 0..3 {
                let base = (k * 3 + d1) * 2;
                let (re, im) = expected[k * 3 + d1];
                assert_close(out[0].data[base], re, &format!("k={k} d1={d1} re"));
                assert_close(out[0].data[base + 1], im, &format!("k={k} d1={d1} im"));
            }
        }
    }
}
