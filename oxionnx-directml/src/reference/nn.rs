//! CPU oracle for the Wave-4 neural-network ops: `Softmax`, `Reduce` and `Conv`.
//!
//! A child of [`crate::reference`], obeying every rule its parent does.  Each function
//! reproduces exactly what the corresponding [`crate::hlsl`] shader (Softmax, Reduce)
//! or `DML_*_OPERATOR_DESC` (Conv) computes: the same accumulation order, the same f32
//! arithmetic, the same max-subtraction.  These are the executable specifications the
//! GPU path is diffed against by `DirectMLContext::self_check` and by
//! `OXIONNX_DIRECTML_VERIFY=1`.
//!
//! # The per-op tolerance, again, is the point
//!
//! | Op | Policy | Why |
//! |---|---|---|
//! | `Softmax` | [`Tolerance::Approx`], transcendental × √axis_len | `exp` is a hardware approximation on the GPU; the sum of `axis_len` of them drifts like `√axis_len`. |
//! | `ReduceSum`, `ReduceMean` | [`Tolerance::Approx`], scaled by √axis_len | a length-`axis_len` f32 sum the GPU may contract to `mad`, exactly like a matmul's dot product. |
//! | `ReduceMax`, `ReduceMin` | [`Tolerance::Exact`] | selection, not arithmetic — no legitimate source of drift. |
//! | `Conv` | [`Tolerance::Approx`], scaled by √(C_in/group · kH · kW) | a convolution is a matmul in disguise; its dot product has that length. |

use crate::error::{DirectMLError, Result};
use crate::plan::{numel, ConvPlan, ReduceKind, ReducePlan, SoftmaxPlan};

use super::{
    check_len, compare, ComparisonReport, Tolerance, MATMUL_ULP_BUDGET,
    TRANSCENDENTAL_ABS_TOLERANCE, TRANSCENDENTAL_REL_TOLERANCE,
};

// ─── softmax ─────────────────────────────────────────────────────────────────

/// Numerically-stable single-axis softmax — the executable spec of
/// [`crate::hlsl::SOFTMAX_HLSL`].
///
/// For each of the `outer × inner` rows it seeds the max with the row's first element
/// (as the shader does), subtracts it before every `exp`, sums in axis order, and
/// multiplies by the reciprocal of that sum.  Both `exp` passes exponentiate the
/// max-subtracted value, so the result is bit-identical to a shader that recomputes
/// `exp` for the store — which is precisely what [`crate::hlsl::SOFTMAX_HLSL`] does.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when `a` does not match the plan's shape.
pub fn ref_softmax(plan: &SoftmaxPlan, a: &[f32]) -> Result<Vec<f32>> {
    let elems = plan.output_elems()?;
    check_len("Softmax input", a.len(), elems)?;

    let outer = plan.outer as usize;
    let axis_len = plan.axis_len as usize;
    let inner = plan.inner as usize;

    let mut out = vec![0.0f32; elems];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_len * inner + i;

            // Row max, seeded with the first element exactly as the shader is.
            let mut m = a[base];
            for k in 1..axis_len {
                let v = a[base + k * inner];
                if v > m {
                    m = v;
                }
            }

            // Σ exp(x − m), summed in axis order.
            let mut sum = 0.0f32;
            for k in 0..axis_len {
                sum += (a[base + k * inner] - m).exp();
            }

            // y_k = exp(x_k − m) · (1 / Σ).  `1.0 / sum` matches the shader's `1.0 / sum`
            // (and `f32::recip`, which oxionnx-ops uses — they are the same instruction).
            let inv = 1.0f32 / sum;
            for k in 0..axis_len {
                out[base + k * inner] = (a[base + k * inner] - m).exp() * inv;
            }
        }
    }
    Ok(out)
}

// ─── reduce ──────────────────────────────────────────────────────────────────

/// Single-axis reduction — the executable spec of [`crate::hlsl::REDUCE_HLSL`].
///
/// `Sum` / `Mean` accumulate `acc += x_k` for `k = 0 … axis_len` in that sequential
/// order — the order the shader uses and the order `oxionnx-ops`' `reduce_with` walks
/// for a fixed output — so on integer-valued inputs the three agree bit for bit.
/// `Mean` then divides by `axis_len`.  `Max` / `Min` seed with the first element and
/// select, doing no arithmetic.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when `a` does not match the plan's input shape.
pub fn ref_reduce(plan: &ReducePlan, a: &[f32]) -> Result<Vec<f32>> {
    check_len("Reduce input", a.len(), numel(&plan.input_shape)?)?;

    let outer = plan.outer as usize;
    let axis_len = plan.axis_len as usize;
    let inner = plan.inner as usize;

    let mut out = vec![0.0f32; plan.out_count as usize];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_len * inner + i;
            let value = match plan.kind {
                ReduceKind::Sum => {
                    let mut acc = 0.0f32;
                    for k in 0..axis_len {
                        acc += a[base + k * inner];
                    }
                    acc
                }
                ReduceKind::Mean => {
                    let mut acc = 0.0f32;
                    for k in 0..axis_len {
                        acc += a[base + k * inner];
                    }
                    acc / axis_len as f32
                }
                ReduceKind::Max => {
                    let mut acc = a[base];
                    for k in 1..axis_len {
                        let v = a[base + k * inner];
                        if v > acc {
                            acc = v;
                        }
                    }
                    acc
                }
                ReduceKind::Min => {
                    let mut acc = a[base];
                    for k in 1..axis_len {
                        let v = a[base + k * inner];
                        if v < acc {
                            acc = v;
                        }
                    }
                    acc
                }
            };
            out[o * inner + i] = value;
        }
    }
    Ok(out)
}

// ─── conv ────────────────────────────────────────────────────────────────────

/// Direct 2-D convolution — the verification anchor for the DirectML `Conv` path.
///
/// This is a from-the-definition direct convolution, **not** an im2col+GEMM
/// transcription: for each output `(n, oc, oy, ox)` it sums
/// `input · weight` over `(ic_local, ky, kx)` in that nesting (which is `oxionnx-ops`'
/// im2col column order), skipping taps that fall in the padding.  Group `g = oc /
/// c_out_per_group` selects input channels `[g · c_in_per_group, …)`.  Because it
/// accumulates in a fixed order and `oxionnx-ops` reassociates through `matrixmultiply`,
/// the two agree bit for bit only on integer-valued inputs; on real data they agree
/// within [`Tolerance::for_conv`]'s `√depth`-scaled budget, and the DirectML
/// metacommand is held to the same.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when a buffer does not match its planned shape, or
/// when the plan carries a bias but none was supplied.
pub fn ref_conv(
    plan: &ConvPlan,
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    check_len("Conv input", input.len(), numel(&plan.input_shape)?)?;
    check_len("Conv weight", weight.len(), numel(&plan.weight_shape)?)?;

    let bias = match (plan.has_bias, bias) {
        (true, Some(b)) => {
            check_len("Conv bias", b.len(), plan.c_out as usize)?;
            Some(b)
        }
        (true, None) => {
            return Err(DirectMLError::ShapeMismatch(
                "Conv: the plan carries a bias but no bias buffer was supplied".into(),
            ))
        }
        (false, _) => None,
    };

    let n = plan.batch as usize;
    let c_in = plan.c_in as usize;
    let in_h = plan.in_h as isize;
    let in_w = plan.in_w as isize;
    let c_out = plan.c_out as usize;
    let c_in_per_group = plan.c_in_per_group as usize;
    let c_out_per_group = plan.c_out_per_group as usize;
    let kernel_h = plan.kernel_h as usize;
    let kernel_w = plan.kernel_w as usize;
    let out_h = plan.out_h as usize;
    let out_w = plan.out_w as usize;
    let stride_h = plan.stride_h as isize;
    let stride_w = plan.stride_w as isize;
    let pad_top = plan.pad_top as isize;
    let pad_left = plan.pad_left as isize;
    let dilation_h = plan.dilation_h as isize;
    let dilation_w = plan.dilation_w as isize;

    let in_hw = plan.in_h as usize * plan.in_w as usize;
    let out_hw = out_h * out_w;

    let mut out = vec![0.0f32; plan.output_elems()?];
    for bn in 0..n {
        for oc in 0..c_out {
            let group = oc / c_out_per_group;
            let ic_base = group * c_in_per_group;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = 0.0f32;
                    for ic_local in 0..c_in_per_group {
                        let ic = ic_base + ic_local;
                        let in_channel = (bn * c_in + ic) * in_hw;
                        let w_channel = (oc * c_in_per_group + ic_local) * kernel_h * kernel_w;
                        for ky in 0..kernel_h {
                            let iy = oy as isize * stride_h + ky as isize * dilation_h - pad_top;
                            if iy < 0 || iy >= in_h {
                                continue;
                            }
                            let in_row = in_channel + iy as usize * plan.in_w as usize;
                            let w_row = w_channel + ky * kernel_w;
                            for kx in 0..kernel_w {
                                let ix =
                                    ox as isize * stride_w + kx as isize * dilation_w - pad_left;
                                if ix < 0 || ix >= in_w {
                                    continue;
                                }
                                acc += input[in_row + ix as usize] * weight[w_row + kx];
                            }
                        }
                    }
                    if let Some(b) = bias {
                        acc += b[oc];
                    }
                    out[(bn * c_out + oc) * out_hw + oy * out_w + ox] = acc;
                }
            }
        }
    }
    Ok(out)
}

// ─── tolerance policy ────────────────────────────────────────────────────────

impl Tolerance {
    /// The policy for a `Softmax`.
    ///
    /// `exp` is a hardware approximation on the GPU and `libm`'s reference here, so the
    /// per-element error is transcendental; summing `axis_len` of them widens the
    /// relative budget by `√axis_len`.  The absolute floor is the transcendental one,
    /// which matters because softmax outputs of a long axis are individually small.
    #[must_use]
    pub fn for_softmax(plan: &SoftmaxPlan) -> Self {
        let scale = (f64::from(plan.axis_len.max(1)).sqrt()) as f32;
        Self::Approx {
            rel: TRANSCENDENTAL_REL_TOLERANCE * scale,
            abs: TRANSCENDENTAL_ABS_TOLERANCE,
        }
    }

    /// The policy for a `Reduce`.
    ///
    /// `Max` / `Min` are [`Tolerance::Exact`] — they select an element.  `Sum` / `Mean`
    /// accumulate a length-`axis_len` f32 sum the GPU may contract to `mad`, so they get
    /// the same `√axis_len`-scaled budget a matmul's dot product does.
    #[must_use]
    pub fn for_reduce(plan: &ReducePlan) -> Self {
        if plan.kind.is_exact() {
            Self::Exact
        } else {
            let k = (f64::from(plan.axis_len.max(1)).sqrt()) as f32;
            let rel = MATMUL_ULP_BUDGET * f32::EPSILON * k;
            Self::Approx { rel, abs: rel }
        }
    }

    /// The policy for a `Conv`, scaled by the dot-product length `C_in/group · kH · kW`.
    ///
    /// A convolution is a matmul whose inner dimension is that length, so it gets the
    /// matmul budget scaled by its square root — for the same reason (`mad`
    /// contraction, denormal flush) and by the same formula as [`Tolerance::for_matmul`].
    #[must_use]
    pub fn for_conv(plan: &ConvPlan) -> Self {
        let depth =
            u64::from(plan.c_in_per_group) * u64::from(plan.kernel_h) * u64::from(plan.kernel_w);
        let k = ((depth.max(1)) as f64).sqrt() as f32;
        let rel = MATMUL_ULP_BUDGET * f32::EPSILON * k;
        Self::Approx { rel, abs: rel }
    }
}

// ─── shadow verification ─────────────────────────────────────────────────────

/// Shadow-compare a GPU `Softmax` result against the oracle.
///
/// # Errors
/// Whatever [`ref_softmax`] returns, plus [`DirectMLError::ShapeMismatch`] when `gpu`
/// is not `plan.output_elems()` long.
pub fn verify_softmax(plan: &SoftmaxPlan, a: &[f32], gpu: &[f32]) -> Result<ComparisonReport> {
    let oracle = ref_softmax(plan, a)?;
    compare("Softmax", gpu, &oracle, Tolerance::for_softmax(plan))
}

/// Shadow-compare a GPU `Reduce` result against the oracle.
///
/// # Errors
/// Whatever [`ref_reduce`] returns, plus [`DirectMLError::ShapeMismatch`] when `gpu`
/// is not `plan.out_count` long.
pub fn verify_reduce(plan: &ReducePlan, a: &[f32], gpu: &[f32]) -> Result<ComparisonReport> {
    let oracle = ref_reduce(plan, a)?;
    compare(
        plan.kind.as_str(),
        gpu,
        &oracle,
        Tolerance::for_reduce(plan),
    )
}

/// Shadow-compare a GPU `Conv` result against the oracle.
///
/// # Errors
/// Whatever [`ref_conv`] returns, plus [`DirectMLError::ShapeMismatch`] when `gpu` is
/// not `plan.output_elems()` long.
pub fn verify_conv(
    plan: &ConvPlan,
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    gpu: &[f32],
) -> Result<ComparisonReport> {
    let oracle = ref_conv(plan, input, weight, bias)?;
    compare("Conv", gpu, &oracle, Tolerance::for_conv(plan))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::plan::ReduceKind;

    // ── ref_softmax ──────────────────────────────────────────────────────────

    #[test]
    fn softmax_of_a_single_row_sums_to_one_and_matches_hand_values() {
        // softmax([1, 2, 3]) = exp(k) normalised.
        let plan = SoftmaxPlan::softmax(&[3], 0).unwrap();
        let out = ref_softmax(&plan, &[1.0, 2.0, 3.0]).unwrap();
        let sum: f32 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1.0e-6,
            "softmax must normalise to 1, got {sum}"
        );
        // Monotone inputs → monotone outputs.
        assert!(out[0] < out[1] && out[1] < out[2]);
        let e = [1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()];
        let denom = e[0] + e[1] + e[2];
        for k in 0..3 {
            assert!((out[k] - e[k] / denom).abs() < 1.0e-6);
        }
    }

    #[test]
    fn softmax_is_shift_invariant_and_stable_on_large_inputs() {
        // The whole reason for the max-subtraction: a row with a huge positive value must
        // not overflow to NaN, and softmax(x) == softmax(x + c).
        let plan = SoftmaxPlan::softmax(&[4], 0).unwrap();
        let small = ref_softmax(&plan, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let shifted = ref_softmax(&plan, &[1001.0, 1002.0, 1003.0, 1004.0]).unwrap();
        for (a, b) in small.iter().zip(shifted.iter()) {
            assert!((a - b).abs() < 1.0e-6, "softmax must be shift-invariant");
            assert!(
                !b.is_nan(),
                "the max-subtraction must prevent overflow to NaN"
            );
        }
    }

    #[test]
    fn softmax_over_a_middle_axis_normalises_each_row_independently() {
        // [2, 3] over axis 0: each column is a softmax row (inner = 3).
        let plan = SoftmaxPlan::softmax(&[2, 3], 0).unwrap();
        assert_eq!(plan.inner, 3);
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = ref_softmax(&plan, &a).unwrap();
        // Each of the 3 columns (indices {0,3}, {1,4}, {2,5}) sums to 1.
        for i in 0..3 {
            let s = out[i] + out[i + 3];
            assert!((s - 1.0).abs() < 1.0e-6, "column {i} sums to {s}");
        }
    }

    #[test]
    fn softmax_rejects_a_mis_sized_buffer() {
        let plan = SoftmaxPlan::softmax(&[2, 3], 1).unwrap();
        let err = ref_softmax(&plan, &[1.0, 2.0]).expect_err("too short");
        assert!(matches!(err, DirectMLError::ShapeMismatch(_)), "{err}");
    }

    // ── ref_reduce ───────────────────────────────────────────────────────────

    #[test]
    fn reduce_sum_and_mean_over_the_last_axis() {
        let plan = ReducePlan::reduce(ReduceKind::Sum, &[2, 3], &[1], false).unwrap();
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(ref_reduce(&plan, &a).unwrap(), vec![6.0, 15.0]);

        let plan = ReducePlan::reduce(ReduceKind::Mean, &[2, 3], &[1], false).unwrap();
        assert_eq!(ref_reduce(&plan, &a).unwrap(), vec![2.0, 5.0]);
    }

    #[test]
    fn reduce_over_a_strided_axis_uses_the_right_offsets() {
        // [2, 3] reduce axis 0 → inner = 3, each output gathers rows 0 and 1 of a column.
        let plan = ReducePlan::reduce(ReduceKind::Sum, &[2, 3], &[0], false).unwrap();
        assert_eq!(plan.inner, 3);
        assert_eq!(plan.axis_len, 2);
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // column sums: 1+4, 2+5, 3+6.
        assert_eq!(ref_reduce(&plan, &a).unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_max_and_min_select_the_extreme() {
        let a = [1.0, -2.0, 3.0, 0.5, 5.0, -6.0];
        let max = ReducePlan::reduce(ReduceKind::Max, &[2, 3], &[1], false).unwrap();
        assert_eq!(ref_reduce(&max, &a).unwrap(), vec![3.0, 5.0]);
        let min = ReducePlan::reduce(ReduceKind::Min, &[2, 3], &[1], false).unwrap();
        assert_eq!(ref_reduce(&min, &a).unwrap(), vec![-2.0, -6.0]);
    }

    #[test]
    fn reduce_sum_accumulates_in_axis_order() {
        // The same catastrophic-cancellation case the matmul oracle pins: a sequential
        // left-associated f32 sum gives 0, a reassociating one might give 1.
        let plan = ReducePlan::reduce(ReduceKind::Sum, &[3], &[0], false).unwrap();
        let a = [1.0f32, 1.0e8, -1.0e8];
        assert_eq!(
            ref_reduce(&plan, &a).unwrap(),
            vec![0.0],
            "the oracle must reproduce the shader's sequential k-major rounding"
        );
    }

    // ── ref_conv ─────────────────────────────────────────────────────────────

    #[test]
    fn conv_identity_kernel_copies_the_input() {
        // 1x1x3x3 input, a single 1x1 kernel of value 1 → output == input.
        let plan = ConvPlan::conv(&[1, 1, 3, 3], &[1, 1, 1, 1], None, &[], &[], &[], 1).unwrap();
        let input: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let out = ref_conv(&plan, &input, &[1.0], None).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn conv_3x3_sum_kernel_matches_a_hand_computed_window() {
        // A 3x3 kernel of all ones over a 3x3 input, no padding → one output = the sum.
        let plan = ConvPlan::conv(&[1, 1, 3, 3], &[1, 1, 3, 3], None, &[], &[], &[], 1).unwrap();
        assert_eq!(plan.output_shape, vec![1, 1, 1, 1]);
        let input: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let weight = vec![1.0f32; 9];
        let out = ref_conv(&plan, &input, &weight, None).unwrap();
        assert_eq!(out, vec![45.0], "sum of 1..=9");
    }

    #[test]
    fn conv_padding_pulls_in_zeros_at_the_border() {
        // 3x3 kernel of ones over a 2x2 input, pad 1 all sides → 2x2 output; each output
        // is the sum of the in-bounds taps.
        let plan = ConvPlan::conv(
            &[1, 1, 2, 2],
            &[1, 1, 3, 3],
            None,
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            1,
        )
        .unwrap();
        assert_eq!(plan.output_shape, vec![1, 1, 2, 2]);
        // input = [[1,2],[3,4]]; each 3x3 window sums the whole 2x2 (all four visible at
        // every output position because pad 1 keeps the 2x2 inside every window).
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; 9];
        let out = ref_conv(&plan, &input, &weight, None).unwrap();
        assert_eq!(
            out,
            vec![10.0, 10.0, 10.0, 10.0],
            "every window sees all of 1+2+3+4"
        );
    }

    #[test]
    fn conv_bias_is_added_per_output_channel() {
        let plan =
            ConvPlan::conv(&[1, 1, 3, 3], &[2, 1, 3, 3], Some(&[2]), &[], &[], &[], 1).unwrap();
        let input: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        // Two filters: all-ones (sum = 45) and all-twos (sum = 90).
        let mut weight = vec![1.0f32; 9];
        weight.extend(vec![2.0f32; 9]);
        let out = ref_conv(&plan, &input, &weight, Some(&[10.0, -5.0])).unwrap();
        assert_eq!(out, vec![45.0 + 10.0, 90.0 - 5.0]);
    }

    #[test]
    fn conv_groups_keep_channels_separate() {
        // group 2: C_in 2, C_out 2, each filter sees one input channel.
        let plan = ConvPlan::conv(&[1, 2, 2, 2], &[2, 1, 2, 2], None, &[], &[], &[], 2).unwrap();
        assert_eq!(plan.output_shape, vec![1, 2, 1, 1]);
        // channel 0 = [1,2,3,4], channel 1 = [5,6,7,8].
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // filter 0 (ones) applies to channel 0 → 10; filter 1 (ones) to channel 1 → 26.
        let weight = vec![1.0f32; 8];
        let out = ref_conv(&plan, &input, &weight, None).unwrap();
        assert_eq!(out, vec![10.0, 26.0]);
    }

    #[test]
    fn conv_rejects_a_bias_the_plan_expects_but_the_caller_omits() {
        let plan =
            ConvPlan::conv(&[1, 1, 3, 3], &[1, 1, 3, 3], Some(&[1]), &[], &[], &[], 1).unwrap();
        let err = ref_conv(&plan, &[0.0f32; 9], &[0.0f32; 9], None)
            .expect_err("the plan has a bias but none was passed");
        assert!(matches!(err, DirectMLError::ShapeMismatch(_)), "{err}");
    }

    // ── tolerance policy ─────────────────────────────────────────────────────

    #[test]
    fn reduce_max_min_are_exact_and_sum_mean_are_not() {
        let max = ReducePlan::reduce(ReduceKind::Max, &[2, 3], &[1], false).unwrap();
        assert_eq!(Tolerance::for_reduce(&max), Tolerance::Exact);
        let min = ReducePlan::reduce(ReduceKind::Min, &[2, 3], &[1], false).unwrap();
        assert_eq!(Tolerance::for_reduce(&min), Tolerance::Exact);

        let sum = ReducePlan::reduce(ReduceKind::Sum, &[2, 3], &[1], false).unwrap();
        assert!(matches!(
            Tolerance::for_reduce(&sum),
            Tolerance::Approx { .. }
        ));
        let mean = ReducePlan::reduce(ReduceKind::Mean, &[2, 3], &[1], false).unwrap();
        assert!(matches!(
            Tolerance::for_reduce(&mean),
            Tolerance::Approx { .. }
        ));
    }

    #[test]
    fn reduce_sum_tolerance_scales_with_the_axis_length() {
        let short = ReducePlan::reduce(ReduceKind::Sum, &[2, 3], &[1], false).unwrap();
        let long = ReducePlan::reduce(ReduceKind::Sum, &[2, 4096], &[1], false).unwrap();
        let (Tolerance::Approx { rel: r_s, .. }, Tolerance::Approx { rel: r_l, .. }) =
            (Tolerance::for_reduce(&short), Tolerance::for_reduce(&long))
        else {
            panic!("sum is never exact");
        };
        assert!(r_l > r_s * 30.0, "√(4096/3) ≈ 37: {r_l} vs {r_s}");
    }

    #[test]
    fn conv_tolerance_scales_with_the_reduction_depth() {
        let shallow = ConvPlan::conv(&[1, 1, 5, 5], &[1, 1, 1, 1], None, &[], &[], &[], 1).unwrap();
        let deep = ConvPlan::conv(&[1, 64, 8, 8], &[1, 64, 7, 7], None, &[], &[], &[], 1).unwrap();
        let (Tolerance::Approx { rel: r_s, .. }, Tolerance::Approx { rel: r_d, .. }) =
            (Tolerance::for_conv(&shallow), Tolerance::for_conv(&deep))
        else {
            panic!("conv is never exact");
        };
        // depth 1 vs 64*49 = 3136 → √3136 = 56.
        assert!(
            r_d > r_s * 40.0,
            "a deep conv gets a far wider budget: {r_d} vs {r_s}"
        );
    }

    #[test]
    fn softmax_tolerance_is_transcendental_and_widens_with_the_axis() {
        let short = SoftmaxPlan::softmax(&[8], 0).unwrap();
        let long = SoftmaxPlan::softmax(&[1024], 0).unwrap();
        let (Tolerance::Approx { rel: r_s, .. }, Tolerance::Approx { rel: r_l, .. }) = (
            Tolerance::for_softmax(&short),
            Tolerance::for_softmax(&long),
        ) else {
            panic!("softmax is never exact");
        };
        assert!(r_s > 0.0);
        assert!(
            r_l > r_s,
            "a longer softmax axis sums more exps and drifts more"
        );
    }

    // ── verify_* ─────────────────────────────────────────────────────────────

    #[test]
    fn verify_softmax_passes_on_the_oracle_and_catches_a_perturbation() {
        let plan = SoftmaxPlan::softmax(&[2, 4], 1).unwrap();
        let a: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
        let gpu = ref_softmax(&plan, &a).unwrap();
        let report = verify_softmax(&plan, &a, &gpu).unwrap();
        assert!(report.passed);
        assert_eq!(report.op, "Softmax");

        let mut bad = gpu.clone();
        bad[5] += 0.01;
        assert!(!verify_softmax(&plan, &a, &bad).unwrap().passed);
    }

    #[test]
    fn verify_reduce_reports_the_kind_and_catches_an_exact_violation() {
        let plan = ReducePlan::reduce(ReduceKind::Max, &[2, 4], &[1], false).unwrap();
        let a = [1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0, 5.0];
        let mut gpu = ref_reduce(&plan, &a).unwrap();
        assert!(verify_reduce(&plan, &a, &gpu).unwrap().passed);
        assert_eq!(verify_reduce(&plan, &a, &gpu).unwrap().op, "ReduceMax");
        // Max is exact: a 1e-6 drift is a bug, not noise.
        gpu[0] += 1.0e-6;
        let report = verify_reduce(&plan, &a, &gpu).unwrap();
        assert!(!report.passed);
        assert_eq!(report.first_mismatch.expect("mismatched").index, 0);
    }

    #[test]
    fn verify_conv_passes_on_the_oracle_and_catches_a_perturbation() {
        let plan =
            ConvPlan::conv(&[1, 2, 4, 4], &[3, 2, 3, 3], Some(&[3]), &[], &[], &[], 1).unwrap();
        let input: Vec<f32> = (0..32).map(|i| (i % 5) as f32 - 2.0).collect();
        let weight: Vec<f32> = (0..54).map(|i| (i % 3) as f32 - 1.0).collect();
        let bias = [0.5, -0.5, 1.0];
        let gpu = ref_conv(&plan, &input, &weight, Some(&bias)).unwrap();
        let report = verify_conv(&plan, &input, &weight, Some(&bias), &gpu).unwrap();
        assert!(report.passed);
        assert_eq!(report.op, "Conv");

        let mut bad = gpu.clone();
        bad[0] += 1.0;
        assert!(
            !verify_conv(&plan, &input, &weight, Some(&bias), &bad)
                .unwrap()
                .passed
        );
    }
}
