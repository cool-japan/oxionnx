//! A minimal, deliberately-naive CPU oracle for shadow-verifying CUDA kernel
//! output, gated behind `OXIONNX_CUDA_VERIFY=1`.
//!
//! # Why this exists
//!
//! `try_cuda_dispatch` returns `Ok(Some(wrong_data))` — a *successful* answer
//! — when a kernel is wrong; nothing downstream can tell the difference from
//! a correct one.  This module is the mechanism that can: it recomputes the
//! same op on the CPU with the simplest, most obviously-correct
//! implementation available (naive loops, `f64` accumulation, no batching,
//! no SIMD) and diffs it against what the GPU returned.  See
//! [`crate::context`] for the full activation/verify/strict story.
//!
//! # Why this does not depend on `oxionnx-ops`
//!
//! Mirroring `oxionnx-directml`'s identical layering rule: an execution
//! provider must not depend on the CPU operator library it exists to bypass.
//! A normal dependency on `oxionnx-ops` here would invert that, and would
//! make it possible for a bug to accidentally use `oxionnx-ops` as the
//! *dispatch fallback* instead of returning `Ok(None)` and letting the
//! session runner do it.  So every oracle below is reimplemented from
//! scratch, small enough to read in one sitting, and cross-checked against
//! hand-computed constants in this module's own tests.
//!
//! # What the oracle models
//!
//! Deliberately **not** the ONNX spec in the abstract — the exact formula
//! each `oxicuda_ptx` kernel template computes, including the two ops
//! (`LeakyRelu`, `HardSigmoid`) that hard-code the ONNX *default* constants
//! with no launch-time override, and `Gelu`, whose kernel computes the
//! `tanh` approximation rather than the exact/erf form.  This matches
//! `try_cuda_dispatch`'s dispatch-time guards (see `lib.rs`), which decline
//! to the CPU whenever a node's actual attributes would disagree with these
//! constants — so by the time any of these oracle functions runs, the
//! constants below are exactly what the kernel is computing.  Verified
//! directly against `oxicuda_ptx::templates::elementwise`'s
//! `generate_leaky_relu` / `generate_hard_sigmoid` / `generate_gelu` doc
//! comments and PTX literals (`0f3C23D70A` = 0.01, `0f3E4CCCCD` = 0.2,
//! `0f3F000000` = 0.5).

use oxionnx_core::graph::OpKind;

use crate::context::{parse_env_flag, FailurePolicy};
use crate::conv::ConvParams;
use crate::error::CudaDispatchError;

// ─── the verify gate ────────────────────────────────────────────────────────

/// Set this to shadow-verify every CUDA-dispatched op against this module's
/// CPU oracle: `OXIONNX_CUDA_VERIFY=1`.
///
/// Off by default: computing the oracle roughly doubles the cost of every
/// claimed node, so this is a diagnostic mode for development/CI on real
/// hardware, not something a production build should carry.
pub const VERIFY_ENV_VAR: &str = "OXIONNX_CUDA_VERIFY";

/// Is shadow verification on?
///
/// Read once and cached: the value cannot change within a process, and this
/// is consulted on the dispatch path of every claimed node.
#[must_use]
pub fn verify_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_flag(std::env::var(VERIFY_ENV_VAR).ok().as_deref()))
}

// ─── comparison ─────────────────────────────────────────────────────────────

/// Absolute tolerance for the GPU-vs-oracle comparison.
///
/// `oxicuda_ptx`'s transcendental kernels use `ex2.approx` / `lg2.approx` /
/// `rcp.approx` (PTX's reduced-precision fast-math intrinsics, documented to
/// roughly 22-23 bits of accuracy) rather than a correctly-rounded `exp`, so
/// bit-exact agreement is not the bar — a value that is *wrong for the
/// formula*, not merely differently-rounded, is.
const ATOL: f32 = 1.0e-4;
/// Relative tolerance, applied against the oracle's magnitude. Wide enough to
/// absorb a longer MatMul/ReduceSum accumulation's rounding drift without
/// absorbing a genuinely wrong element.
const RTOL: f32 = 1.0e-3;

/// Do `gpu` and `cpu` agree within tolerance?
///
/// Two `NaN`s agree (both engines saturated/undefined the same way); one
/// `NaN` and one finite value never do. `+inf`/`-inf` must match exactly.
fn nearly_eq(gpu: f32, cpu: f32) -> bool {
    if gpu == cpu {
        return true; // exact match, including matching +-inf and +-0.0.
    }
    if gpu.is_nan() || cpu.is_nan() {
        return gpu.is_nan() && cpu.is_nan();
    }
    if !gpu.is_finite() || !cpu.is_finite() {
        return false; // one side is infinite, the other is not (and not NaN either).
    }
    let diff = (gpu - cpu).abs();
    diff <= ATOL + RTOL * cpu.abs()
}

/// Compare a GPU result against the oracle's, element by element.
///
/// # Errors
/// A message naming the first disagreement (index, both values, and the raw
/// difference) — the single most diagnostic thing a mismatch report can
/// carry: an early index says "the first thread group is wrong"; a late one
/// says "only the tail is wrong".
pub fn compare(gpu: &[f32], cpu: &[f32]) -> Result<(), String> {
    if gpu.len() != cpu.len() {
        return Err(format!(
            "length mismatch: GPU returned {} elements, the CPU oracle computed {}",
            gpu.len(),
            cpu.len()
        ));
    }
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        if !nearly_eq(g, c) {
            return Err(format!(
                "element {i}: GPU={g}, CPU-oracle={c} (|diff|={:.3e}, tolerance={:.3e})",
                (g - c).abs(),
                ATOL + RTOL * c.abs()
            ));
        }
    }
    Ok(())
}

// ─── shadow_verify: the glue between a kernel call and the policy ─────────

/// Shadow-verify a GPU kernel's output against this module's CPU oracle, and
/// decide what the caller should do about it.
///
/// `verify_on` and `policy` are taken as plain parameters — rather than read
/// internally from [`verify_enabled`] / [`FailurePolicy::current`] — purely
/// so this function is unit-testable without mutating the process
/// environment (global, racy under a threaded test runner, and cached by
/// both of those `OnceLock`s besides). Real call sites in `lib.rs` pass the
/// live values.
///
/// `oracle` is a closure — not a plain `&[f32]` — so the (potentially
/// expensive) CPU computation is skipped entirely when `verify_on` is
/// `false`, which is the default and every production dispatch.
///
/// # Returns
/// * `Ok(true)` — verification is off, or on and passed: the caller should
///   trust `gpu` and proceed normally.
/// * `Ok(false)` — verification is on, the comparison disagreed, and
///   `policy` is [`FailurePolicy::Fallback`] (the default). The mismatch has
///   already been logged at `error!`. The caller **must** discard `gpu` and
///   return `Ok(None)` so the real CPU operator recomputes the node.
/// * `Ok(true)` (with a `warn!` logged) — the oracle has no formula for this
///   op/configuration (`oracle` returned `None`). This is a gap in the
///   *oracle*, not a proven GPU bug, so it is never promoted to `Err`, but a
///   silent skip would defeat the purpose of `OXIONNX_CUDA_VERIFY` — it is
///   always audible.
///
/// # Errors
/// [`CudaDispatchError::Verify`] only when `policy` is
/// [`FailurePolicy::Strict`] and the comparison disagreed.
pub(crate) fn shadow_verify(
    op: &str,
    gpu: &[f32],
    verify_on: bool,
    policy: FailurePolicy,
    oracle: impl FnOnce() -> Option<Vec<f32>>,
) -> Result<bool, CudaDispatchError> {
    if !verify_on {
        return Ok(true);
    }
    let Some(cpu) = oracle() else {
        tracing::warn!(
            op,
            "CUDA shadow verification: the CPU oracle has no formula for this op/configuration; \
             skipping the check for this node (this is a gap in the oracle, not a proven GPU bug)"
        );
        return Ok(true);
    };
    match compare(gpu, &cpu) {
        Ok(()) => {
            tracing::debug!(op, "CUDA shadow verification passed");
            Ok(true)
        }
        Err(reason) => {
            tracing::error!(
                op,
                %reason,
                strict = policy == FailurePolicy::Strict,
                "CUDA kernel VERIFY MISMATCH (a GPU kernel bug, not a decline) — the GPU's \
                 output has been discarded, not returned. Set {} to make this fatal.",
                crate::context::STRICT_ENV_VAR,
            );
            match policy {
                FailurePolicy::Strict => Err(CudaDispatchError::Verify(reason)),
                FailurePolicy::Fallback => Ok(false),
            }
        }
    }
}

// ─── the oracle itself ──────────────────────────────────────────────────────

/// Naive `[m, k] x [k, n] -> [m, n]` row-major matmul.
///
/// `O(m*k*n)` with an `f64` accumulator per output element — deliberately
/// unoptimised; this exists to be obviously correct, not fast, and is only
/// ever called behind [`verify_enabled`].
#[must_use]
pub fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

/// Naive NCHW cross-correlation (the `cuDNN`/ONNX `Conv` convention — no
/// 180-degree kernel flip), honouring padding, stride, dilation, and
/// groups, with an optional per-output-channel bias.
///
/// `input` is `[N, C, H, W]` (`in_shape`), `weight` is `[K, C/groups, R, S]`
/// (`weight_shape`) — the exact layout the ONNX `Conv` operator (and
/// [`crate::conv::cuda_conv`], which uploads this same layout to the GPU)
/// uses. `bias`, when present, is `[K]`, added once per output channel and
/// broadcast across every batch and spatial position.
///
/// Only `params.pads`' first two elements (`[pad_top, pad_left]`) are read.
/// That is not a shortcut: every call site in this crate reaches this
/// oracle only after [`crate::conv::cuda_conv`] has already declined any
/// node whose padding is asymmetric (`pads[0] != pads[2] || pads[1] !=
/// pads[3]` — see the [`crate::conv`] module docs' "What still declines"
/// section), so `pads[2]`/`pads[3]` are guaranteed to equal
/// `pads[0]`/`pads[1]` by the time this oracle ever runs in this crate.
///
/// `O(N*K*P*Q*(C/groups)*R*S)` with an `f64` accumulator per output
/// element, cast to `f32` only once, at the very end — deliberately
/// unoptimised, the same discipline as [`ref_matmul`]: this exists to be
/// obviously correct, not fast.
///
/// # Panics
/// Indexes `input`/`weight`/`bias`/`in_shape`/`weight_shape` directly with
/// no bounds- or length-checking, the same discipline as [`ref_matmul`]: a
/// caller-supplied shape/data-length mismatch (or a `params.group` of `0`)
/// panics rather than silently computing garbage. Every call site in this
/// crate derives its arguments from a `ConvProblem` that
/// [`crate::conv::cuda_conv`] has already validated end-to-end (rank-4
/// shapes, non-zero dims, `group` dividing both channel counts, the filter
/// fitting the padded input) before this oracle ever runs, so a panic here
/// means the caller passed shapes it never validated — a bug in the caller,
/// not a normal outcome.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn ref_conv(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_shape: &[usize],
    weight_shape: &[usize],
    params: &ConvParams,
) -> Vec<f32> {
    let n = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let out_channels = weight_shape[0];
    let in_ch_per_group = weight_shape[1];
    let filter_h = weight_shape[2];
    let filter_w = weight_shape[3];

    let [stride_h, stride_w] = params.strides;
    let [pad_h, pad_w, _pad_bottom, _pad_right] = params.pads;
    let [dil_h, dil_w] = params.dilations;
    let group = params.group;
    let out_ch_per_group = out_channels / group;

    // Standard convolution output-size formula (matches
    // `oxicuda_dnn::conv::descriptor::ConvProblem::output_dims`'s
    // `ConvolutionDescriptor::output_size`, and this crate's own
    // `conv::tests::gpu_numeric::ConvCase::out_hw`):
    // `floor((dim + 2*pad - dilation*(k-1) - 1) / stride) + 1`.
    let eff_h = dil_h * (filter_h - 1) + 1;
    let eff_w = dil_w * (filter_w - 1) + 1;
    let out_h = (in_h + 2 * pad_h - eff_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - eff_w) / stride_w + 1;

    let mut out = vec![0.0_f32; n * out_channels * out_h * out_w];
    for ni in 0..n {
        for ki in 0..out_channels {
            // Which group this output channel belongs to -- `weight`'s
            // leading `[K, ...]` dim is laid out group-major (the first
            // `out_ch_per_group` filters belong to group 0, the next
            // `out_ch_per_group` to group 1, and so on), matching ONNX's
            // `Conv` spec and `oxicuda_dnn`'s own kernel bodies.
            let g = ki / out_ch_per_group;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0_f64;
                    for cg in 0..in_ch_per_group {
                        let ci = g * in_ch_per_group + cg;
                        for ri in 0..filter_h {
                            // Implicit zero-padding: an input coordinate
                            // that lands outside `[0, in_h)`/`[0, in_w)`
                            // contributes nothing, rather than being an
                            // error -- this is what makes the padding
                            // "same"-style output sizes correct.
                            let ih = oh as isize * stride_h as isize - pad_h as isize
                                + ri as isize * dil_h as isize;
                            if ih < 0 || ih as usize >= in_h {
                                continue;
                            }
                            let ih = ih as usize;
                            for si in 0..filter_w {
                                let iw = ow as isize * stride_w as isize - pad_w as isize
                                    + si as isize * dil_w as isize;
                                if iw < 0 || iw as usize >= in_w {
                                    continue;
                                }
                                let iw = iw as usize;
                                let in_idx = ((ni * in_channels + ci) * in_h + ih) * in_w + iw;
                                let f_idx =
                                    ((ki * in_ch_per_group + cg) * filter_h + ri) * filter_w + si;
                                acc += f64::from(input[in_idx]) * f64::from(weight[f_idx]);
                            }
                        }
                    }
                    if let Some(bv) = bias {
                        acc += f64::from(bv[ki]);
                    }
                    let o_idx = ((ni * out_channels + ki) * out_h + oh) * out_w + ow;
                    out[o_idx] = acc as f32;
                }
            }
        }
    }
    out
}

/// Naive per-axis `ReduceSum` / `ReduceMax`.
///
/// `shape` is decomposed as `[outer, axis_len, inner]` around `axis`,
/// matching [`crate::reduce::cuda_reduce`]'s own layout. Returns `None` for
/// an out-of-range axis or an op this oracle has no formula for (the caller
/// treats that as "skip the check", not "the GPU is wrong").
#[must_use]
pub fn ref_reduce(op: &OpKind, data: &[f32], shape: &[usize], axis: usize) -> Option<Vec<f32>> {
    if axis >= shape.len() {
        return None;
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    if outer == 0 || axis_len == 0 || inner == 0 {
        return None;
    }

    let mut out = vec![0.0_f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            match op {
                OpKind::ReduceSum => {
                    let mut acc = 0.0_f64;
                    for a in 0..axis_len {
                        acc += f64::from(data[(o * axis_len + a) * inner + i]);
                    }
                    out[o * inner + i] = acc as f32;
                }
                OpKind::ReduceMax => {
                    let mut acc = f32::NEG_INFINITY;
                    for a in 0..axis_len {
                        acc = acc.max(data[(o * axis_len + a) * inner + i]);
                    }
                    out[o * inner + i] = acc;
                }
                _ => return None,
            }
        }
    }
    Some(out)
}

/// Naive Softmax over the last dimension: the standard
/// max-subtraction-then-normalise formula, `f64` accumulation for the
/// denominator.
///
/// `shape` must be non-empty (mirrors [`crate::softmax::cuda_softmax`]'s own
/// precondition; the caller already declined an empty shape before this
/// would ever run).
#[must_use]
pub fn ref_softmax(data: &[f32], shape: &[usize]) -> Option<Vec<f32>> {
    let &row = shape.last()?;
    if row == 0 {
        return Some(Vec::new());
    }
    let rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    if rows.checked_mul(row) != Some(data.len()) {
        return None;
    }

    let mut out = vec![0.0_f32; data.len()];
    let mut exps = vec![0.0_f64; row];
    for r in 0..rows {
        let base = r * row;
        let slice = &data[base..base + row];
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f64;
        for (i, &x) in slice.iter().enumerate() {
            let e = f64::from(x - max).exp();
            exps[i] = e;
            sum += e;
        }
        for (i, &e) in exps.iter().enumerate() {
            out[base + i] = (e / sum) as f32;
        }
    }
    Some(out)
}

/// The elementwise unary formula matching exactly what the corresponding
/// `oxicuda_ptx` kernel computes — see the [module docs](self) for why this
/// is not always the plain ONNX-spec formula.
///
/// `f64` intermediate arithmetic throughout; the final cast to `f32` is the
/// only place precision is dropped, so the oracle's own rounding never
/// competes with the GPU's approximate transcendentals for `compare`'s
/// (crate-private) tolerance budget.
///
/// Returns `None` for any `op` this oracle has no formula for.
#[must_use]
pub fn ref_unary(op: &OpKind, x: f32) -> Option<f32> {
    let xf = f64::from(x);
    let y: f64 = match op {
        OpKind::Relu => xf.max(0.0),
        OpKind::Sigmoid => 1.0 / (1.0 + (-xf).exp()),
        // GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) — the tanh
        // approximation, matching `oxicuda_ptx::generate_gelu`'s own doc comment, NOT
        // ONNX's exact/erf default.
        OpKind::Gelu => {
            let x3 = xf * xf * xf;
            let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (xf + 0.044_715 * x3);
            0.5 * xf * (1.0 + inner.tanh())
        }
        OpKind::Tanh => xf.tanh(),
        OpKind::Exp => xf.exp(),
        OpKind::Sqrt => xf.sqrt(),
        OpKind::Abs => xf.abs(),
        OpKind::Neg => -xf,
        OpKind::Log => xf.ln(),
        OpKind::Ceil => xf.ceil(),
        OpKind::Floor => xf.floor(),
        // clamp(0.2*x + 0.5, 0, 1) — ONNX's default alpha/beta, hard-coded in the kernel.
        OpKind::HardSigmoid => (0.2 * xf + 0.5).clamp(0.0, 1.0),
        // x * clamp(x + 3, 0, 6) / 6.
        OpKind::HardSwish => xf * (xf + 3.0).clamp(0.0, 6.0) / 6.0,
        OpKind::SiLU => xf * (1.0 / (1.0 + (-xf).exp())),
        OpKind::Softplus => (1.0 + xf.exp()).ln(),
        // x >= 0 ? x : 0.01*x — ONNX's default alpha, hard-coded in the kernel.
        OpKind::LeakyRelu => {
            if xf >= 0.0 {
                xf
            } else {
                0.01 * xf
            }
        }
        _ => return None,
    };
    Some(y as f32)
}

/// Map `ref_unary` over every element of `data`.
///
/// `None` if `op` has no formula — `Option<Vec<_>>`'s `FromIterator` impl
/// short-circuits the whole collection to `None` on the first element
/// [`ref_unary`] cannot compute, rather than silently passing that element
/// through unchanged.
#[must_use]
pub fn ref_unary_vec(op: &OpKind, data: &[f32]) -> Option<Vec<f32>> {
    data.iter().map(|&x| ref_unary(op, x)).collect()
}

/// The elementwise binary formula for `Add` / `Sub` / `Mul` / `Div`.
///
/// Returns `None` for any other `op`.
#[must_use]
pub fn ref_binary(op: &OpKind, a: f32, b: f32) -> Option<f32> {
    match op {
        OpKind::Add => Some(a + b),
        OpKind::Sub => Some(a - b),
        OpKind::Mul => Some(a * b),
        OpKind::Div => Some(a / b),
        _ => None,
    }
}

/// Map `ref_binary` over two equal-length operand slices.
///
/// `None` if the lengths disagree or `op` has no formula.
#[must_use]
pub fn ref_binary_vec(op: &OpKind, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    if a.len() != b.len() {
        return None;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ref_binary(op, x, y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── verify_enabled / parse_env_flag plumbing ───────────────────────────

    #[test]
    fn verify_enabled_never_panics() {
        let _ = verify_enabled();
    }

    // ── nearly_eq / compare ─────────────────────────────────────────────────

    #[test]
    fn nearly_eq_accepts_exact_and_small_rounding_drift() {
        assert!(nearly_eq(1.0, 1.0));
        assert!(nearly_eq(0.0, 0.0));
        assert!(nearly_eq(1.000_05, 1.0)); // well within ATOL.
        assert!(nearly_eq(1000.5, 1000.0)); // within RTOL of a large magnitude.
    }

    #[test]
    fn nearly_eq_rejects_a_genuine_mismatch() {
        assert!(!nearly_eq(1.0, 2.0));
        assert!(!nearly_eq(0.0, 1.0));
    }

    #[test]
    fn nearly_eq_treats_matching_non_finite_values_as_agreement() {
        assert!(nearly_eq(f32::NAN, f32::NAN));
        assert!(nearly_eq(f32::INFINITY, f32::INFINITY));
        assert!(nearly_eq(f32::NEG_INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn nearly_eq_rejects_mismatched_non_finite_values() {
        assert!(!nearly_eq(f32::NAN, 1.0));
        assert!(!nearly_eq(1.0, f32::NAN));
        assert!(!nearly_eq(f32::INFINITY, f32::NEG_INFINITY));
        assert!(!nearly_eq(f32::INFINITY, 1.0e30));
    }

    #[test]
    fn compare_reports_the_first_mismatched_index() {
        let gpu = [1.0_f32, 2.0, 99.0, 4.0];
        let cpu = [1.0_f32, 2.0, 3.0, 4.0];
        let err = compare(&gpu, &cpu).unwrap_err();
        assert!(err.contains("element 2"), "got: {err}");
        assert!(err.contains("99"), "got: {err}");
    }

    #[test]
    fn compare_reports_a_length_mismatch_distinctly() {
        let err = compare(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(err.contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn compare_passes_identical_slices() {
        let v = [1.0_f32, -2.5, 0.0, 3.75];
        assert!(compare(&v, &v).is_ok());
    }

    // ── ref_matmul ───────────────────────────────────────────────────────────

    #[test]
    fn ref_matmul_identity() {
        // [[1,0],[0,1]] @ [[5,6],[7,8]] = [[5,6],[7,8]].
        let a = [1.0_f32, 0.0, 0.0, 1.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        assert_eq!(ref_matmul(&a, &b, 2, 2, 2), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn ref_matmul_hand_verified_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2).
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [7+18+33, 8+20+36] = [58, 64]
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [28+45+66, 32+50+72] = [139, 154]
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(ref_matmul(&a, &b, 2, 3, 2), vec![58.0, 64.0, 139.0, 154.0]);
    }

    // ── ref_conv ─────────────────────────────────────────────────────────────
    //
    // Every expected output below was hand-derived AND independently
    // cross-checked with a from-scratch Python re-implementation of the same
    // naive algorithm before being pasted in here, catching (during that
    // cross-check) one hand-arithmetic mistake of my own on a discarded
    // extra case -- a reminder of exactly why this whole module exists.

    fn unit_params() -> ConvParams {
        ConvParams {
            strides: [1, 1],
            pads: [0, 0, 0, 0],
            dilations: [1, 1],
            group: 1,
        }
    }

    #[test]
    fn ref_conv_is_cross_correlation_not_true_convolution() {
        // input (3x3): [[1,2,3],[4,5,6],[7,8,9]], weight (2x2): [[1,2],[3,4]].
        // Deliberately asymmetric so a 180-degree kernel flip would change
        // the answer -- proving this computes cross-correlation (the ONNX
        // `Conv` / cuDNN convention), not textbook convolution.
        //   out[0][0] = 1*1+2*2+4*3+5*4 = 1+4+12+20 = 37
        //   out[0][1] = 2*1+3*2+5*3+6*4 = 2+6+15+24 = 47
        //   out[1][0] = 4*1+5*2+7*3+8*4 = 4+10+21+32 = 67
        //   out[1][1] = 5*1+6*2+8*3+9*4 = 5+12+24+36 = 77
        // (a flipped kernel [[4,3],[2,1]] would instead give 23 at [0][0].)
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = [1.0_f32, 2.0, 3.0, 4.0];
        let out = ref_conv(
            &input,
            &weight,
            None,
            &[1, 1, 3, 3],
            &[1, 1, 2, 2],
            &unit_params(),
        );
        assert_eq!(out, vec![37.0, 47.0, 67.0, 77.0]);
    }

    #[test]
    fn ref_conv_adds_bias_once_per_output_channel() {
        // Same geometry as above, plus bias=[100] on the single output channel:
        // every element of the no-bias result shifts by exactly 100.
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = [1.0_f32, 2.0, 3.0, 4.0];
        let bias = [100.0_f32];
        let out = ref_conv(
            &input,
            &weight,
            Some(&bias),
            &[1, 1, 3, 3],
            &[1, 1, 2, 2],
            &unit_params(),
        );
        assert_eq!(out, vec![137.0, 147.0, 167.0, 177.0]);
    }

    #[test]
    fn ref_conv_honours_stride() {
        // 5x5 input (row-major 1..25), 2x2 kernel [[1,0],[0,1]], stride=2, no
        // padding: out[oh][ow] = in[2*oh][2*ow] + in[2*oh+1][2*ow+1].
        //   out[0][0] = in[0][0]+in[1][1] =  1+ 7 =  8
        //   out[0][1] = in[0][2]+in[1][3] =  3+ 9 = 12
        //   out[1][0] = in[2][0]+in[3][1] = 11+17 = 28
        //   out[1][1] = in[2][2]+in[3][3] = 13+19 = 32
        let input: Vec<f32> = (1..=25).map(|v| v as f32).collect();
        let weight = [1.0_f32, 0.0, 0.0, 1.0];
        let mut params = unit_params();
        params.strides = [2, 2];
        let out = ref_conv(&input, &weight, None, &[1, 1, 5, 5], &[1, 1, 2, 2], &params);
        assert_eq!(out, vec![8.0, 12.0, 28.0, 32.0]);
    }

    #[test]
    fn ref_conv_zero_pads_the_border_with_correct_offset_direction() {
        // 2x2 input [[1,2],[3,4]], 3x3 kernel that is a one-hot at tap
        // (r=0, s=0) (all other taps zero), pad=1, stride=1 ("same" output
        // size, 2x2). With `ih = oh - 1 + 0` / `iw = ow - 1 + 0`, the
        // top-left tap only ever lands inside the real 2x2 image (rather
        // than the zero border) when oh=ow=1, reading `input[0][0]=1`;
        // every other output position reads purely padding, i.e. 0. This
        // pins down both that out-of-range taps contribute zero AND that
        // `pad` is subtracted (not added) when computing the input
        // coordinate -- a sign error here would shift which corner is
        // nonzero instead of merely failing everywhere.
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        #[rustfmt::skip]
        let weight = [
            1.0_f32, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let mut params = unit_params();
        params.pads = [1, 1, 1, 1];
        let out = ref_conv(&input, &weight, None, &[1, 1, 2, 2], &[1, 1, 3, 3], &params);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn ref_conv_honours_dilation() {
        // Same 5x5 input as the stride test, 2x2 kernel [[1,0],[0,1]],
        // dilation=2 (effective kernel size 3), stride=1, no padding -> 3x3
        // output. out[oh][ow] = in[oh][ow] + in[oh+2][ow+2] (the dilated tap
        // skips one row/col instead of the stride test's adjacent one):
        //   out[0][0]=in[0][0]+in[2][2]= 1+13=14   out[0][1]= 2+14=16   out[0][2]= 3+15=18
        //   out[1][0]=in[1][0]+in[3][2]= 6+18=24   out[1][1]= 7+19=26   out[1][2]= 8+20=28
        //   out[2][0]=in[2][0]+in[4][2]=11+23=34   out[2][1]=12+24=36   out[2][2]=13+25=38
        let input: Vec<f32> = (1..=25).map(|v| v as f32).collect();
        let weight = [1.0_f32, 0.0, 0.0, 1.0];
        let mut params = unit_params();
        params.dilations = [2, 2];
        let out = ref_conv(&input, &weight, None, &[1, 1, 5, 5], &[1, 1, 2, 2], &params);
        assert_eq!(
            out,
            vec![14.0, 16.0, 18.0, 24.0, 26.0, 28.0, 34.0, 36.0, 38.0]
        );
    }

    #[test]
    fn ref_conv_honours_groups_keeping_each_groups_channels_independent() {
        // 1x1 spatial, in_channels=4, out_channels=4, groups=2: group 0 owns
        // input channels [0,1] and output channels [0,1]; group 1 owns input
        // channels [2,3] and output channels [2,3]. input=[1,2,3,4],
        // weight[k][cg] = k0:[1,1] k1:[2,2] k2:[3,3] k3:[4,4].
        //   out[0] = in[0]*1+in[1]*1 = 1+2 =  3   (group 0, sees channels 0,1)
        //   out[1] = in[0]*2+in[1]*2 = 2+4 =  6   (group 0, sees channels 0,1)
        //   out[2] = in[2]*3+in[3]*3 = 9+12= 21   (group 1, sees channels 2,3)
        //   out[3] = in[2]*4+in[3]*4 =12+16= 28   (group 1, sees channels 2,3)
        // A broken group offset (e.g. every output channel reading input
        // channels [0,1] regardless of its group) would instead give
        // [3, 6, 9, 12] for out[2..4] -- a different, wrong answer, so this
        // discriminates a real groups bug rather than passing vacuously.
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let mut params = unit_params();
        params.group = 2;
        let out = ref_conv(&input, &weight, None, &[1, 4, 1, 1], &[4, 2, 1, 1], &params);
        assert_eq!(out, vec![3.0, 6.0, 21.0, 28.0]);
    }

    // ── ref_reduce ───────────────────────────────────────────────────────────

    #[test]
    fn ref_reduce_sum_whole_1d_tensor_over_256_elements() {
        // The exact motivating shape from finding a8-1: a 1-D axis longer than the old
        // 256-thread block size. data[i] = i, sum_{i=0}^{1023} i = 1023*1024/2 = 523776.
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let out = ref_reduce(&OpKind::ReduceSum, &data, &[1024], 0).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 523_776.0).abs() < 1.0);
    }

    #[test]
    fn ref_reduce_max_over_a_middle_axis() {
        // shape [2, 3, 2], axis=1 (outer=2, axis_len=3, inner=2).
        // data laid out row-major: [[[0,1],[2,3],[4,5]], [[6,7],[8,9],[10,11]]]
        // max over axis 1 -> [[4,5],[10,11]]
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = ref_reduce(&OpKind::ReduceMax, &data, &[2, 3, 2], 1).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn ref_reduce_declines_out_of_range_axis_and_unknown_op() {
        assert_eq!(ref_reduce(&OpKind::ReduceSum, &[1.0, 2.0], &[2], 5), None);
        assert_eq!(ref_reduce(&OpKind::Relu, &[1.0, 2.0], &[2], 0), None);
    }

    // ── ref_softmax ──────────────────────────────────────────────────────────

    #[test]
    fn ref_softmax_uniform_row_is_uniform() {
        let out = ref_softmax(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        for v in out {
            assert!((v - 0.25).abs() < 1.0e-6, "got {v}");
        }
    }

    #[test]
    fn ref_softmax_rows_sum_to_one() {
        let out = ref_softmax(&[1.0, 2.0, 3.0, -1.0, 0.0, 5.0], &[2, 3]).unwrap();
        let row0: f32 = out[0..3].iter().sum();
        let row1: f32 = out[3..6].iter().sum();
        assert!((row0 - 1.0).abs() < 1.0e-6, "row0 sums to {row0}");
        assert!((row1 - 1.0).abs() < 1.0e-6, "row1 sums to {row1}");
    }

    #[test]
    fn ref_softmax_is_shift_invariant_and_does_not_overflow() {
        // Large inputs would overflow a naive exp() without the max-subtraction; this must
        // not produce NaN/Inf.
        let out = ref_softmax(&[1000.0, 1000.0, 1000.0], &[3]).unwrap();
        for v in &out {
            assert!(
                v.is_finite(),
                "softmax must stay finite on large shifted inputs"
            );
        }
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
    }

    // ── ref_unary: hand-verified constants ──────────────────────────────────

    #[test]
    fn ref_unary_relu() {
        assert_eq!(ref_unary(&OpKind::Relu, -3.0), Some(0.0));
        assert_eq!(ref_unary(&OpKind::Relu, 3.0), Some(3.0));
    }

    #[test]
    fn ref_unary_leaky_relu_matches_the_hard_coded_kernel_alpha() {
        // oxicuda_ptx::generate_leaky_relu hard-codes alpha=0.01 (0f3C23D70A).
        assert_eq!(ref_unary(&OpKind::LeakyRelu, 2.0), Some(2.0));
        let y = ref_unary(&OpKind::LeakyRelu, -2.0).unwrap();
        assert!((y - (-0.02)).abs() < 1.0e-6, "got {y}");
    }

    #[test]
    fn ref_unary_hard_sigmoid_matches_the_hard_coded_kernel_constants() {
        // clamp(0.2*x + 0.5, 0, 1): x=0 -> 0.5; x=-10 -> clamp(-1.5,0,1)=0; x=10 -> clamp(2.5,0,1)=1.
        assert!((ref_unary(&OpKind::HardSigmoid, 0.0).unwrap() - 0.5).abs() < 1.0e-6);
        assert_eq!(ref_unary(&OpKind::HardSigmoid, -10.0), Some(0.0));
        assert_eq!(ref_unary(&OpKind::HardSigmoid, 10.0), Some(1.0));
    }

    #[test]
    fn ref_unary_sigmoid_hand_verified_at_zero() {
        let y = ref_unary(&OpKind::Sigmoid, 0.0).unwrap();
        assert!((y - 0.5).abs() < 1.0e-6, "sigmoid(0) must be 0.5, got {y}");
    }

    #[test]
    fn ref_unary_gelu_is_the_tanh_approximation_not_the_exact_form() {
        // At x=0 both the exact erf-based GELU and the tanh approximation give exactly 0,
        // so this checks a nonzero point where they'd disagree if the formula were wrong.
        // GELU_tanh(1.0) = 0.5*1*(1+tanh(sqrt(2/pi)*(1+0.044715))) ~= 0.8411919906...
        let y = ref_unary(&OpKind::Gelu, 1.0).unwrap();
        assert!((y - 0.841_192).abs() < 1.0e-4, "got {y}");
    }

    #[test]
    fn ref_unary_softplus_hand_verified_at_zero() {
        // softplus(0) = ln(1 + e^0) = ln(2) ~= 0.693147.
        let y = ref_unary(&OpKind::Softplus, 0.0).unwrap();
        assert!((y - std::f32::consts::LN_2).abs() < 1.0e-5, "got {y}");
    }

    #[test]
    fn ref_unary_unknown_op_is_none() {
        assert_eq!(ref_unary(&OpKind::MatMul, 1.0), None);
    }

    #[test]
    fn ref_unary_vec_maps_every_element() {
        let out = ref_unary_vec(&OpKind::Neg, &[1.0, -2.0, 3.0]).unwrap();
        assert_eq!(out, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn ref_unary_vec_unknown_op_is_none() {
        assert_eq!(ref_unary_vec(&OpKind::MatMul, &[1.0, 2.0]), None);
    }

    // ── ref_binary ───────────────────────────────────────────────────────────

    #[test]
    fn ref_binary_all_four_ops() {
        assert_eq!(ref_binary(&OpKind::Add, 2.0, 3.0), Some(5.0));
        assert_eq!(ref_binary(&OpKind::Sub, 2.0, 3.0), Some(-1.0));
        assert_eq!(ref_binary(&OpKind::Mul, 2.0, 3.0), Some(6.0));
        assert_eq!(ref_binary(&OpKind::Div, 6.0, 3.0), Some(2.0));
    }

    #[test]
    fn ref_binary_unknown_op_is_none() {
        assert_eq!(ref_binary(&OpKind::MatMul, 1.0, 2.0), None);
    }

    #[test]
    fn ref_binary_vec_maps_pairwise() {
        let out = ref_binary_vec(&OpKind::Mul, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(out, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn ref_binary_vec_rejects_length_mismatch() {
        assert_eq!(ref_binary_vec(&OpKind::Add, &[1.0], &[1.0, 2.0]), None);
    }

    // ── shadow_verify: the FailurePolicy branching ──────────────────────────

    #[test]
    fn shadow_verify_skips_the_oracle_entirely_when_verify_is_off() {
        let called = std::cell::Cell::new(false);
        let result = shadow_verify("Test", &[1.0], false, FailurePolicy::Strict, || {
            called.set(true);
            Some(vec![999.0]) // would fail comparison if it were ever run
        });
        // `CudaDispatchError` deliberately does not derive `PartialEq` (it wraps an opaque
        // external driver error type), so `Result<bool, _>` is compared by pattern, not
        // `assert_eq!`, throughout this test group.
        assert!(matches!(result, Ok(true)), "got {result:?}");
        assert!(
            !called.get(),
            "the oracle closure must not run when verify_on is false"
        );
    }

    #[test]
    fn shadow_verify_passes_a_matching_result() {
        let gpu = [1.0_f32, 2.0, 3.0];
        let result = shadow_verify("Add", &gpu, true, FailurePolicy::Fallback, || {
            Some(vec![1.0, 2.0, 3.0])
        });
        assert!(matches!(result, Ok(true)), "got {result:?}");
    }

    #[test]
    fn shadow_verify_mismatch_under_fallback_discards_without_erroring() {
        let gpu = [1.0_f32, 2.0, 99.0];
        let result = shadow_verify("Add", &gpu, true, FailurePolicy::Fallback, || {
            Some(vec![1.0, 2.0, 3.0])
        });
        assert!(
            matches!(result, Ok(false)),
            "a mismatch under Fallback must be Ok(false) (discard GPU numbers, run on CPU), not \
             Err; got {result:?}"
        );
    }

    #[test]
    fn shadow_verify_mismatch_under_strict_is_a_hard_error() {
        let gpu = [1.0_f32, 2.0, 99.0];
        let result = shadow_verify("Add", &gpu, true, FailurePolicy::Strict, || {
            Some(vec![1.0, 2.0, 3.0])
        });
        match result {
            Err(CudaDispatchError::Verify(msg)) => {
                assert!(msg.contains("element 2"), "got: {msg}");
            }
            other => panic!("expected Err(Verify(_)) under Strict, got {other:?}"),
        }
    }

    #[test]
    fn shadow_verify_an_oracle_that_declines_is_ok_true_not_a_silent_pass_disguised_as_failure() {
        // The oracle returning `None` (no formula for this op) must never be treated as a
        // GPU failure, under either policy -- it is a gap in the oracle.
        for policy in [FailurePolicy::Fallback, FailurePolicy::Strict] {
            let result = shadow_verify("Unknown", &[1.0], true, policy, || None);
            assert!(
                matches!(result, Ok(true)),
                "policy {policy:?} got {result:?}"
            );
        }
    }
}
