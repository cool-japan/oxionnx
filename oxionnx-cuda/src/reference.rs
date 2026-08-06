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
