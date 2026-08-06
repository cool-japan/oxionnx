//! CPU oracle — the executable specification of what the GPU path must compute.
//!
//! Platform-neutral.  Every function here reproduces exactly what the corresponding
//! HLSL shader / DirectML operator computes: the same accumulation order, the same
//! f32 arithmetic, the same broadcast semantics.
//!
//! # What this is for — and what it is *not* for
//!
//! It is **not** the dispatch fallback.  [`crate::try_directml_dispatch`] returns
//! `Ok(None)` when a backend declines, so `oxionnx-ops`' tuned CPU kernel runs;
//! calling these naive loops instead would be strictly slower.  Its three real jobs are:
//!
//! 1. **Test oracle.**  On Linux we cannot run a shader, but we *can* assert that
//!    [`crate::plan`] + these functions produce the numerically correct answer for the
//!    exact inputs the GPU path would be handed.
//! 2. **The hardware gate.**  `DirectMLContext::self_check` runs the real GPU path on
//!    fixed inputs and diffs it against these functions.  That is the **only**
//!    mechanism that can validate this crate's Windows-only code, because this
//!    repository has no Windows host and no D3D12 GPU.  Ship it, run it on real
//!    hardware, paste the output into the PR.
//! 3. **Shadow verification.**  With [`VERIFY_ENV_VAR`] set, every dispatched op is
//!    recomputed here and the two results are compared by [`compare`].  That turns the
//!    first user with a real GPU into a consenting test harness — see [`verify_binary`],
//!    [`verify_unary`], [`verify_matmul`].
//!
//! If you change a shader in [`crate::hlsl`], change the matching function here in the
//! same commit, or the self-check will (correctly) fail.
//!
//! # The tolerance policy is the point. Do not flatten it.
//!
//! An oracle that is merely "a correct matmul" catches nothing.  What catches bugs is
//! knowing, per op, *exactly how much* the GPU is allowed to disagree — and refusing
//! anything more:
//!
//! | Op | Policy | Why |
//! |---|---|---|
//! | `Add`, `Sub`, `Mul` | [`Tolerance::Exact`] | D3D requires fp32 add/sub/mul to be correctly rounded (0.5 ULP), and these kernels are index-parallel with no accumulation, so there is no reassociation to hide behind. Any disagreement is a **bug**, not noise. |
//! | `Relu` | [`Tolerance::Exact`] | `max(0, x)` selects an operand; it does no arithmetic at all. |
//! | `Div` | [`Tolerance::Approx`], ~1 ULP | D3D only requires fp32 divide to be within **1.0 ULP**, not correctly rounded. A bit-exact assertion here would fail on conforming hardware. |
//! | `Sigmoid`, `Tanh` | [`Tolerance::Approx`], ~1e-6 | `exp` is a hardware approximation, `tanh` is expanded by the shader compiler into one, and DirectML may use a different polynomial entirely. |
//! | `MatMul`, `Gemm` | [`Tolerance::Approx`], scaled by `sqrt(K)` | The GPU accumulates a length-`K` dot product with mad-contraction and may flush denormals; we accumulate the same order without contraction. The drift is real, bounded, and grows with `K`. |
//!
//! Using one loose tolerance everywhere would hide precisely the class of bug this
//! module exists to catch: an `Add` kernel that reads the wrong element still lands
//! within 1e-3 of the right answer surprisingly often.

use core::fmt;
use std::sync::OnceLock;

use crate::backend::BackendKind;
use crate::error::{DirectMLError, Result};
use crate::plan::{
    apply_gemm_epilogue, broadcast_expand, numel, transpose_2d, BinaryOp, ElementwisePlan,
    MatMulPlan, UnaryOp,
};

/// Wave-4 neural-network oracles (`Softmax`, `Reduce`, `Conv`), kept in a child module
/// so this file stays under the 2000-line refactor ceiling.  Everything it exports is
/// re-exported here, so the public contract is `reference::ref_softmax`, not
/// `reference::nn::ref_softmax`.
pub mod nn;

pub use nn::{ref_conv, ref_reduce, ref_softmax, verify_conv, verify_reduce, verify_softmax};

// ─── scalar kernels: one Rust line per HLSL line ─────────────────────────────

/// `max(0.0, x)` — [`crate::hlsl::ELEMENTWISE_UNARY_HLSL`]'s `main_relu`.
///
/// # The two pinned edge cases
///
/// HLSL's `max` and Rust's [`f32::max`] are *not* the same function, and neither is
/// fully specified on these inputs, so this oracle **chooses** — and
/// `relu_pins_its_nan_and_negative_zero_behaviour` locks the choice in:
///
/// * **`relu(NaN) == 0.0`.**  Rust's `f32::max` returns the non-NaN operand (IEEE-754
///   `maxNum`), so `0.0f32.max(NaN)` is `0.0`; HLSL's `max` is implementation-defined
///   on NaN and real drivers both propagate it and swallow it.  We take the `maxNum`
///   answer because it is what `oxionnx-ops`' CPU `relu` (`v.max(0.0)`) also produces,
///   so the oracle agrees with the kernel it is cross-validated against.  A GPU that
///   propagates the NaN instead will be reported as a mismatch by [`compare`] — that is
///   correct behaviour for a verifier, and a NaN reaching a `Relu` is a broken model in
///   the first place.
/// * **`relu(-0.0) == +0.0`.**  LLVM's `maxnum` may return *either* zero, so we do not
///   route through `f32::max` at all; the sign is pinned here.  This is invisible to
///   [`compare`] (`-0.0 == 0.0`), and matters only to anyone comparing bit patterns.
#[must_use]
pub fn relu(x: f32) -> f32 {
    // Deliberately not `0.0f32.max(x)`: that leaves the sign of a zero result up to
    // LLVM.  `x > 0.0` is false for NaN, for -0.0 and for +0.0, so all three fall
    // through to a positive zero.
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// `1 / (1 + exp(-x))` — [`crate::hlsl::ELEMENTWISE_UNARY_HLSL`]'s `main_sigmoid`.
///
/// This is the **direct** form, not the numerically-stable two-branch one
/// (`x >= 0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x))`).  Parity with the shader is the whole
/// point of this module, and the direct form needs no stabilisation anyway: `exp`
/// saturates to `+inf` below about `-88`, and `1/(1+inf)` is `0.0`, not a NaN.
/// `sigmoid_saturates_cleanly_and_never_produces_nan` pins that.
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `tanh(x)` — [`crate::hlsl::ELEMENTWISE_UNARY_HLSL`]'s `main_tanh`.
///
/// Note that HLSL's `tanh` is not a DXIL intrinsic: the shader compiler expands it into
/// an `exp`-based approximation, and DirectML's `DML_OPERATOR_ACTIVATION_TANH` is free
/// to use yet another one.  This is `libm`'s correctly-rounded-ish `tanh`, which is the
/// *reference*, not a bit-for-bit prediction — hence [`Tolerance::for_unary`] gives
/// `Tanh` an approximate tolerance and not an exact one.
#[must_use]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// One element of a binary kernel: `C[i] = A[i] ⊕ B[i]`.
///
/// `Div` by zero yields `±inf` (or `NaN` for `0/0`), exactly as the shader does — HLSL
/// float division is IEEE and does not trap.
#[must_use]
pub fn apply_binary(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
    }
}

/// One element of a unary kernel: `C[i] = f(A[i])`.
#[must_use]
pub fn apply_unary(op: UnaryOp, x: f32) -> f32 {
    match op {
        UnaryOp::Relu => relu(x),
        UnaryOp::Sigmoid => sigmoid(x),
        UnaryOp::Tanh => tanh(x),
    }
}

// ─── the oracle ──────────────────────────────────────────────────────────────

/// `Y = alpha · op(A) · op(B) + beta · C`, batched, with numpy broadcasting on `C`.
///
/// # Fidelity
///
/// This is a transcription of [`crate::hlsl::MATMUL_HLSL`] plus the HLSL backend's CPU
/// epilogue, not an independent implementation:
///
/// 1. **Transposes** are materialised with [`transpose_2d`] first, exactly as the HLSL
///    backend does when [`MatMulPlan::needs_cpu_transpose`] is true.  (The DirectML
///    backend instead sets `DML_GEMM_OPERATOR_DESC::TransA`/`TransB` and transposes
///    on-device; the *result* is the same, the rounding may not be, which is why
///    MatMul gets an approximate tolerance.)
/// 2. **Offsets** come from [`MatMulPlan::constants_for_slice`] — the same `u32`s that
///    are pushed into the shader's root constants.  Nothing here recomputes them, so a
///    batch-broadcast operand (`a_batch_stride == 0`) is read from offset 0 for every
///    slice, exactly as on the GPU.
/// 3. **Accumulation** is a sequential `k`-major `acc += A[…] * B[…]` in f32, in the
///    shader's order.  It is *not* an f64 accumulation and *not* a blocked/pairwise
///    sum — `accumulates_in_the_shaders_k_major_order` pins that with a case where the
///    two disagree.  This is what makes the oracle a specification of the shader rather
///    than merely of the mathematics.
/// 4. **`alpha` / `beta`** are applied by [`apply_gemm_epilogue`], the same function the
///    HLSL backend calls.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when a buffer does not match its planned shape.
/// [`DirectMLError::Declined`] when the plan asks for a transpose of an operand that is
/// not 2-D, or when an index computation overflows `usize`.
pub fn ref_matmul(plan: &MatMulPlan, a: &[f32], b: &[f32], c: Option<&[f32]>) -> Result<Vec<f32>> {
    check_len("MatMul A", a.len(), numel(&plan.a_stored_shape)?)?;
    check_len("MatMul B", b.len(), numel(&plan.b_stored_shape)?)?;

    // Step 1 — the HLSL backend's CPU pre-transpose.
    let a_eff = maybe_transpose(a, &plan.a_stored_shape, plan.trans_a, "A")?;
    let b_eff = maybe_transpose(b, &plan.b_stored_shape, plan.trans_b, "B")?;

    let m = plan.m as usize;
    let k = plan.k as usize;
    let n = plan.n as usize;
    let a_slice_elems = mul("M * K", m, k)?;
    let b_slice_elems = mul("K * N", k, n)?;
    let c_slice_elems = mul("M * N", m, n)?;

    let mut out = vec![0.0f32; plan.output_elems()?];

    for slice in 0..plan.batch {
        // Step 2 — the shader's own offsets, not a re-derivation of them.
        let consts = plan.constants_for_slice(slice)?;
        let a_lo = consts.a_offset as usize;
        let b_lo = consts.b_offset as usize;
        let c_lo = consts.c_offset as usize;

        let a_mat = window(&a_eff, a_lo, a_slice_elems, "A", slice)?;
        let b_mat = window(&b_eff, b_lo, b_slice_elems, "B", slice)?;
        let c_hi = add("C slice end", c_lo, c_slice_elems)?;
        let c_mat = out
            .get_mut(c_lo..c_hi)
            .ok_or_else(|| out_of_range("output", slice, c_lo, c_slice_elems))?;

        // Step 3 — the shader's loop, verbatim.  `row` is `tid.y`, `col` is `tid.x`.
        for row in 0..m {
            // `row < m` and `k` elements follow, so this row lies inside `a_mat`
            // (`len == m * k`) by construction.
            let a_row = &a_mat[row * k..row * k + k];
            for col in 0..n {
                let mut acc = 0.0f32;
                for (kk, &a_v) in a_row.iter().enumerate() {
                    // `kk < k` and `col < n`, so `kk * n + col < k * n == b_mat.len()`.
                    acc += a_v * b_mat[kk * n + col];
                }
                c_mat[row * n + col] = acc;
            }
        }
    }

    // Step 4 — `alpha` and `beta * C`, through the backend's own epilogue.
    apply_gemm_epilogue(plan, &mut out, c)?;
    Ok(out)
}

/// Broadcasting binary elementwise: `C[i] = A[i] ⊕ B[i]` over the plan's output shape.
///
/// The shaders are index-parallel and have no notion of a shape, so any operand that is
/// smaller than the output is densely expanded first with [`broadcast_expand`] — the
/// same function the HLSL backend would use.  (Today
/// [`ElementwisePlan::binary`] declines every non-identical shape pair, so the expansion
/// is a no-op `Cow::Borrowed` on every plan the router can actually produce.  The oracle
/// still implements the general case, because it is the *specification*: when a later
/// wave lifts that restriction, the answer it must match is already written down here
/// and tested.)
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when `plan` is a unary plan, when a buffer does not
/// match its planned shape, or when the operands do not broadcast to the output shape.
pub fn ref_binary(plan: &ElementwisePlan, op: BinaryOp, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    let b_shape = plan.b_shape.as_ref().ok_or_else(|| {
        DirectMLError::ShapeMismatch(format!(
            "ref_binary({}): the plan has no B operand — it is a unary plan",
            op.as_str()
        ))
    })?;

    let a_dense = broadcast_expand(a, &plan.a_shape, &plan.output_shape)?;
    let b_dense = broadcast_expand(b, b_shape, &plan.output_shape)?;
    let expected = plan.elem_count as usize;
    check_len("binary A (expanded)", a_dense.len(), expected)?;
    check_len("binary B (expanded)", b_dense.len(), expected)?;

    Ok(a_dense
        .iter()
        .zip(b_dense.iter())
        .map(|(&x, &y)| apply_binary(op, x, y))
        .collect())
}

/// Unary elementwise: `C[i] = f(A[i])`.
///
/// See [`relu`], [`sigmoid`] and [`tanh`] for the exact per-element semantics and for
/// the NaN / signed-zero decisions this oracle pins.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when `a` does not match its planned shape.
pub fn ref_unary(plan: &ElementwisePlan, op: UnaryOp, a: &[f32]) -> Result<Vec<f32>> {
    let a_dense = broadcast_expand(a, &plan.a_shape, &plan.output_shape)?;
    check_len("unary A", a_dense.len(), plan.elem_count as usize)?;
    Ok(a_dense.iter().map(|&x| apply_unary(op, x)).collect())
}

/// `"MatMul"` for a plain product, `"Gemm"` when any Gemm attribute is in play.
///
/// The op name a [`ComparisonReport`] carries has to be a `&'static str` (it ends up in
/// [`SelfCheckReport::deviations`]), and a [`MatMulPlan`] does not record which ONNX
/// node it came from — so it is recovered from the attributes that only `Gemm` can set.
#[must_use]
pub fn matmul_op_name(plan: &MatMulPlan) -> &'static str {
    if plan.alpha == 1.0 && plan.beta == 0.0 && !plan.trans_a && !plan.trans_b {
        "MatMul"
    } else {
        "Gemm"
    }
}

// ─── tolerance policy ────────────────────────────────────────────────────────

/// Relative tolerance for `Div`: D3D requires fp32 divide to be within **1.0 ULP**, not
/// correctly rounded, so a bit-exact assertion would fail on *conforming* hardware.
/// Four ULPs of headroom (`4 · 2⁻²³ ≈ 4.8e-7`) covers the requirement plus a
/// reciprocal-and-multiply implementation.
pub const DIV_REL_TOLERANCE: f32 = 4.0 * f32::EPSILON;

/// Relative tolerance for `Sigmoid` and `Tanh`.  `exp` is a hardware approximation,
/// HLSL's `tanh` is expanded by the compiler into another one, and DirectML may use a
/// third; a handful of ULPs is expected and a *departure of 1e-3 is not*.
pub const TRANSCENDENTAL_REL_TOLERANCE: f32 = 1.0e-6;

/// Absolute floor for `Sigmoid` and `Tanh`, so that a result of ~1e-30 (where the
/// relative error of two different approximations is meaningless) is not flagged.
pub const TRANSCENDENTAL_ABS_TOLERANCE: f32 = 1.0e-7;

/// ULP budget per accumulation step of a MatMul, before the `sqrt(K)` scaling in
/// [`Tolerance::for_matmul`].
///
/// The GPU is permitted to contract `acc += a * b` into a single `mad` (one rounding
/// instead of two) and to flush denormal intermediates to zero; we do neither.  The
/// per-step divergence is therefore a couple of ULPs, and a length-`K` sum of
/// independent roundings grows like `sqrt(K)`, not `K`.
pub const MATMUL_ULP_BUDGET: f32 = 4.0;

/// How much a GPU result is allowed to differ from the oracle, for one op.
///
/// Pick it with [`Tolerance::for_binary`], [`Tolerance::for_unary`] or
/// [`Tolerance::for_matmul`] — never by hand at a call site, or the policy stops being a
/// policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    /// Every element must be **equal**, with exactly two escape hatches, both of which
    /// are properties of the hardware and not of the kernel:
    ///
    /// * `NaN ≡ NaN` — a GPU NaN carries a different payload, and `NaN != NaN` anyway.
    /// * **denormal ≡ zero** — D3D permits fp32 denormals to be flushed to zero, so an
    ///   oracle result of `1e-40` against a GPU result of `0.0` is *conforming*.  Both
    ///   sides must be zero-or-denormal for this to apply; a denormal against a normal
    ///   number is still a failure.
    ///
    /// Anything else — a single ULP — is a **bug**.  `Add`, `Sub`, `Mul` and `Relu` are
    /// held to this, because D3D requires fp32 add/sub/mul to be correctly rounded and
    /// these kernels do no accumulation: there is no legitimate source of drift.
    Exact,
    /// `|gpu - cpu| <= abs + rel * |cpu|`.
    Approx {
        /// Relative term, scaled by the magnitude of the oracle's value.
        rel: f32,
        /// Absolute floor.  Absorbs the case where the oracle's value is ~0 through
        /// cancellation while the summands were ~1 — the regime every normalised
        /// activation lives in.  If your tensors are scaled to ~1e6 this floor is far
        /// too tight and verify mode will complain; that is the intended direction of
        /// error, since a false positive costs a log line and a missed mismatch costs a
        /// wrong inference.
        abs: f32,
    },
}

impl Tolerance {
    /// The policy for a binary elementwise op.
    ///
    /// `Add`/`Sub`/`Mul` are [`Tolerance::Exact`]; `Div` is not — see
    /// [`DIV_REL_TOLERANCE`].
    #[must_use]
    pub fn for_binary(op: BinaryOp) -> Self {
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => Self::Exact,
            BinaryOp::Div => Self::Approx {
                rel: DIV_REL_TOLERANCE,
                abs: 0.0,
            },
        }
    }

    /// The policy for a unary elementwise op.
    ///
    /// `Relu` is [`Tolerance::Exact`] — it selects an operand and does no arithmetic.
    /// `Sigmoid` and `Tanh` are transcendental approximations on both sides.
    #[must_use]
    pub fn for_unary(op: UnaryOp) -> Self {
        match op {
            UnaryOp::Relu => Self::Exact,
            UnaryOp::Sigmoid | UnaryOp::Tanh => Self::Approx {
                rel: TRANSCENDENTAL_REL_TOLERANCE,
                abs: TRANSCENDENTAL_ABS_TOLERANCE,
            },
        }
    }

    /// The policy for a MatMul / Gemm, scaled by the inner dimension.
    ///
    /// `rel = MATMUL_ULP_BUDGET · f32::EPSILON · sqrt(K)`, and the absolute floor is the
    /// same number.  A `K = 3` product is held to ~8e-7; a `K = 4096` product to ~3e-5.
    /// Scaling matters: a fixed 1e-5 would be far too loose for a 3-term dot product
    /// (hiding a genuinely wrong `B` column) and too tight for a 4096-term one (failing
    /// on correct hardware).
    #[must_use]
    pub fn for_matmul(plan: &MatMulPlan) -> Self {
        let k = f64::from(plan.k.max(1)).sqrt();
        let rel = MATMUL_ULP_BUDGET * f32::EPSILON * (k as f32);
        Self::Approx { rel, abs: rel }
    }

    /// Does this tolerance accept `gpu` where the oracle produced `cpu`?
    #[must_use]
    pub fn accepts(self, gpu: f32, cpu: f32) -> bool {
        if let Some(agreement) = non_finite_agreement(gpu, cpu) {
            return agreement;
        }
        match self {
            Self::Exact => gpu == cpu || (is_flushable(gpu) && is_flushable(cpu)),
            Self::Approx { rel, abs } => (gpu - cpu).abs() <= abs + rel * cpu.abs(),
        }
    }
}

impl fmt::Display for Tolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => write!(f, "exact"),
            Self::Approx { rel, abs } => write!(f, "|d| <= {abs:e} + {rel:e} * |cpu|"),
        }
    }
}

/// `true` for a value D3D is allowed to have flushed to zero: `±0` and every denormal.
fn is_flushable(v: f32) -> bool {
    v == 0.0 || v.abs() < f32::MIN_POSITIVE
}

/// `Some(agree)` when at least one of the two values is non-finite, `None` when both are
/// finite and ordinary arithmetic applies.
///
/// * `NaN` vs `NaN` → agree (payloads differ between vendors; `NaN != NaN` regardless).
/// * `NaN` vs a number → disagree.
/// * `±inf` vs the same `±inf` → agree; against anything else → disagree.
fn non_finite_agreement(gpu: f32, cpu: f32) -> Option<bool> {
    if gpu.is_nan() || cpu.is_nan() {
        return Some(gpu.is_nan() && cpu.is_nan());
    }
    if gpu.is_infinite() || cpu.is_infinite() {
        return Some(gpu == cpu);
    }
    None
}

// ─── comparison ──────────────────────────────────────────────────────────────

/// One element's disagreement between the GPU and the oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deviation {
    /// Linear index into the output buffer.  This is the index the shader's
    /// `i = (gid.y * GroupsX + gid.x) * 256 + lid.x` (elementwise) or
    /// `COff + row * N + col` (matmul) computes, so it can be read straight back as a
    /// thread id.
    pub index: usize,
    /// What the GPU produced.
    pub gpu: f32,
    /// What the oracle produced.
    pub cpu: f32,
    /// `|gpu - cpu|`, or `f32::INFINITY` when exactly one of the two is non-finite.
    pub abs: f32,
    /// `|gpu - cpu| / |cpu|`.
    ///
    /// `0.0` when they agree exactly, and `f32::INFINITY` when the oracle produced `0.0`
    /// and the GPU did not.  **An infinite relative deviation is not by itself a
    /// failure** — the [`Tolerance`]'s absolute floor is what decides that, and a
    /// cancelling dot product legitimately lands on an exact zero.
    pub rel: f32,
}

impl fmt::Display for Deviation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] gpu={:e} cpu={:e} |d|={:e} rel={:e}",
            self.index, self.gpu, self.cpu, self.abs, self.rel
        )
    }
}

/// The result of shadow-comparing one op's GPU output against the oracle.
///
/// This is what [`VERIFY_ENV_VAR`] mode reports.  It deliberately is **not** a `bool`: a
/// bare "mismatch" tells a user with a GPU we cannot reproduce nothing at all, whereas
/// `Add: MISMATCH — 4096 element(s) … worst |d| 3.2e1 at index 2047` localises the bug to
/// a thread index in one line.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonReport {
    /// The op that was compared, e.g. `"Add"`, `"Relu"`, `"MatMul"`.
    pub op: &'static str,
    /// Number of elements compared.
    pub elem_count: usize,
    /// The policy that was applied.
    pub tolerance: Tolerance,
    /// How many elements exceeded [`Self::tolerance`].
    pub mismatches: usize,
    /// The element with the largest absolute deviation, over **all** elements — present
    /// even when the comparison passed, because "passed, worst drift 1e-7" and "passed,
    /// worst drift 9e-6" are very different things to see in a log.
    pub worst_abs: Option<Deviation>,
    /// The element with the largest relative deviation, over all elements.
    pub worst_rel: Option<Deviation>,
    /// The **first** element that exceeded the tolerance, in linear index order.  This
    /// is usually the most diagnostic single number in the report: a first mismatch at
    /// index 256 says "the second thread group is wrong", and at index `N/2` says "half
    /// the dispatch grid never ran".
    pub first_mismatch: Option<Deviation>,
    /// `true` when [`Self::mismatches`] is 0.
    pub passed: bool,
}

impl ComparisonReport {
    /// The largest absolute deviation over all elements, or `0.0` for an empty buffer.
    #[must_use]
    pub fn max_abs_deviation(&self) -> f32 {
        self.worst_abs.map_or(0.0, |d| d.abs)
    }

    /// The largest relative deviation over all elements, or `0.0` for an empty buffer.
    /// May be `f32::INFINITY`; see [`Deviation::rel`].
    #[must_use]
    pub fn max_rel_deviation(&self) -> f32 {
        self.worst_rel.map_or(0.0, |d| d.rel)
    }
}

impl fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} — {} element(s), tolerance {}",
            self.op,
            if self.passed { "OK" } else { "MISMATCH" },
            self.elem_count,
            self.tolerance
        )?;
        if let Some(worst) = self.worst_abs {
            write!(f, ", worst |d| {worst}")?;
        }
        if !self.passed {
            write!(f, ", {} mismatched", self.mismatches)?;
            if let Some(first) = self.first_mismatch {
                write!(f, ", first at {first}")?;
            }
        }
        Ok(())
    }
}

/// Compare a GPU buffer against an oracle buffer under `tolerance`.
///
/// This is the entry point [`VERIFY_ENV_VAR`] mode and `DirectMLContext::self_check`
/// both call.  It walks every element — the worst deviation is not knowable from a
/// prefix, and a kernel that computes the first thread group correctly and the rest as
/// garbage is a real and common failure mode.
///
/// # Errors
/// [`DirectMLError::ShapeMismatch`] when the two buffers are of different lengths.  A
/// GPU buffer of the wrong *length* is a structural failure, not a numerical one, and
/// reporting it as `passed: false` would understate it.
pub fn compare(
    op: &'static str,
    gpu: &[f32],
    oracle: &[f32],
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    if gpu.len() != oracle.len() {
        return Err(DirectMLError::ShapeMismatch(format!(
            "{op}: the GPU returned {} elements, the oracle {}",
            gpu.len(),
            oracle.len()
        )));
    }

    let mut mismatches = 0usize;
    let mut worst_abs: Option<Deviation> = None;
    let mut worst_rel: Option<Deviation> = None;
    let mut first_mismatch: Option<Deviation> = None;

    for (index, (&g, &c)) in gpu.iter().zip(oracle.iter()).enumerate() {
        let deviation = deviation_at(index, g, c);

        // `>` and not `>=`, so the *first* index wins a tie — a run of identical
        // deviations almost always starts at the interesting one.
        if worst_abs.map_or(true, |w| deviation.abs > w.abs) {
            worst_abs = Some(deviation);
        }
        if worst_rel.map_or(true, |w| deviation.rel > w.rel) {
            worst_rel = Some(deviation);
        }
        if !tolerance.accepts(g, c) {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(deviation);
            }
        }
    }

    Ok(ComparisonReport {
        op,
        elem_count: gpu.len(),
        tolerance,
        mismatches,
        worst_abs,
        worst_rel,
        first_mismatch,
        passed: mismatches == 0,
    })
}

/// The deviation of one element pair.  Non-finite disagreement is `INFINITY`, so it
/// dominates every `max` and cannot be quietly averaged away.
fn deviation_at(index: usize, gpu: f32, cpu: f32) -> Deviation {
    let (abs, rel) = match non_finite_agreement(gpu, cpu) {
        Some(true) => (0.0, 0.0),
        Some(false) => (f32::INFINITY, f32::INFINITY),
        None => {
            let abs = (gpu - cpu).abs();
            let rel = if abs == 0.0 {
                0.0
            } else if cpu == 0.0 {
                f32::INFINITY
            } else {
                abs / cpu.abs()
            };
            (abs, rel)
        }
    };
    Deviation {
        index,
        gpu,
        cpu,
        abs,
        rel,
    }
}

// ─── shadow verification (`OXIONNX_DIRECTML_VERIFY=1`) ───────────────────────

/// Set this environment variable to shadow-compare **every** dispatched op against the
/// oracle and report any mismatch.
///
/// ```text
/// OXIONNX_DIRECTML_VERIFY=1 my_inference_binary
/// ```
///
/// It roughly doubles the cost of every claimed node — the GPU result is computed *and*
/// the CPU one — so it is off by default and is meant for one run on a new machine, not
/// for production.  It exists because this repository has no Windows host and no D3D12
/// GPU: it is how a user with real hardware can tell us, in one line of output, that our
/// shaders are wrong.
pub const VERIFY_ENV_VAR: &str = "OXIONNX_DIRECTML_VERIFY";

/// Is shadow verification on?
///
/// Read once and cached: the value cannot change within a process, and this is called on
/// the hot path of every dispatched node.
///
/// `1`, `true`, `yes`, `on` (any case) enable it.  Unset, empty, `0`, `false`, `no` and
/// `off` disable it.  Anything else is treated as *enabled*, because a user who typed
/// `OXIONNX_DIRECTML_VERIFY=please` wants verification, and silently ignoring them would
/// hand back a false all-clear.
#[must_use]
pub fn verify_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_verify_flag(std::env::var(VERIFY_ENV_VAR).ok().as_deref()))
}

/// The pure core of [`verify_enabled`], so the policy can be tested without touching the
/// process environment (which is shared, racy under a threaded test runner, and cached
/// by the `OnceLock` above).
///
/// Delegates to [`crate::context::parse_env_flag`], which is the crate's single definition
/// of "truthy".  All three of `OXIONNX_DIRECTML`, `OXIONNX_DIRECTML_VERIFY` and
/// `OXIONNX_DIRECTML_STRICT` answer to exactly the same spellings; a user who learns one
/// has learned all three, and a user who writes `=please` gets the feature rather than a
/// silent no-op.
fn parse_verify_flag(value: Option<&str>) -> bool {
    crate::context::parse_env_flag(value)
}

/// Shadow-compare a GPU MatMul / Gemm result against the oracle.
///
/// # Errors
/// Whatever [`ref_matmul`] returns, plus [`DirectMLError::ShapeMismatch`] when `gpu` is
/// not `plan.output_elems()` long.
pub fn verify_matmul(
    plan: &MatMulPlan,
    a: &[f32],
    b: &[f32],
    c: Option<&[f32]>,
    gpu: &[f32],
) -> Result<ComparisonReport> {
    let oracle = ref_matmul(plan, a, b, c)?;
    compare(
        matmul_op_name(plan),
        gpu,
        &oracle,
        Tolerance::for_matmul(plan),
    )
}

/// Shadow-compare a GPU binary elementwise result against the oracle.
///
/// # Errors
/// Whatever [`ref_binary`] returns, plus [`DirectMLError::ShapeMismatch`] when `gpu` is
/// not `plan.elem_count` long.
pub fn verify_binary(
    plan: &ElementwisePlan,
    op: BinaryOp,
    a: &[f32],
    b: &[f32],
    gpu: &[f32],
) -> Result<ComparisonReport> {
    let oracle = ref_binary(plan, op, a, b)?;
    compare(op.as_str(), gpu, &oracle, Tolerance::for_binary(op))
}

/// Shadow-compare a GPU unary elementwise result against the oracle.
///
/// # Errors
/// Whatever [`ref_unary`] returns, plus [`DirectMLError::ShapeMismatch`] when `gpu` is
/// not `plan.elem_count` long.
pub fn verify_unary(
    plan: &ElementwisePlan,
    op: UnaryOp,
    a: &[f32],
    gpu: &[f32],
) -> Result<ComparisonReport> {
    let oracle = ref_unary(plan, op, a)?;
    compare(op.as_str(), gpu, &oracle, Tolerance::for_unary(op))
}

// ─── the self-check report ───────────────────────────────────────────────────

/// Result of `DirectMLContext::self_check` — the only report that can tell us whether
/// this crate's Windows code actually works.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfCheckReport {
    /// Which backend answered.
    pub backend: BackendKind,
    /// DXGI adapter description.
    pub adapter: String,
    /// Per-op maximum absolute deviation from the oracle, in dispatch order.
    pub deviations: Vec<(&'static str, f32)>,
    /// The tolerance that was applied.
    pub tolerance: f32,
    /// `true` when every deviation is finite and `<= tolerance`.
    pub passed: bool,
}

impl SelfCheckReport {
    /// An empty, so-far-passing report for `backend` on `adapter`.
    ///
    /// `tolerance` is the caller's blunt, single-number gate (the one
    /// `examples/directml_self_check.rs` takes on the command line).  It is applied *in
    /// addition to* the per-op policy in [`Tolerance`], never instead of it — see
    /// [`Self::record`].
    #[must_use]
    pub fn new(backend: BackendKind, adapter: String, tolerance: f32) -> Self {
        Self {
            backend,
            adapter,
            deviations: Vec::new(),
            tolerance,
            passed: true,
        }
    }

    /// Fold one op's [`ComparisonReport`] into this report.
    ///
    /// An op passes only if it satisfies **both** gates:
    ///
    /// * the per-op policy ([`ComparisonReport::passed`], from [`Tolerance`]), and
    /// * this report's blunt `tolerance`.
    ///
    /// The conjunction is deliberate.  Taking only the blunt gate would let a caller who
    /// passes `1e-3` sail past an `Add` kernel that is off in the sixth decimal — which
    /// is exactly the bug (a mis-indexed read of a neighbouring element) that the exact
    /// policy exists to catch.  Taking only the per-op policy would ignore what the
    /// caller asked for.  Both must hold.
    pub fn record(&mut self, comparison: &ComparisonReport) {
        let deviation = comparison.max_abs_deviation();
        self.deviations.push((comparison.op, deviation));
        self.passed = self.passed
            && comparison.passed
            && deviation.is_finite()
            && deviation <= self.tolerance;
    }

    /// The worst deviation across all ops, or `0.0` when none ran.
    ///
    /// `f32::NAN` if any op recorded a NaN deviation — a NaN is the worst possible
    /// outcome and must not be swallowed by `f32::max`, which returns the *other*
    /// operand when one is NaN.
    #[must_use]
    pub fn max_deviation(&self) -> f32 {
        let mut worst = 0.0f32;
        for &(_, deviation) in &self.deviations {
            if deviation.is_nan() {
                return f32::NAN;
            }
            if deviation > worst {
                worst = deviation;
            }
        }
        worst
    }
}

impl fmt::Display for SelfCheckReport {
    /// The one-screen summary a user pastes into a bug report.
    ///
    /// Everything a reader needs in order to act is here and nothing else is: **which**
    /// backend answered (DirectML and HLSL are different code paths with different bugs),
    /// **which adapter** (a bug that reproduces only on one IHV is the norm, not the
    /// exception, for a missing barrier), and **which op** drifted and by how much.
    ///
    /// # What this cannot tell you, and does not pretend to
    ///
    /// [`Self::deviations`] records a *magnitude* per op, not a verdict.  [`Self::passed`]
    /// is the conjunction of the blunt `tolerance` **and** the per-op [`Tolerance`] policy
    /// (`Add`/`Sub`/`Mul`/`Relu` are held to bit-exactness, whatever the caller passed),
    /// and that second gate is not recoverable from the numbers kept here.  So a report can
    /// legitimately read `FAIL` with every op inside `tolerance` — the `> tol` markers
    /// below flag only the *first* gate.  The footer says so rather than leaving a reader
    /// to conclude the report contradicts itself, and
    /// `DirectMLContext::self_check_reports` hands back the per-op [`ComparisonReport`]s
    /// that do carry the verdict.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DirectML self-check: {}",
            if self.passed { "PASS" } else { "FAIL" }
        )?;
        writeln!(f, "  backend:   {}", self.backend.as_str())?;
        writeln!(f, "  adapter:   {}", self.adapter)?;
        writeln!(f, "  tolerance: {:e}", self.tolerance)?;
        writeln!(f, "  worst |d|: {:e}", self.max_deviation())?;

        if self.deviations.is_empty() {
            // Not the same thing as a pass, and it must never read like one: zero ops ran.
            return writeln!(f, "  ops:       (none ran)");
        }

        writeln!(f, "  ops:")?;
        let mut over_tolerance = 0usize;
        for &(op, deviation) in &self.deviations {
            let within = deviation.is_finite() && deviation <= self.tolerance;
            if !within {
                over_tolerance += 1;
            }
            let marker = if within { "     " } else { "> tol" };
            writeln!(f, "    {op:<8} worst |d| {deviation:e} {marker}")?;
        }

        if !self.passed && over_tolerance == 0 {
            writeln!(
                f,
                "  note:      every op is within the tolerance you asked for, but at least one \
                 failed the\n             per-op policy (Add/Sub/Mul/Relu are held to \
                 bit-exactness).  Print the\n             ComparisonReports from \
                 self_check_reports() to see which, and where."
            )?;
        }
        Ok(())
    }
}

// ─── internal helpers ────────────────────────────────────────────────────────

/// `Err(ShapeMismatch)` unless `actual == expected`.
fn check_len(what: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(DirectMLError::ShapeMismatch(format!(
            "{what}: buffer of {actual} elements, expected {expected}"
        )))
    }
}

/// `a * b`, or a [`DirectMLError::Declined`] naming the product that overflowed.
fn mul(what: &str, a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| DirectMLError::Declined(format!("{what} overflows usize")))
}

/// `a + b`, or a [`DirectMLError::Declined`] naming the sum that overflowed.
fn add(what: &str, a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| DirectMLError::Declined(format!("{what} overflows usize")))
}

/// The HLSL backend's CPU pre-transpose, applied only when the plan asks for it.
///
/// Returns a borrowed slice on the overwhelmingly common untransposed path, so a
/// `MatMul` copies nothing.
fn maybe_transpose<'a>(
    src: &'a [f32],
    stored_shape: &[usize],
    transposed: bool,
    operand: &str,
) -> Result<std::borrow::Cow<'a, [f32]>> {
    if !transposed {
        return Ok(std::borrow::Cow::Borrowed(src));
    }
    // ONNX `Gemm` — the only op with `transA`/`transB` — is 2-D by definition, and
    // `transpose_2d` has no meaning for anything else.  Decline rather than guess at a
    // batched transpose the shader could not consume anyway.
    let [rows, cols] = stored_shape else {
        return Err(DirectMLError::Declined(format!(
            "MatMul {operand}: a transposed operand must be 2-D, got {stored_shape:?}"
        )));
    };
    Ok(std::borrow::Cow::Owned(transpose_2d(src, *rows, *cols)?))
}

/// `&buf[lo .. lo + len]`, as a [`DirectMLError::ShapeMismatch`] rather than a panic.
fn window<'a>(
    buf: &'a [f32],
    lo: usize,
    len: usize,
    operand: &str,
    slice: u32,
) -> Result<&'a [f32]> {
    let hi = add("slice end", lo, len)?;
    buf.get(lo..hi)
        .ok_or_else(|| out_of_range(operand, slice, lo, len))
}

/// The error a batch slice that runs off the end of its buffer produces.
fn out_of_range(operand: &str, slice: u32, lo: usize, len: usize) -> DirectMLError {
    DirectMLError::ShapeMismatch(format!(
        "MatMul {operand}: batch slice {slice} needs elements [{lo}..{}], which is past \
         the end of the buffer",
        lo + len
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // ── plan builders ────────────────────────────────────────────────────────
    //
    // `MatMulPlan::matmul` accepts 2-D x 2-D only and `ElementwisePlan::binary` accepts
    // identical shapes only, so a *batched* or *broadcasting* plan cannot be built
    // through the public constructors today.  The oracle is the specification of what
    // the GPU must compute once those restrictions lift, so these tests construct such
    // plans through the (fully public) struct literals.  Every field is filled exactly
    // as `plan.rs` documents it.

    /// `dims` is `[batch, m, k, n]`; `strides` is `[a_batch_stride, b_batch_stride]`,
    /// where a `0` means "this operand is batch-broadcast".
    fn batched_plan(
        dims: [u32; 4],
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        strides: [u32; 2],
    ) -> MatMulPlan {
        let [batch, m, k, n] = dims;
        let [a_batch_stride, b_batch_stride] = strides;
        MatMulPlan {
            m,
            k,
            n,
            batch,
            batch_shape: vec![batch as usize],
            output_shape: vec![batch as usize, m as usize, n as usize],
            a_batch_stride,
            b_batch_stride,
            a_stored_shape: a_shape,
            b_stored_shape: b_shape,
            c_shape: None,
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    fn broadcast_binary_plan(
        output_shape: Vec<usize>,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    ) -> ElementwisePlan {
        let elems: usize = output_shape.iter().product();
        ElementwisePlan {
            elem_count: u32::try_from(elems).expect("test shape fits u32"),
            a_needs_broadcast: a_shape != output_shape,
            b_needs_broadcast: b_shape != output_shape,
            output_shape,
            a_shape,
            b_shape: Some(b_shape),
        }
    }

    /// An independent naive batched matmul, written from the *mathematical* definition
    /// rather than derived from `ref_matmul`, so that agreement between the two means
    /// something.  It reads its dimensions off the plan (there is no other honest source
    /// for them) but shares no arithmetic with the oracle.
    fn naive_batched_matmul(a: &[f32], b: &[f32], plan: &MatMulPlan) -> Vec<f32> {
        let batch = plan.batch as usize;
        let m = plan.m as usize;
        let k = plan.k as usize;
        let n = plan.n as usize;
        let a_stride = plan.a_batch_stride as usize;
        let b_stride = plan.b_batch_stride as usize;

        let mut out = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[bi * a_stride + i * k + p] * b[bi * b_stride + p * n + j];
                    }
                    out[bi * m * n + i * n + j] = sum;
                }
            }
        }
        out
    }

    // ── ref_matmul ───────────────────────────────────────────────────────────

    #[test]
    fn matmul_2x3_by_3x2_matches_a_hand_computed_literal() {
        // A = [[1,2,3],        B = [[7, 8],
        //      [4,5,6]]             [9,10],
        //                           [11,12]]
        // A·B = [[1·7+2·9+3·11, 1·8+2·10+3·12],   = [[ 58,  64],
        //        [4·7+5·9+6·11, 4·8+5·10+6·12]]      [139, 154]]
        let plan = MatMulPlan::matmul(&[2, 3], &[3, 2]).expect("2x3 by 3x2 is planable");
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let out = ref_matmul(&plan, &a, &b, None).expect("oracle runs");
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(plan.output_shape, vec![2, 2]);
    }

    #[test]
    fn matmul_accumulates_in_the_shaders_k_major_order() {
        // The whole reason this module exists.  A sequential left-associated f32 sum —
        // which is what `MATMUL_HLSL`'s `for (k) acc += …` performs — gives:
        //
        //     ((0 + 1) + 1e8) + (-1e8)
        //   =  (1 + 1e8)              → 1e8   (the 1.0 falls off the mantissa: ulp(1e8) = 8)
        //   +  (-1e8)                 → 0.0
        //
        // A reassociating implementation (pairwise, blocked, or a vectorised
        // `matrixmultiply` micro-kernel) may instead group `1e8 + (-1e8) = 0` first and
        // return **1.0**.  Both are "correct matmuls".  Only one is what the shader does,
        // and pinning it is what makes this an oracle rather than a second opinion.
        let plan = MatMulPlan::matmul(&[1, 3], &[3, 1]).expect("1x3 by 3x1 is planable");
        let a = [1.0, 1.0e8, -1.0e8];
        let b = [1.0, 1.0, 1.0];
        let out = ref_matmul(&plan, &a, &b, None).expect("oracle runs");
        assert_eq!(
            out,
            vec![0.0],
            "the oracle must reproduce the shader's sequential k-major rounding, not a \
             mathematically nicer one"
        );
    }

    #[test]
    fn matmul_batched_matches_an_independent_triple_loop() {
        let (batch, m, k, n) = (3usize, 4usize, 5usize, 2usize);
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| (i as f32) * 0.25 - 3.0)
            .collect();
        let b: Vec<f32> = (0..batch * k * n)
            .map(|i| 1.5 - (i as f32) * 0.125)
            .collect();

        let plan = batched_plan(
            [batch as u32, m as u32, k as u32, n as u32],
            vec![batch, m, k],
            vec![batch, k, n],
            [(m * k) as u32, (k * n) as u32],
        );

        let got = ref_matmul(&plan, &a, &b, None).expect("oracle runs");
        let want = naive_batched_matmul(&a, &b, &plan);
        assert_eq!(got, want);
        assert_eq!(got.len(), batch * m * n);
    }

    #[test]
    fn matmul_batch_broadcast_b_reads_slice_zero_for_every_batch() {
        // `b_batch_stride == 0` is the entire batch-broadcast implementation: the shader
        // leaves `BOff` at 0 for every slice.  A `b` buffer holding only ONE k x n matrix
        // must therefore be reused across all batches — and must not be indexed past its
        // end, which is what a naive `slice * k * n` offset would do.
        let (batch, m, k, n) = (3usize, 2usize, 3usize, 2usize);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) + 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();

        let plan = batched_plan(
            [batch as u32, m as u32, k as u32, n as u32],
            vec![batch, m, k],
            vec![k, n],
            [(m * k) as u32, 0], // ← 0 = B is batch-broadcast
        );

        let got = ref_matmul(&plan, &a, &b, None).expect("oracle runs");
        let want = naive_batched_matmul(&a, &b, &plan);
        assert_eq!(got, want);

        // And each slice really did see the same B: slice i's output equals A_i · B.
        for bi in 0..batch {
            let a_i = &a[bi * m * k..(bi + 1) * m * k];
            let single = MatMulPlan::matmul(&[m, k], &[k, n]).expect("planable");
            let want_i = ref_matmul(&single, a_i, &b, None).expect("oracle runs");
            assert_eq!(&got[bi * m * n..(bi + 1) * m * n], &want_i[..]);
        }
    }

    #[test]
    fn matmul_rejects_a_buffer_that_does_not_match_its_planned_shape() {
        let plan = MatMulPlan::matmul(&[2, 3], &[3, 2]).expect("planable");
        let err = ref_matmul(&plan, &[1.0, 2.0], &[7.0; 6], None).expect_err("A is too short");
        assert!(matches!(err, DirectMLError::ShapeMismatch(_)), "{err}");
    }

    // ── ref_matmul: Gemm ─────────────────────────────────────────────────────

    #[test]
    fn gemm_applies_alpha_and_a_broadcast_beta_c() {
        // Y = 2·(A·B) + 3·C, with C a row vector broadcast over both rows.
        let plan = MatMulPlan::gemm(&[2, 3], &[3, 2], Some(&[2]), 2.0, 3.0, false, false)
            .expect("gemm is planable");
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = [100.0, 200.0];
        let out = ref_matmul(&plan, &a, &b, Some(&c)).expect("oracle runs");
        // A·B = [[58, 64], [139, 154]]
        assert_eq!(
            out,
            vec![
                2.0 * 58.0 + 3.0 * 100.0,
                2.0 * 64.0 + 3.0 * 200.0,
                2.0 * 139.0 + 3.0 * 100.0,
                2.0 * 154.0 + 3.0 * 200.0,
            ]
        );
    }

    #[test]
    fn gemm_transposes_both_operands_exactly_as_the_hlsl_backend_does() {
        // A_stored is 3x2 = Aᵀ, B_stored is 2x3 = Bᵀ, so Y = Aᵀᵀ·Bᵀᵀ … i.e. the same
        // 2x3 · 3x2 product as `matmul_2x3_by_3x2_matches_a_hand_computed_literal`.
        let plan = MatMulPlan::gemm(&[3, 2], &[2, 3], None, 1.0, 0.0, true, true)
            .expect("gemm is planable");
        assert_eq!((plan.m, plan.k, plan.n), (2, 3, 2));
        assert!(plan.needs_cpu_transpose());

        // Aᵀ (3x2), column-major view of [[1,2,3],[4,5,6]].
        let a_t = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        // Bᵀ (2x3), column-major view of [[7,8],[9,10],[11,12]].
        let b_t = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
        let out = ref_matmul(&plan, &a_t, &b_t, None).expect("oracle runs");
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn gemm_with_beta_zero_ignores_c_entirely() {
        // ONNX says `beta * C`; `0 * C` is nothing, and `MatMulPlan::build` drops the C
        // shape.  The oracle must not then read the C buffer it was handed.
        let plan = MatMulPlan::gemm(&[2, 3], &[3, 2], Some(&[2, 2]), 1.0, 0.0, false, false)
            .expect("planable");
        assert!(!plan.has_bias());
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let out = ref_matmul(&plan, &a, &b, Some(&[f32::NAN; 4])).expect("oracle runs");
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_op_name_distinguishes_matmul_from_gemm() {
        let mm = MatMulPlan::matmul(&[2, 3], &[3, 2]).expect("planable");
        assert_eq!(matmul_op_name(&mm), "MatMul");
        let gemm =
            MatMulPlan::gemm(&[2, 3], &[3, 2], None, 2.0, 0.0, false, false).expect("planable");
        assert_eq!(matmul_op_name(&gemm), "Gemm");
    }

    // ── ref_binary ───────────────────────────────────────────────────────────

    #[test]
    fn binary_ops_are_elementwise_over_identical_shapes() {
        let plan = ElementwisePlan::binary(&[2, 2], &[2, 2]).expect("planable");
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.5, -2.0, 8.0, 0.0];
        assert_eq!(
            ref_binary(&plan, BinaryOp::Add, &a, &b).expect("runs"),
            vec![1.5, 0.0, 11.0, 4.0]
        );
        assert_eq!(
            ref_binary(&plan, BinaryOp::Sub, &a, &b).expect("runs"),
            vec![0.5, 4.0, -5.0, 4.0]
        );
        assert_eq!(
            ref_binary(&plan, BinaryOp::Mul, &a, &b).expect("runs"),
            vec![0.5, -4.0, 24.0, 0.0]
        );
        let div = ref_binary(&plan, BinaryOp::Div, &a, &b).expect("runs");
        assert_eq!(div[0], 2.0);
        assert_eq!(div[1], -1.0);
        assert_eq!(div[2], 0.375);
        // The shader does not guard against division by zero, and neither does the
        // oracle: IEEE says +inf, and DirectML agrees.
        assert_eq!(div[3], f32::INFINITY);
    }

    #[test]
    fn binary_broadcasts_both_operands_up_to_the_output_shape() {
        // Not reachable through `ElementwisePlan::binary` today (it declines), but this
        // is the answer the GPU will have to produce when that restriction lifts.
        let plan = broadcast_binary_plan(vec![2, 3], vec![2, 1], vec![3]);
        let a = [10.0, 20.0];
        let b = [1.0, 2.0, 3.0];
        assert_eq!(
            ref_binary(&plan, BinaryOp::Add, &a, &b).expect("runs"),
            vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
        );
    }

    #[test]
    fn binary_on_a_unary_plan_is_a_shape_error_not_a_panic() {
        let plan = ElementwisePlan::unary(&[4]).expect("planable");
        let err = ref_binary(&plan, BinaryOp::Add, &[1.0; 4], &[1.0; 4])
            .expect_err("a unary plan has no B operand");
        assert!(matches!(err, DirectMLError::ShapeMismatch(_)), "{err}");
    }

    // ── ref_unary ────────────────────────────────────────────────────────────

    #[test]
    fn relu_pins_its_nan_and_negative_zero_behaviour() {
        assert_eq!(relu(2.5), 2.5);
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(0.0), 0.0);

        // Pinned choice #1: NaN in → +0.0 out (IEEE `maxNum`, matching Rust's `f32::max`
        // and `oxionnx-ops`' `v.max(0.0)`).  A GPU that propagates the NaN instead will
        // be *reported* by `compare` — see the module docs.
        assert_eq!(relu(f32::NAN), 0.0);
        assert!(!relu(f32::NAN).is_nan());

        // Pinned choice #2: -0.0 in → +0.0 out, bit for bit.  `f32::max` leaves the sign
        // of a zero result to LLVM, so we do not use it.
        assert_eq!(relu(-0.0).to_bits(), 0.0f32.to_bits());

        assert_eq!(relu(f32::INFINITY), f32::INFINITY);
        assert_eq!(relu(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn sigmoid_saturates_cleanly_and_never_produces_nan() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(f32::NEG_INFINITY), 0.0);
        assert_eq!(sigmoid(f32::INFINITY), 1.0);

        // The direct form is used precisely because it does *not* need stabilising:
        // exp(+100) is finite, exp(+1000) is +inf, and 1/(1+inf) is 0 — not a NaN.
        assert_eq!(sigmoid(-100.0), 0.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(100.0), 1.0);
        for x in [-1000.0f32, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0] {
            assert!(!sigmoid(x).is_nan(), "sigmoid({x}) must not be NaN");
            assert!(
                (0.0..=1.0).contains(&sigmoid(x)),
                "sigmoid({x}) must be in [0, 1]"
            );
        }
        assert!(sigmoid(f32::NAN).is_nan(), "a NaN in still means a NaN out");
    }

    #[test]
    fn tanh_saturates_at_both_infinities() {
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(tanh(f32::INFINITY), 1.0);
        assert_eq!(tanh(f32::NEG_INFINITY), -1.0);
        assert!((tanh(1.0) - 0.761_594_2).abs() < 1.0e-6);
    }

    #[test]
    fn unary_ops_run_over_the_whole_buffer() {
        let plan = ElementwisePlan::unary(&[2, 3]).expect("planable");
        let a = [-2.0, -0.5, 0.0, 0.5, 2.0, 100.0];
        let relu_out = ref_unary(&plan, UnaryOp::Relu, &a).expect("runs");
        assert_eq!(relu_out, vec![0.0, 0.0, 0.0, 0.5, 2.0, 100.0]);

        let sig = ref_unary(&plan, UnaryOp::Sigmoid, &a).expect("runs");
        assert_eq!(sig.len(), 6);
        assert_eq!(sig[2], 0.5);

        let th = ref_unary(&plan, UnaryOp::Tanh, &a).expect("runs");
        assert_eq!(th[2], 0.0);
    }

    #[test]
    fn unary_rejects_a_buffer_of_the_wrong_length() {
        let plan = ElementwisePlan::unary(&[2, 3]).expect("planable");
        let err = ref_unary(&plan, UnaryOp::Relu, &[1.0, 2.0]).expect_err("too short");
        assert!(matches!(err, DirectMLError::ShapeMismatch(_)), "{err}");
    }

    // ── tolerance policy ─────────────────────────────────────────────────────

    #[test]
    fn add_sub_mul_and_relu_are_held_to_bit_exactness() {
        assert_eq!(Tolerance::for_binary(BinaryOp::Add), Tolerance::Exact);
        assert_eq!(Tolerance::for_binary(BinaryOp::Sub), Tolerance::Exact);
        assert_eq!(Tolerance::for_binary(BinaryOp::Mul), Tolerance::Exact);
        assert_eq!(Tolerance::for_unary(UnaryOp::Relu), Tolerance::Exact);

        // A single ULP is a failure for these ops, and that is the entire point: a
        // kernel that reads A[i+1] instead of A[i] usually lands well within any loose
        // tolerance, and only an exact comparison catches it.
        let x = 1.0f32;
        let one_ulp_up = f32::from_bits(x.to_bits() + 1);
        assert!(!Tolerance::Exact.accepts(one_ulp_up, x));
        assert!(Tolerance::Exact.accepts(x, x));
    }

    #[test]
    fn div_sigmoid_and_tanh_are_not_held_to_bit_exactness() {
        // D3D permits 1 ULP of error in fp32 divide, and `exp`/`tanh` are approximations
        // on both sides.  Demanding exactness here would fail on *conforming* hardware —
        // a false alarm that would teach users to ignore verify mode.
        assert!(matches!(
            Tolerance::for_binary(BinaryOp::Div),
            Tolerance::Approx { .. }
        ));
        assert!(matches!(
            Tolerance::for_unary(UnaryOp::Sigmoid),
            Tolerance::Approx { .. }
        ));
        assert!(matches!(
            Tolerance::for_unary(UnaryOp::Tanh),
            Tolerance::Approx { .. }
        ));

        let x = 1.0f32;
        let one_ulp_up = f32::from_bits(x.to_bits() + 1);
        assert!(Tolerance::for_binary(BinaryOp::Div).accepts(one_ulp_up, x));
        // But a 1e-3 departure is still a bug, not noise.
        assert!(!Tolerance::for_binary(BinaryOp::Div).accepts(1.001, x));
        assert!(!Tolerance::for_unary(UnaryOp::Tanh).accepts(1.001, x));
    }

    #[test]
    fn exact_tolerance_allows_denormal_flush_to_zero_and_nothing_else() {
        // D3D is allowed to flush fp32 denormals to zero, so this is conforming:
        let denormal = f32::from_bits(1); // 1.4e-45
        assert!(denormal.is_subnormal());
        assert!(Tolerance::Exact.accepts(0.0, denormal));
        assert!(Tolerance::Exact.accepts(denormal, 0.0));
        assert!(Tolerance::Exact.accepts(-0.0, 0.0));

        // …but flushing a *normal* number is not.
        assert!(!Tolerance::Exact.accepts(0.0, f32::MIN_POSITIVE));
        assert!(!Tolerance::Exact.accepts(0.0, 1.0e-30));
    }

    #[test]
    fn nan_equals_nan_and_infinities_must_match_in_sign() {
        for tolerance in [
            Tolerance::Exact,
            Tolerance::Approx {
                rel: 1.0,
                abs: 1.0e9,
            },
        ] {
            assert!(
                tolerance.accepts(f32::NAN, f32::NAN),
                "NaN payloads differ per vendor"
            );
            assert!(!tolerance.accepts(f32::NAN, 1.0));
            assert!(!tolerance.accepts(1.0, f32::NAN));
            assert!(tolerance.accepts(f32::INFINITY, f32::INFINITY));
            assert!(!tolerance.accepts(f32::INFINITY, f32::NEG_INFINITY));
            // A huge `abs` term must not paper over "the GPU returned infinity".
            assert!(!tolerance.accepts(f32::INFINITY, 1.0));
        }
    }

    #[test]
    fn matmul_tolerance_scales_with_the_inner_dimension() {
        let small = MatMulPlan::matmul(&[2, 3], &[3, 2]).expect("planable");
        let large = MatMulPlan::matmul(&[2, 4096], &[4096, 2]).expect("planable");
        let (Tolerance::Approx { rel: r_small, .. }, Tolerance::Approx { rel: r_large, .. }) =
            (Tolerance::for_matmul(&small), Tolerance::for_matmul(&large))
        else {
            panic!("matmul is never held to exactness — the GPU may contract to `mad`");
        };
        assert!(r_small > 0.0);
        assert!(
            r_large > r_small * 30.0,
            "sqrt(4096/3) ≈ 37, so a K=4096 product must get a far wider budget than a \
             K=3 one: {r_large} vs {r_small}"
        );
        // ~1e-6 for a small K, as the design predicts.
        assert!(r_small < 1.0e-5, "K=3 must stay tight: {r_small}");
    }

    // ── compare ──────────────────────────────────────────────────────────────

    #[test]
    fn compare_on_identical_buffers_passes_with_zero_deviation() {
        let buf = [1.0f32, -2.0, 3.5, 0.0];
        let report = compare("Add", &buf, &buf, Tolerance::Exact).expect("same length");
        assert!(report.passed);
        assert_eq!(report.mismatches, 0);
        assert_eq!(report.elem_count, 4);
        assert_eq!(report.max_abs_deviation(), 0.0);
        assert_eq!(report.max_rel_deviation(), 0.0);
        assert!(report.first_mismatch.is_none());
        assert!(format!("{report}").contains("OK"));
    }

    #[test]
    fn compare_names_the_op_the_worst_deviation_and_the_index() {
        let cpu = [1.0f32, 2.0, 4.0, 8.0];
        let gpu = [1.0f32, 2.5, 4.0, 8.5]; // index 1: |d| 0.5, rel 0.25
                                           //                                    index 3: |d| 0.5, rel 0.0625
        let report = compare("Mul", &gpu, &cpu, Tolerance::Exact).expect("same length");
        assert!(!report.passed);
        assert_eq!(report.op, "Mul");
        assert_eq!(report.mismatches, 2);

        let first = report.first_mismatch.expect("two elements mismatched");
        assert_eq!(
            first.index, 1,
            "the first bad index localises the bad thread"
        );
        assert_eq!(first.gpu, 2.5);
        assert_eq!(first.cpu, 2.0);

        // Ties on |d| go to the earliest index; the relative winner is a different
        // element, which is exactly why both are reported.
        let worst_abs = report.worst_abs.expect("non-empty");
        assert_eq!(worst_abs.index, 1);
        assert_eq!(worst_abs.abs, 0.5);
        let worst_rel = report.worst_rel.expect("non-empty");
        assert_eq!(worst_rel.index, 1);
        assert_eq!(worst_rel.rel, 0.25);

        let text = format!("{report}");
        assert!(text.contains("MISMATCH"), "{text}");
        assert!(text.contains("Mul"), "{text}");
        assert!(text.contains("[1]"), "{text}");
    }

    #[test]
    fn compare_reports_an_infinite_relative_deviation_against_a_zero_oracle() {
        // Cancellation legitimately lands a dot product on exactly 0.0, and then *any*
        // GPU drift is an infinite relative error.  The report must say so honestly —
        // and the absolute floor, not the relative term, must decide pass/fail.
        let cpu = [0.0f32];
        let gpu = [1.0e-9f32];
        let loose = Tolerance::Approx {
            rel: 1.0e-6,
            abs: 1.0e-6,
        };
        let report = compare("MatMul", &gpu, &cpu, loose).expect("same length");
        assert!(report.passed, "the absolute floor must absorb this");
        assert_eq!(report.max_rel_deviation(), f32::INFINITY);
        assert_eq!(report.max_abs_deviation(), 1.0e-9);
    }

    #[test]
    fn compare_treats_a_length_mismatch_as_a_structural_error() {
        let err = compare("Add", &[1.0, 2.0], &[1.0], Tolerance::Exact)
            .expect_err("a wrong-length GPU buffer is not a numerical problem");
        assert!(matches!(err, DirectMLError::ShapeMismatch(_)), "{err}");
    }

    #[test]
    fn compare_walks_the_whole_buffer_not_just_a_prefix() {
        // The classic shader bug: the first thread group is right and nothing else ran.
        let cpu = vec![1.0f32; 1024];
        let mut gpu = cpu.clone();
        for v in gpu.iter_mut().skip(256) {
            *v = 0.0;
        }
        let report = compare("Relu", &gpu, &cpu, Tolerance::Exact).expect("same length");
        assert!(!report.passed);
        assert_eq!(report.mismatches, 768);
        assert_eq!(
            report.first_mismatch.expect("mismatched").index,
            256,
            "a first mismatch at exactly one thread-group boundary is the signature of a \
             dispatch-grid bug"
        );
    }

    // ── verify_* ─────────────────────────────────────────────────────────────

    #[test]
    fn verify_passes_when_the_gpu_agrees_with_the_oracle() {
        let plan = ElementwisePlan::binary(&[4], &[4]).expect("planable");
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.5, 0.5, 0.5, 0.5];
        let gpu = ref_binary(&plan, BinaryOp::Add, &a, &b).expect("oracle runs");
        let report = verify_binary(&plan, BinaryOp::Add, &a, &b, &gpu).expect("verifies");
        assert!(report.passed);
        assert_eq!(report.op, "Add");
        assert_eq!(report.tolerance, Tolerance::Exact);
    }

    #[test]
    fn verify_catches_a_single_perturbed_element() {
        let plan = ElementwisePlan::unary(&[8]).expect("planable");
        let a: Vec<f32> = (0..8).map(|i| i as f32 - 4.0).collect();
        let mut gpu = ref_unary(&plan, UnaryOp::Relu, &a).expect("oracle runs");
        gpu[6] += 1.0e-6; // a departure no loose tolerance would ever flag
        let report = verify_unary(&plan, UnaryOp::Relu, &a, &gpu).expect("verifies");
        assert!(!report.passed, "Relu is exact: 1e-6 is a bug, not noise");
        assert_eq!(report.mismatches, 1);
        assert_eq!(report.first_mismatch.expect("mismatched").index, 6);
    }

    #[test]
    fn verify_matmul_uses_the_k_scaled_tolerance_and_reports_the_op_name() {
        let plan = MatMulPlan::matmul(&[2, 3], &[3, 2]).expect("planable");
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut gpu = ref_matmul(&plan, &a, &b, None).expect("oracle runs");

        let report = verify_matmul(&plan, &a, &b, None, &gpu).expect("verifies");
        assert!(report.passed);
        assert_eq!(report.op, "MatMul");
        assert_eq!(report.tolerance, Tolerance::for_matmul(&plan));

        // A drift within the mad-contraction budget passes …
        gpu[0] += 58.0 * 1.0e-7;
        assert!(
            verify_matmul(&plan, &a, &b, None, &gpu)
                .expect("verifies")
                .passed
        );

        // … and a real error does not.
        gpu[3] += 1.0;
        let report = verify_matmul(&plan, &a, &b, None, &gpu).expect("verifies");
        assert!(!report.passed);
        assert_eq!(report.first_mismatch.expect("mismatched").index, 3);
    }

    #[test]
    fn verify_rejects_a_gpu_buffer_of_the_wrong_length() {
        let plan = ElementwisePlan::unary(&[8]).expect("planable");
        let a = [0.0f32; 8];
        let err = verify_unary(&plan, UnaryOp::Relu, &a, &[0.0f32; 7])
            .expect_err("the GPU returned 7 of 8 elements");
        assert!(matches!(err, DirectMLError::ShapeMismatch(_)), "{err}");
    }

    // ── the verify-mode switch ───────────────────────────────────────────────

    #[test]
    fn the_verify_flag_is_parsed_the_way_a_user_would_expect() {
        assert!(!parse_verify_flag(None));
        for off in ["", "0", "false", "no", "off", " OFF ", "False"] {
            assert!(
                !parse_verify_flag(Some(off)),
                "{off:?} must not enable verify"
            );
        }
        for on in ["1", "true", "yes", "on", "TRUE", "please"] {
            assert!(parse_verify_flag(Some(on)), "{on:?} must enable verify");
        }
    }

    #[test]
    fn verify_enabled_never_panics() {
        // Whatever the ambient environment says, reading it must not blow up on a hot
        // dispatch path.  (The *policy* is tested above, without touching the shared,
        // racy process environment.)
        let _ = verify_enabled();
        assert_eq!(VERIFY_ENV_VAR, "OXIONNX_DIRECTML_VERIFY");
    }

    // ── SelfCheckReport ──────────────────────────────────────────────────────

    #[test]
    fn self_check_report_tracks_the_worst_deviation_across_ops() {
        let mut report = SelfCheckReport::new(BackendKind::Hlsl, "Test Adapter".into(), 1.0e-3);
        assert_eq!(report.max_deviation(), 0.0, "no ops ran yet");
        assert!(report.passed);

        report.deviations.push(("MatMul", 1.0e-6));
        report.deviations.push(("Add", 0.0));
        report.deviations.push(("Relu", 4.0e-6));
        assert_eq!(report.max_deviation(), 4.0e-6);
    }

    #[test]
    fn self_check_report_max_deviation_never_swallows_a_nan() {
        // `f32::max` returns the *other* operand when one is NaN, so a naive fold would
        // report a clean 1e-6 for a GPU that produced NaNs.  That is the single most
        // dangerous way this report could lie.
        let mut report = SelfCheckReport::new(BackendKind::DirectMl, "x".into(), 1.0e-3);
        report.deviations.push(("MatMul", 1.0e-6));
        report.deviations.push(("Sigmoid", f32::NAN));
        assert!(report.max_deviation().is_nan());
    }

    #[test]
    fn self_check_report_record_applies_both_gates() {
        let cpu = [1.0f32, 2.0];

        // Gate 1 — the per-op policy.  `Add` is exact, so a 1e-6 drift fails even though
        // the caller's blunt tolerance is a thousand times looser.
        let mut report = SelfCheckReport::new(BackendKind::Hlsl, "x".into(), 1.0e-3);
        let drifted = compare(
            "Add",
            &[1.0f32 + 1.0e-6, 2.0],
            &cpu,
            Tolerance::for_binary(BinaryOp::Add),
        )
        .expect("same length");
        report.record(&drifted);
        assert!(
            !report.passed,
            "a loose --tolerance must not be able to wave through an exact op"
        );
        assert_eq!(report.deviations.len(), 1);

        // Gate 2 — the caller's blunt tolerance.  A `Sigmoid` drift of 5e-7 satisfies the
        // per-op policy but not a caller who demanded 1e-9.
        let mut strict = SelfCheckReport::new(BackendKind::Hlsl, "x".into(), 1.0e-9);
        let sigmoid_ish = compare(
            "Sigmoid",
            &[1.0f32 + 5.0e-7, 2.0],
            &cpu,
            Tolerance::for_unary(UnaryOp::Sigmoid),
        )
        .expect("same length");
        assert!(
            sigmoid_ish.passed,
            "5e-7 is within the transcendental budget"
        );
        strict.record(&sigmoid_ish);
        assert!(!strict.passed, "…but not within the caller's 1e-9");

        // Both gates satisfied.
        let mut happy = SelfCheckReport::new(BackendKind::Hlsl, "x".into(), 1.0e-3);
        happy.record(&compare("Add", &cpu, &cpu, Tolerance::Exact).expect("same length"));
        assert!(happy.passed);
        assert_eq!(happy.max_deviation(), 0.0);
        assert_eq!(happy.deviations, vec![("Add", 0.0)]);
    }

    #[test]
    fn self_check_report_record_rejects_a_non_finite_deviation() {
        let mut report = SelfCheckReport::new(BackendKind::Hlsl, "x".into(), f32::INFINITY);
        let nan_gpu = compare("Relu", &[f32::NAN], &[1.0], Tolerance::Exact).expect("same length");
        report.record(&nan_gpu);
        assert!(
            !report.passed,
            "an infinite tolerance must still not accept a GPU NaN"
        );
        assert!(report.max_deviation().is_infinite());
    }
}
