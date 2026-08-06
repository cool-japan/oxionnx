//! `MatMul` / `Gemm` kernels.  **Platform-neutral** — no `#[cfg]`, no FFI.
//!
//! Deliberately thin.  Every dimension, every range check, every dispatch grid and every
//! buffer size comes off a [`MatMulPlan`]; this module computes **nothing** shape-derived
//! and contains no `as u32`.  The three steps are:
//!
//! 1. Build the plan from the operand *shapes*.
//! 2. Hand the plan plus the raw `&[f32]` slices to the backend.
//! 3. Check the returned buffer's length, shadow-verify it if asked, wrap it in a `Tensor`.
//!
//! A [`DirectMLError::Declined`] from step 1 or 2 propagates to `dispatch::route`, which
//! turns it into `Ok(None)` and lets the CPU kernel run.  That is the correct outcome, not
//! a failure — see `dispatch`'s module docs for the contract.

use oxionnx_core::Tensor;

use crate::backend::Backend;
use crate::error::{DirectMLError, Result};
use crate::plan::MatMulPlan;
use crate::reference;

/// ONNX `MatMul`.
///
/// # Errors
/// [`DirectMLError::Declined`] when the operands are not 2-D × 2-D, are empty, or are too
/// large — see [`MatMulPlan::matmul`].
/// [`DirectMLError::ShapeMismatch`] when the inner dimensions disagree.
/// Anything else is a genuine GPU failure.
pub(crate) fn dml_matmul(a: &Tensor, b: &Tensor, backend: &Backend) -> Result<Tensor> {
    let plan = MatMulPlan::matmul(&a.shape, &b.shape)?;
    execute(&plan, a, b, None, backend)
}

/// ONNX `Gemm`: `Y = alpha · op(A) · op(B) + beta · C`.
///
/// # Errors
/// As [`dml_matmul`], plus [`DirectMLError::ShapeMismatch`] when `c` does not broadcast to
/// `[m, n]`.
#[allow(clippy::too_many_arguments)] // Mirrors ONNX `Gemm`'s attribute set 1:1.
pub(crate) fn dml_gemm(
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    backend: &Backend,
) -> Result<Tensor> {
    let plan = MatMulPlan::gemm(
        &a.shape,
        &b.shape,
        c.map(|t| t.shape.as_slice()),
        alpha,
        beta,
        trans_a,
        trans_b,
    )?;
    execute(&plan, a, b, c, backend)
}

/// Dispatch a planned MatMul / Gemm and turn the result back into a `Tensor`.
///
/// # Errors
/// Whatever the backend returns, plus [`DirectMLError::DispatchFailed`] when the backend
/// hands back a buffer of the wrong length, or when shadow verification is on and the GPU
/// disagrees with the oracle.
fn execute(
    plan: &MatMulPlan,
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    backend: &Backend,
) -> Result<Tensor> {
    let c_data = c.map(|t| t.data.as_slice());
    let gpu = backend.matmul(plan, &a.data, &b.data, c_data)?;

    let op = reference::matmul_op_name(plan);
    let expected = plan.output_elems()?;
    check_len(op, gpu.len(), expected)?;

    if reference::verify_enabled() {
        let comparison = reference::verify_matmul(plan, &a.data, &b.data, c_data, &gpu)?;
        verified(op, &comparison)?;
    }

    Ok(Tensor::new(gpu, plan.output_shape.clone()))
}

/// Reject a buffer whose length does not match the plan, **before** anything else looks at
/// it.
///
/// Two reasons this is not paranoia:
///
/// * `Tensor::new` carries a `debug_assert_eq!(data.len(), shape.product())`.  A backend
///   that returned the wrong length would therefore *panic* in a debug build and construct
///   a silently inconsistent `Tensor` in a release one — the worst possible split.
/// * It has to happen before [`reference::compare`], which reports a length mismatch as
///   [`DirectMLError::ShapeMismatch`].  The router reads `ShapeMismatch` as "the model is
///   malformed" and defers to the CPU operator, which is precisely the wrong conclusion: a
///   GPU that returns the wrong number of elements is a **GPU failure**, and must be
///   classified as one.
///
/// # Errors
/// [`DirectMLError::DispatchFailed`] when `actual != expected`.
pub(crate) fn check_len(op: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    Err(DirectMLError::DispatchFailed(format!(
        "{op}: the backend returned {actual} elements but the plan says {expected}.  A buffer \
         of the wrong length is a GPU failure, not a shape error."
    )))
}

/// Turn a shadow-verification result into `Ok(())` or a loud failure.
///
/// A mismatch is deliberately a **failure**, not a warning that returns the numbers anyway.
/// The GPU has been *proved* wrong on this exact input; handing the caller a tensor full of
/// values we know are incorrect, so that they can propagate through the rest of the graph,
/// would defeat the entire purpose of having an oracle.  Returning `Err` sends the node
/// down the CPU path (or aborts the run, under `OXIONNX_DIRECTML_STRICT`), and either way
/// `dispatch::route` logs it at `error!`.
///
/// # Errors
/// [`DirectMLError::DispatchFailed`] when `comparison.passed` is false.
pub(crate) fn verified(op: &str, comparison: &reference::ComparisonReport) -> Result<()> {
    if comparison.passed {
        tracing::debug!(%op, report = %comparison, "DirectML shadow verification passed");
        return Ok(());
    }
    Err(DirectMLError::DispatchFailed(format!(
        "VERIFY MISMATCH — the GPU disagrees with the CPU oracle, so its output has been \
         discarded rather than returned.  {comparison}"
    )))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{check_len, verified};
    use crate::error::DirectMLError;
    use crate::plan::{BinaryOp, ElementwisePlan};
    use crate::reference;

    #[test]
    fn a_correct_length_passes() {
        assert!(check_len("MatMul", 20, 20).is_ok());
    }

    #[test]
    fn a_wrong_length_is_a_dispatch_failure_and_never_a_shape_error() {
        let err = check_len("MatMul", 19, 20).unwrap_err();
        assert!(
            matches!(err, DirectMLError::DispatchFailed(_)),
            "a GPU that returns the wrong element count has FAILED; classifying it as a \
             ShapeMismatch would make the router blame the user's model.  Got {err:?}"
        );
        assert!(format!("{err}").contains("19"));
        assert!(format!("{err}").contains("20"));
    }

    #[test]
    fn a_passing_comparison_is_accepted() {
        let plan = ElementwisePlan::binary(&[4], &[4]).unwrap();
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 0.5, 0.5, 0.5];
        let oracle = reference::ref_binary(&plan, BinaryOp::Add, &a, &b).unwrap();
        let comparison = reference::verify_binary(&plan, BinaryOp::Add, &a, &b, &oracle).unwrap();

        assert!(comparison.passed);
        assert!(verified("Add", &comparison).is_ok());
    }

    #[test]
    fn a_failing_comparison_discards_the_gpu_numbers() {
        let plan = ElementwisePlan::binary(&[4], &[4]).unwrap();
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 0.5, 0.5, 0.5];
        // A "GPU" that is wrong in the third element — the shape is right, the length is
        // right, and every value is plausible.  This is exactly what a mis-indexed shader
        // produces, and exactly what nothing but the oracle can catch.
        let gpu = [1.5f32, 2.5, 99.0, 4.5];
        let comparison = reference::verify_binary(&plan, BinaryOp::Add, &a, &b, &gpu).unwrap();

        assert!(!comparison.passed);
        let err = verified("Add", &comparison).unwrap_err();
        assert!(
            matches!(err, DirectMLError::DispatchFailed(_)),
            "a verified-wrong result is a GPU FAILURE, so strict mode can abort on it and \
             the router logs it at error!; got {err:?}"
        );

        let message = format!("{err}");
        assert!(message.contains("VERIFY MISMATCH"), "got: {message}");
        assert!(
            message.contains("discarded"),
            "the message must say the numbers were thrown away, not returned: {message}"
        );
        assert!(
            message.contains('2'),
            "the message must carry the index of the first bad element: {message}"
        );
    }
}
