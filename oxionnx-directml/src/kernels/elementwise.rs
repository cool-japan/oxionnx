//! Elementwise kernels.  **Platform-neutral** — no `#[cfg]`, no FFI.
//!
//! Same shape as [`crate::kernels::matmul`]: build an [`ElementwisePlan`] from the shapes,
//! hand it plus the raw slices to the backend, check the length, shadow-verify, wrap.
//!
//! # Read this before "adding broadcasting"
//!
//! [`ElementwisePlan::binary`] **declines every pair of non-identical shapes**, even
//! perfectly broadcastable ones such as `[2, 3, 4]` and `[1, 4]`.  That decline reaches
//! `dispatch::route` as an `Err`, becomes `Ok(None)`, and `oxionnx-ops`' CPU kernel — which
//! broadcasts correctly — runs.
//!
//! The obvious-looking fix is to dispatch `max(a.numel(), b.numel())` threads.  It is
//! wrong, and it is wrong in the way that matters: the HLSL kernels are index-parallel
//! (`C[i] = A[i] ⊕ B[i]`) with no notion of a shape at all, so 24 threads over a 4-element
//! `B` read `B[0..24]` — past the end of the buffer, into whatever the allocator left there
//! — and hand back a tensor of exactly the right *shape* full of the wrong *values*.  No
//! bounds check fires.  No test that only asserts shapes catches it.
//!
//! The two correct ways to lift the restriction are already implemented and tested
//! ([`crate::plan::broadcast_expand`] for the HLSL path, and
//! [`crate::layout::DmlTensorLayout::broadcast_to`]'s 0-strides for the DirectML path,
//! which copies nothing).  Neither is *wired in*, because neither has been run on hardware.
//! Wire one in, verify it with `DirectMLContext::self_check` on a real GPU, and relax
//! `ElementwisePlan::binary` — in that order.

use oxionnx_core::Tensor;

use crate::backend::Backend;
use crate::error::Result;
use crate::kernels::matmul::{check_len, verified};
use crate::plan::{BinaryOp, ElementwisePlan, UnaryOp};
use crate::reference;

/// A binary elementwise op (`Add`, `Sub`, `Mul`, `Div`).
///
/// # Errors
/// [`crate::DirectMLError::Declined`] when the shapes are not identical, are empty, exceed
/// [`crate::layout::DML_RANK`], or are too large.
/// [`crate::DirectMLError::ShapeMismatch`] when the shapes are not even broadcastable.
/// Anything else is a genuine GPU failure.
pub(crate) fn dml_binary(
    a: &Tensor,
    b: &Tensor,
    op: BinaryOp,
    backend: &Backend,
) -> Result<Tensor> {
    let plan = ElementwisePlan::binary(&a.shape, &b.shape)?;
    let gpu = backend.binary(&plan, op, &a.data, &b.data)?;

    check_len(op.as_str(), gpu.len(), elems(&plan))?;

    if reference::verify_enabled() {
        let comparison = reference::verify_binary(&plan, op, &a.data, &b.data, &gpu)?;
        verified(op.as_str(), &comparison)?;
    }

    Ok(Tensor::new(gpu, plan.output_shape.clone()))
}

/// A unary elementwise op (`Relu`, `Sigmoid`, `Tanh`).
///
/// # Errors
/// [`crate::DirectMLError::Declined`] when the operand is empty, exceeds
/// [`crate::layout::DML_RANK`], or is too large.  Anything else is a genuine GPU failure.
pub(crate) fn dml_unary(a: &Tensor, op: UnaryOp, backend: &Backend) -> Result<Tensor> {
    let plan = ElementwisePlan::unary(&a.shape)?;
    let gpu = backend.unary(&plan, op, &a.data)?;

    check_len(op.as_str(), gpu.len(), elems(&plan))?;

    if reference::verify_enabled() {
        let comparison = reference::verify_unary(&plan, op, &a.data, &gpu)?;
        verified(op.as_str(), &comparison)?;
    }

    Ok(Tensor::new(gpu, plan.output_shape.clone()))
}

/// The plan's element count as a `usize`.
///
/// [`ElementwisePlan::elem_count`] is a `u32` that `plan.rs` has already range-checked, so
/// this widening is total and lossless on every target this crate builds for — it is the
/// one direction of integer conversion that cannot lose information.
fn elems(plan: &ElementwisePlan) -> usize {
    plan.elem_count as usize
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::elems;
    use crate::error::DirectMLError;
    use crate::plan::ElementwisePlan;

    #[test]
    fn elems_widens_the_pre_checked_u32_losslessly() {
        let plan = ElementwisePlan::unary(&[2, 3, 4]).unwrap();
        assert_eq!(elems(&plan), 24);
        assert_eq!(elems(&plan), plan.output_shape.iter().product::<usize>());
    }

    #[test]
    fn a_broadcastable_but_non_identical_pair_is_declined_not_mis_planned() {
        // The regression test for the "just dispatch max(numel) threads" bug.  If this ever
        // starts returning `Ok`, the shaders will read past the end of the smaller operand.
        let err = ElementwisePlan::binary(&[2, 3, 4], &[1, 4]).unwrap_err();
        assert!(
            matches!(err, DirectMLError::Declined(_)),
            "a broadcastable pair must DECLINE (→ CPU, which broadcasts correctly), not be \
             planned; got {err:?}"
        );
    }

    #[test]
    fn a_non_broadcastable_pair_is_a_shape_error_the_cpu_op_will_also_raise() {
        let err = ElementwisePlan::binary(&[2, 3], &[4, 5]).unwrap_err();
        assert!(
            matches!(err, DirectMLError::ShapeMismatch(_)),
            "a malformed model is a ShapeMismatch, not a Declined — the two are routed \
             identically today but mean opposite things; got {err:?}"
        );
    }
}
