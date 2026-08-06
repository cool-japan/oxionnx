//! `Softmax` kernel.  **Platform-neutral** — no `#[cfg]`, no FFI.
//!
//! Same three-step shape as [`crate::kernels::matmul`] and [`crate::kernels::elementwise`]:
//!
//! 1. Build a [`SoftmaxPlan`] from the operand shape and the node's `axis`.
//! 2. Hand the plan plus the raw `&[f32]` slice to the backend, which routes it to the
//!    genuine `DML_ACTIVATION_SOFTMAX` operator or the HLSL `SOFTMAX_HLSL` shader.
//! 3. Check the returned buffer's length, shadow-verify it if asked, wrap it in a `Tensor`
//!    of the (softmax-preserving) input shape.
//!
//! A [`crate::DirectMLError::Declined`] from step 1 or 2 propagates to `dispatch::route`,
//! becomes `Ok(None)`, and the tuned CPU kernel runs — the correct outcome, not a failure.
//! The non-terminal-axis case that the classic `DML_ACTIVATION_SOFTMAX_OPERATOR_DESC`
//! cannot express is one such decline, and it is the backend's to make (via
//! [`SoftmaxPlan::reduces_last_axis`]); this kernel is axis-agnostic.

use oxionnx_core::Tensor;

use crate::backend::Backend;
use crate::error::Result;
use crate::kernels::matmul::{check_len, verified};
use crate::plan::SoftmaxPlan;
use crate::reference;

/// ONNX `Softmax` over a single `axis` (opset-13 semantics).
///
/// `axis` is resolved (a negative value counts from the end) and validated by
/// [`SoftmaxPlan::softmax`]; the caller passes the raw attribute value straight through.
///
/// # Errors
/// [`crate::DirectMLError::ShapeMismatch`] when `axis` is out of range for the rank (a
/// malformed node the CPU operator rejects too).
/// [`crate::DirectMLError::Declined`] when the tensor is empty, a size overflows `u32`, or
/// the backend cannot express this axis — the router turns each into a CPU fallback.
/// Anything else is a genuine GPU failure.
pub(crate) fn dml_softmax(a: &Tensor, axis: i64, backend: &Backend) -> Result<Tensor> {
    let plan = SoftmaxPlan::softmax(&a.shape, axis)?;
    let gpu = backend.softmax(&plan, &a.data)?;

    check_len("Softmax", gpu.len(), plan.output_elems()?)?;

    if reference::verify_enabled() {
        let comparison = reference::verify_softmax(&plan, &a.data, &gpu)?;
        verified("Softmax", &comparison)?;
    }

    // Softmax is shape-preserving: output shape == input shape.
    Ok(Tensor::new(gpu, plan.shape.clone()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use crate::error::DirectMLError;
    use crate::plan::SoftmaxPlan;

    // The kernel itself is one straight-line call into the backend; the behaviour worth
    // pinning on Linux is that the *plan* the router feeds it decodes the ONNX `axis`
    // attribute exactly, because a wrong axis produces a right-shaped tensor of wrong
    // numbers that no shape assertion would catch — the same trap the elementwise kernel
    // guards against.  The end-to-end decline-to-`Ok(None)` path is exercised through
    // `dispatch::route` in `dispatch::tests`.

    fn declined(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::Declined(_))
    }
    fn mismatch(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::ShapeMismatch(_))
    }

    #[test]
    fn the_default_axis_minus_one_normalises_the_last_dimension() {
        // The opset-13 default the router passes when the node omits `axis`.
        let plan = SoftmaxPlan::softmax(&[2, 3, 4], -1).unwrap();
        assert_eq!(plan.axis, 2, "-1 resolves to the trailing axis");
        assert_eq!(plan.inner, 1);
        assert!(plan.reduces_last_axis());
    }

    #[test]
    fn a_non_terminal_axis_is_planned_but_flagged_for_the_backend() {
        // A middle axis is a valid plan — the HLSL path handles it — but the classic DML
        // softmax operator cannot, so `reduces_last_axis()` is the flag the backend reads
        // to decide whether to decline.  The kernel stays axis-agnostic.
        let plan = SoftmaxPlan::softmax(&[2, 3, 4], 1).unwrap();
        assert_eq!(plan.axis, 1);
        assert_eq!(plan.inner, 4);
        assert!(!plan.reduces_last_axis());
    }

    #[test]
    fn an_out_of_range_axis_is_a_shape_error_the_cpu_op_also_raises() {
        assert!(mismatch(&SoftmaxPlan::softmax(&[2, 3], 2).unwrap_err()));
    }

    #[test]
    fn an_empty_tensor_is_declined_rather_than_sent_to_a_zero_width_buffer() {
        assert!(declined(&SoftmaxPlan::softmax(&[0, 4], 1).unwrap_err()));
    }
}
