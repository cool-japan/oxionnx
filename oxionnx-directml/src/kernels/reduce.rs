//! `Reduce{Sum,Mean,Max,Min}` kernel.  **Platform-neutral** — no `#[cfg]`, no FFI.
//!
//! Same three-step shape as [`crate::kernels::matmul`]: build a [`ReducePlan`] from the
//! operand shape, the resolved [`ReduceKind`], the `axes` attribute and `keepdims`; hand it
//! plus the raw `&[f32]` slice to the backend; check the length, shadow-verify, wrap into a
//! `Tensor` of the plan's (per-`keepdims`) output shape.
//!
//! # Single-axis only, and that boundary lives in the plan
//!
//! [`ReducePlan::reduce`] **declines** every `axes` list that resolves to more than one
//! distinct axis — including an empty `axes` on a rank ≥ 2 tensor, which ONNX reads as "all
//! axes".  A flat, index-parallel reduction shader cannot walk several independent reduced
//! axes without a per-thread nested loop over a shape it does not carry, so the multi-axis
//! case is a [`crate::DirectMLError::Declined`] → `dispatch::route` → `Ok(None)` → the CPU
//! kernel, which reduces correctly.  This kernel therefore passes the whole `axes` slice
//! down untouched and lets the plan be the single place that draws the line.

use oxionnx_core::Tensor;

use crate::backend::Backend;
use crate::error::Result;
use crate::kernels::matmul::{check_len, verified};
use crate::plan::{ReduceKind, ReducePlan};
use crate::reference;

/// ONNX `Reduce{Sum,Mean,Max,Min}` over the axes named in `axes`, with `keepdims`.
///
/// `axes` is the raw ONNX attribute (empty means "all axes", negative entries count from
/// the end); [`ReducePlan::reduce`] resolves and validates it, and declines the multi-axis
/// case to the CPU.
///
/// # Errors
/// [`crate::DirectMLError::ShapeMismatch`] when an axis is out of range for the rank (the
/// CPU operator rejects the same input).
/// [`crate::DirectMLError::Declined`] when the resolved axis set is not exactly one axis,
/// the tensor is empty, or a size overflows `u32` — each routes to a CPU fallback.
/// Anything else is a genuine GPU failure.
pub(crate) fn dml_reduce(
    a: &Tensor,
    kind: ReduceKind,
    axes: &[i64],
    keepdims: bool,
    backend: &Backend,
) -> Result<Tensor> {
    let plan = ReducePlan::reduce(kind, &a.shape, axes, keepdims)?;
    let gpu = backend.reduce(&plan, &a.data)?;

    check_len(kind.as_str(), gpu.len(), plan.output_elems()?)?;

    if reference::verify_enabled() {
        let comparison = reference::verify_reduce(&plan, &a.data, &gpu)?;
        verified(kind.as_str(), &comparison)?;
    }

    Ok(Tensor::new(gpu, plan.output_shape.clone()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use crate::error::DirectMLError;
    use crate::plan::{ReduceKind, ReducePlan};

    // As with the softmax kernel, the body is one straight-line backend call; what is worth
    // pinning on Linux is that the plan the router builds honours `axes` and `keepdims` and
    // declines multi-axis to the CPU.  The decline-to-`Ok(None)` path runs through
    // `dispatch::route` in `dispatch::tests`.

    fn declined(e: &DirectMLError) -> bool {
        matches!(e, DirectMLError::Declined(_))
    }

    #[test]
    fn keepdims_true_leaves_a_size_one_axis_and_false_removes_it() {
        let keep = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[1], true).unwrap();
        assert_eq!(keep.output_shape, vec![2, 1, 4]);
        let squeeze = ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[1], false).unwrap();
        assert_eq!(squeeze.output_shape, vec![2, 4]);
    }

    #[test]
    fn a_multi_axis_reduce_is_declined_to_the_cpu_not_mis_executed() {
        // Two explicit axes.
        assert!(declined(
            &ReducePlan::reduce(ReduceKind::Sum, &[2, 3, 4], &[0, 2], false).unwrap_err()
        ));
        // Empty axes over rank >= 2 means "all axes" == multi-axis.
        assert!(declined(
            &ReducePlan::reduce(ReduceKind::Mean, &[2, 3], &[], false).unwrap_err()
        ));
    }

    #[test]
    fn every_reduce_kind_maps_to_its_stable_name() {
        assert_eq!(ReduceKind::Sum.as_str(), "ReduceSum");
        assert_eq!(ReduceKind::Mean.as_str(), "ReduceMean");
        assert_eq!(ReduceKind::Max.as_str(), "ReduceMax");
        assert_eq!(ReduceKind::Min.as_str(), "ReduceMin");
    }
}
