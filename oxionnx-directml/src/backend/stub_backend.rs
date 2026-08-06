//! The non-Windows backend: permanently unavailable, permanently declining.
//!
//! Mirrors `oxionnx-coreml`'s `package/stub_impl.rs`.  It exists so that `dispatch.rs`,
//! `kernels/*`, `context.rs` and every other neutral module can be **compiled and tested
//! on Linux** against a real `Backend` type with the real signatures.  That is not a
//! cosmetic convenience: the routing contract — *"a declining backend must become
//! `Ok(None)`, never an `Err`"* — is the single most important invariant this crate has,
//! and it is the one thing about the GPU path that CI here can actually check.
//!
//! # Why nothing constructs one outside `cfg(test)`
//!
//! Because nothing can.  On a non-Windows target a `Backend` genuinely cannot exist:
//! [`Backend::try_new`] returns `None` unconditionally, so [`crate::DirectMLContext::try_new`]
//! returns `None`, so `oxionnx::Session` holds `dml: None` and never reaches this code at
//! all.  The `#[cfg(test)] fn declining_for_tests` below is the *only* way to get one, and
//! it exists so `dispatch::tests` can drive a declining backend through the router.

use crate::backend::BackendKind;
use crate::error::{DirectMLError, Result};
use crate::plan::{
    BinaryOp, ConvPlan, ElementwisePlan, MatMulPlan, ReducePlan, SoftmaxPlan, UnaryOp,
};

/// The non-Windows backend.  Never available; always declines.
///
/// The private field is what makes `declining_for_tests` the sole constructor: only this
/// module and its descendants can name it.
pub(crate) struct Backend {
    /// Blocks construction from outside this module.
    _private: (),
}

impl Backend {
    /// Always `None`: this target has no D3D12, and never will.
    ///
    /// This is the honest answer, not a placeholder.  Returning `Some` of a permanently
    /// declining backend would be strictly worse than `None`: `oxionnx`'s session runner
    /// keys "this node is GPU-eligible" off `dml.is_some()`, so an inactive-but-present
    /// context would drag every claimed node into the runner's **serial** GPU phase, watch
    /// it decline, and then run it on the CPU anyway — turning parallel CPU work into
    /// serial CPU work.
    pub(crate) fn try_new() -> Option<Self> {
        None
    }

    /// Always [`BackendKind::Unavailable`].
    pub(crate) fn kind(&self) -> BackendKind {
        BackendKind::Unavailable
    }

    /// Always `"none"` — there is no DXGI adapter to describe.
    pub(crate) fn adapter_name(&self) -> String {
        "none".to_string()
    }

    /// Always declines.
    ///
    /// # Errors
    /// Always [`DirectMLError::Declined`], which `dispatch::route` turns into `Ok(None)`
    /// and the session runner turns into a correct CPU execution.
    pub(crate) fn matmul(
        &self,
        plan: &MatMulPlan,
        a: &[f32],
        b: &[f32],
        c: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let _ = (plan, a, b, c);
        Err(Self::declined("MatMul/Gemm"))
    }

    /// Always declines.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn binary(
        &self,
        plan: &ElementwisePlan,
        op: BinaryOp,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        let _ = (plan, a, b);
        Err(Self::declined(op.as_str()))
    }

    /// Always declines.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn unary(&self, plan: &ElementwisePlan, op: UnaryOp, a: &[f32]) -> Result<Vec<f32>> {
        let _ = (plan, a);
        Err(Self::declined(op.as_str()))
    }

    /// Always declines.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn softmax(&self, plan: &SoftmaxPlan, a: &[f32]) -> Result<Vec<f32>> {
        let _ = (plan, a);
        Err(Self::declined("Softmax"))
    }

    /// Always declines.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn reduce(&self, plan: &ReducePlan, a: &[f32]) -> Result<Vec<f32>> {
        let _ = a;
        Err(Self::declined(plan.kind.as_str()))
    }

    /// Always declines.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn conv(
        &self,
        plan: &ConvPlan,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let _ = (plan, input, weight, bias);
        Err(Self::declined("Conv"))
    }

    /// The one decline message, so every path spells it the same way.
    ///
    /// [`DirectMLError::Declined`] and **not** [`DirectMLError::DispatchFailed`]: nothing
    /// has failed here.  There is no GPU on this target, which is an ordinary, expected
    /// state of the world, and the router must read it as "run this on the CPU" rather
    /// than as "the GPU broke".
    fn declined(op: &str) -> DirectMLError {
        DirectMLError::Declined(format!(
            "{op}: no DirectML backend on this target (Windows + D3D12 only)"
        ))
    }
}

#[cfg(test)]
impl Backend {
    /// Test-only constructor.
    ///
    /// Lets `dispatch::tests` exercise the **routing contract** in CI on Linux, where no
    /// GPU exists: the router must turn a declining backend into `Ok(None)`, never into an
    /// `Err`.  The private field means only this module and its descendants can build one.
    pub(crate) fn declining_for_tests() -> Self {
        Self { _private: () }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::Backend;
    use crate::backend::BackendKind;
    use crate::error::DirectMLError;
    use crate::plan::{
        BinaryOp, ConvPlan, ElementwisePlan, MatMulPlan, ReduceKind, ReducePlan, SoftmaxPlan,
        UnaryOp,
    };

    #[test]
    fn try_new_is_always_none() {
        assert!(
            Backend::try_new().is_none(),
            "a non-Windows target has no D3D12 device and must report so"
        );
    }

    #[test]
    fn kind_is_unavailable_and_not_a_gpu() {
        let backend = Backend::declining_for_tests();
        assert_eq!(backend.kind(), BackendKind::Unavailable);
        assert!(!backend.kind().is_gpu());
    }

    #[test]
    fn adapter_name_is_none_not_a_fabricated_device() {
        assert_eq!(Backend::declining_for_tests().adapter_name(), "none");
    }

    #[test]
    fn matmul_declines_rather_than_failing() {
        let backend = Backend::declining_for_tests();
        let plan = MatMulPlan::matmul(&[2, 3], &[3, 4]).unwrap();
        let a = vec![0.0f32; 6];
        let b = vec![0.0f32; 12];

        let err = backend.matmul(&plan, &a, &b, None).unwrap_err();
        assert!(
            matches!(err, DirectMLError::Declined(_)),
            "the stub must DECLINE (→ CPU fallback), not FAIL (→ a logged GPU error); got {err:?}"
        );
    }

    #[test]
    fn binary_declines_for_every_op() {
        let backend = Backend::declining_for_tests();
        let plan = ElementwisePlan::binary(&[2, 2], &[2, 2]).unwrap();
        let a = vec![1.0f32; 4];

        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            let err = backend.binary(&plan, op, &a, &a).unwrap_err();
            assert!(
                matches!(err, DirectMLError::Declined(_)),
                "{op:?} must decline, got {err:?}"
            );
            assert!(
                format!("{err}").contains(op.as_str()),
                "the decline must name the op it declined"
            );
        }
    }

    #[test]
    fn unary_declines_for_every_op() {
        let backend = Backend::declining_for_tests();
        let plan = ElementwisePlan::unary(&[4]).unwrap();
        let a = vec![1.0f32; 4];

        for op in [UnaryOp::Relu, UnaryOp::Sigmoid, UnaryOp::Tanh] {
            let err = backend.unary(&plan, op, &a).unwrap_err();
            assert!(
                matches!(err, DirectMLError::Declined(_)),
                "{op:?} must decline, got {err:?}"
            );
        }
    }

    #[test]
    fn softmax_declines_rather_than_failing() {
        let backend = Backend::declining_for_tests();
        let plan = SoftmaxPlan::softmax(&[2, 3], -1).unwrap();
        let a = vec![0.0f32; 6];

        let err = backend.softmax(&plan, &a).unwrap_err();
        assert!(
            matches!(err, DirectMLError::Declined(_)),
            "the stub must DECLINE (→ CPU fallback), not FAIL; got {err:?}"
        );
    }

    #[test]
    fn reduce_declines_for_every_kind() {
        let backend = Backend::declining_for_tests();
        let a = vec![1.0f32; 6];

        for kind in [
            ReduceKind::Sum,
            ReduceKind::Mean,
            ReduceKind::Max,
            ReduceKind::Min,
        ] {
            let plan = ReducePlan::reduce(kind, &[2, 3], &[1], false).unwrap();
            let err = backend.reduce(&plan, &a).unwrap_err();
            assert!(
                matches!(err, DirectMLError::Declined(_)),
                "{kind:?} must decline, got {err:?}"
            );
            assert!(
                format!("{err}").contains(kind.as_str()),
                "the decline must name the reduction it declined"
            );
        }
    }

    #[test]
    fn conv_declines_rather_than_failing() {
        let backend = Backend::declining_for_tests();
        let plan = ConvPlan::conv(&[1, 1, 5, 5], &[1, 1, 3, 3], None, &[], &[], &[], 1).unwrap();
        let input = vec![0.0f32; 25];
        let weight = vec![0.0f32; 9];

        let err = backend.conv(&plan, &input, &weight, None).unwrap_err();
        assert!(
            matches!(err, DirectMLError::Declined(_)),
            "the stub must DECLINE (→ CPU fallback), not FAIL; got {err:?}"
        );
    }
}
