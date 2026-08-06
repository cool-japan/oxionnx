//! # oxionnx-directml
//!
//! **DirectML / Direct3D 12** execution provider for the OxiONNX inference engine.
//!
//! On Windows this crate dispatches a small set of ONNX nodes to the GPU.  On every
//! other platform it is a fully-typed, zero-overhead no-op: the same function
//! signatures exist, [`DirectMLContext::try_new`] returns `None`, and
//! [`try_directml_dispatch`] returns `Ok(None)`, so callers in `oxionnx` carry no
//! `#[cfg]` noise at all.
//!
//! ## Architecture: two backends, DirectML first
//!
//! ```text
//! DirectMLContext                       (Send + Sync, Mutex-guarded)
//!   └─ Mutex<Backend>
//!        ├─ D3d12Core     device + COMPUTE queue + allocator + list + fence + event
//!        └─ Engine
//!             ├─ DmlEngine    IDMLDevice — genuine DirectML operators
//!             └─ HlslEngine   D3DCompile'd compute shaders — the fallback
//! ```
//!
//! `DMLCreateDevice` is resolved with `LoadLibraryW` + `GetProcAddress`, never
//! statically linked.  A static import is resolved by the loader at *process start*,
//! so if `DirectML.dll` is absent the host process would fail to **launch** — and
//! the HLSL fallback, which exists precisely for that case, would be unreachable.
//!
//! ## Where the logic lives
//!
//! Almost all of it is **platform-neutral** and unit-tested on Linux, where there is
//! no Windows host and no D3D12 GPU:
//!
//! | Module | Compiled on | What it owns |
//! |---|---|---|
//! | [`plan`] | every target | shape validation, dispatch-grid math, buffer sizing, root constants |
//! | [`layout`] | every target | DirectML tensor descriptors, strides, `TotalTensorSizeInBytes`, the op cache key |
//! | [`mod@reference`] | every target | the CPU oracle the GPU path is diffed against |
//! | [`hlsl`] | every target | the shader sources |
//! | `backend::*` | Windows only | thin FFI glue, and nothing else |
//!
//! The Windows-only glue is *type-checked and lint-checked* from Linux via
//! `cargo clippy --target x86_64-pc-windows-gnu`, which needs no linker and no
//! Windows host.  It is **not** executed by anything in this repository.  See the
//! crate README for exactly what is verified and what is not.
//!
//! ## The `Ok(None)` contract
//!
//! ```text
//! try_directml_dispatch
//!   ├─ Ok(Some(tensors))  the GPU computed it
//!   ├─ Ok(None)           this backend declined, or the kernel failed → run the CPU op
//!   └─ Err(_)             a structural problem the CPU op would hit too (missing input)
//! ```
//!
//! A *declined* op is a correct op: `oxionnx-ops`' tuned CPU kernel runs and produces
//! the right answer.  This crate declines rather than guesses — see
//! [`plan::MatMulPlan::matmul`] and [`plan::ElementwisePlan::binary`] for the two
//! places where guessing would have produced plausible, wrong numbers.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::module_name_repetitions)]

mod backend;
mod context;
mod dispatch;
mod kernels;

pub mod error;
pub mod hlsl;
pub mod layout;
pub mod plan;
pub mod reference;

pub use backend::BackendKind;
pub use context::{Activation, DirectMLContext, FailurePolicy, ACTIVATION_ENV_VAR, STRICT_ENV_VAR};
pub use error::{DirectMLError, Result};
pub use reference::{SelfCheckReport, VERIFY_ENV_VAR};

use std::collections::HashMap;

use oxionnx_core::{
    graph::{Node, OpKind},
    OnnxError, Tensor,
};

/// Does this provider's dispatch table claim `op`?
///
/// A cheap, pure predicate with no device, no lock and no allocation.
///
/// # Why this exists
///
/// `oxionnx`'s parallel session runner marks a node "GPU-eligible" — and therefore
/// routes it through the **serial** GPU phase — based purely on whether an execution
/// provider *exists*.  Without this predicate, merely constructing a
/// [`DirectMLContext`] would drag **every node in the graph** into that serial phase,
/// including the 90 % of them this crate has never had a kernel for.  The result is a
/// GPU provider that makes inference slower.  The runner calls this instead, and only
/// serialises the nodes that can actually be claimed.
///
/// # The invariant
///
/// `dispatch::route` must claim **exactly** the ops for which this returns `true`.
/// Over-reporting drags nodes into the serial phase for nothing; under-reporting
/// silently gives up GPU work.  If you teach the router a new op, teach this function
/// the same op **in the same commit** — `is_supported_op_agrees_with_the_router_table`
/// in this module's tests enumerates every routed [`OpKind`] and fails the moment one is
/// routed without being claimed here (or claimed without being routed).  The next
/// candidates are the ops [`plan`] and [`layout`] already model but the router still
/// declines outright — `LogSoftmax`, `ReduceProd`, `ConvTranspose` — and
/// `is_supported_op_does_not_over_claim` names representatives of the unclaimed set so a
/// premature claim trips a test rather than a graph.
///
/// Note that a `true` here is a claim about the **op kind**, not about the specific
/// node: a `MatMul` with 3-D operands answers `true` here and is still declined by
/// [`plan::MatMulPlan::matmul`] at dispatch time, falling back to CPU.  That is the
/// correct split — this predicate must stay `O(1)` and shape-blind.
#[must_use]
pub fn is_supported_op(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::MatMul
            | OpKind::Gemm
            | OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Relu
            | OpKind::Sigmoid
            | OpKind::Tanh
            | OpKind::Softmax
            | OpKind::ReduceSum
            | OpKind::ReduceMean
            | OpKind::ReduceMax
            | OpKind::ReduceMin
            | OpKind::Conv
    )
}

/// Attempt to dispatch a single ONNX graph node to the DirectML backend.
///
/// # Returns
///
/// - `Ok(Some(tensors))` — the GPU executed the op.
/// - `Ok(None)` — the op is not claimed by this provider, or the backend declined it,
///   or the kernel failed.  The caller must fall through to the CPU path.
/// - `Err(_)` — a structural failure the CPU path would hit too (a required input
///   tensor that is simply not there).
///
/// On non-Windows targets `ctx.is_active()` is a monomorphic `false`, so this whole
/// function folds away to `Ok(None)`.
///
/// # Errors
/// [`oxionnx_core::OnnxError::TensorNotFound`] when a required input is missing.
pub fn try_directml_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &DirectMLContext,
) -> core::result::Result<Option<Vec<Tensor>>, OnnxError> {
    if !ctx.is_active() {
        return Ok(None);
    }
    ctx.with_backend(|backend| dispatch::route(node, weights, intermediates, backend))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn try_new_never_panics_and_is_none_off_windows() {
        let ctx = DirectMLContext::try_new();
        #[cfg(not(target_os = "windows"))]
        assert!(ctx.is_none(), "must be None on every non-Windows target");
        // On Windows this only asserts that acquisition never panics — a machine
        // with no D3D12 adapter legitimately yields `None`.
        let _ = ctx.is_some();
    }

    /// The exact set `dispatch::route` handles, as claimed by `is_supported_op`.
    ///
    /// Kept next to the two tests that consume it so the claimed set is written **once**:
    /// `is_supported_op_claims_exactly_the_documented_set` proves every entry is claimed,
    /// and `is_supported_op_agrees_with_the_router_table` proves the router reaches each.
    const ROUTED_OPS: &[OpKind] = &[
        OpKind::MatMul,
        OpKind::Gemm,
        OpKind::Add,
        OpKind::Sub,
        OpKind::Mul,
        OpKind::Div,
        OpKind::Relu,
        OpKind::Sigmoid,
        OpKind::Tanh,
        OpKind::Softmax,
        OpKind::ReduceSum,
        OpKind::ReduceMean,
        OpKind::ReduceMax,
        OpKind::ReduceMin,
        OpKind::Conv,
    ];

    #[test]
    fn is_supported_op_claims_exactly_the_documented_set() {
        // These fifteen are exactly `dispatch::route`'s table.  The two lists are the same
        // list, and `dispatch::tests::a_declining_backend_becomes_ok_none_and_never_an_err`
        // drives every one of them through the router, so an op added here and forgotten
        // there (or the reverse) fails a test rather than quietly regressing a graph.
        for op in ROUTED_OPS {
            assert!(is_supported_op(op), "{op:?} must be claimed");
        }
    }

    #[test]
    fn is_supported_op_agrees_with_the_router_table() {
        // The invariant, asserted directly: every op the router actually routes MUST be
        // claimed (else the session runner never offers it and the GPU work is silently
        // given up), and a representative op the router declines outright MUST NOT be
        // claimed (else it is dragged into the serial GPU phase to fall back to CPU anyway).
        for op in ROUTED_OPS {
            assert!(is_supported_op(op), "{op:?} is routed but not claimed");
        }
        assert!(
            !is_supported_op(&OpKind::Identity),
            "Identity has no dispatch arm and must not be claimed"
        );
    }

    #[test]
    fn is_supported_op_does_not_over_claim() {
        // Over-claiming is the harmful direction: it drags a node into the session
        // runner's SERIAL GPU phase, where it then falls back to CPU anyway — i.e.
        // it turns a parallel CPU node into a serial CPU node.
        //
        // The ops to watch now are the near-neighbours the router still declines outright:
        // `LogSoftmax` (a distinct activation), `ReduceProd` (a fifth reduction with no
        // plan), and `ConvTranspose` (a different `DML_*_OPERATOR_DESC`).  Each is one small
        // step from something already routed, so each is the most likely accidental claim;
        // they stay unclaimed until a real kernel and a `directml_self_check` land.
        for op in [
            OpKind::Identity,
            OpKind::LogSoftmax,
            OpKind::ReduceProd,
            OpKind::ConvTranspose,
            OpKind::Reshape,
            OpKind::Transpose,
            OpKind::Gather,
            OpKind::LayerNorm,
            OpKind::Unknown("Frobnicate".into()),
        ] {
            assert!(!is_supported_op(&op), "{op:?} must NOT be claimed");
        }
    }

    #[test]
    fn is_supported_op_is_pure() {
        // No device, no context, no lock: callable from a rayon closure per node.
        assert_eq!(
            is_supported_op(&OpKind::MatMul),
            is_supported_op(&OpKind::MatMul)
        );
    }
}
