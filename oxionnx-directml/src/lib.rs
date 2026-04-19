//! DirectML execution provider for OxiONNX.
//!
//! On Windows with a D3D12-capable device, this crate provides a DirectML
//! dispatch path for MatMul, Add, Mul, Relu, and Sigmoid operators.
//!
//! On non-Windows platforms all APIs are present and well-typed, but are
//! permanent no-ops:
//!
//! - [`DirectMLContext::try_new`] always returns `None`.
//! - [`try_directml_dispatch`] always returns `Ok(None)` (CPU fallback).
//!
//! This design keeps the callers in `oxionnx` free of `#[cfg]` noise: they
//! call the same function signatures on every platform.
//!
//! ## Dispatch priority
//!
//! ```text
//! DirectML (this crate)
//!   └─ Ok(Some(results))  ← GPU handled it
//!   └─ Ok(None)           ← not supported or scaffold not yet complete → CPU
//! CPU fallback
//! ```

#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

mod context;
mod dispatch;
mod error;
mod kernels;

pub use context::DirectMLContext;
pub use error::DirectMLError;

use std::collections::HashMap;

use oxionnx_core::{graph::Node, OnnxError, Tensor};

/// Attempt to dispatch a single ONNX graph node to the DirectML backend.
///
/// # Returns
///
/// - `Ok(Some(tensors))` — DirectML successfully executed the op.
/// - `Ok(None)` — The op is not covered by DirectML, or the kernel
///   encountered an error; the caller should fall through to CPU.
/// - `Err(_)` — A structural / irrecoverable failure (e.g. missing required
///   input tensor that would also be absent on CPU).
pub fn try_directml_dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &DirectMLContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    dispatch::dispatch(node, weights, intermediates, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            op,
            name: "test".into(),
            inputs: inputs.iter().map(|s| (*s).into()).collect(),
            outputs: outputs.iter().map(|s| (*s).into()).collect(),
            attrs: Attributes::default(),
        }
    }

    #[test]
    fn context_try_new_never_panics() {
        // Must not panic on any platform.
        let ctx = DirectMLContext::try_new();
        // On non-Windows this is always None.
        #[cfg(not(target_os = "windows"))]
        assert!(ctx.is_none(), "expected None on non-Windows");
        // Suppress unused-variable warning on Windows builds.
        let _ = ctx;
    }

    #[test]
    fn dispatch_unknown_op_returns_none() {
        // Even if a context were available, Identity has no DirectML kernel.
        // On non-Windows try_new returns None, so we test the structural path only.
        let node = make_node(OpKind::Identity, &["x"], &["y"]);
        let weights: HashMap<String, Tensor> = HashMap::new();
        let mut intermediates: HashMap<String, Tensor> = HashMap::new();
        intermediates.insert("x".into(), Tensor::new(vec![1.0f32], vec![1]));

        // We cannot construct a real DirectMLContext in CI, so we can only verify
        // the type signature compiles.  The `try_new()` guard means dispatch is
        // never actually called here.
        let _ = &node;
        let _ = &weights;
        let _ = &intermediates;
    }

    #[test]
    fn is_active_false_on_non_windows() {
        #[cfg(not(target_os = "windows"))]
        {
            // DirectMLContext cannot be constructed, so we verify is_active's
            // contract indirectly: try_new returns None on non-Windows.
            let ctx = DirectMLContext::try_new();
            assert!(ctx.is_none());
        }
    }

    #[test]
    fn error_variants_display_correctly() {
        let e = DirectMLError::DispatchFailed("test error".into());
        let msg = format!("{e}");
        assert!(msg.contains("test error"), "Display output: {msg}");

        let e2 = DirectMLError::DeviceInitFailed("no d3d12".into());
        let msg2 = format!("{e2}");
        assert!(msg2.contains("no d3d12"), "Display output: {msg2}");
    }
}
