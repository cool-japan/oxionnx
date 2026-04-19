//! DirectML dispatch router.
//!
//! Matches each supported ONNX op to its DirectML kernel.  On non-Windows
//! platforms this function is a no-op that always returns `Ok(None)`.
//!
//! On Windows, kernel functions currently return `Err` (scaffold wave).
//! Those errors are **swallowed** here and converted to `Ok(None)` so that
//! the session runner always has a clean CPU fallback path.  A hard
//! `Err(OnnxError::…)` is only propagated for irrecoverable structural
//! problems (missing inputs that are also absent on CPU).

use std::collections::HashMap;

use oxionnx_core::{graph::Node, OnnxError, Tensor};

use crate::DirectMLContext;

/// Route a single ONNX node to the appropriate DirectML kernel.
///
/// # Returns
///
/// - `Ok(Some(tensors))` — DirectML handled the op successfully.
/// - `Ok(None)` — op is not supported, or the kernel returned an error;
///   the caller should fall back to wgpu/CPU.
/// - `Err(_)` — structural error (missing inputs) that would also fail on CPU.
pub fn dispatch(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &DirectMLContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    dispatch_impl(node, weights, intermediates, ctx)
}

/// Non-Windows implementation: always yield to the next provider.
#[cfg(not(target_os = "windows"))]
fn dispatch_impl(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &DirectMLContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    let _ = (node, weights, intermediates, ctx);
    Ok(None)
}

/// Windows implementation: route to the DirectML kernel, swallowing errors
/// as `Ok(None)` for CPU fallback.
#[cfg(target_os = "windows")]
fn dispatch_impl(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &DirectMLContext,
) -> Result<Option<Vec<Tensor>>, OnnxError> {
    use oxionnx_core::graph::OpKind;

    use crate::kernels;

    let resolve = |name: &str| -> Option<&Tensor> {
        if name.is_empty() {
            None
        } else {
            intermediates.get(name).or_else(|| weights.get(name))
        }
    };

    match &node.op {
        OpKind::MatMul => {
            let a = resolve(node.inputs.first().map(String::as_str).unwrap_or(""))
                .ok_or_else(|| OnnxError::TensorNotFound("MatMul: missing input A".into()))?;
            let b = resolve(node.inputs.get(1).map(String::as_str).unwrap_or(""))
                .ok_or_else(|| OnnxError::TensorNotFound("MatMul: missing input B".into()))?;
            // Swallow kernel errors — fall back to CPU on any failure.
            Ok(kernels::matmul::dml_matmul(a, b, ctx).ok().map(|t| vec![t]))
        }

        OpKind::Add => {
            let a = resolve(node.inputs.first().map(String::as_str).unwrap_or(""))
                .ok_or_else(|| OnnxError::TensorNotFound("Add: missing input A".into()))?;
            let b = resolve(node.inputs.get(1).map(String::as_str).unwrap_or(""))
                .ok_or_else(|| OnnxError::TensorNotFound("Add: missing input B".into()))?;
            Ok(kernels::elementwise::dml_add(a, b, ctx)
                .ok()
                .map(|t| vec![t]))
        }

        OpKind::Mul => {
            let a = resolve(node.inputs.first().map(String::as_str).unwrap_or(""))
                .ok_or_else(|| OnnxError::TensorNotFound("Mul: missing input A".into()))?;
            let b = resolve(node.inputs.get(1).map(String::as_str).unwrap_or(""))
                .ok_or_else(|| OnnxError::TensorNotFound("Mul: missing input B".into()))?;
            Ok(kernels::elementwise::dml_mul(a, b, ctx)
                .ok()
                .map(|t| vec![t]))
        }

        OpKind::Relu => {
            let a = resolve(node.inputs.first().map(String::as_str).unwrap_or(""))
                .ok_or_else(|| OnnxError::TensorNotFound("Relu: missing input".into()))?;
            Ok(kernels::elementwise::dml_relu(a, ctx).ok().map(|t| vec![t]))
        }

        OpKind::Sigmoid => {
            let a = resolve(node.inputs.first().map(String::as_str).unwrap_or(""))
                .ok_or_else(|| OnnxError::TensorNotFound("Sigmoid: missing input".into()))?;
            Ok(kernels::elementwise::dml_sigmoid(a, ctx)
                .ok()
                .map(|t| vec![t]))
        }

        // All other ops: not handled by DirectML — fall back to CPU.
        _ => Ok(None),
    }
}
