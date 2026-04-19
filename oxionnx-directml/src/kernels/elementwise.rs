//! DirectML elementwise kernels: Add, Mul, Relu, Sigmoid.
//!
//! All functions are scaffolded and return `Err` in this wave so that the
//! dispatch layer falls back to CPU.  Wave 3 will replace each stub with a
//! compiled HLSL pipeline bound to the `DirectMLContext` device.
//!
//! Every public item in this module is gated behind
//! `#[cfg(target_os = "windows")]` because the dispatch layer only calls
//! these functions on Windows.  The HLSL shader constants are always available
//! for tooling / documentation purposes.

#[cfg(target_os = "windows")]
use oxionnx_core::Tensor;

#[cfg(target_os = "windows")]
use crate::{DirectMLContext, DirectMLError};

/// Element-wise addition: `out[i] = a[i] + b[i]`.
///
/// Shapes of `a` and `b` must match (broadcasting is deferred to a later wave).
#[cfg(target_os = "windows")]
pub fn dml_add(a: &Tensor, b: &Tensor, _ctx: &DirectMLContext) -> Result<Tensor, DirectMLError> {
    if a.shape != b.shape {
        return Err(DirectMLError::DispatchFailed(format!(
            "Add: shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    // TODO(Wave3): dispatch ADD_HLSL shader.
    Err(DirectMLError::DispatchFailed(
        "Add HLSL shader not yet compiled — falling back to CPU".into(),
    ))
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
///
/// Shapes of `a` and `b` must match (broadcasting is deferred to a later wave).
#[cfg(target_os = "windows")]
pub fn dml_mul(a: &Tensor, b: &Tensor, _ctx: &DirectMLContext) -> Result<Tensor, DirectMLError> {
    if a.shape != b.shape {
        return Err(DirectMLError::DispatchFailed(format!(
            "Mul: shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    // TODO(Wave3): dispatch MUL_HLSL shader.
    Err(DirectMLError::DispatchFailed(
        "Mul HLSL shader not yet compiled — falling back to CPU".into(),
    ))
}

/// Rectified Linear Unit: `out[i] = max(0, a[i])`.
#[cfg(target_os = "windows")]
pub fn dml_relu(a: &Tensor, _ctx: &DirectMLContext) -> Result<Tensor, DirectMLError> {
    let _ = a;
    // TODO(Wave3): dispatch RELU_HLSL shader.
    Err(DirectMLError::DispatchFailed(
        "Relu HLSL shader not yet compiled — falling back to CPU".into(),
    ))
}

/// Sigmoid activation: `out[i] = 1 / (1 + exp(-a[i]))`.
#[cfg(target_os = "windows")]
pub fn dml_sigmoid(a: &Tensor, _ctx: &DirectMLContext) -> Result<Tensor, DirectMLError> {
    let _ = a;
    // TODO(Wave3): dispatch SIGMOID_HLSL shader.
    Err(DirectMLError::DispatchFailed(
        "Sigmoid HLSL shader not yet compiled — falling back to CPU".into(),
    ))
}

/// HLSL source for element-wise binary ops (Add, Mul).
///
/// Swap the operator in the body and recompile for each variant.
#[allow(dead_code)]
pub const ELEMENTWISE_BINARY_HLSL: &str = r"
cbuffer Constants : register(b0) { uint N; uint _pad0; uint _pad1; uint _pad2; }
StructuredBuffer<float>   A : register(t0);
StructuredBuffer<float>   B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(256, 1, 1)]
void main_add(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    C[i] = A[i] + B[i];
}

[numthreads(256, 1, 1)]
void main_mul(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    C[i] = A[i] * B[i];
}
";

/// HLSL source for element-wise unary activation ops (Relu, Sigmoid).
#[allow(dead_code)]
pub const ELEMENTWISE_UNARY_HLSL: &str = r"
cbuffer Constants : register(b0) { uint N; uint _pad0; uint _pad1; uint _pad2; }
StructuredBuffer<float>   A : register(t0);
RWStructuredBuffer<float> C : register(u0);

[numthreads(256, 1, 1)]
void main_relu(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    C[i] = max(0.0, A[i]);
}

[numthreads(256, 1, 1)]
void main_sigmoid(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    C[i] = 1.0 / (1.0 + exp(-A[i]));
}
";
