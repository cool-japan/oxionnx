//! DirectML MatMul kernel.
//!
//! The HLSL compute-shader pipeline is scaffolded here.  In the current wave
//! every entry-point returns `Err` so the dispatch layer falls back to CPU.
//! Wave 3 will fill in the real D3D12 resource allocation + shader dispatch.
//!
//! The public function is gated behind `#[cfg(target_os = "windows")]`
//! because the dispatch layer only calls it on Windows.

#[cfg(target_os = "windows")]
use oxionnx_core::Tensor;

#[cfg(target_os = "windows")]
use crate::{DirectMLContext, DirectMLError};

/// Execute a batched MatMul on DirectML.
///
/// `a` has shape `[…, M, K]`, `b` has shape `[…, K, N]`; the result has
/// shape `[…, M, N]`.
#[cfg(target_os = "windows")]
pub fn dml_matmul(a: &Tensor, b: &Tensor, _ctx: &DirectMLContext) -> Result<Tensor, DirectMLError> {
    // Validate shapes eagerly so that shape errors surface even in scaffold mode.
    if a.shape.len() < 2 {
        return Err(DirectMLError::DispatchFailed(format!(
            "MatMul: 'a' must be at least 2-D, got {}D",
            a.shape.len()
        )));
    }
    if b.shape.len() < 2 {
        return Err(DirectMLError::DispatchFailed(format!(
            "MatMul: 'b' must be at least 2-D, got {}D",
            b.shape.len()
        )));
    }

    let k_a = a.shape[a.shape.len() - 1];
    let k_b = b.shape[b.shape.len() - 2];
    if k_a != k_b {
        return Err(DirectMLError::DispatchFailed(format!(
            "MatMul: inner dimension mismatch — a[-1]={k_a} vs b[-2]={k_b}"
        )));
    }

    // TODO(Wave3): compile MATMUL_HLSL, bind SRVs for A/B and UAV for C,
    // dispatch ceil(M/16) × ceil(N/16) × 1 thread groups, wait on fence.
    Err(DirectMLError::DispatchFailed(
        "MatMul HLSL shader not yet compiled — falling back to CPU".into(),
    ))
}

/// HLSL source for the tile-based MatMul compute shader.
///
/// cbuffer layout: `uint M; uint K; uint N; uint _pad;`
/// Resources: `StructuredBuffer<float> A : t0`, `B : t1`,
/// `RWStructuredBuffer<float> C : u0`.
/// Dispatch: `ceil(M/16) × ceil(N/16) × 1`.
#[allow(dead_code)]
pub const MATMUL_HLSL: &str = r"
cbuffer Constants : register(b0) {
    uint M; uint K; uint N; uint _pad;
}
StructuredBuffer<float>   A : register(t0);
StructuredBuffer<float>   B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.y;
    uint col = tid.x;
    if (row >= M || col >= N) return;
    float acc = 0.0;
    for (uint k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
";
