//! Execution Provider compatibility layer for migration from `ort`.
//!
//! `ort` 2.x exposes a rich set of execution provider (EP) types such as
//! `CUDAExecutionProvider`, `CoreMLExecutionProvider`, etc., that allow
//! callers to configure hardware acceleration at session build time.
//!
//! oxionnx selects its backend at compile time via Cargo feature flags
//! (`gpu`, `cuda`).  These stub types mirror the `ort` EP API surface so
//! that code written against `ort` can compile against oxionnx with only
//! a `use` path change — no call-site edits required.
//!
//! Every `build()` call returns an [`ExecutionProviderDispatch`] no-op token.
//! The actual backend selection is governed by the crate's feature flags.

use oxionnx_core::graph::OpKind;
use std::collections::HashMap;

/// Opaque no-op token returned by EP `.build()` calls.
///
/// Passed to [`crate::SessionBuilder::with_execution_providers`], which
/// accepts but ignores the list.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExecutionProviderDispatch;

// ── CPU ─────────────────────────────────────────────────────────────────────

/// CPU execution provider stub (always active in oxionnx).
#[derive(Debug, Clone, Default)]
pub struct CPUExecutionProvider;

impl CPUExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── CUDA ────────────────────────────────────────────────────────────────────

/// CUDA execution provider stub.
///
/// When the `cuda` feature is enabled, oxionnx routes eligible ops through
/// the CUDA backend automatically.  This type is accepted at the API level
/// for ort-compatible code.
#[derive(Debug, Clone, Default)]
pub struct CUDAExecutionProvider;

impl CUDAExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── CoreML ──────────────────────────────────────────────────────────────────

/// Apple CoreML execution provider stub.
#[derive(Debug, Clone, Default)]
pub struct CoreMLExecutionProvider;

impl CoreMLExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── DirectML ────────────────────────────────────────────────────────────────

/// DirectML (Windows GPU) execution provider stub.
#[derive(Debug, Clone, Default)]
pub struct DirectMLExecutionProvider;

impl DirectMLExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── TensorRT ────────────────────────────────────────────────────────────────

/// NVIDIA TensorRT execution provider stub.
#[derive(Debug, Clone, Default)]
pub struct TensorRTExecutionProvider;

impl TensorRTExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── OpenVINO ────────────────────────────────────────────────────────────────

/// Intel OpenVINO execution provider stub.
#[derive(Debug, Clone, Default)]
pub struct OpenVINOExecutionProvider;

impl OpenVINOExecutionProvider {
    /// Finalise configuration and return an [`ExecutionProviderDispatch`].
    pub fn build(self) -> ExecutionProviderDispatch {
        ExecutionProviderDispatch
    }
}

// ── Operator Placement ──────────────────────────────────────────────────────

/// Controls how operators are assigned to execution providers.
#[derive(Debug, Clone, Default)]
pub enum OpPlacement {
    /// All ops on CPU (default when no GPU feature).
    #[default]
    CpuOnly,
    /// Auto-select based on op type and tensor size thresholds.
    Auto {
        /// Minimum output tensor bytes for GPU dispatch (default: 65536 = 64KB).
        gpu_threshold_bytes: usize,
    },
    /// Manual per-operator placement.
    Manual(HashMap<OpKind, ProviderKind>),
}

/// Which provider to use for an operator invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    /// Pure-Rust CPU execution (always available as fallback).
    Cpu,
    /// wgpu / WebGPU compute backend (requires `gpu` feature).
    #[cfg(feature = "gpu")]
    Gpu,
    /// NVIDIA CUDA backend (requires `cuda` feature).
    #[cfg(feature = "cuda")]
    Cuda,
    /// Microsoft DirectML backend (requires `directml` feature; no-op off Windows).
    #[cfg(feature = "directml")]
    DirectMl,
}

/// Decide placement for a specific operator invocation.
pub fn decide_placement(op: &OpKind, output_bytes: usize, placement: &OpPlacement) -> ProviderKind {
    match placement {
        OpPlacement::CpuOnly => ProviderKind::Cpu,
        OpPlacement::Auto {
            gpu_threshold_bytes,
        } => {
            if output_bytes >= *gpu_threshold_bytes && is_gpu_capable(op) {
                #[cfg(feature = "gpu")]
                return ProviderKind::Gpu;
                #[cfg(not(feature = "gpu"))]
                return ProviderKind::Cpu;
            }
            ProviderKind::Cpu
        }
        OpPlacement::Manual(map) => map.get(op).copied().unwrap_or(ProviderKind::Cpu),
    }
}

/// Check if an operator has a GPU implementation.
pub fn is_gpu_capable(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::MatMul
            | OpKind::Gemm
            | OpKind::Conv
            | OpKind::Add
            | OpKind::Mul
            | OpKind::Sub
            | OpKind::Relu
            | OpKind::Sigmoid
            | OpKind::Softmax
            | OpKind::LayerNorm
            | OpKind::BatchNorm
            | OpKind::Transpose
            | OpKind::ReduceMean
    )
}
