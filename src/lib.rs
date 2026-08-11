//! oxionnx — Pure Rust ONNX inference engine.
//!
//! Zero C/C++ dependencies. Supports a wide subset of ONNX operators
//! sufficient to run transformer-based models (BERT, T5, GPT-2 family).
//!
//! # Quick start
//!
//! ```no_run
//! use std::collections::HashMap;
//! use oxionnx::{Session, Tensor};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let session = Session::from_file("model.onnx".as_ref())?;
//!     let mut inputs = HashMap::new();
//!     inputs.insert("input_ids", Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]));
//!     let outputs = session.run(&inputs)?;
//!     Ok(())
//! }
//! ```

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use oxionnx_gpu::*;
}
#[cfg(feature = "cuda")]
pub mod cuda {
    //! CUDA-accelerated dispatch for ONNX ops.
    pub use oxionnx_cuda::CudaContext;
    pub use oxionnx_cuda::CudaError;
}
#[cfg(feature = "cuda")]
pub use oxionnx_cuda::CudaContext;
#[cfg(feature = "directml")]
pub mod directml {
    //! DirectML (Windows D3D12 GPU) dispatch for ONNX ops.
    pub use oxionnx_directml::*;
}
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub mod coreml {
    //! Apple CoreML whole-model dispatch for pre-converted `.mlpackage` bundles.
    //!
    //! Re-exports the public surface of [`oxionnx_coreml`].  The
    //! [`MlPackageSession`] alias lifts the inner `MlPackageModel` type to
    //! the user-facing `*Session` naming convention used elsewhere in this
    //! crate (see [`crate::Session`]).
    pub use oxionnx_coreml::{ComputePlanSummary, CoreMLError, MlComputeUnits, MlPackageModel};

    /// Sibling of [`crate::Session`] for CoreML-backed `.mlpackage` models.
    ///
    /// `MlPackageSession` exposes the same I/O contract as
    /// [`crate::Session`] (input/output names + a `predict`-style call) so
    /// callers can swap between the two backends with minimal churn.  It
    /// is implemented as a type alias over
    /// [`oxionnx_coreml::MlPackageModel`].
    pub type MlPackageSession = MlPackageModel;
}
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub use coreml::{ComputePlanSummary, CoreMLError, MlComputeUnits, MlPackageSession};
pub mod graph {
    pub use oxionnx_core::graph::*;
}
pub mod model {
    pub use oxionnx_proto::model::*;
}
pub mod benchmark_select;
pub mod compat;
pub mod execution_providers;
pub mod io_binding;
pub mod macros;
pub mod memory;
pub mod memory_tracker;
#[cfg(feature = "ndarray")]
pub mod ndarray_compat;
pub mod optimizer;
/// Cross-platform monotonic clock (`Instant` panics at runtime on
/// wasm32-unknown-unknown; see the module docs). Internal only.
mod time_compat;
pub mod ops {
    pub use oxionnx_ops::*;
}
pub mod proto {
    pub use oxionnx_proto::parser::*;
    pub use oxionnx_proto::streaming_parser::*;
    pub use oxionnx_proto::types::*;
}
#[cfg(feature = "encryption")]
pub mod encryption;
#[cfg(feature = "wasm")]
pub mod wasm;
#[cfg(feature = "mmap")]
pub mod mmap {
    pub use oxionnx_proto::mmap_loader::*;
}
pub mod opset_coverage;
pub mod session;
pub mod streaming;
pub mod tolerance;
pub mod typed_session;
pub mod tensor {
    pub use oxionnx_core::tensor::*;
}

pub use benchmark_select::{
    benchmark_elementwise, benchmark_matmul, benchmark_suite, format_results, BenchmarkResult,
    ExecutionPath,
};
pub use compat::{GraphOptimizationLevel, SessionOutputs};
pub use execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProviderDispatch, OpenVINOExecutionProvider, ProviderKind,
    TensorRTExecutionProvider,
};
pub use io_binding::IoBinding;
pub use memory::{BufferPool, MemoryPlan, PoolStats, SizeClassPool};
pub use memory_tracker::MemoryTracker;
pub use oxionnx_core::OnnxError;
pub use oxionnx_core::OnnxError as Error;
pub use oxionnx_core::Tensor;
pub use oxionnx_core::{DType, NodeInfo, TensorInfo, TensorStorage, TypedTensor};
#[cfg(feature = "gpu")]
pub use session::GpuExecutionProvider;
pub use session::{
    CancellationToken, ModelInfo, ModelMetadata, NodeProfile, OptLevel, Session, SessionBuilder,
    SessionCacheHeader, SESSION_CACHE_FORMAT_VERSION, SESSION_CACHE_MAGIC,
};
// `run_async`/`spawn_run`/`block_on` start inference on a plain `std::thread`,
// unsupported on wasm32-unknown-unknown (see `session::async_run`'s module
// gate) -- not re-exported there so a caller reaching for them gets a
// compile-time error instead of the runtime panic that used to happen.
#[cfg(not(target_arch = "wasm32"))]
pub use session::{block_on, RunFuture, RunHandle};
pub use streaming::{GenerationConfig, KvCacheBinding, StreamStep, TokenStream};
pub use tolerance::{compare_tensors, tensors_close, ToleranceReport};
pub use typed_session::{Shape, TypedSession};

// Re-export core types for convenience
pub use oxionnx_core::{
    Attributes, Dim, Graph, Node, OpContext, OpKind, Operator, OperatorRegistry,
};

// Re-export registry factory
pub use oxionnx_ops::default_registry;
