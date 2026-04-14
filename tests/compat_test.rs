//! Tests for the ort-compatibility layer (Task 0.3 + 0.4).
//!
//! Verifies that type aliases, execution provider stubs, and
//! SessionBuilder compat methods compile and behave correctly.

use oxionnx::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProviderDispatch, GraphOptimizationLevel,
    OpenVINOExecutionProvider, OptLevel, Session, SessionBuilder, TensorRTExecutionProvider,
};

// ── GraphOptimizationLevel alias ───────────────────────────────────────────

#[test]
fn test_graph_optimization_level_alias() {
    let _level: GraphOptimizationLevel = OptLevel::All;
    let _level2: GraphOptimizationLevel = GraphOptimizationLevel::Basic;
    // Alias is identical to OptLevel — comparison must succeed.
    assert_eq!(OptLevel::All, GraphOptimizationLevel::All);
}

// ── SessionOutputs type alias ──────────────────────────────────────────────

#[test]
fn test_session_outputs_type() {
    use oxionnx::{SessionOutputs, Tensor};
    let mut outputs: SessionOutputs<'_> = std::collections::HashMap::new();
    outputs.insert("logits".to_string(), Tensor::zeros(&[1, 10]));
    assert_eq!(outputs.len(), 1);
}

// ── Execution provider stubs compile ──────────────────────────────────────

#[test]
fn test_execution_provider_dispatch_default() {
    let _ep: ExecutionProviderDispatch = ExecutionProviderDispatch;
}

#[test]
fn test_cpu_ep_build() {
    let _ep: ExecutionProviderDispatch = CPUExecutionProvider.build();
}

#[test]
fn test_cuda_ep_build() {
    let _ep: ExecutionProviderDispatch = CUDAExecutionProvider.build();
}

#[test]
fn test_coreml_ep_build() {
    let _ep: ExecutionProviderDispatch = CoreMLExecutionProvider.build();
}

#[test]
fn test_directml_ep_build() {
    let _ep: ExecutionProviderDispatch = DirectMLExecutionProvider.build();
}

#[test]
fn test_tensorrt_ep_build() {
    let _ep: ExecutionProviderDispatch = TensorRTExecutionProvider.build();
}

#[test]
fn test_openvino_ep_build() {
    let _ep: ExecutionProviderDispatch = OpenVINOExecutionProvider.build();
}

// ── SessionBuilder compat methods ─────────────────────────────────────────

#[test]
fn test_session_builder_with_intra_threads_noop() {
    // Should not panic or change behaviour — just return self.
    let _builder: SessionBuilder = SessionBuilder::new().with_intra_threads(4);
}

#[test]
fn test_session_builder_with_inter_threads_noop() {
    let _builder: SessionBuilder = SessionBuilder::new().with_inter_threads(2);
}

#[test]
fn test_session_builder_with_execution_providers_noop() {
    let providers = vec![CUDAExecutionProvider.build(), CPUExecutionProvider.build()];
    let _builder: SessionBuilder = SessionBuilder::new().with_execution_providers(providers);
}

#[test]
fn test_session_builder_with_execution_providers_empty() {
    let _builder: SessionBuilder =
        SessionBuilder::new().with_execution_providers(std::iter::empty());
}

// ── commit_from_memory (alias for load_from_bytes) ────────────────────────

#[test]
fn test_commit_from_memory_invalid_bytes() {
    let bad_bytes = b"not-a-valid-onnx-model";
    let result: Result<Session, _> = SessionBuilder::new().commit_from_memory(bad_bytes);
    // Should fail gracefully, not panic.
    assert!(
        result.is_err(),
        "commit_from_memory with invalid bytes should return Err"
    );
}

#[test]
fn test_commit_from_file_missing_path() {
    let result: Result<Session, _> =
        SessionBuilder::new().commit_from_file("/non/existent/path.onnx");
    assert!(
        result.is_err(),
        "commit_from_file with missing file should return Err"
    );
}
