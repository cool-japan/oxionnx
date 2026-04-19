//! Tests for Phase E: DirectML execution provider API surface.
//!
//! These tests verify that the DirectML API is structurally sound and behaves
//! as documented (no-op on non-Windows), without requiring a real GPU device.

use oxionnx::{DirectMLExecutionProvider, SessionBuilder};
use oxionnx_core::{Attributes, Graph, Node, OpKind};
use std::collections::HashMap;

// ── DirectMLExecutionProvider struct tests (always-on, no feature flag) ─────

/// DirectMLExecutionProvider can be constructed and converted to a dispatch.
#[test]
fn test_directml_execution_provider_builds() {
    let ep = DirectMLExecutionProvider;
    let _ = ep.build();
}

/// DirectMLExecutionProvider can be cloned and compared for debug formatting.
#[test]
fn test_directml_execution_provider_debug_clone() {
    let ep = DirectMLExecutionProvider;
    let cloned = ep.clone();
    let dbg = format!("{:?}", cloned);
    assert!(
        dbg.contains("DirectML"),
        "Debug output should contain 'DirectML'"
    );
}

/// SessionBuilder::with_execution_provider accepts a DirectML provider without panicking.
#[test]
fn test_session_builder_with_directml_ep() {
    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Identity,
            name: "id".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    let ep = DirectMLExecutionProvider;
    let dispatch = ep.build();

    let session = SessionBuilder::new()
        .with_execution_providers(vec![dispatch])
        .build_from_graph(graph, HashMap::new())
        .expect("SessionBuilder with DirectML EP should not fail");

    // Session should still be usable for inference on CPU
    use oxionnx::Tensor;
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(vec![1.0f32, 2.0], vec![2]));
    let outputs = session.run(&inputs).expect("run after DirectML EP config");
    let y = outputs.get("y").expect("output 'y'");
    assert_eq!(y.data, vec![1.0f32, 2.0]);
}

// ── DirectMLContext API tests (require `directml` feature) ───────────────────

/// DirectMLContext::try_new never panics; returns None on non-Windows.
#[cfg(feature = "directml")]
#[test]
fn test_directml_context_try_new_non_windows() {
    use oxionnx::directml::DirectMLContext;
    let ctx = DirectMLContext::try_new();
    #[cfg(not(target_os = "windows"))]
    assert!(
        ctx.is_none(),
        "DirectMLContext::try_new must return None on non-Windows"
    );
    let _ = ctx;
}

/// On non-Windows, try_new returns None — is_active is unreachable but
/// the type's API surface compiles and can be called through Option.
#[cfg(feature = "directml")]
#[test]
fn test_directml_context_is_active_false_on_non_windows() {
    use oxionnx::directml::DirectMLContext;
    #[cfg(not(target_os = "windows"))]
    {
        let ctx = DirectMLContext::try_new();
        // We can verify: try_new → None implies is_active would be false
        // by checking the Option is None (is_active is on the struct, not Option).
        assert!(
            ctx.is_none(),
            "No context on non-Windows means no active GPU"
        );
    }
    #[cfg(target_os = "windows")]
    {
        // On Windows, we only verify it compiles and doesn't panic.
        let ctx = DirectMLContext::try_new();
        if let Some(ref c) = ctx {
            let _ = c.is_active();
        }
    }
}

/// try_directml_dispatch returns Ok(None) on non-Windows (CPU fallback signal).
#[cfg(feature = "directml")]
#[test]
fn test_try_directml_dispatch_noop_on_non_windows() {
    #[cfg(not(target_os = "windows"))]
    {
        use oxionnx::directml::DirectMLContext;
        use oxionnx_core::{graph::Node, Tensor};

        // try_new returns None on non-Windows; we cannot call try_directml_dispatch
        // without a context, but the function signature must compile.
        let ctx_opt = DirectMLContext::try_new();
        assert!(ctx_opt.is_none());

        // Confirm the dispatch function is importable at the feature path
        let _ = oxionnx::directml::try_directml_dispatch
            as fn(
                &Node,
                &HashMap<String, Tensor>,
                &HashMap<String, Tensor>,
                &DirectMLContext,
            ) -> Result<Option<Vec<Tensor>>, oxionnx_core::OnnxError>;
    }
}

/// DirectMLError variants display non-empty messages.
#[cfg(feature = "directml")]
#[test]
fn test_directml_error_display() {
    use oxionnx::directml::DirectMLError;

    let e1 = DirectMLError::DispatchFailed("kernel unavailable".into());
    let msg1 = format!("{e1}");
    assert!(!msg1.is_empty(), "DispatchFailed display must be non-empty");
    assert!(msg1.contains("kernel unavailable"), "msg: {msg1}");

    let e2 = DirectMLError::DeviceInitFailed("no d3d12".into());
    let msg2 = format!("{e2}");
    assert!(
        !msg2.is_empty(),
        "DeviceInitFailed display must be non-empty"
    );
    assert!(msg2.contains("no d3d12"), "msg: {msg2}");
}

// ── Phase E: session-level dispatch block integration tests ──────────────────
//
// The following tests exercise the DirectML dispatch block that was wired into
// `session/run.rs` by Wave 3a.  On non-Windows (and on Windows until the HLSL
// pipeline is complete), `dml` is `None` and every op falls back to CPU — so
// all assertions are about CPU-correct output, not GPU acceleration.
//
// These tests run unconditionally (no `#[cfg(feature = "directml")]`) because
// the `DirectMLExecutionProvider` builder stub is always available.

/// Session built with DirectML EP accepts a multi-op graph and produces
/// numerically correct CPU output (GPU path is `Ok(None)` on non-Windows).
#[test]
fn test_directml_session_matmul_add_graph() {
    use oxionnx::Tensor;

    // Graph: input [1, 3] @ W [3, 2] → mm [1, 2]  then  mm + bias [2] → out [1, 2]
    let graph = Graph {
        nodes: vec![
            Node {
                op: OpKind::MatMul,
                name: "mm".to_string(),
                inputs: vec!["x".to_string(), "W".to_string()],
                outputs: vec!["mm_out".to_string()],
                attrs: Attributes::default(),
            },
            Node {
                op: OpKind::Add,
                name: "add".to_string(),
                inputs: vec!["mm_out".to_string(), "bias".to_string()],
                outputs: vec!["out".to_string()],
                attrs: Attributes::default(),
            },
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["out".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    let mut weights = HashMap::new();
    // W = identity-like [3, 2]: [[1, 0], [0, 1], [1, 1]]
    weights.insert(
        "W".to_string(),
        Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]),
    );
    // bias = [0.5, -0.5]
    weights.insert("bias".to_string(), Tensor::new(vec![0.5, -0.5], vec![2]));

    let ep = DirectMLExecutionProvider;
    let dispatch = ep.build();

    let session = SessionBuilder::new()
        .with_execution_providers(vec![dispatch])
        .with_optimization_level(oxionnx::OptLevel::None)
        .build_from_graph(graph, weights)
        .expect("Session with DirectML EP must build successfully");

    // x = [1, 2, 3]
    // mm = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    // out = [4+0.5, 5-0.5] = [4.5, 4.5]
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(vec![1.0f32, 2.0, 3.0], vec![1, 3]));
    let outputs = session.run(&inputs).expect("run with DirectML EP session");

    let out = outputs.get("out").expect("output 'out'");
    assert_eq!(out.shape, vec![1, 2]);
    for (i, (&v, &expected)) in out.data.iter().zip([4.5f32, 4.5].iter()).enumerate() {
        assert!(
            (v - expected).abs() < 1e-5,
            "out[{i}]: got {v}, expected {expected}"
        );
    }
}

/// Session with DirectML EP runs Relu correctly via CPU fallback.
#[test]
fn test_directml_session_relu_cpu_fallback() {
    use oxionnx::Tensor;

    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    let ep = DirectMLExecutionProvider;
    let session = SessionBuilder::new()
        .with_execution_providers(vec![ep.build()])
        .with_optimization_level(oxionnx::OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x",
        Tensor::new(vec![-3.0f32, -1.0, 0.0, 2.0, 5.0], vec![5]),
    );
    let outputs = session.run(&inputs).expect("run");
    let y = outputs.get("y").expect("output 'y'");
    assert_eq!(y.data, vec![0.0f32, 0.0, 0.0, 2.0, 5.0]);
    assert_eq!(y.shape, vec![5]);
}

/// Running with the DirectML EP multiple times on the same session is safe.
///
/// The dispatch block must not consume or corrupt internal state between calls.
#[test]
fn test_directml_session_multiple_runs_stable() {
    use oxionnx::Tensor;

    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Identity,
            name: "id".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    let session = SessionBuilder::new()
        .with_execution_providers(vec![DirectMLExecutionProvider.build()])
        .with_optimization_level(oxionnx::OptLevel::None)
        .build_from_graph(graph, HashMap::new())
        .expect("build");

    for run_idx in 0..3u32 {
        let val = run_idx as f32;
        let mut inputs = HashMap::new();
        inputs.insert("x", Tensor::new(vec![val, val + 1.0, val + 2.0], vec![3]));
        let outputs = session.run(&inputs).expect("run {run_idx}");
        let y = outputs.get("y").expect("output 'y'");
        assert_eq!(
            y.data,
            vec![val, val + 1.0, val + 2.0],
            "run {run_idx} mismatch"
        );
    }
}
