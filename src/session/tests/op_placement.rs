use super::super::types::OptLevel;
use super::super::SessionBuilder;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

#[test]
fn test_op_placement_cpu_only() {
    use crate::execution_providers::{decide_placement, OpPlacement, ProviderKind};
    let placement = OpPlacement::CpuOnly;
    let ops = [
        OpKind::MatMul,
        OpKind::Conv,
        OpKind::Add,
        OpKind::Reshape,
        OpKind::Softmax,
        OpKind::Relu,
    ];
    for op in &ops {
        let result = decide_placement(op, 1_000_000, &placement);
        assert_eq!(
            result,
            ProviderKind::Cpu,
            "CpuOnly must always return Cpu for {:?}",
            op
        );
    }
}

#[test]
fn test_op_placement_auto_small_input() {
    use crate::execution_providers::{decide_placement, OpPlacement, ProviderKind};
    // Threshold 64KB; input is only 100 bytes → should stay on CPU
    let placement = OpPlacement::Auto {
        gpu_threshold_bytes: 65536,
    };
    let result = decide_placement(&OpKind::MatMul, 100, &placement);
    assert_eq!(result, ProviderKind::Cpu);
}

#[test]
fn test_op_placement_auto_threshold() {
    use crate::execution_providers::{decide_placement, OpPlacement, ProviderKind};
    let placement = OpPlacement::Auto {
        gpu_threshold_bytes: 1024,
    };

    // Below threshold → CPU, regardless of which accelerators are compiled in.
    // `Auto`'s threshold binds *every* provider equally — CUDA and DirectML
    // included, not just the wgpu path.
    let below = decide_placement(&OpKind::MatMul, 512, &placement);
    assert_eq!(below, ProviderKind::Cpu);

    // At/above threshold → the highest-priority *compiled-in* accelerator that
    // actually implements MatMul, per the mandated `Cuda > DirectMl > Gpu`
    // priority order. MatMul has a kernel in all three backends, so exactly
    // one of the branches below survives per feature combination.
    let at = decide_placement(&OpKind::MatMul, 1024, &placement);
    #[cfg(feature = "cuda")]
    assert_eq!(
        at,
        ProviderKind::Cuda,
        "cuda outranks directml and gpu whenever it is compiled in"
    );
    #[cfg(all(not(feature = "cuda"), feature = "directml"))]
    assert_eq!(
        at,
        ProviderKind::DirectMl,
        "directml outranks gpu when cuda is not compiled in"
    );
    #[cfg(all(not(feature = "cuda"), not(feature = "directml"), feature = "gpu"))]
    assert_eq!(
        at,
        ProviderKind::Gpu,
        "gpu is the lowest-priority accelerator, selected only when cuda and directml are absent"
    );
    #[cfg(not(any(feature = "cuda", feature = "directml", feature = "gpu")))]
    assert_eq!(
        at,
        ProviderKind::Cpu,
        "with no accelerator compiled in, Auto can only ever offer Cpu"
    );

    // Non-GPU-capable op above threshold → still CPU: no backend, of any
    // kind, implements Reshape as an accelerated kernel.
    let reshape = decide_placement(&OpKind::Reshape, 2048, &placement);
    assert_eq!(reshape, ProviderKind::Cpu);
}

#[test]
fn test_op_placement_manual() {
    use crate::execution_providers::{decide_placement, OpPlacement, ProviderKind};
    let mut map = HashMap::new();
    #[cfg(feature = "gpu")]
    {
        map.insert(OpKind::MatMul, ProviderKind::Gpu);
    }
    #[cfg(not(feature = "gpu"))]
    {
        // Without gpu feature, just map to Cpu to test lookup works
        map.insert(OpKind::MatMul, ProviderKind::Cpu);
    }
    let placement = OpPlacement::Manual(map);

    // 65_536 bytes is well above `MIN_GPU_DISPATCH_BYTES` (4096), so the
    // accelerator pin is honoured rather than overridden by the hard size
    // floor that `Manual` now enforces on every accelerator pin.
    let matmul_result = decide_placement(&OpKind::MatMul, 65_536, &placement);
    #[cfg(feature = "gpu")]
    assert_eq!(matmul_result, ProviderKind::Gpu);
    #[cfg(not(feature = "gpu"))]
    assert_eq!(matmul_result, ProviderKind::Cpu);

    // Unmapped op defaults to Cpu regardless of size.
    let reshape_result = decide_placement(&OpKind::Reshape, 65_536, &placement);
    assert_eq!(reshape_result, ProviderKind::Cpu);
}

#[test]
fn test_decide_placement_default() {
    use crate::execution_providers::{decide_placement, OpPlacement, ProviderKind};
    let placement = OpPlacement::default();
    let result = decide_placement(&OpKind::Add, 999999, &placement);
    assert_eq!(result, ProviderKind::Cpu);
}

#[test]
fn test_is_gpu_capable_matmul() {
    use crate::execution_providers::is_gpu_capable;
    assert!(is_gpu_capable(&OpKind::MatMul));
    // [a7-19] `is_gpu_capable` is now derived from `GPU_DISPATCH_OPS`, the
    // exact set of ops `try_gpu_dispatch` has a real match arm for. `Gemm`
    // has no arm (it falls through to the `_ => Ok(None)` catch-all), so
    // asserting it GPU-capable encoded the old, three-way-disagreeing
    // behaviour where `is_gpu_capable` claimed ops the dispatcher would
    // immediately bounce back to CPU. See
    // `execution_providers::is_gpu_capable_matches_try_gpu_dispatch_arms`
    // for the exhaustive version of this check.
    assert!(!is_gpu_capable(&OpKind::Gemm));
    assert!(is_gpu_capable(&OpKind::Conv));
    assert!(is_gpu_capable(&OpKind::Softmax));
    assert!(is_gpu_capable(&OpKind::Relu));
    assert!(is_gpu_capable(&OpKind::ReduceMean));
}

#[test]
fn test_is_gpu_capable_reshape() {
    use crate::execution_providers::is_gpu_capable;
    assert!(!is_gpu_capable(&OpKind::Reshape));
    assert!(!is_gpu_capable(&OpKind::Squeeze));
    assert!(!is_gpu_capable(&OpKind::Flatten));
    assert!(!is_gpu_capable(&OpKind::Gather));
    assert!(!is_gpu_capable(&OpKind::Shape));
}

#[test]
fn test_builder_op_placement_api() {
    use crate::execution_providers::OpPlacement;
    let builder = SessionBuilder::new().with_op_placement(OpPlacement::Auto {
        gpu_threshold_bytes: 4096,
    });
    match &builder.op_placement {
        OpPlacement::Auto {
            gpu_threshold_bytes,
        } => {
            assert_eq!(*gpu_threshold_bytes, 4096);
        }
        other => panic!("Expected Auto, got {:?}", other),
    }

    // Build a simple session with placement to verify end-to-end wiring
    let graph = Graph {
        nodes: vec![Node {
            name: "relu0".to_string(),
            op: OpKind::Relu,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        ..Default::default()
    };
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_op_placement(OpPlacement::Auto {
            gpu_threshold_bytes: 1024,
        })
        .build_from_graph(graph, HashMap::new())
        .expect("build with op placement");

    // Session should run correctly with placement configured
    let input = Tensor::new(vec![-1.0, 2.0, -3.0], vec![1, 3]);
    let out = session.run_one("input", input).expect("run");
    let y = out.get("output").expect("output");
    assert_eq!(y.data, vec![0.0, 2.0, 0.0]);
}

/// Verify that `with_provider_kinds([ProviderKind::Cpu])` stores one provider.
///
/// This test does NOT require GPU hardware — it only verifies the API stores
/// the provider list correctly and that a CPU-only session executes correctly.
#[test]
fn test_with_provider_kinds_cpu_stores_and_runs() {
    use crate::execution_providers::ProviderKind;
    let builder = SessionBuilder::new().with_provider_kinds([ProviderKind::Cpu]);
    assert_eq!(builder.providers.len(), 1, "providers must have 1 element");
    assert_eq!(
        builder.providers[0],
        ProviderKind::Cpu,
        "first provider must be Cpu"
    );

    // Build and run a simple session to verify CPU fallback works correctly
    // when the provider list is [Cpu].
    let graph = Graph {
        nodes: vec![Node {
            name: "relu_ep".to_string(),
            op: OpKind::Relu,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .with_provider_kinds([ProviderKind::Cpu])
        .build_from_graph(graph, HashMap::new())
        .expect("build with CPU provider kind");

    let input = Tensor::new(vec![-2.0f32, 1.0, -3.0, 4.0], vec![4]);
    let out = session.run_one("x", input).expect("run with provider-list");
    let y = out.get("y").expect("output y");
    assert_eq!(y.data, vec![0.0, 1.0, 0.0, 4.0]);
    assert_eq!(y.shape, vec![4]);
}

/// Verify that an empty provider list (default) preserves legacy behavior.
///
/// Calling `build_from_graph` without `with_provider_kinds` must still work
/// correctly — the providers list is empty and the legacy dispatch path is used.
#[test]
fn test_empty_provider_list_uses_legacy_dispatch() {
    let graph = Graph {
        nodes: vec![Node {
            name: "relu_legacy".to_string(),
            op: OpKind::Relu,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let session = SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        // No with_provider_kinds call — providers is empty Vec
        .build_from_graph(graph, HashMap::new())
        .expect("build with empty provider list (legacy)");

    // providers must be empty
    assert!(
        session.providers.is_empty(),
        "default build must have empty providers list"
    );

    let input = Tensor::new(vec![-5.0f32, 0.0, 3.0], vec![3]);
    let out = session
        .run_one("x", input)
        .expect("run with empty providers");
    let y = out.get("y").expect("output y");
    assert_eq!(y.data, vec![0.0, 0.0, 3.0]);
}

/// Verify `with_provider_kinds` with multiple providers stores them in order.
#[test]
fn test_with_provider_kinds_multiple_providers_order() {
    use crate::execution_providers::ProviderKind;
    // Only Cpu is guaranteed to be available without feature flags, but we
    // can verify the storage order by using Cpu twice (or once and checking).
    let builder = SessionBuilder::new().with_provider_kinds([ProviderKind::Cpu, ProviderKind::Cpu]);
    assert_eq!(builder.providers.len(), 2);
    assert_eq!(builder.providers[0], ProviderKind::Cpu);
    assert_eq!(builder.providers[1], ProviderKind::Cpu);
}
