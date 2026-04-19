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

    // Below threshold → CPU
    let below = decide_placement(&OpKind::MatMul, 512, &placement);
    assert_eq!(below, ProviderKind::Cpu);

    // At threshold → GPU-capable op should request GPU (returns Cpu without feature)
    let at = decide_placement(&OpKind::MatMul, 1024, &placement);
    // Without the gpu feature, result is Cpu; with gpu feature, result is Gpu
    #[cfg(feature = "gpu")]
    assert_eq!(at, ProviderKind::Gpu);
    #[cfg(not(feature = "gpu"))]
    assert_eq!(at, ProviderKind::Cpu);

    // Non-GPU-capable op above threshold → still CPU
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

    let matmul_result = decide_placement(&OpKind::MatMul, 0, &placement);
    #[cfg(feature = "gpu")]
    assert_eq!(matmul_result, ProviderKind::Gpu);
    #[cfg(not(feature = "gpu"))]
    assert_eq!(matmul_result, ProviderKind::Cpu);

    // Unmapped op defaults to Cpu
    let reshape_result = decide_placement(&OpKind::Reshape, 0, &placement);
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
    assert!(is_gpu_capable(&OpKind::Gemm));
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
