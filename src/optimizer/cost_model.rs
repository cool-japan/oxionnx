//! Graph cost model and critical-path scheduling for ONNX inference.
//!
//! Estimates per-node computation cost (approximately proportional to FLOPs)
//! and computes the critical path from each node to graph outputs. This
//! information is used to sort nodes within each topological depth level so
//! that the heaviest work starts first under rayon's work-stealing scheduler.

use crate::graph::{Node, OpKind};
use std::collections::HashMap;

/// Estimated cost units (approximately proportional to FLOPs).
///
/// When a `shape_cache` is provided the output-tensor volume is used to scale
/// the per-op multiplier.  Without shape information a default volume of 1 is
/// assumed, so the relative ordering across op types is still meaningful.
pub fn estimate_node_cost(node: &Node, shape_cache: Option<&HashMap<String, Vec<usize>>>) -> u64 {
    let output_volume = node
        .outputs
        .first()
        .and_then(|name| shape_cache.and_then(|c| c.get(name)))
        .map(|s| s.iter().copied().product::<usize>().max(1))
        .unwrap_or(1) as u64; // acceptable in cost estimation – non-critical

    match &node.op {
        OpKind::MatMul | OpKind::Gemm => output_volume * 100,
        OpKind::Conv | OpKind::ConvAddRelu => output_volume * 200,
        OpKind::ConvTranspose => output_volume * 200,
        OpKind::BatchNorm | OpKind::LayerNorm | OpKind::GroupNorm | OpKind::RMSNorm => {
            output_volume * 10
        }
        OpKind::Softmax | OpKind::LogSoftmax => output_volume * 20,
        OpKind::Relu
        | OpKind::Sigmoid
        | OpKind::Tanh
        | OpKind::Gelu
        | OpKind::SiLU
        | OpKind::HardSigmoid
        | OpKind::HardSwish
        | OpKind::LeakyRelu
        | OpKind::PRelu
        | OpKind::Mish
        | OpKind::Celu
        | OpKind::Elu
        | OpKind::Selu
        | OpKind::Softplus
        | OpKind::Softsign
        | OpKind::ThresholdedRelu
        | OpKind::Erf => output_volume * 2,
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => output_volume,
        OpKind::Reshape
        | OpKind::Squeeze
        | OpKind::Unsqueeze
        | OpKind::Transpose
        | OpKind::Flatten
        | OpKind::Identity
        | OpKind::Shape
        | OpKind::Cast => 1,
        OpKind::ReduceSum
        | OpKind::ReduceMean
        | OpKind::ReduceMax
        | OpKind::ReduceMin
        | OpKind::ReduceProd
        | OpKind::ReduceL1
        | OpKind::ReduceL2
        | OpKind::ReduceLogSum
        | OpKind::ReduceLogSumExp
        | OpKind::ReduceSumSquare => output_volume * 5,
        OpKind::Attention | OpKind::MultiHeadAttention => output_volume * 300,
        OpKind::LSTM | OpKind::GRU => output_volume * 150,
        OpKind::Einsum => output_volume * 100,
        _ => output_volume * 5,
    }
}

/// Compute the critical-path cost from each node to graph outputs.
///
/// Returns a `Vec<u64>` parallel to `nodes` where `result[i]` is the cost of
/// the longest remaining path from node `i` to any graph sink (inclusive of
/// node `i`'s own cost).
///
/// Algorithm: reverse topological traversal.  For each node:
///
/// ```text
/// critical[i] = estimate_node_cost(i) + max(critical[j] for j in successors(i))
/// ```
pub fn compute_critical_path_costs(
    nodes: &[Node],
    shape_cache: Option<&HashMap<String, Vec<usize>>>,
) -> Vec<u64> {
    let n = nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Build a successor map: node_index -> list of successor node indices.
    // A successor of node `i` is any node that consumes one of `i`'s outputs.
    let mut output_to_node: HashMap<&str, usize> = HashMap::with_capacity(n);
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            if !out.is_empty() {
                output_to_node.insert(out.as_str(), i);
            }
        }
    }

    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, node) in nodes.iter().enumerate() {
        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            if let Some(&producer_idx) = output_to_node.get(inp.as_str()) {
                if producer_idx != j {
                    successors[producer_idx].push(j);
                }
            }
        }
    }

    // Reverse traversal: process nodes from last to first (assumes topological order).
    let mut critical = vec![0u64; n];
    for i in (0..n).rev() {
        let own_cost = estimate_node_cost(&nodes[i], shape_cache);
        let max_successor_cost = successors[i]
            .iter()
            .map(|&j| critical[j])
            .max()
            .unwrap_or(0);
        critical[i] = own_cost + max_successor_cost;
    }

    critical
}

/// Compute an execution schedule: nodes grouped by topological depth,
/// sorted within each level by critical-path cost (descending, heaviest first).
///
/// This is exported for use by the session runtime and for testing.
pub fn compute_execution_schedule(
    sorted_nodes: &[Node],
    weights: &HashMap<String, crate::tensor::Tensor>,
    shape_cache: Option<&HashMap<String, Vec<usize>>>,
) -> Vec<Vec<usize>> {
    if sorted_nodes.is_empty() {
        return Vec::new();
    }

    use crate::session::Session;

    let depths = Session::compute_node_depths(sorted_nodes, weights);
    let mut groups = Session::group_by_depth(&depths);
    let critical = compute_critical_path_costs(sorted_nodes, shape_cache);

    for group in &mut groups {
        group.sort_by(|&a, &b| critical[b].cmp(&critical[a]));
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Graph, Node, OpKind};
    use crate::session::Session;
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    /// Helper: create a minimal node for cost-model testing.
    fn make_node(op: OpKind, name: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> Node {
        Node {
            op,
            name: name.to_string(),
            inputs: inputs.into_iter().map(|s| s.to_string()).collect(),
            outputs: outputs.into_iter().map(|s| s.to_string()).collect(),
            attrs: Attributes::default(),
        }
    }

    /// Helper: build a shape cache entry.
    fn shape_entry(name: &str, shape: Vec<usize>) -> (String, Vec<usize>) {
        (name.to_string(), shape)
    }

    // ---------------------------------------------------------------
    // 1. MatMul cost > Relu cost for same output shape
    // ---------------------------------------------------------------
    #[test]
    fn test_cost_matmul_higher_than_relu() {
        let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["x"], vec!["relu_out"]);
        let cache: HashMap<String, Vec<usize>> = [
            shape_entry("mm_out", vec![4, 64]),
            shape_entry("relu_out", vec![4, 64]),
        ]
        .into_iter()
        .collect();

        let cost_mm = estimate_node_cost(&matmul, Some(&cache));
        let cost_relu = estimate_node_cost(&relu, Some(&cache));
        assert!(
            cost_mm > cost_relu,
            "MatMul cost {} should be > Relu cost {}",
            cost_mm,
            cost_relu
        );
    }

    // ---------------------------------------------------------------
    // 2. Conv cost > MatMul cost for same output shape
    // ---------------------------------------------------------------
    #[test]
    fn test_cost_conv_highest() {
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
        let cache: HashMap<String, Vec<usize>> = [
            shape_entry("conv_out", vec![1, 64, 32, 32]),
            shape_entry("mm_out", vec![1, 64, 32, 32]),
        ]
        .into_iter()
        .collect();

        let cost_conv = estimate_node_cost(&conv, Some(&cache));
        let cost_mm = estimate_node_cost(&matmul, Some(&cache));
        assert!(
            cost_conv > cost_mm,
            "Conv cost {} should be > MatMul cost {}",
            cost_conv,
            cost_mm
        );
    }

    // ---------------------------------------------------------------
    // 3. Reshape/Transpose cost is minimal (== 1)
    // ---------------------------------------------------------------
    #[test]
    fn test_cost_reshape_minimal() {
        let reshape = make_node(OpKind::Reshape, "r", vec!["x", "s"], vec!["r_out"]);
        let transpose = make_node(OpKind::Transpose, "t", vec!["x"], vec!["t_out"]);
        let cache: HashMap<String, Vec<usize>> = [
            shape_entry("r_out", vec![2, 3, 4]),
            shape_entry("t_out", vec![4, 3, 2]),
        ]
        .into_iter()
        .collect();

        assert_eq!(estimate_node_cost(&reshape, Some(&cache)), 1);
        assert_eq!(estimate_node_cost(&transpose, Some(&cache)), 1);
    }

    // ---------------------------------------------------------------
    // 4. Linear chain A→B→C: critical_path(A) = cost(A)+cost(B)+cost(C)
    // ---------------------------------------------------------------
    #[test]
    fn test_critical_path_linear_chain() {
        // A(Relu) -> B(MatMul) -> C(Add)
        let nodes = vec![
            make_node(OpKind::Relu, "a", vec!["input"], vec!["a_out"]),
            make_node(OpKind::MatMul, "b", vec!["a_out", "w"], vec!["b_out"]),
            make_node(OpKind::Add, "c", vec!["b_out", "bias"], vec!["c_out"]),
        ];

        let critical = compute_critical_path_costs(&nodes, None);

        let cost_a = estimate_node_cost(&nodes[0], None);
        let cost_b = estimate_node_cost(&nodes[1], None);
        let cost_c = estimate_node_cost(&nodes[2], None);

        assert_eq!(critical[0], cost_a + cost_b + cost_c);
        assert_eq!(critical[1], cost_b + cost_c);
        assert_eq!(critical[2], cost_c);
    }

    // ---------------------------------------------------------------
    // 5. Diamond: A→{B,C}→D: critical_path(A) = cost(A)+max(B,C)+cost(D)
    // ---------------------------------------------------------------
    #[test]
    fn test_critical_path_diamond() {
        // A produces a_out consumed by both B and C;
        // B (MatMul) and C (Relu) produce b_out / c_out consumed by D (Add).
        let nodes = vec![
            make_node(OpKind::Identity, "a", vec!["input"], vec!["a_out"]),
            make_node(OpKind::MatMul, "b", vec!["a_out", "w"], vec!["b_out"]),
            make_node(OpKind::Relu, "c", vec!["a_out"], vec!["c_out"]),
            make_node(OpKind::Add, "d", vec!["b_out", "c_out"], vec!["d_out"]),
        ];

        let critical = compute_critical_path_costs(&nodes, None);

        let cost_a = estimate_node_cost(&nodes[0], None);
        let cost_b = estimate_node_cost(&nodes[1], None);
        let cost_c = estimate_node_cost(&nodes[2], None);
        let cost_d = estimate_node_cost(&nodes[3], None);

        let max_bc = cost_b.max(cost_c);
        assert_eq!(critical[0], cost_a + max_bc + cost_d);
        assert_eq!(critical[3], cost_d);
    }

    // ---------------------------------------------------------------
    // 6. Heavier nodes within a level come first after scheduling
    // ---------------------------------------------------------------
    #[test]
    fn test_schedule_sorts_by_cost() {
        // Two independent paths from "input": Relu and MatMul at same depth.
        let nodes = vec![
            make_node(OpKind::Relu, "relu", vec!["input"], vec!["relu_out"]),
            make_node(OpKind::MatMul, "mm", vec!["input", "w"], vec!["mm_out"]),
        ];
        let weights: HashMap<String, Tensor> =
            [("w".to_string(), Tensor::new(vec![1.0; 4], vec![2, 2]))]
                .into_iter()
                .collect();

        let schedule = compute_execution_schedule(&nodes, &weights, None);
        // Both nodes are at depth 0; MatMul should come before Relu.
        assert_eq!(schedule.len(), 1);
        let level = &schedule[0];
        assert_eq!(level.len(), 2);
        // First element should be the MatMul (index 1)
        assert_eq!(level[0], 1, "MatMul (heavier) should be scheduled first");
        assert_eq!(level[1], 0, "Relu (lighter) should be scheduled second");
    }

    // ---------------------------------------------------------------
    // 7. Sorted execution still produces correct results
    // ---------------------------------------------------------------
    #[test]
    fn test_schedule_preserves_correctness() {
        // Build a small graph and verify outputs match sequential execution.
        let node_a = make_node(OpKind::Relu, "relu1", vec!["x"], vec!["r1"]);
        let node_b = make_node(OpKind::Relu, "relu2", vec!["r1"], vec!["out"]);

        let graph = Graph {
            nodes: vec![node_a, node_b],
            input_names: vec!["x".to_string()],
            output_names: vec!["out".to_string()],
            ..Default::default()
        };
        let weights = HashMap::new();

        let session = Session::from_graph(graph, weights).expect("from_graph should succeed");
        let input = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
        let outputs = session.run_one("x", input).expect("run should succeed");
        let out = outputs.get("out").expect("output should exist");
        // Relu(Relu(x)) = Relu(max(0, x)) = max(0, x)
        assert_eq!(out.data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    // ---------------------------------------------------------------
    // 8. Attention op has highest per-element cost
    // ---------------------------------------------------------------
    #[test]
    fn test_cost_attention_very_high() {
        let attn = make_node(
            OpKind::Attention,
            "attn",
            vec!["q", "k", "v"],
            vec!["attn_out"],
        );
        let conv = make_node(OpKind::Conv, "conv", vec!["x", "w"], vec!["conv_out"]);
        let matmul = make_node(OpKind::MatMul, "mm", vec!["a", "b"], vec!["mm_out"]);
        let cache: HashMap<String, Vec<usize>> = [
            shape_entry("attn_out", vec![1, 8, 64]),
            shape_entry("conv_out", vec![1, 8, 64]),
            shape_entry("mm_out", vec![1, 8, 64]),
        ]
        .into_iter()
        .collect();

        let cost_attn = estimate_node_cost(&attn, Some(&cache));
        let cost_conv = estimate_node_cost(&conv, Some(&cache));
        let cost_mm = estimate_node_cost(&matmul, Some(&cache));

        assert!(
            cost_attn > cost_conv,
            "Attention {} should be > Conv {}",
            cost_attn,
            cost_conv
        );
        assert!(
            cost_attn > cost_mm,
            "Attention {} should be > MatMul {}",
            cost_attn,
            cost_mm
        );
    }

    // ---------------------------------------------------------------
    // 9. Single-node graph
    // ---------------------------------------------------------------
    #[test]
    fn test_schedule_single_node() {
        let nodes = vec![make_node(
            OpKind::Relu,
            "only",
            vec!["input"],
            vec!["output"],
        )];
        let weights = HashMap::new();
        let schedule = compute_execution_schedule(&nodes, &weights, None);
        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule[0], vec![0]);
    }

    // ---------------------------------------------------------------
    // 10. Empty graph
    // ---------------------------------------------------------------
    #[test]
    fn test_schedule_empty_graph() {
        let nodes: Vec<Node> = Vec::new();
        let weights = HashMap::new();
        let schedule = compute_execution_schedule(&nodes, &weights, None);
        assert!(schedule.is_empty());
    }
}
