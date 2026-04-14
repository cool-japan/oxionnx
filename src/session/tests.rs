/// Build equal-sized split chunks for a given axis length and count.
#[cfg(test)]
fn equal_split(axis_len: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let chunk = axis_len.div_ceil(n);
    (0..n)
        .map(|i| {
            let start = i * chunk;
            (start + chunk).min(axis_len).saturating_sub(start)
        })
        .filter(|&s| s > 0)
        .collect()
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::types::OptLevel;
    use super::super::Session;
    use super::super::SessionBuilder;
    use super::equal_split;
    use crate::graph::{Attributes, Graph, Node, OpKind};
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    #[test]
    fn test_session_from_empty_bytes() {
        let session = Session::from_bytes(&[]).expect("should load empty model");
        let inputs = HashMap::new();
        let outputs = session.run(&inputs).expect("should run empty model");
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_equal_split_helper() {
        assert_eq!(equal_split(6, 3), vec![2, 2, 2]);
        assert_eq!(equal_split(7, 3), vec![3, 3, 1]);
        assert_eq!(equal_split(4, 1), vec![4]);
    }

    #[test]
    fn test_from_graph_identity() {
        // Build a simple Identity graph: input -> Identity -> output
        let node = Node {
            op: OpKind::Identity,
            name: "id_node".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };
        let weights = HashMap::new();

        let session = Session::from_graph(graph, weights).expect("from_graph should succeed");
        assert_eq!(session.input_names(), &["input".to_string()]);
        assert_eq!(session.output_names(), &["output".to_string()]);

        let input_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let outputs = session
            .run_one("input", input_tensor.clone())
            .expect("run should succeed");
        let out = outputs.get("output").expect("output should exist");
        assert_eq!(out.data, input_tensor.data);
        assert_eq!(out.shape, input_tensor.shape);
    }

    #[test]
    fn test_builder_load_from_empty_bytes() {
        let result = Session::builder()
            .with_optimization_level(OptLevel::None)
            .load_from_bytes(&[]);
        assert!(result.is_ok());
        let session = result.expect("builder should load empty model");
        assert!(session.input_names().is_empty());
    }

    #[test]
    fn test_model_info() {
        let node1 = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["r1".to_string()],
            attrs: Attributes::default(),
        };
        let node2 = Node {
            op: OpKind::Relu,
            name: "relu2".to_string(),
            inputs: vec!["r1".to_string()],
            outputs: vec!["out".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node1, node2],
            input_names: vec!["x".to_string()],
            output_names: vec!["out".to_string()],
            ..Default::default()
        };
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0; 12], vec![3, 4]));

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build_from_graph");
        let info = session.model_info();
        assert_eq!(info.node_count, 2);
        assert_eq!(info.parameter_count, 12);
        assert_eq!(info.weight_bytes, 48); // 12 * 4
        assert_eq!(info.op_histogram.get("Relu").copied().unwrap_or(0), 2);
    }

    #[test]
    fn test_export_dot() {
        let node = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["out".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["out".to_string()],
            ..Default::default()
        };
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0], vec![1]));

        let session = Session::from_graph(graph, weights).expect("from_graph");
        let dot = session.export_dot();

        assert!(dot.starts_with("digraph model {"));
        assert!(dot.ends_with("}\n"));
        assert!(dot.contains("relu1"));
        assert!(dot.contains("Relu"));
        // Weight node should appear as ellipse
        assert!(dot.contains("\"w\""));
        assert!(dot.contains("ellipse"));
        // Edges
        assert!(dot.contains("\"x\" -> \"relu1\""));
        assert!(dot.contains("\"relu1\" -> \"out\""));
    }

    #[test]
    fn test_profiling() {
        let node = Node {
            op: OpKind::Identity,
            name: "id_node".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };
        let weights = HashMap::new();

        let session = Session::builder()
            .with_profiling()
            .build_from_graph(graph, weights)
            .expect("build should succeed");

        // Before running, profiling data should be empty
        let initial = session.profiling_results().expect("profiling enabled");
        assert!(initial.is_empty());

        let input_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let _outputs = session
            .run_one("input", input_tensor)
            .expect("run should succeed");

        let profiles = session.profiling_results().expect("profiling enabled");
        assert!(!profiles.is_empty());
        assert_eq!(profiles[0].node_name, "id_node");
        assert_eq!(profiles[0].op_type, "Identity");
        assert_eq!(profiles[0].output_shapes, vec![vec![1, 3]]);
    }

    #[test]
    fn test_profiling_disabled_returns_none() {
        let session = Session::from_bytes(&[]).expect("load empty model");
        assert!(session.profiling_results().is_none());
    }

    #[test]
    fn test_builder_default() {
        let builder = SessionBuilder::default();
        assert_eq!(builder.opt_level, OptLevel::All);
        assert!(!builder.enable_profiling);
        assert!(builder.enable_memory_pool);
        assert!(builder.registry.is_none());
    }

    #[test]
    fn test_opt_level_variants() {
        assert_ne!(OptLevel::None, OptLevel::Basic);
        assert_ne!(OptLevel::Basic, OptLevel::Extended);
        assert_ne!(OptLevel::Extended, OptLevel::All);
        // Clone + Copy
        let level = OptLevel::All;
        let cloned = level;
        assert_eq!(level, cloned);
    }

    // ── Parallel execution tests ────────────────────────────────────────

    /// Build a graph with two independent branches at the same depth:
    ///   input -> Relu(branch_a) -> output_a
    ///   input -> Relu(branch_b) -> output_b
    /// Both Relu nodes are at depth 0 and should run in parallel.
    #[test]
    fn test_parallel_execution_basic() {
        let node_a = Node {
            op: OpKind::Relu,
            name: "relu_a".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_a".to_string()],
            attrs: Attributes::default(),
        };
        let node_b = Node {
            op: OpKind::Relu,
            name: "relu_b".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_b".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node_a, node_b],
            input_names: vec!["input".to_string()],
            output_names: vec!["out_a".to_string(), "out_b".to_string()],
            ..Default::default()
        };

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(true)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let input = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
        let outputs = session.run_one("input", input).expect("run");

        let expected = vec![0.0, 2.0, 0.0, 4.0];
        let out_a = outputs.get("out_a").expect("out_a");
        let out_b = outputs.get("out_b").expect("out_b");
        assert_eq!(out_a.data, expected);
        assert_eq!(out_b.data, expected);
    }

    /// All nodes sequential (linear chain). Parallel mode should not break anything.
    #[test]
    fn test_parallel_single_node_levels() {
        let node1 = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["mid".to_string()],
            attrs: Attributes::default(),
        };
        let node2 = Node {
            op: OpKind::Relu,
            name: "relu2".to_string(),
            inputs: vec!["mid".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node1, node2],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_parallel_execution(true)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let input = Tensor::new(vec![-1.0, 5.0, -2.0], vec![1, 3]);
        let outputs = session.run_one("input", input).expect("run");
        let out = outputs.get("output").expect("output");
        assert_eq!(out.data, vec![0.0, 5.0, 0.0]);
    }

    // ── In-place execution tests ────────────────────────────────────────

    /// Verify ReLU produces correct output (in-place path used when ref_count==1).
    #[test]
    fn test_inplace_relu() {
        let node = Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            ..Default::default()
        };

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let input = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);
        let outputs = session.run_one("x", input).expect("run");
        let y = outputs.get("y").expect("y");
        assert_eq!(y.data, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
    }

    /// Verify element-wise Add works in-place when shapes match.
    #[test]
    fn test_inplace_add_same_shape() {
        // x -> Add(x, w) -> y   where x and w have same shape
        let node = Node {
            op: OpKind::Add,
            name: "add".to_string(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "w".to_string(),
            Tensor::new(vec![10.0, 20.0, 30.0], vec![3]),
        );

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let outputs = session.run_one("x", input).expect("run");
        let y = outputs.get("y").expect("y");
        assert_eq!(y.data, vec![11.0, 22.0, 33.0]);
    }

    /// Verify broadcast Add falls back to regular path (shapes differ).
    #[test]
    fn test_inplace_fallback_broadcast() {
        // x [2,3] + w [3] -> y [2,3]   (broadcasting needed, inplace should fallback)
        let node = Node {
            op: OpKind::Add,
            name: "add".to_string(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "w".to_string(),
            Tensor::new(vec![10.0, 20.0, 30.0], vec![3]),
        );

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let outputs = session.run_one("x", input).expect("run");
        let y = outputs.get("y").expect("y");
        assert_eq!(y.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        assert_eq!(y.shape, vec![2, 3]);
    }

    /// A tensor consumed by 2 nodes should NOT be modified in-place.
    #[test]
    fn test_inplace_respects_refcount() {
        // input -> relu_a -> out_a
        // input -> relu_b -> out_b
        // "input" has refcount 2, so neither relu should modify it in-place.
        let node_a = Node {
            op: OpKind::Relu,
            name: "relu_a".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_a".to_string()],
            attrs: Attributes::default(),
        };
        let node_b = Node {
            op: OpKind::Relu,
            name: "relu_b".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_b".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node_a, node_b],
            input_names: vec!["input".to_string()],
            output_names: vec!["out_a".to_string(), "out_b".to_string()],
            ..Default::default()
        };

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, HashMap::new())
            .expect("build");

        let input = Tensor::new(vec![-2.0, 3.0, -1.0, 5.0], vec![2, 2]);
        let outputs = session.run_one("input", input).expect("run");

        let expected = vec![0.0, 3.0, 0.0, 5.0];
        let out_a = outputs.get("out_a").expect("out_a");
        let out_b = outputs.get("out_b").expect("out_b");
        assert_eq!(out_a.data, expected);
        assert_eq!(out_b.data, expected);
    }

    /// Test depth computation helper directly.
    #[test]
    fn test_compute_node_depths() {
        // A linear chain: input -> relu1 -> relu2 -> output
        let node1 = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["mid".to_string()],
            attrs: Attributes::default(),
        };
        let node2 = Node {
            op: OpKind::Relu,
            name: "relu2".to_string(),
            inputs: vec!["mid".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let nodes = vec![node1, node2];
        let weights = HashMap::new();
        let depths = Session::compute_node_depths(&nodes, &weights);
        assert_eq!(depths, vec![0, 1]);
    }

    /// Test depth computation with independent branches.
    #[test]
    fn test_compute_node_depths_parallel_branches() {
        // input -> relu_a -> out_a  (depth 0)
        // input -> relu_b -> out_b  (depth 0)
        let node_a = Node {
            op: OpKind::Relu,
            name: "relu_a".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_a".to_string()],
            attrs: Attributes::default(),
        };
        let node_b = Node {
            op: OpKind::Relu,
            name: "relu_b".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out_b".to_string()],
            attrs: Attributes::default(),
        };
        let nodes = vec![node_a, node_b];
        let weights = HashMap::new();
        let depths = Session::compute_node_depths(&nodes, &weights);
        assert_eq!(depths, vec![0, 0]);
    }

    #[test]
    fn test_group_by_depth() {
        let depths = vec![0, 0, 1, 2, 1];
        let groups = Session::group_by_depth(&depths);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![0, 1]);
        assert_eq!(groups[1], vec![2, 4]);
        assert_eq!(groups[2], vec![3]);
    }

    // ── Mixed precision tests ───────────────────────────────────────────

    /// Basic mixed precision: Relu → Add → MatMul → Relu graph.
    /// Verify output matches f32-only within tolerance.
    #[test]
    fn test_mixed_precision_relu_add_matmul_relu() {
        // Graph: input [2,3] → Relu → relu_out → Add(relu_out, bias) → add_out
        //        → MatMul(add_out, weight [3,2]) → mm_out → Relu → output [2,2]
        let relu1 = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["relu_out".to_string()],
            attrs: Attributes::default(),
        };
        let add = Node {
            op: OpKind::Add,
            name: "add1".to_string(),
            inputs: vec!["relu_out".to_string(), "bias".to_string()],
            outputs: vec!["add_out".to_string()],
            attrs: Attributes::default(),
        };
        let matmul = Node {
            op: OpKind::MatMul,
            name: "matmul1".to_string(),
            inputs: vec!["add_out".to_string(), "weight".to_string()],
            outputs: vec!["mm_out".to_string()],
            attrs: Attributes::default(),
        };
        let relu2 = Node {
            op: OpKind::Relu,
            name: "relu2".to_string(),
            inputs: vec!["mm_out".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };

        let graph = Graph {
            nodes: vec![relu1, add, matmul, relu2],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "bias".to_string(),
            Tensor::new(vec![0.5, 0.5, 0.5], vec![3]),
        );
        weights.insert(
            "weight".to_string(),
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]),
        );

        // Run with mixed precision
        let session_mp = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_mixed_precision(true)
            .build_from_graph(graph.clone(), weights.clone())
            .expect("build mixed precision session");

        // Run without mixed precision (reference)
        let session_f32 = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build f32 session");

        let input = Tensor::new(vec![-1.0, 2.0, 0.5, 3.0, -0.5, 1.0], vec![2, 3]);
        let out_mp = session_mp.run_one("input", input.clone()).expect("run mp");
        let out_f32 = session_f32.run_one("input", input).expect("run f32");

        let mp_data = &out_mp.get("output").expect("mp output").data;
        let f32_data = &out_f32.get("output").expect("f32 output").data;

        // Mixed precision should match f32 within tolerance (f16 has ~0.1% relative error)
        assert_eq!(mp_data.len(), f32_data.len());
        for (i, (&mp_val, &f32_val)) in mp_data.iter().zip(f32_data.iter()).enumerate() {
            let abs_err = (mp_val - f32_val).abs();
            let rel_tol = f32_val.abs() * 0.01 + 0.01; // 1% relative + 0.01 absolute
            assert!(
                abs_err < rel_tol,
                "Output[{i}]: mp={mp_val}, f32={f32_val}, err={abs_err} > tol={rel_tol}"
            );
        }
    }

    /// Verify f16-safe ops actually execute in f16 by checking profiling data.
    #[test]
    fn test_mixed_precision_profiling_shows_f16() {
        let relu = Node {
            op: OpKind::Relu,
            name: "relu1".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["relu_out".to_string()],
            attrs: Attributes::default(),
        };
        let add = Node {
            op: OpKind::Add,
            name: "add1".to_string(),
            inputs: vec!["relu_out".to_string(), "bias".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };

        let graph = Graph {
            nodes: vec![relu, add],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "bias".to_string(),
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        );

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_profiling()
            .with_mixed_precision(true)
            .build_from_graph(graph, weights)
            .expect("build");

        let input = Tensor::new(vec![-1.0, 2.0, 3.0], vec![1, 3]);
        let _outputs = session.run_one("input", input).expect("run");

        let profiles = session.profiling_results().expect("profiling enabled");
        assert_eq!(profiles.len(), 2);
        // Relu has native f16 path — profiled as "Relu(f16)"
        assert_eq!(profiles[0].op_type, "Relu(f16)");
        // Add has native f16 path — profiled as "Add(f16)"
        assert_eq!(profiles[1].op_type, "Add(f16)");
    }

    /// Verify MatMul uses f32 accumulation even with mixed precision enabled.
    #[test]
    fn test_mixed_precision_matmul_stays_f32() {
        let matmul = Node {
            op: OpKind::MatMul,
            name: "mm".to_string(),
            inputs: vec!["input".to_string(), "weight".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };

        let graph = Graph {
            nodes: vec![matmul],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
        );

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_profiling()
            .with_mixed_precision(true)
            .build_from_graph(graph, weights)
            .expect("build");

        let input = Tensor::new(vec![3.0, 7.0, 5.0, 11.0], vec![2, 2]);
        let outputs = session.run_one("input", input.clone()).expect("run");

        let out = outputs.get("output").expect("output");
        // Identity matrix multiplication: result == input (exact f32)
        assert_eq!(out.data, vec![3.0, 7.0, 5.0, 11.0]);

        // Profiling should show "MatMul" (not "MatMul(f16)")
        let profiles = session.profiling_results().expect("profiling enabled");
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].op_type, "MatMul");
    }

    /// Verify mixed precision session builds and runs without error.
    #[test]
    fn test_mixed_precision_builder() {
        let session = Session::builder()
            .with_mixed_precision(true)
            .load_from_bytes(&[]);
        assert!(session.is_ok());
        let session = session.expect("should build");
        assert!(session.mixed_precision);
    }

    /// Verify f16 rounding for ops without native f16 path (e.g., Softmax).
    #[test]
    fn test_mixed_precision_f16_rounding_fallback() {
        // Softmax is f16-safe but has no native f16 path in execute_elementwise_f16.
        // With mixed precision, its output should be rounded to f16 precision.
        let softmax = Node {
            op: OpKind::Softmax,
            name: "sm".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };

        let graph = Graph {
            nodes: vec![softmax],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };

        let session_mp = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_mixed_precision(true)
            .build_from_graph(graph.clone(), HashMap::new())
            .expect("build mp");

        let session_f32 = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, HashMap::new())
            .expect("build f32");

        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out_mp = session_mp.run_one("input", input.clone()).expect("run mp");
        let out_f32 = session_f32.run_one("input", input).expect("run f32");

        let mp_data = &out_mp.get("output").expect("mp output").data;
        let f32_data = &out_f32.get("output").expect("f32 output").data;

        // Outputs should be close but mp data should be f16-rounded
        for (&mp_val, &f32_val) in mp_data.iter().zip(f32_data.iter()) {
            let abs_err = (mp_val - f32_val).abs();
            assert!(abs_err < 0.01, "mp={mp_val}, f32={f32_val}, err={abs_err}");
            // Verify mp_val is exactly representable in f16
            let roundtrip = half::f16::from_f32(mp_val).to_f32();
            assert_eq!(
                mp_val, roundtrip,
                "mp output should be exactly f16-representable"
            );
        }
    }

    /// Chain of consecutive f16-safe ops: Relu → Add → Sigmoid.
    /// Verifies multi-op f16 execution works end-to-end.
    #[test]
    fn test_mixed_precision_consecutive_f16_ops() {
        let relu = Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["relu_out".to_string()],
            attrs: Attributes::default(),
        };
        let add = Node {
            op: OpKind::Add,
            name: "add".to_string(),
            inputs: vec!["relu_out".to_string(), "bias".to_string()],
            outputs: vec!["add_out".to_string()],
            attrs: Attributes::default(),
        };
        let sigmoid = Node {
            op: OpKind::Sigmoid,
            name: "sig".to_string(),
            inputs: vec!["add_out".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };

        let graph = Graph {
            nodes: vec![relu, add, sigmoid],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "bias".to_string(),
            Tensor::new(vec![-0.5, 0.0, 0.5], vec![3]),
        );

        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_profiling()
            .with_mixed_precision(true)
            .build_from_graph(graph, weights)
            .expect("build");

        let input = Tensor::new(vec![-2.0, 1.0, 3.0], vec![1, 3]);
        let outputs = session.run_one("input", input).expect("run");
        let out = outputs.get("output").expect("output");

        // All ops are f16-safe and have native f16 paths
        let profiles = session.profiling_results().expect("profiling");
        assert_eq!(profiles.len(), 3);
        assert_eq!(profiles[0].op_type, "Relu(f16)");
        assert_eq!(profiles[1].op_type, "Add(f16)");
        assert_eq!(profiles[2].op_type, "Sigmoid(f16)");

        // Output should be sigmoid values in [0, 1]
        for &v in &out.data {
            assert!((0.0..=1.0).contains(&v), "sigmoid output {v} out of range");
        }
    }

    // ── Per-session thread pool tests ────────────────────────────────────────

    #[test]
    fn test_with_intra_threads_loads_empty_model() {
        let session = Session::builder()
            .with_intra_threads(2)
            .load_from_bytes(&[]);
        assert!(session.is_ok());
        let session = session.expect("should build with intra_threads");
        assert!(session.parallel);
        #[cfg(not(target_arch = "wasm32"))]
        assert!(session.thread_pool.is_some());
    }

    #[test]
    fn test_with_inter_threads_accepted() {
        let session = Session::builder()
            .with_inter_threads(3)
            .load_from_bytes(&[]);
        assert!(session.is_ok());
        let session = session.expect("should build with inter_threads");
        assert!(session.parallel);
        #[cfg(not(target_arch = "wasm32"))]
        assert!(session.thread_pool.is_some());
    }

    #[test]
    fn test_thread_pool_single_thread_same_results() {
        // Build a simple Relu graph and verify 1-thread pool produces same results
        let node = Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            ..Default::default()
        };

        // Default session (no thread pool)
        let session_default =
            Session::from_graph(graph.clone(), HashMap::new()).expect("default build");
        let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let out_default = session_default
            .run_one("input", input.clone())
            .expect("default run");

        // Session with 1 thread
        let session_1t = Session::builder()
            .with_intra_threads(1)
            .build_from_graph(graph, HashMap::new())
            .expect("1-thread build");
        let out_1t = session_1t.run_one("input", input).expect("1-thread run");

        let d = out_default.get("output").expect("default output");
        let t = out_1t.get("output").expect("1-thread output");
        assert_eq!(d.data, t.data);
        assert_eq!(d.shape, t.shape);
    }

    #[test]
    fn test_thread_pool_four_threads_same_results() {
        // Build a graph with two independent branches to exercise parallel path.
        // Use different inputs so CSE doesn't merge the two Relu nodes.
        let relu_a = Node {
            op: OpKind::Relu,
            name: "relu_a".to_string(),
            inputs: vec!["input_a".to_string()],
            outputs: vec!["out_a".to_string()],
            attrs: Attributes::default(),
        };
        let relu_b = Node {
            op: OpKind::Relu,
            name: "relu_b".to_string(),
            inputs: vec!["input_b".to_string()],
            outputs: vec!["out_b".to_string()],
            attrs: Attributes::default(),
        };
        let graph = Graph {
            nodes: vec![relu_a, relu_b],
            input_names: vec!["input_a".to_string(), "input_b".to_string()],
            output_names: vec!["out_a".to_string(), "out_b".to_string()],
            ..Default::default()
        };

        let input_a = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let input_b = Tensor::new(vec![3.0, -4.0, 5.0, -6.0, 7.0], vec![5]);

        // Default session
        let session_default =
            Session::from_graph(graph.clone(), HashMap::new()).expect("default build");
        let mut inputs_default = HashMap::new();
        inputs_default.insert("input_a", input_a.clone());
        inputs_default.insert("input_b", input_b.clone());
        let out_default = session_default.run(&inputs_default).expect("default run");

        // Session with 4 threads
        let session_4t = Session::builder()
            .with_intra_threads(4)
            .build_from_graph(graph, HashMap::new())
            .expect("4-thread build");
        let mut inputs_4t = HashMap::new();
        inputs_4t.insert("input_a", input_a);
        inputs_4t.insert("input_b", input_b);
        let out_4t = session_4t.run(&inputs_4t).expect("4-thread run");

        let da = out_default.get("out_a").expect("default out_a");
        let db = out_default.get("out_b").expect("default out_b");
        let ta = out_4t.get("out_a").expect("4t out_a");
        let tb = out_4t.get("out_b").expect("4t out_b");
        assert_eq!(da.data, ta.data);
        assert_eq!(db.data, tb.data);
    }

    #[test]
    fn test_no_thread_pool_by_default() {
        let session = Session::from_bytes(&[]).expect("load empty model");
        #[cfg(not(target_arch = "wasm32"))]
        assert!(session.thread_pool.is_none());
        assert!(!session.parallel);
    }

    // ── Dynamic shape runtime resolution tests ──────────────────────────

    /// Helper: build a graph with input_infos that have symbolic dims.
    /// Graph: input [batch_size, 3] -> Relu -> output [batch_size, 3]
    fn build_dynamic_relu_graph() -> (Graph, HashMap<String, Tensor>) {
        use oxionnx_core::{DType, TensorInfo};

        let node = Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };

        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            input_infos: vec![TensorInfo {
                name: "input".to_string(),
                dtype: DType::F32,
                shape: vec![None, Some(3)],
                dim_params: vec![Some("batch_size".to_string()), None],
            }],
            output_infos: vec![TensorInfo {
                name: "output".to_string(),
                dtype: DType::F32,
                shape: vec![None, Some(3)],
                dim_params: vec![Some("batch_size".to_string()), None],
            }],
            ..Default::default()
        };

        (graph, HashMap::new())
    }

    /// Helper: build a graph with two symbolic dims (batch_size and seq_len).
    /// Graph: input [batch_size, seq_len] -> Relu -> output [batch_size, seq_len]
    fn build_dynamic_batch_seq_graph() -> (Graph, HashMap<String, Tensor>) {
        use oxionnx_core::{DType, TensorInfo};

        let node = Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attrs: Attributes::default(),
        };

        let graph = Graph {
            nodes: vec![node],
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            input_infos: vec![TensorInfo {
                name: "input".to_string(),
                dtype: DType::F32,
                shape: vec![None, None],
                dim_params: vec![Some("batch_size".to_string()), Some("seq_len".to_string())],
            }],
            output_infos: vec![],
            ..Default::default()
        };

        (graph, HashMap::new())
    }

    /// Helper: build a graph with two inputs sharing the same "batch_size" symbol.
    /// input_a [batch_size, 3] -> Relu -> out_a
    /// input_b [batch_size, 5] -> Relu -> out_b
    fn build_dual_input_dynamic_graph() -> (Graph, HashMap<String, Tensor>) {
        use oxionnx_core::{DType, TensorInfo};

        let node_a = Node {
            op: OpKind::Relu,
            name: "relu_a".to_string(),
            inputs: vec!["input_a".to_string()],
            outputs: vec!["out_a".to_string()],
            attrs: Attributes::default(),
        };
        let node_b = Node {
            op: OpKind::Relu,
            name: "relu_b".to_string(),
            inputs: vec!["input_b".to_string()],
            outputs: vec!["out_b".to_string()],
            attrs: Attributes::default(),
        };

        let graph = Graph {
            nodes: vec![node_a, node_b],
            input_names: vec!["input_a".to_string(), "input_b".to_string()],
            output_names: vec!["out_a".to_string(), "out_b".to_string()],
            input_infos: vec![
                TensorInfo {
                    name: "input_a".to_string(),
                    dtype: DType::F32,
                    shape: vec![None, Some(3)],
                    dim_params: vec![Some("batch_size".to_string()), None],
                },
                TensorInfo {
                    name: "input_b".to_string(),
                    dtype: DType::F32,
                    shape: vec![None, Some(5)],
                    dim_params: vec![Some("batch_size".to_string()), None],
                },
            ],
            output_infos: vec![],
            ..Default::default()
        };

        (graph, HashMap::new())
    }

    #[test]
    fn test_dynamic_batch_size() {
        let (graph, weights) = build_dynamic_relu_graph();
        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        // batch=1
        let input1 = Tensor::new(vec![-1.0, 2.0, 3.0], vec![1, 3]);
        let out1 = session.run_one("input", input1).expect("batch=1 run");
        let y1 = out1.get("output").expect("output");
        assert_eq!(y1.shape, vec![1, 3]);
        assert_eq!(y1.data, vec![0.0, 2.0, 3.0]);

        // batch=4 on the same session
        let input4 = Tensor::new(
            vec![
                -1.0, 2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 9.0, 10.0, -11.0, 12.0,
            ],
            vec![4, 3],
        );
        let out4 = session.run_one("input", input4).expect("batch=4 run");
        let y4 = out4.get("output").expect("output");
        assert_eq!(y4.shape, vec![4, 3]);
        assert_eq!(
            y4.data,
            vec![0.0, 2.0, 3.0, 4.0, 0.0, 6.0, 0.0, 8.0, 9.0, 10.0, 0.0, 12.0]
        );

        // Verify dynamic_dims updated
        let dims = session.dynamic_dims();
        assert_eq!(dims.get("batch_size"), Some(&4));
    }

    #[test]
    fn test_dynamic_seq_len() {
        let (graph, weights) = build_dynamic_batch_seq_graph();
        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        // seq_len=2
        let input_s2 = Tensor::new(vec![-1.0, 2.0, 3.0, -4.0], vec![2, 2]);
        let out_s2 = session.run_one("input", input_s2).expect("seq=2 run");
        let y_s2 = out_s2.get("output").expect("output");
        assert_eq!(y_s2.shape, vec![2, 2]);

        // seq_len=5
        let input_s5 = Tensor::new(
            vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0],
            vec![2, 5],
        );
        let out_s5 = session.run_one("input", input_s5).expect("seq=5 run");
        let y_s5 = out_s5.get("output").expect("output");
        assert_eq!(y_s5.shape, vec![2, 5]);
        assert_eq!(
            y_s5.data,
            vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0]
        );

        let dims = session.dynamic_dims();
        assert_eq!(dims.get("batch_size"), Some(&2));
        assert_eq!(dims.get("seq_len"), Some(&5));
    }

    #[test]
    fn test_shape_validation_rank_error() {
        let (graph, weights) = build_dynamic_relu_graph();
        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        // Expected rank 2 [batch_size, 3], provide rank 3
        let bad_input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
        let result = session.run_one("input", bad_input);
        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("should error"));
        assert!(
            err_msg.contains("rank"),
            "Error should mention rank: {err_msg}"
        );
    }

    #[test]
    fn test_shape_validation_dim_error() {
        let (graph, weights) = build_dynamic_relu_graph();
        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        // Expected [batch_size, 3], provide [2, 5] — dim 1 is static=3 but got 5
        let bad_input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![2, 5],
        );
        let result = session.run_one("input", bad_input);
        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("should error"));
        assert!(
            err_msg.contains("static dim"),
            "Error should mention static dim: {err_msg}"
        );
    }

    #[test]
    fn test_shape_validation_symbolic_consistency() {
        let (graph, weights) = build_dual_input_dynamic_graph();
        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .build_from_graph(graph, weights)
            .expect("build");

        // input_a has batch_size=2, input_b has batch_size=3 — should fail
        let input_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let input_b = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
            vec![3, 5],
        );

        let mut inputs = HashMap::new();
        inputs.insert("input_a", input_a);
        inputs.insert("input_b", input_b);
        let result = session.run(&inputs);
        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("should error"));
        assert!(
            err_msg.contains("inconsistent") || err_msg.contains("conflicting"),
            "Error should mention inconsistency: {err_msg}"
        );
    }

    #[test]
    fn test_resolve_dynamic_shapes_basic() {
        use oxionnx_core::{DType, TensorInfo};

        let infos = vec![
            TensorInfo {
                name: "x".to_string(),
                dtype: DType::F32,
                shape: vec![None, Some(768)],
                dim_params: vec![Some("batch_size".to_string()), None],
            },
            TensorInfo {
                name: "y".to_string(),
                dtype: DType::F32,
                shape: vec![None, None],
                dim_params: vec![Some("batch_size".to_string()), Some("seq_len".to_string())],
            },
        ];

        let tensor_x = Tensor::new(vec![0.0; 4 * 768], vec![4, 768]);
        let tensor_y = Tensor::new(vec![0.0; 4 * 128], vec![4, 128]);
        let mut inputs: HashMap<&str, &Tensor> = HashMap::new();
        inputs.insert("x", &tensor_x);
        inputs.insert("y", &tensor_y);

        let dim_map =
            Session::resolve_dynamic_shapes(&infos, &inputs).expect("resolve should succeed");
        assert_eq!(dim_map.get("batch_size"), Some(&4));
        assert_eq!(dim_map.get("seq_len"), Some(&128));
    }

    #[test]
    fn test_run_after_shape_change() {
        let (graph, weights) = build_dynamic_relu_graph();
        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_memory_pool(true)
            .build_from_graph(graph, weights)
            .expect("build");

        // Run with batch=2
        let input2 = Tensor::new(vec![-1.0, 2.0, 3.0, -4.0, 5.0, -6.0], vec![2, 3]);
        let out2 = session.run_one("input", input2).expect("batch=2");
        let y2 = out2.get("output").expect("output");
        assert_eq!(y2.data, vec![0.0, 2.0, 3.0, 0.0, 5.0, 0.0]);

        // Run with batch=1 — shapes changed, results should still be correct
        let input1 = Tensor::new(vec![10.0, -20.0, 30.0], vec![1, 3]);
        let out1 = session.run_one("input", input1).expect("batch=1");
        let y1 = out1.get("output").expect("output");
        assert_eq!(y1.data, vec![10.0, 0.0, 30.0]);
        assert_eq!(y1.shape, vec![1, 3]);

        // Run with batch=3 — shapes changed again
        let input3 = Tensor::new(
            vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0],
            vec![3, 3],
        );
        let out3 = session.run_one("input", input3).expect("batch=3");
        let y3 = out3.get("output").expect("output");
        assert_eq!(y3.shape, vec![3, 3]);
        assert_eq!(y3.data, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0]);
    }

    #[test]
    fn test_dynamic_shape_memory_reallocation() {
        let (graph, weights) = build_dynamic_relu_graph();
        let session = Session::builder()
            .with_optimization_level(OptLevel::None)
            .with_memory_pool(true)
            .build_from_graph(graph, weights)
            .expect("build");

        // Small batch
        let small = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out_small = session.run_one("input", small).expect("small batch");
        assert_eq!(out_small.get("output").expect("output").shape, vec![1, 3]);

        // Large batch — pool should handle the size increase
        let large_data: Vec<f32> = (0..300).map(|i| i as f32 - 150.0).collect();
        let large = Tensor::new(large_data, vec![100, 3]);
        let out_large = session.run_one("input", large).expect("large batch");
        let y_large = out_large.get("output").expect("output");
        assert_eq!(y_large.shape, vec![100, 3]);

        // Verify output correctness: ReLU maps negatives to 0
        for (i, &val) in y_large.data.iter().enumerate() {
            let original = i as f32 - 150.0;
            let expected = if original < 0.0 { 0.0 } else { original };
            assert!(
                (val - expected).abs() < 1e-6,
                "Mismatch at index {i}: got {val}, expected {expected}"
            );
        }

        // Back to small — the pool still works
        let small2 = Tensor::new(vec![-5.0, 10.0, -15.0], vec![1, 3]);
        let out_small2 = session.run_one("input", small2).expect("small batch 2");
        let y2 = out_small2.get("output").expect("output");
        assert_eq!(y2.data, vec![0.0, 10.0, 0.0]);
    }

    // ── Operator Placement Tests ────────────────────────────────────────

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
}
