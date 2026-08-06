//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::graph::OpKind;
    use crate::optimizer::fusion::simplify::{
        cancel_consecutive_reshape, cancel_consecutive_transpose, eliminate_dropout_inference,
        fuse_div_sqrt_to_rsqrt, fuse_gather_composition, fuse_mul_sigmoid_to_silu,
        simplify_transpose_reshape,
    };
    use crate::optimizer::test_utils::make_node;
    use crate::tensor::Tensor;
    use std::collections::HashMap;
    #[test]
    fn test_cancel_consecutive_transpose_identity() {
        let mut node1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node1.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let mut node2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
        node2.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let nodes = vec![node1, node2];
        let result = cancel_consecutive_transpose(nodes, &[]);
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_cancel_consecutive_transpose_non_identity() {
        let mut node1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node1
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![2, 0, 1]);
        let mut node2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
        node2
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 2, 0]);
        let nodes = vec![node1, node2];
        let result = cancel_consecutive_transpose(nodes, &[]);
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn test_cancel_consecutive_transpose_compose() {
        let mut node1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node1
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 2, 0]);
        let mut node2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
        node2
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 2, 0]);
        let nodes = vec![node1, node2];
        let result = cancel_consecutive_transpose(nodes, &[]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Transpose));
        let perm = result[0].attrs.int_lists.get("perm").expect("perm attr");
        assert_eq!(perm, &vec![2, 0, 1]);
    }
    #[test]
    fn test_cancel_consecutive_transpose_redirect() {
        let mut node1 = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node1.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let mut node2 = make_node(OpKind::Transpose, "t2", vec!["t1_out"], vec!["t2_out"]);
        node2.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let relu = make_node(OpKind::Relu, "relu", vec!["t2_out"], vec!["out"]);
        let nodes = vec![node1, node2, relu];
        let result = cancel_consecutive_transpose(nodes, &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "relu");
        assert_eq!(result[0].inputs[0], "x");
    }
    #[test]
    fn test_cancel_single_transpose() {
        let mut node = make_node(OpKind::Transpose, "t1", vec!["x"], vec!["t1_out"]);
        node.attrs.int_lists.insert("perm".to_string(), vec![1, 0]);
        let nodes = vec![node];
        let result = cancel_consecutive_transpose(nodes, &[]);
        assert_eq!(result.len(), 1);
    }
    #[test]
    fn test_cancel_consecutive_reshape_collapse() {
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "shape1"], vec!["r1_out"]);
        let r2 = make_node(
            OpKind::Reshape,
            "r2",
            vec!["r1_out", "shape2"],
            vec!["r2_out"],
        );
        let nodes = vec![r1, r2];
        let mut weights = HashMap::new();
        weights.insert("shape2".to_string(), Tensor::new(vec![2.0, 3.0], vec![2]));
        let result = cancel_consecutive_reshape(nodes, &weights, &HashMap::new(), &[]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Reshape));
        assert_eq!(result[0].inputs[0], "x");
        assert_eq!(result[0].inputs[1], "shape2");
        assert_eq!(result[0].outputs[0], "r2_out");
    }
    #[test]
    fn test_cancel_consecutive_reshape_same_shape_eliminates_both() {
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "shape_a"], vec!["r1_out"]);
        let r2 = make_node(
            OpKind::Reshape,
            "r2",
            vec!["r1_out", "shape_a"],
            vec!["r2_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["r2_out"], vec!["out"]);
        let nodes = vec![r1, r2, relu];
        // Both Reshapes disappear only when the second provably restores `x`'s
        // own shape — matching shape *names* proves nothing.
        let mut weights = HashMap::new();
        weights.insert("shape_a".to_string(), Tensor::new(vec![2.0, 3.0], vec![2]));
        let mut shapes = HashMap::new();
        shapes.insert("x".to_string(), vec![2, 3]);
        let result = cancel_consecutive_reshape(nodes, &weights, &shapes, &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "relu");
        assert_eq!(result[0].inputs[0], "x");
    }
    #[test]
    fn test_cancel_consecutive_reshape_no_cancel_multiple_consumers() {
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "shape1"], vec!["r1_out"]);
        let r2 = make_node(
            OpKind::Reshape,
            "r2",
            vec!["r1_out", "shape2"],
            vec!["r2_out"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["r1_out"], vec!["relu_out"]);
        let nodes = vec![r1, r2, relu];
        let result = cancel_consecutive_reshape(nodes, &HashMap::new(), &HashMap::new(), &[]);
        assert_eq!(result.len(), 3);
    }
    #[test]
    fn test_cancel_consecutive_reshape_single_node() {
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "shape1"], vec!["r1_out"]);
        let nodes = vec![r1];
        let result = cancel_consecutive_reshape(nodes, &HashMap::new(), &HashMap::new(), &[]);
        assert_eq!(result.len(), 1);
    }
    #[test]
    fn test_cancel_consecutive_reshape_three_reshapes() {
        let r1 = make_node(OpKind::Reshape, "r1", vec!["x", "s1"], vec!["r1_out"]);
        let r2 = make_node(OpKind::Reshape, "r2", vec!["r1_out", "s2"], vec!["r2_out"]);
        let r3 = make_node(OpKind::Reshape, "r3", vec!["r2_out", "s3"], vec!["r3_out"]);
        let nodes = vec![r1, r2, r3];
        let mut weights = HashMap::new();
        weights.insert("s2".to_string(), Tensor::new(vec![2.0, 3.0], vec![2]));
        weights.insert("s3".to_string(), Tensor::new(vec![6.0], vec![1]));
        let result = cancel_consecutive_reshape(nodes, &weights, &HashMap::new(), &[]);
        assert!(result.len() <= 2);
        let last = result.last().expect("should have at least one node");
        assert_eq!(last.outputs[0], "r3_out");
    }
    #[test]
    fn test_fuse_mul_sigmoid_to_silu_basic() {
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["x"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["mul_out"]);
        let nodes = vec![sigmoid, mul];
        let result = fuse_mul_sigmoid_to_silu(nodes, &[]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::SiLU));
        assert_eq!(result[0].inputs, vec!["x"]);
        assert_eq!(result[0].outputs, vec!["mul_out"]);
    }
    #[test]
    fn test_fuse_mul_sigmoid_to_silu_reversed_mul_inputs() {
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["x"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["sig_out", "x"], vec!["mul_out"]);
        let nodes = vec![sigmoid, mul];
        let result = fuse_mul_sigmoid_to_silu(nodes, &[]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::SiLU));
        assert_eq!(result[0].inputs, vec!["x"]);
    }
    #[test]
    fn test_fuse_mul_sigmoid_to_silu_no_fusion_multiple_consumers() {
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["x"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["mul_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["sig_out"], vec!["relu_out"]);
        let nodes = vec![sigmoid, mul, relu];
        let result = fuse_mul_sigmoid_to_silu(nodes, &[]);
        assert_eq!(result.len(), 3);
    }
    #[test]
    fn test_fuse_mul_sigmoid_to_silu_no_fusion_different_input() {
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["y"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["mul_out"]);
        let nodes = vec![sigmoid, mul];
        let result = fuse_mul_sigmoid_to_silu(nodes, &[]);
        assert_eq!(result.len(), 2);
    }
    #[test]
    fn test_fuse_mul_sigmoid_to_silu_preserves_downstream() {
        let sigmoid = make_node(OpKind::Sigmoid, "sigmoid", vec!["x"], vec!["sig_out"]);
        let mul = make_node(OpKind::Mul, "mul", vec!["x", "sig_out"], vec!["mul_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["mul_out"], vec!["relu_out"]);
        let nodes = vec![sigmoid, mul, relu];
        let result = fuse_mul_sigmoid_to_silu(nodes, &[]);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::SiLU));
        assert_eq!(result[0].outputs, vec!["mul_out"]);
        assert_eq!(result[1].inputs, vec!["mul_out"]);
    }
    #[test]
    fn test_fuse_div_sqrt_to_rsqrt_basic() {
        let sqrt = make_node(OpKind::Sqrt, "sqrt", vec!["x"], vec!["sqrt_out"]);
        let div = make_node(OpKind::Div, "div", vec!["one", "sqrt_out"], vec!["div_out"]);
        let nodes = vec![sqrt, div];
        let mut weights = HashMap::new();
        weights.insert("one".to_string(), Tensor::new(vec![1.0], vec![1]));
        let result = fuse_div_sqrt_to_rsqrt(nodes, &weights);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Sqrt));
        assert!(matches!(result[1].op, OpKind::Reciprocal));
        assert_eq!(result[1].inputs, vec!["sqrt_out"]);
        assert_eq!(result[1].outputs, vec!["div_out"]);
    }
    #[test]
    fn test_fuse_div_sqrt_to_rsqrt_not_const_one() {
        let sqrt = make_node(OpKind::Sqrt, "sqrt", vec!["x"], vec!["sqrt_out"]);
        let div = make_node(OpKind::Div, "div", vec!["two", "sqrt_out"], vec!["div_out"]);
        let nodes = vec![sqrt, div];
        let mut weights = HashMap::new();
        weights.insert("two".to_string(), Tensor::new(vec![2.0], vec![1]));
        let result = fuse_div_sqrt_to_rsqrt(nodes, &weights);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[1].op, OpKind::Div));
    }
    #[test]
    fn test_fuse_div_sqrt_to_rsqrt_not_sqrt() {
        let relu = make_node(OpKind::Relu, "relu", vec!["x"], vec!["relu_out"]);
        let div = make_node(OpKind::Div, "div", vec!["one", "relu_out"], vec!["div_out"]);
        let nodes = vec![relu, div];
        let mut weights = HashMap::new();
        weights.insert("one".to_string(), Tensor::new(vec![1.0], vec![1]));
        let result = fuse_div_sqrt_to_rsqrt(nodes, &weights);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[1].op, OpKind::Div));
    }
    #[test]
    fn test_fuse_div_sqrt_to_rsqrt_sqrt_multiple_consumers() {
        let sqrt = make_node(OpKind::Sqrt, "sqrt", vec!["x"], vec!["sqrt_out"]);
        let div = make_node(OpKind::Div, "div", vec!["one", "sqrt_out"], vec!["div_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["sqrt_out"], vec!["relu_out"]);
        let nodes = vec![sqrt, div, relu];
        let mut weights = HashMap::new();
        weights.insert("one".to_string(), Tensor::new(vec![1.0], vec![1]));
        let result = fuse_div_sqrt_to_rsqrt(nodes, &weights);
        assert_eq!(result.len(), 3);
        assert!(matches!(result[1].op, OpKind::Div));
    }
    #[test]
    fn test_fuse_gather_composition_basic() {
        let mut gather1 = make_node(OpKind::Gather, "g1", vec!["data", "idx1"], vec!["g1_out"]);
        gather1.attrs.ints.insert("axis".to_string(), 0);
        let mut gather2 = make_node(OpKind::Gather, "g2", vec!["g1_out", "idx2"], vec!["g2_out"]);
        gather2.attrs.ints.insert("axis".to_string(), 0);
        let nodes = vec![gather1, gather2];
        let mut weights = HashMap::new();
        weights.insert(
            "idx1".to_string(),
            Tensor::new(vec![2.0, 0.0, 1.0], vec![3]),
        );
        weights.insert("idx2".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
        let result = fuse_gather_composition(nodes, &mut weights, &[]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Gather));
        assert_eq!(result[0].inputs[0], "data");
        assert_eq!(result[0].outputs[0], "g2_out");
        let composed = weights
            .get(&result[0].inputs[1])
            .expect("composed indices initializer");
        assert_eq!(composed.data, vec![0.0, 1.0]);
    }
    #[test]
    fn test_fuse_gather_composition_no_fusion_different_axis() {
        let mut gather1 = make_node(OpKind::Gather, "g1", vec!["data", "idx1"], vec!["g1_out"]);
        gather1.attrs.ints.insert("axis".to_string(), 0);
        let mut gather2 = make_node(OpKind::Gather, "g2", vec!["g1_out", "idx2"], vec!["g2_out"]);
        gather2.attrs.ints.insert("axis".to_string(), 1);
        let nodes = vec![gather1, gather2];
        let mut weights = HashMap::new();
        weights.insert("idx1".to_string(), Tensor::new(vec![0.0, 1.0], vec![2]));
        weights.insert("idx2".to_string(), Tensor::new(vec![0.0], vec![1]));
        let result = fuse_gather_composition(nodes, &mut weights, &[]);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Gather));
        assert!(matches!(result[1].op, OpKind::Gather));
    }
    #[test]
    fn test_fuse_gather_composition_no_fusion_multiple_consumers() {
        let mut gather1 = make_node(OpKind::Gather, "g1", vec!["data", "idx1"], vec!["g1_out"]);
        gather1.attrs.ints.insert("axis".to_string(), 0);
        let mut gather2 = make_node(OpKind::Gather, "g2", vec!["g1_out", "idx2"], vec!["g2_out"]);
        gather2.attrs.ints.insert("axis".to_string(), 0);
        let relu = make_node(OpKind::Relu, "relu", vec!["g1_out"], vec!["relu_out"]);
        let nodes = vec![gather1, gather2, relu];
        let mut weights = HashMap::new();
        weights.insert("idx1".to_string(), Tensor::new(vec![0.0, 1.0], vec![2]));
        weights.insert("idx2".to_string(), Tensor::new(vec![0.0], vec![1]));
        let result = fuse_gather_composition(nodes, &mut weights, &[]);
        assert_eq!(result.len(), 3);
    }
    #[test]
    fn test_fuse_gather_composition_no_fusion_non_constant_indices() {
        let mut gather1 = make_node(
            OpKind::Gather,
            "g1",
            vec!["data", "dynamic_idx1"],
            vec!["g1_out"],
        );
        gather1.attrs.ints.insert("axis".to_string(), 0);
        let mut gather2 = make_node(
            OpKind::Gather,
            "g2",
            vec!["g1_out", "dynamic_idx2"],
            vec!["g2_out"],
        );
        gather2.attrs.ints.insert("axis".to_string(), 0);
        let nodes = vec![gather1, gather2];
        let mut weights = HashMap::new();
        let result = fuse_gather_composition(nodes, &mut weights, &[]);
        assert_eq!(result.len(), 2);
    }
    #[test]
    fn test_eliminate_dropout_inference_basic() {
        let dropout = make_node(OpKind::Dropout, "dropout", vec!["x"], vec!["dropout_out"]);
        let relu = make_node(OpKind::Relu, "relu", vec!["dropout_out"], vec!["out"]);
        let nodes = vec![dropout, relu];
        let result = eliminate_dropout_inference(nodes, &HashMap::new(), &[]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Relu));
        assert_eq!(result[0].inputs[0], "x");
    }
    #[test]
    fn test_eliminate_dropout_after_softmax() {
        let softmax = make_node(OpKind::Softmax, "softmax", vec!["x"], vec!["sm_out"]);
        let dropout = make_node(
            OpKind::Dropout,
            "dropout",
            vec!["sm_out"],
            vec!["dropout_out"],
        );
        let matmul = make_node(
            OpKind::MatMul,
            "matmul",
            vec!["dropout_out", "v"],
            vec!["out"],
        );
        let nodes = vec![softmax, dropout, matmul];
        let result = eliminate_dropout_inference(nodes, &HashMap::new(), &[]);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Softmax));
        assert!(matches!(result[1].op, OpKind::MatMul));
        assert_eq!(result[1].inputs[0], "sm_out");
    }
    #[test]
    fn test_eliminate_dropout_training_mode_not_eliminated() {
        let mut dropout = make_node(OpKind::Dropout, "dropout", vec!["x"], vec!["dropout_out"]);
        dropout.attrs.ints.insert("training_mode".to_string(), 1);
        let relu = make_node(OpKind::Relu, "relu", vec!["dropout_out"], vec!["out"]);
        let nodes = vec![dropout, relu];
        let result = eliminate_dropout_inference(nodes, &HashMap::new(), &[]);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Dropout));
    }
    #[test]
    fn test_eliminate_dropout_mask_output_used() {
        let dropout = make_node(
            OpKind::Dropout,
            "dropout",
            vec!["x"],
            vec!["dropout_out", "dropout_mask"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["dropout_out"], vec!["out1"]);
        let mask_user = make_node(
            OpKind::Identity,
            "mask_user",
            vec!["dropout_mask"],
            vec!["out2"],
        );
        let nodes = vec![dropout, relu, mask_user];
        let result = eliminate_dropout_inference(nodes, &HashMap::new(), &[]);
        assert_eq!(result.len(), 3);
        assert!(matches!(result[0].op, OpKind::Dropout));
    }
    #[test]
    fn test_simplify_transpose_reshape_identity_perm() {
        let mut transpose = make_node(OpKind::Transpose, "transpose", vec!["x"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![0, 1, 2]);
        let reshape = make_node(
            OpKind::Reshape,
            "reshape",
            vec!["t_out", "target_shape"],
            vec!["out"],
        );
        let nodes = vec![transpose, reshape];
        let weights = HashMap::new();
        let result = simplify_transpose_reshape(nodes, &weights, &HashMap::new(), &[]);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].op, OpKind::Reshape));
        assert_eq!(result[0].inputs[0], "x");
        assert_eq!(result[0].inputs[1], "target_shape");
        assert_eq!(result[0].outputs[0], "out");
    }
    /// A real permutation followed by a flatten is **not** a memory no-op:
    /// dropping the Transpose would flatten a different layout.
    #[test]
    fn test_simplify_transpose_reshape_flatten_after_transpose_is_kept() {
        let mut transpose = make_node(OpKind::Transpose, "transpose", vec!["x"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![0, 2, 1]);
        let reshape = make_node(
            OpKind::Reshape,
            "reshape",
            vec!["t_out", "flat_shape"],
            vec!["out"],
        );
        let nodes = vec![transpose, reshape];
        let mut weights = HashMap::new();
        weights.insert("flat_shape".to_string(), Tensor::new(vec![-1.0], vec![1]));
        let mut shapes = HashMap::new();
        shapes.insert("x".to_string(), vec![2, 3, 4]);
        let result = simplify_transpose_reshape(nodes, &weights, &shapes, &[]);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Transpose));
        assert!(matches!(result[1].op, OpKind::Reshape));
    }
    #[test]
    fn test_simplify_transpose_reshape_no_simplification_non_trivial() {
        let mut transpose = make_node(OpKind::Transpose, "transpose", vec!["x"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![1, 0]);
        let reshape = make_node(
            OpKind::Reshape,
            "reshape",
            vec!["t_out", "shape_2d"],
            vec!["out"],
        );
        let nodes = vec![transpose, reshape];
        let mut weights = HashMap::new();
        weights.insert("shape_2d".to_string(), Tensor::new(vec![3.0, 4.0], vec![2]));
        let result = simplify_transpose_reshape(nodes, &weights, &HashMap::new(), &[]);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, OpKind::Transpose));
        assert!(matches!(result[1].op, OpKind::Reshape));
    }
    #[test]
    fn test_simplify_transpose_reshape_no_simplification_multiple_consumers() {
        let mut transpose = make_node(OpKind::Transpose, "transpose", vec!["x"], vec!["t_out"]);
        transpose
            .attrs
            .int_lists
            .insert("perm".to_string(), vec![0, 1, 2]);
        let reshape = make_node(
            OpKind::Reshape,
            "reshape",
            vec!["t_out", "shape"],
            vec!["out1"],
        );
        let relu = make_node(OpKind::Relu, "relu", vec!["t_out"], vec!["out2"]);
        let nodes = vec![transpose, reshape, relu];
        let weights = HashMap::new();
        let result = simplify_transpose_reshape(nodes, &weights, &HashMap::new(), &[]);
        assert_eq!(result.len(), 3);
    }
}
