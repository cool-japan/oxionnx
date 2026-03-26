//! Graph optimization passes for ONNX inference.
//! Applied after model loading but before topological sort and execution.

pub mod constant_fold;
pub mod cse;
pub mod dead_code;
pub mod fusion;
#[allow(dead_code)]
pub mod graph_diff;
#[allow(dead_code)]
pub mod shape_inference;
#[allow(dead_code)]
pub(crate) mod shape_inference_ext;
#[allow(dead_code)]
pub mod symbolic_shape;

use crate::graph::Node;
use crate::tensor::Tensor;
use oxionnx_core::OperatorRegistry;
use std::collections::HashMap;

/// Apply all optimization passes to the graph.
/// Modifies weights in place (for fusion passes that fold parameters).
pub fn optimize(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
    registry: &OperatorRegistry,
) -> Vec<Node> {
    let nodes = constant_fold::constant_fold(nodes, weights, registry);
    let nodes = dead_code::dead_node_elimination(nodes, output_names);
    let nodes = cse::eliminate_common_subexpressions(nodes);
    let nodes = fusion::fuse_matmul_add(nodes, weights);
    let nodes = fusion::fuse_conv_batchnorm(nodes, weights);
    let nodes = fusion::fuse_conv_relu(nodes);
    let nodes = fusion::fuse_layer_norm(nodes, weights);
    fusion::cancel_consecutive_transpose(nodes)
}

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::graph::{Attributes, Node, OpKind};
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    pub fn make_node(op: OpKind, name: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> Node {
        Node {
            op,
            name: name.to_string(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
            attrs: Attributes::default(),
        }
    }

    #[allow(dead_code)]
    pub fn make_graph(nodes: Vec<Node>) -> Vec<Node> {
        nodes
    }

    pub fn make_layer_norm_pattern(with_scale_bias: bool) -> (Vec<Node>, HashMap<String, Tensor>) {
        let mut weights = HashMap::new();

        let mut reduce_mean1 =
            make_node(OpKind::ReduceMean, "reduce_mean1", vec!["X"], vec!["mean"]);
        reduce_mean1
            .attrs
            .int_lists
            .insert("axes".to_string(), vec![-1]);

        let sub = make_node(OpKind::Sub, "sub", vec!["X", "mean"], vec!["diff"]);

        let pow = make_node(OpKind::Pow, "pow", vec!["diff", "pow_exp"], vec!["sq"]);
        weights.insert("pow_exp".to_string(), Tensor::new(vec![2.0], vec![1]));

        let mut reduce_mean2 =
            make_node(OpKind::ReduceMean, "reduce_mean2", vec!["sq"], vec!["var"]);
        reduce_mean2
            .attrs
            .int_lists
            .insert("axes".to_string(), vec![-1]);

        let add_eps = make_node(OpKind::Add, "add_eps", vec!["var", "eps"], vec!["var_eps"]);
        weights.insert("eps".to_string(), Tensor::new(vec![1e-5], vec![1]));

        let sqrt = make_node(OpKind::Sqrt, "sqrt", vec!["var_eps"], vec!["std"]);

        let div = make_node(OpKind::Div, "div", vec!["diff", "std"], vec!["normalized"]);

        let mut nodes = vec![reduce_mean1, sub, pow, reduce_mean2, add_eps, sqrt, div];

        if with_scale_bias {
            let mul = make_node(
                OpKind::Mul,
                "mul",
                vec!["normalized", "scale"],
                vec!["scaled"],
            );
            weights.insert("scale".to_string(), Tensor::new(vec![1.0; 4], vec![4]));

            let add_bias = make_node(
                OpKind::Add,
                "add_bias",
                vec!["scaled", "bias"],
                vec!["output"],
            );
            weights.insert("bias".to_string(), Tensor::new(vec![0.0; 4], vec![4]));

            nodes.push(mul);
            nodes.push(add_bias);
        }

        (nodes, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use test_utils::make_node;

    #[test]
    fn test_optimize_empty_graph() {
        let nodes: Vec<Node> = vec![];
        let mut weights = HashMap::new();
        let output_names: Vec<String> = vec![];
        let registry = OperatorRegistry::new();
        let result = optimize(nodes, &mut weights, &output_names, &registry);
        assert!(result.is_empty());
    }

    #[test]
    fn test_optimize_single_node() {
        let nodes = vec![make_node(OpKind::Relu, "relu", vec!["x"], vec!["out"])];
        let mut weights = HashMap::new();
        let output_names = vec!["out".to_string()];
        let registry = OperatorRegistry::new();
        let result = optimize(nodes, &mut weights, &output_names, &registry);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "relu");
    }
}
