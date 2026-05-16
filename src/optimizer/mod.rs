//! Graph optimization passes for ONNX inference.
//! Applied after model loading but before topological sort and execution.

pub mod constant_fold;
pub mod cost_model;
pub mod cse;
pub mod dead_code;
pub mod fusion;
pub mod graph_diff;
pub mod shape_inference;
pub(crate) mod shape_inference_ext;
pub mod symbolic_shape;

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use oxionnx_core::OperatorRegistry;
use std::collections::HashMap;

/// Apply all optimization passes to the graph.
/// Modifies weights in place (for fusion passes that fold parameters).
///
/// Pass order:
///  1. **Shape inference** — infer tensor shapes from weights and propagate
///     through the graph.  For every `Shape` node whose input has a known
///     shape, the output is materialised as a constant weight so that
///     downstream ops (Reshape, Gather, …) become constant-foldable.
///  2. **Constant folding** — evaluate nodes whose inputs are all constants.
///  3. **Dead-node elimination** — remove nodes not reachable from outputs.
///  4. **CSE** — merge duplicate sub-expressions.
///  5. **Fusion passes** — MatMul+Add, Conv+BN, Conv+Relu, Conv+ReLU6,
///     SiLU fusion, Div+Sqrt→Rsqrt, standalone BN folding, LayerNorm,
///     transpose cancellation.
pub fn optimize(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
    registry: &OperatorRegistry,
) -> Vec<Node> {
    // Phase 1: Shape inference.
    // Even without runtime input shapes we can propagate shapes that originate
    // from constant weights. This lets us materialise Shape op outputs so that
    // constant folding can evaluate their consumers.
    let input_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let known_shapes = shape_inference::infer_shapes(&nodes, weights, &input_shapes);
    materialize_shape_ops(&nodes, weights, &known_shapes);

    // Phase 2–N: existing optimisation pipeline.
    let nodes = constant_fold::constant_fold(nodes, weights, registry);
    let nodes = dead_code::dead_node_elimination(nodes, output_names);
    let nodes = cse::eliminate_common_subexpressions(nodes);
    let nodes = fusion::fuse_matmul_add(nodes, weights);
    let nodes = fusion::fuse_conv_batchnorm(nodes, weights);
    let nodes = fusion::fuse_conv_relu(nodes);
    let nodes = fusion::fuse_conv_clip_to_conv_relu6(nodes);
    let nodes = fusion::fuse_mul_sigmoid_to_silu(nodes);
    let nodes = fusion::fuse_div_sqrt_to_rsqrt(nodes, weights);
    // Re-infer shapes after upstream fusions so that the standalone
    // BatchNorm fold can size its synthesized `factor`/`shift` constants
    // to broadcast correctly against the BN input (e.g. `[1, C, 1, 1]`
    // for a 4-D `[N, C, H, W]` input).  Without per-input rank the fold
    // would emit `[C]`-shaped constants that fail strict NumPy alignment.
    let pre_fold_shapes = shape_inference::infer_shapes(&nodes, weights, &input_shapes);
    let nodes = fusion::fold_batch_norm_inference(nodes, weights, &pre_fold_shapes);
    let nodes = fusion::fuse_layer_norm(nodes, weights);
    let nodes = fusion::cancel_consecutive_transpose(nodes);
    let nodes = fusion::fuse_matmul_transpose(nodes);
    let nodes = fusion::fuse_add_matmul_to_gemm(nodes, weights);
    fusion::cancel_consecutive_reshape(nodes)
}

/// For each `Shape` node whose input has a known shape, store the shape
/// vector as a constant weight tensor.  This allows `constant_fold` to
/// evaluate downstream consumers that depend on Shape outputs (e.g.
/// `Shape → Gather → Reshape` chains).
fn materialize_shape_ops(
    nodes: &[Node],
    weights: &mut HashMap<String, Tensor>,
    known_shapes: &HashMap<String, Vec<usize>>,
) {
    for node in nodes {
        if node.op != OpKind::Shape {
            continue;
        }
        let input_name = match node.inputs.first() {
            Some(name) if !name.is_empty() => name,
            _ => continue,
        };
        let shape = match known_shapes.get(input_name) {
            Some(s) => s,
            None => continue,
        };
        let output_name = match node.outputs.first() {
            Some(name) if !name.is_empty() => name,
            _ => continue,
        };
        // Don't overwrite an already-known constant.
        if weights.contains_key(output_name) {
            continue;
        }
        // Store as a 1-D tensor.  Shape output is int64 per ONNX spec;
        // we use f32 since our Tensor stores f32 data.
        let shape_data: Vec<f32> = shape.iter().map(|&d| d as f32).collect();
        let len = shape_data.len();
        weights.insert(output_name.clone(), Tensor::new(shape_data, vec![len]));
    }
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
