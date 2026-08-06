//! Graph optimization passes for ONNX inference.
//! Applied after model loading but before topological sort and execution.

pub mod constant_fold;
pub mod cost_model;
pub mod cse;
pub mod dead_code;
pub mod fusion;
pub mod graph_diff;
pub(crate) mod graph_utils;
pub mod shape_inference;
pub(crate) mod shape_inference_ext;
pub mod symbolic_shape;

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use oxionnx_core::OperatorRegistry;
use std::collections::HashMap;

/// Which optimisation passes [`optimize_with_level`] runs.
///
/// Mirrors the session-level `OptLevel`, and each variant runs exactly what its
/// documentation says — selecting a lower level is the supported way to opt out
/// of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassLevel {
    /// Dead-node elimination only.
    Basic,
    /// Dead-node elimination + operator fusions.
    Extended,
    /// Shape materialisation, constant folding, dead-node elimination, CSE and
    /// every fusion.
    All,
}

/// Apply the full optimization pipeline ([`PassLevel::All`]) to the graph.
/// Modifies weights in place (for fusion passes that fold parameters).
pub fn optimize(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
    registry: &OperatorRegistry,
) -> Vec<Node> {
    optimize_with_level(nodes, weights, output_names, registry, PassLevel::All)
}

/// Apply the optimization passes selected by `level`.
///
/// Pass order at [`PassLevel::All`]:
///  1. **Shape inference** — infer tensor shapes from weights and propagate
///     through the graph.  For every `Shape` node whose input has a known
///     shape, the output is materialised as a constant weight so that
///     downstream ops (Reshape, Gather, …) become constant-foldable.
///  2. **Constant folding** — evaluate nodes whose inputs are all constants.
///  3. **Dead-node elimination** — remove nodes not reachable from outputs.
///  4. **CSE** — merge duplicate sub-expressions.
///  5. **Fusion passes** — inference Dropout removal, MatMul+Add, Conv+BN,
///     Conv+Relu, Conv+ReLU6, SiLU, Div+Sqrt→Rsqrt, standalone BN folding,
///     LayerNorm, Transpose/Reshape cancellation, MatMul+Transpose,
///     Add+MatMul→Gemm, Gather composition, Conv+Add+Relu.
///
/// [`PassLevel::Extended`] runs step 3 and step 5; [`PassLevel::Basic`] runs
/// step 3 only.  Every pass keeps the declared `output_names` produced.
pub fn optimize_with_level(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
    registry: &OperatorRegistry,
    level: PassLevel,
) -> Vec<Node> {
    // Even without runtime input shapes we can propagate shapes that originate
    // from constant weights. This lets us materialise Shape op outputs so that
    // constant folding can evaluate their consumers.
    optimize_with_input_shapes(
        nodes,
        weights,
        output_names,
        registry,
        level,
        &HashMap::new(),
    )
}

/// [`optimize_with_level`], seeded with the graph's declared **input** shapes.
///
/// Several passes are gated on a provably rank-2 activation (`MatMul + Add →
/// Gemm`, `Add + MatMul → Gemm`, `MatMul + Transpose → Gemm`) and others size
/// synthesised constants from the inferred shapes (`fold_batch_norm_inference`,
/// `simplify_transpose_reshape`, `cancel_consecutive_reshape`). Run with an
/// empty seed, shape inference never learns the rank of a graph *input*, so on
/// a real model those passes almost never fire — coverage traded away for
/// soundness when the rank gates were added.
///
/// # Only fully static inputs may be seeded
///
/// `input_shapes` must contain **concrete** dimensions only. Substituting a
/// placeholder for a symbolic axis (`"batch"`) would not merely lose coverage:
/// the shape-consuming passes above would then size real synthesised constants
/// from a fabricated dimension. `Session::build_from_graph` therefore seeds an
/// input only when *every* one of its dims is statically known.
pub fn optimize_with_input_shapes(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
    registry: &OperatorRegistry,
    level: PassLevel,
    input_shapes: &HashMap<String, Vec<usize>>,
) -> Vec<Node> {
    let nodes = if level == PassLevel::All {
        let known_shapes = shape_inference::infer_shapes(&nodes, weights, input_shapes);
        materialize_shape_ops(&nodes, weights, &known_shapes);
        constant_fold::constant_fold(nodes, weights, registry, output_names)
    } else {
        nodes
    };

    let nodes = dead_code::dead_node_elimination(nodes, output_names);
    if level == PassLevel::Basic {
        return nodes;
    }

    let nodes = if level == PassLevel::All {
        cse::eliminate_common_subexpressions(nodes, output_names)
    } else {
        nodes
    };

    // Inference-mode Dropout is the identity; removing it first lets the
    // downstream patterns (e.g. Softmax → Dropout → MatMul) match.
    let nodes = fusion::eliminate_dropout_inference(nodes, weights, output_names);

    let shapes = shape_inference::infer_shapes(&nodes, weights, input_shapes);
    let nodes = fusion::fuse_matmul_add(nodes, weights, &shapes, output_names);
    let nodes = fusion::fuse_conv_batchnorm(nodes, weights, output_names);
    let nodes = fusion::fuse_conv_relu(nodes, weights, output_names);
    let nodes = fusion::fuse_conv_clip_to_conv_relu6(nodes, weights, output_names);
    let nodes = fusion::fuse_mul_sigmoid_to_silu(nodes, output_names);
    let nodes = fusion::fuse_div_sqrt_to_rsqrt(nodes, weights);
    // Re-infer shapes after upstream fusions so that the standalone
    // BatchNorm fold can size its synthesized `factor`/`shift` constants
    // to broadcast correctly against the BN input (e.g. `[1, C, 1, 1]`
    // for a 4-D `[N, C, H, W]` input).  Without per-input rank the fold
    // would emit `[C]`-shaped constants that fail strict NumPy alignment.
    let pre_fold_shapes = shape_inference::infer_shapes(&nodes, weights, input_shapes);
    let nodes = fusion::fold_batch_norm_inference(nodes, weights, &pre_fold_shapes);
    let nodes = fusion::fuse_layer_norm(nodes, weights, &pre_fold_shapes, output_names);
    let nodes = fusion::cancel_consecutive_transpose(nodes, output_names);
    let nodes = fusion::fuse_matmul_transpose(nodes, output_names);
    let nodes = fusion::fuse_add_matmul_to_gemm(nodes, weights, &pre_fold_shapes, output_names);
    let nodes = fusion::fuse_gather_composition(nodes, weights, output_names);
    // `ConvAddRelu` is an optimizer-generated op: only emit it when the active
    // registry actually provides a kernel for it, otherwise the fused node
    // would fail to dispatch at run time.
    let nodes = if registry.get("ConvAddRelu").is_some() {
        fusion::fuse_conv_add_relu(nodes, output_names)
    } else {
        nodes
    };
    let nodes = fusion::simplify_transpose_reshape(nodes, weights, &pre_fold_shapes, output_names);
    fusion::cancel_consecutive_reshape(nodes, weights, &pre_fold_shapes, output_names)
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
mod wave1_tests;

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
