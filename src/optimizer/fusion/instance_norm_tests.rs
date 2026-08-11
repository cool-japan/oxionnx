//! Unit tests for [`super::fuse_instance_norm`].

use super::*;
use crate::optimizer::graph_utils::missing_graph_outputs;
use crate::optimizer::test_utils::make_node;

/// `[N, C, H, W]` shapes for every tensor the synthetic chains use, so
/// `reduces_over_spatial_axes` can resolve rank.
fn nchw_shapes() -> HashMap<String, Vec<usize>> {
    let mut shapes = HashMap::new();
    for name in ["X", "diff", "sq", "norm", "scaled", "biased", "other"] {
        shapes.insert(name.to_string(), vec![2, 3, 4, 5]);
    }
    shapes
}

fn reduce_mean(name: &str, input: &str, output: &str, axes: Vec<i64>) -> Node {
    let mut node = make_node(OpKind::ReduceMean, name, vec![input], vec![output]);
    node.attrs.int_lists.insert("axes".to_string(), axes);
    node.attrs.ints.insert("keepdims".to_string(), 1);
    node
}

/// How the chain spells the squared deviation and the reciprocal, matching the
/// variants the pass documents.
struct ChainSpec {
    pow_square: bool,
    /// `None` = direct `Div(diff, std)`; `Some(true)` = `Reciprocal(std)`;
    /// `Some(false)` = `Div(one, std)`.
    reciprocal: Option<bool>,
    /// Put `rstd` in the final `Mul`'s slot 0 rather than slot 1.
    swap_mul_operands: bool,
    /// Put `eps` in the `Add`'s slot 0 rather than slot 1.
    swap_eps_operands: bool,
}

impl Default for ChainSpec {
    fn default() -> Self {
        Self {
            pow_square: false,
            reciprocal: Some(true),
            swap_mul_operands: false,
            swap_eps_operands: false,
        }
    }
}

/// Build the chain `X → … → norm`, plus the weights it references.
fn make_chain(spec: &ChainSpec) -> (Vec<Node>, HashMap<String, Tensor>) {
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("eps".to_string(), Tensor::new(vec![1e-8], vec![1]));

    let mut nodes = vec![
        reduce_mean("rm_mean", "X", "mean", vec![2, 3]),
        make_node(OpKind::Sub, "sub", vec!["X", "mean"], vec!["diff"]),
    ];

    if spec.pow_square {
        weights.insert("two".to_string(), Tensor::new(vec![2.0], vec![1]));
        nodes.push(make_node(
            OpKind::Pow,
            "pow",
            vec!["diff", "two"],
            vec!["sq"],
        ));
    } else {
        nodes.push(make_node(
            OpKind::Mul,
            "square",
            vec!["diff", "diff"],
            vec!["sq"],
        ));
    }

    nodes.push(reduce_mean("rm_var", "sq", "var", vec![2, 3]));
    let add_inputs = if spec.swap_eps_operands {
        vec!["eps", "var"]
    } else {
        vec!["var", "eps"]
    };
    nodes.push(make_node(OpKind::Add, "add_eps", add_inputs, vec!["vare"]));
    nodes.push(make_node(OpKind::Sqrt, "sqrt", vec!["vare"], vec!["std"]));

    match spec.reciprocal {
        None => nodes.push(make_node(
            OpKind::Div,
            "normalize",
            vec!["diff", "std"],
            vec!["norm"],
        )),
        Some(as_reciprocal) => {
            if as_reciprocal {
                nodes.push(make_node(
                    OpKind::Reciprocal,
                    "recip",
                    vec!["std"],
                    vec!["rstd"],
                ));
            } else {
                weights.insert("one".to_string(), Tensor::new(vec![1.0], vec![1]));
                nodes.push(make_node(
                    OpKind::Div,
                    "recip",
                    vec!["one", "std"],
                    vec!["rstd"],
                ));
            }
            let mul_inputs = if spec.swap_mul_operands {
                vec!["rstd", "diff"]
            } else {
                vec!["diff", "rstd"]
            };
            nodes.push(make_node(
                OpKind::Mul,
                "normalize",
                mul_inputs,
                vec!["norm"],
            ));
        }
    }

    (nodes, weights)
}

/// The chain plus the runtime affine pair the pass must leave alone.
fn make_chain_with_affine(spec: &ChainSpec) -> (Vec<Node>, HashMap<String, Tensor>) {
    let (mut nodes, weights) = make_chain(spec);
    nodes.push(make_node(
        OpKind::Mul,
        "affine_mul",
        vec!["style_scale", "norm"],
        vec!["scaled"],
    ));
    nodes.push(make_node(
        OpKind::Add,
        "affine_add",
        vec!["scaled", "style_shift"],
        vec!["biased"],
    ));
    (nodes, weights)
}

fn outputs() -> Vec<String> {
    vec!["norm".to_string()]
}

fn fused_nodes(nodes: &[Node]) -> Vec<&Node> {
    nodes
        .iter()
        .filter(|n| matches!(n.op, OpKind::OxiInstanceNorm))
        .collect()
}

fn run(spec: &ChainSpec, output_names: &[String]) -> Vec<Node> {
    let (nodes, weights) = make_chain(spec);
    fuse_instance_norm(nodes, &weights, &nchw_shapes(), output_names)
}

#[test]
fn fuses_reciprocal_mul_self_variant() {
    let result = run(&ChainSpec::default(), &outputs());
    // 8 nodes (mean, sub, square, var, add, sqrt, recip, mul) → 1.
    assert_eq!(result.len(), 1, "{result:?}");
    let fused = fused_nodes(&result);
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].inputs, vec!["X".to_string()]);
    assert_eq!(fused[0].outputs, vec!["norm".to_string()]);
    assert!((fused[0].attrs.f("epsilon", 0.0) - 1e-8).abs() < 1e-12);
}

#[test]
fn fuses_div_one_reciprocal_variant() {
    let spec = ChainSpec {
        reciprocal: Some(false),
        ..ChainSpec::default()
    };
    let result = run(&spec, &outputs());
    assert_eq!(result.len(), 1, "{result:?}");
    assert_eq!(fused_nodes(&result).len(), 1);
}

#[test]
fn fuses_direct_div_variant() {
    let spec = ChainSpec {
        reciprocal: None,
        ..ChainSpec::default()
    };
    let result = run(&spec, &outputs());
    // 7 nodes here (no reciprocal node) → 1.
    assert_eq!(result.len(), 1, "{result:?}");
    assert_eq!(fused_nodes(&result).len(), 1);
}

#[test]
fn fuses_pow_square_variant() {
    let spec = ChainSpec {
        pow_square: true,
        ..ChainSpec::default()
    };
    let result = run(&spec, &outputs());
    assert_eq!(result.len(), 1, "{result:?}");
    assert_eq!(fused_nodes(&result).len(), 1);
}

/// Operand order is an exporter detail, not part of the semantics: `Mul` is
/// commutative and so is the `Add` that introduces epsilon.
#[test]
fn fuses_regardless_of_commutative_operand_order() {
    let spec = ChainSpec {
        swap_mul_operands: true,
        swap_eps_operands: true,
        ..ChainSpec::default()
    };
    let result = run(&spec, &outputs());
    assert_eq!(result.len(), 1, "{result:?}");
    assert_eq!(fused_nodes(&result).len(), 1);
    assert_eq!(fused_nodes(&result)[0].inputs, vec!["X".to_string()]);
}

/// The affine pair takes its scale/shift from runtime tensors, so it cannot
/// move into the fused node — and must survive untouched.
#[test]
fn leaves_the_runtime_affine_nodes_outside() {
    let spec = ChainSpec::default();
    let (nodes, weights) = make_chain_with_affine(&spec);
    let output_names = vec!["biased".to_string()];
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &output_names);
    // 10 nodes → 3 (fused + affine Mul + affine Add).
    assert_eq!(result.len(), 3, "{result:?}");
    assert_eq!(fused_nodes(&result).len(), 1);
    assert!(result
        .iter()
        .any(|n| n.name == "affine_mul" && n.inputs == vec!["style_scale", "norm"]));
    assert!(result
        .iter()
        .any(|n| n.name == "affine_add" && n.inputs == vec!["scaled", "style_shift"]));
    assert!(missing_graph_outputs(&result, &weights, &output_names).is_empty());
}

/// Two independent chains in one graph both fuse, and neither steals a node
/// from the other.
#[test]
fn fuses_two_independent_chains() {
    let (first, weights) = make_chain(&ChainSpec::default());
    let mut nodes = first;
    let mut renamed: Vec<Node> = Vec::new();
    for node in make_chain(&ChainSpec::default()).0 {
        let mut node = node;
        node.name = format!("{}_b", node.name);
        node.inputs = node
            .inputs
            .iter()
            .map(|n| {
                if n == "eps" {
                    n.clone()
                } else {
                    format!("{n}_b")
                }
            })
            .collect();
        node.outputs = node.outputs.iter().map(|n| format!("{n}_b")).collect();
        renamed.push(node);
    }
    nodes.extend(renamed);

    let mut shapes = nchw_shapes();
    shapes.insert("X_b".to_string(), vec![2, 3, 4, 5]);
    shapes.insert("diff_b".to_string(), vec![2, 3, 4, 5]);
    shapes.insert("sq_b".to_string(), vec![2, 3, 4, 5]);

    let output_names = vec!["norm".to_string(), "norm_b".to_string()];
    let result = fuse_instance_norm(nodes, &weights, &shapes, &output_names);
    assert_eq!(result.len(), 2, "{result:?}");
    assert_eq!(fused_nodes(&result).len(), 2);
    assert!(missing_graph_outputs(&result, &weights, &output_names).is_empty());
}

/// `axes=[1,2,3]` normalises over the channel axis too — that is a different
/// operator, and folding it into `OxiInstanceNorm` would silently change the
/// numbers.
#[test]
fn declines_when_reduction_includes_the_channel_axis() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    for node in nodes.iter_mut() {
        if matches!(node.op, OpKind::ReduceMean) {
            node.attrs
                .int_lists
                .insert("axes".to_string(), vec![1, 2, 3]);
        }
    }
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// `axes=[3]` leaves axis 2 un-normalised: a partial spatial reduction, not
/// this pattern.
#[test]
fn declines_on_partial_spatial_reduction() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    for node in nodes.iter_mut() {
        if matches!(node.op, OpKind::ReduceMean) {
            node.attrs.int_lists.insert("axes".to_string(), vec![3]);
        }
    }
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// `keepdims=0` squeezes the reduction result, so `Sub(X, mean)` would not
/// broadcast the way the fused op assumes.
#[test]
fn declines_when_keepdims_is_zero() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    for node in nodes.iter_mut() {
        if matches!(node.op, OpKind::ReduceMean) {
            node.attrs.ints.insert("keepdims".to_string(), 0);
        }
    }
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// Without a rank the pass cannot tell `[2, 3]` from "some trailing axes", so
/// it declines rather than guessing.
#[test]
fn declines_when_rank_is_unknown() {
    let (nodes, weights) = make_chain(&ChainSpec::default());
    let result = fuse_instance_norm(nodes, &weights, &HashMap::new(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// An extra reader of `diff` would lose its producer when the chain collapses.
#[test]
fn declines_when_an_intermediate_has_an_extra_consumer() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    nodes.push(make_node(OpKind::Relu, "spy", vec!["diff"], vec!["other"]));
    let output_names = vec!["norm".to_string(), "other".to_string()];
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &output_names);
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
    assert!(missing_graph_outputs(&result, &weights, &output_names).is_empty());
}

/// A declared graph output inside the chain must keep being produced.
#[test]
fn declines_when_an_intermediate_is_a_graph_output() {
    let (nodes, weights) = make_chain(&ChainSpec::default());
    let output_names = vec!["norm".to_string(), "std".to_string()];
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &output_names);
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
    assert!(missing_graph_outputs(&result, &weights, &output_names).is_empty());
}

/// `Mul(diff, something_else)` is not a square, whatever the rest looks like.
#[test]
fn declines_when_the_square_multiplies_two_different_tensors() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    for node in nodes.iter_mut() {
        if node.name == "square" {
            node.inputs[1] = "other".to_string();
        }
    }
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// `Pow(diff, 3)` is a third moment, not a variance.
#[test]
fn declines_on_a_non_square_exponent() {
    let spec = ChainSpec {
        pow_square: true,
        ..ChainSpec::default()
    };
    let (nodes, mut weights) = make_chain(&spec);
    weights.insert("two".to_string(), Tensor::new(vec![3.0], vec![1]));
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// `Div(2, std)` is not a reciprocal, so the trailing `Mul` is not a
/// normalisation.
#[test]
fn declines_when_the_reciprocal_numerator_is_not_one() {
    let spec = ChainSpec {
        reciprocal: Some(false),
        ..ChainSpec::default()
    };
    let (nodes, mut weights) = make_chain(&spec);
    weights.insert("one".to_string(), Tensor::new(vec![2.0], vec![1]));
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// Epsilon must be a compile-time scalar; a runtime tensor there means the
/// denominator is not the one the fused op computes.
#[test]
fn declines_when_epsilon_is_not_a_constant() {
    let (nodes, mut weights) = make_chain(&ChainSpec::default());
    weights.remove("eps");
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// A negative epsilon is not a stabiliser; decline rather than reason about
/// which side produces `NaN` first.
#[test]
fn declines_on_a_negative_epsilon() {
    let (nodes, mut weights) = make_chain(&ChainSpec::default());
    weights.insert("eps".to_string(), Tensor::new(vec![-1e-8], vec![1]));
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// A large epsilon is unusual but semantically fine — `(x - mean) /
/// sqrt(var + k)` is exactly what the fused op computes for any `k >= 0`, so
/// the pass must not impose an arbitrary magnitude window.
#[test]
fn fuses_with_an_unusually_large_epsilon() {
    let (nodes, mut weights) = make_chain(&ChainSpec::default());
    weights.insert("eps".to_string(), Tensor::new(vec![0.5], vec![1]));
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert_eq!(fused_nodes(&result).len(), 1, "{result:?}");
    assert!((fused_nodes(&result)[0].attrs.f("epsilon", 0.0) - 0.5).abs() < 1e-9);
}

/// `Sub(mean, X)` is the negated deviation; fusing it would flip the sign of
/// every output.
#[test]
fn declines_when_the_subtraction_is_reversed() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    for node in nodes.iter_mut() {
        if node.name == "sub" {
            node.inputs = vec!["mean".to_string(), "X".to_string()];
        }
    }
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// The two reductions must run over the same tensor family: a mean of `X` and
/// a variance of something unrelated is not a normalisation.
#[test]
fn declines_when_the_mean_is_taken_of_a_different_tensor() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    for node in nodes.iter_mut() {
        if node.name == "rm_mean" {
            node.inputs = vec!["other".to_string()];
        }
    }
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert!(fused_nodes(&result).is_empty(), "{result:?}");
}

/// A graph too short to contain the pattern is returned untouched, and the
/// early-out must not lose nodes.
#[test]
fn short_graph_is_returned_unchanged() {
    let nodes = vec![make_node(OpKind::Relu, "relu", vec!["X"], vec!["Y"])];
    let result = fuse_instance_norm(nodes, &HashMap::new(), &nchw_shapes(), &["Y".to_string()]);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0].op, OpKind::Relu));
}

/// Rank 3 (`[N, C, L]`) normalises over the single spatial axis.
#[test]
fn fuses_rank_3_over_the_single_spatial_axis() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    for node in nodes.iter_mut() {
        if matches!(node.op, OpKind::ReduceMean) {
            node.attrs.int_lists.insert("axes".to_string(), vec![2]);
        }
    }
    let mut shapes = HashMap::new();
    for name in ["X", "diff", "sq"] {
        shapes.insert(name.to_string(), vec![2, 3, 8]);
    }
    let result = fuse_instance_norm(nodes, &weights, &shapes, &outputs());
    assert_eq!(fused_nodes(&result).len(), 1, "{result:?}");
}

/// Negative axes (`[-2, -1]` on a rank-4 input) name the same spatial pair.
#[test]
fn fuses_with_negative_axes() {
    let (mut nodes, weights) = make_chain(&ChainSpec::default());
    for node in nodes.iter_mut() {
        if matches!(node.op, OpKind::ReduceMean) {
            node.attrs
                .int_lists
                .insert("axes".to_string(), vec![-2, -1]);
        }
    }
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert_eq!(fused_nodes(&result).len(), 1, "{result:?}");
}

/// opset 18+ moves `axes` from an attribute to input slot 1.
#[test]
fn fuses_with_opset_18_axes_as_input() {
    let (mut nodes, mut weights) = make_chain(&ChainSpec::default());
    weights.insert(
        "spatial_axes".to_string(),
        Tensor::new(vec![2.0, 3.0], vec![2]),
    );
    for node in nodes.iter_mut() {
        if matches!(node.op, OpKind::ReduceMean) {
            node.attrs.int_lists.remove("axes");
            node.inputs.push("spatial_axes".to_string());
        }
    }
    let result = fuse_instance_norm(nodes, &weights, &nchw_shapes(), &outputs());
    assert_eq!(fused_nodes(&result).len(), 1, "{result:?}");
}

/// The fused node must land where the chain started, so a topological order
/// built from the rewritten list still puts it after its producer and before
/// its consumers.
#[test]
fn fused_node_replaces_the_earliest_chain_node() {
    let spec = ChainSpec::default();
    let (nodes, weights) = make_chain_with_affine(&spec);
    let produce_x = make_node(OpKind::Relu, "produce_x", vec!["in"], vec!["X"]);
    let mut all = vec![produce_x];
    all.extend(nodes);
    let output_names = vec!["biased".to_string()];
    let result = fuse_instance_norm(all, &weights, &nchw_shapes(), &output_names);
    let fused_pos = result
        .iter()
        .position(|n| matches!(n.op, OpKind::OxiInstanceNorm))
        .expect("fused node present");
    let x_pos = result
        .iter()
        .position(|n| n.name == "produce_x")
        .expect("producer present");
    let affine_pos = result
        .iter()
        .position(|n| n.name == "affine_mul")
        .expect("affine present");
    assert!(x_pos < fused_pos && fused_pos < affine_pos, "{result:?}");
}
