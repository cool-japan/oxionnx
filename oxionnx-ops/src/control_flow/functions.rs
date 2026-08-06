//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxionnx_core::error::OnnxError;
use oxionnx_core::graph::Graph;
use oxionnx_core::operator::{OpContext, OperatorRegistry};
use oxionnx_core::tensor::Tensor;
use std::collections::HashMap;

/// Execute a subgraph with the given inputs and outer scope.
///
/// The function performs a topological sort on the subgraph nodes, then
/// executes each node in order, resolving inputs from (in priority order):
/// 1. `subgraph_inputs` — explicit inputs to the subgraph
/// 2. `intermediates` — outputs produced by earlier nodes in this subgraph
/// 3. `outer_scope` — tensors from the enclosing graph scope
/// 4. `weights` — model weights
///
/// Returns the output tensors in the order specified by `graph.output_names`.
///
/// # Cost: this runs once per `Loop`/`Scan` **iteration**
///
/// `LoopOp` calls this once per trip and `ScanOp` once per sequence element, with
/// `outer_scope` bound to the *entire* live tensor map of the enclosing run and
/// `weights` to every model initializer.  Two things in here used to be
/// proportional to those maps rather than to the subgraph, and so quietly
/// quadratic in a decoder loop:
///
/// * **the merged scope** — `outer_scope.clone()` plus a clone of every
///   intermediate produced so far, rebuilt *per node*.  A 20-node body run for
///   512 iterations performed ~10 000 deep copies of every live activation in the
///   model, purely as scaffolding, and only `If`/`Loop`/`Scan` ever read it.  It
///   is now built only for a node that actually carries a subgraph attribute (see
///   the loop body below), which for an ordinary body of arithmetic is *never*.
/// * **`known_names`** — a `String` clone of every key of `subgraph_inputs`,
///   `outer_scope` **and `weights`**, i.e. one allocation per model parameter per
///   iteration.  [`Graph::topological_sort`] only ever asks whether a name it
///   found in *this* graph's `node.inputs` is known, so the list is now restricted
///   to exactly those names.  Every membership answer is identical; the size is
///   now proportional to the subgraph, not to the model.
pub(super) fn execute_subgraph(
    graph: &Graph,
    subgraph_inputs: HashMap<String, Tensor>,
    outer_scope: &HashMap<String, Tensor>,
    weights: &HashMap<String, Tensor>,
    registry: &OperatorRegistry,
) -> Result<Vec<Tensor>, OnnxError> {
    // The names `topological_sort` can actually ask about: this graph's node
    // inputs that some enclosing scope already provides.  Anything else in those
    // scopes is unreachable from here and cannot change the sort.
    let mut known_names: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for node in &graph.nodes {
        for input in &node.inputs {
            if input.is_empty() || !seen.insert(input.as_str()) {
                continue;
            }
            if subgraph_inputs.contains_key(input)
                || outer_scope.contains_key(input)
                || weights.contains_key(input)
            {
                known_names.push(input.clone());
            }
        }
    }
    let order = graph.topological_sort(&known_names);
    let mut intermediates: HashMap<String, Tensor> = subgraph_inputs;
    for idx in order {
        let node = graph
            .nodes
            .get(idx)
            .ok_or_else(|| OnnxError::InvalidModel("subgraph node index out of bounds".into()))?;
        let operator = registry.get(node.op.as_str()).ok_or_else(|| {
            OnnxError::UnsupportedOp(format!(
                "operator '{}' not found in registry for subgraph execution",
                node.op.as_str()
            ))
        })?;
        let resolved_inputs: Vec<Option<&Tensor>> = node
            .inputs
            .iter()
            .map(|name| {
                if name.is_empty() {
                    None
                } else {
                    intermediates
                        .get(name)
                        .or_else(|| outer_scope.get(name))
                        .or_else(|| weights.get(name))
                }
            })
            .collect();
        // The merged scope — enclosing scope + everything this body has produced
        // so far — is what a *nested* subgraph operator resolves its free names
        // out of, and nothing else reads it: `IfOp`, `LoopOp` and `ScanOp` are the
        // only operators in this workspace that touch `ctx.outer_scope`, and all
        // three carry their bodies in `attrs.graphs`.  Building it for the
        // ordinary nodes of the body — the `Add`s and `Relu`s that make up
        // essentially all of it — deep-copied the entire enclosing scope once per
        // node, per iteration, for nobody.  So it is built only when the node
        // being executed can actually observe it; every other node is handed the
        // enclosing scope directly, at zero cost.
        let merged_scope: Option<HashMap<String, Tensor>> = if node.attrs.graphs.is_empty() {
            None
        } else {
            let mut merged = HashMap::with_capacity(outer_scope.len() + intermediates.len());
            for (k, v) in outer_scope {
                merged.insert(k.clone(), v.clone());
            }
            // Body-local names shadow the enclosing scope, matching the input
            // resolution order above (`intermediates` before `outer_scope`).
            for (k, v) in &intermediates {
                merged.insert(k.clone(), v.clone());
            }
            Some(merged)
        };
        let ctx = OpContext {
            node,
            inputs: resolved_inputs,
            outer_scope: Some(merged_scope.as_ref().unwrap_or(outer_scope)),
            weights: Some(weights),
            registry: Some(registry),
        };
        let results = operator.execute(&ctx)?;
        for (out_name, tensor) in node.outputs.iter().zip(results) {
            if !out_name.is_empty() {
                intermediates.insert(out_name.clone(), tensor);
            }
        }
    }
    let mut outputs = Vec::with_capacity(graph.output_names.len());
    for name in &graph.output_names {
        let tensor = intermediates.remove(name).ok_or_else(|| {
            OnnxError::TensorNotFound(format!(
                "subgraph output '{}' not found after execution",
                name
            ))
        })?;
        outputs.push(tensor);
    }
    Ok(outputs)
}
/// Stack tensors along a new axis 0.
///
/// Each tensor in the slice must have the same shape. The result has shape
/// `[N, ...original_shape]` where N is the number of tensors.
pub(super) fn stack_tensors_axis0(tensors: &[Tensor]) -> Result<Tensor, OnnxError> {
    if tensors.is_empty() {
        return Ok(Tensor::new(vec![], vec![0]));
    }
    let first = &tensors[0];
    let elem_shape = &first.shape;
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.shape != *elem_shape {
            return Err(OnnxError::ShapeMismatch(format!(
                "stack_tensors_axis0: shape mismatch at index {}: {:?} vs {:?}",
                i, t.shape, elem_shape
            )));
        }
    }
    let mut new_shape = Vec::with_capacity(1 + elem_shape.len());
    new_shape.push(tensors.len());
    new_shape.extend_from_slice(elem_shape);
    let total_elems: usize = new_shape.iter().product();
    let mut data = Vec::with_capacity(total_elems);
    for t in tensors {
        data.extend_from_slice(&t.data);
    }
    Ok(Tensor::new(data, new_shape))
}
/// Relocate axis 0 of `tensor` to position `target` (0-indexed in the final
/// shape), shifting the intervening axes left by one -- equivalent to
/// `numpy.moveaxis(tensor, 0, target)`.
///
/// `stack_tensors_axis0` always materializes the new axis at position 0; this
/// implements the ONNX Scan `scan_output_axes` attribute, which lets a model
/// place that axis anywhere in the per-iteration shape.
pub(super) fn move_axis0_to(tensor: &Tensor, target: usize) -> Result<Tensor, OnnxError> {
    let rank = tensor.shape.len();
    if target >= rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "move_axis0_to: target axis {} >= rank {}",
            target, rank
        )));
    }
    if target == 0 {
        return Ok(Tensor::new(tensor.data.clone(), tensor.shape.clone()));
    }
    // Permutation: new dim d takes its size/coordinate from old dim perm[d].
    // [1, 2, ..., target, 0, target+1, ..., rank-1]
    let mut perm: Vec<usize> = Vec::with_capacity(rank);
    perm.extend(1..=target);
    perm.push(0);
    perm.extend((target + 1)..rank);

    let old_shape = &tensor.shape;
    let new_shape: Vec<usize> = perm.iter().map(|&d| old_shape[d]).collect();

    let mut old_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
    }
    let mut new_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    let total: usize = new_shape.iter().product();
    let mut data = vec![0.0f32; total];
    let mut coord = vec![0usize; rank];
    for (new_flat, dst) in data.iter_mut().enumerate() {
        let mut rem = new_flat;
        for d in 0..rank {
            coord[d] = rem / new_strides[d];
            rem %= new_strides[d];
        }
        let mut old_flat = 0usize;
        for (new_dim, &old_dim) in perm.iter().enumerate() {
            old_flat += coord[new_dim] * old_strides[old_dim];
        }
        *dst = tensor.data[old_flat];
    }
    Ok(Tensor::new(data, new_shape))
}
/// Slice a tensor along a given axis at a specific index, removing that dim.
///
/// For example, slicing shape \[3, 4, 5\] along axis 0 at index 1 => \[4, 5\].
pub(super) fn slice_along_axis(
    tensor: &Tensor,
    axis: usize,
    index: usize,
) -> Result<Tensor, OnnxError> {
    if axis >= tensor.shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "slice_along_axis: axis {} >= rank {}",
            axis,
            tensor.shape.len()
        )));
    }
    if index >= tensor.shape[axis] {
        return Err(OnnxError::ShapeMismatch(format!(
            "slice_along_axis: index {} >= dim {} at axis {}",
            index, tensor.shape[axis], axis
        )));
    }
    let rank = tensor.shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * tensor.shape[i + 1];
    }
    let mut new_shape: Vec<usize> = Vec::with_capacity(rank - 1);
    for (i, &dim) in tensor.shape.iter().enumerate() {
        if i != axis {
            new_shape.push(dim);
        }
    }
    let new_size: usize = if new_shape.is_empty() {
        1
    } else {
        new_shape.iter().product()
    };
    let mut data = Vec::with_capacity(new_size);
    let total_elements: usize = tensor.shape.iter().product();
    for flat_idx in 0..total_elements {
        let coord_at_axis = (flat_idx / strides[axis]) % tensor.shape[axis];
        if coord_at_axis == index {
            if let Some(&val) = tensor.data.get(flat_idx) {
                data.push(val);
            }
        }
    }
    if new_shape.is_empty() {
        new_shape.push(1);
    }
    Ok(Tensor::new(data, new_shape))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_flow::{IfOp, LoopOp, ScanOp};
    use oxionnx_core::graph::{Attributes, Graph, Node, OpKind};
    use oxionnx_core::operator::{Operator, OperatorRegistry};
    fn default_attrs() -> Attributes {
        Attributes::default()
    }
    fn test_registry() -> OperatorRegistry {
        let mut r = OperatorRegistry::new();
        r.register(Box::new(crate::registry::nn_ops::ReluOp));
        r.register(Box::new(crate::registry::nn_ops::SigmoidOp));
        r.register(Box::new(crate::registry::misc_ops::IdentityOp));
        r.register(Box::new(crate::registry::math_ops::AddOp));
        r.register(Box::new(crate::registry::math_ops::MulOp));
        r.register(Box::new(IfOp));
        r.register(Box::new(LoopOp));
        r.register(Box::new(ScanOp));
        r
    }
    #[test]
    fn test_if_op_true_branch() {
        let registry = test_registry();
        let then_graph = Graph {
            nodes: vec![Node {
                op: OpKind::Relu,
                name: "relu_node".into(),
                inputs: vec!["X".into()],
                outputs: vec!["Y".into()],
                attrs: default_attrs(),
            }],
            input_names: vec![],
            output_names: vec!["Y".into()],
            ..Default::default()
        };
        let else_graph = Graph {
            nodes: vec![Node {
                op: OpKind::Sigmoid,
                name: "sigmoid_node".into(),
                inputs: vec!["X".into()],
                outputs: vec!["Y".into()],
                attrs: default_attrs(),
            }],
            input_names: vec![],
            output_names: vec!["Y".into()],
            ..Default::default()
        };
        let mut attrs = default_attrs();
        attrs.graphs.insert("then_branch".into(), then_graph);
        attrs.graphs.insert("else_branch".into(), else_graph);
        let node = Node {
            op: OpKind::If,
            name: "if_node".into(),
            inputs: vec!["cond".into()],
            outputs: vec!["result".into()],
            attrs,
        };
        let cond_true = Tensor::scalar(1.0);
        let x_tensor = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
        let inputs: Vec<Option<&Tensor>> = vec![Some(&cond_true)];
        let mut outer_scope = HashMap::new();
        outer_scope.insert("X".into(), x_tensor);
        let ctx = OpContext {
            node: &node,
            inputs,
            outer_scope: Some(&outer_scope),
            weights: None,
            registry: Some(&registry),
        };
        let results = IfOp.execute(&ctx).expect("If op should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, vec![0.0, 2.0, 0.0, 4.0]);
    }
    #[test]
    fn test_if_op_false_branch() {
        let registry = test_registry();
        let then_graph = Graph {
            nodes: vec![Node {
                op: OpKind::Relu,
                name: "relu_node".into(),
                inputs: vec!["X".into()],
                outputs: vec!["Y".into()],
                attrs: default_attrs(),
            }],
            input_names: vec![],
            output_names: vec!["Y".into()],
            ..Default::default()
        };
        let else_graph = Graph {
            nodes: vec![Node {
                op: OpKind::Sigmoid,
                name: "sigmoid_node".into(),
                inputs: vec!["X".into()],
                outputs: vec!["Y".into()],
                attrs: default_attrs(),
            }],
            input_names: vec![],
            output_names: vec!["Y".into()],
            ..Default::default()
        };
        let mut attrs = default_attrs();
        attrs.graphs.insert("then_branch".into(), then_graph);
        attrs.graphs.insert("else_branch".into(), else_graph);
        let node = Node {
            op: OpKind::If,
            name: "if_node".into(),
            inputs: vec!["cond".into()],
            outputs: vec!["result".into()],
            attrs,
        };
        let cond_false = Tensor::scalar(0.0);
        let x_tensor = Tensor::new(vec![0.0], vec![1]);
        let inputs: Vec<Option<&Tensor>> = vec![Some(&cond_false)];
        let mut outer_scope = HashMap::new();
        outer_scope.insert("X".into(), x_tensor);
        let ctx = OpContext {
            node: &node,
            inputs,
            outer_scope: Some(&outer_scope),
            weights: None,
            registry: Some(&registry),
        };
        let results = IfOp.execute(&ctx).expect("If op should succeed");
        assert_eq!(results.len(), 1);
        let expected = 1.0 / (1.0 + (-0.0_f32).exp());
        assert!((results[0].data[0] - expected).abs() < 1e-6);
    }
    #[test]
    fn test_loop_op_count_to_5() {
        let registry = test_registry();
        let body = Graph {
            nodes: vec![
                Node {
                    op: OpKind::Add,
                    name: "add_node".into(),
                    inputs: vec!["accum".into(), "one".into()],
                    outputs: vec!["accum_out".into()],
                    attrs: default_attrs(),
                },
                Node {
                    op: OpKind::Identity,
                    name: "cond_pass".into(),
                    inputs: vec!["cond_in".into()],
                    outputs: vec!["cond_out".into()],
                    attrs: default_attrs(),
                },
                Node {
                    op: OpKind::Identity,
                    name: "scan_pass".into(),
                    inputs: vec!["accum_out".into()],
                    outputs: vec!["scan_out".into()],
                    attrs: default_attrs(),
                },
            ],
            input_names: vec!["iter_num".into(), "cond_in".into(), "accum".into()],
            output_names: vec!["cond_out".into(), "accum_out".into(), "scan_out".into()],
            ..Default::default()
        };
        let mut attrs = default_attrs();
        attrs.graphs.insert("body".into(), body);
        let node = Node {
            op: OpKind::Loop,
            name: "loop_node".into(),
            inputs: vec!["max_trip".into(), "init_cond".into(), "init_accum".into()],
            outputs: vec!["final_accum".into(), "scan_values".into()],
            attrs,
        };
        let max_trip = Tensor::scalar(5.0);
        let init_cond = Tensor::scalar(1.0);
        let init_accum = Tensor::scalar(0.0);
        let inputs: Vec<Option<&Tensor>> =
            vec![Some(&max_trip), Some(&init_cond), Some(&init_accum)];
        let mut outer_scope = HashMap::new();
        outer_scope.insert("one".into(), Tensor::scalar(1.0));
        let ctx = OpContext {
            node: &node,
            inputs,
            outer_scope: Some(&outer_scope),
            weights: None,
            registry: Some(&registry),
        };
        let results = LoopOp.execute(&ctx).expect("Loop op should succeed");
        assert_eq!(results[0].data, vec![5.0]);
        assert_eq!(results[1].data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // Regression for a3-5: scan outputs are STACKED along a new leading
        // axis (rank+1), not concatenated. Each iteration's scan_out here has
        // shape [1] (Identity of a scalar-shaped [1] tensor), so 5 iterations
        // must stack to [5, 1], not collapse to [5].
        assert_eq!(
            results[1].shape,
            vec![5, 1],
            "Loop scan output must be stacked to [num_iterations, 1], not concatenated to [num_iterations]"
        );
    }
    #[test]
    fn test_loop_op_zero_iterations() {
        let registry = test_registry();
        let body = Graph {
            nodes: vec![Node {
                op: OpKind::Identity,
                name: "pass_cond".into(),
                inputs: vec!["cond_in".into()],
                outputs: vec!["cond_out".into()],
                attrs: default_attrs(),
            }],
            input_names: vec!["iter_num".into(), "cond_in".into()],
            output_names: vec!["cond_out".into()],
            ..Default::default()
        };
        let mut attrs = default_attrs();
        attrs.graphs.insert("body".into(), body);
        let node = Node {
            op: OpKind::Loop,
            name: "loop_node".into(),
            inputs: vec!["max_trip".into(), "init_cond".into()],
            outputs: vec![],
            attrs,
        };
        let max_trip = Tensor::scalar(0.0);
        let init_cond = Tensor::scalar(1.0);
        let inputs: Vec<Option<&Tensor>> = vec![Some(&max_trip), Some(&init_cond)];
        let ctx = OpContext {
            node: &node,
            inputs,
            outer_scope: None,
            weights: None,
            registry: Some(&registry),
        };
        let results = LoopOp.execute(&ctx).expect("Loop op should succeed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_scan_op_relu_sequence() {
        let registry = test_registry();
        let body = Graph {
            nodes: vec![
                Node {
                    op: OpKind::Identity,
                    name: "state_pass".into(),
                    inputs: vec!["state_in".into()],
                    outputs: vec!["state_out".into()],
                    attrs: default_attrs(),
                },
                Node {
                    op: OpKind::Relu,
                    name: "relu_node".into(),
                    inputs: vec!["scan_elem".into()],
                    outputs: vec!["scan_out".into()],
                    attrs: default_attrs(),
                },
            ],
            input_names: vec!["state_in".into(), "scan_elem".into()],
            output_names: vec!["state_out".into(), "scan_out".into()],
            ..Default::default()
        };
        let mut attrs = default_attrs();
        attrs.graphs.insert("body".into(), body);
        attrs.ints.insert("num_scan_inputs".into(), 1);
        let node = Node {
            op: OpKind::Scan,
            name: "scan_node".into(),
            inputs: vec!["init_state".into(), "sequence".into()],
            outputs: vec!["final_state".into(), "scan_output".into()],
            attrs,
        };
        let init_state = Tensor::scalar(0.0);
        let sequence = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4, 1]);
        let inputs: Vec<Option<&Tensor>> = vec![Some(&init_state), Some(&sequence)];
        let ctx = OpContext {
            node: &node,
            inputs,
            outer_scope: None,
            weights: None,
            registry: Some(&registry),
        };
        let results = ScanOp.execute(&ctx).expect("Scan op should succeed");
        assert_eq!(results[0].data, vec![0.0]);
        assert_eq!(results[1].data, vec![0.0, 2.0, 0.0, 4.0]);
        assert_eq!(results[1].shape, vec![4, 1]);
    }
    /// Regression for a3-6: `scan_input_axes` accepts spec-legal negative
    /// values ("counting from the back"). Before the fix, `-1 as usize`
    /// wrapped to `usize::MAX` and always failed the `>= rank` bounds check,
    /// rejecting the model outright. Here axis -1 on a rank-3 [2, 3, 4] input
    /// must normalize to axis 2 (size 4) -- a genuinely non-zero axis, so
    /// this also exercises `slice_along_axis` at axis != 0 from Scan for the
    /// first time.
    #[test]
    fn test_scan_negative_input_axis_resolves_nonzero() {
        let registry = test_registry();
        let body = Graph {
            nodes: vec![Node {
                op: OpKind::Identity,
                name: "pass".into(),
                inputs: vec!["scan_elem".into()],
                outputs: vec!["scan_out".into()],
                attrs: default_attrs(),
            }],
            input_names: vec!["scan_elem".into()],
            output_names: vec!["scan_out".into()],
            ..Default::default()
        };
        let mut attrs = default_attrs();
        attrs.graphs.insert("body".into(), body);
        attrs.ints.insert("num_scan_inputs".into(), 1);
        attrs.int_lists.insert("scan_input_axes".into(), vec![-1]);
        let node = Node {
            op: OpKind::Scan,
            name: "scan_node".into(),
            inputs: vec!["seq".into()],
            outputs: vec!["mapped".into()],
            attrs,
        };
        // shape [2, 3, 4]; data[i] = i so slicing axis 2 at index k picks
        // every element whose flat index ≡ k (mod 4).
        let seq_data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let seq = Tensor::new(seq_data, vec![2, 3, 4]);
        let inputs: Vec<Option<&Tensor>> = vec![Some(&seq)];
        let ctx = OpContext {
            node: &node,
            inputs,
            outer_scope: None,
            weights: None,
            registry: Some(&registry),
        };
        let results = ScanOp
            .execute(&ctx)
            .expect("Scan with negative scan_input_axes should succeed");
        // 4 steps along axis 2 (size 4), each step shape [2, 3] -> stacked [4, 2, 3].
        assert_eq!(results[0].shape, vec![4, 2, 3]);
        assert_eq!(
            results[0].data,
            vec![
                0.0, 4.0, 8.0, 12.0, 16.0, 20.0, // step 0 (axis-2 index 0)
                1.0, 5.0, 9.0, 13.0, 17.0, 21.0, // step 1
                2.0, 6.0, 10.0, 14.0, 18.0, 22.0, // step 2
                3.0, 7.0, 11.0, 15.0, 19.0, 23.0, // step 3
            ]
        );
    }
    /// Regression for a3-6: `scan_output_axes` and `scan_output_directions`
    /// composed together on the same output, so an ordering mistake between
    /// "reverse then place axis" vs "place axis then reverse" would be
    /// caught (each attribute tested in isolation would not catch it).
    #[test]
    fn test_scan_output_axes_and_directions_combined() {
        let registry = test_registry();
        let body = Graph {
            nodes: vec![Node {
                op: OpKind::Identity,
                name: "pass".into(),
                inputs: vec!["scan_elem".into()],
                outputs: vec!["scan_out".into()],
                attrs: default_attrs(),
            }],
            input_names: vec!["scan_elem".into()],
            output_names: vec!["scan_out".into()],
            ..Default::default()
        };
        let mut attrs = default_attrs();
        attrs.graphs.insert("body".into(), body);
        attrs.ints.insert("num_scan_inputs".into(), 1);
        // Place the new iteration axis at position 1 (not the default 0),
        // and prepend each iteration's value (reverse accumulation order).
        attrs.int_lists.insert("scan_output_axes".into(), vec![1]);
        attrs
            .int_lists
            .insert("scan_output_directions".into(), vec![1]);
        let node = Node {
            op: OpKind::Scan,
            name: "scan_node".into(),
            inputs: vec!["seq".into()],
            outputs: vec!["mapped".into()],
            attrs,
        };
        // 3 steps, each a length-2 vector: step0=[1,2], step1=[3,4], step2=[5,6].
        let seq = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let inputs: Vec<Option<&Tensor>> = vec![Some(&seq)];
        let ctx = OpContext {
            node: &node,
            inputs,
            outer_scope: None,
            weights: None,
            registry: Some(&registry),
        };
        let results = ScanOp
            .execute(&ctx)
            .expect("Scan with output axes+directions should succeed");
        // Forward + axis 0 would stack to [3,2] = [[1,2],[3,4],[5,6]].
        // direction=1 (prepend) reverses accumulation to [[5,6],[3,4],[1,2]]
        // (still shape [3,2]). axis=1 then moves the new axis from 0 to 1,
        // transposing to shape [2,3]: [[5,3,1],[6,4,2]] flattened.
        assert_eq!(results[0].shape, vec![2, 3]);
        assert_eq!(results[0].data, vec![5.0, 3.0, 1.0, 6.0, 4.0, 2.0]);
    }
    #[test]
    fn test_if_op_with_outer_scope() {
        let registry = test_registry();
        let then_graph = Graph {
            nodes: vec![Node {
                op: OpKind::Mul,
                name: "mul_node".into(),
                inputs: vec!["X".into(), "scale".into()],
                outputs: vec!["Y".into()],
                attrs: default_attrs(),
            }],
            input_names: vec![],
            output_names: vec!["Y".into()],
            ..Default::default()
        };
        let else_graph = Graph {
            nodes: vec![Node {
                op: OpKind::Identity,
                name: "id_node".into(),
                inputs: vec!["X".into()],
                outputs: vec!["Y".into()],
                attrs: default_attrs(),
            }],
            input_names: vec![],
            output_names: vec!["Y".into()],
            ..Default::default()
        };
        let mut attrs = default_attrs();
        attrs.graphs.insert("then_branch".into(), then_graph);
        attrs.graphs.insert("else_branch".into(), else_graph);
        let node = Node {
            op: OpKind::If,
            name: "if_node".into(),
            inputs: vec!["cond".into()],
            outputs: vec!["result".into()],
            attrs,
        };
        let cond = Tensor::scalar(1.0);
        let x_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let scale_tensor = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]);
        let mut outer_scope = HashMap::new();
        outer_scope.insert("X".into(), x_tensor);
        outer_scope.insert("scale".into(), scale_tensor);
        let inputs: Vec<Option<&Tensor>> = vec![Some(&cond)];
        let ctx = OpContext {
            node: &node,
            inputs,
            outer_scope: Some(&outer_scope),
            weights: None,
            registry: Some(&registry),
        };
        let results = IfOp.execute(&ctx).expect("If op should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data, vec![2.0, 4.0, 6.0]);
    }
    #[test]
    fn test_move_axis0_to_zero_is_identity() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let result = move_axis0_to(&t, 0).expect("should move");
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
    #[test]
    fn test_move_axis0_to_transpose_rank2() {
        // [[1,2],[3,4],[5,6]] shape [3,2] -> moveaxis(0,1) -> shape [2,3]
        // = [[1,3,5],[2,4,6]]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let result = move_axis0_to(&t, 1).expect("should move");
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }
    #[test]
    fn test_move_axis0_to_rank3_middle() {
        // shape [2,2,2], values 0..8 (N=2 stacked axis0) -> moveaxis(0,1) -> shape [2,2,2]
        // old[n][a][b] -> new[a][n][b]
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 2, 2]);
        let result = move_axis0_to(&t, 1).expect("should move");
        assert_eq!(result.shape, vec![2, 2, 2]);
        // new[a][n][b] = old[n][a][b]:
        // new[0][0][*]=old[0][0][*]=[0,1]; new[0][1][*]=old[1][0][*]=[4,5]
        // new[1][0][*]=old[0][1][*]=[2,3]; new[1][1][*]=old[1][1][*]=[6,7]
        assert_eq!(result.data, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }
    #[test]
    fn test_slice_along_axis() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let result = slice_along_axis(&tensor, 0, 1).expect("should slice");
        assert_eq!(result.data, vec![3.0, 4.0]);
        assert_eq!(result.shape, vec![2]);
    }
    #[test]
    fn test_slice_along_axis_1() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let result = slice_along_axis(&tensor, 1, 0).expect("should slice");
        assert_eq!(result.data, vec![1.0, 3.0, 5.0]);
        assert_eq!(result.shape, vec![3]);
    }
}
