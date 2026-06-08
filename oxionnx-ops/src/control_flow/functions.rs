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
pub(super) fn execute_subgraph(
    graph: &Graph,
    subgraph_inputs: HashMap<String, Tensor>,
    outer_scope: &HashMap<String, Tensor>,
    weights: &HashMap<String, Tensor>,
    registry: &OperatorRegistry,
) -> Result<Vec<Tensor>, OnnxError> {
    let mut known_names: Vec<String> = subgraph_inputs.keys().cloned().collect();
    for name in outer_scope.keys() {
        known_names.push(name.clone());
    }
    for name in weights.keys() {
        known_names.push(name.clone());
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
        let mut merged_scope = outer_scope.clone();
        for (k, v) in &intermediates {
            merged_scope.insert(k.clone(), v.clone());
        }
        let ctx = OpContext {
            node,
            inputs: resolved_inputs,
            outer_scope: Some(&merged_scope),
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
/// Concatenate a list of tensors along axis 0.
/// Each tensor must have the same shape except possibly along axis 0.
pub(super) fn concatenate_tensors_axis0(tensors: &[Tensor]) -> Result<Tensor, OnnxError> {
    if tensors.is_empty() {
        return Ok(Tensor::new(vec![], vec![0]));
    }
    let first = &tensors[0];
    if first.shape.is_empty() {
        let data: Vec<f32> = tensors
            .iter()
            .flat_map(|t| t.data.iter().copied())
            .collect();
        let len = data.len();
        return Ok(Tensor::new(data, vec![len]));
    }
    let mut new_shape = first.shape.clone();
    let mut total_dim0 = 0usize;
    for t in tensors {
        if t.shape.len() != first.shape.len() {
            return Err(OnnxError::ShapeMismatch(
                "concatenate_tensors_axis0: rank mismatch".into(),
            ));
        }
        for (i, (&a, &b)) in first.shape.iter().zip(t.shape.iter()).enumerate() {
            if i != 0 && a != b {
                return Err(OnnxError::ShapeMismatch(format!(
                    "concatenate_tensors_axis0: dim {} mismatch: {} vs {}",
                    i, a, b
                )));
            }
        }
        total_dim0 += t.shape[0];
    }
    new_shape[0] = total_dim0;
    let mut data = Vec::with_capacity(new_shape.iter().product());
    for t in tensors {
        data.extend_from_slice(&t.data);
    }
    Ok(Tensor::new(data, new_shape))
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
    fn test_concatenate_tensors_axis0_scalars() {
        let tensors = vec![
            Tensor::scalar(1.0),
            Tensor::scalar(2.0),
            Tensor::scalar(3.0),
        ];
        let result = concatenate_tensors_axis0(&tensors).expect("should concat");
        assert_eq!(result.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.shape, vec![3]);
    }
    #[test]
    fn test_concatenate_tensors_axis0_matrices() {
        let tensors = vec![
            Tensor::new(vec![1.0, 2.0], vec![1, 2]),
            Tensor::new(vec![3.0, 4.0], vec![1, 2]),
            Tensor::new(vec![5.0, 6.0], vec![1, 2]),
        ];
        let result = concatenate_tensors_axis0(&tensors).expect("should concat");
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(result.shape, vec![3, 2]);
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
