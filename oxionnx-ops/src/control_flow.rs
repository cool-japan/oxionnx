//! Control flow operators: If, Loop, Scan.
//!
//! These operators execute subgraphs conditionally or iteratively, enabling
//! dynamic control flow within ONNX models.

use oxionnx_core::error::OnnxError;
use oxionnx_core::graph::Graph;
use oxionnx_core::operator::{OpContext, Operator, OperatorRegistry};
use oxionnx_core::tensor::Tensor;
use std::collections::HashMap;

// ─── Subgraph execution helper ─────────────────────────────────────────────

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
fn execute_subgraph(
    graph: &Graph,
    subgraph_inputs: HashMap<String, Tensor>,
    outer_scope: &HashMap<String, Tensor>,
    weights: &HashMap<String, Tensor>,
    registry: &OperatorRegistry,
) -> Result<Vec<Tensor>, OnnxError> {
    // Build the set of initially known names for topological sort
    let mut known_names: Vec<String> = subgraph_inputs.keys().cloned().collect();
    for name in outer_scope.keys() {
        known_names.push(name.clone());
    }
    for name in weights.keys() {
        known_names.push(name.clone());
    }

    let order = graph.topological_sort(&known_names);

    // Intermediates: start with subgraph inputs
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

        // Resolve inputs: intermediates -> outer_scope -> weights
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

        // Build a merged outer scope for nested subgraphs
        let mut merged_scope = outer_scope.clone();
        for (k, v) in &intermediates {
            merged_scope.insert(k.clone(), v.clone());
        }

        let ctx = OpContext {
            node,
            inputs: resolved_inputs,
            outer_scope: Some(&merged_scope),
            registry: Some(registry),
        };

        let results = operator.execute(&ctx)?;

        // Store outputs
        for (out_name, tensor) in node.outputs.iter().zip(results) {
            if !out_name.is_empty() {
                intermediates.insert(out_name.clone(), tensor);
            }
        }
    }

    // Collect outputs in the order specified by graph.output_names
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

// ─── If operator ────────────────────────────────────────────────────────────

/// ONNX If operator.
///
/// Conditionally executes one of two subgraphs based on a boolean condition.
/// - Input 0: condition (scalar bool-like tensor; data\[0\] != 0.0 means true)
/// - Attributes: "then_branch" (Graph), "else_branch" (Graph)
/// - Outputs: the outputs of the selected branch subgraph
pub struct IfOp;

impl Operator for IfOp {
    fn op_type(&self) -> &str {
        "If"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let cond = ctx.input(0)?;
        let condition_val = cond
            .data
            .first()
            .ok_or_else(|| OnnxError::InvalidModel("If: condition tensor is empty".into()))?;
        let is_true = *condition_val != 0.0;

        let branch_name = if is_true {
            "then_branch"
        } else {
            "else_branch"
        };
        let graph = ctx.attrs().graph(branch_name).ok_or_else(|| {
            OnnxError::InvalidModel(format!("If: missing '{}' attribute", branch_name))
        })?;

        let registry = ctx.registry.ok_or_else(|| {
            OnnxError::InvalidModel("If: registry not available for subgraph execution".into())
        })?;

        // Build outer scope from the context's outer_scope (if any)
        let empty_scope = HashMap::new();
        let outer = ctx.outer_scope.unwrap_or(&empty_scope);

        // If subgraphs receive no explicit inputs (they read from outer scope)
        let subgraph_inputs = HashMap::new();
        let weights = HashMap::new();

        execute_subgraph(graph, subgraph_inputs, outer, &weights, registry)
    }
}

// ─── Loop operator ──────────────────────────────────────────────────────────

/// ONNX Loop operator.
///
/// Repeatedly executes a body subgraph until a condition becomes false or the
/// maximum trip count is reached.
///
/// - Input 0: max_trip_count (scalar i64-like; empty string name means infinite)
/// - Input 1: initial condition (scalar bool-like; empty string name means true)
/// - Inputs 2..N: initial values for loop-carried dependencies
/// - Attribute: "body" (Graph)
/// - Body inputs: (iteration_num, condition, ...carried_deps)
/// - Body outputs: (condition_out, ...carried_deps_out, ...scan_outputs)
/// - Final outputs: final carried deps + concatenated scan outputs
pub struct LoopOp;

impl Operator for LoopOp {
    fn op_type(&self) -> &str {
        "Loop"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        // Parse max trip count
        let max_trip_count: Option<i64> = ctx
            .optional_input(0)
            .and_then(|t| t.data.first().map(|v| *v as i64));

        // Parse initial condition
        let initial_cond: bool = ctx
            .optional_input(1)
            .map(|t| t.data.first().copied().unwrap_or(1.0) != 0.0)
            .unwrap_or(true);

        // Gather initial loop-carried dependencies
        let num_total_inputs = ctx.inputs.len();
        let mut carried_deps: Vec<Tensor> = Vec::new();
        for i in 2..num_total_inputs {
            if let Some(t) = ctx.optional_input(i) {
                carried_deps.push(t.clone());
            }
        }

        let body = ctx
            .attrs()
            .graph("body")
            .ok_or_else(|| OnnxError::InvalidModel("Loop: missing 'body' attribute".into()))?;

        let registry = ctx.registry.ok_or_else(|| {
            OnnxError::InvalidModel("Loop: registry not available for subgraph execution".into())
        })?;

        let empty_scope = HashMap::new();
        let outer = ctx.outer_scope.unwrap_or(&empty_scope);
        let weights = HashMap::new();

        // Number of body outputs = 1 (condition) + num_carried + num_scan
        let num_carried = carried_deps.len();
        let num_body_outputs = body.output_names.len();
        if num_body_outputs < 1 + num_carried {
            return Err(OnnxError::InvalidModel(format!(
                "Loop: body has {} outputs but expected at least {} (1 cond + {} carried)",
                num_body_outputs,
                1 + num_carried,
                num_carried
            )));
        }
        let num_scan_outputs = num_body_outputs - 1 - num_carried;

        // Accumulate scan outputs (one Vec per scan output)
        let mut scan_accumulators: Vec<Vec<Tensor>> = vec![Vec::new(); num_scan_outputs];

        let mut condition = initial_cond;
        let mut iteration: i64 = 0;

        loop {
            // Check termination conditions
            if !condition {
                break;
            }
            if let Some(max) = max_trip_count {
                if iteration >= max {
                    break;
                }
            }

            // Build subgraph inputs
            let mut subgraph_inputs = HashMap::new();

            // Input 0: iteration number (as f32 scalar)
            if let Some(iter_name) = body.input_names.first() {
                if !iter_name.is_empty() {
                    subgraph_inputs.insert(iter_name.clone(), Tensor::scalar(iteration as f32));
                }
            }

            // Input 1: condition (as f32 scalar)
            if let Some(cond_name) = body.input_names.get(1) {
                if !cond_name.is_empty() {
                    let cond_val = if condition { 1.0_f32 } else { 0.0_f32 };
                    subgraph_inputs.insert(cond_name.clone(), Tensor::scalar(cond_val));
                }
            }

            // Inputs 2..N: carried dependencies
            for (i, dep) in carried_deps.iter().enumerate() {
                if let Some(name) = body.input_names.get(2 + i) {
                    if !name.is_empty() {
                        subgraph_inputs.insert(name.clone(), dep.clone());
                    }
                }
            }

            let outputs = execute_subgraph(body, subgraph_inputs, outer, &weights, registry)?;

            // Parse outputs
            // Output 0: new condition
            let new_cond = outputs
                .first()
                .ok_or_else(|| OnnxError::InvalidModel("Loop: body produced no outputs".into()))?;
            condition = new_cond.data.first().copied().unwrap_or(0.0) != 0.0;

            // Outputs 1..1+num_carried: updated carried deps
            carried_deps.clear();
            for i in 0..num_carried {
                let dep = outputs.get(1 + i).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Loop: body missing carried dep output at index {}",
                        1 + i
                    ))
                })?;
                carried_deps.push(dep.clone());
            }

            // Remaining outputs: scan outputs
            for (i, accumulator) in scan_accumulators.iter_mut().enumerate() {
                let scan_out = outputs.get(1 + num_carried + i).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Loop: body missing scan output at index {}",
                        1 + num_carried + i
                    ))
                })?;
                accumulator.push(scan_out.clone());
            }

            iteration += 1;

            // Safety: prevent infinite loops with a very high bound
            if iteration > 1_000_000 {
                return Err(OnnxError::InvalidModel(
                    "Loop: exceeded 1,000,000 iterations safety limit".into(),
                ));
            }
        }

        // Build final outputs: carried deps + concatenated scan outputs
        let mut final_outputs = carried_deps;

        for accumulator in scan_accumulators {
            if accumulator.is_empty() {
                // Empty scan output: return an empty tensor
                final_outputs.push(Tensor::new(vec![], vec![0]));
            } else {
                let concatenated = concatenate_tensors_axis0(&accumulator)?;
                final_outputs.push(concatenated);
            }
        }

        Ok(final_outputs)
    }
}

/// Concatenate a list of tensors along axis 0.
/// Each tensor must have the same shape except possibly along axis 0.
fn concatenate_tensors_axis0(tensors: &[Tensor]) -> Result<Tensor, OnnxError> {
    if tensors.is_empty() {
        return Ok(Tensor::new(vec![], vec![0]));
    }

    let first = &tensors[0];

    if first.shape.is_empty() {
        // Scalars: stack into a 1D tensor
        let data: Vec<f32> = tensors
            .iter()
            .flat_map(|t| t.data.iter().copied())
            .collect();
        let len = data.len();
        return Ok(Tensor::new(data, vec![len]));
    }

    // Build new shape: [sum of dim0, ...rest]
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
fn stack_tensors_axis0(tensors: &[Tensor]) -> Result<Tensor, OnnxError> {
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

// ─── Scan operator ──────────────────────────────────────────────────────────

/// ONNX Scan operator.
///
/// Iterates over a sequence dimension of the scan inputs, executing the body
/// subgraph once per element. State tensors are carried across iterations.
///
/// - Inputs 0..M-1: initial state tensors
/// - Inputs M..N-1: scan input sequences
/// - Attribute: "body" (Graph), "num_scan_inputs" (int)
/// - Attribute: "scan_input_axes" (int list, default all 0)
/// - Attribute: "scan_input_directions" (int list, default all 0=forward)
/// - Body inputs: (state_0, ..., state_M-1, scan_elem_0, ..., scan_elem_K-1)
/// - Body outputs: (state_0_out, ..., state_M-1_out, scan_out_0, ...)
/// - Final outputs: final state tensors + scan output sequences (concatenated)
pub struct ScanOp;

impl Operator for ScanOp {
    fn op_type(&self) -> &str {
        "Scan"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let num_scan_inputs = ctx.attrs().i("num_scan_inputs", 0) as usize;
        if num_scan_inputs == 0 {
            return Err(OnnxError::InvalidModel(
                "Scan: num_scan_inputs must be > 0".into(),
            ));
        }

        let total_inputs = ctx.num_inputs();
        if total_inputs < num_scan_inputs {
            return Err(OnnxError::InvalidModel(format!(
                "Scan: expected at least {} inputs (num_scan_inputs), got {}",
                num_scan_inputs, total_inputs
            )));
        }
        let num_state = total_inputs - num_scan_inputs;

        // Gather initial state tensors
        let mut states: Vec<Tensor> = Vec::with_capacity(num_state);
        for i in 0..num_state {
            states.push(ctx.input(i)?.clone());
        }

        // Gather scan input tensors
        let mut scan_inputs: Vec<&Tensor> = Vec::with_capacity(num_scan_inputs);
        for i in num_state..total_inputs {
            scan_inputs.push(ctx.input(i)?);
        }

        // Parse scan input axes (default: all 0)
        let scan_input_axes_attr = ctx.attrs().ints("scan_input_axes");
        let scan_input_axes: Vec<usize> = if scan_input_axes_attr.is_empty() {
            vec![0; num_scan_inputs]
        } else {
            scan_input_axes_attr.iter().map(|&x| x as usize).collect()
        };

        // Parse scan input directions (default: all 0 = forward)
        let scan_input_dirs_attr = ctx.attrs().ints("scan_input_directions");
        let scan_input_dirs: Vec<i64> = if scan_input_dirs_attr.is_empty() {
            vec![0; num_scan_inputs]
        } else {
            scan_input_dirs_attr.to_vec()
        };

        // Determine sequence length from the first scan input
        let first_scan = scan_inputs
            .first()
            .ok_or_else(|| OnnxError::InvalidModel("Scan: no scan inputs".into()))?;
        let scan_axis = scan_input_axes.first().copied().unwrap_or(0);
        if scan_axis >= first_scan.shape.len() {
            return Err(OnnxError::InvalidModel(format!(
                "Scan: scan_input_axis {} >= rank {}",
                scan_axis,
                first_scan.shape.len()
            )));
        }
        let seq_len = first_scan.shape[scan_axis];

        let body = ctx
            .attrs()
            .graph("body")
            .ok_or_else(|| OnnxError::InvalidModel("Scan: missing 'body' attribute".into()))?;

        let registry = ctx.registry.ok_or_else(|| {
            OnnxError::InvalidModel("Scan: registry not available for subgraph execution".into())
        })?;

        let empty_scope = HashMap::new();
        let outer = ctx.outer_scope.unwrap_or(&empty_scope);
        let weights = HashMap::new();

        // Body outputs: num_state states + scan_outputs
        let num_body_outputs = body.output_names.len();
        if num_body_outputs < num_state {
            return Err(OnnxError::InvalidModel(format!(
                "Scan: body has {} outputs, expected >= {} state outputs",
                num_body_outputs, num_state
            )));
        }
        let num_scan_outputs = num_body_outputs - num_state;
        let mut scan_accumulators: Vec<Vec<Tensor>> = vec![Vec::new(); num_scan_outputs];

        for step in 0..seq_len {
            let mut subgraph_inputs = HashMap::new();

            // State inputs
            for (i, state) in states.iter().enumerate() {
                if let Some(name) = body.input_names.get(i) {
                    if !name.is_empty() {
                        subgraph_inputs.insert(name.clone(), state.clone());
                    }
                }
            }

            // Scan element inputs: slice each scan input along its axis
            for (si, scan_tensor) in scan_inputs.iter().enumerate() {
                let axis = scan_input_axes.get(si).copied().unwrap_or(0);
                let direction = scan_input_dirs.get(si).copied().unwrap_or(0);
                let actual_step = if direction != 0 {
                    seq_len - 1 - step
                } else {
                    step
                };

                let element = slice_along_axis(scan_tensor, axis, actual_step)?;

                if let Some(name) = body.input_names.get(num_state + si) {
                    if !name.is_empty() {
                        subgraph_inputs.insert(name.clone(), element);
                    }
                }
            }

            let outputs = execute_subgraph(body, subgraph_inputs, outer, &weights, registry)?;

            // Update states
            states.clear();
            for i in 0..num_state {
                let state = outputs.get(i).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Scan: body missing state output at index {}",
                        i
                    ))
                })?;
                states.push(state.clone());
            }

            // Accumulate scan outputs
            for (i, accumulator) in scan_accumulators.iter_mut().enumerate() {
                let scan_out = outputs.get(num_state + i).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Scan: body missing scan output at index {}",
                        num_state + i
                    ))
                })?;
                accumulator.push(scan_out.clone());
            }
        }

        // Build final outputs: states + concatenated scan outputs
        let mut final_outputs = states;

        for accumulator in scan_accumulators {
            if accumulator.is_empty() {
                final_outputs.push(Tensor::new(vec![], vec![0]));
            } else {
                let stacked = stack_tensors_axis0(&accumulator)?;
                final_outputs.push(stacked);
            }
        }

        Ok(final_outputs)
    }
}

/// Slice a tensor along a given axis at a specific index, removing that dim.
///
/// For example, slicing shape \[3, 4, 5\] along axis 0 at index 1 => \[4, 5\].
fn slice_along_axis(tensor: &Tensor, axis: usize, index: usize) -> Result<Tensor, OnnxError> {
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

    // Compute strides
    let rank = tensor.shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * tensor.shape[i + 1];
    }

    // New shape: remove the axis dimension
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

    // Select elements where the axis coordinate == index
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
        // Scalar result
        new_shape.push(1);
    }

    Ok(Tensor::new(data, new_shape))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Graph, Node, OpKind};

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

    // ── If op tests ─────────────────────────────────────────────────────

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
            registry: Some(&registry),
        };

        let results = IfOp.execute(&ctx).expect("If op should succeed");
        assert_eq!(results.len(), 1);
        // Relu of [-1, 2, -3, 4] = [0, 2, 0, 4]
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
            registry: Some(&registry),
        };

        let results = IfOp.execute(&ctx).expect("If op should succeed");
        assert_eq!(results.len(), 1);
        // Sigmoid(0.0) = 0.5
        let expected = 1.0 / (1.0 + (-0.0_f32).exp());
        assert!((results[0].data[0] - expected).abs() < 1e-6);
    }

    // ── Loop op tests ───────────────────────────────────────────────────

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
            registry: Some(&registry),
        };

        let results = LoopOp.execute(&ctx).expect("Loop op should succeed");

        // Output 0: final accum = 5.0
        assert_eq!(results[0].data, vec![5.0]);

        // Output 1: scan output = [1.0, 2.0, 3.0, 4.0, 5.0]
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
            registry: Some(&registry),
        };

        let results = LoopOp.execute(&ctx).expect("Loop op should succeed");
        assert!(results.is_empty());
    }

    // ── Scan op tests ───────────────────────────────────────────────────

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
            registry: Some(&registry),
        };

        let results = ScanOp.execute(&ctx).expect("Scan op should succeed");

        // Output 0: final state (unchanged)
        assert_eq!(results[0].data, vec![0.0]);

        // Output 1: Relu of each element = [0, 2, 0, 4], shape [4, 1]
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
            registry: Some(&registry),
        };

        let results = IfOp.execute(&ctx).expect("If op should succeed");
        assert_eq!(results.len(), 1);
        // X * scale = [2.0, 4.0, 6.0]
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
