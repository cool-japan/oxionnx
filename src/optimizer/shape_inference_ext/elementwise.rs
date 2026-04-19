//! Element-wise and variadic broadcast helpers.

use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

use crate::optimizer::shape_inference::get_input_shape;

/// Broadcast all variadic inputs together.
pub(super) fn infer_variadic_broadcast(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    if node.inputs.is_empty() {
        return None;
    }
    let mut result = get_input_shape(node, 0, known)?;
    for i in 1..node.inputs.len() {
        let shape = get_input_shape(node, i, known)?;
        result = Tensor::broadcast_shape(&result, &shape).ok()?;
    }
    Some(vec![result])
}
