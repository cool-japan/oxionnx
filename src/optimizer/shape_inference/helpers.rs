//! Shared helper utilities for shape inference.

use crate::graph::Node;
use std::collections::HashMap;

/// Get the shape of the i-th input, or None if unavailable.
pub(crate) fn get_input_shape(
    node: &Node,
    idx: usize,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<usize>> {
    let name = node.inputs.get(idx)?;
    if name.is_empty() {
        return None;
    }
    known.get(name).cloned()
}
