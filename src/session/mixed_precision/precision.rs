//! Precision conversion and graph consumer analysis utilities.

use crate::tensor::Tensor;

use super::classify::should_use_f16;

/// Convert a f32 tensor's data to f16 precision (stored as f32 values).
///
/// Each f32 value is rounded to the nearest f16 representable value.
/// This simulates f16 storage precision loss while keeping the f32 container.
pub fn round_to_f16_precision(tensor: &Tensor) -> Tensor {
    let data: Vec<f32> = tensor
        .data
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    Tensor::new(data, tensor.shape.clone())
}

/// Determine whether all downstream consumers of a node's outputs are f16-safe.
///
/// Used to decide whether intermediate results should be kept in f16 precision
/// or promoted back to f32.
pub fn next_consumers_all_f16(
    node_outputs: &[String],
    all_nodes: &[crate::graph::Node],
    current_node_idx: usize,
) -> bool {
    for output_name in node_outputs {
        if output_name.is_empty() {
            continue;
        }
        for node in all_nodes.iter().skip(current_node_idx + 1) {
            if node.inputs.contains(output_name) && !should_use_f16(node.op.as_str()) {
                return false;
            }
        }
    }
    true
}
