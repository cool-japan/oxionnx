//! Shape inference for spatial transformation operators.
//!
//! Covers DepthToSpace, SpaceToDepth, GridSample, and RoiAlign.

use crate::graph::Node;
use std::collections::HashMap;

use crate::optimizer::shape_inference::get_input_shape;

/// DepthToSpace: [N, C, H, W] -> [N, C/(block^2), H*block, W*block]
pub(super) fn infer_depth_to_space_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.len() != 4 {
        return None;
    }

    let blocksize = node.attrs.i("blocksize", 0);
    if blocksize <= 0 {
        return None;
    }
    let bs = blocksize as usize;

    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    if c % (bs * bs) != 0 {
        return None;
    }

    Some(vec![vec![n, c / (bs * bs), h * bs, w * bs]])
}

/// SpaceToDepth: [N, C, H, W] -> [N, C*(block^2), H/block, W/block]
pub(super) fn infer_space_to_depth_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    if input_shape.len() != 4 {
        return None;
    }

    let blocksize = node.attrs.i("blocksize", 0);
    if blocksize <= 0 {
        return None;
    }
    let bs = blocksize as usize;

    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    if h % bs != 0 || w % bs != 0 {
        return None;
    }

    Some(vec![vec![n, c * bs * bs, h / bs, w / bs]])
}

/// GridSample: output shape [N, C, H_out, W_out] where H_out, W_out come from grid.
pub(super) fn infer_grid_sample_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let grid_shape = get_input_shape(node, 1, known)?;

    if input_shape.len() != 4 || grid_shape.len() != 4 {
        return None;
    }

    // Input: [N, C, H_in, W_in], Grid: [N, H_out, W_out, 2]
    Some(vec![vec![
        input_shape[0],
        input_shape[1],
        grid_shape[1],
        grid_shape[2],
    ]])
}

/// RoiAlign: output shape [num_rois, C, output_height, output_width]
pub(super) fn infer_roi_align_shape(
    node: &Node,
    known: &HashMap<String, Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    let input_shape = get_input_shape(node, 0, known)?;
    let rois_shape = get_input_shape(node, 1, known)?;

    if input_shape.len() != 4 || rois_shape.len() != 2 {
        return None;
    }

    let num_rois = rois_shape[0];
    let c = input_shape[1];
    let output_height = node.attrs.i("output_height", 1) as usize;
    let output_width = node.attrs.i("output_width", 1) as usize;

    Some(vec![vec![num_rois, c, output_height, output_width]])
}
