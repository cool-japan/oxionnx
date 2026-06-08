//! ONNX-ML tree ensemble operator implementations.
//!
//! Covers TreeEnsembleClassifier and TreeEnsembleRegressor.

use oxionnx_core::{OnnxError, OpContext, Tensor};

use crate::ml::{apply_post_transform, PostTransform};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Node traversal mode constants.
const MODE_BRANCH_LEQ: u8 = 0;
const MODE_BRANCH_LT: u8 = 1;
const MODE_BRANCH_GTE: u8 = 2;
const MODE_BRANCH_GT: u8 = 3;
const MODE_BRANCH_EQ: u8 = 4;
const MODE_BRANCH_NEQ: u8 = 5;
const MODE_LEAF: u8 = 6;

/// Parse a mode string into its numeric encoding.
fn parse_mode(s: &str) -> u8 {
    match s {
        "BRANCH_LEQ" => MODE_BRANCH_LEQ,
        "BRANCH_LT" => MODE_BRANCH_LT,
        "BRANCH_GTE" => MODE_BRANCH_GTE,
        "BRANCH_GT" => MODE_BRANCH_GT,
        "BRANCH_EQ" => MODE_BRANCH_EQ,
        "BRANCH_NEQ" => MODE_BRANCH_NEQ,
        "LEAF" => MODE_LEAF,
        _ => MODE_BRANCH_LEQ, // default
    }
}

/// Evaluate whether the branch comparison is true.
#[inline]
fn branch_true(mode: u8, feature_val: f32, threshold: f32) -> bool {
    match mode {
        MODE_BRANCH_LEQ => feature_val <= threshold,
        MODE_BRANCH_LT => feature_val < threshold,
        MODE_BRANCH_GTE => feature_val >= threshold,
        MODE_BRANCH_GT => feature_val > threshold,
        MODE_BRANCH_EQ => (feature_val - threshold).abs() < f32::EPSILON,
        MODE_BRANCH_NEQ => (feature_val - threshold).abs() >= f32::EPSILON,
        _ => false, // LEAF - should not be called
    }
}

/// Build a lookup from (tree_id, node_id) -> flat index for efficient traversal.
fn build_node_index(
    tree_ids: &[i64],
    node_ids: &[i64],
) -> std::collections::HashMap<(i64, i64), usize> {
    let mut map = std::collections::HashMap::new();
    for (idx, (&tid, &nid)) in tree_ids.iter().zip(node_ids.iter()).enumerate() {
        map.insert((tid, nid), idx);
    }
    map
}

// ── TreeEnsembleClassifier ─────────────────────────────────────────────────

/// ONNX-ML TreeEnsembleClassifier operator.
///
/// Input 0: X \[N, features\] (2D tensor)
/// Output 0: predicted labels \[N\] (as f32)
/// Output 1: class scores \[N, num_classes\]
pub fn tree_ensemble_classifier(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    // Parse tree structure from attributes
    let feature_ids = attrs.ints("nodes_featureids");
    let thresholds = attrs
        .float_lists
        .get("nodes_values")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let tree_ids = attrs.ints("nodes_treeids");
    let node_ids = attrs.ints("nodes_nodeids");
    let false_ids = attrs.ints("nodes_falsenodeids");
    let true_ids = attrs.ints("nodes_truenodeids");

    // Parse modes: either from string_lists or int_lists
    let mode_strings = attrs.string_list("nodes_modes");
    let mode_ints = attrs.ints("nodes_modes_int");
    let modes: Vec<u8> = if !mode_strings.is_empty() {
        mode_strings.iter().map(|s| parse_mode(s)).collect()
    } else if !mode_ints.is_empty() {
        mode_ints.iter().map(|&v| v as u8).collect()
    } else {
        // Fallback: try string attribute (comma-separated) or default all to BRANCH_LEQ
        let mode_str = attrs.s("nodes_modes");
        if !mode_str.is_empty() {
            mode_str.split(',').map(|s| parse_mode(s.trim())).collect()
        } else {
            vec![MODE_BRANCH_LEQ; tree_ids.len()]
        }
    };

    // Parse leaf info
    let class_ids = attrs.ints("class_ids");
    let class_node_ids = attrs.ints("class_nodeids");
    let class_tree_ids = attrs.ints("class_treeids");
    let class_weights = attrs
        .float_lists
        .get("class_weights")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let post_transform_str = attrs.s("post_transform");
    let post_transform = PostTransform::parse(post_transform_str);

    let base_values = attrs
        .float_lists
        .get("base_values")
        .cloned()
        .unwrap_or_default();

    let class_labels = attrs.ints("classlabels_int64s");

    // Determine dimensions
    let n = x.shape[0];
    let features = if x.shape.len() > 1 {
        x.shape[1]
    } else {
        x.numel() / n
    };

    // Determine number of classes
    let num_classes = if !class_labels.is_empty() {
        class_labels.len()
    } else {
        // Infer from max class_id + 1
        let max_class = class_ids.iter().copied().max().unwrap_or(0);
        (max_class + 1) as usize
    };

    // Determine number of trees
    let num_trees = tree_ids.iter().copied().max().map(|m| m + 1).unwrap_or(0) as usize;

    // Build node index for efficient lookup
    let node_index = build_node_index(tree_ids, node_ids);

    // Build leaf lookup: (tree_id, node_id) -> list of (class_id, weight)
    let mut leaf_lookup: std::collections::HashMap<(i64, i64), Vec<(usize, f32)>> =
        std::collections::HashMap::new();
    for i in 0..class_node_ids.len() {
        if i < class_tree_ids.len() && i < class_ids.len() && i < class_weights.len() {
            leaf_lookup
                .entry((class_tree_ids[i], class_node_ids[i]))
                .or_default()
                .push((class_ids[i] as usize, class_weights[i]));
        }
    }

    // Allocate output scores
    let mut all_scores = vec![0.0f32; n * num_classes];

    // Process each sample
    for sample_idx in 0..n {
        let x_offset = sample_idx * features;
        let score_offset = sample_idx * num_classes;

        // Initialize with base values
        for c in 0..num_classes {
            if c < base_values.len() {
                all_scores[score_offset + c] = base_values[c];
            }
        }

        // Traverse each tree
        for tree_id in 0..num_trees {
            let tid = tree_id as i64;

            // Find root node (nodeids == 0) for this tree
            let root_idx = match node_index.get(&(tid, 0)) {
                Some(&idx) => idx,
                None => continue, // Skip trees with no root
            };

            // Traverse the tree
            let mut current_idx = root_idx;
            loop {
                if current_idx >= modes.len() {
                    break;
                }

                let mode = modes[current_idx];
                if mode == MODE_LEAF {
                    // Accumulate leaf weights
                    if let Some(leaves) = leaf_lookup.get(&(tid, node_ids[current_idx])) {
                        for &(class_id, weight) in leaves {
                            if class_id < num_classes {
                                all_scores[score_offset + class_id] += weight;
                            }
                        }
                    }
                    break;
                }

                // Non-leaf: get feature value and compare
                let feat_idx = feature_ids[current_idx] as usize;
                let feature_val = if feat_idx < features {
                    x.data[x_offset + feat_idx]
                } else {
                    0.0
                };
                let threshold = if current_idx < thresholds.len() {
                    thresholds[current_idx]
                } else {
                    0.0
                };

                let next_node_id = if branch_true(mode, feature_val, threshold) {
                    true_ids[current_idx]
                } else {
                    false_ids[current_idx]
                };

                // Look up the next node index
                match node_index.get(&(tid, next_node_id)) {
                    Some(&idx) => current_idx = idx,
                    None => break, // Could not find next node
                }
            }
        }
    }

    // Apply post-transform
    apply_post_transform(&mut all_scores, n, num_classes, post_transform);

    // Compute predicted labels via argmax
    let mut labels = vec![0.0f32; n];
    for (i, label) in labels.iter_mut().enumerate() {
        let row_offset = i * num_classes;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..num_classes {
            if all_scores[row_offset + j] > best_val {
                best_val = all_scores[row_offset + j];
                best_idx = j;
            }
        }
        if !class_labels.is_empty() && best_idx < class_labels.len() {
            *label = class_labels[best_idx] as f32;
        } else {
            *label = best_idx as f32;
        }
    }

    let label_tensor = Tensor::new(labels, vec![n]);
    let score_tensor = Tensor::new(all_scores, vec![n, num_classes]);

    Ok(vec![label_tensor, score_tensor])
}

// ── TreeEnsembleRegressor ──────────────────────────────────────────────────

/// ONNX-ML TreeEnsembleRegressor operator.
///
/// Input 0: X \[N, features\] (2D tensor)
/// Output 0: Y \[N, n_targets\]
pub fn tree_ensemble_regressor(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    // Parse tree structure from attributes
    let feature_ids = attrs.ints("nodes_featureids");
    let thresholds = attrs
        .float_lists
        .get("nodes_values")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let tree_ids = attrs.ints("nodes_treeids");
    let node_ids = attrs.ints("nodes_nodeids");
    let false_ids = attrs.ints("nodes_falsenodeids");
    let true_ids = attrs.ints("nodes_truenodeids");

    // Parse modes
    let mode_strings = attrs.string_list("nodes_modes");
    let mode_ints = attrs.ints("nodes_modes_int");
    let modes: Vec<u8> = if !mode_strings.is_empty() {
        mode_strings.iter().map(|s| parse_mode(s)).collect()
    } else if !mode_ints.is_empty() {
        mode_ints.iter().map(|&v| v as u8).collect()
    } else {
        let mode_str = attrs.s("nodes_modes");
        if !mode_str.is_empty() {
            mode_str.split(',').map(|s| parse_mode(s.trim())).collect()
        } else {
            vec![MODE_BRANCH_LEQ; tree_ids.len()]
        }
    };

    // Parse target info (regressor uses target_* instead of class_*)
    let target_ids = attrs.ints("target_ids");
    let target_node_ids = attrs.ints("target_nodeids");
    let target_tree_ids = attrs.ints("target_treeids");
    let target_weights = attrs
        .float_lists
        .get("target_weights")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let post_transform_str = attrs.s("post_transform");
    let post_transform = PostTransform::parse(post_transform_str);

    let base_values = attrs
        .float_lists
        .get("base_values")
        .cloned()
        .unwrap_or_default();

    let n_targets = attrs.i("n_targets", 1) as usize;
    let aggregate_function = attrs.s("aggregate_function");

    // Determine dimensions
    let n = x.shape[0];
    let features = if x.shape.len() > 1 {
        x.shape[1]
    } else {
        x.numel() / n
    };

    let num_trees = tree_ids.iter().copied().max().map(|m| m + 1).unwrap_or(0) as usize;

    // Build node index
    let node_index = build_node_index(tree_ids, node_ids);

    // Build leaf lookup: (tree_id, node_id) -> list of (target_id, weight)
    let mut leaf_lookup: std::collections::HashMap<(i64, i64), Vec<(usize, f32)>> =
        std::collections::HashMap::new();
    for i in 0..target_node_ids.len() {
        if i < target_tree_ids.len() && i < target_ids.len() && i < target_weights.len() {
            leaf_lookup
                .entry((target_tree_ids[i], target_node_ids[i]))
                .or_default()
                .push((target_ids[i] as usize, target_weights[i]));
        }
    }

    // Allocate output
    let mut output = vec![0.0f32; n * n_targets];

    // Process each sample
    for sample_idx in 0..n {
        let x_offset = sample_idx * features;
        let out_offset = sample_idx * n_targets;

        // Initialize with base values
        for t in 0..n_targets {
            if t < base_values.len() {
                output[out_offset + t] = base_values[t];
            }
        }

        // Traverse each tree
        for tree_id in 0..num_trees {
            let tid = tree_id as i64;

            let root_idx = match node_index.get(&(tid, 0)) {
                Some(&idx) => idx,
                None => continue,
            };

            let mut current_idx = root_idx;
            loop {
                if current_idx >= modes.len() {
                    break;
                }

                let mode = modes[current_idx];
                if mode == MODE_LEAF {
                    if let Some(leaves) = leaf_lookup.get(&(tid, node_ids[current_idx])) {
                        for &(target_id, weight) in leaves {
                            if target_id < n_targets {
                                output[out_offset + target_id] += weight;
                            }
                        }
                    }
                    break;
                }

                let feat_idx = feature_ids[current_idx] as usize;
                let feature_val = if feat_idx < features {
                    x.data[x_offset + feat_idx]
                } else {
                    0.0
                };
                let threshold = if current_idx < thresholds.len() {
                    thresholds[current_idx]
                } else {
                    0.0
                };

                let next_node_id = if branch_true(mode, feature_val, threshold) {
                    true_ids[current_idx]
                } else {
                    false_ids[current_idx]
                };

                match node_index.get(&(tid, next_node_id)) {
                    Some(&idx) => current_idx = idx,
                    None => break,
                }
            }
        }

        // Apply aggregate function (AVERAGE divides by number of trees)
        if aggregate_function == "AVERAGE" && num_trees > 0 {
            let divisor = num_trees as f32;
            for t in 0..n_targets {
                output[out_offset + t] /= divisor;
            }
        }
    }

    // Apply post-transform
    apply_post_transform(&mut output, n, n_targets, post_transform);

    Ok(vec![Tensor::new(output, vec![n, n_targets])])
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn make_context(
        op: OpKind,
        inputs: Vec<Option<&Tensor>>,
        attrs: Attributes,
    ) -> (Node, Vec<Option<&Tensor>>) {
        let node = Node {
            op,
            name: "test_node".to_string(),
            inputs: vec![],
            outputs: vec![],
            attrs,
        };
        (node, inputs)
    }

    fn ctx_from<'a>(node: &'a Node, inputs: &'a [Option<&'a Tensor>]) -> OpContext<'a> {
        OpContext {
            node,
            inputs: inputs.to_vec(),
            outer_scope: None,
            weights: None,
            registry: None,
        }
    }

    #[test]
    fn test_tree_ensemble_classifier_2tree_2class() {
        // Build a simple 2-tree ensemble for binary classification (2 features).
        //
        // Tree 0: if x[0] <= 0.5 -> leaf(class=0, w=1.0) else leaf(class=1, w=1.0)
        //   Node 0: feature=0, threshold=0.5, true->1, false->2
        //   Node 1: leaf (class=0, weight=1.0)
        //   Node 2: leaf (class=1, weight=1.0)
        //
        // Tree 1: if x[1] <= 0.5 -> leaf(class=0, w=1.0) else leaf(class=1, w=1.0)
        //   Node 0: feature=1, threshold=0.5, true->1, false->2
        //   Node 1: leaf (class=0, weight=1.0)
        //   Node 2: leaf (class=1, weight=1.0)

        let x = Tensor::new(
            vec![
                0.0, 0.0, // sample 0: both features < 0.5 => class 0 (2 votes)
                1.0, 1.0, // sample 1: both features > 0.5 => class 1 (2 votes)
                0.0, 1.0, // sample 2: split votes => tied, argmax picks 0
            ],
            vec![3, 2],
        );

        let mut attrs = Attributes::default();
        // nodes_treeids: tree0 has 3 nodes, tree1 has 3 nodes
        attrs
            .int_lists
            .insert("nodes_treeids".into(), vec![0, 0, 0, 1, 1, 1]);
        attrs
            .int_lists
            .insert("nodes_nodeids".into(), vec![0, 1, 2, 0, 1, 2]);
        attrs
            .int_lists
            .insert("nodes_featureids".into(), vec![0, 0, 0, 1, 0, 0]);
        attrs
            .float_lists
            .insert("nodes_values".into(), vec![0.5, 0.0, 0.0, 0.5, 0.0, 0.0]);
        attrs
            .int_lists
            .insert("nodes_truenodeids".into(), vec![1, 0, 0, 1, 0, 0]);
        attrs
            .int_lists
            .insert("nodes_falsenodeids".into(), vec![2, 0, 0, 2, 0, 0]);

        // Modes: BRANCH_LEQ for node 0 in each tree, LEAF for nodes 1,2
        attrs.string_lists.insert(
            "nodes_modes".into(),
            vec![
                "BRANCH_LEQ".into(),
                "LEAF".into(),
                "LEAF".into(),
                "BRANCH_LEQ".into(),
                "LEAF".into(),
                "LEAF".into(),
            ],
        );

        // Leaf info: 4 leaf entries
        // Tree 0, node 1 -> class 0, weight 1.0
        // Tree 0, node 2 -> class 1, weight 1.0
        // Tree 1, node 1 -> class 0, weight 1.0
        // Tree 1, node 2 -> class 1, weight 1.0
        attrs
            .int_lists
            .insert("class_treeids".into(), vec![0, 0, 1, 1]);
        attrs
            .int_lists
            .insert("class_nodeids".into(), vec![1, 2, 1, 2]);
        attrs.int_lists.insert("class_ids".into(), vec![0, 1, 0, 1]);
        attrs
            .float_lists
            .insert("class_weights".into(), vec![1.0, 1.0, 1.0, 1.0]);

        attrs
            .int_lists
            .insert("classlabels_int64s".into(), vec![0, 1]);
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::TreeEnsembleClassifier, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = tree_ensemble_classifier(&ctx).expect("tree_ensemble_classifier failed");

        assert_eq!(result.len(), 2);

        let labels = &result[0];
        assert_eq!(labels.shape, vec![3]);
        // Sample 0: class 0 gets 2 votes, class 1 gets 0
        assert!((labels.data[0] - 0.0).abs() < 1e-5);
        // Sample 1: class 0 gets 0 votes, class 1 gets 2
        assert!((labels.data[1] - 1.0).abs() < 1e-5);
        // Sample 2: class 0 gets 1 vote (tree 0), class 1 gets 1 vote (tree 1) -> tie, first wins
        assert!((labels.data[2] - 0.0).abs() < 1e-5);

        let scores = &result[1];
        assert_eq!(scores.shape, vec![3, 2]);
        // Sample 0: [2.0, 0.0]
        assert!((scores.data[0] - 2.0).abs() < 1e-5);
        assert!((scores.data[1] - 0.0).abs() < 1e-5);
        // Sample 1: [0.0, 2.0]
        assert!((scores.data[2] - 0.0).abs() < 1e-5);
        assert!((scores.data[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_tree_ensemble_regressor_single_tree() {
        // Single tree with 2 features:
        // if x[0] <= 1.0 -> leaf(target=0, weight=10.0)
        // else -> leaf(target=0, weight=20.0)
        //
        // Tree 0:
        //   Node 0: feature=0, threshold=1.0, true->1, false->2
        //   Node 1: leaf
        //   Node 2: leaf

        let x = Tensor::new(
            vec![
                0.5, 0.0, // sample 0: x[0] <= 1.0 => 10.0
                2.0, 0.0, // sample 1: x[0] > 1.0 => 20.0
            ],
            vec![2, 2],
        );

        let mut attrs = Attributes::default();
        attrs
            .int_lists
            .insert("nodes_treeids".into(), vec![0, 0, 0]);
        attrs
            .int_lists
            .insert("nodes_nodeids".into(), vec![0, 1, 2]);
        attrs
            .int_lists
            .insert("nodes_featureids".into(), vec![0, 0, 0]);
        attrs
            .float_lists
            .insert("nodes_values".into(), vec![1.0, 0.0, 0.0]);
        attrs
            .int_lists
            .insert("nodes_truenodeids".into(), vec![1, 0, 0]);
        attrs
            .int_lists
            .insert("nodes_falsenodeids".into(), vec![2, 0, 0]);

        attrs.string_lists.insert(
            "nodes_modes".into(),
            vec!["BRANCH_LEQ".into(), "LEAF".into(), "LEAF".into()],
        );

        attrs.int_lists.insert("target_treeids".into(), vec![0, 0]);
        attrs.int_lists.insert("target_nodeids".into(), vec![1, 2]);
        attrs.int_lists.insert("target_ids".into(), vec![0, 0]);
        attrs
            .float_lists
            .insert("target_weights".into(), vec![10.0, 20.0]);

        attrs.ints.insert("n_targets".into(), 1);
        attrs.strings.insert("post_transform".into(), "NONE".into());

        let (node, inputs) = make_context(OpKind::TreeEnsembleRegressor, vec![Some(&x)], attrs);
        let ctx = ctx_from(&node, &inputs);
        let result = tree_ensemble_regressor(&ctx).expect("tree_ensemble_regressor failed");

        assert_eq!(result.len(), 1);
        let y = &result[0];
        assert_eq!(y.shape, vec![2, 1]);
        assert!((y.data[0] - 10.0).abs() < 1e-5);
        assert!((y.data[1] - 20.0).abs() < 1e-5);
    }
}
