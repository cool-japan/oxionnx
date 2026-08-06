//! ONNX-ML tree ensemble operator implementations.
//!
//! Covers TreeEnsembleClassifier and TreeEnsembleRegressor.
//!
//! The node tables (`nodes_treeids`, `nodes_nodeids`, `nodes_featureids`,
//! `nodes_values`, `nodes_truenodeids`, `nodes_falsenodeids`, `nodes_modes`,
//! `nodes_missing_value_tracks_true`) are parallel arrays. They are validated
//! once up front, so the traversal can index them without further checks, and
//! every traversal is bounded by the number of nodes in its tree — a malformed
//! model with a cycle (or a self-referencing leaf) yields
//! [`OnnxError::InvalidModel`] instead of hanging the process.

use std::collections::HashMap;

use oxionnx_core::graph::Attributes;
use oxionnx_core::{OnnxError, OpContext, Tensor};

use crate::ml::{apply_post_transform, batch_dims, PostTransform};

// ── Node modes ─────────────────────────────────────────────────────────────

/// Node traversal mode (`nodes_modes` entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeMode {
    BranchLeq,
    BranchLt,
    BranchGte,
    BranchGt,
    BranchEq,
    BranchNeq,
    Leaf,
}

impl NodeMode {
    /// Parse a `nodes_modes` string entry.
    fn parse(s: &str, op: &str) -> Result<Self, OnnxError> {
        match s.trim() {
            "BRANCH_LEQ" => Ok(Self::BranchLeq),
            "BRANCH_LT" => Ok(Self::BranchLt),
            "BRANCH_GTE" => Ok(Self::BranchGte),
            "BRANCH_GT" => Ok(Self::BranchGt),
            "BRANCH_EQ" => Ok(Self::BranchEq),
            "BRANCH_NEQ" => Ok(Self::BranchNeq),
            "LEAF" => Ok(Self::Leaf),
            other => Err(OnnxError::InvalidModel(format!(
                "{op}: unknown nodes_modes entry '{other}'"
            ))),
        }
    }

    /// Parse the numeric encoding accepted through `nodes_modes_int`.
    fn from_i64(value: i64, op: &str) -> Result<Self, OnnxError> {
        match value {
            0 => Ok(Self::BranchLeq),
            1 => Ok(Self::BranchLt),
            2 => Ok(Self::BranchGte),
            3 => Ok(Self::BranchGt),
            4 => Ok(Self::BranchEq),
            5 => Ok(Self::BranchNeq),
            6 => Ok(Self::Leaf),
            other => Err(OnnxError::InvalidModel(format!(
                "{op}: unknown nodes_modes_int entry {other}"
            ))),
        }
    }

    /// Evaluate the branch comparison. NaN feature values make every ordered
    /// comparison false and `BRANCH_NEQ` true, exactly as in onnxruntime.
    #[inline]
    fn branch_true(self, value: f32, threshold: f32) -> bool {
        match self {
            Self::BranchLeq => value <= threshold,
            Self::BranchLt => value < threshold,
            Self::BranchGte => value >= threshold,
            Self::BranchGt => value > threshold,
            Self::BranchEq => value == threshold,
            Self::BranchNeq => value != threshold,
            Self::Leaf => false,
        }
    }
}

// ── Aggregation ────────────────────────────────────────────────────────────

/// `aggregate_function` of TreeEnsembleRegressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Aggregate {
    Sum,
    Average,
    Min,
    Max,
}

impl Aggregate {
    fn parse(s: &str) -> Result<Self, OnnxError> {
        match s {
            "" | "SUM" => Ok(Self::Sum),
            "AVERAGE" => Ok(Self::Average),
            "MIN" => Ok(Self::Min),
            "MAX" => Ok(Self::Max),
            other => Err(OnnxError::InvalidModel(format!(
                "TreeEnsembleRegressor: unknown aggregate_function '{other}' \
                 (expected SUM, AVERAGE, MIN or MAX)"
            ))),
        }
    }

    /// Fold one leaf weight into the running per-target accumulator.
    #[inline]
    fn fold(self, acc: &mut f32, seen: &mut bool, weight: f32) {
        match self {
            Self::Sum | Self::Average => {
                *acc += weight;
            }
            Self::Min => {
                if !*seen || weight < *acc {
                    *acc = weight;
                }
            }
            Self::Max => {
                if !*seen || weight > *acc {
                    *acc = weight;
                }
            }
        }
        *seen = true;
    }
}

// ── Node tables ────────────────────────────────────────────────────────────

/// Root of one tree of the ensemble.
struct TreeRoot {
    tree_id: i64,
    /// Flat index of the root node.
    index: usize,
    /// Number of nodes belonging to this tree; doubles as the traversal cap.
    node_count: usize,
}

/// Validated parallel node tables shared by both tree ensemble operators.
struct TreeEnsemble<'a> {
    feature_ids: &'a [i64],
    thresholds: &'a [f32],
    node_ids: &'a [i64],
    true_ids: &'a [i64],
    false_ids: &'a [i64],
    modes: Vec<NodeMode>,
    missing_tracks_true: Vec<bool>,
    node_index: HashMap<(i64, i64), usize>,
    roots: Vec<TreeRoot>,
}

impl<'a> TreeEnsemble<'a> {
    /// Read and validate every `nodes_*` attribute.
    fn parse(attrs: &'a Attributes, op: &str) -> Result<Self, OnnxError> {
        let tree_ids = attrs.ints("nodes_treeids");
        let node_ids = attrs.ints("nodes_nodeids");
        let feature_ids = attrs.ints("nodes_featureids");
        let true_ids = attrs.ints("nodes_truenodeids");
        let false_ids = attrs.ints("nodes_falsenodeids");
        let thresholds = attrs
            .float_lists
            .get("nodes_values")
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let n_nodes = tree_ids.len();
        let modes = parse_modes(attrs, n_nodes, op)?;

        // One validation pass over every parallel array: the traversal below
        // indexes all of them with the same node index.
        check_len("nodes_nodeids", node_ids.len(), n_nodes, op)?;
        check_len("nodes_featureids", feature_ids.len(), n_nodes, op)?;
        check_len("nodes_truenodeids", true_ids.len(), n_nodes, op)?;
        check_len("nodes_falsenodeids", false_ids.len(), n_nodes, op)?;
        check_len("nodes_values", thresholds.len(), n_nodes, op)?;
        check_len("nodes_modes", modes.len(), n_nodes, op)?;

        for (i, &feature) in feature_ids.iter().enumerate() {
            if feature < 0 {
                return Err(OnnxError::InvalidModel(format!(
                    "{op}: nodes_featureids[{i}] is negative ({feature})"
                )));
            }
        }

        let missing = attrs.ints("nodes_missing_value_tracks_true");
        let missing_tracks_true = if missing.is_empty() {
            vec![false; n_nodes]
        } else {
            check_len(
                "nodes_missing_value_tracks_true",
                missing.len(),
                n_nodes,
                op,
            )?;
            missing.iter().map(|&v| v != 0).collect()
        };

        // (tree_id, node_id) -> flat index, plus per-tree roots and sizes.
        let mut node_index: HashMap<(i64, i64), usize> = HashMap::with_capacity(n_nodes);
        let mut tree_slots: HashMap<i64, usize> = HashMap::new();
        let mut roots: Vec<TreeRoot> = Vec::new();
        for (idx, (&tid, &nid)) in tree_ids.iter().zip(node_ids.iter()).enumerate() {
            node_index.entry((tid, nid)).or_insert(idx);
            match tree_slots.get(&tid) {
                Some(&slot) => {
                    if let Some(root) = roots.get_mut(slot) {
                        root.node_count += 1;
                    }
                }
                None => {
                    tree_slots.insert(tid, roots.len());
                    roots.push(TreeRoot {
                        tree_id: tid,
                        index: idx,
                        node_count: 1,
                    });
                }
            }
        }
        // Node id 0 is the conventional root; fall back to the first node of
        // the tree in array order when it is absent.
        for root in roots.iter_mut() {
            if let Some(&idx) = node_index.get(&(root.tree_id, 0)) {
                root.index = idx;
            }
        }

        Ok(Self {
            feature_ids,
            thresholds,
            node_ids,
            true_ids,
            false_ids,
            modes,
            missing_tracks_true,
            node_index,
            roots,
        })
    }

    /// Walk one tree for one sample and return the flat index of the leaf.
    ///
    /// Returns `Ok(None)` when a branch points at a node that does not exist
    /// (the tree contributes nothing), and [`OnnxError::InvalidModel`] when the
    /// walk exceeds the node count of the tree, which can only happen when the
    /// node table contains a cycle.
    fn leaf_of(&self, tree: &TreeRoot, row: &[f32], op: &str) -> Result<Option<usize>, OnnxError> {
        let mut idx = tree.index;
        let mut steps = 0usize;

        loop {
            // Safe: every parallel array was checked to have `modes.len()`
            // entries, and `idx` always comes from `node_index`.
            let mode = self.modes[idx];
            if mode == NodeMode::Leaf {
                return Ok(Some(idx));
            }
            if steps >= tree.node_count {
                return Err(OnnxError::InvalidModel(format!(
                    "{op}: traversal of tree {} exceeded its {} nodes; \
                     the node table contains a cycle",
                    tree.tree_id, tree.node_count
                )));
            }
            steps += 1;

            let feature = self.feature_ids[idx] as usize;
            let value = row.get(feature).copied().unwrap_or(0.0);
            let threshold = self.thresholds[idx];
            let go_true = mode.branch_true(value, threshold)
                || (self.missing_tracks_true[idx] && value.is_nan());
            let next_id = if go_true {
                self.true_ids[idx]
            } else {
                self.false_ids[idx]
            };

            match self.node_index.get(&(tree.tree_id, next_id)) {
                Some(&next) => idx = next,
                None => return Ok(None),
            }
        }
    }

    /// Node id of a flat node index (leaf lookup key).
    #[inline]
    fn node_id(&self, idx: usize) -> i64 {
        self.node_ids[idx]
    }
}

/// Read `nodes_modes` from the STRINGS attribute, the numeric fallback, or a
/// single comma separated string.
fn parse_modes(attrs: &Attributes, n_nodes: usize, op: &str) -> Result<Vec<NodeMode>, OnnxError> {
    let mode_strings = attrs.string_list("nodes_modes");
    if !mode_strings.is_empty() {
        return mode_strings
            .iter()
            .map(|s| NodeMode::parse(s, op))
            .collect();
    }

    let mode_ints = attrs.ints("nodes_modes_int");
    if !mode_ints.is_empty() {
        return mode_ints
            .iter()
            .map(|&v| NodeMode::from_i64(v, op))
            .collect();
    }

    let mode_str = attrs.s("nodes_modes");
    if !mode_str.is_empty() {
        return mode_str
            .split(',')
            .map(|s| NodeMode::parse(s, op))
            .collect();
    }

    if n_nodes == 0 {
        return Ok(Vec::new());
    }

    // Defaulting to BRANCH_LEQ would turn every leaf into a branch whose
    // children are node 0, i.e. an infinite loop back to the root.
    Err(OnnxError::InvalidModel(format!(
        "{op}: required attribute 'nodes_modes' is missing for {n_nodes} nodes"
    )))
}

/// Reject a parallel array whose length differs from the node count.
fn check_len(name: &str, actual: usize, expected: usize, op: &str) -> Result<(), OnnxError> {
    if actual == expected {
        Ok(())
    } else {
        Err(OnnxError::InvalidModel(format!(
            "{op}: '{name}' has {actual} entries but the ensemble declares {expected} nodes"
        )))
    }
}

/// Leaf weights of the ensemble: `(tree_id, node_id) -> [(target_id, weight)]`.
type LeafTable = HashMap<(i64, i64), Vec<(usize, f32)>>;

/// Build the leaf weight table from the `class_*` / `target_*` attributes.
fn build_leaf_table(attrs: &Attributes, prefix: &str, op: &str) -> Result<LeafTable, OnnxError> {
    let ids = attrs.ints(&format!("{prefix}_ids"));
    let node_ids = attrs.ints(&format!("{prefix}_nodeids"));
    let tree_ids = attrs.ints(&format!("{prefix}_treeids"));
    let weights = attrs
        .float_lists
        .get(&format!("{prefix}_weights"))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let count = ids.len();
    check_len(&format!("{prefix}_nodeids"), node_ids.len(), count, op)?;
    check_len(&format!("{prefix}_treeids"), tree_ids.len(), count, op)?;
    check_len(&format!("{prefix}_weights"), weights.len(), count, op)?;

    let mut table: LeafTable = HashMap::new();
    for i in 0..count {
        let id = ids[i];
        if id < 0 {
            return Err(OnnxError::InvalidModel(format!(
                "{op}: '{prefix}_ids[{i}]' is negative ({id})"
            )));
        }
        table
            .entry((tree_ids[i], node_ids[i]))
            .or_default()
            .push((id as usize, weights[i]));
    }
    Ok(table)
}

// ── TreeEnsembleClassifier ─────────────────────────────────────────────────

/// ONNX-ML TreeEnsembleClassifier operator.
///
/// Input 0: X \[N, features\] (a 1-D \[C\] input is one sample with C features)
/// Output 0: predicted labels \[N\] (as f32)
/// Output 1: class scores \[N, num_classes\]
pub fn tree_ensemble_classifier(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    const OP: &str = "TreeEnsembleClassifier";

    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let ensemble = TreeEnsemble::parse(attrs, OP)?;
    let leaf_table = build_leaf_table(attrs, "class", OP)?;

    let post_transform = PostTransform::parse(attrs.s("post_transform"), OP)?;
    let base_values = attrs
        .float_lists
        .get("base_values")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let class_labels = attrs.ints("classlabels_int64s");

    let (n, features) = batch_dims(x, OP)?;

    // Number of classes: labels if present, otherwise the largest class id + 1.
    let num_classes = if !class_labels.is_empty() {
        class_labels.len()
    } else {
        let label_strings = attrs.string_list("classlabels_strings").len();
        if label_strings > 0 {
            label_strings
        } else {
            let max_class = attrs.ints("class_ids").iter().copied().max().unwrap_or(0);
            // onnxruntime caps the class count at 65535; an inferred count
            // beyond that comes from a corrupt `class_ids` list.
            if max_class >= 65535 {
                return Err(OnnxError::InvalidModel(format!(
                    "{OP}: inferred class count {} exceeds the supported maximum",
                    max_class as i128 + 1
                )));
            }
            max_class.max(0) as usize + 1
        }
    };
    if num_classes == 0 {
        return Err(OnnxError::InvalidModel(format!("{OP}: zero classes")));
    }

    let use_base_values = base_values.len() == num_classes;
    let score_count = n.checked_mul(num_classes).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!("{OP}: score buffer size overflows usize"))
    })?;
    let mut all_scores = vec![0.0f32; score_count];

    for sample_idx in 0..n {
        let row = &x.data[sample_idx * features..sample_idx * features + features];
        let score_offset = sample_idx * num_classes;

        for tree in &ensemble.roots {
            let Some(leaf) = ensemble.leaf_of(tree, row, OP)? else {
                continue;
            };
            if let Some(weights) = leaf_table.get(&(tree.tree_id, ensemble.node_id(leaf))) {
                for &(class_id, weight) in weights {
                    if class_id < num_classes {
                        // Classification always accumulates (SUM).
                        all_scores[score_offset + class_id] += weight;
                    }
                }
            }
        }

        if use_base_values {
            for c in 0..num_classes {
                all_scores[score_offset + c] += base_values[c];
            }
        }
    }

    // onnxruntime's `TreeAggregatorClassifier::FinalizeScores` picks the
    // predicted label from the raw aggregated scores (`get_max_weight` then
    // `*Y = class_labels_[...]`) and only afterwards calls `write_scores` to
    // apply `post_transform` to the score output. SOFTMAX/LOGISTIC/PROBIT are
    // monotonic and would not change the label either way, but SOFTMAX_ZERO
    // is deliberately not rank-order-preserving between a class that never
    // received a leaf contribution (left at raw `0.0`) and one that did (so
    // labels must be decided before, not after, it runs). This also matches
    // the convention `svm_classifier` already follows: its label comes from
    // `votes` over the raw decision values, computed before
    // `apply_post_transform` touches the score buffer.
    let labels = argmax_labels(&all_scores, n, num_classes, class_labels);

    apply_post_transform(&mut all_scores, n, num_classes, post_transform);

    Ok(vec![
        Tensor::new(labels, vec![n]),
        Tensor::new(all_scores, vec![n, num_classes]),
    ])
}

/// Pick the highest scoring class per row and map it through the label list.
fn argmax_labels(scores: &[f32], n: usize, num_classes: usize, class_labels: &[i64]) -> Vec<f32> {
    let mut labels = vec![0.0f32; n];
    for (i, label) in labels.iter_mut().enumerate() {
        let row_offset = i * num_classes;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..num_classes {
            let value = scores[row_offset + j];
            if value > best_val {
                best_val = value;
                best_idx = j;
            }
        }
        *label = match class_labels.get(best_idx) {
            Some(&l) => l as f32,
            None => best_idx as f32,
        };
    }
    labels
}

// ── TreeEnsembleRegressor ──────────────────────────────────────────────────

/// ONNX-ML TreeEnsembleRegressor operator.
///
/// Input 0: X \[N, features\] (a 1-D \[C\] input is one sample with C features)
/// Output 0: Y \[N, n_targets\]
pub fn tree_ensemble_regressor(ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
    const OP: &str = "TreeEnsembleRegressor";

    let x = ctx.input(0)?;
    let attrs = ctx.attrs();

    let ensemble = TreeEnsemble::parse(attrs, OP)?;
    let leaf_table = build_leaf_table(attrs, "target", OP)?;

    let post_transform = PostTransform::parse(attrs.s("post_transform"), OP)?;
    let base_values = attrs
        .float_lists
        .get("base_values")
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let aggregate = Aggregate::parse(attrs.s("aggregate_function"))?;

    let n_targets = attrs.i("n_targets", 1);
    if n_targets <= 0 {
        return Err(OnnxError::InvalidModel(format!(
            "{OP}: n_targets must be positive, got {n_targets}"
        )));
    }
    let n_targets = n_targets as usize;

    let (n, features) = batch_dims(x, OP)?;
    let num_trees = ensemble.roots.len();
    let use_base_values = base_values.len() == n_targets;

    let output_count = n.checked_mul(n_targets).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!("{OP}: output buffer size overflows usize"))
    })?;
    let mut output = vec![0.0f32; output_count];
    let mut seen = vec![false; n_targets];

    for sample_idx in 0..n {
        let row = &x.data[sample_idx * features..sample_idx * features + features];
        let out_offset = sample_idx * n_targets;
        seen.iter_mut().for_each(|s| *s = false);

        for tree in &ensemble.roots {
            let Some(leaf) = ensemble.leaf_of(tree, row, OP)? else {
                continue;
            };
            if let Some(weights) = leaf_table.get(&(tree.tree_id, ensemble.node_id(leaf))) {
                for &(target_id, weight) in weights {
                    if target_id < n_targets {
                        aggregate.fold(
                            &mut output[out_offset + target_id],
                            &mut seen[target_id],
                            weight,
                        );
                    }
                }
            }
        }

        // onnxruntime finalizes with the aggregation first and adds
        // `base_values` afterwards, so the base is never averaged.
        for t in 0..n_targets {
            let slot = &mut output[out_offset + t];
            if !seen[t] {
                *slot = 0.0;
            } else if aggregate == Aggregate::Average && num_trees > 0 {
                *slot /= num_trees as f32;
            }
            if use_base_values {
                *slot += base_values[t];
            }
        }
    }

    apply_post_transform(&mut output, n, n_targets, post_transform);

    Ok(vec![Tensor::new(output, vec![n, n_targets])])
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
